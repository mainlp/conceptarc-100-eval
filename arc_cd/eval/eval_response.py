"""
Functions and helpers to evaluate model responses to ARC-style data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Optional, cast

from arc_cd.data_prep import Grid
from arc_cd.eval.parsing import parse_grid_from_generation
from arc_cd.utils import optional_int_tuple, safe_divide, string_tuple


@dataclass(frozen=True)
class CaseEvaluation:
    """Evaluation result for one generated response.

    Attributes:
        index: Row index in the generation JSONL.
        case_id: Stable ARC-style case ID.
        task_id: Stable ARC-style task ID.
        concept: Concept name, if available.
        sample_id: Sample ID, if available.
        test_index: Test query index inside the task or labeled pool.
        token_index: Optional index into a token sidecar archive.
        correct: Whether the predicted grid exactly matches the target grid.
        parse_ok: Whether a valid predicted grid was parsed.
        parse_status: Parser status.
        predicted_grid: Parsed predicted grid, if available.
        target_grid: Ground-truth target grid.
        metadata: Extra generation metadata.
        finish_reason: vLLM finish reason, if available.
        stop_reason: vLLM stop reason, if available.
        response_finished: Whether the response stopped naturally.
        hit_output_token_budget: Whether generation hit the output budget.
        reasoning_budget: Reasoning-budget metadata.
        hit_thinking_token_budget: Whether the reasoning budget was reached.
        forced_reasoning_stop: Whether reasoning was forced to stop.
        reasoning_over_budget: Whether observed reasoning exceeded the budget.
        diagnostic_parse_ok: Whether diagnostic parsing found a grid.
        diagnostic_parse_status: Diagnostic parser status.
        diagnostic_predicted_grid: Diagnostic parsed grid, if available.
        is_resampled_episode: Whether this case belongs to a resampled episode.
        episode_id: Resampled support-set identifier, if available.
        support_indices: Labeled-pool indices used as support, if available.
        query_index: Labeled-pool index used as query, if available.
        support_sources: Original source names for support pairs, if available.
        query_source: Original source name for the query pair, if available.
    """

    index: int
    case_id: str
    task_id: str
    concept: Optional[str]
    sample_id: str
    test_index: int
    token_index: Optional[int]
    correct: bool
    parse_ok: bool
    parse_status: str
    predicted_grid: Optional[Grid]
    target_grid: Grid
    metadata: dict[str, Any]
    finish_reason: Optional[str]
    stop_reason: str | int | None
    response_finished: bool
    hit_output_token_budget: bool
    reasoning_budget: dict[str, Any]
    hit_thinking_token_budget: bool
    forced_reasoning_stop: bool
    reasoning_over_budget: Optional[bool]
    diagnostic_parse_ok: bool
    diagnostic_parse_status: str
    diagnostic_predicted_grid: Optional[Grid]
    is_resampled_episode: bool
    episode_id: Optional[str]
    support_indices: Optional[tuple[int, ...]]
    query_index: Optional[int]
    support_sources: tuple[str, ...]
    query_source: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the evaluation to a JSON-serializable dictionary.

        Returns:
            JSON-serializable evaluation record.
        """
        return asdict(self)


def _optional_str(value: Any) -> Optional[str]:
    """Return a non-empty string or None.

    Args:
        value: Value to normalize.

    Returns:
        Normalized string or None.
    """
    return value if isinstance(value, str) and value else None


def _optional_int(value: Any) -> Optional[int]:
    """Return an integer or None.

    Args:
        value: Value to normalize.

    Returns:
        Normalized integer or None.
    """
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    return None


def _optional_bool(value: Any) -> Optional[bool]:
    """Return a boolean or None.

    Args:
        value: Value to normalize.

    Returns:
        Boolean value or None.
    """
    return value if isinstance(value, bool) else None


def _optional_stop_reason(value: Any) -> str | int | None:
    """Return a stop reason or None.

    Args:
        value: Value to normalize.

    Returns:
        Stop reason string, integer token ID, or None.
    """
    if isinstance(value, bool):
        return None

    if isinstance(value, (str, int)):
        return value

    return None


def _bool_or_false(value: Any) -> bool:
    """Return a boolean value, defaulting malformed values to False.

    Args:
        value: Value to normalize.

    Returns:
        Boolean value.
    """
    return value if isinstance(value, bool) else False


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return generation metadata as a dictionary.

    Args:
        record: Generation record.

    Returns:
        Metadata dictionary. Returns an empty dictionary if metadata is absent
        or malformed.
    """
    metadata = record.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _reasoning_budget(record: dict[str, Any]) -> dict[str, Any]:
    """Return reasoning-budget metadata as a dictionary.

    Args:
        record: Generation record.

    Returns:
        Reasoning-budget metadata dictionary. Returns an empty dictionary if
        metadata is absent or malformed.
    """
    reasoning_budget = record.get("reasoning_budget")
    return dict(reasoning_budget) if isinstance(reasoning_budget, dict) else {}


def _finish_reason_label(finish_reason: Optional[str]) -> str:
    """Return a stable summary label for a finish reason.

    Args:
        finish_reason: Raw finish reason.

    Returns:
        Summary label.
    """
    return finish_reason if finish_reason is not None else "<missing>"


def evaluate_records(
    records: list[dict[str, Any]],
    end_think_token: str = "</think>",
    require_arc_colors: bool = True,
) -> tuple[list[CaseEvaluation], dict[str, Any]]:
    """Evaluate generated ARC-style records.

    Args:
        records: Generation records loaded from JSONL.
        end_think_token: End-thinking marker.
        require_arc_colors: Whether predicted grid values must be in [0, 9].

    Returns:
        Pair of per-case evaluations and summary metrics.

    Raises:
        ValueError: If a record is missing a target grid.
    """
    evaluations: list[CaseEvaluation] = []
    seen_cases: set[str] = set()

    for index, record in enumerate(records):
        case_id = str(record.get("case_id", index))
        if case_id in seen_cases:
            raise ValueError(f"Duplicate case_id {case_id!r} in one evaluation run.")
        seen_cases.add(case_id)
        target_grid = record.get("target_grid")
        if target_grid is None:
            raise ValueError(f"Record {index} has no target_grid; cannot evaluate.")

        parsed = parse_grid_from_generation(
            record,
            end_think_token=end_think_token,
            require_arc_colors=require_arc_colors,
        )

        metadata = _metadata(record)
        reasoning_budget = _reasoning_budget(record)
        finish_reason = _optional_str(record.get("finish_reason"))
        response_finished = finish_reason == "stop"
        correct = parsed.ok and parsed.grid == target_grid

        evaluations.append(
            CaseEvaluation(
                index=index,
                case_id=case_id,
                task_id=str(record.get("task_id", "")),
                concept=_optional_str(record.get("concept")),
                sample_id=str(record.get("sample_id", "")),
                test_index=int(record.get("test_index", 0)),
                token_index=record.get("token_index"),
                correct=bool(correct),
                parse_ok=parsed.ok,
                parse_status=parsed.status,
                predicted_grid=parsed.grid,
                target_grid=cast(Grid, target_grid),
                metadata=metadata,
                finish_reason=finish_reason,
                stop_reason=_optional_stop_reason(record.get("stop_reason")),
                response_finished=response_finished,
                hit_output_token_budget=finish_reason == "length",
                reasoning_budget=reasoning_budget,
                hit_thinking_token_budget=_bool_or_false(
                    reasoning_budget.get("hit_thinking_token_budget")
                ),
                forced_reasoning_stop=_bool_or_false(
                    reasoning_budget.get("forced_stop")
                ),
                reasoning_over_budget=_optional_bool(
                    reasoning_budget.get("over_budget")
                ),
                diagnostic_parse_ok=parsed.diagnostic_ok,
                diagnostic_parse_status=parsed.diagnostic_status,
                diagnostic_predicted_grid=parsed.diagnostic_grid,
                is_resampled_episode=bool(metadata.get("is_resampled_episode", False)),
                episode_id=_optional_str(metadata.get("episode_id")),
                support_indices=optional_int_tuple(metadata.get("support_indices")),
                query_index=_optional_int(metadata.get("query_index")),
                support_sources=string_tuple(metadata.get("support_sources")),
                query_source=_optional_str(metadata.get("query_source")),
            )
        )

    return evaluations, summarize_evaluations(evaluations)


def summarize_group(evaluations: list[CaseEvaluation]) -> dict[str, Any]:
    """Summarize one group of evaluations.

    Args:
        evaluations: Evaluations to summarize.

    Returns:
        Summary metrics.
    """
    n = len(evaluations)
    n_correct = sum(case.correct for case in evaluations)
    n_finished = sum(case.response_finished for case in evaluations)
    n_unfinished = n - n_finished
    n_parse_error = sum(
        case.response_finished and not case.parse_ok for case in evaluations
    )
    n_error = n - n_correct
    n_correct_finished = sum(
        case.correct for case in evaluations if case.response_finished
    )
    n_output_budget_hit = sum(case.hit_output_token_budget for case in evaluations)
    n_hit_thinking_token_budget = sum(
        case.hit_thinking_token_budget for case in evaluations
    )
    n_forced_reasoning_stop = sum(case.forced_reasoning_stop for case in evaluations)
    n_reasoning_over_budget = sum(
        case.reasoning_over_budget is True for case in evaluations
    )

    return {
        "n": n,
        "n_correct": n_correct,
        "n_error": n_error,
        "n_finished": n_finished,
        "n_unfinished": n_unfinished,
        "n_parse_error": n_parse_error,
        "n_output_budget_hit": n_output_budget_hit,
        "n_hit_thinking_token_budget": n_hit_thinking_token_budget,
        "n_forced_reasoning_stop": n_forced_reasoning_stop,
        "n_reasoning_over_budget": n_reasoning_over_budget,
        "accuracy": safe_divide(n_correct, n),
        "accuracy_finished": safe_divide(n_correct_finished, n_finished),
        "finish_rate": safe_divide(n_finished, n),
        "parse_error_rate": safe_divide(n_parse_error, n),
        "parse_error_fraction_of_errors": safe_divide(n_parse_error, n_error),
        "finish_reason_counts": dict(
            sorted(
                Counter(
                    _finish_reason_label(case.finish_reason) for case in evaluations
                ).items()
            )
        ),
        "parse_status_counts": dict(
            sorted(Counter(case.parse_status for case in evaluations).items())
        ),
    }


def summarize_by_attribute(
    evaluations: list[CaseEvaluation],
    attribute: str,
) -> dict[str, dict[str, Any]]:
    """Summarize evaluations grouped by an attribute.

    Args:
        evaluations: Per-case evaluations.
        attribute: CaseEvaluation attribute name to group by.

    Returns:
        Mapping from attribute value to summary metrics. Cases with ``None`` or
        empty-string values are skipped.
    """
    groups: dict[str, list[CaseEvaluation]] = defaultdict(list)

    for evaluation in evaluations:
        value = getattr(evaluation, attribute)
        if value is None or value == "":
            continue
        groups[str(value)].append(evaluation)

    return {
        group_id: summarize_group(group) for group_id, group in sorted(groups.items())
    }


def summarize_evaluations(evaluations: list[CaseEvaluation]) -> dict[str, Any]:
    """Compute overall and grouped evaluation metrics.

    Args:
        evaluations (list[CaseEvaluation]): Scored test queries.

    Returns:
        dict[str, Any]: Accuracy by concept and task, with diagnostics that
        identify unfinished responses and unparseable grids.
    """
    parse_error_cases = [
        case for case in evaluations if case.response_finished and not case.parse_ok
    ]
    unfinished_cases = [case for case in evaluations if not case.response_finished]

    return {
        "overall": summarize_group(evaluations),
        "per_concept": summarize_by_attribute(evaluations, "concept"),
        "per_task": summarize_by_attribute(evaluations, "task_id"),
        "parse_error_indices": [case.index for case in parse_error_cases],
        "parse_error_case_ids": [case.case_id for case in parse_error_cases],
        "parse_error_token_indices": [
            case.token_index
            for case in parse_error_cases
            if case.token_index is not None
        ],
        "unfinished_indices": [case.index for case in unfinished_cases],
        "unfinished_case_ids": [case.case_id for case in unfinished_cases],
        "unfinished_token_indices": [
            case.token_index
            for case in unfinished_cases
            if case.token_index is not None
        ],
    }
