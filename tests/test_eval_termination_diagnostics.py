from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arc_cd.eval.eval_response import evaluate_records
from arc_cd.utils import load_jsonl
from scripts.eval_arc_acc import evaluate_generation_file, get_eval_output_paths


def _record(
    *,
    generated_text: str,
    finish_reason: str,
    target_grid: list[list[int]] | None = None,
    reasoning_budget: dict[str, Any] | None = None,
    case_id: str = "case-0",
) -> dict[str, Any]:
    """Build a minimal generation record for eval tests.

    Args:
        generated_text: Raw generated response text.
        finish_reason: vLLM finish reason.
        target_grid: Ground-truth target grid.
        reasoning_budget: Reasoning-budget metadata.
        case_id: Case identifier.

    Returns:
        Minimal generation record.
    """
    return {
        "case_id": case_id,
        "task_id": "task-0",
        "sample_id": "sample-0",
        "test_index": 0,
        "generated_text": generated_text,
        "finish_reason": finish_reason,
        "stop_reason": None,
        "reasoning_budget": reasoning_budget or {},
        "target_grid": target_grid or [[1, 2], [3, 4]],
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to a JSONL file.

    Args:
        path: Output path.
        records: JSON-serializable records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_duplicate_case_ids_are_rejected() -> None:
    """Each query should contribute at most once to a run."""
    record = _record(generated_text="[[1]]", finish_reason="stop")
    with pytest.raises(ValueError, match="Duplicate case_id 'case-0'"):
        evaluate_records([record, record])


def test_stopped_valid_response_parses_and_scores_correct() -> None:
    records = [
        _record(
            generated_text="<think>done</think>\n\n[1 2]\n[3 4]",
            finish_reason="stop",
        )
    ]

    evaluations, summary = evaluate_records(records)

    case = evaluations[0]
    assert case.response_finished is True
    assert case.parse_ok is True
    assert case.correct is True
    assert case.predicted_grid == [[1, 2], [3, 4]]
    assert case.parse_status == "parsed_last_row_grid"
    assert summary["overall"]["n_finished"] == 1
    assert summary["overall"]["n_parse_error"] == 0
    assert summary["overall"]["accuracy"] == 1.0
    assert summary["overall"]["accuracy_finished"] == 1.0


def test_stopped_malformed_response_is_parse_error() -> None:
    records = [
        _record(
            generated_text="<think>done</think>\n\nnot a grid",
            finish_reason="stop",
        )
    ]

    evaluations, summary = evaluate_records(records)

    case = evaluations[0]
    assert case.response_finished is True
    assert case.parse_ok is False
    assert case.parse_status == "parse_failed"
    assert case.diagnostic_parse_ok is False
    assert summary["overall"]["n_parse_error"] == 1
    assert summary["parse_error_indices"] == [0]


def test_non_stop_response_is_only_diagnostically_parsed() -> None:
    records = [
        _record(
            generated_text="<think>still going</think>\n\n[1 2]\n[3 4]",
            finish_reason="length",
        )
    ]

    evaluations, summary = evaluate_records(records)

    case = evaluations[0]
    assert case.response_finished is False
    assert case.hit_output_token_budget is True
    assert case.parse_ok is False
    assert case.correct is False
    assert case.predicted_grid is None
    assert case.parse_status == "unfinished_generation"
    assert case.diagnostic_parse_ok is True
    assert case.diagnostic_parse_status == "parsed_last_row_grid"
    assert case.diagnostic_predicted_grid == [[1, 2], [3, 4]]
    assert summary["overall"]["n_unfinished"] == 1
    assert summary["overall"]["n_output_budget_hit"] == 1
    assert summary["overall"]["n_parse_error"] == 0
    assert summary["overall"]["finish_reason_counts"] == {"length": 1}
    assert summary["unfinished_indices"] == [0]


def test_reasoning_budget_hit_does_not_block_stopped_parse() -> None:
    records = [
        _record(
            generated_text="<think>done</think>\n\n[1 2]\n[3 4]",
            finish_reason="stop",
            reasoning_budget={
                "thinking_token_budget": 4,
                "reasoning_tokens": 4,
                "hit_thinking_token_budget": True,
                "forced_stop": True,
                "over_budget": False,
            },
        )
    ]

    evaluations, summary = evaluate_records(records)

    case = evaluations[0]
    assert case.parse_ok is True
    assert case.correct is True
    assert case.hit_thinking_token_budget is True
    assert case.forced_reasoning_stop is True
    assert case.reasoning_over_budget is False
    assert summary["overall"]["n_hit_thinking_token_budget"] == 1
    assert summary["overall"]["n_forced_reasoning_stop"] == 1
    assert summary["overall"]["n_reasoning_over_budget"] == 0


def test_eval_script_writes_error_case_files_separately(
    tmp_path: Path,
) -> None:
    input_jsonl = (
        tmp_path
        / "concept_arc_100"
        / "qwen35_27b"
        / "generations"
        / "seed_7"
        / "concept_arc_100_qwen35_27b_seed_7_generations.jsonl"
    )
    records = [
        _record(
            generated_text="<think>done</think>\n\nnot a grid",
            finish_reason="stop",
            case_id="parse-error",
        ),
        _record(
            generated_text="<think>still going</think>\n\n[1 2]\n[3 4]",
            finish_reason="length",
            case_id="unfinished",
        ),
        _record(
            generated_text="<think>done</think>\n\n[1 2]\n[3 0]",
            finish_reason="stop",
            case_id="model-error",
        ),
    ]
    _write_jsonl(input_jsonl, records)

    summary = evaluate_generation_file(input_jsonl, end_think_token="</think>")
    output_paths = get_eval_output_paths(input_jsonl)

    parse_errors = load_jsonl(output_paths["parse_errors"])
    unfinished = load_jsonl(output_paths["unfinished"])
    model_errors = load_jsonl(output_paths["model_errors"])

    assert summary["overall"]["n_parse_error"] == 1
    assert summary["overall"]["n_unfinished"] == 1
    assert len(parse_errors) == 1
    assert parse_errors[0]["case_id"] == "parse-error"
    assert parse_errors[0]["response_finished"] is True
    assert len(unfinished) == 1
    assert unfinished[0]["case_id"] == "unfinished"
    assert unfinished[0]["response_finished"] is False
    assert unfinished[0]["diagnostic_parse_ok"] is True
    assert len(model_errors) == 1
    assert model_errors[0]["case_id"] == "model-error"
    assert model_errors[0]["response_finished"] is True
    assert model_errors[0]["parse_ok"] is True
    assert model_errors[0]["correct"] is False


def test_moved_run_uses_its_saved_reasoning_marker(tmp_path: Path) -> None:
    """Score an archived run without its preset and write beside the copied file.

    Args:
        tmp_path (Path): Temporary location independent of the generation layout.
    """
    path = tmp_path / "moved" / "custom_generations.jsonl"
    reasoning_only = _record(
        generated_text="[1 2] [3 4] <end> no answer", finish_reason="stop"
    )
    _write_jsonl(path, [reasoning_only])
    (path.parent / "custom_config.json").write_text(
        json.dumps({"config": {"reasoning": {"reasoning_end_marker": "<end>"}}})
    )
    summary = evaluate_generation_file(path, model_config_dir=tmp_path / "missing")
    assert summary["overall"]["n_parse_error"] == 1
    assert (path.parent / "eval/custom_generations_eval_cases.jsonl").is_file()


def test_eval_script_uses_seeded_output_dir_and_names(tmp_path: Path) -> None:
    input_jsonl = (
        tmp_path
        / "concept_arc_100"
        / "qwen35_27b"
        / "generations"
        / "seed_7"
        / "concept_arc_100_qwen35_27b_seed_7_generations.jsonl"
    )

    output_paths = get_eval_output_paths(input_jsonl)

    assert output_paths["summary"].parent == (
        tmp_path.resolve() / "concept_arc_100" / "qwen35_27b" / "eval" / "seed_7"
    )
    assert output_paths["summary"].name == (
        "concept_arc_100_qwen35_27b_seed_7_generations_eval_summary.json"
    )


def test_eval_script_infers_gpt_oss_harmony_marker(tmp_path: Path) -> None:
    input_jsonl = (
        tmp_path
        / "concept_arc_100"
        / "gpt_oss_20b"
        / "generations"
        / "seed_7"
        / (
            "concept_arc_100_gpt_oss_20b_reasoning_effort_medium_seed_7"
            "_generations.jsonl"
        )
    )
    records = [
        _record(
            generated_text=(
                "<|channel|>analysis<|message|>[1 2]"
                "<|channel|>final<|message|>not a grid"
            ),
            finish_reason="stop",
            case_id="gpt-oss-parse-error",
        )
    ]
    _write_jsonl(input_jsonl, records)

    summary = evaluate_generation_file(input_jsonl)

    assert summary["overall"]["n_parse_error"] == 1
    assert summary["overall"]["parse_status_counts"] == {"parse_failed": 1}
