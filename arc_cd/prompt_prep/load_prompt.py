"""
Functions and modules to prepare and load prompts.
"""

from __future__ import annotations

from arc_cd.data_prep import Grid, GridPair
from arc_cd.data_prep.concept_arc import ConceptARCTestCase
from arc_cd.prompt_prep import PromptRecord
from arc_cd.prompt_prep.prompt_format import PromptTemplates


def _format_row(row: list[int]) -> str:
    """Format one grid row.

    Args:
        row: Grid row.

    Returns:
        Formatted row.
    """
    return "[" + " ".join(str(value) for value in row) + "]"


def format_grid(grid: Grid) -> str:
    """Format a grid using the paper-style text representation.

    Each row is represented as a bracketed, space-separated list. Rows are
    separated by spaces.

    Args:
        grid: ARC grid.

    Returns:
        Text representation of the grid.

    Example:
        ``[[2, 1, 0, 1], [0, 0, 3, 3]]`` becomes
        ``"[2 1 0 1] [0 0 3 3]"``.
    """
    return " ".join(_format_row(row) for row in grid)


def build_examples_with_char_spans(
    train: tuple[GridPair, ...],
) -> tuple[str, dict[str, list[int]]]:
    """Build examples with pair spans and final output-bracket markers.

    Args:
        train (tuple[GridPair, ...]): Training input-output grid pairs.

    Returns:
        tuple[str, dict[str, list[int]]]: Text and half-open character spans.
        Bracket markers contain exactly the final closing bracket of each
        support output grid.

    Raises:
        ValueError: If a training example lacks an output grid.
    """
    chunks: list[str] = []
    spans_by_pair: list[list[int]] = []
    support_spans: dict[str, list[int]] = {}
    cursor = 0

    for i, pair in enumerate(train, start=1):
        if pair.output is None:
            raise ValueError(f"Training example {i} is missing an output grid.")

        if chunks:
            chunks.append("\n\n")
            cursor += 2

        pair_text = (
            f"input {i}: {format_grid(pair.input)}\n\n"
            f"output {i}: {format_grid(pair.output)}"
        )

        start = cursor
        chunks.append(pair_text)
        cursor += len(pair_text)
        spans_by_pair.append([start, cursor])
        support_spans[f"support_{i}_full"] = [start, cursor]
        if pair_text.endswith("]"):
            support_spans[f"support_{i}_output_closing_bracket"] = [cursor - 1, cursor]

    if not spans_by_pair:
        raise ValueError("ARC prompts require at least one support example.")

    return "".join(chunks), {
        **support_spans,
        "first_support_full": spans_by_pair[0],
        "last_support_full": spans_by_pair[-1],
    }


def build_user_prompt_with_char_spans(
    test_case: ConceptARCTestCase,
    template: str,
) -> tuple[str, dict[str, list[int]]]:
    """Render the user prompt and support-pair character spans.

    Args:
        test_case (ConceptARCTestCase): Flattened ARC-style test case.
        template (str): User prompt template.

    Returns:
        tuple[str, dict[str, list[int]]]: Rendered user prompt and character
        spans relative to the rendered user prompt.

    Raises:
        ValueError: If the formatted examples or query input cannot be located in the prompt.
    """
    examples, example_spans = build_examples_with_char_spans(test_case.train)
    test_input = format_grid(test_case.test_input)

    user_prompt = template.format(
        examples=examples,
        query_number=len(test_case.train) + 1,
        test_input=test_input,
        concept=test_case.concept,
        sample_id=test_case.sample_id,
        task_id=test_case.task_id,
        case_id=test_case.case_id,
        test_index=test_case.test_index,
    )

    examples_start = user_prompt.find(examples)
    if examples_start < 0:
        raise ValueError("Could not locate formatted examples in user prompt.")

    prompt_spans = {
        name: [examples_start + span[0], examples_start + span[1]]
        for name, span in example_spans.items()
    }

    query_input_start = user_prompt.rfind(test_input)
    if query_input_start < 0:
        raise ValueError("Could not locate query input in user prompt.")
    prompt_spans["query_input_full"] = [
        query_input_start,
        query_input_start + len(test_input),
    ]

    return user_prompt, prompt_spans


def build_prompt_record(
    test_case: ConceptARCTestCase,
    templates: PromptTemplates,
) -> PromptRecord:
    """Build a model-ready prompt record for one test case.

    Args:
        test_case (ConceptARCTestCase): Flattened ARC-style test case.
        templates (PromptTemplates): Loaded prompt templates.

    Returns:
        PromptRecord: Prompt record containing messages and metadata.
    """
    target_text = (
        format_grid(test_case.test_output)
        if test_case.test_output is not None
        else None
    )

    user_prompt, prompt_char_spans = build_user_prompt_with_char_spans(
        test_case,
        templates.user,
    )

    return PromptRecord(
        case_id=test_case.case_id,
        task_id=test_case.task_id,
        concept=test_case.concept,
        sample_id=test_case.sample_id,
        test_index=test_case.test_index,
        source_path=str(test_case.path),
        messages=[
            {"role": "system", "content": templates.system},
            {"role": "user", "content": user_prompt},
        ],
        target_grid=test_case.test_output,
        target_text=target_text,
        metadata={
            "is_resampled_episode": False,
            "prompt_char_spans": prompt_char_spans,
        },
    )
