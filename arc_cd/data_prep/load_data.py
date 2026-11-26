"""
Modules to load and prepare ARC-style data.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

from arc_cd.data_prep import Grid, GridPair
from arc_cd.data_prep.concept_arc import ConceptARCTask, ConceptARCTestCase
from arc_cd.utils import resolve_directory


def _parse_grid(raw_grid: Any, location: str) -> Grid:
    """Validate and parse one ARC grid.

    Args:
        raw_grid: Raw grid loaded from JSON.
        location: Human-readable JSON location for error messages.

    Returns:
        Parsed grid.

    Raises:
        ValueError: If the grid is malformed.
    """
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ValueError(f"{location}: grid must be a non-empty list of rows.")

    if not all(isinstance(row, list) and row for row in raw_grid):
        raise ValueError(f"{location}: every grid row must be a non-empty list.")

    width = len(raw_grid[0])

    for row_index, row in enumerate(raw_grid):
        if len(row) != width:
            raise ValueError(
                f"{location}: ragged grid; row 0 has width {width}, "
                f"but row {row_index} has width {len(row)}."
            )

        for col_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{location}: expected integer at row {row_index}, "
                    f"col {col_index}; got {value!r}."
                )

            if not (0 <= value <= 9):
                raise ValueError(
                    f"{location}: expected integer at row {row_index}, "
                    f"col {col_index}; got {value!r}, but value should be within range 0 <= val <= 9."
                )

    return cast(Grid, raw_grid)


def _parse_grid_pair(
    raw_pair: Any,
    path: Path,
    split: str,
    index: int,
    require_output: bool,
) -> GridPair:
    """Parse one ARC-style input-output pair from raw JSON.

    Args:
        raw_pair: Raw pair loaded from JSON.
        path: Source JSON path.
        split: Split name, usually ``train`` or ``test``.
        index: Pair index within the split.
        require_output: Whether the pair must contain an output grid.

    Returns:
        Parsed grid pair.

    Raises:
        ValueError: If the pair is malformed.
    """
    location = f"{path}:{split}[{index}]"

    if not isinstance(raw_pair, dict):
        raise ValueError(f"{location}: expected object.")

    if "input" not in raw_pair:
        raise ValueError(f"{location}: missing 'input' field.")

    if require_output and "output" not in raw_pair:
        raise ValueError(f"{location}: missing required 'output' field.")

    input_grid = _parse_grid(raw_pair["input"], location=f"{location}.input")

    output_grid = None
    if "output" in raw_pair:
        output_grid = _parse_grid(raw_pair["output"], location=f"{location}.output")

    return GridPair(input=input_grid, output=output_grid)


def _load_arc_json(
    path: Path,
    *,
    require_test_outputs: bool,
) -> tuple[tuple[GridPair, ...], tuple[GridPair, ...]]:
    """Load train and test pairs from one ARC-style JSON file.

    Args:
        path: Source JSON path.
        require_test_outputs: Whether every test example must contain output.

    Returns:
        A ``(train, test)`` tuple of parsed grid-pair tuples.

    Raises:
        ValueError: If the JSON file has an invalid ARC task structure.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected top-level JSON object.")

    if "train" not in raw or "test" not in raw:
        raise ValueError(f"{path}: expected top-level 'train' and 'test' fields.")

    train_raw = raw["train"]
    test_raw = raw["test"]

    if not isinstance(train_raw, list):
        raise ValueError(f"{path}: 'train' must be a list.")
    if not isinstance(test_raw, list):
        raise ValueError(f"{path}: 'test' must be a list.")

    train = tuple(
        _parse_grid_pair(
            raw_pair=pair,
            path=path,
            split="train",
            index=i,
            require_output=True,
        )
        for i, pair in enumerate(train_raw)
    )

    test = tuple(
        _parse_grid_pair(
            raw_pair=pair,
            path=path,
            split="test",
            index=i,
            require_output=require_test_outputs,
        )
        for i, pair in enumerate(test_raw)
    )

    return train, test


def _load_concept_arc_task_json(
    path: Path,
    concept: str,
    *,
    require_test_outputs: bool,
) -> ConceptARCTask:
    """Load a single ConceptARC JSON file.

    Args:
        path: Source JSON path.
        concept: Concept name inferred from the parent directory.
        require_test_outputs: Whether every test example must contain output.

    Returns:
        Loaded ConceptARC task.
    """
    train, test = _load_arc_json(
        path,
        require_test_outputs=require_test_outputs,
    )

    return ConceptARCTask(
        concept=concept,
        sample_id=path.stem,
        path=path,
        train=train,
        test=test,
    )


def load_concept_arc(
    root: str | Path,
    *,
    require_test_outputs: bool = False,
) -> tuple[ConceptARCTask, ...]:
    """Load all ConceptARC tasks from a directory.

    Expects the following structure:

        ConceptARC/
            AboveBelow/
                1.json
                2.json
                ...
            Center/
                1.json
                2.json
                ...

    Args:
        root: Path to the ConceptARC root directory.
        require_test_outputs: Whether every test example must contain an
            ``output`` field.

    Returns:
        A tuple of loaded ConceptARC tasks.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` is not a directory.
        ValueError: If a JSON file has an invalid ARC task structure.
    """
    root_path = resolve_directory(root, dataset_name="ConceptARC")
    tasks: list[ConceptARCTask] = []

    for concept_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        for json_path in sorted(concept_dir.glob("*.json")):
            tasks.append(
                _load_concept_arc_task_json(
                    path=json_path,
                    concept=concept_dir.name,
                    require_test_outputs=require_test_outputs,
                )
            )

    return tuple(tasks)


def iter_test_cases(tasks: Iterable[ConceptARCTask]) -> Iterator[ConceptARCTestCase]:
    """Yield each ConceptARC test query in its original order.

    Args:
        tasks (Iterable[ConceptARCTask]): Tasks with their support and test pairs.

    Yields:
        ConceptARCTestCase: One query with its task's support examples.
    """
    for task in tasks:
        for index, pair in enumerate(task.test):
            yield ConceptARCTestCase(
                concept=task.concept,
                sample_id=task.sample_id,
                test_index=index,
                path=task.path,
                train=task.train,
                test_input=pair.input,
                test_output=pair.output,
            )
