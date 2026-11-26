"""Support-grid controls with explicit, stable input and full feature profiles."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from arc_cd.data_prep import Grid
from arc_cd.data_prep.concept_arc import ConceptARCTask
from arc_cd.data_prep.load_data import load_concept_arc
from arc_cd.eval.visualize import grid_pair_to_example
from arc_cd.repr_analysis.explorer import pack_support_grid
from arc_cd.repr_analysis.visual_features import (
    pair_feature_layout,
    preprocess_task_features,
)
from arc_cd.repr_analysis.visual_runner import extract_visual_features

PROFILES = ("input", "full")
QUANTILES = (0.25, 0.5, 0.75)


def validate_support_examples(
    metadata_dir: Path, versions: Iterable[tuple[str, list[dict[str, Grid]]]]
) -> None:
    """Check the actual support grids against the frozen explorer snapshot.

    Args:
        metadata_dir (Path): Metadata explorer published for this population.
        versions (Iterable[tuple[str, list[dict[str, Grid]]]]): Version IDs and
            ordered support pairs from the dataset or feature manifest.
    """
    snapshot = json.loads((metadata_dir / "supports.json").read_text())
    expected = {row["task_id"]: row["supports"] for row in snapshot}
    if len(expected) != len(snapshot):
        raise ValueError("Duplicate task version in the metadata support snapshot.")
    for task_id, pairs in versions:
        grids = [
            {side: pack_support_grid(grid, task_id) for side, grid in pair.items()}
            for pair in pairs
        ]
        if expected.pop(task_id, None) != grids:
            raise ValueError(
                f"Support grids differ from frozen metadata for {task_id}."
            )
    if expected:
        raise ValueError("Some task versions in the metadata have no support grids.")


def load_matching_tasks(
    data_dir: Path, metadata_dir: Path
) -> tuple[ConceptARCTask, ...]:
    """Load the same support pairs used to extract the task representations.

    Args:
        data_dir (Path): Eight-version task dataset.
        metadata_dir (Path): Frozen metadata explorer with the extracted support grids.

    Returns:
        tuple[ConceptARCTask, ...]: Tasks with validated support examples.
    """
    tasks = load_concept_arc(data_dir)
    validate_support_examples(
        metadata_dir,
        (
            (task.task_id, [grid_pair_to_example(pair) for pair in task.train])
            for task in tasks
        ),
    )
    return tasks


def extract_controls(tasks: Sequence[ConceptARCTask]) -> dict[str, np.ndarray]:
    """Extract ordered supports and freeze preprocessing across all parents.

    Args:
        tasks (Sequence[ConceptARCTask]): Eight versions of every parent task.

    Returns:
        dict[str, np.ndarray]: Parent identities, raw/prepared profile vectors
        and their coordinate names and preprocessing parameters.
    """
    source = extract_visual_features(
        tasks, "concat", "both", ("geometry", "color", "pixel_change")
    )
    original_names = source["pair_feature_names"].tolist()
    original = source["pair_features"]
    grid_names, grid_blocks = pair_feature_layout("both", ("geometry", "color"))
    names = [*grid_names, "changed_count", "changed_fraction"]
    pairs = np.concatenate(
        (
            original[..., [original_names.index(name) for name in grid_names]],
            original[..., original_names.index("changed_count"), None],
            (
                original[..., original_names.index("changed_count")]
                / original[..., original_names.index("input_area")]
            )[..., None],
        ),
        axis=-1,
    )
    blocks = [*grid_blocks, "pixel_change", "pixel_change"]
    result = {
        key: source[key]
        for key in (
            "source_task_id",
            "concept",
            "version_task_ids",
            "support_set_ids",
        )
    }
    for profile in PROFILES:
        columns = (
            [i for i, name in enumerate(names) if name.startswith("input_")]
            if profile == "input"
            else list(range(len(names)))
        )
        pair_values = pairs[..., columns]
        version_values = pair_values.reshape(len(pairs), 8, -1)
        task_values = version_values.mean(axis=1)
        feature_names = tuple(
            f"support_{support}/{names[column]}"
            for support in (1, 2, 3)
            for column in columns
        )
        feature_blocks = tuple(blocks[column] for _ in range(3) for column in columns)
        prepared = preprocess_task_features(task_values, feature_blocks)
        result.update(
            {
                f"{profile}_task_features": task_values,
                f"{profile}_feature_names": np.asarray(feature_names),
                f"{profile}_feature_blocks": np.asarray(feature_blocks),
                **{f"{profile}_{key}": value for key, value in prepared.items()},
            }
        )
    return result
