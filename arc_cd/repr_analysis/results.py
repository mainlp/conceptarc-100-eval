"""Traceable result metadata and disk-backed derived representations."""

from __future__ import annotations

import json
import platform
from contextlib import ExitStack
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from arc_cd.repr_analysis.data import FeatureBundle
from arc_cd.repr_analysis.geometry import DistanceMetric
from arc_cd.repr_analysis.population import AnalysisPopulation


def write_json(path: Path, value: Any) -> None:
    """Write portable JSON, using null for undefined numerical values.

    Args:
        path (Path): Destination file.
        value (Any): JSON-compatible data, optionally containing NumPy values.
    """

    def convert(item: Any) -> Any:
        """Convert numerical containers recursively to strict JSON values.

        Args:
            item (Any): Nested value.

        Returns:
            Any: JSON-compatible value.
        """
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.generic):
            return convert(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        return item

    path.write_text(
        json.dumps(convert(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def software_versions() -> dict[str, str | None]:
    """Record the numerical and plotting software used by the analysis.

    Returns:
        dict[str, str | None]: Python and installed package versions.
    """
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in (
        "numpy",
        "scipy",
        "matplotlib",
        "pillow",
        "threadpoolctl",
        "scikit-learn",
        "umap-learn",
        "numba",
        "conceptarc-100-eval",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def save_axes(
    stage: Path,
    bundle: FeatureBundle,
    positions: list[str],
    metrics: tuple[str, ...],
    *,
    population: AnalysisPopulation | None = None,
    distance_metric: DistanceMetric | None = None,
) -> None:
    """Save identities and axis contracts independent of numerical arrays.

    Args:
        stage (Path): Analysis staging directory.
        bundle (FeatureBundle): Validated source grouping.
        positions (list[str]): Selected positions in order.
        metrics (tuple[str, ...]): Ordered names in score arrays.
        population (AnalysisPopulation | None): Parents selected from this bundle,
            or None to use all parents.
        distance_metric (DistanceMetric | None): Separation metric, when applicable.

    Raises:
        ValueError: If the population refers to a different source bundle.
    """
    if population is not None and population.source is not bundle:
        raise ValueError("Population must refer to the supplied feature bundle.")
    selected: AnalysisPopulation | FeatureBundle = (
        population if population is not None else bundle
    )
    records, rows = selected.records, selected.rows
    write_json(
        stage / "axes.json",
        {
            "positions": positions,
            "layers": bundle.layers,
            "concepts": selected.concepts,
            "task_ids": selected.task_ids,
            "labels": selected.labels,
            "metrics": metrics,
            **(
                {"distance_metric": distance_metric}
                if distance_metric is not None
                else {}
            ),
            "version_rows": rows,
            **(
                {
                    "population": population.snapshot(),
                    "source_parent_indices": population.parent_indices,
                    "source_version_rows": population.source_rows,
                    "source_record_indices": population.record_indices,
                }
                if population is not None
                else {}
            ),
            "version_task_ids": [
                [records[row]["task_id"] for row in group] for group in rows
            ],
            "support_set_ids": [
                [records[row]["metadata"]["support_set_id"] for row in group]
                for group in rows
            ],
            "case_ids": [[records[row]["case_id"] for row in group] for group in rows],
            "array_axes": {
                "task_scores": ["position", "layer", "task", "metric"],
                "concept_mean": ["position", "layer", "concept", "metric"],
                "macro": ["position", "layer", "metric"],
                "task_means.npy": ["task", "layer", "hidden"],
                "grand_means.npy": ["layer", "hidden"],
                "concept_means.npy": ["concept", "layer", "hidden"],
            },
        },
    )


def representation_arrays(
    stage: Path,
    position: str,
    bundle: FeatureBundle,
    hidden_size: int,
    resources: ExitStack,
    *,
    population: AnalysisPopulation | None = None,
) -> dict[str, np.memmap]:
    """Allocate FP32 derived means without retaining all layers in RAM.

    Args:
        stage (Path): Output staging directory.
        position (str): Support position.
        bundle (FeatureBundle): Validated identities and layers.
        hidden_size (int): Model hidden dimension.
        resources (ExitStack): Close mappings before staging cleanup on failure.
        population (AnalysisPopulation | None): Parents selected from this bundle,
            or None to use all parents.

    Returns:
        dict[str, np.memmap]: Writable grand/task/concept mean arrays.

    Raises:
        ValueError: If the population refers to a different source bundle.
    """
    if population is not None and population.source is not bundle:
        raise ValueError("Population must refer to the supplied feature bundle.")
    selected: AnalysisPopulation | FeatureBundle = (
        population if population is not None else bundle
    )
    directory = stage / position
    directory.mkdir()
    shapes = {
        "grand_means": (len(bundle.layers), hidden_size),
        "task_means": (len(selected.task_ids), len(bundle.layers), hidden_size),
        "concept_means": (len(selected.concepts), len(bundle.layers), hidden_size),
    }
    arrays: dict[str, np.memmap] = {}
    for name, shape in shapes.items():
        array = np.lib.format.open_memmap(
            directory / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape
        )
        resources.callback(array._mmap.close)
        arrays[name] = array
    return arrays
