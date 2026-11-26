"""Helpers for generation-derived evaluation artifact names."""

from __future__ import annotations

from pathlib import Path


def infer_dataset_label(artifact_path: str | Path, model_name: str) -> str:
    """Infer the dataset/split label from an artifact filename.

    Args:
        artifact_path (str | Path): Generation-derived artifact path.
        model_name (str): Model config name embedded in the artifact name.

    Returns:
        str: Dataset/split label, such as ``concept_arc`` or ``concept_arc_100``.

    Raises:
        ValueError: If the model name cannot be found in the artifact name.
    """
    stem = Path(artifact_path).stem
    model_marker = f"_{model_name}"
    model_start = stem.find(model_marker)
    if model_start <= 0:
        raise ValueError(
            f"Could not infer dataset label from artifact path: {artifact_path}. "
            f"Expected filename to contain model name {model_name!r}."
        )

    return stem[:model_start]
