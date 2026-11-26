"""Float64 task geometry and balanced task-level summaries."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist

DistanceMetric = Literal["cosine", "euclidean"]
DISTANCE_METRICS = ("cosine", "euclidean")


def separation_similarity(
    vectors: np.ndarray, distance_metric: DistanceMetric = "cosine"
) -> np.ndarray:
    """Compute pairwise scores where larger values mean closer task vectors.

    Args:
        vectors (np.ndarray): Finite [tasks, features] representations, already
            preprocessed. An empty feature axis is allowed for constant data.
        distance_metric (DistanceMetric): Cosine similarity or negative Euclidean
            distance. Euclidean uses vector magnitudes without row normalization.

    Returns:
        np.ndarray: Float64 [task, task] scores. Zero vectors give undefined
        cosines but valid Euclidean distances, including all-zero distances when
        every feature is constant. Euclidean uses direct coordinate differences
        to avoid cancellation when subtracting large squared vector norms.
    """
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite task vectors with a nonempty task axis.")
    if distance_metric == "cosine":
        return cosine_from_gram(values @ values.T)
    if distance_metric == "euclidean":
        return -cdist(values, values, metric="euclidean")
    raise ValueError(f"Unknown distance metric: {distance_metric}")


def cosine_from_gram(gram: np.ndarray) -> np.ndarray:
    """Convert a square dot-product matrix to cosine, preserving undefined pairs.

    Args:
        gram (np.ndarray): Square float64 Gram matrix.

    Returns:
        np.ndarray: Cosines, with NaN for zero-norm vectors.
    """
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = norms[:, None] * norms[None, :]
    return np.clip(
        np.divide(
            gram, denominator, out=np.full_like(gram, np.nan), where=denominator > 0
        ),
        -1.0,
        1.0,
    )


def finite_mean(values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Average finite entries and return valid comparison counts.

    Args:
        values (np.ndarray): Values containing possible undefined comparisons.
        axis (int): Axis to average.

    Returns:
        tuple[np.ndarray, np.ndarray]: Means and valid counts.
    """
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0).sum(axis=axis)
    return (
        np.divide(
            total,
            count,
            out=np.full_like(total, np.nan, dtype=np.float64),
            where=count > 0,
        ),
        count,
    )


def summarize(
    scores: np.ndarray, labels: np.ndarray, n_concepts: int
) -> dict[str, np.ndarray]:
    """Aggregate task scores within concepts, then equally over concepts.

    Args:
        scores (np.ndarray): [tasks, metrics] scores.
        labels (np.ndarray): Integer concept indices.
        n_concepts (int): Number of concepts in the saved vocabulary.

    Returns:
        dict[str, np.ndarray]: Concept means, valid task counts, macro means,
        and valid concept counts. Macro means require every concept to be valid.
    """
    means = np.full((n_concepts, scores.shape[1]), np.nan)
    valid_counts = np.zeros_like(means, dtype=np.int64)
    for concept in range(n_concepts):
        means[concept], valid_counts[concept] = finite_mean(
            scores[labels == concept], axis=0
        )
    return {
        "concept_mean": means,
        "concept_valid_tasks": valid_counts,
        "macro": means.mean(axis=0),
        "valid_concepts": np.isfinite(means).sum(axis=0),
    }
