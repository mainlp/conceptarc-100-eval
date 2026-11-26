"""Exact same-concept versus different-concept separation by distance ranks."""

from __future__ import annotations

import numpy as np

METRICS = ("strict_win_probability", "tie_adjusted_probability")


def separation_probability(
    similarity: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute triplet ranks using distinct anchor, positive, and negative parents.

    Args:
        similarity (np.ndarray): Pairwise scores with larger values indicating
            closer tasks; cosine similarities or negated Euclidean distances.
        labels (np.ndarray): Concept index per task.

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]]: Per-anchor strict and half-tie
        probabilities, eligible pair counts, and tie probabilities.
    """
    scores = np.full((len(labels), 2), np.nan)
    pair_counts = np.zeros(len(labels), dtype=np.int64)
    tie_probability = np.full(len(labels), np.nan)
    for anchor in range(len(labels)):
        eligible = np.isfinite(similarity[anchor])
        eligible[anchor] = False
        same = similarity[anchor, eligible & (labels == labels[anchor])]
        other = similarity[anchor, eligible & (labels != labels[anchor])]
        denominator = same.size * other.size
        pair_counts[anchor] = denominator
        if denominator == 0:
            continue
        other.sort()
        left = np.searchsorted(other, same, side="left")
        right = np.searchsorted(other, same, side="right")
        wins = left.sum()
        ties = (right - left).sum()
        scores[anchor] = (wins / denominator, (wins + 0.5 * ties) / denominator)
        tie_probability[anchor] = ties / denominator
    return scores, {
        "valid_triplet_comparisons": pair_counts,
        "tie_probability": tie_probability,
    }
