"""Family-balanced separation and the original separation reference."""

from __future__ import annotations

import numpy as np
from numba import njit

from arc_cd.repr_analysis.geometry import finite_mean


@njit(nogil=True, cache=True)
def score_groups(
    similarity: np.ndarray,
    membership: np.ndarray,
    anchor_tasks: np.ndarray,
    anchor_families: np.ndarray,
    edge_b: np.ndarray,
    edge_c: np.ndarray,
    offsets: np.ndarray,
    flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score distinct-parent comparisons without materializing triplet tensors.

    Args:
        similarity (np.ndarray): Pairwise similarities; larger values mean closer.
        membership (np.ndarray): Boolean parent/target-family memberships.
        anchor_tasks (np.ndarray): Parent index for each anchor membership.
        anchor_families (np.ndarray): Target family for each anchor membership.
        edge_b (np.ndarray): Layout-matched positive endpoints, grouped by family.
        edge_c (np.ndarray): Corresponding negative endpoints.
        offsets (np.ndarray): Family boundaries in edge arrays.
        flags (np.ndarray): Seven eligibility bits: layout and six visual cutoffs.

    Returns:
        tuple[np.ndarray, np.ndarray]: Group scores [condition, anchor membership,
        negative family, strict/tie-adjusted] and original scores [anchor, metric].
    """
    n, n_families = membership.shape
    n_anchors = len(anchor_tasks)
    scores = np.zeros((8, n_anchors, n_families, 2), dtype=np.float64)
    denominators = np.zeros((8, n_anchors, n_families), dtype=np.int64)
    original = np.full((n_anchors, 2), np.nan)
    for row in range(n_anchors):
        a, f = anchor_tasks[row], anchor_families[row]
        if not np.isfinite(similarity[a, a]):
            continue
        # Sorted ranks avoid enumerating every unrestricted reference pair.
        order = np.argsort(similarity[a])
        below = np.zeros(n_families, dtype=np.int64)
        below_all, positive_count, wins, ties = 0, 0, 0, 0
        start = 0
        while start < n:
            value = similarity[a, order[start]]
            end = start + 1
            while end < n and similarity[a, order[end]] == value:
                end += 1
            tied = np.zeros(n_families, dtype=np.int64)
            tied_all = 0
            if np.isfinite(value):
                for i in range(start, end):
                    c = order[i]
                    if c != a and not membership[c, f]:
                        tied_all += 1
                        for g in range(n_families):
                            if membership[c, g]:
                                tied[g] += 1
                for i in range(start, end):
                    b = order[i]
                    if b != a and membership[b, f]:
                        positive_count += 1
                        wins += below_all
                        ties += tied_all
                        for g in range(n_families):
                            scores[0, row, g, 0] += below[g]
                            scores[0, row, g, 1] += below[g] + 0.5 * tied[g]
                below += tied
                below_all += tied_all
            start = end
        total = positive_count * below_all
        if total:
            original[row, 0] = wins / total
            original[row, 1] = (wins + 0.5 * ties) / total
        denominators[0, row] = positive_count * below
        for edge in range(offsets[f], offsets[f + 1]):
            b, c = edge_b[edge], edge_c[edge]
            if b == a or c == a:
                continue
            sb, sc = similarity[a, b], similarity[a, c]
            if not np.isfinite(sb) or not np.isfinite(sc):
                continue
            win = 1.0 if sb > sc else 0.0
            adjusted = win + (0.5 if sb == sc else 0.0)
            for g in range(n_families):
                if membership[c, g]:
                    for condition in range(1, 8):
                        if flags[edge] & (1 << (condition - 1)):
                            denominators[condition, row, g] += 1
                            scores[condition, row, g, 0] += win
                            scores[condition, row, g, 1] += adjusted
    for condition in range(8):
        for row in range(n_anchors):
            for g in range(n_families):
                denominator = denominators[condition, row, g]
                if denominator:
                    scores[condition, row, g] /= denominator
                else:
                    scores[condition, row, g] = np.nan
    return scores, original


def evaluate_groups(
    similarity: np.ndarray, design: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the compiled scorer to a validated matching design.

    Args:
        similarity (np.ndarray): Pairwise similarities; larger values mean closer.
        design (dict[str, np.ndarray]): Frozen graph and target-family memberships.

    Returns:
        tuple[np.ndarray, np.ndarray]: Balanced group scores and original scores.
    """
    n = len(design["membership"])
    if similarity.shape != (n, n):
        raise ValueError("Representations and matching parents have different axes.")
    return score_groups(
        similarity,
        design["membership"],
        design["anchor_tasks"],
        design["anchor_families"],
        design["edge_b"],
        design["edge_c"],
        design["edge_offsets"],
        design["edge_flags"],
    )


def aggregate_scores(
    groups: np.ndarray,
    original: np.ndarray,
    design: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Average negative families, anchors and target families equally in that order.

    Args:
        groups (np.ndarray): [condition, anchor membership, negative family, metric].
        original (np.ndarray): Original unbalanced anchor scores.
        design (dict[str, np.ndarray]): Frozen memberships and headline masks.

    Returns:
        dict[str, np.ndarray]: Family/macro scores,
        original reference scores and valid-anchor masks for coverage summaries.
    """
    anchors, _ = finite_mean(groups, axis=2)
    n_families = design["membership"].shape[1]
    family = np.full((8, n_families, 2), np.nan)
    original_family = np.full((n_families, 2), np.nan)
    for f in range(n_families):
        selected = design["anchor_families"] == f
        family[:, f], _ = finite_mean(anchors[:, selected], axis=1)
        original_family[f], _ = finite_mean(original[selected], axis=0)
    macro = np.full((8, 2), np.nan)
    for condition in range(8):
        included = design["headline"][condition]
        if included.any():
            # An undefined required family must not silently change the population.
            macro[condition] = family[condition, included].mean(axis=0)
    included = design["original_headline"]
    original_macro = (
        original_family[included].mean(axis=0) if included.any() else np.full(2, np.nan)
    )
    return {
        "valid_anchors": np.isfinite(anchors[..., 0]),
        "original_valid_anchors": np.isfinite(original[:, 0]),
        "family_scores": family,
        "macro": macro,
        "original_family_scores": original_family,
        "original_macro": original_macro,
    }
