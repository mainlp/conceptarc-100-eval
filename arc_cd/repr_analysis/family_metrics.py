"""Overlapping-family metrics on pairwise task similarities."""

from __future__ import annotations

import numpy as np

from arc_cd.repr_analysis.geometry import finite_mean

SEPARATION_METRICS = ("strict_win", "tie_adjusted")
REASONS = {
    "not_member": 1,
    "undefined_anchor_cosine": 2,
    "no_positive_reference": 4,
    "no_negative_reference": 8,
    "concept_reference_not_applicable": 32,
}


def eligible_references(
    similarity: np.ndarray, labels: np.ndarray, scope: str
) -> np.ndarray:
    """Select finite leave-task-out references in the requested population.

    Args:
        similarity (np.ndarray): Square task similarity matrix; larger is closer.
        labels (np.ndarray): Original concept labels.
        scope (str): global or cross_concept.

    Returns:
        np.ndarray: Boolean [anchor, candidate] eligibility.
    """
    if scope not in ("global", "cross_concept"):
        raise ValueError(f"Unknown comparison scope: {scope}")
    eligible = np.isfinite(similarity)
    np.fill_diagonal(eligible, False)
    if scope == "cross_concept":
        eligible &= labels[:, None] != labels[None, :]
    return eligible


def reference_counts(
    similarity: np.ndarray, eligible: np.ndarray, membership: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Count positive/negative finite references and encode missing-score reasons.

    Args:
        similarity (np.ndarray): Task similarity matrix; nonfinite diagonal values
            mark undefined anchors using the legacy undefined_anchor_cosine bit.
        eligible (np.ndarray): Leave-task-out candidate mask.
        membership (np.ndarray): Boolean [task, family] tags.

    Returns:
        tuple[np.ndarray, np.ndarray]: [task, family, positive/negative] counts
        and uint8 reason bitmasks. Zero means a valid member anchor.
    """
    positive = eligible.astype(np.int64) @ membership.astype(np.int64)
    negative = eligible.sum(axis=1)[:, None] - positive
    counts = np.stack((positive, negative), axis=-1)
    reason = np.where(membership, 0, REASONS["not_member"]).astype(np.uint8)
    reason |= (
        (~np.isfinite(np.diag(similarity)))[:, None]
        * REASONS["undefined_anchor_cosine"]
    ).astype(np.uint8)
    reason |= ((positive == 0) * REASONS["no_positive_reference"]).astype(np.uint8)
    reason |= ((negative == 0) * REASONS["no_negative_reference"]).astype(np.uint8)
    return counts, reason


def family_separation(
    similarity: np.ndarray, eligible: np.ndarray, membership: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute exact strict/tie-adjusted separation using shared sorted ranks.

    Args:
        similarity (np.ndarray): Float64 task similarities, with larger values
            meaning closer vectors (cosines or negative Euclidean distances).
        eligible (np.ndarray): Finite candidate mask after scope restrictions.
        membership (np.ndarray): Overlapping binary [task, family] memberships.

    Returns:
        dict[str, np.ndarray]: Scores/reasons, eligible counts, exact strict/tie
        pair counts, and comparison denominators aligned to member anchors.
    """
    counts, base_reason = reference_counts(similarity, eligible, membership)
    shape = membership.shape
    wins = np.zeros(shape, dtype=np.int64)
    ties = np.zeros(shape, dtype=np.int64)
    for anchor in range(len(similarity)):
        candidates = np.flatnonzero(eligible[anchor])
        candidates = candidates[
            np.argsort(similarity[anchor, candidates], kind="stable")
        ]
        values = similarity[anchor, candidates]
        tags = membership[candidates].astype(np.int64)
        negative_prefix = np.vstack(
            (np.zeros(shape[1], dtype=np.int64), np.cumsum(1 - tags, axis=0))
        )
        lower = np.searchsorted(values, values, side="left")
        upper = np.searchsorted(values, values, side="right")
        wins[anchor] = (tags * negative_prefix[lower]).sum(axis=0)
        ties[anchor] = (tags * (negative_prefix[upper] - negative_prefix[lower])).sum(
            axis=0
        )
    comparisons = counts[..., 0] * counts[..., 1]
    valid = base_reason == 0
    scores = np.full((*shape, 2), np.nan)
    np.divide(wins, comparisons, out=scores[..., 0], where=valid)
    np.divide(wins + 0.5 * ties, comparisons, out=scores[..., 1], where=valid)
    return {
        "scores": scores,
        "reasons": np.repeat(base_reason[..., None], 2, axis=-1),
        "reference_counts": counts,
        "strict_win_counts": np.where(valid, wins, 0),
        "tie_counts": np.where(valid, ties, 0),
        "comparison_counts": np.where(valid, comparisons, 0),
    }


def aggregate_families(
    scores: np.ndarray,
    membership: np.ndarray,
    roles: list[str],
    min_tasks: int,
    role_order: list[str] | None = None,
    *,
    included: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Average anchors within families, then equally over eligible families.

    Args:
        scores (np.ndarray): [task, family, metric] scores with missing entries.
        membership (np.ndarray): Binary task memberships.
        roles (list[str]): Role per family, in column order.
        min_tasks (int): Minimum parent count for metadata headline summaries.
        role_order (list[str] | None): Explicit output roles, including unmapped roles.
        included (np.ndarray | None): Optional Boolean family mask for headline means;
            all per-family statistics remain available.

    Returns:
        dict[str, np.ndarray]: Family means/counts and role macros/counts. Role
        order is first occurrence in roles; concept references retain all groups.
    """
    means, valid = finite_mean(scores, axis=0)
    sizes = membership.sum(axis=0)
    eligible = (sizes >= min_tasks) | (np.asarray(roles) == "concept")
    if included is not None:
        if included.dtype != bool or included.shape != sizes.shape:
            raise ValueError(
                "Headline inclusion mask must be Boolean and match the family axis."
            )
        eligible &= included
    vocabulary = role_order if role_order is not None else list(dict.fromkeys(roles))
    macro = np.full((len(vocabulary), scores.shape[-1]), np.nan)
    valid_families = np.zeros(macro.shape, dtype=np.int64)
    for index, role in enumerate(vocabulary):
        selected = (np.asarray(roles) == role) & eligible
        macro[index], valid_families[index] = finite_mean(means[selected], axis=0)
    return {
        "family_mean": means,
        "family_valid_anchors": valid,
        "macro": macro,
        "macro_valid_families": valid_families,
    }
