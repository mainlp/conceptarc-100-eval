"""Model-independent visual matching and compact family-level diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from arc_cd.repr_analysis.matched_features import PROFILES, QUANTILES

CONDITIONS = (
    "frequency_balanced",
    "layout_only",
    "input_q25",
    "input_q50",
    "input_q75",
    "full_q25",
    "full_q50",
    "full_q75",
)


def build_matching_design(
    membership: np.ndarray,
    layouts: np.ndarray,
    vectors: dict[str, np.ndarray],
    min_tasks: int = 10,
    *,
    cutoffs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Freeze exact-layout edges, visual cutoffs and distinct-parent eligibility.

    Args:
        membership (np.ndarray): Boolean [parent, target family] memberships.
        layouts (np.ndarray): Exact generator layout operator per parent.
        vectors (dict[str, np.ndarray]): Prepared input/full parent vectors.
        min_tasks (int): Minimum distinct parents in each headline role.
        cutoffs (np.ndarray | None): Shared [input/full, quartile] thresholds;
            None calibrates them from this target's nearest negative matches.

    Returns:
        dict[str, np.ndarray]: Matching graph, cutoffs and family coverage.
        Edge distances are used for diagnostics and need not be persisted.
    """
    membership = np.asarray(membership, dtype=bool)
    if membership.ndim != 2 or not all(membership.shape):
        raise ValueError("Expected nonempty parent/action memberships.")
    n, families = membership.shape
    if min_tasks < 1 or layouts.shape != (n,):
        raise ValueError("Expected aligned layouts and a positive minimum.")
    if cutoffs is not None:
        cutoffs = np.asarray(cutoffs, dtype=np.float64)
        if (
            cutoffs.shape != (2, 3)
            or np.isinf(cutoffs).any()
            or (cutoffs < 0).any()
            or (np.diff(cutoffs, axis=1) < 0).any()
            or any(np.isnan(row).any() and not np.isnan(row).all() for row in cutoffs)
        ):
            raise ValueError(
                "Expected nonnegative ordered cutoffs or unavailable rows."
            )
    for profile in PROFILES:
        values = vectors[profile]
        if values.ndim != 2 or len(values) != n or not np.isfinite(values).all():
            raise ValueError(f"Invalid {profile} visual-feature matrix.")
    distances = {name: cdist(vectors[name], vectors[name]) for name in PROFILES}
    sizes = membership.sum(axis=0)
    anchor_tasks, anchor_families = np.nonzero(membership)
    offsets = [0]
    positives, negatives = [], []
    nearest: list[list[float]] = [[], []]
    for family in tqdm(
        range(families), desc="Matching families", unit="family", disable=None
    ):
        eligible = (
            membership[:, family, None]
            & ~membership[None, :, family]
            & (layouts[:, None] == layouts[None, :])
        )
        if sizes[family] < 2:
            eligible[:] = False
        b, c = np.nonzero(eligible)
        positives.extend(b.tolist())
        negatives.extend(c.tolist())
        offsets.append(len(positives))
        if cutoffs is None and sizes[family] >= min_tasks:
            for p, profile in enumerate(PROFILES):
                values = np.min(np.where(eligible, distances[profile], np.inf), axis=1)
                nearest[p].extend(values[np.isfinite(values)].tolist())
    cuts = np.full((2, 3), np.nan) if cutoffs is None else cutoffs.copy()
    if cutoffs is None:
        for p in range(2):
            if nearest[p]:
                cuts[p] = np.quantile(nearest[p], QUANTILES, method="linear")
    b, c = np.asarray(positives, dtype=np.int32), np.asarray(negatives, dtype=np.int32)
    flags = np.ones(len(b), dtype=np.uint8)
    pair_distances = np.asarray([distances[name][b, c] for name in PROFILES])
    for p in range(2):
        for q in range(3):
            flags |= (pair_distances[p] < cuts[p, q]).astype(np.uint8) << (
                1 + 3 * p + q
            )
    role_counts = np.zeros((8, families, 3), dtype=np.int64)
    pair_counts = np.zeros((8, families), dtype=np.int64)
    tagged = membership.any(axis=1)
    for family in range(families):
        members = np.flatnonzero(membership[:, family])
        if len(members) < 2:
            continue
        negative_count = np.count_nonzero(~membership[:, family] & tagged)
        if negative_count:
            role_counts[0, family] = (len(members), len(members), negative_count)
            pair_counts[0, family] = len(members) * negative_count
        sl = slice(offsets[family], offsets[family + 1])
        for condition in range(1, len(CONDITIONS)):
            keep = ((flags[sl] & (1 << (condition - 1))) != 0) & tagged[c[sl]]
            bb, cc = b[sl][keep], c[sl][keep]
            if len(bb):
                valid_anchors = len(bb) - np.bincount(bb, minlength=n)[members] > 0
                role_counts[condition, family] = (
                    valid_anchors.sum(),
                    len(np.unique(bb)),
                    len(np.unique(cc)),
                )
                pair_counts[condition, family] = len(bb)
    return {
        "membership": membership,
        "anchor_tasks": anchor_tasks.astype(np.int32),
        "anchor_families": anchor_families.astype(np.int32),
        "edge_b": b,
        "edge_c": c,
        "edge_offsets": np.asarray(offsets, dtype=np.int64),
        "edge_flags": flags,
        "edge_visual_distances": pair_distances,
        "cutoffs": cuts,
        "role_counts": role_counts,
        "pair_counts": pair_counts,
        "headline": (role_counts >= min_tasks).all(axis=2),
        "original_headline": sizes >= min_tasks,
    }


def matching_summary(
    design: dict[str, np.ndarray], families: list[str], *, family_key: str = "actions"
) -> dict[str, Any]:
    """Summarize coverage and B-C visual distances once per condition and family.

    Args:
        design (dict[str, np.ndarray]): Matching graph with visual distances.
        families (list[str]): Target names in membership order.
        family_key (str): Key for per-family coverage and distances.

    Returns:
        dict[str, Any]: Coverage and distances grouped by condition and family,
        with the cutoff recorded once per condition.
        Each distance summary weights qualifying B-C pairs equally.
    """
    summary = {}
    for condition, name in enumerate(CONDITIONS):
        entry: dict[str, Any] = {}
        if condition >= 2:
            entry["cutoff"] = design["cutoffs"][
                (condition - 2) // 3, (condition - 2) % 3
            ]
        entry[family_key] = {}
        for family, action in enumerate(families):
            row = {
                "headline": bool(design["headline"][condition, family]),
                **dict(
                    zip(
                        ("anchors", "positives", "negatives"),
                        design["role_counts"][condition, family],
                    )
                ),
                "reference_pairs": int(design["pair_counts"][condition, family]),
            }
            if condition:
                sl = slice(*design["edge_offsets"][family : family + 2])
                keep = (design["edge_flags"][sl] & (1 << (condition - 1))) != 0
                keep &= design["membership"][design["edge_c"][sl]].any(axis=1)
                for p, profile in enumerate(PROFILES):
                    values = design["edge_visual_distances"][p, sl][keep]
                    for quantile, label in ((0.5, "median"), (0.9, "p90")):
                        row[f"{profile}_distance_{label}"] = (
                            float(np.quantile(values, quantile))
                            if len(values)
                            else np.nan
                        )
            entry[family_key][action] = row
        summary[name] = entry
    return summary
