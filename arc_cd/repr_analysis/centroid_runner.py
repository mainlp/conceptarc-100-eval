"""Family-separation analysis of task centroids without synthetic layers."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from arc_cd.probes.prompt_feature_io import staged_output_directory
from arc_cd.repr_analysis.centroid_data import (
    CONCEPT_ALIASES,
    TaskCentroids,
    load_task_centroids,
)
from arc_cd.repr_analysis.data import fingerprint
from arc_cd.repr_analysis.family_data import family_comparison_mask
from arc_cd.repr_analysis.family_metrics import (
    REASONS,
    SEPARATION_METRICS,
    aggregate_families,
    eligible_references,
    family_separation,
)
from arc_cd.repr_analysis.family_runner import _coverage, _write_csv
from arc_cd.repr_analysis.geometry import (
    DISTANCE_METRICS,
    DistanceMetric,
    separation_similarity,
)
from arc_cd.repr_analysis.results import software_versions, write_json


def write_centroid_separation(
    stage: Path,
    data: TaskCentroids,
    similarity: np.ndarray,
    *,
    population_name: str,
    comparison_scopes: list[str],
    min_family_tasks: int,
    include_structural_selectors: bool,
    distance_metric: DistanceMetric = "cosine",
    extra_axes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save shared family-separation scores for one representation per parent.

    Args:
        stage (Path): Existing output directory.
        data (TaskCentroids): Aligned parent identities and frozen family labels.
        similarity (np.ndarray): Parent similarity matrix; larger values mean
            closer vectors. Cosine uses cosines, Euclidean uses negated distances.
        population_name (str): Selected population name.
        comparison_scopes (list[str]): Global and/or cross-concept references.
        min_family_tasks (int): Minimum family size for headline statistics.
        include_structural_selectors (bool): Include structural selector families.
        distance_metric (DistanceMetric): Metric describing the supplied matrix.
        extra_axes (dict[str, Any] | None): Additional representation descriptions;
            array_axes entries update individual array descriptions.

    Returns:
        dict[str, Any]: Coverage information saved alongside the statistics.
    """
    if distance_metric not in DISTANCE_METRICS:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    reason_bits = {
        (
            "undefined_anchor_distance"
            if distance_metric == "euclidean" and name == "undefined_anchor_cosine"
            else name
        ): bit
        for name, bit in REASONS.items()
    }
    families = data.families
    included = family_comparison_mask(
        families, include_structural_selectors=include_structural_selectors
    )
    sizes = families.membership.sum(axis=0)
    headline = included & (
        (sizes >= min_family_tasks) | (np.asarray(families.roles) == "concept")
    )
    exclusions = [
        (
            f"structural_{role}"
            if not included[i]
            else "below_min_family_tasks" if not headline[i] else ""
        )
        for i, role in enumerate(families.roles)
    ]
    coverage = _coverage(families, min_family_tasks, included)
    population = {
        "name": population_name,
        "concepts": data.concepts,
        "num_parents": len(data.task_ids),
        "num_support_sets": int(data.instance_counts.sum()),
        "source_num_parents": data.source_num_tasks,
    }
    axes = {
        "task_ids": data.task_ids,
        "source_task_ids": data.source_task_ids,
        "source_rules": data.source_rules,
        "source_rows": data.source_rows,
        "instance_counts": data.instance_counts,
        "concepts": data.concepts,
        "labels": data.labels,
        "population": population,
        "metrics": SEPARATION_METRICS,
        "distance_metric": distance_metric,
        "scopes": comparison_scopes,
        "roles": families.role_order,
        "family_ids": families.ids,
        "family_names": families.names,
        "family_roles": families.roles,
        "family_sizes": sizes,
        "comparison_included": included,
        "headline_eligible": headline,
        "headline_exclusion_reasons": exclusions,
        "reason_bits": reason_bits,
        "array_axes": {
            "task_means.npy": ["task", "hidden"],
            "grand_mean.npy": ["hidden"],
            "memberships.npy": ["task", "family"],
            "macro": ["scope", "role", "metric"],
            "macro_valid_families": ["scope", "role", "metric"],
            "family_mean": ["scope", "family", "metric"],
            "family_valid_anchors": ["scope", "family", "metric"],
            "scores": ["scope", "task", "family", "metric"],
            "reasons": ["scope", "task", "family", "metric"],
            "reference_counts": ["scope", "task", "family", "positive_negative"],
            "comparison_counts": ["scope", "task", "family"],
            "strict_win_counts": ["scope", "task", "family"],
            "tie_counts": ["scope", "task", "family"],
        },
    }
    if extra_axes:
        if (extra_axes.keys() & axes.keys()) - {"array_axes"}:
            raise ValueError("Extra axes cannot replace shared centroid axes.")
        axes["array_axes"].update(extra_axes.get("array_axes", {}))
        axes.update(
            {key: value for key, value in extra_axes.items() if key != "array_axes"}
        )
    np.save(stage / "memberships.npy", families.membership)
    write_json(stage / "axes.json", axes)
    write_json(stage / "coverage.json", coverage)
    results: dict[str, list[np.ndarray]] = {}
    details: dict[str, list[np.ndarray]] = {}
    macro_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for scope in comparison_scopes:
        eligible = eligible_references(similarity, data.labels, scope)
        output = family_separation(similarity, eligible, families.membership)
        if scope == "cross_concept":
            concepts = np.asarray(families.roles) == "concept"
            output["scores"][:, concepts] = np.nan
            output["reasons"][:, concepts] |= REASONS[
                "concept_reference_not_applicable"
            ]
        summary = aggregate_families(
            output["scores"],
            families.membership,
            families.roles,
            min_family_tasks,
            families.role_order,
            included=included,
        )
        for key, values in summary.items():
            results.setdefault(key, []).append(values)
        for key, values in output.items():
            details.setdefault(key, []).append(values)
        for f, family in enumerate(families.ids):
            for m, metric in enumerate(SEPARATION_METRICS):
                reason = output["reasons"][:, f, m]
                members = families.membership[:, f]
                family_rows.append(
                    {
                        "scope": scope,
                        "family": family,
                        "metric": metric,
                        "value": summary["family_mean"][f, m],
                        "parent_count": int(sizes[f]),
                        "comparison_included": bool(included[f]),
                        "headline_eligible": bool(headline[f]),
                        "headline_exclusion_reason": exclusions[f],
                        "valid_anchors": int(summary["family_valid_anchors"][f, m]),
                        **{
                            name: int(((reason & bit != 0) & members).sum())
                            for name, bit in reason_bits.items()
                            if name != "not_member"
                        },
                    }
                )
        for r, role in enumerate(families.role_order):
            selected = (np.asarray(families.roles) == role) & headline
            for m, metric in enumerate(SEPARATION_METRICS):
                valid = np.isfinite(output["scores"][:, selected, m])
                macro_rows.append(
                    {
                        "scope": scope,
                        "role": role,
                        "metric": metric,
                        "value": summary["macro"][r, m],
                        "valid_families": int(summary["macro_valid_families"][r, m]),
                        "eligible_families": int(selected.sum()),
                        "all_families": int(sum(np.asarray(families.roles) == role)),
                        "valid_anchor_memberships": int(valid.sum()),
                        "valid_unique_anchors": int(valid.any(axis=1).sum()),
                        "eligible_unique_anchors": int(
                            families.membership[:, selected].any(axis=1).sum()
                        ),
                        "total_parents": len(data.task_ids),
                    }
                )
        logging.info(
            "Centroid %s distance: %s (%d tasks)",
            distance_metric,
            scope,
            len(data.task_ids),
        )
    np.savez_compressed(
        stage / "metrics.npz",
        allow_pickle=False,
        **{key: np.stack(values) for key, values in results.items()},
    )
    np.savez_compressed(
        stage / "scores.npz",
        allow_pickle=False,
        **{key: np.stack(values) for key, values in details.items()},
    )
    _write_csv(stage / "summary.csv", macro_rows)
    _write_csv(stage / "family_summary.csv", family_rows)
    write_json(stage / "summary.json", {"coverage": coverage, "rows": macro_rows})
    return coverage


def run_centroid_analysis(args: argparse.Namespace) -> Path:
    """Publish family-separation statistics for one centroid per task.

    Args:
        args (argparse.Namespace): Validated separation CLI arguments.

    Returns:
        Path: Published analysis directory with numerical results and PNG/PDF plots.
    """
    started = time.perf_counter()
    analysis = f"{args.distance_metric}_distance"
    source = args.centroid_file.expanduser().resolve()
    metadata_dir = args.metadata_dir.expanduser().resolve()
    metadata_paths = [
        metadata_dir / name
        for name in (
            "run_config.json",
            "axes.json",
            "task_metadata.json",
            "taxonomy.json",
        )
    ]
    stats = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in [source, *metadata_paths]
    }
    source_fingerprint = fingerprint(source)
    metadata_fingerprints = {path.name: fingerprint(path) for path in metadata_paths}
    data = load_task_centroids(source, metadata_dir, args.family_roles, args.population)
    families = data.families
    population = {
        "name": args.population,
        "concepts": data.concepts,
        "num_parents": len(data.task_ids),
        "num_support_sets": int(data.instance_counts.sum()),
        "source_num_parents": data.source_num_tasks,
    }
    destination = (
        args.output_root.expanduser().resolve()
        / data.dataset_id
        / args.model_name
        / "repr_analysis"
        / args.run_name
        / args.population
        / "metadata"
        / analysis
    )
    config = {
        "schema_version": 1,
        "analysis": analysis,
        "distance_metric": args.distance_metric,
        "label_mode": "metadata",
        "input_kind": "task_centroids",
        "model_name": args.model_name,
        "dataset_id": data.dataset_id,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "population": population,
        "centroid_file": str(source),
        "centroid_fingerprint": source_fingerprint,
        "metadata_fingerprints": metadata_fingerprints,
        "taxonomy_version": families.taxonomy["version"],
        "concept_aliases": CONCEPT_ALIASES,
        "software": software_versions(),
        "conventions": {
            "representation": "input task centroids over eight three-example support sets; no layer or support-position axis",
            "support_set_provenance": "producer-confirmed matching support sets; archive contains counts but no per-support-set IDs or vectors",
            "centering": "subtract unweighted centroid mean over the selected population",
            "normalization": (
                "none; Euclidean distance between task means without unit normalization or feature scaling"
                if args.distance_metric == "euclidean"
                else "only for cosine, after centering input centroids"
            ),
            "calculation_dtype": "float64",
            "global": "finite references from the selected population, excluding the anchor task",
            "cross_concept": "finite references excluding the anchor's entire original concept",
            "comparisons": "all eligible positive-negative pairs per member task and family",
            "ties": "strict wins and half-credit-for-ties scores both retained",
            "aggregation": "mean valid anchors per family, then equal mean over eligible families; all concept categories retained",
            "structural_actions": "excluded from headline means; retained in detailed results",
            "structural_selectors": "excluded from headline means unless explicitly enabled; retained in detailed results",
            "missing": "NaN/null with shared reason bitmasks; no synthetic values for undefined comparisons",
            "interpretation": "descriptive alignment, not an independent or causal explanation",
            "bootstrap": False,
        },
    }
    with (
        staged_output_directory(destination) as stage,
        threadpool_limits(limits=args.num_threads),
    ):
        grand_mean = data.embeddings.mean(axis=0, dtype=np.float64)
        means = data.embeddings - grand_mean
        similarity = separation_similarity(means, args.distance_metric)
        np.save(stage / "task_means.npy", means)
        np.save(stage / "grand_mean.npy", grand_mean)
        write_json(stage / "task_metadata.json", families.metadata)
        write_json(stage / "metadata_source_config.json", families.config)
        shutil.copyfile(metadata_dir / "taxonomy.json", stage / "taxonomy.json")
        config["coverage"] = write_centroid_separation(
            stage,
            data,
            similarity,
            population_name=args.population,
            comparison_scopes=args.comparison_scopes,
            min_family_tasks=args.min_family_tasks,
            include_structural_selectors=args.include_structural_selectors,
            distance_metric=args.distance_metric,
            extra_axes={
                "representation_label": f"Task centroids · {args.distance_metric} distance",
            },
        )
        config["elapsed_seconds_before_plotting"] = time.perf_counter() - started
        write_json(stage / "run_config.json", config)
        from arc_cd.repr_analysis.visualization import plot_analysis

        plot_analysis(stage)
        for path, previous in stats.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != previous:
                raise RuntimeError(f"Analysis input changed during the run: {path}")
        config["elapsed_seconds"] = time.perf_counter() - started
        write_json(stage / "run_config.json", config)
    logging.info("Published %s in %.1fs", destination, time.perf_counter() - started)
    return destination
