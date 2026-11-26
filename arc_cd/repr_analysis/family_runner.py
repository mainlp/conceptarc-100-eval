"""Staged, reproducible metadata analyses sharing the existing feature pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from arc_cd.probes.prompt_feature_io import staged_output_directory
from arc_cd.repr_analysis.data import center_versions, fingerprint, load_bundle
from arc_cd.repr_analysis.family_data import (
    FamilyData,
    family_comparison_mask,
    load_family_data,
)
from arc_cd.repr_analysis.family_metrics import (
    REASONS,
    SEPARATION_METRICS,
    aggregate_families,
    eligible_references,
    family_separation,
)
from arc_cd.repr_analysis.geometry import DistanceMetric, separation_similarity
from arc_cd.repr_analysis.population import select_population
from arc_cd.repr_analysis.results import (
    representation_arrays,
    save_axes,
    software_versions,
    write_json,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a flat numerical summary, leaving undefined values empty.

    Args:
        path (Path): Destination CSV.
        rows (list[dict[str, Any]]): Ordered rows with a common schema.
    """
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if isinstance(value, (float, np.floating))
                        and not np.isfinite(value)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _coverage(
    families: FamilyData, min_tasks: int, included: np.ndarray
) -> dict[str, Any]:
    """Describe parent coverage, overlapping memberships, and rare families.

    Args:
        families (FamilyData): Aligned annotations.
        min_tasks (int): Headline metadata-family threshold.
        included (np.ndarray): Families included in headline means.

    Returns:
        dict[str, Any]: Coverage by role, including exact family parent counts.
    """
    sizes = families.membership.sum(axis=0)
    result = {}
    for role in families.role_order:
        indices = np.flatnonzero(np.asarray(families.roles) == role)
        compared = indices[included[indices]]
        result[role] = {
            "total_parents": len(families.membership),
            "annotated_parents": int(families.membership[:, indices].any(axis=1).sum()),
            "families": len(indices),
            "comparison_families": len(compared),
            "comparison_annotated_parents": int(
                families.membership[:, compared].any(axis=1).sum()
            ),
            "excluded_families": [
                families.names[i] for i in indices if not included[i]
            ],
            "headline_eligible_families": int(
                sum(sizes[i] >= min_tasks or role == "concept" for i in compared)
            ),
            "family_parent_counts": {families.names[i]: int(sizes[i]) for i in indices},
        }
    return result


def run_metadata_analysis(args: argparse.Namespace) -> Path:
    """Run a descriptive family analysis and publish only complete artifacts.

    Args:
        args (argparse.Namespace): Validated shared runner arguments.

    Returns:
        Path: Published metadata sub-analysis, independently regenerable.
    """
    started = time.perf_counter()
    distance_metric: DistanceMetric = args.distance_metric
    output_analysis = f"{distance_metric}_distance"
    reason_bits = {
        (
            "undefined_anchor_distance"
            if distance_metric == "euclidean" and name == "undefined_anchor_cosine"
            else name
        ): bit
        for name, bit in REASONS.items()
    }
    bundle = load_bundle(args.feature_dir, args.layers)
    population = select_population(bundle, args.population)
    metadata_dir = args.metadata_dir.expanduser().resolve()
    destination = (
        args.output_root.expanduser().resolve()
        / bundle.config["dataset"]["id"]
        / bundle.config["extraction"]["model_name"]
        / "repr_analysis"
        / args.run_name
        / population.name
        / "metadata"
        / output_analysis
    )
    feature_paths = [
        bundle.directory / name
        for name in (
            "run_config.json",
            "manifest.jsonl",
            "tokens.npz",
            *(f"{p}.npz" for p in args.positions),
        )
    ]
    metadata_paths = [
        metadata_dir / name
        for name in (
            "run_config.json",
            "axes.json",
            "task_metadata.json",
            "taxonomy.json",
        )
    ]
    input_paths = feature_paths + metadata_paths
    stats = {
        path: (path.stat().st_size, path.stat().st_mtime_ns) for path in input_paths
    }
    with (
        staged_output_directory(destination) as stage,
        ExitStack() as maps,
        threadpool_limits(limits=args.num_threads),
    ):
        hashes = {path.name: fingerprint(path) for path in feature_paths}
        families = load_family_data(
            metadata_dir, bundle, args.family_roles, hashes, population=population
        )
        included = family_comparison_mask(
            families, include_structural_selectors=args.include_structural_selectors
        )
        coverage = _coverage(families, args.min_family_tasks, included)
        roles = families.role_order
        sizes = families.membership.sum(axis=0)
        headline_eligible = included & (
            (sizes >= args.min_family_tasks) | (np.asarray(families.roles) == "concept")
        )
        exclusion_reasons = [
            (
                f"structural_{families.roles[i]}"
                if not included[i]
                else "below_min_family_tasks"
                if not headline_eligible[i]
                else ""
            )
            for i in range(len(included))
        ]
        metrics = SEPARATION_METRICS
        save_axes(
            stage,
            bundle,
            args.positions,
            metrics,
            population=population,
            distance_metric=distance_metric,
        )
        axes = json.loads((stage / "axes.json").read_text())
        axes.update(
            {
                "family_ids": families.ids,
                "family_names": families.names,
                "family_roles": families.roles,
                "roles": roles,
                "scopes": args.comparison_scopes,
                "family_sizes": sizes,
                "comparison_included": included,
                "headline_eligible": headline_eligible,
                "headline_exclusion_reasons": exclusion_reasons,
                "reason_bits": reason_bits,
                "array_axes": {
                    **axes["array_axes"],
                    "memberships.npy": ["task", "family"],
                    "family_overlap.npy": ["family", "family"],
                    "layer.scores": ["scope", "task", "family", "metric"],
                    "layer.reasons": ["scope", "task", "family", "metric"],
                    "layer.reference_counts": [
                        "scope",
                        "task",
                        "family",
                        "positive_negative",
                    ],
                    "layer.comparison_counts": ["scope", "task", "family"],
                    "layer.strict_win_counts": ["scope", "task", "family"],
                    "layer.tie_counts": ["scope", "task", "family"],
                    "family_mean": ["position", "layer", "scope", "family", "metric"],
                    "family_valid_anchors": [
                        "position",
                        "layer",
                        "scope",
                        "family",
                        "metric",
                    ],
                    "macro": ["position", "layer", "scope", "role", "metric"],
                    "macro_valid_families": [
                        "position",
                        "layer",
                        "scope",
                        "role",
                        "metric",
                    ],
                },
            }
        )
        write_json(stage / "axes.json", axes)
        np.save(stage / "memberships.npy", families.membership)
        overlap = families.membership.astype(np.int64).T @ families.membership.astype(
            np.int64
        )
        np.save(stage / "family_overlap.npy", overlap)
        shutil.copyfile(metadata_dir / "taxonomy.json", stage / "taxonomy.json")
        write_json(stage / "task_metadata.json", families.metadata)
        write_json(stage / "metadata_source_config.json", families.config)
        write_json(stage / "coverage.json", coverage)
        config = {
            "schema_version": 1,
            "analysis": output_analysis,
            "distance_metric": distance_metric,
            "label_mode": "metadata",
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "feature_dir": str(bundle.directory),
            "extraction_config": bundle.config,
            "population": population.snapshot(),
            "fingerprints": hashes,
            "metadata_fingerprints": {
                path.name: fingerprint(path) for path in metadata_paths
            },
            "taxonomy_version": families.taxonomy["version"],
            "software": software_versions(),
            "coverage": coverage,
            "conventions": {
                "centering": "selected population version rows per layer/position; eight versions averaged per parent",
                "normalization": (
                    "none; Euclidean distance between task means without unit normalization or feature scaling"
                    if distance_metric == "euclidean"
                    else "only for cosine, after task/family averaging"
                ),
                "calculation_dtype": "float64",
                "representation_storage_dtype": "float32",
                "families": "task-level memberships from the frozen taxonomy, including explicit concept-derived action labels; nested operation annotations retain provenance; no duplicate filtering",
                "global": f"selected population finite {distance_metric} references excluding the anchor parent",
                "cross_concept": f"selected population finite {distance_metric} references excluding the anchor's entire original concept",
                "missing": "NaN/null; missing positives, negatives, defined anchor distance; reasons are additive bitmasks",
                "ties": "half credit for separation ties",
                "aggregation": "mean valid anchors per family, then equal mean over included families meeting min_family_tasks; all concept groups eligible",
                "structural_selectors": "taxonomy-marked structural selector tags are retained in detailed statistics; excluded from headline means unless include_structural_selectors is enabled",
                "structural_actions": "taxonomy-marked control-flow tags are retained as metadata and excluded from headline means; branch actions remain declared alternatives",
                "interpretation": "descriptive alignment, not independent or causal explanation of geometry",
                "bootstrap": False,
            },
        }
        write_json(stage / "run_config.json", config)
        logging.info(
            "Selected %s population: %d parents / %d versions; taxonomy %s; coverage %s",
            population.name,
            len(population.task_ids),
            len(population.records),
            families.taxonomy["version"],
            {r: c["headline_eligible_families"] for r, c in coverage.items()},
        )
        shape = (len(args.positions), len(bundle.layers), len(args.comparison_scopes))
        results: dict[str, np.ndarray] = {}
        for key in (
            "family_mean",
            "family_valid_anchors",
            "macro",
            "macro_valid_families",
        ):
            size = len(families.ids) if key.startswith("family_") else len(roles)
            results[key] = (
                np.full((*shape, size, len(metrics)), np.nan)
                if key in ("family_mean", "macro")
                else np.zeros((*shape, size, len(metrics)), dtype=np.int64)
            )
        macro_rows: list[dict[str, Any]] = []
        family_rows: list[dict[str, Any]] = []
        for p, position in enumerate(args.positions):
            raw = bundle.load_position(position)
            arrays = representation_arrays(
                stage, position, bundle, raw.shape[-1], maps, population=population
            )
            for layer_index, layer in enumerate(bundle.layers):
                raw_index = bundle.config["extraction"]["layers"].index(layer)
                grand, versions, means = center_versions(
                    raw[:, raw_index], population.source_rows
                )
                del versions
                arrays["grand_means"][layer_index] = grand
                arrays["task_means"][:, layer_index] = means
                arrays["concept_means"][:, layer_index] = np.stack(
                    [
                        means[population.labels == i].mean(axis=0)
                        for i in range(len(population.concepts))
                    ]
                )
                similarity = separation_similarity(means, distance_metric)
                details: dict[str, list[np.ndarray]] = {}
                for scope_index, scope in enumerate(args.comparison_scopes):
                    index = (p, layer_index, scope_index)
                    common = {"position": position, "layer": layer, "scope": scope}
                    eligible = eligible_references(similarity, population.labels, scope)
                    output = family_separation(
                        similarity, eligible, families.membership
                    )
                    if scope == "cross_concept":
                        concept_columns = np.asarray(families.roles) == "concept"
                        output["scores"][:, concept_columns] = np.nan
                        output["reasons"][:, concept_columns] |= REASONS[
                            "concept_reference_not_applicable"
                        ]
                    summary = aggregate_families(
                        output["scores"],
                        families.membership,
                        families.roles,
                        args.min_family_tasks,
                        roles,
                        included=included,
                    )
                    for key, values in summary.items():
                        results[key][index] = values
                    for key, values in output.items():
                        details.setdefault(key, []).append(values)
                    for f, family in enumerate(families.ids):
                        for m, metric in enumerate(metrics):
                            reason = output["reasons"][:, f, m]
                            members = families.membership[:, f]
                            family_rows.append(
                                {
                                    **common,
                                    "family": family,
                                    "metric": metric,
                                    "value": summary["family_mean"][f, m],
                                    "parent_count": int(sizes[f]),
                                    "comparison_included": bool(included[f]),
                                    "headline_eligible": bool(headline_eligible[f]),
                                    "headline_exclusion_reason": exclusion_reasons[f],
                                    "valid_anchors": int(
                                        summary["family_valid_anchors"][f, m]
                                    ),
                                    **{
                                        name: int(((reason & bit != 0) & members).sum())
                                        for name, bit in reason_bits.items()
                                        if name != "not_member"
                                    },
                                }
                            )
                    for r, role in enumerate(roles):
                        selected = (
                            np.asarray(families.roles) == role
                        ) & headline_eligible
                        for m, metric in enumerate(metrics):
                            valid = np.isfinite(output["scores"][:, selected, m])
                            macro_rows.append(
                                {
                                    **common,
                                    "role": role,
                                    "metric": metric,
                                    "value": summary["macro"][r, m],
                                    "valid_families": int(
                                        summary["macro_valid_families"][r, m]
                                    ),
                                    "eligible_families": int(selected.sum()),
                                    "all_families": int(
                                        sum(np.asarray(families.roles) == role)
                                    ),
                                    "valid_anchor_memberships": int(valid.sum()),
                                    "valid_unique_anchors": int(
                                        valid.any(axis=1).sum()
                                    ),
                                    "eligible_unique_anchors": int(
                                        families.membership[:, selected]
                                        .any(axis=1)
                                        .sum()
                                    ),
                                    "total_parents": len(population.task_ids),
                                }
                            )
                if details:
                    np.savez_compressed(
                        stage / position / f"layer_{layer}.npz",
                        allow_pickle=False,
                        **{key: np.stack(values) for key, values in details.items()},
                    )
                if layer_index % 8 == 0 or layer_index == len(bundle.layers) - 1:
                    logging.info(
                        "%s: %s layer %d (%d/%d)",
                        output_analysis,
                        position,
                        layer,
                        layer_index + 1,
                        len(bundle.layers),
                    )
            for array in arrays.values():
                array.flush()
            del raw
        np.savez_compressed(stage / "metrics.npz", allow_pickle=False, **results)
        _write_csv(stage / "summary.csv", macro_rows)
        _write_csv(stage / "family_summary.csv", family_rows)
        write_json(stage / "summary.json", {"coverage": coverage, "rows": macro_rows})
        config["elapsed_seconds_before_plotting"] = time.perf_counter() - started
        write_json(stage / "run_config.json", config)
        from arc_cd.repr_analysis.visualization import plot_analysis

        plot_analysis(stage, pdf=args.pdf)
        for path, previous in stats.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != previous:
                raise RuntimeError(f"Analysis input changed during the run: {path}")
        config["elapsed_seconds"] = time.perf_counter() - started
        write_json(stage / "run_config.json", config)
    logging.info("Published %s in %.1fs", destination, time.perf_counter() - started)
    return destination
