"""Traceable visual baselines using the existing parent-level separation metric."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from arc_cd.data_prep.concept_arc import ConceptARCTask
from arc_cd.data_prep.dataset_ids import infer_concept_arc_dataset_id
from arc_cd.data_prep.load_data import load_concept_arc
from arc_cd.probes.prompt_feature_io import staged_output_directory
from arc_cd.probes.prompt_features import task_support_identity
from arc_cd.repr_analysis.centroid_data import load_task_centroids
from arc_cd.repr_analysis.centroid_runner import write_centroid_separation
from arc_cd.repr_analysis.data import fingerprint
from arc_cd.repr_analysis.geometry import DISTANCE_METRICS, separation_similarity
from arc_cd.repr_analysis.population import POPULATIONS
from arc_cd.repr_analysis.results import software_versions, write_json
from arc_cd.repr_analysis.visual_features import (
    DEFAULT_FEATURES,
    FEATURE_DEFINITIONS,
    FEATURE_GROUPS,
    GRID_SCOPES,
    PAIR_FEATURE_NAMES,
    GridScope,
    SupportAggregation,
    aggregate_support_features,
    basic_pair_features,
    canonical_feature_groups,
    pair_feature_layout,
    preprocess_task_features,
    support_feature_layout,
)

LOGGER = logging.getLogger(__name__)


def extract_visual_features(
    tasks: Sequence[ConceptARCTask],
    aggregation: SupportAggregation,
    grid_scope: GridScope = "both",
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> dict[str, np.ndarray]:
    """Keep ordered pair/version features and average eight versions per parent.

    Args:
        tasks (Sequence[ConceptARCTask]): Complete source dataset in any order.
        aggregation (SupportAggregation): concat or mean_std within each version.
        grid_scope (GridScope): Use input grids, output grids, or both.
        features (tuple[str, ...]): Requested feature groups.

    Returns:
        dict[str, np.ndarray]: Raw centroid-compatible arrays plus auditable pair
        features [parent, version, support, descriptor], version vectors, and identities.
    """
    features = canonical_feature_groups(features, grid_scope)
    pair_names, pair_blocks = pair_feature_layout(grid_scope, features)
    names, blocks = support_feature_layout(aggregation, grid_scope, features)
    groups: dict[str, dict[str, ConceptARCTask]] = {}
    for task in tasks:
        identity = task_support_identity(task)
        parent, support = identity["parent_task_id"], identity["support_set_id"]
        versions = groups.setdefault(parent, {})
        if support in versions:
            raise ValueError(f"Duplicate support version: {task.task_id}")
        if len(task.train) != 3 or (
            grid_scope != "input" and any(pair.output is None for pair in task.train)
        ):
            raise ValueError(
                f"Exactly three supports with the selected grids required: {task.task_id}"
            )
        versions[support] = task
    if not groups:
        raise ValueError("No visual-feature tasks found.")
    parent_ids = sorted(groups)
    ordered: list[list[ConceptARCTask]] = []
    support_ids: list[list[str]] = []
    for parent in parent_ids:
        versions = groups[parent]
        if len(versions) != 8 or "canonical" in versions:
            raise ValueError(f"Expected eight distinct support versions for {parent}.")
        support_ids.append(sorted(versions))
        ordered.append([versions[version] for version in sorted(versions)])
    pairs = np.empty((len(ordered), 8, 3, len(pair_names)), dtype=np.float64)
    LOGGER.info(
        "Extracting %d coordinates per support pair for %d parents (8 versions, 3 supports each).",
        len(pair_names),
        len(ordered),
    )
    for parent_index, parent_versions in enumerate(ordered):
        for version_index, task in enumerate(parent_versions):
            for support_index, pair in enumerate(task.train):
                try:
                    pairs[parent_index, version_index, support_index] = (
                        basic_pair_features(
                            pair.input,
                            pair.output,
                            grid_scope,
                            features,
                        )
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Feature extraction failed for {task.task_id}, "
                        f"support {support_index + 1}: {error}"
                    ) from error
        if (parent_index + 1) % 100 == 0 or parent_index + 1 == len(ordered):
            LOGGER.info("Extracted %d/%d parents.", parent_index + 1, len(ordered))
    version_features = np.asarray(
        [
            [
                aggregate_support_features(pair_set, aggregation, grid_scope, features)
                for pair_set in parent
            ]
            for parent in pairs
        ]
    )
    return {
        "embeddings": version_features.mean(axis=1),
        "source_task_id": np.asarray(parent_ids),
        "concept": np.asarray([parent.split("/")[0] for parent in parent_ids]),
        "source_rule": np.asarray([parent.split("/")[1] for parent in parent_ids]),
        "instance_count": np.full(len(parent_ids), 8, dtype=np.int64),
        "pair_features": pairs,
        "version_features": version_features,
        "version_task_ids": np.asarray([[t.task_id for t in v] for v in ordered]),
        "support_set_ids": np.asarray(support_ids),
        "source_paths": np.asarray([[str(t.path) for t in v] for v in ordered]),
        "feature_names": np.asarray(names),
        "feature_blocks": np.asarray(blocks),
        "pair_feature_names": np.asarray(pair_names),
        "pair_feature_blocks": np.asarray(pair_blocks),
    }


def log_feature_samples(
    arrays: dict[str, np.ndarray],
    source_rows: np.ndarray,
    selection: str,
) -> None:
    """Preview the first support-set version's complete vector for selected parents.

    Args:
        arrays (dict[str, np.ndarray]): Extracted features and aligned identities;
            version vectors already aggregate all three support pairs.
        source_rows (np.ndarray): Parent rows belonging to the analysis population.
        selection (str): Number of samples (0 through 10), or per_concept to show
            the first parent of every selected concept. Numeric previews prefer
            distinct concepts before filling with additional parents.
    """
    if selection == "0" or not LOGGER.isEnabledFor(logging.INFO):
        return
    first_by_concept: dict[str, int] = {}
    for row in source_rows.tolist():
        first_by_concept.setdefault(str(arrays["concept"][row]), row)
    rows = list(first_by_concept.values())
    if selection != "per_concept":
        seen = set(rows)
        rows.extend(row for row in source_rows.tolist() if row not in seen)
        rows = rows[: int(selection)]
    LOGGER.info(
        "Previewing %d feature vectors from the first support-set version only "
        "(all three supports; before version averaging, exclusions, or standardization).",
        len(rows),
    )
    names = arrays["feature_names"]
    for row in rows:
        LOGGER.info(
            "Feature sample: concept=%s parent_task_id=%s task_id=%s support_set_id=%s",
            arrays["concept"][row],
            arrays["source_task_id"][row],
            arrays["version_task_ids"][row, 0],
            arrays["support_set_ids"][row, 0],
        )
        values = arrays["version_features"][row, 0]
        LOGGER.info(
            "  feature_vector (%d coordinates): [%s]",
            len(values),
            ", ".join(f"{name}={value:.8g}" for name, value in zip(names, values)),
        )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse feature selection and standard separation-analysis options.

    Args:
        argv (list[str] | None): Explicit arguments, or process arguments.

    Returns:
        argparse.Namespace: Validated inputs, aggregation, and output controls.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--metadata_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("output"))
    parser.add_argument("--run_name", default="visual_v2")
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURE_GROUPS,
        default=list(DEFAULT_FEATURES),
        help="Descriptor groups to compute; pixel_change is a separate optional reference.",
    )
    parser.add_argument(
        "--support_aggregation", choices=("concat", "mean_std"), default="concat"
    )
    parser.add_argument(
        "--distance_metric",
        choices=DISTANCE_METRICS,
        default="cosine",
        help="Distance on standardized task vectors; Euclidean keeps vector magnitudes without unit-length normalization.",
    )
    parser.add_argument(
        "--grid_scope",
        choices=GRID_SCOPES,
        default="both",
        help="Use only input grids, only output grids, or both; pixel_change requires both.",
    )
    parser.add_argument(
        "--exclude_feature_groups",
        nargs="+",
        choices=FEATURE_GROUPS,
        default=[],
        help="Omit entire descriptor groups before distance analysis; keep at least one group.",
    )
    parser.add_argument(
        "--exclude_pair_features",
        nargs="+",
        choices=PAIR_FEATURE_NAMES,
        default=[],
        help="Omit named descriptors from every support slot or both mean/SD coordinates.",
    )
    parser.add_argument("--population", choices=POPULATIONS, default="all")
    parser.add_argument(
        "--family_roles",
        nargs="+",
        choices=("action", "layout", "selector"),
        default=["action", "layout", "selector"],
    )
    parser.add_argument(
        "--comparison_scopes",
        nargs="+",
        choices=("global", "cross_concept"),
        default=["global", "cross_concept"],
    )
    parser.add_argument("--min_family_tasks", type=int, default=10)
    parser.add_argument("--include_structural_selectors", action="store_true")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console logging level; progress and feature previews use INFO.",
    )
    parser.add_argument(
        "--log_feature_samples",
        choices=(*map(str, range(11)), "per_concept"),
        default="per_concept",
        help="Preview only the first support-set version's complete vector for this many parents, or one per selected concept; 0 disables previews.",
    )
    args = parser.parse_args(argv)
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        parser.error("--run_name must be a single directory name.")
    if min(args.min_family_tasks, args.num_threads) < 1:
        parser.error("Family size and thread count must be positive.")
    for key in (
        "family_roles",
        "comparison_scopes",
        "exclude_feature_groups",
        "exclude_pair_features",
    ):
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            parser.error(f"--{key} must contain unique values.")
    try:
        args.features = list(
            canonical_feature_groups(tuple(args.features), args.grid_scope)
        )
    except ValueError as error:
        parser.error(str(error))
    pair_names, pair_blocks = pair_feature_layout(args.grid_scope, tuple(args.features))
    available_groups = set(pair_blocks)
    unavailable = set(args.exclude_feature_groups) - available_groups
    if unavailable:
        parser.error(
            f"Feature groups {sorted(unavailable)} were not requested in --features."
        )
    if not available_groups - set(args.exclude_feature_groups):
        parser.error("At least one available feature group must remain selected.")
    args.exclude_feature_groups = [
        group for group in FEATURE_GROUPS if group in args.exclude_feature_groups
    ]
    selected_names = {
        name
        for name, block in zip(pair_names, pair_blocks)
        if block not in args.exclude_feature_groups
    }
    if not set(args.exclude_pair_features) <= selected_names:
        parser.error(
            "Excluded pair features must be available in the selected groups and grid scope."
        )
    if not selected_names - set(args.exclude_pair_features):
        parser.error("At least one pair feature must remain selected.")
    args.exclude_pair_features = [
        name for name in pair_names if name in args.exclude_pair_features
    ]
    return args


def run_visual_baseline(argv: list[str] | None = None) -> Path:
    """Extract, standardize, and evaluate selected visual features on CPU.

    Args:
        argv (list[str] | None): Explicit CLI arguments, or process arguments.

    Returns:
        Path: Published result directory with features, scores, and PNG/PDF plots.
    """
    args = parse_arguments(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    started = time.perf_counter()
    data_dir = args.data_dir.expanduser().resolve()
    metadata_dir = args.metadata_dir.expanduser().resolve()
    dataset_id = infer_concept_arc_dataset_id(data_dir)
    destination = (
        args.output_root.expanduser().resolve()
        / dataset_id
        / "visual_features"
        / "repr_analysis"
        / args.run_name
        / ("features_" + "_".join(args.features))
        / args.support_aggregation
    )
    if args.grid_scope != "both":
        destination /= f"{args.grid_scope}_only"
    if args.exclude_feature_groups:
        destination /= "without_" + "_".join(args.exclude_feature_groups)
    if args.exclude_pair_features:
        destination = destination.joinpath(
            "without_features", *args.exclude_pair_features
        )
    analysis = f"{args.distance_metric}_distance"
    destination = destination / args.population / "metadata" / analysis
    LOGGER.info(
        "Visual baseline: features=%s grid_scope=%s aggregation=%s distance=%s population=%s",
        ",".join(args.features),
        args.grid_scope,
        args.support_aggregation,
        args.distance_metric,
        args.population,
    )
    LOGGER.info("Results will be saved to %s", destination)
    with (
        staged_output_directory(destination) as stage,
        threadpool_limits(args.num_threads),
    ):
        metadata_paths = [
            metadata_dir / name
            for name in (
                "run_config.json",
                "axes.json",
                "task_metadata.json",
                "taxonomy.json",
            )
        ]
        metadata_hashes = {p.name: fingerprint(p) for p in metadata_paths}
        metadata_config = json.loads((metadata_dir / "run_config.json").read_text())
        expected_hashes = metadata_config["extraction_config"]["dataset"]["task_hashes"]
        paths = sorted(data_dir.glob("*/*.json"))
        path_by_id = {f"{p.parent.name}/{p.stem}": p for p in paths}
        if not paths or set(path_by_id) != set(expected_hashes):
            raise ValueError(
                "Source task versions do not match the frozen feature inventory."
            )
        LOGGER.info("Verifying %d source versions against frozen metadata.", len(paths))
        for task, path in path_by_id.items():
            if fingerprint(path)["sha256"] != expected_hashes[task]:
                raise ValueError(
                    f"Source hash differs from model feature extraction: {task}"
                )
        arrays = extract_visual_features(
            load_concept_arc(data_dir),
            args.support_aggregation,
            args.grid_scope,
            tuple(args.features),
        )
        arrays["source_sha256"] = np.asarray(
            [
                [expected_hashes[task] for task in row]
                for row in arrays["version_task_ids"]
            ]
        )
        np.savez_compressed(stage / "features.npz", allow_pickle=False, **arrays)
        data = load_task_centroids(
            stage / "features.npz", metadata_dir, args.family_roles, args.population
        )
        log_feature_samples(arrays, data.source_rows, args.log_feature_samples)
        names, blocks = support_feature_layout(
            args.support_aggregation, args.grid_scope, tuple(args.features)
        )
        prepared = preprocess_task_features(
            data.embeddings,
            blocks,
            tuple(args.exclude_feature_groups),
            feature_names=names,
            exclude_pair_features=tuple(args.exclude_pair_features),
        )
        means = prepared.pop("vectors")
        LOGGER.info(
            "Preprocessing %d parents: %d raw, %d included, %d nonconstant coordinates.",
            len(data.task_ids),
            len(names),
            prepared["included"].sum(),
            means.shape[1],
        )
        np.savez_compressed(stage / "preprocessing.npz", allow_pickle=False, **prepared)
        np.save(stage / "raw_task_means.npy", data.embeddings)
        np.save(stage / "task_means.npy", means)
        np.save(stage / "grand_mean.npy", prepared["mean"])
        LOGGER.info(
            "Computing %s distances and family separation probabilities.",
            args.distance_metric,
        )
        similarity = separation_similarity(means, args.distance_metric)
        label = f"Visual features · {args.support_aggregation} · three supports"
        label += "\n" + ", ".join(args.features)
        label += f"\n{args.distance_metric.capitalize()} distance"
        if args.grid_scope != "both":
            label += f"\n{args.grid_scope.capitalize()} grids only"
        if args.exclude_feature_groups:
            label += "\nWithout " + ", ".join(args.exclude_feature_groups)
        if args.exclude_pair_features:
            label += "\nWithout descriptors: " + ", ".join(args.exclude_pair_features)
        coverage = write_centroid_separation(
            stage,
            data,
            similarity,
            population_name=args.population,
            comparison_scopes=args.comparison_scopes,
            min_family_tasks=args.min_family_tasks,
            include_structural_selectors=args.include_structural_selectors,
            distance_metric=args.distance_metric,
            extra_axes={
                "representation_label": label,
                "probability_limits": [0, 1],
                "grid_scope": args.grid_scope,
                "geometry_layout": "per_grid",
                "features": args.features,
                "feature_names": names,
                "feature_blocks": blocks,
                "included_feature_names": np.asarray(names)[prepared["included"]],
                "retained_feature_names": np.asarray(names)[prepared["retained"]],
                "excluded_feature_groups": args.exclude_feature_groups,
                "excluded_pair_features": args.exclude_pair_features,
                "array_axes": {
                    "raw_task_means.npy": ["task", "feature"],
                    "task_means.npy": ["task", "retained_feature"],
                    "grand_mean.npy": ["feature"],
                },
                "feature_archive_axes": {
                    "pair_features": [
                        "source_parent",
                        "version",
                        "support",
                        "pair_feature",
                    ],
                    "version_features": ["source_parent", "version", "feature"],
                    "embeddings": ["source_parent", "feature"],
                },
            },
        )
        write_json(stage / "task_metadata.json", data.families.metadata)
        write_json(stage / "metadata_source_config.json", data.families.config)
        shutil.copyfile(metadata_dir / "taxonomy.json", stage / "taxonomy.json")
        config: dict[str, Any] = {
            "schema_version": 1,
            "analysis": analysis,
            "distance_metric": args.distance_metric,
            "label_mode": "metadata",
            "input_kind": "visual_features",
            "feature_set": "visual_v2",
            "features": args.features,
            "feature_definitions": {
                name: FEATURE_DEFINITIONS[name] for name in args.features
            },
            "object_extraction": (
                {
                    "extractor": "arc_cd.repr_analysis.visual_objects.visual_scene_objects"
                }
                if "objects" in args.features
                else None
            ),
            "grid_scope": args.grid_scope,
            "geometry_layout": "per_grid",
            "excluded_pair_features": args.exclude_pair_features,
            "feature_groups": {
                "available": [group for group in FEATURE_GROUPS if group in blocks],
                "included": [
                    group
                    for group in FEATURE_GROUPS
                    if group in np.asarray(blocks)[prepared["included"]]
                ],
                "excluded": args.exclude_feature_groups,
                "raw_dimension": len(names),
                "included_dimension": int(prepared["included"].sum()),
                "retained_dimension": means.shape[1],
            },
            "model_name": "visual_features",
            "dataset_id": dataset_id,
            "metadata_dataset_id": data.dataset_id,
            "taxonomy_version": data.families.taxonomy["version"],
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "data_dir": str(data_dir),
            "metadata_dir": str(metadata_dir),
            "metadata_fingerprints": metadata_hashes,
            "feature_fingerprint": fingerprint(stage / "features.npz"),
            "source_task_hashes": expected_hashes,
            "population": {
                "name": args.population,
                "concepts": data.concepts,
                "num_parents": len(data.task_ids),
                "num_support_sets": int(data.instance_counts.sum()),
                "source_num_parents": data.source_num_tasks,
            },
            "coverage": coverage,
            "software": software_versions(),
            "conventions": {
                "background_color": 0,
                "pair_features": arrays["pair_feature_names"].tolist(),
                "geometry": "separate input/output height, width, area, and aspect ratio; both still requires equal input/output sizes",
                "spatial": "foreground is color != 0; inclusive bounding boxes; centroids use zero-based coordinates; empty extents are 0, empty/singleton-axis centroids are 0.5",
                "objects": "local visual_scene_objects: same-color eight-connected components; singleton components grouped by four-connectivity of their 3x3 dilation masks; statistics use original visible pixels and equally weight objects; empty statistics are 0",
                "support_aggregation": args.support_aggregation,
                "support_order": "original train order; concat is order-sensitive",
                "standard_deviation": "population (ddof=0), within support sets before version averaging",
                "versions": "equal arithmetic mean of exactly eight distinct support versions per parent",
                "centering": "subtract coordinate mean over selected parent representations after version averaging",
                "scaling": "divide by selected-parent population SD; drop constant coordinates; multiply each block by inverse sqrt of retained dimension count",
                "feature_ablation": "exclude all coordinates of selected groups and pair descriptors before distance calculation; raw archives retain the requested feature groups for grid_scope; retained coordinates keep their original centering and scaling; block weights use the remaining nonconstant coordinate count",
                "normalization": (
                    "cosine after aggregation and preprocessing; no per-pair or per-version normalization"
                    if args.distance_metric == "cosine"
                    else "Euclidean on aggregated, centered, standardized, block-weighted task vectors; no unit-length normalization"
                ),
                "comparison_similarity": (
                    "cosine similarity; higher means closer"
                    if args.distance_metric == "cosine"
                    else "negative Euclidean distance computed from coordinate differences; higher means closer"
                ),
                "zero_vectors": (
                    "prepared vectors within the float64 centering-error bound are zero; preprocessing.npz records zero_norm_threshold after accounting for sample count, feature magnitude, scaling, and block weights; "
                    + (
                        "their cosine is undefined"
                        if args.distance_metric == "cosine"
                        else "their Euclidean distances remain valid; all-constant representations produce ties"
                    )
                ),
                "global": "exclude anchor parent",
                "cross_concept": "exclude anchor's entire original concept",
                "ties": "strict wins and half-credit-for-ties scores both saved",
                "aggregation": "mean valid anchors per family, then equal mean over eligible families",
                "missing": "undefined comparisons remain NaN/null with shared reason bits",
                "calculation_dtype": "float64",
                "bootstrap": False,
            },
            "elapsed_seconds_before_plotting": time.perf_counter() - started,
        }
        write_json(stage / "run_config.json", config)
        from arc_cd.repr_analysis.visualization import plot_analysis

        LOGGER.info("Rendering figures and verifying source files before publication.")
        plot_analysis(stage)
        if sorted(data_dir.glob("*/*.json")) != paths:
            raise RuntimeError("Source dataset inventory changed during the run.")
        for task, path in path_by_id.items():
            if fingerprint(path)["sha256"] != expected_hashes[task]:
                raise RuntimeError(f"Source dataset changed during the run: {task}")
        for path in metadata_paths:
            if fingerprint(path) != metadata_hashes[path.name]:
                raise RuntimeError(f"Frozen metadata changed during the run: {path}")
        config["elapsed_seconds"] = time.perf_counter() - started
        write_json(stage / "run_config.json", config)
    LOGGER.info(
        "Published %d parents, %d raw / %d retained features: %s",
        len(data.task_ids),
        len(names),
        means.shape[1],
        destination,
    )
    return destination
