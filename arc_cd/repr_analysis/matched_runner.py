"""Prepare frozen visual matches and evaluate existing task representations."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from arc_cd.probes.prompt_feature_io import staged_output_directory
from arc_cd.repr_analysis.centroid_data import load_centroid_archive
from arc_cd.repr_analysis.data import (
    FeatureBundle,
    center_versions,
    load_bundle,
)
from arc_cd.repr_analysis.geometry import (
    DISTANCE_METRICS,
    cosine_from_gram,
    separation_similarity,
)
from arc_cd.repr_analysis.matched_design import (
    CONDITIONS,
    build_matching_design,
    matching_summary,
)
from arc_cd.repr_analysis.matched_examples import (
    DEFAULT_NUM_QUALITATIVE_EXAMPLES,
    save_qualitative_examples,
)
from arc_cd.repr_analysis.matched_features import (
    PROFILES,
    extract_controls,
    load_matching_tasks,
    validate_support_examples,
)
from arc_cd.repr_analysis.matched_metrics import (
    aggregate_scores,
    evaluate_groups,
)
from arc_cd.repr_analysis.results import write_json

LOGGER = logging.getLogger(__name__)
METRICS = ("strict_win", "tie_adjusted")


def action_memberships(
    records: list[dict[str, Any]], taxonomy: dict[str, Any]
) -> tuple[list[str], np.ndarray]:
    """Build action labels and their task-by-action membership matrix.

    Args:
        records (list[dict[str, Any]]): Task metadata in matching order.
        taxonomy (dict[str, Any]): Definitions of structural and headline action tags.

    Returns:
        tuple[list[str], np.ndarray]: Action names and Boolean task memberships.
    """
    tags = [record["families"]["action"] for record in records]
    families = taxonomy["families"]["action"]
    names = sorted(
        {
            tag
            for row in tags
            for tag in row
            if families[tag].get("kind") != "structural"
        }
    )
    return names, np.asarray(
        [[name in row for name in names] for row in tags], dtype=bool
    )


def prepare_matches(
    data_dir: Path,
    metadata_dir: Path,
    output_dir: Path,
    *,
    min_tasks: int = 10,
    expected_parents: int = 1600,
) -> Path:
    """Save reusable matches and a compact coverage/visual-distance summary.

    Args:
        data_dir (Path): Eight-support-set source dataset.
        metadata_dir (Path): Metadata explorer containing task/action annotations.
        output_dir (Path): Matching directory; existing output is replaced on success.
        min_tasks (int): Minimum distinct parents per headline role.
        expected_parents (int): Required parent count; smaller values support fixtures.

    Returns:
        Path: Published matching directory.
    """
    data_dir, metadata_dir, output_dir = (
        p.expanduser().resolve() for p in (data_dir, metadata_dir, output_dir)
    )
    LOGGER.info("Reading task annotations from %s.", metadata_dir)
    source_axes = json.loads((metadata_dir / "axes.json").read_text())
    records = json.loads((metadata_dir / "task_metadata.json").read_text())["records"]
    taxonomy = json.loads((metadata_dir / "taxonomy.json").read_text())
    task_ids = source_axes["task_ids"]
    concepts, labels = source_axes["concepts"], source_axes["labels"]
    if len(task_ids) != expected_parents:
        raise ValueError(f"Expected {expected_parents} parents, found {len(task_ids)}.")
    if (
        len(set(task_ids)) != len(task_ids)
        or [record["task_id"] for record in records] != task_ids
        or len(labels) != len(task_ids)
        or any(
            type(label) is not int or not 0 <= label < len(concepts) for label in labels
        )
    ):
        raise ValueError(
            "Metadata parent identities or concept labels are inconsistent."
        )
    with staged_output_directory(output_dir, overwrite=True) as stage:
        LOGGER.info("Loading support-set tasks from %s.", data_dir)
        tasks = load_matching_tasks(data_dir, metadata_dir)
        LOGGER.info(
            "Extracting visual controls from %d support-set versions.", len(tasks)
        )
        arrays = extract_controls(tasks)
        if set(arrays["source_task_id"]) != set(task_ids):
            raise ValueError("Source parent identities differ from metadata.")
        by_id = {task: i for i, task in enumerate(arrays["source_task_id"])}
        order = np.asarray([by_id[task] for task in task_ids])
        for record, label, source in zip(records, labels, order):
            if (
                record["concept_name"] != concepts[label]
                or arrays["concept"][source] != concepts[label]
            ):
                raise ValueError("Source concept labels differ from metadata.")
        identities = {
            key: arrays[key][order] for key in ("version_task_ids", "support_set_ids")
        }
        if any(source_axes[key] != value.tolist() for key, value in identities.items()):
            raise ValueError(
                "Metadata support-set identities differ from source tasks."
            )
        names, membership = action_memberships(records, taxonomy)
        LOGGER.info(
            "Matching %d parents across %d action families.", len(records), len(names)
        )
        layouts = np.asarray([r["root_ops"]["layout"] for r in records])
        vectors = {p: arrays[f"{p}_vectors"][order] for p in PROFILES}
        design = build_matching_design(membership, layouts, vectors, min_tasks)
        LOGGER.info("Writing matching design and coverage summaries.")
        write_json(stage / "matching_summary.json", matching_summary(design, names))
        axes: dict[str, Any] = {
            "task_ids": task_ids,
            "concepts": concepts,
            "labels": labels,
            "families": names,
            "conditions": CONDITIONS,
        }
        design.pop("edge_visual_distances")
        np.savez_compressed(stage / "matches.npz", **design, **identities)
        write_json(stage / "axes.json", axes)
        write_json(
            stage / "run_config.json",
            {
                "analysis": "matched_action_design",
                "data_dir": str(data_dir),
                "metadata_dir": str(metadata_dir),
                "min_tasks": min_tasks,
                "num_parents": len(task_ids),
                "target_roles": ["action"],
                "conventions": {
                    "matching": "all B-C pairs with equal root layout op; A unrestricted; distinct parents",
                    "features": "ordered three-support descriptors averaged over eight support sets; input: geometry/color; full: both sides plus changed count/fraction",
                    "scaling": "population z-scores; constants removed; blocks weighted by inverse square root of dimension; Euclidean distance",
                    "cutoffs": "strict distance < pooled nearest-negative quartile over original headline family/B memberships",
                },
            },
        )
        for i, condition in enumerate(CONDITIONS):
            LOGGER.info(
                "%s: %d headline actions.", condition, design["headline"][i].sum()
            )
    LOGGER.info("Saved matching design to %s.", output_dir)
    return output_dir


def load_matching_design(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    """Load reusable matches and check their task/action axes.

    Args:
        directory (Path): Published matching directory.

    Returns:
        tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]: Config, axes, graph.
    """
    config = json.loads((directory / "run_config.json").read_text())
    axes = json.loads((directory / "axes.json").read_text())
    with np.load(directory / "matches.npz", allow_pickle=False) as archive:
        design = {name: archive[name] for name in archive.files}
    if not {"version_task_ids", "support_set_ids"} <= design.keys():
        raise ValueError("Matching design lacks support-set identities; rerun prepare.")
    n = len(axes["task_ids"])
    if (
        axes["conditions"] != list(CONDITIONS)
        or len(set(axes["task_ids"])) != n
        or len(axes["labels"]) != n
        or any(
            type(label) is not int or not 0 <= label < len(axes["concepts"])
            for label in axes["labels"]
        )
        or design["membership"].shape != (n, len(axes["families"]))
        or design["version_task_ids"].shape != (n, 8)
        or design["support_set_ids"].shape != (n, 8)
    ):
        raise ValueError("Matching axes are inconsistent.")
    return config, axes, design


def align_feature_bundle(
    bundle: FeatureBundle, design: dict[str, np.ndarray], axes: dict[str, Any]
) -> np.ndarray:
    """Join model parents and check concept labels and ordered support-set IDs.

    Args:
        bundle (FeatureBundle): Validated complete model feature bundle.
        design (dict[str, np.ndarray]): Matching graph and support-set identities.
        axes (dict[str, Any]): Matching parent identities and labels.

    Returns:
        np.ndarray: Model parent indices in matching order.
    """
    if set(bundle.task_ids) != set(axes["task_ids"]):
        raise ValueError(
            "Feature parent identities differ from the matching population."
        )
    by_id = {task: i for i, task in enumerate(bundle.task_ids)}
    order = np.asarray([by_id[task] for task in axes["task_ids"]])
    records = [[bundle.records[row] for row in bundle.rows[source]] for source in order]
    identities = {
        "version_task_ids": [[r["task_id"] for r in rows] for rows in records],
        "support_set_ids": [
            [r["metadata"]["support_set_id"] for r in rows] for rows in records
        ],
    }
    if any(
        not np.array_equal(values, design[key]) for key, values in identities.items()
    ):
        raise ValueError("Feature support-set identities differ from matching.")
    for i, source in enumerate(order):
        if (
            bundle.concepts[bundle.labels[source]]
            != axes["concepts"][axes["labels"][i]]
        ):
            raise ValueError("Feature concept labels differ from matching.")
    return order


def _reference_values(
    directory: Path | None, bundle: FeatureBundle, axes: dict[str, Any]
) -> dict[tuple[str, int, str], float]:
    """Read an optional existing global action result after checking its provenance.

    Args:
        directory (Path | None): Original family-separation directory.
        bundle (FeatureBundle): Evaluation feature source.
        axes (dict[str, Any]): Frozen matching identities.

    Returns:
        dict[tuple[str, int, str], float]: Position/layer/metric reference values.
    """
    if directory is None:
        return {}
    config = json.loads((directory / "run_config.json").read_text())
    original_axes = json.loads((directory / "axes.json").read_text())
    if (
        config["extraction_config"]["extraction"] != bundle.config["extraction"]
        or original_axes["task_ids"] != axes["task_ids"]
        or config.get("analysis") != "cosine_distance"
    ):
        raise ValueError(
            "Reference analysis does not match these features and metadata."
        )
    with (directory / "summary.csv").open() as stream:
        return {
            (r["position"], int(r["layer"]), r["metric"]): float(r["value"])
            for r in csv.DictReader(stream)
            if r["role"] == "action" and r["scope"] == "global" and r["value"]
        }


def evaluate_matches(args: argparse.Namespace) -> Path:
    """Score saved representations and publish overall and per-action comparisons.

    Args:
        args (argparse.Namespace): Validated evaluation CLI arguments.

    Returns:
        Path: Published summary JSONs, run configuration and figures.
    """
    design_dir = args.match_dir.expanduser().resolve()
    LOGGER.info("Loading matching design from %s.", design_dir)
    match_config, axes, design = load_matching_design(design_dir)
    LOGGER.info("Loading and validating feature metadata from %s.", args.feature_dir)
    bundle = load_bundle(args.feature_dir, args.layers)
    order = align_feature_bundle(bundle, design, axes)
    if args.metadata_dir is not None:
        validate_support_examples(
            args.metadata_dir.expanduser(),
            ((row["task_id"], row["support_examples"]) for row in bundle.records),
        )
    LOGGER.info(
        "Evaluating %d parents, %d action families, %d positions and %d layers with %d CPU threads.",
        len(order),
        len(axes["families"]),
        len(args.positions),
        len(bundle.layers),
        args.num_threads,
    )
    reference = _reference_values(args.reference_analysis_dir, bundle, axes)
    output_dir = args.output_dir.expanduser().resolve()
    summary: dict[str, Any] = {}
    families: dict[str, Any] = {}
    with (
        staged_output_directory(output_dir, overwrite=True) as stage,
        threadpool_limits(limits=args.num_threads),
    ):
        shutil.copyfile(
            design_dir / "matching_summary.json", stage / "matching_summary.json"
        )
        for position in args.positions:
            summary[position], families[position] = {}, {}
            LOGGER.info("Loading %s feature archive into memory.", position)
            raw = bundle.load_position(position)
            for layer in tqdm(
                bundle.layers, desc=f"Scoring {position}", unit="layer", disable=None
            ):
                LOGGER.info("Scoring %s layer %d.", position, layer)
                raw_layer = bundle.config["extraction"]["layers"].index(layer)
                _, versions, means = center_versions(
                    raw[:, raw_layer], bundle.rows[order]
                )
                del versions
                groups, original = evaluate_groups(
                    cosine_from_gram(means @ means.T), design
                )
                point = aggregate_scores(groups, original, design)
                for metric, value in zip(METRICS, point["original_macro"]):
                    key = (position, layer, metric)
                    if args.reference_analysis_dir is not None and (
                        key not in reference
                        or not np.isclose(value, reference[key], rtol=0, atol=1e-10)
                    ):
                        raise ValueError(
                            f"Original reference missing or mismatched: {key}"
                        )
                summary[position][str(layer)], families[position][str(layer)] = (
                    _summaries(axes, design, point)
                )
            del raw
        LOGGER.info("Writing result summaries.")
        write_json(stage / "summary.json", summary)
        write_json(stage / "family_summary.json", families)
        write_json(
            stage / "run_config.json",
            {
                "analysis": "matched_action_separation",
                "match_dir": str(design_dir),
                "feature_dir": str(bundle.directory),
                "model_name": bundle.config["extraction"]["model_name"],
                "positions": args.positions,
                "layers": bundle.layers,
                "reference_analysis_dir": (
                    str(args.reference_analysis_dir)
                    if args.reference_analysis_dir
                    else None
                ),
                "conventions": {
                    "geometry": "float64 cosine of globally centered eight-version parent means",
                    "aggregation": "equal negative actions within anchor, equal anchors within target action, equal eligible target actions; overlaps retained",
                    "difference_from_original": "condition minus original convention at the same position/layer; each uses its own eligible population",
                    "missing": "undefined groups/anchors omitted; undefined required target action makes overall score undefined",
                },
            },
        )
        from arc_cd.repr_analysis.matched_visualization import plot_matched_analysis

        plot_matched_analysis(stage)
        if args.qualitative_examples:
            save_qualitative_examples(
                args.data_dir or Path(match_config["data_dir"]),
                axes,
                design,
                stage / "qualitative_examples",
                metadata_dir=args.metadata_dir or Path(match_config["metadata_dir"]),
                num_examples=args.num_qualitative_examples,
            )
    LOGGER.info("Saved analysis and figures to %s.", output_dir)
    return output_dir


def load_visual_vectors(
    directory: Path, design: dict[str, np.ndarray], axes: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    """Align prepared visual-baseline vectors without repeating preprocessing.

    Args:
        directory (Path): Completed run_visual_baseline output directory.
        design (dict[str, np.ndarray]): Matching support-set identities.
        axes (dict[str, Any]): Matching parent IDs and concept labels.

    Returns:
        tuple[dict[str, Any], np.ndarray]: Visual settings and aligned task vectors.
    """
    config = json.loads((directory / "run_config.json").read_text())
    visual_axes = json.loads((directory / "axes.json").read_text())
    if config.get("input_kind") != "visual_features":
        raise ValueError("Expected a completed visual baseline directory.")
    task_ids = visual_axes["task_ids"]
    if len(set(task_ids)) != len(task_ids) or set(task_ids) != set(axes["task_ids"]):
        raise ValueError("Visual parents must match the complete matching population.")
    vectors = np.load(directory / "task_means.npy", allow_pickle=False)
    if (
        vectors.ndim != 2
        or len(vectors) != len(task_ids)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Expected finite visual vectors aligned to the parent axis.")
    by_id = {task: row for row, task in enumerate(task_ids)}
    order = np.asarray([by_id[task] for task in axes["task_ids"]])
    for target, source in enumerate(order):
        if (
            visual_axes["concepts"][visual_axes["labels"][source]]
            != axes["concepts"][axes["labels"][target]]
        ):
            raise ValueError("Visual concept labels differ from matching.")
    # Raw features use their own parent order, independent of task_means.npy.
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        source_ids = archive["source_task_id"].tolist()
        if len(set(source_ids)) != len(source_ids) or set(source_ids) != set(task_ids):
            raise ValueError(
                "Visual feature archive parent identities are inconsistent."
            )
        by_id = {task: row for row, task in enumerate(source_ids)}
        source_order = np.asarray([by_id[task] for task in axes["task_ids"]])
        for key in ("version_task_ids", "support_set_ids"):
            if not np.array_equal(archive[key][source_order], design[key]):
                raise ValueError("Visual support-set identities differ from matching.")
    return config, np.asarray(vectors[order], dtype=np.float64)


def load_centroid_vectors(path: Path, axes: dict[str, Any]) -> np.ndarray:
    """Align task centroids to the complete matching population and center them.

    Args:
        path (Path): Eight-support-set centroid NPZ, such as the VARC embeddings.
        axes (dict[str, Any]): Frozen matching parent IDs and concept labels.

    Returns:
        np.ndarray: Float64 parent vectors with the population mean subtracted.
    """
    arrays, task_ids = load_centroid_archive(path)
    if set(task_ids) != set(axes["task_ids"]):
        raise ValueError(
            "Centroid parents must match the complete matching population."
        )
    by_id = {task: row for row, task in enumerate(task_ids)}
    order = np.asarray([by_id[task] for task in axes["task_ids"]])
    for task, label in zip(axes["task_ids"], axes["labels"]):
        if task.split("/")[0] != axes["concepts"][label]:
            raise ValueError("Centroid concept labels differ from matching.")
    vectors = np.asarray(arrays["embeddings"][order], dtype=np.float64)
    return vectors - vectors.mean(axis=0)


def evaluate_vector_matches(args: argparse.Namespace) -> Path:
    """Apply frozen controls to visual features or one embedding per parent.

    Args:
        args (argparse.Namespace): Validated visual or centroid evaluation arguments.

    Returns:
        Path: Nine-condition scores, per-action results and figures.
    """
    design_dir = args.match_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    LOGGER.info("Loading matching design from %s.", design_dir)
    match_config, axes, design = load_matching_design(design_dir)
    if args.command == "evaluate-visual":
        visual_dir = args.visual_dir.expanduser().resolve()
        LOGGER.info("Loading prepared visual vectors from %s.", visual_dir)
        config, vectors = load_visual_vectors(visual_dir, design, axes)
        distance = args.distance_metric or config["distance_metric"]
        source_config = {
            "input_kind": "visual_features",
            "visual_dir": str(visual_dir),
            "features": config["feature_groups"]["included"],
            "excluded_pair_features": config["excluded_pair_features"],
            "grid_scope": config["grid_scope"],
            "support_aggregation": config["arguments"]["support_aggregation"],
        }
        conventions = {
            "representation": "saved visual baseline task vectors; preprocessing reused unchanged",
        }
    else:
        centroid_file = args.centroid_file.expanduser().resolve()
        LOGGER.info("Loading and centering task centroids from %s.", centroid_file)
        vectors = load_centroid_vectors(centroid_file, axes)
        distance = args.distance_metric
        source_config = {
            "input_kind": "task_centroids",
            "centroid_file": str(centroid_file),
            "model_name": args.model_name,
        }
        conventions = {
            "representation": "float64 task centroids over eight support sets; subtract the unweighted mean over all matching parents; no feature scaling",
            "support_sets": "counts checked to equal eight; archive has no per-support-set IDs, so exact support-set correspondence cannot be verified",
        }
    LOGGER.info(
        "Scoring %d parents with %d coordinates using %s distance under all nine conditions.",
        len(vectors),
        vectors.shape[1],
        distance,
    )
    with (
        staged_output_directory(output_dir, overwrite=True) as stage,
        threadpool_limits(args.num_threads),
    ):
        groups, original = evaluate_groups(
            separation_similarity(vectors, distance), design
        )
        point = aggregate_scores(groups, original, design)
        summary, families = _summaries(axes, design, point)
        write_json(stage / "summary.json", summary)
        write_json(stage / "family_summary.json", families)
        shutil.copyfile(
            design_dir / "matching_summary.json", stage / "matching_summary.json"
        )
        write_json(
            stage / "run_config.json",
            {
                "analysis": "matched_action_separation",
                **source_config,
                "match_dir": str(design_dir),
                "distance_metric": distance,
                "num_parents": len(vectors),
                "num_features": vectors.shape[1],
                "conventions": {
                    **conventions,
                    "matching": "reuse the same parent pairs, visual cutoffs and family eligibility as the model analysis",
                    "aggregation": "equal negative actions within anchor, equal anchors within target action, equal eligible target actions; overlaps retained",
                    "difference_from_original": "condition minus original convention; each uses its own eligible population",
                    "missing": "undefined comparisons omitted; undefined required target action makes the overall score undefined",
                },
            },
        )
        from arc_cd.repr_analysis.matched_visualization import plot_matched_analysis

        plot_matched_analysis(stage)
        if args.qualitative_examples:
            save_qualitative_examples(
                args.data_dir or Path(match_config["data_dir"]),
                axes,
                design,
                stage / "qualitative_examples",
                metadata_dir=args.metadata_dir or Path(match_config["metadata_dir"]),
                num_examples=args.num_qualitative_examples,
            )
    LOGGER.info("Saved controlled representation analysis to %s.", output_dir)
    return output_dir


def _score_summary(
    scores: np.ndarray, original: np.ndarray | None = None
) -> dict[str, Any]:
    """Group scores and their differences from the original convention.

    Args:
        scores (np.ndarray): Strict and tie-adjusted probabilities.
        original (np.ndarray | None): Original-convention reference probabilities.

    Returns:
        dict[str, Any]: Scores and, for controlled conditions, original differences.
    """
    result: dict[str, Any] = dict(zip(METRICS, scores))
    if original is not None:
        result["difference_from_original"] = dict(zip(METRICS, scores - original))
    return result


def _summaries(
    axes: dict[str, Any],
    design: dict[str, np.ndarray],
    point: dict[str, np.ndarray],
    *,
    family_count_key: str = "headline_actions",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summarize one representation by condition and target family.

    Args:
        axes (dict[str, Any]): Task/action names.
        design (dict[str, np.ndarray]): Matching graph and eligibility masks.
        point (dict[str, np.ndarray]): Aggregated scores and valid-anchor masks.
        family_count_key (str): Field name for the number of eligible families.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Overall and per-action summaries.
    """
    summary: dict[str, Any] = {}
    families: dict[str, Any] = {}
    for condition in range(-1, len(CONDITIONS)):
        original = condition == -1
        name = "original_figure4" if original else CONDITIONS[condition]
        included = (
            design["original_headline"] if original else design["headline"][condition]
        )
        valid = (
            point["original_valid_anchors"]
            if original
            else point["valid_anchors"][condition]
        )
        selected = included[design["anchor_families"]] & valid
        scores = point["original_macro"] if original else point["macro"][condition]
        baseline = None if original else point["original_macro"]
        summary[name] = {
            **_score_summary(scores, baseline),
            family_count_key: int(included.sum()),
            "headline_anchors": len(np.unique(design["anchor_tasks"][selected])),
        }
        family_scores = (
            point["original_family_scores"]
            if original
            else point["family_scores"][condition]
        )
        families[name] = {}
        for f, action in enumerate(axes["families"]):
            baseline = None if original else point["original_family_scores"][f]
            families[name][action] = {
                **_score_summary(family_scores[f], baseline),
                "headline": bool(included[f]),
                "valid_anchors": int((valid & (design["anchor_families"] == f)).sum()),
            }
    return summary, families


def main(argv: list[str] | None = None) -> Path:
    """Dispatch preparation, CPU evaluation or plot regeneration.

    Args:
        argv (list[str] | None): Explicit command arguments or process arguments.

    Returns:
        Path: Published artifact or figure directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Freeze model-independent visual matches."
    )
    prepare.add_argument("--data_dir", type=Path, required=True)
    prepare.add_argument("--metadata_dir", type=Path, required=True)
    prepare.add_argument("--output_dir", type=Path, required=True)
    prepare.add_argument("--min_tasks", type=int, default=10)
    prepare.add_argument("--expected_parents", type=int, default=1600)
    evaluate = commands.add_parser(
        "evaluate", help="Score saved hidden states and summarize matched comparisons."
    )
    evaluate.add_argument("--feature_dir", type=Path, required=True)
    evaluate.add_argument("--reference_analysis_dir", type=Path)
    evaluate.add_argument(
        "--positions", nargs="+", choices=("S1", "S2", "S3"), default=["S3"]
    )
    evaluate.add_argument(
        "--layers",
        default="39",
        help='Layer indices: "all", "0-63:2", or "0,31,63" (default: 39).',
    )
    visual = commands.add_parser(
        "evaluate-visual",
        help="Score saved visual-baseline vectors with the same controls.",
    )
    visual.add_argument("--visual_dir", type=Path, required=True)
    visual.add_argument(
        "--distance_metric",
        choices=DISTANCE_METRICS,
        help="Defaults to the saved visual baseline's distance metric.",
    )
    centroids = commands.add_parser(
        "evaluate-centroids",
        help="Score eight-support-set task centroids, such as VARC, with the same controls.",
    )
    centroids.add_argument("--centroid_file", type=Path, required=True)
    centroids.add_argument("--model_name", required=True)
    centroids.add_argument(
        "--distance_metric", choices=DISTANCE_METRICS, default="cosine"
    )
    for command in (evaluate, visual, centroids):
        command.add_argument("--match_dir", type=Path, required=True)
        command.add_argument("--output_dir", type=Path, required=True)
        command.add_argument(
            "--data_dir",
            type=Path,
            help="Current eight-version dataset location for qualitative examples. Defaults to the preparation path.",
        )
        command.add_argument(
            "--metadata_dir",
            type=Path,
            help="Metadata explorer for qualitative examples and, with evaluate, checking the feature support grids.",
        )
        command.add_argument("--num_threads", type=int, default=8)
        command.add_argument(
            "--qualitative_examples",
            action="store_true",
            help="Save first-variant support triplets as PNG/JSON pairs for full q25/q50/q75, near each distance band's midpoint and with varied anchor action operators.",
        )
        command.add_argument(
            "--num_qualitative_examples",
            type=int,
            default=DEFAULT_NUM_QUALITATIVE_EXAMPLES,
            help=f"Maximum examples per eligible action and condition when --qualitative_examples is enabled (default: {DEFAULT_NUM_QUALITATIVE_EXAMPLES}).",
        )
    plot = commands.add_parser("plot", help="Regenerate figures from saved summaries.")
    plot.add_argument("--analysis_dir", type=Path, required=True)
    plot.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    if (
        args.command in ("evaluate", "evaluate-visual", "evaluate-centroids")
        and args.num_qualitative_examples < 1
    ):
        parser.error("--num_qualitative_examples must be a positive integer.")
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("arc_cd.repr_analysis").setLevel(logging.INFO)
    with logging_redirect_tqdm():
        if args.command == "prepare":
            if args.min_tasks < 1 or args.expected_parents < 1:
                parser.error("Parent count and minimum role coverage must be positive.")
            return prepare_matches(
                args.data_dir,
                args.metadata_dir,
                args.output_dir,
                min_tasks=args.min_tasks,
                expected_parents=args.expected_parents,
            )
        if args.command in ("evaluate-visual", "evaluate-centroids"):
            if args.num_threads < 1:
                parser.error("Require a positive thread count.")
            return evaluate_vector_matches(args)
        if args.command == "evaluate":
            if args.num_threads < 1 or len(set(args.positions)) != len(args.positions):
                parser.error("Require positive threads and unique positions.")
            return evaluate_matches(args)
        from arc_cd.repr_analysis.matched_visualization import plot_matched_analysis

        return plot_matched_analysis(args.analysis_dir, args.output_dir)
