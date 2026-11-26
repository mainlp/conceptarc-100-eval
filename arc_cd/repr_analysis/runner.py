"""Shared command-line execution, preprocessing, and staged analysis publication."""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from arc_cd.probes.prompt_feature_io import staged_output_directory
from arc_cd.repr_analysis import cosine_distance
from arc_cd.repr_analysis.data import center_versions, fingerprint, load_bundle
from arc_cd.repr_analysis.geometry import (
    DISTANCE_METRICS,
    separation_similarity,
    summarize,
)
from arc_cd.repr_analysis.population import POPULATIONS, select_population
from arc_cd.repr_analysis.results import (
    representation_arrays,
    save_axes,
    software_versions,
    write_json,
)


def parse_arguments(analysis: str, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse shared options and analysis-specific statistical controls.

    Args:
        analysis (str): Name of the requested representation analysis.
        argv (list[str] | None): Explicit argument list, or process arguments.

    Returns:
        argparse.Namespace: Validated CLI options.
    """
    parser = argparse.ArgumentParser(
        description=f"ConceptARC {'separation probability' if analysis == 'cosine_distance' else analysis.replace('_', ' ')} from task representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--feature_dir", type=Path)
    if analysis == "cosine_distance":
        parser.add_argument(
            "--distance_metric",
            choices=DISTANCE_METRICS,
            default="cosine",
            help="Task separation metric; Euclidean uses unnormalized task means.",
        )
        inputs.add_argument(
            "--centroid_file",
            type=Path,
            help="Task-centroid NPZ; requires --metadata_dir and --model_name.",
        )
        parser.add_argument(
            "--model_name", help="Output model identifier for task-centroid inputs."
        )
    parser.add_argument("--output_root", type=Path, default=Path("output"))
    parser.add_argument("--run_name", default="baseline")
    parser.add_argument(
        "--positions", nargs="+", choices=("S1", "S2", "S3"), default=None
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="all, 0,31,63, or 0-63:2; original transformer indices. For dimensionality reduction, representative selects first/middle/last.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="CPU BLAS threads; no GPU is required.",
    )
    parser.add_argument(
        "--population",
        choices=POPULATIONS,
        default="all",
        help="Select complete predicate/action concept groups, or every task in the supplied bundle.",
    )
    if analysis == "cosine_distance":
        parser.add_argument(
            "--pdf",
            action="store_true",
            help="Also export static figures as PDFs (automatic for metadata separation plots).",
        )
        parser.add_argument(
            "--metadata_dir",
            type=Path,
            default=None,
            help="Published metadata explorer; enables overlapping-family analyses.",
        )
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
        parser.add_argument(
            "--include_structural_selectors",
            action="store_true",
            help="Include taxonomy-marked structural selector tags (extraction, set operations, and count measurements) in headline averages. Detailed memberships and family statistics always retain them.",
        )
    if analysis == "dimensionality_reduction":
        parser.set_defaults(layers="representative")
        parser.add_argument(
            "--methods", nargs="+", choices=("umap", "pca"), default=["umap", "pca"]
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--n_neighbors", type=int, default=15)
        parser.add_argument("--min_dist", type=float, default=0.1)
        parser.add_argument("--min_cluster_size", type=int, default=10)
        parser.add_argument("--min_samples", type=int, default=5)
        parser.add_argument(
            "--cluster_selection_method", choices=("eom", "leaf"), default="eom"
        )
    args = parser.parse_args(argv)
    centroid_input = getattr(args, "centroid_file", None) is not None
    if centroid_input:
        if args.metadata_dir is None or not args.model_name:
            parser.error("--centroid_file requires --metadata_dir and --model_name.")
        if Path(args.model_name).name != args.model_name or args.model_name in (
            ".",
            "..",
        ):
            parser.error("--model_name must be a single directory name.")
        if args.positions is not None or args.layers != "all":
            parser.error(
                "Task centroids have no selectable support positions or layers."
            )
    elif getattr(args, "model_name", None) is not None:
        parser.error("--model_name is only used with --centroid_file.")
    if args.positions is None:
        args.positions = [] if centroid_input else ["S1", "S2", "S3"]
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        parser.error("--run_name must be a single directory name.")
    if len(set(args.positions)) != len(args.positions):
        parser.error("--positions must be unique.")
    if args.num_threads < 1:
        parser.error("--num_threads must be positive.")
    if analysis == "dimensionality_reduction":
        from arc_cd.repr_analysis.embedding import EmbeddingOptions

        try:
            EmbeddingOptions(
                methods=tuple(args.methods),
                seed=args.seed,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                cluster_selection_method=args.cluster_selection_method,
            )
        except ValueError as error:
            parser.error(str(error))
    if (
        getattr(args, "include_structural_selectors", False)
        and args.metadata_dir is None
    ):
        parser.error("--include_structural_selectors requires --metadata_dir.")
    if getattr(args, "metadata_dir", None) is not None:
        if args.min_family_tasks < 1:
            parser.error("--min_family_tasks must be positive.")
        if len(set(args.family_roles)) != len(args.family_roles) or len(
            set(args.comparison_scopes)
        ) != len(args.comparison_scopes):
            parser.error("--family_roles and --comparison_scopes must be unique.")
    return args


def run_analysis(analysis: str, argv: list[str] | None = None) -> Path:
    """Run one analysis with common validated preprocessing and result contracts.

    Args:
        analysis (str): Use ``cosine_distance`` for concept or metadata-family separation.
        argv (list[str] | None): Optional explicit CLI arguments.

    Returns:
        Path: Published sub-analysis directory.
    """
    if analysis != "cosine_distance":
        raise ValueError(f"Unknown analysis: {analysis}")
    args = parse_arguments(analysis, argv)
    output_analysis = f"{args.distance_metric}_distance"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if getattr(args, "centroid_file", None) is not None:
        from arc_cd.repr_analysis.centroid_runner import run_centroid_analysis

        return run_centroid_analysis(args)
    if getattr(args, "metadata_dir", None) is not None:
        from arc_cd.repr_analysis.family_runner import run_metadata_analysis

        return run_metadata_analysis(args)
    started = time.perf_counter()
    bundle = load_bundle(args.feature_dir, args.layers)
    population = select_population(bundle, args.population)
    n_tasks, n_concepts = len(population.task_ids), len(population.concepts)
    logging.info(
        "Selected %s population: %d versions / %d parents / %d concepts; %s, layers %s.",
        population.name,
        len(population.records),
        n_tasks,
        n_concepts,
        args.positions,
        bundle.layers,
    )
    destination = (
        args.output_root.expanduser().resolve()
        / bundle.config["dataset"]["id"]
        / bundle.config["extraction"]["model_name"]
        / "repr_analysis"
        / args.run_name
        / population.name
        / output_analysis
    )
    metrics = cosine_distance.METRICS
    input_paths = [
        bundle.directory / name
        for name in (
            "run_config.json",
            "manifest.jsonl",
            "tokens.npz",
            *(f"{p}.npz" for p in args.positions),
        )
    ]
    stats = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in input_paths
    }
    with (
        staged_output_directory(destination) as stage,
        ExitStack() as output_maps,
        threadpool_limits(limits=args.num_threads),
    ):
        config: dict[str, Any] = {
            "schema_version": 1,
            "analysis": output_analysis,
            "distance_metric": args.distance_metric,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "feature_dir": str(bundle.directory),
            "extraction_config": bundle.config,
            "population": population.snapshot(),
            "fingerprints": {path.name: fingerprint(path) for path in input_paths},
            "software": software_versions(),
            "conventions": {
                "versions_per_task": 8,
                "centering": "mean over selected support-set versions at each position and layer",
                "normalization": (
                    "none; Euclidean distance between task means without unit normalization or feature scaling"
                    if output_analysis == "euclidean_distance"
                    else "only for cosine, after task/concept averaging"
                ),
                "calculation_dtype": "float64",
                "representation_storage_dtype": "float32",
                "undefined": "NaN in numerical arrays; null in JSON; no substitution; macro requires all selected concepts",
                "duplicate_filtering": False,
            },
        }
        save_axes(
            stage,
            bundle,
            args.positions,
            metrics,
            population=population,
            distance_metric=getattr(args, "distance_metric", None),
        )
        all_results: dict[str, list[np.ndarray]] = {}
        hidden_size = None
        for position in args.positions:
            logging.info("Loading and validating %s features.", position)
            raw = bundle.load_position(position)
            if hidden_size is not None and raw.shape[2] != hidden_size:
                raise ValueError("Hidden dimensions differ across positions.")
            hidden_size = raw.shape[2]
            representations = representation_arrays(
                stage, position, bundle, hidden_size, output_maps, population=population
            )
            result_layers: dict[str, list[np.ndarray]] = {}
            for layer_index, layer in enumerate(bundle.layers):
                raw_index = bundle.config["extraction"]["layers"].index(layer)
                mean, versions, task_means = center_versions(
                    raw[:, raw_index, :], population.source_rows
                )
                centroids = np.asarray(
                    [
                        task_means[population.labels == c].mean(axis=0)
                        for c in range(n_concepts)
                    ]
                )
                representations["grand_means"][layer_index] = mean
                representations["task_means"][:, layer_index] = task_means
                representations["concept_means"][:, layer_index] = centroids
                similarity = separation_similarity(task_means, args.distance_metric)
                scores, details = cosine_distance.separation_probability(
                    similarity, population.labels
                )

                result = {
                    "task_scores": scores,
                    **details,
                    **summarize(scores, population.labels, n_concepts),
                }
                for key, value in result.items():
                    result_layers.setdefault(key, []).append(value)
                del versions, task_means
                if layer_index % 8 == 0 or layer_index == len(bundle.layers) - 1:
                    logging.info(
                        "%s %s: layer %d (%d/%d), %.1fs elapsed.",
                        output_analysis,
                        position,
                        layer,
                        layer_index + 1,
                        len(bundle.layers),
                        time.perf_counter() - started,
                    )
            for array in representations.values():
                array.flush()
                array._mmap.close()
            del array, raw, representations
            for key, values in result_layers.items():
                all_results.setdefault(key, []).append(np.stack(values))
        results = {key: np.stack(values) for key, values in all_results.items()}
        if results:
            np.savez_compressed(stage / "metrics.npz", allow_pickle=False, **results)
            write_json(
                stage / "summary.json",
                {
                    "metrics": metrics,
                    "positions": args.positions,
                    "layers": bundle.layers,
                    "macro": results["macro"],
                    "concept_mean": results["concept_mean"],
                    "valid_concepts": results["valid_concepts"],
                    "chance": results.get("chance"),
                },
            )
        for path in input_paths:
            if stats[str(path)] != (path.stat().st_size, path.stat().st_mtime_ns):
                raise ValueError(f"Input changed during analysis: {path}")
        config["elapsed_seconds_before_plotting"] = time.perf_counter() - started
        write_json(stage / "run_config.json", config)
        from arc_cd.repr_analysis.visualization import plot_analysis

        plot_analysis(stage, pdf=args.pdf)
        config["elapsed_seconds"] = time.perf_counter() - started
        write_json(stage / "run_config.json", config)
    logging.info(
        "Published %s in %.1fs: %s",
        output_analysis,
        time.perf_counter() - started,
        destination,
    )
    return destination
