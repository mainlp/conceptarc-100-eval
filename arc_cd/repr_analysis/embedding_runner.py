"""Independent, staged dimensionality-reduction runs from validated support features."""

from __future__ import annotations

import json
import logging
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from arc_cd.probes.prompt_feature_io import staged_output_directory
from arc_cd.repr_analysis.data import center_versions, fingerprint, load_bundle
from arc_cd.repr_analysis.embedding import (
    EmbeddingOptions,
    embed_tasks,
    representative_versions,
    within_task_dispersion,
)
from arc_cd.repr_analysis.explorer import save_explorer_supports, write_explorer
from arc_cd.repr_analysis.population import select_population
from arc_cd.repr_analysis.results import (
    representation_arrays,
    save_axes,
    software_versions,
    write_json,
)
from arc_cd.repr_analysis.runner import parse_arguments


def run_embedding(argv: list[str] | None = None) -> Path:
    """Save task maps, original-space groups, and a standalone HTML explorer.

    Args:
        argv (list[str] | None): Explicit CLI arguments or process arguments.

    Returns:
        Path: Published dimensionality_reduction directory.
    """
    args = parse_arguments("dimensionality_reduction", argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    started = time.perf_counter()
    bundle = load_bundle(
        args.feature_dir, "all" if args.layers == "representative" else args.layers
    )
    if args.layers == "representative":
        bundle.layers = sorted(
            {
                bundle.layers[0],
                bundle.layers[(len(bundle.layers) - 1) // 2],
                bundle.layers[-1],
            }
        )
    population = select_population(bundle, args.population)
    options = EmbeddingOptions(
        methods=tuple(args.methods),
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
    )
    destination = (
        args.output_root.expanduser().resolve()
        / bundle.config["dataset"]["id"]
        / bundle.config["extraction"]["model_name"]
        / "repr_analysis"
        / args.run_name
        / population.name
        / "dimensionality_reduction"
    )
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
        path: (path.stat().st_size, path.stat().st_mtime_ns) for path in input_paths
    }
    logging.info(
        "Selected %s population: %d parent tasks; %s at layers %s.",
        population.name,
        len(population.task_ids),
        args.positions,
        bundle.layers,
    )
    with (
        staged_output_directory(destination) as stage,
        ExitStack() as resources,
        threadpool_limits(limits=args.num_threads),
    ):
        config: dict[str, Any] = {
            "schema_version": 1,
            "analysis": "dimensionality_reduction",
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
                "point_unit": "parent_task_mean",
                "versions_per_task": 8,
                "centering": "grand mean of selected population version rows per position/layer, followed by task averaging",
                "normalization": "unit norm after averaging; no per-feature standardization",
                "pca": "full SVD, with PCA recentering of unit-normalized task means",
                "umap": "original cosine distances; seeded random initialization; one job",
                "clustering": "HDBSCAN on original cosine distances, independently per position/layer; no concept labels",
                "cluster_ids": "local to position/layer, ordered by first task; -1 noise, -2 undefined cosine",
                "representative_variant": "maximum cosine to centered task mean; first tie in saved version order",
                "undefined": "missing coordinates for zero norms, no spread, or too few vectors; no fabricated projections",
                "calculation_dtype": "float64 preprocessing/PCA/clustering; UMAP computes its layout in float32",
                "duplicate_filtering": False,
                "interpretation": "exploratory groups; projected distances and cluster labels are not statistical evidence",
            },
        }
        save_axes(stage, bundle, args.positions, (), population=population)
        axes_path = stage / "axes.json"
        axes = json.loads(axes_path.read_text())
        axes["methods"] = list(options.methods)
        axes["array_axes"].update(
            {
                "coordinates": ["method", "task", "xy"],
                "cluster_labels": ["task"],
                "cluster_strengths": ["task"],
                "representative_versions": ["task"],
                "within_task_dispersion": ["task"],
            }
        )
        write_json(axes_path, axes)
        save_explorer_supports(stage, population.records)
        summaries = []
        hidden_size = None
        for position in args.positions:
            logging.info("Loading and validating %s features.", position)
            raw = bundle.load_position(position)
            if hidden_size is not None and raw.shape[2] != hidden_size:
                raise ValueError("Hidden dimensions differ across positions.")
            hidden_size = raw.shape[2]
            representations = representation_arrays(
                stage, position, bundle, hidden_size, resources, population=population
            )
            for index, layer in enumerate(bundle.layers):
                raw_index = bundle.config["extraction"]["layers"].index(layer)
                mean, versions, task_means = center_versions(
                    raw[:, raw_index, :], population.source_rows
                )
                representations["grand_means"][index] = mean
                representations["task_means"][:, index] = task_means
                representations["concept_means"][:, index] = np.asarray(
                    [
                        task_means[population.labels == label].mean(axis=0)
                        for label in range(len(population.concepts))
                    ]
                )
                logging.info(
                    "Embedding %s layer %d (%s).",
                    position,
                    layer,
                    ", ".join(options.methods),
                )
                result = embed_tasks(task_means, options)
                result["representative_versions"] = representative_versions(
                    versions, task_means
                )
                result["within_task_dispersion"] = within_task_dispersion(versions)
                np.savez_compressed(
                    stage / position / f"embedding_layer_{layer}.npz",
                    allow_pickle=False,
                    **result,
                )
                labels, sizes = np.unique(result["cluster_labels"], return_counts=True)
                summaries.append(
                    {
                        "position": position,
                        "layer": layer,
                        "cluster_counts": {
                            str(label): int(size) for label, size in zip(labels, sizes)
                        },
                        "projected_tasks": {
                            method: int(
                                np.isfinite(result["coordinates"][m]).all(axis=1).sum()
                            )
                            for m, method in enumerate(options.methods)
                        },
                        "effective_n_neighbors": result["effective_n_neighbors"],
                        "pca_explained_variance_ratio": result[
                            "pca_explained_variance_ratio"
                        ],
                    }
                )
            for array in representations.values():
                array.flush()
                getattr(array, "_mmap").close()
            del raw, versions, task_means
        for path, previous in stats.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != previous:
                raise ValueError(f"Input changed during analysis: {path}")
        write_json(stage / "summary.json", summaries)
        write_json(stage / "run_config.json", config)
        write_explorer(stage)
        config["elapsed_seconds"] = time.perf_counter() - started
        write_json(stage / "run_config.json", config)
    logging.info(
        "Published %s in %.1f seconds.", destination, time.perf_counter() - started
    )
    return destination
