"""Stage a metadata-enriched explorer without replaying models or fitting maps."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from arc_cd.probes.prompt_feature_io import staged_output_directory
from arc_cd.repr_analysis.data import fingerprint
from arc_cd.repr_analysis.explorer import load_explorer_data, write_explorer
from arc_cd.repr_analysis.results import software_versions, write_json
from arc_cd.repr_analysis.spec_metadata import (
    DEFAULT_TAXONOMY,
    load_spec_metadata,
    load_taxonomy,
)


def run_metadata_explorer(argv: list[str] | None = None) -> Path:
    """Join task specs to an embedding, inheriting its selected population.

    Args:
        argv (list[str] | None): Explicit CLI arguments or process arguments.

    Returns:
        Path: Published metadata_explorer directory.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        required=True,
        help="Existing dimensionality_reduction results; inherits their selected population.",
    )
    parser.add_argument("--spec_dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Defaults to metadata_explorer beside the input analysis.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    started = time.perf_counter()
    source, specs, taxonomy_path = (
        path.expanduser().resolve()
        for path in (args.analysis_dir, args.spec_dir, args.taxonomy)
    )
    destination = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else source.parent / "metadata_explorer"
    )
    source_config = json.loads((source / "run_config.json").read_text())
    if (
        source_config.get("analysis") != "dimensionality_reduction"
        or source_config.get("schema_version") != 1
    ):
        raise ValueError(
            "--analysis_dir must contain completed dimensionality-reduction results."
        )
    display = load_explorer_data(source)
    axes = display["axes"]
    del display
    paths = [
        Path("axes.json"),
        Path("supports.json"),
        *(
            Path(p) / f"embedding_layer_{layer}.npz"
            for p in axes["positions"]
            for layer in axes["layers"]
        ),
    ]
    input_fingerprints = {
        path.as_posix(): fingerprint(source / path)
        for path in [Path("run_config.json"), *paths]
    }
    taxonomy_fingerprint = fingerprint(taxonomy_path)
    taxonomy = load_taxonomy(taxonomy_path)
    bundle = load_spec_metadata(
        specs, axes["task_ids"], axes["concepts"], axes["labels"], taxonomy
    )
    # Validate copied assets through the same loader used for regeneration.
    logging.info(
        "Matched %d parents; %d unused specs; %d unmapped operation entries.",
        len(bundle.records),
        len(bundle.inventory["unused_task_ids"]),
        len(bundle.inventory["unmapped_operators"]),
    )
    with staged_output_directory(destination) as stage:
        for relative in paths:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
            if fingerprint(target) != input_fingerprints[relative.as_posix()]:
                raise ValueError(f"Embedding input changed while copying: {relative}")
        shutil.copyfile(taxonomy_path, stage / "taxonomy.json")
        if fingerprint(stage / "taxonomy.json") != taxonomy_fingerprint:
            raise ValueError("Taxonomy changed during annotation.")
        write_json(
            stage / "task_metadata.json",
            {
                "schema_version": 1,
                "taxonomy_version": taxonomy["version"],
                "taxonomy": taxonomy,
                "records": bundle.records,
                "fields": bundle.fields,
            },
        )
        write_json(stage / "metadata_inventory.json", bundle.inventory)
        config: dict[str, Any] = {
            "schema_version": 1,
            "analysis": "metadata_explorer",
            "arguments": {
                "analysis_dir": str(source),
                "spec_dir": str(specs),
                "taxonomy": str(taxonomy_path),
                "output_dir": str(destination),
            },
            "embedding_arguments": source_config["arguments"],
            "extraction_config": source_config["extraction_config"],
            **(
                {"population": source_config["population"]}
                if "population" in source_config
                else {}
            ),
            "source_fingerprints": input_fingerprints,
            "feature_fingerprints": source_config["fingerprints"],
            "spec_fingerprints": bundle.fingerprints,
            "taxonomy_fingerprint": taxonomy_fingerprint,
            "taxonomy_version": taxonomy["version"],
            "software": software_versions(),
            "conventions": {
                "unit": "one parent task; metadata inherited by all eight variants",
                "families": "explicit overlapping tags; both conditional branches describe declared operations",
                "filters": "OR within fields; AND across fields; original coordinates and clusters retained",
                "composition": "unique selected parents per tag versus visible parents; descriptive only",
            },
        }
        write_json(stage / "run_config.json", config)
        write_explorer(stage)
        for relative_name, expected in input_fingerprints.items():
            if fingerprint(source / relative_name) != expected:
                raise ValueError(
                    f"Source changed during metadata annotation: {relative_name}"
                )
        for relative_name, expected in bundle.fingerprints.items():
            if fingerprint(specs / relative_name) != expected:
                raise ValueError(
                    f"Spec changed during metadata annotation: {relative_name}"
                )
        if fingerprint(taxonomy_path) != taxonomy_fingerprint:
            raise ValueError("Taxonomy changed during metadata annotation.")
        config["elapsed_seconds"] = time.perf_counter() - started
        write_json(stage / "run_config.json", config)
    logging.info(
        "Published %s in %.1f seconds.", destination, time.perf_counter() - started
    )
    return destination
