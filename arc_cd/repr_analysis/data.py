"""Validated support bundles and shared, layer-wise task preprocessing."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from arc_cd.probes.feature_io import load_feature_archive
from arc_cd.probes.layers import parse_layers
from arc_cd.probes.prompt_feature_io import (
    _load_shard_records,
    _validate_position_archive,
)


@dataclass
class FeatureBundle:
    """Validated manifest, extraction contract, and concept-ordered task rows.

    Attributes:
        directory (Path): Complete feature bundle.
        config (dict[str, Any]): Original extraction configuration.
        records (list[dict[str, Any]]): Aligned manifest, without large token lists.
        concepts (list[str]): Saved concept vocabulary order.
        task_ids (list[str]): Concept-scoped parent identities, grouped by concept.
        labels (np.ndarray): Integer concept index per parent task.
        rows (np.ndarray): Manifest indices, shaped [tasks, eight versions].
        layers (list[int]): Selected transformer layer indices.
    """

    directory: Path
    config: dict[str, Any]
    records: list[dict[str, Any]]
    concepts: list[str]
    task_ids: list[str]
    labels: np.ndarray
    rows: np.ndarray
    layers: list[int]

    def load_position(self, position: str) -> np.ndarray:
        """Load and validate one raw position tensor against the common manifest.

        Args:
            position (str): S1, S2, or S3.

        Returns:
            np.ndarray: Original FP16 [versions, extracted layers, hidden] tensor.
        """
        archive = load_feature_archive(self.directory / f"{position}.npz")
        _validate_position_archive(
            archive, self.records, position, self.config["extraction"]
        )
        return archive["x"]


def fingerprint(path: Path) -> dict[str, Any]:
    """Hash an input file in bounded memory.

    Args:
        path (Path): File to fingerprint.

    Returns:
        dict[str, Any]: SHA-256 and byte count.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def load_bundle(directory: Path, layer_spec: str = "all") -> FeatureBundle:
    """Validate a complete support bundle and require eight versions per parent.

    Args:
        directory (Path): Extracted-feature bundle directory.
        layer_spec (str): Existing layer syntax, or all extracted layers.

    Returns:
        FeatureBundle: Validated, deterministic parent-task grouping.
    """
    directory = directory.expanduser().resolve()
    config = json.loads((directory / "run_config.json").read_text())
    extraction = config["extraction"]
    if config["schema_version"] != 1 or config["selection"] != {"concept": None}:
        raise ValueError(
            "Analysis requires a complete dataset bundle (not a concept shard)."
        )
    if (
        extraction.get("feature_dtype") != "float16"
        or extraction.get("position_definition")
        != "token_containing_final_support_output_closing_bracket"
    ):
        raise ValueError("Unsupported extraction dtype or support-position definition.")
    for name in (config["dataset"]["id"], extraction["model_name"]):
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in (".", "..")
        ):
            raise ValueError(
                "Dataset/model identifiers must be single path components."
            )
    concepts = extraction["concepts"]
    available = extraction["layers"]
    if not concepts or len(set(concepts)) != len(concepts):
        raise ValueError("Concept vocabulary must be nonempty and unique.")
    if (
        not available
        or len(set(available)) != len(available)
        or any(type(layer) is not int or layer < 0 for layer in available)
    ):
        raise ValueError("Extracted layers must be unique nonnegative integers.")
    layers = available.copy() if layer_spec == "all" else parse_layers(layer_spec)
    if not set(layers) <= set(available):
        raise ValueError(f"Requested layers are not in the archive: {layers}.")
    records = _load_shard_records(directory)
    ids = [record["task_id"] for record in records]
    if (
        not records
        or len(records) != config["num_records"]
        or len(set(ids)) != len(ids)
        or set(ids) != set(config["dataset"]["task_hashes"])
    ):
        raise ValueError("Missing, duplicate, or unexpected task versions in manifest.")
    groups: dict[str, list[int]] = {}
    parent_concepts: dict[str, str] = {}
    for row, record in enumerate(records):
        concept, sample = record["concept"], record["sample_id"]
        match = re.fullmatch(r"(.+)_(\d+)", sample)
        parent, version = match.groups() if match else (sample, "canonical")
        parent_id = f"{concept}/{parent}"
        if (
            concept not in concepts
            or record["task_id"] != f"{concept}/{sample}"
            or record["case_id"] != f"{concept}/{sample}/support"
            or record["test_index"] != -1
            or record["metadata"].get("parent_task_id") != parent_id
            or record["metadata"].get("support_set_id") != version
            or record["metadata"].get("prompt_only") is not True
            or len(record["support_examples"]) != 3
            or any(
                not pair.get("input") or not pair.get("output")
                for pair in record["support_examples"]
            )
        ):
            raise ValueError(
                f"Invalid support identity/provenance: {record['case_id']}"
            )
        groups.setdefault(parent_id, []).append(row)
        parent_concepts[parent_id] = concept
        record.pop("prompt_token_ids", None)
        record.pop("generated_token_ids", None)
    task_ids = sorted(
        groups, key=lambda task: (concepts.index(parent_concepts[task]), task)
    )
    rows = []
    for task in task_ids:
        indices = sorted(
            groups[task], key=lambda i: records[i]["metadata"]["support_set_id"]
        )
        versions = {records[i]["metadata"]["support_set_id"] for i in indices}
        if len(indices) != 8 or len(versions) != 8:
            raise ValueError(
                f"Expected eight distinct version IDs for {task}; found {len(indices)}."
            )
        rows.append(indices)
    labels = np.asarray(
        [concepts.index(parent_concepts[task]) for task in task_ids], dtype=np.int32
    )
    if set(labels.tolist()) != set(range(len(concepts))):
        raise ValueError("Some configured concepts have no parent tasks.")
    for position in ("S1", "S2", "S3"):
        if not (directory / f"{position}.npz").is_file():
            raise ValueError(f"Missing complete-bundle position: {position}")
    return FeatureBundle(
        directory, config, records, concepts, task_ids, labels, np.asarray(rows), layers
    )


def center_versions(
    raw: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grand-center version vectors before computing unnormalized task means.

    Args:
        raw (np.ndarray): Single-layer [version rows, hidden] features.
        rows (np.ndarray): [tasks, eight versions] manifest indices.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Float64 grand mean, centered
        [tasks, versions, hidden] vectors, and [tasks, hidden] task means.
    """
    versions = np.asarray(raw[rows], dtype=np.float64)
    grand_mean = versions.mean(axis=(0, 1), dtype=np.float64)
    versions -= grand_mean
    return grand_mean, versions, versions.mean(axis=1, dtype=np.float64)
