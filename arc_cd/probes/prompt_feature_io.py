"""Transactional support-feature archives and validated concept-shard merging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from arc_cd.data_prep.dataset_ids import infer_concept_arc_dataset_id
from arc_cd.data_prep.load_data import load_concept_arc
from arc_cd.models.generation_io import (
    load_generation_jsonl,
    load_token_archive_npz,
    write_generation_outputs,
)
from arc_cd.probes.feature_io import load_feature_archive, save_feature_archive
from arc_cd.probes.hidden import HiddenStateFeatures
from arc_cd.probes.prompt_features import (
    SUPPORT_POSITIONS,
    task_inventory,
    task_support_identity,
)

LOGGER = logging.getLogger(__name__)


@contextmanager
def staged_output_directory(
    destination: Path, *, overwrite: bool = False
) -> Iterator[Path]:
    """Publish an entire output directory only after a successful run.

    Args:
        destination (Path): Final output directory.
        overwrite (bool): Warn and replace existing output after a successful run.

    Yields:
        Path: Temporary sibling directory for this run's artifacts.

    Raises:
        FileExistsError: If output exists without overwrite, or a writer is active.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    stage: Path | None = None
    try:
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Output already exists: {destination}")
            if not destination.is_dir():
                raise NotADirectoryError(f"Output is not a directory: {destination}")
            LOGGER.warning(
                "Output already exists: %s. It will be replaced after a successful run.",
                destination,
            )
        stage = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        yield stage
        previous: Path | None = None
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Output already exists: {destination}")
            previous = stage.with_name(stage.name + ".previous")
            destination.rename(previous)
        try:
            stage.rename(destination)
        except BaseException:
            if previous is not None:
                previous.rename(destination)
            raise
        if previous is not None:
            shutil.rmtree(previous)
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        lock.unlink()


def position_metadata(
    records: list[dict[str, Any]], position: str
) -> list[dict[str, Any]]:
    """Build row metadata shared by saving and alignment validation.

    Args:
        records (list[dict[str, Any]]): Aligned manifest rows.
        position (str): S1, S2 or S3.

    Returns:
        list[dict[str, Any]]: Per-feature identities and replay boundaries.
    """
    return [
        {
            **record["metadata"],
            "case_id": record["case_id"],
            "task_id": record["task_id"],
            "token_index": row,
            "token_position": position,
            "original_absolute_token_index": record["selected_token_indices"][position],
            "prefix_sequence_len": record["selected_token_indices"][position] + 1,
        }
        for row, record in enumerate(records)
    ]


def save_position_features(
    stage: Path,
    position: str,
    x: np.ndarray,
    records: list[dict[str, Any]],
    concepts: list[str],
    layers: list[int],
) -> None:
    """Save one position using the existing analysis archive format.

    Args:
        stage (Path): Output staging directory.
        position (str): Support position name.
        x (np.ndarray): Features, optionally backed by a disk memmap.
        records (list[dict[str, Any]]): Aligned manifest rows.
        concepts (list[str]): Dataset-wide ordered label vocabulary.
        layers (list[int]): Extracted layer indices.
    """
    selected = np.asarray(
        [record["selected_token_indices"][position] for record in records],
        dtype=np.int32,
    )
    labels = np.asarray(
        [
            [int(record["concept"] == concept) for concept in concepts]
            for record in records
        ],
        dtype=np.int64,
    )
    save_feature_archive(
        stage / f"{position}.npz",
        features=HiddenStateFeatures(
            x, selected, selected + 1, position_metadata(records, position)
        ),
        y=labels,
        concepts=concepts,
        layers=layers,
        records=records,
        token_position=position,
    )


def save_prompt_manifest(
    stage: Path, records: list[dict[str, Any]], run_config: dict[str, Any]
) -> None:
    """Write aligned source records, exact token sidecar and configuration.

    Args:
        stage (Path): Output staging directory.
        records (list[dict[str, Any]]): Rows containing prefix token IDs.
        run_config (dict[str, Any]): Resolved extraction and dataset snapshot.
    """
    for row, record in enumerate(records):
        record["feature_rows"] = {position: row for position in SUPPORT_POSITIONS}
    write_generation_outputs(records, stage / "manifest.jsonl", stage / "tokens.npz")
    (stage / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_prompt_feature_run(
    stage: Path,
    records: list[dict[str, Any]],
    states: Iterator[dict[str, np.ndarray]],
    run_config: dict[str, Any],
) -> None:
    """Stream all three feature tensors to disk before compressing them.

    Args:
        stage (Path): Output staging directory.
        records (list[dict[str, Any]]): Ordered support records.
        states (Iterator[dict[str, np.ndarray]]): One set of states per row.
        run_config (dict[str, Any]): Resolved extraction and dataset snapshot.

    Raises:
        ValueError: If feature rows, shapes, positions or values are invalid.
    """
    layers = run_config["extraction"]["layers"]
    concepts = run_config["extraction"]["concepts"]
    arrays: dict[str, np.memmap] = {}
    count = 0
    for row, row_states in enumerate(states):
        if row >= len(records) or set(row_states) != set(SUPPORT_POSITIONS):
            raise ValueError("Feature row/position mismatch.")
        for position in SUPPORT_POSITIONS:
            values = row_states[position]
            if (
                values.ndim != 2
                or values.shape[0] != len(layers)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"Invalid {position} feature shape or values at row {row}."
                )
            if row == 0:
                arrays[position] = np.lib.format.open_memmap(
                    stage / f".{position}.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(len(records), *values.shape),
                )
            if values.shape != arrays[position].shape[1:]:
                raise ValueError(f"Inconsistent {position} feature shape at row {row}.")
            arrays[position][row] = values
        if len({array.shape for array in arrays.values()}) != 1:
            raise ValueError("Support positions have different feature dimensions.")
        count += 1
    if count != len(records) or count == 0:
        raise ValueError(f"Expected {len(records)} feature rows, received {count}.")
    for position in SUPPORT_POSITIONS:
        array = arrays.pop(position)
        array.flush()
        save_position_features(stage, position, array, records, concepts, layers)
        del array
        (stage / f".{position}.npy").unlink()
    save_prompt_manifest(stage, records, run_config)


def _load_shard_records(directory: Path) -> list[dict[str, Any]]:
    """Validate manifest/token alignment and restore prefix token IDs.

    Args:
        directory (Path): Completed concept shard directory.

    Returns:
        list[dict[str, Any]]: Records including verified prefix token IDs.

    Raises:
        ValueError: If sidecar offsets, identities or selected positions differ.
    """
    records = load_generation_jsonl(directory / "manifest.jsonl")
    archive = load_token_archive_npz(directory / "tokens.npz")
    if tuple(record["case_id"] for record in records) != archive.case_ids:
        raise ValueError(f"Manifest/token case alignment mismatch: {directory}")
    for ragged in (archive.prompt_token_ids, archive.generated_token_ids):
        if (
            ragged.offsets.shape != (len(records) + 1,)
            or ragged.offsets.dtype.kind not in "iu"
            or ragged.data.ndim != 1
            or ragged.data.dtype.kind not in "iu"
            or ragged.offsets[0] != 0
            or ragged.offsets[-1] != len(ragged.data)
            or np.any(np.diff(ragged.offsets) < 0)
        ):
            raise ValueError(f"Invalid token offsets: {directory}")
    for row, record in enumerate(records):
        if (
            record["token_index"] != row
            or record["token_archive"] != "tokens.npz"
            or record["feature_rows"]
            != {position: row for position in SUPPORT_POSITIONS}
        ):
            raise ValueError(f"Manifest row alignment mismatch: {directory}, row {row}")
        tokens = archive.get_by_index(row)
        prefix = tokens["prompt_token_ids"]
        positions = record["selected_token_indices"]
        indices = [positions[name] for name in SUPPORT_POSITIONS]
        digest = hashlib.sha256(",".join(map(str, prefix)).encode("ascii")).hexdigest()
        if (
            tokens["generated_token_ids"]
            or not prefix
            or indices != sorted(set(indices))
            or indices[0] < 0
            or indices[-1] != len(prefix) - 1
            or digest != record["prefix_sha256"]
        ):
            raise ValueError(f"Invalid support prefix: {directory}, row {row}")
        for number, position in enumerate(SUPPORT_POSITIONS, start=1):
            index = positions[position]
            key = f"support_{number}_output_closing_bracket"
            start, end = record["prompt_char_spans"][key]
            if (
                record["token_spans"][key] != [index, index + 1]
                or record["prompt"][start:end] != "]"
                or record["selected_token_ids"][position] != prefix[index]
            ):
                raise ValueError(
                    f"Bracket/token alignment mismatch: {directory}, row {row}"
                )
        record.update(tokens)
    return records


def _validate_position_archive(
    archive: dict[str, Any],
    records: list[dict[str, Any]],
    position: str,
    extraction: dict[str, Any],
) -> None:
    """Verify feature identity, label and boundary arrays against the manifest.

    Args:
        archive (dict[str, Any]): Loaded position archive.
        records (list[dict[str, Any]]): Corresponding shard manifest.
        position (str): Expected support position.
        extraction (dict[str, Any]): Common extraction configuration.

    Raises:
        ValueError: If feature shapes, values or alignment are inconsistent.
    """
    concepts, layers = extraction["concepts"], extraction["layers"]
    selected = [record["selected_token_indices"][position] for record in records]
    expected = {
        "concepts": concepts,
        "layers": layers,
        "token_position": position,
        "case_ids": [record["case_id"] for record in records],
        "task_ids": [record["task_id"] for record in records],
        "primary_concepts": [record["concept"] for record in records],
        "parent_task_ids": [record["metadata"]["parent_task_id"] for record in records],
        "support_set_ids": [record["metadata"]["support_set_id"] for record in records],
        "test_indices": [-1] * len(records),
        "selected_token_indices": selected,
        "sequence_lengths": [index + 1 for index in selected],
        "parse_status": ["prompt_only"] * len(records),
        "y": [
            [int(record["concept"] == concept) for concept in concepts]
            for record in records
        ],
    }
    for key, values in expected.items():
        if key not in archive or not np.array_equal(archive[key], np.asarray(values)):
            raise ValueError(f"{position} feature alignment mismatch: {key}")
    metadata = [json.loads(value) for value in archive["metadata_json"].tolist()]
    if metadata != position_metadata(records, position):
        raise ValueError(f"{position} feature metadata alignment mismatch.")
    x = archive["x"]
    if (
        x.ndim != 3
        or x.shape[:2] != (len(records), len(layers))
        or x.shape[2] == 0
        or x.dtype != np.float16
        or not np.isfinite(x).all()
    ):
        raise ValueError(f"Invalid {position} feature tensor.")


def _prompt_feature_fingerprints(directory: Path) -> dict[str, dict[str, str | int]]:
    """Hash source artifacts for provenance and mutation detection.

    Args:
        directory (Path): Complete feature bundle or concept shard.

    Returns:
        dict[str, dict[str, str | int]]: SHA-256 and byte count per file.
    """
    result: dict[str, dict[str, str | int]] = {}
    for name in (
        "run_config.json",
        "manifest.jsonl",
        "tokens.npz",
        *(f"{position}.npz" for position in SUPPORT_POSITIONS),
    ):
        path = directory / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result[name] = {"sha256": digest, "bytes": path.stat().st_size}
    return result


def merge_prompt_feature_shards(
    data_dir: Path,
    shard_dir: Path,
    output_dir: Path | None = None,
    *,
    base_feature_dir: Path | None = None,
) -> Path:
    """Merge concept shards, optionally extending a complete earlier dataset.

    A base bundle must cover whole concepts whose source files are unchanged.
    Shards must cover exactly the remaining concepts in the current dataset.
    Only the concept vocabulary may differ between extraction configurations.
    Feature values and prefix tokens are preserved in canonical dataset order.

    Args:
        data_dir (Path): Current source ConceptARC dataset.
        shard_dir (Path): Directory containing the required concept shards.
        output_dir (Path | None): Final directory; defaults to sibling features
            for shard-only merges. Required when extending a base bundle.
        base_feature_dir (Path | None): Complete bundle for an earlier subset
            of the current dataset's concepts.

    Returns:
        Path: Published merged directory. All source artifacts are retained.

    Raises:
        ValueError: If coverage, configurations, sources or aligned arrays differ,
            or a base bundle is supplied without an explicit output directory.
        FileExistsError: If merged output already exists.
    """
    if base_feature_dir is not None and output_dir is None:
        raise ValueError("output_dir is required when extending a base feature bundle.")
    destination = (
        (output_dir if output_dir is not None else shard_dir.parent / "features")
        .expanduser()
        .resolve()
    )
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    tasks = load_concept_arc(data_dir)
    if not tasks:
        raise ValueError("No ConceptARC tasks found.")
    concepts = sorted({task.concept for task in tasks})
    inventory = task_inventory(tasks)
    dataset_id = infer_concept_arc_dataset_id(data_dir)
    sources: list[tuple[Path, str | None, dict[str, Any]]] = []
    base_concepts: list[str] = []
    if base_feature_dir is not None:
        base_feature_dir = base_feature_dir.expanduser().resolve()
        base_config = json.loads((base_feature_dir / "run_config.json").read_text())
        base_concepts = base_config["extraction"]["concepts"]
        if (
            not isinstance(base_concepts, list)
            or not base_concepts
            or any(not isinstance(concept, str) for concept in base_concepts)
            or base_concepts != sorted(set(base_concepts))
            or not set(base_concepts) < set(concepts)
        ):
            raise ValueError(
                "Base concepts must be a nonempty sorted subset of the expanded dataset."
            )
        sources.append((base_feature_dir, None, base_config))
    expected_concepts = sorted(set(concepts) - set(base_concepts))
    found = sorted(
        path.name
        for path in shard_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if found != expected_concepts:
        raise ValueError(
            f"Concept shard coverage mismatch: expected {expected_concepts}, found {found}."
        )
    for concept in expected_concepts:
        directory = (shard_dir / concept).expanduser().resolve()
        config = json.loads((directory / "run_config.json").read_text())
        sources.append((directory, concept, config))
    task_by_id = {task.task_id: task for task in tasks}
    source_records: dict[Path, list[dict[str, Any]]] = {}
    source_fingerprints: dict[Path, dict[str, dict[str, str | int]]] = {}
    common: dict[str, Any] | None = None
    for directory, selection, config in sources:
        selected_concepts = base_concepts if selection is None else [selection]
        source_concepts = base_concepts if selection is None else concepts
        source_inventory = (
            {
                task.task_id: inventory[task.task_id]
                for task in tasks
                if task.concept in base_concepts
            }
            if selection is None
            else inventory
        )
        extraction = {**config["extraction"], "concepts": concepts}
        if common is None:
            common = {
                "schema_version": 1,
                "extraction": extraction,
                "dataset": {"id": dataset_id, "task_hashes": inventory},
            }
        if (
            extraction != common["extraction"]
            or config["schema_version"] != 1
            or config["selection"] != {"concept": selection}
            or config["dataset"] != {"id": dataset_id, "task_hashes": source_inventory}
            or config["extraction"]["concepts"] != source_concepts
        ):
            raise ValueError(
                f"Incompatible configuration or changed dataset: {directory}"
            )
        source_fingerprints[directory] = _prompt_feature_fingerprints(directory)
        if json.loads((directory / "run_config.json").read_text()) != config:
            raise ValueError(f"Source configuration changed during merge: {directory}")
        records = _load_shard_records(directory)
        expected_ids = [
            task.task_id for task in tasks if task.concept in selected_concepts
        ]
        if [record["task_id"] for record in records] != expected_ids or config[
            "num_records"
        ] != len(records):
            raise ValueError(
                f"Missing, duplicate or unordered task versions: {directory}"
            )
        for record in records:
            task = task_by_id[record["task_id"]]
            identity = task_support_identity(task)
            if (
                record["case_id"] != f"{task.task_id}/support"
                or record["sample_id"] != task.sample_id
                or record["concept"] != task.concept
                or record["test_index"] != -1
                or record["metadata"].get("prompt_only") is not True
                or any(
                    record["metadata"].get(key) != value
                    for key, value in identity.items()
                )
                or record["support_examples"]
                != [{"input": pair.input, "output": pair.output} for pair in task.train]
            ):
                raise ValueError(
                    f"Source support identity mismatch: {record['case_id']}"
                )
        source_records[directory] = records
    assert common is not None
    records_by_id = {
        record["task_id"]: record
        for records in source_records.values()
        for record in records
    }
    combined = [records_by_id[task.task_id] for task in tasks]
    row_by_id = {record["task_id"]: row for row, record in enumerate(combined)}
    extraction = common["extraction"]
    hidden_size: int | None = None
    with staged_output_directory(destination) as stage, ExitStack() as maps:
        for position in SUPPORT_POSITIONS:
            merged: np.memmap | None = None
            for directory, _, config in sources:
                archive = load_feature_archive(directory / f"{position}.npz")
                records = source_records[directory]
                _validate_position_archive(
                    archive, records, position, config["extraction"]
                )
                x = archive["x"]
                if hidden_size is None:
                    hidden_size = x.shape[2]
                if x.shape[2] != hidden_size:
                    raise ValueError("Incompatible feature hidden dimensions.")
                if merged is None:
                    merged = np.lib.format.open_memmap(
                        stage / f".{position}.npy",
                        mode="w+",
                        dtype=np.float16,
                        shape=(len(combined), len(extraction["layers"]), hidden_size),
                    )
                    maps.callback(merged._mmap.close)
                rows = [row_by_id[record["task_id"]] for record in records]
                merged[rows] = x
                del x, archive
            assert merged is not None
            merged.flush()
            save_position_features(
                stage, position, merged, combined, concepts, extraction["layers"]
            )
            merged._mmap.close()
            del merged
            (stage / f".{position}.npy").unlink()
        save_prompt_manifest(
            stage,
            combined,
            {
                **common,
                "selection": {"concept": None},
                "num_records": len(combined),
                "merged_from": [str(directory) for directory, _, _ in sources],
                "merge_sources": [
                    {
                        "directory": str(directory),
                        "config": config,
                        "fingerprints": source_fingerprints[directory],
                    }
                    for directory, _, config in sources
                ],
            },
        )
        for directory, _, _ in sources:
            if (
                _prompt_feature_fingerprints(directory)
                != source_fingerprints[directory]
            ):
                raise ValueError(
                    f"Source feature artifacts changed during merge: {directory}"
                )
        if task_inventory(load_concept_arc(data_dir)) != inventory:
            raise ValueError("Source dataset changed during merge.")
    return destination
