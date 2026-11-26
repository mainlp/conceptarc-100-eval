"""Validate and merge canonical ConceptARC generation shards."""

from __future__ import annotations

import copy
import json
import logging
import re
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from arc_cd.data_prep.concept_arc import ConceptARCTestCase
from arc_cd.data_prep.dataset_ids import infer_concept_arc_dataset_id
from arc_cd.data_prep.load_data import iter_test_cases, load_concept_arc
from arc_cd.models.generation_io import (
    TokenArchive,
    load_generation_jsonl,
    load_token_archive_npz,
)
from arc_cd.prompt_prep.load_prompt import build_prompt_record
from arc_cd.prompt_prep.prompt_format import PromptTemplates
from arc_cd.utils import DEFAULT_CONCEPT_ARC_PROMPT_DIR, load_json, save_json


def _load_shard_config(path: Path) -> dict[str, Any]:
    """Read the settings shared by generation and merge.

    Args:
        path (Path): Saved generation configuration.

    Returns:
        dict[str, Any]: Configuration with validated settings and path sections.
    """
    config = load_json(path)
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("config"), dict)
        or any(
            not isinstance(config["config"].get(section), dict)
            for section in ("model", "generation", "reasoning", "chat")
        )
        or not isinstance(config.get("paths"), dict)
    ):
        raise ValueError(f"Invalid shard configuration: {path}")
    return config


def _load_concept_shard(
    directory: Path,
    run_name: str,
    cases: list[ConceptARCTestCase],
    data_dir: Path,
    config: dict[str, Any],
    templates: PromptTemplates,
) -> tuple[list[dict[str, Any]], TokenArchive]:
    """Validate dataset identity, relative source paths, queries, and tokens.

    Args:
        directory (Path): One concept's shard directory.
        run_name (str): Common artifact stem.
        cases (list[ConceptARCTestCase]): Expected cases in canonical order.
        data_dir (Path): Resolved source dataset directory.
        config (dict[str, Any]): Saved shard configuration.
        templates (PromptTemplates): Templates used to check the archived messages.

    Returns:
        tuple[list[dict[str, Any]], TokenArchive]: Ordered records and their token archive.

    Raises:
        ValueError: If a shard is incomplete or its metadata/tokens disagree.
        FileNotFoundError: If a required artifact is missing.
    """
    jsonl_path = directory / f"{run_name}_generations.jsonl"
    token_path = directory / f"{run_name}_tokens.npz"
    selection = config.get("data_selection")
    if (
        not isinstance(selection, dict)
        or set(selection) != {"data_dir", "concept"}
        or selection["concept"] != directory.name
        or not isinstance(selection["data_dir"], str)
        or not Path(selection["data_dir"]).is_absolute()
    ):
        raise ValueError(f"Invalid shard data selection: {directory}")
    saved_data_dir = Path(selection["data_dir"])
    if infer_concept_arc_dataset_id(saved_data_dir) != infer_concept_arc_dataset_id(data_dir):
        raise ValueError(
            f"Shard dataset variant differs from the requested dataset: "
            f"{saved_data_dir} versus {data_dir}"
        )

    records = load_generation_jsonl(jsonl_path)
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or case_id in records_by_id:
            raise ValueError(
                f"Missing or duplicate case_id in {jsonl_path}: {case_id!r}"
            )
        records_by_id[case_id] = record
    if set(records_by_id) != {case.case_id for case in cases} or config.get(
        "num_records"
    ) != len(cases):
        raise ValueError(
            f"Shard does not cover exactly the expected queries: {jsonl_path}"
        )

    archive = load_token_archive_npz(token_path)
    if len(archive.case_ids) != len(cases):
        raise ValueError(f"Token archive case count mismatch: {token_path}")
    for tokens in (archive.prompt_token_ids, archive.generated_token_ids):
        offsets = tokens.offsets
        if (
            tokens.data.ndim != 1
            or not np.issubdtype(tokens.data.dtype, np.integer)
            or offsets.ndim != 1
            or not np.issubdtype(offsets.dtype, np.integer)
            or len(offsets) != len(cases) + 1
            or offsets[0] != 0
            or offsets[-1] != len(tokens.data)
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError(f"Malformed token array or offsets: {token_path}")

    ordered_records = []
    for case in cases:
        record = records_by_id[case.case_id]
        source_path = record.get("source_path")
        if not isinstance(source_path, str):
            raise ValueError(f"Missing source_path for {case.case_id} in {jsonl_path}")
        try:
            relative_source = Path(source_path).relative_to(saved_data_dir)
        except ValueError as error:
            raise ValueError(
                f"Source path is outside the saved dataset for {case.case_id}"
            ) from error
        if relative_source != case.path.relative_to(data_dir):
            raise ValueError(
                f"Mismatched relative source path for {case.case_id} in {jsonl_path}"
            )
        for field, expected in (
            ("task_id", case.task_id),
            ("concept", case.concept),
            ("sample_id", case.sample_id),
            ("test_index", case.test_index),
            ("target_grid", case.test_output),
        ):
            if record.get(field) != expected:
                raise ValueError(
                    f"Mismatched {field} for {case.case_id} in {jsonl_path}"
                )
        if record.get("messages") != build_prompt_record(case, templates).messages:
            raise ValueError(f"Mismatched prompt for {case.case_id} in {jsonl_path}")
        token_index = record.get("token_index")
        if (
            not isinstance(token_index, int)
            or isinstance(token_index, bool)
            or not 0 <= token_index < len(archive.case_ids)
            or archive.case_ids[token_index] != case.case_id
            or record.get("token_archive") != token_path.name
        ):
            raise ValueError(
                f"Token alignment mismatch for {case.case_id} in {jsonl_path}"
            )
        ordered_records.append(record)
    return ordered_records, archive


def _normalized_merge_config(
    config: dict[str, Any], model_path_aliases: dict[str, str]
) -> dict[str, Any]:
    """Compare effective settings while retaining original shard snapshots.

    Args:
        config (dict[str, Any]): Saved model, generation, reasoning, and chat settings.
        model_path_aliases (dict[str, str]): Explicitly confirmed checkpoint aliases.

    Returns:
        dict[str, Any]: Copy with the legacy chat default and model aliases resolved.
    """
    normalized = copy.deepcopy(config)
    normalized["chat"].setdefault("prepend_system_to_user", False)
    model_path = normalized["model"].get("model")
    if isinstance(model_path, str):
        normalized["model"]["model"] = model_path_aliases.get(model_path, model_path)
    return normalized


def _publish_merged_artifacts(
    staging: Path, destinations: tuple[Path, ...], *, overwrite: bool
) -> None:
    """Publish completed artifacts and restore prior files if publication fails.

    Args:
        staging (Path): Complete, validated files and temporary rollback storage.
        destinations (tuple[Path, ...]): Canonical token, generation, and config paths.
        overwrite (bool): Replace existing files after validation succeeds.
    """
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for path in destinations:
            if overwrite and path.exists():
                backup = staging / "previous" / path.name
                backup.parent.mkdir(exist_ok=True)
                path.replace(backup)
                backups[path] = backup
            path.hardlink_to(staging / path.name)
            published.append(path)
    except BaseException:
        for path in reversed(published):
            path.unlink()
        for path, backup in backups.items():
            backup.replace(path)
        raise


def merge_concept_generations(
    data_dir: str | Path,
    shard_dir: str | Path,
    *,
    run_name: str | None = None,
    overwrite: bool = False,
    allow_partial: bool = False,
    model_path_aliases: dict[str, str] | None = None,
) -> Path:
    """Merge validated concepts into the standard generation directory for one seed.

    Tokens are copied through temporary files one concept at a time to bound
    memory use. The merge preserves canonical case order and all record fields
    except the rebuilt token index and archive filename. Original shards remain
    available. All shards must have equivalent model and prompt configurations.
    Replacement starts only after all included shards have been validated.
    Partial mode skips wholly absent concepts, still requires complete included
    shards, records coverage, and refuses to drop previously merged queries.

    Args:
        data_dir (str | Path): ConceptARC dataset root used for inference.
        shard_dir (str | Path): Model's generation_shards/seed_<seed> directory.
        run_name (str | None): Artifact stem, inferred if exactly one is present.
        overwrite (bool): Replace existing canonical artifacts after validation.
        allow_partial (bool): Allow entirely missing concepts; default requires all.
        model_path_aliases (dict[str, str] | None): Explicit mappings between copies
            of the same checkpoint. All other
            model settings must still match. Applied aliases are saved in the
            merged configuration alongside the original shard configurations.

    Returns:
        Path: Merged generation JSONL path, ready for evaluation or extraction.

    Raises:
        ValueError: If selection, completeness, configuration, or tokens disagree.
        FileNotFoundError: If any concept artifact is missing.
        FileExistsError: If a destination exists and overwrite is disabled.
    """
    aliases = model_path_aliases or {}
    if any(
        not isinstance(value, str) or not value
        for pair in aliases.items()
        for value in pair
    ):
        raise ValueError(
            "Model path aliases must map nonempty strings to nonempty strings."
        )
    data_root = Path(data_dir).expanduser().resolve()
    source = Path(shard_dir).expanduser().resolve()
    if source.parent.name != "generation_shards" or not re.fullmatch(
        r"seed_\d+", source.name
    ):
        raise ValueError(
            "shard_dir must be a model's generation_shards/seed_<seed> folder."
        )
    model_dir = source.parent.parent
    dataset_id = model_dir.parent.name
    base_dataset_id = infer_concept_arc_dataset_id(data_root)
    if dataset_id != base_dataset_id:
        raise ValueError("Shard dataset ID does not match the requested dataset.")
    if run_name is None:
        names = sorted(
            {
                path.name.removesuffix("_generations.jsonl")
                for path in source.glob("*/*_generations.jsonl")
            }
        )
        if len(names) != 1:
            raise ValueError(f"Specify --run_name; found artifact stems: {names}")
        run_name = names[0]
    pattern = (
        rf"{re.escape(dataset_id)}_{re.escape(model_dir.name)}"
        rf"(?:_reasoning_effort_(low|medium|high))?_{re.escape(source.name)}"
    )
    match = re.fullmatch(pattern, run_name)
    if match is None:
        raise ValueError(
            f"Run name does not match the dataset, model, and seed: {run_name}"
        )

    cases_by_concept: dict[str, list[ConceptARCTestCase]] = defaultdict(list)
    tasks = load_concept_arc(data_root, require_test_outputs=True)
    for case in cast(list[ConceptARCTestCase], list(iter_test_cases(tasks))):
        cases_by_concept[case.concept].append(case)
    if not cases_by_concept:
        raise ValueError("No ConceptARC queries found in the dataset root.")

    expected_concepts = list(cases_by_concept)
    expected_num_records = sum(len(cases) for cases in cases_by_concept.values())
    missing_concepts = [
        concept
        for concept in expected_concepts
        if not any(
            (source / concept / f"{run_name}_{suffix}").exists()
            for suffix in ("config.json", "generations.jsonl", "tokens.npz")
        )
    ]
    if allow_partial:
        for concept in missing_concepts:
            del cases_by_concept[concept]
        if not cases_by_concept:
            raise ValueError(f"No concept shards found for run {run_name}.")

    output_dir = model_dir / "generations" / source.name
    jsonl_path = output_dir / f"{run_name}_generations.jsonl"
    token_path = output_dir / f"{run_name}_tokens.npz"
    config_path = output_dir / f"{run_name}_config.json"
    destinations = (jsonl_path, token_path, config_path)
    for path in destinations:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Merged output already exists: {path}")
    if allow_partial and overwrite and jsonl_path.exists():
        selected_case_ids = {
            case.case_id for cases in cases_by_concept.values() for case in cases
        }
        with jsonl_path.open(encoding="utf-8") as previous:
            for line in previous:
                if not line.strip():
                    continue
                case_id = json.loads(line).get("case_id")
                if case_id not in selected_case_ids:
                    raise ValueError(
                        "Partial merge would drop a previously merged query: "
                        f"{case_id}. Restore its source shard before updating."
                    )
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        concept: _load_shard_config(source / concept / f"{run_name}_config.json")
        for concept in cases_by_concept
    }
    snapshots = [cfg["templates"] for cfg in configs.values() if "templates" in cfg]
    if snapshots:
        if any(snapshot != snapshots[0] for snapshot in snapshots):
            raise ValueError("Mismatched prompt templates across generation shards.")
        templates = PromptTemplates(**snapshots[0])
    else:
        # The generation CLI used the packaged paper prompts before it saved their text.
        templates = PromptTemplates.from_dir(DEFAULT_CONCEPT_ARC_PROMPT_DIR)

    reference_config: dict[str, Any] | None = None
    reference_settings: dict[str, Any] | None = None
    source_configs: list[dict[str, Any]] = []
    applied_aliases: dict[str, str] = {}
    case_ids: list[str] = []
    prompt_offsets = [0]
    generated_offsets = [0]
    source_paths: list[str] = []
    with tempfile.TemporaryDirectory(prefix=".concept_merge_", dir=output_dir) as temp:
        staging = Path(temp)
        prompt_data = staging / "prompt.bin"
        generated_data = staging / "generated.bin"
        with (
            (staging / jsonl_path.name).open("w", encoding="utf-8") as text_file,
            prompt_data.open("wb") as prompt_file,
            generated_data.open("wb") as generated_file,
        ):
            for concept, cases in cases_by_concept.items():
                config = configs[concept]
                records, archive = _load_concept_shard(
                    source / concept, run_name, cases, data_root, config, templates
                )
                settings = _normalized_merge_config(config["config"], aliases)
                model_path = config["config"]["model"].get("model")
                if isinstance(model_path, str) and model_path in aliases:
                    applied_aliases[model_path] = aliases[model_path]
                source_configs.append(
                    {"directory": str(source / concept), "config": config}
                )
                if reference_config is None:
                    reference_config = config
                    reference_settings = settings
                    snapshot = config["config"]
                    seed = int(source.name.removeprefix("seed_"))
                    if (
                        snapshot.get("model", {}).get("seed") != seed
                        or snapshot.get("generation", {}).get("seed") != seed
                        or snapshot.get("reasoning", {}).get("reasoning_effort")
                        != match.group(1)
                    ):
                        raise ValueError(
                            "Saved seed or reasoning effort disagrees with run name."
                        )
                elif settings != reference_settings:
                    raise ValueError(f"Mismatched model configuration: {concept}")
                source_paths.append(str(source / concept / jsonl_path.name))
                for record in records:
                    token_index = record["token_index"]
                    for tokens, handle, offsets in (
                        (archive.prompt_token_ids, prompt_file, prompt_offsets),
                        (
                            archive.generated_token_ids,
                            generated_file,
                            generated_offsets,
                        ),
                    ):
                        start = int(tokens.offsets[token_index])
                        end = int(tokens.offsets[token_index + 1])
                        np.asarray(tokens.data[start:end], dtype=np.int64).tofile(
                            handle
                        )
                        offsets.append(offsets[-1] + end - start)
                    record["token_index"] = len(case_ids)
                    record["token_archive"] = token_path.name
                    case_ids.append(record["case_id"])
                    text_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                del records, archive, tokens

        arrays = {}
        for name, path, offsets in (
            ("prompt_token_ids", prompt_data, prompt_offsets),
            ("generated_token_ids", generated_data, generated_offsets),
        ):
            arrays[f"{name}_data"] = (
                np.memmap(path, dtype=np.int64, mode="r")
                if offsets[-1]
                else np.empty(0, dtype=np.int64)
            )
            arrays[f"{name}_offsets"] = np.asarray(offsets, dtype=np.int64)
        np.savez_compressed(
            staging / token_path.name,
            case_ids=np.asarray(case_ids, dtype=str),
            prompt_token_ids_data=arrays["prompt_token_ids_data"],
            prompt_token_ids_offsets=arrays["prompt_token_ids_offsets"],
            generated_token_ids_data=arrays["generated_token_ids_data"],
            generated_token_ids_offsets=arrays["generated_token_ids_offsets"],
        )
        assert reference_config is not None
        merged_config = {
            **reference_config,
            "config": reference_settings,
            "templates": asdict(templates),
            "model_path_aliases": applied_aliases,
            "merge_sources": source_configs,
            "paths": {
                **reference_config["paths"],
                "generation_jsonl": str(jsonl_path),
                "token_npz": str(token_path),
            },
            "num_records": len(case_ids),
            "data_selection": {
                "data_dir": str(data_root),
                "concepts": list(cases_by_concept),
            },
            "merged_from": source_paths,
            "merge_coverage": {
                "is_complete": not missing_concepts,
                "expected_concepts": expected_concepts,
                "included_concepts": list(cases_by_concept),
                "missing_concepts": missing_concepts,
                "expected_num_records": expected_num_records,
                "num_records": len(case_ids),
            },
        }
        save_json(merged_config, staging / config_path.name)
        _publish_merged_artifacts(
            staging, (token_path, jsonl_path, config_path), overwrite=overwrite
        )
    logging.info(
        "Merged %d/%d concepts and %d/%d queries. Missing concepts: %s",
        len(cases_by_concept),
        len(expected_concepts),
        len(case_ids),
        expected_num_records,
        ", ".join(missing_concepts) or "none",
    )
    return jsonl_path
