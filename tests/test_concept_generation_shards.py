"""Test concept generation, shard merging, and query alignment."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from arc_cd.data_prep.load_data import iter_test_cases, load_concept_arc
from arc_cd.eval.eval_response import evaluate_records
from arc_cd.models.concept_shards import merge_concept_generations
from arc_cd.models.generation_io import load_generation_jsonl, load_token_archive_npz
from arc_cd.prompt_prep import PromptRecord
from scripts import merge_concept_generations as merge_cli
from scripts import run_arc_vLLM
from scripts.eval_arc_acc import evaluate_generation_file

RUN_NAME = "concept_arc_100_qwen3_8b_seed_42"


def _generate_records(
    *, records: Sequence[PromptRecord], **kwargs: Any
) -> list[dict[str, Any]]:
    """Simulate generation while preserving the actual prompt identities.

    Args:
        records (Sequence[PromptRecord]): Real records built by the runner.
        **kwargs (Any): Unused model and tokenizer arguments.

    Returns:
        list[dict[str, Any]]: Generations with distinctive, variable-length tokens.
    """
    return [
        {
            "case_id": record.case_id,
            "task_id": record.task_id,
            "concept": record.concept,
            "sample_id": record.sample_id,
            "test_index": record.test_index,
            "source_path": record.source_path,
            "metadata": record.metadata,
            "messages": record.messages,
            "target_grid": record.target_grid,
            "generated_text": "</think>\n" + json.dumps(record.target_grid),
            "finish_reason": "stop",
            "prompt_token_ids": [101, index, 103],
            "generated_token_ids": [200 + record.test_index] * record.test_index,
            "token_spans": {"query_input": [1, 2]},
        }
        for index, record in enumerate(records)
    ]


@pytest.fixture
def generation_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Mock]:
    """Set up two concepts and replace GPU inference with deterministic outputs.

    Args:
        tmp_path (Path): Temporary dataset and artifact root.
        monkeypatch (pytest.MonkeyPatch): Model import and seed patcher.

    Returns:
        tuple[Path, Path, Mock]: Dataset root, output root, and tokenizer loader.
    """
    data_dir = tmp_path / "ConceptARC-100"
    for concept in ("Center", "AboveBelow"):
        directory = data_dir / concept
        directory.mkdir(parents=True)
        for task in ("task2", "task10"):
            (directory / f"{task}.json").write_text(
                json.dumps(
                    {
                        "train": [{"input": [[1]], "output": [[2]]}] * 3,
                        "test": [{"input": [[1]], "output": [[2]]}] * 10,
                    }
                )
            )
    fake_vllm = ModuleType("arc_cd.models.vllm_setup")
    loader = Mock(return_value=object())
    fake_vllm.load_tokenizer = loader  # type: ignore[attr-defined]
    fake_vllm.load_vllm = Mock(return_value=object())  # type: ignore[attr-defined]
    fake_vllm.generate_from_records = _generate_records  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "arc_cd.models.vllm_setup", fake_vllm)
    monkeypatch.setattr(run_arc_vLLM, "set_seed", lambda seed: None)
    return data_dir, tmp_path / "output", loader


def _run_generation(
    data_dir: Path,
    output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *selection: str,
) -> None:
    """Run the real generation entry point with optional concept selection.

    Args:
        data_dir (Path): Dataset root.
        output_dir (Path): Output root.
        monkeypatch (pytest.MonkeyPatch): CLI argument patcher.
        *selection (str): Additional CLI arguments.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_arc_vLLM.py",
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(output_dir),
            "--model_name",
            "qwen3_8b",
            "--seed",
            "42",
            *selection,
        ],
    )
    run_arc_vLLM.main()


def test_concept_runs_merge_with_exact_tokens_and_canonical_order(
    generation_setup: tuple[Path, Path, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merge shuffled JSONL rows while retaining tokens, spans, and query IDs.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and fake inference.
        monkeypatch (pytest.MonkeyPatch): CLI argument patcher.
    """
    data_dir, output_dir, _ = generation_setup
    for index in (1, 0):
        _run_generation(
            data_dir, output_dir, monkeypatch, "--concept_index", str(index)
        )
    shard_dir = output_dir / "concept_arc_100/qwen3_8b/generation_shards/seed_42"
    original_records = {}
    original_tokens = {}
    for path in shard_dir.glob("*/*_generations.jsonl"):
        archive = load_token_archive_npz(path.with_name(f"{RUN_NAME}_tokens.npz"))
        records = load_generation_jsonl(path)
        for record in records:
            original_records[record["case_id"]] = record
            original_tokens[record["case_id"]] = archive.get_by_index(
                record["token_index"]
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in reversed(records)))

    result_path = merge_concept_generations(data_dir, shard_dir)
    records = load_generation_jsonl(result_path)
    archive = load_token_archive_npz(result_path.with_name(f"{RUN_NAME}_tokens.npz"))
    expected_ids = [
        case.case_id for case in iter_test_cases(load_concept_arc(data_dir))
    ]
    assert [record["case_id"] for record in records] == expected_ids
    assert list(archive.case_ids) == expected_ids
    for index, record in enumerate(records):
        assert record["token_index"] == index
        assert archive.get_by_index(index) == original_tokens[record["case_id"]]
        assert record == {**original_records[record["case_id"]], "token_index": index}
    assert (
        result_path.parent
        == output_dir / "concept_arc_100/qwen3_8b/generations/seed_42"
    )
    config = json.loads(result_path.with_name(f"{RUN_NAME}_config.json").read_text())
    assert config["num_records"] == 40
    assert config["data_selection"]["concepts"] == ["AboveBelow", "Center"]
    assert config["paths"]["generation_jsonl"] == str(result_path)
    assert len(config["merged_from"]) == 2
    _, summary = evaluate_records(records)
    assert summary["overall"]["n"] == summary["overall"]["n_correct"] == 40
    assert all(metrics["n"] == 10 for metrics in summary["per_task"].values())
    with pytest.raises(FileExistsError):
        merge_concept_generations(data_dir, shard_dir)


@pytest.mark.parametrize("legacy_shards", [0, 1, 2])
def test_merge_checks_prompt_content_across_installations(
    generation_setup: tuple[Path, Path, Mock],
    monkeypatch: pytest.MonkeyPatch,
    legacy_shards: int,
) -> None:
    """Merge relocated prompts and reject changed grids or templates.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and simulated inference.
        monkeypatch (pytest.MonkeyPatch): CLI argument patcher.
        legacy_shards (int): Number of shards that saved only the template path.
    """
    data_dir, output_dir, _ = generation_setup
    shard_dir = output_dir / "concept_arc_100/qwen3_8b/generation_shards/seed_42"
    for index, concept in enumerate(("AboveBelow", "Center")):
        _run_generation(data_dir, output_dir, monkeypatch, "--concept", concept)
        path = shard_dir / concept / f"{RUN_NAME}_config.json"
        config = json.loads(path.read_text())
        worker = "old-install" if legacy_shards == 2 else f"worker-{index}"
        config["paths"]["prompt_dir"] = str(output_dir / worker / "prompts")
        if index < legacy_shards:
            del config["templates"]
        path.write_text(json.dumps(config))

    merged = merge_concept_generations(data_dir, shard_dir)
    assert len(load_generation_jsonl(merged)) == 40
    original = merged.read_bytes()
    task_path = data_dir / "Center/task2.json"
    task = json.loads(task_path.read_text())
    task["train"][0]["input"] = [[9]]
    task_path.write_text(json.dumps(task))
    with pytest.raises(ValueError, match="Mismatched prompt for"):
        merge_concept_generations(data_dir, shard_dir, overwrite=True)
    if legacy_shards == 0:
        path = shard_dir / "Center" / f"{RUN_NAME}_config.json"
        config = json.loads(path.read_text())
        config["templates"]["user"] += " changed"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="Mismatched prompt templates"):
            merge_concept_generations(data_dir, shard_dir, overwrite=True)
    assert merged.read_bytes() == original


def test_full_dataset_output_contract_is_unchanged(
    generation_setup: tuple[Path, Path, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep unfiltered runs in the existing generation tree and config schema.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and fake inference.
        monkeypatch (pytest.MonkeyPatch): CLI argument patcher.
    """
    data_dir, output_dir, _ = generation_setup
    _run_generation(data_dir, output_dir, monkeypatch)
    path = (
        output_dir
        / "concept_arc_100/qwen3_8b/generations/seed_42"
        / f"{RUN_NAME}_config.json"
    )
    config = json.loads(path.read_text())
    assert config["num_records"] == 40
    assert "data_selection" not in config
    assert not list(output_dir.rglob("generation_shards"))


def test_merge_accepts_empty_generated_token_sequences(
    generation_setup: tuple[Path, Path, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve zero-length generations even when the entire token array is empty.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and fake inference.
        monkeypatch (pytest.MonkeyPatch): CLI argument patcher.
    """
    data_dir, output_dir, _ = generation_setup
    for path in data_dir.glob("*/*.json"):
        task = json.loads(path.read_text())
        task["test"] = task["test"][:1]
        path.write_text(json.dumps(task))
    for index in (0, 1):
        _run_generation(
            data_dir, output_dir, monkeypatch, "--concept_index", str(index)
        )
    shard_dir = output_dir / "concept_arc_100/qwen3_8b/generation_shards/seed_42"
    result_path = merge_concept_generations(data_dir, shard_dir)
    archive = load_token_archive_npz(result_path.with_name(f"{RUN_NAME}_tokens.npz"))
    assert archive.generated_token_ids.data.size == 0
    assert archive.generated_token_ids.offsets.tolist() == [0, 0, 0, 0, 0]


def test_completed_concept_is_not_overwritten(
    generation_setup: tuple[Path, Path, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject repeated concept runs before loading the model again.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and fake inference.
        monkeypatch (pytest.MonkeyPatch): CLI argument patcher.
    """
    data_dir, output_dir, loader = generation_setup
    _run_generation(data_dir, output_dir, monkeypatch, "--concept", "AboveBelow")
    loader.reset_mock()
    with pytest.raises(FileExistsError):
        _run_generation(data_dir, output_dir, monkeypatch, "--concept", "AboveBelow")
    loader.assert_not_called()


@pytest.mark.parametrize(
    "selection",
    [("--concept", "Unknown"), ("--concept_index", "-1"), ("--concept_index", "2")],
)
def test_invalid_selection_fails_before_model_loading(
    generation_setup: tuple[Path, Path, Mock],
    monkeypatch: pytest.MonkeyPatch,
    selection: tuple[str, str],
) -> None:
    """Reject invalid concept names and indices before allocating the model.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and fake inference.
        monkeypatch (pytest.MonkeyPatch): CLI argument patcher.
        selection (tuple[str, str]): Invalid selection option and value.
    """
    data_dir, output_dir, loader = generation_setup
    with pytest.raises(ValueError):
        _run_generation(data_dir, output_dir, monkeypatch, *selection)
    loader.assert_not_called()


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "duplicate",
        "truncated",
        "config",
        "index",
        "offsets",
        "target",
        "seed",
    ],
)
@pytest.mark.parametrize("allow_partial", [False, True])
def test_merge_rejects_incomplete_or_inconsistent_shards(
    generation_setup: tuple[Path, Path, Mock],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    allow_partial: bool,
) -> None:
    """Reject corrupt inputs without publishing partial canonical artifacts.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and fake inference.
        monkeypatch (pytest.MonkeyPatch): CLI argument patcher.
        damage (str): Shard corruption to introduce.
        allow_partial (bool): Permit absent concepts without permitting corrupt shards.
    """
    data_dir, output_dir, _ = generation_setup
    for concept in ("AboveBelow", "Center"):
        _run_generation(data_dir, output_dir, monkeypatch, "--concept", concept)
    shard_dir = output_dir / "concept_arc_100/qwen3_8b/generation_shards/seed_42"
    directory = shard_dir / "Center"
    jsonl_path = directory / f"{RUN_NAME}_generations.jsonl"
    config_path = directory / f"{RUN_NAME}_config.json"
    token_path = directory / f"{RUN_NAME}_tokens.npz"
    records = load_generation_jsonl(jsonl_path)
    if damage == "missing":
        config_path.unlink()
    elif damage in {"config", "seed"}:
        config = json.loads(config_path.read_text())
        field = "temperature" if damage == "config" else "seed"
        config["config"]["generation"][field] += 1
        config_path.write_text(json.dumps(config))
    elif damage == "offsets":
        with np.load(token_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["prompt_token_ids_offsets"][-1] += 1
        np.savez_compressed(token_path, **arrays)
    else:
        if damage == "duplicate":
            records[1] = records[0]
        elif damage == "truncated":
            records.pop()
        elif damage == "index":
            records[0]["token_index"] = 1
        elif damage == "target":
            records[0]["target_grid"] = [[9]]
        jsonl_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises((ValueError, FileNotFoundError)):
        merge_concept_generations(data_dir, shard_dir, allow_partial=allow_partial)
    destination = output_dir / "concept_arc_100/qwen3_8b/generations/seed_42"
    assert not list(destination.iterdir())


@pytest.mark.parametrize("conflict", [None, "chat", "temperature"])
def test_merge_checkpoint_aliases_and_legacy_chat_default(
    generation_setup: tuple[Path, Path, Mock],
    monkeypatch: pytest.MonkeyPatch,
    conflict: str | None,
) -> None:
    """Resolve confirmed path aliases and defaults without hiding changed settings.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and simulated inference.
        monkeypatch (pytest.MonkeyPatch): Generation argument patcher.
        conflict (str | None): Genuine chat or sampling mismatch to retain.
    """
    data_dir, output_dir, _ = generation_setup
    shard_dir = output_dir / "concept_arc_100/qwen3_8b/generation_shards/seed_42"
    paths = ["/models/first/Qwen3-8B", "/models/second/Qwen3-8B"]
    originals = {}
    for concept, model_path in zip(("AboveBelow", "Center"), paths, strict=True):
        _run_generation(data_dir, output_dir, monkeypatch, "--concept", concept)
        path = shard_dir / concept / f"{RUN_NAME}_config.json"
        config = json.loads(path.read_text())
        config["config"]["model"]["model"] = model_path
        if concept == "AboveBelow":
            config["config"]["chat"].pop("prepend_system_to_user", None)
        else:
            config["config"]["chat"]["prepend_system_to_user"] = conflict == "chat"
            if conflict == "temperature":
                config["config"]["generation"]["temperature"] += 1
        path.write_text(json.dumps(config))
        originals[path] = path.read_bytes()
    with pytest.raises(ValueError, match="Mismatched model configuration"):
        merge_concept_generations(data_dir, shard_dir)
    aliases = {paths[1]: paths[0]}
    if conflict:
        with pytest.raises(ValueError, match="Mismatched model configuration"):
            merge_concept_generations(data_dir, shard_dir, model_path_aliases=aliases)
    else:
        result = merge_concept_generations(
            data_dir, shard_dir, model_path_aliases=aliases
        )
        saved = json.loads(result.with_name(f"{RUN_NAME}_config.json").read_text())
        assert saved["config"]["model"]["model"] == paths[0]
        assert saved["config"]["chat"]["prepend_system_to_user"] is False
        assert saved["model_path_aliases"] == aliases
        assert [
            source["config"]["config"]["model"]["model"]
            for source in saved["merge_sources"]
        ] == paths
    assert all(path.read_bytes() == original for path, original in originals.items())


@pytest.mark.parametrize("failure", [None, "missing_shard", "publication", "interrupt"])
def test_merge_overwrite_preserves_old_artifacts_until_success(
    generation_setup: tuple[Path, Path, Mock],
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    """Replace a complete run, retaining old bytes if validation or publication fails.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and simulated inference.
        monkeypatch (pytest.MonkeyPatch): Generation and publication patcher.
        failure (str | None): Failure before or during canonical publication.
    """
    data_dir, output_dir, _ = generation_setup
    for concept in ("AboveBelow", "Center"):
        _run_generation(data_dir, output_dir, monkeypatch, "--concept", concept)
    shard_dir = output_dir / "concept_arc_100/qwen3_8b/generation_shards/seed_42"
    original = merge_concept_generations(data_dir, shard_dir)
    old_bytes = {path: path.read_bytes() for path in original.parent.iterdir()}
    shard = shard_dir / "Center" / original.name
    records = load_generation_jsonl(shard)
    records[0]["merge_test_revision"] = 2
    shard.write_text("".join(json.dumps(record) + "\n" for record in records))
    if failure == "missing_shard":
        (shard_dir / "Center" / f"{RUN_NAME}_config.json").unlink()
    elif failure in ("publication", "interrupt"):
        hardlink = Path.hardlink_to

        def fail_publication(path: Path, target: str | Path) -> None:
            """Fail after replacing tokens so rollback must restore multiple files.

            Args:
                path (Path): Destination artifact.
                target (str | Path): Staged replacement.
            """
            if path == original:
                if failure == "interrupt":
                    raise KeyboardInterrupt()
                raise OSError("simulated publication error")
            hardlink(path, target)

        monkeypatch.setattr(Path, "hardlink_to", fail_publication)
    if failure is None:
        result = merge_concept_generations(data_dir, shard_dir, overwrite=True)
        assert result == original and result.read_bytes() != old_bytes[result]
        assert any(
            record.get("merge_test_revision") == 2
            for record in load_generation_jsonl(result)
        )
        archive = load_token_archive_npz(result.with_name(f"{RUN_NAME}_tokens.npz"))
        for record in load_generation_jsonl(result):
            assert archive.case_ids[record["token_index"]] == record["case_id"]
    else:
        error = KeyboardInterrupt if failure == "interrupt" else OSError
        with pytest.raises(error):
            merge_concept_generations(data_dir, shard_dir, overwrite=True)
        assert all(path.read_bytes() == content for path, content in old_bytes.items())
    assert set(original.parent.iterdir()) == set(old_bytes)


@pytest.mark.parametrize(
    "damage", [None, "variant", "concept", "root", "source", "target", "tokens"]
)
def test_merge_relocated_dataset_retains_identity_checks(
    generation_setup: tuple[Path, Path, Mock],
    monkeypatch: pytest.MonkeyPatch,
    damage: str | None,
) -> None:
    """Accept another cluster's dataset root while rejecting actual identity errors.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and simulated inference.
        monkeypatch (pytest.MonkeyPatch): Generation argument patcher.
        damage (str | None): Selection, source, content, or alignment error to inject.
    """
    data_dir, output_dir, _ = generation_setup
    shard_dir = output_dir / "concept_arc_100/qwen3_8b/generation_shards/seed_42"
    saved_root = Path("/other-location/ConceptARC-100")
    originals = {}
    for concept in ("AboveBelow", "Center"):
        _run_generation(data_dir, output_dir, monkeypatch, "--concept", concept)
        directory = shard_dir / concept
        config_path = directory / f"{RUN_NAME}_config.json"
        config = json.loads(config_path.read_text())
        config["data_selection"]["data_dir"] = str(saved_root)
        records_path = directory / f"{RUN_NAME}_generations.jsonl"
        records = load_generation_jsonl(records_path)
        for record in records:
            record["source_path"] = str(
                saved_root / Path(record["source_path"]).relative_to(data_dir)
            )
        if concept == "Center":
            if damage == "variant":
                config["data_selection"]["data_dir"] = "/other-location/ConceptARC-100-10"
            elif damage == "concept":
                config["data_selection"]["concept"] = "AboveBelow"
            elif damage == "root":
                records[0][
                    "source_path"
                ] = "/elsewhere/ConceptARC-100/Center/task2.json"
            elif damage == "source":
                records[0]["source_path"] = str(saved_root / "Center/wrong_task.json")
            elif damage == "target":
                records[0]["target_grid"] = [[9]]
            elif damage == "tokens":
                records[0]["token_index"] = 1
        config_path.write_text(json.dumps(config))
        records_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        originals[config_path] = config_path.read_bytes()
        originals[records_path] = records_path.read_bytes()
    if damage:
        with pytest.raises(ValueError):
            merge_concept_generations(data_dir, shard_dir)
        destination = output_dir / "concept_arc_100/qwen3_8b/generations/seed_42"
        assert not list(destination.iterdir())
    else:
        merged = merge_concept_generations(data_dir, shard_dir)
        config = json.loads(merged.with_name(f"{RUN_NAME}_config.json").read_text())
        assert config["data_selection"]["data_dir"] == str(data_dir)
        assert all(
            source["config"]["data_selection"]["data_dir"] == str(saved_root)
            for source in config["merge_sources"]
        )
        records = load_generation_jsonl(merged)
        assert len(records) == 40
        assert all(
            record["source_path"].startswith(str(saved_root) + "/")
            for record in records
        )
        archive = load_token_archive_npz(merged.with_name(f"{RUN_NAME}_tokens.npz"))
        assert list(archive.case_ids) == [record["case_id"] for record in records]
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_partial_merge_cli_evaluation_and_later_completion(
    generation_setup: tuple[Path, Path, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluate covered concepts and preserve tokens when later shards arrive.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and simulated inference.
        monkeypatch (pytest.MonkeyPatch): Generation and merge CLI argument patcher.
    """
    data_dir, output_dir, _ = generation_setup
    (data_dir / "Count").mkdir()
    for source in (data_dir / "Center").glob("*.json"):
        (data_dir / "Count" / source.name).write_bytes(source.read_bytes())
    _run_generation(data_dir, output_dir, monkeypatch, "--concept", "Center")
    shard_dir = output_dir / "concept_arc_100/qwen3_8b/generation_shards/seed_42"
    shard = shard_dir / "Center" / f"{RUN_NAME}_generations.jsonl"
    records = load_generation_jsonl(shard)
    records[0]["generated_text"] = "</think>\n[[9]]"
    records[1]["generated_text"] = "</think>\nnot a grid"
    shard.write_text("".join(json.dumps(row) + "\n" for row in reversed(records)))
    original_tokens = load_token_archive_npz(shard.with_name(f"{RUN_NAME}_tokens.npz"))
    originals = {record["case_id"]: record for record in records}
    with pytest.raises(FileNotFoundError):
        merge_concept_generations(data_dir, shard_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_concept_generations.py",
            "--data_dir",
            str(data_dir),
            "--shard_dir",
            str(shard_dir),
            "--allow_partial",
        ],
    )
    merge_cli.main()
    result = (
        output_dir
        / f"concept_arc_100/qwen3_8b/generations/seed_42/{RUN_NAME}_generations.jsonl"
    )
    config_path = result.with_name(f"{RUN_NAME}_config.json")
    config = json.loads(config_path.read_text())
    assert config["num_records"] == 20
    assert config["data_selection"]["concepts"] == ["Center"]
    assert config["merge_coverage"] == {
        "is_complete": False,
        "expected_concepts": ["AboveBelow", "Center", "Count"],
        "included_concepts": ["Center"],
        "missing_concepts": ["AboveBelow", "Count"],
        "expected_num_records": 60,
        "num_records": 20,
    }
    summary = evaluate_generation_file(result)
    assert summary["overall"]["n"] == 20
    assert summary["overall"]["accuracy"] == pytest.approx(0.9)
    assert set(summary["per_concept"]) == {"Center"}
    center_accuracy = summary["per_concept"]["Center"]
    partial_records = load_generation_jsonl(result)
    partial_tokens = load_token_archive_npz(result.with_name(f"{RUN_NAME}_tokens.npz"))
    assert list(partial_tokens.case_ids) == [row["case_id"] for row in partial_records]
    for record in partial_records:
        original = originals[record["case_id"]]
        assert partial_tokens.get_by_index(
            record["token_index"]
        ) == original_tokens.get_by_index(original["token_index"])
        assert record == {**original, "token_index": record["token_index"]}

    # Adding an earlier concept moves all existing token indices; identities stay aligned.
    _run_generation(data_dir, output_dir, monkeypatch, "--concept", "AboveBelow")
    assert (
        merge_concept_generations(
            data_dir, shard_dir, overwrite=True, allow_partial=True
        )
        == result
    )
    expanded = json.loads(config_path.read_text())
    assert expanded["merge_coverage"]["is_complete"] is False
    assert expanded["merge_coverage"]["missing_concepts"] == ["Count"]
    assert expanded["merge_coverage"]["num_records"] == 40
    expanded_records = load_generation_jsonl(result)
    expanded_tokens = load_token_archive_npz(result.with_name(f"{RUN_NAME}_tokens.npz"))
    assert list(expanded_tokens.case_ids) == [
        row["case_id"] for row in expanded_records
    ]
    for index, record in enumerate(expanded_records):
        assert record["token_index"] == index
        if record["concept"] == "Center":
            original = originals[record["case_id"]]
            assert index >= 20
            assert expanded_tokens.get_by_index(index) == original_tokens.get_by_index(
                original["token_index"]
            )
            assert record == {**original, "token_index": index}
    expanded_summary = evaluate_generation_file(result)
    assert expanded_summary["overall"]["n"] == 40
    assert expanded_summary["overall"]["accuracy"] == pytest.approx(0.95)
    assert expanded_summary["per_concept"]["Center"] == center_accuracy

    _run_generation(data_dir, output_dir, monkeypatch, "--concept", "Count")
    assert merge_concept_generations(data_dir, shard_dir, overwrite=True) == result
    completed = json.loads(config_path.read_text())
    assert completed["merge_coverage"]["is_complete"] is True
    assert completed["merge_coverage"]["missing_concepts"] == []
    assert completed["merge_coverage"]["num_records"] == 60
    full_summary = evaluate_generation_file(result)
    assert full_summary["overall"]["n"] == 60
    assert full_summary["overall"]["accuracy"] == pytest.approx(58 / 60)
    assert full_summary["per_concept"]["Center"] == center_accuracy


def test_partial_merge_rejects_empty_run(
    generation_setup: tuple[Path, Path, Mock],
) -> None:
    """Ignore other artifact stems and refuse to publish an empty partial merge.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and artifact roots.
    """
    data_dir, output_dir, _ = generation_setup
    shard_dir = output_dir / "concept_arc_100/qwen3_8b/generation_shards/seed_42"
    (shard_dir / "Center").mkdir(parents=True)
    (shard_dir / "Center/other_config.json").write_text("{}")
    with pytest.raises(ValueError, match="No concept shards found"):
        merge_concept_generations(
            data_dir, shard_dir, run_name=RUN_NAME, allow_partial=True
        )
    assert not (shard_dir.parent.parent / "generations").exists()


def test_partial_merge_cannot_drop_previously_merged_queries(
    generation_setup: tuple[Path, Path, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain previous artifacts when a missing source would shrink coverage.

    Args:
        generation_setup (tuple[Path, Path, Mock]): Dataset and simulated inference.
        monkeypatch (pytest.MonkeyPatch): Generation CLI argument patcher.
    """
    data_dir, output_dir, _ = generation_setup
    for concept in ("AboveBelow", "Center"):
        _run_generation(data_dir, output_dir, monkeypatch, "--concept", concept)
    shard_dir = output_dir / "concept_arc_100/qwen3_8b/generation_shards/seed_42"
    result = merge_concept_generations(data_dir, shard_dir)
    originals = {path: path.read_bytes() for path in result.parent.iterdir()}
    for path in (shard_dir / "Center").iterdir():
        path.unlink()
    with pytest.raises(ValueError, match="would drop a previously merged query"):
        merge_concept_generations(
            data_dir, shard_dir, allow_partial=True, overwrite=True
        )
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert set(result.parent.iterdir()) == set(originals)


