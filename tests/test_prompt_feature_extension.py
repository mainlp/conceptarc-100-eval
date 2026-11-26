"""Preserve existing support features while adding newly extracted concepts."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arc_cd.models.generation_io import load_generation_jsonl, load_token_archive_npz
from arc_cd.probes import prompt_feature_io
from arc_cd.probes.feature_io import load_feature_archive
from arc_cd.probes.prompt_feature_io import merge_prompt_feature_shards
from arc_cd.probes.prompt_features import SUPPORT_POSITIONS
from arc_cd.repr_analysis.data import load_bundle
from scripts.repr_analysis.merge_prompt_features import main as merge_main
from tests.test_prompt_features import build_records, fixture_config, write_fixture_run


def _write_concept(dataset: Path, concept: str, color: int) -> None:
    """Write one parent with eight distinguishable support versions.

    Args:
        dataset (Path): Dataset root.
        concept (str): Concept directory and sample prefix.
        color (int): Color distinguishing the concept's source examples.
    """
    directory = dataset / concept
    directory.mkdir()
    for version in range(1, 9):
        task = {
            "train": [
                {"input": [[color, version]], "output": [[i, color, version]]}
                for i in (3, 4, 5)
            ],
            "test": [{"input": [[color]], "output": [[color]]}],
        }
        (directory / f"{concept}1_{version:03d}.json").write_text(json.dumps(task))


def _config(dataset: Path) -> dict[str, Any]:
    """Build an analysis-compatible extraction contract with explicit settings.

    Args:
        dataset (Path): Current source dataset.

    Returns:
        dict[str, Any]: Fixture configuration including the current inventory.
    """
    config = fixture_config(dataset)
    config["extraction"].update(
        {
            "model_name": "fixture",
            "config": {"model": {"model": "fixture", "dtype": "bfloat16"}},
            "templates": {"system": "fixture", "user": "fixture"},
            "tokenizer_sha256": "fixture",
            "tokenizer_chat_template": "fixture",
            "hf_model_config": {"hidden_size": 4},
            "chunk_size": 4096,
            "feature_dtype": "float16",
            "position_definition": "token_containing_final_support_output_closing_bracket",
            "software": {"torch": "fixture"},
            "torch_backends": {"cudnn_sdp": False},
        }
    )
    config["runtime"] = {"data_dir": str(dataset), "hf_device_map": "auto"}
    return config


@pytest.fixture
def extension_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write an old complete bundle followed by two new concept shards.

    Args:
        tmp_path (Path): Isolated test directory.

    Returns:
        tuple[Path, Path, Path]: Expanded dataset, base bundle, and new shards.
    """
    dataset = tmp_path / "ConceptARC-100-10"
    dataset.mkdir()
    for color, concept in enumerate(("AboveBelow", "FilledNotFilled"), start=1):
        _write_concept(dataset, concept, color)
    base = tmp_path / "prompt_features" / "features"
    records, _ = build_records(dataset)
    write_fixture_run(base, records, _config(dataset))
    for color, concept in enumerate(("Count", "Order"), start=6):
        _write_concept(dataset, concept, color)
    records, _ = build_records(dataset)
    config = _config(dataset)
    config["runtime"]["hf_device_map"] = "cuda:0"
    shards = base.parent / "shards"
    for concept in ("Count", "Order"):
        selected = [record for record in records if record["concept"] == concept]
        write_fixture_run(
            shards / concept,
            selected,
            {**config, "selection": {"concept": concept}, "num_records": len(selected)},
        )
    return dataset, base, shards


def test_extended_merge_matches_full_run(
    extension_sources: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI extension preserves values and tokens while rebuilding global axes.

    Args:
        extension_sources (tuple[Path, Path, Path]): Dataset and source features.
        monkeypatch (pytest.MonkeyPatch): Command-line argument patcher.
    """
    dataset, base, shards = extension_sources
    sources = [base, shards / "Count", shards / "Order"]
    original = {p: p.read_bytes() for source in sources for p in source.iterdir()}
    destination = base.parent / "features_all"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_prompt_features",
            "--data_dir",
            str(dataset),
            "--base_feature_dir",
            str(base),
            "--shard_dir",
            str(shards),
            "--output_dir",
            str(destination),
        ],
    )
    merge_main()
    records, _ = build_records(dataset)
    expected = base.parent / "expected"
    write_fixture_run(expected, records, _config(dataset))
    for position in SUPPORT_POSITIONS:
        actual_archive = load_feature_archive(destination / f"{position}.npz")
        expected_archive = load_feature_archive(expected / f"{position}.npz")
        assert actual_archive.keys() == expected_archive.keys()
        for key in actual_archive:
            np.testing.assert_array_equal(
                actual_archive[key], expected_archive[key], err_msg=key
            )
    assert load_generation_jsonl(
        destination / "manifest.jsonl"
    ) == load_generation_jsonl(expected / "manifest.jsonl")
    tokens = load_token_archive_npz(destination / "tokens.npz")
    for row, record in enumerate(records):
        assert tokens.get_by_index(row) == {
            "prompt_token_ids": record["prompt_token_ids"],
            "generated_token_ids": [],
        }
    config = json.loads((destination / "run_config.json").read_text())
    assert config["dataset"] == _config(dataset)["dataset"]
    assert config["extraction"]["concepts"] == [
        "AboveBelow",
        "Count",
        "FilledNotFilled",
        "Order",
    ]
    assert config["merged_from"] == list(map(str, sources))
    for source, provenance in zip(sources, config["merge_sources"], strict=True):
        assert provenance["config"] == json.loads(original[source / "run_config.json"])
        assert provenance["directory"] == str(source)
        assert set(provenance["fingerprints"]) == {p.name for p in source.iterdir()}
        for name, fingerprint in provenance["fingerprints"].items():
            raw = original[source / name]
            assert fingerprint == {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
    assert {p: p.read_bytes() for p in original} == original
    bundle = load_bundle(destination)
    assert bundle.rows.shape == (4, 8)
    assert bundle.load_position("S3").shape == (32, 2, 4)


@pytest.mark.parametrize(
    "field",
    [
        "config",
        "templates",
        "tokenizer_sha256",
        "tokenizer_chat_template",
        "hf_model_config",
        "chunk_size",
        "layers",
        "software",
        "torch_backends",
    ],
)
def test_extension_rejects_extraction_mismatch(
    extension_sources: tuple[Path, Path, Path], field: str
) -> None:
    """Dataset growth does not relax numerical or prompt compatibility checks.

    Args:
        extension_sources (tuple[Path, Path, Path]): Dataset and source features.
        field (str): Extraction setting to corrupt in a new shard.
    """
    dataset, base, shards = extension_sources
    path = shards / "Count" / "run_config.json"
    config = json.loads(path.read_text())
    config["extraction"][field] = "changed"
    path.write_text(json.dumps(config))
    destination = base.parent / "features_all"
    with pytest.raises(ValueError, match="Incompatible configuration"):
        merge_prompt_feature_shards(dataset, shards, destination, base_feature_dir=base)
    assert not destination.exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "selection",
        "vocabulary",
        "dataset",
        "num_records",
        "manifest",
        "tokens",
        "labels",
        "features",
        "support_identity",
    ],
)
def test_extension_rejects_invalid_base(
    extension_sources: tuple[Path, Path, Path], corruption: str
) -> None:
    """Validate old bundles as strictly as new shards, without publishing errors.

    Args:
        extension_sources (tuple[Path, Path, Path]): Dataset and source features.
        corruption (str): Base contract or alignment failure to introduce.
    """
    dataset, base, shards = extension_sources
    if corruption in ("selection", "vocabulary", "dataset", "num_records"):
        path = base / "run_config.json"
        config = json.loads(path.read_text())
        if corruption == "selection":
            config["selection"]["concept"] = "AboveBelow"
        elif corruption == "vocabulary":
            config["extraction"]["concepts"] = ["AboveBelow"] * 2
        elif corruption == "dataset":
            key = next(iter(config["dataset"]["task_hashes"]))
            config["dataset"]["task_hashes"][key] = "changed"
        else:
            config["num_records"] -= 1
        path.write_text(json.dumps(config))
    elif corruption in ("manifest", "support_identity"):
        path = base / "manifest.jsonl"
        rows = load_generation_jsonl(path)
        if corruption == "manifest":
            rows[0]["token_index"] = 1
        else:
            rows[0]["support_examples"][0]["output"] = [[9]]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        path = base / ("tokens.npz" if corruption == "tokens" else "S2.npz")
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if corruption == "tokens":
            arrays["prompt_token_ids_data"][0] += 1
        elif corruption == "labels":
            arrays["y"] = arrays["y"][:, ::-1]
        else:
            arrays["x"] = arrays["x"][:, :, :-1]
        np.savez_compressed(path, **arrays)
    destination = base.parent / "features_all"
    with pytest.raises(ValueError):
        merge_prompt_feature_shards(dataset, shards, destination, base_feature_dir=base)
    assert not destination.exists()
    assert not (destination.parent / ".features_all.lock").exists()


@pytest.mark.parametrize(
    "change", ["missing", "overlap", "unexpected", "changed_task", "added_version"]
)
def test_extension_rejects_invalid_coverage(
    extension_sources: tuple[Path, Path, Path], change: str
) -> None:
    """Require unchanged old concepts and complete, disjoint new concepts.

    Args:
        extension_sources (tuple[Path, Path, Path]): Dataset and source features.
        change (str): Source coverage change to introduce.
    """
    dataset, base, shards = extension_sources
    if change == "missing":
        shutil.rmtree(shards / "Order")
    elif change == "overlap":
        shutil.copytree(base, shards / "AboveBelow")
    elif change == "unexpected":
        (shards / "Unexpected").mkdir()
    else:
        path = next((dataset / "AboveBelow").glob("*.json"))
        if change == "changed_task":
            path.write_text(path.read_text() + "\n")
        else:
            shutil.copyfile(path, path.with_name("AboveBelow1_009.json"))
    destination = base.parent / "features_all"
    with pytest.raises(ValueError):
        merge_prompt_feature_shards(dataset, shards, destination, base_feature_dir=base)
    assert not destination.exists()


@pytest.mark.parametrize("source", ["features", "dataset"])
def test_extension_rejects_inputs_changed_during_merge(
    extension_sources: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """Discard staged output when a source changes after initial validation.

    Args:
        extension_sources (tuple[Path, Path, Path]): Dataset and source features.
        monkeypatch (pytest.MonkeyPatch): Feature reader patcher.
        source (str): Feature artifacts or source tasks to mutate during the merge.
    """
    dataset, base, shards = extension_sources
    read_archive = prompt_feature_io.load_feature_archive

    def read_and_mutate(path: Path) -> dict[str, Any]:
        """Read one archive and then simulate a concurrent source modification.

        Args:
            path (Path): Feature archive requested by the merger.

        Returns:
            dict[str, Any]: Original archive arrays.
        """
        archive = read_archive(path)
        if path == base / "S3.npz":
            target = (
                base / "run_config.json"
                if source == "features"
                else next((dataset / "AboveBelow").glob("*.json"))
            )
            target.write_text(target.read_text() + "\n")
        return archive

    monkeypatch.setattr(prompt_feature_io, "load_feature_archive", read_and_mutate)
    destination = base.parent / "features_all"
    with pytest.raises(ValueError, match="changed during merge"):
        merge_prompt_feature_shards(dataset, shards, destination, base_feature_dir=base)
    assert not destination.exists()
    assert not (destination.parent / ".features_all.lock").exists()


def test_extension_requires_new_explicit_destination(
    extension_sources: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both CLI and API require an output and refuse to overwrite the base.

    Args:
        extension_sources (tuple[Path, Path, Path]): Dataset and source features.
        monkeypatch (pytest.MonkeyPatch): Command-line argument patcher.
    """
    dataset, base, shards = extension_sources
    with pytest.raises(ValueError, match="output_dir is required"):
        merge_prompt_feature_shards(dataset, shards, base_feature_dir=base)
    before = (base / "run_config.json").read_bytes()
    with pytest.raises(FileExistsError):
        merge_prompt_feature_shards(dataset, shards, base, base_feature_dir=base)
    assert (base / "run_config.json").read_bytes() == before
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_prompt_features",
            "--data_dir",
            str(dataset),
            "--shard_dir",
            str(shards),
            "--base_feature_dir",
            str(base),
        ],
    )
    with pytest.raises(SystemExit) as error:
        merge_main()
    assert error.value.code == 2


def test_failed_merge_closes_maps_before_cleanup(
    extension_sources: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close staged mappings on write failure even when the caller retains references.

    Args:
        extension_sources (tuple[Path, Path, Path]): Dataset and source features.
        monkeypatch (pytest.MonkeyPatch): Simulate a disk quota failure.
    """
    dataset, base, shards = extension_sources
    opened: list[np.memmap] = []

    def fail_flush(array: np.memmap) -> None:
        """Retain the live mapping and simulate an interrupted output write.

        Args:
            array (np.memmap): Staged output mapping.
        """
        opened.append(array)
        raise OSError("simulated disk quota exceeded")

    monkeypatch.setattr(np.memmap, "flush", fail_flush)
    destination = base.parent / "features_all"
    with pytest.raises(OSError, match="simulated disk quota"):
        merge_prompt_feature_shards(dataset, shards, destination, base_feature_dir=base)
    try:
        assert opened and all(array._mmap.closed for array in opened)
        assert not destination.exists()
        assert not list(destination.parent.glob(".features_all.*"))
    finally:
        for array in opened:
            array._mmap.close()
