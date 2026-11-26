"""Support-boundary replay, version identity and concept-shard contracts."""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from arc_cd.data_prep.load_data import load_concept_arc
from arc_cd.models.config import ChatTemplateConfig
from arc_cd.models.generation_io import load_generation_jsonl, load_token_archive_npz
from arc_cd.probes.feature_io import load_feature_archive
from arc_cd.probes.hidden import PromptReplaySample
from arc_cd.probes.hidden_hf import iter_prompt_hidden_states_hf
from arc_cd.probes.prompt_feature_io import (
    merge_prompt_feature_shards,
    staged_output_directory,
    write_prompt_feature_run,
)
from arc_cd.probes.prompt_features import (
    SUPPORT_POSITIONS,
    build_support_prompts,
    task_inventory,
)
from arc_cd.prompt_prep.prompt_format import PromptTemplates


class CharacterTokenizer:
    """Deterministic offset tokenizer, optionally merging bracket/newline tokens."""

    def __init__(self, merge_newlines: bool = False) -> None:
        self.merge_newlines = merge_newlines
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: "fixture_tokenizer")
        self.chat_template = "fixture_template"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "\n".join(message["content"] for message in messages) + "\nassistant:"

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[Any]]:
        ids, offsets = [], []
        cursor = 0
        while cursor < len(text):
            length = (
                3 if self.merge_newlines and text.startswith("]\n\n", cursor) else 1
            )
            ids.append(256 if length == 3 else ord(text[cursor]))
            offsets.append((cursor, cursor + length))
            cursor += length
        return {"input_ids": ids, "offset_mapping": offsets}


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Create two concepts with parent tasks, support versions and ten queries."""
    root = tmp_path / "ConceptARC-100-10"
    for concept in ("AboveBelow", "FilledNotFilled"):
        directory = root / concept
        directory.mkdir(parents=True)
        for stem in (
            f"{concept}1_005",
            f"{concept}1_006",
            f"{concept}2_005",
            f"{concept}2_006",
            f"{concept}3",
        ):
            value = 1 if stem.endswith("005") else 2
            source = {
                "train": [
                    {"input": [[value, 0], [0, value]], "output": [[i, 0], [0, i]]}
                    for i in (3, 4, 5)
                ],
                "test": [{"input": [[i]], "output": [[i]]} for i in range(10)],
            }
            (directory / f"{stem}.json").write_text(json.dumps(source))
    return root


def build_records(
    dataset: Path, merge_newlines: bool = False
) -> tuple[list[dict[str, Any]], list[PromptReplaySample]]:
    """Build fixture records with the production formatter and span mapping."""
    return build_support_prompts(
        load_concept_arc(dataset),
        PromptTemplates.from_dir("prompts/concept_arc"),
        CharacterTokenizer(merge_newlines),
        ChatTemplateConfig(),
    )


@pytest.mark.parametrize("merge_newlines", [False, True])
def test_support_brackets_and_query_independence(
    dataset: Path, merge_newlines: bool
) -> None:
    """One row per version uses the exact output brackets, independent of queries."""
    records, samples = build_records(dataset, merge_newlines)
    assert len(records) == len(samples) == 10
    assert len({record["metadata"]["parent_task_id"] for record in records}) == 6
    assert records[0]["metadata"]["parent_task_id"] == "AboveBelow/AboveBelow1"
    assert records[0]["metadata"]["support_set_id"] == "005"
    assert records[4]["metadata"]["support_set_id"] == "canonical"
    assert all(
        record["test_index"] == -1 and record["case_id"].endswith("/support")
        for record in records
    )
    for record, sample in zip(records, samples, strict=True):
        assert "input 4:" not in record["prompt"]
        assert len(sample.token_ids) == sample.positions["S3"] + 1
        assert record["support_examples"][0]["output"] == [[3, 0], [0, 3]]
        for number, position in enumerate(SUPPORT_POSITIONS, start=1):
            start, end = record["prompt_char_spans"][f"support_{number}_full"]
            text = record["prompt"][start:end]
            assert text.startswith(f"input {number}:")
            assert text.endswith(f"[0 {number + 2}]")
            bracket = record["prompt_char_spans"][
                f"support_{number}_output_closing_bracket"
            ]
            assert bracket == [end - 1, end]
            assert sample.token_ids[sample.positions[position]] == (
                256 if merge_newlines else ord("]")
            )
        assert (
            record["token_spans"]["first_support_full"]
            == record["token_spans"]["support_1_full"]
        )
        assert (
            record["token_spans"]["last_support_full"]
            == record["token_spans"]["support_3_full"]
        )
    task = load_concept_arc(dataset)[0]
    for query in task.test:
        changed = replace(task, test=(replace(query, output=[[9, 9]]),))
        _, replay = build_support_prompts(
            [changed],
            PromptTemplates.from_dir("prompts/concept_arc"),
            CharacterTokenizer(merge_newlines),
            ChatTemplateConfig(),
        )
        assert replay[0] == samples[0]


def test_rejects_wrong_support_count(dataset: Path) -> None:
    """Require three supports rather than silently changing position meanings."""
    task = load_concept_arc(dataset)[0]
    with pytest.raises(ValueError, match="three complete supports"):
        build_support_prompts(
            [replace(task, train=task.train[:2])],
            PromptTemplates("", ""),
            CharacterTokenizer(),
            ChatTemplateConfig(),
        )


def test_cached_traversal_matches_independent_replays() -> None:
    """Every input token is processed once and each state matches its own replay."""
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=64,
            n_embd=16,
            n_layer=3,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    ).eval()
    samples = [
        PromptReplaySample(
            "A/support", list(range(1, 20)), {"S1": 4, "S2": 10, "S3": 18}
        ),
        PromptReplaySample(
            "B/support", list(range(2, 15)), {"S1": 0, "S2": 4, "S3": 12}
        ),
    ]
    calls: list[tuple[list[int], bool, bool]] = []

    def record_call(
        module: torch.nn.Module, args: tuple, kwargs: dict[str, Any]
    ) -> None:
        calls.append(
            (
                kwargs["input_ids"][0].tolist(),
                kwargs["output_hidden_states"],
                torch.is_grad_enabled(),
            )
        )

    handle = model.register_forward_pre_hook(record_call, with_kwargs=True)
    actual = list(
        iter_prompt_hidden_states_hf(model, samples, layers=[0, 2], chunk_size=3)
    )
    handle.remove()
    assert [token for tokens, _, _ in calls for token in tokens] == samples[
        0
    ].token_ids + samples[1].token_ids
    assert sum(capture for _, capture, _ in calls) == 6
    assert all(len(tokens) == 1 for tokens, capture, _ in calls if capture)
    assert not any(grad for _, _, grad in calls)
    assert model.training is False
    for sample, states in zip(samples, actual, strict=True):
        for position, index in sample.positions.items():
            with torch.inference_mode():
                output = model(
                    input_ids=torch.tensor([sample.token_ids[: index + 1]]),
                    use_cache=False,
                    output_hidden_states=True,
                )
            expected = torch.stack(
                [output.hidden_states[layer + 1][0, -1] for layer in (0, 2)]
            )
            np.testing.assert_allclose(
                states[position], expected.numpy(), atol=2e-3, rtol=2e-3
            )
            assert states[position].dtype == np.float16
            assert states[position].shape == (2, 16)
            assert np.isfinite(states[position]).all()


def fixture_config(dataset: Path) -> dict[str, Any]:
    """Create a common extraction contract for fixture archives."""
    tasks = load_concept_arc(dataset)
    return {
        "schema_version": 1,
        "extraction": {
            "layers": [0, 2],
            "concepts": sorted({task.concept for task in tasks}),
        },
        "dataset": {"id": "concept_arc_100_10", "task_hashes": task_inventory(tasks)},
        "selection": {"concept": None},
        "num_records": len(tasks),
    }


def write_fixture_run(
    destination: Path, records: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    """Write distinguishable feature rows through the production archive path."""
    states = (
        {
            position: np.full(
                (2, 4), sum(record["prompt_token_ids"]) % 300 + number, dtype=np.float16
            )
            for number, position in enumerate(SUPPORT_POSITIONS)
        }
        for record in records
    )
    with staged_output_directory(destination) as stage:
        write_prompt_feature_run(stage, copy.deepcopy(records), states, config)


@pytest.fixture
def shards(dataset: Path, tmp_path: Path) -> Path:
    """Write both fixture concept shards with the global label vocabulary."""
    records, _ = build_records(dataset)
    config = fixture_config(dataset)
    root = tmp_path / "prompt_features" / "shards"
    for concept in config["extraction"]["concepts"]:
        selected = [record for record in records if record["concept"] == concept]
        write_fixture_run(
            root / concept,
            selected,
            {**config, "selection": {"concept": concept}, "num_records": len(selected)},
        )
    return root


def test_merge_matches_unsharded_run(
    dataset: Path, shards: Path, tmp_path: Path
) -> None:
    """Merged rows, identities, labels, tokens and features match a whole run."""
    records, _ = build_records(dataset)
    whole = tmp_path / "whole"
    write_fixture_run(whole, records, fixture_config(dataset))
    merged = merge_prompt_feature_shards(dataset, shards)
    for position in SUPPORT_POSITIONS:
        actual, expected = load_feature_archive(
            merged / f"{position}.npz"
        ), load_feature_archive(whole / f"{position}.npz")
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        assert actual["x"].shape == (10, 2, 4)
        assert actual["y"].shape == (10, 2)
    assert load_generation_jsonl(merged / "manifest.jsonl") == load_generation_jsonl(
        whole / "manifest.jsonl"
    )
    actual_tokens = load_token_archive_npz(merged / "tokens.npz")
    for row, record in enumerate(records):
        assert (
            actual_tokens.get_by_index(row)["prompt_token_ids"]
            == record["prompt_token_ids"]
        )
    assert (shards / "AboveBelow" / "S1.npz").exists()
    with pytest.raises(FileExistsError):
        merge_prompt_feature_shards(dataset, shards)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "duplicate",
        "config",
        "dataset",
        "token_index",
        "token_ids",
        "feature_ids",
        "labels",
        "feature_metadata",
    ],
)
def test_merge_rejects_invalid_shards(
    dataset: Path, shards: Path, corruption: str
) -> None:
    """Reject incomplete, incompatible or misaligned inputs without publishing."""
    directory = shards / "AboveBelow"
    manifest = directory / "manifest.jsonl"
    if corruption == "missing":
        shutil.rmtree(directory)
    elif corruption == "duplicate":
        manifest.write_text(
            manifest.read_text() + manifest.read_text().splitlines()[0] + "\n"
        )
    elif corruption == "config":
        path = directory / "run_config.json"
        config = json.loads(path.read_text())
        config["extraction"]["layers"] = [1, 2]
        path.write_text(json.dumps(config))
    elif corruption == "dataset":
        path = next((dataset / "AboveBelow").glob("*.json"))
        path.write_text(path.read_text() + "\n")
    elif corruption == "token_index":
        rows = load_generation_jsonl(manifest)
        rows[0]["token_index"] = 1
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif corruption == "token_ids":
        with np.load(directory / "tokens.npz") as source:
            data = {key: source[key] for key in source.files}
        data["prompt_token_ids_data"][0] += 1
        np.savez_compressed(directory / "tokens.npz", **data)
    else:
        path = directory / "S2.npz"
        data = load_feature_archive(path)
        if corruption == "feature_ids":
            data["case_ids"] = data["case_ids"][::-1]
        elif corruption == "labels":
            data["y"] = data["y"][:, ::-1]
        else:
            data["metadata_json"] = data["metadata_json"][::-1]
        np.savez_compressed(path, **data)
    with pytest.raises(ValueError):
        merge_prompt_feature_shards(dataset, shards)
    assert not (shards.parent / "features").exists()
    assert not (shards.parent / ".features.lock").exists()


def test_failed_writer_cleans_stage_and_prevents_overwrites(tmp_path: Path) -> None:
    """An interrupted extraction can be retried without replacing completed data."""
    destination = tmp_path / "features"
    with pytest.raises(RuntimeError):
        with staged_output_directory(destination) as stage:
            (stage / "partial").write_text("partial")
            raise RuntimeError("failed replay")
    assert list(tmp_path.iterdir()) == []
    with staged_output_directory(destination) as stage:
        (stage / "complete").write_text("complete")
    with pytest.raises(FileExistsError):
        with staged_output_directory(destination):
            pytest.fail("Existing output must not be overwritten.")
    assert (destination / "complete").read_text() == "complete"


def test_overwrite_replaces_output_after_success(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Warn, retain old results during work, and discard stale files on success.

    Args:
        tmp_path (Path): Temporary output root.
        caplog (pytest.LogCaptureFixture): Captured overwrite warning.
    """
    destination = tmp_path / "results"
    (destination / "figures").mkdir(parents=True)
    (destination / "figures/obsolete.png").write_text("old figure")
    (destination / "summary.json").write_text("old summary")
    with staged_output_directory(destination, overwrite=True) as stage:
        assert "will be replaced after a successful run" in caplog.text
        assert (destination / "summary.json").read_text() == "old summary"
        (stage / "summary.json").write_text("new summary")
        with pytest.raises(FileExistsError):
            with staged_output_directory(destination, overwrite=True):
                pytest.fail("Overwrite must not bypass the active writer lock.")
    assert (destination / "summary.json").read_text() == "new summary"
    assert list(destination.iterdir()) == [destination / "summary.json"]
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("failure", ["run", "publish"])
def test_failed_overwrite_preserves_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Preserve old results when computation or publishing fails.

    Args:
        tmp_path (Path): Temporary output root.
        monkeypatch (pytest.MonkeyPatch): Simulate a publishing failure.
        failure (str): Stage at which the replacement fails.
    """
    destination = tmp_path / "results"
    destination.mkdir()
    (destination / "summary.json").write_text("old summary")
    rename = Path.rename
    with pytest.raises(RuntimeError if failure == "run" else OSError):
        with staged_output_directory(destination, overwrite=True) as stage:
            (stage / "summary.json").write_text("partial replacement")
            if failure == "run":
                raise RuntimeError("failed analysis")

            def fail_publish(path: Path, target: str | Path) -> Path:
                """Fail the new output rename while allowing rollback.

                Args:
                    path (Path): Rename source.
                    target (str | Path): Rename destination.

                Returns:
                    Path: Destination of an allowed rename.
                """
                if path == stage:
                    raise OSError("failed publishing")
                return rename(path, target)

            monkeypatch.setattr(Path, "rename", fail_publish)
    assert (destination / "summary.json").read_text() == "old summary"
    assert list(destination.iterdir()) == [destination / "summary.json"]
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.integration
def test_real_qwen_brackets_and_query_independence(dataset: Path) -> None:
    """Check all bracket positions and ten query variants with the local tokenizer."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", local_files_only=True)
    except OSError as error:
        pytest.skip(f"Qwen tokenizer is unavailable in the local cache: {error}")
    task = load_concept_arc(dataset)[0]
    templates = PromptTemplates.from_dir("prompts/concept_arc")
    original_records, original_samples = build_support_prompts(
        [task], templates, tokenizer, ChatTemplateConfig()
    )
    record, sample = original_records[0], original_samples[0]
    for number, position in enumerate(SUPPORT_POSITIONS, start=1):
        index = sample.positions[position]
        token = tokenizer.decode([sample.token_ids[index]])
        assert "]" in token
        decoded_prefix = tokenizer.decode(sample.token_ids[: index + 1])
        assert decoded_prefix.rstrip().endswith(f"[0 {number + 2}]")
        assert f"input {number + 1}:" not in decoded_prefix
        assert record["token_spans"][f"support_{number}_full"][1] - 1 == index
    for query in task.test:
        _, samples = build_support_prompts(
            [replace(task, test=(replace(query, output=None),))],
            templates,
            tokenizer,
            ChatTemplateConfig(),
        )
        assert samples == original_samples


def test_qwen_hybrid_cache_matches_full_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen's recurrent and attention caches agree with independent full forwards."""
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    for name in (
        "FusedRMSNormGated",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    ):
        monkeypatch.setattr(qwen, name, None)
    torch.manual_seed(5)
    torch.set_num_threads(1)
    model = qwen.Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [2, 1, 1],
            },
        )
    ).eval()
    sample = PromptReplaySample(
        "Qwen/support", list(range(1, 18)), {"S1": 4, "S2": 8, "S3": 16}
    )
    layers = [0, 1, 3]
    actual = list(
        iter_prompt_hidden_states_hf(model, [sample], layers=layers, chunk_size=3)
    )[0]
    with torch.inference_mode():
        for position, index in sample.positions.items():
            output = model(
                input_ids=torch.tensor([sample.token_ids[: index + 1]]),
                use_cache=False,
                output_hidden_states=True,
            )
            expected = np.stack(
                [
                    output.hidden_states[layer + 1][0, -1].to(torch.float16).numpy()
                    for layer in layers
                ]
            )
            np.testing.assert_allclose(actual[position], expected, atol=2e-3, rtol=2e-3)


def test_extraction_cli_selects_concept_and_loads_model_once(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the CLI through model loading, replay and publication on CPU."""
    import sys

    from scripts.repr_analysis import extract_prompt_features

    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=257,
            n_positions=2048,
            n_embd=16,
            n_layer=3,
            n_head=2,
        )
    ).eval()
    tokenizer = CharacterTokenizer()
    loads = []

    def load_model(*args: Any, **kwargs: Any) -> GPT2LMHeadModel:
        loads.append(kwargs)
        return model

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "tiny.yaml").write_text("model:\n  model: fixture\n")
    output = tmp_path / "cli_features"
    monkeypatch.setattr(
        extract_prompt_features, "load_tokenizer", lambda *args, **kwargs: tokenizer
    )
    monkeypatch.setattr(extract_prompt_features, "load_hf_model", load_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_prompt_features",
            "--data_dir",
            str(dataset),
            "--model_name",
            "tiny",
            "--model_config_dir",
            str(configs),
            "--layers",
            "0,2",
            "--hf_device_map",
            "cpu",
            "--concept_index",
            "1",
            "--output_dir",
            str(output),
            "--hf_chunk_size",
            "64",
        ],
    )
    extract_prompt_features.main()
    assert len(loads) == 1
    destination = output / "shards" / "FilledNotFilled"
    assert not (output / "shards" / "AboveBelow").exists()
    for position in SUPPORT_POSITIONS:
        archive = load_feature_archive(destination / f"{position}.npz")
        assert archive["x"].shape == (5, 2, 16)
        np.testing.assert_array_equal(archive["y"], [[0, 1]] * 5)
        assert np.isfinite(archive["x"]).all()
    config = json.loads((destination / "run_config.json").read_text())
    assert config["extraction"]["layers"] == [0, 2]
    assert config["extraction"]["hf_model_config"]["n_embd"] == 16
    assert (
        config["extraction"]["attention_implementation"]
        == model.config._attn_implementation
    )
    assert (
        config["extraction"]["torch_backends"]["cudnn_sdp"]
        == torch.backends.cuda.cudnn_sdp_enabled()
    )
    assert {"vllm", "flash_linear_attention", "causal_conv1d"} <= config["extraction"][
        "software"
    ].keys()
    assert config["extraction"]["config"]["model"]["model"] == "fixture"
    assert config["num_records"] == 5
    with pytest.raises(FileExistsError):
        extract_prompt_features.main()
    assert len(loads) == 1


def test_rejects_query_before_supports(dataset: Path) -> None:
    """A custom template cannot leak the test input into support features."""
    with pytest.raises(ValueError, match="Test query precedes"):
        build_support_prompts(
            load_concept_arc(dataset)[:1],
            PromptTemplates("", "query: {test_input}\n\n{examples}"),
            CharacterTokenizer(),
            ChatTemplateConfig(),
        )
