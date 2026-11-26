from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import BatchUpdate

from arc_cd.eval.eval_response import evaluate_records
from arc_cd.eval.parsing import parse_grid_from_generation
from arc_cd.models import harmony_budget
from arc_cd.models.config import (
    ChatTemplateConfig,
    VLLMReasoningConfig,
    load_experiment_config,
)
from arc_cd.models.generation_io import (
    load_generation_jsonl,
    load_token_archive_npz,
    write_generation_outputs,
)
from arc_cd.models.harmony_budget import (
    HARMONY_BUDGET_KEY,
    HarmonyThinkingBudgetProcessor,
    HarmonyTokenIds,
)
from arc_cd.models.vllm_setup import (
    build_harmony_token_spans,
    build_prompt_token_spans,
    generate_from_records,
    render_prompts,
)
from arc_cd.prompt_prep.prompt_format import PromptRecord
from arc_cd.utils import (
    infer_model_name_from_generation_path,
    resolve_reasoning_end_marker,
)
from scripts.run_arc_vLLM import build_run_name

HARMONY_ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"


@pytest.fixture
def gpt_oss_tokenizer() -> PreTrainedTokenizerBase:
    """Load the local GPT-OSS tokenizer if it is available.

    Returns:
        PreTrainedTokenizerBase: Cached GPT-OSS tokenizer.
    """
    try:
        return AutoTokenizer.from_pretrained("openai/gpt-oss-20b", local_files_only=True)
    except OSError as error:
        pytest.skip(f"GPT-OSS tokenizer is unavailable in the local cache: {error}")


def _record() -> PromptRecord:
    """Build a small ARC-style prompt record.

    Returns:
        PromptRecord: Prompt record with support character spans.
    """
    first_support = "input 1: [1 0]\n\noutput 1: [2 2]"
    last_support = "input 2: [3 0]\n\noutput 2: [4 4]"
    user_prompt = f"{first_support}\n\n{last_support}\n\ninput 3: [5 0]\n\noutput 3:"
    last_start = user_prompt.index(last_support)

    return PromptRecord(
        case_id="mock/test_0",
        task_id="mock",
        sample_id="mock",
        test_index=0,
        source_path="mock.json",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful ARC solver. Only give the answer.",
            },
            {"role": "user", "content": user_prompt},
        ],
        target_grid=[[6, 6]],
        target_text="[6 6]",
        concept="mock",
        metadata={
            "prompt_char_spans": {
                "first_support_full": [0, len(first_support)],
                "last_support_full": [
                    last_start,
                    last_start + len(last_support),
                ],
            }
        },
    )


@pytest.mark.parametrize("model_name", ["gpt_oss_20b", "gpt_oss_120b"])
def test_gpt_oss_configs_load_reasoning_budget(model_name: str) -> None:
    """Load both GPT-OSS configs and verify their reasoning budgets.

    Args:
        model_name (str): GPT-OSS model configuration name.
    """
    config = load_experiment_config(
        Path(f"arc_cd/models/model_config/{model_name}.yaml"),
        seed=7,
    )

    assert Path(config.model.model).name == model_name.replace("_", "-")
    assert config.model.seed == 7
    assert config.chat.renderer == "harmony"
    assert config.generation.stop_token_ids == [200002, 200012]
    assert config.generation.skip_special_tokens is False
    assert config.reasoning.reasoning_parser == "openai_gptoss"
    assert config.generation.thinking_token_budget == 49152
    assert config.generation.max_tokens == 65536
    assert config.reasoning.reasoning_effort == "high"
    assert config.reasoning.ensure_reasoning_start_in_prompt is False
    assert config.reasoning.reasoning_end_str == (
        "<|end|><|start|>assistant" + HARMONY_FINAL_MARKER
    )
    assert config.reasoning.reasoning_end_marker == HARMONY_FINAL_MARKER


def test_reasoning_effort_mid_normalizes_to_medium(tmp_path: Path) -> None:
    """Normalize the YAML alias ``mid`` to ``medium``."""
    config_path = tmp_path / "model.yaml"
    config_path.write_text("reasoning:\n  reasoning_effort: mid\n", encoding="utf-8")

    config = load_experiment_config(config_path)

    assert config.reasoning.reasoning_effort == "medium"


def test_invalid_reasoning_effort_raises(tmp_path: Path) -> None:
    """Reject unsupported reasoning-effort values."""
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        "reasoning:\n  reasoning_effort: extreme\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reasoning_effort"):
        load_experiment_config(config_path)


def test_hf_harmony_rendering_preserves_prompt_effort_and_token_ids(
    gpt_oss_tokenizer,
) -> None:
    """Render GPT-OSS prompts with HF's shipped Harmony chat template."""
    record = _record()
    rendered_prompts = render_prompts(
        [record],
        gpt_oss_tokenizer,
        ChatTemplateConfig(renderer="harmony"),
        reasoning_config=VLLMReasoningConfig(
            reasoning_parser="openai_gptoss",
            reasoning_effort="high",
            reasoning_start_str=HARMONY_ANALYSIS_MARKER,
            reasoning_end_marker=HARMONY_FINAL_MARKER,
            reasoning_end_str=HARMONY_FINAL_MARKER,
            ensure_reasoning_start_in_prompt=False,
        ),
    )
    rendered = rendered_prompts[0]
    prompt_token_ids = gpt_oss_tokenizer(
        rendered,
        add_special_tokens=False,
    )["input_ids"]
    template_token_ids = gpt_oss_tokenizer.apply_chat_template(
        record.messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        reasoning_effort="high",
    )

    assert prompt_token_ids == template_token_ids
    assert "Reasoning: high" in rendered
    assert "You are a helpful ARC solver" in rendered
    assert "<|start|>developer<|message|># Instructions" in rendered
    assert record.messages[-1]["content"] in rendered

    token_spans = build_prompt_token_spans(
        record,
        rendered,
        prompt_token_ids,
        gpt_oss_tokenizer,
    )

    assert token_spans["first_support_full"][0] < token_spans["first_support_full"][1]
    assert token_spans["last_support_full"][0] < token_spans["last_support_full"][1]
    first_span = token_spans["first_support_full"]
    last_span = token_spans["last_support_full"]
    first_text = gpt_oss_tokenizer.decode(
        prompt_token_ids[first_span[0] : first_span[1]],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    last_text = gpt_oss_tokenizer.decode(
        prompt_token_ids[last_span[0] : last_span[1]],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    assert "input 1:" in first_text
    assert "output 1:" in first_text
    assert "input 2:" in last_text
    assert "output 2:" in last_text


def test_harmony_token_spans_and_final_answer_parse(gpt_oss_tokenizer) -> None:
    """Build analysis/final token spans from a GPT-OSS completion."""
    prompt_token_ids = gpt_oss_tokenizer.encode(
        "<|start|>assistant",
        add_special_tokens=False,
    )
    generated_text = (
        f"{HARMONY_ANALYSIS_MARKER}reason carefully"
        "<|end|><|start|>assistant"
        f"{HARMONY_FINAL_MARKER}[1 2] [3 4]<|return|>"
    )
    generated_token_ids = gpt_oss_tokenizer.encode(
        generated_text,
        add_special_tokens=False,
    )

    spans = build_harmony_token_spans(
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        tokenizer=gpt_oss_tokenizer,
        reasoning_start_str=HARMONY_ANALYSIS_MARKER,
        reasoning_end_marker=HARMONY_FINAL_MARKER,
        thinking_enabled=True,
    )

    assert spans["reasoning_content_generated"] is not None
    assert spans["final_content_generated"] is not None
    assert spans["generated_content_generated"] is not None

    reasoning_start, reasoning_end = spans["reasoning_content_generated"]
    final_start, final_end = spans["final_content_generated"]
    generated_start, generated_end = spans["generated_content_generated"]

    assert (
        gpt_oss_tokenizer.decode(
            generated_token_ids[reasoning_start:reasoning_end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        == "reason carefully"
    )
    assert (
        gpt_oss_tokenizer.decode(
            generated_token_ids[final_start:final_end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        == "[1 2] [3 4]"
    )
    assert generated_start == reasoning_start
    assert generated_end == final_end

    for name, start, end in (
        ("generated_content_full", generated_start, generated_end),
        ("reasoning_content_full", reasoning_start, reasoning_end),
        ("final_content_full", final_start, final_end),
    ):
        assert spans[name] == [
            len(prompt_token_ids) + start,
            len(prompt_token_ids) + end,
        ]

    parsed = parse_grid_from_generation(
        {
            "generated_text": gpt_oss_tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "finish_reason": "stop",
        },
        end_think_token=HARMONY_FINAL_MARKER,
    )

    assert parsed.grid == [[1, 2], [3, 4]]


def test_gpt_oss_output_name_and_model_inference() -> None:
    """Include reasoning effort in run names while preserving model inference."""
    run_name = build_run_name(
        dataset_id="arc_agi_1",
        split_spec="_training",
        model_name="gpt_oss_20b",
        seed=42,
        reasoning_config=VLLMReasoningConfig(reasoning_effort="high"),
    )
    path = (
        Path("output")
        / "arc_agi_1"
        / "gpt_oss_20b"
        / "generations"
        / "seed_42"
        / f"{run_name}_generations.jsonl"
    )

    assert run_name == "arc_agi_1_training_gpt_oss_20b_reasoning_effort_high_seed_42"
    assert infer_model_name_from_generation_path(path) == "gpt_oss_20b"
    assert resolve_reasoning_end_marker(path) == HARMONY_FINAL_MARKER


@pytest.mark.parametrize("force_transition", [False, True])
def test_harmony_budget_generation_io_and_evaluation(
    gpt_oss_tokenizer: PreTrainedTokenizerBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force_transition: bool,
) -> None:
    """Exercise real Harmony tokens through CPU logits, generation I/O, and eval.

    Args:
        gpt_oss_tokenizer (PreTrainedTokenizerBase): Local GPT-OSS tokenizer.
        tmp_path (Path): Temporary artifact directory.
        monkeypatch (pytest.MonkeyPatch): Engine tokenizer patch fixture.
        force_transition (bool): Whether body tokens exhaust the reasoning cap.
    """
    config = load_experiment_config(
        Path("arc_cd/models/model_config/gpt_oss_20b.yaml"), seed=42
    )
    markers = HarmonyTokenIds.from_tokenizer(gpt_oss_tokenizer)
    body = gpt_oss_tokenizer.encode(" cat\n", add_special_tokens=False)
    assert len(body) == 2  # The standalone newline must consume one budget token.
    budget = len(body) if force_transition else len(body) + 1
    generation_config = replace(config.generation, thinking_token_budget=budget)
    record = _record()
    expected_prompt = render_prompts(
        [record], gpt_oss_tokenizer, config.chat, config.reasoning
    )[0]
    assert expected_prompt.endswith("<|start|>assistant")
    monkeypatch.setattr(
        harmony_budget,
        "cached_tokenizer_from_config",
        lambda **kwargs: gpt_oss_tokenizer,
    )
    engine_config = cast(
        VllmConfig, SimpleNamespace(speculative_config=None, model_config=object())
    )
    processor = HarmonyThinkingBudgetProcessor(
        engine_config, torch.device("cpu"), False
    )
    accepted_tokens: list[int] = []
    prompt_ids = gpt_oss_tokenizer.encode(expected_prompt, add_special_tokens=False)
    final_ids = gpt_oss_tokenizer.encode("[6 6]<|return|>", add_special_tokens=False)
    calls = 0

    def generate(
        prompts: Sequence[str], sampling_params: SamplingParams, use_tqdm: bool
    ) -> list[SimpleNamespace]:
        """Replace model logits with deterministic proposals on the real vocabulary.

        Args:
            prompts (Sequence[str]): Rendered inference prompts.
            sampling_params (SamplingParams): Configured request parameters.
            use_tqdm (bool): Ignored progress-bar flag.

        Returns:
            list[SimpleNamespace]: One vLLM-compatible completion.
        """
        nonlocal calls
        calls += 1
        assert list(prompts) == [expected_prompt]
        assert sampling_params.max_tokens == 65536
        assert sampling_params.thinking_token_budget is None
        assert sampling_params.extra_args == {HARMONY_BUDGET_KEY: budget}
        processor.update_state(
            BatchUpdate(1, [], [(0, sampling_params, prompt_ids, accepted_tokens)], [])
        )
        proposals = [*markers.analysis, *body]
        proposals += (
            [body[0]] * len(markers.transition)
            if force_transition
            else list(markers.transition)
        )
        proposals += final_ids
        logits = torch.zeros((1, len(gpt_oss_tokenizer)))
        for proposed in proposals:
            logits.fill_(0.0)
            logits[0, proposed] = 10.0
            accepted_tokens.append(int(processor.apply(logits)[0].argmax()))
            processor.update_state(None)
        assert accepted_tokens == [
            *markers.analysis,
            *body,
            *markers.transition,
            *final_ids,
        ]
        return [
            SimpleNamespace(
                prompt_token_ids=prompt_ids,
                outputs=[
                    SimpleNamespace(
                        token_ids=accepted_tokens,
                        text="",
                        finish_reason="stop",
                        stop_reason=final_ids[-1],
                    )
                ],
            )
        ]

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        results = generate_from_records(
            llm=cast(LLM, SimpleNamespace(generate=generate)),
            records=[record],
            tokenizer=gpt_oss_tokenizer,
            chat_config=config.chat,
            generation_config=generation_config,
            reasoning_config=config.reasoning,
            use_tqdm=False,
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert calls == 1
    result = results[0]
    assert result["reasoning_budget"] == {
        "thinking_token_budget": budget,
        "reasoning_tokens": len(body),
        "hit_thinking_token_budget": force_transition,
        "forced_stop": force_transition,
        "over_budget": False,
        "reasoning_effort": "low",
    }
    spans = result["token_spans"]
    start, end = spans["final_content_generated"]
    assert gpt_oss_tokenizer.decode(accepted_tokens[start:end]) == "[6 6]"
    assert spans["final_content_full"] == [
        len(prompt_ids) + start,
        len(prompt_ids) + end,
    ]
    jsonl_path, npz_path = tmp_path / "generations.jsonl", tmp_path / "tokens.npz"
    write_generation_outputs(results, jsonl_path, npz_path)
    saved = load_generation_jsonl(jsonl_path)
    archive = load_token_archive_npz(npz_path)
    assert saved[0]["case_id"] == record.case_id == archive.case_ids[0]
    assert saved[0]["token_index"] == 0
    assert saved[0]["token_spans"] == spans
    assert saved[0]["generated_text"] == gpt_oss_tokenizer.decode(
        accepted_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert archive.prompt_token_ids.get(0) == prompt_ids
    assert archive.generated_token_ids.get(0) == accepted_tokens
    evaluations, summary = evaluate_records(saved, end_think_token=HARMONY_FINAL_MARKER)
    assert evaluations[0].correct is True
    assert evaluations[0].forced_reasoning_stop is force_transition
    assert summary["overall"]["n_reasoning_over_budget"] == 0

    unfinished = [{**saved[0], "finish_reason": "length"}]
    evaluations, summary = evaluate_records(
        unfinished, end_think_token=HARMONY_FINAL_MARKER
    )
    assert evaluations[0].correct is False
    assert summary["overall"]["n_unfinished"] == 1
