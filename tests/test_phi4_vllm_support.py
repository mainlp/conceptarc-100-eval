from __future__ import annotations

import os
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.config import ModelConfig, ReasoningConfig
from vllm.reasoning import ReasoningParserManager
from vllm.v1.sample import thinking_budget_state
from vllm.v1.sample.logits_processor import BatchUpdate
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

from arc_cd.eval.eval_response import evaluate_records
from arc_cd.models.config import (
    ChatTemplateConfig,
    GenerationConfig,
    VLLMReasoningConfig,
    load_experiment_config,
)
from arc_cd.models.generation_io import (
    load_generation_jsonl,
    load_token_archive_npz,
    write_generation_outputs,
)
from arc_cd.models.vllm_setup import (
    build_prompt_token_spans,
    build_reasoning_budget_metadata,
    build_token_spans,
    generate_from_records,
    load_vllm,
    make_sampling_params,
    prepare_prompts_for_reasoning_budget,
    render_prompt,
)
from arc_cd.prompt_prep.prompt_format import Message, PromptRecord
from arc_cd.utils import (
    infer_model_name_from_generation_path,
    resolve_reasoning_end_marker,
)
from scripts.eval_arc_acc import evaluate_generation_file
from scripts.summarize_eval_seed_stats import (
    aggregate_model_summaries,
    discover_seed_summaries,
)

CONFIG_PATH = Path("arc_cd/models/model_config/phi4_reasoning_plus.yaml")


def _record() -> PromptRecord:
    """Build an ARC query with two traceable support examples.

    Returns:
        PromptRecord: Small query with support character spans.
    """
    first = "input 1: [1 0]\n\noutput 1: [2 2]"
    last = "input 2: [3 0]\n\noutput 2: [4 4]"
    user = f"{first}\n\n{last}\n\ninput 3: [5 0]\n\noutput 3:"
    return PromptRecord(
        case_id="MockConcept/1/test_0",
        task_id="MockConcept/1",
        sample_id="1",
        test_index=0,
        source_path="mock.json",
        concept="MockConcept",
        messages=[
            {
                "role": "system",
                "content": "Solve the ARC puzzle. Only give the answer.",
            },
            {"role": "user", "content": user},
        ],
        target_grid=[[6, 6]],
        target_text="[6 6]",
        metadata={
            "prompt_char_spans": {
                "first_support_full": [0, len(first)],
                "last_support_full": [user.index(last), user.index(last) + len(last)],
            }
        },
    )


@pytest.fixture(scope="module")
def phi_tokenizer() -> PreTrainedTokenizerBase:
    """Load Phi's tokenizer from local files without downloading weights.

    Returns:
        PreTrainedTokenizerBase: Tokenizer from PHI4_TOKENIZER_NAME or the YAML path.
    """
    name = os.environ.get(
        "PHI4_TOKENIZER_NAME", load_experiment_config(CONFIG_PATH).model.model
    )
    try:
        return AutoTokenizer.from_pretrained(
            name, trust_remote_code=False, local_files_only=True
        )
    except OSError as error:
        pytest.skip(f"Phi tokenizer unavailable; set PHI4_TOKENIZER_NAME: {error}")


def test_phi_config_and_vllm_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass Phi's limits, sampler, seed, and reasoning settings to vLLM.

    Args:
        monkeypatch (pytest.MonkeyPatch): Model-construction patch fixture.
    """
    config = load_experiment_config(CONFIG_PATH, seed=44)
    assert Path(config.model.model).name == "Phi-4-reasoning-plus"
    assert config.model.max_model_len == 32768
    assert config.model.seed == config.generation.seed == 44
    assert config.chat.prepend_system_to_user is True
    assert config.chat.chat_template_kwargs == {}
    assert config.reasoning.ensure_reasoning_start_in_prompt is True
    assert config.reasoning.reasoning_end_marker == "</think>"
    params = make_sampling_params(config.generation, config.reasoning)
    assert params.max_tokens == 18432
    assert params.thinking_token_budget == 15360
    assert (params.temperature, params.top_p, params.top_k) == (0.8, 0.95, 50)
    assert params.presence_penalty == 0.0
    assert params.repetition_penalty == 1.0
    assert params.stop_token_ids == [100265]
    assert params.skip_special_tokens is False
    assert params.seed == 44

    llm_factory = Mock()
    monkeypatch.setattr("arc_cd.models.vllm_setup.LLM", llm_factory)
    assert load_vllm(config.model, config.reasoning) is llm_factory.return_value
    kwargs = llm_factory.call_args.kwargs
    assert kwargs["max_model_len"] == 32768
    assert kwargs["trust_remote_code"] is False
    reasoning = kwargs["reasoning_config"]
    assert reasoning.reasoning_parser == "deepseek_r1"
    assert reasoning.reasoning_start_str == "<think>"
    assert reasoning.reasoning_end_str == "\n</think>\n\n"


@pytest.mark.parametrize("prepend", [False, True])
def test_system_instructions_preserve_original_messages(prepend: bool) -> None:
    """Move instructions only when enabled and only into the first user message.

    Args:
        prepend (bool): Whether to enable system-to-user adaptation.
    """
    messages: list[Message] = [
        *_record().messages,
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Next query"},
    ]
    original = deepcopy(messages)
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "rendered"
    assert (
        render_prompt(
            messages, tokenizer, ChatTemplateConfig(prepend_system_to_user=prepend)
        )
        == "rendered"
    )
    rendered = tokenizer.apply_chat_template.call_args.args[0]
    expected = original
    if prepend:
        expected = [
            {
                "role": "user",
                "content": original[0]["content"] + "\n\n" + original[1]["content"],
            },
            *original[2:],
        ]
    assert rendered == expected
    assert messages == original


def test_system_adaptation_requires_user_message() -> None:
    """Reject system instructions that cannot be carried into a user message."""
    with pytest.raises(ValueError, match="without a user message"):
        render_prompt(
            [{"role": "system", "content": "Instructions"}],
            Mock(),
            ChatTemplateConfig(prepend_system_to_user=True),
        )


@pytest.mark.parametrize("remaining", [-1, 0, None])
def test_context_check_includes_prepared_reasoning_prefix(
    remaining: int | None,
) -> None:
    """Reject overflow before generation, accepting equality and an unset limit.

    Args:
        remaining (int | None): Spare context tokens, or no context check.
    """
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "assistant"
    tokenizer.encode.side_effect = lambda text, **kwargs: list(text.encode())
    llm = Mock()
    llm.generate.side_effect = RuntimeError("generation reached")
    final_prompt = "assistant<think>\n"
    required = len(final_prompt) + 32
    limit = None if remaining is None else required + remaining
    expected_error = ValueError if remaining == -1 else RuntimeError
    with pytest.raises(expected_error) as error:
        generate_from_records(
            llm,
            [_record()],
            tokenizer,
            ChatTemplateConfig(),
            GenerationConfig(max_tokens=32, thinking_token_budget=4),
            VLLMReasoningConfig(),
            max_model_len=limit,
        )
    if remaining == -1:
        assert _record().case_id in str(error.value)
        assert f"prompt_tokens={len(final_prompt)}" in str(error.value)
        assert "max_tokens=32" in str(error.value)
        assert f"max_model_len={limit}" in str(error.value)
        llm.generate.assert_not_called()
    else:
        assert str(error.value) == "generation reached"
        assert llm.generate.call_args.args[0] == [final_prompt]
        assert llm.generate.call_args.kwargs["sampling_params"].max_tokens == 32


@pytest.mark.integration
def test_phi_native_template_and_support_spans(
    phi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Keep Phi's native system prompt and align the original ARC support spans.

    Args:
        phi_tokenizer (PreTrainedTokenizerBase): Actual Phi tokenizer.
    """
    config = load_experiment_config(CONFIG_PATH)
    record = _record()
    original = deepcopy(record.messages)
    prompt = render_prompt(
        record.messages, phi_tokenizer, config.chat, config.reasoning
    )
    expected = phi_tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": original[0]["content"] + "\n\n" + original[1]["content"],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert prompt == expected
    assert "You are Phi, a language model trained by Microsoft" in prompt
    assert prompt.count(original[0]["content"]) == 1
    prompt = prepare_prompts_for_reasoning_budget(
        [prompt], config.chat, config.generation, config.reasoning
    )[0]
    assert prompt.endswith("<|im_start|>assistant<|im_sep|><think>\n")
    assert prepare_prompts_for_reasoning_budget(
        [prompt], config.chat, config.generation, config.reasoning
    ) == [prompt]
    assert record.messages == original
    assert phi_tokenizer.encode("<think>\n", add_special_tokens=False) == [100350]
    assert phi_tokenizer.encode(
        config.reasoning.reasoning_end_str, add_special_tokens=False
    ) == [100351]
    assert phi_tokenizer.eos_token_id == 100265
    parser = ReasoningParserManager.get_reasoning_parser("deepseek_r1")(phi_tokenizer)
    assert (parser.start_token_id, parser.end_token_id) == (100350, 100351)

    prompt_ids = phi_tokenizer.encode(prompt, add_special_tokens=False)
    assert phi_tokenizer.encode(prompt, add_special_tokens=True) == prompt_ids
    spans = build_prompt_token_spans(record, prompt, prompt_ids, phi_tokenizer)
    for key in ("first_support_full", "last_support_full"):
        char_start, char_end = record.metadata["prompt_char_spans"][key]
        start, end = spans[key]
        # Phi can merge trailing newlines into the token containing the final ].
        assert (
            phi_tokenizer.decode(prompt_ids[start:end]).rstrip()
            == original[1]["content"][char_start:char_end]
        )


@pytest.mark.integration
def test_phi_context_overflow_rejects_entire_batch(
    phi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Validate every adapted prompt before submitting any request to vLLM.

    Args:
        phi_tokenizer (PreTrainedTokenizerBase): Actual Phi tokenizer.
    """
    config = load_experiment_config(CONFIG_PATH)
    record = _record()
    longer = replace(
        record,
        case_id="MockConcept/2/test_0",
        messages=[
            record.messages[0],
            {
                "role": "user",
                "content": record.messages[1]["content"] + " extra context",
            },
        ],
    )
    prompt = render_prompt(
        record.messages, phi_tokenizer, config.chat, config.reasoning
    )
    prompt = prepare_prompts_for_reasoning_budget(
        [prompt], config.chat, config.generation, config.reasoning
    )[0]
    limit = (
        len(phi_tokenizer.encode(prompt, add_special_tokens=False))
        + config.generation.max_tokens
    )
    llm = Mock()
    with pytest.raises(ValueError, match=longer.case_id):
        generate_from_records(
            llm,
            [record, longer],
            phi_tokenizer,
            config.chat,
            config.generation,
            config.reasoning,
            max_model_len=limit,
        )
    llm.generate.assert_not_called()


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["forced", "natural", "zero", "repeated_start"])
def test_phi_controller_and_artifact_pipeline(
    phi_tokenizer: PreTrainedTokenizerBase,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    """Exercise real vLLM budget forcing through generation, evaluation, and stats.

    Args:
        phi_tokenizer (PreTrainedTokenizerBase): Actual Phi tokenizer.
        monkeypatch (pytest.MonkeyPatch): CPU transfer and tokenizer patch fixture.
        tmp_path (Path): Temporary generation and evaluation directory.
        mode (str): Forced, natural, zero-budget, or repeated-marker reasoning.
    """
    config = load_experiment_config(CONFIG_PATH, seed=42)
    body = phi_tokenizer.encode(" cat cat cat cat", add_special_tokens=False)
    assert len(body) == 4
    if mode == "zero":
        body = []
    elif mode == "repeated_start":
        body = [body[0], 100350, *body[1:]]
    budget = len(body) + (2 if mode == "natural" else 0)
    generation = replace(config.generation, thinking_token_budget=budget)
    reason_config = ReasoningConfig(
        reasoning_parser=config.reasoning.reasoning_parser,
        reasoning_start_str=config.reasoning.reasoning_start_str,
        reasoning_end_str=config.reasoning.reasoning_end_str,
    )
    monkeypatch.setattr(
        "vllm.config.reasoning.cached_tokenizer_from_config",
        lambda **kwargs: phi_tokenizer,
    )
    reason_config.initialize_token_ids(cast(ModelConfig, object()))
    # Only replace GPU transfer; state updates and logits forcing are vLLM's code.
    monkeypatch.setattr(
        thinking_budget_state,
        "async_tensor_h2d",
        lambda data, dtype, device: torch.tensor(data, dtype=dtype, device=device),
    )
    holder = ThinkingBudgetStateHolder(reason_config, 1, 0, torch.device("cpu"), False)
    record = _record()
    output_ids: list[int] = []
    final_ids = phi_tokenizer.encode("[6 6]<|im_end|>", add_special_tokens=False)

    def generate(
        prompts: Sequence[str], sampling_params: SamplingParams, use_tqdm: bool
    ) -> list[SimpleNamespace]:
        """Drive the actual sampler state with deterministic token proposals.

        Args:
            prompts (Sequence[str]): Final prepared prompt strings.
            sampling_params (SamplingParams): Phi's sampling parameters.
            use_tqdm (bool): Progress flag, unused by this simulation.

        Returns:
            list[SimpleNamespace]: One vLLM-compatible completed request.
        """
        prompt_ids = phi_tokenizer.encode(prompts[0], add_special_tokens=False)
        holder.sync_batch(
            BatchUpdate(1, [], [(0, sampling_params, prompt_ids, output_ids)], [])
        )
        close_proposal = 100351 if mode == "natural" else 42
        logits = torch.zeros((1, len(phi_tokenizer)))
        for proposal in [*body, close_proposal, *final_ids]:
            holder.update_state([output_ids], None)
            logits.zero_()
            logits[0, proposal] = 10.0
            output_ids.append(
                int(holder.apply_to_logits(logits, False, None).argmax(-1)[0])
            )
        assert output_ids == [*body, 100351, *final_ids]
        return [
            SimpleNamespace(
                prompt_token_ids=prompt_ids,
                outputs=[
                    SimpleNamespace(
                        token_ids=output_ids,
                        text=phi_tokenizer.decode(
                            output_ids,
                            skip_special_tokens=sampling_params.skip_special_tokens,
                        ),
                        finish_reason="stop",
                        stop_reason=100265,
                    )
                ],
            )
        ]

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        results = generate_from_records(
            cast(LLM, SimpleNamespace(generate=generate)),
            [record],
            phi_tokenizer,
            config.chat,
            generation,
            config.reasoning,
            max_model_len=config.model.max_model_len,
            use_tqdm=False,
        )
    finally:
        torch.set_num_threads(previous_threads)
    result = results[0]
    forced = mode != "natural"
    assert result["reasoning_budget"] == {
        "thinking_token_budget": budget,
        "reasoning_tokens": len(body),
        "hit_thinking_token_budget": forced,
        "forced_stop": forced,
        "over_budget": False,
    }
    spans = result["token_spans"]
    assert spans["reasoning_marker_generated"] == [len(body), len(body) + 1]
    start, end = spans["final_content_generated"]
    assert phi_tokenizer.decode(output_ids[start:end]) == "[6 6]"
    prompt_len = len(result["prompt_token_ids"])
    assert spans["final_content_full"] == [prompt_len + start, prompt_len + end]
    model_dir = tmp_path / "concept_arc_100" / "phi4_reasoning_plus"
    for seed in (42, 44):
        run_name = f"concept_arc_100_phi4_reasoning_plus_seed_{seed}"
        jsonl = (
            model_dir / "generations" / f"seed_{seed}" / f"{run_name}_generations.jsonl"
        )
        npz = jsonl.with_name(f"{run_name}_tokens.npz")
        write_generation_outputs(results, jsonl, npz)
        saved = load_generation_jsonl(jsonl)
        archive = load_token_archive_npz(npz)
        assert saved[0]["case_id"] == archive.case_ids[0] == record.case_id
        assert saved[0]["token_index"] == 0
        assert saved[0]["token_spans"] == spans
        assert saved[0]["messages"] == record.messages
        assert archive.prompt_token_ids.get(0) == result["prompt_token_ids"]
        assert archive.generated_token_ids.get(0) == output_ids
        assert infer_model_name_from_generation_path(jsonl) == "phi4_reasoning_plus"
        assert resolve_reasoning_end_marker(jsonl) == "</think>"
        summary = evaluate_generation_file(jsonl)
        assert summary["overall"]["accuracy"] == 1.0
        assert summary["overall"]["n_forced_reasoning_stop"] == int(forced)
        assert summary["overall"]["n_reasoning_over_budget"] == 0
    summaries = discover_seed_summaries(model_dir / "eval", "*_eval_summary.json")
    aggregate = aggregate_model_summaries(summaries, model_dir / "eval")
    assert aggregate["model_name"] == "phi4_reasoning_plus"
    assert aggregate["seeds"] == [42, 44]
    assert aggregate["overall_scalar_stats"]["accuracy"]["mean"] == 1.0
    evaluations, summary = evaluate_records([{**results[0], "finish_reason": "length"}])
    assert evaluations[0].correct is False
    assert summary["overall"]["n_unfinished"] == 1


@pytest.mark.integration
def test_phi_answer_marker_does_not_move_reasoning_start(
    phi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Keep the first reasoning close when the answer mentions an opening tag.

    Args:
        phi_tokenizer (PreTrainedTokenizerBase): Actual Phi tokenizer.
    """
    config = load_experiment_config(CONFIG_PATH)
    prompt_ids = phi_tokenizer.encode("<think>\n", add_special_tokens=False)
    body = phi_tokenizer.encode(" cat", add_special_tokens=False)
    answer = phi_tokenizer.encode("[6 6] <think>", add_special_tokens=False)
    output_ids = [*body, 100351, *answer]
    spans = build_token_spans(prompt_ids, output_ids, phi_tokenizer)
    assert spans["reasoning_content_generated"] == [0, len(body)]
    assert spans["reasoning_marker_generated"] == [len(body), len(body) + 1]
    assert spans["final_content_generated"] == [len(body) + 1, len(output_ids)]
    metadata = build_reasoning_budget_metadata(
        prompt_ids, output_ids, phi_tokenizer, config.generation, config.reasoning
    )
    assert metadata["reasoning_tokens"] == len(body)
    assert metadata["forced_stop"] is False
    assert metadata["over_budget"] is False
