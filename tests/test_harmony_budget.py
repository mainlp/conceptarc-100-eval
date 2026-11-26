"""Check Harmony budget boundaries, request isolation, and vLLM wiring on CPU."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality

from arc_cd.models import harmony_budget, vllm_setup
from arc_cd.models.config import GenerationConfig, VLLMModelConfig, VLLMReasoningConfig
from arc_cd.models.harmony_budget import (
    HARMONY_ANALYSIS_MARKER,
    HARMONY_BUDGET_KEY,
    HARMONY_FINAL_MARKER,
    HARMONY_FINAL_TRANSITION,
    HarmonyBudgetRequest,
    HarmonyReasoningState,
    HarmonyThinkingBudgetProcessor,
    HarmonyTokenIds,
    harmony_budget_metadata,
)

MARKERS = HarmonyTokenIds(
    analysis=(3, 12, 4),
    final=(3, 13, 4),
    transition=(1, 2, 7, 3, 13, 4),
    message_end=1,
    turn_ends=frozenset({0, 5}),
)
PROMPT = [2, 7]
BODY_TOKEN = 20
CONFIG = VLLMReasoningConfig(
    reasoning_parser="openai_gptoss",
    reasoning_start_str=HARMONY_ANALYSIS_MARKER,
    reasoning_end_marker=HARMONY_FINAL_MARKER,
    reasoning_end_str=HARMONY_FINAL_TRANSITION,
    ensure_reasoning_start_in_prompt=False,
)


def sample_token(
    request: HarmonyBudgetRequest, output: list[int], proposed: int
) -> int:
    """Sample one deterministic model proposal through the real logits controller.

    Args:
        request (HarmonyBudgetRequest): Request controller.
        output (list[int]): Previously accepted tokens.
        proposed (int): Token preferred by the mock model.

    Returns:
        int: Accepted token, including any enforced transition.
    """
    logits = torch.zeros(32)
    logits[proposed] = 10
    return int(request(PROMPT, output, logits).argmax())


@pytest.mark.parametrize("budget", [0, 1, 4, 49152])
def test_forces_complete_transition_at_exact_body_budget(budget: int) -> None:
    """Leave no extra reasoning token and release the model after final framing.

    Args:
        budget (int): Allowed body-token count.
    """
    request = HarmonyBudgetRequest(MARKERS, budget)
    output = list(MARKERS.analysis) + [BODY_TOKEN] * budget
    for expected in MARKERS.transition:
        token = sample_token(request, output, BODY_TOKEN)
        assert token == expected
        output.append(token)
    assert sample_token(request, output, 21) == 21
    assert harmony_budget_metadata(PROMPT, output, MARKERS, budget) == {
        "reasoning_tokens": budget,
        "hit_thinking_token_budget": True,
        "forced_stop": True,
        "over_budget": False,
    }


@pytest.mark.parametrize("body_length", [0, 3, 4, 5, 7])
def test_natural_transition_crossing_numeric_boundary_is_not_duplicated(
    body_length: int,
) -> None:
    """Exclude every header token even if the natural transition crosses the cap.

    Args:
        body_length (int): Number of body tokens before a natural transition.
    """
    request = HarmonyBudgetRequest(MARKERS, 8)
    expected = (
        list(MARKERS.analysis)
        + [BODY_TOKEN] * body_length
        + list(MARKERS.transition)
        + [21, 0]
    )
    output: list[int] = []
    for proposed in expected:
        output.append(sample_token(request, output, proposed))
    assert output == expected
    metadata = harmony_budget_metadata(PROMPT, output, MARKERS, 8)
    assert metadata["reasoning_tokens"] == body_length
    assert metadata["hit_thinking_token_budget"] is False
    assert metadata["forced_stop"] is False


def test_budget_accumulates_across_analysis_messages() -> None:
    """Opening a second analysis message must not reset the budget."""
    request = HarmonyBudgetRequest(MARKERS, 4)
    output = (
        list(MARKERS.analysis)
        + [BODY_TOKEN] * 2
        + [1, 2, 7]
        + list(MARKERS.analysis)
        + [BODY_TOKEN] * 2
    )
    for expected in MARKERS.transition:
        assert sample_token(request, output, BODY_TOKEN) == expected
        output.append(expected)
    assert harmony_budget_metadata(PROMPT, output, MARKERS, 4)["reasoning_tokens"] == 4


def test_direct_final_answer_is_untouched_with_zero_budget() -> None:
    """A zero budget does not force an unnecessary analysis message."""
    request = HarmonyBudgetRequest(MARKERS, 0)
    output: list[int] = []
    for token in [*MARKERS.final, BODY_TOKEN, 0]:
        assert sample_token(request, output, token) == token
        output.append(token)
    assert harmony_budget_metadata(PROMPT, output, MARKERS, 0) == {
        "reasoning_tokens": 0,
        "hit_thinking_token_budget": False,
        "forced_stop": False,
        "over_budget": False,
    }


def test_open_prompt_analysis_counts_but_previous_messages_do_not() -> None:
    """Ignore closed prompt messages while supporting an unfinished analysis prefix."""
    historical = [2, 7, *MARKERS.analysis, BODY_TOKEN, 1, 2, 7, *MARKERS.final, 21, 1]
    prompt = historical + [2, 7, *MARKERS.analysis, BODY_TOKEN]
    state = HarmonyReasoningState(MARKERS, prompt)
    assert state.reasoning_tokens == 1
    assert state.in_analysis
    request = HarmonyBudgetRequest(MARKERS, 2)
    logits = request(prompt, [BODY_TOKEN], torch.zeros(32))
    assert int(logits.argmax()) == MARKERS.message_end
    assert harmony_budget_metadata(
        prompt, [BODY_TOKEN, *MARKERS.transition], MARKERS, 2
    )["forced_stop"]


@pytest.mark.parametrize("close_length", [0, 1, 3, 5])
def test_incomplete_transition_is_not_a_completed_forced_stop(
    close_length: int,
) -> None:
    """Reaching the total output limit during transition remains incomplete.

    Args:
        close_length (int): Number of available transition tokens.
    """
    output = [*MARKERS.analysis, BODY_TOKEN, *MARKERS.transition[:close_length]]
    metadata = harmony_budget_metadata(PROMPT, output, MARKERS, 1)
    assert metadata["reasoning_tokens"] == 1
    assert metadata["hit_thinking_token_budget"] is True
    assert metadata["forced_stop"] is False


def test_over_budget_records_remain_detectable() -> None:
    """Diagnostic counting must not clamp already-generated over-budget records."""
    output = [*MARKERS.analysis, *([BODY_TOKEN] * 5), *MARKERS.transition]
    metadata = harmony_budget_metadata(PROMPT, output, MARKERS, 4)
    assert metadata["reasoning_tokens"] == 5
    assert metadata["over_budget"] is True
    assert metadata["forced_stop"] is False


@pytest.fixture
def processor(monkeypatch: pytest.MonkeyPatch) -> HarmonyThinkingBudgetProcessor:
    """Build the actual vLLM batch adapter with a tiny marker vocabulary.

    Args:
        monkeypatch (pytest.MonkeyPatch): Tokenizer factory patch fixture.

    Returns:
        HarmonyThinkingBudgetProcessor: CPU controller.
    """
    monkeypatch.setattr(
        harmony_budget, "cached_tokenizer_from_config", lambda **kwargs: None
    )
    monkeypatch.setattr(
        HarmonyTokenIds, "from_tokenizer", classmethod(lambda cls, tokenizer: MARKERS)
    )
    config = cast(
        VllmConfig, SimpleNamespace(speculative_config=None, model_config=object())
    )
    return HarmonyThinkingBudgetProcessor(config, torch.device("cpu"), False)


def test_adapter_keeps_budgets_with_batch_moves_and_request_reuse(
    processor: HarmonyThinkingBudgetProcessor,
) -> None:
    """Requests retain their own counters when vLLM swaps, moves, or removes rows.

    Args:
        processor (HarmonyThinkingBudgetProcessor): CPU batch adapter.
    """
    params = SamplingParams(extra_args={HARMONY_BUDGET_KEY: 1})
    exhausted = [*MARKERS.analysis, BODY_TOKEN]
    fresh = list(MARKERS.analysis)
    processor.update_state(
        BatchUpdate(
            2, [], [(0, params, PROMPT, exhausted), (1, params, PROMPT, fresh)], []
        )
    )
    logits = processor.apply(torch.ones(2, 32))
    assert torch.isfinite(logits[0]).sum() == 1
    assert torch.isfinite(logits[1]).all()
    exhausted.append(MARKERS.transition[0])
    processor.update_state(BatchUpdate(2, [], [], [(0, 1, MoveDirectionality.SWAP)]))
    logits = processor.apply(torch.ones(2, 32))
    assert torch.isfinite(logits[0]).all()
    assert int(logits[1].argmax()) == MARKERS.transition[1]
    processor.update_state(
        BatchUpdate(1, [0], [], [(1, 0, MoveDirectionality.UNIDIRECTIONAL)])
    )
    assert int(processor.apply(torch.ones(1, 32))[0].argmax()) == MARKERS.transition[1]
    replacement = list(MARKERS.analysis)
    zero_budget = SamplingParams(extra_args={HARMONY_BUDGET_KEY: 0})
    processor.update_state(
        BatchUpdate(1, [0], [(0, zero_budget, PROMPT, replacement)], [])
    )
    logits = processor.apply(torch.ones(1, 32))
    assert torch.isfinite(logits[0]).sum() == 1
    assert int(logits[0].argmax()) == MARKERS.message_end
    processor.update_state(BatchUpdate(1, [0], [(0, SamplingParams(), PROMPT, [])], []))
    assert torch.isfinite(processor.apply(torch.ones(1, 32))).all()


@pytest.mark.parametrize("budget", [-1, True, 2.5])
def test_adapter_rejects_invalid_budgets(budget: object) -> None:
    """Reject configurations that cannot represent a token count.

    Args:
        budget (object): Invalid requested budget.
    """
    with pytest.raises(ValueError, match="nonnegative integer"):
        HarmonyThinkingBudgetProcessor.validate_params(
            SamplingParams(extra_args={HARMONY_BUDGET_KEY: budget})
        )


def test_adapter_rejects_generic_controller_and_interrupted_transition(
    processor: HarmonyThinkingBudgetProcessor,
) -> None:
    """Fail explicitly for conflicting controllers or stop tokens.

    Args:
        processor (HarmonyThinkingBudgetProcessor): CPU batch adapter.
    """
    with pytest.raises(ValueError, match="only the Harmony controller"):
        processor.new_req_logits_processor(
            SamplingParams(thinking_token_budget=4, extra_args={HARMONY_BUDGET_KEY: 4})
        )
    with pytest.raises(ValueError, match="Stop tokens"):
        processor.new_req_logits_processor(
            SamplingParams(stop_token_ids=[1], extra_args={HARMONY_BUDGET_KEY: 4})
        )
    with pytest.raises(ValueError, match="speculative"):
        HarmonyThinkingBudgetProcessor(
            cast(VllmConfig, SimpleNamespace(speculative_config=object())),
            torch.device("cpu"),
            False,
        )


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_sampling_params_route_gpt_oss_budget(effort: str) -> None:
    """Keep effort independent of the hard cap and preserve sampling settings.

    Args:
        effort (str): GPT-OSS reasoning effort.
    """
    params = vllm_setup.make_sampling_params(
        GenerationConfig(max_tokens=65536, thinking_token_budget=49152, seed=42),
        replace(CONFIG, reasoning_effort=effort),
    )
    assert params.thinking_token_budget is None
    assert params.extra_args == {HARMONY_BUDGET_KEY: 49152}
    assert params.max_tokens == 65536
    assert params.seed == 42
    assert params.temperature == 0.6


def test_unbudgeted_replay_and_non_harmony_sampling_are_unchanged() -> None:
    """Preserve ordinary vLLM budgets and disable the controller for answer replays."""
    replay = vllm_setup.make_sampling_params(GenerationConfig(), CONFIG)
    assert replay.thinking_token_budget is None
    assert not replay.extra_args
    params = vllm_setup.make_sampling_params(
        GenerationConfig(thinking_token_budget=4), VLLMReasoningConfig()
    )
    assert params.thinking_token_budget == 4
    assert not params.extra_args
    with pytest.raises(ValueError, match="reasoning_end_str"):
        vllm_setup.make_sampling_params(
            GenerationConfig(thinking_token_budget=4),
            replace(CONFIG, reasoning_end_str=HARMONY_FINAL_MARKER),
        )


def test_engine_registers_controller_only_for_gpt_oss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check engine construction without loading weights or initializing a GPU.

    Args:
        monkeypatch (pytest.MonkeyPatch): LLM constructor patch fixture.
    """
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(vllm_setup, "LLM", lambda **kwargs: calls.append(kwargs))
    for reasoning in (CONFIG, VLLMReasoningConfig(), replace(CONFIG, enabled=False)):
        vllm_setup.load_vllm(VLLMModelConfig(), reasoning_config=reasoning)
    assert calls[0]["logits_processors"] == [
        "arc_cd.models.harmony_budget:HarmonyThinkingBudgetProcessor"
    ]
    assert "logits_processors" not in calls[1]
    assert "logits_processors" not in calls[2]
