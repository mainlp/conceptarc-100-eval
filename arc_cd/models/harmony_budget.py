"""Enforce GPT-OSS reasoning budgets without interrupting Harmony headers."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor, BatchUpdate

from arc_cd.models.config import VLLMReasoningConfig

HARMONY_BUDGET_KEY = "arc_cd_harmony_thinking_token_budget"
HARMONY_ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"
HARMONY_FINAL_TRANSITION = "<|end|><|start|>assistant" + HARMONY_FINAL_MARKER


def validate_harmony_reasoning_config(config: VLLMReasoningConfig) -> None:
    """Require the channel markers understood by the GPT-OSS controller.

    Args:
        config (VLLMReasoningConfig): GPT-OSS reasoning configuration.

    Raises:
        ValueError: If the configured markers would produce invalid framing.
    """
    if (
        config.reasoning_start_str != HARMONY_ANALYSIS_MARKER
        or config.reasoning_end_marker != HARMONY_FINAL_MARKER
        or config.reasoning_end_str != HARMONY_FINAL_TRANSITION
    ):
        raise ValueError(
            "GPT-OSS budget enforcement requires the standard analysis/final "
            f"markers and reasoning_end_str={HARMONY_FINAL_TRANSITION!r}."
        )


@dataclass(frozen=True)
class HarmonyTokenIds:
    """Tokenizer-specific Harmony framing.

    Args:
        analysis (tuple[int, ...]): Analysis-channel header.
        final (tuple[int, ...]): Final-channel header.
        transition (tuple[int, ...]): Complete analysis-to-final transition.
        message_end (int): Message-ending token.
        turn_ends (frozenset[int]): Assistant completion and tool-handoff tokens.
    """

    analysis: tuple[int, ...]
    final: tuple[int, ...]
    transition: tuple[int, ...]
    message_end: int
    turn_ends: frozenset[int]

    @classmethod
    def from_tokenizer(
        cls, tokenizer: PreTrainedTokenizerBase | TokenizerLike
    ) -> HarmonyTokenIds:
        """Resolve framing IDs without assuming a particular vocabulary.

        Args:
            tokenizer (PreTrainedTokenizerBase | TokenizerLike): Model tokenizer.

        Returns:
            HarmonyTokenIds: Encoded channel and message markers.
        """
        endings = [
            tokenizer.encode(marker, add_special_tokens=False)
            for marker in ("<|end|>", "<|return|>", "<|call|>")
        ]
        if any(len(ids) != 1 for ids in endings):
            raise ValueError("GPT-OSS message endings must each encode to one token.")
        return cls(
            analysis=tuple(
                tokenizer.encode(HARMONY_ANALYSIS_MARKER, add_special_tokens=False)
            ),
            final=tuple(
                tokenizer.encode(HARMONY_FINAL_MARKER, add_special_tokens=False)
            ),
            transition=tuple(
                tokenizer.encode(HARMONY_FINAL_TRANSITION, add_special_tokens=False)
            ),
            message_end=endings[0][0],
            turn_ends=frozenset(ids[0] for ids in endings[1:]),
        )


class HarmonyReasoningState:
    """Count analysis-body tokens across messages in the current completion.

    Args:
        markers (HarmonyTokenIds): Tokenized Harmony framing.
        prompt_token_ids (Sequence[int]): Prompt, possibly ending in open analysis.
    """

    def __init__(
        self, markers: HarmonyTokenIds, prompt_token_ids: Sequence[int]
    ) -> None:
        self.markers = markers
        self.reasoning_tokens = 0
        self.in_analysis = False
        self.saw_analysis = False
        self.final_started = False
        self.ended = False
        self._header: deque[int] = deque(
            maxlen=max(len(markers.analysis), len(markers.final))
        )
        # Only an unfinished prompt message can contribute to this request's budget.
        start = len(prompt_token_ids)
        while (
            start
            and prompt_token_ids[start - 1] not in markers.turn_ends
            and prompt_token_ids[start - 1] != markers.message_end
        ):
            start -= 1
        for token_id in prompt_token_ids[start:]:
            self.advance(token_id)

    def advance(self, token_id: int) -> None:
        """Consume one token, excluding Harmony framing from the body count.

        Args:
            token_id (int): Next prompt or generated token.
        """
        if self.final_started or self.ended:
            return
        if token_id in self.markers.turn_ends:
            self.ended = True
            self.in_analysis = False
        elif token_id == self.markers.message_end:
            self.in_analysis = False
            self._header.clear()
        elif self.in_analysis:
            self.reasoning_tokens += 1
        else:
            self._header.append(token_id)
            header = tuple(self._header)
            if header[-len(self.markers.analysis) :] == self.markers.analysis:
                self.in_analysis = self.saw_analysis = True
                self._header.clear()
            elif header[-len(self.markers.final) :] == self.markers.final:
                self.final_started = True


def harmony_budget_metadata(
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    markers: HarmonyTokenIds,
    budget: int,
) -> dict[str, int | bool]:
    """Summarize the same token count used by the Harmony controller.

    Args:
        prompt_token_ids (Sequence[int]): Exact prompt IDs.
        generated_token_ids (Sequence[int]): Exact generated IDs.
        markers (HarmonyTokenIds): Tokenized Harmony framing.
        budget (int): Maximum number of analysis-body tokens.

    Returns:
        dict[str, int | bool]: Counts and budget diagnostics. A forced stop
            requires a complete transition immediately at the budget boundary.
    """
    state = HarmonyReasoningState(markers, prompt_token_ids)
    boundary: int | None = None
    for index, token_id in enumerate(generated_token_ids):
        if boundary is None and state.in_analysis and state.reasoning_tokens == budget:
            boundary = index
        state.advance(token_id)
    return {
        "reasoning_tokens": state.reasoning_tokens,
        "hit_thinking_token_budget": state.saw_analysis
        and state.reasoning_tokens >= budget,
        "forced_stop": boundary is not None
        and tuple(generated_token_ids[boundary : boundary + len(markers.transition)])
        == markers.transition,
        "over_budget": state.reasoning_tokens > budget,
    }


class HarmonyBudgetRequest:
    """Incrementally constrain a single request at its reasoning budget.

    Args:
        markers (HarmonyTokenIds): Tokenized Harmony framing.
        budget (int): Maximum number of analysis-body tokens.
    """

    def __init__(self, markers: HarmonyTokenIds, budget: int) -> None:
        self.markers = markers
        self.budget = budget
        self.state: HarmonyReasoningState | None = None
        self._processed = 0
        self._force_index: int | None = None

    def __call__(
        self, prompt_ids: list[int], output_ids: list[int], logits: torch.Tensor
    ) -> torch.Tensor:
        """Force only the next transition token when the analysis cap is reached.

        Args:
            prompt_ids (list[int]): Exact prompt IDs supplied by vLLM.
            output_ids (list[int]): Live list of generated IDs.
            logits (torch.Tensor): This request's next-token logits.

        Returns:
            torch.Tensor: Logits, modified in place only during forced transition.
        """
        if self.state is None:
            self.state = HarmonyReasoningState(self.markers, prompt_ids)
            if self.state.in_analysis and self.state.reasoning_tokens >= self.budget:
                self._force_index = 0
        if len(output_ids) < self._processed:
            raise RuntimeError(
                "Harmony budget enforcement requires non-speculative decoding."
            )
        for token_id in output_ids[self._processed :]:
            if self._force_index is not None:
                if token_id != self.markers.transition[self._force_index]:
                    raise RuntimeError(
                        "vLLM did not preserve the forced Harmony transition."
                    )
                self._force_index += 1
                if self._force_index == len(self.markers.transition):
                    self._force_index = None
            self.state.advance(token_id)
            if (
                self._force_index is None
                and self.state.in_analysis
                and self.state.reasoning_tokens >= self.budget
            ):
                self._force_index = 0
        self._processed = len(output_ids)
        if self._force_index is not None:
            logits.fill_(-float("inf"))
            logits[self.markers.transition[self._force_index]] = 0.0
        return logits


class HarmonyThinkingBudgetProcessor(AdapterLogitsProcessor):
    """vLLM adapter managing an independent Harmony budget for each request."""

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ) -> None:
        """Initialize the controller using the engine's cached tokenizer.

        Args:
            vllm_config (VllmConfig): Engine configuration.
            device (torch.device): Sampling device.
            is_pin_memory (bool): vLLM pinned-memory setting.
        """
        super().__init__(vllm_config, device, is_pin_memory)
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "GPT-OSS budget enforcement does not support speculative decoding."
            )
        if vllm_config.model_config is None:
            raise ValueError("GPT-OSS budget enforcement requires a model tokenizer.")
        self.markers = HarmonyTokenIds.from_tokenizer(
            cached_tokenizer_from_config(model_config=vllm_config.model_config)
        )

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        """Reject malformed budgets and simultaneous generic enforcement.

        Args:
            sampling_params (SamplingParams): Request sampling configuration.
        """
        budget = (sampling_params.extra_args or {}).get(HARMONY_BUDGET_KEY)
        if budget is None:
            return
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError(
                "The GPT-OSS thinking budget must be a nonnegative integer."
            )
        if sampling_params.thinking_token_budget is not None:
            raise ValueError(
                "Use only the Harmony controller for GPT-OSS thinking budgets."
            )

    def is_argmax_invariant(self) -> bool:
        """Indicate that forcing a transition can change greedy decoding.

        Returns:
            bool: False, so vLLM also applies this controller to greedy requests.
        """
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        """Apply removals before creating requests that can reuse their slots.

        Args:
            batch_update (BatchUpdate | None): vLLM batch membership changes.
        """
        if batch_update is None:
            return
        # Some vLLM adapters apply additions before removals, deleting replacements.
        super().update_state(
            BatchUpdate(batch_update.batch_size, batch_update.removed, [], [])
        )
        super().update_state(
            BatchUpdate(
                batch_update.batch_size, [], batch_update.added, batch_update.moved
            )
        )

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> HarmonyBudgetRequest | None:
        """Create request state only when a Harmony budget is configured.

        Args:
            params (SamplingParams): Request sampling configuration.

        Returns:
            HarmonyBudgetRequest | None: Controller, or None for unbudgeted requests.
        """
        self.validate_params(params)
        budget = (params.extra_args or {}).get(HARMONY_BUDGET_KEY)
        if budget is None:
            return None
        if set(self.markers.transition).intersection(params.all_stop_token_ids):
            raise ValueError(
                "Stop tokens must not interrupt the forced Harmony transition."
            )
        return HarmonyBudgetRequest(self.markers, budget)
