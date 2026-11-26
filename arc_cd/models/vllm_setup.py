"""
Functions and modules to run vLLM generation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Optional, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig

from arc_cd.models.config import (
    ChatTemplateConfig,
    GenerationConfig,
    VLLMModelConfig,
    VLLMReasoningConfig,
)
from arc_cd.prompt_prep import Message, PromptRecord


def load_tokenizer(
    config: VLLMModelConfig,
    cache_dir: Optional[str] = None,
) -> PreTrainedTokenizerBase:
    """Load the tokenizer used to render chat prompts.

    Args:
        config: vLLM model configuration.
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        Hugging Face tokenizer.
    """
    tokenizer_name = config.tokenizer or config.model
    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=config.tokenizer_revision or config.revision,
        trust_remote_code=config.trust_remote_code,
        cache_dir=cache_dir,
    )


def load_vllm(
    config: VLLMModelConfig,
    reasoning_config: Optional[VLLMReasoningConfig] = None,
    cache_dir: Optional[str] = None,
) -> LLM:
    """Load a vLLM model.

    Args:
        config (VLLMModelConfig): vLLM model configuration.
        reasoning_config (Optional[VLLMReasoningConfig]): Reasoning parser
            configuration passed to vLLM.
        cache_dir (Optional[str]): Optional model download/cache directory.

    Returns:
        LLM: Loaded vLLM instance.
    """
    kwargs: dict[str, Any] = {
        "model": config.model,
        "tokenizer": config.tokenizer,
        "dtype": config.dtype,
        "trust_remote_code": config.trust_remote_code,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "tensor_parallel_size": config.tensor_parallel_size,
        "seed": config.seed,
    }

    if config.revision is not None:
        kwargs["revision"] = config.revision
    if config.tokenizer_revision is not None:
        kwargs["tokenizer_revision"] = config.tokenizer_revision
    if config.max_model_len is not None:
        kwargs["max_model_len"] = config.max_model_len

    if reasoning_config is not None and reasoning_config.enabled:
        if reasoning_config.reasoning_parser == "openai_gptoss":
            from arc_cd.models.harmony_budget import validate_harmony_reasoning_config

            validate_harmony_reasoning_config(reasoning_config)
            kwargs["logits_processors"] = [
                "arc_cd.models.harmony_budget:HarmonyThinkingBudgetProcessor"
            ]
        kwargs["reasoning_config"] = ReasoningConfig(
            reasoning_parser=reasoning_config.reasoning_parser,
            reasoning_start_str=reasoning_config.reasoning_start_str,
            reasoning_end_str=reasoning_config.reasoning_end_str,
        )

    return LLM(**kwargs, download_dir=cache_dir)


def make_sampling_params(
    config: GenerationConfig,
    reasoning_config: Optional[VLLMReasoningConfig] = None,
) -> SamplingParams:
    """Create vLLM sampling parameters.

    Args:
        config (GenerationConfig): Generation configuration.
        reasoning_config (Optional[VLLMReasoningConfig]): Selects Harmony-aware
            enforcement for enabled GPT-OSS reasoning.

    Returns:
        SamplingParams: vLLM request parameters.
    """
    kwargs: dict[str, Any] = {
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "min_p": config.min_p,
        "presence_penalty": config.presence_penalty,
        "repetition_penalty": config.repetition_penalty,
    }

    if config.thinking_token_budget is not None:
        if (
            reasoning_config is not None
            and reasoning_config.enabled
            and reasoning_config.reasoning_parser == "openai_gptoss"
        ):
            from arc_cd.models.harmony_budget import (
                HARMONY_BUDGET_KEY,
                validate_harmony_reasoning_config,
            )

            validate_harmony_reasoning_config(reasoning_config)
            kwargs["extra_args"] = {
                HARMONY_BUDGET_KEY: config.thinking_token_budget,
            }
        else:
            kwargs["thinking_token_budget"] = config.thinking_token_budget
    if config.stop is not None:
        kwargs["stop"] = config.stop
    if config.stop_token_ids is not None:
        kwargs["stop_token_ids"] = config.stop_token_ids
    if config.skip_special_tokens is not None:
        kwargs["skip_special_tokens"] = config.skip_special_tokens
    if config.seed is not None:
        kwargs["seed"] = config.seed

    return SamplingParams(**kwargs)


def encode_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    """Encode text without adding special tokens.

    Args:
        tokenizer (PreTrainedTokenizerBase): Tokenizer used for generation.
        text (str): Text to encode.

    Returns:
        list[int]: Token IDs for ``text``.
    """
    return list(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def render_prompt(
    messages: list[Message],
    tokenizer: PreTrainedTokenizerBase,
    config: ChatTemplateConfig,
    reasoning_config: Optional[VLLMReasoningConfig] = None,
) -> str:
    """Render chat messages into a model prompt.

    Args:
        messages (list[Message]): Chat messages, preserved without modification.
        tokenizer (PreTrainedTokenizerBase): Tokenizer with a chat template.
        config (ChatTemplateConfig): Chat rendering and system-message handling.
        reasoning_config (Optional[VLLMReasoningConfig]): Reasoning configuration.
            For Harmony templates, forwards ``reasoning_effort`` from this config.

    Returns:
        str: Rendered prompt string.

    Raises:
        ValueError: If moving system instructions requires a missing user message.
    """
    render_messages = messages
    if config.prepend_system_to_user:
        system_text = "\n\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        render_messages = [
            dict(message) for message in messages if message["role"] != "system"
        ]
        if system_text:
            first_user = next(
                (message for message in render_messages if message["role"] == "user"),
                None,
            )
            if first_user is None:
                raise ValueError(
                    "Cannot prepend system instructions without a user message."
                )
            first_user["content"] = f"{system_text}\n\n{first_user['content']}"

    chat_template_kwargs = dict(config.chat_template_kwargs)
    if config.renderer == "harmony" and reasoning_config is not None:
        if reasoning_config.reasoning_effort is not None:
            chat_template_kwargs["reasoning_effort"] = reasoning_config.reasoning_effort

    return cast(
        str,
        tokenizer.apply_chat_template(
            render_messages,
            tokenize=False,
            add_generation_prompt=config.add_generation_prompt,
            **chat_template_kwargs,
        ),
    )


def render_prompts(
    records: Sequence[PromptRecord],
    tokenizer: PreTrainedTokenizerBase,
    config: ChatTemplateConfig,
    reasoning_config: Optional[VLLMReasoningConfig] = None,
) -> list[str]:
    """Render prompt records into prompt strings.

    Args:
        records: ARC-style prompt records.
        tokenizer: Tokenizer with a chat template.
        config: Chat-template configuration.
        reasoning_config: Optional reasoning configuration.

    Returns:
        Rendered prompt strings, in the same order as ``records``.
    """
    return [
        render_prompt(
            messages=record.messages,
            tokenizer=tokenizer,
            config=config,
            reasoning_config=reasoning_config,
        )
        for record in records
    ]


def has_open_reasoning_span(
    text: str,
    *,
    reasoning_start_str: str,
    reasoning_end_marker: str,
) -> bool:
    """Return whether text ends inside an unclosed reasoning span.

    Args:
        text (str): Prompt or completion text to inspect.
        reasoning_start_str (str): Text marker that opens the reasoning span.
        reasoning_end_marker (str): Text marker that closes the reasoning span.

    Returns:
        bool: Whether the last reasoning start marker appears after the last
            reasoning end marker.
    """
    if not reasoning_start_str:
        return False

    last_start = text.rfind(reasoning_start_str)
    if last_start < 0:
        return False

    if not reasoning_end_marker:
        return True

    last_end = text.rfind(reasoning_end_marker)
    if last_end >= 0 and last_start > last_end:
        logging.warning(
            "Prompt contains a closed reasoning span followed by a new open "
            "reasoning span. Treating the final span as open."
        )

    return last_start > last_end


def ensure_reasoning_start_in_prompt(
    prompt: str,
    reasoning_config: VLLMReasoningConfig,
) -> str:
    """Ensure vLLM can see an open reasoning span before generation.

    vLLM's thinking-token budget state starts counting only after the prompt or
    generated output contains ``reasoning_start_str``. Qwen3.5 templates usually
    put ``<think>`` in the prompt; older Qwen3 templates often do not.

    Args:
        prompt (str): Rendered chat prompt.
        reasoning_config (VLLMReasoningConfig): Reasoning marker configuration.

    Returns:
        str: Prompt with an open reasoning start marker.
    """
    if has_open_reasoning_span(
        prompt,
        reasoning_start_str=reasoning_config.reasoning_start_str,
        reasoning_end_marker=reasoning_config.reasoning_end_marker,
    ):
        return prompt

    return f"{prompt}{reasoning_config.reasoning_start_str}\n"


def prepare_prompts_for_reasoning_budget(
    prompts: Sequence[str],
    chat_config: ChatTemplateConfig,
    generation_config: GenerationConfig,
    reasoning_config: Optional[VLLMReasoningConfig],
) -> list[str]:
    """Prepare prompts so vLLM can enforce the thinking budget.

    Args:
        prompts (Sequence[str]): Rendered chat prompts.
        chat_config (ChatTemplateConfig): Chat-template configuration used to
            render prompts.
        generation_config (GenerationConfig): Sampling/generation configuration.
        reasoning_config (Optional[VLLMReasoningConfig]): Reasoning marker
            configuration.

    Returns:
        list[str]: Prompts, with ``reasoning_start_str`` appended when needed.
    """
    if (
        generation_config.thinking_token_budget is None
        or reasoning_config is None
        or not reasoning_config.enabled
        or not reasoning_config.ensure_reasoning_start_in_prompt
        or chat_config.chat_template_kwargs.get("enable_thinking") is False
    ):
        return list(prompts)

    return [
        ensure_reasoning_start_in_prompt(prompt, reasoning_config) for prompt in prompts
    ]


def find_first_subsequence(
    sequence: Sequence[int],
    pattern: Sequence[int],
) -> Optional[int]:
    """Find the first start index of a subsequence.

    Args:
        sequence (Sequence[int]): Token sequence to search.
        pattern (Sequence[int]): Token subsequence to find.

    Returns:
        Optional[int]: First start index of ``pattern`` in ``sequence``, or
            ``None``.
    """
    if not pattern or len(pattern) > len(sequence):
        return None

    for index in range(len(sequence) - len(pattern) + 1):
        if list(sequence[index : index + len(pattern)]) == list(pattern):
            return index

    return None


def find_last_subsequence(
    sequence: Sequence[int],
    pattern: Sequence[int],
) -> Optional[int]:
    """Find the last start index of a subsequence.

    Args:
        sequence: Token sequence to search.
        pattern: Token subsequence to find.

    Returns:
        Last start index of ``pattern`` in ``sequence``, or ``None``.
    """
    if not pattern or len(pattern) > len(sequence):
        return None

    for index in range(len(sequence) - len(pattern), -1, -1):
        if list(sequence[index : index + len(pattern)]) == list(pattern):
            return index

    return None


def find_first_subsequence_at_or_after(
    sequence: Sequence[int],
    pattern: Sequence[int],
    start: int,
) -> Optional[int]:
    """Find the first subsequence start at or after ``start``.

    Args:
        sequence (Sequence[int]): Token sequence to search.
        pattern (Sequence[int]): Token subsequence to find.
        start (int): Earliest index where the match may begin.

    Returns:
        Optional[int]: First start index of ``pattern``, or ``None``.
    """
    if not pattern or len(pattern) > len(sequence):
        return None

    start_index = max(0, int(start))
    for index in range(start_index, len(sequence) - len(pattern) + 1):
        if list(sequence[index : index + len(pattern)]) == list(pattern):
            return index

    return None


def count_open_prompt_reasoning_tokens(
    prompt_token_ids: Sequence[int],
    reasoning_start_ids: Sequence[int],
    reasoning_end_ids: Sequence[int],
) -> int | None:
    """Count prompt tokens inside an open reasoning span.

    Args:
        prompt_token_ids (Sequence[int]): Prompt token IDs.
        reasoning_start_ids (Sequence[int]): Token IDs opening reasoning.
        reasoning_end_ids (Sequence[int]): Token IDs closing reasoning.

    Returns:
        int | None: Number of tokens after the last unclosed reasoning start,
        including zero for an empty open span, or None if no span is open.
    """
    last_start = find_last_subsequence(prompt_token_ids, reasoning_start_ids)
    if last_start is None:
        return None

    last_end = find_last_subsequence(prompt_token_ids, reasoning_end_ids)
    if last_end is not None and last_start <= last_end:
        return None

    return max(0, len(prompt_token_ids) - last_start - len(reasoning_start_ids))


def is_content_token(
    token_id: int,
    tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """Return whether a token should count as real generated content.

    Content tokens exclude tokenizer-declared special tokens and tokens that
    decode to whitespace only.

    Args:
        token_id: Token ID to inspect.
        tokenizer: Tokenizer used for generation.

    Returns:
        Whether the token is a non-special, non-whitespace token.
    """
    if int(token_id) in set(tokenizer.all_special_ids):
        return False

    text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return bool(cast(str, text).strip())


def trim_content_span(
    token_ids: Sequence[int],
    start: int,
    end: int,
    tokenizer: PreTrainedTokenizerBase,
) -> Optional[list[int]]:
    """Trim a span to its first and last content tokens.

    Args:
        token_ids: Token IDs for the generated completion.
        start: Inclusive generated-token start index.
        end: Exclusive generated-token end index.
        tokenizer: Tokenizer used for generation.

    Returns:
        Trimmed half-open span, or ``None`` if the span has no content tokens.
    """
    while start < end and not is_content_token(token_ids[start], tokenizer):
        start += 1

    while end > start and not is_content_token(token_ids[end - 1], tokenizer):
        end -= 1

    if start >= end:
        return None

    return [start, end]


def offset_span(
    span: Optional[list[int]],
    offset: int,
) -> Optional[list[int]]:
    """Convert a generated-relative span into a full-sequence span.

    Args:
        span: Generated-relative half-open span, or ``None``.
        offset: Number of prompt tokens.

    Returns:
        Full-sequence half-open span, or ``None``.
    """
    if span is None:
        return None

    return [offset + int(span[0]), offset + int(span[1])]


def char_span_to_token_span(
    offsets: Sequence[tuple[int, int]],
    char_span: Sequence[int],
) -> list[int]:
    """Convert a character span to a half-open token span.

    Args:
        offsets: Token character offsets in the rendered prompt.
        char_span: Half-open character span ``[start, end]``.

    Returns:
        Half-open token span.

    Raises:
        ValueError: If the character span overlaps no tokens.
    """
    char_start, char_end = int(char_span[0]), int(char_span[1])

    token_indices = [
        i
        for i, (token_start, token_end) in enumerate(offsets)
        if int(token_end) > char_start and int(token_start) < char_end
    ]

    if not token_indices:
        raise ValueError(f"Character span {list(char_span)} overlaps no tokens.")

    return [token_indices[0], token_indices[-1] + 1]


def build_prompt_token_spans(
    record: PromptRecord,
    prompt: str,
    prompt_token_ids: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, list[int]]:
    """Build support token spans inside the rendered prompt.

    Args:
        record: Prompt record containing user-prompt character spans.
        prompt: Rendered chat prompt.
        prompt_token_ids: Prompt token IDs returned by vLLM.
        tokenizer: Tokenizer used to render and tokenize prompts.

    Returns:
        Prompt-side token spans relative to the full prompt.

    Raises:
        ValueError: If prompt character spans or tokenizer offsets are invalid.
    """
    prompt_char_spans = record.metadata.get("prompt_char_spans")
    if not isinstance(prompt_char_spans, dict):
        raise ValueError(f"Missing prompt_char_spans for {record.case_id}.")

    user_content = record.messages[-1]["content"]
    user_start = prompt.find(user_content)
    if user_start < 0:
        raise ValueError(f"Could not locate user content for {record.case_id}.")

    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    token_ids = encoded["input_ids"]

    if len(token_ids) != len(prompt_token_ids):
        raise ValueError(
            f"Prompt-token mismatch for {record.case_id}: "
            f"{len(token_ids)} tokenizer tokens vs {len(prompt_token_ids)} vLLM tokens."
        )

    token_spans: dict[str, list[int]] = {}

    for name, char_span in prompt_char_spans.items():
        if not isinstance(char_span, list | tuple) or len(char_span) != 2:
            raise ValueError(
                f"Malformed prompt character span {name!r}: {char_span!r}."
            )

        absolute_char_span = [
            user_start + int(char_span[0]),
            user_start + int(char_span[1]),
        ]
        token_spans[name] = char_span_to_token_span(offsets, absolute_char_span)

    return token_spans


def build_token_spans(
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
    *,
    reasoning_start_str: str = "<think>",
    reasoning_end_marker: str = "</think>",
    thinking_enabled: bool = True,
) -> dict[str, Optional[list[int]]]:
    """Build raw and content-token spans for a generated completion.

    Spans are half-open. ``*_generated`` spans are relative to generated tokens.
    ``*_full`` spans are relative to ``prompt_token_ids + generated_token_ids``.

    The reasoning content span excludes the reasoning end marker. Content spans
    also trim tokenizer-declared special tokens and whitespace-only tokens from
    their boundaries.

    Args:
        prompt_token_ids (Sequence[int]): Prompt token IDs.
        generated_token_ids (Sequence[int]): Generated token IDs.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used for generation.
        reasoning_start_str (str): Text marker starting the reasoning span.
        reasoning_end_marker (str): Text marker ending the reasoning span.
        thinking_enabled (bool): Whether thinking mode is enabled by the model.

    Returns:
        dict[str, Optional[list[int]]]: Token span metadata.
    """
    prompt_len = len(prompt_token_ids)
    generated_len = len(generated_token_ids)

    start_token_ids = tokenizer.encode(
        reasoning_start_str,
        add_special_tokens=False,
    )
    marker_token_ids = tokenizer.encode(
        reasoning_end_marker,
        add_special_tokens=False,
    )
    prompt_reasoning_tokens = count_open_prompt_reasoning_tokens(
        prompt_token_ids,
        start_token_ids,
        marker_token_ids,
    )

    reasoning_start = 0
    if prompt_reasoning_tokens is None:
        start_index = find_first_subsequence(generated_token_ids, start_token_ids)
        if start_index is not None:
            reasoning_start = start_index + len(start_token_ids)

    marker_start = find_first_subsequence_at_or_after(
        generated_token_ids,
        marker_token_ids,
        reasoning_start,
    )

    generated_content = trim_content_span(
        generated_token_ids,
        0,
        generated_len,
        tokenizer,
    )

    reasoning_marker = None
    reasoning_content = None
    final_content = generated_content

    if marker_start is not None:
        marker_end = marker_start + len(marker_token_ids)
        reasoning_marker = [marker_start, marker_end]

        reasoning_content = trim_content_span(
            generated_token_ids,
            reasoning_start,
            marker_start,
            tokenizer,
        )
        final_content = trim_content_span(
            generated_token_ids,
            marker_end,
            generated_len,
            tokenizer,
        )
    elif thinking_enabled:
        reasoning_content = trim_content_span(
            generated_token_ids,
            reasoning_start,
            generated_len,
            tokenizer,
        )
        final_content = None

    return {
        "prompt": [0, prompt_len],
        "generated": [prompt_len, prompt_len + generated_len],
        "generated_content_generated": generated_content,
        "generated_content_full": offset_span(generated_content, prompt_len),
        "reasoning_marker_generated": reasoning_marker,
        "reasoning_marker_full": offset_span(reasoning_marker, prompt_len),
        "reasoning_content_generated": reasoning_content,
        "reasoning_content_full": offset_span(reasoning_content, prompt_len),
        "final_content_generated": final_content,
        "final_content_full": offset_span(final_content, prompt_len),
    }


def _first_stop_marker_index(
    token_ids: Sequence[int],
    stop_patterns: Sequence[Sequence[int]],
    start: int,
) -> Optional[int]:
    """Find the first stop marker at or after ``start``.

    Args:
        token_ids (Sequence[int]): Token sequence to search.
        stop_patterns (Sequence[Sequence[int]]): Stop-token patterns.
        start (int): Earliest generated-token index where a match may begin.

    Returns:
        Optional[int]: First stop-marker start index, or ``None``.
    """
    matches = [
        index
        for pattern in stop_patterns
        if (index := find_first_subsequence_at_or_after(token_ids, pattern, start))
        is not None
    ]

    return min(matches) if matches else None


def build_harmony_token_spans(
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
    *,
    reasoning_start_str: str,
    reasoning_end_marker: str,
    thinking_enabled: bool = True,
) -> dict[str, Optional[list[int]]]:
    """Build token spans for GPT-OSS Harmony completions.

    Harmony completions carry reasoning in the ``analysis`` channel and answers
    in the ``final`` channel. This parser excludes Harmony channel framing from
    both content spans.

    Args:
        prompt_token_ids (Sequence[int]): Prompt token IDs.
        generated_token_ids (Sequence[int]): Generated token IDs.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used for generation.
        reasoning_start_str (str): Harmony analysis-channel marker.
        reasoning_end_marker (str): Harmony final-channel marker.
        thinking_enabled (bool): Whether missing final content should be treated
            as open reasoning.

    Returns:
        dict[str, Optional[list[int]]]: Token span metadata.
    """
    prompt_len = len(prompt_token_ids)
    generated_len = len(generated_token_ids)

    analysis_marker_ids = tokenizer.encode(
        reasoning_start_str,
        add_special_tokens=False,
    )
    final_marker_ids = tokenizer.encode(
        reasoning_end_marker,
        add_special_tokens=False,
    )
    message_end_ids = tokenizer.encode("<|end|>", add_special_tokens=False)
    stop_patterns = [
        tokenizer.encode(stop_marker, add_special_tokens=False)
        for stop_marker in ("<|return|>", "<|call|>", "<|end|>")
    ]

    raw_generated_content = trim_content_span(
        generated_token_ids,
        0,
        generated_len,
        tokenizer,
    )

    analysis_marker_start = find_first_subsequence(
        generated_token_ids,
        analysis_marker_ids,
    )
    final_marker_start = find_first_subsequence(
        generated_token_ids,
        final_marker_ids,
    )

    reasoning_content = None
    if analysis_marker_start is not None:
        reasoning_start = analysis_marker_start + len(analysis_marker_ids)
        analysis_end = find_first_subsequence_at_or_after(
            generated_token_ids,
            message_end_ids,
            reasoning_start,
        )
        if analysis_end is None or (
            final_marker_start is not None and final_marker_start < analysis_end
        ):
            reasoning_end = (
                final_marker_start if final_marker_start is not None else generated_len
            )
        else:
            reasoning_end = analysis_end

        reasoning_content = trim_content_span(
            generated_token_ids,
            reasoning_start,
            reasoning_end,
            tokenizer,
        )
    elif final_marker_start is None and thinking_enabled:
        reasoning_content = trim_content_span(
            generated_token_ids,
            0,
            generated_len,
            tokenizer,
        )

    reasoning_marker = None
    final_content = None
    if final_marker_start is not None:
        final_start = final_marker_start + len(final_marker_ids)
        final_end = _first_stop_marker_index(
            generated_token_ids,
            stop_patterns,
            final_start,
        )
        if final_end is None:
            final_end = generated_len

        reasoning_marker = [final_marker_start, final_start]
        final_content = trim_content_span(
            generated_token_ids,
            final_start,
            final_end,
            tokenizer,
        )
    elif not thinking_enabled:
        final_content = raw_generated_content

    content_spans = [
        span for span in (reasoning_content, final_content) if span is not None
    ]
    generated_content = (
        [
            min(span[0] for span in content_spans),
            max(span[1] for span in content_spans),
        ]
        if content_spans
        else raw_generated_content
    )

    return {
        "prompt": [0, prompt_len],
        "generated": [prompt_len, prompt_len + generated_len],
        "generated_content_generated": generated_content,
        "generated_content_full": offset_span(generated_content, prompt_len),
        "reasoning_marker_generated": reasoning_marker,
        "reasoning_marker_full": offset_span(reasoning_marker, prompt_len),
        "reasoning_content_generated": reasoning_content,
        "reasoning_content_full": offset_span(reasoning_content, prompt_len),
        "final_content_generated": final_content,
        "final_content_full": offset_span(final_content, prompt_len),
    }


def build_reasoning_budget_metadata(
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
    generation_config: GenerationConfig,
    reasoning_config: Optional[VLLMReasoningConfig],
) -> dict[str, Any]:
    """Build compact reasoning-budget metadata for one completion.

    Args:
        prompt_token_ids (Sequence[int]): Prompt token IDs.
        generated_token_ids (Sequence[int]): Generated completion token IDs.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used for generation.
        generation_config (GenerationConfig): Sampling/generation configuration.
        reasoning_config (Optional[VLLMReasoningConfig]): Reasoning marker
            configuration.

    Returns:
        dict[str, Any]: JSON-serializable reasoning-budget metadata.
    """
    budget = generation_config.thinking_token_budget

    metadata: dict[str, Any] = {
        "thinking_token_budget": budget,
        "reasoning_tokens": None,
        "hit_thinking_token_budget": False,
        "forced_stop": False,
        "over_budget": None,
    }
    if reasoning_config is not None and reasoning_config.reasoning_effort is not None:
        metadata["reasoning_effort"] = reasoning_config.reasoning_effort

    if budget is None or reasoning_config is None or not reasoning_config.enabled:
        return metadata

    if reasoning_config.reasoning_parser == "openai_gptoss":
        from arc_cd.models.harmony_budget import (
            HarmonyTokenIds,
            harmony_budget_metadata,
        )

        metadata.update(
            harmony_budget_metadata(
                prompt_token_ids,
                generated_token_ids,
                HarmonyTokenIds.from_tokenizer(tokenizer),
                budget,
            )
        )
        return metadata

    start_ids = encode_text(tokenizer, reasoning_config.reasoning_start_str)
    end_marker_ids = encode_text(tokenizer, reasoning_config.reasoning_end_marker)
    forced_end_ids = encode_text(tokenizer, reasoning_config.reasoning_end_str)

    prompt_reasoning_tokens = count_open_prompt_reasoning_tokens(
        prompt_token_ids,
        start_ids,
        end_marker_ids,
    )

    reasoning_start = 0
    if prompt_reasoning_tokens is None:
        start_index = find_first_subsequence(generated_token_ids, start_ids)
        if start_index is not None:
            reasoning_start = start_index + len(start_ids)

    forced_end_index = find_first_subsequence_at_or_after(
        generated_token_ids,
        forced_end_ids,
        reasoning_start,
    )
    marker_index = find_first_subsequence_at_or_after(
        generated_token_ids,
        end_marker_ids,
        reasoning_start,
    )
    has_distinct_forced_end = (
        reasoning_config.reasoning_end_str.replace(
            reasoning_config.reasoning_end_marker,
            "",
            1,
        ).strip()
        != ""
    )

    forced_close_index = (
        forced_end_index
        if forced_end_index is not None
        and (marker_index is None or forced_end_index <= marker_index)
        else None
    )

    if forced_close_index is not None:
        reasoning_end = forced_close_index
        forced_stop = has_distinct_forced_end
    else:
        reasoning_end = (
            marker_index if marker_index is not None else len(generated_token_ids)
        )
        forced_stop = False

    generated_reasoning_tokens = max(0, reasoning_end - reasoning_start)
    reasoning_tokens = (prompt_reasoning_tokens or 0) + generated_reasoning_tokens
    hit_budget = reasoning_tokens >= budget
    if not forced_stop:
        forced_stop = (
            marker_index is not None or forced_close_index is not None
        ) and reasoning_tokens == budget

    metadata["reasoning_tokens"] = reasoning_tokens
    metadata["hit_thinking_token_budget"] = hit_budget
    metadata["forced_stop"] = forced_stop
    metadata["over_budget"] = reasoning_tokens > budget

    return metadata


def _build_generation_result(
    record: PromptRecord,
    prompt: str,
    output: Any,
    tokenizer: PreTrainedTokenizerBase,
    chat_config: ChatTemplateConfig,
    generation_config: GenerationConfig,
    reasoning_config: Optional[VLLMReasoningConfig] = None,
) -> dict[str, Any]:
    """Convert a vLLM request output into a JSON-serializable record.

    Args:
        record (PromptRecord): ARC-style prompt record.
        prompt (str): Rendered prompt string sent to vLLM.
        output (Any): vLLM request output.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used for generation.
        chat_config (ChatTemplateConfig): Prompt rendering configuration.
        generation_config (GenerationConfig): Sampling/generation configuration.
        reasoning_config (Optional[VLLMReasoningConfig]): Reasoning marker
            configuration.

    Returns:
        dict[str, Any]: JSON-serializable generation record.
    """
    if len(output.outputs) != 1:
        raise ValueError(
            f"Expected exactly one completion for {record.case_id}; "
            f"got {len(output.outputs)}."
        )
    completion = output.outputs[0]

    if output.prompt_token_ids is None:
        raise ValueError(f"Missing prompt token IDs for {record.case_id}.")

    prompt_token_ids = list(output.prompt_token_ids)
    generated_token_ids = list(completion.token_ids)
    generated_text = completion.text
    if chat_config.renderer == "harmony":
        generated_text = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    reasoning_end_marker = (
        reasoning_config.reasoning_end_marker
        if reasoning_config is not None
        else "</think>"
    )

    thinking_enabled = (
        reasoning_config.enabled if reasoning_config is not None else True
    )

    span_builder = (
        build_harmony_token_spans
        if chat_config.renderer == "harmony"
        else build_token_spans
    )
    token_spans = span_builder(
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        tokenizer=tokenizer,
        reasoning_start_str=(
            reasoning_config.reasoning_start_str
            if reasoning_config is not None
            else "<think>"
        ),
        reasoning_end_marker=reasoning_end_marker,
        thinking_enabled=thinking_enabled,
    )

    token_spans.update(
        build_prompt_token_spans(
            record=record,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            tokenizer=tokenizer,
        )
    )

    reasoning_budget = build_reasoning_budget_metadata(
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        tokenizer=tokenizer,
        generation_config=generation_config,
        reasoning_config=reasoning_config,
    )

    metadata = dict(record.metadata)
    if reasoning_config is not None and reasoning_config.reasoning_effort is not None:
        metadata["reasoning_effort"] = reasoning_config.reasoning_effort

    return {
        "case_id": record.case_id,
        "task_id": record.task_id,
        "concept": record.concept,
        "sample_id": record.sample_id,
        "test_index": record.test_index,
        "source_path": record.source_path,
        "metadata": metadata,
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "finish_reason": completion.finish_reason,
        "stop_reason": getattr(completion, "stop_reason", None),
        "reasoning_budget": reasoning_budget,
        "target_text": record.target_text,
        "target_grid": record.target_grid,
        "messages": record.messages,
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "token_spans": token_spans,
    }


def generate_from_records(
    llm: LLM,
    records: Sequence[PromptRecord],
    tokenizer: PreTrainedTokenizerBase,
    chat_config: ChatTemplateConfig,
    generation_config: GenerationConfig,
    reasoning_config: Optional[VLLMReasoningConfig] = None,
    *,
    max_model_len: int | None = None,
    use_tqdm: bool = True,
) -> list[dict[str, Any]]:
    """Generate model responses for ARC-style prompt records.

    Args:
        llm (LLM): Loaded vLLM model.
        records (Sequence[PromptRecord]): ARC-style prompt records.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used to render chat
            prompts.
        chat_config (ChatTemplateConfig): Chat-template configuration.
        generation_config (GenerationConfig): Sampling configuration.
        reasoning_config (Optional[VLLMReasoningConfig]): Reasoning marker
            configuration.
        max_model_len (int | None): Configured context limit. When supplied,
            require every final prompt plus the full output budget to fit.
        use_tqdm (bool): Whether vLLM should display a progress bar.

    Returns:
        list[dict[str, Any]]: Generation result dictionaries aligned with
            ``records``.

    Raises:
        ValueError: If a prompt plus the requested output budget exceeds the
            supplied context limit.
    """
    rendered_prompts = render_prompts(
        records,
        tokenizer,
        chat_config,
        reasoning_config=reasoning_config,
    )
    final_prompts = prepare_prompts_for_reasoning_budget(
        rendered_prompts,
        chat_config=chat_config,
        generation_config=generation_config,
        reasoning_config=reasoning_config,
    )

    if max_model_len is not None:
        for record, prompt in zip(records, final_prompts, strict=True):
            prompt_tokens = len(encode_text(tokenizer, prompt))
            required_tokens = prompt_tokens + generation_config.max_tokens
            if required_tokens > max_model_len:
                raise ValueError(
                    f"Context budget exceeded for {record.case_id}: "
                    f"prompt_tokens={prompt_tokens} + "
                    f"max_tokens={generation_config.max_tokens} = "
                    f"{required_tokens} > max_model_len={max_model_len}. "
                    "Reduce the configured output budget or prompt length."
                )

    sampling_params = make_sampling_params(generation_config, reasoning_config)

    outputs = llm.generate(
        final_prompts,
        sampling_params=sampling_params,
        use_tqdm=use_tqdm,
    )

    return [
        _build_generation_result(
            record=record,
            prompt=prompt,
            output=output,
            tokenizer=tokenizer,
            chat_config=chat_config,
            generation_config=generation_config,
            reasoning_config=reasoning_config,
        )
        for record, prompt, output in zip(records, final_prompts, outputs, strict=True)
    ]
