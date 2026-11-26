from __future__ import annotations

import logging

import pytest

from arc_cd.models.config import (
    ChatTemplateConfig,
    GenerationConfig,
    VLLMReasoningConfig,
)
from arc_cd.models.vllm_setup import (
    build_reasoning_budget_metadata,
    build_token_spans,
    count_open_prompt_reasoning_tokens,
    ensure_reasoning_start_in_prompt,
    has_open_reasoning_span,
    prepare_prompts_for_reasoning_budget,
)


class TinyReasoningTokenizer:
    """Small tokenizer with single-token think markers for budget tests."""

    all_special_ids: list[int] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens

        token_ids: list[int] = []
        index = 0
        while index < len(text):
            if text.startswith("<think>", index):
                token_ids.append(1)
                index += len("<think>")
            elif text.startswith("</think>", index):
                token_ids.append(2)
                index += len("</think>")
            elif text.startswith("FORCED", index):
                token_ids.append(4)
                index += len("FORCED")
            else:
                token_ids.append(1000 + ord(text[index]))
                index += 1

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces

        pieces: list[str] = []
        for token_id in token_ids:
            if token_id == 1:
                pieces.append("<think>")
            elif token_id == 2:
                pieces.append("</think>")
            elif token_id == 4:
                pieces.append("FORCED")
            else:
                pieces.append(chr(token_id - 1000))

        return "".join(pieces)


def _metadata(
    *,
    prompt: str = "",
    generated: str,
    budget: int = 4,
    reasoning_end_str: str = "</think>\n\n",
) -> dict[str, object]:
    tokenizer = TinyReasoningTokenizer()

    return build_reasoning_budget_metadata(
        prompt_token_ids=tokenizer.encode(prompt, add_special_tokens=False),
        generated_token_ids=tokenizer.encode(generated, add_special_tokens=False),
        tokenizer=tokenizer,  # type: ignore[arg-type]
        generation_config=GenerationConfig(thinking_token_budget=budget),
        reasoning_config=VLLMReasoningConfig(
            reasoning_end_str=reasoning_end_str,
        ),
    )


def test_ensure_reasoning_start_in_prompt_appends_missing_open_span() -> None:
    config = VLLMReasoningConfig(reasoning_end_str="</think>\n\n")

    assert ensure_reasoning_start_in_prompt("assistant\n", config) == (
        "assistant\n<think>\n"
    )
    assert ensure_reasoning_start_in_prompt("assistant\n<think>\n", config) == (
        "assistant\n<think>\n"
    )


def test_has_open_reasoning_span_warns_on_reopened_span(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        is_open = has_open_reasoning_span(
            "old</think>\nnew<think>\n",
            reasoning_start_str="<think>",
            reasoning_end_marker="</think>",
        )

    assert is_open is True
    assert "closed reasoning span followed by a new open reasoning span" in caplog.text


def test_prepare_prompts_for_reasoning_budget_respects_disabled_thinking() -> None:
    prompts = prepare_prompts_for_reasoning_budget(
        ["assistant\n"],
        chat_config=ChatTemplateConfig(
            chat_template_kwargs={"enable_thinking": False},
        ),
        generation_config=GenerationConfig(thinking_token_budget=4),
        reasoning_config=VLLMReasoningConfig(reasoning_end_str="</think>\n\n"),
    )

    assert prompts == ["assistant\n"]


def test_conventional_end_marker_budget_boundary_is_inferred_forced() -> None:
    metadata = _metadata(generated="abcd</think>\n\nfinal")

    assert metadata["thinking_token_budget"] == 4
    assert "reasoning_effort" not in metadata
    assert metadata["reasoning_tokens"] == 4
    assert metadata["hit_thinking_token_budget"] is True
    assert metadata["forced_stop"] is True
    assert metadata["over_budget"] is False


def test_conventional_end_marker_before_budget_is_natural() -> None:
    metadata = _metadata(generated="ab</think>\n\nfinal")

    assert metadata["reasoning_tokens"] == 2
    assert metadata["hit_thinking_token_budget"] is False
    assert metadata["forced_stop"] is False
    assert metadata["over_budget"] is False


def test_prompt_open_reasoning_tokens_count_toward_budget() -> None:
    metadata = _metadata(prompt="<think>\n", generated="abc</think>\n\nfinal")

    assert metadata["reasoning_tokens"] == 4
    assert metadata["hit_thinking_token_budget"] is True
    assert metadata["forced_stop"] is True
    assert metadata["over_budget"] is False


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("assistant", None),
        ("<think>abc</think>", None),
        ("<think>", 0),
        ("<think>\n", 1),
        ("<think>abc</think><think>", 0),
    ],
)
def test_prompt_reasoning_count_distinguishes_empty_open_span(
    prompt: str, expected: int | None
) -> None:
    """Distinguish an empty open reasoning span from absent or closed reasoning.

    Args:
        prompt (str): Prompt to tokenize.
        expected (int | None): Open-span token count, or no open span.
    """
    tokenizer = TinyReasoningTokenizer()
    assert (
        count_open_prompt_reasoning_tokens(
            tokenizer.encode(prompt),
            tokenizer.encode("<think>"),
            tokenizer.encode("</think>"),
        )
        == expected
    )


def test_whitespace_prefixed_end_string_does_not_count_forced_prefix() -> None:
    metadata = _metadata(
        prompt="<think>\n",
        generated="abc\n</think>\n\nfinal",
        reasoning_end_str="\n</think>\n\n",
    )

    assert metadata["reasoning_tokens"] == 4
    assert metadata["hit_thinking_token_budget"] is True
    assert metadata["forced_stop"] is True
    assert metadata["over_budget"] is False


def test_duplicate_end_marker_uses_first_reasoning_close() -> None:
    metadata = _metadata(
        prompt="<think>\n",
        generated="abc</think>\n\nfinal says </think>\n\nagain",
    )

    assert metadata["reasoning_tokens"] == 4
    assert metadata["hit_thinking_token_budget"] is True
    assert metadata["forced_stop"] is True
    assert metadata["over_budget"] is False


def test_token_spans_use_first_reasoning_close() -> None:
    tokenizer = TinyReasoningTokenizer()
    prompt_token_ids = tokenizer.encode("<think>\n", add_special_tokens=False)
    generated_token_ids = tokenizer.encode(
        "abc</think>\n\nfinal says </think>\n\nagain",
        add_special_tokens=False,
    )

    spans = build_token_spans(
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        reasoning_start_str="<think>",
        reasoning_end_marker="</think>",
        thinking_enabled=True,
    )

    assert spans["reasoning_marker_generated"] == [3, 4]
    assert spans["reasoning_content_generated"] == [0, 3]


def test_missing_end_marker_past_budget_is_recorded_as_over_budget() -> None:
    metadata = _metadata(generated="abcde")

    assert metadata["reasoning_tokens"] == 5
    assert metadata["hit_thinking_token_budget"] is True
    assert metadata["forced_stop"] is False
    assert metadata["over_budget"] is True


def test_distinct_forced_end_string_is_exact_forced_stop() -> None:
    metadata = _metadata(
        generated="abcdFORCED</think>\n\nfinal",
        reasoning_end_str="FORCED</think>\n\n",
    )

    assert metadata["reasoning_tokens"] == 4
    assert metadata["hit_thinking_token_budget"] is True
    assert metadata["forced_stop"] is True
