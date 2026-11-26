"""Build support-only replay samples using the canonical inference prompt."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from arc_cd.data_prep.concept_arc import ConceptARCTask, ConceptARCTestCase
from arc_cd.models.config import ChatTemplateConfig, VLLMReasoningConfig
from arc_cd.probes.hidden import PromptReplaySample
from arc_cd.prompt_prep.load_prompt import build_prompt_record
from arc_cd.prompt_prep.prompt_format import PromptTemplates

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


SUPPORT_POSITIONS = ("S1", "S2", "S3")


def task_support_identity(task: ConceptARCTask) -> dict[str, str]:
    """Identify a parent task and its support-set version from the filename.

    Args:
        task (ConceptARCTask): Source task, including its concept and filename.

    Returns:
        dict[str, str]: Parent identity scoped by concept and support-set ID
        local to that parent. Numeric suffixes retain leading zeros.
    """
    match = re.fullmatch(r"(.+)_(\d+)", task.sample_id)
    parent, support_set = match.groups() if match else (task.sample_id, "canonical")
    return {
        "parent_task_id": f"{task.concept}/{parent}",
        "support_set_id": support_set,
    }


def task_inventory(tasks: Sequence[ConceptARCTask]) -> dict[str, str]:
    """Fingerprint source files to detect dataset changes across shards.

    Args:
        tasks (Sequence[ConceptARCTask]): Ordered dataset tasks.

    Returns:
        dict[str, str]: Task IDs mapped to SHA-256 hashes of source bytes.
    """
    return {
        task.task_id: hashlib.sha256(task.path.read_bytes()).hexdigest()
        for task in tasks
    }


def build_support_prompts(
    tasks: Sequence[ConceptARCTask],
    templates: PromptTemplates,
    tokenizer: PreTrainedTokenizerBase,
    chat_config: ChatTemplateConfig,
    reasoning_config: VLLMReasoningConfig | None = None,
) -> tuple[list[dict[str, Any]], list[PromptReplaySample]]:
    """Render once per task version and truncate at its third output bracket.

    Args:
        tasks (Sequence[ConceptARCTask]): One task per support-set version.
        templates (PromptTemplates): Canonical system and user templates.
        tokenizer (PreTrainedTokenizerBase): Inference tokenizer.
        chat_config (ChatTemplateConfig): Inference chat rendering options.
        reasoning_config (VLLMReasoningConfig | None): Chat reasoning options.

    Returns:
        tuple[list[dict[str, Any]], list[PromptReplaySample]]: Aligned manifest
        rows and replay prefixes, without query or generated token sequences.

    Raises:
        ValueError: If a task lacks three supports, a render query or valid
            bracket positions, or a boundary token includes later content.
    """
    from arc_cd.models.vllm_setup import build_prompt_token_spans, render_prompt

    records: list[dict[str, Any]] = []
    samples: list[PromptReplaySample] = []
    for task in tasks:
        if len(task.train) != 3 or any(pair.output is None for pair in task.train):
            raise ValueError(
                f"{task.task_id}: exactly three complete supports required."
            )
        if not task.test:
            raise ValueError(
                f"{task.task_id}: a test input is needed to render the prompt."
            )
        render_case = ConceptARCTestCase(
            concept=task.concept,
            sample_id=task.sample_id,
            test_index=0,
            path=task.path,
            train=task.train,
            test_input=task.test[0].input,
        )
        prompt_record = build_prompt_record(render_case, templates)
        prompt = render_prompt(
            prompt_record.messages,
            tokenizer,
            chat_config,
            reasoning_config=reasoning_config,
        )
        encoded = tokenizer(
            prompt, add_special_tokens=False, return_offsets_mapping=True
        )
        token_ids = list(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        token_spans = build_prompt_token_spans(
            prompt_record, prompt, token_ids, tokenizer
        )
        user_start = prompt.find(prompt_record.messages[-1]["content"])
        char_spans = {
            name: [user_start + start, user_start + end]
            for name, (start, end) in prompt_record.metadata[
                "prompt_char_spans"
            ].items()
            if name != "query_input_full"
        }
        positions: dict[str, int] = {}
        for number, position in enumerate(SUPPORT_POSITIONS, start=1):
            key = f"support_{number}_output_closing_bracket"
            if key not in char_spans:
                raise ValueError(
                    f"Missing closing bracket for {task.task_id}/{position}."
                )
            start, end = char_spans[key]
            span = token_spans[key]
            if prompt[start:end] != "]" or span[1] - span[0] != 1:
                raise ValueError(
                    f"Ambiguous closing bracket for {task.task_id}/{position}."
                )
            index = span[0]
            token_end = offsets[index][1]
            if prompt[end:token_end].strip():
                raise ValueError(
                    f"Boundary token includes later content: {task.task_id}/{position}."
                )
            positions[position] = index
        selected = list(positions.values())
        if selected != sorted(set(selected)):
            raise ValueError(f"Support positions are not ordered for {task.task_id}.")
        if token_spans["query_input_full"][0] <= positions["S3"]:
            raise ValueError(
                f"Test query precedes the end of the supports: {task.task_id}."
            )
        prefix_ids = token_ids[: positions["S3"] + 1]
        prefix_text = prompt[: offsets[positions["S3"]][1]]
        case_id = f"{task.task_id}/support"
        row = len(records)
        metadata = {
            **task_support_identity(task),
            "prompt_only": True,
            "render_case_id": render_case.case_id,
            "render_test_index": 0,
        }
        records.append(
            {
                "case_id": case_id,
                "task_id": task.task_id,
                "sample_id": task.sample_id,
                "concept": task.concept,
                "test_index": -1,
                "source_path": str(task.path),
                "support_examples": [
                    {"input": pair.input, "output": pair.output} for pair in task.train
                ],
                "prompt": prefix_text,
                "prompt_char_spans": char_spans,
                "token_spans": {name: token_spans[name] for name in char_spans},
                "selected_token_indices": positions,
                "selected_token_ids": {
                    name: token_ids[index] for name, index in positions.items()
                },
                "prefix_sha256": hashlib.sha256(
                    ",".join(map(str, prefix_ids)).encode("ascii")
                ).hexdigest(),
                "feature_rows": {name: row for name in SUPPORT_POSITIONS},
                "metadata": metadata,
                "prompt_token_ids": prefix_ids,
                "generated_token_ids": [],
            }
        )
        samples.append(PromptReplaySample(case_id, prefix_ids, positions))
    return records, samples
