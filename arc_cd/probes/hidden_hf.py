"""
Modules and functions to load and process hidden states via HF transformers.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel

from arc_cd.probes.hidden import PromptReplaySample
from arc_cd.utils import input_device


def _validate_layers(model: PreTrainedModel, layers: list[int]) -> None:
    """Validate requested transformer layer indices.

    Args:
        model (PreTrainedModel): Hugging Face causal language model.
        layers (list[int]): Transformer block layer indices.

    Raises:
        ValueError: If any layer index is invalid.
    """
    config = getattr(model.config, "text_config", model.config)
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Model config does not expose num_hidden_layers.")

    invalid = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(
            f"Invalid layer indices {invalid}; model has {num_layers} layers."
        )


@torch.inference_mode()
def iter_prompt_hidden_states_hf(
    model: PreTrainedModel,
    samples: Sequence[PromptReplaySample],
    *,
    layers: list[int],
    chunk_size: int = 4096,
) -> Iterator[dict[str, np.ndarray]]:
    """Extract multiple boundary states in one cached traversal per sample.

    Args:
        model (PreTrainedModel): Hugging Face causal language model.
        samples (Sequence[PromptReplaySample]): Ordered support prefixes.
        layers (list[int]): Transformer block indices, excluding embeddings.
        chunk_size (int): Maximum number of non-boundary tokens per call.

    Yields:
        dict[str, np.ndarray]: FP16 arrays of shape ``[layers, hidden_size]``
        keyed by position, aligned with the input samples.

    Raises:
        ValueError: If layers, positions or chunk size are invalid.
        RuntimeError: If the model fails to return a cache or hidden states.
    """
    if chunk_size <= 0 or not layers:
        raise ValueError("chunk_size and the number of layers must be positive.")
    _validate_layers(model, layers)
    model.eval()
    device = input_device(model)
    # Avoid allocating vocabulary logits for every prefix token when supported.
    forward_options = (
        {"logits_to_keep": 1}
        if "logits_to_keep" in inspect.signature(model.forward).parameters
        else {}
    )
    for sample in tqdm(samples, desc="Extracting support hidden states"):
        indices = list(sample.positions.values())
        if (
            not indices
            or indices != sorted(set(indices))
            or indices[0] < 0
            or indices[-1] != len(sample.token_ids) - 1
        ):
            raise ValueError(f"Invalid support positions for {sample.case_id}.")
        past_key_values = None
        cursor = 0
        states: dict[str, np.ndarray] = {}
        for position, index in sample.positions.items():
            while cursor <= index:
                capture = cursor == index
                end = cursor + 1 if capture else min(cursor + chunk_size, index)
                outputs = model(
                    input_ids=torch.tensor(
                        [sample.token_ids[cursor:end]], dtype=torch.long, device=device
                    ),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=capture,
                    return_dict=True,
                    **forward_options,
                )
                past_key_values = outputs.past_key_values
                if past_key_values is None:
                    raise RuntimeError("Model did not return a replay cache.")
                if capture:
                    if outputs.hidden_states is None:
                        raise RuntimeError("Model did not return hidden states.")
                    vectors = [
                        outputs.hidden_states[layer + 1][0, -1] for layer in layers
                    ]
                    selected = torch.stack(
                        [vector.to(vectors[0].device) for vector in vectors]
                    )
                    states[position] = selected.to(
                        device="cpu", dtype=torch.float16
                    ).numpy()
                    del vectors, selected
                    if not np.isfinite(states[position]).all():
                        raise ValueError(
                            f"Non-finite features at {sample.case_id}/{position}."
                        )
                cursor = end
                del outputs
        del past_key_values
        yield states
