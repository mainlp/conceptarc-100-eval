"""
Modules and functions to load models via HF transformers backend.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from arc_cd.models.config import VLLMModelConfig


def resolve_torch_dtype(dtype: str) -> str | torch.dtype:
    """Resolve config dtype for Hugging Face model loading.

    Args:
        dtype: Dtype string from the model config.

    Returns:
        Hugging Face-compatible dtype value.
    """
    if dtype == "auto":
        return "auto"
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype in {"float32", "fp32"}:
        return torch.float32

    raise ValueError(f"Unsupported dtype for Hugging Face backend: {dtype!r}.")


def load_hf_model(
    config: VLLMModelConfig,
    *,
    cache_dir: Optional[str] = None,
    device_map: str | dict[str, Any] = "auto",
) -> PreTrainedModel:
    """Load a Hugging Face model using the shared model config.

    Args:
        config: Model-loading configuration used by the generation run.
        cache_dir: Optional Hugging Face cache directory.
        device_map: Device map for large-model loading.

    Returns:
        Loaded Hugging Face causal language model.
    """
    return AutoModelForCausalLM.from_pretrained(
        config.model,
        revision=config.revision,
        torch_dtype=resolve_torch_dtype(config.dtype),
        device_map=device_map,
        trust_remote_code=config.trust_remote_code,
        cache_dir=cache_dir,
    )
