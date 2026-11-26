from __future__ import annotations

from arc_cd.models.config import VLLMModelConfig


def resolve_layers(
    spec: str,
    model_config: VLLMModelConfig,
    *,
    cache_dir: str | None,
) -> list[int]:
    """Resolve a layer spec into concrete layer indices.

    Args:
        spec: Layer spec, such as ``"all"``, ``"0-63"``, or ``"20,28,36"``.
        model_config: Model configuration with model-loading fields.
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        Sorted layer indices.

    Raises:
        ValueError: If the model config does not expose a layer count.
    """
    if spec.lower() != "all":
        return parse_layers(spec)

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(
        model_config.model,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        cache_dir=cache_dir,
    )
    txt_config = getattr(hf_config, "text_config", hf_config)

    num_layers = getattr(txt_config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(txt_config, "n_layer", None)

    if num_layers is None:
        raise ValueError("Could not infer number of hidden layers from model config.")

    return list(range(int(num_layers)))


def parse_layers(spec: str) -> list[int]:
    """Parse a layer specification.

    Args:
        spec: Layer spec such as "0-63", "16-47", "0-63:2", or "20,28,36".

    Returns:
        Sorted unique layer indices.

    Raises:
        ValueError: If no valid layers are specified.
    """
    layers: set[int] = set()

    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue

        if "-" not in item:
            layers.add(int(item))
            continue

        range_part, _, step_part = item.partition(":")
        start_s, end_s = range_part.split("-", maxsplit=1)
        step = int(step_part) if step_part else 1

        layers.update(range(int(start_s), int(end_s) + 1, step))

    if not layers:
        raise ValueError("At least one layer must be specified.")

    return sorted(layers)
