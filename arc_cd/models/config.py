"""
Dataclasses for model and tokenizer configs.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, TypeVar

import yaml

ConfigT = TypeVar("ConfigT")


def normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
    """Normalize a GPT-OSS reasoning-effort label.

    Args:
        value (Optional[str]): Reasoning effort from config. Accepted values are
            ``"low"``, ``"mid"``, ``"medium"``, ``"high"``, or ``None``.

    Returns:
        Optional[str]: Normalized effort, with ``"mid"`` mapped to
        ``"medium"``.

    Raises:
        ValueError: If ``value`` is not a supported effort label.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            "reasoning_effort must be one of 'low', 'mid', 'medium', 'high', "
            f"or null; got {value!r}."
        )

    effort = value.lower()
    if effort == "mid":
        return "medium"
    if effort in {"low", "medium", "high"}:
        return effort

    raise ValueError(
        "reasoning_effort must be one of 'low', 'mid', 'medium', 'high', "
        f"or null; got {value!r}."
    )


@dataclass(frozen=True)
class VLLMModelConfig:
    """Configuration for loading a vLLM model.

    Attributes:
        model: Hugging Face model name or local model path.
        tokenizer: Optional tokenizer name or path. If omitted, uses ``model``.
        tensor_parallel_size: Number of GPUs used for tensor parallelism.
        dtype: vLLM dtype setting, such as ``"auto"``, ``"bfloat16"``, or
            ``"float16"``.
        trust_remote_code: Whether to trust custom model/tokenizer code.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        max_model_len: Optional maximum context length override.
        seed: Random seed used by vLLM.
    """

    model: str = "Qwen/Qwen3-8B"
    tokenizer: Optional[str] = None
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    trust_remote_code: bool = True
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None
    seed: int = 0


@dataclass(frozen=True)
class ChatTemplateConfig:
    """Configuration for rendering chat messages.

    Attributes:
        renderer: Prompt renderer to use. ``"chat_template"`` uses the
            Hugging Face tokenizer chat template; ``"harmony"`` uses that same
            chat-template path with GPT-OSS Harmony effort and span handling.
        add_generation_prompt: Whether to append the model's assistant prefix.
        chat_template_kwargs: Extra kwargs passed to the tokenizer chat
            template. For Qwen3 thinking mode, use
            ``{"enable_thinking": True}``.
        prepend_system_to_user: Whether to move system instructions into the
            first user message for templates that ignore supplied system messages.
    """

    renderer: str = "chat_template"
    add_generation_prompt: bool = True
    chat_template_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"enable_thinking": True}
    )
    prepend_system_to_user: bool = False

    def __post_init__(self) -> None:
        """Validate the prompt renderer."""
        if self.renderer not in {"chat_template", "harmony"}:
            raise ValueError(
                "chat.renderer must be either 'chat_template' or 'harmony'; "
                f"got {self.renderer!r}."
            )


@dataclass(frozen=True)
class VLLMReasoningConfig:
    """Configuration for vLLM reasoning-budget control and span parsing.

    Attributes:
        enabled: Whether to pass a vLLM reasoning config.
        reasoning_parser: vLLM reasoning parser name, e.g. "qwen3",
            "deepseek_r1", "granite", etc.
        reasoning_start_str: String marking the start of reasoning.
        reasoning_end_str: String vLLM should force when the thinking budget is hit.
            This may include a transition phrase plus the actual end marker.
        reasoning_end_marker: Minimal marker used by local span parsing. This should
            usually be just "</think>", not the full forced transition phrase.
        ensure_reasoning_start_in_prompt: Whether to append ``reasoning_start_str``
            when thinking is enabled but the rendered chat template does not leave
            the prompt inside a reasoning span. This is needed for vLLM's thinking
            budget to engage on templates where the model would otherwise generate
            ``<think>`` itself.
        reasoning_effort: Optional GPT-OSS reasoning effort. Accepted values are
            ``"low"``, ``"mid"``, ``"medium"``, ``"high"``, or ``None``.
    """

    enabled: bool = True
    reasoning_parser: str = "qwen3"
    reasoning_start_str: str = "<think>"
    reasoning_end_str: str = (
        "\n\nConsidering the limited time by the user, "
        "I have to give the solution based on the thinking directly now.\n"
        "</think>\n\n"
    )
    reasoning_end_marker: str = "</think>"
    ensure_reasoning_start_in_prompt: bool = True
    reasoning_effort: Optional[str] = None

    def __post_init__(self) -> None:
        """Normalize the optional reasoning effort label."""
        object.__setattr__(
            self,
            "reasoning_effort",
            normalize_reasoning_effort(self.reasoning_effort),
        )


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling configuration for vLLM generation.

    Attributes:
        max_tokens: Maximum number of total generated tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling parameter.
        min_p: Minimum probability sampling parameter.
        presence_penalty: Presence penalty.
        repetition_penalty: Repetition penalty.
        thinking_token_budget: Maximum number of generated reasoning tokens.
        stop: Optional stop strings.
        stop_token_ids: Optional token IDs that stop generation.
        skip_special_tokens: Optional vLLM detokenization setting. Leave unset
            to use vLLM's default.
        seed: Optional per-request sampling seed.
    """

    max_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    thinking_token_budget: Optional[int] = None
    stop: Optional[list[str]] = None
    stop_token_ids: Optional[list[int]] = None
    skip_special_tokens: Optional[bool] = None
    seed: Optional[int] = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Full model inference configuration.

    Attributes:
        model: vLLM model-loading configuration.
        chat: Chat-template rendering configuration.
        generation: Sampling/generation configuration.
        reasoning: Reasoning configuration.
    """

    model: VLLMModelConfig = field(default_factory=VLLMModelConfig)
    chat: ChatTemplateConfig = field(default_factory=ChatTemplateConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reasoning: VLLMReasoningConfig = field(default_factory=VLLMReasoningConfig)


def _dataclass_from_dict(
    cls: type[ConfigT],
    values: dict[str, Any],
    section: str,
) -> ConfigT:
    """Instantiate a dataclass from a partial dictionary.

    Args:
        cls: Dataclass type to instantiate.
        values: Partial field values loaded from YAML.
        section: YAML section name, used for error messages.

    Returns:
        Dataclass instance.

    Raises:
        ValueError: If ``values`` is not a dictionary or contains unknown fields.
    """
    if values is None:
        values = {}

    if not isinstance(values, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")

    field_names = {field.name for field in fields(cls)}  # type: ignore
    unknown_fields = set(values) - field_names

    if unknown_fields:
        raise ValueError(
            f"Unknown field(s) in config section '{section}': "
            f"{sorted(unknown_fields)}"
        )

    return cls(**values)


def load_experiment_config(path: str | Path, seed: int = 0) -> ExperimentConfig:
    """Load an experiment configuration from YAML.

    Missing fields fall back to the defaults defined by the corresponding
    dataclass.

    Args:
        path (str | Path): Path to a YAML config file.
        seed (int): Runtime seed used for vLLM model loading and sampling.

    Returns:
        ExperimentConfig: Experiment configuration.

    Raises:
        ValueError: If the YAML file is not a mapping or contains unknown fields.
    """
    config_path = Path(path).expanduser().resolve()

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML object at top level: {config_path}")

    allowed_sections = {"model", "chat", "generation", "reasoning"}
    unknown_sections = set(raw) - allowed_sections
    if unknown_sections:
        raise ValueError(
            f"Unknown config section(s) in {config_path}: "
            f"{sorted(unknown_sections)}"
        )

    model_values = raw.get("model", {})
    if model_values is None:
        model_values = {}
    if isinstance(model_values, dict):
        model_values = {**model_values, "seed": seed}

    generation_values = raw.get("generation", {})
    if generation_values is None:
        generation_values = {}
    if isinstance(generation_values, dict):
        generation_values = {**generation_values, "seed": seed}

    return ExperimentConfig(
        model=_dataclass_from_dict(
            VLLMModelConfig,
            model_values,
            section="model",
        ),
        chat=_dataclass_from_dict(
            ChatTemplateConfig,
            raw.get("chat", {}),
            section="chat",
        ),
        generation=_dataclass_from_dict(
            GenerationConfig,
            generation_values,
            section="generation",
        ),
        reasoning=_dataclass_from_dict(
            VLLMReasoningConfig,
            raw.get("reasoning", {}),
            section="reasoning",
        ),
    )
