"""
Utility functions and modules.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CONFIG_DIR = PACKAGE_DIR / "models" / "model_config"
DEFAULT_CONCEPT_ARC_PROMPT_DIR = PACKAGE_DIR.parent / "prompts" / "concept_arc"


def setup_logging(verbosity: int) -> None:
    """
    Set up logging based on the verbosity level.

    Args:
    verbosity (int): Verbosity level from command line arguments.
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


def set_seed(seed: int = 0) -> None:
    """Seed the random generators used by inference and support extraction.

    Args:
        seed (int): Random seed in NumPy's unsigned 32-bit range.
    """
    import numpy as np
    import torch

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_directory(root: str | Path, dataset_name: str) -> Path:
    """Resolve and validate a dataset root directory.

    Args:
        root: Dataset root directory.
        dataset_name: Dataset name used in error messages.

    Returns:
        Resolved root path.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` is not a directory.
    """
    root_path = Path(root).expanduser().resolve()

    if not root_path.exists():
        raise FileNotFoundError(f"{dataset_name} root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"{dataset_name} root is not a directory: {root_path}")

    return root_path


def optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    """Return a tuple of integers or ``None``.

    Args:
        value (Any): Value to normalize.

    Returns:
        tuple[int, ...] | None: Tuple of integers, or ``None`` if the value is
        absent or malformed.
    """
    if value is None:
        return None

    if not isinstance(value, (list, tuple)):
        return None

    values: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        values.append(item)

    return tuple(values)


def string_tuple(value: Any) -> tuple[str, ...]:
    """Return a tuple of strings.

    Args:
        value (Any): Value to normalize.

    Returns:
        tuple[str, ...]: Tuple of strings. Returns an empty tuple if the value
        is absent or malformed.
    """
    if not isinstance(value, (list, tuple)):
        return ()

    return tuple(str(item) for item in value)


def load_json(file_path: str | Path) -> Any:
    """Read a JSON value from disk.

    Args:
        file_path (str | Path): Input JSON file.

    Returns:
        Any: The decoded JSON value.
    """
    with Path(file_path).expanduser().open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file.

    Args:
        path: JSONL path.

    Returns:
        List of records.
    """
    samples: list[dict[str, Any]] = []

    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    return samples


def save_json(
    data: dict[str, Any] | list[dict[str, Any]], file_path: str | Path
) -> None:
    """
    Saves a given list of dictionaries to a JSON file.

    Args:
        data (dict[str, Any] | list[dict[str, Any]]): Data to serialize.
        file_path (str | Path): Destination JSON file.
    """
    file_path = Path(file_path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write records to a JSONL file.

    Args:
        records: JSON-serializable records.
        path: Output path.
    """
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def infer_model_name_from_generation_path(
    input_jsonl: str | Path,
    model_config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR,
) -> str:
    """Infer a model config name from a generation JSONL path.

    The preferred layout is
    ``.../{model_name}/generations/seed_<id>/{run}.jsonl`` or
    ``.../{model_name}/truncated_generations/seed_<id>/{run}.jsonl``. If that
    does not point to an existing YAML config, this falls back to matching known
    config names inside the filename stem.

    Args:
        input_jsonl (str | Path): Generation JSONL path.
        model_config_dir (str | Path): Directory containing model YAML configs.

    Returns:
        str: Inferred model config name without ``.yaml``.

    Raises:
        ValueError: If no matching config can be inferred.
    """
    input_path = Path(input_jsonl).expanduser().resolve()
    config_dir = Path(model_config_dir).expanduser().resolve()

    if input_path.parent.parent.name in {"generations", "truncated_generations"}:
        directory_candidate = input_path.parent.parent.parent.name
        if (config_dir / f"{directory_candidate}.yaml").exists():
            return directory_candidate

    config_names = sorted(
        (path.stem for path in config_dir.glob("*.yaml")),
        key=len,
        reverse=True,
    )
    stem = input_path.stem
    padded_stem = f"_{stem}_"

    for config_name in config_names:
        if f"_{config_name}_" in padded_stem:
            return config_name

    valid = ", ".join(config_names)
    raise ValueError(
        f"Could not infer model config from generation path: {input_path}. "
        "Expected a path like "
        ".../<model_name>/generations/seed_<id>/<run>.jsonl, "
        ".../<model_name>/truncated_generations/seed_<id>/<run>.jsonl, "
        "or a filename "
        f"containing one of: {valid}."
    )


def resolve_reasoning_end_marker(
    input_jsonl: str | Path,
    model_config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR,
) -> str:
    """Read the marker saved with a run, using the preset for older artifacts.

    Args:
        input_jsonl (str | Path): Generation JSONL path.
        model_config_dir (str | Path): Directory containing model YAML configs.

    Returns:
        str: Marker separating reasoning from the final answer.
    """
    from arc_cd.models.config import load_experiment_config

    path = Path(input_jsonl).expanduser()
    if path.name.endswith("_generations.jsonl"):
        run_name = path.name.removesuffix("_generations.jsonl")
        run_config = path.with_name(f"{run_name}_config.json")
        if run_config.is_file():
            reasoning = load_json(run_config).get("config", {}).get("reasoning", {})
            if "reasoning_end_marker" in reasoning:
                marker = reasoning["reasoning_end_marker"]
                if not isinstance(marker, str) or not marker:
                    raise ValueError(f"Invalid reasoning_end_marker in {run_config}")
                return marker

    model_name = infer_model_name_from_generation_path(
        input_jsonl,
        model_config_dir=model_config_dir,
    )
    config_path = Path(model_config_dir).expanduser().resolve() / f"{model_name}.yaml"
    return load_experiment_config(config_path).reasoning.reasoning_end_marker


def safe_divide(numerator: int, denominator: int) -> float:
    """Safely divide two integers."""
    return float(numerator / denominator) if denominator else 0.0


def input_device(model: PreTrainedModel) -> torch.device:
    """Return the device expected for input IDs.

    Args:
        model: Hugging Face causal language model.

    Returns:
        Device of the model's input embeddings.
    """
    return model.get_input_embeddings().weight.device
