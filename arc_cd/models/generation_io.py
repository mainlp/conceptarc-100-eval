from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from arc_cd.models.config import ExperimentConfig
from arc_cd.prompt_prep.prompt_format import PromptTemplates
from arc_cd.utils import load_jsonl, save_json

load_generation_jsonl = load_jsonl


@dataclass(frozen=True)
class RaggedTokenArray:
    """Compact representation of variable-length token sequences.

    Attributes:
        data: Flattened token IDs.
        offsets: Sequence boundaries. Sequence i is stored in
            ``data[offsets[i]:offsets[i + 1]]``.
    """

    data: np.ndarray
    offsets: np.ndarray

    def __len__(self) -> int:
        """Return the number of token sequences."""
        return len(self.offsets) - 1

    def get(self, index: int) -> list[int]:
        """Return one token sequence.

        Args:
            index: Sequence index.

        Returns:
            Token IDs for the requested sequence.
        """
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.data[start:end].astype(int).tolist()


@dataclass(frozen=True)
class TokenArchive:
    """Token sidecar loaded from disk.

    Attributes:
        case_ids: Case IDs aligned with token arrays.
        prompt_token_ids: Prompt token IDs.
        generated_token_ids: Generated token IDs.
    """

    case_ids: tuple[str, ...]
    prompt_token_ids: RaggedTokenArray
    generated_token_ids: RaggedTokenArray

    def get_by_index(self, index: int) -> dict[str, list[int]]:
        """Return token IDs for one generation by index.

        Args:
            index: Generation index.

        Returns:
            Dictionary containing prompt and generated token IDs.
        """
        return {
            "prompt_token_ids": self.prompt_token_ids.get(index),
            "generated_token_ids": self.generated_token_ids.get(index),
        }

    def get_by_case_id(self, case_id: str) -> dict[str, list[int]]:
        """Return token IDs for one generation by case ID.

        Args:
            case_id: ARC-style case ID.

        Returns:
            Dictionary containing prompt and generated token IDs.

        Raises:
            KeyError: If the case ID is absent.
        """
        try:
            index = self.case_ids.index(case_id)
        except ValueError as exc:
            raise KeyError(f"Unknown case_id: {case_id}") from exc

        return self.get_by_index(index)


def _flatten_token_sequences(
    sequences: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten variable-length token sequences.

    Args:
        sequences: Token ID sequences.

    Returns:
        Pair of ``data`` and ``offsets`` arrays.
    """
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)

    for i, sequence in enumerate(sequences):
        offsets[i + 1] = offsets[i] + len(sequence)

    data = np.empty(int(offsets[-1]), dtype=np.int64)

    cursor = 0
    for sequence in sequences:
        length = len(sequence)
        data[cursor : cursor + length] = np.asarray(sequence, dtype=np.int64)
        cursor += length

    return data, offsets


def write_token_archive_npz(
    records: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write prompt and generated token IDs to a compressed NPZ archive.

    Args:
        records: Generation result records.
        path: Output NPZ path.
    """
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    case_ids = [str(record["case_id"]) for record in records]
    prompt_token_ids = [record["prompt_token_ids"] for record in records]
    generated_token_ids = [record["generated_token_ids"] for record in records]

    prompt_data, prompt_offsets = _flatten_token_sequences(prompt_token_ids)
    generated_data, generated_offsets = _flatten_token_sequences(generated_token_ids)

    np.savez_compressed(
        output_path,
        case_ids=np.asarray(case_ids, dtype=str),
        prompt_token_ids_data=prompt_data,
        prompt_token_ids_offsets=prompt_offsets,
        generated_token_ids_data=generated_data,
        generated_token_ids_offsets=generated_offsets,
    )


def write_generation_outputs(
    records: list[dict[str, Any]],
    jsonl_path: str | Path,
    token_npz_path: str | Path,
) -> None:
    """Write generation outputs as text JSONL plus token-ID sidecar.

    Args:
        records: Generation result records. Each record should contain
            ``prompt_token_ids`` and ``generated_token_ids``.
        jsonl_path: Output path for human-readable JSONL.
        token_npz_path: Output path for compressed token arrays.
    """
    jsonl = Path(jsonl_path).expanduser().resolve()
    token_npz = Path(token_npz_path).expanduser().resolve()

    jsonl.parent.mkdir(parents=True, exist_ok=True)
    token_npz.parent.mkdir(parents=True, exist_ok=True)

    write_token_archive_npz(records, token_npz)

    with jsonl.open("w", encoding="utf-8") as f:
        for token_index, record in enumerate(records):
            text_record = dict(record)
            text_record.pop("prompt_token_ids", None)
            text_record.pop("generated_token_ids", None)

            text_record["token_index"] = token_index
            text_record["token_archive"] = token_npz.name

            f.write(json.dumps(text_record, ensure_ascii=False) + "\n")


def load_token_archive_npz(path: str | Path) -> TokenArchive:
    """Load a compressed token archive.

    Args:
        path: Path to NPZ token archive.

    Returns:
        Loaded token archive.
    """
    archive_path = Path(path).expanduser().resolve()

    with np.load(archive_path, allow_pickle=False) as data:
        return TokenArchive(
            case_ids=tuple(str(x) for x in data["case_ids"].tolist()),
            prompt_token_ids=RaggedTokenArray(
                data=data["prompt_token_ids_data"],
                offsets=data["prompt_token_ids_offsets"],
            ),
            generated_token_ids=RaggedTokenArray(
                data=data["generated_token_ids_data"],
                offsets=data["generated_token_ids_offsets"],
            ),
        )


def write_run_config(
    config: ExperimentConfig,
    path: str | Path,
    *,
    prompt_dir: str | Path | None = None,
    prompt_templates: PromptTemplates | None = None,
    generation_jsonl_path: str | Path | None = None,
    token_npz_path: str | Path | None = None,
    num_records: int | None = None,
    data_selection: dict[str, str] | None = None,
) -> None:
    """Write a minimal run configuration JSON file.

    Args:
        config (ExperimentConfig): Configuration used for generation.
        path (str | Path): Output path for the run config JSON.
        prompt_dir (str | Path | None): Directory containing prompt templates.
        prompt_templates (PromptTemplates | None): Exact templates used to generate the prompts.
        generation_jsonl_path (str | Path | None): Generation-output JSONL path.
        token_npz_path (str | Path | None): Token archive NPZ path.
        num_records (int | None): Number of generated records.
        data_selection (dict[str, str] | None): Dataset root and selected concept.
    """
    run_config: dict[str, Any] = {
        "config": asdict(config),
        "paths": {
            "prompt_dir": str(prompt_dir) if prompt_dir is not None else None,
            "generation_jsonl": (
                str(generation_jsonl_path)
                if generation_jsonl_path is not None
                else None
            ),
            "token_npz": str(token_npz_path) if token_npz_path is not None else None,
        },
        "num_records": num_records,
    }
    if prompt_templates is not None:
        run_config["templates"] = asdict(prompt_templates)
    if data_selection is not None:
        run_config["data_selection"] = data_selection

    save_json(run_config, path)
