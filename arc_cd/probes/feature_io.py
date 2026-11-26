"""
I/O for support-position hidden states and their aligned metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from arc_cd.probes.hidden import HiddenStateFeatures


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return generation metadata as a dictionary.

    Args:
        record: Generation record.

    Returns:
        Metadata dictionary, or an empty dictionary if unavailable.
    """
    metadata = record.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _metadata_bool(
    records: list[dict[str, Any]],
    key: str,
) -> np.ndarray:
    """Extract a boolean metadata field from records.

    Args:
        records: Generation records.
        key: Metadata key.

    Returns:
        Boolean array aligned with records.
    """
    return np.asarray(
        [bool(_record_metadata(record).get(key, False)) for record in records],
        dtype=np.bool_,
    )


def _metadata_int(
    records: list[dict[str, Any]],
    key: str,
    *,
    missing_value: int = -1,
) -> np.ndarray:
    """Extract an integer metadata field from records.

    Args:
        records: Generation records.
        key: Metadata key.
        missing_value: Value used when the field is absent or malformed.

    Returns:
        Integer array aligned with records.
    """
    values: list[int] = []

    for record in records:
        value = _record_metadata(record).get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            values.append(missing_value)
        else:
            values.append(value)

    return np.asarray(values, dtype=np.int32)


def _metadata_str(
    records: list[dict[str, Any]],
    key: str,
) -> np.ndarray:
    """Extract a string metadata field from records.

    Args:
        records: Generation records.
        key: Metadata key.

    Returns:
        String array aligned with records. Missing values become empty strings.
    """
    values = []

    for record in records:
        value = _record_metadata(record).get(key)
        values.append(value if isinstance(value, str) else "")

    return np.asarray(values, dtype=str)


def _metadata_json(
    records: list[dict[str, Any]],
    key: str,
) -> np.ndarray:
    """Extract a JSON-serialized metadata field from records.

    Args:
        records: Generation records.
        key: Metadata key.

    Returns:
        JSON string array aligned with records.
    """
    return np.asarray(
        [
            json.dumps(_record_metadata(record).get(key), ensure_ascii=False)
            for record in records
        ],
        dtype=str,
    )


def compute_prompt_only_diagnostic_arrays(
    records: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Build neutral diagnostic arrays for prompt-only feature archives.

    Args:
        records (list[dict[str, Any]]): Prompt-only records aligned with the
            extracted features.

    Returns:
        dict[str, np.ndarray]: Diagnostic arrays with the same keys as regular
        generation feature archives.
    """
    n_records = len(records)
    prompt_only_labels = np.asarray(["prompt_only"] * n_records, dtype=str)

    return {
        "answer_correct": np.zeros(n_records, dtype=np.bool_),
        "parse_ok": np.zeros(n_records, dtype=np.bool_),
        "parse_status": prompt_only_labels,
        "response_finished": np.zeros(n_records, dtype=np.bool_),
        "hit_output_token_budget": np.zeros(n_records, dtype=np.bool_),
        "finish_reason": prompt_only_labels,
        "diagnostic_parse_ok": np.zeros(n_records, dtype=np.bool_),
        "diagnostic_parse_status": prompt_only_labels,
        "hit_thinking_token_budget": np.zeros(n_records, dtype=np.bool_),
        "forced_reasoning_stop": np.zeros(n_records, dtype=np.bool_),
        "reasoning_over_budget": np.zeros(n_records, dtype=np.bool_),
        "has_reasoning_content": np.zeros(n_records, dtype=np.bool_),
        "has_closed_reasoning": np.zeros(n_records, dtype=np.bool_),
        "has_final_content": np.zeros(n_records, dtype=np.bool_),
    }


def save_feature_archive(
    path: str | Path,
    *,
    features: HiddenStateFeatures,
    y: np.ndarray,
    concepts: list[str],
    layers: list[int],
    records: list[dict[str, Any]],
    token_position: str,
) -> None:
    """Save hidden states and aligned metadata.

    Args:
        path (str | Path): Output NPZ path.
        features (HiddenStateFeatures): Extracted hidden-state features.
        y (np.ndarray): Target matrix aligned with features.
        concepts (list[str]): Concept names aligned with columns of ``y``.
        layers (list[int]): Extracted layer indices.
        records (list[dict[str, Any]]): Records aligned with features. Optional
            parent_task_id/support_set_id metadata adds aligned identity arrays.
        token_position (str): Token position name.
    """
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    identity_arrays = {
        plural: _metadata_str(records, singular)
        for singular, plural in (
            ("parent_task_id", "parent_task_ids"),
            ("support_set_id", "support_set_ids"),
        )
        if any(singular in _record_metadata(record) for record in records)
    }

    np.savez_compressed(
        output_path,
        x=features.x,
        y=y,
        concepts=np.asarray(concepts, dtype=str),
        layers=np.asarray(layers, dtype=np.int32),
        case_ids=np.asarray([str(r["case_id"]) for r in records], dtype=str),
        task_ids=np.asarray([str(r["task_id"]) for r in records], dtype=str),
        primary_concepts=np.asarray(
            [str(r.get("concept") or "") for r in records],
            dtype=str,
        ),
        test_indices=np.asarray(
            [int(r["test_index"]) for r in records],
            dtype=np.int32,
        ),
        is_resampled_episode=_metadata_bool(records, "is_resampled_episode"),
        episode_ids=_metadata_str(records, "episode_id"),
        query_indices=_metadata_int(records, "query_index"),
        support_indices_json=_metadata_json(records, "support_indices"),
        support_sources_json=_metadata_json(records, "support_sources"),
        query_sources=_metadata_str(records, "query_source"),
        **compute_prompt_only_diagnostic_arrays(records),
        **identity_arrays,
        token_position=np.asarray(token_position, dtype=str),
        selected_token_indices=features.selected_token_indices,
        sequence_lengths=features.sequence_lengths,
        metadata_json=np.asarray(
            [json.dumps(m, ensure_ascii=False) for m in features.metadata],
            dtype=str,
        ),
    )


def load_feature_archive(path: str | Path) -> dict[str, Any]:
    """Load the arrays for one support position.

    Args:
        path: Feature archive NPZ path.

    Returns:
        Dictionary of archive arrays.
    """
    archive_path = Path(path).expanduser().resolve()

    with np.load(archive_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
