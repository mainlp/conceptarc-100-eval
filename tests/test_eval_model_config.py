from __future__ import annotations

from pathlib import Path

from arc_cd.utils import (
    infer_model_name_from_generation_path,
    resolve_reasoning_end_marker,
)


def test_infer_model_name_from_seeded_generation_directory() -> None:
    path = (
        Path("output")
        / "arc_agi_1"
        / "qwen35_27b"
        / "generations"
        / "seed_7"
        / "arc_agi_1_training_qwen35_27b_seed_7_generations.jsonl"
    )

    assert infer_model_name_from_generation_path(path) == "qwen35_27b"
    assert resolve_reasoning_end_marker(path) == "</think>"


def test_infer_model_name_from_seeded_truncated_generation_directory() -> None:
    path = (
        Path("output")
        / "arc_agi_1"
        / "qwen35_27b"
        / "truncated_generations"
        / "seed_7"
        / "arc_agi_1_training_qwen35_27b_seed_7_generations"
        "_reasoning_truncation_profile.jsonl"
    )

    assert infer_model_name_from_generation_path(path) == "qwen35_27b"
    assert resolve_reasoning_end_marker(path) == "</think>"


def test_infer_model_name_from_generation_filename() -> None:
    path = (
        Path("tmp") / "arc_agi_1_training_qwen3_32b_generations"
        "_reasoning_truncation_profile.jsonl"
    )

    assert infer_model_name_from_generation_path(path) == "qwen3_32b"
