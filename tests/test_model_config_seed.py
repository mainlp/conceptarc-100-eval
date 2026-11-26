from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from arc_cd.models.config import load_experiment_config
from arc_cd.utils import set_seed


def test_zero_seeds_each_inference_rng() -> None:
    """Extraction uses seed zero for Python, NumPy and PyTorch."""
    with (
        patch("random.seed") as python_seed,
        patch("numpy.random.seed") as numpy_seed,
        patch("torch.manual_seed") as torch_seed,
    ):
        set_seed(0)
    for seed in (python_seed, numpy_seed, torch_seed):
        seed.assert_called_once_with(0)


def test_load_experiment_config_applies_seed_to_model_and_generation(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
model:
  seed: 1
generation:
  seed: 2
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path, seed=7)

    assert config.model.seed == 7
    assert config.generation.seed == 7
