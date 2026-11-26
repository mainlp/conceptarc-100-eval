"""Check how the downloader handles an incomplete benchmark split."""

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.download_data import download_config


def test_short_download_leaves_destination_available_for_retry(tmp_path: Path) -> None:
    """A short stream must not publish a partial dataset or leave a lock."""
    pair = {"input": [[1]], "output": [[1]]}
    row = {
        "concept": "Count",
        "task_id": "Count1",
        "version": 1,
        "train": [pair] * 3,
        "test": [pair] * 10,
    }
    with patch("datasets.load_dataset", return_value=[row]):
        with pytest.raises(ValueError, match="Expected 1,600 rows.*received 1"):
            download_config("ConceptARC-100", tmp_path)
    assert not list(tmp_path.iterdir())
