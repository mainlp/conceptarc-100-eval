"""
Modules and functions to store and process hidden states from LLMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HiddenStateFeatures:
    """Extracted selected-token hidden-state features.

    Attributes:
        x: Feature tensor with shape ``[n_samples, n_layers, hidden_size]``.
        selected_token_indices: Selected indices in the full original sequence.
        sequence_lengths: Replay prefix lengths.
        metadata: Per-sample metadata.
    """

    x: np.ndarray
    selected_token_indices: np.ndarray
    sequence_lengths: np.ndarray
    metadata: list[dict[str, Any]]


@dataclass(frozen=True)
class PromptReplaySample:
    """One support prefix with multiple selected token positions.

    Attributes:
        case_id (str): Support-specific case identity.
        token_ids (list[int]): Exact inference token prefix through S3.
        positions (dict[str, int]): Ordered position names and absolute indices.
    """

    case_id: str
    token_ids: list[int]
    positions: dict[str, int]
