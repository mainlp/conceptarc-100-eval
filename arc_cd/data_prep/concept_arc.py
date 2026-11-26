"""
Classes and modules to represent ConceptARC data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from arc_cd.data_prep import Grid, GridPair

CANONICAL_CONCEPT_LABELS = (
    "AboveBelow",
    "Center",
    "CleanUp",
    "CompleteShape",
    "Copy",
    "Count",
    "ExtendToBoundary",
    "ExtractObjects",
    "FilledNotFilled",
    "HorizontalVertical",
    "InsideOutside",
    "MoveToBoundary",
    "Order",
    "SameDifferent",
    "TopBottom2D",
    "TopBottom3D",
)


@dataclass(frozen=True)
class ConceptARCTask:
    """A ConceptARC task loaded from one JSON file.

    Attributes:
        concept: Concept name, inferred from the parent folder name.
        sample_id: Sample identifier, inferred from the JSON filename stem.
        path: Path to the source JSON file.
        train: Few-shot training input-output examples.
        test: Test input-output examples. Outputs may be absent.
    """

    concept: str
    sample_id: str
    path: Path
    train: tuple[GridPair, ...]
    test: tuple[GridPair, ...]

    @property
    def task_id(self) -> str:
        """Stable task identifier."""
        return f"{self.concept}/{self.sample_id}"

    @property
    def n_train(self) -> int:
        """Number of few-shot training examples."""
        return len(self.train)

    @property
    def n_test(self) -> int:
        """Number of test examples."""
        return len(self.test)


@dataclass(frozen=True)
class ConceptARCTestCase:
    """A flattened test case from a ConceptARC task.

    This is the convenient unit for model inference: one prompt, one test query.

    Attributes:
        concept: Concept name.
        sample_id: Sample identifier.
        test_index: Index of this test query within the task.
        path: Path to the source JSON file.
        train: Few-shot training examples shared by the task.
        test_input: Test query input grid.
        test_output: Ground-truth output grid, if available.
    """

    concept: str
    sample_id: str
    test_index: int
    path: Path
    train: tuple[GridPair, ...]
    test_input: Grid
    test_output: Optional[Grid] = None

    @property
    def task_id(self) -> str:
        """Stable task identifier."""
        return f"{self.concept}/{self.sample_id}"

    @property
    def case_id(self) -> str:
        """Stable test-case identifier."""
        return f"{self.concept}/{self.sample_id}/test_{self.test_index}"
