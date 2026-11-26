"""Shared parent-task populations with explicit mappings to source feature rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from arc_cd.repr_analysis.data import FeatureBundle

CONCEPT_GROUPS: dict[str, tuple[str, ...]] = {
    "predicate": (
        "AboveBelow",
        "Center",
        "FilledNotFilled",
        "HorizontalVertical",
        "InsideOutside",
        "SameDifferent",
        "TopBottom2D",
        "TopBottom3D",
    ),
    "action": (
        "CleanUp",
        "CompleteShape",
        "Copy",
        "Count",
        "ExtendToBoundary",
        "ExtractObjects",
        "MoveToBoundary",
        "Order",
    ),
}
POPULATIONS: tuple[str, ...] = ("predicate", "action", "all")


@dataclass(frozen=True)
class AnalysisPopulation:
    """Selected parents and local labels, retaining the complete source contract.

    Attributes:
        source (FeatureBundle): Complete source used for archive validation.
        name (str): Requested population name.
        parent_indices (np.ndarray): Selected indices into the source parent axis.
        concepts (list[str]): Selected vocabulary in original source order.
        labels (np.ndarray): Contiguous local concept index per selected parent.
    """

    source: FeatureBundle
    name: str
    parent_indices: np.ndarray
    concepts: list[str]
    labels: np.ndarray

    @property
    def task_ids(self) -> list[str]:
        """Return selected parent identities in source order.

        Returns:
            list[str]: Concept-scoped parent task IDs.
        """
        return [self.source.task_ids[i] for i in self.parent_indices]

    @property
    def source_rows(self) -> np.ndarray:
        """Return archive indices for all eight versions of each selected parent.

        Returns:
            np.ndarray: Original feature indices, shaped [parents, eight versions].
        """
        return self.source.rows[self.parent_indices]

    @property
    def record_indices(self) -> np.ndarray:
        """Return selected manifest indices in original archive order.

        Returns:
            np.ndarray: Source row for each locally saved support record.
        """
        return np.sort(self.source_rows.ravel())

    @property
    def rows(self) -> np.ndarray:
        """Return dense local indices for saved support manifests and galleries.

        Returns:
            np.ndarray: Local support indices, shaped [parents, eight versions].
        """
        return np.searchsorted(self.record_indices, self.source_rows)

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return selected source records without altering their saved identities.

        Returns:
            list[dict[str, Any]]: Source records in local support-manifest order.
        """
        return [self.source.records[i] for i in self.record_indices]

    def snapshot(self) -> dict[str, Any]:
        """Describe the resolved population and the group definition used.

        Returns:
            dict[str, Any]: JSON-compatible population provenance.
        """
        return {
            "name": self.name,
            "concept_groups": {
                name: list(group) for name, group in CONCEPT_GROUPS.items()
            },
            "concepts": self.concepts,
            "num_parents": len(self.parent_indices),
            "num_versions": self.source_rows.size,
            "source_num_parents": len(self.source.task_ids),
            "source_num_versions": len(self.source.records),
        }


def select_population(bundle: FeatureBundle, name: str = "all") -> AnalysisPopulation:
    """Select whole concepts without changing archive validation or source labels.

    Args:
        bundle (FeatureBundle): Complete validated source bundle.
        name (str): Predicate, action, or all available concepts.

    Returns:
        AnalysisPopulation: Selected parents and source/local row mappings.

    Raises:
        ValueError: If the mode is unknown or an explicit group is incomplete or
            the bundle contains unrecognized concept names for a named selection.
    """
    if name not in POPULATIONS:
        raise ValueError(f"Unknown population: {name}; expected one of {POPULATIONS}.")
    if name == "all":
        concepts = list(bundle.concepts)
    else:
        known = {concept for group in CONCEPT_GROUPS.values() for concept in group}
        missing = sorted(set(CONCEPT_GROUPS[name]) - set(bundle.concepts))
        unknown = sorted(set(bundle.concepts) - known)
        if missing or unknown:
            raise ValueError(
                f"Cannot select {name} population: missing concepts {missing}; "
                f"unrecognized concepts {unknown}."
            )
        concepts = [
            concept for concept in bundle.concepts if concept in CONCEPT_GROUPS[name]
        ]
    source_labels = [bundle.concepts.index(concept) for concept in concepts]
    parents = np.flatnonzero(np.isin(bundle.labels, source_labels))
    remapping = {label: index for index, label in enumerate(source_labels)}
    labels = np.asarray(
        [remapping[int(bundle.labels[i])] for i in parents], dtype=np.int32
    )
    return AnalysisPopulation(bundle, name, parents, concepts, labels)


def population_label(config: dict[str, Any]) -> str:
    """Describe the saved population in figures, preserving legacy plot labels.

    Args:
        config (dict[str, Any]): Saved run configuration or axes.

    Returns:
        str: Population and counts, or an empty string for legacy artifacts.
    """
    population = config.get("population")
    if population is None:
        return ""
    return (
        f"{population['name'].capitalize()} population · "
        f"{len(population['concepts'])} concepts · {population['num_parents']} tasks"
    )
