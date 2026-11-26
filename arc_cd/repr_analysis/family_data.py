"""Validated frozen metadata memberships for quantitative representation analyses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from arc_cd.repr_analysis.data import FeatureBundle, fingerprint
from arc_cd.repr_analysis.population import AnalysisPopulation, select_population
from arc_cd.repr_analysis.spec_metadata import (
    _field_inventory,
    load_taxonomy,
    validate_saved_metadata,
)


@dataclass
class FamilyData:
    """Frozen annotations and overlapping memberships aligned to parent tasks.

    Attributes:
        metadata (dict[str, Any]): Frozen annotations in selected parent order.
        taxonomy (dict[str, Any]): Validated taxonomy snapshot.
        config (dict[str, Any]): Metadata explorer provenance.
        ids (list[str]): Qualified concept/family identifiers.
        roles (list[str]): Role of each membership column, including concept.
        names (list[str]): Unqualified column names.
        membership (np.ndarray): Boolean [task, family] matrix.
        role_order (list[str]): Requested roles, including roles with no mapped tags.
    """

    metadata: dict[str, Any]
    taxonomy: dict[str, Any]
    config: dict[str, Any]
    ids: list[str]
    roles: list[str]
    names: list[str]
    membership: np.ndarray
    role_order: list[str]


def load_family_data(
    directory: Path,
    bundle: FeatureBundle,
    roles: list[str],
    feature_fingerprints: dict[str, Any],
    *,
    population: AnalysisPopulation | None = None,
) -> FamilyData:
    """Validate a frozen metadata population, then select requested analysis parents.

    Args:
        directory (Path): Published metadata explorer directory.
        bundle (FeatureBundle): Complete validated feature bundle.
        roles (list[str]): Requested metadata roles.
        feature_fingerprints (dict[str, Any]): Hashes of used feature inputs.
        population (AnalysisPopulation | None): Requested parents, all of which must
            exist in the metadata. Validate all supplied metadata before selection.

    Returns:
        FamilyData: Validated annotations and overlapping binary tags. Displayed
        embedding layers do not constrain the layers available for analysis.
    """
    if population is not None and population.source is not bundle:
        raise ValueError("Population must refer to the supplied feature bundle.")
    config = json.loads((directory / "run_config.json").read_text())
    axes = json.loads((directory / "axes.json").read_text())
    metadata = json.loads((directory / "task_metadata.json").read_text())
    taxonomy = load_taxonomy(directory / "taxonomy.json")
    if (
        config.get("analysis") != "metadata_explorer"
        or config.get("schema_version") != 1
    ):
        raise ValueError("--metadata_dir must be a published metadata explorer.")
    if config.get("extraction_config") != bundle.config:
        raise ValueError("Metadata extraction configuration differs from features.")
    if config.get("taxonomy_version") != taxonomy["version"] or config.get(
        "taxonomy_fingerprint"
    ) != fingerprint(directory / "taxonomy.json"):
        raise ValueError("Metadata taxonomy fingerprint/version mismatch.")
    for name, digest in feature_fingerprints.items():
        if config.get("feature_fingerprints", {}).get(name) != digest:
            raise ValueError(f"Metadata feature fingerprint mismatch: {name}")
    validate_saved_metadata(metadata, taxonomy)
    snapshot = config.get("population")
    if snapshot != axes.get("population"):
        raise ValueError("Metadata population differs between configuration and axes.")
    metadata_population = None
    if snapshot is not None:
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("name"), str):
            raise ValueError("Invalid metadata population snapshot.")
        metadata_population = select_population(bundle, snapshot["name"])
        if snapshot != metadata_population.snapshot():
            raise ValueError("Metadata population conflicts with the feature source.")
    metadata_source = metadata_population if metadata_population is not None else bundle
    records_source, rows_source = metadata_source.records, metadata_source.rows
    expected_axes = {
        "task_ids": metadata_source.task_ids,
        "concepts": metadata_source.concepts,
        "labels": metadata_source.labels.tolist(),
        "version_rows": rows_source.tolist(),
        "version_task_ids": [
            [records_source[r]["task_id"] for r in rows] for rows in rows_source
        ],
        "support_set_ids": [
            [records_source[r]["metadata"]["support_set_id"] for r in rows]
            for rows in rows_source
        ],
        "case_ids": [
            [records_source[r]["case_id"] for r in rows] for rows in rows_source
        ],
    }
    if metadata_population is not None:
        expected_axes.update(
            {
                "source_parent_indices": metadata_population.parent_indices.tolist(),
                "source_version_rows": metadata_population.source_rows.tolist(),
                "source_record_indices": metadata_population.record_indices.tolist(),
            }
        )
    for key, expected in expected_axes.items():
        if axes.get(key) != expected:
            raise ValueError(f"Metadata parent/variant axis mismatch: {key}")
    records = metadata["records"]
    if [record.get("task_id") for record in records] != metadata_source.task_ids:
        raise ValueError(
            "Metadata parent identities are duplicated, missing, or misaligned."
        )
    spec_dir = Path(config["arguments"]["spec_dir"])
    for record, label in zip(records, metadata_source.labels):
        if record["concept_name"] != metadata_source.concepts[label]:
            raise ValueError("Metadata concept labels conflict with feature labels.")
        try:
            relative = Path(record["spec_path"]).relative_to(spec_dir).as_posix()
        except ValueError as error:
            raise ValueError(
                "Metadata spec provenance is outside its recorded source."
            ) from error
        expected_fingerprint = config.get("spec_fingerprints", {}).get(relative)
        if (
            not isinstance(expected_fingerprint, dict)
            or record.get("source_fingerprint") != expected_fingerprint
        ):
            raise ValueError(f"Metadata spec fingerprint mismatch: {record['task_id']}")
    selected = population if population is not None else bundle
    by_id = {record["task_id"]: record for record in records}
    missing = sorted(set(selected.task_ids) - by_id.keys())
    if missing:
        raise ValueError(
            f"Metadata does not cover the requested population: {len(missing)} missing "
            "parents. Use a matching population or an all-population metadata explorer."
        )
    if population is not None:
        records = [by_id[task] for task in population.task_ids]
        metadata = {
            **metadata,
            "records": records,
            "fields": _field_inventory(records, taxonomy),
        }
    return build_family_data(
        metadata, taxonomy, config, selected.concepts, selected.labels, roles
    )


def build_family_data(
    metadata: dict[str, Any],
    taxonomy: dict[str, Any],
    config: dict[str, Any],
    concepts: list[str],
    labels: np.ndarray,
    roles: list[str],
) -> FamilyData:
    """Build memberships from validated annotations in the requested task order.

    Args:
        metadata (dict[str, Any]): Validated annotations in selected task order.
        taxonomy (dict[str, Any]): Validated frozen taxonomy.
        config (dict[str, Any]): Annotation-source provenance.
        concepts (list[str]): Selected concept vocabulary.
        labels (np.ndarray): Concept indices aligned to the metadata records.
        roles (list[str]): Requested metadata roles.

    Returns:
        FamilyData: Concept and overlapping-family memberships in task order.
    """
    records = metadata["records"]
    names = list(concepts)
    column_roles = ["concept"] * len(names)
    columns = [labels == index for index in range(len(names))]
    for role in roles:
        vocabulary = sorted(
            {tag for record in records for tag in record["families"][role]}
        )
        names.extend(vocabulary)
        column_roles.extend([role] * len(vocabulary))
        columns.extend(
            np.array(
                [name in record["families"][role] for record in records], dtype=bool
            )
            for name in vocabulary
        )
    return FamilyData(
        metadata,
        taxonomy,
        config,
        [f"{role}/{name}" for role, name in zip(column_roles, names)],
        column_roles,
        names,
        np.stack(columns, axis=1),
        ["concept", *roles],
    )


def family_comparison_mask(
    families: FamilyData, *, include_structural_selectors: bool = False
) -> np.ndarray:
    """Select criteria for headline comparisons while retaining all detailed tags.

    Structural action tags are always excluded. The existing selector option
    controls only structural selectors. Frozen taxonomies without explicit kinds
    retain their old behavior.

    Args:
        families (FamilyData): Validated family columns and taxonomy snapshot.
        include_structural_selectors (bool): Include extraction/set-operation tags.

    Returns:
        np.ndarray: Boolean mask aligned to the complete family axis.
    """
    return np.asarray(
        [
            role == "concept"
            or families.taxonomy["families"][role][name].get("kind", "criterion")
            != "structural"
            or (role == "selector" and include_structural_selectors)
            for role, name in zip(families.roles, families.names)
        ],
        dtype=bool,
    )
