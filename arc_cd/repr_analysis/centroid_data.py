"""Task-centroid inputs aligned to frozen, model-independent task annotations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arc_cd.repr_analysis.data import fingerprint
from arc_cd.repr_analysis.family_data import FamilyData, build_family_data
from arc_cd.repr_analysis.population import CONCEPT_GROUPS, POPULATIONS
from arc_cd.repr_analysis.spec_metadata import (
    _field_inventory,
    load_taxonomy,
    validate_saved_metadata,
)

CONCEPT_ALIASES = {"Cleanup": "CleanUp", "Extract": "ExtractObjects"}


@dataclass
class TaskCentroids:
    """Selected centroid rows with canonical identities and source provenance.

    Attributes:
        embeddings (np.ndarray): Float64 [task, hidden] input centroids.
        task_ids (list[str]): Canonical parent IDs in metadata order.
        source_task_ids (list[str]): Original IDs for the selected input rows.
        source_rules (list[str]): Original rule names for the selected rows.
        source_rows (np.ndarray): Selected indices into the input NPZ.
        instance_counts (np.ndarray): Number of support sets per centroid.
        concepts (list[str]): Selected canonical concept vocabulary.
        labels (np.ndarray): Concept index per selected task.
        dataset_id (str): Dataset identifier from the annotation source.
        source_num_tasks (int): Number of tasks before population selection.
        families (FamilyData): Validated annotations and aligned memberships.
    """

    embeddings: np.ndarray
    task_ids: list[str]
    source_task_ids: list[str]
    source_rules: list[str]
    source_rows: np.ndarray
    instance_counts: np.ndarray
    concepts: list[str]
    labels: np.ndarray
    dataset_id: str
    source_num_tasks: int
    families: FamilyData


def load_centroid_archive(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Validate eight-support-set centroids and normalize parent identities.

    Args:
        path (Path): NPZ containing embeddings, source_task_id, concept,
            source_rule, and instance_count arrays.

    Returns:
        tuple[dict[str, np.ndarray], list[str]]: Source arrays and canonical parent
        IDs in archive order. Per-support-set identities are not in this format.
    """
    required = {
        "embeddings",
        "source_task_id",
        "concept",
        "source_rule",
        "instance_count",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Centroid archive is missing arrays: {sorted(missing)}")
        arrays = {key: archive[key] for key in required}
    vectors = arrays["embeddings"]
    if (
        vectors.ndim != 2
        or min(vectors.shape) < 1
        or vectors.dtype.kind != "f"
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Expected finite floating-point [task, hidden] embeddings.")
    count = len(vectors)
    for key in ("source_task_id", "concept", "source_rule"):
        if arrays[key].shape != (count,) or arrays[key].dtype.kind not in "US":
            raise ValueError(f"Expected one string {key} per centroid.")
    counts = arrays["instance_count"]
    if counts.shape != (count,) or counts.dtype.kind not in "iu" or np.any(counts != 8):
        raise ValueError("Each centroid must represent exactly eight support sets.")
    source_ids = arrays["source_task_id"].astype(str).tolist()
    source_concepts = arrays["concept"].astype(str).tolist()
    source_rules = arrays["source_rule"].astype(str).tolist()
    canonical_ids = []
    for task, concept, rule in zip(source_ids, source_concepts, source_rules):
        if task != f"{concept}/{rule}":
            raise ValueError(
                f"Centroid identity conflicts with concept/source_rule: {task}"
            )
        if concept in CONCEPT_ALIASES:
            match = re.fullmatch(re.escape(concept) + r"(\d+)", rule)
            if match is None:
                raise ValueError(f"Cannot normalize aliased centroid rule: {task}")
            concept = CONCEPT_ALIASES[concept]
            task = f"{concept}/{concept}{match.group(1)}"
        canonical_ids.append(task)
    if len(set(canonical_ids)) != count:
        raise ValueError("Duplicate centroid task IDs after alias normalization.")
    return arrays, canonical_ids


def load_task_centroids(
    path: Path,
    metadata_dir: Path,
    roles: list[str],
    population: str = "all",
) -> TaskCentroids:
    """Validate eight-support-set centroids and join frozen metadata by parent ID.

    Args:
        path (Path): NPZ containing embeddings, source_task_id, concept,
            source_rule, and instance_count arrays.
        metadata_dir (Path): Published metadata explorer covering the input tasks.
        roles (list[str]): Requested action, layout, and/or selector roles.
        population (str): All tasks or a complete predicate/action concept group.

    Returns:
        TaskCentroids: Validated, population-selected inputs in metadata task order.

    Raises:
        ValueError: If arrays, identities, counts, or metadata provenance conflict.
    """
    if population not in POPULATIONS:
        raise ValueError(f"Unknown population: {population}")
    arrays, canonical_ids = load_centroid_archive(path)
    vectors, counts = arrays["embeddings"], arrays["instance_count"]
    source_ids = arrays["source_task_id"].astype(str).tolist()
    source_rules = arrays["source_rule"].astype(str).tolist()
    count = len(vectors)

    config = json.loads((metadata_dir / "run_config.json").read_text())
    axes = json.loads((metadata_dir / "axes.json").read_text())
    metadata = json.loads((metadata_dir / "task_metadata.json").read_text())
    taxonomy = load_taxonomy(metadata_dir / "taxonomy.json")
    if (
        config.get("analysis") != "metadata_explorer"
        or config.get("schema_version") != 1
    ):
        raise ValueError("--metadata_dir must be a published metadata explorer.")
    if config.get("taxonomy_version") != taxonomy["version"] or config.get(
        "taxonomy_fingerprint"
    ) != fingerprint(metadata_dir / "taxonomy.json"):
        raise ValueError("Metadata taxonomy fingerprint/version mismatch.")
    validate_saved_metadata(metadata, taxonomy)
    if config.get("population") != axes.get("population"):
        raise ValueError("Metadata population differs between configuration and axes.")
    records = metadata["records"]
    task_ids = axes["task_ids"]
    concepts = axes["concepts"]
    labels = axes["labels"]
    if (
        len(set(task_ids)) != len(task_ids)
        or [record.get("task_id") for record in records] != task_ids
        or not concepts
        or len(set(concepts)) != len(concepts)
        or len(labels) != len(task_ids)
        or any(
            type(label) is not int or not 0 <= label < len(concepts) for label in labels
        )
    ):
        raise ValueError("Metadata parent identities or concept axes are invalid.")
    if set(canonical_ids) != set(task_ids):
        missing_ids = sorted(set(task_ids) - set(canonical_ids))
        extra_ids = sorted(set(canonical_ids) - set(task_ids))
        raise ValueError(
            "Centroid and metadata task sets must match before population selection: "
            f"{len(missing_ids)} missing ({missing_ids[:3]}), "
            f"{len(extra_ids)} extra ({extra_ids[:3]})."
        )
    spec_dir = Path(config["arguments"]["spec_dir"])
    for record, label in zip(records, labels):
        if (
            record["concept_name"] != concepts[label]
            or record["task_id"].split("/")[0] != concepts[label]
        ):
            raise ValueError("Metadata concepts conflict with parent identities.")
        try:
            relative = Path(record["spec_path"]).relative_to(spec_dir).as_posix()
        except ValueError as error:
            raise ValueError(
                "Metadata spec provenance is outside its recorded source."
            ) from error
        expected = config.get("spec_fingerprints", {}).get(relative)
        if (
            not isinstance(expected, dict)
            or record.get("source_fingerprint") != expected
        ):
            raise ValueError(f"Metadata spec fingerprint mismatch: {record['task_id']}")
    dataset_id = config["extraction_config"]["dataset"]["id"]
    if (
        not isinstance(dataset_id, str)
        or Path(dataset_id).name != dataset_id
        or dataset_id in ("", ".", "..")
    ):
        raise ValueError("Metadata dataset identifier must be a single directory name.")
    if population != "all":
        known = {name for group in CONCEPT_GROUPS.values() for name in group}
        if set(CONCEPT_GROUPS[population]) - set(concepts) or set(concepts) - known:
            raise ValueError(
                f"Cannot select incomplete or unrecognized {population} concepts."
            )
        concepts = [name for name in concepts if name in CONCEPT_GROUPS[population]]
    selected = [record for record in records if record["concept_name"] in concepts]
    task_ids = [record["task_id"] for record in selected]
    labels_array = np.asarray(
        [concepts.index(record["concept_name"]) for record in selected], dtype=np.int32
    )
    row_by_id = {task: row for row, task in enumerate(canonical_ids)}
    source_rows = np.asarray([row_by_id[task] for task in task_ids], dtype=np.int64)
    metadata = {
        **metadata,
        "records": selected,
        "fields": _field_inventory(selected, taxonomy),
    }
    families = build_family_data(
        metadata, taxonomy, config, concepts, labels_array, roles
    )
    return TaskCentroids(
        vectors[source_rows].astype(np.float64),
        task_ids,
        [source_ids[row] for row in source_rows],
        [source_rules[row] for row in source_rows],
        source_rows,
        counts[source_rows],
        concepts,
        labels_array,
        dataset_id,
        count,
        families,
    )
