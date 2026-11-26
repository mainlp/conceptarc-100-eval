"""Population isolation, source alignment, and analysis publication contracts."""

from __future__ import annotations

import copy
import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from arc_cd.probes.feature_io import load_feature_archive
from arc_cd.probes.prompt_feature_io import (
    _load_shard_records,
    merge_prompt_feature_shards,
    write_prompt_feature_run,
)
from arc_cd.probes.prompt_features import SUPPORT_POSITIONS
from arc_cd.repr_analysis import visualization
from arc_cd.repr_analysis.data import center_versions, fingerprint, load_bundle
from arc_cd.repr_analysis.embedding_runner import run_embedding
from arc_cd.repr_analysis.explorer import load_explorer_data, write_explorer
from arc_cd.repr_analysis.family_data import load_family_data
from arc_cd.repr_analysis.metadata_runner import run_metadata_explorer
from arc_cd.repr_analysis.population import (
    CONCEPT_GROUPS,
    POPULATIONS,
    population_label,
    select_population,
)
from arc_cd.repr_analysis.results import representation_arrays, save_axes, write_json
from arc_cd.repr_analysis.runner import parse_arguments, run_analysis
from arc_cd.repr_analysis.spec_metadata import (
    _field_inventory,
    annotate_program,
    validate_saved_metadata,
)
from tests.test_family_analysis import metadata_bundle as _metadata_bundle
from tests.test_prompt_feature_extension import _config, _write_concept
from tests.test_prompt_features import build_records

metadata_bundle = _metadata_bundle


@pytest.fixture
def population_sources(tmp_path: Path) -> tuple[Path, Path]:
    """Write matching combined and original predicate bundles with varied states.

    Args:
        tmp_path (Path): Isolated source and output root.

    Returns:
        tuple[Path, Path]: Combined and predicate-only feature directories.
    """
    dataset = tmp_path / "ConceptARC-100-10"
    dataset.mkdir()
    concepts = sorted(c for group in CONCEPT_GROUPS.values() for c in group)
    for index, concept in enumerate(concepts):
        _write_concept(dataset, concept, index % 8 + 1)
        for path in list((dataset / concept).glob("*.json")):
            (
                path.parent / path.name.replace(f"{concept}1_", f"{concept}2_")
            ).write_bytes(path.read_bytes())
    records, _ = build_records(dataset)
    config = _config(dataset)
    rng = np.random.default_rng(219)
    states = [
        {
            position: (
                rng.normal(size=(2, 4))
                + (5 if record["concept"] in CONCEPT_GROUPS["action"] else 0)
            ).astype(np.float16)
            for position in SUPPORT_POSITIONS
        }
        for record in records
    ]
    complete, predicate = tmp_path / "features_all", tmp_path / "features_predicate"
    complete.mkdir()
    write_prompt_feature_run(complete, copy.deepcopy(records), iter(states), config)
    selected = [
        i for i, r in enumerate(records) if r["concept"] in CONCEPT_GROUPS["predicate"]
    ]
    old_config = copy.deepcopy(config)
    old_config["extraction"]["concepts"] = list(CONCEPT_GROUPS["predicate"])
    old_config["dataset"]["task_hashes"] = {
        records[i]["task_id"]: config["dataset"]["task_hashes"][records[i]["task_id"]]
        for i in selected
    }
    old_config["num_records"] = len(selected)
    predicate.mkdir()
    write_prompt_feature_run(
        predicate,
        [copy.deepcopy(records[i]) for i in selected],
        iter([states[i] for i in selected]),
        old_config,
    )
    return complete, predicate


@pytest.fixture
def feature_bundle(population_sources: tuple[Path, Path]) -> Path:
    """Supply the combined bundle to the existing frozen-metadata fixture.

    Args:
        population_sources (tuple[Path, Path]): Combined and predicate bundles.

    Returns:
        Path: Combined feature directory.
    """
    return population_sources[0]


def _arguments(feature_dir: Path, output: Path, mode: str) -> list[str]:
    """Request one position and layer for the selected population.

    Args:
        feature_dir (Path): Complete source features.
        output (Path): Analysis output root.
        mode (str): Population to select.

    Returns:
        list[str]: Shared runner arguments.
    """
    return [
        "--feature_dir",
        str(feature_dir),
        "--output_root",
        str(output),
        "--population",
        mode,
        "--positions",
        "S2",
        "--layers",
        "2",
        "--num_threads",
        "1",
    ]


def _assert_numerical_outputs_equal(left: Path, right: Path) -> None:
    """Compare the numerical artifacts for two equivalent task populations.

    Args:
        left (Path): First published analysis.
        right (Path): Equivalent analysis from independent source grouping.
    """
    files = sorted(
        p.relative_to(left) for p in left.rglob("*") if p.suffix in (".npz", ".npy")
    )
    assert files
    assert files == sorted(
        p.relative_to(right) for p in right.rglob("*") if p.suffix in (".npz", ".npy")
    )
    for relative in files:
        if relative.suffix == ".npy":
            np.testing.assert_array_equal(
                np.load(left / relative),
                np.load(right / relative),
                err_msg=str(relative),
            )
        else:
            with np.load(left / relative) as a, np.load(right / relative) as b:
                assert a.files == b.files
                for key in a.files:
                    np.testing.assert_array_equal(
                        a[key], b[key], err_msg=f"{relative}: {key}"
                    )


def test_population_partition_and_identity(
    population_sources: tuple[Path, Path],
) -> None:
    """Groups partition the source, preserve order, and map local rows to archives.

    Args:
        population_sources (tuple[Path, Path]): Combined and predicate bundles.
    """
    bundle = load_bundle(population_sources[0])
    before = copy.deepcopy(bundle)
    selected = {mode: select_population(bundle, mode) for mode in POPULATIONS}
    predicate, action, whole = (selected[mode] for mode in POPULATIONS)
    assert not set(predicate.task_ids) & set(action.task_ids)
    assert set(predicate.task_ids) | set(action.task_ids) == set(whole.task_ids)
    for mode, population in selected.items():
        n_concepts = 16 if mode == "all" else 8
        assert population.concepts == [
            c for c in bundle.concepts if mode == "all" or c in CONCEPT_GROUPS[mode]
        ]
        assert population.source_rows.shape == (n_concepts * 2, 8)
        assert set(population.labels) == set(range(n_concepts))
        np.testing.assert_array_equal(
            population.record_indices[population.rows], population.source_rows
        )
        assert sorted(population.rows.ravel()) == list(range(n_concepts * 16))
        for parent, rows in enumerate(population.rows):
            for version, row in enumerate(rows):
                record = population.records[row]
                assert (
                    record["metadata"]["parent_task_id"] == population.task_ids[parent]
                )
                assert record["metadata"]["support_set_id"] == f"{version + 1:03d}"
                assert (
                    record["concept"] == population.concepts[population.labels[parent]]
                )
                assert record is bundle.records[population.source_rows[parent, version]]
    np.testing.assert_array_equal(whole.rows, bundle.rows)
    assert whole.records == bundle.records
    assert bundle.config == before.config and bundle.records == before.records
    np.testing.assert_array_equal(bundle.labels, before.labels)
    np.testing.assert_array_equal(bundle.rows, before.rows)
    old = load_bundle(population_sources[1])
    assert select_population(old, "predicate").task_ids == predicate.task_ids
    assert select_population(old).task_ids == predicate.task_ids
    with pytest.raises(ValueError, match="missing concepts"):
        select_population(old, "action")
    with pytest.raises(ValueError, match="Unknown population"):
        select_population(bundle, "typo")
    unknown = replace(bundle, concepts=[*bundle.concepts[:-1], "Unrecognized"])
    with pytest.raises(ValueError, match="unrecognized concepts"):
        select_population(unknown, "action")
    assert select_population(unknown).concepts[-1] == "Unrecognized"


def test_reordered_source_axes(population_sources: tuple[Path, Path]) -> None:
    """Local axes remain correct when stored labels and manifest rows are noncanonical.

    Args:
        population_sources (tuple[Path, Path]): Combined and predicate bundles.
    """
    bundle = load_bundle(population_sources[0])
    reordered = replace(
        bundle,
        concepts=bundle.concepts[::-1],
        labels=15 - bundle.labels,
        records=bundle.records[::-1],
        rows=len(bundle.records) - 1 - bundle.rows,
    )
    selected = select_population(reordered, "predicate")
    assert selected.concepts == list(CONCEPT_GROUPS["predicate"])[::-1]
    for parent, rows in enumerate(selected.rows):
        for version, row in enumerate(rows):
            record = selected.records[row]
            assert record["metadata"]["parent_task_id"] == selected.task_ids[parent]
            assert record["concept"] == selected.concepts[selected.labels[parent]]
            assert record is reordered.records[selected.source_rows[parent, version]]


@pytest.mark.parametrize("mode", POPULATIONS)
@pytest.mark.parametrize("analysis", ["cosine_distance"])
def test_population_runner(
    population_sources: tuple[Path, Path],
    tmp_path: Path,
    mode: str,
    analysis: str,
) -> None:
    """Publish selected analyses and reproduce the original predicate-only results.

    Args:
        population_sources (tuple[Path, Path]): Combined and predicate bundles.
        tmp_path (Path): Isolated output root.
        mode (str): Population selector.
        analysis (str): Analysis to exercise end to end.
    """
    source, old = population_sources
    directory = run_analysis(analysis, _arguments(source, tmp_path / "out", mode))
    assert directory.parts[-2:] == (mode, analysis)
    bundle = load_bundle(source)
    population = select_population(bundle, mode)
    axes = json.loads((directory / "axes.json").read_text())
    config = json.loads((directory / "run_config.json").read_text())
    assert axes["population"] == config["population"] == population.snapshot()
    assert config["extraction_config"] == bundle.config
    assert axes["task_ids"] == population.task_ids
    assert axes["concepts"] == population.concepts
    assert axes["source_parent_indices"] == population.parent_indices.tolist()
    assert axes["source_version_rows"] == population.source_rows.tolist()
    assert axes["version_rows"] == population.rows.tolist()
    assert axes["source_record_indices"] == population.record_indices.tolist()
    assert list((directory / "figures").glob("*.png"))
    raw = bundle.load_position("S2")
    mean, _, tasks = center_versions(raw[:, 1], population.source_rows)
    np.testing.assert_array_equal(
        np.load(directory / "S2/grand_means.npy")[0], mean.astype(np.float32)
    )
    np.testing.assert_array_equal(
        np.load(directory / "S2/task_means.npy")[:, 0], tasks.astype(np.float32)
    )
    if mode == "predicate":
        original = run_analysis(analysis, _arguments(old, tmp_path / "original", "all"))
        _assert_numerical_outputs_equal(directory, original)


@pytest.mark.parametrize("mode", POPULATIONS)
@pytest.mark.parametrize("analysis", ["cosine_distance"])
def test_population_metadata_runner(
    feature_bundle: Path,
    metadata_bundle: Path,
    tmp_path: Path,
    mode: str,
    analysis: str,
) -> None:
    """Use selected parents for metadata axes, coverage, and reference counts.

    Args:
        feature_bundle (Path): Combined support features.
        metadata_bundle (Path): Frozen full-source annotations.
        tmp_path (Path): Output root.
        mode (str): Population selector.
        analysis (str): Metadata analysis name.
    """
    args = _arguments(feature_bundle, tmp_path / "out", mode) + [
        "--metadata_dir",
        str(metadata_bundle),
        "--min_family_tasks",
        "3",
    ]
    directory = run_analysis(analysis, args)
    assert directory.parts[-3:] == (mode, "metadata", analysis)
    source = load_bundle(feature_bundle)
    population = select_population(source, mode)
    config = json.loads((directory / "run_config.json").read_text())
    axes = json.loads((directory / "axes.json").read_text())
    saved = json.loads((directory / "task_metadata.json").read_text())
    original = json.loads((metadata_bundle / "task_metadata.json").read_text())
    assert saved["records"] == [
        original["records"][i] for i in population.parent_indices
    ]
    validate_saved_metadata(saved, saved["taxonomy"])
    assert axes["task_ids"] == population.task_ids
    memberships = np.load(directory / "memberships.npy")
    assert memberships.shape[0] == len(population.task_ids)
    assert axes["family_sizes"] == memberships.sum(axis=0).tolist()
    assert axes["headline_eligible"] == [
        included and (size >= 3 or role == "concept")
        for included, size, role in zip(
            axes["comparison_included"], axes["family_sizes"], axes["family_roles"]
        )
    ]
    assert set(config["coverage"]) == {"concept", "action", "layout", "selector"}
    assert all(
        c["total_parents"] == len(population.task_ids)
        for c in config["coverage"].values()
    )
    assert list((directory / "figures").rglob("*.png"))
    with np.load(directory / "S2/layer_2.npz") as result:
        assert result["scores"].shape[1] == len(population.task_ids)


@pytest.mark.parametrize("metadata", [False, True])
def test_excluded_states_cannot_change_results(
    feature_bundle: Path,
    metadata_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: bool,
) -> None:
    """Excluded but valid states cannot influence centering or separation scores.

    Args:
        feature_bundle (Path): Combined feature bundle to perturb.
        metadata_bundle (Path): Frozen metadata whose feature digest is refreshed.
        tmp_path (Path): Analysis output root.
        monkeypatch (pytest.MonkeyPatch): Suppress duplicate plot rendering.
        metadata (bool): Exercise the family-analysis branch as well.
    """
    monkeypatch.setattr(visualization, "plot_analysis", lambda *args, **kwargs: None)
    args = _arguments(feature_bundle, tmp_path / "out", "predicate")
    if metadata:
        args += ["--metadata_dir", str(metadata_bundle)]
    before = run_analysis("cosine_distance", args + ["--run_name", "before"])
    bundle = load_bundle(feature_bundle)
    action = select_population(bundle, "action")
    path = feature_bundle / "S2.npz"
    arrays = load_feature_archive(path)
    arrays["x"][action.record_indices] += np.float16(500)
    np.savez_compressed(path, **arrays)
    config_path = metadata_bundle / "run_config.json"
    config = json.loads(config_path.read_text())
    config["feature_fingerprints"]["S2.npz"] = fingerprint(path)
    write_json(config_path, config)
    after = run_analysis("cosine_distance", args + ["--run_name", "after"])
    _assert_numerical_outputs_equal(before, after)


@pytest.mark.parametrize(
    "corruption", ["annotations", "axes", "fingerprint", "feature_values"]
)
def test_excluded_corruption_is_still_rejected(
    feature_bundle: Path,
    metadata_bundle: Path,
    tmp_path: Path,
    corruption: str,
) -> None:
    """Selection cannot conceal invalid source metadata or excluded feature rows.

    Args:
        feature_bundle (Path): Combined features.
        metadata_bundle (Path): Frozen metadata snapshot to corrupt.
        tmp_path (Path): Output root.
        corruption (str): Full-source validation failure to introduce.
    """
    bundle = load_bundle(feature_bundle)
    action = select_population(bundle, "action")
    parent = int(action.parent_indices[0])
    if corruption == "feature_values":
        path = feature_bundle / "S2.npz"
        values = load_feature_archive(path)
        values["x"][action.source_rows[0, 0], 0, 0] = np.nan
        np.savez_compressed(path, **values)
        with pytest.raises(ValueError, match="feature tensor"):
            run_analysis(
                "cosine_distance",
                _arguments(feature_bundle, tmp_path / "out", "predicate"),
            )
        assert not list((tmp_path / "out").rglob("metrics.npz"))
        return
    filename = {
        "annotations": "task_metadata.json",
        "axes": "axes.json",
        "fingerprint": "run_config.json",
    }[corruption]
    path = metadata_bundle / filename
    data = json.loads(path.read_text())
    if corruption == "annotations":
        data["records"][parent]["families"]["action"] = ["invalid_tag"]
    elif corruption == "axes":
        data["version_rows"][parent][0] = -1
    else:
        data["spec_fingerprints"][f"{bundle.task_ids[parent]}.json"] = {
            "sha256": "wrong",
            "bytes": 1,
        }
    write_json(path, data)
    with pytest.raises(ValueError):
        load_family_data(
            metadata_bundle,
            bundle,
            ["action"],
            {},
            population=select_population(bundle, "predicate"),
        )


def test_missing_tags_preserve_selected_tasks(
    feature_bundle: Path, metadata_bundle: Path, tmp_path: Path
) -> None:
    """Unknown operations remain unannotated without removing their selected parents.

    Args:
        feature_bundle (Path): Combined features.
        metadata_bundle (Path): Frozen annotations to leave partially unmapped.
        tmp_path (Path): Isolated analysis output root.
    """
    bundle = load_bundle(feature_bundle)
    population = select_population(bundle, "predicate")
    path = metadata_bundle / "task_metadata.json"
    data = json.loads(path.read_text())
    for i in population.parent_indices:
        record = data["records"][i]
        record["program"]["action"] = {"op": "unmapped_population_test"}
        record.update(
            annotate_program(
                record["program"], data["taxonomy"], concept_name=record["concept_name"]
            )
        )
    data["fields"] = _field_inventory(data["records"], data["taxonomy"])
    write_json(path, data)
    selected = load_family_data(
        metadata_bundle,
        bundle,
        ["action", "layout", "selector"],
        {},
        population=population,
    )
    assert len(selected.metadata["records"]) == len(population.task_ids)
    assert not any(role == "action" for role in selected.roles)
    assert "action" in selected.role_order
    validate_saved_metadata(selected.metadata, selected.taxonomy)
    args = _arguments(feature_bundle, tmp_path / "out", "predicate")
    directory = run_analysis(
        "cosine_distance", args + ["--metadata_dir", str(metadata_bundle)]
    )
    coverage = json.loads((directory / "coverage.json").read_text())
    assert coverage["action"]["annotated_parents"] == 0
    assert coverage["action"]["total_parents"] == len(population.task_ids)


def test_population_cli_and_legacy_labels() -> None:
    """Keep default all and expose the same population selector for embeddings."""
    assert (
        parse_arguments("cosine_distance", ["--feature_dir", "unused"]).population
        == "all"
    )
    assert (
        parse_arguments(
            "dimensionality_reduction",
            ["--feature_dir", "unused", "--population", "action"],
        ).population
        == "action"
    )
    assert population_label({}) == ""


@pytest.mark.parametrize("writer", ["axes", "representations"])
def test_result_writers_reject_another_source(
    population_sources: tuple[Path, Path],
    tmp_path: Path,
    writer: str,
) -> None:
    """Reject mixed source contracts before publishing axes or allocating arrays.

    Args:
        population_sources (tuple[Path, Path]): Combined and predicate bundles.
        tmp_path (Path): Isolated output directory.
        writer (str): Shared result helper to exercise.
    """
    complete, predicate = (load_bundle(path) for path in population_sources)
    population = select_population(complete, "predicate")
    destination = tmp_path / "results"
    destination.mkdir()
    with ExitStack() as resources:
        with pytest.raises(ValueError, match="Population must refer"):
            if writer == "axes":
                save_axes(destination, predicate, ["S2"], (), population=population)
            else:
                representation_arrays(
                    destination, "S2", predicate, 4, resources, population=population
                )
    assert not list(destination.iterdir())


def test_merge_then_select_populations(population_sources: tuple[Path, Path]) -> None:
    """Extend the original bundle and recover identical features in every population.

    Args:
        population_sources (tuple[Path, Path]): Combined reference and predicate bundle.
    """
    complete, base = population_sources
    reference = load_bundle(complete)
    original = load_bundle(base)
    records = _load_shard_records(complete)
    features = {
        position: reference.load_position(position) for position in SUPPORT_POSITIONS
    }
    shards = complete.parent / "shards"
    for concept in CONCEPT_GROUPS["action"]:
        indices = [
            i for i, record in enumerate(records) if record["concept"] == concept
        ]
        directory = shards / concept
        directory.mkdir(parents=True)
        write_prompt_feature_run(
            directory,
            [copy.deepcopy(records[i]) for i in indices],
            (
                {position: features[position][i] for position in SUPPORT_POSITIONS}
                for i in indices
            ),
            {
                **reference.config,
                "selection": {"concept": concept},
                "num_records": len(indices),
            },
        )
    destination = merge_prompt_feature_shards(
        complete.parent / "ConceptARC-100-10",
        shards,
        complete.parent / "merged",
        base_feature_dir=base,
    )
    merged = load_bundle(destination)
    assert _load_shard_records(destination) == records
    for position in SUPPORT_POSITIONS:
        actual = load_feature_archive(destination / f"{position}.npz")
        expected = load_feature_archive(complete / f"{position}.npz")
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        for mode in POPULATIONS:
            selected = select_population(merged, mode)
            expected_population = select_population(reference, mode)
            assert selected.task_ids == expected_population.task_ids
            np.testing.assert_array_equal(selected.labels, expected_population.labels)
            np.testing.assert_array_equal(
                actual["x"][selected.source_rows],
                features[position][expected_population.source_rows],
            )
            if mode == "predicate":
                assert selected.task_ids == original.task_ids
                np.testing.assert_array_equal(
                    actual["x"][selected.source_rows],
                    original.load_position(position)[original.rows],
                )


@pytest.mark.parametrize("mode", POPULATIONS)
def test_population_embedding_explorer_and_analysis_round_trip(
    population_sources: tuple[Path, Path],
    metadata_bundle: Path,
    tmp_path: Path,
    mode: str,
) -> None:
    """Publish population maps and feed their metadata to every numerical analysis.

    Args:
        population_sources (tuple[Path, Path]): Combined and predicate-only features.
        metadata_bundle (Path): Full-source metadata and matching spec fixture.
        tmp_path (Path): Isolated output directory.
        mode (str): Predicate, action, or combined population.
    """
    source, old = population_sources
    args = _arguments(source, tmp_path / "out", mode)
    embedding_args = args + ["--methods", "pca", "umap"]
    embedding = run_embedding(embedding_args)
    assert embedding.parts[-2:] == (mode, "dimensionality_reduction")
    bundle = load_bundle(source)
    population = select_population(bundle, mode)
    display = load_explorer_data(embedding)
    assert display["axes"]["population"] == population.snapshot()
    assert display["axes"]["task_ids"] == population.task_ids
    assert len(display["supports"]) == population.source_rows.size
    assert display["axes"]["source_version_rows"] == population.source_rows.tolist()
    for row, record in enumerate(population.records):
        assert display["supports"][row]["case_id"] == record["case_id"]
    mean, _, tasks = center_versions(
        bundle.load_position("S2")[:, 1], population.source_rows
    )
    np.testing.assert_array_equal(
        np.load(embedding / "S2/grand_means.npy")[0], mean.astype(np.float32)
    )
    np.testing.assert_array_equal(
        np.load(embedding / "S2/task_means.npy")[:, 0], tasks.astype(np.float32)
    )
    coordinates = np.asarray(display["frames"][0]["coordinates"])
    assert coordinates.shape == (2, len(population.task_ids), 2)
    assert np.isfinite(coordinates).all()
    if mode == "predicate":
        original_args = _arguments(old, tmp_path / "original", "all")
        original = run_embedding(original_args + ["--methods", "pca", "umap"])
        _assert_numerical_outputs_equal(embedding, original)
    source_config = json.loads((metadata_bundle / "run_config.json").read_text())
    explorer = run_metadata_explorer(
        [
            "--analysis_dir",
            str(embedding),
            "--spec_dir",
            source_config["arguments"]["spec_dir"],
        ]
    )
    assert explorer.parts[-2:] == (mode, "metadata_explorer")
    exported = load_explorer_data(explorer)
    assert exported["axes"] == display["axes"]
    assert exported["supports"] == display["supports"]
    assert exported["frames"] == display["frames"]
    assert [
        record["task_id"] for record in exported["metadata"]["records"]
    ] == population.task_ids
    config = json.loads((explorer / "run_config.json").read_text())
    assert config["population"] == population.snapshot()
    html = (explorer / "explorer.html").read_text()
    write_explorer(explorer)
    assert (explorer / "explorer.html").read_text() == html
    roles = ["action", "layout", "selector"]
    from_full = load_family_data(
        metadata_bundle, bundle, roles, {}, population=population
    )
    from_selected = load_family_data(explorer, bundle, roles, {}, population=population)
    assert from_selected.metadata == from_full.metadata
    assert from_selected.ids == from_full.ids
    np.testing.assert_array_equal(from_selected.membership, from_full.membership)
    output = run_analysis(
        "cosine_distance",
        args + ["--metadata_dir", str(explorer), "--min_family_tasks", "2"],
    )
    axes = json.loads((output / "axes.json").read_text())
    assert axes["task_ids"] == population.task_ids
    np.testing.assert_array_equal(
        np.load(output / "memberships.npy"), from_full.membership
    )
    assert list((output / "figures").rglob("*.png"))
    if mode != "all":
        with pytest.raises(ValueError, match="does not cover the requested population"):
            load_family_data(
                explorer, bundle, roles, {}, population=select_population(bundle, "all")
            )


@pytest.mark.parametrize("corruption", ["snapshot", "parents", "versions", "records"])
def test_selected_metadata_rejects_corrupt_source_mapping(
    feature_bundle: Path,
    metadata_bundle: Path,
    corruption: str,
) -> None:
    """Reject inconsistent selected metadata even when local parent IDs still match.

    Args:
        feature_bundle (Path): Combined feature bundle.
        metadata_bundle (Path): Frozen metadata to restrict and then corrupt.
        corruption (str): Source provenance field to alter.
    """
    bundle = load_bundle(feature_bundle)
    population = select_population(bundle, "action")
    selected = load_family_data(
        metadata_bundle, bundle, ["action"], {}, population=population
    )
    write_json(metadata_bundle / "task_metadata.json", selected.metadata)
    save_axes(metadata_bundle, bundle, ["S1"], (), population=population)
    config_path = metadata_bundle / "run_config.json"
    config = json.loads(config_path.read_text())
    config["population"] = population.snapshot()
    write_json(config_path, config)
    load_family_data(metadata_bundle, bundle, ["action"], {}, population=population)
    axes_path = metadata_bundle / "axes.json"
    axes = json.loads(axes_path.read_text())
    if corruption == "snapshot":
        axes["population"]["name"] = "predicate"
    elif corruption == "parents":
        axes["source_parent_indices"].reverse()
    elif corruption == "versions":
        axes["source_version_rows"][0][0] = -1
    else:
        axes["source_record_indices"].reverse()
    write_json(axes_path, axes)
    with pytest.raises(ValueError, match="population|axis mismatch"):
        load_family_data(metadata_bundle, bundle, ["action"], {}, population=population)
