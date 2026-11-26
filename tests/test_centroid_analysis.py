"""Centroid alignment, input failures, numerical parity, and offline plotting."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from arc_cd.repr_analysis import visualization
from arc_cd.repr_analysis.centroid_data import load_task_centroids
from arc_cd.repr_analysis.data import load_bundle
from arc_cd.repr_analysis.results import write_json
from arc_cd.repr_analysis.runner import parse_arguments, run_analysis
from tests.test_repr_population import feature_bundle as _feature_bundle
from tests.test_repr_population import metadata_bundle as _metadata_bundle
from tests.test_repr_population import population_sources as _population_sources

feature_bundle = _feature_bundle
metadata_bundle = _metadata_bundle
population_sources = _population_sources


@pytest.fixture
def centroid_file(feature_bundle: Path, tmp_path: Path) -> Path:
    """Save shuffled, aliased centroids of the existing S3/layer-2 fixture.

    Args:
        feature_bundle (Path): Complete sixteen-concept feature fixture.
        tmp_path (Path): Temporary archive directory.

    Returns:
        Path: NPZ with genuine means of eight distinct support vectors per task.
    """
    bundle = load_bundle(feature_bundle)
    raw = bundle.load_position("S3")[:, bundle.layers.index(2)].astype(np.float64)
    means = raw[bundle.rows].mean(axis=1)
    aliases = {"CleanUp": "Cleanup", "ExtractObjects": "Extract"}
    ids, concepts, rules = [], [], []
    for task in bundle.task_ids:
        concept, rule = task.split("/")
        replacement = aliases.get(concept, concept)
        rule = replacement + rule[len(concept) :]
        ids.append(f"{replacement}/{rule}")
        concepts.append(replacement)
        rules.append(rule)
    order = np.random.default_rng(42).permutation(len(ids))
    path = tmp_path / "centroids.npz"
    np.savez_compressed(
        path,
        embeddings=means[order],
        source_task_id=np.asarray(ids)[order],
        concept=np.asarray(concepts)[order],
        source_rule=np.asarray(rules)[order],
        instance_count=np.full(len(ids), 8, dtype=np.int64),
    )
    return path


def _arguments(source: Path, metadata: Path, output: Path) -> list[str]:
    """Build centroid CLI arguments with small-fixture family thresholds.

    Args:
        source (Path): Centroid NPZ.
        metadata (Path): Frozen annotations.
        output (Path): Experiment output root.

    Returns:
        list[str]: Valid centroid-analysis arguments.
    """
    return [
        "--centroid_file",
        str(source),
        "--metadata_dir",
        str(metadata),
        "--model_name",
        "varc",
        "--output_root",
        str(output),
        "--min_family_tasks",
        "2",
        "--num_threads",
        "1",
    ]


@pytest.mark.parametrize("population", ["all", "predicate", "action"])
@pytest.mark.parametrize("distance_metric", ["cosine", "euclidean"])
def test_centroid_scores_match_support_pipeline(
    feature_bundle: Path,
    metadata_bundle: Path,
    centroid_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    population: str,
    distance_metric: str,
) -> None:
    """Match every score and reason after shuffling, aliasing, and population selection.

    Args:
        feature_bundle (Path): Original eight-version feature bundle.
        metadata_bundle (Path): Frozen metadata shared by both inputs.
        centroid_file (Path): Shuffled and aliased task means.
        tmp_path (Path): Output root.
        monkeypatch (pytest.MonkeyPatch): Skip rendering during numerical checks.
        population (str): Selected population to compare.
        distance_metric (str): Distance used for separation.
    """
    monkeypatch.setattr(visualization, "plot_analysis", lambda *args, **kwargs: None)
    common = [
        "--distance_metric",
        distance_metric,
        "--metadata_dir",
        str(metadata_bundle),
        "--output_root",
        str(tmp_path / "out"),
        "--population",
        population,
        "--min_family_tasks",
        "2",
        "--num_threads",
        "1",
    ]
    reference = run_analysis(
        "cosine_distance",
        [
            "--feature_dir",
            str(feature_bundle),
            "--positions",
            "S3",
            "--layers",
            "2",
            *common,
        ],
    )
    actual = run_analysis(
        "cosine_distance",
        ["--centroid_file", str(centroid_file), "--model_name", "varc", *common],
    )
    with (
        np.load(reference / "metrics.npz") as expected,
        np.load(actual / "metrics.npz") as observed,
    ):
        assert set(expected.files) == set(observed.files)
        for key in expected.files:
            np.testing.assert_allclose(observed[key], expected[key][0, 0], err_msg=key)
    with (
        np.load(reference / "S3/layer_2.npz") as expected,
        np.load(actual / "scores.npz") as observed,
    ):
        for key in expected.files:
            np.testing.assert_allclose(observed[key], expected[key], err_msg=key)
    np.testing.assert_allclose(
        np.load(actual / "task_means.npy"),
        np.load(reference / "S3/task_means.npy")[:, 0],
        atol=1e-6,
    )
    axes = json.loads((actual / "axes.json").read_text())
    expected_axes = json.loads((reference / "axes.json").read_text())
    assert actual.name == reference.name == f"{distance_metric}_distance"
    assert (
        axes["distance_metric"] == expected_axes["distance_metric"] == distance_metric
    )
    if distance_metric == "euclidean":
        assert "undefined_anchor_distance" in axes["reason_bits"]
        assert "undefined_anchor_cosine" not in axes["reason_bits"]
    assert axes["task_ids"] == expected_axes["task_ids"]
    assert axes["family_ids"] == expected_axes["family_ids"]
    assert "positions" not in axes and "layers" not in axes
    assert len(axes["task_ids"]) == (32 if population == "all" else 16)
    with np.load(centroid_file) as source:
        assert (
            source["source_task_id"][axes["source_rows"]].tolist()
            == axes["source_task_ids"]
        )
    if population != "predicate":
        assert any(task.startswith("Cleanup/") for task in axes["source_task_ids"])
        assert any(task.startswith("Extract/") for task in axes["source_task_ids"])
    with pytest.raises(FileExistsError):
        run_analysis(
            "cosine_distance",
            ["--centroid_file", str(centroid_file), "--model_name", "varc", *common],
        )


@pytest.mark.parametrize(
    "problem",
    ["nonfinite", "counts", "duplicate", "identity", "missing_task", "missing_array"],
)
def test_centroid_input_failures(
    centroid_file: Path, metadata_bundle: Path, problem: str
) -> None:
    """Reject invalid vectors, provenance, and incomplete task sets.

    Args:
        centroid_file (Path): Valid archive to corrupt.
        metadata_bundle (Path): Valid frozen metadata.
        problem (str): Corruption to introduce.
    """
    with np.load(centroid_file) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if problem == "nonfinite":
        arrays["embeddings"][0, 0] = np.nan
    elif problem == "counts":
        arrays["instance_count"][0] = 7
    elif problem == "duplicate":
        for key in ("source_task_id", "concept", "source_rule"):
            arrays[key][1] = arrays[key][0]
    elif problem == "identity":
        arrays["concept"][0] = "Center"
        arrays["source_task_id"][0] = "wrong"
    elif problem == "missing_task":
        arrays = {key: values[1:] for key, values in arrays.items()}
    else:
        del arrays["source_rule"]
    np.savez_compressed(centroid_file, **arrays)
    with pytest.raises(ValueError):
        load_task_centroids(centroid_file, metadata_bundle, ["action"])


@pytest.mark.parametrize("problem", ["taxonomy", "concept_axis", "spec_fingerprint"])
def test_centroid_metadata_failures(
    centroid_file: Path, metadata_bundle: Path, problem: str
) -> None:
    """Keep frozen-taxonomy and task-provenance checks for external embeddings.

    Args:
        centroid_file (Path): Valid centroid archive.
        metadata_bundle (Path): Metadata to corrupt.
        problem (str): Snapshot inconsistency to introduce.
    """
    filename = {
        "taxonomy": "run_config.json",
        "concept_axis": "axes.json",
        "spec_fingerprint": "task_metadata.json",
    }[problem]
    path = metadata_bundle / filename
    payload = json.loads(path.read_text())
    if problem == "taxonomy":
        payload["taxonomy_fingerprint"]["sha256"] = "invalid"
    elif problem == "concept_axis":
        payload["labels"][0] = 1
    else:
        payload["records"][0]["source_fingerprint"]["sha256"] = "invalid"
    write_json(path, payload)
    with pytest.raises(ValueError):
        load_task_centroids(centroid_file, metadata_bundle, ["action"])


@pytest.mark.parametrize("distance_metric", ["cosine", "euclidean"])
def test_centroid_plots_reproduce_without_inputs(
    centroid_file: Path, metadata_bundle: Path, tmp_path: Path, distance_metric: str
) -> None:
    """Export real PDFs and replot solely from the published numerical artifacts.

    Args:
        centroid_file (Path): Centroid input to remove after analysis.
        metadata_bundle (Path): Annotation input to remove after analysis.
        tmp_path (Path): Output root.
        distance_metric (str): Distance used for separation.
    """
    args = _arguments(centroid_file, metadata_bundle, tmp_path / "out")
    directory = run_analysis(
        "cosine_distance",
        args
        + [
            "--comparison_scopes",
            "global",
            "--family_roles",
            "action",
            "--distance_metric",
            distance_metric,
        ],
    )
    figures = sorted(
        path.relative_to(directory) for path in (directory / "figures").rglob("*.png")
    )
    assert len(figures) == 3
    for name in figures:
        assert (directory / name.with_suffix(".pdf")).read_bytes().startswith(b"%PDF-")
    with np.load(directory / "metrics.npz") as archive:
        before = archive["macro"].copy()
    centroid_file.unlink()
    shutil.rmtree(metadata_bundle)
    shutil.rmtree(directory / "figures")
    visualization.plot_analysis(directory)
    for name in figures:
        assert (directory / name).exists()
        assert (directory / name.with_suffix(".pdf")).exists()
    with np.load(directory / "metrics.npz") as archive:
        np.testing.assert_array_equal(archive["macro"], before)


@pytest.mark.parametrize(
    "extra",
    [
        ["--positions", "S3"],
        ["--layers", "2"],
        ["--feature_dir", "features"],
        ["--model_name", "../varc"],
    ],
)
def test_centroid_cli_rejects_incompatible_options(extra: list[str]) -> None:
    """Reject synthetic axes, ambiguous inputs, and invalid output identifiers.

    Args:
        extra (list[str]): Incompatible options added to valid base arguments.
    """
    with pytest.raises(SystemExit):
        parse_arguments(
            "cosine_distance",
            [
                "--centroid_file",
                "centroids.npz",
                "--metadata_dir",
                "metadata",
                "--model_name",
                "varc",
                *extra,
            ],
        )
