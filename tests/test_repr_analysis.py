"""Numerical references and end-to-end checks for representation analysis."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from arc_cd.probes.prompt_feature_io import write_prompt_feature_run
from arc_cd.repr_analysis.cosine_distance import separation_probability
from arc_cd.repr_analysis.data import center_versions, load_bundle
from arc_cd.repr_analysis.embedding import (
    representative_versions,
    within_task_dispersion,
)
from arc_cd.repr_analysis.geometry import (
    DistanceMetric,
    separation_similarity,
    summarize,
)
from arc_cd.repr_analysis.runner import parse_arguments, run_analysis
from arc_cd.repr_analysis.visualization import plot_analysis


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a direct cosine reference with explicit undefined zero vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        float: Cosine, or NaN for a zero norm.
    """
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / norm, -1, 1)) if norm > 0 else np.nan


def explicit_separation(
    means: np.ndarray,
    labels: np.ndarray,
    distance_metric: DistanceMetric = "cosine",
) -> np.ndarray:
    """Enumerate all sampled triplets as an independent numerical reference.

    Args:
        means (np.ndarray): Recentered original task means.
        labels (np.ndarray): Concept labels.
        distance_metric (DistanceMetric): Use direct cosines or negative squared L2.

    Returns:
        np.ndarray: Strict and tie-adjusted anchor probabilities.
    """
    result = np.full((len(labels), 2), np.nan)
    for anchor, label in enumerate(labels):
        same = [
            index
            for index, other in enumerate(labels)
            if other == label and index != anchor
        ]
        other = np.flatnonzero(labels != label)
        pairs = [
            (
                (cosine(means[anchor], means[j]), cosine(means[anchor], means[k]))
                if distance_metric == "cosine"
                else (
                    -np.sum((means[anchor] - means[j]) ** 2),
                    -np.sum((means[anchor] - means[k]) ** 2),
                )
            )
            for j in same
            for k in other
        ]
        pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
        if pairs:
            result[anchor] = np.mean(
                [(a > b, float(a > b) + 0.5 * (a == b)) for a, b in pairs], axis=0
            )
    return result


@pytest.mark.parametrize("distance_metric", ["cosine", "euclidean"])
def test_separation_zero_vectors_and_ties(distance_metric: DistanceMetric) -> None:
    """Match explicit triplets with zero anchors, and duplicate points.

    Args:
        distance_metric (DistanceMetric): Metric under test.
    """
    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [2.0, 0.0],
        ]
    )
    labels = np.repeat([0, 1], 4)
    actual, details = separation_probability(
        separation_similarity(vectors, distance_metric), labels
    )
    expected = explicit_separation(vectors, labels, distance_metric)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(
        details["tie_probability"], 2 * (expected[:, 1] - expected[:, 0]), atol=1e-14
    )
    for anchor in range(len(vectors)):
        good = (
            np.ones(8, dtype=bool)
            if distance_metric == "euclidean"
            else np.any(vectors != 0, axis=1)
        )
        good[anchor] = False
        denominator = (good & (labels == labels[anchor])).sum() * (
            good & (labels != labels[anchor])
        ).sum()
        if distance_metric == "cosine" and not np.any(vectors[anchor]):
            denominator = 0
        assert details["valid_triplet_comparisons"][anchor] == denominator
    if distance_metric == "euclidean":
        shifted, _ = separation_probability(
            separation_similarity(vectors + [8.0, -16.0], distance_metric),
            labels,
        )
        np.testing.assert_array_equal(shifted, actual)
        assert np.isfinite(actual[[0, 4]]).all()
    else:
        assert np.isnan(actual[[0, 4]]).all()


def test_distance_metric_cli() -> None:
    """Default to cosine and reject unsupported metrics or unrelated analyses."""
    assert (
        parse_arguments("cosine_distance", ["--feature_dir", "unused"]).distance_metric
        == "cosine"
    )
    with pytest.raises(SystemExit):
        parse_arguments(
            "cosine_distance",
            ["--feature_dir", "unused", "--distance_metric", "manhattan"],
        )


def test_hidden_distance_runs_coexist(
    feature_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the metric preserves task means and existing cosine artifacts.

    Args:
        feature_bundle (Path): Eight-version hidden-state fixture.
        tmp_path (Path): Output root.
        monkeypatch (pytest.MonkeyPatch): Skip rendering for this numerical check.
    """
    monkeypatch.setattr(
        "arc_cd.repr_analysis.visualization.plot_analysis", lambda *args, **kwargs: None
    )
    args = [
        "--feature_dir",
        str(feature_bundle),
        "--output_root",
        str(tmp_path / "out"),
        "--positions",
        "S3",
        "--layers",
        "2",
        "--num_threads",
        "1",
    ]
    cosine_dir = run_analysis("cosine_distance", args)
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in cosine_dir.rglob("*")
        if p.is_file()
    }
    euclidean_dir = run_analysis(
        "cosine_distance", args + ["--distance_metric", "euclidean"]
    )
    assert cosine_dir.parent == euclidean_dir.parent
    assert (
        cosine_dir.name == "cosine_distance"
        and euclidean_dir.name == "euclidean_distance"
    )
    for name in ("grand_means.npy", "task_means.npy", "concept_means.npy"):
        np.testing.assert_array_equal(
            np.load(cosine_dir / "S3" / name), np.load(euclidean_dir / "S3" / name)
        )
    assert before == {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in cosine_dir.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        run_analysis("cosine_distance", args + ["--distance_metric", "euclidean"])


def test_centering_dispersion_and_averaging() -> None:
    """Check global centering, the 28 support-set pairs, and mean norms."""
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(32, 7)) * rng.uniform(1, 20, size=(32, 1))
    rows = np.arange(32).reshape(4, 8)[::-1]
    grand, versions, means = center_versions(raw, rows)
    np.testing.assert_allclose(grand, raw.mean(axis=0))
    np.testing.assert_allclose(versions.mean(axis=(0, 1)), 0, atol=1e-14)
    np.testing.assert_allclose(means, raw[rows].mean(axis=1) - grand)
    assert not np.allclose(
        means, (versions / np.linalg.norm(versions, axis=2, keepdims=True)).mean(axis=1)
    )
    dispersion = within_task_dispersion(versions)
    for task, vectors in enumerate(versions):
        expected = np.mean(
            [1 - cosine(vectors[a], vectors[b]) for a, b in combinations(range(8), 2)]
        )
        np.testing.assert_allclose(dispersion[task], expected, atol=1e-13)


def test_summary_requires_valid_tasks_for_each_concept() -> None:
    """A missing concept score makes the macro-average undefined."""
    summary = summarize(np.array([[1.0], [np.nan]]), np.array([0, 1]), 2)
    assert np.isnan(summary["macro"]).all()
    np.testing.assert_array_equal(summary["concept_valid_tasks"].ravel(), [1, 0])


@pytest.fixture
def feature_bundle(tmp_path: Path) -> Path:
    """Write a tiny complete feature bundle with real archive/token contracts.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        Path: Complete synthetic support-feature bundle.
    """
    records = []
    concepts = ["AboveBelow", "Center"]
    tokens = [60, 10, 60, 10, 60]
    for concept in concepts:
        for parent in range(6):
            for version in range(8):
                sample = f"{concept}{parent}_{version:03}"
                task_id = f"{concept}/{sample}"
                records.append(
                    {
                        "case_id": f"{task_id}/support",
                        "task_id": task_id,
                        "sample_id": sample,
                        "concept": concept,
                        "test_index": -1,
                        "source_path": f"/synthetic/{task_id}.json",
                        "prompt": "] ] ]",
                        "prompt_token_ids": tokens,
                        "generated_token_ids": [],
                        "prefix_sha256": hashlib.sha256(
                            ",".join(map(str, tokens)).encode()
                        ).hexdigest(),
                        "selected_token_indices": {
                            f"S{i + 1}": 2 * i for i in range(3)
                        },
                        "selected_token_ids": {f"S{i + 1}": 60 for i in range(3)},
                        "prompt_char_spans": {
                            f"support_{i + 1}_output_closing_bracket": [
                                2 * i,
                                2 * i + 1,
                            ]
                            for i in range(3)
                        },
                        "token_spans": {
                            f"support_{i + 1}_output_closing_bracket": [
                                2 * i,
                                2 * i + 1,
                            ]
                            for i in range(3)
                        },
                        "metadata": {
                            "parent_task_id": f"{concept}/{concept}{parent}",
                            "support_set_id": f"{version:03}",
                            "prompt_only": True,
                        },
                        "support_examples": [
                            {"input": [[i, version % 10]], "output": [[parent % 10, i]]}
                            for i in range(3)
                        ],
                    }
                )
    config = {
        "schema_version": 1,
        "extraction": {
            "concepts": concepts,
            "layers": [0, 2],
            "model_name": "tiny",
            "feature_dtype": "float16",
            "position_definition": "token_containing_final_support_output_closing_bracket",
        },
        "dataset": {
            "id": "concept_arc_100_10",
            "task_hashes": {record["task_id"]: "synthetic" for record in records},
        },
        "selection": {"concept": None},
        "num_records": len(records),
    }
    rng = np.random.default_rng(37)
    features = [
        {
            position: rng.normal(size=(2, 5)).astype(np.float16)
            for position in ("S1", "S2", "S3")
        }
        for _ in records
    ]
    directory = tmp_path / "features"
    directory.mkdir()
    write_prompt_feature_run(directory, records, iter(features), config)
    return directory


def test_bundle_validation(feature_bundle: Path) -> None:
    """Reject missing versions, corrupted row metadata and unextracted layers."""
    bundle = load_bundle(feature_bundle)
    assert bundle.rows.shape == (12, 8)
    assert bundle.load_position("S1").shape == (96, 2, 5)
    with pytest.raises(ValueError, match="Requested layers"):
        load_bundle(feature_bundle, "1")
    archive_path = feature_bundle / "S1.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values["parent_task_ids"][0] = "bad"
    np.savez_compressed(archive_path, **values)
    with pytest.raises(ValueError, match="parent_task_ids"):
        bundle.load_position("S1")
    config_path = feature_bundle / "run_config.json"
    config = json.loads(config_path.read_text())
    config["num_records"] -= 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Missing, duplicate"):
        load_bundle(feature_bundle)


@pytest.mark.parametrize(
    "analysis,distance_metric",
    [
        ("cosine_distance", "cosine"),
        ("cosine_distance", "euclidean"),
    ],
)
def test_runner_artifacts_and_plots(
    feature_bundle: Path,
    tmp_path: Path,
    analysis: str,
    distance_metric: DistanceMetric,
) -> None:
    """Check saved separation scores against explicit triplets and regenerate the plots.

    Args:
        feature_bundle (Path): Eight-version hidden-state fixture.
        tmp_path (Path): Isolated output root.
        analysis (str): Analysis entrypoint.
        distance_metric (DistanceMetric): Separation metric.
    """
    arguments = [
        "--feature_dir",
        str(feature_bundle),
        "--output_root",
        str(tmp_path / "out"),
        "--num_threads",
        "1",
    ]
    arguments.extend(["--distance_metric", distance_metric])
    directory = run_analysis(analysis, arguments)
    axes = json.loads((directory / "axes.json").read_text())
    config = json.loads((directory / "run_config.json").read_text())
    assert directory.name == config["analysis"] == f"{distance_metric}_distance"
    assert axes["distance_metric"] == config["distance_metric"] == distance_metric
    assert config["elapsed_seconds"] > config["elapsed_seconds_before_plotting"] >= 0
    assert axes["layers"] == [0, 2] and len(axes["task_ids"]) == 12
    assert list((directory / "figures").glob("*.png"))
    assert not list((directory / "figures").glob("*.pdf"))
    saved = np.load(directory / "S1/task_means.npy", allow_pickle=False)
    assert saved.shape == (12, 2, 5) and saved.dtype == np.float32
    bundle = load_bundle(feature_bundle)
    raw = bundle.load_position("S1")
    grand, _, means = center_versions(raw[:, 0], bundle.rows)
    np.testing.assert_allclose(saved[:, 0], means, atol=1e-7)
    np.testing.assert_allclose(
        np.load(directory / "S1/grand_means.npy")[0], grand, atol=1e-7
    )
    with np.load(directory / "metrics.npz") as result:
        assert result["task_scores"].shape[:3] == (3, 2, 12)
        for pos_idx, position in enumerate(axes["positions"]):
            raw_position = bundle.load_position(position)
            for layer_idx in range(len(bundle.layers)):
                _, _, task_means = center_versions(
                    raw_position[:, layer_idx], bundle.rows
                )
                np.testing.assert_allclose(
                    result["task_scores"][pos_idx, layer_idx],
                    explicit_separation(task_means, bundle.labels, distance_metric),
                    atol=1e-13,
                )
    with pytest.raises(FileExistsError):
        run_analysis(analysis, arguments)
    existing_pdfs = {
        path: path.read_bytes() for path in (directory / "figures").glob("*.pdf")
    }
    plot_analysis(directory)
    assert {
        path: path.read_bytes() for path in (directory / "figures").glob("*.pdf")
    } == existing_pdfs


@pytest.mark.parametrize(
    "script",
    ["run_dimensionality_reduction", "run_metadata_explorer", "run_cosine_distance"],
)
def test_analysis_cli_without_vllm(script: str) -> None:
    """Start CPU analysis commands without importing the GPU inference engine.

    Args:
        script (str): Representation-analysis entry point to start.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys\n"
            "sys.modules['vllm'] = None\n"
            "sys.argv = [sys.argv[1], '--help']\n"
            "runpy.run_module(sys.argv[0], run_name='__main__')\n",
            f"scripts.repr_analysis.{script}",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_missing_eighth_version(feature_bundle: Path) -> None:
    """Reject a coherent bundle whose last parent has only seven versions.

    Args:
        feature_bundle (Path): Synthetic feature bundle to shorten.
    """
    from arc_cd.probes.prompt_feature_io import (
        _load_shard_records,
        save_prompt_manifest,
    )

    records = _load_shard_records(feature_bundle)
    config = json.loads((feature_bundle / "run_config.json").read_text())
    removed = records.pop()
    config["dataset"]["task_hashes"].pop(removed["task_id"])
    config["num_records"] -= 1
    save_prompt_manifest(feature_bundle, records, config)
    with pytest.raises(ValueError, match="eight distinct version IDs"):
        load_bundle(feature_bundle)


def test_subset_axes_and_other_analysis_directory(
    feature_bundle: Path, tmp_path: Path
) -> None:
    """Select a noninitial position/layer and preserve a sibling analysis.

    Args:
        feature_bundle (Path): Source fixture.
        tmp_path (Path): Output root.
    """
    sibling = (
        tmp_path / "out/concept_arc_100_10/tiny/repr_analysis/subset/all/other_analysis"
    )
    sibling.mkdir(parents=True)
    (sibling / "keep.txt").write_text("existing result")
    result = run_analysis(
        "cosine_distance",
        [
            "--feature_dir",
            str(feature_bundle),
            "--output_root",
            str(tmp_path / "out"),
            "--run_name",
            "subset",
            "--positions",
            "S2",
            "--layers",
            "2",
            "--num_threads",
            "1",
        ],
    )
    axes = json.loads((result / "axes.json").read_text())
    assert axes["positions"] == ["S2"] and axes["layers"] == [2]
    assert (sibling / "keep.txt").read_text() == "existing result"
    with np.load(result / "metrics.npz") as archive:
        assert archive["task_scores"].shape == (1, 1, 12, 2)


def test_anchor_cannot_supply_its_own_positive_reference() -> None:
    """A concept with one parent has no positive comparison for that anchor."""
    means = np.array([[1.0, 1.0], [0.0, 1.0], [-1.0, -1.0]])
    labels = np.array([0, 1, 1])
    separation, counts_detail = separation_probability(
        separation_similarity(means), labels
    )
    assert counts_detail["valid_triplet_comparisons"][0] == 0
    assert np.isnan(separation[0]).all()


def test_write_failure_closes_maps_before_cleanup(
    feature_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close live memory maps and release staging locks after a quota-like error.

    Args:
        feature_bundle (Path): Source fixture.
        tmp_path (Path): Output root.
        monkeypatch (pytest.MonkeyPatch): Temporary flush failure.
    """
    opened = []

    def fail_flush(array: np.memmap) -> None:
        """Simulate an output flush failure while retaining the array reference.

        Args:
            array (np.memmap): Live output map.
        """
        opened.append(array)
        raise OSError("simulated disk quota exceeded")

    monkeypatch.setattr(np.memmap, "flush", fail_flush)
    with pytest.raises(OSError, match="simulated disk quota"):
        run_analysis(
            "cosine_distance",
            [
                "--feature_dir",
                str(feature_bundle),
                "--output_root",
                str(tmp_path / "out"),
                "--positions",
                "S1",
                "--layers",
                "0",
                "--num_threads",
                "1",
            ],
        )
    assert opened and all(array._mmap.closed for array in opened)
    root = tmp_path / "out/concept_arc_100_10/tiny/repr_analysis/baseline/all"
    assert not list(root.iterdir())


def test_embedding_runner_and_offline_round_trip(
    feature_bundle: Path, tmp_path: Path
) -> None:
    """Keep embedding identities, support grids, preprocessing and HTML reproducible.

    Args:
        feature_bundle (Path): Complete synthetic support-feature bundle.
        tmp_path (Path): Isolated output directory.
    """
    from arc_cd.repr_analysis.embedding import EmbeddingOptions, embed_tasks
    from arc_cd.repr_analysis.embedding_runner import run_embedding

    arguments = [
        "--feature_dir",
        str(feature_bundle),
        "--output_root",
        str(tmp_path / "output"),
        "--methods",
        "pca",
        "--positions",
        "S1",
        "S3",
        "--min_cluster_size",
        "3",
        "--min_samples",
        "2",
    ]
    directory = run_embedding(arguments)
    axes = json.loads((directory / "axes.json").read_text())
    assert axes["layers"] == [0, 2] and axes["methods"] == ["pca"]
    bundle = load_bundle(feature_bundle)
    packed = json.loads((directory / "supports.json").read_text())
    for row, record in enumerate(bundle.records):
        assert packed[row]["case_id"] == record["case_id"]
        for i, pair in enumerate(record["support_examples"]):
            for kind in ("input", "output"):
                grid = packed[row]["supports"][i][kind]
                restored = np.array(list(grid["cells"]), dtype=int).reshape(
                    grid["rows"], grid["cols"]
                )
                np.testing.assert_array_equal(restored, pair[kind])
    for position in axes["positions"]:
        raw = bundle.load_position(position)
        for layer_index, layer in enumerate(axes["layers"]):
            _, versions, means = center_versions(raw[:, layer_index], bundle.rows)
            with np.load(
                directory / position / f"embedding_layer_{layer}.npz",
                allow_pickle=False,
            ) as saved:
                expected = embed_tasks(
                    means,
                    EmbeddingOptions(
                        methods=("pca",), min_cluster_size=3, min_samples=2
                    ),
                )
                for key in expected:
                    np.testing.assert_allclose(saved[key], expected[key])
                np.testing.assert_array_equal(
                    saved["representative_versions"],
                    representative_versions(versions, means),
                )
                np.testing.assert_allclose(
                    saved["within_task_dispersion"],
                    within_task_dispersion(versions),
                )
    page = directory / "explorer.html"
    initial = page.read_text()
    assert 'src="http' not in initial and 'href="http' not in initial
    assert "Show all 8 variants" in initial and "pointerup" in initial
    payload = json.loads(
        initial.split('<script type="application/json" id="explorer-data">', 1)[
            1
        ].split("</script>", 1)[0]
    )
    assert len(payload["frames"]) == 4
    assert len(payload["supports"]) == len(bundle.records)
    page.unlink()
    plot_analysis(directory)
    assert page.read_text() == initial
    with pytest.raises(FileExistsError):
        run_embedding(arguments)
    assert not list(directory.parent.glob(".*staging*"))
