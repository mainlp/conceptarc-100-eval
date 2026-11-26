"""Spatial descriptors, local object grouping, and appearance-run contracts."""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from arc_cd.repr_analysis import visualization
from arc_cd.repr_analysis.data import fingerprint
from arc_cd.repr_analysis.geometry import DISTANCE_METRICS, separation_similarity
from arc_cd.repr_analysis.visual_features import (
    DEFAULT_FEATURES,
    FEATURE_GROUPS,
    GRID_SCOPES,
    aggregate_support_features,
    basic_pair_features,
    pair_feature_layout,
    preprocess_task_features,
    support_feature_layout,
)
from arc_cd.repr_analysis.visual_runner import (
    log_feature_samples,
    parse_arguments,
    run_visual_baseline,
)
from tests import test_visual_baseline

feature_bundle = test_visual_baseline.feature_bundle
metadata_bundle = test_visual_baseline.metadata_bundle
population_sources = test_visual_baseline.population_sources


@pytest.mark.parametrize(
    "grid,expected",
    [
        ([[0, 0, 0, 0], [0, 2, 2, 0]], [0.5, 0.5, 0.5, 1]),
        ([[0, 0, 0, 0, 2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [1 / 3, 0.2, 1, 0]),
        ([[0, 0], [0, 0]], [0, 0, 0.5, 0.5]),
        ([[0, 2, 0, 0]], [1, 0.25, 1 / 3, 0.5]),
        ([[0], [2], [2]], [2 / 3, 1, 0.5, 0.75]),
        ([[2]], [1, 1, 0.5, 0.5]),
        ([[0]], [0, 0, 0.5, 0.5]),
    ],
)
def test_spatial_features(grid: list[list[int]], expected: list[float]) -> None:
    """Check normalized extent and zero-based centroids against hand calculations.

    Args:
        grid (list[list[int]]): Rectangular, empty, or singleton-axis example.
        expected (list[float]): Four manually calculated spatial descriptors.
    """
    actual = basic_pair_features(grid, None, "input", ("spatial",))
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_array_equal(
        actual, basic_pair_features(None, grid, "output", ("spatial",))
    )


@pytest.mark.parametrize(
    "grid,expected",
    [
        ([[2, 2, 3, 3]], [2, 0.5, 1, 0.5]),
        ([[2, 3, 2, 3]], [2, 0.5, 1, 0.75]),
        ([[2, 3, 2, 3], [3, 2, 3, 2]], [2, 0.5, 1, 1]),
        ([[2, 3, 2, 3], [2, 3, 2, 3]], [4, 0.25, 1, 0.25]),
        ([[2, 0, 0], [0, 2, 0], [0, 0, 2]], [1, 1 / 3, 1, 1]),
        ([[2, 0, 2, 0, 2]], [1, 0.6, 1, 1]),
        ([[2, 2, 0, 2, 2]], [2, 0.4, 1, 0.4]),
        ([[2, 2, 0, 2]], [2, 0.5, 1, 0.375]),
        ([[2, 0, 0, 2]], [1, 0.5, 1, 1]),
        ([[2, 0, 0, 0, 2]], [2, 0.2, 1, 0.2]),
        ([[0, 0], [0, 0]], [0, 0, 0, 0]),
        ([[2]], [1, 1, 1, 1]),
        ([[2, 2, 0, 3]], [2, 0.5, 1, 0.375]),
    ],
)
def test_object_features(
    grid: list[list[int]],
    expected: list[float],
) -> None:
    """Check the local extractor's grouping and unweighted object means.

    Args:
        grid (list[list[int]]): Hand-designed visible pattern.
        expected (list[float]): Count, largest fraction, and mean box extents.
    """
    actual = basic_pair_features(grid, None, "input", ("objects",))
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    recolored = [[{0: 0, 2: 7, 3: 1}[c] for c in row] for row in grid]
    np.testing.assert_array_equal(
        actual,
        basic_pair_features(None, recolored, "output", ("objects",)),
    )


@pytest.mark.parametrize("grid_scope", GRID_SCOPES)
@pytest.mark.parametrize(
    "groups", [c for size in range(1, 6) for c in combinations(FEATURE_GROUPS, size)]
)
def test_selected_layouts(groups: tuple[str, ...], grid_scope: str) -> None:
    """Give every valid selection aligned coordinates and reject unpaired changes.

    Args:
        groups (tuple[str, ...]): Requested feature subset.
        grid_scope (str): Selected grid side(s).
    """
    if grid_scope != "both" and "pixel_change" in groups:
        with pytest.raises(ValueError, match="pixel_change"):
            pair_feature_layout(grid_scope, groups)
        return
    names, blocks = pair_feature_layout(grid_scope, groups)
    sides = 2 if grid_scope == "both" else 1
    dimensions = {
        "geometry": 4 * sides,
        "color": 10 * sides,
        "spatial": 4 * sides,
        "objects": 4 * sides,
        "pixel_change": 4,
    }
    assert (
        len(names)
        == len(blocks)
        == len(set(names))
        == sum(dimensions[g] for g in groups)
    )
    assert set(blocks) == set(groups)
    assert (names, blocks) == pair_feature_layout(grid_scope, groups[::-1])
    for mode, multiplier in (("concat", 3), ("mean_std", 2)):
        support_names, support_blocks = support_feature_layout(mode, grid_scope, groups)
        values = np.tile(np.arange(len(names)), (3, 1)).astype(float)
        values[1] += 3
        values[2] += 6
        actual = aggregate_support_features(values, mode, grid_scope, groups)
        expected = (
            values.ravel()
            if mode == "concat"
            else np.r_[values[1], np.full(len(names), np.sqrt(6))]
        )
        np.testing.assert_allclose(actual, expected)
        assert (
            len(support_names)
            == len(support_blocks)
            == len(actual)
            == multiplier * len(names)
        )


@pytest.mark.parametrize("mode", ["concat", "mean_std"])
@pytest.mark.parametrize("group", FEATURE_GROUPS)
def test_new_group_and_descriptor_ablations(mode: str, group: str) -> None:
    """Remove all derived coordinates and rebalance only the retained group sizes.

    Args:
        mode (str): Support aggregation.
        group (str): Entire descriptor group to exclude.
    """
    names, blocks = support_feature_layout(mode, features=FEATURE_GROUPS)
    values = np.random.default_rng(84).normal(size=(20, len(names)))
    single = (
        "input_foreground_centroid_x" if group != "spatial" else "output_object_count"
    )
    output = preprocess_task_features(
        values, blocks, (group,), feature_names=names, exclude_pair_features=(single,)
    )
    expected = np.array(
        [
            block != group and name.split("/")[-1] != single
            for block, name in zip(blocks, names)
        ]
    )
    np.testing.assert_array_equal(output["included"], expected)
    for retained_group in FEATURE_GROUPS:
        selected = expected & (np.asarray(blocks) == retained_group)
        if selected.any():
            np.testing.assert_allclose(
                output["block_weights"][selected], 1 / np.sqrt(selected.sum())
            )


def test_feature_selection_cli() -> None:
    """Expose appearance defaults and validate explicit groups before extraction."""
    base = ["--data_dir", "unused", "--metadata_dir", "unused"]
    args = parse_arguments(base)
    assert args.features == list(DEFAULT_FEATURES)
    assert args.run_name == "visual_v2"
    assert args.log_level == "INFO"
    assert args.log_feature_samples == "per_concept"
    assert parse_arguments(
        base + ["--features", "objects", "spatial", "geometry", "color"]
    ).features == list(DEFAULT_FEATURES)
    assert parse_arguments(base + ["--features", "pixel_change"]).features == [
        "pixel_change"
    ]
    for options in (
        ["--features"],
        ["--features", "spatial", "spatial"],
        ["--features", "change"],
        ["--features", "pixel_change", "--grid_scope", "input"],
        ["--features", "color", "--exclude_feature_groups", "objects"],
        ["--exclude_pair_features", "changed_count"],
        ["--features", "objects", "--exclude_feature_groups", "objects"],
        ["--grid_scope", "input", "--exclude_pair_features", "output_object_count"],
        ["--grid_scope", "input", "--exclude_pair_features", "output_height"],
        ["--exclude_pair_features", "height"],
        ["--log_feature_samples", "-1"],
        ["--log_feature_samples", "11"],
        ["--log_feature_samples", "1.5"],
        ["--log_level", "unknown"],
    ):
        with pytest.raises(SystemExit):
            parse_arguments(base + options)
    assert parse_arguments(
        base + ["--exclude_pair_features", "output_width", "input_height"]
    ).exclude_pair_features == ["input_height", "output_width"]
    for count in range(11):
        assert parse_arguments(
            base + ["--log_feature_samples", str(count)]
        ).log_feature_samples == str(count)


def test_paired_appearance_contains_each_grid() -> None:
    """Match every coordinate to its grid side, including duplicated geometry."""
    source = [[0, 2, 2], [0, 0, 0]]
    target = [[3, 0, 0], [3, 0, 0]]
    paired_names, paired_blocks = pair_feature_layout()
    paired = basic_pair_features(source, target)
    assert len(paired) == 44
    assert paired_blocks.count("geometry") == 8
    for side in ("input", "output"):
        names, _ = pair_feature_layout(side)
        single = basic_pair_features(source, target, side)
        indices = [paired_names.index(name) for name in names]
        assert len(single) == 22
        np.testing.assert_array_equal(paired[indices], single)
    np.testing.assert_array_equal(paired[:4], paired[4:8])


@pytest.mark.parametrize("mode", ["concat", "mean_std"])
@pytest.mark.parametrize("metric", DISTANCE_METRICS)
def test_duplicate_geometry_preserves_group_weight(mode: str, metric: str) -> None:
    """Duplicating every geometry coordinate must not double its distance weight.

    Args:
        mode (str): Layout of task coordinates.
        metric (str): Cosine or Euclidean distance.
    """
    names, blocks = support_feature_layout(mode)
    raw = np.random.default_rng(55).uniform(size=(30, len(names)))
    shared = np.ones(len(names), dtype=bool)
    for index, (name, block) in enumerate(zip(names, blocks)):
        if block == "geometry" and "/output_" in name:
            raw[:, index] = raw[:, names.index(name.replace("/output_", "/input_"))]
            shared[index] = False
    doubled = preprocess_task_features(raw, blocks)["vectors"]
    original = preprocess_task_features(
        raw[:, shared], tuple(np.asarray(blocks)[shared])
    )["vectors"]
    np.testing.assert_allclose(
        separation_similarity(doubled, metric),
        separation_similarity(original, metric),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("0", []),
        ("1", [2]),
        ("2", [2, 3]),
        ("3", [2, 3, 0]),
        ("10", [2, 3, 0]),
        ("per_concept", [2, 3]),
    ],
)
@pytest.mark.parametrize("mode", ["concat", "mean_std"])
def test_feature_preview_selection(
    caplog: pytest.LogCaptureFixture,
    selection: str,
    expected: list[int],
    mode: str,
) -> None:
    """Preview one full first-version vector per parent with aligned identities.

    Args:
        caplog (pytest.LogCaptureFixture): Captured console log records.
        selection (str): Sample count or per-concept mode.
        expected (list[int]): Expected source rows in preview order.
        mode (str): Aggregation of all three support pairs within each version.
    """
    caplog.set_level(logging.INFO, logger="arc_cd.repr_analysis.visual_runner")
    parents = ["A/1", "Excluded/1", "A/2", "B/1"]
    pairs = np.arange(4 * 8 * 3 * 2).reshape(4, 8, 3, 2)
    prefixes = (
        ("support_1", "support_2", "support_3") if mode == "concat" else ("mean", "std")
    )
    names = [
        f"{prefix}/{name}"
        for prefix in prefixes
        for name in ("input_height", "output_height")
    ]
    vectors = (
        pairs.reshape(4, 8, 6)
        if mode == "concat"
        else np.concatenate((pairs.mean(axis=2), pairs.std(axis=2)), axis=2)
    )
    arrays = {
        "concept": np.array(["A", "Excluded", "A", "B"]),
        "source_task_id": np.array(parents),
        "version_task_ids": np.array(
            [
                [f"{parent}_{version:03d}" for version in range(1, 9)]
                for parent in parents
            ]
        ),
        "support_set_ids": np.tile(
            [f"{version:03d}" for version in range(1, 9)], (4, 1)
        ),
        "feature_names": np.array(names),
        "version_features": vectors,
    }
    log_feature_samples(arrays, np.array([2, 0, 3]), selection)
    headers = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Feature sample:")
    ]
    features = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("  feature_vector")
    ]
    assert len(headers) == len(features) == len(expected)
    for row, header, values in zip(expected, headers, features):
        assert f"concept={arrays['concept'][row]} " in header
        assert f"parent_task_id={arrays['source_task_id'][row]} " in header
        assert (
            f"task_id={arrays['version_task_ids'][row, 0]} support_set_id=001" in header
        )
        expected_values = ", ".join(
            f"{name}={value:.8g}" for name, value in zip(names, vectors[row, 0])
        )
        assert (
            values
            == f"  feature_vector ({len(names)} coordinates): [{expected_values}]"
        )
    if expected:
        assert "first support-set version only" in caplog.text
        assert "all three supports; before version averaging" in caplog.text
    else:
        assert not caplog.records


@pytest.mark.parametrize("scope", GRID_SCOPES)
@pytest.mark.parametrize("mode", ["concat", "mean_std"])
@pytest.mark.parametrize("metric", DISTANCE_METRICS)
def test_appearance_pipeline(
    feature_bundle: Path,
    metadata_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    mode: str,
    metric: str,
) -> None:
    """Preserve eight-version averaging, provenance, metrics, and offline figures.

    Args:
        feature_bundle (Path): Original frozen hidden-state fixture.
        metadata_bundle (Path): Reference task annotations.
        tmp_path (Path): Dataset and output root.
        monkeypatch (pytest.MonkeyPatch): Avoid repeated figure rendering.
        scope (str): Input, output, or both grids.
        mode (str): Support aggregation.
        metric (str): Distance for separation.
    """
    render = (scope, mode, metric) == ("both", "concat", "euclidean")
    if not render:
        monkeypatch.setattr(visualization, "plot_analysis", lambda *a, **k: None)
    directory = run_visual_baseline(
        [
            "--data_dir",
            str(tmp_path / "ConceptARC-100-10"),
            "--metadata_dir",
            str(metadata_bundle),
            "--output_root",
            str(tmp_path / "output"),
            "--support_aggregation",
            mode,
            "--grid_scope",
            scope,
            "--distance_metric",
            metric,
            "--min_family_tasks",
            "2",
            "--num_threads",
            "1",
        ]
    )
    assert "visual_features" in directory.parts and "visual_v2" in directory.parts
    assert "features_geometry_color_spatial_objects" in directory.parts
    config = json.loads((directory / "run_config.json").read_text())
    axes = json.loads((directory / "axes.json").read_text())
    assert config["features"] == axes["features"] == list(DEFAULT_FEATURES)
    assert config["feature_set"] == "visual_v2"
    assert config["object_extraction"] == {
        "extractor": "arc_cd.repr_analysis.visual_objects.visual_scene_objects"
    }
    assert config["geometry_layout"] == axes["geometry_layout"] == "per_grid"
    width = 44 if scope == "both" else 22
    assert config["feature_groups"]["raw_dimension"] == width * (
        3 if mode == "concat" else 2
    )
    with np.load(directory / "features.npz") as archive:
        pairs = archive["pair_features"]
        assert pairs.shape == (32, 8, 3, width)
        expected_versions = (
            pairs.reshape(32, 8, 3 * width)
            if mode == "concat"
            else np.concatenate((pairs.mean(axis=2), pairs.std(axis=2, ddof=0)), axis=2)
        )
        np.testing.assert_allclose(
            archive["version_features"], expected_versions, atol=1e-15
        )
        np.testing.assert_allclose(
            archive["embeddings"], expected_versions.mean(axis=1), atol=1e-15
        )
        assert np.isfinite(pairs).all()
        assert len(np.unique(archive["version_task_ids"])) == 256
    assert "pixel_change" not in axes["feature_blocks"]
    if render:
        assert list((directory / "figures").rglob("*.pdf"))
        before = fingerprint(directory / "metrics.npz")
        monkeypatch.setattr(
            "arc_cd.repr_analysis.visual_features.visual_scene_objects",
            lambda *a: pytest.fail("Offline plots must not extract objects"),
        )
        visualization.plot_analysis(directory)
        assert fingerprint(directory / "metrics.npz") == before


def test_unrequested_objects_are_not_computed(
    feature_bundle: Path,
    metadata_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip object extraction for spatial and pixel-change feature selections.

    Args:
        feature_bundle (Path): Original hidden-state fixture.
        metadata_bundle (Path): Frozen annotations.
        tmp_path (Path): Dataset and output root.
        monkeypatch (pytest.MonkeyPatch): Fail on any unintended object extraction.
    """
    monkeypatch.setattr(visualization, "plot_analysis", lambda *a, **k: None)
    monkeypatch.setattr(
        "arc_cd.repr_analysis.visual_features.visual_scene_objects",
        lambda *a: pytest.fail("Unrequested objects must not be computed"),
    )
    for groups in (("geometry", "color", "spatial"), ("pixel_change",)):
        directory = run_visual_baseline(
            [
                "--data_dir",
                str(tmp_path / "ConceptARC-100-10"),
                "--metadata_dir",
                str(metadata_bundle),
                "--output_root",
                str(tmp_path / "out"),
                "--features",
                *groups,
                "--min_family_tasks",
                "2",
                "--num_threads",
                "1",
            ]
        )
        config = json.loads((directory / "run_config.json").read_text())
        assert config["object_extraction"] is None
