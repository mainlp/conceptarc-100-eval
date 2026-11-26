"""Visual descriptors, ordered aggregation, provenance, and separation parity."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from arc_cd.data_prep import Grid
from arc_cd.data_prep.load_data import load_concept_arc
from arc_cd.repr_analysis import visualization
from arc_cd.repr_analysis.family_metrics import eligible_references, family_separation
from arc_cd.repr_analysis.geometry import (
    DISTANCE_METRICS,
    DistanceMetric,
    cosine_from_gram,
    separation_similarity,
)
from arc_cd.repr_analysis.visual_features import (
    DEFAULT_FEATURES,
    GRID_SCOPES,
    PIXEL_CHANGE_NAMES,
    GridScope,
    SupportAggregation,
    aggregate_support_features,
    basic_pair_features,
    pair_feature_layout,
    preprocess_task_features,
    support_feature_layout,
)
from arc_cd.repr_analysis.visual_runner import (
    extract_visual_features,
    parse_arguments,
    run_visual_baseline,
)
from tests import test_repr_population

feature_bundle = test_repr_population.feature_bundle
metadata_bundle = test_repr_population.metadata_bundle

BASIC_GROUPS = ("geometry", "color", "pixel_change")
BASIC_PAIR_NAMES = (
    *(
        f"{side}_{name}"
        for side in ("input", "output")
        for name in ("height", "width", "area", "aspect_ratio")
    ),
    *(f"input_color_{color}_fraction" for color in range(10)),
    *(f"output_color_{color}_fraction" for color in range(10)),
    *PIXEL_CHANGE_NAMES,
)


def basic_groups(grid_scope: str) -> tuple[str, ...]:
    """Select geometry/color/pixel-change descriptor groups available for a grid scope.

    Args:
        grid_scope (str): Input, output, or both.

    Returns:
        tuple[str, ...]: Original geometry/color groups plus paired pixel changes.
    """
    return BASIC_GROUPS if grid_scope == "both" else BASIC_GROUPS[:2]


def _write_visual_concept(dataset: Path, concept: str, color: int) -> None:
    """Write equal-size support pairs for the existing population fixtures.

    Args:
        dataset (Path): Fixture dataset directory.
        concept (str): Concept and parent name prefix.
        color (int): Color used by this concept's examples.
    """
    directory = dataset / concept
    directory.mkdir()
    for version in range(1, 9):
        supports = []
        for index in range(3):
            source = np.zeros((2 + version % 2, 3 + index), dtype=int)
            source[0, :2] = color
            target = source.copy()
            target[-1, index] = version
            target[0, index % 2] = (color + index) % 10
            supports.append({"input": source.tolist(), "output": target.tolist()})
        row = {"train": supports, "test": [{"input": [[9]], "output": [[0]]}]}
        (directory / f"{concept}1_{version:03d}.json").write_text(json.dumps(row))


@pytest.fixture
def population_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Use valid visual grids in the shared sixteen-concept metadata fixture.

    Args:
        tmp_path (Path): Temporary source root.
        monkeypatch (pytest.MonkeyPatch): Replace only the fixture's grid generator.

    Returns:
        tuple[Path, Path]: Combined and predicate feature fixture directories.
    """
    monkeypatch.setattr(test_repr_population, "_write_concept", _write_visual_concept)
    return test_repr_population.population_sources.__wrapped__(tmp_path)


def test_exact_pair_features() -> None:
    """Check all 32 values against a hand-counted rectangular support pair."""
    features = basic_pair_features(
        [[0, 1, 2], [3, 0, 4]], [[5, 0, 6], [3, 0, 4]], features=BASIC_GROUPS
    )
    expected = np.r_[
        [2, 3, 6, 1.5],
        [2, 3, 6, 1.5],
        np.array([2, 1, 1, 1, 1, 0, 0, 0, 0, 0]) / 6,
        np.array([2, 0, 0, 1, 1, 1, 1, 0, 0, 0]) / 6,
        [3, 1 / 6, 1 / 6, 1 / 6],
    ]
    assert len(BASIC_PAIR_NAMES) == len(set(BASIC_PAIR_NAMES)) == 32
    assert pair_feature_layout(features=BASIC_GROUPS)[0] == BASIC_PAIR_NAMES
    assert features.dtype == np.float64
    np.testing.assert_array_equal(features, expected)
    assert features[-3:].sum() == features[-4] / features[2]


@pytest.mark.parametrize("grid_scope", ["input", "output"])
def test_single_grid_features(grid_scope: GridScope) -> None:
    """Use only the selected grid, including its own size and histogram denominator.

    Args:
        grid_scope (GridScope): Side supplying the 14 descriptors.
    """
    source = [[0, 1, 1], [2, 0, 2]]
    target = [[4], [4], [0]]
    expected = (
        np.r_[[2, 3, 6, 1.5], np.array([2, 2, 2, 0, 0, 0, 0, 0, 0, 0]) / 6]
        if grid_scope == "input"
        else np.r_[[3, 1, 3, 1 / 3], np.array([1, 0, 0, 0, 2, 0, 0, 0, 0, 0]) / 3]
    )
    np.testing.assert_array_equal(
        basic_pair_features(
            source, target, grid_scope, features=basic_groups(grid_scope)
        ),
        expected,
    )
    ignored_grids: tuple[Grid | None, ...] = (None, [], [[True]], [[9, 8, 7, 6]])
    for ignored in ignored_grids:
        actual = (
            basic_pair_features(
                source, ignored, grid_scope, features=basic_groups(grid_scope)
            )
            if grid_scope == "input"
            else basic_pair_features(
                ignored, target, grid_scope, features=basic_groups(grid_scope)
            )
        )
        np.testing.assert_array_equal(actual, expected)
    names, blocks = pair_feature_layout(grid_scope, features=basic_groups(grid_scope))
    assert len(names) == 14 and blocks == ("geometry",) * 4 + ("color",) * 10
    assert names[:4] == tuple(
        f"{grid_scope}_{name}" for name in ("height", "width", "area", "aspect_ratio")
    )
    assert names[4:] == tuple(f"{grid_scope}_color_{i}_fraction" for i in range(10))
    with pytest.raises(ValueError):
        basic_pair_features(None, None, grid_scope, features=basic_groups(grid_scope))


@pytest.mark.parametrize("grid_scope", ["input", "output"])
@pytest.mark.parametrize("mode", ["concat", "mean_std"])
def test_single_grid_aggregation(
    grid_scope: GridScope, mode: SupportAggregation
) -> None:
    """Match names, dimensions, means, and population SD for 14-value supports.

    Args:
        grid_scope (GridScope): Single support side.
        mode (SupportAggregation): Concatenation or mean and standard deviation.
    """
    values = np.arange(14, dtype=float)[None, :] + np.array([0, 2, 4])[:, None]
    actual = aggregate_support_features(
        values, mode, grid_scope, feature_groups=basic_groups(grid_scope)
    )
    expected = (
        values.ravel()
        if mode == "concat"
        else np.r_[np.arange(14) + 2, np.full(14, np.sqrt(8 / 3))]
    )
    np.testing.assert_allclose(actual, expected)
    names, blocks = support_feature_layout(
        mode, grid_scope, features=basic_groups(grid_scope)
    )
    assert len(actual) == len(names) == len(blocks) == (42 if mode == "concat" else 28)
    with pytest.raises(ValueError):
        aggregate_support_features(
            np.zeros((3, 28)),
            mode,
            grid_scope,
            feature_groups=basic_groups(grid_scope),
        )


def test_invalid_grid_scope() -> None:
    """Reject unknown grid selections in all feature entry points."""
    invalid_scope = cast(GridScope, "invalid")
    features = basic_groups(invalid_scope)
    with pytest.raises(ValueError, match="Unknown grid scope"):
        pair_feature_layout(invalid_scope, features=features)
    with pytest.raises(ValueError, match="Unknown grid scope"):
        basic_pair_features([[0]], [[0]], invalid_scope, features=features)
    with pytest.raises(ValueError, match="Unknown grid scope"):
        extract_visual_features([], "concat", invalid_scope, features=features)


@pytest.mark.parametrize(
    "source,target,changes",
    [
        ([[0, 0]], [[0, 0]], [0, 0, 0, 0]),
        ([[1, 2]], [[1, 2]], [0, 0, 0, 0]),
        ([[0, 0]], [[1, 2]], [2, 1, 0, 0]),
        ([[1, 2]], [[0, 0]], [2, 0, 1, 0]),
        ([[1, 2]], [[2, 1]], [2, 0, 0, 1]),
        ([[1, 0]], [[0, 1]], [2, 0.5, 0.5, 0]),
    ],
)
def test_pixel_change_categories(
    source: Any, target: Any, changes: list[float]
) -> None:
    """Distinguish painting, erasing, recoloring, and movement-like pixel changes.

    Args:
        source (Any): Small input grid.
        target (Any): Small output grid.
        changes (list[float]): Hand-computed final four features.
    """
    values = basic_pair_features(source, target, features=BASIC_GROUPS)
    np.testing.assert_array_equal(values[-4:], changes)
    assert values[8:18].sum() == pytest.approx(1)
    assert values[18:28].sum() == pytest.approx(1)


@pytest.mark.parametrize(
    "grid", [[], [[]], [[0], [0, 1]], [[True, 0]], [[1.0]], [[-1]], [[10]]]
)
def test_invalid_grids(grid: Any) -> None:
    """Reject malformed grids and invalid categorical cell values.

    Args:
        grid (Any): Invalid input grid.
    """
    with pytest.raises(ValueError):
        basic_pair_features(grid, grid, features=BASIC_GROUPS)
    with pytest.raises(ValueError, match="equal input/output"):
        basic_pair_features([[0]], [[0, 0]], features=BASIC_GROUPS)


def test_aggregation_order_and_population_sd() -> None:
    """Concatenation preserves slots; means and population SD ignore their order."""
    pairs = np.full((3, 32), 0.1)
    pairs[:, 0] = [1, 3, 5]
    concatenated = aggregate_support_features(
        pairs, "concat", feature_groups=BASIC_GROUPS
    )
    assert concatenated.shape == (96,)
    np.testing.assert_array_equal(concatenated[[0, 32, 64]], [1, 3, 5])
    moments = aggregate_support_features(pairs, "mean_std", feature_groups=BASIC_GROUPS)
    assert moments.shape == (64,)
    assert moments[0] == 3
    assert moments[32] == pytest.approx(np.sqrt(8 / 3))
    np.testing.assert_array_equal(moments[33:], 0)
    np.testing.assert_array_equal(
        aggregate_support_features(
            pairs[::-1], "mean_std", feature_groups=BASIC_GROUPS
        ),
        moments,
    )
    assert not np.array_equal(
        aggregate_support_features(pairs[::-1], "concat", feature_groups=BASIC_GROUPS),
        concatenated,
    )
    names, blocks = support_feature_layout("concat", features=BASIC_GROUPS)
    assert names[32] == "support_2/input_height" and blocks[32] == "geometry"
    names, blocks = support_feature_layout("mean_std", features=BASIC_GROUPS)
    assert names[32] == "std/input_height" and blocks[32] == "geometry"
    with pytest.raises(ValueError):
        aggregate_support_features(pairs[:2], "concat", feature_groups=BASIC_GROUPS)


def test_pair_features_do_not_encode_pixel_locations() -> None:
    """Joint pixel permutations preserve descriptors; transpose changes geometry only."""
    rng = np.random.default_rng(73)
    source = rng.integers(0, 10, (7, 11))
    target = rng.integers(0, 10, (7, 11))
    expected = basic_pair_features(
        source.tolist(), target.tolist(), features=BASIC_GROUPS
    )
    permutation = rng.permutation(source.size)
    observed = basic_pair_features(
        source.ravel()[permutation].reshape(source.shape).tolist(),
        target.ravel()[permutation].reshape(target.shape).tolist(),
        features=BASIC_GROUPS,
    )
    np.testing.assert_array_equal(observed, expected)
    transposed = basic_pair_features(
        source.T.tolist(), target.T.tolist(), features=BASIC_GROUPS
    )
    np.testing.assert_array_equal(transposed[:8], [11, 7, 77, 7 / 11] * 2)
    np.testing.assert_array_equal(transposed[8:], expected[8:])


@pytest.mark.parametrize("mode", ["concat", "mean_std"])
@pytest.mark.parametrize("excluded", BASIC_GROUPS)
def test_ablation_removes_all_group_coordinates(
    mode: SupportAggregation, excluded: str
) -> None:
    """Omitted values cannot affect the remaining coordinates or their weights.

    Args:
        mode (SupportAggregation): Support aggregation defining the coordinate layout.
        excluded (str): Entire descriptor group to omit.
    """
    _, blocks = support_feature_layout(mode, features=BASIC_GROUPS)
    raw = np.random.default_rng(35).normal(size=(20, len(blocks)))
    raw[:, 0] = 4.0
    full = preprocess_task_features(raw, blocks)
    ablated = preprocess_task_features(raw, blocks, (excluded,))
    included = np.asarray(blocks) != excluded
    np.testing.assert_array_equal(ablated["included"], included)
    np.testing.assert_array_equal(ablated["retained"], full["retained"] & included)
    np.testing.assert_array_equal(
        ablated["vectors"], full["vectors"][:, included[full["retained"]]]
    )
    np.testing.assert_array_equal(ablated["block_weights"][~included], 0)
    perturbed = raw.copy()
    perturbed[:, ~included] = 1e9 * raw[::-1, ~included]
    rerun = preprocess_task_features(perturbed, blocks, (excluded,))
    np.testing.assert_array_equal(ablated["vectors"], rerun["vectors"])
    remaining = ablated["retained"]
    np.testing.assert_array_equal(ablated["scale"][remaining], full["scale"][remaining])
    np.testing.assert_array_equal(
        ablated["block_weights"][remaining], full["block_weights"][remaining]
    )


def test_ablation_with_constant_remaining_groups() -> None:
    """Discarding the only varying group yields undefined rather than chance cosines."""
    raw = np.array([[1, 0.1, 0], [2, 0.1, 0], [3, 0.1, 0]])
    result = preprocess_task_features(raw, BASIC_GROUPS, ("geometry",))
    assert result["included"].sum() == 2 and not result["retained"].any()
    assert result["vectors"].shape == (3, 0)
    assert np.isnan(cosine_from_gram(result["vectors"] @ result["vectors"].T)).all()
    geometry_only = preprocess_task_features(
        raw, BASIC_GROUPS, ("color", "pixel_change")
    )
    assert geometry_only["vectors"].shape == (3, 1)
    for invalid in (BASIC_GROUPS, ("geometry", "geometry"), ("invalid",)):
        with pytest.raises(ValueError, match="Excluded groups"):
            preprocess_task_features(raw, BASIC_GROUPS, invalid)


@pytest.mark.parametrize("mode", ["concat", "mean_std"])
@pytest.mark.parametrize("excluded", BASIC_PAIR_NAMES[-4:])
def test_pair_ablation_removes_every_derived_coordinate(
    mode: SupportAggregation, excluded: str
) -> None:
    """Drop every copy of one descriptor without letting its values affect cosine.

    Args:
        mode (SupportAggregation): Support aggregation defining the coordinate layout.
        excluded (str): Pixel-change descriptor omitted from all supports.
    """
    names, blocks = support_feature_layout(mode, features=BASIC_GROUPS)
    raw = np.random.default_rng(35).normal(size=(20, len(names)))
    selected = np.array(
        [
            block == "pixel_change" and name.split("/")[-1] != excluded
            for name, block in zip(names, blocks)
        ]
    )
    expected = raw[:, selected]
    expected = (expected - expected.mean(axis=0)) / expected.std(axis=0)
    expected /= np.sqrt(selected.sum())
    for values in (raw, np.where(selected, raw, raw[::-1] * 1e9)):
        result = preprocess_task_features(
            values,
            blocks,
            ("geometry", "color"),
            feature_names=names,
            exclude_pair_features=(excluded,),
        )
        np.testing.assert_array_equal(result["included"], selected)
        np.testing.assert_array_equal(result["retained"], selected)
        np.testing.assert_array_equal(result["block_weights"][~selected], 0)
        np.testing.assert_allclose(result["vectors"], expected, rtol=1e-14, atol=1e-14)
    assert selected.sum() == (9 if mode == "concat" else 6)


def test_invalid_pair_ablation() -> None:
    """Reject ambiguous exclusions, absent names, and an empty selected feature set."""
    names, blocks = support_feature_layout("concat", features=BASIC_GROUPS)
    raw = np.random.default_rng(12).normal(size=(8, len(names)))
    for provided_names, excluded in (
        (None, ("changed_count",)),
        (names[:-1], ("changed_count",)),
        (names, ("changed_count", "changed_count")),
        (names, ("input_height",)),
        (names, ("unknown",)),
        (names, BASIC_PAIR_NAMES[-4:]),
    ):
        with pytest.raises(ValueError):
            preprocess_task_features(
                raw,
                blocks,
                ("geometry", "color"),
                feature_names=provided_names,
                exclude_pair_features=excluded,
            )


def test_preprocessing_and_degenerate_vectors() -> None:
    """Give blocks equal expected squared norm and preserve undefined cosines."""
    raw = np.array([[1, 10, 0.1, 0], [2, 30, 0.1, 1], [3, 20, 0.1, 2]])
    result = preprocess_task_features(
        raw, ("geometry", "geometry", "color", "pixel_change")
    )
    np.testing.assert_array_equal(result["retained"], [True, True, False, True])
    np.testing.assert_allclose(
        result["block_weights"], [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 1]
    )
    np.testing.assert_allclose(result["vectors"].mean(axis=0), 0, atol=1e-15)
    assert np.square(result["vectors"][:, :2]).sum(axis=1).mean() == pytest.approx(1)
    assert np.square(result["vectors"][:, 2]).mean() == pytest.approx(1)
    shifted = preprocess_task_features(
        raw + 10, ("geometry", "geometry", "color", "pixel_change")
    )
    np.testing.assert_allclose(shifted["vectors"], result["vectors"])
    empty = preprocess_task_features(np.full((4, 2), 0.1), ("color", "color"))[
        "vectors"
    ]
    assert empty.shape == (4, 0)
    assert np.isnan(cosine_from_gram(empty @ empty.T)).all()
    centered = preprocess_task_features(np.array([[0.0], [1.0], [2.0]]), ("geometry",))[
        "vectors"
    ]
    assert np.isnan(cosine_from_gram(centered @ centered.T)[1]).all()


@pytest.mark.parametrize("width", [1, 3])
def test_centering_roundoff_does_not_create_cosine(width: int) -> None:
    """Keep a fractional mean anchor undefined despite floating-point residuals.

    Args:
        width (int): Number of repeated color coordinates to balance.
    """
    raw = np.repeat(np.array([[0.0], [0.1], [0.2]]), width, axis=1)
    vectors = preprocess_task_features(raw, ("color",) * width)["vectors"]
    np.testing.assert_array_equal(vectors[1], np.zeros(width))
    cosine = cosine_from_gram(vectors @ vectors.T)
    assert np.isnan(cosine[1]).all()
    assert cosine[0, 2] == pytest.approx(-1)


@pytest.mark.parametrize("repeats", [1, 533])
@pytest.mark.parametrize("excluded", [(), ("geometry",)])
def test_centering_roundoff_at_dataset_scale(
    repeats: int, excluded: tuple[str, ...]
) -> None:
    """Repeated histogram values must not create directions for mean anchors.

    Args:
        repeats (int): Number of copies of three balanced descriptor rows.
        excluded (tuple[str, ...]): Optional omission of the geometry block.
    """
    raw = np.tile([[10, 0.0, 1.0], [20, 0.1, 0.9], [30, 0.2, 0.8]], (repeats, 1))
    blocks = ("geometry", "color", "color")
    result = preprocess_task_features(raw, blocks, excluded)
    np.testing.assert_array_equal(result["vectors"][1::3], 0)
    vectors = result["vectors"][:3]
    cosine = cosine_from_gram(vectors @ vectors.T)
    assert np.isnan(cosine[1]).all()
    assert cosine[0, 2] == pytest.approx(-1)
    assert result["zero_norm_threshold"].shape == ()
    raw[1, 1] += 1e-9
    raw[1, 2] -= 1e-9
    shifted = preprocess_task_features(raw, blocks, excluded)
    assert np.linalg.norm(shifted["vectors"][1]) > shifted["zero_norm_threshold"]


@pytest.mark.parametrize("mode", ["concat", "mean_std"])
@pytest.mark.parametrize("grid_scope", GRID_SCOPES)
def test_extraction_identity_and_queries(
    tmp_path: Path, mode: SupportAggregation, grid_scope: GridScope
) -> None:
    """Group eight versions deterministically and never featurize test queries.

    Args:
        tmp_path (Path): Temporary dataset directory.
        mode (SupportAggregation): Support aggregation mode.
        grid_scope (GridScope): Selected side or both sides of the supports.
    """
    _write_visual_concept(tmp_path, "Count", 3)
    tasks = load_concept_arc(tmp_path)
    pair_width = 32 if grid_scope == "both" else 14
    width = pair_width * (3 if mode == "concat" else 2)
    actual = extract_visual_features(
        tasks, mode, grid_scope, features=basic_groups(grid_scope)
    )
    assert actual["pair_features"].shape == (1, 8, 3, pair_width)
    assert actual["version_features"].shape == (1, 8, width)
    assert actual["embeddings"].shape == (1, width)
    assert actual["support_set_ids"].tolist() == [[f"{i:03d}" for i in range(1, 9)]]
    shuffled = extract_visual_features(
        [replace(task, test=()) for task in tasks[::-1]],
        mode,
        grid_scope,
        features=basic_groups(grid_scope),
    )
    for key in actual:
        np.testing.assert_array_equal(actual[key], shuffled[key])
    np.testing.assert_allclose(
        actual["embeddings"], actual["version_features"].sum(axis=1) / 8
    )
    if mode == "concat":
        np.testing.assert_allclose(
            actual["embeddings"].reshape(3, pair_width),
            actual["pair_features"].mean(axis=1)[0],
        )
    if grid_scope == "both":
        default = extract_visual_features(tasks, mode, features=BASIC_GROUPS)
        for key in actual:
            np.testing.assert_array_equal(actual[key], default[key])
    else:
        altered = [
            replace(
                task,
                train=tuple(
                    (
                        replace(pair, output=None)
                        if grid_scope == "input"
                        else replace(pair, input=[[9]])
                    )
                    for pair in task.train
                ),
            )
            for task in tasks
        ]
        ignored_side_changed = extract_visual_features(
            altered, mode, grid_scope, features=basic_groups(grid_scope)
        )
        for key in actual:
            np.testing.assert_array_equal(actual[key], ignored_side_changed[key])
    for invalid in (
        tasks[:-1],
        (*tasks, tasks[0]),
        (replace(tasks[0], train=tasks[0].train[:2]), *tasks[1:]),
    ):
        with pytest.raises(ValueError):
            extract_visual_features(
                invalid, mode, grid_scope, features=basic_groups(grid_scope)
            )


def _arguments(
    dataset: Path,
    metadata: Path,
    root: Path,
    mode: str,
    population: str = "all",
    grid_scope: str = "both",
) -> list[str]:
    """Create arguments for small-fixture separation runs.

    Args:
        dataset (Path): Source grids.
        metadata (Path): Frozen label bundle.
        root (Path): Result root.
        mode (str): Support aggregation.
        population (str): Requested task group.
        grid_scope (str): Selected grids for the geometry/color/pixel-change set.

    Returns:
        list[str]: Complete CLI arguments.
    """
    return [
        "--features",
        *basic_groups(grid_scope),
        "--data_dir",
        str(dataset),
        "--metadata_dir",
        str(metadata),
        "--output_root",
        str(root),
        "--support_aggregation",
        mode,
        "--population",
        population,
        "--min_family_tasks",
        "2",
        "--num_threads",
        "1",
    ]


@pytest.mark.parametrize("distance_metric", DISTANCE_METRICS)
@pytest.mark.parametrize("mode", ["concat", "mean_std"])
@pytest.mark.parametrize(
    "grid_scope,excluded,excluded_pair,population",
    [
        (scope, excluded, (), population)
        for scope in GRID_SCOPES
        for excluded in [(), ("geometry",), ("color",), ("pixel_change",)]
        for population in ("all", "predicate", "action")
        if scope == "both" or excluded != ("pixel_change",)
    ]
    + [
        ("both", ("geometry", "color"), (name,), "all")
        for name in BASIC_PAIR_NAMES[-4:]
    ]
    + [
        pytest.param(
            "both",
            ("geometry", "color"),
            tuple(name for name in BASIC_PAIR_NAMES[-4:] if name != kept),
            "all",
            id=f"only_{kept}",
        )
        for kept in BASIC_PAIR_NAMES[-4:]
    ],
)
def test_runner_against_explicit_triplets(
    feature_bundle: Path,
    metadata_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    population: str,
    excluded: tuple[str, ...],
    excluded_pair: tuple[str, ...],
    grid_scope: str,
    distance_metric: str,
) -> None:
    """Match saved probabilities to direct comparisons of every eligible triplet.

    Args:
        feature_bundle (Path): Sixteen-concept feature fixture.
        metadata_bundle (Path): Frozen annotation fixture.
        tmp_path (Path): Dataset and result root.
        monkeypatch (pytest.MonkeyPatch): Skip repeated plotting.
        mode (str): Support aggregation.
        population (str): Requested concept population.
        excluded (tuple[str, ...]): Descriptor group omitted from the comparison.
        excluded_pair (tuple[str, ...]): Individual pair descriptors omitted.
        grid_scope (str): Support grid selection.
        distance_metric (str): Pairwise metric; cosine exercises the CLI default.
    """
    monkeypatch.setattr(visualization, "plot_analysis", lambda *args, **kwargs: None)
    source = tmp_path / "ConceptARC-100-10"
    dataset = tmp_path / "ConceptARC-100-8"
    source.rename(dataset)
    args = _arguments(
        dataset, metadata_bundle, tmp_path / "output", mode, population, grid_scope
    )
    scope_args = ["--grid_scope", grid_scope]
    exclusion_args = ["--exclude_feature_groups", *excluded] if excluded else []
    if excluded_pair:
        exclusion_args += ["--exclude_pair_features", *excluded_pair]
    args += scope_args + exclusion_args
    if distance_metric != "cosine":
        args += ["--distance_metric", distance_metric]
    result = run_visual_baseline(args)
    assert "concept_arc_100_8" in result.parts
    axes = json.loads((result / "axes.json").read_text())
    config = json.loads((result / "run_config.json").read_text())
    assert config["metadata_dataset_id"] == "concept_arc_100_10"
    assert config["input_kind"] == "visual_features"
    assert config["distance_metric"] == axes["distance_metric"] == distance_metric
    assert config["analysis"] == result.name == f"{distance_metric}_distance"
    assert config["arguments"]["distance_metric"] == distance_metric
    assert f"{distance_metric.capitalize()} distance" in axes["representation_label"]
    undefined_key = (
        "undefined_anchor_cosine"
        if distance_metric == "cosine"
        else "undefined_anchor_distance"
    )
    assert axes["reason_bits"][undefined_key] == 2
    assert config["grid_scope"] == axes["grid_scope"] == grid_scope
    available_groups = (
        ["geometry", "color", "pixel_change"]
        if grid_scope == "both"
        else ["geometry", "color"]
    )
    assert config["feature_groups"]["available"] == available_groups
    assert config["feature_groups"]["included"] == [
        group for group in available_groups if group not in excluded
    ]
    if grid_scope != "both":
        assert f"{grid_scope}_only" in result.parts
        assert "pixel_change" not in axes["feature_blocks"]
        unselected = "output" if grid_scope == "input" else "input"
        assert not any(f"{unselected}_color_" in name for name in axes["feature_names"])
    assert config["feature_groups"]["excluded"] == list(excluded)
    assert axes["excluded_feature_groups"] == list(excluded)
    assert (
        config["excluded_pair_features"]
        == axes["excluded_pair_features"]
        == list(excluded_pair)
    )
    if excluded:
        assert "without_" + "_".join(excluded) in result.parts
    if excluded_pair:
        assert "without_features" in result.parts
        assert all(name in result.parts for name in excluded_pair)
    assert "layers" not in axes and "positions" not in axes
    assert axes["array_axes"]["raw_task_means.npy"] == ["task", "feature"]
    assert axes["array_axes"]["task_means.npy"] == ["task", "retained_feature"]
    assert axes["array_axes"]["grand_mean.npy"] == ["feature"]
    assert axes["array_axes"]["scores"] == ["scope", "task", "family", "metric"]
    assert len(axes["task_ids"]) == (32 if population == "all" else 16)
    with np.load(result / "features.npz") as features:
        assert features["pair_features"].shape == (
            32,
            8,
            3,
            32 if grid_scope == "both" else 14,
        )
        np.testing.assert_array_equal(
            features["source_task_id"][axes["source_rows"]], axes["task_ids"]
        )
        raw = features["embeddings"][axes["source_rows"]]
        np.testing.assert_array_equal(raw, np.load(result / "raw_task_means.npy"))
    with np.load(result / "preprocessing.npz") as preprocessing:
        mask = preprocessing["retained"]
        expected = (
            (raw[:, mask] - preprocessing["mean"][mask])
            / preprocessing["scale"][mask]
            * preprocessing["block_weights"][mask]
        )
        expected[
            np.linalg.norm(expected, axis=1) <= preprocessing["zero_norm_threshold"]
        ] = 0
        np.testing.assert_array_equal(expected, np.load(result / "task_means.npy"))
        assert len(axes["feature_names"]) == len(mask)
        assert len(axes["retained_feature_names"]) == int(mask.sum())
        included = np.array(
            [
                block not in excluded and name.split("/")[-1] not in excluded_pair
                for name, block in zip(axes["feature_names"], axes["feature_blocks"])
            ]
        )
        np.testing.assert_array_equal(preprocessing["included"], included)
        assert not (mask & ~included).any()
        assert len(axes["included_feature_names"]) == int(included.sum())
        assert config["feature_groups"]["included_dimension"] == int(included.sum())
        assert not mask.all()
    similarity = (
        cosine_from_gram(expected @ expected.T)
        if distance_metric == "cosine"
        else -np.linalg.norm(expected[:, None, :] - expected[None, :, :], axis=-1)
    )
    membership = np.load(result / "memberships.npy")
    labels = np.asarray(axes["labels"])
    with np.load(result / "scores.npz") as scores:
        for scope_index, scope in enumerate(axes["scopes"]):
            for anchor in range(len(raw)):
                candidates = np.isfinite(similarity[anchor]) & (
                    np.arange(len(raw)) != anchor
                )
                if scope == "cross_concept":
                    candidates &= labels != labels[anchor]
                for family in range(membership.shape[1]):
                    positive = similarity[anchor, candidates & membership[:, family]]
                    negative = similarity[anchor, candidates & ~membership[:, family]]
                    observed = scores["scores"][scope_index, anchor, family]
                    if (
                        not membership[anchor, family]
                        or not len(positive)
                        or not len(negative)
                    ):
                        assert np.isnan(observed).all()
                    else:
                        wins = positive[:, None] > negative
                        ties = positive[:, None] == negative
                        np.testing.assert_allclose(
                            observed, [wins.mean(), (wins + 0.5 * ties).mean()]
                        )
    repeat = run_visual_baseline(
        _arguments(
            dataset, metadata_bundle, tmp_path / "repeat", mode, population, grid_scope
        )
        + scope_args
        + exclusion_args
        + ["--distance_metric", distance_metric]
    )
    with (
        np.load(result / "metrics.npz") as first,
        np.load(repeat / "metrics.npz") as second,
    ):
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])
    with pytest.raises(FileExistsError):
        run_visual_baseline(args)


@pytest.mark.parametrize(
    "problem",
    [
        "changed_source",
        "missing_version",
        "extra_version",
        "bad_metadata",
        "during_run",
    ],
)
def test_provenance_rejection(
    feature_bundle: Path,
    metadata_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    """Reject changed datasets or labels without publishing a partial result.

    Args:
        feature_bundle (Path): Source feature fixture.
        metadata_bundle (Path): Matching annotations.
        tmp_path (Path): Dataset root.
        monkeypatch (pytest.MonkeyPatch): Simulate mutation during plotting.
        problem (str): Input corruption to introduce.
    """
    dataset = tmp_path / "ConceptARC-100-10"
    path = next(dataset.glob("*/*.json"))
    if problem == "changed_source":
        path.write_text(path.read_text() + "\n")
    elif problem == "missing_version":
        path.unlink()
    elif problem == "extra_version":
        path.with_name(path.stem + "9.json").write_bytes(path.read_bytes())
    elif problem == "bad_metadata":
        taxonomy = metadata_bundle / "taxonomy.json"
        taxonomy.write_text(taxonomy.read_text() + "\n")
    else:

        def change_source(*args: Any, **kwargs: Any) -> None:
            """Simulate a concurrent writer to a source grid."""
            path.write_text(path.read_text() + "\n")

        monkeypatch.setattr(visualization, "plot_analysis", change_source)
    output = tmp_path / "output"
    with pytest.raises((ValueError, RuntimeError)):
        run_visual_baseline(_arguments(dataset, metadata_bundle, output, "concat"))
    assert not list(output.rglob("metrics.npz"))
    assert not list(output.rglob("*.lock"))


@pytest.mark.parametrize("distance_metric", DISTANCE_METRICS)
def test_visual_figures(
    feature_bundle: Path, metadata_bundle: Path, tmp_path: Path, distance_metric: str
) -> None:
    """Generate and regenerate standalone PNG/PDFs without layer axes.

    Args:
        feature_bundle (Path): Source feature fixture.
        metadata_bundle (Path): Frozen annotations.
        tmp_path (Path): Dataset root.
        distance_metric (str): Metric named in saved metadata and figure titles.
    """
    output = run_visual_baseline(
        _arguments(
            tmp_path / "ConceptARC-100-10",
            metadata_bundle,
            tmp_path / "output",
            "concat",
        )
        + ["--distance_metric", distance_metric]
    )
    plot = output / "figures/comparisons/global/centroid_tie_adjusted"
    assert plot.with_suffix(".png").is_file() and plot.with_suffix(".pdf").is_file()
    visualization.plot_analysis(output)
    assert plot.with_suffix(".pdf").stat().st_size > 1000


@pytest.mark.parametrize("mode", ["concat", "mean_std"])
def test_distance_runs_coexist_without_changing_representations(
    feature_bundle: Path,
    metadata_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """Switching metrics preserves representations and previously published results.

    Args:
        feature_bundle (Path): Frozen source features for the fixture.
        metadata_bundle (Path): Frozen task metadata.
        tmp_path (Path): Dataset and output root.
        monkeypatch (pytest.MonkeyPatch): Skip plotting already covered separately.
        mode (str): Support aggregation defining the task representations.
    """
    monkeypatch.setattr(visualization, "plot_analysis", lambda *args, **kwargs: None)
    args = _arguments(
        tmp_path / "ConceptARC-100-10", metadata_bundle, tmp_path / "output", mode
    )
    args += ["--exclude_feature_groups", "geometry", "color"]
    cosine = run_visual_baseline(args)
    hashes = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in cosine.iterdir()
        if path.is_file()
    }
    euclidean = run_visual_baseline(args + ["--distance_metric", "euclidean"])
    assert euclidean.parent == cosine.parent
    assert cosine.name == "cosine_distance" and euclidean.name == "euclidean_distance"
    assert hashes == {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in cosine.iterdir()
        if path.is_file()
    }
    for name in ("raw_task_means.npy", "task_means.npy", "grand_mean.npy"):
        np.testing.assert_array_equal(np.load(cosine / name), np.load(euclidean / name))
    with (
        np.load(cosine / "preprocessing.npz") as original,
        np.load(euclidean / "preprocessing.npz") as other,
    ):
        assert original.files == other.files
        for name in original.files:
            np.testing.assert_array_equal(original[name], other[name])
    for directory, metric in ((cosine, "cosine"), (euclidean, "euclidean")):
        config = json.loads((directory / "run_config.json").read_text())
        assert config["distance_metric"] == metric
    with pytest.raises(FileExistsError):
        run_visual_baseline(args + ["--distance_metric", "cosine"])
    with pytest.raises(FileExistsError):
        run_visual_baseline(args + ["--distance_metric", "euclidean"])


def test_cli_defaults() -> None:
    """Expose the agreed default and reject ambiguous selections."""
    base = ["--data_dir", "ConceptARC-100-8", "--metadata_dir", "metadata"]
    args = parse_arguments(base)
    assert args.support_aggregation == "concat" and args.population == "all"
    assert args.grid_scope == "both"
    assert args.distance_metric == "cosine"
    assert args.features == list(DEFAULT_FEATURES)
    assert args.run_name == "visual_v2"
    assert (
        parse_arguments(base + ["--distance_metric", "euclidean"]).distance_metric
        == "euclidean"
    )
    assert args.exclude_feature_groups == []
    assert args.exclude_pair_features == []
    base += ["--features", *BASIC_GROUPS]
    assert parse_arguments(
        base + ["--exclude_pair_features", "recolored_fraction", "changed_count"]
    ).exclude_pair_features == ["changed_count", "recolored_fraction"]
    assert parse_arguments(
        base + ["--exclude_feature_groups", "pixel_change", "geometry"]
    ).exclude_feature_groups == ["geometry", "pixel_change"]
    for extra in (
        ["--support_aggregation", "mean"],
        ["--distance_metric", "unknown"],
        ["--run_name", "../bad"],
        ["--num_threads", "0"],
        ["--comparison_scopes", "global", "global"],
        ["--exclude_feature_groups", "unknown"],
        ["--exclude_feature_groups", "geometry", "geometry"],
        ["--exclude_feature_groups", *BASIC_GROUPS],
        ["--grid_scope", "invalid"],
        ["--grid_scope", "input", "--exclude_feature_groups", "pixel_change"],
        ["--grid_scope", "output", "--exclude_feature_groups", "geometry", "color"],
        ["--exclude_pair_features", "unknown"],
        ["--exclude_pair_features", "changed_count", "changed_count"],
        ["--grid_scope", "input", "--exclude_pair_features", "changed_count"],
        [
            "--exclude_feature_groups",
            "pixel_change",
            "--exclude_pair_features",
            "changed_count",
        ],
        [
            "--exclude_feature_groups",
            "geometry",
            "color",
            "--exclude_pair_features",
            *BASIC_PAIR_NAMES[-4:],
        ],
    ):
        with pytest.raises(SystemExit):
            parse_arguments(base + extra)


def test_euclidean_preserves_magnitudes_and_zero_anchors() -> None:
    """A magnitude-sensitive triplet reverses cosine's result; zero anchors stay valid."""
    vectors = np.array([[1.0, 0.0], [10.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    euclidean = separation_similarity(vectors, "euclidean")
    expected = -np.linalg.norm(vectors[:, None] - vectors[None, :], axis=-1)
    np.testing.assert_allclose(euclidean, expected, rtol=0, atol=1e-15)
    assert euclidean[0, 1] == -9 and euclidean[0, 2] == -1
    cosine = separation_similarity(vectors, "cosine")
    assert cosine[0, 1] > cosine[0, 2]
    assert np.isnan(cosine[3]).all() and np.isfinite(euclidean[3]).all()
    np.testing.assert_array_equal(euclidean, euclidean.T)
    np.testing.assert_array_equal(np.diag(euclidean), 0)
    # Translating the same vectors leaves their Euclidean distances unchanged.
    np.testing.assert_array_equal(
        separation_similarity(vectors + 1e12, "euclidean"), euclidean
    )


@pytest.mark.parametrize("width", [0, 1, 4])
def test_identical_euclidean_vectors_receive_half_credit(width: int) -> None:
    """Constant or empty representations retain valid references and receive only ties.

    Args:
        width (int): Number of constant feature coordinates, including none.
    """
    vectors = np.zeros((4, width))
    similarity = separation_similarity(vectors, "euclidean")
    labels = np.array([0, 0, 1, 1])
    membership = labels[:, None] == np.arange(2)
    result = family_separation(
        similarity, eligible_references(similarity, labels, "global"), membership
    )
    np.testing.assert_array_equal(similarity, 0)
    np.testing.assert_array_equal(
        result["scores"][membership], np.tile([0, 0.5], (4, 1))
    )
    np.testing.assert_array_equal(result["tie_counts"][membership], 2)
    np.testing.assert_array_equal(result["reasons"][membership], 0)
    assert np.isnan(separation_similarity(vectors)).all()


def test_scalar_euclidean_distances_and_cosine_compatibility() -> None:
    """Scalar Euclidean retains differences on the same side of zero."""
    scalar = np.array([[-2.0], [0.0], [3.0], [5.0]])
    expected = -np.array([[0, 2, 5, 7], [2, 0, 3, 5], [5, 3, 0, 2], [7, 5, 2, 0]])
    np.testing.assert_array_equal(separation_similarity(scalar, "euclidean"), expected)
    assert separation_similarity(scalar)[2, 3] == 1
    vectors = np.random.default_rng(19).normal(size=(30, 84))
    vectors[0] = 0
    np.testing.assert_array_equal(
        separation_similarity(vectors), cosine_from_gram(vectors @ vectors.T)
    )


def test_invalid_separation_geometry() -> None:
    """Reject malformed vectors and unsupported distance metrics."""
    for invalid in (
        np.ones(4),
        np.ones((0, 2)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
    ):
        with pytest.raises(ValueError, match="finite task vectors"):
            separation_similarity(invalid, "euclidean")
    with pytest.raises(ValueError, match="Unknown distance metric"):
        separation_similarity(np.ones((3, 2)), cast(DistanceMetric, "invalid"))
