"""Brute-force scores, compact reports and matching/feature alignment contracts."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.distance import cdist

from arc_cd.data_prep.load_data import load_concept_arc
from arc_cd.repr_analysis.data import center_versions, load_bundle
from arc_cd.repr_analysis.explorer import pack_support_grid
from arc_cd.repr_analysis.family_metrics import family_separation
from arc_cd.repr_analysis.geometry import (
    DistanceMetric,
    cosine_from_gram,
    separation_similarity,
)
from arc_cd.repr_analysis.matched_design import (
    CONDITIONS,
    build_matching_design,
    matching_summary,
)
from arc_cd.repr_analysis.matched_examples import (
    _draw_triplet,
    _select_triplets,
    save_qualitative_examples,
)
from arc_cd.repr_analysis.matched_features import extract_controls
from arc_cd.repr_analysis.matched_metrics import (
    aggregate_scores,
    evaluate_groups,
)
from arc_cd.repr_analysis.matched_runner import (
    _summaries,
    align_feature_bundle,
    load_centroid_vectors,
    load_matching_design,
    load_visual_vectors,
    main,
    prepare_matches,
)
from arc_cd.repr_analysis.matched_visualization import (
    LABELS,
    VISUAL_LABELS,
    _save,
    plot_matched_analysis,
)
from arc_cd.repr_analysis.results import write_json
from arc_cd.repr_analysis.visual_features import preprocess_task_features
from tests import test_centroid_analysis, test_visual_baseline

centroid_file = test_centroid_analysis.centroid_file
feature_bundle = test_visual_baseline.feature_bundle
metadata_bundle = test_visual_baseline.metadata_bundle
population_sources = test_visual_baseline.population_sources


def toy_design() -> dict[str, np.ndarray]:
    """Create overlapping families and layout/caliper variation.

    Returns:
        dict[str, np.ndarray]: Small model-independent matching design.
    """
    rng = np.random.default_rng(10)
    membership = rng.random((18, 4)) < 0.42
    membership[np.arange(18), np.arange(18) % 4] = True
    layouts = np.asarray(["x", "x", "y"] * 6)
    vectors = {"input": rng.normal(size=(18, 4)), "full": rng.normal(size=(18, 6))}
    return build_matching_design(membership, layouts, vectors, min_tasks=2)


def brute_groups(sim: np.ndarray, design: dict[str, np.ndarray]) -> np.ndarray:
    """Enumerate distinct parents independently of optimized scoring.

    Args:
        sim (np.ndarray): Model similarities.
        design (dict[str, np.ndarray]): Frozen graph.

    Returns:
        np.ndarray: Explicit comparison means per negative-action group.
    """
    membership = design["membership"]
    out = np.full((8, len(design["anchor_tasks"]), membership.shape[1], 2), np.nan)
    for row, (a, f) in enumerate(
        zip(design["anchor_tasks"], design["anchor_families"])
    ):
        if not np.isfinite(sim[a, a]):
            continue
        sl = slice(*design["edge_offsets"][f : f + 2])
        flags = {
            (int(b), int(c)): int(flag)
            for b, c, flag in zip(
                design["edge_b"][sl], design["edge_c"][sl], design["edge_flags"][sl]
            )
        }
        for condition in range(8):
            for g in range(membership.shape[1]):
                comparisons = []
                for b in np.flatnonzero(membership[:, f]):
                    for c in np.flatnonzero(~membership[:, f] & membership[:, g]):
                        if (
                            a == b
                            or a == c
                            or b == c
                            or not np.isfinite(sim[a, b])
                            or not np.isfinite(sim[a, c])
                        ):
                            continue
                        if condition and not (
                            flags.get((b, c), 0) & (1 << (condition - 1))
                        ):
                            continue
                        win, tie = sim[a, b] > sim[a, c], sim[a, b] == sim[a, c]
                        comparisons.append([float(win), float(win) + 0.5 * float(tie)])
                if comparisons:
                    out[condition, row, g] = np.mean(comparisons, axis=0)
    return out


@pytest.mark.parametrize("seed", [11, 42])
def test_group_scores_against_explicit_triplets(seed: int) -> None:
    """Check overlap, reuse, exact ties, zero norms and all conditions.

    Args:
        seed (int): Random geometry seed.
    """
    design = toy_design()
    rng = np.random.default_rng(seed)
    x = rng.integers(-1, 2, (18, 5)).astype(float)
    x[0] = 0
    x[1:4] = x[4]
    sim = cosine_from_gram(x @ x.T)
    actual, original = evaluate_groups(sim, design)
    np.testing.assert_allclose(actual, brute_groups(sim, design), atol=1e-15)
    eligible = np.isfinite(sim) & ~np.eye(18, dtype=bool)
    expected = family_separation(sim, eligible, design["membership"])["scores"]
    np.testing.assert_allclose(
        original, expected[design["anchor_tasks"], design["anchor_families"]]
    )


def test_calipers_layouts_coverage_and_constants() -> None:
    """Calibrate once per family/B and retain every strict-cutoff match."""
    membership = np.array([[1, 0]] * 4 + [[0, 1]] * 4, dtype=bool)
    layouts = np.array(["x", "x", "y", "y"] * 2)
    x = np.array([[0.0], [1.0], [3.0], [5.0], [0.1], [0.2], [4.0], [6.0]])
    design = build_matching_design(membership, layouts, {"input": x, "full": x}, 2)
    distance = cdist(x, x)
    nearest = []
    for f in range(2):
        for b in np.flatnonzero(membership[:, f]):
            c = np.flatnonzero(~membership[:, f] & (layouts == layouts[b]))
            nearest.append(distance[b, c].min())
    expected_cutoffs = np.quantile(nearest, [0.25, 0.5, 0.75])
    np.testing.assert_array_equal(design["cutoffs"][0], expected_cutoffs)
    for f in range(2):
        sl = slice(*design["edge_offsets"][f : f + 2])
        b, c = design["edge_b"][sl], design["edge_c"][sl]
        assert np.all(layouts[b] == layouts[c])
        for q, cutoff in enumerate(expected_cutoffs):
            np.testing.assert_array_equal(
                (design["edge_flags"][sl] & (1 << (q + 1))) != 0,
                distance[b, c] < cutoff,
            )
    prepared = preprocess_task_features(
        np.column_stack((np.ones(8), x[:, 0], 2 * x[:, 0])),
        ("geometry", "geometry", "color"),
    )
    assert prepared["retained"].tolist() == [False, True, True]
    np.testing.assert_array_equal(prepared["block_weights"], [0, 1, 1])
    zero = build_matching_design(
        membership, layouts, {"input": np.empty((8, 0)), "full": np.zeros((8, 2))}, 2
    )
    assert not zero["headline"][2:].any()  # Strict d < 0 intentionally has no matches.
    assert (zero["cutoffs"] == 0).all()


def test_balancing() -> None:
    """Compare equal negative-group, anchor and target-family averages."""
    design = toy_design()
    x = np.random.default_rng(4).normal(size=(18, 5))
    groups, original = evaluate_groups(cosine_from_gram(x @ x.T), design)
    actual = aggregate_scores(groups, original, design)
    for condition in range(8):
        expected_family = []
        for f in np.flatnonzero(design["headline"][condition]):
            anchors = []
            for row in np.flatnonzero(design["anchor_families"] == f):
                valid = np.isfinite(groups[condition, row, :, 0])
                if valid.any():
                    anchors.append(groups[condition, row, valid].mean(axis=0))
            expected_family.append(
                np.mean(anchors, axis=0) if anchors else [np.nan, np.nan]
            )
        if expected_family:
            np.testing.assert_allclose(
                actual["macro"][condition], np.mean(expected_family, axis=0)
            )
    changed = copy.deepcopy(design)
    changed["headline"][2] = True
    broken = groups.copy()
    broken[2, design["anchor_families"] == 0] = np.nan
    assert np.isnan(aggregate_scores(broken, original, changed)["macro"][2]).all()


def test_undefined_geometry() -> None:
    """Keep unavailable scores undefined and reject misaligned similarities."""
    design = toy_design()
    groups, original = evaluate_groups(np.full((18, 18), np.nan), design)
    result = aggregate_scores(groups, original, design)
    assert np.isnan(result["macro"]).all()
    assert not result["valid_anchors"].any()
    assert not result["original_valid_anchors"].any()
    with pytest.raises(ValueError, match="different axes"):
        evaluate_groups(np.eye(17), design)


def test_compact_comparisons_and_valid_coverage(tmp_path: Path) -> None:
    """Report finite matched comparisons and count only usable original anchors.

    Args:
        tmp_path (Path): Portable summary and figure destination.
    """
    design = toy_design()
    x = np.random.default_rng(42).normal(size=(18, 5))
    x[0] = 0
    groups, original = evaluate_groups(cosine_from_gram(x @ x.T), design)
    point = aggregate_scores(groups, original, design)
    axes = {"families": [f"action_{i}" for i in range(4)]}
    summary, families = _summaries(axes, design, point)
    assert summary["original_figure4"]["headline_anchors"] == 17
    matched = [
        summary[name]
        for name in CONDITIONS
        if np.isfinite(summary[name]["tie_adjusted"])
    ]
    assert matched
    for row in matched:
        assert "layout_baseline" not in row
        assert row["difference_from_original"]["tie_adjusted"] == pytest.approx(
            row["tie_adjusted"] - summary["original_figure4"]["tie_adjusted"]
        )
    for name in CONDITIONS:
        for action, row in families[name].items():
            expected = (
                row["tie_adjusted"]
                - families["original_figure4"][action]["tie_adjusted"]
            )
            np.testing.assert_allclose(
                row["difference_from_original"]["tie_adjusted"], expected
            )
    summaries = {"S3": {"0": summary, "2": copy.deepcopy(summary)}}
    # Replotting uses the saved scores, independently of old derived comparisons.
    summaries["S3"]["2"]["original_figure4"]["tie_adjusted"] -= 0.1
    write_json(tmp_path / "summary.json", summaries)
    write_json(tmp_path / "family_summary.json", {"S3": {"0": families, "2": families}})
    write_json(
        tmp_path / "matching_summary.json", matching_summary(design, axes["families"])
    )
    (tmp_path / "run_config.json").write_text(
        json.dumps({"model_name": "fixture", "positions": ["S3"], "layers": [0, 2]})
    )
    with patch("arc_cd.repr_analysis.matched_visualization._save", wraps=_save) as save:
        plots = plot_matched_analysis(tmp_path)
    assert {p.stem for p in plots.glob("*.png")} == {
        "S3_layer_0_comparison",
        "S3_layer_2_comparison",
        "S3_layers",
        "S3_layers_visual_difference",
        "matching_quality",
    }
    for call in save.call_args_list:
        fig = call.args[0]
        assert fig._suptitle is None
        assert all(not panel.get_title() for panel in fig.axes)
    layer_figure = next(
        call.args[0] for call in save.call_args_list if call.args[1].name == "S3_layers"
    )
    difference_figure = next(
        call.args[0]
        for call in save.call_args_list
        if call.args[1].name == "S3_layers_visual_difference"
    )
    assert len(difference_figure.axes) == 1
    for panel, difference, labels in (
        (layer_figure.axes[0], False, LABELS),
        (layer_figure.axes[1], True, {name: LABELS[name] for name in CONDITIONS}),
        (difference_figure.axes[0], True, VISUAL_LABELS),
    ):
        curves = {
            line.get_label(): line
            for line in panel.lines
            if line.get_label() in labels.values()
        }
        assert set(curves) == set(labels.values())
        for name, label in labels.items():
            curve = curves[label]
            expected = []
            for layer in (0, 2):
                values = summaries["S3"][str(layer)]
                score = values[name]["tie_adjusted"]
                if difference:
                    score -= values["original_figure4"]["tie_adjusted"]
                expected.append(100 * score)
            np.testing.assert_array_equal(curve.get_xdata(), [0, 2])
            np.testing.assert_allclose(curve.get_ydata(), expected)


def test_prepare_evaluate_round_trip(
    feature_bundle: Path,
    metadata_bundle: Path,
    tmp_path: Path,
) -> None:
    """Exercise task alignment, model-independent design, JSON reports and replotting.

    Args:
        feature_bundle (Path): Complete synthetic sixteen-concept feature bundle.
        metadata_bundle (Path): Frozen taxonomy annotations.
        tmp_path (Path): Isolated outputs.
    """
    bundle = load_bundle(feature_bundle)
    data_dir = Path(bundle.config["runtime"]["data_dir"])
    destination = tmp_path / "matches"
    destination.mkdir()
    (destination / "obsolete.json").write_text("old matching output")
    prepare_matches(
        data_dir, metadata_bundle, destination, min_tasks=1, expected_parents=32
    )
    config, axes, design = load_matching_design(destination)
    np.testing.assert_array_equal(
        align_feature_bundle(bundle, design, axes), np.arange(32)
    )
    reordered = copy.deepcopy(bundle)
    reordered.task_ids = bundle.task_ids[::-1]
    reordered.rows = bundle.rows[::-1]
    reordered.labels = bundle.labels[::-1]
    np.testing.assert_array_equal(
        align_feature_bundle(reordered, design, axes), np.arange(32)[::-1]
    )
    wrong_versions = copy.deepcopy(bundle)
    wrong_versions.rows[0] = wrong_versions.rows[0, ::-1]
    with pytest.raises(ValueError, match="support-set identities"):
        align_feature_bundle(wrong_versions, design, axes)
    assert {p.name for p in destination.iterdir()} == {
        "run_config.json",
        "axes.json",
        "matches.npz",
        "matching_summary.json",
    }
    bookkeeping = {
        "schema_version",
        "taxonomy_version",
        "source_digest",
        "support_digest",
        "version_digest",
        "metadata_fingerprints",
        "artifacts",
        "software",
    }
    assert bookkeeping.isdisjoint(config)
    (destination / "run_config.json").write_text(
        json.dumps(
            {
                **config,
                "data_dir": str(tmp_path / "previous-data-location"),
                "metadata_dir": str(tmp_path / "previous-metadata-location"),
            }
        )
    )
    before = (destination / "matches.npz").read_bytes()
    output = tmp_path / "results"
    output.mkdir()
    (output / "obsolete.json").write_text("old model output")
    main(
        [
            "evaluate",
            "--match_dir",
            str(destination),
            "--feature_dir",
            str(feature_bundle),
            "--output_dir",
            str(output),
            "--layers",
            "all",
            "--num_threads",
            "1",
            "--qualitative_examples",
            "--data_dir",
            str(data_dir),
            "--metadata_dir",
            str(metadata_bundle),
            "--num_qualitative_examples",
            "2",
        ]
    )
    assert (destination / "matches.npz").read_bytes() == before
    assert (output / "figures/S3_layer_0_comparison.pdf").is_file()
    assert (output / "figures/S3_layer_2_comparison.pdf").is_file()
    assert (output / "figures/S3_layers.pdf").is_file()
    main(
        [
            "plot",
            "--analysis_dir",
            str(output),
            "--output_dir",
            str(tmp_path / "replot"),
        ]
    )
    assert (tmp_path / "replot/S3_layer_0_comparison.png").is_file()
    assert {p.name for p in output.iterdir()} == {
        "run_config.json",
        "summary.json",
        "family_summary.json",
        "matching_summary.json",
        "figures",
        "qualitative_examples",
    }
    assert not list(output.rglob("*.npz"))
    assert not list(output.rglob("*.csv"))
    examples = json.loads((output / "qualitative_examples/summary.json").read_text())
    assert examples["requested_examples_per_action"] == 2
    assert set(examples["conditions"]) == {"full_q25", "full_q50", "full_q75"}
    assert bookkeeping.isdisjoint(json.loads((output / "run_config.json").read_text()))
    summaries = json.loads((output / "summary.json").read_text())
    assert set(summaries["S3"]) == {"0", "2"}
    for layer in summaries["S3"].values():
        assert set(layer) == {"original_figure4", *CONDITIONS}
        assert all(
            "layout_baseline" not in entry and "difference" not in entry
            for entry in layer.values()
        )
        for name in CONDITIONS:
            row = layer[name]
            for metric in ("strict_win", "tie_adjusted"):
                if row[metric] is not None:
                    assert row["difference_from_original"][metric] == pytest.approx(
                        row[metric] - layer["original_figure4"][metric]
                    )
    assert (output / "matching_summary.json").read_bytes() == (
        destination / "matching_summary.json"
    ).read_bytes()
    # A different model name, layer, and state values reuse exactly the same design.
    other = tmp_path / "other_model"
    shutil.copytree(feature_bundle, other)
    other_config = json.loads((other / "run_config.json").read_text())
    other_config["extraction"]["model_name"] = "another_model"
    (other / "run_config.json").write_text(json.dumps(other_config))
    with np.load(other / "S3.npz") as archive:
        states = {key: archive[key] for key in archive.files}
    states["x"] = (
        np.random.default_rng(42).normal(size=states["x"].shape).astype(np.float16)
    )
    np.savez_compressed(other / "S3.npz", **states)
    main(
        [
            "evaluate",
            "--match_dir",
            str(destination),
            "--feature_dir",
            str(other),
            "--output_dir",
            str(tmp_path / "other_results"),
            "--layers",
            "2",
            "--num_threads",
            "1",
        ]
    )
    assert (destination / "matches.npz").read_bytes() == before
    changed = copy.deepcopy(bundle)
    changed.labels[0] = (changed.labels[0] + 1) % len(changed.concepts)
    with pytest.raises(ValueError, match="concept labels"):
        align_feature_bundle(changed, design, axes)
    changed = copy.deepcopy(bundle)
    changed.task_ids[0] = "unknown/task"
    with pytest.raises(ValueError, match="parent identities"):
        align_feature_bundle(changed, design, axes)
    incompatible = tmp_path / "incompatible_metadata"
    shutil.copytree(metadata_bundle, incompatible)
    metadata_axes = json.loads((incompatible / "axes.json").read_text())
    metadata_axes["labels"][0] = (metadata_axes["labels"][0] + 1) % len(
        metadata_axes["concepts"]
    )
    write_json(incompatible / "axes.json", metadata_axes)
    with pytest.raises(ValueError, match="concept labels"):
        prepare_matches(
            data_dir, incompatible, tmp_path / "rejected", expected_parents=32
        )
    assert not (tmp_path / "rejected").exists()
    # Source-file formatting does not affect scientific results.
    source_path = next(data_dir.glob("*/*.json"))
    original_source = source_path.read_text()
    try:
        source_path.write_text(original_source + "\n")
        destination2 = tmp_path / "reformatted_source"
        prepare_matches(
            data_dir, metadata_bundle, destination2, min_tasks=1, expected_parents=32
        )
        _, _, repeated = load_matching_design(destination2)
        for key in design:
            np.testing.assert_array_equal(design[key], repeated[key])
        changed_task = json.loads(original_source)
        grid = changed_task["train"][0]["output"]
        grid[0][0] = (grid[0][0] + 1) % 10
        source_path.write_text(json.dumps(changed_task))
        with pytest.raises(ValueError, match="Support grids differ"):
            prepare_matches(
                data_dir,
                metadata_bundle,
                destination2,
                min_tasks=1,
                expected_parents=32,
            )
        _, _, preserved = load_matching_design(destination2)
        np.testing.assert_array_equal(design["edge_flags"], preserved["edge_flags"])
    finally:
        source_path.write_text(original_source)
    for key, invalid in (
        ("families", axes["families"][:-1]),
        ("labels", axes["labels"][:-1]),
        ("labels", [-1, *axes["labels"][1:]]),
        ("labels", [len(axes["concepts"]), *axes["labels"][1:]]),
    ):
        broken_axes = copy.deepcopy(axes)
        broken_axes[key] = invalid
        write_json(destination / "axes.json", broken_axes)
        with pytest.raises(ValueError, match="axes are inconsistent"):
            load_matching_design(destination)


@pytest.mark.parametrize("distance_metric", ["cosine", "euclidean"])
def test_visual_evaluation(distance_metric: DistanceMetric, tmp_path: Path) -> None:
    """Check cached-vector alignment and all nine scores against explicit triplets.

    Args:
        distance_metric (DistanceMetric): Visual similarity measure.
        tmp_path (Path): Isolated matching, visual-baseline and result directories.
    """
    design = toy_design()
    task_ids = [f"Toy/Toy{i}" for i in range(18)]
    axes: dict[str, Any] = {
        "task_ids": task_ids,
        "concepts": ["Toy"],
        "labels": [0] * 18,
        "families": [f"action_{i}" for i in range(4)],
        "conditions": CONDITIONS,
    }
    design["version_task_ids"] = np.asarray(
        [[f"{task}_{i:03d}" for i in range(1, 9)] for task in task_ids]
    )
    design["support_set_ids"] = np.asarray([[f"{i:03d}" for i in range(1, 9)]] * 18)
    matches, visual = tmp_path / "matches", tmp_path / "visual"
    matches.mkdir()
    visual.mkdir()
    write_json(matches / "run_config.json", {"min_tasks": 2})
    write_json(matches / "axes.json", axes)
    write_json(
        matches / "matching_summary.json", matching_summary(design, axes["families"])
    )
    np.savez_compressed(matches / "matches.npz", **design)
    rng = np.random.default_rng(31)
    vectors = preprocess_task_features(
        rng.normal(size=(18, 6)), ("geometry",) * 3 + ("color",) * 3
    )["vectors"]
    prepared_order, raw_order = rng.permutation(18), rng.permutation(18)
    np.save(visual / "task_means.npy", vectors[prepared_order])
    write_json(
        visual / "axes.json",
        {
            "task_ids": [task_ids[i] for i in prepared_order],
            "concepts": ["Toy"],
            "labels": [0] * 18,
        },
    )
    identities = {
        "source_task_id": np.asarray(task_ids)[raw_order],
        **{
            key: design[key][raw_order]
            for key in ("version_task_ids", "support_set_ids")
        },
    }
    np.savez_compressed(visual / "features.npz", **identities)
    write_json(
        visual / "run_config.json",
        {
            "input_kind": "visual_features",
            "distance_metric": "cosine",
            "grid_scope": "both",
            "feature_groups": {"included": ["geometry", "color"]},
            "excluded_pair_features": [],
            "arguments": {"support_aggregation": "concat"},
        },
    )
    _, aligned = load_visual_vectors(visual, design, axes)
    np.testing.assert_array_equal(aligned, vectors)
    output = tmp_path / "results"
    output.mkdir()
    (output / "obsolete.json").write_text("old visual output")
    argv = [
        "evaluate-visual",
        "--match_dir",
        str(matches),
        "--visual_dir",
        str(visual),
        "--output_dir",
        str(output),
        "--num_threads",
        "1",
    ]
    if distance_metric == "euclidean":
        argv += ["--distance_metric", distance_metric]
    main(argv)
    similarity = separation_similarity(vectors, distance_metric)
    original = family_separation(
        similarity,
        np.isfinite(similarity) & ~np.eye(18, dtype=bool),
        design["membership"],
    )["scores"]
    expected = aggregate_scores(
        brute_groups(similarity, design),
        original[design["anchor_tasks"], design["anchor_families"]],
        design,
    )
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary) == {"original_figure4", *CONDITIONS}
    for i, name in enumerate(["original_figure4", *CONDITIONS]):
        scores = expected["original_macro"] if i == 0 else expected["macro"][i - 1]
        np.testing.assert_allclose(
            [summary[name][metric] for metric in ("strict_win", "tie_adjusted")], scores
        )
    config = json.loads((output / "run_config.json").read_text())
    assert config["distance_metric"] == distance_metric
    assert "positions" not in config and "layers" not in config
    assert {path.name for path in output.iterdir()} == {
        "summary.json",
        "family_summary.json",
        "matching_summary.json",
        "run_config.json",
        "figures",
    }
    assert {path.stem for path in (output / "figures").glob("*.png")} == {
        "visual_comparison",
        "matching_quality",
    }
    assert (output / "figures/visual_comparison.pdf").is_file()
    main(
        [
            "plot",
            "--analysis_dir",
            str(output),
            "--output_dir",
            str(tmp_path / "replot"),
        ]
    )
    assert (tmp_path / "replot/visual_comparison.png").is_file()
    np.testing.assert_array_equal(
        np.load(visual / "task_means.npy"), vectors[prepared_order]
    )
    identities["support_set_ids"][0, 0] = "999"
    np.savez_compressed(visual / "features.npz", **identities)
    with pytest.raises(ValueError, match="support-set identities"):
        load_visual_vectors(visual, design, axes)


def test_feature_order_and_dimension_rejection(feature_bundle: Path) -> None:
    """Reject undefined pixel alignment and preserve identity under source reordering.

    Args:
        feature_bundle (Path): Synthetic equal-grid source fixture.
    """
    from dataclasses import replace

    from arc_cd.data_prep import GridPair
    from arc_cd.data_prep.load_data import load_concept_arc

    config = json.loads((feature_bundle / "run_config.json").read_text())
    tasks = load_concept_arc(config["runtime"]["data_dir"])
    first, reordered = extract_controls(tasks), extract_controls(tasks[::-1])
    for key in first:
        np.testing.assert_array_equal(first[key], reordered[key])
    bad = replace(
        tasks[0], train=(GridPair(input=[[1]], output=[[1, 2]]), *tasks[0].train[1:])
    )
    with pytest.raises(ValueError, match="equal input/output"):
        extract_controls((bad, *tasks[1:]))
    with pytest.raises(ValueError, match="eight distinct"):
        extract_controls(tasks[1:])


def test_hand_counted_profiles() -> None:
    """Check every coordinate, saved support order, version means, and block weights."""
    from arc_cd.data_prep import GridPair
    from arc_cd.data_prep.concept_arc import ConceptARCTask

    tasks, expected = [], []
    for parent in range(1, 4):
        versions = []
        for version in range(1, 9):
            supports, pairs = [], []
            for support in range(3):
                height, width = parent + 1, support + 3
                source = np.zeros((height, width), dtype=int)
                source.flat[:version] = parent
                target = source.copy()
                target.flat[-1] = 9
                supports.append(GridPair(input=source.tolist(), output=target.tolist()))
                hist = np.zeros(10)
                area = height * width
                filled = min(version, area)
                hist[0], hist[parent] = area - filled, filled
                output_hist = hist.copy()
                output_hist[source.flat[-1]] -= 1
                output_hist[9] += 1
                pairs.append(
                    np.r_[
                        [height, width, area, width / height] * 2,
                        hist / area,
                        output_hist / area,
                        [1, 1 / area],
                    ]
                )
            name = f"Toy{parent}_{version:03d}"
            tasks.append(
                ConceptARCTask(
                    "Toy", name, Path(f"Toy/{name}.json"), tuple(supports), ()
                )
            )
            versions.append(pairs)
        expected.append(versions)
    arrays = extract_controls(tasks)
    full = np.asarray(expected)
    input_columns = [*range(4), *range(8, 18)]
    for profile, values in (("full", full), ("input", full[..., input_columns])):
        raw = values.reshape(3, 8, -1).mean(axis=1)
        np.testing.assert_array_equal(arrays[f"{profile}_task_features"], raw)
        retained = np.ptp(raw, axis=0) > 0
        blocks = arrays[f"{profile}_feature_blocks"]
        weights = np.zeros(raw.shape[1])
        for block in set(blocks):
            mask = retained & (blocks == block)
            if mask.any():
                weights[mask] = 1 / np.sqrt(mask.sum())
        expected_vectors = (raw[:, retained] - raw.mean(axis=0)[retained]) / raw.std(
            axis=0
        )[retained]
        expected_vectors *= weights[retained]
        np.testing.assert_array_equal(arrays[f"{profile}_retained"], retained)
        np.testing.assert_allclose(arrays[f"{profile}_block_weights"], weights)
        np.testing.assert_allclose(arrays[f"{profile}_vectors"], expected_vectors)


def test_qualitative_triplet_selection() -> None:
    """Select near band midpoints with diverse anchor operators and valid negatives."""
    membership = np.array([[1, 0]] * 4 + [[0, 1]] * 4 + [[0, 0]], dtype=bool)
    operators = np.array(["remove", "remove", "fill", "translate"] + ["other"] * 5)
    distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.79, 0.8])
    cutoffs = np.array([[0.35, 0.55, 0.8]] * 2)
    flags = np.ones(9, dtype=np.uint8)
    for q in range(3):
        flags |= (distances < cutoffs[1, q]).astype(np.uint8) << (4 + q)
    design = {
        "membership": membership,
        "edge_b": np.array([0, 0, 1, 2, 3, 3, 2, 0, 1]),
        "edge_c": np.array([4, 5, 5, 6, 7, 4, 5, 8, 7]),
        "edge_offsets": np.array([0, 9, 9]),
        "edge_flags": flags,
        "cutoffs": cutoffs,
        "headline": np.array([[True, False]] * 8),
    }
    pair_distances = dict(zip(zip(design["edge_b"], design["edge_c"]), distances))
    for q, expected in enumerate(
        ([0.2, 0.1, 0.3], [0.4, 0.5, 0.3, 0.1, 0.2], [0.7, 0.6, 0.4, 0.5, 0.3])
    ):
        triplets = _select_triplets(design, distances, 0, q, operators, 5)
        assert triplets == _select_triplets(design, distances, 0, q, operators, 5)
        assert len(triplets) == len(expected)
        assert len({(b, c) for _, b, c in triplets}) == len(expected)
        assert len({a for a, _, _ in triplets}) == min(4, len(expected))
        assert len({operators[a] for a, _, _ in triplets}) == 3
        assert [pair_distances[b, c] for _, b, c in triplets] == expected
        for a, b, c in triplets:
            assert len({a, b, c}) == 3
            assert membership[a, 0] and membership[b, 0]
            assert not membership[c, 0] and membership[c].any()
            assert pair_distances[b, c] < cutoffs[1, q]
        assert _select_triplets(design, distances, 1, q, operators, 5) == []
    all_triplets = _select_triplets(design, distances, 0, 2, operators, 10)
    assert len(all_triplets) == 7
    for num_examples in (1, 2, 5):
        assert (
            _select_triplets(design, distances, 0, 2, operators, num_examples)
            == all_triplets[:num_examples]
        )
    single_operator = _select_triplets(design, distances, 0, 0, np.full(9, "same"), 5)
    assert len({a for a, _, _ in single_operator}) == 3
    design["edge_flags"][1:] = 1
    assert len(_select_triplets(design, distances, 0, 0, operators, 10)) == 1


@pytest.mark.parametrize("num_examples", [2, 10])
def test_qualitative_support_export(tmp_path: Path, num_examples: int) -> None:
    """Export matching PNG/JSON examples with exact first-variant grids and stats.

    Args:
        tmp_path (Path): Temporary dataset and example outputs.
        num_examples (int): Requested examples per action and condition.
    """
    dataset = tmp_path / "dataset"
    (dataset / "Toy").mkdir(parents=True)
    rng = np.random.default_rng(23)
    for parent in range(24):
        for version in range(1, 9):
            supports = []
            for support in range(3):
                source = rng.integers(0, 10, (3 + parent % 3, 4 + support))
                target = source.copy()
                target[0, 0] = (source[0, 0] + version) % 10
                supports.append({"input": source.tolist(), "output": target.tolist()})
            write_json(
                dataset / "Toy" / f"Toy{parent}_{version:03d}.json",
                {"train": supports, "test": [{"input": [[0]], "output": [[1]]}]},
            )
    tasks = load_concept_arc(dataset)
    controls = extract_controls(tasks)
    order = np.arange(24)[::-1]
    axes = {
        "task_ids": controls["source_task_id"][order].tolist(),
        "families": ["action_0", "action_1"],
    }
    vectors = {p: controls[f"{p}_vectors"][order] for p in ("input", "full")}
    design = build_matching_design(
        np.eye(2, dtype=bool)[np.arange(24) % 2], np.zeros(24), vectors, min_tasks=2
    )
    design.update(
        {key: controls[key][order] for key in ("version_task_ids", "support_set_ids")}
    )
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    operators = [f"operator_{i // 2 % 5}" for i in range(24)]
    records = [
        {
            "task_id": task,
            "root_ops": {"action": operators[i]},
            "families": {"action": [axes["families"][i % 2]]},
        }
        for i, task in enumerate(axes["task_ids"])
    ]
    write_json(metadata / "task_metadata.json", {"records": records[::-1]})
    write_json(
        metadata / "supports.json",
        [
            {
                "task_id": task.task_id,
                "supports": [
                    {
                        "input": pack_support_grid(pair.input, task.task_id),
                        "output": pack_support_grid(pair.output, task.task_id),
                    }
                    for pair in task.train
                    if pair.output is not None
                ],
            }
            for task in tasks
        ],
    )
    output = tmp_path / "examples"
    with patch(
        "arc_cd.repr_analysis.matched_examples._draw_triplet", wraps=_draw_triplet
    ) as renderer:
        save_qualitative_examples(
            dataset,
            axes,
            design,
            output,
            metadata_dir=metadata,
            num_examples=num_examples,
        )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["requested_examples_per_action"] == num_examples
    by_id = {task.task_id: task for task in tasks}
    first_ids = design["version_task_ids"][:, 0].tolist()
    full_distances = cdist(vectors["full"], vectors["full"])
    count = 0
    for q, condition in enumerate(("full_q25", "full_q50", "full_q75")):
        entry = summary["conditions"][condition]
        eligible = np.flatnonzero(design["headline"][5 + q])
        assert set(entry["actions"]) == {axes["families"][f] for f in eligible}
        expected_band = [
            design["cutoffs"][1, q - 1] if q else 0,
            design["cutoffs"][1, q],
        ]
        assert entry["distance_band"] == expected_band
        assert entry["target_distance"] == pytest.approx(np.mean(expected_band))
        for action, records in entry["actions"].items():
            family = axes["families"].index(action)
            assert len(records) == min(
                num_examples, design["pair_counts"][5 + q, family]
            )
            assert len(
                {record["action_operators"]["anchor"] for record in records}
            ) == min(5, len(records))
            assert len({record["anchor"] for record in records}) == len(records)
            for number, record in enumerate(records, start=1):
                a, b, c = [
                    first_ids.index(record[role])
                    for role in ("anchor", "positive", "negative")
                ]
                assert len({a, b, c}) == 3
                assert record["distance"] == pytest.approx(full_distances[b, c])
                assert record["distance"] < entry["distance_band"][1]
                assert record["action_operators"] == dict(
                    zip(
                        ("anchor", "positive", "negative"),
                        (operators[a], operators[b], operators[c]),
                    )
                )
                image_path = output / condition / action / f"example_{number:02d}.png"
                with Image.open(image_path) as image:
                    assert image.width >= 1230 and image.height > 300
                saved = json.loads(image_path.with_suffix(".json").read_text())
                assert saved["action_group"] == action
                assert saved["condition"] == condition
                assert saved["stats"] == {
                    "positive_negative_distance": record["distance"],
                    "distance_band": entry["distance_band"],
                    "target_distance": entry["target_distance"],
                    "within_band": entry["distance_band"][0]
                    <= record["distance"]
                    < entry["distance_band"][1],
                }
                for role in ("anchor", "positive", "negative"):
                    task = by_id[record[role]]
                    assert saved[role] == {
                        "task_id": task.task_id,
                        "action_operator": record["action_operators"][role],
                        "support_examples": [
                            {"input": pair.input, "output": pair.output}
                            for pair in task.train
                        ],
                    }
                count += 1
    assert count == renderer.call_count > 0
    assert len(list(output.rglob("example_*.png"))) == count
    assert len(list(output.rglob("example_*.json"))) == count
    assert list(output.rglob(f"example_{num_examples:02d}.json"))
    for call in renderer.call_args_list:
        assert all(task == by_id[task.task_id] for task in call.args[0])
        assert all(task.task_id.endswith("_001") for task in call.args[0])
        assert call.args[5] == tuple(
            operators[first_ids.index(task.task_id)] for task in call.args[0]
        )
    stale = {**design, "edge_flags": design["edge_flags"].copy()}
    stale["edge_flags"][0] ^= 1 << 4
    with pytest.raises(ValueError, match="no longer reproduce"):
        save_qualitative_examples(
            dataset, axes, stale, tmp_path / "stale", metadata_dir=metadata
        )
    mismatched = {**design, "support_set_ids": design["support_set_ids"].copy()}
    mismatched["support_set_ids"][0, 0] = "999"
    with pytest.raises(ValueError, match="support-set identities"):
        save_qualitative_examples(
            dataset, axes, mismatched, tmp_path / "mismatched", metadata_dir=metadata
        )
    annotations = json.loads((metadata / "task_metadata.json").read_text())
    annotations["records"][0]["families"]["action"] = []
    write_json(metadata / "task_metadata.json", annotations)
    with pytest.raises(ValueError, match="action annotations differ"):
        save_qualitative_examples(
            dataset, axes, design, tmp_path / "wrong_actions", metadata_dir=metadata
        )


@pytest.mark.parametrize("distance_metric", ["cosine", "euclidean"])
def test_centroid_evaluation(
    feature_bundle: Path,
    metadata_bundle: Path,
    centroid_file: Path,
    tmp_path: Path,
    distance_metric: DistanceMetric,
) -> None:
    """Match shuffled, aliased task centroids to the centered support-state scores.

    Args:
        feature_bundle (Path): Original eight-version hidden-state fixture.
        metadata_bundle (Path): Frozen task annotations.
        centroid_file (Path): Shuffled means with VARC concept aliases.
        tmp_path (Path): Isolated matching and result directories.
        distance_metric (DistanceMetric): Cosine or Euclidean evaluation.
    """
    bundle = load_bundle(feature_bundle)
    matches = prepare_matches(
        Path(bundle.config["runtime"]["data_dir"]),
        metadata_bundle,
        tmp_path / "matches",
        min_tasks=1,
        expected_parents=32,
    )
    _, axes, design = load_matching_design(matches)
    raw = bundle.load_position("S3")[:, bundle.layers.index(2)]
    _, _, means = center_versions(raw, bundle.rows)
    np.testing.assert_allclose(load_centroid_vectors(centroid_file, axes), means)
    before = (matches / "matches.npz").read_bytes()
    output = tmp_path / "results"
    argv = [
        "evaluate-centroids",
        "--match_dir",
        str(matches),
        "--centroid_file",
        str(centroid_file),
        "--model_name",
        "varc",
        "--output_dir",
        str(output),
        "--num_threads",
        "1",
    ]
    if distance_metric == "euclidean":
        argv += ["--distance_metric", distance_metric]
    main(argv)
    similarity = separation_similarity(means, distance_metric)
    original = family_separation(
        similarity,
        np.isfinite(similarity) & ~np.eye(len(means), dtype=bool),
        design["membership"],
    )["scores"]
    expected = aggregate_scores(
        brute_groups(similarity, design),
        original[design["anchor_tasks"], design["anchor_families"]],
        design,
    )
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary) == {"original_figure4", *CONDITIONS}
    for i, name in enumerate(["original_figure4", *CONDITIONS]):
        scores = expected["original_macro"] if i == 0 else expected["macro"][i - 1]
        observed = [
            np.nan if summary[name][metric] is None else summary[name][metric]
            for metric in ("strict_win", "tie_adjusted")
        ]
        np.testing.assert_allclose(observed, scores)
    assert (matches / "matches.npz").read_bytes() == before
    config = json.loads((output / "run_config.json").read_text())
    assert config["input_kind"] == "task_centroids"
    assert config["model_name"] == "varc"
    assert config["distance_metric"] == distance_metric
    assert "positions" not in config and "layers" not in config
    assert not list(output.rglob("*.csv"))
    assert {path.stem for path in (output / "figures").glob("*.png")} == {
        "centroid_comparison",
        "matching_quality",
    }
    assert (output / "figures/centroid_comparison.pdf").is_file()
    main(
        [
            "plot",
            "--analysis_dir",
            str(output),
            "--output_dir",
            str(tmp_path / "replot"),
        ]
    )
    assert (tmp_path / "replot/centroid_comparison.png").is_file()
    wrong_axes = copy.deepcopy(axes)
    wrong_axes["task_ids"][0] = "unknown/task"
    with pytest.raises(ValueError, match="complete matching population"):
        load_centroid_vectors(centroid_file, wrong_axes)
    wrong_axes = copy.deepcopy(axes)
    wrong_axes["labels"][0] = (wrong_axes["labels"][0] + 1) % len(axes["concepts"])
    with pytest.raises(ValueError, match="concept labels"):
        load_centroid_vectors(centroid_file, wrong_axes)



@pytest.mark.parametrize(
    "command", ["evaluate", "evaluate-visual", "evaluate-centroids"]
)
@pytest.mark.parametrize("count", [None, "3", "0", "-1"])
def test_qualitative_count_cli(command: str, count: str | None, tmp_path: Path) -> None:
    """Parse the default/custom example count and reject nonpositive counts early.

    Args:
        command (str): Hidden-state, visual, or centroid evaluation command.
        count (str | None): Explicit count, or None to test the default.
        tmp_path (Path): Placeholder input/output paths; no data is loaded.
    """
    source_option = {
        "evaluate": "--feature_dir",
        "evaluate-visual": "--visual_dir",
        "evaluate-centroids": "--centroid_file",
    }[command]
    argv = [
        command,
        "--match_dir",
        str(tmp_path),
        source_option,
        str(tmp_path),
        "--output_dir",
        str(tmp_path / "results"),
        "--qualitative_examples",
    ]
    if command == "evaluate-centroids":
        argv += ["--model_name", "varc"]
    if count is not None:
        argv += ["--num_qualitative_examples", count]
    function = (
        "evaluate_matches" if command == "evaluate" else "evaluate_vector_matches"
    )
    with patch(f"arc_cd.repr_analysis.matched_runner.{function}") as evaluate:
        if count is not None and int(count) < 1:
            with pytest.raises(SystemExit) as error:
                main(argv)
            assert error.value.code == 2
            evaluate.assert_not_called()
        else:
            main(argv)
            evaluate.assert_called_once()
            args = evaluate.call_args.args[0]
            assert args.qualitative_examples
            assert args.num_qualitative_examples == (
                10 if count is None else int(count)
            )


def test_ten_distinct_parents_per_role() -> None:
    """Reject nine B or C parents even with enough anchors and many reused references."""
    membership = np.array([[1, 0]] * 12 + [[0, 1]] * 12, dtype=bool)
    vectors = {
        profile: np.arange(24, dtype=float)[:, None] for profile in ("input", "full")
    }
    for positives, negatives, included in (
        (9, 12, False),
        (12, 9, False),
        (10, 10, True),
    ):
        layouts = np.array([f"unique_{i}" for i in range(24)])
        layouts[:positives] = "shared"
        layouts[12 : 12 + negatives] = "shared"
        design = build_matching_design(membership, layouts, vectors)
        np.testing.assert_array_equal(
            design["role_counts"][1, 0], [12, positives, negatives]
        )
        assert bool(design["headline"][1, 0]) == included
        assert design["pair_counts"][1, 0] == positives * negatives
        # The remaining anchors have unique layouts but are eligible nonetheless.
        anchors = design["anchor_tasks"][design["anchor_families"] == 0]
        assert len(anchors) == 12


@pytest.mark.parametrize("case", ["sparse", "universal", "untagged", "unique_layouts"])
def test_sparse_populations_against_triplets(case: str) -> None:
    """Check sparse-family coverage and scores using explicit valid triplets.

    Args:
        case (str): Degenerate membership or layout configuration.
    """
    rng = np.random.default_rng(7)
    membership = rng.random((10, 5)) < 0.4
    membership[:, :2] = False
    membership[1, 1] = True  # Empty and singleton target families.
    membership[0] = False  # Untagged negatives belong only to the original baseline.
    if case == "universal":
        membership[:, -1] = True
    elif case == "untagged":
        membership[:] = False
    layouts = np.arange(10) if case == "unique_layouts" else np.arange(10) % 3
    vectors = {p: rng.normal(size=(10, 3)) for p in ("input", "full")}
    design = build_matching_design(membership, layouts, vectors, min_tasks=2)
    distances = [cdist(vectors[p], vectors[p]) for p in ("input", "full")]
    for condition in range(len(CONDITIONS)):
        for f in range(membership.shape[1]):
            members = np.flatnonzero(membership[:, f])
            negatives = np.flatnonzero(~membership[:, f] & membership.any(axis=1))
            triplets = []
            for a in members:
                for b in members:
                    for c in negatives:
                        if a == b or (condition and layouts[b] != layouts[c]):
                            continue
                        if condition >= 2:
                            p, q = divmod(condition - 2, 3)
                            if not distances[p][b, c] < design["cutoffs"][p, q]:
                                continue
                        triplets.append((a, b, c))
            expected_roles = [len({t[i] for t in triplets}) for i in range(3)]
            np.testing.assert_array_equal(
                design["role_counts"][condition, f], expected_roles
            )
            assert design["pair_counts"][condition, f] == len(
                {(b, c) for _, b, c in triplets}
            )
            assert design["headline"][condition, f] == all(
                count >= 2 for count in expected_roles
            )
    x = rng.integers(-1, 2, (10, 4)).astype(float)
    x[2] = 0
    sim = cosine_from_gram(x @ x.T)
    groups, original = evaluate_groups(sim, design)
    np.testing.assert_allclose(groups, brute_groups(sim, design), atol=1e-15)
    expected = family_separation(
        sim, np.isfinite(sim) & ~np.eye(10, dtype=bool), membership
    )["scores"]
    np.testing.assert_allclose(
        original, expected[design["anchor_tasks"], design["anchor_families"]]
    )
    point = aggregate_scores(groups, original, design)
    assert not np.isfinite(point["macro"][~design["headline"].any(axis=1)]).any()


@pytest.mark.parametrize("option", ["--n_bootstrap", "--confidence_level", "--seed"])
def test_removed_bootstrap_options(option: str, tmp_path: Path) -> None:
    """Reject removed options instead of silently ignoring requested uncertainty.

    Args:
        option (str): Removed command-line option.
        tmp_path (Path): Placeholder paths; no inputs should be loaded.
    """
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--match_dir",
                str(tmp_path),
                "--feature_dir",
                str(tmp_path),
                "--output_dir",
                str(tmp_path / "results"),
                option,
                "42",
            ]
        )
    assert error.value.code == 2
