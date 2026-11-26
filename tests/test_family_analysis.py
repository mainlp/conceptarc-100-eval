"""Reference calculations, provenance failures, and metadata-run round trips."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arc_cd.repr_analysis.cosine_distance import separation_probability
from arc_cd.repr_analysis.data import center_versions, fingerprint, load_bundle
from arc_cd.repr_analysis.explorer import save_explorer_supports
from arc_cd.repr_analysis.family_data import family_comparison_mask, load_family_data
from arc_cd.repr_analysis.family_metrics import (
    aggregate_families,
    eligible_references,
    family_separation,
)
from arc_cd.repr_analysis.geometry import (
    DISTANCE_METRICS,
    DistanceMetric,
    cosine_from_gram,
    separation_similarity,
)
from arc_cd.repr_analysis.results import save_axes, write_json
from arc_cd.repr_analysis.runner import parse_arguments, run_analysis
from arc_cd.repr_analysis.spec_metadata import (
    DEFAULT_TAXONOMY,
    load_spec_metadata,
    load_taxonomy,
)
from scripts.repr_analysis.plot_analysis import main as plot_main
from tests.test_repr_analysis import feature_bundle as _feature_bundle

feature_bundle = _feature_bundle


@pytest.mark.parametrize("scope", ["global", "cross_concept"])
@pytest.mark.parametrize("seed", [0, 13, 42])
@pytest.mark.parametrize("distance_metric", DISTANCE_METRICS)
def test_metrics_against_brute_force(
    scope: str, seed: int, distance_metric: DistanceMetric
) -> None:
    """Compare overlapping-tag scores with explicit tie weights and all triplets.

    Args:
        scope (str): Candidate reference population.
        seed (int): Deterministic overlapping-family fixture seed.
        distance_metric (DistanceMetric): Cosine or Euclidean task distances.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.integers(-1, 2, size=(21, 4)).astype(float)
    vectors[0] = 0
    vectors[1:4] = vectors[4]  # exact ties across several original tasks
    similarity = separation_similarity(vectors, distance_metric)
    labels = np.repeat([0, 1, 2], [5, 7, 9])
    tags = rng.random((21, 7)) < np.linspace(0.1, 0.9, 7)
    tags[:, 0] = labels == 0  # concept-exclusive family
    tags[:, 1] = False
    tags[3, 1] = True  # singleton
    tags[:, 2] = True  # no negatives
    eligible = eligible_references(similarity, labels, scope)
    if distance_metric == "euclidean":
        expected_eligible = ~np.eye(len(vectors), dtype=bool)
        if scope == "cross_concept":
            expected_eligible &= labels[:, None] != labels[None, :]
        np.testing.assert_array_equal(eligible, expected_eligible)
    separation = family_separation(similarity, eligible, tags)
    for anchor in range(len(vectors)):
        candidates = np.flatnonzero(eligible[anchor])
        for family in range(tags.shape[1]):
            positives = candidates[tags[candidates, family]]
            negatives = candidates[~tags[candidates, family]]
            valid = tags[anchor, family] and len(positives) > 0 and len(negatives) > 0
            if not valid:
                assert np.isnan(separation["scores"][anchor, family]).all()
                continue
            if distance_metric == "euclidean":
                # Integer squared distances give an independent exact ordering,
                # including zero anchors, duplicate points, and equal-length edits.
                a = ((vectors[positives] - vectors[anchor]) ** 2).sum(axis=1)[:, None]
                b = ((vectors[negatives] - vectors[anchor]) ** 2).sum(axis=1)[None, :]
                wins, ties = a < b, a == b
            else:
                a = similarity[anchor, positives, None]
                b = similarity[anchor, negatives][None, :]
                wins, ties = a > b, a == b
            expected = [np.mean(wins), np.mean(wins + 0.5 * ties)]
            np.testing.assert_allclose(separation["scores"][anchor, family], expected)
            assert separation["strict_win_counts"][anchor, family] == np.sum(wins)
            assert separation["tie_counts"][anchor, family] == np.sum(ties)
            assert separation["comparison_counts"][anchor, family] == a.size * b.size
    assert np.array_equal(np.isfinite(separation["scores"]), separation["reasons"] == 0)


def test_rare_and_undefined_family_aggregation() -> None:
    """Detailed rare groups survive while headline macros weight eligible families equally."""
    tags = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=bool)
    scores = np.array(
        [[1, 0, 0.4], [0.5, np.nan, np.nan], [0, 0, np.nan], [np.nan, 0.9, np.nan]]
    )[..., None]
    output = aggregate_families(scores, tags, ["action"] * 3, 2, ["action", "layout"])
    np.testing.assert_allclose(output["family_mean"][:, 0], [0.5, 0.3, 0.4])
    assert output["macro"][0, 0] == pytest.approx(0.4)
    assert output["macro_valid_families"][0, 0] == 2
    assert np.isnan(output["macro"][1, 0])
    assert output["macro_valid_families"][1, 0] == 0


def test_concept_references_match_existing_metrics() -> None:
    """The global one-hot reference reproduces legacy concept-only calculations."""
    rng = np.random.default_rng(83)
    means = rng.normal(size=(17, 6))
    means -= means.mean(axis=0)
    labels = np.repeat([0, 1, 2], [4, 6, 7])
    tags = labels[:, None] == np.arange(3)
    gram = means @ means.T
    similarity = cosine_from_gram(gram)
    eligible = eligible_references(similarity, labels, "global")
    separation = family_separation(similarity, eligible, tags)
    old_sep, _ = separation_probability(similarity, labels)
    np.testing.assert_allclose(separation["scores"][np.arange(17), labels], old_sep)


@pytest.fixture
def metadata_bundle(feature_bundle: Path, tmp_path: Path) -> Path:
    """Publish a small frozen spec snapshot aligned to all eight-version features.

    Args:
        feature_bundle (Path): Existing realistic archive fixture.
        tmp_path (Path): Isolated temporary directory.

    Returns:
        Path: Portable metadata explorer snapshot with a restricted layer window.
    """
    bundle = load_bundle(feature_bundle)
    directory, specs = tmp_path / "metadata", tmp_path / "specs"
    directory.mkdir()
    for index, parent in enumerate(bundle.task_ids):
        path = specs / f"{parent}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        action: dict[str, Any] = {
            "op": "recolor_group" if index % 2 else "remove_group",
            "color": 3,
        }
        if index == 0:
            action = {
                "op": "conditional",
                "if_true": action,
                "if_false": {"op": "recolor_group", "color": 3},
            }
        write_json(
            path,
            {
                "task_id": parent,
                "concept_name": parent.split("/")[0],
                "program_kind": "conceptarc_dsl_v1",
                "description": "fixture",
                "program": {
                    "layout": {
                        "op": "scattered_objects" if index % 3 else "line_boundary"
                    },
                    "selector": {
                        "op": (
                            "objects_by_orientation"
                            if index % 2
                            else "inside_container_objects"
                        )
                    },
                    "action": action,
                },
            },
        )
    taxonomy = load_taxonomy()
    metadata = load_spec_metadata(
        specs, bundle.task_ids, bundle.concepts, bundle.labels.tolist(), taxonomy
    )
    write_json(
        directory / "task_metadata.json",
        {
            "schema_version": 1,
            "taxonomy_version": taxonomy["version"],
            "taxonomy": taxonomy,
            "records": metadata.records,
            "fields": metadata.fields,
        },
    )
    shutil.copyfile(DEFAULT_TAXONOMY, directory / "taxonomy.json")
    save_axes(directory, bundle, ["S1"], ())
    save_explorer_supports(directory, bundle.records)
    axes = json.loads((directory / "axes.json").read_text())
    axes["layers"] = [0]
    write_json(directory / "axes.json", axes)
    write_json(
        directory / "run_config.json",
        {
            "schema_version": 1,
            "analysis": "metadata_explorer",
            "arguments": {"spec_dir": str(specs)},
            "extraction_config": bundle.config,
            "feature_fingerprints": {
                name: fingerprint(feature_bundle / name)
                for name in (
                    "run_config.json",
                    "manifest.jsonl",
                    "tokens.npz",
                    "S1.npz",
                    "S2.npz",
                    "S3.npz",
                )
            },
            "taxonomy_version": taxonomy["version"],
            "taxonomy_fingerprint": fingerprint(directory / "taxonomy.json"),
            "spec_fingerprints": metadata.fingerprints,
        },
    )
    return directory


def test_frozen_alignment_and_layer_independence(
    metadata_bundle: Path, feature_bundle: Path
) -> None:
    """Validate nested tags and allow layers absent from the explorer window.

    Args:
        metadata_bundle (Path): Frozen annotations with a narrow displayed window.
        feature_bundle (Path): Complete support-feature archives.
    """
    bundle = load_bundle(feature_bundle, "2")
    hashes = {
        name: fingerprint(feature_bundle / name)
        for name in ("run_config.json", "manifest.jsonl", "tokens.npz", "S3.npz")
    }
    families = load_family_data(
        metadata_bundle, bundle, ["action", "layout", "selector"], hashes
    )
    assert bundle.layers == [2]
    assert all(
        families.membership[0, families.ids.index(f"action/{tag}")]
        for tag in ["conditional", "recolor", "removal"]
    )
    assert families.membership.shape[0] == 12
    hashes["S3.npz"] = {"sha256": "wrong", "bytes": 1}
    with pytest.raises(ValueError, match="fingerprint"):
        load_family_data(metadata_bundle, bundle, ["action"], hashes)


@pytest.mark.parametrize(
    "mutation",
    [
        "labels",
        "version_rows",
        "support_set_ids",
        "case_ids",
        "task_ids",
        "taxonomy",
        "spec_hash",
        "duplicate",
        "concept",
        "annotations",
        "extraction",
    ],
)
def test_metadata_rejects_conflicting_provenance(
    metadata_bundle: Path, feature_bundle: Path, mutation: str
) -> None:
    """Reject inconsistent axes, modified tags, and incompatible provenance.

    Args:
        metadata_bundle (Path): Frozen annotations to alter.
        feature_bundle (Path): Matching feature archives.
        mutation (str): Provenance field to invalidate.
    """
    name = "axes.json"
    if mutation in ("spec_hash", "extraction"):
        name = "run_config.json"
    elif mutation in ("duplicate", "concept", "annotations"):
        name = "task_metadata.json"
    elif mutation == "taxonomy":
        name = "taxonomy.json"
    payload = json.loads((metadata_bundle / name).read_text())
    if mutation == "support_set_ids":
        payload[mutation][0] = payload[mutation][0][::-1]
    elif name == "axes.json":
        payload[mutation] = payload[mutation][::-1]
    elif mutation == "spec_hash":
        payload["spec_fingerprints"] = {}
    elif mutation == "extraction":
        payload["extraction_config"]["extraction"]["model_name"] = "wrong"
    elif mutation == "taxonomy":
        payload["version"] = "wrong"
    elif mutation == "duplicate":
        payload["records"][1] = payload["records"][0]
    elif mutation == "concept":
        payload["records"][0]["concept_name"] = "Wrong"
    else:
        payload["records"][0]["families"]["action"] = ["invented"]
    write_json(metadata_bundle / name, payload)
    with pytest.raises(ValueError):
        load_family_data(metadata_bundle, load_bundle(feature_bundle), ["action"], {})


@pytest.mark.parametrize(
    "analysis,distance_metric",
    [
        ("cosine_distance", "cosine"),
        ("cosine_distance", "euclidean"),
    ],
)
def test_metadata_runner_round_trip(
    metadata_bundle: Path,
    feature_bundle: Path,
    tmp_path: Path,
    analysis: str,
    distance_metric: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish independent outputs and regenerate plots without source features.

    Args:
        metadata_bundle (Path): Frozen metadata fixture.
        feature_bundle (Path): Complete support features.
        tmp_path (Path): Isolated publication directory.
        analysis (str): Metadata analysis to run.
        distance_metric (str | None): Optional separation metric.
        monkeypatch (pytest.MonkeyPatch): Set the regeneration CLI arguments.
    """
    arguments = [
        "--feature_dir",
        str(feature_bundle),
        "--metadata_dir",
        str(metadata_bundle),
        "--output_root",
        str(tmp_path / "out"),
        "--positions",
        "S3",
        "--layers",
        "2",
        "--family_roles",
        "action",
        "--min_family_tasks",
        "2",
        "--num_threads",
        "1",
    ]
    if analysis == "cosine_distance":
        arguments.append("--pdf")
    if distance_metric is not None:
        arguments.extend(["--distance_metric", distance_metric])
    directory = run_analysis(analysis, arguments)
    output_analysis = f"{distance_metric}_distance" if distance_metric else analysis
    assert directory.parts[-2:] == ("metadata", output_analysis)
    axes = json.loads((directory / "axes.json").read_text())
    assert axes["layers"] == [2] and len(axes["task_ids"]) == 12
    assert axes["scopes"] == ["global", "cross_concept"]
    assert np.load(directory / "memberships.npy").dtype == bool
    assert np.load(directory / "S3/task_means.npy").dtype == np.float32
    config = json.loads((directory / "run_config.json").read_text())
    assert config["analysis"] == output_analysis
    if distance_metric is not None:
        assert axes["distance_metric"] == config["distance_metric"] == distance_metric
    if distance_metric == "euclidean":
        assert "undefined_anchor_distance" in axes["reason_bits"]
        assert (
            "undefined_anchor_cosine"
            not in (directory / "family_summary.csv").read_text()
        )
    assert config["elapsed_seconds"] >= config["elapsed_seconds_before_plotting"] > 0
    with np.load(directory / "metrics.npz", allow_pickle=False) as results:
        assert np.isnan(results["macro"][0, 0, 1, 0]).all()
        conditional = axes["family_ids"].index("action/conditional")
        assert not axes["headline_eligible"][conditional]
        with np.load(directory / "S3/layer_2.npz", allow_pickle=False) as layer:
            assert np.isfinite(layer["scores"]).sum() > 0
            assert layer["reasons"].dtype == np.uint8
            if distance_metric == "euclidean":
                bundle = load_bundle(feature_bundle)
                _, _, means = center_versions(
                    bundle.load_position("S3")[:, bundle.layers.index(2)],
                    bundle.rows,
                )
                tags = np.load(directory / "memberships.npy")
                for s, scope in enumerate(axes["scopes"]):
                    for anchor in range(len(means)):
                        candidates = np.arange(len(means)) != anchor
                        if scope == "cross_concept":
                            candidates &= bundle.labels != bundle.labels[anchor]
                        for f, role in enumerate(axes["family_roles"]):
                            if not tags[anchor, f] or (
                                scope == "cross_concept" and role == "concept"
                            ):
                                continue
                            positive = means[candidates & tags[:, f]]
                            negative = means[candidates & ~tags[:, f]]
                            if not len(positive) or not len(negative):
                                assert np.isnan(layer["scores"][s, anchor, f]).all()
                                continue
                            pos = ((positive - means[anchor]) ** 2).sum(axis=1)[:, None]
                            neg = ((negative - means[anchor]) ** 2).sum(axis=1)[None, :]
                            wins, ties = pos < neg, pos == neg
                            np.testing.assert_allclose(
                                layer["scores"][s, anchor, f],
                                [wins.mean(), (wins + 0.5 * ties).mean()],
                            )
                            assert layer["comparison_counts"][s, anchor, f] == wins.size
                            assert (
                                layer["strict_win_counts"][s, anchor, f] == wins.sum()
                            )
                            assert layer["tie_counts"][s, anchor, f] == ties.sum()
    assert list((directory / "figures").rglob("*.png"))
    assert list((directory / "figures").rglob("*.pdf"))
    with pytest.raises(FileExistsError):
        run_analysis(analysis, arguments)
    shutil.rmtree(feature_bundle)
    shutil.rmtree(metadata_bundle)
    shutil.rmtree(directory / "figures")
    (directory / "figures").mkdir()
    legacy = directory / "figures" / f"S3_global_{axes['metrics'][0]}.png"
    legacy.write_bytes(b"previous flat figure")
    monkeypatch.setattr(
        "sys.argv", ["plot_analysis", "--analysis_dir", str(directory), "--pdf"]
    )
    plot_main()
    assert not legacy.exists()
    assert not list((directory / "figures").glob("*.png"))
    assert list((directory / "figures" / "comparisons" / "global").glob("*.png"))
    figures = list((directory / "figures").rglob("*.png"))
    assert figures
    for figure in figures:
        assert figure.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
    assert (directory / "summary.csv").exists()


@pytest.mark.parametrize("all_ineligible", [False, True])
def test_family_trends_filter_without_reassigning_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, all_ineligible: bool
) -> None:
    """Omit rare/undefined families while preserving gaps, colors and peak ordering.

    Args:
        tmp_path (Path): Unwritten figure destination.
        monkeypatch (pytest.MonkeyPatch): Capture the figure before export.
        all_ineligible (bool): Also exercise a role without any eligible family.
    """
    import matplotlib.pyplot as plt

    from arc_cd.repr_analysis import family_visualization
    from arc_cd.repr_analysis.figure_style import family_line_style

    axes = {
        "metrics": ["tie_adjusted"],
        "positions": ["S3"],
        "scopes": ["cross_concept"],
        "layers": [0, 1, 2],
        "family_names": ["low", "rare", "high", "undefined"],
        "family_display_names": ["Low", "Rare", "High", "Undefined"],
        "headline_eligible": (
            [False] * 4 if all_ineligible else [True, False, True, True]
        ),
    }
    values = np.array(
        [[0.1, 0.9, np.nan, np.nan], [0.2, 0.9, 0.6, np.nan], [0.3, 0.9, 0.4, np.nan]]
    )[None, :, None, :, None]
    before = values.copy()
    figures = []
    monkeypatch.setattr(
        family_visualization,
        "save_figure",
        lambda figure, *args, **kwargs: figures.append(figure),
    )
    family_visualization._plot_family_trends(
        tmp_path,
        axes,
        {"family_mean": values},
        0,
        0,
        "action",
        np.arange(4),
        metric_name="tie_adjusted",
    )
    figure = figures[0]
    try:
        panel, key = figure.axes
        assert not figure.texts
        legend = key.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == (
            [] if all_ineligible else ["High", "Low"]
        )
        lines = {line.get_label(): line for line in panel.lines}
        assert "Rare" not in lines and "Undefined" not in lines
        if not all_ineligible:
            for name, original_index in (("Low", 0), ("High", 2)):
                assert (
                    lines[name].get_color()
                    == family_line_style(original_index, 3)["color"]
                )
            assert np.isnan(lines["High"].get_ydata()[0])
        np.testing.assert_array_equal(values, before)
    finally:
        plt.close(figure)


def test_missing_spec_fingerprints_rejected(
    metadata_bundle: Path, feature_bundle: Path
) -> None:
    """Missing hashes on both sides cannot masquerade as matching provenance.

    Args:
        metadata_bundle (Path): Frozen metadata to invalidate.
        feature_bundle (Path): Complete original support features.
    """
    metadata = json.loads((metadata_bundle / "task_metadata.json").read_text())
    config = json.loads((metadata_bundle / "run_config.json").read_text())
    for record in metadata["records"]:
        record.pop("source_fingerprint")
    config["spec_fingerprints"] = {}
    write_json(metadata_bundle / "task_metadata.json", metadata)
    write_json(metadata_bundle / "run_config.json", config)
    with pytest.raises(ValueError, match="fingerprint"):
        load_family_data(metadata_bundle, load_bundle(feature_bundle), ["action"], {})


def test_metadata_bootstrap_rejected() -> None:
    """Metadata mode cannot silently accept an unimplemented inferential option."""
    with pytest.raises(SystemExit):
        parse_arguments(
            "cosine_distance",
            ["--feature_dir", "features", "--metadata_dir", "metadata", "--bootstrap"],
        )
    args = parse_arguments("cosine_distance", ["--feature_dir", "features"])
    assert args.metadata_dir is None


@pytest.fixture
def structural_metadata_bundle(metadata_bundle: Path) -> Path:
    """Add structural selectors, including two parents with no semantic criterion.

    Args:
        metadata_bundle (Path): Portable metadata fixture to annotate again.

    Returns:
        Path: Updated fixture with consistent spec fingerprints and annotations.
    """
    config = json.loads((metadata_bundle / "run_config.json").read_text())
    axes = json.loads((metadata_bundle / "axes.json").read_text())
    metadata = json.loads((metadata_bundle / "task_metadata.json").read_text())
    for i, row in enumerate(metadata["records"]):
        path = Path(row["spec_path"])
        spec = json.loads(path.read_text())
        prior = spec["program"]["selector"]
        spec["program"]["selector"] = (
            {"op": "intersection", "selectors": [{"op": "all_objects"}, prior]}
            if i < 10
            else {"op": "all_objects"}
        )
        write_json(path, spec)
    annotated = load_spec_metadata(
        Path(config["arguments"]["spec_dir"]),
        axes["task_ids"],
        axes["concepts"],
        axes["labels"],
        load_taxonomy(metadata_bundle / "taxonomy.json"),
    )
    metadata.update(records=annotated.records, fields=annotated.fields)
    config["spec_fingerprints"] = annotated.fingerprints
    write_json(metadata_bundle / "task_metadata.json", metadata)
    write_json(metadata_bundle / "run_config.json", config)
    return metadata_bundle


def test_headline_mask_retains_family_statistics() -> None:
    """Excluding a family changes macro means without deleting its detailed scores."""
    scores = np.broadcast_to(
        np.asarray([0.2, 0.8, 0.9])[None, :, None], (4, 3, 1)
    ).copy()
    membership = np.ones((4, 3), dtype=bool)
    roles = ["selector"] * 3
    full = aggregate_families(scores, membership, roles, 2)
    filtered = aggregate_families(
        scores, membership, roles, 2, included=np.asarray([True, False, False])
    )
    np.testing.assert_array_equal(full["family_mean"], filtered["family_mean"])
    np.testing.assert_allclose(filtered["macro"], [[0.2]])
    np.testing.assert_array_equal(filtered["macro_valid_families"], [[1]])
    assert full["macro"][0, 0] != filtered["macro"][0, 0]
    empty = aggregate_families(
        scores, membership, roles, 2, included=np.zeros(3, dtype=bool)
    )
    assert np.isnan(empty["macro"]).all() and not empty["macro_valid_families"].any()
    for invalid in (np.ones(3, dtype=int), np.ones(2, dtype=bool)):
        with pytest.raises(ValueError, match="mask"):
            aggregate_families(scores, membership, roles, 2, included=invalid)


def test_structural_kind_is_frozen_in_taxonomy(
    structural_metadata_bundle: Path,
    feature_bundle: Path,
) -> None:
    """The option follows explicit taxonomy kinds; older snapshots include all tags.

    Args:
        structural_metadata_bundle (Path): Annotated structural-selector fixture.
        feature_bundle (Path): Matching feature archives.
    """
    families = load_family_data(
        structural_metadata_bundle, load_bundle(feature_bundle), ["selector"], {}
    )
    mask = family_comparison_mask(families)
    assert {families.ids[i] for i in np.flatnonzero(~mask)} == {
        "selector/composition",
        "selector/collections",
    }
    assert family_comparison_mask(families, include_structural_selectors=True).all()
    for rule in families.taxonomy["families"]["selector"].values():
        rule.pop("kind", None)
    assert family_comparison_mask(families).all()


@pytest.mark.parametrize("analysis", ["cosine_distance"])
def test_structural_selector_option_round_trip(
    analysis: str,
    structural_metadata_bundle: Path,
    feature_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI consistently filters macro averages and keeps aligned detailed archives.

    Args:
        analysis (str): Metadata analysis runner.
        structural_metadata_bundle (Path): Structural and criterion memberships.
        feature_bundle (Path): Matching hidden-state archives.
        tmp_path (Path): Isolated output root.
        monkeypatch (pytest.MonkeyPatch): Disable redundant figure rendering here.
    """
    monkeypatch.setattr(
        "arc_cd.repr_analysis.visualization.plot_analysis", lambda *args, **kwargs: None
    )
    outputs = []
    for include in (False, True):
        arguments = [
            "--feature_dir",
            str(feature_bundle),
            "--metadata_dir",
            str(structural_metadata_bundle),
            "--output_root",
            str(tmp_path / "out"),
            "--run_name",
            str(include),
            "--layers",
            "2",
            "--positions",
            "S3",
            "--family_roles",
            "selector",
            "--min_family_tasks",
            "2",
            "--num_threads",
            "1",
        ]
        if include:
            arguments.append("--include_structural_selectors")
        directory = run_analysis(analysis, arguments)
        axes = json.loads((directory / "axes.json").read_text())
        config = json.loads((directory / "run_config.json").read_text())
        columns = [
            axes["family_ids"].index("selector/" + name)
            for name in ("collections", "composition")
        ]
        assert all(axes["comparison_included"][i] == include for i in columns)
        assert all(axes["headline_eligible"][i] == include for i in columns)
        expected = [] if include else ["collections", "composition"]
        assert config["coverage"]["selector"]["excluded_families"] == expected
        assert config["coverage"]["selector"]["comparison_annotated_parents"] == (
            12 if include else 10
        )
        with np.load(directory / "metrics.npz") as metrics:
            outputs.append({key: metrics[key].copy() for key in metrics.files})
            selector = axes["roles"].index("selector")
            keep = (np.asarray(axes["family_roles"]) == "selector") & np.asarray(
                axes["headline_eligible"]
            )
            values = metrics["family_mean"][..., keep, :]
            valid = np.isfinite(values)
            expected_macro = np.divide(
                np.where(valid, values, 0).sum(axis=-2),
                valid.sum(axis=-2),
                out=np.full(values.shape[:-2] + values.shape[-1:], np.nan),
                where=valid.sum(axis=-2) > 0,
            )
            np.testing.assert_allclose(
                metrics["macro"][..., selector, :], expected_macro, equal_nan=True
            )
    if outputs:
        np.testing.assert_allclose(
            outputs[0]["family_mean"], outputs[1]["family_mean"], equal_nan=True
        )
        np.testing.assert_array_equal(
            outputs[0]["family_valid_anchors"], outputs[1]["family_valid_anchors"]
        )


@pytest.mark.parametrize("analysis", ["cosine_distance"])
def test_structural_selector_option_requires_metadata(analysis: str) -> None:
    """Reject the metadata-only flag instead of silently ignoring it in concept mode.

    Args:
        analysis (str): Shared runner name.
    """
    with pytest.raises(SystemExit):
        parse_arguments(
            analysis, ["--feature_dir", "unused", "--include_structural_selectors"]
        )
