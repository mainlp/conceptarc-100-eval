"""Spec identity, taxonomy semantics, and portable explorer regression tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arc_cd.repr_analysis.explorer import load_explorer_data
from arc_cd.repr_analysis.metadata_runner import run_metadata_explorer
from arc_cd.repr_analysis.spec_metadata import (
    annotate_program,
    load_spec_metadata,
    load_taxonomy,
)
from arc_cd.repr_analysis.visualization import plot_analysis


def program(action: dict | None = None) -> dict:
    """Create a small, valid DSL program.

    Args:
        action (dict | None): Optional action override.

    Returns:
        dict: Program with explicit layout, selector, and action roots.
    """
    return {
        "layout": {"op": "scattered_objects", "orientation": "mixed"},
        "selector": {"op": "objects_by_orientation", "orientation": "vertical"},
        "action": action
        or {
            "op": "recolor_object_group_pattern",
            "color": 7,
            "pattern": {"op": "alternating", "mode": "horizontal"},
        },
    }


def test_nested_roles_and_declared_branch_families() -> None:
    """Keep branch actions, embedded selectors, helpers, and parameter paths distinct."""
    example = program(
        {
            "op": "conditional",
            "condition": {
                "op": "object_pair_equivalent",
                "key": {"op": "object_shape"},
            },
            "if_true": {
                "op": "recolor_group",
                "color": {
                    "op": "selector_color",
                    "selector": {"op": "inside_container_objects"},
                },
            },
            "if_false": {"op": "remove_group"},
        }
    )
    result = annotate_program(example, load_taxonomy())
    assert result["families"]["action"] == ["conditional", "recolor", "removal"]
    assert result["families"]["selector"] == ["containment", "orientation"]
    nodes = {node["path"]: node for node in result["operations"]}
    assert nodes["program.action.if_true.color.selector"]["role"] == "selector"
    assert nodes["program.action.condition.key"]["status"] == "helper"
    assert result["values"]["parameter.program.layout.orientation"] == ['"mixed"']
    assert result["values"]["parameter.program.selector.orientation"] == ['"vertical"']
    assert result["values"]["parameter.program.action.condition.key.op"] == [
        '"object_shape"'
    ]
    assert example["action"]["op"] == "conditional"


def test_composition_tags_explicit_rendering_and_keeps_provenance() -> None:
    """A canvas composition retains placement/render paths and deduplicated tags."""
    placement = {
        "op": "place_selection",
        "row": {"op": "selection_min_row"},
        "col": {"op": "selection_min_col"},
        "selector": {
            "op": "completion_region_cells",
            "state": "incomplete",
            "part": "missing",
        },
        "render": {"op": "alternating", "mode": "checkerboard", "colors": [3, 4]},
    }
    example = program(
        {
            "op": "compose_canvas",
            "canvas": {"op": "input_canvas"},
            "placements": [placement, copy.deepcopy(placement)],
        }
    )
    result = annotate_program(example, load_taxonomy())
    assert result["families"]["action"] == ["pattern_filling"]
    assert "completion_reconstruction" in result["families"]["selector"]
    assert result["values"]["parameter.program.action.placements[].render.mode"] == [
        '"checkerboard"'
    ]
    assert any(
        p["path"] == "program.action.placements[1].render.mode"
        for p in result["parameters"]
    )
    assert result["values"]["parameter.program.action.canvas.op"] == ['"input_canvas"']


def test_unknown_operations_and_explicit_overrides(tmp_path: Path) -> None:
    """Retain unmapped names until explicitly covered by a taxonomy revision.

    Args:
        tmp_path (Path): Isolated taxonomy directory.
    """
    taxonomy = load_taxonomy()
    example = program({"op": "recolorish_but_unknown"})
    result = annotate_program(example, taxonomy)
    assert result["families"]["action"] == []
    assert result["root_ops"]["action"] == "recolorish_but_unknown"
    assert result["operations"][-1]["status"] == "unmapped"
    taxonomy["families"]["action"]["recolor"]["operations"].append(
        "recolorish_but_unknown"
    )
    taxonomy["version"] = "test-override"
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(taxonomy))
    assert annotate_program(example, load_taxonomy(path))["families"]["action"] == [
        "recolor"
    ]
    taxonomy["parameter_keys"].append(taxonomy["parameter_keys"][0])
    path.write_text(json.dumps(taxonomy))
    with pytest.raises(ValueError, match="unique"):
        load_taxonomy(path)


@pytest.fixture
def metadata_source(tmp_path: Path) -> tuple[Path, Path]:
    """Create independent small embedding artifacts and corresponding parent specs.

    Args:
        tmp_path (Path): Isolated source directories.

    Returns:
        tuple[Path, Path]: Embedding and spec directories.
    """
    source, specs = tmp_path / "source", tmp_path / "specs"
    source.mkdir()
    concepts = ["Alpha", "Beta"]
    tasks = [f"{concept}/{concept}{i}" for concept in concepts for i in range(4)]
    rows = np.arange(len(tasks) * 8).reshape(len(tasks), 8).tolist()
    variants = [[f"{task}_{v:03}" for v in range(8)] for task in tasks]
    case_ids = [[f"{task}/support" for task in versions] for versions in variants]
    axes = {
        "task_ids": tasks,
        "labels": [0] * 4 + [1] * 4,
        "concepts": concepts,
        "positions": ["S1", "S3"],
        "layers": [0, 2],
        "methods": ["umap", "pca"],
        "version_rows": rows,
        "version_task_ids": variants,
        "case_ids": case_ids,
        "support_set_ids": [[f"{v:03}" for v in range(8)] for _ in tasks],
    }
    (source / "axes.json").write_text(json.dumps(axes))
    supports = []
    for task, parent in enumerate(tasks):
        path = specs / f"{parent}.json"
        path.parent.mkdir(exist_ok=True, parents=True)
        example = program() if task % 2 else program({"op": "remove_group"})
        example["layout"]["op"] = (
            "scattered_objects" if task % 3 else "gapped_aligned_objects"
        )
        path.write_text(
            json.dumps(
                {
                    "task_id": parent,
                    "concept_name": parent.split("/")[0],
                    "program_kind": "conceptarc_dsl_v1",
                    "description": "A synthetic </script><script>alert('unsafe')</script> task",
                    "seed": 42,
                    "program": example,
                }
            )
        )
        for version in range(8):
            supports.append(
                {
                    "task_id": variants[task][version],
                    "case_id": case_ids[task][version],
                    "parent_task_id": parent,
                    "support_set_id": f"{version:03}",
                    "source_path": f"/synthetic/{variants[task][version]}.json",
                    "selected_token_indices": {"S1": 0, "S2": 2, "S3": 4},
                    "supports": [
                        {
                            "input": {"rows": 1, "cols": 2, "cells": "12"},
                            "output": {"rows": 1, "cols": 2, "cells": "34"},
                        }
                    ]
                    * 3,
                }
            )
    (source / "supports.json").write_text(json.dumps(supports))
    config = {
        "schema_version": 1,
        "analysis": "dimensionality_reduction",
        "arguments": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "min_cluster_size": 3,
            "min_samples": 2,
            "cluster_selection_method": "eom",
        },
        "extraction_config": {
            "dataset": {"id": "synthetic"},
            "extraction": {"model_name": "tiny"},
        },
        "fingerprints": {"input": {"sha256": "fixture", "bytes": 0}},
    }
    (source / "run_config.json").write_text(json.dumps(config))
    rng = np.random.default_rng(12)
    for position in axes["positions"]:
        (source / position).mkdir()
        for layer in axes["layers"]:
            np.savez_compressed(
                source / position / f"embedding_layer_{layer}.npz",
                coordinates=rng.normal(size=(2, len(tasks), 2)),
                cluster_labels=np.array([0, 0, 0, -1, 1, 1, 1, -1]),
                cluster_strengths=np.ones(len(tasks)),
                representative_versions=np.arange(len(tasks)),
                within_task_dispersion=np.arange(len(tasks)) / 10,
                valid_vectors=np.ones(len(tasks), dtype=bool),
                pca_explained_variance_ratio=np.array([0.3, 0.2]),
            )
    return source, specs


def test_parent_join_counts_unused_and_missing(
    metadata_source: tuple[Path, Path],
) -> None:
    """Align by identity, count parent tags once, report unused and reject missing specs.

    Args:
        metadata_source (tuple[Path, Path]): Embeddings and specs.
    """
    source, specs = metadata_source
    axes = json.loads((source / "axes.json").read_text())
    original = json.loads((specs / "Alpha/Alpha0.json").read_text())
    extra = {**original, "task_id": "Alpha/Unused"}
    (specs / "Alpha/Unused.json").write_text(json.dumps(extra))
    renamed = specs / "arbitrary_filename.json"
    (specs / "Alpha/Alpha0.json").rename(renamed)
    bundle = load_spec_metadata(
        specs,
        axes["task_ids"][::-1],
        axes["concepts"],
        axes["labels"][::-1],
        load_taxonomy(),
    )
    assert [row["task_id"] for row in bundle.records] == axes["task_ids"][::-1]
    assert bundle.inventory["unused_task_ids"] == ["Alpha/Unused"]
    field = next(field for field in bundle.fields if field["id"] == "family.action")
    assert (
        next(value for value in field["values"] if value["value"] == "pattern_recolor")[
            "count"
        ]
        == 4
    )
    renamed.unlink()
    with pytest.raises(ValueError, match="Missing specs"):
        load_spec_metadata(
            specs, axes["task_ids"], axes["concepts"], axes["labels"], load_taxonomy()
        )


@pytest.mark.parametrize("change", ["duplicate", "concept", "program"])
def test_reject_conflicting_specs(
    metadata_source: tuple[Path, Path], change: str
) -> None:
    """Reject identity, concept, and program violations.

    Args:
        metadata_source (tuple[Path, Path]): Embeddings and specs.
        change (str): Corruption to introduce.
    """
    source, specs = metadata_source
    axes = json.loads((source / "axes.json").read_text())
    path = specs / "Alpha/Alpha1.json"
    value = json.loads(path.read_text())
    if change == "duplicate":
        value["task_id"] = "Alpha/Alpha0"
    elif change == "concept":
        value["concept_name"] = "Beta"
    else:
        value["program"]["action"] = {"op": 10}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_spec_metadata(
            specs, axes["task_ids"], axes["concepts"], axes["labels"], load_taxonomy()
        )


def test_runner_artifact_preservation_and_regeneration(
    metadata_source: tuple[Path, Path],
) -> None:
    """Copy only display assets, preserve source bytes, and regenerate without inputs.

    Args:
        metadata_source (tuple[Path, Path]): Embeddings and specs.
    """
    source, specs = metadata_source
    hashes = {
        p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file()
    }
    arguments = ["--analysis_dir", str(source), "--spec_dir", str(specs)]
    destination = run_metadata_explorer(arguments)
    for relative, expected in hashes.items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected
        if relative != Path("run_config.json"):
            assert (
                hashlib.sha256((destination / relative).read_bytes()).hexdigest()
                == expected
            )
    assert not list(destination.rglob("*means.npy"))
    data = load_explorer_data(destination)
    assert len(data["metadata"]["records"]) == 8
    assert len(data["frames"]) == 4
    html = (destination / "explorer.html").read_text()
    assert "</script><script>alert" not in html
    assert "Task metadata" in html and "Selection metadata" in html
    assert 'src="http' not in html and 'href="http' not in html
    with pytest.raises(FileExistsError):
        run_metadata_explorer(arguments)
    source.rename(source.with_name("source-moved"))
    specs.rename(specs.with_name("specs-moved"))
    (destination / "explorer.html").unlink()
    plot_analysis(destination)
    assert (destination / "explorer.html").read_text() == html
    records = json.loads((destination / "task_metadata.json").read_text())
    records["records"].reverse()
    (destination / "task_metadata.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="align"):
        load_explorer_data(destination)


@pytest.mark.parametrize(
    "value",
    [None, [], 7, {"schema_version": 1, "version": "test", "families": None}],
)
def test_reject_malformed_taxonomy_objects(tmp_path: Path, value: Any) -> None:
    """Reject malformed override roots with a useful validation error.

    Args:
        tmp_path (Path): Isolated override directory.
        value (Any): Invalid JSON taxonomy value.
    """
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Taxonomy"):
        load_taxonomy(path)


@pytest.mark.parametrize("value", [None, [], 7])
def test_reject_nonobject_specs(metadata_source: tuple[Path, Path], value: Any) -> None:
    """Reject malformed spec roots before accessing identity fields.

    Args:
        metadata_source (tuple[Path, Path]): Embeddings and specs.
        value (Any): Invalid JSON spec value.
    """
    source, specs = metadata_source
    axes = json.loads((source / "axes.json").read_text())
    (specs / "Alpha/Alpha0.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="JSON object"):
        load_spec_metadata(
            specs, axes["task_ids"], axes["concepts"], axes["labels"], load_taxonomy()
        )


@pytest.mark.parametrize(
    "change",
    ["version", "snapshot", "annotations", "counts", "missing_field", "config_version"],
)
def test_reject_inconsistent_metadata_snapshot(
    metadata_source: tuple[Path, Path], change: str
) -> None:
    """Reject inconsistent saved annotations before rendering an incorrect explorer.

    Args:
        metadata_source (tuple[Path, Path]): Embeddings and specs.
        change (str): Saved metadata inconsistency to introduce.
    """
    source, specs = metadata_source
    destination = run_metadata_explorer(
        ["--analysis_dir", str(source), "--spec_dir", str(specs)]
    )
    path = destination / "task_metadata.json"
    metadata = json.loads(path.read_text())
    if change == "version":
        metadata["taxonomy_version"] = "different"
    elif change == "snapshot":
        metadata["taxonomy"]["version"] = "different"
    elif change == "annotations":
        metadata["records"][0]["values"]["op.action"] = ["wrong_action"]
    elif change == "counts":
        metadata["fields"][0]["values"][0]["count"] += 1
    elif change == "missing_field":
        metadata["fields"].pop()
    else:
        config_path = destination / "run_config.json"
        config = json.loads(config_path.read_text())
        config["taxonomy_version"] = "different"
        config_path.write_text(json.dumps(config))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata|taxonomy"):
        load_explorer_data(destination)


@pytest.mark.parametrize(
    ("role", "op", "expected"),
    [
        ("layout", "row_level_shape", {"row_level_structure"}),
        ("layout", "solid_rectangle_with_scattered_cells", {"corrupted_surfaces"}),
        ("layout", "paired_objects", {"paired_objects"}),
        ("selector", "object_pair", {"collections"}),
        ("selector", "same_color_object_pair", {"collections", "color", "equivalence"}),
        (
            "selector",
            "most_separated_parallel_lines",
            {"spatial_relations", "extrema", "orientation"},
        ),
        ("selector", "nearest_object", {"spatial_relations", "extrema"}),
        ("selector", "largest_object", {"size", "extrema"}),
        (
            "selector",
            "matching_template_objects",
            {"equivalence", "color", "size", "extrema"},
        ),
        ("action", "swap_shape_layer_colors", {"recolor"}),
    ],
)
def test_audited_operation_semantics(role: str, op: str, expected: set[str]) -> None:
    """Keep family assignments tied to DSL criteria rather than suggestive names.

    Args:
        role (str): Root expression role.
        op (str): Audited DSL primitive.
        expected (set[str]): Semantically justified family membership.
    """
    example = program()
    example[role] = {"op": op}
    assert set(annotate_program(example, load_taxonomy())["families"][role]) == expected


@pytest.mark.parametrize(
    ("key", "extra"),
    [
        ("object_line_direction", {"orientation"}),
        ("object_color", {"color"}),
        ("object_color_signature", {"color"}),
        ("object_shape", {"shape_geometry"}),
        ("object_dimensions", {"shape_geometry", "size"}),
        ("object_raster", {"shape_geometry", "color"}),
        ("unrecognized_key", set()),
    ],
)
def test_equivalence_key_families_and_evidence(key: str, extra: set[str]) -> None:
    """Resolve exact equivalence criteria, including selectors inside action branches.

    Args:
        key (str): Declared equivalence-key primitive.
        extra (set[str]): Additional families justified by that key.
    """
    example = program(
        {
            "op": "conditional",
            "if_true": {
                "op": "compose_canvas",
                "placements": [
                    {
                        "op": "place_selection",
                        "selector": {
                            "op": "objects_by_equivalence_class",
                            "key": {"op": key},
                        },
                    }
                ],
            },
            "if_false": {"op": "identity"},
        }
    )
    example["selector"] = {"op": "all_objects"}
    result = annotate_program(example, load_taxonomy())
    assert set(result["families"]["selector"]) == extra | {"collections", "equivalence"}
    node = next(
        n for n in result["operations"] if n["op"] == "objects_by_equivalence_class"
    )
    assert {e["family"] for e in node.get("parameter_evidence", [])} == extra
    assert all(
        e["path"] == node["path"] + ".key.op"
        for e in node.get("parameter_evidence", [])
    )
    assert result["values"]["op.action"] == ["conditional"]
    assert result["values"]["operation.action"] == [
        "compose_canvas",
        "conditional",
        "identity",
        "place_selection",
    ]
    assert "objects_by_equivalence_class" in result["values"]["operation.selector"]


def test_parameter_family_rules_require_the_correct_operator_and_role() -> None:
    """Do not apply a key or orientation rule to an unrelated operation or role."""
    example = program()
    example["selector"] = {"op": "line_objects", "orientation": "horizontal"}
    example["layout"]["key"] = {"op": "object_line_direction"}
    result = annotate_program(example, load_taxonomy())
    # line_objects ignores an orientation field in the DSL; it does not filter it.
    assert result["families"]["selector"] == ["collections"]
    example["selector"] = {"op": "spanning_line", "orientation": "horizontal"}
    assert (
        "orientation"
        in annotate_program(example, load_taxonomy())["families"]["selector"]
    )
    example["selector"] = {"op": "nearest_object", "anchor": {"op": "grid_center"}}
    assert (
        "centers_anchors"
        in annotate_program(example, load_taxonomy())["families"]["selector"]
    )


def test_complete_parameters_preserve_ranges_offsets_and_unknown_literals() -> None:
    """Expose literal metadata without flattening ordered numeric vectors into tags."""
    example = program({"op": "translate_object_group", "offset": [2, 0]})
    example["layout"].update(
        {
            "grid_height_range": [12, 18],
            "shape_styles": ["rectangle", "diamond"],
            "new_policy": "explicit_only",
            "empty_options": [],
            "object_connectivity": "orthogonal",
        }
    )
    result = annotate_program(example, load_taxonomy())
    values = result["values"]
    assert values["parameter.program.action.offset"] == ["[2, 0]"]
    assert values["parameter.program.layout.grid_height_range"] == ["[12, 18]"]
    assert values["parameter.program.layout.shape_styles"] == [
        '"diamond"',
        '"rectangle"',
    ]
    assert values["parameter.program.layout.empty_options"] == ["[]"]
    assert values["parameter.program.layout.new_policy"] == ['"explicit_only"']
    assert values["parameter.program.layout.object_connectivity"] == ['"orthogonal"']
    assert next(p for p in result["parameters"] if p["path"].endswith(".offset"))[
        "value"
    ] == [2, 0]


def test_legacy_taxonomy_parameter_behavior_is_preserved() -> None:
    """Old snapshots retain scalar-list tags and the original annotation shape."""
    taxonomy = load_taxonomy()
    for key in ("parameter_mode", "parameter_families", "index_all_operations"):
        taxonomy.pop(key)
    example = program({"op": "translate_object_group", "offset": [2, 0]})
    example["layout"]["unlisted_parameter"] = "value"
    result = annotate_program(example, taxonomy)
    assert result["values"]["parameter.program.action.offset"] == ["0", "2"]
    assert "parameter.program.layout.unlisted_parameter" not in result["values"]
    assert not any(field.startswith("operation.") for field in result["values"])
    assert all("parameter_evidence" not in node for node in result["operations"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "expression"),
        ("family", "unknown_family"),
        ("parameter", "key..op"),
        ("operations", ["not_registered"]),
        ("values", []),
        ("description", ""),
    ],
)
def test_reject_invalid_parameter_family_rules(
    tmp_path: Path, field: str, value: Any
) -> None:
    """Reject ambiguous or misspelled taxonomy overrides before annotation.

    Args:
        tmp_path (Path): Isolated override directory.
        field (str): Rule field to corrupt.
        value (Any): Invalid field value.
    """
    taxonomy = load_taxonomy()
    taxonomy["parameter_families"][0][field] = value
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(taxonomy))
    with pytest.raises(ValueError, match="[Pp]arameter family"):
        load_taxonomy(path)


def test_ordered_string_parameters_distinguish_opposite_layer_orders() -> None:
    """Preserve sort priorities while keeping categorical options as membership tags."""
    taxonomy = load_taxonomy()
    example = program(
        {
            "op": "reorder_shape_layers",
            "order": {
                "op": "sort",
                "key": {"op": "shape_orientation"},
                "value_order": ["vertical", "horizontal"],
            },
        }
    )
    example["layout"]["shape_styles"] = ["rectangle", "diamond"]
    before = copy.deepcopy(example)
    first = annotate_program(example, taxonomy)
    assert example == before
    example["action"]["order"]["value_order"].reverse()
    second = annotate_program(example, taxonomy)
    field = "parameter.program.action.order.value_order"
    assert first["values"][field] == ['["vertical", "horizontal"]']
    assert second["values"][field] == ['["horizontal", "vertical"]']
    assert first["families"] == second["families"]
    assert first["values"]["parameter.program.layout.shape_styles"] == [
        '"diamond"',
        '"rectangle"',
    ]
    # Taxonomy 1.1.0 snapshots keep their original membership-style annotations.
    taxonomy.pop("ordered_parameter_keys")
    assert annotate_program(example, taxonomy)["values"][field] == [
        '"horizontal"',
        '"vertical"',
    ]


@pytest.mark.parametrize("value", [None, ["value_order", "value_order"]])
def test_reject_invalid_ordered_parameter_keys(tmp_path: Path, value: Any) -> None:
    """Reject malformed ordering overrides instead of silently changing semantics.

    Args:
        tmp_path (Path): Isolated taxonomy override directory.
        value (Any): Invalid ordering-key configuration.
    """
    taxonomy = load_taxonomy()
    taxonomy["ordered_parameter_keys"] = value
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(taxonomy))
    with pytest.raises(ValueError, match="ordered_parameter_keys"):
        load_taxonomy(path)


def composition(
    placement: dict[str, Any], canvas: str = "blank_canvas"
) -> dict[str, Any]:
    """Build a composition with an explicitly selected output canvas.

    Args:
        placement (dict[str, Any]): Single declared placement.
        canvas (str): Canvas expression name.

    Returns:
        dict[str, Any]: Composition action for annotation tests.
    """
    placement.setdefault("row", {"op": "selection_min_row"})
    placement.setdefault("col", {"op": "selection_min_col"})
    return {"op": "compose_canvas", "canvas": {"op": canvas}, "placements": [placement]}


@pytest.mark.parametrize(
    "canvas,family",
    [("blank_canvas", "isolation"), ("input_canvas", "selection_overlay")],
)
@pytest.mark.parametrize("render", [None, {}, {"op": "preserve"}])
def test_preserved_composition_uses_explicit_original_coordinates(
    canvas: str, family: str, render: Any
) -> None:
    """Distinguish isolation/overlay without guessing omitted placement coordinates.

    Args:
        canvas (str): Canvas type.
        family (str): Expected semantic family.
        render (Any): Supported source-color-preserving render declaration.
    """
    placement = {
        "op": "place_selection",
        "row": {"op": "selection_min_row"},
        "col": {"op": "selection_min_col"},
        "render": render,
    }
    example = program(composition(placement, canvas))
    before = copy.deepcopy(example)
    result = annotate_program(example, load_taxonomy())
    assert result["families"]["action"] == [family]
    assert example == before
    node = next(n for n in result["operations"] if n["op"] == "compose_canvas")
    assert node["composition_evidence"][0]["path"] == "program.action.placements"
    for field in ("row", "col"):
        missing = copy.deepcopy(example)
        missing["action"]["placements"][0].pop(field)
        assert not annotate_program(missing, load_taxonomy())["families"]["action"]
    placement["target_position"] = {"op": "grid_center"}
    assert not annotate_program(example, load_taxonomy())["families"]["action"]


def test_corner_extraction_and_declared_empty_branch() -> None:
    """Keep corner placement distinct from unchanged input and empty output branches."""
    example = program(
        {
            "op": "conditional",
            "if_true": composition(
                {
                    "op": "place_selection",
                    "location": {
                        "op": "selection_grid_corner",
                        "corner": "lower_right",
                    },
                }
            ),
            "if_false": {
                "op": "compose_canvas",
                "canvas": {"op": "blank_canvas"},
                "placements": [],
                "allow_empty": True,
            },
        }
    )
    result = annotate_program(example, load_taxonomy())
    assert result["families"]["action"] == [
        "conditional",
        "empty_output",
        "extract_place",
    ]
    nodes = {n["path"]: n for n in result["operations"]}
    assert (
        nodes["program.action.if_true"]["composition_evidence"][0]["case"]
        == "extract_to_corner"
    )
    assert (
        nodes["program.action.if_false"]["composition_evidence"][0]["case"]
        == "empty_canvas"
    )
    empty = example["action"]["if_false"]
    empty["canvas"]["op"] = "input_canvas"
    assert annotate_program(program(empty), load_taxonomy())["families"]["action"] == [
        "identity"
    ]
    empty.pop("allow_empty")
    assert not annotate_program(program(empty), load_taxonomy())["families"]["action"]


def test_composition_rendering_is_contextual_and_overlapping() -> None:
    """Classify declared render nodes, not a selector's solid/alternating model."""
    action = composition(
        {"op": "place_selection", "render": {"op": "solid", "color": 3}}, "input_canvas"
    )
    action["placements"].append(
        {
            "op": "place_selection",
            "render": {"op": "alternating", "colors": [2, 4]},
            "selector": {
                "op": "completion_region_cells",
                "part": "object",
                "model": {"op": "solid"},
            },
            "row": {"op": "selection_min_row"},
            "col": {"op": "selection_min_col"},
        }
    )
    result = annotate_program(program(action), load_taxonomy())
    assert result["families"]["action"] == ["pattern_recolor", "recolor"]
    evidence = next(n for n in result["operations"] if n["op"] == "compose_canvas")[
        "color_edit_evidence"
    ]
    assert {e["case"] for e in evidence} == {"object_uniform", "object_pattern"}
    assert all(".placements[" in e["path"] for e in evidence)
    for p in action["placements"]:
        p.pop("render")
    result = annotate_program(program(action), load_taxonomy())
    assert result["families"]["action"] == ["selection_overlay"]
    assert any(
        n["op"] == "solid" and n["role"] == "expression" for n in result["operations"]
    )


@pytest.mark.parametrize("change", ["canvas", "placement", "render", "coordinates"])
def test_unsupported_compositions_remain_unmapped(change: str) -> None:
    """Do not manufacture a semantic family for an unsupported composition.

    Args:
        change (str): Unsupported part of the composition.
    """
    action = composition(
        {
            "op": "place_selection",
            "row": {"op": "selection_min_row"},
            "col": {"op": "selection_min_col"},
        }
    )
    if change == "canvas":
        action["canvas"]["op"] = "repeated_selection_canvas"
    elif change == "placement":
        action["placements"][0]["op"] = "repeat_selection"
    elif change == "render":
        action["placements"][0]["render"] = {"op": "unknown_render"}
    else:
        action["placements"][0]["row"] = 0
    result = annotate_program(program(action), load_taxonomy())
    assert not result["families"]["action"]
    node = next(n for n in result["operations"] if n["op"] == "compose_canvas")
    assert node["status"] == "unmapped"


def test_legacy_canvas_taxonomy_retains_original_annotations() -> None:
    """Frozen pre-split snapshots still produce their original broad family."""
    taxonomy = load_taxonomy()
    taxonomy.pop("composition_rules")
    taxonomy.pop("color_edit_rules")
    taxonomy["families"]["action"]["canvas_composition"] = {
        "label": "Canvas composition",
        "description": "Legacy operator family",
        "operations": ["compose_canvas", "place_selection"],
    }
    example = program(
        composition(
            {"op": "place_selection", "render": {"op": "alternating", "colors": [2, 3]}}
        )
    )
    result = annotate_program(example, taxonomy)
    assert result["families"]["action"] == ["canvas_composition"]
    assert all("composition_evidence" not in n for n in result["operations"])


@pytest.mark.parametrize(
    "change", ["unknown_case", "unknown_family", "empty_families", "no_description"]
)
def test_reject_invalid_composition_taxonomy(tmp_path: Path, change: str) -> None:
    """Reject unsupported composition rules and invalid family targets.

    Args:
        tmp_path (Path): Temporary taxonomy directory.
        change (str): Invalid rule mutation.
    """
    taxonomy = load_taxonomy()
    rules = taxonomy["composition_rules"]
    if change == "unknown_case":
        rules["guess_by_name"] = rules.pop("isolate_in_place")
    elif change == "unknown_family":
        rules["isolate_in_place"]["families"] = ["not_a_family"]
    elif change == "empty_families":
        rules["isolate_in_place"]["families"] = []
    else:
        rules["isolate_in_place"]["description"] = ""
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(taxonomy))
    with pytest.raises(ValueError, match="[Cc]omposition"):
        load_taxonomy(path)


@pytest.mark.parametrize(
    "render,expected",
    [
        ({"op": "solid", "color": 0}, ["removal"]),
        (
            {
                "op": "solid",
                "color": {"op": "selector_color", "selector": {"op": "all_objects"}},
            },
            ["recolor"],
        ),
        ({"op": "alternating", "colors": [3, 4]}, ["pattern_recolor"]),
        ({"op": "solid"}, []),
        ({"op": "alternating", "colors": [3]}, []),
    ],
)
def test_composition_render_families_require_explicit_color_parameters(
    render: dict[str, Any], expected: list[str]
) -> None:
    """Recognize solid and alternating renderers independently, including color expressions.

    Args:
        render (dict[str, Any]): Declared renderer.
        expected (list[str]): Expected action-family membership.
    """
    example = program(
        composition({"op": "place_selection", "render": render}, "input_canvas")
    )
    assert annotate_program(example, load_taxonomy())["families"]["action"] == expected


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ({"op": "translate_object_group", "offset": [0, 2]}, ["translation_alignment"]),
        (
            {"op": "translate_shape_layers", "offset": [-1, 1]},
            ["translation_alignment"],
        ),
        (
            {
                "op": "translate_to",
                "target_position": {"op": "align_to_reference", "offset": 2},
            },
            ["translation_alignment"],
        ),
        (
            {"op": "translate_each_to_reference", "axis": "row", "offset": 1},
            ["translation_alignment"],
        ),
        (
            {"op": "move_to_anchor", "anchor": {"op": "grid_center"}},
            ["translation_alignment"],
        ),
        (
            {"op": "move_to_parallel_midline", "lines": {"op": "object_pair"}},
            ["translation_alignment"],
        ),
        ({"op": "swap_shape_layer_positions"}, ["reordering"]),
        (
            {"op": "center_markers_on_lines"},
            ["translation_alignment"],
        ),
    ],
)
def test_movement_families_preserve_whole_transformations(
    action: dict[str, Any], expected: list[str]
) -> None:
    """Keep marker relocation whole and classify position swapping as reordering.

    Args:
        action (dict[str, Any]): Declared movement, swap or marker action.
        expected (list[str]): Semantic families implied by the operation.
    """
    result = annotate_program(program(action), load_taxonomy())
    assert result["families"]["action"] == expected
    secondary = {
        "center_markers_on_lines": ["mark_center", "removal"],
        "swap_shape_layer_positions": ["translation_alignment"],
    }
    if action["op"] in secondary:
        node = next(n for n in result["operations"] if n["path"] == "program.action")
        assert node["secondary_families"] == secondary[action["op"]]
        assert not set(node["secondary_families"]) & set(
            result["values"]["family.action"]
        )
    assert not {"fixed_translation", "alignment_centering"} & set(
        result["families"]["action"]
    )


def test_mixed_movement_branches_preserve_parameter_namespaces() -> None:
    """Combine movement branches while keeping displacement and target-gap parameters distinct."""
    example = program(
        {
            "op": "conditional",
            "condition": {
                "op": "object_pair_equivalent",
                "key": {"op": "object_shape"},
            },
            "if_true": {"op": "translate_object_group", "offset": [0, 2]},
            "if_false": {
                "op": "translate_to",
                "target_position": {
                    "op": "align_to_reference",
                    "axis": "col",
                    "offset": 2,
                    "self_edge": "left",
                    "reference_edge": "right",
                },
            },
        }
    )
    result = annotate_program(example, load_taxonomy())
    assert result["families"]["action"] == ["conditional", "translation_alignment"]
    values = result["values"]
    assert json.loads(values["parameter.program.action.if_true.offset"][0]) == [0, 2]
    assert values["parameter.program.action.if_false.target_position.offset"] == ["2"]
    assert values[
        "parameter.program.action.if_false.target_position.reference_edge"
    ] == ['"right"']


@pytest.mark.parametrize(
    "operation", ["mark_anchor", "mark_cross", "fill_center", "fill_each_center"]
)
def test_center_marking_keeps_single_and_multiple_cell_markers(operation: str) -> None:
    """Group center marks independently of their point, cross or per-object renderer.

    Args:
        operation (str): Declared center-marking operation.
    """
    example = program({"op": operation, "color": 3, "anchor": {"op": "object_center"}})
    result = annotate_program(example, load_taxonomy())
    assert result["families"]["action"] == ["mark_center"]


@pytest.mark.parametrize("dashed", [False, True])
def test_canvas_edge_coloring_stays_distinct_from_object_part_coloring(
    dashed: bool,
) -> None:
    """Distinguish the grid border from a selected object's own edge.

    Args:
        dashed (bool): Use a dashed rather than solid grid-border stripe.
    """
    action: dict[str, Any] = {
        "op": "mark_grid_edge",
        "color": 3,
        "edge": "top",
        "thickness": 1,
    }
    if dashed:
        action["pattern"] = {
            "op": "dashed",
            "dash_length": 1,
            "gap_length": 2,
            "gap_color": 0,
        }
    result = annotate_program(program(action), load_taxonomy())
    assert result["families"]["action"] == ["color_grid_edge"]
    example = program({"op": "recolor_cells", "color": 3})
    example["selector"] = {
        "op": "object_extreme_cells",
        "axis": "row",
        "mode": "min",
        "object_selector": {"op": "largest_object"},
    }
    assert annotate_program(example, load_taxonomy())["families"]["action"] == [
        "cell_coloring"
    ]
