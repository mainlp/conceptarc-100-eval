"""Action-concept taxonomy semantics, evidence paths, and snapshot compatibility."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arc_cd.repr_analysis.data import load_bundle
from arc_cd.repr_analysis.family_data import FamilyData, family_comparison_mask
from arc_cd.repr_analysis.population import CONCEPT_GROUPS
from arc_cd.repr_analysis.results import write_json
from arc_cd.repr_analysis.runner import run_analysis
from arc_cd.repr_analysis.spec_metadata import (
    annotate_program,
    load_spec_metadata,
    load_taxonomy,
    validate_saved_metadata,
)
from tests.test_family_analysis import metadata_bundle as _metadata_bundle
from tests.test_repr_population import feature_bundle as _feature_bundle
from tests.test_repr_population import population_sources as _population_sources

feature_bundle = _feature_bundle
metadata_bundle = _metadata_bundle
population_sources = _population_sources


ACTION_FAMILIES = {
    "CleanUp": "clean_up",
    "CompleteShape": "complete_shape",
    "Copy": "copy",
    "Count": "count",
    "ExtendToBoundary": "extend_to_boundary",
    "ExtractObjects": "extract_objects",
    "MoveToBoundary": "move_to_boundary",
    "Order": "order",
}


def annotate(
    action: dict[str, Any],
    selector: dict[str, Any] | None = None,
    layout: dict[str, Any] | None = None,
    taxonomy: dict[str, Any] | None = None,
    concept_name: str | None = None,
) -> dict[str, Any]:
    """Annotate independent DSL fragments while checking source immutability.

    Args:
        action (dict[str, Any]): Action expression under test.
        selector (dict[str, Any] | None): Optional selector expression.
        layout (dict[str, Any] | None): Optional input-layout expression.
        taxonomy (dict[str, Any] | None): Optional frozen taxonomy.
        concept_name (str | None): Optional declared task concept.

    Returns:
        dict[str, Any]: Exact operations, parameter evidence, and family tags.
    """
    program = {
        "layout": layout or {"op": "scattered_objects"},
        "selector": selector or {"op": "all_objects"},
        "action": action,
    }
    before = copy.deepcopy(program)
    result = annotate_program(
        program,
        taxonomy if taxonomy is not None else load_taxonomy(),
        concept_name=concept_name,
    )
    assert program == before
    return result


@pytest.mark.parametrize("concept", list(ACTION_FAMILIES))
def test_action_concept_is_the_final_task_label(concept: str) -> None:
    """Concept identity determines one action family without losing branch provenance.

    Args:
        concept (str): Declared action concept, independent of rendering details.
    """
    taxonomy = load_taxonomy()
    assert taxonomy["concept_action_families"] == ACTION_FAMILIES
    assert set(ACTION_FAMILIES) == set(CONCEPT_GROUPS["action"])
    action = {
        "op": "conditional",
        "if_true": {"op": "recolor_group", "color": 3},
        "if_false": {"op": "remove_group"},
    }
    raw = annotate(action)
    result = annotate(action, concept_name=concept)
    family = ACTION_FAMILIES[concept]
    assert result["families"]["action"] == [family]
    assert result["values"]["family.action"] == [family]
    assert taxonomy["families"]["action"][family]["label"] == concept
    assert result["action_family_evidence"] == {
        "source": "concept_name",
        "value": concept,
        "family": family,
    }
    assert result["operations"] == raw["operations"]
    assert result["parameters"] == raw["parameters"]
    for role in ("layout", "selector"):
        assert result["families"][role] == raw["families"][role]
    assert not any("operational" in field for field in result["values"])


def test_frozen_taxonomy_without_concept_labels_keeps_program_semantics() -> None:
    """Concept context does not retrofit action labels into old taxonomy snapshots."""
    taxonomy = load_taxonomy()
    taxonomy.pop("concept_action_families")
    action = {"op": "count", "renderer": {"op": "overlay_bar", "color": 2}}
    assert annotate(action, taxonomy=taxonomy, concept_name="Count") == annotate(
        action, taxonomy=taxonomy
    )
    assert annotate(action, concept_name="AboveBelow") == annotate(action)


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        [],
        {"": "count"},
        dict[str, Any](Count=[]),
        {"Count": "missing"},
        {"Count": "conditional"},
    ],
)
def test_invalid_concept_action_mapping(tmp_path: Path, mapping: Any) -> None:
    """Reject malformed concept rules and targets that are structural metadata.

    Args:
        tmp_path (Path): Temporary taxonomy directory.
        mapping (Any): Deliberately invalid concept-to-action mapping.
    """
    taxonomy = load_taxonomy()
    taxonomy["concept_action_families"] = mapping
    path = tmp_path / "taxonomy.json"
    write_json(path, taxonomy)
    with pytest.raises(ValueError, match="[Cc]oncept"):
        load_taxonomy(path)


@pytest.mark.parametrize("change", ["action_family", "evidence", "concept"])
def test_saved_action_labels_require_matching_concept(
    metadata_bundle: Path, change: str
) -> None:
    """Validate final labels and their concept provenance on frozen metadata reload.

    Args:
        metadata_bundle (Path): Complete metadata fixture with all concept groups.
        change (str): Annotation or concept field to corrupt.
    """
    metadata = json.loads((metadata_bundle / "task_metadata.json").read_text())
    taxonomy = load_taxonomy(metadata_bundle / "taxonomy.json")
    validate_saved_metadata(metadata, taxonomy)
    record = next(r for r in metadata["records"] if r["concept_name"] == "Copy")
    if change == "action_family":
        record["families"]["action"] = ["copying"]
    elif change == "evidence":
        record["action_family_evidence"]["value"] = "Count"
    else:
        record["concept_name"] = "Count"
    with pytest.raises(ValueError, match="annotations conflict"):
        validate_saved_metadata(metadata, taxonomy)


def test_conditional_is_excluded_independently_of_selector_option() -> None:
    """Control flow stays structural while all declared branch outcomes survive."""
    taxonomy = load_taxonomy()
    result = annotate(
        {
            "op": "conditional",
            "if_true": {"op": "identity"},
            "if_false": {"op": "remove_group"},
        },
        concept_name="SameDifferent",
    )
    assert result["families"]["action"] == ["conditional", "identity", "removal"]
    nodes = {node["path"]: node for node in result["operations"]}
    assert nodes["program.action.if_true"]["families"] == ["identity"]
    assert nodes["program.action.if_false"]["families"] == ["removal"]
    families = FamilyData(
        {},
        taxonomy,
        {},
        [
            "concept/SameDifferent",
            "action/conditional",
            "action/identity",
            "action/removal",
            "selector/collections",
        ],
        ["concept", "action", "action", "action", "selector"],
        ["SameDifferent", "conditional", "identity", "removal", "collections"],
        np.ones((1, 5), dtype=bool),
        ["concept", "action", "selector"],
    )
    np.testing.assert_array_equal(
        family_comparison_mask(families), [True, False, True, True, False]
    )
    np.testing.assert_array_equal(
        family_comparison_mask(families, include_structural_selectors=True),
        [True, False, True, True, True],
    )
    taxonomy["families"]["action"]["conditional"].pop("kind")
    assert family_comparison_mask(families, include_structural_selectors=True).all()


def test_secondary_details_do_not_remove_explicit_branch_actions() -> None:
    """Keep an explicit removal branch even when relocation also mentions removal."""
    result = annotate(
        {
            "op": "conditional",
            "if_true": {"op": "center_markers_on_lines"},
            "if_false": {"op": "remove_group"},
        },
        concept_name="SameDifferent",
    )
    assert result["families"]["action"] == [
        "conditional",
        "removal",
        "translation_alignment",
    ]
    assert result["values"]["family.action"] == result["families"]["action"]
    nodes = {node["path"]: node for node in result["operations"]}
    assert nodes["program.action.if_true"]["families"] == ["translation_alignment"]
    assert nodes["program.action.if_true"]["secondary_families"] == [
        "mark_center",
        "removal",
    ]
    assert nodes["program.action.if_false"]["families"] == ["removal"]


@pytest.mark.parametrize("change", ["missing", "altered", "primary_membership"])
def test_secondary_provenance_is_validated_on_reload(
    tmp_path: Path, change: str
) -> None:
    """Reject altered provenance and attempts to promote details to action labels.

    Args:
        tmp_path (Path): Temporary spec directory.
        change (str): Saved provenance or membership field to corrupt.
    """
    taxonomy = load_taxonomy()
    program = {
        "layout": {"op": "scattered_lines"},
        "selector": {"op": "line_objects"},
        "action": {"op": "center_markers_on_lines"},
    }
    write_json(
        tmp_path / "Center5.json",
        {
            "task_id": "Center/Center5",
            "concept_name": "Center",
            "program_kind": "conceptarc_dsl_v1",
            "description": "marker relocation",
            "program": program,
        },
    )
    bundle = load_spec_metadata(tmp_path, ["Center/Center5"], ["Center"], [0], taxonomy)
    metadata = {
        "schema_version": 1,
        "taxonomy_version": taxonomy["version"],
        "taxonomy": taxonomy,
        "records": bundle.records,
        "fields": bundle.fields,
    }
    validate_saved_metadata(metadata, taxonomy)
    record = metadata["records"][0]
    node = next(n for n in record["operations"] if n["path"] == "program.action")
    if change == "missing":
        node.pop("secondary_families")
    elif change == "altered":
        node["secondary_families"] = ["recolor"]
    else:
        record["families"]["action"].append("removal")
        record["values"]["family.action"].append("removal")
    with pytest.raises(ValueError, match="annotations conflict"):
        validate_saved_metadata(metadata, taxonomy)


def compose(placement: dict[str, Any], canvas: str = "input_canvas") -> dict[str, Any]:
    """Build a canvas composition with one explicit placement.

    Args:
        placement (dict[str, Any]): Reviewed or deliberately unsupported placement.
        canvas (str): Destination canvas operator.

    Returns:
        dict[str, Any]: Canvas action expression.
    """
    return {"op": "compose_canvas", "canvas": {"op": canvas}, "placements": [placement]}


@pytest.mark.parametrize(
    "action,expected",
    [
        ({"op": "count", "renderer": {"op": "overlay_bar", "color": 1}}, "counting"),
        (
            {
                "op": "draw_line",
                "mode": "ray",
                "stop": {"op": "before_selector", "selector": {"op": "largest_object"}},
            },
            "line_drawing",
        ),
        (
            {"op": "draw_lines", "lines": [{"op": "draw_line"}, {"op": "draw_line"}]},
            "line_drawing",
        ),
        ({"op": "draw_line_each", "line": {"op": "draw_line"}}, "line_drawing"),
        ({"op": "pack_object_patches", "canvas_mode": "input_size"}, "extract_place"),
        (
            {
                "op": "translate_each_to",
                "target_position": {"op": "adjacent_to_reference"},
            },
            "translation_alignment",
        ),
        (
            {"op": "translate_each_until_blocked", "direction": "right"},
            "translation_alignment",
        ),
        (
            {
                "op": "reorder_objects",
                "key": {"op": "object_area"},
                "order": "ascending",
            },
            "reordering",
        ),
        (
            {
                "op": "render_object_sequence",
                "attribute": {"op": "object_color"},
                "source_order": "selection",
                "renderer": {"op": "overlay_sequence"},
            },
            "sequence_rendering",
        ),
    ],
)
def test_action_meanings(action: dict[str, Any], expected: str) -> None:
    """Keep counting, drawing, extraction, movement, sorting, and encoding distinct.

    Args:
        action (dict[str, Any]): Action expression and relevant nested operators.
        expected (str): Semantic action family, independent of any concept label.
    """
    result = annotate(action)
    assert result["families"]["action"] == [expected]
    assert not [node for node in result["operations"] if node["status"] == "unmapped"]
    for node in result["operations"]:
        if node["op"] == "draw_line":
            assert node["role"] == "action"
        if node["op"] in ("object_area", "overlay_bar", "overlay_sequence"):
            assert node["role"] == "expression" and node["status"] == "helper"


@pytest.mark.parametrize(
    "placement",
    [
        {
            "op": "place_selection",
            "location": {"op": "selection_offset", "direction": "right", "offset": 2},
        },
        {
            "op": "place_selection",
            "location": {
                "op": "selection_grid_corner_pair",
                "corners": ["lower_left", "lower_right"],
            },
        },
        {
            "op": "place_selection",
            "target_position": {"op": "grid_boundary", "edge": "right"},
        },
        {
            "op": "place_selection",
            "anchor": {"op": "object_center", "selector": {"op": "smallest_marker"}},
        },
        {
            "op": "place_selection",
            "location": {
                "op": "matching_markers",
                "selector": {"op": "singleton_marker_cells"},
            },
        },
        {
            "op": "place_selection",
            "row": {
                "op": "add",
                "values": [
                    {"op": "input_height"},
                    {"op": "mul", "factor": -1, "value": {"op": "selection_height"}},
                ],
            },
            "col": {"op": "selection_min_col"},
        },
        {
            "op": "repeat_selection",
            "axis": "col",
            "copy_count": 3,
            "row": {"op": "selection_min_row"},
            "col": {"op": "selection_min_col"},
        },
        {
            "op": "repeat_selection",
            "axis": {
                "op": "aspect_axis",
                "choice": "longer",
                "source": "selection",
                "tie_axis": "row",
            },
            "copy_count": 2,
            "row": 0,
            "col": 0,
        },
        {"op": "place_each_object_at_matching_centers"},
    ],
)
def test_preserved_copy_placement(placement: dict[str, Any]) -> None:
    """Recognize explicit placement without inventing recoloring or moving tags.

    Args:
        placement (dict[str, Any]): Copy form used by the new concept specs.
    """
    result = annotate(compose(placement))
    assert result["families"]["action"] == ["copying"]
    root = next(
        node for node in result["operations"] if node["path"] == "program.action"
    )
    assert root["composition_evidence"][0]["case"] == "copy_to_input"
    assert root["composition_evidence"][0]["path"] == "program.action.placements"
    assert "color_edit_issues" not in root and "color_edit_evidence" not in root
    legacy = load_taxonomy()
    del legacy["composition_rules"]["copy_to_input"]
    assert annotate(compose(placement), taxonomy=legacy)["families"]["action"] == []


def test_canvas_and_target_precedence() -> None:
    """Canvas choice and effective target distinguish copying, extraction, and overlay."""
    placement = {
        "op": "place_selection",
        "target_position": {"op": "grid_boundary", "edge": "top"},
    }
    assert annotate(compose(placement))["families"]["action"] == ["copying"]
    assert annotate(compose(placement, "blank_canvas"))["families"]["action"] == [
        "extract_place"
    ]
    placement["location"] = {"op": "unknown_ignored_location"}
    assert annotate(compose(placement))["families"]["action"] == ["copying"]
    placement["target_position"] = {"op": "unknown_target"}
    placement["location"] = {
        "op": "selection_offset",
        "direction": "right",
        "offset": 2,
    }
    assert annotate(compose(placement))["families"]["action"] == []
    in_place = {
        "op": "place_selection",
        "row": {"op": "selection_min_row"},
        "col": {"op": "selection_min_col"},
    }
    assert annotate(compose(in_place))["families"]["action"] == ["selection_overlay"]
    assert annotate(compose(in_place, "blank_canvas"))["families"]["action"] == [
        "isolation"
    ]


@pytest.mark.parametrize(
    "placement",
    [
        {"op": "place_selection"},
        {"op": "place_selection", "row": 1},
        {"op": "place_selection", "row": True, "col": 1},
        {"op": "place_selection", "row": {"op": "add", "values": []}, "col": 0},
        {
            "op": "place_selection",
            "row": {"op": "mul", "factor": "unknown", "value": 2},
            "col": 0,
        },
        {"op": "place_selection", "location": {"op": "unknown_location"}},
        {
            "op": "place_selection",
            "anchor": {"op": "object_center"},
            "align": "unsupported",
        },
        {
            "op": "place_selection",
            "location": {"op": "selection_offset"},
            "render": {"op": "unknown_render"},
        },
        {"op": "repeat_selection", "axis": "col", "row": 0, "col": 0},
        {
            "op": "repeat_selection",
            "axis": "col",
            "copy_count": True,
            "row": 0,
            "col": 0,
        },
        {"op": "repeat_selection", "axis": "col", "copy_count": 1, "row": 0, "col": 0},
        {
            "op": "repeat_selection",
            "axis": "unknown",
            "copy_count": 2,
            "row": 0,
            "col": 0,
        },
        {"op": "place_each_object_at_matching_centers", "render": {"op": "preserve"}},
    ],
)
def test_unsupported_copy_forms_remain_unmapped(placement: dict[str, Any]) -> None:
    """Missing settings, unknown expressions, and unsupported rendering stay explicit.

    Args:
        placement (dict[str, Any]): Unsupported or insufficient placement evidence.
    """
    result = annotate(compose(placement))
    assert not result["families"]["action"]
    root = next(
        node for node in result["operations"] if node["path"] == "program.action"
    )
    assert root["status"] == "unmapped"


def test_mixed_copy_and_paint_does_not_claim_pure_copying() -> None:
    """A mixed composition retains paint evidence without a whole-action copy claim."""
    action = compose(
        {
            "op": "place_selection",
            "location": {"op": "selection_offset", "direction": "right", "offset": 2},
        }
    )
    action["placements"].append(
        {
            "op": "place_selection",
            "row": {"op": "selection_min_row"},
            "col": {"op": "selection_min_col"},
            "render": {"op": "solid", "color": 4},
        }
    )
    assert annotate(action)["families"]["action"] == ["recolor"]


@pytest.mark.parametrize(
    "key,expected",
    [
        ("object_area", "size"),
        ("object_width", "size"),
        ("object_height", "size"),
        ("object_non_dominant_color_count", "color"),
        ("object_outline_side_count", "shape_geometry"),
    ],
)
def test_extremum_measurement_evidence(key: str, expected: str) -> None:
    """Keep the measurement criterion separate from the min/max operation.

    Args:
        key (str): Exact DSL measurement operator.
        expected (str): Additional selection criterion.
    """
    result = annotate(
        {"op": "remove_group"},
        {
            "op": "extreme_object_by_key",
            "mode": "min",
            "key": {"op": key},
            "source": {"op": "all_objects"},
        },
    )
    assert set(result["families"]["selector"]) == {"collections", "extrema", expected}
    node = next(n for n in result["operations"] if n["path"] == "program.selector")
    assert node["parameter_evidence"][0]["path"] == "program.selector.key.op"
    assert node["parameter_evidence"][0]["value"] == key


def test_nested_selector_and_model_roles() -> None:
    """Reconstruction models remain expressions; embedded selectors keep their role."""
    selector = {
        "op": "inferred_footprint_cells",
        "part": "full",
        "source": {"op": "translated_pattern_target_cells", "axis": "col"},
        "model": {
            "op": "template_projection",
            "template_selector": {"op": "color_cluster"},
        },
        "fill": {"op": "model_correspondence"},
    }
    result = annotate({"op": "isolate_same_canvas"}, selector)
    nodes = {n["path"]: n for n in result["operations"]}
    assert nodes["program.selector.model"]["role"] == "expression"
    assert nodes["program.selector.model"]["status"] == "helper"
    assert nodes["program.selector.model.template_selector"]["role"] == "selector"
    assert "color" in result["families"]["selector"]
    assert "completion_reconstruction" in result["families"]["selector"]
    assert result["families"]["action"] == ["isolation"]


@pytest.mark.parametrize("count_groups", [False, True])
def test_repeated_pattern_selection_keeps_criteria(count_groups: bool) -> None:
    """Retain pattern-matching criteria beneath a structural count wrapper.

    Args:
        count_groups (bool): Wrap the pattern selector in a semantic group count.
    """
    selector = {
        "op": "repeated_multicolor_patterns",
        "background_color": "infer",
        "min_repetitions": 2,
    }
    result = annotate(
        {"op": "count"} if count_groups else {"op": "identity"},
        (
            {"op": "cell_patch_group_count", "source": selector}
            if count_groups
            else selector
        ),
    )
    criteria = {"color", "equivalence", "shape_geometry"}
    expected = criteria | ({"count_measurement"} if count_groups else set())
    assert set(result["families"]["selector"]) == expected
    pattern = next(
        node
        for node in result["operations"]
        if node["op"] == "repeated_multicolor_patterns"
    )
    assert pattern["path"] == (
        "program.selector.source" if count_groups else "program.selector"
    )
    assert pattern["role"] == "selector" and pattern["status"] == "mapped"
    assert set(pattern["families"]) == criteria
    assert result["values"]["parameter." + pattern["path"] + ".background_color"] == [
        '"infer"'
    ]
    assert not [node for node in result["operations"] if node["status"] == "unmapped"]


def test_semantic_count_extraction_is_structural() -> None:
    """Exclude measurement plumbing from headline selector criteria by default."""
    taxonomy = load_taxonomy()
    result = annotate({"op": "count"}, {"op": "distinct_colors"})
    assert result["families"]["selector"] == ["color", "count_measurement"]
    families = FamilyData(
        {},
        taxonomy,
        {},
        ["selector/color", "selector/count_measurement"],
        ["selector", "selector"],
        ["color", "count_measurement"],
        np.ones((1, 2), dtype=bool),
        ["selector"],
    )
    np.testing.assert_array_equal(family_comparison_mask(families), [True, False])
    np.testing.assert_array_equal(
        family_comparison_mask(families, include_structural_selectors=True),
        [True, True],
    )


@pytest.mark.parametrize(
    "layout,expected",
    [
        ({"op": "corner_missing_rectangles"}, {"fill_completion"}),
        ({"op": "polygon_outline_sequence"}, {"aligned_objects"}),
        ({"op": "directional_objects"}, {"directional_shapes"}),
        (
            {
                "op": "hollow_rectangles_with_scattered_cells",
                "corruption_region": "background",
            },
            {"containers_boundaries", "pattern_corruption"},
        ),
        (
            {"op": "periodic_pattern_with_overlaid_object"},
            {"overlap_occlusion", "repeated_patterns"},
        ),
        ({"op": "marker_decorated_panels"}, {"panels", "references_markers"}),
    ],
)
def test_new_layouts_do_not_imply_actions(
    layout: dict[str, Any], expected: set[str]
) -> None:
    """Describe input structures independently of the concept and transformation.

    Args:
        layout (dict[str, Any]): Declared input constructor and relevant settings.
        expected (set[str]): Overlapping layout families.
    """
    result = annotate({"op": "identity"}, layout=layout)
    assert set(result["families"]["layout"]) == expected
    assert result["families"]["action"] == ["identity"]


@pytest.mark.parametrize("analysis", ["cosine_distance"])
@pytest.mark.parametrize("population", ["predicate", "action", "all"])
def test_new_families_in_population_analyses(
    feature_bundle: Path,
    metadata_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    analysis: str,
    population: str,
) -> None:
    """Use new frozen annotations in all three quantitative population analyses.

    Args:
        feature_bundle (Path): Combined predicate/action feature fixture.
        metadata_bundle (Path): Metadata export with matching source identities.
        tmp_path (Path): Isolated analysis output directory.
        monkeypatch (pytest.MonkeyPatch): Suppress unrelated plot rendering.
        analysis (str): Analysis receiving the expanded family vocabulary.
        population (str): Shared predicate, action, or combined population.
    """
    monkeypatch.setattr(
        "arc_cd.repr_analysis.visualization.plot_analysis", lambda *args, **kwargs: None
    )
    config_path = metadata_bundle / "run_config.json"
    config = json.loads(config_path.read_text())
    specs = Path(config["arguments"]["spec_dir"])
    actions = {
        "Copy": compose(
            {
                "op": "place_selection",
                "location": {
                    "op": "selection_offset",
                    "direction": "right",
                    "offset": 2,
                },
            }
        ),
        "Count": {"op": "count", "renderer": {"op": "overlay_bar", "color": 3}},
        "ExtendToBoundary": {
            "op": "draw_line",
            "mode": "ray",
            "direction": "right",
            "stop": {"op": "grid_boundary"},
        },
        "ExtractObjects": {"op": "pack_object_patches"},
        "MoveToBoundary": {"op": "translate_each_until_blocked", "direction": "right"},
        "Order": {"op": "render_object_sequence", "attribute": {"op": "object_color"}},
    }
    for path in specs.glob("*/*.json"):
        spec = json.loads(path.read_text())
        concept = spec["concept_name"]
        if concept in actions:
            spec["program"]["action"] = actions[concept]
            if concept == "Count":
                spec["program"]["selector"] = {
                    "op": "cell_patch_group_count",
                    "source": {
                        "op": "repeated_multicolor_patterns",
                        "background_color": "infer",
                        "min_repetitions": 2,
                    },
                }
            write_json(path, spec)
    bundle = load_bundle(feature_bundle)
    taxonomy = load_taxonomy(metadata_bundle / "taxonomy.json")
    metadata = load_spec_metadata(
        specs, bundle.task_ids, bundle.concepts, bundle.labels.tolist(), taxonomy
    )
    write_json(
        metadata_bundle / "task_metadata.json",
        {
            "schema_version": 1,
            "taxonomy_version": taxonomy["version"],
            "taxonomy": taxonomy,
            "records": metadata.records,
            "fields": metadata.fields,
        },
    )
    config["spec_fingerprints"] = metadata.fingerprints
    write_json(config_path, config)
    destination = run_analysis(
        analysis,
        [
            "--feature_dir",
            str(feature_bundle),
            "--metadata_dir",
            str(metadata_bundle),
            "--output_root",
            str(tmp_path / "out"),
            "--population",
            population,
            "--positions",
            "S2",
            "--layers",
            "2",
            "--num_threads",
            "1",
            "--min_family_tasks",
            "1",
        ],
    )
    axes = json.loads((destination / "axes.json").read_text())
    memberships = np.load(destination / "memberships.npy")
    assert axes["population"]["num_parents"] == (32 if population == "all" else 16)
    if population in ("action", "all"):
        for concept, family in ACTION_FAMILIES.items():
            index = axes["family_ids"].index(f"action/{family}")
            selected = [task.split("/")[0] == concept for task in axes["task_ids"]]
            np.testing.assert_array_equal(memberships[:, index], selected)
        action_columns = np.asarray(axes["family_roles"]) == "action"
        action_rows = [
            task.split("/")[0] in ACTION_FAMILIES for task in axes["task_ids"]
        ]
        np.testing.assert_array_equal(
            memberships[action_rows][:, action_columns].sum(axis=1), 1
        )
        index = axes["family_ids"].index("selector/count_measurement")
        assert axes["family_sizes"][index] == 2
        assert not axes["comparison_included"][index]
        assert not axes["headline_eligible"][index]
        count_rows = [task.split("/")[0] == "Count" for task in axes["task_ids"]]
        for criterion in ("color", "equivalence", "shape_geometry"):
            index = axes["family_ids"].index(f"selector/{criterion}")
            assert memberships[count_rows, index].all()
            assert axes["comparison_included"][index]
    if population in ("predicate", "all"):
        index = axes["family_ids"].index("action/conditional")
        assert not axes["comparison_included"][index]
        assert not axes["headline_eligible"][index]
        assert axes["headline_exclusion_reasons"][index] == "structural_action"
