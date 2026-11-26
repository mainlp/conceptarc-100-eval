"""Direction-aware fill families, semantic equivalence, and frozen compatibility."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from arc_cd.repr_analysis.spec_metadata import annotate_program, load_taxonomy


def annotate(
    layout: dict[str, Any], selector: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    """Annotate a fixture and verify input immutability.

    Args:
        layout (dict[str, Any]): Input layout constraints.
        selector (dict[str, Any]): Selected objects or completion regions.
        action (dict[str, Any]): Declared action.

    Returns:
        dict[str, Any]: Full annotations including contextual evidence.
    """
    program = {"layout": layout, "selector": selector, "action": action}
    before = copy.deepcopy(program)
    result = annotate_program(program, load_taxonomy())
    assert program == before
    return result


def placement(selector: dict[str, Any], color: int | None) -> dict[str, Any]:
    """Build an in-place completion edit.

    Args:
        selector (dict[str, Any]): Explicit placement target.
        color (int | None): Solid paint color or preserved/inferred source colors.

    Returns:
        dict[str, Any]: Placement expression.
    """
    return {
        "op": "place_selection",
        "row": {"op": "selection_min_row"},
        "col": {"op": "selection_min_col"},
        "selector": selector,
        **({"render": {"op": "solid", "color": color}} if color is not None else {}),
    }


@pytest.mark.parametrize(
    "layout,selector,expected",
    [
        (
            {"op": "scattered_objects"},
            {"op": "objects_by_fill_state", "state": "unfilled"},
            "filling",
        ),
        (
            {"op": "scattered_objects"},
            {"op": "objects_by_fill_state", "state": "filled"},
            "hollowing",
        ),
        (
            {"op": "scattered_objects", "shape_styles": ["hollow_rectangle"]},
            {"op": "all_objects"},
            "filling",
        ),
        (
            {"op": "scattered_objects", "shape_styles": ["filled_rectangle"]},
            {"op": "all_objects"},
            "hollowing",
        ),
        (
            {
                "op": "corner_stacked_objects",
                "shape_styles": ["filled_rectangle", "hollow_rectangle"],
            },
            {"op": "all_objects"},
            "fill_toggle",
        ),
        (
            {
                "op": "central_object_with_distractors",
                "center_fill_states": ["filled", "unfilled"],
            },
            {"op": "central_object"},
            "fill_toggle",
        ),
        (
            {
                "op": "gapped_aligned_objects",
                "larger_shape_styles": ["hollow_rectangle"],
                "smaller_shape_styles": ["filled_rectangle"],
            },
            {"op": "all_objects"},
            "fill_toggle",
        ),
        (
            {
                "op": "hollow_container_with_inside_outside_objects",
                "object_shape": "geometry_safe_rectangles",
            },
            {"op": "inside_container_objects"},
            "hollowing",
        ),
    ],
)
def test_toggle_directions(
    layout: dict[str, Any], selector: dict[str, Any], expected: str
) -> None:
    """Equivalent DSL toggles acquire only the directions allowed by their context.

    Args:
        layout (dict[str, Any]): Layout constraints.
        selector (dict[str, Any]): Target selector.
        expected (str): Single direction family.
    """
    result = annotate(layout, selector, {"op": "toggle_object_group_fill"})
    assert result["families"]["action"] == [expected]
    node = next(n for n in result["operations"] if n["role"] == "action")
    assert node["fill_state_evidence"] and not node.get("fill_state_issues")


def test_reference_fill_state_does_not_constrain_candidates() -> None:
    """A filled reference does not turn a hollow-candidate operation into hollowing."""
    result = annotate(
        {"op": "reference_object_with_objects"},
        {
            "op": "objects_relative_to_reference",
            "reference_selector": {"op": "objects_by_fill_state", "state": "filled"},
            "candidate_selector": {"op": "all", "of": {"op": "hollow_objects"}},
        },
        {"op": "toggle_object_group_fill"},
    )
    assert result["families"]["action"] == ["filling"]


@pytest.mark.parametrize(
    "depth,expected", [("top", "hollowing"), ("bottom", "fill_toggle")]
)
def test_layer_toggle_uses_selected_depth(depth: str, expected: str) -> None:
    """Foreground and reconstructed lower layers can have different allowed states.

    Args:
        depth (str): Selected layer depth.
        expected (str): Contextual action family.
    """
    result = annotate(
        {
            "op": "layered_objects",
            "count": 2,
            "overlap": "required",
            "upper_shape_styles": ["filled_rectangle"],
            "lower_shape_styles": ["filled_rectangle", "hollow_rectangle"],
        },
        {"op": "shape_instance_by_depth", "depth": depth},
        {"op": "toggle_shape_layer_fill"},
    )
    assert result["families"]["action"] == [expected]


@pytest.mark.parametrize(
    "geometry,expected",
    [
        ("enclosed_region", "fill_toggle"),
        ("open_row_span", "fill_toggle"),
        ("parallel_strip", "filling+removal"),
    ],
)
def test_inversion_preserves_boundary_distinction(geometry: str, expected: str) -> None:
    """Erasing a strip's required cells removes its boundary, unlike container interiors.

    Args:
        geometry (str): Completion-region geometry.
        expected (str): Expected family names separated by plus signs.
    """
    selector = {
        "op": "completion_region_cells",
        "geometry": geometry,
        "part": "required",
        "state": "complete",
    }
    missing = {**selector, "part": "missing", "state": "incomplete"}
    result = annotate(
        {"op": "scattered_objects"},
        selector,
        {
            "op": "compose_canvas",
            "canvas": {"op": "input_canvas"},
            "placements": [placement(selector, 0), placement(missing, None)],
        },
    )
    assert result["families"]["action"] == expected.split("+")


def test_patterned_inversion_retains_pattern_provenance() -> None:
    """A patterned inverse task belongs only to switching among the direction families."""
    selector = {
        "op": "completion_region_cells",
        "part": "required",
        "state": "complete",
        "model": {"op": "alternating", "axis": "checkerboard"},
    }
    result = annotate(
        {"op": "scattered_objects"},
        selector,
        {
            "op": "compose_canvas",
            "canvas": {"op": "input_canvas"},
            "placements": [
                placement(selector, 0),
                placement({**selector, "part": "missing", "state": "incomplete"}, None),
            ],
        },
    )
    assert result["families"]["action"] == ["fill_toggle"]
    node = next(n for n in result["operations"] if n["op"] == "compose_canvas")
    assert any(e["patterned"] for e in node["color_edit_evidence"])
    assert any(
        e["case"] == "bidirectional_program" for e in node["fill_state_evidence"]
    )


@pytest.mark.parametrize("patterned", [False, True])
def test_missing_patch_alone_is_isolation(patterned: bool) -> None:
    """A standalone inferred patch does not complete an object in the output scene.

    Args:
        patterned (bool): Whether the inferred patch follows an alternating model.
    """
    selector = {
        "op": "completion_region_cells",
        "part": "missing",
        "model": {"op": "alternating" if patterned else "solid"},
    }
    result = annotate(
        {"op": "scattered_objects"},
        selector,
        {
            "op": "compose_canvas",
            "canvas": {"op": "blank_canvas"},
            "placements": [placement(selector, None)],
        },
    )
    assert result["families"]["action"] == ["isolation"]


def test_unknown_and_contradictory_contexts_are_flagged() -> None:
    """Unsupported states never silently imply that both directions are possible."""
    cases: list[tuple[dict[str, Any], dict[str, Any]]] = [
        ({"op": "scattered_objects"}, {"op": "all_objects"}),
        (
            {"op": "scattered_objects", "shape_styles": ["filled_rectangle"]},
            {"op": "hollow_objects"},
        ),
    ]
    for layout, selector in cases:
        result = annotate(layout, selector, {"op": "toggle_object_fill"})
        assert not result["families"]["action"]
        assert any(n.get("fill_state_issues") for n in result["operations"])


def test_conditional_directions_form_one_parent_family() -> None:
    """Declared branch alternatives stay traceable under one bidirectional task tag."""
    selector = {
        "op": "completion_region_cells",
        "part": "interior",
        "state": "incomplete",
    }
    result = annotate(
        {"op": "scattered_objects"},
        selector,
        {
            "op": "conditional",
            "if_true": {"op": "recolor_cells", "color": 3},
            "if_false": {"op": "recolor_cells", "color": 0},
        },
    )
    assert result["families"]["action"] == ["conditional", "fill_toggle"]
    assert result["values"]["family.action"] == ["conditional", "fill_toggle"]


def test_legacy_profile_retains_toggle_and_isolated_fill_memberships() -> None:
    """Opting out reproduces both legacy memberships without changing the program."""
    taxonomy = load_taxonomy()
    taxonomy.pop("fill_state_rules")
    program: dict[str, Any] = {
        "layout": {"op": "hollow_objects"},
        "selector": {"op": "all_objects"},
        "action": {"op": "toggle_object_fill"},
    }
    assert annotate_program(program, taxonomy)["families"]["action"] == ["fill_toggle"]
    selector = {"op": "completion_region_cells", "part": "missing"}
    program["selector"] = selector
    program["action"] = {
        "op": "compose_canvas",
        "canvas": {"op": "blank_canvas"},
        "placements": [placement(selector, None)],
    }
    assert annotate_program(program, taxonomy)["families"]["action"] == [
        "filling",
        "isolation",
    ]


@pytest.mark.parametrize(
    "change", ["version", "family", "duplicate", "shape", "layout"]
)
def test_invalid_fill_profiles(tmp_path: Path, change: str) -> None:
    """Reject malformed contextual profiles before annotating any tasks.

    Args:
        tmp_path (Path): Temporary taxonomy path.
        change (str): Invalid field to introduce.
    """
    taxonomy = load_taxonomy()
    rules = taxonomy["fill_state_rules"]
    if change == "version":
        rules["version"] = 2
    elif change == "family":
        rules["families"]["fill"] = "unknown"
    elif change == "duplicate":
        rules["families"]["fill"] = rules["families"]["switch"]
    elif change == "shape":
        rules["shape_states"]["filled_rectangle"] = "unknown"
    else:
        rules["layout_shape_parameters"]["scattered_objects"] = "shape_styles"
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(taxonomy))
    with pytest.raises(ValueError, match="Fill-state"):
        load_taxonomy(path)


@pytest.mark.parametrize("depth", ["top", "bottom"])
def test_pair_fill_constraints_override_generic_role_parameters(depth: str) -> None:
    """Dedicated pair styles determine fill direction even with inactive role fields.

    Args:
        depth (str): Selected pair member.
    """
    result = annotate(
        {
            "op": "layered_objects",
            "count": 2,
            "object_pair_equivalence_key": "object_dimensions",
            "placement_relation_modes": ["overlapping"],
            "object_shape_styles": ["hollow_rectangle"],
            "upper_shape_styles": ["filled_rectangle"],
            "lower_shape_styles": ["filled_rectangle"],
        },
        {"op": "shape_instance_by_depth", "depth": depth},
        {"op": "toggle_shape_layer_fill"},
    )
    assert result["families"]["action"] == ["filling"]


@pytest.mark.parametrize(
    "parameters",
    [
        {"count": 2, "overlap": "never"},
        {"count": 2, "overlap": "optional"},
        {"count": 2},
        {"count": 1, "overlap": "required"},
        {"count": 3, "overlap": "required"},
        {"min_count": 2, "max_count": 5, "overlap": "required"},
    ],
)
def test_inactive_or_ambiguous_layer_roles_do_not_imply_direction(
    parameters: dict[str, Any],
) -> None:
    """Flag unsupported role constraints rather than inventing a fill direction.

    Args:
        parameters (dict[str, Any]): Count and overlap declarations.
    """
    result = annotate(
        {
            "op": "layered_objects",
            "upper_shape_styles": ["filled_rectangle"],
            **parameters,
        },
        {"op": "shape_instance_by_depth", "depth": "top"},
        {"op": "toggle_shape_layer_fill"},
    )
    assert result["families"]["action"] == []
    assert any(n.get("fill_state_issues") for n in result["operations"])


def test_generic_layer_catalog_does_not_use_pair_style_field() -> None:
    """The generic generator ignores object_shape_styles when choosing its catalog."""
    result = annotate(
        {
            "op": "layered_objects",
            "count": 2,
            "overlap": "required",
            "object_shape_styles": ["filled_rectangle"],
        },
        {"op": "shape_instance_by_depth", "depth": "top"},
        {"op": "toggle_shape_layer_fill"},
    )
    assert result["families"]["action"] == []
    assert any(n.get("fill_state_issues") for n in result["operations"])


@pytest.mark.parametrize("policy", ["required", "always", "true"])
def test_required_pair_overlap_aliases_preserve_fill_direction(policy: str) -> None:
    """Resolve valid overlap aliases and a fixed range when count is null.

    Args:
        policy (str): Required-overlap spelling accepted by the generator.
    """
    result = annotate(
        {
            "op": "layered_objects",
            "count": None,
            "min_count": 2,
            "max_count": 2,
            "overlap": policy,
            "upper_shape_styles": ["filled_rectangle"],
        },
        {"op": "shape_instances_by_occlusion", "relation": "fully_visible"},
        {"op": "toggle_shape_layer_fill"},
    )
    assert result["families"]["action"] == ["hollowing"]
