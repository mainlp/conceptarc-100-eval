"""Behavioral distinctions between object recoloring, filling, marking, and erasure."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from arc_cd.repr_analysis.spec_metadata import annotate_program, load_taxonomy


def annotation(action: dict[str, Any], selector: dict[str, Any]) -> dict[str, Any]:
    """Annotate a small program without mutating its declarations.

    Args:
        action (dict[str, Any]): Action under test.
        selector (dict[str, Any]): Program-level target selector.

    Returns:
        dict[str, Any]: Validated semantic annotations.
    """
    program = {
        "layout": {"op": "scattered_objects"},
        "selector": selector,
        "action": action,
    }
    before = copy.deepcopy(program)
    result = annotate_program(program, load_taxonomy())
    assert program == before
    return result


def canvas_action(
    render: dict[str, Any] | None, canvas: str = "input_canvas"
) -> dict[str, Any]:
    """Build one placement with explicit unchanged coordinates.

    Args:
        render (dict[str, Any] | None): Renderer or inferred/preserved source colors.
        canvas (str): Canvas expression.

    Returns:
        dict[str, Any]: Compose action with one placement.
    """
    return {
        "op": "compose_canvas",
        "canvas": {"op": canvas},
        "placements": [
            {
                "op": "place_selection",
                "row": {"op": "selection_min_row"},
                "col": {"op": "selection_min_col"},
                **({"render": render} if render is not None else {}),
            }
        ],
    }


@pytest.mark.parametrize(
    "op,parameters,expected",
    [
        ("recolor", {"color": 4}, {"recolor"}),
        ("recolor_group", {"color": 0}, {"removal"}),
        (
            "recolor_shape_layers",
            {"color": {"op": "literal_color", "color": 7}},
            {"recolor"},
        ),
        (
            "recolor_group_by_orientation",
            {"horizontal_color": 0, "vertical_color": 2},
            {"recolor", "removal"},
        ),
        (
            "recolor_object_pattern",
            {"color": 2, "pattern": {"op": "alternating", "mode": "horizontal"}},
            {"pattern_recolor"},
        ),
        (
            "recolor_object_group_pattern",
            {"color": 2, "pattern": {"op": "alternating", "mode": "vertical"}},
            {"pattern_recolor"},
        ),
        (
            "recolor_shape_layers_pattern",
            {"color": 2, "pattern": {"op": "alternating", "mode": "checkerboard"}},
            {"pattern_recolor"},
        ),
        ("swap_shape_layer_colors", {}, {"recolor"}),
    ],
)
def test_whole_object_edits(
    op: str, parameters: dict[str, Any], expected: set[str]
) -> None:
    """Patterned operations do not imply uniform recoloring; background means erasure.

    Args:
        op (str): Declared color primitive.
        parameters (dict[str, Any]): Palette and pattern.
        expected (set[str]): Semantic memberships.
    """
    result = annotation({"op": op, **parameters}, {"op": "all_objects"})
    assert set(result["families"]["action"]) == expected


@pytest.mark.parametrize(
    "selector,expected",
    [
        ({"op": "oriented_color_band_cells", "orientation": "horizontal"}, "recolor"),
        ({"op": "object_extreme_cells", "axis": "col", "mode": "max"}, "cell_coloring"),
        (
            {"op": "cells_relative_to_reference", "through": "non_background"},
            "cell_coloring",
        ),
        (
            {"op": "cells_relative_to_reference", "through": "any", "mode": "line"},
            "region_coloring",
        ),
        (
            {
                "op": "intersection",
                "selectors": [
                    {"op": "region_cells", "region": "non_background"},
                    {"op": "cells_relative_to_reference", "through": "occupied"},
                ],
            },
            "cell_coloring",
        ),
    ],
)
def test_cell_selectors_determine_scope(
    selector: dict[str, Any], expected: str
) -> None:
    """A cell primitive can select whole objects, object parts, or spatial regions.

    Args:
        selector (dict[str, Any]): Effective target.
        expected (str): Non-patterned family.
    """
    assert annotation({"op": "recolor_cells", "color": 3}, selector)["families"][
        "action"
    ] == [expected]
    patterned = annotation(
        {
            "op": "recolor_cells_pattern",
            "colors": [2, 4],
            "pattern": {"op": "alternating", "mode": "row"},
        },
        selector,
    )
    assert patterned["families"]["action"] == [
        "pattern_recolor" if expected == "recolor" else expected
    ]


@pytest.mark.parametrize(
    "part,state,model,render,expected",
    [
        ("object", "incomplete", "solid", {"op": "solid", "color": 3}, "recolor"),
        (
            "object",
            "incomplete",
            "solid",
            {"op": "alternating", "colors": [2, 4]},
            "pattern_recolor",
        ),
        ("present", "complete", "solid", {"op": "solid", "color": 3}, "cell_coloring"),
        ("required", "complete", "solid", {"op": "solid", "color": 3}, "cell_coloring"),
        ("missing", "incomplete", "solid", {"op": "solid", "color": 3}, "filling"),
        ("missing", "incomplete", "solid", None, "filling"),
        ("missing", "incomplete", "alternating", None, "pattern_filling"),
        (
            "missing",
            "incomplete",
            "solid",
            {"op": "alternating", "colors": [2, 4]},
            "pattern_filling",
        ),
        ("interior", "incomplete", "solid", {"op": "solid", "color": 3}, "filling"),
        (
            "interior",
            "incomplete",
            "solid",
            {"op": "alternating", "colors": [2, 4]},
            "pattern_filling",
        ),
        (
            "interior",
            "complete",
            "alternating",
            {"op": "alternating", "colors": [2, 4]},
            "pattern_filling",
        ),
        ("present", "incomplete", "solid", {"op": "solid", "color": 0}, "hollowing"),
    ],
)
def test_completion_targets_and_palette(
    part: str, state: str, model: str, render: dict[str, Any] | None, expected: str
) -> None:
    """Completion state and selected part control interpretation of identical renderers.

    Args:
        part (str): Selected portion of the region.
        state (str): Declared completion state.
        model (str): Completion model.
        render (dict[str, Any] | None): Palette or preserved/inferred colors.
        expected (str): Expected family.
    """
    selector = {
        "op": "completion_region_cells",
        "geometry": "enclosed_region",
        "part": part,
        "state": state,
        "model": {"op": model},
    }
    result = annotation(canvas_action(render), selector)
    assert result["families"]["action"] == [expected]
    evidence = next(n for n in result["operations"] if n["op"] == "compose_canvas")[
        "color_edit_evidence"
    ]
    assert all(e["selector_path"] == "program.selector" for e in evidence)


def test_blank_canvas_keeps_output_context() -> None:
    """An inferred patch alone is isolation, with its completion provenance retained."""
    action = canvas_action({"op": "solid", "color": 3}, "blank_canvas")
    selector = {
        "op": "completion_region_cells",
        "part": "missing",
        "state": "incomplete",
    }
    result = annotation(action, selector)
    assert result["families"]["action"] == ["isolation"]
    action["placements"][0]["render"]["color"] = 0
    result = annotation(action, selector)
    assert "removal" not in result["families"]["action"]
    assert any(n.get("color_edit_issues") for n in result["operations"])


def test_mixed_edits_and_inherited_branch_selector() -> None:
    """Preserve separate edits and exact selector provenance in conditional branches."""
    action = canvas_action({"op": "solid", "color": 0})
    action["placements"][0]["selector"] = {
        "op": "completion_region_cells",
        "part": "object",
        "state": "complete",
    }
    action["placements"].append(canvas_action(None)["placements"][0])
    selector = {
        "op": "completion_region_cells",
        "part": "missing",
        "state": "incomplete",
        "model": {"op": "alternating"},
    }
    result = annotation(
        {
            "op": "conditional",
            "if_true": action,
            "if_false": {"op": "recolor_group", "color": 3},
        },
        selector,
    )
    assert result["families"]["action"] == [
        "conditional",
        "pattern_filling",
        "recolor",
        "removal",
    ]
    node = next(n for n in result["operations"] if n["op"] == "compose_canvas")
    paths = {e["case"]: e["selector_path"] for e in node["color_edit_evidence"]}
    assert paths == {
        "erase": "program.action.if_true.placements[0].selector",
        "fill_pattern": "program.selector",
    }


def test_partial_unknown_edit_stays_visible_for_review() -> None:
    """One mapped placement must not hide an unresolved second placement."""
    action = canvas_action({"op": "solid", "color": 3})
    second = copy.deepcopy(action["placements"][0])
    second["selector"] = {"op": "unknown_selector"}
    action["placements"].append(second)
    result = annotation(action, {"op": "all_objects"})
    assert result["families"]["action"] == ["recolor"]
    node = next(n for n in result["operations"] if n["op"] == "compose_canvas")
    assert node["color_edit_issues"] == [
        {
            "path": "program.action.placements[1]",
            "reason": "unsupported_target_selector",
        }
    ]


@pytest.mark.parametrize("change", ["version", "case", "family", "selector"])
def test_invalid_semantic_profiles(tmp_path: Path, change: str) -> None:
    """Reject incompatible or misspelled overrides before annotation.

    Args:
        tmp_path (Path): Temporary taxonomy path.
        change (str): Invalid field to introduce.
    """
    taxonomy = load_taxonomy()
    rules = taxonomy["color_edit_rules"]
    if change == "version":
        rules["version"] = 2
    elif change == "case":
        rules["cases"].pop("fill_uniform")
    elif change == "family":
        rules["cases"]["fill_uniform"]["family"] = "missing_family"
    else:
        rules["object_selectors"].append(rules["object_selectors"][0])
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(taxonomy))
    with pytest.raises(ValueError, match="[Cc]olor-edit"):
        load_taxonomy(path)


def test_v1_2_render_only_rules_remain_reproducible() -> None:
    """Frozen render-only profiles retain their original overlapping labels."""
    taxonomy = load_taxonomy()
    taxonomy.pop("color_edit_rules")
    taxonomy.pop("fill_state_rules")
    taxonomy["version"] = "1.2.0"
    taxonomy["composition_rules"]["alternating_render"] = {
        "families": ["recolor", "pattern_recolor"],
        "description": "Legacy render-only rule",
    }
    result = annotate_program(
        {
            "layout": {"op": "scattered_objects"},
            "selector": {"op": "completion_region_cells", "part": "missing"},
            "action": canvas_action({"op": "alternating", "colors": [2, 3]}),
        },
        taxonomy,
    )
    assert result["families"]["action"] == ["pattern_recolor", "recolor"]
    assert all(
        "color_edit_evidence" not in n and "color_edit_issues" not in n
        for n in result["operations"]
    )


@pytest.mark.parametrize(
    "part,render,expected",
    [
        ("interior", {"op": "solid", "color": 3}, ["filling"]),
        ("required", None, ["pattern_filling"]),
        ("missing", {"op": "solid", "color": 0}, []),
    ],
)
def test_pattern_precedence_and_missing_cell_erasure(
    part: str, render: dict[str, Any] | None, expected: list[str]
) -> None:
    """Uniformly painting an interior overrides its model; empty cells cannot be erased.

    Args:
        part (str): Selected completion part.
        render (dict[str, Any] | None): Palette or preserved colors.
        expected (list[str]): Expected family tags.
    """
    selector = {
        "op": "completion_region_cells",
        "state": "incomplete",
        "part": part,
        "model": {"op": "alternating"},
    }
    result = annotation(canvas_action(render), selector)
    assert result["families"]["action"] == expected


def test_background_paint_can_be_skipped_by_placement() -> None:
    """A renderer whose background pixels are skipped must not acquire removal tags."""
    action = canvas_action({"op": "solid", "color": 0})
    action["placements"][0]["include_background"] = False
    result = annotation(action, {"op": "all_objects"})
    assert "removal" not in result["families"]["action"]
    assert any(n.get("color_edit_issues") for n in result["operations"])
