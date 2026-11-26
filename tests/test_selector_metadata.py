"""Selector criteria, overlapping families, and explicit color-filter provenance."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from arc_cd.repr_analysis.spec_metadata import annotate_program, load_taxonomy


def annotate(selector: dict[str, Any]) -> dict[str, Any]:
    """Annotate one selector without modifying its source program.

    Args:
        selector (dict[str, Any]): Root selector expression.

    Returns:
        dict[str, Any]: Complete operation and parameter annotations.
    """
    program = {
        "layout": {"op": "scattered_objects"},
        "selector": selector,
        "action": {"op": "identity"},
    }
    before = copy.deepcopy(program)
    result = annotate_program(program, load_taxonomy())
    assert program == before
    assert result["families"]["action"] == ["identity"]
    assert result["families"]["layout"] == ["scattered_objects"]
    return result


@pytest.mark.parametrize(
    "op,expected",
    [
        ("extreme_object_by_position", {"spatial_relations", "extrema"}),
        ("object_extreme_cells", {"spatial_relations", "extrema"}),
        ("central_object", {"spatial_relations", "centers_anchors"}),
        ("off_center_marker", {"spatial_relations", "centers_anchors"}),
        ("panel_deviation_objects", {"collections"}),
        ("same_color_object_pair", {"collections", "color", "equivalence"}),
        ("object_pair", {"collections"}),
        (
            "complete_shape_color_objects",
            {"color", "completion_reconstruction", "depth_occlusion", "shape_geometry"},
        ),
        ("objects_by_fill_state", {"fill_state"}),
        ("hollow_objects", {"fill_state"}),
        ("completion_region_cells", {"completion_reconstruction"}),
        ("inferred_footprint_cells", {"completion_reconstruction"}),
        ("inferred_pattern_cells", {"completion_reconstruction"}),
    ],
)
def test_audited_selector_criteria(op: str, expected: set[str]) -> None:
    """Keep matching, spatial, and completion criteria distinct from extraction.

    Args:
        op (str): Reviewed DSL operation.
        expected (set[str]): Its exact overlapping families.
    """
    assert set(annotate({"op": op})["families"]["selector"]) == expected


def test_nested_fill_and_completion_are_both_preserved() -> None:
    """A reconstruction using hollow-object candidates carries both criteria."""
    result = annotate(
        {
            "op": "inferred_footprint_cells",
            "source": {"op": "hollow_objects"},
            "part": "missing",
        }
    )
    assert result["families"]["selector"] == ["completion_reconstruction", "fill_state"]
    assert result["values"]["family.selector"] == result["families"]["selector"]


@pytest.mark.parametrize(
    "parameters,paths",
    [
        ({}, []),
        ({"color": 0}, ["color"]),
        ({"color": 8}, ["color"]),
        ({"color": {"op": "literal_color", "color": 8}}, ["color"]),
        ({"color": 8, "color_selector": None}, []),
        (
            {"color": 8, "color_selector": {"op": "literal_color", "color": 3}},
            ["color_selector"],
        ),
        ({"color_selector": None, "exclude_color": 0}, ["exclude_color"]),
        ({"color": 8, "exclude_color": 0}, ["color", "exclude_color"]),
        ({"color": True}, []),
        ({"color": 12}, []),
    ],
)
def test_color_filter_precedence(parameters: dict[str, Any], paths: list[str]) -> None:
    """Use only effective color fields and retain their literal/expression values.

    Args:
        parameters (dict[str, Any]): Optional color criteria.
        paths (list[str]): Effective field names in evidence order.
    """
    result = annotate({"op": "region_cells", "region": "non_background", **parameters})
    root = next(n for n in result["operations"] if n["path"] == "program.selector")
    evidence = root.get("parameter_evidence", [])
    assert [e["path"] for e in evidence] == [f"program.selector.{p}" for p in paths]
    assert [e["value"] for e in evidence] == [parameters[p] for p in paths]
    assert ("color" in root["families"]) is bool(paths)


def test_color_filter_in_nested_action_selector() -> None:
    """Parameter criteria retain the selector role even inside a canvas action."""
    program = {
        "layout": {"op": "scattered_objects"},
        "selector": {"op": "all_objects"},
        "action": {
            "op": "compose_canvas",
            "canvas": {"op": "input_canvas"},
            "placements": [
                {
                    "op": "place_selection",
                    "row": {"op": "selection_min_row"},
                    "col": {"op": "selection_min_col"},
                    "selector": {
                        "op": "region_cells",
                        "region": "non_background",
                        "color": 8,
                    },
                }
            ],
        },
    }
    result = annotate_program(program, load_taxonomy())
    path = "program.action.placements[0].selector"
    node = next(n for n in result["operations"] if n["path"] == path)
    assert node["role"] == "selector" and node["families"] == ["collections", "color"]
    assert node["parameter_evidence"][0]["path"] == path + ".color"


def test_legacy_snapshot_does_not_gain_parameter_tags() -> None:
    """The new color criteria are opt-in through the frozen taxonomy snapshot."""
    taxonomy = load_taxonomy()
    taxonomy.pop("selector_rules")
    result = annotate_program(
        {
            "layout": {"op": "scattered_objects"},
            "selector": {"op": "region_cells", "color": 8},
            "action": {"op": "identity"},
        },
        taxonomy,
    )
    assert result["families"]["selector"] == ["collections"]


@pytest.mark.parametrize("change", ["version", "family", "operations", "kind"])
def test_invalid_selector_rules(tmp_path: Path, change: str) -> None:
    """Reject unsupported semantic profiles and structural kinds before annotation.

    Args:
        tmp_path (Path): Temporary taxonomy directory.
        change (str): Invalid field to introduce.
    """
    taxonomy = load_taxonomy()
    if change == "version":
        taxonomy["selector_rules"]["version"] = 2
    elif change == "family":
        taxonomy["selector_rules"]["color_family"] = "unknown"
    elif change == "operations":
        taxonomy["selector_rules"]["color_filter_operations"] = ["unknown"]
    else:
        taxonomy["families"]["selector"]["collections"]["kind"] = "unknown"
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(taxonomy))
    with pytest.raises(ValueError, match="Selector rules|family kind"):
        load_taxonomy(path)
