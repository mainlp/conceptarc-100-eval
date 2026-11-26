"""Layout semantics, generator precedence, and frozen-taxonomy compatibility."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from arc_cd.repr_analysis.spec_metadata import annotate_program, load_taxonomy


def annotation(layout: dict[str, Any]) -> dict[str, Any]:
    """Annotate one input layout without changing the source program.

    Args:
        layout (dict[str, Any]): Declared layout and parameters.

    Returns:
        dict[str, Any]: Complete metadata annotation.
    """
    program = {
        "layout": layout,
        "selector": {"op": "all_objects"},
        "action": {"op": "identity"},
    }
    before = copy.deepcopy(program)
    result = annotate_program(program, load_taxonomy())
    assert program == before
    assert result["families"]["selector"] == ["collections"]
    assert result["families"]["action"] == ["identity"]
    return result


@pytest.mark.parametrize(
    "op,expected",
    [
        ("separator_delimited_panels", {"panels"}),
        ("single_color_banded_region", {"color_bands"}),
        ("banded_regions_with_path", {"color_bands", "lines_paths"}),
        ("symmetric_objects_with_marker", {"symmetry", "references_markers"}),
        ("symmetric_pattern_with_missing_cells", {"symmetry", "fill_completion"}),
        ("central_object_with_distractors", {"centering", "references_markers"}),
        ("center_object_in_frame", {"centering", "containers_boundaries"}),
        ("axis_reference_with_objects", {"references_markers", "lines_paths"}),
        ("edge_reference_with_objects", {"references_markers", "lines_paths"}),
    ],
)
def test_split_layout_families(op: str, expected: set[str]) -> None:
    """Split only the intended structures and retain their other layout properties.

    Args:
        op (str): Layout operator.
        expected (set[str]): Expected overlapping layout memberships.
    """
    assert set(annotation({"op": op})["families"]["layout"]) == expected


@pytest.mark.parametrize(
    "policy,present", [("required", True), ("optional", True), ("never", False)]
)
def test_overlap_policy_and_pair_membership(policy: str, present: bool) -> None:
    """Possible overlap is distinguished from required overlap and explicit absence.

    Args:
        policy (str): Declared overlap policy.
        present (bool): Whether overlap is possible under the declaration.
    """
    result = annotation({"op": "layered_objects", "count": 2, "overlap": policy})
    assert ("overlap_occlusion" in result["families"]["layout"]) is present
    assert "paired_objects" in result["families"]["layout"]
    root = result["operations"][0]
    evidence = next(e for e in root["layout_evidence"] if "overlap" in e)
    assert evidence["overlap"] == policy and evidence["defaulted"] is False


@pytest.mark.parametrize(
    "modes,expected",
    [
        (["separated"], "never"),
        (["touching", "overlapping"], "optional"),
        (["overlapping"], "required"),
    ],
)
def test_pair_placement_modes_override_generic_overlap(
    modes: list[str], expected: str
) -> None:
    """The dedicated pair generator uses placement modes rather than generic policy.

    Args:
        modes (list[str]): Allowed pair-placement modes.
        expected (str): Effective overlap status.
    """
    result = annotation(
        {
            "op": "layered_objects",
            "count": 2,
            "object_pair_equivalence_key": "object_dimensions",
            "overlap": "never",
            "placement_relation_modes": modes,
        }
    )
    evidence = next(
        e for e in result["operations"][0]["layout_evidence"] if "overlap" in e
    )
    assert evidence["overlap"] == expected
    assert evidence["paths"] == ["program.layout.placement_relation_modes"]


def test_overlap_defaults_are_explicit_and_invalid_values_flagged() -> None:
    """Defaults retain provenance; unsupported values never imply guaranteed overlap."""
    result = annotation({"op": "layered_objects", "count": 2})
    evidence = next(
        e for e in result["operations"][0]["layout_evidence"] if "overlap" in e
    )
    assert evidence["overlap"] == "optional" and evidence["defaulted"] is True
    result = annotation({"op": "layered_objects", "count": 2, "overlap": "unknown"})
    assert result["families"]["layout"] == ["paired_objects"]
    assert result["operations"][0]["layout_issues"]


def test_filled_body_marks_are_not_containers() -> None:
    """Body-marked filled silhouettes differ from hollow or boundary-marked shapes."""
    filled = annotation(
        {"op": "shape_with_region_cells", "shape_kind": "filled", "region": "body"}
    )
    assert filled["families"]["layout"] == ["references_markers"]
    for parameters in (
        {},
        {"shape_kind": "hollow", "region": "body"},
        {"shape_kind": "filled", "region": "boundary"},
    ):
        other = annotation({"op": "shape_with_region_cells", **parameters})
        assert set(other["families"]["layout"]) == {
            "references_markers",
            "containers_boundaries",
        }


@pytest.mark.parametrize(
    "presence,region,expected",
    [
        ("always", "background", "background_noise"),
        ("always", "rectangle", "corrupted_surfaces"),
        ("never", "background", "corrupted_surfaces"),
        ("mixed", "background", "corrupted_surfaces"),
    ],
)
def test_background_noise_is_contextual(
    presence: str, region: str, expected: str
) -> None:
    """Rename only scenes guaranteed to contain an object with noise outside it.

    Args:
        presence (str): Rectangle presence policy.
        region (str): Region containing corruption cells.
        expected (str): Appropriate layout family.
    """
    result = annotation(
        {
            "op": "solid_rectangle_with_scattered_cells",
            "rectangle_presence": presence,
            "corruption_region": region,
        }
    )
    assert result["families"]["layout"] == [expected]


@pytest.mark.parametrize(
    "layout,expected",
    [
        (
            {
                "op": "scattered_objects",
                "shape_styles": ["filled_rectangle", "hollow_rectangle"],
                "color_mode": "distinct",
                "object_count_range": [4, 4],
                "fill_state_count_options": [[3, 1]],
            },
            True,
        ),
        (
            {
                "op": "scattered_objects",
                "shape_styles": ["filled_rectangle", "hollow_rectangle"],
            },
            True,
        ),
        (
            {
                "op": "scattered_objects",
                "shape_styles": ["filled_rectangle", "hollow_rectangle"],
                "fill_state_count_options": [[3, 0]],
            },
            False,
        ),
        (
            {
                "op": "scattered_objects",
                "shape_styles": ["filled_rectangle", "unknown_shape"],
            },
            False,
        ),
        (
            {
                "op": "gapped_aligned_objects",
                "size_relation": "any",
                "larger_shape_styles": ["hollow_rectangle"],
                "smaller_shape_styles": ["filled_rectangle"],
            },
            True,
        ),
        (
            {
                "op": "central_object_with_distractors",
                "center_fill_states": ["filled", "unfilled"],
            },
            True,
        ),
        (
            {"op": "central_object_with_distractors", "center_fill_states": ["filled"]},
            False,
        ),
        (
            {
                "op": "layered_objects",
                "lower_shape_styles": ["filled_rectangle", "hollow_rectangle"],
                "upper_shape_styles": ["filled_rectangle"],
            },
            True,
        ),
        (
            {
                "op": "layered_objects",
                "object_shape_styles": ["filled_rectangle", "hollow_rectangle"],
                "lower_shape_styles": ["filled_rectangle"],
                "upper_shape_styles": ["filled_rectangle"],
            },
            False,
        ),
        (
            {
                "op": "scattered_objects",
                "equivalence_key": "object_shape",
                "object_shape_styles": ["hollow_rectangle"],
                "shape_styles": ["filled_rectangle", "hollow_rectangle"],
            },
            False,
        ),
    ],
)
def test_explicit_fill_variation(layout: dict[str, Any], expected: bool) -> None:
    """Use active generator fields and quotas; never infer states from fuzzy names.

    Args:
        layout (dict[str, Any]): Layout parameters.
        expected (bool): Whether both fill states are explicitly permitted.
    """
    result = annotation(layout)
    assert ("fill_completion" in result["families"]["layout"]) is expected
    if expected:
        assert any(
            e["case"] == "mixed_fill_states" and e["paths"]
            for e in result["operations"][0]["layout_evidence"]
        )


@pytest.mark.parametrize(
    "layout,expected",
    [
        ({"op": "scattered_objects", "object_count_range": [2, 2]}, True),
        ({"op": "scattered_objects", "object_count_range": [2, 4]}, False),
        (
            {
                "op": "scattered_objects",
                "equivalence_key": "object_shape",
                "object_count_range": [2, 2],
            },
            False,
        ),
        ({"op": "gapped_aligned_objects", "size_relation": "any"}, True),
        (
            {
                "op": "gapped_aligned_objects",
                "size_relation": "any",
                "object_count_range": [3, 4],
            },
            False,
        ),
        (
            {
                "op": "gapped_aligned_objects",
                "reference_mode": "central_largest",
                "size_relation": "any",
                "object_count_range": [2, 2],
            },
            False,
        ),
        ({"op": "layered_objects", "min_count": 2, "max_count": 2}, True),
        (
            {
                "op": "separator_delimited_panels",
                "object_pair_equivalence_key": "object_shape",
            },
            False,
        ),
    ],
)
def test_explicit_pairs_follow_generator_precedence(
    layout: dict[str, Any], expected: bool
) -> None:
    """Do not confuse pairwise comparisons or two targets plus a reference with a pair.

    Args:
        layout (dict[str, Any]): Scene generator and count declarations.
        expected (bool): Whether the generator explicitly guarantees an object pair.
    """
    assert ("paired_objects" in annotation(layout)["families"]["layout"]) is expected


def test_central_reference_and_line_parameters() -> None:
    """Keep axis placement, reference presence, and line structure as separate tags."""
    result = annotation(
        {
            "op": "gapped_aligned_objects",
            "reference_mode": "central_largest",
            "object_shape": "line_segments",
        }
    )
    assert set(result["families"]["layout"]) == {
        "aligned_objects",
        "references_markers",
        "lines_paths",
    }


def test_no_layout_evidence_is_inferred_from_the_action() -> None:
    """A filling action does not make an otherwise unspecified scene a fill-state layout."""
    program = {
        "layout": {"op": "scattered_objects"},
        "selector": {"op": "objects_by_fill_state", "state": "unfilled"},
        "action": {"op": "toggle_object_fill"},
    }
    result = annotate_program(program, load_taxonomy())
    assert result["families"]["action"] == ["filling"]
    assert result["families"]["layout"] == ["scattered_objects"]


def test_legacy_layout_profile_remains_unchanged() -> None:
    """Older snapshots retain original operator-only assignments and family names."""
    taxonomy = load_taxonomy()
    taxonomy.pop("layout_rules")
    taxonomy["families"]["layout"]["panels_bands"] = taxonomy["families"]["layout"].pop(
        "panels"
    )
    program = {
        "layout": {"op": "layered_objects", "overlap": "never", "count": 2},
        "selector": {"op": "all_objects"},
        "action": {"op": "identity"},
    }
    result = annotate_program(program, taxonomy)
    assert result["families"]["layout"] == ["overlap_occlusion"]
    assert all("layout_evidence" not in n for n in result["operations"])
    program["layout"] = {"op": "separator_delimited_panels"}
    assert annotate_program(program, taxonomy)["families"]["layout"] == ["panels_bands"]


@pytest.mark.parametrize("change", ["version", "family", "duplicate", "shape"])
def test_invalid_layout_profiles(tmp_path: Path, change: str) -> None:
    """Reject malformed taxonomy overrides before metadata generation.

    Args:
        tmp_path (Path): Temporary taxonomy path.
        change (str): Invalid field to introduce.
    """
    taxonomy = load_taxonomy()
    rules = taxonomy["layout_rules"]
    if change == "version":
        rules["version"] = 2
    elif change == "family":
        rules["families"]["pair"] = "unknown"
    elif change == "duplicate":
        rules["families"]["pair"] = rules["families"]["fill"]
    else:
        rules["shape_states"]["filled_rectangle"] = "unknown"
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(taxonomy))
    with pytest.raises(ValueError, match="Layout rules"):
        load_taxonomy(path)


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("always", "required"),
        ("true", "required"),
        ("none", "never"),
        ("false", "never"),
    ],
)
def test_explicit_generator_overlap_aliases(policy: str, expected: str) -> None:
    """Resolve only aliases explicitly accepted by the generator.

    Args:
        policy (str): Declared generator alias.
        expected (str): Canonical overlap status.
    """
    result = annotation({"op": "layered_objects", "count": 2, "overlap": policy})
    root = result["operations"][0]
    assert not root.get("layout_issues")
    assert root["layout_evidence"][0]["overlap"] == expected
    assert ("overlap_occlusion" in result["families"]["layout"]) is (
        expected == "required"
    )


@pytest.mark.parametrize(
    "counts", [{"count": 1}, {"max_count": 1}, {"count": None, "max_count": 1}]
)
def test_single_object_cannot_overlap(counts: dict[str, Any]) -> None:
    """A single object never activates overlap or overlap-role shape restrictions.

    Args:
        counts (dict[str, Any]): Explicit single-object declaration.
    """
    result = annotation(
        {
            "op": "layered_objects",
            "overlap": "required",
            "lower_shape_styles": ["filled_rectangle", "hollow_rectangle"],
            "upper_shape_styles": ["filled_rectangle"],
            **counts,
        }
    )
    assert result["families"]["layout"] == []
    root = result["operations"][0]
    assert root["layout_evidence"][0]["case"] == "single_object"
    assert root["layout_evidence"][0]["overlap"] == "never"


@pytest.mark.parametrize("pair", [False, True])
def test_object_shape_styles_belong_to_dedicated_pair_generator(pair: bool) -> None:
    """Generic layered scenes use their shape catalog rather than the pair field.

    Args:
        pair (bool): Whether the dedicated pair generator is active.
    """
    layout: dict[str, Any] = {
        "op": "layered_objects",
        "count": 2,
        "overlap": "required",
        "object_shape_styles": ["filled_rectangle", "hollow_rectangle"],
    }
    if pair:
        layout["object_pair_equivalence_key"] = "object_dimensions"
    assert ("fill_completion" in annotation(layout)["families"]["layout"]) is pair


def test_no_overlap_ignores_overlap_role_shape_styles() -> None:
    """Role-specific styles cannot supply fill evidence in the non-overlap branch."""
    result = annotation(
        {
            "op": "layered_objects",
            "count": 2,
            "overlap": "never",
            "lower_shape_styles": ["filled_rectangle", "hollow_rectangle"],
            "upper_shape_styles": ["filled_rectangle"],
        }
    )
    assert result["families"]["layout"] == ["paired_objects"]


def test_null_explicit_count_uses_declared_pair_range() -> None:
    """The generator treats a null count as an instruction to sample its range."""
    result = annotation(
        {"op": "layered_objects", "count": None, "min_count": 2, "max_count": 2}
    )
    assert "paired_objects" in result["families"]["layout"]


def test_quota_alone_does_not_enable_configured_scattered_branch() -> None:
    """The unconfigured scattered generator does not read the fill quota field."""
    result = annotation(
        {"op": "scattered_objects", "fill_state_count_options": [[3, 1]]}
    )
    assert result["families"]["layout"] == ["scattered_objects"]


@pytest.mark.parametrize(
    "op", ["scattered_objects", "gapped_aligned_objects", "corner_stacked_objects"]
)
@pytest.mark.parametrize("bounds", [[2, 2], [3, 3], [4, 4]])
def test_only_feasible_fill_quotas_contribute(op: str, bounds: list[int]) -> None:
    """A mixed-fill quota outside the object's count range cannot generate a scene.

    Args:
        op (str): Configured object generator.
        bounds (list[int]): Fixed number of generated objects.
    """
    result = annotation(
        {
            "op": op,
            "shape_styles": ["filled_rectangle", "hollow_rectangle"],
            "object_count_range": bounds,
            "fill_state_count_options": [[2, 0], [2, 1], [4, 0]],
        }
    )
    assert ("fill_completion" in result["families"]["layout"]) is (bounds[0] == 3)
