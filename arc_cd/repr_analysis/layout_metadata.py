"""Parameter-aware input-layout tags with explicit DSL provenance."""

from __future__ import annotations

from typing import Any


def validate_layout_rules(taxonomy: dict[str, Any]) -> None:
    """Validate the optional layout profile used by newer taxonomy snapshots.

    Args:
        taxonomy (dict[str, Any]): Loaded family definitions and semantic rules.
    """
    if "layout_rules" not in taxonomy:
        return
    rules = taxonomy["layout_rules"]
    if (
        not isinstance(rules, dict)
        or type(rules.get("version")) is not int
        or rules["version"] != 1
    ):
        raise ValueError("Layout rules require version 1.")
    families = rules.get("families")
    if (
        not isinstance(families, dict)
        or set(families)
        != {
            "overlap",
            "containers",
            "markers",
            "fill",
            "pair",
            "noise",
            "corruption",
            "lines",
        }
        or any(
            not isinstance(f, str) or f not in taxonomy["families"]["layout"]
            for f in families.values()
        )
        or len(set(families.values())) != len(families)
    ):
        raise ValueError("Layout rules require distinct known layout families.")
    shapes = rules.get("shape_states")
    if (
        not isinstance(shapes, dict)
        or not shapes
        or any(
            not isinstance(k, str) or v not in ("filled", "unfilled")
            for k, v in shapes.items()
        )
    ):
        raise ValueError("Layout rules require explicit shape-state mappings.")


def _mixed_fill_paths(layout: dict[str, Any], shapes: dict[str, str]) -> list[str]:
    """Find explicit declarations allowing both filled and hollow object states.

    Args:
        layout (dict[str, Any]): Layout expression.
        shapes (dict[str, str]): Exact shape-style to fill-state mapping.

    Returns:
        list[str]: Contributing parameter paths, or an empty list for no evidence.
    """
    op = layout["op"]
    keys: list[str] = []
    counts = False
    if op == "scattered_objects" and "object_groups" not in layout:
        keys = (
            ["object_shape_styles"] if "equivalence_key" in layout else ["shape_styles"]
        )
        # A quota alone does not activate the configured scattered generator;
        # valid mixed-fill scenes require explicit rectangle shape styles.
        counts = "equivalence_key" not in layout and "shape_styles" in layout
    elif (
        op == "gapped_aligned_objects"
        and layout.get("reference_mode") != "central_largest"
    ):
        if "object_count_range" in layout:
            keys, counts = ["shape_styles"], True
        elif "size_relation" in layout:
            keys = ["larger_shape_styles", "smaller_shape_styles"]
    elif op == "corner_stacked_objects":
        keys, counts = ["shape_styles"], True
    elif op == "layered_objects":
        if "object_pair_equivalence_key" in layout:
            keys = ["object_shape_styles"]
        elif _single_object_path(layout) is None and layout.get(
            "overlap", "optional"
        ) in ("required", "always", "true", "optional"):
            # Generic layered scenes use the catalog when role styles are absent;
            # object_shape_styles belongs only to the dedicated pair generator.
            keys = ["lower_shape_styles", "upper_shape_styles"]
    elif op == "central_object_with_distractors":
        keys = ["center_fill_states"]
    elif op == "single_shape_with_fill_state":
        keys = ["fill_states"]
    elif op == "objects_with_fill_states":
        counts = True
    states: set[str] = set()
    paths = []
    for key in keys:
        values = layout.get(key)
        if not isinstance(values, list):
            continue
        observed: set[str] = set()
        for value in values:
            if isinstance(value, str):
                state = value if key.endswith("fill_states") else shapes.get(value)
                if state is not None and state in ("filled", "unfilled"):
                    observed.add(state)
        if observed:
            states.update(observed)
            paths.append(f"program.layout.{key}")
    if counts and "fill_state_count_options" in layout:
        options = layout["fill_state_count_options"]
        if (
            isinstance(options, list)
            and options
            and all(
                isinstance(pair, list)
                and len(pair) == 2
                and all(type(n) is int and n >= 0 for n in pair)
                and sum(pair) > 0
                for pair in options
            )
        ):
            # The generator samples only quotas matching the object count.
            defaults = {
                "scattered_objects": [1, 3],
                "corner_stacked_objects": [3, 4],
                "gapped_aligned_objects": [2, 5],
            }
            if op in defaults:
                bounds = layout.get("object_count_range", defaults[op])
                if (
                    not isinstance(bounds, list)
                    or len(bounds) != 2
                    or not all(type(n) is int and n > 0 for n in bounds)
                    or bounds[0] > bounds[1]
                ):
                    return []
                options = [
                    pair for pair in options if bounds[0] <= sum(pair) <= bounds[1]
                ]
            # Count quotas constrain the allowed styles and take precedence.
            states = {
                state
                for i, state in enumerate(("filled", "unfilled"))
                if any(pair[i] > 0 for pair in options)
            }
            paths = ["program.layout.fill_state_count_options"]
        else:
            return []
    return paths if states == {"filled", "unfilled"} else []


def _single_object_path(layout: dict[str, Any]) -> str | None:
    """Identify explicit declarations limiting a layered scene to one object.

    Args:
        layout (dict[str, Any]): Layered-layout expression.

    Returns:
        str | None: Count evidence path, or no single-object declaration.
    """
    count = layout.get("count")
    if type(count) is int and count == 1:
        return "program.layout.count"
    if (
        count is None
        and type(layout.get("max_count")) is int
        and layout["max_count"] == 1
    ):
        return "program.layout.max_count"
    return None


def _pair_path(layout: dict[str, Any]) -> str | None:
    """Recognize explicit two-object declarations in supported generator branches.

    Args:
        layout (dict[str, Any]): Layout expression.

    Returns:
        str | None: Evidence path for a guaranteed pair, or no additional tag.
    """
    op = layout["op"]
    if op == "layered_objects":
        if layout.get("count") is not None:
            return (
                "program.layout.count"
                if type(layout["count"]) is int and layout["count"] == 2
                else None
            )
        if all(
            type(layout.get(k)) is int and layout[k] == 2
            for k in ("min_count", "max_count")
        ):
            return "program.layout.min_count"
    if op == "scattered_objects" and any(
        k in layout for k in ("object_groups", "equivalence_key")
    ):
        return None
    if op == "gapped_aligned_objects":
        if layout.get("reference_mode") == "central_largest":
            return None
        if "object_count_range" not in layout and "size_relation" in layout:
            return "program.layout.size_relation"
    if op in ("scattered_objects", "gapped_aligned_objects", "corner_stacked_objects"):
        value = layout.get("object_count_range")
        if (
            isinstance(value, list)
            and len(value) == 2
            and all(type(n) is int and n == 2 for n in value)
        ):
            return "program.layout.object_count_range"
    return None


def apply_layout_rules(
    layout: dict[str, Any], operations: list[dict[str, Any]], rules: dict[str, Any]
) -> set[str]:
    """Refine the root layout annotation without altering actions or raw parameters.

    Optional overlap describes an allowed scene property, not a guarantee for
    every support example. Documented defaults are recorded explicitly in evidence.

    Args:
        layout (dict[str, Any]): Original layout expression.
        operations (list[dict[str, Any]]): Local operation annotations to update.
        rules (dict[str, Any]): Validated layout semantic profile.

    Returns:
        set[str]: Final layout-family memberships.
    """
    node = next(n for n in operations if n["path"] == "program.layout")
    families = set(node["families"])
    targets = rules["families"]
    evidence: list[dict[str, Any]] = []

    def change(
        case: str,
        paths: list[str],
        add: str | None = None,
        remove: str | None = None,
        **details: Any,
    ) -> None:
        """Record one contextual family decision.

        Args:
            case (str): Semantic case identifier.
            paths (list[str]): Contributing program paths.
            add (str | None): Profile family key to add.
            remove (str | None): Profile family key to remove.
            **details (Any): Additional resolved values and default provenance.
        """
        added = [targets[add]] if add is not None else []
        removed = [targets[remove]] if remove is not None else []
        families.difference_update(removed)
        families.update(added)
        evidence.append(
            {
                "case": case,
                "paths": paths,
                "added_families": added,
                "removed_families": removed,
                **details,
            }
        )

    op = layout["op"]
    if op == "layered_objects":
        if "object_pair_equivalence_key" in layout:
            modes = layout.get(
                "placement_relation_modes", ["separated", "touching", "overlapping"]
            )
            path = "program.layout.placement_relation_modes"
            defaulted = "placement_relation_modes" not in layout
            valid = (
                isinstance(modes, list)
                and bool(modes)
                and all(
                    isinstance(m, str) and m in {"separated", "touching", "overlapping"}
                    for m in modes
                )
            )
            status = (
                (
                    "required"
                    if set(modes) == {"overlapping"}
                    else "optional" if "overlapping" in modes else "never"
                )
                if valid
                else None
            )
        else:
            status = layout.get("overlap", "optional")
            path = "program.layout.overlap"
            defaulted = "overlap" not in layout
            if isinstance(status, str):
                status = {
                    "always": "required",
                    "true": "required",
                    "none": "never",
                    "false": "never",
                }.get(status, status)
        single = _single_object_path(layout)
        if single is not None:
            change("single_object", [single], remove="overlap", overlap="never")
        elif status == "never":
            change(
                "overlap_forbidden",
                [path],
                remove="overlap",
                overlap="never",
                defaulted=defaulted,
            )
        elif status in ("required", "optional"):
            change(
                "overlap_policy",
                [path],
                add="overlap",
                overlap=status,
                defaulted=defaulted,
            )
        else:
            change("unresolved_overlap", [path], remove="overlap")
            node["layout_issues"] = [
                {"path": path, "reason": "unsupported_overlap_policy"}
            ]
    if (
        op == "shape_with_region_cells"
        and layout.get("shape_kind") == "filled"
        and layout.get("region") == "body"
    ):
        change(
            "filled_body_markers",
            ["program.layout.shape_kind", "program.layout.region"],
            add="markers",
            remove="containers",
        )
    if (
        op == "solid_rectangle_with_scattered_cells"
        and layout.get("rectangle_presence") == "always"
        and layout.get("corruption_region") == "background"
    ):
        change(
            "object_with_background_noise",
            ["program.layout.rectangle_presence", "program.layout.corruption_region"],
            add="noise",
            remove="corruption",
        )
    if (
        op == "gapped_aligned_objects"
        and layout.get("reference_mode") == "central_largest"
    ):
        change("central_reference", ["program.layout.reference_mode"], add="markers")
        if layout.get("object_shape") == "line_segments":
            change("reference_with_lines", ["program.layout.object_shape"], add="lines")
    paths = _mixed_fill_paths(layout, rules["shape_states"])
    if paths:
        change("mixed_fill_states", paths, add="fill")
    pair = _pair_path(layout)
    if pair is not None:
        change("explicit_object_pair", [pair], add="pair")
    node["families"] = sorted(families)
    if evidence:
        node["layout_evidence"] = evidence
    if families:
        node["status"] = "mapped"
    elif evidence:
        node["status"] = "unmapped"
    return families
