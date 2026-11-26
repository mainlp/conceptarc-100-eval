"""Contextual filling, hollowing, and bidirectional task-family assignments."""

from __future__ import annotations

from typing import Any

TOGGLE_OPS = frozenset(
    {"toggle_object_fill", "toggle_object_group_fill", "toggle_shape_layer_fill"}
)


def validate_fill_state_rules(taxonomy: dict[str, Any]) -> None:
    """Validate the optional profile without changing frozen older taxonomies.

    Args:
        taxonomy (dict[str, Any]): Loaded taxonomy with optional fill-state rules.
    """
    rules = taxonomy.get("fill_state_rules")
    if rules is None:
        return
    if (
        not isinstance(rules, dict)
        or type(rules.get("version")) is not int
        or rules["version"] != 1
        or not taxonomy.get("color_edit_rules")
    ):
        raise ValueError("Fill-state rules require version 1 and color-edit rules.")
    families = rules.get("families")
    if (
        not isinstance(families, dict)
        or set(families) != {"fill", "pattern_fill", "hollow", "switch", "isolate"}
        or any(
            not isinstance(f, str) or f not in taxonomy["families"]["action"]
            for f in families.values()
        )
        or len(set(families.values())) != len(families)
    ):
        raise ValueError("Fill-state rules require distinct known family targets.")
    shapes = rules.get("shape_states")
    layouts = rules.get("layout_shape_parameters")
    if (
        not isinstance(shapes, dict)
        or not shapes
        or any(
            not isinstance(k, str) or v not in ("filled", "unfilled")
            for k, v in shapes.items()
        )
        or not isinstance(layouts, dict)
        or any(
            not isinstance(k, str)
            or not isinstance(v, list)
            or not v
            or any(not isinstance(p, str) or not p for p in v)
            for k, v in layouts.items()
        )
    ):
        raise ValueError("Fill-state rules require explicit shape and layout mappings.")


def _selector_states(selector: dict[str, Any]) -> set[str] | None:
    """Read constraints on selected objects, excluding unrelated reference objects.

    Args:
        selector (dict[str, Any]): Effective target selector.

    Returns:
        set[str] | None: Declared input fill states, or unknown.
    """
    op = selector.get("op", "")
    if op == "hollow_objects":
        return {"unfilled"}
    if op == "objects_by_fill_state" and selector.get("state") in (
        "filled",
        "unfilled",
    ):
        return {selector["state"]}
    child_key = {
        "all": "of",
        "objects_relative_to_reference": "candidate_selector",
        "objects_by_equivalence_class": "source",
    }.get(op)
    if child_key is not None and isinstance(selector.get(child_key), dict):
        return _selector_states(selector[child_key])
    if op == "union":
        children = selector.get("selectors")
        if (
            isinstance(children, list)
            and children
            and all(isinstance(s, dict) for s in children)
        ):
            states = [_selector_states(s) for s in children]
            if all(s is not None for s in states):
                return set().union(*(s for s in states if s is not None))
    return None


def _layout_states(
    layout: dict[str, Any], selector: dict[str, Any], rules: dict[str, Any]
) -> set[str] | None:
    """Resolve exact layout constraints for the selected object or layer role.

    Args:
        layout (dict[str, Any]): Declared scene generator.
        selector (dict[str, Any]): Target selector, including depth/center roles.
        rules (dict[str, Any]): Explicit shape-state and layout-parameter registries.

    Returns:
        set[str] | None: Possible input states, or unknown for unsupported context.
    """
    op = layout.get("op")
    if op == "hollow_objects":
        return {"unfilled"}
    if op == "central_object_with_distractors":
        states = layout.get("center_fill_states")
        if (
            selector.get("op") == "central_object"
            and isinstance(states, list)
            and states
            and all(s in ("filled", "unfilled") for s in states)
        ):
            return set(states)
        return None
    keys = rules["layout_shape_parameters"].get(op, [])
    if op == "layered_objects":
        if "object_pair_equivalence_key" in layout:
            # The dedicated pair generator ignores generic depth-role styles.
            keys = ["object_shape_styles"]
        else:
            count = layout.get("count")
            if count is None and layout.get("min_count") == layout.get("max_count"):
                count = layout.get("min_count")
            if layout.get("overlap", "optional") not in ("required", "always", "true"):
                return None
            if type(count) is not int or count < 2:
                return None
            if count != 2 and layout.get("overlap_topology", "single_pair") != "chain":
                return None
            depth = (
                selector.get("depth")
                if selector.get("op") == "shape_instance_by_depth"
                else None
            )
            if (
                selector.get("op") == "shape_instances_by_occlusion"
                and selector.get("relation") == "fully_visible"
                and count == 2
            ):
                depth = "top"
            key = {"top": "upper_shape_styles", "bottom": "lower_shape_styles"}.get(
                depth if isinstance(depth, str) else ""
            )
            if key is None or key not in layout:
                return None
            # Unrestricted roles use the catalog, not object_shape_styles.
            keys = [key]
    if op in (
        "concave_region_with_inside_outside_objects",
        "hollow_container_with_inside_outside_objects",
        "irregular_outline_with_inside_outside_objects",
        "parallel_lines_with_inside_outside_objects",
    ) and selector.get("op") not in (
        "inside_container_objects",
        "outside_container_objects",
    ):
        return None
    shapes: list[str] = []
    for key in keys:
        value = layout.get(key)
        if value is None:
            continue
        entries = value if isinstance(value, list) else [value]
        if not entries or any(
            not isinstance(s, str) or s not in rules["shape_states"] for s in entries
        ):
            return None
        shapes.extend(entries)
    return {rules["shape_states"][s] for s in shapes} if shapes else None


def _node_at_path(program: dict[str, Any], path: str) -> dict[str, Any]:
    """Resolve an internally generated program path for provenance checks.

    Args:
        program (dict[str, Any]): Original program.
        path (str): Absolute dotted path, optionally containing array indices.

    Returns:
        dict[str, Any]: Referenced expression node.
    """
    value: Any = program
    for key in (
        path.removeprefix("program.").replace("[", ".").replace("]", "").split(".")
    ):
        value = value[int(key)] if isinstance(value, list) else value[key]
    if not isinstance(value, dict):
        raise ValueError(
            f"Fill-state evidence does not reference an expression: {path}"
        )
    return value


def apply_fill_state_rules(
    program: dict[str, Any], operations: list[dict[str, Any]], rules: dict[str, Any]
) -> set[str]:
    """Refine local annotations and collapse bidirectional parent tasks into one family.

    Branches contribute their declared directions; no claim is made that every
    support example executes both. Original color evidence remains available,
    alongside explicit fill-state evidence explaining any superseded family.

    Args:
        program (dict[str, Any]): Original layout, selector, and action.
        operations (list[dict[str, Any]]): Local annotation records to refine.
        rules (dict[str, Any]): Validated, versioned fill-state profile.

    Returns:
        set[str]: Final parent-level action memberships.
    """
    families = rules["families"]
    for node in operations:
        if node["role"] != "action":
            continue
        matches = set(node["families"])
        evidence = []
        if node["op"] in TOGGLE_OPS:
            matches.discard(families["switch"])
            selected = _selector_states(program["selector"])
            layout = _layout_states(program["layout"], program["selector"], rules)
            states = (
                selected
                if layout is None
                else layout if selected is None else selected & layout
            )
            if not states:
                node["fill_state_issues"] = [
                    {"path": node["path"], "reason": "unresolved_toggle_input_states"}
                ]
            else:
                direction = (
                    "switch"
                    if len(states) == 2
                    else "fill" if "unfilled" in states else "hollow"
                )
                matches.add(families[direction])
                evidence.append(
                    {
                        "case": "toggle_input_states",
                        "family": families[direction],
                        "path": node["path"],
                        "selector_path": "program.selector",
                        "layout_path": "program.layout",
                        "input_states": sorted(states),
                    }
                )
        color_evidence = node.get("color_edit_evidence", [])
        if color_evidence:
            matches.difference_update(e["family"] for e in color_evidence)
        for original in color_evidence:
            family = original["family"]
            case = original["case"]
            if (
                case in ("fill_uniform", "fill_pattern")
                and original["canvas"] == "blank_canvas"
            ):
                family = families["isolate"]
                case = "isolated_completion_patch"
            elif case == "erase" and original["canvas"] == "input_canvas":
                selector = _node_at_path(program, original["selector_path"])
                if (
                    selector.get("op") == "completion_region_cells"
                    and selector.get("geometry", "enclosed_region")
                    in ("enclosed_region", "open_row_span")
                    and selector.get("part", "missing")
                    in ("present", "required", "interior")
                    and not selector.get("include_boundary", False)
                ):
                    family = families["hollow"]
                    case = "clear_interior_keep_boundary"
            matches.add(family)
            if family != original["family"]:
                evidence.append(
                    {
                        "case": case,
                        "family": family,
                        "previous_family": original["family"],
                        "path": original["path"],
                        "selector_path": original["selector_path"],
                        "patterned": original["patterned"],
                    }
                )
        node["families"] = sorted(matches)
        if evidence:
            node["fill_state_evidence"] = evidence
        if node["op"] in TOGGLE_OPS:
            node["status"] = "mapped" if matches else "unmapped"

    found = {f for n in operations if n["role"] == "action" for f in n["families"]}
    fills = {families["fill"], families["pattern_fill"]}
    if families["switch"] in found or (found & fills and families["hollow"] in found):
        components = fills | {families["hollow"]}
        for node in operations:
            if node["role"] != "action":
                continue
            previous = set(node["families"]) & components
            if previous:
                node["families"] = sorted(
                    (set(node["families"]) - components) | {families["switch"]}
                )
                node.setdefault("fill_state_evidence", []).append(
                    {
                        "case": "bidirectional_program",
                        "family": families["switch"],
                        "path": node["path"],
                        "component_families": sorted(previous),
                    }
                )
        found = (found - components) | {families["switch"]}
    return found
