"""Declared object recoloring, missing-content filling, and cell-paint semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OBJECT_UNIFORM_OPS = frozenset(
    {
        "recolor",
        "recolor_group",
        "recolor_shape_layers",
        "recolor_non_background_from_selection",
        "recolor_group_by_orientation",
        "swap_shape_layer_colors",
    }
)
OBJECT_PATTERN_OPS = frozenset(
    {
        "recolor_object_pattern",
        "recolor_object_group_pattern",
        "recolor_shape_layers_pattern",
    }
)
COLOR_EDIT_OPS = (
    OBJECT_UNIFORM_OPS
    | OBJECT_PATTERN_OPS
    | {"recolor_cells", "recolor_cells_pattern", "compose_canvas"}
)
COLOR_EDIT_CASES = frozenset(
    {
        "object_uniform",
        "object_pattern",
        "fill_uniform",
        "fill_pattern",
        "localized",
        "region",
        "erase",
        "isolate",
    }
)


@dataclass
class ColorEditAnnotation:
    """Semantic evidence and unresolved declarations for one action node.

    Attributes:
        evidence (list[dict[str, Any]]): Family decisions with source paths.
        issues (list[dict[str, str]]): Unsupported declarations requiring review.
    """

    evidence: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, str]] = field(default_factory=list)


def validate_color_edit_rules(taxonomy: dict[str, Any]) -> None:
    """Validate the optional semantic profile without changing legacy snapshots.

    Args:
        taxonomy (dict[str, Any]): Loaded taxonomy with optional color_edit_rules.
    """
    if "color_edit_rules" not in taxonomy:
        return
    rules = taxonomy["color_edit_rules"]
    if (
        not isinstance(rules, dict)
        or type(rules.get("version")) is not int
        or rules["version"] != 1
    ):
        raise ValueError("Unsupported color-edit rule version.")
    cases = rules.get("cases")
    if not isinstance(cases, dict) or set(cases) != COLOR_EDIT_CASES:
        raise ValueError("Color-edit rules require the supported semantic cases.")
    for rule in cases.values():
        if (
            not isinstance(rule, dict)
            or not isinstance(rule.get("family"), str)
            or rule["family"] not in taxonomy["families"]["action"]
            or not isinstance(rule.get("description"), str)
            or not rule["description"]
        ):
            raise ValueError(
                "Color-edit cases require known action families and explanations."
            )
    selectors = rules.get("object_selectors")
    if (
        not isinstance(selectors, list)
        or any(not isinstance(s, str) or not s for s in selectors)
        or len(set(selectors)) != len(selectors)
    ):
        raise ValueError("Color-edit object selectors must be unique operator names.")


def _op(value: Any) -> str | None:
    """Read an exact operator name.

    Args:
        value (Any): Candidate DSL expression.

    Returns:
        str | None: Declared operator or None.
    """
    return value.get("op") if isinstance(value, dict) else None


def _literal_color(value: Any) -> int | None:
    """Read a literal ARC color without executing input-dependent expressions.

    Args:
        value (Any): Color literal or expression.

    Returns:
        int | None: Literal color in 0..9, if available.
    """
    if _op(value) == "literal_color":
        value = value.get("color")
    return value if type(value) is int and 0 <= value <= 9 else None


def _selector_target(
    selector: dict[str, Any], rules: dict[str, Any]
) -> tuple[str, bool]:
    """Resolve the declared target, preserving the distinction from render patterns.

    Args:
        selector (dict[str, Any]): Effective selection expression.
        rules (dict[str, Any]): Explicit object-selector registry.

    Returns:
        tuple[str, bool]: Target kind and whether missing cells follow an alternating model.
    """
    op = selector.get("op")
    if op in rules["object_selectors"] or op == "oriented_color_band_cells":
        return "object", False
    if op == "completion_region_cells":
        part, state = selector.get("part", "missing"), selector.get(
            "state", "incomplete"
        )
        model = _op(selector.get("model", {"op": "solid"}))
        if model not in ("solid", "alternating") or state not in (
            "complete",
            "incomplete",
        ):
            return "unknown", False
        if part == "object":
            return "object", False  # Observed interior pixels plus the boundary.
        if part == "missing" or (
            part in ("required", "interior") and state == "incomplete"
        ):
            target = {
                "missing": "missing",
                "required": "completion",
                "interior": "interior",
            }[part]
            return target, model == "alternating"
        if part == "interior" and model == "alternating":
            return (
                "gapped_interior",
                True,
            )  # Complete required phases can still leave gaps.
        if part in ("present", "required", "interior"):
            if (
                state == "complete"
                and model == "solid"
                and (
                    selector.get("include_boundary") is True
                    or selector.get("geometry") == "parallel_strip"
                )
            ):
                return "object", False
            return "cells", False
    if op == "inferred_footprint_cells" and selector.get("part") == "missing":
        return "missing", False
    if op == "object_extreme_cells":
        return "cells", False
    if op == "region_cells":
        region = selector.get("region")
        if region == "non_background":
            return "cells", False
        if region == "background":
            return "background", False
        if region == "all":
            return "region", False
    if op == "cells_relative_to_reference":
        candidate = selector.get("candidate_selector")
        if isinstance(candidate, dict) and _selector_target(candidate, rules)[0] in (
            "object",
            "cells",
        ):
            return "cells", False
        through = selector.get("through", "non_background")
        if through in ("non_background", "occupied"):
            return "cells", False
        if through == "any":
            return "region", False
    if op in ("union", "intersection"):
        children = selector.get("selectors")
        if (
            isinstance(children, list)
            and children
            and all(
                isinstance(child, dict) and _selector_target(child, rules)[0] == "cells"
                for child in children
            )
        ):
            return "cells", False
    return "unknown", False


def _in_place(placement: dict[str, Any]) -> bool:
    """Check explicit source coordinates, accounting for DSL location precedence.

    Args:
        placement (dict[str, Any]): Placement declaration.

    Returns:
        bool: Whether the source patch stays at its original coordinates.
    """
    return (
        all(
            placement.get(key) is None
            for key in ("target_position", "location", "anchor")
        )
        and placement.get("row") == {"op": "selection_min_row"}
        and placement.get("col") == {"op": "selection_min_col"}
    )


def annotate_color_edit(
    action: dict[str, Any], selector: dict[str, Any], path: str, rules: dict[str, Any]
) -> ColorEditAnnotation:
    """Classify declared edits with effective-selector and rendering provenance.

    Source geometry is judged relative to the input, including when selected
    content is rendered on a blank canvas. Conditional branches remain declared
    alternatives. Missing colors/unsupported selections never imply recoloring.

    Args:
        action (dict[str, Any]): Supported color primitive or compose_canvas node.
        selector (dict[str, Any]): Program-level selector inherited by placements.
        path (str): Absolute path of this action in the saved program.
        rules (dict[str, Any]): Validated semantic profile.

    Returns:
        ColorEditAnnotation: Decisions and unresolved cases with source paths.
    """
    result = ColorEditAnnotation()

    def emit(
        case: str,
        node_path: str,
        target: str,
        selector_path: str,
        pattern: bool = False,
        canvas: str = "input_canvas",
    ) -> None:
        """Append one decision and its exact program context.

        Args:
            case (str): Semantic case from the taxonomy.
            node_path (str): Source edit path.
            target (str): Resolved selection kind.
            selector_path (str): Effective selector path.
            pattern (bool): Whether the edit follows an alternating pattern.
            canvas (str): Destination canvas type.
        """
        rule = rules["cases"][case]
        result.evidence.append(
            {
                "case": case,
                "family": rule["family"],
                "path": node_path,
                "selector_path": selector_path,
                "target": target,
                "patterned": pattern,
                "canvas": canvas,
                "description": rule["description"],
            }
        )

    def paint(
        target: str,
        source_pattern: bool,
        render: dict[str, Any] | None,
        node_path: str,
        selector_path: str,
        canvas: str = "input_canvas",
        background: int = 0,
        include_background: bool = True,
    ) -> None:
        """Resolve palette and target jointly without assuming black means recoloring.

        Args:
            target (str): Resolved target kind.
            source_pattern (bool): Alternating completion model for missing cells.
            render (dict[str, Any] | None): Explicit renderer or preserved source colors.
            node_path (str): Source path.
            selector_path (str): Effective selector path.
            canvas (str): Destination canvas kind.
            background (int): Background used by the canvas.
            include_background (bool): Whether background-colored cells are written.
        """
        mode = render.get("op", "preserve") if isinstance(render, dict) else "preserve"
        pattern = mode == "alternating"
        if mode == "preserve":
            if target not in ("missing", "completion", "interior"):
                return
        elif mode == "solid":
            color = render.get("color") if render is not None else None
            literal = _literal_color(color)
            if literal == background:
                if (
                    canvas == "input_canvas"
                    and include_background
                    and target
                    in (
                        "object",
                        "cells",
                        "interior",
                        "completion",
                        "region",
                        "gapped_interior",
                    )
                ):
                    emit("erase", node_path, target, selector_path, canvas=canvas)
                else:
                    result.issues.append(
                        {
                            "path": node_path,
                            "reason": "background_render_does_not_erase_input",
                        }
                    )
                return
            if literal is None and _op(color) != "selector_color":
                result.issues.append(
                    {"path": node_path, "reason": "unsupported_or_missing_color"}
                )
                return
        elif mode == "alternating":
            colors = render.get("colors") if render is not None else None
            if (
                render is None
                or render.get("mode", "checkerboard")
                not in ("row", "col", "checkerboard")
                or not isinstance(colors, list)
                or len(colors) != 2
                or colors[0] == colors[1]
                or any(
                    _literal_color(c) in (0, background)
                    or (_literal_color(c) is None and _op(c) != "selector_color")
                    for c in colors
                )
            ):
                result.issues.append(
                    {"path": node_path, "reason": "unsupported_or_missing_palette"}
                )
                return
        else:
            result.issues.append({"path": node_path, "reason": "unsupported_renderer"})
            return
        if target == "object":
            case = "object_pattern" if pattern else "object_uniform"
        elif target in ("missing", "completion", "interior", "gapped_interior"):
            pattern = pattern or (
                source_pattern
                and (mode == "preserve" or target in ("missing", "completion"))
            )
            case = "fill_pattern" if pattern else "fill_uniform"
        elif target == "cells":
            case = "localized"
        elif target in ("region", "background"):
            case = "region"
        else:
            result.issues.append(
                {"path": node_path, "reason": "unsupported_target_selector"}
            )
            return
        emit(case, node_path, target, selector_path, pattern, canvas)

    op = action.get("op")
    selector_path = "program.selector"
    if op in OBJECT_UNIFORM_OPS:
        if op in ("swap_shape_layer_colors", "recolor_non_background_from_selection"):
            emit("object_uniform", path, "object", selector_path)
        else:
            keys = (
                ("horizontal_color", "vertical_color")
                if op == "recolor_group_by_orientation"
                else ("color",)
            )
            for key in keys:
                paint(
                    "object",
                    False,
                    {"op": "solid", "color": action.get(key)},
                    f"{path}.{key}",
                    selector_path,
                )
    elif op in OBJECT_PATTERN_OPS:
        pattern = action.get("pattern")
        color = action.get("color")
        if (
            not isinstance(pattern, dict)
            or _op(pattern) != "alternating"
            or pattern.get("mode", "checkerboard")
            not in (
                "horizontal",
                "vertical",
                "checkerboard",
            )
        ):
            result.issues.append(
                {"path": path, "reason": "unsupported_or_missing_pattern"}
            )
        elif _literal_color(color) == 0:
            emit("erase", f"{path}.color", "object", selector_path, pattern=True)
        elif _literal_color(color) is not None or _op(color) == "selector_color":
            emit("object_pattern", path, "object", selector_path, pattern=True)
        else:
            result.issues.append(
                {"path": path, "reason": "unsupported_or_missing_color"}
            )
    elif op in ("recolor_cells", "recolor_cells_pattern"):
        target, source_pattern = _selector_target(selector, rules)
        if (
            op == "recolor_cells_pattern"
            and _op(action.get("pattern")) != "alternating"
        ):
            result.issues.append(
                {"path": path, "reason": "unsupported_or_missing_pattern"}
            )
            return result
        cell_render = (
            {"op": "solid", "color": action.get("color")}
            if op == "recolor_cells"
            else {
                "op": "alternating",
                "colors": action.get("colors"),
                "mode": action["pattern"].get("mode", "checkerboard"),
            }
        )
        paint(target, source_pattern, cell_render, path, selector_path)
    elif op == "compose_canvas":
        canvas = _op(action.get("canvas"))
        placements = action.get("placements")
        if canvas not in ("input_canvas", "blank_canvas") or not isinstance(
            placements, list
        ):
            result.issues.append(
                {"path": path, "reason": "unsupported_canvas_or_placements"}
            )
            return result
        background = _literal_color(action["canvas"].get("background_color", 0))
        if background is None:
            result.issues.append(
                {"path": path, "reason": "unsupported_canvas_background"}
            )
            return result
        for i, placement in enumerate(placements):
            node_path = f"{path}.placements[{i}]"
            if (
                not isinstance(placement, dict)
                or placement.get("op") != "place_selection"
            ):
                result.issues.append(
                    {"path": node_path, "reason": "unsupported_placement"}
                )
                continue
            selected = placement.get("selector", selector)
            selected_path = (
                f"{node_path}.selector" if "selector" in placement else selector_path
            )
            if not isinstance(selected, dict):
                result.issues.append(
                    {"path": node_path, "reason": "unsupported_target_selector"}
                )
                continue
            target, source_pattern = _selector_target(selected, rules)
            render = placement.get("render")
            if render is not None and not isinstance(render, dict):
                result.issues.append(
                    {"path": node_path, "reason": "unsupported_renderer"}
                )
                continue
            preserves = render is None or render.get("op", "preserve") == "preserve"
            if not _in_place(placement):
                if not preserves or target in (
                    "missing",
                    "completion",
                    "interior",
                    "gapped_interior",
                ):
                    result.issues.append(
                        {
                            "path": node_path,
                            "reason": "color_edit_with_changed_or_unspecified_position",
                        }
                    )
                continue
            before = len(result.evidence)
            paint(
                target,
                source_pattern,
                render,
                node_path,
                selected_path,
                canvas,
                background,
                placement.get("include_background", True),
            )
            if canvas == "blank_canvas" and len(result.evidence) > before:
                emit("isolate", node_path, target, selected_path, canvas=canvas)
    return result
