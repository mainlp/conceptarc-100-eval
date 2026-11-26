"""Explicit selector-parameter criteria with source-field provenance."""

from __future__ import annotations

from typing import Any


def validate_selector_rules(taxonomy: dict[str, Any]) -> None:
    """Validate optional parameter rules without changing older snapshots.

    Args:
        taxonomy (dict[str, Any]): Loaded family definitions and optional rules.
    """
    if "selector_rules" not in taxonomy:
        return
    rules = taxonomy["selector_rules"]
    if (
        not isinstance(rules, dict)
        or type(rules.get("version")) is not int
        or rules["version"] != 1
        or not isinstance(rules.get("color_family"), str)
        or rules["color_family"] not in taxonomy["families"]["selector"]
        or rules.get("color_filter_operations") != ["region_cells"]
    ):
        raise ValueError(
            "Selector rules require version 1, a known color family, and region_cells."
        )


def selector_parameter_evidence(
    selector: dict[str, Any], path: str, rules: dict[str, Any]
) -> list[dict[str, Any]]:
    """Annotate explicit color criteria using the generator's field precedence.

    Args:
        selector (dict[str, Any]): Selector expression, at any program depth.
        path (str): Exact selector path in the original program.
        rules (dict[str, Any]): Validated optional selector profile.

    Returns:
        list[dict[str, Any]]: Family evidence retaining effective fields and values.
    """
    if selector["op"] not in rules["color_filter_operations"]:
        return []
    field = "color_selector" if "color_selector" in selector else "color"
    evidence = []
    for key in (field, "exclude_color"):
        value = selector.get(key)
        literal = type(value) is int and 0 <= value <= 9
        expression = (
            key != "exclude_color"
            and isinstance(value, dict)
            and isinstance(value.get("op"), str)
            and bool(value["op"])
        )
        if literal or expression:
            evidence.append(
                {
                    "family": rules["color_family"],
                    "path": f"{path}.{key}",
                    "value": value,
                    "description": (
                        "Explicit color exclusion."
                        if key == "exclude_color"
                        else "Effective color filter; color_selector takes precedence over color."
                    ),
                }
            )
    return evidence
