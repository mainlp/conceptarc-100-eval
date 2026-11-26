"""Traceable task-spec metadata and conservative, explicit DSL family mappings."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc_cd.repr_analysis.color_edit_metadata import (
    COLOR_EDIT_OPS,
    annotate_color_edit,
    validate_color_edit_rules,
)
from arc_cd.repr_analysis.fill_state_metadata import (
    apply_fill_state_rules,
    validate_fill_state_rules,
)
from arc_cd.repr_analysis.layout_metadata import (
    apply_layout_rules,
    validate_layout_rules,
)
from arc_cd.repr_analysis.selector_metadata import (
    selector_parameter_evidence,
    validate_selector_rules,
)

DEFAULT_TAXONOMY = Path(__file__).with_name("spec_taxonomy.json")
ROLES = ("layout", "selector", "action")


@dataclass
class MetadataBundle:
    """Aligned parent metadata and its descriptive inventory.

    Attributes:
        records (list[dict[str, Any]]): One annotated spec per saved parent task.
        fields (list[dict[str, Any]]): Searchable categorical fields and value counts.
        inventory (dict[str, Any]): Coverage, per-concept counts, and unmapped operators.
        fingerprints (dict[str, dict[str, Any]]): Exact input-spec hashes and sizes.
    """

    records: list[dict[str, Any]]
    fields: list[dict[str, Any]]
    inventory: dict[str, Any]
    fingerprints: dict[str, dict[str, Any]]


def load_taxonomy(path: Path = DEFAULT_TAXONOMY) -> dict[str, Any]:
    """Read and validate an explicit JSON taxonomy.

    Args:
        path (Path): Taxonomy file; defaults to the bundled vocabulary.

    Returns:
        dict[str, Any]: Validated family rules and parameter vocabulary.
    """
    taxonomy = json.loads(path.read_text())
    if (
        not isinstance(taxonomy, dict)
        or taxonomy.get("schema_version") != 1
        or not isinstance(taxonomy.get("version"), str)
        or not taxonomy["version"]
    ):
        raise ValueError("Taxonomy requires schema_version 1 and a nonempty version.")
    if not isinstance(taxonomy.get("families"), dict) or set(
        taxonomy["families"]
    ) != set(ROLES):
        raise ValueError("Taxonomy must define layout, selector, and action families.")
    for role, families in taxonomy["families"].items():
        if not isinstance(families, dict):
            raise ValueError(f"Invalid family mapping for {role}.")
        for name, rule in families.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not isinstance(rule, dict):
                raise ValueError(f"Invalid family identifier: {role}/{name}")
            if any(
                not isinstance(rule.get(key), str) or not rule[key]
                for key in ("label", "description")
            ):
                raise ValueError(f"Family {role}/{name} needs a label and explanation.")
            if rule.get("kind", "criterion") not in ("criterion", "structural"):
                raise ValueError(f"Invalid family kind: {role}/{name}")
            _validate_strings(rule.get("operations"), f"{role}/{name} operations")
    for key in ("helper_operations", "parameter_keys", "expression_parameters"):
        _validate_strings(taxonomy.get(key), key)
    if taxonomy.get("parameter_mode", "listed") not in ("listed", "all_literals"):
        raise ValueError("Unsupported taxonomy parameter_mode.")
    if not isinstance(taxonomy.get("index_all_operations", False), bool):
        raise ValueError("index_all_operations must be a boolean.")
    _validate_strings(
        taxonomy.get("ordered_parameter_keys", []), "ordered_parameter_keys"
    )
    _validate_parameter_families(taxonomy)
    _validate_composition_rules(taxonomy)
    _validate_concept_action_families(taxonomy)
    _validate_secondary_action_families(taxonomy)
    validate_color_edit_rules(taxonomy)
    validate_fill_state_rules(taxonomy)
    validate_layout_rules(taxonomy)
    validate_selector_rules(taxonomy)
    return taxonomy


def _validate_concept_action_families(taxonomy: dict[str, Any]) -> None:
    """Validate explicit concept-derived task action labels.

    Args:
        taxonomy (dict[str, Any]): Taxonomy with an optional concept-to-family map.
    """
    mapping = taxonomy.get("concept_action_families", {})
    if not isinstance(mapping, dict):
        raise ValueError("concept_action_families must be an object.")
    families = taxonomy["families"]["action"]
    for concept, family in mapping.items():
        if (
            not isinstance(concept, str)
            or not concept
            or not isinstance(family, str)
            or family not in families
            or families[family].get("kind", "criterion") == "structural"
        ):
            raise ValueError(
                "Concept action labels require nonempty concepts and known "
                "non-structural action families."
            )


def _validate_secondary_action_families(taxonomy: dict[str, Any]) -> None:
    """Validate operation details that never contribute task family memberships.

    Args:
        taxonomy (dict[str, Any]): Taxonomy with optional secondary action details.
    """
    mapping = taxonomy.get("secondary_action_families", {})
    if not isinstance(mapping, dict):
        raise ValueError("secondary_action_families must be an object.")
    families = taxonomy["families"]["action"]
    operations = {op for rule in families.values() for op in rule["operations"]}
    for op, details in mapping.items():
        if op not in operations:
            raise ValueError(
                "Secondary action details require a known action operator."
            )
        _validate_strings(details, "Secondary action families")
        if not details or not set(details) <= set(families):
            raise ValueError("Secondary action details require known action families.")


def _validate_parameter_families(taxonomy: dict[str, Any]) -> None:
    """Validate explicit operator-and-parameter family rules.

    Args:
        taxonomy (dict[str, Any]): Taxonomy containing optional parameter rules.
    """
    rules = taxonomy.get("parameter_families", [])
    if not isinstance(rules, list):
        raise ValueError("parameter_families must be a list.")
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("Each parameter family rule must be an object.")
        role, family = rule.get("role"), rule.get("family")
        if (
            not isinstance(role, str)
            or role not in ROLES
            or not isinstance(family, str)
            or family not in taxonomy["families"][role]
            or not isinstance(rule.get("description"), str)
            or not rule["description"]
            or not isinstance(rule.get("parameter"), str)
            or not re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*", rule["parameter"])
        ):
            raise ValueError(
                "Invalid parameter family role, family, path, or explanation."
            )
        _validate_strings(rule.get("operations"), "parameter family operations")
        _validate_strings(rule.get("values"), "parameter family values")
        known = {
            op
            for entry in taxonomy["families"][role].values()
            for op in entry["operations"]
        }
        if (
            not rule["operations"]
            or not rule["values"]
            or not set(rule["operations"]).issubset(known)
        ):
            raise ValueError(
                "Parameter family rules require known operators and values."
            )


def _validate_composition_rules(taxonomy: dict[str, Any]) -> None:
    """Validate the opt-in, DSL-specific composition cases and their family targets.

    Args:
        taxonomy (dict[str, Any]): Taxonomy with optional composition rules.
    """
    rules = taxonomy.get("composition_rules", {})
    known = {
        "solid_render",
        "alternating_render",
        "isolate_in_place",
        "extract_to_corner",
        "overlay_in_place",
        "empty_canvas",
        "unchanged_canvas",
        "copy_to_input",
        "extract_to_position",
    }
    if not isinstance(rules, dict) or not set(rules) <= known:
        raise ValueError("Invalid composition rules: unknown case or mapping.")
    for rule in rules.values():
        if (
            not isinstance(rule, dict)
            or not isinstance(rule.get("description"), str)
            or not rule["description"]
        ):
            raise ValueError("Composition rules require an explanation.")
        _validate_strings(rule.get("families"), "Composition rule families")
        if not rule["families"] or not set(rule["families"]) <= set(
            taxonomy["families"]["action"]
        ):
            raise ValueError("Composition rules require known action families.")


def _composition_cases(action: dict[str, Any]) -> list[tuple[str, str]]:
    """Identify supported canvas semantics without executing a program.

    Original-coordinate placements, explicit relocation, and repeated placement
    are distinguished. Selections can contain inferred colors or geometry;
    placement does not prove literal duplication of observed input pixels.
    Missing coordinates never imply original placement: the DSL defaults to 0.

    Args:
        action (dict[str, Any]): compose_canvas expression.

    Returns:
        list[tuple[str, str]]: Semantic cases and relative evidence paths.
    """
    canvas = action.get("canvas")
    placements = action.get("placements")
    relocated = _relocated_composition_case(action)
    if relocated is not None:
        return [(relocated, "placements")]
    if (
        not isinstance(canvas, dict)
        or canvas.get("op") not in ("blank_canvas", "input_canvas")
        or not isinstance(placements, list)
        or any(
            not isinstance(p, dict) or p.get("op") != "place_selection"
            for p in placements
        )
    ):
        return []
    blank = canvas["op"] == "blank_canvas"
    if not placements:
        if action.get("allow_empty") is True:
            return [("empty_canvas" if blank else "unchanged_canvas", "placements")]
        return []
    cases: list[tuple[str, str]] = []
    preserves = True
    for i, placement in enumerate(placements):
        render = placement.get("render")
        if render is None or (
            isinstance(render, dict) and render.get("op", "preserve") == "preserve"
        ):
            continue
        preserves = False
        if not isinstance(render, dict):
            continue
        if render.get("op") == "solid" and "color" in render:
            cases.append(("solid_render", f"placements[{i}].render"))
        if (
            render.get("op") == "alternating"
            and isinstance(render.get("colors"), list)
            and len(render["colors"]) == 2
        ):
            cases.append(("alternating_render", f"placements[{i}].render"))
    if preserves:
        in_place = all(
            all(p.get(key) is None for key in ("target_position", "location", "anchor"))
            and p.get("row") == {"op": "selection_min_row"}
            and p.get("col") == {"op": "selection_min_col"}
            for p in placements
        )
        to_corner = all(
            p.get("target_position") is None
            and isinstance(p.get("location"), dict)
            and p["location"].get("op") == "selection_grid_corner"
            and p["location"].get("corner")
            in ("upper_left", "upper_right", "lower_left", "lower_right")
            for p in placements
        )
        if in_place:
            cases.append(
                ("isolate_in_place" if blank else "overlay_in_place", "placements")
            )
        elif blank and to_corner:
            cases.append(("extract_to_corner", "placements"))
    return cases


def _is_canvas_coordinate(value: Any) -> bool:
    """Recognize explicit integer coordinates and supported arithmetic expressions.

    Args:
        value (Any): Literal or DSL coordinate expression.

    Returns:
        bool: Whether the expression uses the reviewed coordinate grammar.
    """
    if type(value) is int:
        return True
    if not isinstance(value, dict):
        return False
    op = value.get("op")
    if op in (
        "input_height",
        "input_width",
        "selection_height",
        "selection_width",
        "selection_min_row",
        "selection_min_col",
    ):
        return True
    if op == "mul":
        return _is_canvas_coordinate(value.get("factor")) and _is_canvas_coordinate(
            value.get("value")
        )
    if op == "add":
        terms = value.get("values")
        return (
            isinstance(terms, list)
            and bool(terms)
            and all(_is_canvas_coordinate(term) for term in terms)
        )
    return False


def _placement_destination(placement: dict[str, Any]) -> str | None:
    """Recognize explicit destinations in the DSL's target/location/anchor order.

    Args:
        placement (dict[str, Any]): Source-color-preserving place_selection node.

    Returns:
        str | None: Destination field, or None for unsupported or original placement.
    """
    supported = {
        "target_position": (
            "grid_boundary",
            "enclosing_boundary",
            "align_to_grid_center",
            "align_to_reference",
            "opposite_grid_boundary_on_reference_axis",
        ),
        "location": (
            "selection_offset",
            "selection_bilateral_offset",
            "selection_cardinal_offset",
            "selection_grid_corner",
            "selection_grid_corner_pair",
            "selection_grid_corner_triple",
            "selection_grid_corner_quad",
            "matching_subset",
            "matching_markers",
        ),
        "anchor": ("object_center", "grid_center", "midpoint_between_objects"),
    }
    for field, operations in supported.items():
        value = placement.get(field)
        if value is not None:
            if not isinstance(value, dict) or value.get("op") not in operations:
                return None
            if field == "anchor" and placement.get("align", "center") != "center":
                return None
            return field
    row, col = placement.get("row"), placement.get("col")
    if row == {"op": "selection_min_row"} and col == {"op": "selection_min_col"}:
        return None
    return (
        "coordinates"
        if _is_canvas_coordinate(row) and _is_canvas_coordinate(col)
        else None
    )


def _relocated_composition_case(action: dict[str, Any]) -> str | None:
    """Classify reviewed preserved placements without guessing missing settings.

    Args:
        action (dict[str, Any]): Canvas composition expression.

    Returns:
        str | None: Copy-on-input or explicit extraction case, if fully supported.
    """
    canvas, placements = action.get("canvas"), action.get("placements")
    if (
        not isinstance(canvas, dict)
        or canvas.get("op") not in ("input_canvas", "blank_canvas")
        or not isinstance(placements, list)
        or not placements
    ):
        return None
    on_input = canvas["op"] == "input_canvas"
    for placement in placements:
        if not isinstance(placement, dict):
            return None
        render = placement.get("render")
        if render is not None and (
            not isinstance(render, dict) or render.get("op", "preserve") != "preserve"
        ):
            return None
        op = placement.get("op")
        if op == "place_selection":
            destination = _placement_destination(placement)
            if destination is None or (
                not on_input and destination != "target_position"
            ):
                return None
        elif on_input and op == "repeat_selection":
            count, axis = placement.get("copy_count"), placement.get("axis")
            if (
                type(count) is not int
                or count < 2
                or not (
                    axis in ("row", "col")
                    or isinstance(axis, dict)
                    and axis.get("op") in ("aspect_axis", "object_group_axis")
                )
                or not all(
                    _is_canvas_coordinate(placement.get(key)) for key in ("row", "col")
                )
            ):
                return None
        elif not (
            on_input
            and op == "place_each_object_at_matching_centers"
            and render is None
        ):
            return None
    return "copy_to_input" if on_input else "extract_to_position"


def _validate_strings(value: Any, name: str) -> None:
    """Require a unique list of nonempty strings.

    Args:
        value (Any): Candidate list.
        name (str): Field description for errors.
    """
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a list of unique nonempty strings.")


def annotate_program(
    program: dict[str, Any],
    taxonomy: dict[str, Any],
    *,
    concept_name: str | None = None,
) -> dict[str, Any]:
    """Collect exact operators, nested paths, family evidence, and parameter tags.

    Roles follow explicit operator registries: an action's embedded selector is
    a selector, while colors, measurements, and rendering helpers remain
    expressions. Unknown operators remain recorded without semantic inference.
    Optional parameter rules match exact declared values; absent parameters do
    not imply DSL defaults. Opt-in composition rules inspect declared canvas and
    placement structure, with evidence paths. When concept context is supplied,
    explicit taxonomy mappings determine the task-level action family; operation
    annotations remain provenance. Legacy snapshots retain their annotations.

    Args:
        program (dict[str, Any]): Layout, selector, and action expressions.
        taxonomy (dict[str, Any]): Validated family rules.
        concept_name (str | None): Declared spec concept. Without context, annotate
            only the program operations. Metadata exports always supply this.

    Returns:
        dict[str, Any]: Exact roots, operation records, parameters, family sets,
        and deduplicated values suitable for filtering and parent-level counts.
    """
    if concept_name is not None and (
        not isinstance(concept_name, str) or not concept_name
    ):
        raise ValueError("Concept context must be a nonempty string.")
    if set(program) != set(ROLES) or any(
        not isinstance(program[r], dict)
        or not isinstance(program[r].get("op"), str)
        or not program[r]["op"]
        for r in ROLES
    ):
        raise ValueError(
            "Program must contain layout, selector, and action operator mappings."
        )
    lookup: dict[str, dict[str, list[str]]] = {
        role: defaultdict(list) for role in ROLES
    }
    for role, families in taxonomy["families"].items():
        for family, rule in families.items():
            for op in rule["operations"]:
                lookup[role][op].append(family)
    helpers = set(taxonomy["helper_operations"])
    parameter_keys = set(taxonomy["parameter_keys"])
    expression_keys = set(taxonomy["expression_parameters"])
    all_parameters = taxonomy.get("parameter_mode") == "all_literals"
    ordered_parameters = set(taxonomy.get("ordered_parameter_keys", []))
    index_operations = taxonomy.get("index_all_operations", False)
    parameter_rules = taxonomy.get("parameter_families", [])
    composition_rules = taxonomy.get("composition_rules", {})
    color_rules = taxonomy.get("color_edit_rules")
    secondary_actions = taxonomy.get("secondary_action_families", {})
    operations: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    families_found: dict[str, set[str]] = {role: set() for role in ROLES}
    values: dict[str, set[str]] = defaultdict(set)
    for role in ROLES:
        values[f"op.{role}"].add(program[role]["op"])

    def parameter(path: str, value: Any) -> None:
        """Retain literal values, with explicit ordering for sequence parameters.

        Args:
            path (str): Actual JSON path.
            value (Any): Scalar or flat scalar list.
        """
        items = value if isinstance(value, list) else [value]
        if any(
            not isinstance(item, (str, bool, int, float)) and item is not None
            for item in items
        ):
            return
        if (
            all_parameters
            and isinstance(value, list)
            and (
                path.rsplit(".", 1)[-1] in ordered_parameters
                or not value
                or not all(isinstance(item, str) for item in value)
            )
        ):
            # Preserve ranges, vectors, and declared sequences as complete values.
            items = [value]
        if not items:
            return
        field = "parameter." + re.sub(r"\[\d+\]", "[]", path)
        parameters.append({"path": path, "field": field, "value": value})
        for item in items:
            values[field].add(json.dumps(item, sort_keys=True, allow_nan=False))

    def walk(value: Any, path: str, scope: str, root: bool = False) -> None:
        """Walk expression nodes without executing any task program.

        Args:
            value (Any): Current JSON value.
            path (str): Its full JSON path.
            scope (str): Original layout/selector/action subtree.
            root (bool): Whether this is that subtree's root operator.
        """
        if isinstance(value, dict):
            op = value.get("op")
            if "op" in value:
                if not isinstance(op, str) or not op:
                    raise ValueError(f"Invalid operator at {path}")
                if root:
                    role = scope
                elif (
                    composition_rules
                    and scope == "action"
                    and op in ("compose_canvas", "place_selection")
                ):
                    role = "action"
                else:
                    role = next(
                        (r for r in ("selector", "action") if op in lookup[r]),
                        "expression",
                    )
                matches = set(lookup.get(role, {}).get(op, []))
                evidence = []
                for rule in parameter_rules:
                    if rule["role"] != role or op not in rule["operations"]:
                        continue
                    literal: Any = value
                    for key in rule["parameter"].split("."):
                        literal = (
                            literal.get(key) if isinstance(literal, dict) else None
                        )
                    if isinstance(literal, str) and literal in rule["values"]:
                        matches.add(rule["family"])
                        evidence.append(
                            {
                                "family": rule["family"],
                                "path": f"{path}.{rule['parameter']}",
                                "value": literal,
                                "description": rule["description"],
                            }
                        )
                if role == "selector" and taxonomy.get("selector_rules"):
                    selected_evidence = selector_parameter_evidence(
                        value, path, taxonomy["selector_rules"]
                    )
                    evidence.extend(selected_evidence)
                    matches.update(item["family"] for item in selected_evidence)
                composition_cases = (
                    [
                        (case, relative)
                        for case, relative in _composition_cases(value)
                        if case in composition_rules
                    ]
                    if composition_rules and role == "action" and op == "compose_canvas"
                    else []
                )
                preserved_placement = any(
                    case in ("copy_to_input", "extract_to_position")
                    for case, _ in composition_cases
                )
                color_evidence = []
                color_issues = []
                if (
                    color_rules
                    and role == "action"
                    and op in COLOR_EDIT_OPS
                    and not preserved_placement
                ):
                    color_edit = annotate_color_edit(
                        value, program["selector"], path, color_rules
                    )
                    color_evidence, color_issues = (
                        color_edit.evidence,
                        color_edit.issues,
                    )
                    matches = {item["family"] for item in color_evidence}
                composition_evidence = []
                if composition_cases:
                    for case, relative in composition_cases:
                        if color_rules and (
                            case in ("solid_render", "alternating_render")
                            or (case == "overlay_in_place" and color_evidence)
                        ):
                            continue
                        rule = composition_rules[case]
                        for family in rule["families"]:
                            matches.add(family)
                            composition_evidence.append(
                                {
                                    "case": case,
                                    "family": family,
                                    "path": f"{path}.{relative}",
                                    "description": rule["description"],
                                }
                            )
                status = (
                    "mapped" if matches else "helper" if op in helpers else "unmapped"
                )
                operations.append(
                    {
                        "path": path,
                        "scope": scope,
                        "role": role,
                        "op": op,
                        "families": sorted(matches),
                        "status": status,
                        **(
                            {"secondary_families": list(secondary_actions[op])}
                            if role == "action" and op in secondary_actions
                            else {}
                        ),
                        **({"parameter_evidence": evidence} if evidence else {}),
                        **(
                            {"color_edit_evidence": color_evidence}
                            if color_evidence
                            else {}
                        ),
                        **({"color_edit_issues": color_issues} if color_issues else {}),
                        **(
                            {"composition_evidence": composition_evidence}
                            if composition_evidence
                            else {}
                        ),
                    }
                )
                if role in families_found:
                    families_found[role].update(matches)
                if index_operations:
                    values[f"operation.{role}"].add(op)
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if (all_parameters and key != "op") or key in parameter_keys:
                    parameter(child_path, child)
                if (
                    (all_parameters or key in expression_keys)
                    and isinstance(child, dict)
                    and isinstance(child.get("op"), str)
                ):
                    parameter(child_path + ".op", child["op"])
                walk(child, child_path, scope)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                if all_parameters and isinstance(child, dict) and "op" in child:
                    parameter(f"{path}[{i}].op", child["op"])
                walk(child, f"{path}[{i}]", scope)

    for role in ROLES:
        walk(program[role], f"program.{role}", role, root=True)
        # Families may include selector nodes found later inside an action.
    if taxonomy.get("layout_rules"):
        families_found["layout"] = apply_layout_rules(
            program["layout"], operations, taxonomy["layout_rules"]
        )
    if taxonomy.get("fill_state_rules"):
        families_found["action"] = apply_fill_state_rules(
            program, operations, taxonomy["fill_state_rules"]
        )
    action_family = taxonomy.get("concept_action_families", {}).get(concept_name)
    if action_family is not None:
        families_found["action"] = {action_family}
    for role, names in families_found.items():
        if names:
            values[f"family.{role}"].update(names)
    return {
        "root_ops": {role: program[role]["op"] for role in ROLES},
        **(
            {
                "action_family_evidence": {
                    "source": "concept_name",
                    "value": concept_name,
                    "family": action_family,
                }
            }
            if action_family is not None
            else {}
        ),
        "operations": operations,
        "parameters": parameters,
        "families": {role: sorted(names) for role, names in families_found.items()},
        "values": {field: sorted(items) for field, items in sorted(values.items())},
    }


def load_spec_metadata(
    spec_dir: Path,
    task_ids: list[str],
    concepts: list[str],
    labels: list[int],
    taxonomy: dict[str, Any],
) -> MetadataBundle:
    """Join specs by explicit parent identity, rejecting incomplete/conflicting data.

    Args:
        spec_dir (Path): Directory of exported concept-scoped task specs.
        task_ids (list[str]): Saved parent identities in feature order.
        concepts (list[str]): Saved concept vocabulary.
        labels (list[int]): Concept indices aligned to task_ids.
        taxonomy (dict[str, Any]): Validated semantic rules.

    Returns:
        MetadataBundle: Aligned annotations, category counts, coverage, and hashes.
    """
    if not spec_dir.is_dir():
        raise ValueError(f"Spec directory does not exist: {spec_dir}")
    if (
        not task_ids
        or len(set(task_ids)) != len(task_ids)
        or len(labels) != len(task_ids)
        or len(set(concepts)) != len(concepts)
    ):
        raise ValueError("Invalid or duplicate saved parent/concept axes.")
    if any(
        type(label) is not int or not 0 <= label < len(concepts) for label in labels
    ):
        raise ValueError("Invalid saved concept labels.")
    expected = dict(zip(task_ids, (concepts[label] for label in labels)))
    if any(
        task.split("/")[0] != concept or len(task.split("/")) != 2
        for task, concept in expected.items()
    ):
        raise ValueError("Saved parent identities conflict with concept labels.")
    by_id: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    for path in sorted(spec_dir.rglob("*.json")):
        raw = path.read_bytes()
        spec = json.loads(raw)
        if not isinstance(spec, dict):
            raise ValueError(f"Spec must contain a JSON object: {path}")
        task = spec.get("task_id")
        if not isinstance(task, str) or task in by_id:
            raise ValueError(f"Missing or duplicate spec task_id: {task!r}")
        relative = path.relative_to(spec_dir).as_posix()
        concept = spec.get("concept_name")
        if (
            len(task.split("/")) != 2
            or not all(task.split("/"))
            or concept != task.split("/")[0]
        ):
            raise ValueError(f"Conflicting spec identity/concept: {relative}")
        if (
            spec.get("program_kind") != "conceptarc_dsl_v1"
            or not isinstance(spec.get("description"), str)
            or not isinstance(spec.get("program"), dict)
        ):
            raise ValueError(f"Unsupported or malformed spec: {relative}")
        fingerprints[relative] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        if task in expected and concept != expected[task]:
            raise ValueError(f"Conflicting concept for {task}")
        by_id[task] = {
            **spec,
            "spec_path": str(path.resolve()),
            "source_fingerprint": fingerprints[relative],
        }
    missing = sorted(set(task_ids) - by_id.keys())
    if missing:
        raise ValueError(
            f"Missing specs for {len(missing)} parent tasks: {missing[:10]}"
        )
    records = [
        {
            **by_id[task],
            **annotate_program(
                by_id[task]["program"],
                taxonomy,
                concept_name=by_id[task]["concept_name"],
            ),
        }
        for task in task_ids
    ]
    fields = _field_inventory(records, taxonomy)
    operator_tasks: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    node_counts: Counter[tuple[str, str, str]] = Counter()
    for i, record in enumerate(records):
        for node in record["operations"]:
            key = (node["role"], node["op"], node["status"])
            operator_tasks[key].add(i)
            node_counts[key] += 1
    operators = [
        {
            "role": role,
            "op": op,
            "status": status,
            "task_count": len(indices),
            "node_count": node_counts[(role, op, status)],
            "concept_counts": dict(
                Counter(records[i]["concept_name"] for i in sorted(indices))
            ),
        }
        for (role, op, status), indices in sorted(operator_tasks.items())
    ]
    inventory = {
        "matched_tasks": len(records),
        "spec_files": len(by_id),
        "unused_task_ids": sorted(by_id.keys() - set(task_ids)),
        "root_operation_counts": {
            role: dict(
                sorted(Counter(row["root_ops"][role] for row in records).items())
            )
            for role in ROLES
        },
        "root_family_coverage": {
            role: {
                "mapped_tasks": sum(
                    any(
                        node["path"] == f"program.{role}" and node["families"]
                        for node in row["operations"]
                    )
                    for row in records
                ),
                "total_tasks": len(records),
            }
            for role in ROLES
        },
        "operators": operators,
        "unmapped_operators": [
            item for item in operators if item["status"] == "unmapped"
        ],
        "fields": fields,
    }
    if taxonomy.get("color_edit_rules"):
        inventory["unresolved_color_edits"] = [
            {"task_id": row["task_id"], **issue}
            for row in records
            for node in row["operations"]
            for issue in node.get("color_edit_issues", [])
        ]
    if taxonomy.get("fill_state_rules"):
        inventory["unresolved_fill_states"] = [
            {"task_id": row["task_id"], **issue}
            for row in records
            for node in row["operations"]
            for issue in node.get("fill_state_issues", [])
        ]
    if taxonomy.get("layout_rules"):
        inventory["unresolved_layouts"] = [
            {"task_id": row["task_id"], **issue}
            for row in records
            for node in row["operations"]
            for issue in node.get("layout_issues", [])
        ]
    return MetadataBundle(records, fields, inventory, fingerprints)


def validate_saved_metadata(metadata: dict[str, Any], taxonomy: dict[str, Any]) -> None:
    """Check saved annotations and category counts against their program snapshot.

    Args:
        metadata (dict[str, Any]): Saved metadata payload.
        taxonomy (dict[str, Any]): Validated taxonomy snapshot from the artifact.
    """
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or metadata.get("taxonomy_version") != taxonomy["version"]
        or metadata.get("taxonomy") != taxonomy
    ):
        raise ValueError("Saved metadata conflicts with its taxonomy snapshot.")
    records = metadata.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Saved metadata must contain parent records.")
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("program"), dict)
            or not isinstance(record.get("concept_name"), str)
        ):
            raise ValueError("Malformed saved metadata record.")
        expected = annotate_program(
            record["program"], taxonomy, concept_name=record["concept_name"]
        )
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"Saved metadata annotations conflict with the program: {record.get('task_id')}"
            )
    if metadata.get("fields") != _field_inventory(records, taxonomy):
        raise ValueError(
            "Saved metadata fields or counts conflict with parent records."
        )


def _field_inventory(
    records: list[dict[str, Any]], taxonomy: dict[str, Any]
) -> list[dict[str, Any]]:
    """Count distinct parents per category and concept for searchable controls.

    Args:
        records (list[dict[str, Any]]): Aligned annotated task specs.
        taxonomy (dict[str, Any]): Display labels and family explanations.

    Returns:
        list[dict[str, Any]]: Ordered fields and their observed value vocabulary.
    """
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for record in records:
        for field, values in record["values"].items():
            for value in set(values):
                counts[field][value][record["concept_name"]] += 1
    result = []
    for field, values in sorted(
        counts.items(), key=lambda item: (item[0].split(".")[0] == "parameter", item[0])
    ):
        kind, role = field.split(".", 1)
        label = (
            f"{role.title()} {'family' if kind == 'family' else 'operation'}"
            if kind != "parameter"
            else role.removeprefix("program.")
        )
        if kind == "operation":
            label = f"{role.title()} operations (all depths)"
        entries = []
        for value, concept_counts in sorted(values.items()):
            display, description = value, ""
            if kind == "family":
                rule = taxonomy["families"][role][value]
                display, description = rule["label"], rule["description"]
            elif kind == "parameter":
                literal = json.loads(value)
                display = literal if isinstance(literal, str) else json.dumps(literal)
            entries.append(
                {
                    "value": value,
                    "label": display,
                    "description": description,
                    "count": sum(concept_counts.values()),
                    "concept_counts": dict(sorted(concept_counts.items())),
                }
            )
        result.append({"id": field, "label": label, "kind": kind, "values": entries})
    return result
