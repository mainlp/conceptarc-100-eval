"""Qualitative support-set triplets from the frozen full-profile controls."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from arc_cd.data_prep.concept_arc import ConceptARCTask
from arc_cd.eval.visualize import (
    draw_pair,
    grid_pair_to_example,
    load_font,
    pair_size,
)
from arc_cd.repr_analysis.matched_design import CONDITIONS
from arc_cd.repr_analysis.matched_features import extract_controls, load_matching_tasks
from arc_cd.repr_analysis.results import write_json

LOGGER = logging.getLogger(__name__)
DEFAULT_NUM_QUALITATIVE_EXAMPLES = 10


def _select_triplets(
    design: dict[str, np.ndarray],
    distances: np.ndarray,
    family: int,
    quantile: int,
    action_operators: np.ndarray,
    num_examples: int,
) -> list[tuple[int, int, int]]:
    """Select pairs near the band midpoint with varied anchor operators.

    Args:
        design (dict[str, np.ndarray]): Frozen matching graph and eligibility.
        distances (np.ndarray): Full-profile distances in edge order.
        family (int): Target action index.
        quantile (int): Full-profile cutoff index (0, 1 or 2).
        action_operators (np.ndarray): Exact action operator per matching parent.
        num_examples (int): Maximum number of triplets to select.

    Returns:
        list[tuple[int, int, int]]: Distinct anchor/positive/negative parent indices.
            Selection is deterministic and independent of representation scores.
    """
    condition = CONDITIONS.index("full_q25") + quantile
    if not design["headline"][condition, family]:
        return []
    start, stop = design["edge_offsets"][family : family + 2]
    edges = np.arange(start, stop)
    keep = (design["edge_flags"][edges] & (1 << (condition - 1))) != 0
    keep &= design["membership"][design["edge_c"][edges]].any(axis=1)
    edges = edges[keep]
    lower = float(design["cutoffs"][1, quantile - 1]) if quantile else 0.0
    midpoint = (lower + design["cutoffs"][1, quantile]) / 2
    edges = edges[np.argsort(np.abs(distances[edges] - midpoint), kind="stable")]
    newly_admitted = distances[edges] >= lower
    members = np.flatnonzero(design["membership"][:, family])
    used_a: set[int] = set()
    used_b: set[int] = set()
    used_c: set[int] = set()
    used_actions: set[str] = set()
    triplets: list[tuple[int, int, int]] = []
    for pool in (edges[newly_admitted], edges[~newly_admitted]):
        remaining = pool.tolist()
        while remaining and len(triplets) < num_examples:
            fresh_anchors = [
                a for a in members if action_operators[a] not in used_actions
            ]
            candidates = remaining
            if len(fresh_anchors) == 1:
                # Keep the last unused action available for the anchor.
                candidates = [
                    e for e in remaining if design["edge_b"][e] != fresh_anchors[0]
                ] or remaining
            edge = next(
                (
                    e
                    for e in candidates
                    if int(design["edge_b"][e]) not in used_b
                    and int(design["edge_c"][e]) not in used_c
                ),
                candidates[0],
            )
            remaining.remove(edge)
            b, c = int(design["edge_b"][edge]), int(design["edge_c"][edge])
            anchors = [int(a) for a in members if a != b]
            a = min(
                anchors,
                key=lambda a: (action_operators[a] in used_actions, a in used_a),
            )
            triplets.append((a, b, c))
            used_a.add(a)
            used_b.add(b)
            used_c.add(c)
            used_actions.add(str(action_operators[a]))
    return triplets


def _draw_triplet(
    tasks: tuple[ConceptARCTask, ConceptARCTask, ConceptARCTask],
    action: str,
    condition: str,
    distance: float,
    distance_band: tuple[float, float],
    action_operators: tuple[str, str, str],
    path: Path,
) -> None:
    """Render three support sets in anchor/positive/negative columns.

    Args:
        tasks (tuple[ConceptARCTask, ConceptARCTask, ConceptARCTask]): First variants
            of the anchor, positive and negative parents, in that order.
        action (str): Target action shared by anchor and positive.
        condition (str): Full-profile matching condition.
        distance (float): Distance between eight-variant mean control vectors.
        distance_band (tuple[float, float]): Lower inclusive and upper exclusive
            bounds of the newly admitted distance band.
        action_operators (tuple[str, str, str]): Exact anchor, positive and negative
            root action operators from the task annotations.
        path (Path): Destination PNG.
    """
    examples = [[grid_pair_to_example(pair) for pair in task.train] for task in tasks]
    max_dimension = max(
        max(len(grid), max(map(len, grid), default=0))
        for supports in examples
        for pair in supports
        for grid in pair.values()
    )
    cell = max(4, min(16, 240 // max(1, max_dimension)))
    sizes = [[pair_size(pair, cell) for pair in supports] for supports in examples]
    font, small = load_font(18), load_font(14)
    column_width = max(
        410,
        max(width for column in sizes for width, _ in column) + 40,
        int(max(small.getlength(task.task_id) for task in tasks)) + 40,
        int(max(small.getlength(f"Action operator: {op}") for op in action_operators))
        + 40,
    )
    row_heights = [max(column[row][1] for column in sizes) + 24 for row in range(3)]
    image = Image.new("RGB", (3 * column_width, 200 + sum(row_heights)), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (20, 12), f"Action group: {action}  |  {condition}", fill="black", font=font
    )
    draw.text(
        (20, 42),
        f"Full-profile B-C distance: {distance:.8f}  |  "
        f"target band: [{distance_band[0]:.6f}, {distance_band[1]:.6f})  |  "
        f"midpoint: {sum(distance_band) / 2:.6f}",
        fill="black",
        font=small,
    )
    draw.text(
        (20, 65),
        "First support-set variant shown; matching distances use all eight variants.",
        fill="#555555",
        font=small,
    )
    for column, (task, supports, label) in enumerate(
        zip(
            tasks,
            examples,
            (
                "Anchor (A)",
                "Positive (B): same action group",
                "Negative (C): lacks action group",
            ),
        )
    ):
        x = column * column_width
        if column:
            draw.line((x, 100, x, image.height - 15), fill="#cccccc", width=2)
        draw.text((x + 20, 105), label, fill="black", font=font)
        draw.text((x + 20, 134), task.task_id, fill="#555555", font=small)
        draw.text(
            (x + 20, 160),
            f"Action operator: {action_operators[column]}",
            fill="black",
            font=small,
        )
        y = 195
        for row, pair in enumerate(supports):
            width = sizes[column][row][0]
            draw_pair(
                draw, pair, row + 1, x + (column_width - width) // 2, y, cell, small
            )
            y += row_heights[row]
    image.save(path)


def save_qualitative_examples(
    data_dir: Path,
    axes: dict[str, Any],
    design: dict[str, np.ndarray],
    output_dir: Path,
    *,
    metadata_dir: Path,
    num_examples: int = DEFAULT_NUM_QUALITATIVE_EXAMPLES,
) -> None:
    """Save full-profile triplets per action as PNGs, grid JSONs and an index.

    Args:
        data_dir (Path): Original eight-variant dataset used to prepare matches.
        axes (dict[str, Any]): Parent and action names in frozen matching order.
        design (dict[str, np.ndarray]): Frozen matching graph and support identities.
        output_dir (Path): Destination for paired PNG/JSON examples and summary.json.
        metadata_dir (Path): Task annotations used to prepare the matching design.
        num_examples (int): Positive maximum per eligible action and condition.
    """
    if num_examples < 1:
        raise ValueError("Require a positive number of qualitative examples.")
    records = json.loads((metadata_dir / "task_metadata.json").read_text())["records"]
    by_parent = {record["task_id"]: record for record in records}
    if len(by_parent) != len(records) or set(by_parent) != set(axes["task_ids"]):
        raise ValueError(
            "Qualitative metadata parents differ from the matching design."
        )
    ordered_records = [by_parent[task] for task in axes["task_ids"]]
    membership = np.asarray(
        [
            [action in record["families"]["action"] for action in axes["families"]]
            for record in ordered_records
        ]
    )
    if not np.array_equal(membership, design["membership"]):
        raise ValueError("Qualitative action annotations differ from matching.")
    action_operators = np.asarray([r["root_ops"]["action"] for r in ordered_records])
    LOGGER.info("Loading support grids and reconstructing full matching distances.")
    tasks = load_matching_tasks(data_dir, metadata_dir)
    controls = extract_controls(tasks)
    if set(controls["source_task_id"]) != set(axes["task_ids"]):
        raise ValueError("Qualitative source parents differ from the matching design.")
    control_rows = {task: i for i, task in enumerate(controls["source_task_id"])}
    order = np.asarray([control_rows[task] for task in axes["task_ids"]])
    if any(
        not np.array_equal(controls[key][order], design[key])
        for key in ("version_task_ids", "support_set_ids")
    ):
        raise ValueError("Qualitative support-set identities differ from matching.")
    vectors = controls["full_vectors"][order]
    distances = cdist(vectors, vectors)
    edge_distances = distances[design["edge_b"], design["edge_c"]]
    for q in range(3):
        saved = (design["edge_flags"] & (1 << (4 + q))) != 0
        if not np.array_equal(saved, edge_distances < design["cutoffs"][1, q]):
            raise ValueError("Source grids no longer reproduce the saved full matches.")
    by_task = {task.task_id: task for task in tasks}
    first_variants = [by_task[ids[0]] for ids in design["version_task_ids"]]
    summary: dict[str, Any] = {
        "requested_examples_per_action": num_examples,
        "selection": "Triplets per eligible action near each band's midpoint; prefer newly admitted B-C pairs, different anchor action operators and different B/C parents. Fall back to other allowed pairs or repeated operators when needed. No representation scores used.",
        "display": "First support-set variant; distances use the mean descriptors over all eight variants.",
        "conditions": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for q, condition in enumerate(("full_q25", "full_q50", "full_q75")):
        cutoff = float(design["cutoffs"][1, q])
        lower = float(design["cutoffs"][1, q - 1]) if q else 0.0
        distance_band = (lower, cutoff)
        target_distance = sum(distance_band) / 2
        actions = {}
        for family, action in enumerate(
            tqdm(
                axes["families"],
                desc=f"Examples {condition}",
                unit="action",
                disable=None,
            )
        ):
            triplets = _select_triplets(
                design, edge_distances, family, q, action_operators, num_examples
            )
            if not triplets:
                continue
            directory = output_dir / condition / action
            directory.mkdir(parents=True, exist_ok=True)
            records = []
            for number, (a, b, c) in enumerate(triplets, start=1):
                selected = (first_variants[a], first_variants[b], first_variants[c])
                operators = (
                    str(action_operators[a]),
                    str(action_operators[b]),
                    str(action_operators[c]),
                )
                distance = float(distances[b, c])
                image_path = directory / f"example_{number:02d}.png"
                _draw_triplet(
                    selected,
                    action,
                    condition,
                    distance,
                    distance_band,
                    operators,
                    image_path,
                )
                write_json(
                    image_path.with_suffix(".json"),
                    {
                        "action_group": action,
                        "condition": condition,
                        "distance_basis": "Euclidean distance between prepared full-profile mean descriptors over eight support-set variants; grids below are the first variant shown in the image.",
                        "stats": {
                            "positive_negative_distance": distance,
                            "distance_band": distance_band,
                            "target_distance": target_distance,
                            "within_band": lower <= distance < cutoff,
                        },
                        **{
                            role: {
                                "task_id": task.task_id,
                                "action_operator": operator,
                                "support_examples": [
                                    grid_pair_to_example(pair) for pair in task.train
                                ],
                            }
                            for role, task, operator in zip(
                                ("anchor", "positive", "negative"), selected, operators
                            )
                        },
                    },
                )
                records.append(
                    {
                        "anchor": selected[0].task_id,
                        "positive": selected[1].task_id,
                        "negative": selected[2].task_id,
                        "distance": distance,
                        "action_operators": dict(
                            zip(("anchor", "positive", "negative"), operators)
                        ),
                    }
                )
            actions[action] = records
            total += len(records)
        summary["conditions"][condition] = {
            "distance_band": distance_band,
            "target_distance": target_distance,
            "actions": actions,
        }
        LOGGER.info(
            "%s: saved examples for %d eligible actions.", condition, len(actions)
        )
    write_json(output_dir / "summary.json", summary)
    LOGGER.info("Saved %d qualitative examples.", total)
