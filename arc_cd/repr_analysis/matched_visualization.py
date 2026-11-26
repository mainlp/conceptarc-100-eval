"""Plot controlled separation, differences from original and matching quality."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter
from tqdm.auto import tqdm

from arc_cd.repr_analysis.family_visualization import COLORS, ROLE_LABELS
from arc_cd.repr_analysis.figure_style import ML_CONFERENCE_STYLE, family_line_style
from arc_cd.repr_analysis.matched_design import CONDITIONS

matplotlib.use("Agg")

LOGGER = logging.getLogger(__name__)
LABELS = {
    "original_figure4": "Original (non-constrained)",
    "frequency_balanced": "Action-frequency balanced",
    "layout_only": "Layout matched",
    **{
        f"{profile}_q{q}": f"{label} matched · q{q}"
        for profile, label in (("input", "Input grids"), ("full", "Input-output grids"))
        for q in (25, 50, 75)
    },
}
VISUAL_LABELS = {
    f"{profile}_q{q}": f"{label} · q{q}"
    for profile, label in (("input", "Input"), ("full", "Input & Output"))
    for q in (25, 50, 75)
}
CURVE_STYLES = {
    "original_figure4": {"color": "#333333", "linestyle": "-"},
    "frequency_balanced": {"color": "#009E73", "linestyle": "-"},
    "layout_only": {"color": "#AA4499", "linestyle": "-"},
    **{
        f"{profile}_q{q}": {"color": color, "linestyle": style}
        for profile, color in (("input", "#0072B2"), ("full", "#D55E00"))
        for q, style in zip((25, 50, 75), ("-", "--", ":"))
    },
}


def numeric_value(row: dict[str, Any], key: str) -> float:
    """Read a JSON summary value, preserving unavailable scores.

    Args:
        row (dict[str, Any]): Summary entry.
        key (str): Numerical field name.

    Returns:
        float: Saved value, or NaN for null or absent fields.
    """
    value = row.get(key)
    return float(value) if value is not None else float("nan")


def _save(fig: plt.Figure, path: Path, *, pad_inches: float = 0.1) -> None:
    """Save matching PNG/PDF figures and release their memory.

    Args:
        fig (plt.Figure): Completed figure.
        path (Path): File stem without extension.
        pad_inches (float): White margin around the exported figure.
    """
    fig.savefig(
        path.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=pad_inches
    )
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)


def _plot_comparison(data: dict[str, Any], path: Path) -> None:
    """Plot nine separation scores and eight differences from the original.

    Args:
        data (dict[str, Any]): Condition summaries for one representation.
        path (Path): Output file stem without extension.
    """
    names = ["original_figure4", *CONDITIONS]
    fig, panels = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    for y, name in enumerate(names):
        row = data[name]
        score = 100 * numeric_value(row, "tie_adjusted")
        panels[0].scatter(score, y, color=CURVE_STYLES[name]["color"])
        if not np.isfinite(score):
            panels[0].text(2, y, "Unavailable", va="center", fontsize=8)
    panels[0].set(
        yticks=range(len(names)),
        yticklabels=[
            f"{LABELS[n]}\n{data[n]['headline_actions']} actions; {data[n]['headline_anchors']} anchors"
            for n in names
        ],
        xlim=(0, 100),
        ylim=(-0.5, len(names) - 0.5),
        xlabel="Action separation probability (%)",
    )
    panels[0].invert_yaxis()
    panels[0].axvline(50, color="0.7", linestyle="--", linewidth=1)
    original = numeric_value(data["original_figure4"], "tie_adjusted")
    differences = [
        100 * (numeric_value(data[name], "tie_adjusted") - original)
        for name in CONDITIONS
    ]
    panels[1].barh(
        range(len(CONDITIONS)),
        differences,
        color=[CURVE_STYLES[name]["color"] for name in CONDITIONS],
    )
    for y, value in enumerate(differences):
        if not np.isfinite(value):
            panels[1].text(0, y, "Unavailable", va="center", fontsize=8)
    panels[1].set(
        yticks=range(len(CONDITIONS)),
        yticklabels=[LABELS[n] for n in CONDITIONS],
        ylim=(-0.5, len(CONDITIONS) - 0.5),
        xlabel="Difference from original\n(percentage points)",
    )
    panels[1].invert_yaxis()
    panels[1].axvline(0, color="0.5", linewidth=1)
    _save(fig, path)


def _plot_visual_differences(
    data: dict[str, Any], layers: list[int], path: Path
) -> None:
    """Plot visual-matching differences from the original at each layer.

    Args:
        data (dict[str, Any]): One position's scores indexed by layer and condition.
        layers (list[int]): Sorted transformer layer indices.
        path (Path): Output file stem without extension.
    """
    original = np.asarray(
        [
            numeric_value(data[str(layer)]["original_figure4"], "tie_adjusted")
            for layer in layers
        ]
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.8), layout="constrained")
    for name, label in VISUAL_LABELS.items():
        scores = np.asarray(
            [numeric_value(data[str(layer)][name], "tie_adjusted") for layer in layers]
        )
        style = dict(CURVE_STYLES[name])
        style["color"] = "#008B8B" if name.startswith("input_") else "#7B3294"
        ax.plot(layers, 100 * (scores - original), label=label, **style)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Transformer layer", fontsize=18)
    ax.set_ylabel("$\\Delta$ Action separation probability (%)", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.margins(x=0.015)
    ax.legend(
        loc="upper left",
        fontsize=14,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
        handlelength=2.5,
    )
    _save(fig, path, pad_inches=0.08)


def _plot_roles(config: dict[str, Any], summary: dict[str, Any], output: Path) -> None:
    """Plot action, concept and selector curves separately for each control.

    Args:
        config (dict[str, Any]): Evaluated positions, layers and target roles.
        summary (dict[str, Any]): Scores indexed by position/layer/role/condition.
        output (Path): Parent destination for condition-specific figures.
    """
    layers = sorted(config["layers"])
    with plt.rc_context(ML_CONFERENCE_STYLE):
        for condition in ("original_figure4", *CONDITIONS):
            directory = output / condition
            directory.mkdir(parents=True, exist_ok=True)
            for position in config["positions"]:
                fig, ax = plt.subplots(figsize=(6.8, 4.8), layout="constrained")
                for index, role in enumerate(COLORS):
                    if role not in config["roles"]:
                        continue
                    scores = [
                        100
                        * numeric_value(
                            summary[position][str(layer)][role][condition],
                            "tie_adjusted",
                        )
                        for layer in layers
                    ]
                    style = family_line_style(index, len(layers))
                    style.update(color=COLORS[role], linestyle="-")
                    ax.plot(layers, scores, label=ROLE_LABELS[role], **style)
                ax.axhline(50, color="#666666", linestyle="--", linewidth=1)
                ax.set_ylim(min(45, ax.get_ylim()[0]), 100)
                ax.set_xlabel("Transformer layer", fontsize=18)
                ax.set_ylabel("Separation probability", fontsize=18)
                ax.tick_params(axis="both", labelsize=14)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
                ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
                ax.set_axisbelow(True)
                ax.grid(True, color="#CCD2DA", linestyle="--", linewidth=0.6, alpha=0.7)
                ax.margins(x=0.015)
                ax.legend(loc="upper right", fontsize=14, frameon=False, ncols=1)
                _save(fig, directory / f"{position}_tie_adjusted")


def plot_matched_analysis(directory: Path, output_dir: Path | None = None) -> Path:
    """Regenerate comparison figures using only compact saved summaries.

    Args:
        directory (Path): Completed evaluation directory.
        output_dir (Path | None): Figure destination, defaulting to figures/.

    Returns:
        Path: Figure directory.
    """
    directory = directory.expanduser().resolve()
    output = output_dir.expanduser().resolve() if output_dir else directory / "figures"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((directory / "run_config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if config.get("analysis") == "matched_role_separation":
        LOGGER.info("Plotting action, concept and selector comparisons.")
        _plot_roles(config, summary, output)
        return output
    vector_plot = {
        "visual_features": "visual_comparison",
        "task_centroids": "centroid_comparison",
    }.get(config.get("input_kind", ""))
    if vector_plot:
        LOGGER.info("Rendering the nine conditions for %s.", config["input_kind"])
    else:
        LOGGER.info(
            "Rendering comparisons for %d positions and %d layers.",
            len(config["positions"]),
            len(config["layers"]),
        )
    with plt.rc_context(ML_CONFERENCE_STYLE):
        if vector_plot:
            _plot_comparison(summary, output / vector_plot)
        for position in config.get("positions", []):
            for layer in tqdm(
                config["layers"],
                desc=f"Plotting {position}",
                unit="layer",
                disable=None,
            ):
                data = summary[position][str(layer)]
                _plot_comparison(data, output / f"{position}_layer_{layer}_comparison")
            if len(config["layers"]) > 1:
                LOGGER.info(
                    "Plotting all nine conditions across layers for %s.", position
                )
                fig, panels = plt.subplots(
                    1,
                    2,
                    figsize=(13, 6),
                    layout="constrained",
                    gridspec_kw={"wspace": 0.08},
                )
                layers = sorted(config["layers"])
                original_scores = np.asarray(
                    [
                        numeric_value(
                            summary[position][str(layer)]["original_figure4"],
                            "tie_adjusted",
                        )
                        for layer in layers
                    ]
                )
                for name in ("original_figure4", *CONDITIONS):
                    scores = np.asarray(
                        [
                            numeric_value(
                                summary[position][str(layer)][name], "tie_adjusted"
                            )
                            for layer in layers
                        ]
                    )
                    panels[0].plot(
                        layers, 100 * scores, label=LABELS[name], **CURVE_STYLES[name]
                    )
                    if name != "original_figure4":
                        panels[1].plot(
                            layers,
                            100 * (scores - original_scores),
                            label=LABELS[name],
                            **CURVE_STYLES[name],
                        )
                for panel, reference, label in zip(
                    panels,
                    (50, 0),
                    (
                        "Action separation probability (%)",
                        "Difference from original (%)",
                    ),
                ):
                    panel.axhline(reference, color="grey", linestyle="--")
                    panel.set_xlabel("Transformer layer", fontsize=18)
                    panel.set_ylabel(label, fontsize=18)
                    panel.tick_params(axis="both", labelsize=16)
                    panel.xaxis.set_major_locator(MaxNLocator(integer=True))
                    panel.grid(axis="y", color="0.9", linewidth=0.5)
                panels[0].set_ylim(0, 100)
                handles, labels = panels[0].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="outside upper center",
                    ncol=3,
                    fontsize=16,
                    handlelength=3,
                )
                _save(fig, output / f"{position}_layers")
                _plot_visual_differences(
                    summary[position],
                    layers,
                    output / f"{position}_layers_visual_difference",
                )
        LOGGER.info("Plotting visual matching quality.")
        matching = json.loads((directory / "matching_summary.json").read_text())
        fig, panels = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for panel, profile in zip(panels, ("input", "full")):
            for i, name in enumerate(CONDITIONS[1:]):
                values = [
                    numeric_value(r, f"{profile}_distance_median")
                    for r in matching[name]["actions"].values()
                    if r["headline"]
                ]
                panel.scatter(np.full(len(values), i), values, alpha=0.5, s=16)
            panel.set(
                xticks=range(7),
                xticklabels=[LABELS[n].replace(" · ", "\n") for n in CONDITIONS[1:]],
                ylabel=f"Median B–C {profile} distance",
            )
            panel.tick_params(axis="x", labelsize=8)
        _save(fig, output / "matching_quality")
    LOGGER.info("Finished rendering figures.")
    return output
