"""Layer-wise and centroid separation figures for concepts and task families."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

from arc_cd.repr_analysis.figure_style import (
    ML_CONFERENCE_STYLE,
    curve_canvas,
    display_name,
    family_line_style,
    finish_legend,
    save_figure,
)

matplotlib.use("Agg")

COLORS = {
    "concept": "#333333",
    "action": "#D55E00",
    "layout": "#0072B2",
    "selector": "#009E73",
}
ROLE_LABELS = {
    "concept": "Concepts",
    "action": "Actions",
    "layout": "Layouts",
    "selector": "Selectors",
}


def _plot_family_trends(
    plots: Path,
    axes: dict[str, Any],
    results: dict[str, np.ndarray],
    position_index: int,
    scope_index: int,
    role: str,
    indices: np.ndarray,
    *,
    metric_name: str,
) -> None:
    """Plot eligible families with defined curves, ranking the legend by peak score.

    Args:
        plots (Path): Root figure directory.
        axes (dict[str, Any]): Saved axes and readable family names.
        results (dict[str, np.ndarray]): Family scores and coverage.
        position_index (int): Support-position index.
        scope_index (int): Comparison-scope index.
        role (str): action, layout, or selector.
        indices (np.ndarray): Family columns in stable saved order.
        metric_name (str): Separation metric to plot.
    """
    metric = axes["metrics"].index(metric_name)
    position, scope = axes["positions"][position_index], axes["scopes"][scope_index]
    values = results["family_mean"][position_index, :, scope_index, :, metric][
        :, indices
    ].T
    visible_rows = np.flatnonzero(
        np.asarray(axes["headline_eligible"], dtype=bool)[indices]
        & np.isfinite(values).any(axis=1)
    )
    names = axes.get(
        "family_display_names", [display_name(name) for name in axes["family_names"]]
    )
    fig, ax, key = curve_canvas("", "", len(visible_rows))
    fig.set_size_inches(10.5, max(4.8, 0.30 * len(visible_rows) + 0.6))
    ax.set_ylabel("Separation probability", fontsize=18)
    if axes.get("distance_metric") == "euclidean":
        ax.set_title("Euclidean distance", fontsize=12)
    ax.xaxis.label.set_fontsize(18)
    ax.tick_params(axis="both", labelsize=14)
    for row in visible_rows:
        ax.plot(
            axes["layers"],
            values[row],
            label=names[indices[row]],
            **family_line_style(int(row), len(axes["layers"])),
        )
    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 1)
    ax.set_yticks([0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    peaks = np.max(
        np.where(np.isfinite(values[visible_rows]), values[visible_rows], -np.inf),
        axis=1,
    )
    finish_legend(
        ax,
        key,
        "",
        order=np.argsort(-peaks, kind="stable").tolist(),
        fontsize=14,
    )
    save_figure(
        fig,
        plots / "family_trends" / role / scope / f"{position}_{metric_name}.png",
        plots / "family_heatmaps" / role / scope / f"{position}_{metric_name}.png",
        pdf=True,
    )


def plot_metadata_analysis(
    directory: Path,
) -> None:
    """Regenerate ML-conference figures without changing numerical result artifacts.

    Args:
        directory (Path): Saved family separation results.
    """
    with matplotlib.rc_context(ML_CONFERENCE_STYLE):
        _plot_metadata_figures(directory)


def _plot_metadata_figures(
    directory: Path,
) -> None:
    """Render metadata figures within the shared ML-conference style context.

    Args:
        directory (Path): Numerical result directory.
    """
    axes = json.loads((directory / "axes.json").read_text())
    config = json.loads((directory / "run_config.json").read_text())
    distance_label = config.get(
        "distance_metric", config["analysis"].removesuffix("_distance")
    )
    taxonomy = json.loads((directory / "taxonomy.json").read_text())
    axes["family_display_names"] = [
        taxonomy.get("families", {})
        .get(role, {})
        .get(name, {})
        .get("label", display_name(name))
        for role, name in zip(axes["family_roles"], axes["family_names"])
    ]
    plots = directory / "figures"
    plots.mkdir(exist_ok=True)
    with np.load(directory / "metrics.npz", allow_pickle=False) as archive:
        results = {key: archive[key] for key in archive.files}
    if config.get("input_kind") in ("task_centroids", "visual_features"):
        _plot_centroid_figures(plots, axes, results)
        return
    macro = results["macro"]
    for p, position in enumerate(axes["positions"]):
        for s, scope in enumerate(axes["scopes"]):
            for m, metric in enumerate(axes["metrics"]):
                roles = [
                    (r, role)
                    for r, role in enumerate(axes["roles"])
                    if not (role == "concept" and scope == "cross_concept")
                ]
                fig, ax = plt.subplots(
                    figsize=(6.8, 4.8),
                    layout="constrained",
                )
                if distance_label == "euclidean":
                    ax.set_title("Euclidean distance", fontsize=12)
                ax.set_xlabel("Transformer layer", fontsize=18)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
                ax.tick_params(axis="both", labelsize=14)
                ax.set_axisbelow(True)
                ax.grid(True, color="#CCD2DA", linestyle="--", linewidth=0.6, alpha=0.7)
                ax.margins(x=0.015)
                for r, role in roles:
                    style = family_line_style(r, len(axes["layers"]))
                    style.update(color=COLORS[role], linestyle="-")
                    ax.plot(
                        axes["layers"],
                        macro[p, :, s, r, m],
                        label=ROLE_LABELS[role],
                        **style,
                    )
                ax.set_ylim(0.45, 1)
                ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
                ax.set_ylabel("Separation probability")
                if metric == "tie_adjusted":
                    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
                ax.yaxis.label.set_fontsize(18)
                ax.legend(
                    loc="upper right",
                    fontsize=14,
                    frameon=False,
                    ncols=1,
                )
                save_figure(
                    fig,
                    plots / "comparisons" / scope / f"{position}_{metric}.png",
                    plots / f"{position}_{scope}_{metric}.png",
                    pdf=True,
                )
            for role in config["arguments"]["family_roles"]:
                indices = np.flatnonzero(np.asarray(axes["family_roles"]) == role)
                if not len(indices):
                    continue
                _plot_family_trends(
                    plots,
                    axes,
                    results,
                    p,
                    s,
                    role,
                    indices,
                    metric_name="tie_adjusted",
                )


def _plot_centroid_figures(
    plots: Path, axes: dict[str, Any], results: dict[str, np.ndarray]
) -> None:
    """Plot single-representation scores without inventing layer or position axes.

    Args:
        plots (Path): Figure root directory.
        axes (dict[str, Any]): Canonical family, scope, and role axes.
        results (dict[str, np.ndarray]): Saved centroid macro and family scores.
    """
    for s, scope in enumerate(axes["scopes"]):
        for m, metric in enumerate(axes["metrics"]):
            values = results["macro"][s, :, m]
            indices = np.flatnonzero(np.isfinite(values))
            fig, ax = plt.subplots(figsize=(6.8, 4.8), layout="constrained")
            ax.bar(
                np.arange(len(indices)),
                values[indices],
                color=[COLORS[axes["roles"][r]] for r in indices],
                width=0.65,
            )
            ax.set_xticks(
                np.arange(len(indices)),
                [ROLE_LABELS[axes["roles"][r]] for r in indices],
            )
            ax.set_ylabel("Separation probability", fontsize=18)
            ax.set_ylim(*axes.get("probability_limits", (0.45, 1)))
            if axes.get("representation_label"):
                ax.set_title(
                    f"{axes['representation_label']}\n{scope} · {axes['population']['name']}",
                    fontsize=12,
                )
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.tick_params(axis="both", labelsize=14)
            if metric == "tie_adjusted":
                ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#CCD2DA", linestyle="--", linewidth=0.6, alpha=0.7)
            save_figure(
                fig, plots / "comparisons" / scope / f"centroid_{metric}.png", pdf=True
            )
        metric = axes["metrics"].index("tie_adjusted")
        for role in axes["roles"]:
            if role == "concept":
                continue
            indices = np.flatnonzero(np.asarray(axes["family_roles"]) == role)
            values = results["family_mean"][s, indices, metric]
            visible = np.flatnonzero(
                np.asarray(axes["headline_eligible"], dtype=bool)[indices]
                & np.isfinite(values)
            )
            visible = visible[np.argsort(-values[visible], kind="stable")]
            fig, ax = plt.subplots(
                figsize=(8.0, max(4.8, 0.30 * len(visible) + 1.2)), layout="constrained"
            )
            for y, row in enumerate(visible):
                color = family_line_style(int(row), 1)["color"]
                ax.hlines(y, 0.5, values[row], color=color, linewidth=1.8)
                ax.plot(values[row], y, marker="o", color=color, markersize=5)
            ax.set_yticks(
                np.arange(len(visible)),
                [axes["family_display_names"][indices[row]] for row in visible],
            )
            ax.invert_yaxis()
            ax.set_xlim(*axes.get("probability_limits", (0.45, 1)))
            if axes.get("representation_label"):
                ax.set_title(
                    f"{axes['representation_label']}\n{role} · {scope} · {axes['population']['name']}",
                    fontsize=12,
                )
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.axvline(0.5, color="#666666", linestyle="--", linewidth=1)
            ax.set_xlabel("Separation probability", fontsize=18)
            ax.tick_params(axis="both", labelsize=14)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#CCD2DA", linestyle="--", linewidth=0.6, alpha=0.7)
            save_figure(
                fig,
                plots / "family_scores" / role / scope / "centroid_tie_adjusted.png",
                pdf=True,
            )
