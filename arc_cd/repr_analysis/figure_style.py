"""Compact ML-conference figures with readable typography and optional PDF export."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

ML_CONFERENCE_STYLE: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 13,
    "axes.titlesize": 12.5,
    "axes.titleweight": "normal",
    "axes.labelpad": 6,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Keep family colors and marker shapes consistent across layers and positions.
FAMILY_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#AA4499",
    "#BB7700",
    "#56B4E9",
    "#444444",
    "#CC3311",
    "#332288",
    "#66A61E",
    "#E78AC3",
    "#17BECF",
    "#8C564B",
    "#9467BD",
    "#7F8000",
    "#882255",
    "#1B7837",
    "#B15928",
    "#4D4D9B",
    "#A6761D",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def display_name(name: str) -> str:
    """Convert an internal identifier to a readable fallback label.

    Args:
        name (str): Snake-case or CamelCase identifier.

    Returns:
        str: Readable sentence-case label; taxonomy labels take precedence elsewhere.
    """
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name.replace("_", " "))
    return words[:1].upper() + words[1:]


def curve_canvas(title: str, context: str, n_items: int) -> tuple[Figure, Axes, Axes]:
    """Create a curve panel with one title and a right-hand legend area.

    Args:
        title (str): Main figure title; empty omits the title.
        context (str): Compact position, scope, and neighborhood context.
        n_items (int): Number of legend entries used to reserve vertical space.

    Returns:
        tuple[Figure, Axes, Axes]: Figure, data axes, and right-hand legend axes.
    """
    legend_width = 1.8 if n_items <= 4 else 3.2
    fig, panels = plt.subplots(
        1,
        2,
        figsize=(5.8 + legend_width, max(3.6, 0.19 * n_items + 1.1)),
        layout="constrained",
        gridspec_kw={"width_ratios": [5.4, legend_width], "wspace": 0.04},
    )
    ax, key_ax = panels
    key_ax.set_axis_off()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("Transformer layer")
    ax.set_axisbelow(True)
    ax.grid(True, color="#CCD2DA", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.margins(x=0.015)
    if title:
        fig.suptitle(
            f"{title} — {context}" if context else title,
            fontsize=12.5,
            fontweight="normal",
        )
    return fig, ax, key_ax


def finish_legend(
    ax: Axes,
    key_ax: Axes,
    note: str,
    *,
    order: list[int] | None = None,
    fontsize: float | None = None,
) -> None:
    """Place a single-column legend to the right and a compact note below.

    Args:
        ax (Axes): Data axes with labeled curves.
        key_ax (Axes): Reserved right-hand legend area.
        note (str): One or two short coverage or method notes below the figure.
        order (list[int] | None): Legend-entry indices; defaults to drawing order.
        fontsize (float | None): Legend font size, or the active style default.
    """
    handles, labels = ax.get_legend_handles_labels()
    if order is not None:
        handles = [handles[index] for index in order]
        labels = [labels[index] for index in order]
    key_ax.legend(
        handles,
        labels,
        loc="center left",
        ncols=1,
        borderaxespad=0,
        handlelength=2.6,
        handletextpad=0.65,
        labelspacing=0.4,
        fontsize=fontsize,
    )
    if note:
        key_ax.get_figure().supxlabel(note, fontsize=8.5, ha="left", x=0.02)


def family_line_style(index: int, n_layers: int) -> dict[str, Any]:
    """Give each family a stable color and marker with a solid curve.

    Args:
        index (int): Family index within its role in saved order.
        n_layers (int): Number of plotted layers.

    Returns:
        dict[str, Any]: Matplotlib plot keyword arguments.
    """
    return {
        "color": FAMILY_COLORS[index % len(FAMILY_COLORS)],
        "linestyle": "-",
        "marker": MARKERS[index % len(MARKERS)],
        "markevery": max(1, n_layers // 8),
        "markersize": 3.2,
        "markerfacecolor": "white",
        "markeredgewidth": 0.8,
    }


def save_figure(
    fig: Figure, path: Path, legacy_path: Path | None = None, *, pdf: bool = False
) -> None:
    """Export a 450-dpi PNG and optionally a PDF, preserving previous PDF exports.

    Args:
        fig (Figure): Completed figure; dense heatmaps remain embedded images.
        path (Path): PNG destination within the descriptive figure folders.
        legacy_path (Path | None): Previous flat generated filename to retire.
        pdf (bool): Also write a PDF with embedded text and vector curves.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.08)
        if pdf:
            fig.savefig(
                path.with_suffix(".pdf"), dpi=450, bbox_inches="tight", pad_inches=0.08
            )
    finally:
        plt.close(fig)
    if legacy_path is not None and legacy_path != path:
        legacy_path.unlink(missing_ok=True)
        if pdf:
            legacy_path.with_suffix(".pdf").unlink(missing_ok=True)
