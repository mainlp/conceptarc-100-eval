"""Regenerable analysis figures and self-contained support-grid galleries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from arc_cd.repr_analysis.figure_style import ML_CONFERENCE_STYLE, save_figure
from arc_cd.repr_analysis.population import population_label

matplotlib.use("Agg")

LABELS = {
    "strict_win_probability": "Strict triplet win probability",
    "tie_adjusted_probability": "Triplet probability (half credit for ties)",
}


def plot_analysis(
    directory: Path,
    *,
    pdf: bool = False,
) -> None:
    """Regenerate visualizations with optional PDF exports of static figures.

    Args:
        directory (Path): Completed or staged sub-analysis directory.
        pdf (bool): Also export PDFs; PNGs are always written.
    """
    with matplotlib.rc_context(ML_CONFERENCE_STYLE):
        _plot_analysis(directory, pdf=pdf)


def _plot_analysis(
    directory: Path,
    *,
    pdf: bool,
) -> None:
    """Regenerate figures and galleries entirely from saved numerical artifacts.

    Args:
        directory (Path): A completed or staged sub-analysis directory.
        pdf (bool): Also export PDF figures.
    """
    config = json.loads((directory / "run_config.json").read_text())
    if config.get("label_mode") == "metadata":
        from arc_cd.repr_analysis.family_visualization import plot_metadata_analysis

        plot_metadata_analysis(directory)
        return
    if config["analysis"] in ("dimensionality_reduction", "metadata_explorer"):
        from arc_cd.repr_analysis.explorer import write_explorer

        write_explorer(directory)
        return
    axes = json.loads((directory / "axes.json").read_text())
    context = "\n" + population_label(config) if population_label(config) else ""
    if config["analysis"] in ("cosine_distance", "euclidean_distance"):
        context = f" · {config['analysis'].replace('_', ' ')}{context}"
    plots = directory / "figures"
    plots.mkdir(exist_ok=True)
    with np.load(directory / "metrics.npz", allow_pickle=False) as archive:
        macro = archive["macro"]
    for metric_index, metric in enumerate(axes["metrics"]):
        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        for p, position in enumerate(axes["positions"]):
            ax.plot(
                axes["layers"],
                macro[p, :, metric_index],
                label=position,
                marker="." if len(axes["layers"]) < 12 else None,
            )
        if metric == "tie_adjusted_probability":
            ax.axhline(0.5, color="gray", linestyle="--", label="Chance (0.5)")
        ax.set(
            xlabel="Transformer layer (zero-based)",
            ylabel=LABELS[metric],
            title=f"Macro-average over concepts{context}",
        )
        ax.grid(alpha=0.2)
        ax.legend()
        save_figure(fig, plots / f"{metric}.png", pdf=pdf)
