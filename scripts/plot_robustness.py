"""Plot cumulative task robustness from model-level seed-statistics CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

MODEL_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#6A51A3",
    "#555555",
    "#8C564B",
    "#8A9A00",
)
MODEL_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h", "<", ">")


@dataclass(frozen=True)
class RobustnessCurve:
    """Cumulative task proportions for one model.

    Args:
        label (str): Model label for the legend.
        means (tuple[float, ...]): Mean proportions at thresholds 1 through N.
        stdevs (tuple[float, ...] | None): Standard deviations in proportion
            units, or ``None`` when unavailable.
        n_seed_runs (int | None): Number of contributing seeds, if known.
    """

    label: str
    means: tuple[float, ...]
    stdevs: tuple[float, ...] | None
    n_seed_runs: int | None


def load_robustness_csv(
    path: Path, n_queries: int = 10, label: str | None = None
) -> RobustnessCurve:
    """Read and validate cumulative robustness statistics for one query total.

    Args:
        path (Path): CSV produced by ``summarize_eval_seed_stats.py``.
        n_queries (int): Query total to select when a CSV contains multiple N.
        label (str | None): Legend label; defaults to the filename's model name.

    Returns:
        RobustnessCurve: Values ordered by increasing positive threshold.

    Raises:
        ValueError: If the CSV is incomplete, malformed, or uses exact-count bins.
    """
    if n_queries < 1:
        raise ValueError("n_queries must be positive.")
    required = {
        "n_queries",
        "min_n_correct",
        "n_seed_runs",
        "fraction_mean",
        "fraction_stdev",
    }
    values: dict[int, tuple[float, float, int]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path}: expected a cumulative query_distribution_seed_stats "
                "CSV with min_n_correct. Rerun summarize_eval_seed_stats.py "
                "to regenerate older exact-count tables."
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                if int(row["n_queries"]) != n_queries:
                    continue
                threshold = int(row["min_n_correct"])
                average = float(row["fraction_mean"])
                deviation = float(row["fraction_stdev"])
                n_seeds = int(row["n_seed_runs"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{row_number}: invalid numeric value."
                ) from exc
            if threshold in values or not 1 <= threshold <= n_queries:
                raise ValueError(
                    f"{path}:{row_number}: duplicate or invalid threshold {threshold}."
                )
            if (
                not math.isfinite(average)
                or not 0 <= average <= 1
                or not math.isfinite(deviation)
                or deviation < 0
                or n_seeds < 1
                or (n_seeds == 1 and deviation != 0)
            ):
                raise ValueError(
                    f"{path}:{row_number}: invalid proportion or seed stats."
                )
            values[threshold] = (average, deviation, n_seeds)

    if set(values) != set(range(1, n_queries + 1)):
        raise ValueError(
            f"{path}: expected every threshold from 1/{n_queries} to {n_queries}/{n_queries}."
        )
    ordered = [values[threshold] for threshold in range(1, n_queries + 1)]
    means, stdevs, seed_counts = zip(*ordered)
    if len(set(seed_counts)) != 1:
        raise ValueError(f"{path}: seed counts differ between thresholds.")
    if any(later > earlier + 1e-12 for earlier, later in zip(means, means[1:])):
        raise ValueError(f"{path}: cumulative proportions must not increase.")
    inferred_label = path.stem.split("_query_distribution_seed_stats_", 1)[-1]
    return RobustnessCurve(
        label=label if label is not None else inferred_label.replace("_", " "),
        means=means,
        stdevs=stdevs,
        n_seed_runs=seed_counts[0],
    )


def load_reference_json(path: Path, n_queries: int = 10) -> list[RobustnessCurve]:
    """Load externally supplied curves without inferring missing uncertainty.

    Args:
        path (Path): JSON list of objects with ``label``, ``mean_percent``, and
            optional ``sd_percentage_points`` arrays, ordered by thresholds
            1 through N. Seed counts are unknown for these reference curves.
        n_queries (int): Number of query thresholds in each array.

    Returns:
        list[RobustnessCurve]: Reference curves converted to proportion units.

    Raises:
        ValueError: If labels, arrays, or cumulative statistics are invalid.
    """
    if n_queries < 1:
        raise ValueError("n_queries must be positive.")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a nonempty list of reference curves.")
    curves: list[RobustnessCurve] = []
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("label"), str)
            or not record["label"].strip()
        ):
            raise ValueError(f"{path}: each reference needs a nonempty label.")
        arrays: dict[str, tuple[float, ...] | None] = {}
        for key in ("mean_percent", "sd_percentage_points"):
            values = record.get(key)
            if key == "sd_percentage_points" and values is None:
                arrays[key] = None
                continue
            if not isinstance(values, list) or len(values) != n_queries:
                raise ValueError(f"{path}: {key} must contain {n_queries} values.")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 100
                for value in values
            ):
                raise ValueError(f"{path}: {key} must contain numbers in [0, 100].")
            arrays[key] = tuple(value / 100 for value in values)
        means = arrays["mean_percent"]
        assert means is not None
        if any(later > earlier + 1e-12 for earlier, later in zip(means, means[1:])):
            raise ValueError(f"{path}: cumulative proportions must not increase.")
        curves.append(
            RobustnessCurve(
                record["label"], means, arrays["sd_percentage_points"], None
            )
        )
    return curves


def plot_robustness(
    curves: list[RobustnessCurve],
    output_path: Path,
    colors: list[str] | None = None,
    show_std: bool = False,
    figsize: tuple[float, float] = (12.0, 3.5),
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Save a model comparison as a vector PDF and a high-resolution PNG.

    Args:
        curves (list[RobustnessCurve]): Model curves in legend order.
        output_path (Path): Output stem or PDF/PNG path; both formats are saved.
        colors (list[str] | None): One color per model, or the default palette.
        show_std (bool): Shade mean plus/minus one standard deviation, clipped
            to [0, 1]. Bands are omitted for single-seed curves or missing SDs.
        figsize (tuple[float, float]): Figure width and height in inches.
        dpi (int): Resolution of the PNG preview.

    Returns:
        tuple[Path, Path]: PDF and PNG output paths.

    Raises:
        ValueError: If curves, colors, or output settings are incompatible.
    """
    if not curves or not curves[0].means:
        raise ValueError("Provide at least one nonempty model curve.")
    n_queries = len(curves[0].means)
    if any(
        len(curve.means) != n_queries
        or (curve.stdevs is not None and len(curve.stdevs) != n_queries)
        for curve in curves
    ):
        raise ValueError("All curves must have the same query thresholds.")
    if len({curve.label for curve in curves}) != len(curves):
        raise ValueError(
            "Model labels must be unique; use --labels to distinguish them."
        )
    if colors is not None and len(colors) != len(curves):
        raise ValueError("Provide exactly one color per model.")
    if dpi <= 0 or any(not math.isfinite(size) or size <= 0 for size in figsize):
        raise ValueError("Figure dimensions and DPI must be positive.")
    if output_path.suffix.lower() not in {"", ".pdf", ".png"}:
        raise ValueError("Use an output stem or a .pdf/.png output path.")

    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import is_color_like, to_hex
    from matplotlib.figure import Figure
    from matplotlib.ticker import FormatStrFormatter

    if colors is None:
        colors = (
            list(MODEL_COLORS[: len(curves)])
            if len(curves) <= len(MODEL_COLORS)
            else [
                to_hex(mpl.colormaps["turbo"](index / (len(curves) - 1)))
                for index in range(len(curves))
            ]
        )
    if any(not is_color_like(color) for color in colors):
        raise ValueError("Each color must be a valid Matplotlib color.")
    output_path = output_path.expanduser().resolve()
    pdf_path, png_path = output_path.with_suffix(".pdf"), output_path.with_suffix(
        ".png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with mpl.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 15,
            "font.weight": "normal",
            "text.color": "#222222",
            "axes.labelsize": 20,
            "axes.labelweight": "normal",
            "axes.labelcolor": "#222222",
            "axes.facecolor": "white",
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "xtick.labelcolor": "#222222",
            "ytick.labelcolor": "#222222",
            "legend.fontsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure = Figure(figsize=figsize, layout="constrained", facecolor="white")
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        thresholds = list(range(1, n_queries + 1))
        for index, (curve, color) in enumerate(zip(curves, colors, strict=True)):
            axes.plot(
                thresholds,
                curve.means,
                label=curve.label,
                color=color,
                linewidth=1.8,
                solid_capstyle="round",
                marker=MODEL_MARKERS[index % len(MODEL_MARKERS)],
                markersize=5.5,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.8,
                zorder=3,
            )
            if show_std and curve.stdevs is not None and curve.n_seed_runs != 1:
                lower = [
                    max(0.0, avg - sd) for avg, sd in zip(curve.means, curve.stdevs)
                ]
                upper = [
                    min(1.0, avg + sd) for avg, sd in zip(curve.means, curve.stdevs)
                ]
                axes.fill_between(thresholds, lower, upper, color=color, alpha=0.12)

        axes.set_xlabel("Minimum correct queries per task", labelpad=12)
        axes.set_ylabel("Proportion of tasks", labelpad=12)
        axes.set_xticks(thresholds, [f"{value}/{n_queries}" for value in thresholds])
        axes.set_yticks([value / 5 for value in range(6)])
        axes.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axes.set_xlim(0.7, n_queries + 0.3)
        axes.set_ylim(-0.02, 1.02)
        axes.set_axisbelow(True)
        axes.grid(axis="y", color="#CCD2DA", linestyle="--", linewidth=0.6, alpha=0.7)
        axes.spines[["top", "right"]].set_visible(False)
        for side in ("bottom", "left"):
            axes.spines[side].set_color("#333333")
            axes.spines[side].set_linewidth(0.8)
        axes.tick_params(direction="out", length=3.5, width=0.8, pad=6, color="#333333")
        axes.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            ncols=1,
            frameon=False,
            handlelength=2.4,
        )
        for path in (pdf_path, png_path):
            figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    return pdf_path, png_path


def main() -> None:
    """Read model CSVs and save their cumulative robustness comparison."""
    parser = argparse.ArgumentParser(
        description="Compare at-least-X/N task robustness across models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_csvs",
        type=Path,
        nargs="+",
        required=True,
        help="One query_distribution_seed_stats CSV per model, in plot order.",
    )
    parser.add_argument("--labels", nargs="+", help="Legend labels in input CSV order.")
    parser.add_argument(
        "--reference_json",
        type=Path,
        help=(
            "Append reference curves from a JSON list with label, mean_percent, "
            "and optional sd_percentage_points arrays for thresholds 1 through N."
        ),
    )
    parser.add_argument(
        "--colors", nargs="+", help="Optional colors in CSV then reference order."
    )
    parser.add_argument(
        "--n_queries", type=int, default=10, help="Query total to plot."
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("plots/robustness.pdf"),
        help="Output stem or PDF/PNG path; both formats are always saved.",
    )
    parser.add_argument(
        "--show_std",
        action="store_true",
        help="Shade plus/minus one standard deviation across seeds (not a CI).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(12.0, 3.5),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure dimensions in inches.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.input_csvs):
        parser.error("--labels must provide exactly one label per input CSV.")
    try:
        curves = [
            load_robustness_csv(
                path.expanduser(),
                args.n_queries,
                args.labels[index] if args.labels is not None else None,
            )
            for index, path in enumerate(args.input_csvs)
        ]
        if args.reference_json is not None:
            curves.extend(
                load_reference_json(args.reference_json.expanduser(), args.n_queries)
            )
        paths = plot_robustness(
            curves,
            args.output_path,
            args.colors,
            args.show_std,
            tuple(args.figsize),
            args.dpi,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
