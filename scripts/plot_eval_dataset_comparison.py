"""Plot model performance across multiple evaluation dataset roots."""

from __future__ import annotations

import argparse
import colorsys
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc_cd.utils import setup_logging
from scripts.summarize_eval_seed_stats import (
    aggregate_model_summaries,
    discover_seed_summaries,
    group_seed_summaries,
    is_scalar_number,
    safe_filename_fragment,
)

DEFAULT_SUMMARY_GLOB = "*_eval_summary.json"
SEED_DIR_NAME_PATTERN = re.compile(r"^seed_\d+$")


@dataclass(frozen=True)
class DatasetComparison:
    """Aggregated model metrics for one dataset root.

    Args:
        dataset_root (Path): Dataset output root containing model subfolders.
        dataset_label (str): Dataset label inferred from summary filenames.
        dataset_display_label (str): Label used in the plot legend.
        aggregates_by_model (dict[str, dict[str, Any]]): Aggregate summaries
            keyed by display model name.
    """

    dataset_root: Path
    dataset_label: str
    dataset_display_label: str
    aggregates_by_model: dict[str, dict[str, Any]]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped model-performance bars across two or more eval "
            "dataset roots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_roots",
        nargs="+",
        required=True,
        help=(
            "Dataset output roots with layout "
            "{dataset_root}/{model_name}/eval/seed_*/*_eval_summary.json."
        ),
    )
    parser.add_argument(
        "--dataset_labels",
        nargs="+",
        default=None,
        help=(
            "Optional display labels for the dataset roots. When provided, "
            "the number of labels must match --dataset_roots."
        ),
    )
    parser.add_argument(
        "--summary_glob",
        default=DEFAULT_SUMMARY_GLOB,
        help="Glob used inside each seed_* folder to find summary JSON files.",
    )
    parser.add_argument(
        "--plot_metric",
        default="accuracy",
        help="Scalar metric to compare overall or within each concept.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for all generated artifacts. Defaults to a dedicated "
            "comparison folder in the first dataset root's parent directory."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output plot DPI.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG).",
    )

    return parser.parse_args()


def resolve_dataset_root(dataset_root: str | Path) -> Path:
    """Resolve and validate one dataset root path.

    Args:
        dataset_root (str | Path): Dataset root from the CLI.

    Returns:
        Path: Resolved dataset root path.

    Raises:
        FileNotFoundError: If the dataset root does not exist.
        NotADirectoryError: If the dataset root is not a directory.
    """
    root_path = Path(dataset_root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root_path}")

    return root_path


def resolve_dataset_roots(dataset_roots: Sequence[str | Path]) -> list[Path]:
    """Resolve and validate dataset root paths.

    Args:
        dataset_roots (Sequence[str | Path]): Dataset roots from the CLI.

    Returns:
        list[Path]: Resolved dataset root paths.

    Raises:
        ValueError: If fewer than two roots are provided.
    """
    if len(dataset_roots) < 2:
        raise ValueError("At least two dataset roots are required.")

    resolved_roots = [
        resolve_dataset_root(dataset_root) for dataset_root in dataset_roots
    ]
    return resolved_roots


def has_seed_summary(eval_folder: Path, summary_glob: str) -> bool:
    """Return whether an eval folder contains at least one seed summary.

    Args:
        eval_folder (Path): Candidate model eval folder.
        summary_glob (str): Summary filename glob searched in each seed folder.

    Returns:
        bool: Whether the eval folder contains a matching seed summary.
    """
    for seed_dir in eval_folder.iterdir():
        if (
            not seed_dir.is_dir()
            or SEED_DIR_NAME_PATTERN.fullmatch(seed_dir.name) is None
        ):
            continue
        if any(seed_dir.glob(summary_glob)):
            return True

    return False


def discover_model_eval_folders(dataset_root: Path, summary_glob: str) -> list[Path]:
    """Discover model eval folders under one dataset root.

    Args:
        dataset_root (Path): Dataset output root.
        summary_glob (str): Summary filename glob searched in each seed folder.

    Returns:
        list[Path]: Sorted model eval folders with at least one matching summary.
    """
    eval_folders: list[Path] = []
    for model_dir in sorted(dataset_root.iterdir()):
        eval_folder = model_dir / "eval"
        if not model_dir.is_dir() or not eval_folder.is_dir():
            continue
        if has_seed_summary(eval_folder, summary_glob):
            eval_folders.append(eval_folder)
        else:
            logging.warning(
                "Skipping %s because it has no seed summaries.", eval_folder
            )

    return eval_folders


def format_dataset_display_label(dataset_label: str) -> str:
    """Format a dataset label for a plot legend.

    Args:
        dataset_label (str): Dataset label inferred from summary filenames.

    Returns:
        str: Human-readable dataset label.
    """
    if dataset_label in {"concept_arc", "concept_arc_10"}:
        return "ConceptARC"
    if dataset_label.startswith("concept_arc_"):
        return "ConceptARC-" + dataset_label.removeprefix("concept_arc_").replace(
            "_", "-"
        )
    return dataset_label.replace("_", " ").title()


def load_dataset_comparison(
    dataset_root: str | Path,
    summary_glob: str = DEFAULT_SUMMARY_GLOB,
    dataset_display_label: str | None = None,
) -> DatasetComparison:
    """Aggregate seed summaries under one dataset root.

    Args:
        dataset_root (str | Path): Dataset output root.
        summary_glob (str): Summary filename glob searched in each seed folder.
        dataset_display_label (str | None): Optional legend label override.

    Returns:
        DatasetComparison: Aggregates for the dataset root.

    Raises:
        ValueError: If no summaries are found, multiple dataset labels are found,
            or a model variant is duplicated within the dataset root.
    """
    root_path = resolve_dataset_root(dataset_root)
    aggregates: list[dict[str, Any]] = []
    for eval_folder in discover_model_eval_folders(root_path, summary_glob):
        summaries = discover_seed_summaries(
            input_folder=eval_folder,
            summary_glob=summary_glob,
        )
        for model_summaries in group_seed_summaries(summaries).values():
            aggregates.append(
                aggregate_model_summaries(
                    summaries=model_summaries,
                    input_folder=eval_folder,
                )
            )

    if not aggregates:
        raise ValueError(f"No evaluation summaries found under {root_path}.")

    dataset_labels = sorted(
        {str(aggregate["dataset_label"]) for aggregate in aggregates}
    )
    if len(dataset_labels) != 1:
        raise ValueError(
            f"Expected one dataset label under {root_path}, found: "
            f"{', '.join(dataset_labels)}."
        )

    aggregates_by_model: dict[str, dict[str, Any]] = {}
    for aggregate in aggregates:
        model_name = str(aggregate["model_name"])
        if model_name in aggregates_by_model:
            raise ValueError(
                f"Found duplicate aggregate for model {model_name!r} under "
                f"{root_path}."
            )
        aggregates_by_model[model_name] = aggregate

    dataset_label = dataset_labels[0]
    return DatasetComparison(
        dataset_root=root_path,
        dataset_label=dataset_label,
        dataset_display_label=(
            dataset_display_label
            if dataset_display_label is not None
            else format_dataset_display_label(dataset_label)
        ),
        aggregates_by_model=aggregates_by_model,
    )


def load_dataset_comparisons(
    dataset_roots: Sequence[str | Path],
    dataset_display_labels: Sequence[str] | None = None,
    summary_glob: str = DEFAULT_SUMMARY_GLOB,
) -> list[DatasetComparison]:
    """Load aggregate comparisons for multiple dataset roots.

    Args:
        dataset_roots (Sequence[str | Path]): Dataset output roots.
        dataset_display_labels (Sequence[str] | None): Optional display labels.
        summary_glob (str): Summary filename glob searched in each seed folder.

    Returns:
        list[DatasetComparison]: Aggregated dataset comparisons.

    Raises:
        ValueError: If the number of display labels does not match the roots.
    """
    resolved_roots = resolve_dataset_roots(dataset_roots)
    if dataset_display_labels is not None and (
        len(dataset_display_labels) != len(resolved_roots)
    ):
        raise ValueError(
            "The number of --dataset_labels values must match --dataset_roots."
        )

    comparisons: list[DatasetComparison] = []
    for index, dataset_root in enumerate(resolved_roots):
        display_label = (
            dataset_display_labels[index]
            if dataset_display_labels is not None
            else None
        )
        comparisons.append(
            load_dataset_comparison(
                dataset_root=dataset_root,
                summary_glob=summary_glob,
                dataset_display_label=display_label,
            )
        )

    return comparisons


def get_common_model_names(comparisons: list[DatasetComparison]) -> list[str]:
    """Return model variants that are present in every dataset comparison.

    Args:
        comparisons (list[DatasetComparison]): Dataset comparisons.

    Returns:
        list[str]: Sorted common model names.

    Raises:
        ValueError: If no comparisons are provided.
    """
    if not comparisons:
        raise ValueError("At least one dataset comparison is required.")

    common_model_names = set(comparisons[0].aggregates_by_model)
    for comparison in comparisons[1:]:
        common_model_names &= set(comparison.aggregates_by_model)

    return sorted(common_model_names)


def get_skipped_model_names_by_dataset(
    comparisons: list[DatasetComparison],
    common_model_names: list[str],
) -> dict[str, list[str]]:
    """Return unpaired model variants that will be skipped for each dataset.

    Args:
        comparisons (list[DatasetComparison]): Dataset comparisons.
        common_model_names (list[str]): Model variants present in all datasets.

    Returns:
        dict[str, list[str]]: Skipped model names keyed by dataset label.
    """
    common_model_set = set(common_model_names)
    return {
        comparison.dataset_label: sorted(
            set(comparison.aggregates_by_model) - common_model_set
        )
        for comparison in comparisons
    }


def get_metric_mean_and_stdev(
    aggregate: dict[str, Any],
    metric_name: str,
) -> tuple[float, float]:
    """Read the mean and stdev for a scalar aggregate metric.

    Args:
        aggregate (dict[str, Any]): Aggregate model summary.
        metric_name (str): Overall scalar metric name.

    Returns:
        tuple[float, float]: Metric mean and seed-level stdev.

    Raises:
        ValueError: If the metric is missing or malformed.
    """
    metric_stats = aggregate["overall_scalar_stats"].get(metric_name)
    if not isinstance(metric_stats, dict):
        raise ValueError(
            f"Metric {metric_name!r} is missing for model "
            f"{aggregate['model_name']!r}."
        )

    mean_value = metric_stats.get("mean")
    stdev_value = metric_stats.get("stdev")
    if not is_scalar_number(mean_value) or not is_scalar_number(stdev_value):
        raise ValueError(
            f"Metric {metric_name!r} has malformed mean/stdev values for model "
            f"{aggregate['model_name']!r}."
        )

    return float(mean_value), float(stdev_value)


def sort_model_names_by_average_metric(
    comparisons: list[DatasetComparison],
    model_names: list[str],
    metric_name: str,
) -> list[str]:
    """Sort models by family, then descending average metric.

    Args:
        comparisons (list[DatasetComparison]): Dataset comparisons.
        model_names (list[str]): Model names to sort.
        metric_name (str): Overall scalar metric name.

    Returns:
        list[str]: Sorted model names.
    """

    def sort_key(model_name: str) -> tuple[int, str, float, str]:
        normalized_name = model_name.lower()
        family_match = re.match(r"(.+?)_\d+(?:\.\d+)?b(?:\s|$)", normalized_name)
        family = family_match.group(1) if family_match else normalized_name
        preferred_family_order = {"qwen35": 0, "qwen3": 1}
        means = [
            get_metric_mean_and_stdev(
                comparison.aggregates_by_model[model_name],
                metric_name,
            )[0]
            for comparison in comparisons
        ]
        return (
            preferred_family_order.get(family, 2),
            family,
            -sum(means) / len(means),
            model_name,
        )

    return sorted(model_names, key=sort_key)


def get_model_base_colors(model_names: list[str], plt: Any) -> list[Any]:
    """Return one stable, distinct base color per model.

    Args:
        model_names (list[str]): Model names in plotted order.
        plt (Any): Matplotlib pyplot module.

    Returns:
        list[Any]: Matplotlib-compatible colors.
    """
    color_map_name = "tab10" if len(model_names) <= 10 else "tab20"
    color_map = plt.get_cmap(color_map_name)
    sorted_model_names = sorted(set(model_names))
    color_by_model = {
        model_name: color_map(index % color_map.N)
        for index, model_name in enumerate(sorted_model_names)
    }
    return [color_by_model[model_name] for model_name in model_names]


def get_dataset_lightness(n_datasets: int, dataset_index: int) -> float:
    """Return the lightness level for a dataset shade.

    Args:
        n_datasets (int): Number of datasets in the plot.
        dataset_index (int): Zero-based dataset index.

    Returns:
        float: HLS lightness value.
    """
    if n_datasets <= 1:
        return 0.56

    lightest = 0.72
    darkest = 0.42
    step = (lightest - darkest) / (n_datasets - 1)
    return lightest - step * dataset_index


def shade_model_color(
    base_color: Any,
    dataset_index: int,
    n_datasets: int,
) -> tuple[float, float, float]:
    """Return a dataset shade of a model's base color.

    Args:
        base_color (Any): Matplotlib-compatible base color.
        dataset_index (int): Zero-based dataset index.
        n_datasets (int): Number of datasets in the plot.

    Returns:
        tuple[float, float, float]: RGB color tuple.
    """
    import matplotlib.colors as mcolors

    red, green, blue = mcolors.to_rgb(base_color)
    hue, _, saturation = colorsys.rgb_to_hls(red, green, blue)
    lightness = get_dataset_lightness(n_datasets, dataset_index)
    return colorsys.hls_to_rgb(hue, lightness, saturation)


def get_dataset_legend_color(
    dataset_index: int,
    n_datasets: int,
) -> tuple[float, float, float]:
    """Return a neutral gray legend color for one dataset shade.

    Args:
        dataset_index (int): Zero-based dataset index.
        n_datasets (int): Number of datasets in the plot.

    Returns:
        tuple[float, float, float]: RGB gray color tuple.
    """
    lightness = get_dataset_lightness(n_datasets, dataset_index)
    return (lightness, lightness, lightness)


def plot_eval_dataset_comparison(
    comparisons: list[DatasetComparison],
    model_names: list[str],
    metric_name: str,
    output_path: str | Path,
    dpi: int,
) -> None:
    """Write a grouped bar plot comparing datasets for each model.

    Args:
        comparisons (list[DatasetComparison]): Dataset comparisons to plot.
        model_names (list[str]): Common model names in plotted order.
        metric_name (str): Overall scalar metric name.
        output_path (str | Path): Output PNG path.
        dpi (int): Output plot DPI.

    Raises:
        ValueError: If there are no models or fewer than two datasets.
    """
    if len(comparisons) < 2:
        raise ValueError("At least two dataset comparisons are required.")
    if not model_names:
        raise ValueError("At least one common model is required for plotting.")

    metric_values_by_dataset = [
        [
            get_metric_mean_and_stdev(
                comparison.aggregates_by_model[model_name],
                metric_name,
            )
            for model_name in model_names
        ]
        for comparison in comparisons
    ]
    all_lower_bounds = [
        mean_value - stdev_value
        for dataset_values in metric_values_by_dataset
        for mean_value, stdev_value in dataset_values
    ]
    all_upper_bounds = [
        mean_value + stdev_value
        for dataset_values in metric_values_by_dataset
        for mean_value, stdev_value in dataset_values
    ]
    lower_bound = min(all_lower_bounds)
    upper_bound = max(all_upper_bounds)
    label_offset = max(0.0125, 0.035 * max(abs(upper_bound), 1.0))

    plot_path = Path(output_path).expanduser().resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    n_datasets = len(comparisons)
    group_centers = list(range(len(model_names)))
    bar_width = min(0.28, 0.82 / n_datasets)
    fig_width = max(7.5, 1.15 * len(model_names) + 0.35 * n_datasets + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.9), facecolor="white")
    model_base_colors = get_model_base_colors(model_names, plt)

    for dataset_index, comparison in enumerate(comparisons):
        offset = (dataset_index - (n_datasets - 1) / 2.0) * bar_width
        positions = [center + offset for center in group_centers]
        means = [
            mean_value for mean_value, _ in metric_values_by_dataset[dataset_index]
        ]
        stdevs = [
            stdev_value for _, stdev_value in metric_values_by_dataset[dataset_index]
        ]
        bar_colors = [
            shade_model_color(
                base_color,
                dataset_index=dataset_index,
                n_datasets=n_datasets,
            )
            for base_color in model_base_colors
        ]
        ax.bar(
            positions,
            means,
            width=bar_width,
            yerr=stdevs,
            capsize=4,
            color=bar_colors,
            edgecolor="#2f2f2f",
            linewidth=0.7,
            error_kw={"elinewidth": 1.3, "ecolor": "#303030", "capthick": 1.3},
        )
        for position, mean_value, stdev_value in zip(
            positions,
            means,
            stdevs,
            strict=True,
        ):
            ax.text(
                position,
                mean_value + stdev_value + label_offset,
                f"{mean_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="semibold",
                rotation=90 if n_datasets > 3 else 0,
            )

    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        mpatches.Patch(
            facecolor=get_dataset_legend_color(dataset_index, n_datasets),
            edgecolor="#2f2f2f",
            linewidth=0.7,
            label=comparison.dataset_display_label,
        )
        for dataset_index, comparison in enumerate(comparisons)
    ]
    ax.legend(handles=legend_handles, frameon=False, ncols=min(n_datasets, 4))

    if 0.0 <= lower_bound and upper_bound <= 1.0:
        ax.set_ylim(0.0, max(1.0, upper_bound + 2.5 * label_offset))
    else:
        ax.set_ylim(
            bottom=min(0.0, lower_bound),
            top=upper_bound + 2.5 * label_offset,
        )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def log_unpaired_models(
    comparisons: list[DatasetComparison],
    common_model_names: list[str],
) -> None:
    """Log model variants excluded from the paired comparison.

    Args:
        comparisons (list[DatasetComparison]): Dataset comparisons.
        common_model_names (list[str]): Model variants present in all datasets.
    """
    common_model_set = set(common_model_names)
    all_model_names = sorted(
        {
            model_name
            for comparison in comparisons
            for model_name in comparison.aggregates_by_model
        }
    )
    unpaired_models = [
        model_name
        for model_name in all_model_names
        if model_name not in common_model_set
    ]
    if unpaired_models:
        logging.info(
            "Skipping model variants not present in every dataset: %s",
            ", ".join(unpaired_models),
        )

    for comparison in comparisons:
        missing_models = sorted(
            set(all_model_names) - set(comparison.aggregates_by_model)
        )
        if missing_models:
            logging.info(
                "%s is missing: %s",
                comparison.dataset_display_label,
                ", ".join(missing_models),
            )


def main() -> None:
    """Run dataset comparison plotting."""
    args = parse_arguments()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    comparisons = load_dataset_comparisons(
        dataset_roots=args.dataset_roots,
        dataset_display_labels=args.dataset_labels,
        summary_glob=args.summary_glob,
    )
    common_model_names = get_common_model_names(comparisons)
    if not common_model_names:
        raise ValueError("No model variants are present in every dataset root.")

    log_unpaired_models(comparisons, common_model_names)
    model_names = sort_model_names_by_average_metric(
        comparisons=comparisons,
        model_names=common_model_names,
        metric_name=args.plot_metric,
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (
            Path(args.dataset_roots[0]).expanduser().resolve().parent
            / "dataset_comparison"
        )
    )
    output_path = output_dir / (
        f"dataset_model_{safe_filename_fragment(args.plot_metric)}_comparison.png"
    )
    plot_eval_dataset_comparison(
        comparisons=comparisons,
        model_names=model_names,
        metric_name=args.plot_metric,
        output_path=output_path,
        dpi=args.dpi,
    )
    logging.info("Wrote dataset comparison plot to %s", output_path)


if __name__ == "__main__":
    main()
