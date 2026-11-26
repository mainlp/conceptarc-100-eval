"""Plot model performance by ConceptARC concept for one dataset root."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from arc_cd.repr_analysis.population import CONCEPT_GROUPS
from arc_cd.utils import save_json, setup_logging
from scripts.plot_eval_dataset_comparison import (
    DEFAULT_SUMMARY_GLOB,
    discover_model_eval_folders,
    format_dataset_display_label,
    get_model_base_colors,
    resolve_dataset_root,
    shade_model_color,
)
from scripts.summarize_eval_seed_stats import (
    SeedSummary,
    discover_seed_summaries,
    group_seed_summaries,
    is_scalar_number,
    load_summary_json,
    safe_filename_fragment,
    summarize_values,
)

PREFERRED_MODEL_ORDER = (
    "gemma4_31b",
    "qwen35_27b",
    "qwen35_9b",
    "gpt_oss_120b",
    "gpt_oss_20b",
    "qwen3_32b",
    "qwen3_8b",
    "phi4_reasoning_plus",
)


@dataclass(frozen=True)
class ConceptMetricStats:
    """Metric statistics for one model and concept.

    Args:
        mean (float): Mean metric value across seed summaries.
        stdev (float): Seed-level metric standard deviation.
        n_seed_runs (int): Number of seed summaries contributing values.
    """

    mean: float
    stdev: float
    n_seed_runs: int


@dataclass(frozen=True)
class ModelConceptPerformance:
    """Concept-level performance for one model variant.

    Args:
        base_model_name (str): Base model config name.
        model_name (str): Display model name.
        reasoning_effort (str | None): Reasoning effort for the model variant.
        dataset_label (str): Dataset label inferred from summary filenames.
        input_folder (Path): Model eval folder.
        seeds (tuple[int, ...]): Contributing generation seeds.
        summary_paths (tuple[Path, ...]): Contributing evaluation summaries.
        overall_stats (ConceptMetricStats): Overall metric statistics.
        concept_stats (dict[str, ConceptMetricStats]): Metric stats keyed by
            concept name.
        group_stats (dict[str, ConceptMetricStats]): Equal-weight concept group
            means summarized across seeds, for groups present in every seed.
    """

    base_model_name: str
    model_name: str
    reasoning_effort: str | None
    dataset_label: str
    input_folder: Path
    seeds: tuple[int, ...]
    summary_paths: tuple[Path, ...]
    overall_stats: ConceptMetricStats
    concept_stats: dict[str, ConceptMetricStats]
    group_stats: dict[str, ConceptMetricStats]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped model-performance bars by concept for one evaluation "
            "dataset root."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help=(
            "Dataset output root with layout "
            "{dataset_root}/{model_name}/eval/seed_*/*_eval_summary.json."
        ),
    )
    parser.add_argument(
        "--dataset_label",
        default=None,
        help="Optional display label override for output naming/logging.",
    )
    parser.add_argument(
        "--summary_glob",
        default=DEFAULT_SUMMARY_GLOB,
        help="Glob used inside each seed_* folder to find summary JSON files.",
    )
    parser.add_argument(
        "--plot_metric",
        default="accuracy",
        help="Per-concept scalar metric to compare.",
    )
    parser.add_argument(
        "--sort_models",
        choices=["preferred", "overall_desc", "overall_asc", "name"],
        default="preferred",
        help=(
            "Ordering for model bars and legend entries. Overall modes use the "
            "selected overall metric averaged across available seeds. "
            "Preferred preserves the existing model order."
        ),
    )
    parser.add_argument(
        "--sort_concepts",
        choices=["average_desc", "average_asc", "name"],
        default="average_desc",
        help=(
            "Ordering for concept groups on the x-axis. Average modes use the "
            "mean metric value across all plotted models."
        ),
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help=(
            "Output PNG path. Defaults to "
            "{dataset_root}/concept_model_{metric}_comparison.png."
        ),
    )
    parser.add_argument(
        "--summary_output_path",
        default=None,
        help=(
            "Output JSON path. Defaults to the resolved plot output path with "
            "a .json suffix."
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


def get_summary_concept_values(
    summary: SeedSummary,
    metric_name: str,
) -> dict[str, float]:
    """Return per-concept metric values for one seed summary.

    Args:
        summary (SeedSummary): One seed evaluation summary.
        metric_name (str): Per-concept scalar metric name.

    Returns:
        dict[str, float]: Metric values keyed by concept.

    Raises:
        ValueError: If ``per_concept`` is missing or a metric is malformed.
    """
    summary_json = load_summary_json(summary.summary_path)
    per_concept = summary_json.get("per_concept")
    if not isinstance(per_concept, dict):
        raise ValueError(
            f"Missing object field 'per_concept' in {summary.summary_path}"
        )

    values: dict[str, float] = {}
    for concept_name, concept_metrics in per_concept.items():
        if not isinstance(concept_metrics, dict):
            raise ValueError(
                f"Malformed per_concept metrics for concept {concept_name!r} "
                f"in {summary.summary_path}."
            )
        metric_value = concept_metrics.get(metric_name)
        if not is_scalar_number(metric_value):
            raise ValueError(
                f"Missing numeric per_concept metric {metric_name!r} for "
                f"concept {concept_name!r} in {summary.summary_path}."
            )
        values[str(concept_name)] = float(metric_value)

    return values


def summarize_metric_values(values: list[float]) -> ConceptMetricStats:
    """Summarize metric values contributed by seed runs.

    Args:
        values (list[float]): Metric values to summarize.

    Returns:
        ConceptMetricStats: Mean, sample standard deviation, and run count.
    """
    stats = summarize_values(values)
    return ConceptMetricStats(
        mean=float(stats["mean"]),
        stdev=float(stats["stdev"]),
        n_seed_runs=int(stats["n"]),
    )


def aggregate_model_concept_performance(
    summaries: list[SeedSummary],
    input_folder: str | Path,
    metric_name: str,
) -> ModelConceptPerformance:
    """Aggregate per-concept metric values across seed summaries.

    Args:
        summaries (list[SeedSummary]): Seed summaries for one model variant.
        input_folder (str | Path): Source model eval folder.
        metric_name (str): Per-concept scalar metric name.

    Returns:
        ModelConceptPerformance: Concept-level performance for one model.

    Raises:
        ValueError: If summaries are empty or span multiple model variants.
    """
    if not summaries:
        raise ValueError("Cannot aggregate an empty summary list.")

    variant_keys = {
        (
            summary.base_model_name,
            summary.dataset_label,
            summary.model_name,
            summary.reasoning_effort,
        )
        for summary in summaries
    }
    if len(variant_keys) != 1:
        raise ValueError("Cannot aggregate concept metrics across model variants.")

    sorted_summaries = sorted(summaries, key=lambda summary: summary.seed)
    overall_values: list[float] = []
    values_by_concept: dict[str, list[float]] = {}
    group_means: dict[str, list[float]] = {group: [] for group in CONCEPT_GROUPS}
    for summary in sorted_summaries:
        overall_value = summary.overall.get(metric_name)
        if not is_scalar_number(overall_value):
            raise ValueError(
                f"Missing numeric overall metric {metric_name!r} in "
                f"{summary.summary_path}."
            )
        overall_values.append(float(overall_value))
        per_concept = get_summary_concept_values(summary, metric_name)
        for concept_name, metric_value in per_concept.items():
            values_by_concept.setdefault(concept_name, []).append(metric_value)
        for group, concepts in CONCEPT_GROUPS.items():
            if all(concept in per_concept for concept in concepts):
                group_means[group].append(mean(per_concept[name] for name in concepts))

    concept_stats = {
        concept_name: summarize_metric_values(values)
        for concept_name, values in sorted(values_by_concept.items())
    }
    first_summary = sorted_summaries[0]

    return ModelConceptPerformance(
        base_model_name=first_summary.base_model_name,
        model_name=first_summary.model_name,
        reasoning_effort=first_summary.reasoning_effort,
        dataset_label=first_summary.dataset_label,
        input_folder=Path(input_folder).expanduser().resolve(),
        seeds=tuple(summary.seed for summary in sorted_summaries),
        summary_paths=tuple(summary.summary_path for summary in sorted_summaries),
        overall_stats=summarize_metric_values(overall_values),
        concept_stats=concept_stats,
        group_stats={
            group: summarize_metric_values(values)
            for group, values in group_means.items()
            if len(values) == len(summaries)
        },
    )


def load_concept_performances(
    dataset_root: str | Path,
    metric_name: str,
    summary_glob: str = DEFAULT_SUMMARY_GLOB,
) -> list[ModelConceptPerformance]:
    """Load concept-level model performances from one dataset root.

    Args:
        dataset_root (str | Path): Dataset output root.
        metric_name (str): Per-concept scalar metric name.
        summary_glob (str): Summary filename glob searched in each seed folder.

    Returns:
        list[ModelConceptPerformance]: Performances sorted by model name.

    Raises:
        ValueError: If no concept performances are found or dataset labels vary.
    """
    root_path = resolve_dataset_root(dataset_root)
    performances: list[ModelConceptPerformance] = []
    for eval_folder in discover_model_eval_folders(root_path, summary_glob):
        summaries = discover_seed_summaries(
            input_folder=eval_folder,
            summary_glob=summary_glob,
        )
        for model_summaries in group_seed_summaries(summaries).values():
            performances.append(
                aggregate_model_concept_performance(
                    summaries=model_summaries,
                    input_folder=eval_folder,
                    metric_name=metric_name,
                )
            )

    if not performances:
        raise ValueError(f"No concept performance summaries found under {root_path}.")

    dataset_labels = {performance.dataset_label for performance in performances}
    if len(dataset_labels) != 1:
        raise ValueError(
            f"Expected one dataset label under {root_path}, found: "
            f"{', '.join(sorted(dataset_labels))}."
        )

    return sorted(performances, key=lambda performance: performance.model_name)


def get_common_concept_names(
    performances: list[ModelConceptPerformance],
) -> list[str]:
    """Return concepts represented for every model.

    Args:
        performances (list[ModelConceptPerformance]): Model performances.

    Returns:
        list[str]: Sorted common concept names.

    Raises:
        ValueError: If no performances are provided.
    """
    if not performances:
        raise ValueError("At least one model performance is required.")

    common_concepts = set(performances[0].concept_stats)
    for performance in performances[1:]:
        common_concepts &= set(performance.concept_stats)

    return sorted(common_concepts)


def sort_models_by_average_metric(
    performances: list[ModelConceptPerformance],
    concept_names: list[str],
    sort_mode: str = "preferred",
) -> list[ModelConceptPerformance]:
    """Order models by preference, overall seed-mean performance, or name.

    Args:
        performances (list[ModelConceptPerformance]): Model performances.
        concept_names (list[str]): Concepts included in the plot.
        sort_mode (str): ``preferred``, ``overall_desc``, ``overall_asc``, or
            ``name``. Preferred ordering uses concept means as a secondary key.

    Returns:
        list[ModelConceptPerformance]: Sorted model performances.

    Raises:
        ValueError: If ``sort_mode`` is unsupported.
    """
    if sort_mode == "name":
        return sorted(performances, key=lambda performance: performance.model_name)
    if sort_mode in {"overall_desc", "overall_asc"}:
        direction = -1 if sort_mode == "overall_desc" else 1
        return sorted(
            performances,
            key=lambda performance: (
                direction * performance.overall_stats.mean,
                performance.model_name,
            ),
        )
    if sort_mode != "preferred":
        raise ValueError(f"Unsupported model sort mode: {sort_mode}")

    def average_metric(performance: ModelConceptPerformance) -> float:
        means = [
            performance.concept_stats[concept_name].mean
            for concept_name in concept_names
        ]
        return sum(means) / len(means)

    preferred_order = {
        model_name: index for index, model_name in enumerate(PREFERRED_MODEL_ORDER)
    }
    return sorted(
        performances,
        key=lambda performance: (
            preferred_order.get(performance.base_model_name, len(preferred_order)),
            -average_metric(performance),
            performance.model_name,
        ),
    )


def sort_concept_names(
    performances: list[ModelConceptPerformance],
    concept_names: list[str],
    sort_mode: str,
) -> list[str]:
    """Sort concept names for plotting.

    Args:
        performances (list[ModelConceptPerformance]): Model performances.
        concept_names (list[str]): Common concept names.
        sort_mode (str): One of ``average_desc``, ``average_asc``, or ``name``.

    Returns:
        list[str]: Sorted concept names.

    Raises:
        ValueError: If ``sort_mode`` is unsupported.
    """
    if sort_mode == "name":
        return sorted(concept_names)

    if sort_mode not in {"average_desc", "average_asc"}:
        raise ValueError(f"Unsupported concept sort mode: {sort_mode}")

    reverse = sort_mode == "average_desc"

    def average_metric(concept_name: str) -> float:
        means = [
            performance.concept_stats[concept_name].mean for performance in performances
        ]
        return sum(means) / len(means)

    return sorted(
        concept_names,
        key=lambda concept_name: (
            -average_metric(concept_name) if reverse else average_metric(concept_name),
            concept_name,
        ),
    )


def get_default_output_path(
    dataset_root: str | Path,
    metric_name: str,
) -> Path:
    """Build the default concept-performance plot output path.

    Args:
        dataset_root (str | Path): Dataset output root.
        metric_name (str): Per-concept scalar metric name.

    Returns:
        Path: Default output path.
    """
    root_path = Path(dataset_root).expanduser().resolve()
    return root_path / (
        f"concept_model_{safe_filename_fragment(metric_name)}_comparison.png"
    )


def get_default_summary_output_path(plot_output_path: str | Path) -> Path:
    """Build the default JSON path beside the concept-performance plot.

    Args:
        plot_output_path (str | Path): Resolved or user-provided plot path.

    Returns:
        Path: Plot path with a ``.json`` suffix.
    """
    return Path(plot_output_path).expanduser().resolve().with_suffix(".json")


def metric_stats_to_dict(stats: ConceptMetricStats) -> dict[str, float | int]:
    """Convert metric statistics to the persisted JSON representation.

    Args:
        stats (ConceptMetricStats): Metric statistics to serialize.

    Returns:
        dict[str, float | int]: Mean, sample standard deviation, and run count.
    """
    return {
        "mean": stats.mean,
        "stdev": stats.stdev,
        "n_seed_runs": stats.n_seed_runs,
    }


def build_concept_performance_summary(
    dataset_root: str | Path,
    performances: list[ModelConceptPerformance],
    concept_names: list[str],
    metric_name: str,
    summary_glob: str,
) -> dict[str, Any]:
    """Build a JSON-ready cross-seed concept-performance summary.

    Args:
        dataset_root (str | Path): Dataset root containing model outputs.
        performances (list[ModelConceptPerformance]): Aggregated model results.
        concept_names (list[str]): Concepts included in the plot and summary.
        metric_name (str): Selected overall and per-concept metric.
        summary_glob (str): Input summary glob used for discovery.

    Returns:
        dict[str, Any]: JSON-ready summary payload.

    Raises:
        ValueError: If no model performances are provided or dataset labels
            differ.
    """
    if not performances:
        raise ValueError("At least one model performance is required.")

    dataset_labels = {performance.dataset_label for performance in performances}
    if len(dataset_labels) != 1:
        raise ValueError("Cannot summarize multiple dataset labels together.")

    return {
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "dataset_label": performances[0].dataset_label,
        "metric_name": metric_name,
        "summary_glob": summary_glob,
        "plotted_concepts": list(concept_names),
        "models": [
            {
                "model_name": performance.model_name,
                "base_model_name": performance.base_model_name,
                "reasoning_effort": performance.reasoning_effort,
                "seeds": list(performance.seeds),
                "n_seed_runs": len(performance.seeds),
                "input_folder": str(performance.input_folder),
                "summary_paths": [
                    str(summary_path) for summary_path in performance.summary_paths
                ],
                "overall": metric_stats_to_dict(performance.overall_stats),
                "per_concept_group": {
                    group: metric_stats_to_dict(stats)
                    for group, stats in performance.group_stats.items()
                },
                "per_concept": {
                    concept_name: metric_stats_to_dict(
                        performance.concept_stats[concept_name]
                    )
                    for concept_name in concept_names
                },
            }
            for performance in performances
        ],
    }


def write_concept_performance_summary(
    dataset_root: str | Path,
    performances: list[ModelConceptPerformance],
    concept_names: list[str],
    metric_name: str,
    summary_glob: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write a cross-seed concept-performance JSON summary.

    Args:
        dataset_root (str | Path): Dataset root containing model outputs.
        performances (list[ModelConceptPerformance]): Aggregated model results.
        concept_names (list[str]): Concepts included in the plot and summary.
        metric_name (str): Selected overall and per-concept metric.
        summary_glob (str): Input summary glob used for discovery.
        output_path (str | Path): Destination JSON path.

    Returns:
        dict[str, Any]: Persisted summary payload.
    """
    summary = build_concept_performance_summary(
        dataset_root=dataset_root,
        performances=performances,
        concept_names=concept_names,
        metric_name=metric_name,
        summary_glob=summary_glob,
    )
    save_json(summary, output_path)
    return summary


def warn_if_model_seed_sets_differ(
    performances: list[ModelConceptPerformance],
) -> None:
    """Warn when compared model variants use different seed sets.

    Args:
        performances (list[ModelConceptPerformance]): Aggregated model results.
    """
    if len({performance.seeds for performance in performances}) <= 1:
        return

    seed_details = "; ".join(
        f"{performance.model_name}={list(performance.seeds)}"
        for performance in performances
    )
    logging.warning(
        "Models use different seed sets; aggregating all available seeds: %s",
        seed_details,
    )


def get_model_legend_ncols(n_models: int) -> int:
    """Return one legend column per model for a single-row legend.

    Args:
        n_models (int): Number of plotted models.

    Returns:
        int: Legend column count.
    """
    return n_models


def get_concept_arc_dataset_shade_index(dataset_label: str) -> int | None:
    """Return the two-dataset comparison shade index for a ConceptARC label.

    Args:
        dataset_label (str): Dataset label inferred from summary filenames.

    Returns:
        int | None: ``0`` for ConceptARC 10/plain ConceptARC, ``1`` for other
        ConceptARC variants, or ``None`` for non-ConceptARC datasets.
    """
    if dataset_label in {"concept_arc", "concept_arc_10"}:
        return 0
    if dataset_label.startswith("concept_arc_"):
        return 1
    return None


def get_model_colors_for_dataset(
    model_names: list[str],
    dataset_label: str,
    plt: Any,
) -> list[Any]:
    """Return model colors using the dataset-comparison shade when available.

    Args:
        model_names (list[str]): Model names in plotted order.
        dataset_label (str): Dataset label inferred from summary filenames.
        plt (Any): Matplotlib pyplot module.

    Returns:
        list[Any]: Matplotlib-compatible model colors.
    """
    base_colors = get_model_base_colors(model_names, plt)
    shade_index = get_concept_arc_dataset_shade_index(dataset_label)
    if shade_index is None:
        return base_colors

    return [
        shade_model_color(
            base_color,
            dataset_index=shade_index,
            n_datasets=2,
        )
        for base_color in base_colors
    ]


def plot_concept_performance(
    performances: list[ModelConceptPerformance],
    concept_names: list[str],
    metric_name: str,
    output_path: str | Path,
    dpi: int,
    dataset_display_label: str | None = None,
    dataset_label: str | None = None,
) -> None:
    """Write a subplot grid of concept performance by model.

    Args:
        performances (list[ModelConceptPerformance]): Model performances.
        concept_names (list[str]): Concepts in plotted order.
        metric_name (str): Per-concept scalar metric name.
        output_path (str | Path): Output PNG path.
        dpi (int): Output plot DPI.
        dataset_display_label (str | None): Optional dataset label for logging.
        dataset_label (str | None): Dataset label used to choose ConceptARC
            comparison-shade colors.

    Raises:
        ValueError: If there are no concepts or no models.
    """
    if not performances:
        raise ValueError("At least one model performance is required.")
    if not concept_names:
        raise ValueError("At least one concept is required for plotting.")

    plot_path = Path(output_path).expanduser().resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = [performance.model_name for performance in performances]
    color_dataset_label = dataset_label or performances[0].dataset_label
    colors = get_model_colors_for_dataset(model_names, color_dataset_label, plt)

    metric_bounds = [
        bound
        for performance in performances
        for concept_name in concept_names
        for bound in (
            performance.concept_stats[concept_name].mean
            - performance.concept_stats[concept_name].stdev,
            performance.concept_stats[concept_name].mean
            + performance.concept_stats[concept_name].stdev,
        )
    ]
    lower_bound = min(metric_bounds)
    upper_bound = max(metric_bounds)
    label_offset = max(0.0125, 0.035 * max(abs(upper_bound), 1.0))
    metric_display_label = metric_name.replace("_", " ").title()
    title = f"{metric_display_label} by Concept"
    if dataset_display_label:
        title = f"{dataset_display_label}: {title}"

    if 0.0 <= lower_bound and upper_bound <= 1.0:
        y_limits = (0.0, max(1.0, upper_bound + 2.5 * label_offset))
    else:
        y_limits = (min(0.0, lower_bound), upper_bound + 2.5 * label_offset)

    n_columns = min(4, len(concept_names))
    n_rows = math.ceil(len(concept_names) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.3 * n_columns, 3.6 * n_rows),
        facecolor="white",
        squeeze=False,
        sharey=True,
    )
    model_positions = list(range(len(performances)))
    for concept_index, concept_name in enumerate(concept_names):
        row, column = divmod(concept_index, n_columns)
        ax = axes[row][column]
        for model_index, (performance, color) in enumerate(
            zip(performances, colors, strict=True)
        ):
            stats = performance.concept_stats[concept_name]
            ax.bar(
                model_index,
                stats.mean,
                width=0.7,
                yerr=stats.stdev,
                capsize=4,
                color=color,
                edgecolor="#2f2f2f",
                linewidth=0.7,
                label=performance.model_name,
                error_kw={
                    "elinewidth": 1.3,
                    "ecolor": "#303030",
                    "capthick": 1.3,
                },
            )

        ax.set_title(concept_name)
        ax.set_xticks(model_positions)
        ax.set_xticklabels(model_names, rotation=28, ha="right")
        ax.set_ylim(*y_limits)
        ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if column == 0:
            ax.set_ylabel(metric_display_label)
        if row == n_rows - 1:
            ax.set_xlabel("Model")

    for concept_index in range(len(concept_names), n_rows * n_columns):
        row, column = divmod(concept_index, n_columns)
        axes[row][column].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        ncols=get_model_legend_ncols(len(performances)),
    )
    fig.suptitle(title, y=1.01)

    if dataset_display_label:
        logging.info(
            "Plotting %s concept performance for %d models.",
            dataset_display_label,
            len(performances),
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run concept-performance plotting."""
    args = parse_arguments()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    performances = load_concept_performances(
        dataset_root=args.dataset_root,
        metric_name=args.plot_metric,
        summary_glob=args.summary_glob,
    )
    common_concepts = get_common_concept_names(performances)
    if not common_concepts:
        raise ValueError("No concepts are present for every model.")

    concept_names = sort_concept_names(
        performances=performances,
        concept_names=common_concepts,
        sort_mode=args.sort_concepts,
    )
    sorted_performances = sort_models_by_average_metric(
        performances,
        concept_names,
        sort_mode=args.sort_models,
    )
    inferred_dataset_label = sorted_performances[0].dataset_label
    dataset_display_label = (
        args.dataset_label
        if args.dataset_label is not None
        else format_dataset_display_label(inferred_dataset_label)
    )
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else get_default_output_path(args.dataset_root, args.plot_metric)
    )
    summary_output_path = (
        Path(args.summary_output_path).expanduser().resolve()
        if args.summary_output_path is not None
        else get_default_summary_output_path(output_path)
    )

    warn_if_model_seed_sets_differ(sorted_performances)
    write_concept_performance_summary(
        dataset_root=args.dataset_root,
        performances=sorted_performances,
        concept_names=concept_names,
        metric_name=args.plot_metric,
        summary_glob=args.summary_glob,
        output_path=summary_output_path,
    )
    logging.info("Wrote concept performance summary to %s", summary_output_path)

    plot_concept_performance(
        performances=sorted_performances,
        concept_names=concept_names,
        metric_name=args.plot_metric,
        output_path=output_path,
        dpi=args.dpi,
        dataset_display_label=dataset_display_label,
        dataset_label=inferred_dataset_label,
    )
    logging.info("Wrote concept performance plot to %s", output_path)


if __name__ == "__main__":
    main()
