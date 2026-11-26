"""Summarize ARC evaluation metrics across seed runs."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, TypeGuard, Union

from arc_cd.eval.artifact_names import infer_dataset_label
from arc_cd.utils import load_json, save_json, setup_logging

SEED_DIR_PATTERN = re.compile(r"^seed_(\d+)$")
REASONING_EFFORT_PATTERN = re.compile(r"_reasoning_effort_(low|medium|high)_")
GPT_OSS_MODEL_NAMES = {"gpt_oss_20b", "gpt_oss_120b"}
REASONING_EFFORT_ORDER = {"low": 0, "medium": 1, "high": 2}
GPT_OSS_REASONING_COLORS = {
    "gpt_oss_20b": {
        "low": "#9ecae1",
        "medium": "#4292c6",
        "high": "#2171b5",
    },
    "gpt_oss_120b": {
        "low": "#fdd0a2",
        "medium": "#f16913",
        "high": "#d94801",
    },
}
GPT_OSS_120B_LOW_MODEL_NAME = "gpt_oss_120b low"
GPT_OSS_120B_LOW_VIOLET_COLOR = "#6F4E8C"
OTHER_MODEL_COLORS = [
    "#2ca25f",
    "#756bb1",
    "#de2d26",
    "#636363",
    "#31a354",
    "#54278f",
    "#c51b8a",
    "#525252",
    "#8c6d31",
    "#006d2c",
]
PLOT_TITLE_FONT_SIZE = 16
PLOT_AXIS_LABEL_FONT_SIZE = 14
PLOT_TICK_FONT_SIZE = 12
PLOT_VALUE_FONT_SIZE = 11
NumberStat = Union[float, int]
CountTotals = dict[str, dict[str, float]]
CountStats = dict[str, dict[str, dict[str, NumberStat]]]
PlotRow = tuple[str, float, float, float]
TaskQueryTables = dict[str, list[dict[str, str | int | float]]]


@dataclass(frozen=True)
class SeedSummary:
    """Evaluation summary for one seed run.

    Args:
        base_model_name (str): Model name inferred from the input folder.
        dataset_label (str): Dataset/split label inferred from the summary
            filename.
        model_name (str): Display model name, including reasoning effort when
            applicable.
        reasoning_effort (str | None): GPT-OSS reasoning effort, if present.
        seed (int): Seed value parsed from the seed directory name.
        summary_path (Path): Path to the seed evaluation summary JSON.
        overall (dict[str, Any]): Overall metrics from the summary JSON.
        per_task (dict[str, Any]): Per-task metrics from the summary JSON.
    """

    base_model_name: str
    dataset_label: str
    model_name: str
    reasoning_effort: str | None
    seed: int
    summary_path: Path
    overall: dict[str, Any]
    per_task: dict[str, Any]


@dataclass(frozen=True)
class PlotModelSpec:
    """Configuration for one model/source bar in a comparison plot.

    Args:
        model_name (str): Internal model/source name used in aggregate data.
        display_label (str): Label shown on the plot.
        color (str): Matplotlib-compatible bar color.
    """

    model_name: str
    display_label: str
    color: str


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Summarize ARC eval summary JSON files across seed runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG).",
    )

    # Data configs
    parser.add_argument(
        "--input_folders",
        nargs="+",
        required=True,
        help=(
            "One or more eval folders containing seed_* subfolders, e.g. "
            "output/concept_arc_100/qwen3_32b/eval."
        ),
    )
    parser.add_argument(
        "--summary_glob",
        default="*_eval_summary.json",
        help="Glob used inside each seed_* folder to find one summary JSON.",
    )
    # Plot configs
    parser.add_argument(
        "--plot_metric",
        default="accuracy",
        help="Overall scalar metric to compare in the model plot.",
    )
    parser.add_argument(
        "--plot_format",
        choices=["png", "pdf"],
        default="png",
        help="Comparison plot output format. PDF produces vector output.",
    )
    parser.add_argument(
        "--no_title",
        action="store_true",
        help="Omit the title from comparison plots.",
    )
    parser.add_argument(
        "--plot_model",
        nargs=3,
        action="append",
        metavar=("MODEL_NAME", "DISPLAY_LABEL", "COLOR"),
        help=(
            "Select, order, label, and color one plot source. Repeat for each "
            "source. MODEL_NAME must match an aggregate model name, such as "
            "qwen35_27b or 'gpt_oss_120b high'."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output plot DPI.",
    )

    return parser.parse_args()


def get_global_summary_dir(input_folder: str | Path) -> Path:
    """Build the output directory for an input eval folder.

    Args:
        input_folder (str | Path): Eval folder containing ``seed_*`` subfolders.

    Returns:
        Path: ``global_summary`` directory under the input folder.
    """
    return Path(input_folder).expanduser().resolve() / "global_summary"


def infer_model_name(input_folder: Path) -> str:
    """Infer the model name from an eval folder path.

    Args:
        input_folder (Path): Evaluation folder path.

    Returns:
        str: Inferred model name.
    """
    if input_folder.name == "eval":
        return input_folder.parent.name
    return input_folder.name


def parse_seed_dir(seed_dir: Path) -> int | None:
    """Parse a seed value from a seed directory path.

    Args:
        seed_dir (Path): Candidate seed directory.

    Returns:
        int | None: Parsed seed, or ``None`` when the name is not ``seed_<int>``.
    """
    match = SEED_DIR_PATTERN.fullmatch(seed_dir.name)
    if match is None:
        return None
    return int(match.group(1))


def load_summary_json(path: Path) -> dict[str, Any]:
    """Load one evaluation summary JSON file.

    Args:
        path (Path): Summary JSON path.

    Returns:
        dict[str, Any]: Parsed summary dictionary.
    """
    summary = load_json(path)
    if not isinstance(summary, dict):
        raise ValueError(f"Summary is not a JSON object: {path}")
    return summary


def parse_reasoning_effort(path: Path) -> str | None:
    """Parse a GPT-OSS reasoning effort from a summary filename.

    Args:
        path (Path): Summary path.

    Returns:
        str | None: Reasoning effort, or ``None`` if no effort is present.
    """
    match = REASONING_EFFORT_PATTERN.search(path.name)
    if match is None:
        return None
    return match.group(1)


def make_model_name(
    base_model_name: str,
    reasoning_effort: str | None,
) -> str:
    """Build a plot/stats label for a model variant.

    Args:
        base_model_name (str): Base model name.
        reasoning_effort (str | None): Reasoning effort, if applicable.

    Returns:
        str: Display model name.
    """
    if reasoning_effort is None:
        return base_model_name
    return f"{base_model_name} {reasoning_effort}"


def get_summary_model_variant(
    base_model_name: str,
    summary_path: Path,
) -> tuple[str, str | None]:
    """Infer the display model name and reasoning effort for a summary path.

    Args:
        base_model_name (str): Base model name inferred from the eval folder.
        summary_path (Path): Summary JSON path.

    Returns:
        tuple[str, str | None]: Display model name and reasoning effort.

    Raises:
        ValueError: If a GPT-OSS summary lacks a reasoning-effort filename
            marker.
    """
    reasoning_effort = parse_reasoning_effort(summary_path)
    if base_model_name in GPT_OSS_MODEL_NAMES and reasoning_effort is None:
        raise ValueError(
            f"Expected reasoning_effort_low|medium|high in GPT-OSS summary "
            f"filename: {summary_path}"
        )

    return make_model_name(base_model_name, reasoning_effort), reasoning_effort


def find_seed_summary_paths(
    seed_dir: Path,
    summary_glob: str,
    base_model_name: str,
) -> list[Path]:
    """Find summary JSON files in a seed directory.

    Args:
        seed_dir (Path): Seed directory to search.
        summary_glob (str): Glob pattern for summary files.
        base_model_name (str): Base model name inferred from the eval folder.

    Returns:
        list[Path]: Summary paths, possibly empty.

    Raises:
        ValueError: If multiple non-GPT summaries match, or if one GPT seed has
            duplicate summaries for the same reasoning effort.
    """
    summary_paths = sorted(seed_dir.glob(summary_glob))
    if not summary_paths:
        return []

    seen_variants: set[tuple[str, str | None]] = set()
    for summary_path in summary_paths:
        dataset_label = infer_dataset_label(
            summary_path,
            base_model_name,
        )
        _, reasoning_effort = get_summary_model_variant(
            base_model_name=base_model_name,
            summary_path=summary_path,
        )
        variant_key = (
            dataset_label,
            reasoning_effort if base_model_name in GPT_OSS_MODEL_NAMES else None,
        )
        if variant_key in seen_variants:
            raise ValueError(
                f"Found multiple {base_model_name} summaries for dataset "
                f"{dataset_label!r} and reasoning effort {reasoning_effort!r} "
                f"in {seed_dir}."
            )
        seen_variants.add(variant_key)

    return summary_paths


def discover_seed_summaries(
    input_folder: str | Path,
    summary_glob: str,
) -> list[SeedSummary]:
    """Discover per-seed evaluation summaries below an input folder.

    Args:
        input_folder (str | Path): Folder containing ``seed_*`` subfolders.
        summary_glob (str): Glob pattern used within each seed folder.

    Returns:
        list[SeedSummary]: Seed summaries sorted by seed value.

    Raises:
        FileNotFoundError: If the input folder does not exist.
        NotADirectoryError: If the input folder is not a directory.
        ValueError: If no usable seed summaries are found.
    """
    eval_folder = Path(input_folder).expanduser().resolve()
    if not eval_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {eval_folder}")
    if not eval_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {eval_folder}")

    base_model_name = infer_model_name(eval_folder)
    seed_dirs: list[tuple[int, Path]] = []
    for child in eval_folder.iterdir():
        if not child.is_dir():
            continue
        seed = parse_seed_dir(child)
        if seed is not None:
            seed_dirs.append((seed, child))

    if not seed_dirs:
        raise ValueError(f"No seed_* subfolders found in {eval_folder}.")

    summaries: list[SeedSummary] = []
    for seed, seed_dir in sorted(seed_dirs):
        summary_paths = find_seed_summary_paths(
            seed_dir=seed_dir,
            summary_glob=summary_glob,
            base_model_name=base_model_name,
        )
        if not summary_paths:
            logging.warning("No summary matched %s in %s", summary_glob, seed_dir)
            continue

        for summary_path in summary_paths:
            dataset_label = infer_dataset_label(
                summary_path,
                base_model_name,
            )
            model_name, reasoning_effort = get_summary_model_variant(
                base_model_name=base_model_name,
                summary_path=summary_path,
            )
            summary = load_summary_json(summary_path)
            overall = summary.get("overall")
            if not isinstance(overall, dict):
                raise ValueError(f"Missing object field 'overall' in {summary_path}")
            per_task = summary.get("per_task")
            if not isinstance(per_task, dict):
                raise ValueError(f"Missing object field 'per_task' in {summary_path}")

            summaries.append(
                SeedSummary(
                    base_model_name=base_model_name,
                    dataset_label=dataset_label,
                    model_name=model_name,
                    reasoning_effort=reasoning_effort,
                    seed=seed,
                    summary_path=summary_path,
                    overall=overall,
                    per_task=per_task,
                )
            )

    if not summaries:
        raise ValueError(f"No summary files matched {summary_glob} in {eval_folder}.")

    return summaries


def group_seed_summaries(
    summaries: list[SeedSummary],
) -> dict[tuple[str, str], list[SeedSummary]]:
    """Group seed summaries by dataset label and display model name.

    Args:
        summaries (list[SeedSummary]): Seed summaries to group.

    Returns:
        dict[tuple[str, str], list[SeedSummary]]: Seed summaries keyed by
        dataset label and display model name.
    """
    grouped: dict[tuple[str, str], list[SeedSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary.dataset_label, summary.model_name)].append(summary)

    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: model_group_sort_key(item[1]),
    )
    return {
        model_name: sorted(model_summaries, key=lambda summary: summary.seed)
        for model_name, model_summaries in sorted_groups
    }


def model_group_sort_key(summaries: list[SeedSummary]) -> tuple[str, str, int, str]:
    """Return a stable sort key for one display-model summary group.

    Args:
        summaries (list[SeedSummary]): Summaries for one display model.

    Returns:
        tuple[str, str, int, str]: Sort key using dataset, base model, and
        effort order.
    """
    first_summary = summaries[0]
    effort_rank = REASONING_EFFORT_ORDER.get(
        first_summary.reasoning_effort or "",
        len(REASONING_EFFORT_ORDER),
    )
    return (
        first_summary.dataset_label,
        first_summary.base_model_name,
        effort_rank,
        first_summary.model_name,
    )


def is_scalar_number(value: Any) -> TypeGuard[int | float]:
    """Return whether a value is a finite non-boolean scalar number.

    Args:
        value (Any): Value to check.

    Returns:
        TypeGuard[int | float]: Whether the value is a finite ``int`` or
        ``float``.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def summarize_values(values: list[float]) -> dict[str, float | int]:
    """Compute descriptive statistics for numeric values.

    Args:
        values (list[float]): Numeric values to summarize.

    Returns:
        dict[str, float | int]: Count, mean, stdev, min, max, median, and sum.
    """
    if not values:
        raise ValueError("Cannot summarize an empty list of values.")

    value_stdev = stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean(values),
        "stdev": value_stdev,
        "min": min(values),
        "max": max(values),
        "median": median(values),
        "sum": sum(values),
    }


def summarize_scalar_metrics(
    overalls: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Summarize scalar overall metrics across seed runs.

    Args:
        overalls (list[dict[str, Any]]): Overall metric dictionaries.

    Returns:
        dict[str, dict[str, float | int]]: Descriptive stats keyed by metric name.
    """
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for overall in overalls:
        for key, value in overall.items():
            if is_scalar_number(value):
                values_by_metric[key].append(float(value))

    return {
        metric: summarize_values(values)
        for metric, values in sorted(values_by_metric.items())
    }


def is_numeric_count_dict(value: Any) -> bool:
    """Return whether a value is a dictionary of numeric counts.

    Args:
        value (Any): Value to check.

    Returns:
        bool: Whether the value is a dictionary with numeric values.
    """
    if not isinstance(value, dict):
        return False
    return all(is_scalar_number(item) for item in value.values())


def summarize_count_dict_metrics(
    overalls: list[dict[str, Any]],
) -> tuple[CountTotals, CountStats]:
    """Summarize nested count dictionaries across seed runs.

    Args:
        overalls (list[dict[str, Any]]): Overall metric dictionaries.

    Returns:
        tuple[CountTotals, CountStats]: Total counts and descriptive stats for
            each nested count key.
    """
    count_metric_names = sorted(
        {
            key
            for overall in overalls
            for key, value in overall.items()
            if is_numeric_count_dict(value)
        }
    )

    count_totals: CountTotals = {}
    count_stats: CountStats = {}
    for metric_name in count_metric_names:
        nested_keys = sorted(
            {
                nested_key
                for overall in overalls
                for nested_key in overall.get(metric_name, {}).keys()
                if is_numeric_count_dict(overall.get(metric_name, {}))
            }
        )

        totals: dict[str, float] = {}
        stats: dict[str, dict[str, float | int]] = {}
        for nested_key in nested_keys:
            values = [
                float(overall.get(metric_name, {}).get(nested_key, 0.0))
                for overall in overalls
            ]
            totals[nested_key] = sum(values)
            stats[nested_key] = summarize_values(values)

        count_totals[metric_name] = totals
        count_stats[metric_name] = stats

    return count_totals, count_stats


def summarize_task_success_rates(
    summaries: list[SeedSummary],
) -> list[dict[str, str | float]]:
    """Average per-task accuracy across seed summaries.

    Args:
        summaries (list[SeedSummary]): Seed summaries for one model variant.

    Returns:
        list[dict[str, str | float]]: Task difficulty rows with ``task_id`` and
        ``success_rate`` fields.

    Raises:
        ValueError: If a per-task summary is malformed or lacks ``accuracy``.
    """
    accuracies_by_task: dict[str, list[float]] = defaultdict(list)
    for summary in summaries:
        for task_id, task_metrics in summary.per_task.items():
            if not isinstance(task_metrics, dict):
                raise ValueError(
                    f"Malformed per_task metrics for task {task_id!r} in "
                    f"{summary.summary_path}."
                )
            accuracy = task_metrics.get("accuracy")
            if not is_scalar_number(accuracy):
                raise ValueError(
                    f"Missing numeric per_task accuracy for task {task_id!r} "
                    f"in {summary.summary_path}."
                )
            accuracies_by_task[str(task_id)].append(float(accuracy))  # type: ignore

    return [
        {
            "task_id": task_id,
            "success_rate": mean(task_accuracies),
        }
        for task_id, task_accuracies in sorted(accuracies_by_task.items())
    ]


def summarize_task_query_counts(summaries: list[SeedSummary]) -> TaskQueryTables:
    """Tabulate exact per-task counts and cumulative task robustness.

    For each observed query total N, report tasks with at least X correct
    queries for X = 1 through N. Tasks with different query totals remain
    separate. Fractions use all tasks present in that seed as the denominator,
    including tasks with zero correct queries. Each seed has equal weight in
    the mean and sample standard deviation.

    Args:
        summaries (list[SeedSummary]): Summaries for one dataset/model variant.

    Returns:
        TaskQueryTables: Per-task counts, per-seed threshold counts and fractions,
        and cross-seed fraction statistics, as three named tables.

    Raises:
        ValueError: If summaries are empty, seeds are duplicated, or task
            query counts are missing or invalid.
    """
    if not summaries:
        raise ValueError("Cannot summarize an empty list of seed summaries.")
    if len({summary.seed for summary in summaries}) != len(summaries):
        raise ValueError("Duplicate seed summaries for one dataset/model variant.")

    tables: TaskQueryTables = {
        "task_query_counts": [],
        "seed_query_distribution": [],
        "query_distribution_seed_stats": [],
    }
    counts_by_seed: dict[int, Counter[tuple[int, int]]] = {}
    query_totals: set[int] = set()
    for summary in sorted(summaries, key=lambda item: item.seed):
        if not summary.per_task:
            raise ValueError(f"No per-task counts in {summary.summary_path}.")
        counts: Counter[tuple[int, int]] = Counter()
        for task_id, metrics in sorted(summary.per_task.items()):
            if not isinstance(metrics, dict):
                raise ValueError(
                    f"Malformed per_task metrics for task {task_id!r} in "
                    f"{summary.summary_path}."
                )
            n_queries = metrics.get("n")
            n_correct = metrics.get("n_correct")
            if (
                not isinstance(n_queries, int)
                or isinstance(n_queries, bool)
                or n_queries <= 0
                or not isinstance(n_correct, int)
                or isinstance(n_correct, bool)
                or not 0 <= n_correct <= n_queries
            ):
                raise ValueError(
                    f"Expected integer counts n > 0 and 0 <= n_correct <= n "
                    f"for task {task_id!r} in {summary.summary_path}."
                )
            tables["task_query_counts"].append(
                {
                    "seed": summary.seed,
                    "task_id": task_id,
                    "n_queries": n_queries,
                    "n_correct": n_correct,
                    "solved_queries": f"{n_correct}/{n_queries}",
                }
            )
            counts[(n_queries, n_correct)] += 1
            query_totals.add(n_queries)
        counts_by_seed[summary.seed] = counts

    fractions_by_threshold: dict[tuple[int, int], list[float]] = defaultdict(list)
    for seed, counts in counts_by_seed.items():
        total_tasks = sum(counts.values())
        for n_queries in sorted(query_totals):
            n_tasks = sum(
                counts[(n_queries, n_correct)] for n_correct in range(1, n_queries + 1)
            )
            for min_n_correct in range(1, n_queries + 1):
                fraction = n_tasks / total_tasks
                tables["seed_query_distribution"].append(
                    {
                        "seed": seed,
                        "n_queries": n_queries,
                        "min_n_correct": min_n_correct,
                        "solved_queries": f"at least {min_n_correct}/{n_queries}",
                        "n_tasks": n_tasks,
                        "total_tasks": total_tasks,
                        "fraction": fraction,
                    }
                )
                fractions_by_threshold[(n_queries, min_n_correct)].append(fraction)
                n_tasks -= counts[(n_queries, min_n_correct)]

    for (n_queries, min_n_correct), fractions in fractions_by_threshold.items():
        tables["query_distribution_seed_stats"].append(
            {
                "n_queries": n_queries,
                "min_n_correct": min_n_correct,
                "solved_queries": f"at least {min_n_correct}/{n_queries}",
                "n_seed_runs": len(fractions),
                "fraction_mean": mean(fractions),
                "fraction_stdev": stdev(fractions) if len(fractions) > 1 else 0.0,
            }
        )
    return tables


def aggregate_model_summaries(
    summaries: list[SeedSummary],
    input_folder: str | Path,
) -> dict[str, Any]:
    """Aggregate seed summaries for one model/input folder.

    Args:
        summaries (list[SeedSummary]): Seed summaries for one input folder.
        input_folder (str | Path): Source input folder.

    Returns:
        dict[str, Any]: Aggregate model summary.
    """
    if not summaries:
        raise ValueError("Cannot aggregate an empty summary list.")

    base_model_names = {summary.base_model_name for summary in summaries}
    dataset_labels = {summary.dataset_label for summary in summaries}
    model_names = {summary.model_name for summary in summaries}
    reasoning_efforts = {summary.reasoning_effort for summary in summaries}
    if (
        len(base_model_names) != 1
        or len(dataset_labels) != 1
        or len(model_names) != 1
        or len(reasoning_efforts) != 1
    ):
        raise ValueError("Cannot aggregate summaries from multiple model variants.")

    base_model_name = summaries[0].base_model_name
    dataset_label = summaries[0].dataset_label
    model_name = summaries[0].model_name
    reasoning_effort = summaries[0].reasoning_effort
    overalls = [summary.overall for summary in summaries]
    count_totals, count_stats = summarize_count_dict_metrics(overalls)

    return {
        "base_model_name": base_model_name,
        "dataset_label": dataset_label,
        "model_name": model_name,
        "reasoning_effort": reasoning_effort,
        "input_folder": str(Path(input_folder).expanduser().resolve()),
        "n_seed_runs": len(summaries),
        "seeds": [summary.seed for summary in summaries],
        "summary_paths": [str(summary.summary_path) for summary in summaries],
        "overall_scalar_stats": summarize_scalar_metrics(overalls),
        "overall_count_totals": count_totals,
        "overall_count_stats": count_stats,
        "difficulty_estimates": summarize_task_success_rates(summaries),
        "seed_overalls": [
            {
                "seed": summary.seed,
                "summary_path": str(summary.summary_path),
                "base_model_name": summary.base_model_name,
                "dataset_label": summary.dataset_label,
                "model_name": summary.model_name,
                "reasoning_effort": summary.reasoning_effort,
                "overall": summary.overall,
            }
            for summary in summaries
        ],
    }


def make_unique_labels(labels: list[str]) -> list[str]:
    """Make repeated plot labels unique while preserving readable names.

    Args:
        labels (list[str]): Candidate labels.

    Returns:
        list[str]: Unique labels.
    """
    counts = Counter(labels)
    seen: dict[str, int] = defaultdict(int)
    unique_labels: list[str] = []
    for label in labels:
        if counts[label] == 1:
            unique_labels.append(label)
            continue
        seen[label] += 1
        unique_labels.append(f"{label} ({seen[label]})")
    return unique_labels


def parse_plot_model_specs(
    raw_specs: list[list[str]] | None,
) -> list[PlotModelSpec] | None:
    """Convert repeated CLI plot-model values into validated specifications.

    Args:
        raw_specs (list[list[str]] | None): Raw ``--plot_model`` triples.

    Returns:
        list[PlotModelSpec] | None: Plot specifications in requested order, or
        ``None`` when no source selection was requested.

    Raises:
        ValueError: If an internal model/source name is repeated.
    """
    if raw_specs is None:
        return None

    specs = [PlotModelSpec(*raw_spec) for raw_spec in raw_specs]
    model_name_counts = Counter(spec.model_name for spec in specs)
    duplicate_names = sorted(
        model_name for model_name, count in model_name_counts.items() if count > 1
    )
    if duplicate_names:
        raise ValueError(
            f"Repeated --plot_model MODEL_NAME values: {', '.join(duplicate_names)}"
        )
    return specs


def select_plot_rows(
    plot_rows: list[PlotRow],
    plot_model_specs: list[PlotModelSpec] | None,
) -> tuple[list[PlotRow], list[str], list[str] | None]:
    """Select and format plot rows according to optional source specifications.

    Args:
        plot_rows (list[PlotRow]): Available model/source metric rows.
        plot_model_specs (list[PlotModelSpec] | None): Optional source order,
            labels, and colors.

    Returns:
        tuple[list[PlotRow], list[str], list[str] | None]: Selected metric rows,
        displayed labels, and explicit colors when configured.

    Raises:
        ValueError: If a requested model/source is unavailable or ambiguous.
    """
    if plot_model_specs is None:
        sorted_rows = sorted(plot_rows, key=lambda row: (-row[1], row[0]))
        return sorted_rows, [row[0] for row in sorted_rows], None

    rows_by_model: dict[str, list[PlotRow]] = defaultdict(list)
    for plot_row in plot_rows:
        rows_by_model[plot_row[0]].append(plot_row)

    requested_names = [spec.model_name for spec in plot_model_specs]
    missing_names = [name for name in requested_names if name not in rows_by_model]
    if missing_names:
        available_names = sorted(rows_by_model)
        raise ValueError(
            "Requested plot sources are unavailable: "
            f"{', '.join(missing_names)}. Available sources: "
            f"{', '.join(available_names)}"
        )

    ambiguous_names = [name for name in requested_names if len(rows_by_model[name]) > 1]
    if ambiguous_names:
        raise ValueError(
            "Requested plot sources have multiple metric rows: "
            f"{', '.join(ambiguous_names)}"
        )

    selected_rows = [rows_by_model[name][0] for name in requested_names]
    display_labels = [spec.display_label for spec in plot_model_specs]
    colors = [spec.color for spec in plot_model_specs]
    return selected_rows, display_labels, colors


def format_metric_display_label(metric_name: str) -> str:
    """Return the publication-facing label for an evaluation metric.

    Args:
        metric_name (str): Internal evaluation metric name.

    Returns:
        str: Label to display on the plot axis.
    """
    if metric_name == "accuracy":
        return "pass@1"
    return metric_name


def plot_model_metric_comparison(
    aggregates: list[dict[str, Any]],
    metric_name: str,
    output_path: str | Path,
    dpi: int,
    dataset_labels: list[str],
    plot_model_specs: list[PlotModelSpec] | None = None,
    show_title: bool = True,
) -> None:
    """Plot average model metric values with seed-level stdev error bars.

    Args:
        aggregates (list[dict[str, Any]]): Aggregate model summaries.
        metric_name (str): Overall scalar metric to plot.
        output_path (str | Path): Plot output path. A JSON data sidecar is
            written beside it with the same stem. PDF output also produces a
            same-stem SVG.
        dpi (int): Output image DPI.
        dataset_labels (list[str]): Dataset labels included in the plot.
        plot_model_specs (list[PlotModelSpec] | None): Optional source order,
            display labels, and colors. When provided, unlisted sources are
            excluded.
        show_title (bool): Whether to display the plot title.
    """

    title = "Performance Comparison" if show_title else None

    plot_path = Path(output_path).expanduser().resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    plot_rows: list[PlotRow] = []
    source_stats_by_model: dict[str, dict[str, Any]] = {}
    for aggregate in aggregates:
        metric_stats = aggregate["overall_scalar_stats"].get(metric_name)
        if metric_stats is None:
            raise ValueError(
                f"Metric {metric_name!r} is missing for model "
                f"{aggregate['model_name']!r}."
            )

        model_name = str(aggregate["model_name"])
        metric_mean = float(metric_stats["mean"])
        metric_stdev = float(metric_stats["stdev"])
        plot_rows.append(
            (
                model_name,
                metric_mean,
                metric_stdev,
                metric_stdev,
            )
        )
        source_stats_by_model[model_name] = {
            "value": metric_mean,
            "stdev": metric_stdev,
            "error_bar_type": "seed_stdev",
        }

    plot_rows, display_labels, configured_colors = select_plot_rows(
        plot_rows,
        plot_model_specs,
    )
    model_labels = [row[0] for row in plot_rows]
    labels = make_unique_labels(display_labels)
    means = [row[1] for row in plot_rows]
    lower_errors = [row[2] for row in plot_rows]
    upper_errors = [row[3] for row in plot_rows]

    fig_width = max(7.0, 1.1 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2), facecolor="white")
    positions = list(range(len(labels)))
    colors = configured_colors or get_model_bar_colors(model_labels)
    metric_display_label = format_metric_display_label(metric_name)
    comparison_sources: list[dict[str, Any]] = []
    for model_label, display_label, color in zip(
        model_labels,
        display_labels,
        colors,
        strict=True,
    ):
        source_stats = source_stats_by_model[model_label]
        source_summary = {
            "model_name": model_label,
            "display_label": display_label,
            "color": color,
            metric_display_label: source_stats["value"],
            "stdev": source_stats["stdev"],
            "error_bar_type": source_stats["error_bar_type"],
        }
        comparison_sources.append(source_summary)

    ax.bar(
        positions,
        means,
        yerr=[lower_errors, upper_errors],
        capsize=5,
        color=colors,
        edgecolor="#2f2f2f",
        linewidth=0.7,
        error_kw={"elinewidth": 1.4, "ecolor": "#303030", "capthick": 1.4},
    )

    if title is not None:
        ax.set_title(title, fontsize=PLOT_TITLE_FONT_SIZE, fontweight="semibold")
    ax.set_ylabel(
        metric_display_label,
        fontsize=PLOT_AXIS_LABEL_FONT_SIZE,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        labels,
        rotation=30,
        ha="right",
        fontsize=PLOT_TICK_FONT_SIZE,
    )
    ax.tick_params(axis="y", labelsize=PLOT_TICK_FONT_SIZE)
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    error_bar_tops = [
        mean_value + upper_error
        for mean_value, upper_error in zip(means, upper_errors, strict=True)
    ]
    lower_bound = min(
        mean_value - lower_error
        for mean_value, lower_error in zip(means, lower_errors, strict=True)
    )
    upper_bound = max(error_bar_tops)
    label_offset = max(0.0125, 0.035 * max(upper_bound, 1.0))
    if 0.0 <= lower_bound and upper_bound <= 1.0:
        ax.set_ylim(0.0, max(1.0, upper_bound + 2.5 * label_offset))
    else:
        ax.set_ylim(bottom=min(0.0, lower_bound), top=upper_bound + 2.5 * label_offset)

    for position, value, error_bar_top in zip(
        positions,
        means,
        error_bar_tops,
        strict=True,
    ):
        ax.text(
            position,
            error_bar_top + label_offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=PLOT_VALUE_FONT_SIZE,
            fontweight="semibold",
        )

    fig.tight_layout()
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
        if plot_path.suffix.lower() == ".pdf":
            fig.savefig(plot_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    save_json(
        {
            "title": title,
            "dataset_labels": dataset_labels,
            "metric_name": metric_display_label,
            "internal_metric_name": metric_name,
            "sources": comparison_sources,
        },
        plot_path.with_suffix(".json"),
    )


def get_model_bar_colors(model_labels: list[str]) -> list[str]:
    """Return model-aware colors for performance bars.

    Args:
        model_labels (list[str]): Model labels in plotted order.

    Returns:
        list[str]: Matplotlib-compatible color hex strings.
    """
    colors: list[str] = []
    other_model_index = 0
    for model_label in model_labels:
        gpt_oss_color = get_gpt_oss_reasoning_color(model_label)
        if gpt_oss_color is not None:
            colors.append(gpt_oss_color)
            continue

        colors.append(OTHER_MODEL_COLORS[other_model_index % len(OTHER_MODEL_COLORS)])
        other_model_index += 1

    return colors


def get_gpt_oss_reasoning_color(model_label: str) -> str | None:
    """Return a fixed GPT-OSS reasoning-effort color, if applicable.

    Args:
        model_label (str): Display model label.

    Returns:
        str | None: Hex color for GPT-OSS labels, otherwise ``None``.
    """
    normalized_label = model_label.strip().lower()
    for base_model_name, effort_colors in GPT_OSS_REASONING_COLORS.items():
        prefix = f"{base_model_name} "
        if not normalized_label.startswith(prefix):
            continue
        reasoning_effort = normalized_label.removeprefix(prefix).split()[0]
        return effort_colors.get(reasoning_effort)

    return None


def safe_filename_fragment(value: str) -> str:
    """Convert a metric name into a safe filename fragment.

    Args:
        value (str): Raw metric name.

    Returns:
        str: Filename-safe metric fragment.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "metric"


def get_stats_output_path(
    output_dir: Path,
    aggregate: dict[str, Any],
    n_input_aggregates: int,
) -> Path:
    """Build a stats JSON path for one aggregate.

    Args:
        output_dir (Path): Global summary directory.
        aggregate (dict[str, Any]): Aggregate model summary.
        n_input_aggregates (int): Number of aggregates for the input folder.

    Returns:
        Path: Output JSON path.
    """
    dataset_fragment = safe_filename_fragment(str(aggregate["dataset_label"]))
    if aggregate["reasoning_effort"] is None and n_input_aggregates == 1:
        return output_dir / f"{dataset_fragment}_overall_seed_stats.json"

    model_fragment = safe_filename_fragment(str(aggregate["model_name"]))
    return output_dir / f"{dataset_fragment}_overall_seed_stats_{model_fragment}.json"


def get_difficulty_estimates_output_path(
    output_dir: Path,
    aggregate: dict[str, Any],
    n_input_aggregates: int,
) -> Path:
    """Build a difficulty-estimates CSV path for one aggregate.

    Args:
        output_dir (Path): Global summary directory.
        aggregate (dict[str, Any]): Aggregate model summary.
        n_input_aggregates (int): Number of aggregates for the input folder.

    Returns:
        Path: Output CSV path.
    """
    dataset_fragment = safe_filename_fragment(str(aggregate["dataset_label"]))
    model_fragment = safe_filename_fragment(str(aggregate["model_name"]))
    return output_dir / f"{dataset_fragment}_difficulty_estimates_{model_fragment}.csv"


def write_difficulty_estimates_csv(
    rows: list[dict[str, str | float]],
    output_path: str | Path,
) -> None:
    """Write per-task success rates to CSV.

    Args:
        rows (list[dict[str, str | float]]): Rows with ``task_id`` and
            ``success_rate`` values.
        output_path (str | Path): Destination CSV path.
    """
    csv_path = Path(output_path).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "success_rate"])
        writer.writeheader()
        writer.writerows(rows)


def write_task_query_tables(
    tables: TaskQueryTables,
    output_dir: Path,
    dataset_label: str,
    model_name: str,
) -> list[Path]:
    """Write task counts and query-distribution tables as CSV files.

    Args:
        tables (TaskQueryTables): Nonempty tables from task query analysis.
        output_dir (Path): Global summary output directory.
        dataset_label (str): Dataset/split label used in filenames.
        model_name (str): Model variant name, including reasoning effort.

    Returns:
        list[Path]: Paths of the three written CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_fragment = safe_filename_fragment(dataset_label)
    model_fragment = safe_filename_fragment(model_name)
    paths: list[Path] = []
    for table_name, rows in tables.items():
        path = output_dir / f"{dataset_fragment}_{table_name}_{model_fragment}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    return paths


def get_dataset_labels(aggregates: list[dict[str, Any]]) -> list[str]:
    """Return sorted dataset labels present in aggregate summaries.

    Args:
        aggregates (list[dict[str, Any]]): Aggregate summaries.

    Returns:
        list[str]: Sorted dataset labels.
    """
    return sorted({str(aggregate["dataset_label"]) for aggregate in aggregates})


def count_aggregates_by_dataset(
    aggregates: list[dict[str, Any]],
) -> dict[str, int]:
    """Count aggregate summaries per dataset label.

    Args:
        aggregates (list[dict[str, Any]]): Aggregate summaries.

    Returns:
        dict[str, int]: Number of aggregates keyed by dataset label.
    """
    counts: dict[str, int] = defaultdict(int)
    for aggregate in aggregates:
        counts[str(aggregate["dataset_label"])] += 1
    return dict(counts)


def main() -> None:
    """Run seed-stat aggregation and plotting."""
    args = parse_arguments()
    plot_model_specs = parse_plot_model_specs(args.plot_model)
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    aggregates: list[dict[str, Any]] = []
    aggregates_by_input_folder: dict[str, list[dict[str, Any]]] = {}
    for input_folder in args.input_folders:
        summaries = discover_seed_summaries(
            input_folder=input_folder,
            summary_glob=args.summary_glob,
        )
        input_aggregates: list[dict[str, Any]] = []
        for model_summaries in group_seed_summaries(summaries).values():
            aggregate = aggregate_model_summaries(
                summaries=model_summaries,
                input_folder=input_folder,
            )
            aggregate["task_query_analysis"] = summarize_task_query_counts(
                model_summaries
            )
            input_aggregates.append(aggregate)
            aggregates.append(aggregate)
            logging.info(
                "Aggregated %d seed summaries for %s",
                aggregate["n_seed_runs"],
                aggregate["model_name"],
            )
        aggregates_by_input_folder[str(input_folder)] = input_aggregates

    for input_folder in args.input_folders:
        output_dir = get_global_summary_dir(input_folder)
        input_aggregates = aggregates_by_input_folder[str(input_folder)]
        dataset_labels = get_dataset_labels(input_aggregates)
        dataset_counts = count_aggregates_by_dataset(input_aggregates)
        for aggregate in input_aggregates:
            stats_path = get_stats_output_path(
                output_dir=output_dir,
                aggregate=aggregate,
                n_input_aggregates=dataset_counts[str(aggregate["dataset_label"])],
            )
            output = {
                "input_folder": str(Path(input_folder).expanduser().resolve()),
                "dataset_label": aggregate["dataset_label"],
                "summary_glob": args.summary_glob,
                "plot_metric": args.plot_metric,
                "model": aggregate,
            }
            save_json(output, stats_path)
            logging.info("Wrote aggregate JSON to %s", stats_path)
            difficulty_estimates_path = get_difficulty_estimates_output_path(
                output_dir=output_dir,
                aggregate=aggregate,
                n_input_aggregates=dataset_counts[str(aggregate["dataset_label"])],
            )
            write_difficulty_estimates_csv(
                rows=aggregate["difficulty_estimates"],
                output_path=difficulty_estimates_path,
            )
            logging.info(
                "Wrote difficulty estimates CSV to %s",
                difficulty_estimates_path,
            )
            for table_path in write_task_query_tables(
                tables=aggregate["task_query_analysis"],
                output_dir=output_dir,
                dataset_label=aggregate["dataset_label"],
                model_name=aggregate["model_name"],
            ):
                logging.info("Wrote task query CSV to %s", table_path)

        for dataset_label in dataset_labels:
            plot_dataset_fragment = safe_filename_fragment(dataset_label)
            plot_path = output_dir / (
                f"{plot_dataset_fragment}_model_"
                f"{safe_filename_fragment(args.plot_metric)}_comparison."
                f"{args.plot_format}"
            )
            plot_aggregates = [
                aggregate
                for aggregate in aggregates
                if str(aggregate["dataset_label"]) == dataset_label
            ]
            plot_model_metric_comparison(
                aggregates=plot_aggregates,
                metric_name=args.plot_metric,
                output_path=plot_path,
                dpi=args.dpi,
                dataset_labels=[dataset_label],
                plot_model_specs=plot_model_specs,
                show_title=not args.no_title,
            )
            logging.info("Wrote comparison plot to %s", plot_path)
            if plot_path.suffix.lower() == ".pdf":
                logging.info(
                    "Wrote SVG comparison plot to %s",
                    plot_path.with_suffix(".svg"),
                )
            logging.info(
                "Wrote comparison data to %s",
                plot_path.with_suffix(".json"),
            )


if __name__ == "__main__":
    main()
