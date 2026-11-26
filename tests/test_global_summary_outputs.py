from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import pytest
from matplotlib.axes import Axes

from arc_cd.data_prep.concept_arc import CANONICAL_CONCEPT_LABELS
from scripts.plot_eval_concept_performance import (
    get_common_concept_names,
    get_concept_arc_dataset_shade_index,
    get_model_legend_ncols,
    load_concept_performances,
    plot_concept_performance,
    sort_concept_names,
    warn_if_model_seed_sets_differ,
    write_concept_performance_summary,
)
from scripts.plot_eval_concept_performance import (
    get_default_output_path as get_default_concept_performance_output_path,
)
from scripts.plot_eval_concept_performance import (
    get_default_summary_output_path as get_default_concept_performance_summary_output_path,
)
from scripts.plot_eval_concept_performance import (
    sort_models_by_average_metric as sort_models_by_average_concept_metric,
)
from scripts.plot_eval_dataset_comparison import (
    format_dataset_display_label,
    get_common_model_names,
    get_dataset_legend_color,
    get_skipped_model_names_by_dataset,
    load_dataset_comparisons,
    plot_eval_dataset_comparison,
    shade_model_color,
    sort_model_names_by_average_metric,
)
from scripts.plot_eval_dataset_comparison import (
    main as run_dataset_comparison,
)
from scripts.summarize_eval_seed_stats import (
    SeedSummary,
    count_aggregates_by_dataset,
    format_metric_display_label,
    get_difficulty_estimates_output_path,
    get_gpt_oss_reasoning_color,
    get_model_bar_colors,
    get_stats_output_path,
    parse_plot_model_specs,
    plot_model_metric_comparison,
    select_plot_rows,
    summarize_task_query_counts,
    summarize_task_success_rates,
    write_difficulty_estimates_csv,
)


def test_eval_stats_output_paths_include_dataset_label(tmp_path: Path) -> None:
    """Prefix eval global-summary JSON names with the dataset label."""
    aggregate = {
        "dataset_label": "concept_arc",
        "model_name": "qwen3_32b",
        "reasoning_effort": None,
    }
    gpt_aggregate = {
        "dataset_label": "concept_arc",
        "model_name": "gpt_oss_120b high",
        "reasoning_effort": "high",
    }

    assert get_stats_output_path(tmp_path, aggregate, 1).name == (
        "concept_arc_overall_seed_stats.json"
    )
    assert get_stats_output_path(tmp_path, gpt_aggregate, 3).name == (
        "concept_arc_overall_seed_stats_gpt_oss_120b_high.json"
    )


def test_eval_stats_counts_variants_per_dataset(tmp_path: Path) -> None:
    """Keep train/eval singleton outputs separate without model suffixes."""
    train_aggregate = {
        "dataset_label": "concept_arc",
        "model_name": "qwen3_32b",
        "reasoning_effort": None,
    }
    eval_aggregate = {
        "dataset_label": "concept_arc_100",
        "model_name": "qwen3_32b",
        "reasoning_effort": None,
    }
    counts = count_aggregates_by_dataset([train_aggregate, eval_aggregate])

    assert (
        get_stats_output_path(
            tmp_path,
            train_aggregate,
            counts["concept_arc"],
        ).name
        == "concept_arc_overall_seed_stats.json"
    )
    assert (
        get_stats_output_path(
            tmp_path,
            eval_aggregate,
            counts["concept_arc_100"],
        ).name
        == "concept_arc_100_overall_seed_stats.json"
    )


def test_difficulty_estimate_output_paths_include_dataset_label(
    tmp_path: Path,
) -> None:
    """Name per-task difficulty CSVs by dataset and model variant."""
    aggregate = {
        "dataset_label": "concept_arc",
        "model_name": "qwen3_32b",
        "reasoning_effort": None,
    }
    gpt_aggregate = {
        "dataset_label": "concept_arc",
        "model_name": "gpt_oss_120b high",
        "reasoning_effort": "high",
    }

    assert get_difficulty_estimates_output_path(tmp_path, aggregate, 1).name == (
        "concept_arc_difficulty_estimates_qwen3_32b.csv"
    )
    assert get_difficulty_estimates_output_path(tmp_path, gpt_aggregate, 3).name == (
        "concept_arc_difficulty_estimates_gpt_oss_120b_high.csv"
    )


def test_summarize_and_write_task_success_rates(tmp_path: Path) -> None:
    """Average per-task accuracy across seed summaries and write two columns."""
    summaries = [
        SeedSummary(
            base_model_name="qwen3_32b",
            dataset_label="concept_arc",
            model_name="qwen3_32b",
            reasoning_effort=None,
            seed=1,
            summary_path=Path("seed_1.json"),
            overall={"accuracy": 0.5},
            per_task={
                "task-b": {"accuracy": 1.0},
                "task-a": {"accuracy": 0.0},
            },
        ),
        SeedSummary(
            base_model_name="qwen3_32b",
            dataset_label="concept_arc",
            model_name="qwen3_32b",
            reasoning_effort=None,
            seed=2,
            summary_path=Path("seed_2.json"),
            overall={"accuracy": 0.75},
            per_task={
                "task-b": {"accuracy": 0.5},
                "task-a": {"accuracy": 1.0},
            },
        ),
    ]

    rows = summarize_task_success_rates(summaries)
    assert rows == [
        {"task_id": "task-a", "success_rate": 0.5},
        {"task_id": "task-b", "success_rate": 0.75},
    ]

    csv_path = tmp_path / "difficulty.csv"
    write_difficulty_estimates_csv(rows, csv_path)
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "task_id,success_rate",
        "task-a,0.5",
        "task-b,0.75",
    ]


def _make_query_count_summary(
    seed: int,
    task_counts: dict[str, tuple[int, int]],
) -> Any:
    """Build a seed summary with exact per-task query counts.

    Args:
        seed (int): Run seed.
        task_counts (dict[str, tuple[int, int]]): Task IDs mapped to
            (number correct, number of queries).

    Returns:
        Any: SeedSummary loaded from the summary script.
    """
    return SeedSummary(
        base_model_name="qwen3_32b",
        dataset_label="concept_arc_100",
        model_name="qwen3_32b",
        reasoning_effort=None,
        seed=seed,
        summary_path=Path(f"seed_{seed}.json"),
        overall={},
        per_task={
            task_id: {"n": n, "n_correct": correct, "accuracy": correct / n}
            for task_id, (correct, n) in task_counts.items()
        },
    )


def test_task_query_distribution_counts_none_partial_and_all_correct() -> None:
    """Count tasks cumulatively and include unsolved tasks in the denominator."""
    summaries = [
        _make_query_count_summary(
            44, {"a": (0, 10), "b": (0, 10), "c": (7, 10), "d": (7, 10)}
        ),
        _make_query_count_summary(
            42, {"a": (0, 10), "b": (7, 10), "c": (10, 10), "d": (10, 10)}
        ),
    ]
    tables = summarize_task_query_counts(summaries)
    tasks = tables["task_query_counts"]
    assert len(tasks) == 8
    assert tasks[1] == {
        "seed": 42,
        "task_id": "b",
        "n_queries": 10,
        "n_correct": 7,
        "solved_queries": "7/10",
    }
    rows = tables["seed_query_distribution"]
    assert len(rows) == 20
    for seed, expected in [(42, [3] * 7 + [2] * 3), (44, [2] * 7 + [0] * 3)]:
        seed_rows = [row for row in rows if row["seed"] == seed]
        assert [row["min_n_correct"] for row in seed_rows] == list(range(1, 11))
        assert [row["solved_queries"] for row in seed_rows] == [
            f"at least {threshold}/10" for threshold in range(1, 11)
        ]
        assert [row["n_tasks"] for row in seed_rows] == expected
        fractions = [row["fraction"] for row in seed_rows]
        assert fractions == sorted(fractions, reverse=True)
        for row in seed_rows:
            assert row["total_tasks"] == 4
            assert row["fraction"] == row["n_tasks"] / 4

    stats = tables["query_distribution_seed_stats"]
    assert len(stats) == 10
    assert [row["min_n_correct"] for row in stats] == list(range(1, 11))
    for row in stats[:7]:
        assert row["fraction_mean"] == pytest.approx(0.625)
        assert row["fraction_stdev"] == pytest.approx(stdev([0.75, 0.5]))
    for row in stats[7:]:
        assert row["fraction_mean"] == pytest.approx(0.25)
        assert row["fraction_stdev"] == pytest.approx(stdev([0.5, 0.0]))
    assert all(row["n_seed_runs"] == 2 for row in stats)


def test_task_query_distribution_preserves_denominators_and_seed_weights() -> None:
    """Keep 1/1 distinct from 1/10 and average fractions equally across seeds."""
    summaries = [
        _make_query_count_summary(42, {"a": (1, 1), "b": (7, 10)}),
        _make_query_count_summary(44, {"b": (1, 10)}),
    ]
    tables = summarize_task_query_counts(summaries)
    stats = {
        row["solved_queries"]: row for row in tables["query_distribution_seed_stats"]
    }
    assert len(stats) == 11
    assert stats["at least 1/1"]["fraction_mean"] == pytest.approx(0.25)
    assert stats["at least 1/10"]["fraction_mean"] == pytest.approx(0.75)
    assert stats["at least 1/10"]["fraction_stdev"] == pytest.approx(stdev([0.5, 1.0]))
    for threshold in range(2, 8):
        assert stats[f"at least {threshold}/10"]["fraction_mean"] == pytest.approx(0.25)
    for threshold in range(8, 11):
        row = stats[f"at least {threshold}/10"]
        assert row["fraction_mean"] == row["fraction_stdev"] == 0.0
    single_seed = summarize_task_query_counts(summaries[:1])
    assert all(
        row["n_seed_runs"] == 1 and row["fraction_stdev"] == 0.0
        for row in single_seed["query_distribution_seed_stats"]
    )


@pytest.mark.parametrize("n_queries,n_correct", [(1, 0), (1, 1), (10, 0), (10, 10)])
def test_task_query_distribution_all_or_none_correct(
    n_queries: int, n_correct: int
) -> None:
    """Include every positive threshold when all tasks succeed or all fail.

    Args:
        n_queries (int): Queries per task.
        n_correct (int): Correct queries per task, either zero or all queries.
    """
    tables = summarize_task_query_counts(
        [_make_query_count_summary(42, {"a": (n_correct, n_queries)})]
    )
    rows = tables["seed_query_distribution"]
    assert [row["min_n_correct"] for row in rows] == list(range(1, n_queries + 1))
    assert all(row["fraction"] == float(n_correct > 0) for row in rows)
    assert tables["task_query_counts"][0]["n_correct"] == n_correct


@pytest.mark.parametrize(
    "counts",
    [
        {"accuracy": 0.7},
        {"n": 0, "n_correct": 0},
        {"n": 10, "n_correct": -1},
        {"n": 10, "n_correct": 11},
        {"n": 10, "n_correct": True},
        {"n": True, "n_correct": 1},
        {"n": 10.0, "n_correct": 7},
        {"n": 10, "n_correct": 7.0},
    ],
)
def test_task_query_distribution_rejects_invalid_counts(
    counts: dict[str, int | float | bool],
) -> None:
    """Reject missing and invalid exact counts instead of rounding accuracy.

    Args:
        counts (dict[str, int | float | bool]): Invalid per-task metrics.
    """
    summary = _make_query_count_summary(42, {"a": (7, 10)})
    summary.per_task["a"] = counts
    with pytest.raises(ValueError, match="integer counts") as exc:
        summarize_task_query_counts([summary])
    assert "seed_42.json" in str(exc.value)
    assert "task 'a'" in str(exc.value)


def test_task_query_distribution_rejects_empty_or_duplicate_seeds() -> None:
    """Require tasks and distinct seeds for well-defined seed fractions."""
    summary = _make_query_count_summary(42, {"a": (7, 10)})
    with pytest.raises(ValueError, match="empty list"):
        summarize_task_query_counts([])
    with pytest.raises(ValueError, match="Duplicate seed"):
        summarize_task_query_counts([summary, summary])
    with pytest.raises(ValueError, match="No per-task counts"):
        summarize_task_query_counts([_make_query_count_summary(42, {})])


def test_model_bar_colors_group_gpt_oss_reasoning_efforts() -> None:
    """Use shared hue families for GPT-OSS reasoning efforts."""
    labels = [
        "gpt_oss_20b low",
        "gpt_oss_20b medium",
        "gpt_oss_20b high",
        "gpt_oss_120b low",
        "gpt_oss_120b medium",
        "gpt_oss_120b high",
        "qwen3_32b",
    ]

    assert get_model_bar_colors(labels) == [
        "#9ecae1",
        "#4292c6",
        "#2171b5",
        "#fdd0a2",
        "#f16913",
        "#d94801",
        "#2ca25f",
    ]
    assert get_gpt_oss_reasoning_color("qwen3_32b") is None


def test_plot_model_specs_select_order_label_and_color() -> None:
    """Select only configured sources in their publication order."""
    raw_specs = [
        ["qwen35_27b", "Qwen3.5-27B", "#4C78A8"],
        ["phi4_reasoning_plus", "Phi4", "#F58518"],
        ["gpt_oss_120b high", "GPT-OSS-120B (high)", "#54A24B"],
        ["gpt_oss_120b medium", "GPT-OSS-120B (mid)", "#E45756"],
        ["gpt_oss_120b low", "GPT-OSS-120B (low)", "#72B7B2"],
    ]
    specs = parse_plot_model_specs(raw_specs)
    assert specs is not None
    plot_rows = [
        ("gpt_oss_120b low", 0.1, 0.01, 0.01),
        ("qwen35_27b", 0.5, 0.02, 0.02),
        ("gpt_oss_120b medium", 0.3, 0.03, 0.03),
        ("phi4_reasoning_plus", 0.6, 0.02, 0.02),
        ("unselected_model", 0.9, 0.01, 0.01),
        ("gpt_oss_120b high", 0.4, 0.04, 0.04),
    ]

    selected_rows, display_labels, colors = select_plot_rows(plot_rows, specs)

    assert [row[0] for row in selected_rows] == [spec[0] for spec in raw_specs]
    assert display_labels == [spec[1] for spec in raw_specs]
    assert colors == [spec[2] for spec in raw_specs]


def test_plot_model_specs_reject_duplicates_and_missing_sources() -> None:
    """Reject plot selections that cannot identify one available source."""
    with pytest.raises(ValueError, match="Repeated --plot_model"):
        parse_plot_model_specs(
            [
                ["qwen35_27b", "Qwen", "#4C78A8"],
                ["qwen35_27b", "Qwen duplicate", "#F58518"],
            ]
        )

    specs = parse_plot_model_specs(
        [["gpt_oss_120b high", "GPT-OSS-120B (high)", "#54A24B"]]
    )
    assert specs is not None
    with pytest.raises(ValueError, match="Requested plot sources are unavailable"):
        select_plot_rows([("qwen35_27b", 0.5, 0.02, 0.02)], specs)


def test_accuracy_uses_pass_at_one_display_label() -> None:
    """Use benchmark terminology for the plotted accuracy metric."""
    assert format_metric_display_label("accuracy") == "pass@1"
    assert format_metric_display_label("parse_rate") == "parse_rate"


def test_model_metric_comparison_writes_selected_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the selected model, seed SD, and labels in the exported PDF."""
    specs = parse_plot_model_specs(
        [
            ["qwen35_27b", "Qwen3.5-27B", "#4C78A8"],
        ]
    )
    assert specs is not None
    plot_path = tmp_path / "concept_arc_100_model_accuracy_comparison.pdf"
    axis_text_calls: list[tuple[str, Any]] = []
    y_label_calls: list[tuple[str, Any]] = []
    title_calls: list[tuple[str, Any]] = []
    original_axis_text = Axes.text
    original_set_ylabel = Axes.set_ylabel
    original_set_title = Axes.set_title

    def capture_axis_text(
        axis: Axes,
        x: float,
        y: float,
        text: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        axis_text_calls.append((text, kwargs.get("fontsize")))
        return original_axis_text(axis, x, y, text, *args, **kwargs)

    def capture_ylabel(
        axis: Axes,
        label: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        y_label_calls.append((label, kwargs.get("fontsize")))
        return original_set_ylabel(axis, label, *args, **kwargs)

    def capture_title(
        axis: Axes,
        label: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        title_calls.append((label, kwargs.get("fontsize")))
        return original_set_title(axis, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", capture_axis_text)
    monkeypatch.setattr(Axes, "set_ylabel", capture_ylabel)
    monkeypatch.setattr(Axes, "set_title", capture_title)

    plot_model_metric_comparison(
        aggregates=[
            {
                "model_name": "qwen35_27b",
                "overall_scalar_stats": {"accuracy": {"mean": 0.4, "stdev": 0.03}},
            },
            {
                "model_name": "unselected_model",
                "overall_scalar_stats": {"accuracy": {"mean": 0.8, "stdev": 0.01}},
            },
        ],
        metric_name="accuracy",
        output_path=plot_path,
        dpi=80,
        dataset_labels=["concept_arc_100"],
        plot_model_specs=specs,
    )

    assert plot_path.read_bytes().startswith(b"%PDF")
    svg_text = plot_path.with_suffix(".svg").read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "Performance Comparison" in svg_text
    assert "Qwen3.5-27B" in svg_text
    assert json.loads(plot_path.with_suffix(".json").read_text(encoding="utf-8")) == {
        "title": "Performance Comparison",
        "dataset_labels": ["concept_arc_100"],
        "metric_name": "pass@1",
        "internal_metric_name": "accuracy",
        "sources": [
            {
                "model_name": "qwen35_27b",
                "display_label": "Qwen3.5-27B",
                "color": "#4C78A8",
                "pass@1": 0.4,
                "stdev": 0.03,
                "error_bar_type": "seed_stdev",
            },
        ],
    }
    assert y_label_calls == [("pass@1", 14)]
    assert title_calls == [("Performance Comparison", 16)]
    assert ("0.400", 11) in axis_text_calls


def _write_eval_summary(
    dataset_root: Path,
    dataset_label: str,
    model_name: str,
    seed: int,
    accuracy: float,
) -> None:
    """Write a minimal eval summary for dataset-comparison plot tests.

    Args:
        dataset_root (Path): Output dataset directory.
        dataset_label (str): Dataset label used in filenames.
        model_name (str): Model folder and filename label.
        seed (int): Run seed.
        accuracy (float): Query accuracy, in increments of 0.1.
    """
    seed_dir = dataset_root / model_name / "eval" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = seed_dir / (
        f"{dataset_label}_{model_name}_seed_{seed}_generations_eval_summary.json"
    )
    summary = {
        "overall": {
            "accuracy": accuracy,
            "n": 10,
            "n_correct": round(10 * accuracy),
        },
        "per_task": {
            "task-a": {
                "accuracy": accuracy,
                "n": 10,
                "n_correct": round(10 * accuracy),
            },
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def test_eval_summary_cli_writes_scores_and_pdf(
    tmp_path: Path,
) -> None:
    """Exercise the documented CLI and verify its plot, JSON, and CSV outputs.

    Args:
        tmp_path (Path): Temporary output root.
    """
    dataset_root = tmp_path / "arc_agi_1"
    for seed, accuracy in [(1, 0.4), (2, 0.6)]:
        _write_eval_summary(
            dataset_root,
            "concept_arc_100",
            "qwen35_27b",
            seed,
            accuracy,
        )
    eval_folder = dataset_root / "qwen35_27b" / "eval"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.summarize_eval_seed_stats",
            "--input_folders",
            str(eval_folder),
            "--plot_format",
            "pdf",
            "--no_title",
            "--plot_model",
            "qwen35_27b",
            "Qwen3.5-27B",
            "#4C78A8",
            "--verbose",
            "0",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    plot_path = (
        eval_folder / "global_summary" / "concept_arc_100_model_accuracy_comparison.pdf"
    )
    assert plot_path.read_bytes().startswith(b"%PDF")
    assert "Performance Comparison" not in plot_path.with_suffix(".svg").read_text(
        encoding="utf-8"
    )
    sidecar = json.loads(plot_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["title"] is None
    output_dir = eval_folder / "global_summary"
    report = json.loads(
        (output_dir / "concept_arc_100_overall_seed_stats.json").read_text()
    )
    tables = report["model"]["task_query_analysis"]
    assert [row["solved_queries"] for row in tables["task_query_counts"]] == [
        "4/10",
        "6/10",
    ]
    assert len(tables["seed_query_distribution"]) == 20
    stats = tables["query_distribution_seed_stats"]
    assert [row["fraction_mean"] for row in stats] == [1.0] * 4 + [0.5] * 2 + [0.0] * 4
    assert stats[3]["fraction_stdev"] == stats[6]["fraction_stdev"] == 0.0
    assert stats[4]["fraction_stdev"] == pytest.approx(stdev([1.0, 0.0]))
    for table_name, rows in tables.items():
        csv_path = output_dir / f"concept_arc_100_{table_name}_qwen35_27b.csv"
        with csv_path.open(newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        assert csv_rows == [
            {key: str(value) for key, value in row.items()} for row in rows
        ]


def test_eval_dataset_comparison_filters_common_models(tmp_path: Path) -> None:
    """Compare only model variants that appear in every dataset root."""
    dataset_labels = ["concept_arc_10", "concept_arc_30", "concept_arc_40"]
    dataset_roots = [tmp_path / dataset_label for dataset_label in dataset_labels]
    for dataset_root, dataset_label in zip(
        dataset_roots,
        dataset_labels,
        strict=True,
    ):
        _write_eval_summary(dataset_root, dataset_label, "qwen35_27b", 1, 0.5)
        _write_eval_summary(dataset_root, dataset_label, "qwen35_27b", 2, 0.7)

    _write_eval_summary(dataset_roots[0], dataset_labels[0], "qwen3_8b", 1, 0.2)
    _write_eval_summary(dataset_roots[1], dataset_labels[1], "qwen3_8b", 1, 0.3)
    _write_eval_summary(dataset_roots[2], dataset_labels[2], "qwen3_32b", 1, 0.4)

    comparisons = load_dataset_comparisons(
        dataset_roots,
        None,
        "*_eval_summary.json",
    )
    common_model_names = get_common_model_names(comparisons)
    skipped_by_dataset = get_skipped_model_names_by_dataset(
        comparisons,
        common_model_names,
    )

    assert [comparison.dataset_label for comparison in comparisons] == dataset_labels
    assert [comparison.dataset_display_label for comparison in comparisons] == [
        "ConceptARC",
        "ConceptARC-30",
        "ConceptARC-40",
    ]
    assert common_model_names == ["qwen35_27b"]
    assert skipped_by_dataset == {
        "concept_arc_10": ["qwen3_8b"],
        "concept_arc_30": ["qwen3_8b"],
        "concept_arc_40": ["qwen3_32b"],
    }


def test_eval_dataset_comparison_formats_concept_arc_labels() -> None:
    """Use compact ConceptARC display labels for comparison plots."""
    assert format_dataset_display_label("concept_arc_10") == "ConceptARC"
    assert format_dataset_display_label("concept_arc_40") == "ConceptARC-40"
    assert format_dataset_display_label("concept_arc_50") == "ConceptARC-50"
    assert format_dataset_display_label("concept_arc_100_8") == "ConceptARC-100-8"


def test_eval_dataset_comparison_uses_dataset_shades() -> None:
    """Represent datasets as different shades of the same model hue."""
    light_blue = shade_model_color("#1f77b4", 0, 2)
    dark_blue = shade_model_color("#1f77b4", 1, 2)
    light_gray = get_dataset_legend_color(0, 2)
    dark_gray = get_dataset_legend_color(1, 2)

    assert light_blue != dark_blue
    assert max(light_blue) > max(dark_blue)
    assert light_gray != dark_gray
    assert light_gray[0] == light_gray[1] == light_gray[2]
    assert dark_gray[0] == dark_gray[1] == dark_gray[2]
    assert light_gray[0] > dark_gray[0]


def test_eval_dataset_comparison_plot_writes_grouped_png(tmp_path: Path) -> None:
    """Write a grouped dataset-comparison plot with seed-level stdevs."""
    dataset_labels = ["concept_arc_10", "concept_arc_40"]
    dataset_roots = [tmp_path / dataset_label for dataset_label in dataset_labels]
    for seed, accuracy in [(1, 0.3), (2, 0.4)]:
        _write_eval_summary(
            dataset_roots[0],
            dataset_labels[0],
            "qwen35_27b",
            seed,
            accuracy,
        )
    for seed, accuracy in [(1, 0.4), (2, 0.5)]:
        _write_eval_summary(
            dataset_roots[1],
            dataset_labels[1],
            "qwen35_27b",
            seed,
            accuracy,
        )
    for seed, accuracy in [(1, 0.8), (2, 0.9)]:
        _write_eval_summary(
            dataset_roots[0],
            dataset_labels[0],
            "qwen3_8b",
            seed,
            accuracy,
        )
    for seed, accuracy in [(1, 0.7), (2, 0.8)]:
        _write_eval_summary(
            dataset_roots[1],
            dataset_labels[1],
            "qwen3_8b",
            seed,
            accuracy,
        )

    comparisons = load_dataset_comparisons(
        dataset_roots,
        ["10 examples", "40 examples"],
        "*_eval_summary.json",
    )
    common_model_names = get_common_model_names(comparisons)
    sorted_model_names = sort_model_names_by_average_metric(
        comparisons,
        common_model_names,
        "accuracy",
    )
    plot_path = tmp_path / "comparison.png"
    plot_eval_dataset_comparison(
        comparisons,
        sorted_model_names,
        "accuracy",
        plot_path,
        80,
    )

    assert sorted_model_names == ["qwen35_27b", "qwen3_8b"]
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def _write_concept_eval_summary(
    dataset_root: Path,
    dataset_label: str,
    model_name: str,
    seed: int,
    concept_accuracies: dict[str, float],
) -> None:
    """Write a minimal per-concept eval summary for concept-plot tests."""
    seed_dir = dataset_root / model_name / "eval" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = seed_dir / (
        f"{dataset_label}_{model_name}_seed_{seed}_generations_eval_summary.json"
    )
    total_accuracy = sum(concept_accuracies.values()) / len(concept_accuracies)
    summary = {
        "overall": {
            "accuracy": total_accuracy,
            "n": len(concept_accuracies),
        },
        "per_concept": {
            concept_name: {
                "accuracy": accuracy,
                "n": 1,
            }
            for concept_name, accuracy in concept_accuracies.items()
        },
        "per_task": {
            f"{concept_name}/task-a": {
                "accuracy": accuracy,
            }
            for concept_name, accuracy in concept_accuracies.items()
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def test_overall_comparison_cli_defaults_to_dedicated_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write the overall plot to a dedicated folder beside the first root."""
    roots = [tmp_path / "concept_arc_20", tmp_path / "concept_arc_40"]
    for root, label, accuracy in zip(
        roots,
        ("concept_arc_20", "concept_arc_40"),
        (0.4, 0.6),
        strict=True,
    ):
        _write_eval_summary(root, label, "qwen3_8b", 42, accuracy)
    output_dir = tmp_path / "dataset_comparison"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_eval_dataset_comparison.py",
            "--dataset_roots",
            *(str(root) for root in roots),
            "--dpi",
            "30",
            "--verbose",
            "0",
        ],
    )

    run_dataset_comparison()

    assert (output_dir / "dataset_model_accuracy_comparison.png").stat().st_size > 0


def test_eval_concept_performance_aggregates_and_sorts(tmp_path: Path) -> None:
    """Aggregate per-concept model performance across seed summaries."""
    dataset_root = tmp_path / "concept_arc_40"
    _write_concept_eval_summary(
        dataset_root,
        "concept_arc_40",
        "qwen35_27b",
        1,
        {"AboveBelow": 0.6, "Count": 0.2, "Copy": 0.8},
    )
    _write_concept_eval_summary(
        dataset_root,
        "concept_arc_40",
        "qwen35_27b",
        2,
        {"AboveBelow": 0.8, "Count": 0.4, "Copy": 1.0},
    )
    _write_concept_eval_summary(
        dataset_root,
        "concept_arc_40",
        "qwen3_8b",
        1,
        {"AboveBelow": 0.4, "Count": 0.1, "Copy": 0.5},
    )
    _write_concept_eval_summary(
        dataset_root,
        "concept_arc_40",
        "qwen3_8b",
        2,
        {"AboveBelow": 0.6, "Count": 0.3, "Copy": 0.7},
    )
    for model_name, concept_accuracies in {
        "qwen35_9b": {"AboveBelow": 0.5, "Count": 0.2, "Copy": 0.7},
        "gemma4_31b": {"AboveBelow": 0.9, "Count": 0.8, "Copy": 0.95},
        "qwen3_32b": {"AboveBelow": 0.2, "Count": 0.05, "Copy": 0.3},
    }.items():
        _write_concept_eval_summary(
            dataset_root,
            "concept_arc_40",
            model_name,
            1,
            concept_accuracies,
        )

    performances = load_concept_performances(
        dataset_root,
        "accuracy",
        "*_eval_summary.json",
    )
    common_concepts = get_common_concept_names(performances)
    sorted_concepts = sort_concept_names(
        performances,
        common_concepts,
        "average_desc",
    )
    sorted_models = sort_models_by_average_concept_metric(
        performances,
        sorted_concepts,
    )

    assert common_concepts == ["AboveBelow", "Copy", "Count"]
    assert sorted_concepts == ["Copy", "AboveBelow", "Count"]
    assert [performance.model_name for performance in sorted_models] == [
        "gemma4_31b",
        "qwen35_27b",
        "qwen35_9b",
        "qwen3_32b",
        "qwen3_8b",
    ]
    qwen27 = next(
        performance
        for performance in performances
        if performance.model_name == "qwen35_27b"
    )
    assert qwen27.concept_stats["AboveBelow"].mean == 0.7
    assert qwen27.concept_stats["AboveBelow"].n_seed_runs == 2
    assert get_model_legend_ncols(4) == 4
    assert get_model_legend_ncols(5) == 5
    assert get_concept_arc_dataset_shade_index("concept_arc_10") == 0
    assert get_concept_arc_dataset_shade_index("concept_arc_40") == 1
    assert get_concept_arc_dataset_shade_index("arc_agi_1") is None
    assert get_default_concept_performance_output_path(
        dataset_root,
        "accuracy",
    ) == (dataset_root / "concept_model_accuracy_comparison.png")


def test_group_accuracy_averages_concepts_before_seeds(tmp_path: Path) -> None:
    """Verify seed-level cancellation and omit groups missing a constituent concept."""
    from arc_cd.repr_analysis.population import CONCEPT_GROUPS

    dataset = "concept_arc_100"
    root = tmp_path / dataset
    for offset, seed in enumerate((42, 44)):
        values = {
            name: float((i + offset) % 2)
            for i, name in enumerate(CONCEPT_GROUPS["predicate"])
        }
        values["Copy"] = 1.0
        _write_concept_eval_summary(root, dataset, "qwen3_8b", seed, values)
    (model,) = load_concept_performances(root, "accuracy")
    assert set(model.group_stats) == {"predicate"}
    assert model.group_stats["predicate"].mean == 0.5
    assert model.group_stats["predicate"].stdev == 0.0
    assert model.concept_stats["AboveBelow"].stdev > 0.0


def test_eval_concept_performance_writes_five_seed_summary(tmp_path: Path) -> None:
    """Write overall and per-concept statistics across five seeds."""
    dataset_root = tmp_path / "concept_arc_100"
    dataset_label = "concept_arc_100"
    values_by_seed = {
        42: {"AboveBelow": 0.2, "Count": 0.4},
        44: {"AboveBelow": 0.4, "Count": 0.4},
        46: {"AboveBelow": 0.6, "Count": 0.6},
        48: {"AboveBelow": 0.8, "Count": 0.6},
        50: {"AboveBelow": 1.0, "Count": 0.8},
    }
    for seed, concept_accuracies in values_by_seed.items():
        _write_concept_eval_summary(
            dataset_root,
            dataset_label,
            "qwen35_27b",
            seed,
            concept_accuracies,
        )

    summary_glob = "concept_arc_100_*_eval_summary.json"
    performances = load_concept_performances(
        dataset_root,
        "accuracy",
        summary_glob,
    )
    concept_names = get_common_concept_names(performances)
    summary_path = tmp_path / "custom_summary.json"
    payload = write_concept_performance_summary(
        dataset_root,
        performances,
        concept_names,
        "accuracy",
        summary_glob,
        summary_path,
    )

    above_below_values = [
        concept_accuracies["AboveBelow"]
        for concept_accuracies in values_by_seed.values()
    ]
    overall_values = [
        mean(concept_accuracies.values())
        for concept_accuracies in values_by_seed.values()
    ]
    model_summary = payload["models"][0]
    assert payload["dataset_label"] == dataset_label
    assert payload["metric_name"] == "accuracy"
    assert payload["plotted_concepts"] == ["AboveBelow", "Count"]
    assert model_summary["base_model_name"] == "qwen35_27b"
    assert model_summary["reasoning_effort"] is None
    assert model_summary["seeds"] == [42, 44, 46, 48, 50]
    assert model_summary["n_seed_runs"] == 5
    assert model_summary["overall"] == {
        "mean": mean(overall_values),
        "stdev": stdev(overall_values),
        "n_seed_runs": 5,
    }
    assert model_summary["per_concept"]["AboveBelow"] == {
        "mean": mean(above_below_values),
        "stdev": stdev(above_below_values),
        "n_seed_runs": 5,
    }
    assert summary_path.stat().st_size > 0
    assert json.loads(summary_path.read_text(encoding="utf-8")) == payload
    assert get_default_concept_performance_summary_output_path(
        tmp_path / "concepts.png"
    ) == (tmp_path / "concepts.json")
    assert format_dataset_display_label(dataset_label) == ("ConceptARC-100")


def test_eval_concept_performance_keeps_different_seed_sets(
    tmp_path: Path,
    caplog: Any,
) -> None:
    """Keep every available seed and warn when model seed sets differ."""
    dataset_root = tmp_path / "concept_arc_100"
    dataset_label = "concept_arc_100"
    for seed, accuracy in [(42, 0.4), (44, 0.8)]:
        _write_concept_eval_summary(
            dataset_root,
            dataset_label,
            "qwen35_27b",
            seed,
            {"Count": accuracy},
        )
    _write_concept_eval_summary(
        dataset_root,
        dataset_label,
        "qwen3_8b",
        42,
        {"Count": 0.5},
    )

    performances = load_concept_performances(
        dataset_root,
        "accuracy",
        "*_eval_summary.json",
    )
    caplog.set_level(logging.WARNING)
    warn_if_model_seed_sets_differ(performances)
    payload = write_concept_performance_summary(
        dataset_root,
        performances,
        ["Count"],
        "accuracy",
        "*_eval_summary.json",
        tmp_path / "different_seeds.json",
    )

    models_by_name = {
        model_summary["model_name"]: model_summary
        for model_summary in payload["models"]
    }
    assert models_by_name["qwen35_27b"]["seeds"] == [42, 44]
    assert models_by_name["qwen3_8b"]["seeds"] == [42]
    assert models_by_name["qwen3_8b"]["overall"]["stdev"] == 0.0
    assert "Models use different seed sets" in caplog.text


def test_eval_concept_performance_rejects_missing_metric(tmp_path: Path) -> None:
    """Reject a per-concept summary that lacks the selected metric."""
    dataset_root = tmp_path / "concept_arc_100"
    _write_concept_eval_summary(
        dataset_root,
        "concept_arc_100",
        "qwen35_27b",
        42,
        {"Count": 0.5},
    )
    summary_path = next(dataset_root.glob("**/*_eval_summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    del summary["per_concept"]["Count"]["accuracy"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing numeric per_concept metric"):
        load_concept_performances(
            dataset_root,
            "accuracy",
            "*_eval_summary.json",
        )


def test_eval_concept_performance_plot_writes_png(tmp_path: Path) -> None:
    """Write a grouped concept-performance plot with model bars."""
    dataset_root = tmp_path / "concept_arc_40"
    _write_concept_eval_summary(
        dataset_root,
        "concept_arc_40",
        "qwen35_27b",
        1,
        {"AboveBelow": 0.6, "Count": 0.2},
    )
    _write_concept_eval_summary(
        dataset_root,
        "concept_arc_40",
        "qwen35_27b",
        2,
        {"AboveBelow": 0.8, "Count": 0.4},
    )
    _write_concept_eval_summary(
        dataset_root,
        "concept_arc_40",
        "qwen3_8b",
        1,
        {"AboveBelow": 0.4, "Count": 0.1},
    )
    _write_concept_eval_summary(
        dataset_root,
        "concept_arc_40",
        "qwen3_8b",
        2,
        {"AboveBelow": 0.6, "Count": 0.3},
    )

    performances = load_concept_performances(
        dataset_root,
        "accuracy",
        "*_eval_summary.json",
    )
    common_concepts = get_common_concept_names(performances)
    sorted_models = sort_models_by_average_concept_metric(
        performances,
        common_concepts,
    )
    plot_path = tmp_path / "concepts.png"
    plot_concept_performance(
        sorted_models,
        common_concepts,
        "accuracy",
        plot_path,
        80,
        "ConceptARC-40",
    )

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_eval_concept_performance_uses_four_by_four_layout(tmp_path: Path) -> None:
    """Arrange the 16 canonical concepts in a near-square plot grid."""
    dataset_root = tmp_path / "concept_arc_100"
    _write_concept_eval_summary(
        dataset_root,
        "concept_arc_100",
        "qwen35_27b",
        42,
        {
            concept_name: 0.25 + concept_index / 100
            for concept_index, concept_name in enumerate(CANONICAL_CONCEPT_LABELS)
        },
    )
    performances = load_concept_performances(
        dataset_root,
        "accuracy",
        "*_eval_summary.json",
    )
    plot_path = tmp_path / "concept_grid.png"

    plot_concept_performance(
        performances,
        list(CANONICAL_CONCEPT_LABELS),
        "accuracy",
        plot_path,
        40,
        "ConceptARC-100",
    )

    png_header = plot_path.read_bytes()[:24]
    assert png_header[:8] == b"\x89PNG\r\n\x1a\n"
    width = int.from_bytes(png_header[16:20], byteorder="big")
    height = int.from_bytes(png_header[20:24], byteorder="big")
    assert 0.9 <= width / height <= 1.5
