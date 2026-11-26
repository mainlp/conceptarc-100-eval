"""Validate cumulative CSV interpretation and model-comparison artifacts."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from scripts.plot_robustness import (
    RobustnessCurve,
    load_reference_json,
    load_robustness_csv,
    plot_robustness,
)


def write_stats_csv(path: Path, means: list[float], n_seeds: int = 2) -> None:
    """Write a small cumulative summary fixture.

    Args:
        path (Path): Destination CSV.
        means (list[float]): Mean proportions ordered by threshold.
        n_seeds (int): Number of seed runs.
    """
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "n_queries",
                "min_n_correct",
                "n_seed_runs",
                "fraction_mean",
                "fraction_stdev",
            ]
        )
        for threshold, average in reversed(list(enumerate(means, start=1))):
            writer.writerow(
                [len(means), threshold, n_seeds, average, 0.1 if n_seeds > 1 else 0]
            )


def test_load_robustness_orders_thresholds_and_selects_query_total(
    tmp_path: Path,
) -> None:
    """Select one denominator and preserve the saved across-seed means.

    Args:
        tmp_path (Path): Temporary output directory.
    """
    path = tmp_path / "concept_arc_100_query_distribution_seed_stats_gemma4_31b.csv"
    means = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25]
    write_stats_csv(path, means)
    with path.open("a") as stream:
        stream.write("1,1,2,0.9,0.1\n")
    curve = load_robustness_csv(path)
    assert curve.label == "gemma4 31b"
    assert curve.means == tuple(means)
    assert curve.stdevs == (0.1,) * 10
    assert curve.n_seed_runs == 2


@pytest.mark.parametrize(
    "rows,error",
    [
        ("2,1,2,0.3,0.1\n2,2,2,0.8,0.1\n", "must not increase"),
        ("2,1,2,0.8,0.1\n", "every threshold"),
        ("2,1,2,0.8,0.1\n2,1,2,0.5,0.1\n", "duplicate"),
        ("2,0,2,1.0,0.1\n2,1,2,0.8,0.1\n", "invalid threshold"),
        ("2,1,2,nan,0.1\n2,2,2,0.5,0.1\n", "invalid proportion"),
        ("2,1,2,1.1,0.1\n2,2,2,0.5,0.1\n", "invalid proportion"),
        ("2,1,2,0.8,-0.1\n2,2,2,0.5,0.1\n", "invalid proportion"),
        ("2,1,2,0.8,0.1\n2,2,3,0.5,0.1\n", "seed counts differ"),
    ],
)
def test_load_robustness_rejects_invalid_statistics(
    tmp_path: Path, rows: str, error: str
) -> None:
    """Reject incomplete or inconsistent scientific plotting data.

    Args:
        tmp_path (Path): Temporary input directory.
        rows (str): Malformed CSV data rows.
        error (str): Expected error message fragment.
    """
    path = tmp_path / "invalid.csv"
    path.write_text(
        "n_queries,min_n_correct,n_seed_runs,fraction_mean,fraction_stdev\n" + rows
    )
    with pytest.raises(ValueError, match=error):
        load_robustness_csv(path, n_queries=2)


def test_load_robustness_rejects_old_exact_count_csv(tmp_path: Path) -> None:
    """Avoid silently presenting exact-bin fractions as cumulative values.

    Args:
        tmp_path (Path): Temporary input directory.
    """
    path = tmp_path / "old.csv"
    path.write_text(
        "n_queries,n_correct,n_seed_runs,fraction_mean,fraction_stdev\n10,1,1,0.2,0\n"
    )
    with pytest.raises(ValueError, match="Rerun summarize_eval_seed_stats.py"):
        load_robustness_csv(path)


def test_plot_robustness_model_lines_and_std_bands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check plotted values, model legend, and optional uncertainty semantics.

    Args:
        tmp_path (Path): Temporary plot directory.
        monkeypatch (pytest.MonkeyPatch): Figure capture fixture.
    """
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    write_stats_csv(paths[0], [0.95, 0.6, 0.3])
    write_stats_csv(paths[1], [0.7, 0.4, 0.1], n_seeds=1)
    curves = [
        load_robustness_csv(path, 3, label)
        for path, label in zip(paths, ["First", "Second"])
    ]
    figures: list[Figure] = []
    monkeypatch.setattr(
        Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure)
    )
    plot_robustness(curves, tmp_path / "comparison.pdf", show_std=True)
    axes = figures[0].axes[0]
    assert axes.get_title() == ""
    assert axes.get_xlabel() == "Minimum correct queries per task"
    assert axes.get_ylabel() == "Proportion of tasks"
    assert [tick.get_text() for tick in axes.get_xticklabels()] == ["1/3", "2/3", "3/3"]
    legend = axes.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["First", "Second"]
    assert len(axes.lines) == 2
    assert axes.lines[0].get_color() != axes.lines[1].get_color()
    for line, curve in zip(axes.lines, curves):
        assert line.get_ydata() == pytest.approx(curve.means)
    assert len(axes.collections) == 1  # The single-seed curve has no band.


def test_plot_robustness_cli_writes_pdf_and_png(tmp_path: Path) -> None:
    """Exercise multi-model input and headless vector/raster export through CLI.

    Args:
        tmp_path (Path): Temporary input and output directory.
    """
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    write_stats_csv(first, [0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0])
    write_stats_csv(second, [1.0] * 10)
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps([{"label": "Reference", "mean_percent": [10] * 10}])
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_robustness.py",
            "--input_csvs",
            str(first),
            str(second),
            "--labels",
            "Model A",
            "Model B",
            "--show_std",
            "--reference_json",
            str(reference),
            "--colors",
            "#0072B2",
            "#D55E00",
            "#8C564B",
            "--output_path",
            str(tmp_path / "figure.pdf"),
            "--dpi",
            "72",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "MPLCONFIGDIR": str(tmp_path / "mplconfig"),
            "MPLBACKEND": "Agg",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "figure.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path / "figure.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_reference_curves_preserve_units_uncertainty_and_existing_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding reference models must retain colors and distinguish missing SDs.

    Args:
        tmp_path (Path): Temporary reference and plot directory.
        monkeypatch (pytest.MonkeyPatch): Figure capture fixture.
    """
    path = tmp_path / "references.json"
    path.write_text(
        json.dumps(
            [
                {"label": "TRM", "mean_percent": [35.44, 17.38, 8.69]},
                {
                    "label": "VARC",
                    "mean_percent": [53.667, 37.000, 25.479],
                    "sd_percentage_points": [0.899, 0.821, 1.225],
                },
            ]
        )
    )
    references = load_reference_json(path, n_queries=3)
    trm, varc = references
    assert trm.means == pytest.approx([0.3544, 0.1738, 0.0869])
    assert trm.stdevs is None
    assert varc.means == pytest.approx([0.53667, 0.37, 0.25479])
    assert varc.stdevs == pytest.approx([0.00899, 0.00821, 0.01225])
    assert trm.n_seed_runs is None and varc.n_seed_runs is None
    curves = [
        RobustnessCurve(f"Model {index}", (0.9, 0.5, 0.1), (0.01,) * 3, 3)
        for index in range(8)
    ]
    figures: list[Figure] = []
    monkeypatch.setattr(
        Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure)
    )
    plot_robustness(curves, tmp_path / "original.pdf", show_std=True)
    original_colors = [line.get_color() for line in figures[0].axes[0].lines]
    plot_robustness(curves + references, tmp_path / "extended.pdf", show_std=True)
    axes = figures[-1].axes[0]
    colors = [line.get_color() for line in axes.lines]
    assert colors[:8] == original_colors
    assert len(set(colors)) == 10
    assert axes.lines[-2].get_ydata() == pytest.approx(trm.means)
    assert axes.lines[-1].get_ydata() == pytest.approx(varc.means)
    assert len(axes.collections) == 9  # Eight seed SDs and the supplied VARC SD.
    band_vertices = axes.collections[-1].get_paths()[0].vertices
    first_threshold = band_vertices[band_vertices[:, 0] == 1, 1]
    assert first_threshold.min() == pytest.approx(0.53667 - 0.00899)
    assert first_threshold.max() == pytest.approx(0.53667 + 0.00899)


@pytest.mark.parametrize(
    "reference,error",
    [
        ({"label": "VARC", "mean_percent": [0.5]}, "must contain 2 values"),
        ({"label": "VARC", "mean_percent": [50, 60]}, "must not increase"),
        ({"label": "VARC", "mean_percent": [101, 20]}, "numbers in"),
        (
            {
                "label": "VARC",
                "mean_percent": [50, 20],
                "sd_percentage_points": [-1, 2],
            },
            "numbers in",
        ),
        (
            {"label": "VARC", "mean_percent": [50, 20], "sd_percentage_points": [1]},
            "must contain 2 values",
        ),
    ],
)
def test_reference_curves_reject_invalid_statistics(
    tmp_path: Path, reference: dict[str, object], error: str
) -> None:
    """Reject reference values that would misrepresent cumulative robustness.

    Args:
        tmp_path (Path): Temporary input directory.
        reference (dict[str, object]): Malformed reference curve.
        error (str): Expected error message fragment.
    """
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([reference]))
    with pytest.raises(ValueError, match=error):
        load_reference_json(path, n_queries=2)
