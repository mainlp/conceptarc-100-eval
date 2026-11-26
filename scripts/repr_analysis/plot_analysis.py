"""Regenerate plots and offline galleries from a saved sub-analysis directory."""

import argparse
from pathlib import Path

from arc_cd.repr_analysis.visualization import plot_analysis


def main() -> None:
    """Parse a saved analysis directory and regenerate its visualizations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis_dir", type=Path, required=True)
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also export static figures as PDFs (automatic for metadata cosine-distance plots).",
    )
    args = parser.parse_args()
    try:
        plot_analysis(
            args.analysis_dir.expanduser().resolve(),
            pdf=args.pdf,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
