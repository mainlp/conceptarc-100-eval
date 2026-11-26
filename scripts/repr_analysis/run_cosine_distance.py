"""Analyze cosine or Euclidean separation of LRM support features or task centroids."""

from arc_cd.repr_analysis.runner import run_analysis


def main() -> None:
    """Run this analysis with shared command-line options."""
    run_analysis("cosine_distance")


if __name__ == "__main__":
    main()
