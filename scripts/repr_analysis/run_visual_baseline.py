"""Extract visual features and evaluate cosine or Euclidean task separation."""

from arc_cd.repr_analysis.visual_runner import run_visual_baseline


def main() -> None:
    """Run the visual baseline using the shared family-separation analysis."""
    run_visual_baseline()


if __name__ == "__main__":
    main()
