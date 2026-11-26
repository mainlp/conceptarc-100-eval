"""Annotate an existing representation explorer with task-spec metadata."""

from arc_cd.repr_analysis.metadata_runner import run_metadata_explorer


def main() -> None:
    """Build a separately staged, offline explorer using aligned parent specs."""
    run_metadata_explorer()


if __name__ == "__main__":
    main()
