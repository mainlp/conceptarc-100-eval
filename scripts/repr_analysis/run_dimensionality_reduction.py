"""Build UMAP/PCA task maps and a standalone interactive support-grid explorer."""

from arc_cd.repr_analysis.embedding_runner import run_embedding


def main() -> None:
    """Run dimensionality reduction using the shared feature-bundle interface."""
    run_embedding()


if __name__ == "__main__":
    main()
