"""Merge ConceptARC support-feature shards, optionally extending an old bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from arc_cd.probes.prompt_feature_io import merge_prompt_feature_shards


def main() -> None:
    """Validate and merge concept shards without overwriting existing output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--shard_dir", type=Path, required=True)
    parser.add_argument(
        "--base_feature_dir",
        type=Path,
        default=None,
        help="Existing complete bundle for unchanged concepts; shards supply new concepts.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Final directory; required with --base_feature_dir, otherwise defaults "
            "to sibling features/."
        ),
    )
    args = parser.parse_args()
    if args.base_feature_dir is not None and args.output_dir is None:
        parser.error("--output_dir is required with --base_feature_dir.")
    output = merge_prompt_feature_shards(
        args.data_dir.expanduser().resolve(),
        args.shard_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve() if args.output_dir is not None else None,
        base_feature_dir=(
            args.base_feature_dir.expanduser().resolve()
            if args.base_feature_dir is not None
            else None
        ),
    )
    print(f"Merged S1/S2/S3 features into {output}")


if __name__ == "__main__":
    main()
