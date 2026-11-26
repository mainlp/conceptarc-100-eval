"""Merge per-concept generations, requiring all concepts unless explicitly relaxed."""

import argparse
import logging

from arc_cd.models.concept_shards import merge_concept_generations
from arc_cd.utils import setup_logging


def main() -> None:
    """Parse merge arguments and write validated generation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="ConceptARC dataset root.")
    parser.add_argument(
        "--shard_dir",
        required=True,
        help="Model generation_shards/seed_<seed> directory containing concept folders.",
    )
    parser.add_argument(
        "--run_name",
        help="Artifact stem before _generations.jsonl; required if multiple runs exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing canonical artifacts after validating all included shards.",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help=(
            "Merge available concepts and record missing concepts. Each included "
            "shard must be complete; existing merged queries cannot be dropped."
        ),
    )
    parser.add_argument(
        "--model_path_alias",
        nargs=2,
        action="append",
        default=[],
        metavar=("SAVED_PATH", "CANONICAL_PATH"),
        help="Declare equivalent checkpoint paths explicitly; repeat for more aliases.",
    )
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    args = parser.parse_args()
    aliases: dict[str, str] = {}
    for saved, canonical in args.model_path_alias:
        if saved in aliases and aliases[saved] != canonical:
            parser.error(f"Conflicting model path aliases for {saved}")
        aliases[saved] = canonical
    setup_logging(args.verbose)
    path = merge_concept_generations(
        args.data_dir,
        args.shard_dir,
        run_name=args.run_name,
        overwrite=args.overwrite,
        allow_partial=args.allow_partial,
        model_path_aliases=aliases,
    )
    logging.info("Merged seed run: %s", path)


if __name__ == "__main__":
    main()
