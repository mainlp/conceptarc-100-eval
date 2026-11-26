"""Export the public ConceptARC-100 test splits as concept-scoped JSON files."""

import argparse
import json
from pathlib import Path

from arc_cd.probes.prompt_feature_io import staged_output_directory

DATASET = "mainlp/ConceptARC-100"
CONFIGS = ("ConceptARC-100", "ConceptARC-100-8", "ConceptARC-100-specs")


def download_config(config: str, root: Path, revision: str | None = None) -> None:
    """Write one dataset configuration in the layout used by the evaluation scripts.

    Args:
        config (str): Hugging Face dataset configuration name.
        root (Path): Parent directory for the exported configuration.
        revision (str | None): Optional Hugging Face dataset revision.
    """
    from datasets import load_dataset

    destination = root / config
    with staged_output_directory(destination) as stage:
        rows = load_dataset(
            DATASET, config, split="test", revision=revision, streaming=True
        )
        count = 0
        for row in rows:
            concept, task = row["concept"], row["task_id"]
            if any(
                name in ("", ".", "..") or Path(name).name != name
                for name in (concept, task)
            ):
                raise ValueError(
                    f"Invalid concept or task identifier: {concept!r}, {task!r}"
                )
            if config == "ConceptARC-100-specs":
                filename = f"{task}.json"
                record = {
                    "task_id": row["source_task_id"],
                    "concept_name": concept,
                    "description": row["description"],
                    "program_kind": row["program_kind"],
                    "program": json.loads(row["program_json"]),
                    "seed": row["seed"],
                }
            else:
                filename = f"{task}_{row['version']:03d}.json"
                record = {"train": row["train"], "test": row["test"]}
            folder = stage / concept
            folder.mkdir(exist_ok=True)
            with (folder / filename).open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            count += 1
        expected = 12_800 if config == "ConceptARC-100-8" else 1_600
        if count != expected:
            raise ValueError(
                f"Expected {expected:,} rows in the {config} test split, received "
                f"{count:,}. No files were published to {destination}."
            )
    print(f"Wrote {count:,} files to {destination}")


def main() -> None:
    """Download the requested configurations into a user-selected directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=CONFIGS,
        default=[CONFIGS[0]],
        help="Configurations to export (default: ConceptARC-100).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Parent directory for the exported configurations (default: data).",
    )
    parser.add_argument("--revision", help="Hugging Face dataset tag or commit.")
    args = parser.parse_args()
    for config in dict.fromkeys(args.configs):
        download_config(config, args.output_dir.expanduser(), args.revision)


if __name__ == "__main__":
    main()
