"""
Evaluate model responses for ARC-style data.
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from arc_cd.eval.eval_response import evaluate_records
from arc_cd.eval.parsing import extract_final_answer
from arc_cd.utils import (
    DEFAULT_MODEL_CONFIG_DIR,
    load_jsonl,
    resolve_reasoning_end_marker,
    save_json,
    setup_logging,
    write_jsonl,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluate model responses (accuracy) for ARC-style data."
    )

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    # Data configs
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to jsonl response file.",
    )
    parser.add_argument(
        "--allow_non_arc_colors",
        action="store_true",
        help="Allow parsed grid values outside [0, 9].",
    )

    # Model configs
    parser.add_argument(
        "--model_config_dir",
        type=str,
        default=str(DEFAULT_MODEL_CONFIG_DIR),
        help="Fallback model YAML directory for runs without a saved reasoning marker.",
    )

    return parser.parse_args()


def get_eval_output_paths(
    input_jsonl: str | Path,
) -> dict[str, Path]:
    """Use the run's model and seed layout, or a sibling eval folder for loose files.

    Args:
        input_jsonl (str | Path): Path to the generation JSONL.

    Returns:
        dict[str, Path]: Paths for the summary, per-case scores and error cases.
    """
    input_path = Path(input_jsonl).expanduser().resolve()

    seed_dir = input_path.parent
    if seed_dir.parent.name in {"generations", "truncated_generations"}:
        output_dir = seed_dir.parent.parent / "eval" / seed_dir.name
    else:
        output_dir = seed_dir / "eval"
    run_name = input_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "summary": output_dir / f"{run_name}_eval_summary.json",
        "cases": output_dir / f"{run_name}_eval_cases.jsonl",
        "parse_errors": output_dir / f"{run_name}_parse_error_cases.jsonl",
        "unfinished": output_dir / f"{run_name}_unfinished_cases.jsonl",
        "model_errors": output_dir / f"{run_name}_model_error_cases.jsonl",
    }


def evaluate_generation_file(
    input_jsonl: str | Path,
    end_think_token: str | None = None,
    require_arc_colors: bool = True,
    model_config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR,
) -> dict[str, Any]:
    """Evaluate a generation JSONL file and write evaluation outputs.

    Output files are named using the input JSONL stem. For example, evaluating
    ``model_generations.jsonl`` writes:

        model_generations_eval_summary.json
        model_generations_eval_cases.jsonl
        model_generations_parse_error_cases.jsonl
        model_generations_unfinished_cases.jsonl
        model_generations_model_error_cases.jsonl

    Args:
        input_jsonl: Path to generation JSONL.
        end_think_token: Optional override for the run's saved reasoning end marker.
        require_arc_colors: Whether predicted grid values must be in [0, 9].
        model_config_dir: Directory containing model YAML configs.

    Returns:
        Summary metrics.
    """
    if end_think_token is None:
        end_think_token = resolve_reasoning_end_marker(
            input_jsonl,
            model_config_dir=model_config_dir,
        )

    records = load_jsonl(input_jsonl)
    evaluations, summary = evaluate_records(
        records,
        end_think_token=end_think_token,
        require_arc_colors=require_arc_colors,
    )

    output_paths = get_eval_output_paths(
        input_jsonl=input_jsonl,
    )

    save_json(summary, output_paths["summary"])
    write_jsonl([asdict(case) for case in evaluations], output_paths["cases"])

    parse_error_records = []
    unfinished_records = []
    model_error_records = []
    for case, record in zip(evaluations, records, strict=True):
        record_with_text = {
            **asdict(case),
            "generated_text": record.get("generated_text", ""),
            "final_answer": extract_final_answer(
                record.get("generated_text", ""),
                end_think_token=end_think_token,
            ),
        }
        if case.response_finished and not case.parse_ok:
            parse_error_records.append(record_with_text)
        if not case.response_finished:
            unfinished_records.append(record_with_text)
        if case.response_finished and case.parse_ok and not case.correct:
            model_error_records.append(record_with_text)

    write_jsonl(parse_error_records, output_paths["parse_errors"])
    write_jsonl(unfinished_records, output_paths["unfinished"])
    write_jsonl(model_error_records, output_paths["model_errors"])

    return summary


def main() -> None:
    """
    Main execution flow.
    """
    args = parse_arguments()

    setup_logging(args.verbose)

    summary = evaluate_generation_file(
        input_jsonl=args.input_jsonl,
        require_arc_colors=not args.allow_non_arc_colors,
        model_config_dir=args.model_config_dir,
    )
    logging.info(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
