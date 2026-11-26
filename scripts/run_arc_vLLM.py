"""
Run LRM inference on ARC-style data using vLLM.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from arc_cd.data_prep.concept_arc import ConceptARCTask
from arc_cd.data_prep.dataset_ids import infer_concept_arc_dataset_id
from arc_cd.data_prep.load_data import iter_test_cases, load_concept_arc
from arc_cd.models.config import VLLMReasoningConfig, load_experiment_config
from arc_cd.models.generation_io import write_generation_outputs, write_run_config
from arc_cd.prompt_prep.load_prompt import build_prompt_record
from arc_cd.prompt_prep.prompt_format import PromptTemplates
from arc_cd.utils import (
    DEFAULT_CONCEPT_ARC_PROMPT_DIR,
    DEFAULT_MODEL_CONFIG_DIR,
    set_seed,
    setup_logging,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser("Run inference on ARC-style data via vLLM.")

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed.")

    # Data configs
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Data directory.",
    )
    concept_selection = parser.add_mutually_exclusive_group()
    concept_selection.add_argument(
        "--concept",
        help="Run one ConceptARC concept and save separate concept artifacts.",
    )
    concept_selection.add_argument(
        "--concept_index",
        type=int,
        help="Select a concept by its zero-based index in sorted concept names.",
    )

    # Model configs
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen3_8b",
        help="Model preset name from arc_cd/models/model_config/.",
    )
    parser.add_argument(
        "--model_config_dir",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_DIR,
        help="Directory containing model preset YAML files.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for models and tokenizer.",
    )

    # Output configs
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory.",
    )

    return parser.parse_args()


def build_run_name(
    *,
    dataset_id: str,
    split_spec: str,
    model_name: str,
    seed: int,
    reasoning_config: VLLMReasoningConfig,
) -> str:
    """Build the generation artifact stem for one run.

    Args:
        dataset_id (str): Dataset identifier such as ``"concept_arc_100"``.
        split_spec (str): Optional split suffix, including its leading
            underscore when present.
        model_name (str): Model config name.
        seed (int): Runtime seed.
        reasoning_config (VLLMReasoningConfig): Reasoning config, including
            optional reasoning effort.

    Returns:
        str: Run-name stem used for JSONL, NPZ, and config artifacts.
    """
    name_parts = [f"{dataset_id}{split_spec}", model_name]

    if reasoning_config.reasoning_effort is not None:
        name_parts.append(f"reasoning_effort_{reasoning_config.reasoning_effort}")

    name_parts.append(f"seed_{seed}")
    return "_".join(name_parts)


def select_concept_tasks(
    tasks: Sequence[ConceptARCTask],
    *,
    concept: str | None = None,
    concept_index: int | None = None,
) -> tuple[ConceptARCTask, ...]:
    """Select a concept without changing task identities or task order.

    Args:
        tasks (Sequence[ConceptARCTask]): Tasks loaded from the dataset root.
        concept (str | None): Exact concept directory name.
        concept_index (int | None): Index into sorted, nonempty concept names.

    Returns:
        tuple[ConceptARCTask, ...]: Selected tasks, or all tasks if unfiltered.

    Raises:
        ValueError: If the dataset is empty or the selection is invalid.
    """
    concepts = sorted({task.concept for task in tasks})
    if not concepts:
        raise ValueError("No ConceptARC tasks found; pass the dataset root directory.")
    if concept is not None and concept_index is not None:
        raise ValueError("Specify either concept or concept_index, not both.")
    if concept_index is not None:
        if not 0 <= concept_index < len(concepts):
            raise ValueError(
                f"Concept index {concept_index} is outside 0..{len(concepts) - 1}: "
                f"{concepts}"
            )
        concept = concepts[concept_index]
    if concept is None:
        return tuple(tasks)
    if concept not in concepts:
        raise ValueError(f"Unknown concept {concept!r}; available concepts: {concepts}")
    return tuple(task for task in tasks if task.concept == concept)


def main() -> None:
    """Run ARC-style generation."""
    args = parse_arguments()

    setup_logging(args.verbose)
    set_seed(args.seed)

    dataset_id = infer_concept_arc_dataset_id(args.data_dir)
    tasks = select_concept_tasks(
        load_concept_arc(args.data_dir),
        concept=args.concept,
        concept_index=args.concept_index,
    )
    selected_concept = (
        tasks[0].concept
        if args.concept is not None or args.concept_index is not None
        else None
    )
    task_samples = list(iter_test_cases(tasks))

    logging.info(
        "Prepared %d queries from %d tasks%s.",
        len(task_samples),
        len(tasks),
        f" for concept {selected_concept}" if selected_concept is not None else "",
    )

    # Build prompts
    prompt_templates = PromptTemplates.from_dir(DEFAULT_CONCEPT_ARC_PROMPT_DIR)
    prompt_records = [
        build_prompt_record(sample, prompt_templates) for sample in task_samples
    ]

    # Load config
    config = load_experiment_config(
        args.model_config_dir / f"{args.model_name}.yaml",
        seed=args.seed,
    )

    run_name = build_run_name(
        dataset_id=dataset_id,
        split_spec="",
        model_name=args.model_name,
        seed=args.seed,
        reasoning_config=config.reasoning,
    )
    model_dir = Path(args.output_dir).expanduser() / dataset_id / args.model_name
    seed_label = f"seed_{args.seed}"
    if selected_concept is None:
        out_dir = model_dir / "generations" / seed_label
    else:
        out_dir = model_dir / "generation_shards" / seed_label / selected_concept
    generation_jsonl_path = out_dir / f"{run_name}_generations.jsonl"
    token_npz_path = out_dir / f"{run_name}_tokens.npz"
    config_path = out_dir / f"{run_name}_config.json"
    if selected_concept is not None:
        for path in (generation_jsonl_path, token_npz_path, config_path):
            if path.exists():
                raise FileExistsError(f"Concept output already exists: {path}")

    from arc_cd.models.vllm_setup import (
        generate_from_records,
        load_tokenizer,
        load_vllm,
    )

    # Load tokenizer & model
    tokenizer = load_tokenizer(config.model, cache_dir=args.cache_dir)
    llm = load_vllm(
        config.model, reasoning_config=config.reasoning, cache_dir=args.cache_dir
    )

    # Run inference
    results = generate_from_records(
        llm=llm,
        records=prompt_records,
        tokenizer=tokenizer,
        chat_config=config.chat,
        generation_config=config.generation,
        reasoning_config=config.reasoning,
        max_model_len=config.model.max_model_len,
    )

    # Save to disk
    write_generation_outputs(
        results,
        jsonl_path=generation_jsonl_path,
        token_npz_path=token_npz_path,
    )

    write_run_config(
        config=config,
        path=config_path,
        prompt_dir=DEFAULT_CONCEPT_ARC_PROMPT_DIR,
        prompt_templates=prompt_templates,
        generation_jsonl_path=generation_jsonl_path,
        token_npz_path=token_npz_path,
        num_records=len(results),
        data_selection=(
            {
                "data_dir": str(Path(args.data_dir).expanduser().resolve()),
                "concept": selected_concept,
            }
            if selected_concept is not None
            else None
        ),
    )


if __name__ == "__main__":
    main()
