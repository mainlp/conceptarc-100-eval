"""Extract S1/S2/S3 support features directly from ConceptARC task files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import transformers

from arc_cd.data_prep.dataset_ids import infer_concept_arc_dataset_id
from arc_cd.data_prep.load_data import load_concept_arc
from arc_cd.models.config import load_experiment_config
from arc_cd.models.hf_setup import load_hf_model
from arc_cd.models.vllm_setup import load_tokenizer
from arc_cd.probes.hidden_hf import iter_prompt_hidden_states_hf
from arc_cd.probes.layers import resolve_layers
from arc_cd.probes.prompt_feature_io import (
    staged_output_directory,
    write_prompt_feature_run,
)
from arc_cd.probes.prompt_features import build_support_prompts, task_inventory
from arc_cd.prompt_prep.prompt_format import PromptTemplates
from arc_cd.utils import (
    DEFAULT_CONCEPT_ARC_PROMPT_DIR,
    DEFAULT_MODEL_CONFIG_DIR,
    set_seed,
    setup_logging,
)


def parse_arguments() -> argparse.Namespace:
    """Parse the standalone support extraction interface.

    Returns:
        argparse.Namespace: Dataset, model, selection and output options.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--model_name", default="qwen35_27b", help="Model YAML config name."
    )
    parser.add_argument(
        "--model_config_dir", type=Path, default=DEFAULT_MODEL_CONFIG_DIR
    )
    parser.add_argument("--prompt_dir", type=Path, default=DEFAULT_CONCEPT_ARC_PROMPT_DIR)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--layers", default="all", help='Layer indices: "all", "0-63:2", or "0,31,63".'
    )
    parser.add_argument("--hf_device_map", default="auto")
    parser.add_argument("--hf_chunk_size", type=int, default=4096)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--output_root",
        type=Path,
        default=Path("output"),
        help="Root for DATASET/MODEL/probing/prompt_features/.",
    )
    output.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Prompt-feature root; runs write features/ or shards/CONCEPT/ beneath it.",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--concept", default=None)
    selector.add_argument(
        "--concept_index",
        type=int,
        default=None,
        help="Zero-based sorted concept index.",
    )
    return parser.parse_args()


def _installed_version(distribution: str) -> str | None:
    """Read an optional numerical backend's installed package version.

    Args:
        distribution (str): Python distribution name.

    Returns:
        str | None: Installed version, or None when the backend is absent.
    """
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def main() -> None:
    """Extract all support positions and publish a complete, traceable run."""
    args = parse_arguments()
    setup_logging(args.verbose)
    set_seed(0)
    if args.hf_chunk_size <= 0:
        raise ValueError("--hf_chunk_size must be positive.")
    data_dir = args.data_dir.expanduser().resolve()
    tasks = load_concept_arc(data_dir)
    if not tasks:
        raise ValueError(f"No ConceptARC tasks found in {data_dir}.")
    concepts = sorted({task.concept for task in tasks})
    concept = args.concept
    if args.concept_index is not None:
        if not 0 <= args.concept_index < len(concepts):
            raise ValueError(f"--concept_index must be in 0..{len(concepts) - 1}.")
        concept = concepts[args.concept_index]
    if concept is not None and concept not in concepts:
        raise ValueError(f"Unknown concept {concept!r}; expected one of {concepts}.")
    selected_tasks = [
        task for task in tasks if concept is None or task.concept == concept
    ]
    inventory = task_inventory(tasks)
    dataset_id = infer_concept_arc_dataset_id(data_dir)
    root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.output_root.expanduser().resolve()
        / dataset_id
        / args.model_name
        / "probing"
        / "prompt_features"
    )
    destination = (
        root / "shards" / concept if concept is not None else root / "features"
    )
    config = load_experiment_config(args.model_config_dir / f"{args.model_name}.yaml")
    templates = PromptTemplates.from_dir(args.prompt_dir)
    logging.info(
        "Dataset: %s; %d versions across %d concepts.",
        dataset_id,
        len(tasks),
        len(concepts),
    )
    logging.info(
        "Extracting %d versions for %s into %s.",
        len(selected_tasks),
        concept or "all concepts",
        destination,
    )
    started = time.perf_counter()
    with staged_output_directory(destination) as stage:
        tokenizer = load_tokenizer(config.model, cache_dir=args.cache_dir)
        records, samples = build_support_prompts(
            selected_tasks, templates, tokenizer, config.chat, config.reasoning
        )
        layers = resolve_layers(args.layers, config.model, cache_dir=args.cache_dir)
        model = load_hf_model(
            config.model, cache_dir=args.cache_dir, device_map=args.hf_device_map
        )
        extraction = {
            "model_name": args.model_name,
            "config": asdict(config),
            "templates": asdict(templates),
            "tokenizer_sha256": hashlib.sha256(
                tokenizer.backend_tokenizer.to_str().encode()
            ).hexdigest(),
            "tokenizer_chat_template": tokenizer.chat_template,
            "hf_model_config": model.config.to_dict(),
            "attention_implementation": getattr(
                model.config, "_attn_implementation", None
            ),
            "torch_backends": {
                "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
                "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math_sdp": torch.backends.cuda.math_sdp_enabled(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            },
            "layers": layers,
            "concepts": concepts,
            "chunk_size": args.hf_chunk_size,
            "feature_dtype": "float16",
            "position_definition": "token_containing_final_support_output_closing_bracket",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "vllm": _installed_version("vllm"),
                "flash_linear_attention": _installed_version("flash-linear-attention"),
                "causal_conv1d": _installed_version("causal-conv1d"),
            },
        }
        run_config = {
            "schema_version": 1,
            "extraction": extraction,
            "dataset": {"id": dataset_id, "task_hashes": inventory},
            "selection": {"concept": concept},
            "num_records": len(records),
            "runtime": {"data_dir": str(data_dir), "hf_device_map": args.hf_device_map},
        }
        # Check serializability before the expensive replay begins.
        json.dumps(run_config)
        write_prompt_feature_run(
            stage,
            records,
            iter_prompt_hidden_states_hf(
                model, samples, layers=layers, chunk_size=args.hf_chunk_size
            ),
            run_config,
        )
        if task_inventory(tasks) != inventory:
            raise ValueError(
                "Source dataset changed during extraction; outputs were not published."
            )
    logging.info(
        "Saved S1/S2/S3 for %d versions in %.1f seconds: %s",
        len(records),
        time.perf_counter() - started,
        destination,
    )


if __name__ == "__main__":
    main()
