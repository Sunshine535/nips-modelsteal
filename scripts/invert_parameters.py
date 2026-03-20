#!/usr/bin/env python3
"""
Stage 2 — Progressive Parameter Inversion.

Given a distilled student as initialization, progressively recover the
teacher's weight parameters layer by layer. For each layer (output → deeper):
  1. Fix recovered layers, hypothesize current layer weights
  2. Use gradient-based optimization to match teacher outputs
  3. Active query selection: pick inputs that maximize |student - teacher|

Usage:
    python scripts/invert_parameters.py \
        --teacher_model Qwen/Qwen3.5-9B \
        --student_checkpoint results/distillation/best_student \
        --query_budget 500000 \
        --output_dir results/inversion

    # Multi-GPU:
    torchrun --nproc_per_node=8 scripts/invert_parameters.py ...
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.active_query import ActiveQuerySelector, QueryPool
from src.parameter_inverter import (
    BlackBoxTeacher,
    InversionConfig,
    LayerWiseInverter,
)

logger = logging.getLogger(__name__)


def setup_logging(rank: int = 0):
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[%(asctime)s][Rank {rank}] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def load_config(config_path: str) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def extract_ground_truth_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract teacher's true weights for evaluation (not used during inversion)."""
    gt = {}
    for name, param in model.named_parameters():
        gt[name] = param.data.clone().cpu()
    return gt


def build_query_pool(
    tokenizer,
    pool_size: int,
    max_seq_len: int,
    device: str,
    seed: int = 42,
) -> QueryPool:
    """Build a pool of candidate queries for active selection."""
    pool = QueryPool(
        tokenizer=tokenizer,
        pool_size=pool_size,
        max_seq_len=max_seq_len,
        device=device,
        seed=seed,
    )

    try:
        from datasets import load_dataset

        ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="validation", trust_remote_code=True
        )
        pool.build_from_dataset(ds, text_column="text")
    except Exception as e:
        logger.warning("Could not load dataset for query pool: %s. Using random.", e)
        pool.build_random(tokenizer.vocab_size)

    return pool


def save_recovered_model(
    student_model,
    recovered_state_dict: dict,
    output_dir: str,
    tokenizer,
):
    """Save the model with recovered weights."""
    out = Path(output_dir) / "recovered_model"
    out.mkdir(parents=True, exist_ok=True)

    current_sd = student_model.state_dict()
    for key, value in recovered_state_dict.items():
        if key in current_sd:
            current_sd[key] = value

    student_model.load_state_dict(current_sd)
    student_model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    logger.info("Saved recovered model to %s", out)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Progressive Parameter Inversion")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--student_checkpoint", type=str, default="results/distillation/best_student")
    parser.add_argument("--query_budget", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_steps_per_layer", type=int, default=10000)
    parser.add_argument("--convergence_threshold", type=float, default=1e-6)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "lbfgs"])
    parser.add_argument("--active_query_strategy", type=str, default="gradient_magnitude",
                        choices=["random", "gradient_magnitude", "fisher_information", "divergence"])
    parser.add_argument("--query_pool_size", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--regularization_lambda", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="results/inversion")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_layers", type=int, default=-1,
                        help="Max layers to invert (-1 = all)")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    setup_logging(rank)

    config = load_config(args.config)
    torch.manual_seed(args.seed)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # --- Load teacher (black-box, but we keep weights for evaluation) ---
    logger.info("Loading teacher model: %s", args.teacher_model)
    teacher_raw = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    logger.info("Extracting ground-truth weights for evaluation...")
    ground_truth = extract_ground_truth_weights(teacher_raw)

    teacher = BlackBoxTeacher(model=teacher_raw, device=device)

    # --- Load distilled student ---
    logger.info("Loading student from checkpoint: %s", args.student_checkpoint)
    if Path(args.student_checkpoint).exists():
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
    else:
        logger.warning(
            "Student checkpoint not found at %s; loading base model",
            args.student_checkpoint,
        )
        student_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-0.8B",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Build query pool ---
    logger.info("Building query pool (size=%d)...", args.query_pool_size)
    query_pool = build_query_pool(
        tokenizer, args.query_pool_size, args.max_seq_len, device, args.seed
    )

    # --- Configure inversion ---
    inv_config = InversionConfig(
        query_budget=args.query_budget,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps_per_layer=args.max_steps_per_layer,
        convergence_threshold=args.convergence_threshold,
        optimizer_type=args.optimizer,
        regularization_lambda=args.regularization_lambda,
        active_query_strategy=args.active_query_strategy,
        active_query_pool_size=args.query_pool_size,
        seed=args.seed,
    )

    # --- Run progressive inversion ---
    inverter = LayerWiseInverter(
        student_model=student_model,
        teacher=teacher,
        config=inv_config,
        device=device,
    )
    inverter.set_query_pool(query_pool)

    logger.info("Starting progressive parameter inversion...")
    logger.info("  Query budget: %d", args.query_budget)
    logger.info("  Strategy: %s", args.active_query_strategy)
    logger.info("  Max steps/layer: %d", args.max_steps_per_layer)

    result = inverter.run_progressive_inversion(
        teacher_ground_truth=ground_truth,
        output_dir=args.output_dir,
    )

    # --- Save recovered model ---
    if rank == 0 and result.recovered_state_dict:
        save_recovered_model(
            student_model, result.recovered_state_dict, args.output_dir, tokenizer
        )

        logger.info("=== Inversion Summary ===")
        logger.info("Total queries used: %d", result.total_queries)
        for lr in result.layer_results:
            logger.info(
                "  %s: cos_sim=%.4f, l2=%.4f, loss=%.6f, steps=%d, queries=%d",
                lr.layer_name, lr.cosine_similarity, lr.l2_distance,
                lr.final_loss, lr.num_steps, lr.num_queries_used,
            )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
