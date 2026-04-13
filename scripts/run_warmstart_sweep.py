#!/usr/bin/env python3
"""Warm-start initialization sweep: interpolate between random init and teacher.

Tests the "observability-recoverability gap" hypothesis by starting from
different distances to the teacher weights. If near-teacher starts recover
while far starts fail, this confirms the gap is about optimization distance.

Usage:
    python scripts/run_warmstart_sweep.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --alphas 0.0 0.1 0.3 0.5 0.7 0.9 1.0 \
        --regime oracle \
        --seed 42 \
        --output_dir results/warmstart_sweep
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.parameter_inverter import (
    BlackBoxTeacher, SPSIConfig, TeacherCache, invert_block,
    get_block_param_names,
)
from src.permutation_alignment import compute_aligned_cosine

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_query_pool(tokenizer, pool_size, max_seq_len, seed=42):
    input_ids_list = []
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        for ex in ds:
            if len(input_ids_list) >= pool_size:
                break
            text = ex.get("text", "")
            if len(text.strip()) < 20:
                continue
            tokens = tokenizer(
                text, max_length=max_seq_len, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            input_ids_list.append(tokens["input_ids"].squeeze(0))
    except Exception as e:
        logger.warning("Dataset load failed: %s. Using random tokens.", e)

    remaining = pool_size - len(input_ids_list)
    if remaining > 0:
        rng = torch.Generator().manual_seed(seed + 1)
        random_ids = torch.randint(3, tokenizer.vocab_size, (remaining, max_seq_len), generator=rng)
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    return torch.stack(input_ids_list[:pool_size])


def interpolate_suffix(student, ground_truth, block_idx, alpha):
    """Set student block parameters to alpha * teacher + (1-alpha) * random.

    alpha=0: fully random (standard init)
    alpha=1: teacher weights (oracle init)
    """
    prefix = None
    for name, _ in student.named_parameters():
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == str(block_idx):
                prefix = ".".join(parts[:i + 1])
                break
        if prefix:
            break
    if prefix is None:
        return

    for name, param in student.named_parameters():
        if not name.startswith(prefix):
            continue
        if name in ground_truth:
            teacher_val = ground_truth[name].to(param.device, param.dtype)
            # Random init
            random_val = param.data.clone()
            if param.dim() >= 2:
                torch.nn.init.kaiming_uniform_(random_val)
            elif "norm" in name.lower():
                random_val.fill_(1.0)
            else:
                random_val.zero_()
            # Interpolate
            param.data.copy_(alpha * teacher_val + (1 - alpha) * random_val)


def main():
    parser = argparse.ArgumentParser(description="Warm-start initialization sweep")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--regime", type=str, default="oracle")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    parser.add_argument("--target_block", type=int, default=23)
    parser.add_argument("--logit_suffix_positions", type=int, default=32)
    parser.add_argument("--pool_size", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/warmstart_sweep")
    args = parser.parse_args()

    setup_logging()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=== Warm-Start Initialization Sweep ===")
    logger.info("Alphas: %s", args.alphas)
    logger.info("Target: Block %d, K=%d", args.target_block, args.logit_suffix_positions)

    # Load teacher
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    teacher_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ground_truth = {n: p.data.cpu().clone() for n, p in teacher_model.named_parameters()}
    if getattr(teacher_model.config, "tie_word_embeddings", False):
        lm_head = getattr(teacher_model, "lm_head", None)
        if lm_head is not None and "lm_head.weight" not in ground_truth:
            ground_truth["lm_head.weight"] = lm_head.weight.data.cpu().clone()

    # Build query pool
    query_ids = build_query_pool(tokenizer, args.pool_size, args.max_seq_len, args.seed)

    # Detect num_blocks
    num_blocks = 0
    for name, _ in teacher_model.named_parameters():
        for part in name.split("."):
            if part.isdigit():
                num_blocks = max(num_blocks, int(part) + 1)
                break

    # Build teacher cache once
    teacher_bb = BlackBoxTeacher(model=teacher_model, device=device)

    config = SPSIConfig(
        query_budget=500000,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lm_head_lr=1e-3,
        lm_head_steps=0,
        max_steps_per_block=args.max_steps,
        convergence_threshold=1e-7,
        patience=300,
        alpha=1.0, beta=0.0, gamma=1e-5,
        num_perturbation_positions=1,
        num_replacement_tokens=1,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        init_method="random",
        logit_suffix_positions=args.logit_suffix_positions,
    )

    boundary_layers = [args.target_block - 1, args.target_block]
    cache = TeacherCache(device=device)
    cache.build(teacher_bb, query_ids, config, cache_boundary_layers=boundary_layers)
    logger.info("Teacher cache built. Queries: %d", teacher_bb.query_count)

    num_attention_heads = teacher_model.config.num_attention_heads
    head_dim = teacher_model.config.hidden_size // num_attention_heads

    results = []

    for alpha in args.alphas:
        logger.info("\n=== Alpha = %.2f (%.0f%% teacher) ===", alpha, alpha * 100)

        # Fresh student
        student = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=True,
        )

        # Untie lm_head
        lm_head = getattr(student, "lm_head", None)
        embed = getattr(getattr(student, "model", None), "embed_tokens", None)
        if lm_head and embed and lm_head.weight.data_ptr() == embed.weight.data_ptr():
            lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
            if hasattr(student, "config"):
                student.config.tie_word_embeddings = False

        if hasattr(student, "gradient_checkpointing_disable"):
            student.gradient_checkpointing_disable()

        # Interpolate target block
        interpolate_suffix(student, ground_truth, args.target_block, alpha)

        # Measure pre-training cosine
        pre_params = {n: p.data.cpu().clone() for n, p in student.named_parameters()}
        prefix = None
        for name, _ in student.named_parameters():
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == str(args.target_block):
                    prefix = ".".join(parts[:i + 1])
                    break
            if prefix:
                break

        if prefix:
            _, pre_aligned = compute_aligned_cosine(
                pre_params, ground_truth, prefix, num_attention_heads, head_dim,
            )
            pre_mean = sum(pre_aligned.values()) / len(pre_aligned) if pre_aligned else 0
        else:
            pre_aligned, pre_mean = {}, 0

        logger.info("  Pre-training mean_cos = %.4f", pre_mean)

        # Get block param names
        block_names = get_block_param_names(student, args.target_block)

        # Train
        use_oracle = (args.regime == "oracle")
        br = invert_block(
            student, cache, config, block_names,
            ground_truth=ground_truth,
            boundary_layer_idx=args.target_block,
            use_oracle_boundary=use_oracle,
        )

        # Post-training cosine
        post_params = {n: p.data.cpu().clone() for n, p in student.named_parameters()}
        if prefix:
            _, post_aligned = compute_aligned_cosine(
                post_params, ground_truth, prefix, num_attention_heads, head_dim,
            )
            post_mean = sum(post_aligned.values()) / len(post_aligned) if post_aligned else 0
        else:
            post_aligned, post_mean = {}, 0

        result = {
            "alpha": alpha,
            "pre_mean_cos": pre_mean,
            "post_mean_cos": post_mean,
            "pre_per_matrix": {k: float(v) for k, v in pre_aligned.items()},
            "post_per_matrix": {k: float(v) for k, v in post_aligned.items()},
            "final_loss": float(br.final_loss),
            "num_steps": br.num_steps,
            "converged": br.converged,
        }
        results.append(result)

        logger.info("  Post-training mean_cos = %.4f (loss=%.4f, steps=%d)",
                     post_mean, br.final_loss, br.num_steps)

        # Cleanup
        del student
        torch.cuda.empty_cache()

    # Save results
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    logger.info("\n=== Summary ===")
    logger.info("%-8s | %-12s | %-12s | %-10s", "Alpha", "Pre cos", "Post cos", "Delta")
    logger.info("-" * 50)
    for r in results:
        delta = r["post_mean_cos"] - r["pre_mean_cos"]
        logger.info("%-8.2f | %-12.4f | %-12.4f | %+.4f",
                     r["alpha"], r["pre_mean_cos"], r["post_mean_cos"], delta)

    logger.info("\nResults saved to %s", output_dir)


if __name__ == "__main__":
    main()
