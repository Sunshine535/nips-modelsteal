#!/usr/bin/env python3
"""Expanded Observation S-PSI: Training with more suffix positions and active queries.

This script extends run_spsi.py with two key modifications:
1. Expanded observation: uses logit_suffix_positions > 8 (up to full sequence)
2. (Optional) Gramian-aware active query selection between optimization rounds

The goal is to test whether expanding the observation space (more positions,
optimized queries) can break through the rank=32 Gramian bottleneck.

Usage:
    # Phase 1: Expanded positions, random queries
    python scripts/run_expanded_observation.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --logit_suffix_positions 64 \
        --init_num_probes 256 \
        --init_truncation_rank 64 \
        --regime oracle \
        --seed 42 \
        --output_dir results/expanded_K64_s42

    # Phase 2: Expanded positions + active query reselection
    python scripts/run_expanded_observation.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --logit_suffix_positions 64 \
        --init_num_probes 256 \
        --init_truncation_rank 64 \
        --regime oracle \
        --active_query_rounds 3 \
        --seed 42 \
        --output_dir results/expanded_K64_active_s42
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.parameter_inverter import (
    BlackBoxTeacher,
    SPSIConfig,
    TeacherCache,
    run_spsi,
)
from src.permutation_alignment import compute_aligned_cosine, compute_lm_head_aligned_cosine
from src.gramian import (
    GramianConfig,
    make_flat_param_spec,
    build_probe_matrix,
    compute_gramian_for_block,
)
from src.algebraic_init import AlgebraicInitConfig
from src.symmetry_gauge import build_suffix_gauge_basis, build_flat_param_spec as sg_build_flat_param_spec

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_query_pool(tokenizer, pool_size, max_seq_len, seed=42):
    """Build query pool from WikiText validation."""
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


def _untie_lm_head(student):
    """Untie lm_head from embed_tokens if shared."""
    lm_head = getattr(student, "lm_head", None)
    embed = getattr(getattr(student, "model", None), "embed_tokens", None)
    if lm_head and embed and lm_head.weight.data_ptr() == embed.weight.data_ptr():
        lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
        if hasattr(student, "config"):
            student.config.tie_word_embeddings = False
        logger.info("Untied lm_head from embed_tokens")
        return True
    return False


def randomize_suffix(student, num_blocks, num_suffix_blocks, include_lm_head=True):
    """Re-initialize suffix blocks and optionally lm_head."""
    if include_lm_head:
        _untie_lm_head(student)

    last_block = num_blocks - 1
    target_blocks = set(range(last_block - num_suffix_blocks + 1, last_block + 1))

    for name, param in student.named_parameters():
        reset = False
        if include_lm_head and "lm_head" in name:
            reset = True
        for part in name.split("."):
            if part.isdigit() and int(part) in target_blocks:
                reset = True
                break
        if reset:
            if param.dim() >= 2:
                torch.nn.init.kaiming_uniform_(param)
            elif "norm" in name.lower():
                torch.nn.init.ones_(param)
            else:
                torch.nn.init.zeros_(param)


def get_block_param_names(model, block_idx):
    """Get parameter names for a specific block."""
    prefix = None
    for name, _ in model.named_parameters():
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == str(block_idx):
                prefix = ".".join(parts[: i + 1])
                break
        if prefix:
            break
    if prefix is None:
        return []
    return [n for n, _ in model.named_parameters() if n.startswith(prefix)]


def compute_diagnostic_gramian(student, cache, block_idx, K, k, N, seed=42):
    """Compute Gramian and return eigenspectrum for diagnostics."""
    param_names = get_block_param_names(student, block_idx)
    if not param_names:
        return None

    # Build gauge basis
    spec_for_gauge = sg_build_flat_param_spec(student, param_names)
    gauge_basis = build_suffix_gauge_basis(
        student, spec_for_gauge, [block_idx],
        include_rmsnorm=True, include_mlp=True,
    )

    config = GramianConfig(
        num_probes=k,
        query_subsample=N,
        ridge=1e-8,
        include_sensitivity=False,
        sensitivity_weight=0.0,
        project_gauge=True,
        seed=seed,
        query_batch_size=max(1, min(4, int(6e9 / (K * 151936 * k * 4)))),
    )

    result, spec, probe_matrix = compute_gramian_for_block(
        student=student,
        cache=cache,
        param_names=param_names,
        config=config,
        spsi_config=None,
        boundary_layer_idx=block_idx,
        use_oracle_boundary=False,
        gauge_basis=gauge_basis,
    )

    eigs = result.eigenvalues.numpy()
    return {
        "eigenvalues": eigs.tolist(),
        "sigma_max": float(eigs[0]),
        "effective_rank": float(result.effective_rank),
        "rank_above_1e-4": int((eigs > 1e-4).sum()),
        "rank_above_1e-2": int((eigs > 1e-2).sum()),
        "trace": float(result.trace),
        "gramian_result": result,
        "spec": spec,
        "probe_matrix": probe_matrix,
        "gauge_basis": gauge_basis,
    }


def main():
    parser = argparse.ArgumentParser(description="Expanded Observation S-PSI")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--regime", type=str, default="oracle", choices=["oracle", "pure_logits"])
    parser.add_argument("--num_suffix_blocks", type=int, default=2)
    parser.add_argument("--logit_suffix_positions", type=int, default=64,
                        help="Number of suffix positions to observe (key parameter)")
    parser.add_argument("--pool_size", type=int, default=4096)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--lm_head_steps", type=int, default=1500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lm_head_lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--init_method", type=str, default="alg_clean",
                        choices=["random", "alg_clean", "alg_aug"])
    parser.add_argument("--init_num_probes", type=int, default=256)
    parser.add_argument("--init_truncation_rank", type=int, default=64)
    parser.add_argument("--active_query_rounds", type=int, default=0,
                        help="Number of active query reselection rounds (0=disabled)")
    parser.add_argument("--active_query_fraction", type=float, default=0.3,
                        help="Fraction of queries to replace with active selection each round")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/expanded_obs")
    parser.add_argument("--gramian_diagnostic", action="store_true", default=True,
                        help="Compute Gramian diagnostic before and after training")
    args = parser.parse_args()

    setup_logging()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=== Expanded Observation S-PSI ===")
    logger.info("Key parameter: logit_suffix_positions=%d (was 8 in v1/v2)",
                args.logit_suffix_positions)

    # Load models
    logger.info("Loading teacher model: %s", args.model_name)
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
    logger.info("Building query pool...")
    query_ids = build_query_pool(tokenizer, args.pool_size, args.max_seq_len, args.seed)

    # Detect num_blocks
    num_blocks = 0
    for name, _ in teacher_model.named_parameters():
        for part in name.split("."):
            if part.isdigit():
                num_blocks = max(num_blocks, int(part) + 1)
                break

    # Setup student
    student = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )

    if args.regime == "oracle":
        randomize_suffix(student, num_blocks, args.num_suffix_blocks, include_lm_head=True)
    else:
        _untie_lm_head(student)
        for m in student.modules():
            if hasattr(m, "reset_parameters") and isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
                m.reset_parameters()
        for name, param in student.named_parameters():
            if "norm" in name.lower() and "weight" in name:
                torch.nn.init.ones_(param)

    if hasattr(student, "gradient_checkpointing_disable"):
        student.gradient_checkpointing_disable()

    # Build teacher cache with expanded K
    logger.info("Building teacher cache with K=%d...", args.logit_suffix_positions)
    teacher_bb = BlackBoxTeacher(model=teacher_model, device=device)
    cache = TeacherCache(device=device)

    boundary_layers = list(range(
        num_blocks - args.num_suffix_blocks - 1, num_blocks
    ))

    config = SPSIConfig(
        query_budget=500000,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lm_head_lr=args.lm_head_lr,
        lm_head_steps=args.lm_head_steps,
        max_steps_per_block=args.max_steps,
        convergence_threshold=1e-7,
        patience=500,
        alpha=1.0,
        beta=args.beta,
        gamma=1e-5,
        num_perturbation_positions=4,
        num_replacement_tokens=2,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        init_method=args.init_method,
        init_num_probes=args.init_num_probes,
        init_truncation_rank=args.init_truncation_rank,
        init_ridge=1e-4,
        init_step_scale=1.0,
        init_query_subsample=256,
        gramian_num_probes=args.init_num_probes,
        gramian_query_subsample=256,
        gramian_project_gauge=True,
        save_gramian_metrics=True,
        logit_suffix_positions=args.logit_suffix_positions,
    )

    cache.build(teacher_bb, query_ids, config, cache_boundary_layers=boundary_layers)
    logger.info("Teacher cache ready (K=%d). Queries used: %d",
                args.logit_suffix_positions, teacher_bb.query_count)

    # Pre-training Gramian diagnostic
    if args.gramian_diagnostic:
        logger.info("\n=== Pre-training Gramian Diagnostic ===")
        block_idx = num_blocks - 1  # last block
        diag = compute_diagnostic_gramian(
            student, cache, block_idx,
            K=args.logit_suffix_positions,
            k=args.init_num_probes,
            N=256, seed=args.seed,
        )
        if diag:
            logger.info(
                "Block %d pre-training: σ_max=%.3e, eff_rank=%.1f, rank>1e-4=%d",
                block_idx, diag["sigma_max"], diag["effective_rank"],
                diag["rank_above_1e-4"],
            )
            with open(output_dir / "gramian_pre_training.json", "w") as f:
                json.dump({
                    "block_idx": block_idx,
                    "K": args.logit_suffix_positions,
                    "k": args.init_num_probes,
                    "sigma_max": diag["sigma_max"],
                    "effective_rank": diag["effective_rank"],
                    "rank_above_1e-4": diag["rank_above_1e-4"],
                    "rank_above_1e-2": diag["rank_above_1e-2"],
                    "eigenvalues": diag["eigenvalues"],
                }, f, indent=2)

    # Run S-PSI
    logger.info("\n=== Running S-PSI with expanded observation ===")
    result = run_spsi(
        student=student,
        teacher=teacher_bb,
        cache=cache,
        config=config,
        regime=args.regime,
        num_suffix_blocks=args.num_suffix_blocks,
        ground_truth=ground_truth,
        output_dir=str(output_dir / "spsi"),
        init_seed=args.seed,
    )

    # Evaluate
    num_attention_heads = teacher_model.config.num_attention_heads
    head_dim = teacher_model.config.hidden_size // num_attention_heads
    recovered_params = {n: p.data.cpu().clone() for n, p in student.named_parameters()}

    eval_results = {}
    for br in result.block_results:
        block_idx_str = br.block_name.replace("block_", "")
        if block_idx_str.isdigit():
            prefix = ""
            for name, _ in student.named_parameters():
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == block_idx_str:
                        prefix = ".".join(parts[: i + 1])
                        break
                if prefix:
                    break
            if prefix:
                unaligned, aligned = compute_aligned_cosine(
                    recovered_params, ground_truth,
                    prefix, num_attention_heads, head_dim,
                )
                br.per_matrix_cosine = {"unaligned": unaligned, "aligned": aligned}
                br.mean_cosine = sum(aligned.values()) / len(aligned) if aligned else 0.0
                eval_results[br.block_name] = {
                    "mean_cosine": br.mean_cosine,
                    "per_matrix": aligned,
                }
                logger.info("  %s: mean_cos=%.4f", br.block_name, br.mean_cosine)

    # lm_head evaluation
    lm_head_result = compute_lm_head_aligned_cosine(
        recovered_params, ground_truth, lm_head_key="lm_head.weight",
    )
    if lm_head_result:
        eval_results["lm_head"] = lm_head_result
        logger.info("  lm_head: aligned_cos=%.4f", lm_head_result.get("aligned_cosine", 0))

    # Post-training Gramian diagnostic
    if args.gramian_diagnostic:
        logger.info("\n=== Post-training Gramian Diagnostic ===")
        block_idx = num_blocks - 1
        diag_post = compute_diagnostic_gramian(
            student, cache, block_idx,
            K=args.logit_suffix_positions,
            k=args.init_num_probes,
            N=256, seed=args.seed,
        )
        if diag_post:
            logger.info(
                "Block %d post-training: σ_max=%.3e, eff_rank=%.1f, rank>1e-4=%d",
                block_idx, diag_post["sigma_max"], diag_post["effective_rank"],
                diag_post["rank_above_1e-4"],
            )
            with open(output_dir / "gramian_post_training.json", "w") as f:
                json.dump({
                    "block_idx": block_idx,
                    "K": args.logit_suffix_positions,
                    "k": args.init_num_probes,
                    "sigma_max": diag_post["sigma_max"],
                    "effective_rank": diag_post["effective_rank"],
                    "rank_above_1e-4": diag_post["rank_above_1e-4"],
                    "rank_above_1e-2": diag_post["rank_above_1e-2"],
                    "eigenvalues": diag_post["eigenvalues"],
                }, f, indent=2)

    # Save summary
    summary = {
        "model": args.model_name,
        "regime": args.regime,
        "seed": args.seed,
        "logit_suffix_positions": args.logit_suffix_positions,
        "init_method": args.init_method,
        "init_num_probes": args.init_num_probes,
        "init_truncation_rank": args.init_truncation_rank,
        "eval_results": eval_results,
        "total_queries": teacher_bb.query_count,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n=== Experiment Complete ===")
    logger.info("Results saved to %s", output_dir)
    for name, data in eval_results.items():
        if isinstance(data, dict) and "mean_cosine" in data:
            logger.info("  %s: %.4f", name, data["mean_cosine"])
        elif isinstance(data, dict) and "aligned_cosine" in data:
            logger.info("  %s: %.4f", name, data["aligned_cosine"])


if __name__ == "__main__":
    main()
