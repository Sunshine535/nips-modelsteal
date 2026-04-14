#!/usr/bin/env python3
"""Active query selection experiment: Gramian-aware vs random queries.

Tests whether actively selecting queries that target weak Gramian
eigendirections improves parameter recovery over passive (random) queries.

Usage:
    python scripts/run_active_query.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --pool_size 2048 \
        --active_select 256 \
        --output_dir results/active_query
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
from src.gramian import (
    make_flat_param_spec, compute_sketched_gramian, jvp_logits,
    GramianConfig,
)
from src.active_query import GramianAwareSelector

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


class MinimalCache:
    """Minimal teacher cache for Gramian computation (same pattern as diagnose_gramian_rank.py)."""

    def __init__(self, teacher_model, input_ids, device="cuda"):
        self._full_input_ids = input_ids
        self.input_ids = input_ids
        self.device = device
        self._teacher = teacher_model
        self.clean_logits = None
        self._logit_suffix_k = 0
        self.perturbed_input_ids = None
        self.perturbed_logits = None
        self.boundary_states = {}

    def build_for_K(self, K: int, query_indices: torch.Tensor):
        """Pre-compute teacher logits for specific K and query subset."""
        n = len(query_indices)
        self._logit_suffix_k = K
        self.input_ids = self._full_input_ids[query_indices]

        logits_list = []
        bs = 16
        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch = self.input_ids[start:end].to(self.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = self._teacher(batch).logits.float()
            if K > 0:
                out = out[:, -K:, :]
            logits_list.append(out.cpu())

        self.clean_logits = torch.cat(logits_list)
        logger.info("  Cache built: %d queries, K=%d, shape=%s",
                     n, K, self.clean_logits.shape)


def compute_gramian_for_block(student, teacher_model, block_names, query_ids, args, device):
    """Compute sketched Gramian and return eigendecomposition + probe matrix."""
    spec = make_flat_param_spec(student, block_names)
    K = args.logit_suffix_positions
    k = args.num_probes

    logger.info("Computing Gramian: %d params, k=%d probes, K=%d suffix positions",
                spec.num_params, k, K)

    # Try gauge projection
    probe_matrix = None
    try:
        from src.symmetry_gauge import build_gauge_basis
        gauge_basis = build_gauge_basis(student, block_names)
        if gauge_basis is not None and gauge_basis.shape[1] > 0:
            logger.info("Gauge basis: %d directions (projecting out)", gauge_basis.shape[1])
            P_perp = torch.eye(spec.num_params) - gauge_basis @ gauge_basis.T
            raw = torch.randn(spec.num_params, k)
            projected = P_perp @ raw
            probe_matrix, _ = torch.linalg.qr(projected)
            probe_matrix = probe_matrix[:, :k]
    except Exception as e:
        logger.warning("Gauge projection failed: %s. Using random probes.", e)

    if probe_matrix is None:
        raw = torch.randn(spec.num_params, k)
        probe_matrix, _ = torch.linalg.qr(raw)
        probe_matrix = probe_matrix[:, :k]

    # Subsample queries
    n_queries = min(args.gramian_query_subsample, len(query_ids))
    perm = torch.randperm(len(query_ids))[:n_queries]

    # Build MinimalCache with teacher logits
    cache = MinimalCache(teacher_model, query_ids, device=device)
    cache.build_for_K(K, perm)

    # GramianConfig
    gram_config = GramianConfig(
        num_probes=k,
        query_subsample=n_queries,
        ridge=1e-8,
        include_sensitivity=False,
        sensitivity_weight=0.0,
        project_gauge=False,
        seed=args.seed,
        query_batch_size=2,
    )

    result = compute_sketched_gramian(
        student=student,
        cache=cache,
        spec=spec,
        query_indices=torch.arange(n_queries),
        probe_matrix=probe_matrix,
        config=gram_config,
        spsi_config=None,
        boundary_layer_idx=args.target_block,
        use_oracle_boundary=False,
        gauge_basis=None,
    )

    eigenvalues = result.eigenvalues
    eigenvectors = result.eigenvectors

    logger.info("Gramian: σ_max=%.1f, σ_min=%.1e, κ=%.2f, eff_rank=%.1f",
                eigenvalues[0].item(),
                eigenvalues[-1].item() if eigenvalues[-1] > 0 else 0,
                eigenvalues[0].item() / max(eigenvalues[-1].item(), 1e-12),
                (eigenvalues.sum()**2 / eigenvalues.pow(2).sum()).item())

    return spec, probe_matrix, eigenvalues, eigenvectors


def run_one_condition(student, teacher_bb, query_ids, block_names,
                      block_idx, ground_truth, config, args, device,
                      condition_name, num_attention_heads, head_dim):
    """Run S-PSI training on given queries and evaluate."""
    boundary_layers = [block_idx - 1, block_idx]
    cache = TeacherCache(device=device)
    cache.build(teacher_bb, query_ids, config,
                cache_boundary_layers=boundary_layers)
    queries_used = teacher_bb.query_count

    br = invert_block(
        student, cache, config, block_names,
        ground_truth=ground_truth,
        boundary_layer_idx=block_idx,
        use_oracle_boundary=True,
    )

    # Free cache memory before evaluation
    del cache
    torch.cuda.empty_cache()

    # Move student to CPU for eval
    student.cpu()
    torch.cuda.empty_cache()

    # Evaluate
    post_params = {n: p.data.clone() for n, p in student.named_parameters()}
    prefix = None
    for name in post_params:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == str(block_idx):
                prefix = ".".join(parts[:i + 1])
                break
        if prefix:
            break

    if prefix:
        _, aligned = compute_aligned_cosine(
            post_params, ground_truth, prefix,
            num_attention_heads, head_dim,
        )
        mean_cos = sum(aligned.values()) / len(aligned) if aligned else 0
    else:
        aligned, mean_cos = {}, 0

    logger.info("[%s] Block %d: mean_cos=%.4f, loss=%.4f, steps=%d",
                condition_name, block_idx, mean_cos, br.final_loss, br.num_steps)

    return {
        "condition": condition_name,
        "block_idx": block_idx,
        "mean_cosine": float(mean_cos),
        "per_matrix": {k: float(v) for k, v in aligned.items()},
        "final_loss": float(br.final_loss),
        "num_steps": br.num_steps,
        "converged": br.converged,
        "queries_used": queries_used,
    }


def main():
    parser = argparse.ArgumentParser(description="Active query experiment")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--target_block", type=int, default=23)
    parser.add_argument("--pool_size", type=int, default=2048)
    parser.add_argument("--active_select", type=int, default=256,
                        help="Number of queries to select via active method")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--logit_suffix_positions", type=int, default=32)
    parser.add_argument("--num_probes", type=int, default=128)
    parser.add_argument("--gramian_query_subsample", type=int, default=256)
    parser.add_argument("--num_target_directions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/active_query")
    args = parser.parse_args()

    setup_logging()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=== Active Query Experiment ===")
    logger.info("Pool: %d, Active select: %d, Block: %d",
                args.pool_size, args.active_select, args.target_block)

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

    # Build large query pool
    full_pool = build_query_pool(tokenizer, args.pool_size, args.max_seq_len,
                                 args.seed)
    logger.info("Full pool: %s", full_pool.shape)

    num_attention_heads = teacher_model.config.num_attention_heads
    head_dim = teacher_model.config.hidden_size // num_attention_heads

    block_names = get_block_param_names(teacher_model, args.target_block)

    # SPSIConfig for both conditions
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

    teacher_bb = BlackBoxTeacher(model=teacher_model, device=device)

    # --- Condition 1: Random query subset (control) ---
    logger.info("\n=== Condition 1: RANDOM queries (%d from pool) ===",
                args.active_select)
    torch.manual_seed(args.seed)

    random_perm = torch.randperm(len(full_pool))[:args.active_select]
    random_queries = full_pool[random_perm]

    student_random = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    # Untie
    lm_head = getattr(student_random, "lm_head", None)
    embed = getattr(getattr(student_random, "model", None), "embed_tokens", None)
    if lm_head and embed and lm_head.weight.data_ptr() == embed.weight.data_ptr():
        lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
        student_random.config.tie_word_embeddings = False
    if hasattr(student_random, "gradient_checkpointing_disable"):
        student_random.gradient_checkpointing_disable()

    teacher_bb.query_count = 0
    result_random = run_one_condition(
        student_random, teacher_bb, random_queries, block_names,
        args.target_block, ground_truth, config, args, device,
        "random", num_attention_heads, head_dim,
    )
    del student_random
    torch.cuda.empty_cache()

    # --- Compute Gramian for active selection ---
    logger.info("\n=== Computing Gramian for active query selection ===")

    student_active = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    lm_head = getattr(student_active, "lm_head", None)
    embed = getattr(getattr(student_active, "model", None), "embed_tokens", None)
    if lm_head and embed and lm_head.weight.data_ptr() == embed.weight.data_ptr():
        lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
        student_active.config.tie_word_embeddings = False
    if hasattr(student_active, "gradient_checkpointing_disable"):
        student_active.gradient_checkpointing_disable()

    spec, probe_matrix, eigenvalues, eigenvectors = compute_gramian_for_block(
        student_active, teacher_model, block_names, full_pool, args, device,
    )

    # --- Condition 2: Active (Gramian-aware) query selection ---
    logger.info("\n=== Condition 2: ACTIVE queries (%d from pool) ===",
                args.active_select)

    selector = GramianAwareSelector(
        student=student_active,
        spec=spec,
        probe_matrix=probe_matrix,
        gramian_eigenvectors=eigenvectors,
        gramian_eigenvalues=eigenvalues,
        num_target_directions=args.num_target_directions,
        logit_suffix_positions=args.logit_suffix_positions,
        device=device,
    )

    active_queries, active_indices, active_scores = selector.select(
        full_pool, n_select=args.active_select, batch_size=2,
    )

    teacher_bb.query_count = 0
    result_active = run_one_condition(
        student_active, teacher_bb, active_queries, block_names,
        args.target_block, ground_truth, config, args, device,
        "active", num_attention_heads, head_dim,
    )
    del student_active
    torch.cuda.empty_cache()

    # --- Condition 3: Full pool (upper bound) ---
    logger.info("\n=== Condition 3: FULL pool (%d queries) ===", len(full_pool))

    student_full = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    lm_head = getattr(student_full, "lm_head", None)
    embed = getattr(getattr(student_full, "model", None), "embed_tokens", None)
    if lm_head and embed and lm_head.weight.data_ptr() == embed.weight.data_ptr():
        lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
        student_full.config.tie_word_embeddings = False
    if hasattr(student_full, "gradient_checkpointing_disable"):
        student_full.gradient_checkpointing_disable()

    teacher_bb.query_count = 0
    result_full = run_one_condition(
        student_full, teacher_bb, full_pool, block_names,
        args.target_block, ground_truth, config, args, device,
        "full_pool", num_attention_heads, head_dim,
    )
    del student_full
    torch.cuda.empty_cache()

    # --- Save results ---
    results = {
        "random": result_random,
        "active": result_active,
        "full_pool": result_full,
        "gramian_stats": {
            "sigma_max": float(eigenvalues[0]),
            "sigma_min": float(eigenvalues[-1]),
            "condition_number": float(eigenvalues[0] / max(eigenvalues[-1], 1e-12)),
            "effective_rank": float(eigenvalues.sum()**2 / eigenvalues.pow(2).sum()),
        },
    }

    with open(output_dir / "active_query_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    logger.info("\n=== Summary ===")
    logger.info("%-15s | %-10s | %-10s | %-10s | %-8s",
                "Condition", "N queries", "Mean cos", "Loss", "Steps")
    logger.info("-" * 65)
    for r in [result_random, result_active, result_full]:
        logger.info("%-15s | %-10d | %-10.4f | %-10.4f | %-8d",
                    r["condition"], r["queries_used"],
                    r["mean_cosine"], r["final_loss"], r["num_steps"])

    logger.info("\nResults saved to %s", output_dir)


if __name__ == "__main__":
    main()
