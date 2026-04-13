#!/usr/bin/env python3
"""Diagnostic: How does Gramian rank scale with observation size?

Tests whether rank=32 (observed with K=8 positions, k=64 probes) is a fundamental
structural limit or an artifact of insufficient observation.

Experiment matrix (Block 23, Qwen2.5-0.5B):
  K ∈ {8, 32, 64, 128}  (suffix token positions observed)
  k ∈ {64, 128, 256}     (number of random probes)

For each (K, k) pair, computes the gauge-projected sketched Gramian and reports:
  - Full eigenspectrum
  - Effective rank (trace²/Frobenius²)
  - Number of eigenvalues above threshold (1e-6, 1e-4, 1e-2)
  - σ_max, σ_min(nonzero), condition number

Memory-adaptive: reduces query_batch_size for large K to fit A100-80GB.

Usage:
    python scripts/diagnose_gramian_rank.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --output_dir results/gramian_diagnostic \
        --K_values 8 32 64 128 \
        --k_values 64 128 256 \
        --query_subsample 256 \
        --pool_size 2048
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

from src.gramian import (
    GramianConfig,
    build_probe_matrix,
    compute_sketched_gramian,
    make_flat_param_spec,
)
from src.symmetry_gauge import (
    build_suffix_gauge_basis,
    build_flat_param_spec as sg_build_flat_param_spec,
)

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_block_param_names(model, block_idx: int) -> list:
    """Get parameter names for a specific transformer block."""
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


def build_query_pool(tokenizer, pool_size: int, max_seq_len: int, seed: int = 42):
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
        rng = torch.Generator().manual_seed(seed)
        random_ids = torch.randint(3, tokenizer.vocab_size, (pool_size, max_seq_len), generator=rng)
        for i in range(pool_size):
            input_ids_list.append(random_ids[i])

    remaining = pool_size - len(input_ids_list)
    if remaining > 0:
        rng = torch.Generator().manual_seed(seed + 1)
        random_ids = torch.randint(3, tokenizer.vocab_size, (remaining, max_seq_len), generator=rng)
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    return torch.stack(input_ids_list[:pool_size])


class MinimalCache:
    """Minimal teacher cache that supports variable K (suffix positions).

    Unlike TeacherCache which pre-stores logits at fixed K, this re-queries
    the teacher with the desired K at Gramian computation time.
    """

    def __init__(self, teacher_model, input_ids, device="cuda"):
        self._full_input_ids = input_ids  # immutable full pool
        self.input_ids = input_ids  # may be remapped by build_for_K
        self.device = device
        self._teacher = teacher_model
        # These will be set per-K
        self.clean_logits = None
        self._logit_suffix_k = 0
        self.perturbed_input_ids = None
        self.perturbed_logits = None
        # Attributes accessed by compute_sketched_gramian
        self.boundary_states = {}

    def build_for_K(self, K: int, query_indices: torch.Tensor):
        """Pre-compute teacher logits for specific K and query subset.

        After calling this, input_ids[i] and clean_logits[i] are aligned
        for i in 0..len(query_indices)-1.
        """
        device = self.device
        n = len(query_indices)
        self._logit_suffix_k = K

        # Remap input_ids so indices [0..n) are valid
        self.input_ids = self._full_input_ids[query_indices]

        logits_list = []
        bs = 16
        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch = self.input_ids[start:end].to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = self._teacher(batch).logits.float()
            if K > 0:
                out = out[:, -K:, :]
            logits_list.append(out.cpu())

        self.clean_logits = torch.cat(logits_list)
        logger.info("  Cache built: %d queries, K=%d, shape=%s",
                     n, K, self.clean_logits.shape)


def _memory_efficient_gramian(
    student,
    spec,
    probe_matrix: torch.Tensor,
    cache: MinimalCache,
    K: int,
    N: int,
    actual_k: int,
):
    """Compute k×k Gramian without materializing [B*K*V, k] on GPU.

    For large K, the standard code stores S_clean = [B*K*V, k] which can
    exceed GPU memory. Instead, we process one query at a time:
    1. Compute all k JVPs for one query, store on CPU as float16
    2. Accumulate G += S^T S on CPU in float64

    CPU memory per query: k × K × V × 2 bytes (float16).
    E.g., k=256, K=128, V=152K → ~10 GB. k=64, K=128 → ~2.5 GB.
    """
    from src.gramian import jvp_logits

    device = next(student.parameters()).device
    G_clean = torch.zeros(actual_k, actual_k, dtype=torch.float64, device="cpu")
    rhs_acc = torch.zeros(actual_k, dtype=torch.float64, device="cpu")
    student.eval()

    for qi in range(N):
        if qi % 25 == 0:
            logger.info("    Query %d/%d (memory-efficient)", qi, N)

        input_ids = cache.input_ids[qi:qi+1].to(device)
        teacher_logits = cache.clean_logits[qi:qi+1].to(device).float()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits = student(input_ids).logits.float()
        if K > 0 and student_logits.shape[1] > K:
            student_logits = student_logits[:, -K:, :]

        residual = (teacher_logits - student_logits).reshape(-1).cpu()  # [K*V] on CPU
        obs_dim = residual.shape[0]

        # Compute all k JVPs and store on CPU as float16
        S = torch.zeros(obs_dim, actual_k, dtype=torch.float16, device="cpu")
        for j in range(actual_k):
            v_j = probe_matrix[:, j]
            jvp_out = jvp_logits(student, input_ids, spec, v_j)
            if K > 0 and jvp_out.shape[1] > K:
                jvp_out = jvp_out[:, -K:, :]
            S[:, j] = jvp_out.reshape(-1).cpu().half()
            del jvp_out
            torch.cuda.empty_cache()

        # Accumulate G and RHS on CPU in float64
        S_f64 = S.double()  # [obs_dim, k]
        G_clean += S_f64.T @ S_f64  # [k, k]
        rhs_acc += S_f64.T @ residual.double()  # [k]

        del S, S_f64
        torch.cuda.empty_cache()

    # Normalize
    G_clean /= N
    rhs_acc /= N

    # Ridge
    G_clean.diagonal().add_(1e-8)

    # Eigendecomposition
    G_f32 = G_clean.float()
    eigenvalues, eigenvectors = torch.linalg.eigh(G_f32)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    return eigenvalues.numpy(), rhs_acc.float().numpy()


def compute_gramian_for_config(
    student,
    cache: MinimalCache,
    param_names: list,
    K: int,
    k: int,
    query_subsample: int,
    gauge_basis,
    seed: int = 42,
    block_idx: int = 23,
):
    """Compute Gramian for specific (K, k) configuration.

    Returns dict with eigenspectrum and rank analysis.
    """
    device = next(student.parameters()).device

    # Build flat param spec
    spec = make_flat_param_spec(student, param_names)
    p = spec.num_params

    # Build probes
    actual_k = min(k, p)
    probe_matrix = build_probe_matrix(spec, actual_k, seed, device)

    # Gauge-project probes
    if gauge_basis is not None:
        from src.symmetry_gauge import project_out_gauge
        probe_cpu = probe_matrix.cpu().float()
        projected = project_out_gauge(probe_cpu, gauge_basis)
        # QR re-orthonormalize and drop zero columns
        Q, R = torch.linalg.qr(projected, mode="reduced")
        diag = R.diag().abs()
        valid = diag > 1e-8
        n_lost = (~valid).sum().item()
        Q = Q[:, valid]
        probe_matrix = Q.to(device)
        actual_k = probe_matrix.shape[1]
        logger.info("  Gauge projection: k=%d → %d (lost %d)", k, actual_k, n_lost)

    if actual_k == 0:
        return {"error": "all probes in gauge subspace"}

    # Subsample queries
    N = min(query_subsample, len(cache.input_ids))

    # Build cache with correct K
    cache.build_for_K(K, torch.arange(N))

    vocab_size = cache.clean_logits.shape[-1]

    # Determine if we need memory-efficient path
    # S_clean memory estimate: B * K * V * k * 4 bytes
    # With batch_size=1: K * V * k * 4
    s_clean_bytes = K * vocab_size * actual_k * 4
    gpu_budget = 6 * 1024**3  # 6 GB

    if s_clean_bytes > gpu_budget:
        # Use memory-efficient path (JVPs stored on CPU)
        logger.info("  Using MEMORY-EFFICIENT path (S_clean would be %.1f GB)",
                     s_clean_bytes / 1024**3)
        eigs, rhs = _memory_efficient_gramian(
            student, spec, probe_matrix, cache, K, N, actual_k,
        )
    else:
        # Use standard GPU path
        adaptive_bs = max(1, int(gpu_budget / (K * vocab_size * actual_k * 4)))
        adaptive_bs = min(adaptive_bs, 8)
        logger.info("  Using STANDARD path (batch_size=%d)", adaptive_bs)

        config = GramianConfig(
            num_probes=actual_k,
            query_subsample=N,
            ridge=1e-8,
            include_sensitivity=False,
            sensitivity_weight=0.0,
            project_gauge=False,
            seed=seed,
            query_batch_size=adaptive_bs,
        )

        result = compute_sketched_gramian(
            student=student,
            cache=cache,
            spec=spec,
            query_indices=torch.arange(N),
            probe_matrix=probe_matrix,
            config=config,
            spsi_config=None,
            boundary_layer_idx=block_idx,
            use_oracle_boundary=False,
            gauge_basis=None,
        )
        eigs = result.eigenvalues.numpy()

    # Analyze eigenspectrum
    eigs = np.array(eigs)
    eigs_pos = eigs[eigs > 0]

    # Count eigenvalues above thresholds
    thresholds = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    counts = {f"rank_above_{t:.0e}": int((eigs > t).sum()) for t in thresholds}

    # Effective rank
    trace_val = eigs_pos.sum()
    frob_sq = (eigs_pos**2).sum()
    eff_rank = (trace_val**2 / frob_sq) if frob_sq > 0 else 0.0

    # Condition number (of nonzero part)
    nonzero_mask = eigs > 1e-8
    if nonzero_mask.sum() > 0:
        sigma_max = eigs[0]
        sigma_min_nz = eigs[nonzero_mask][-1]
        cond = sigma_max / max(sigma_min_nz, 1e-30)
    else:
        sigma_max = 0.0
        sigma_min_nz = 0.0
        cond = float("inf")

    analysis = {
        "K": K,
        "k": k,
        "actual_k": actual_k,
        "num_params": p,
        "num_queries": N,
        "sigma_max": float(sigma_max),
        "sigma_min_nonzero": float(sigma_min_nz),
        "condition_number": float(cond),
        "effective_rank": float(eff_rank),
        "trace": float(trace_val),
        **counts,
        "eigenvalues": eigs.tolist(),
        "gauge_dims_lost": k - actual_k,
    }

    logger.info(
        "  Result: K=%d, k=%d → σ_max=%.4e, eff_rank=%.1f, rank>1e-4=%d, rank>1e-2=%d",
        K, actual_k, sigma_max, eff_rank,
        counts["rank_above_1e-04"], counts["rank_above_1e-02"],
    )

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Gramian Rank Diagnostic")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", type=str, default="results/gramian_diagnostic")
    parser.add_argument("--K_values", type=int, nargs="+", default=[8, 32, 64, 128])
    parser.add_argument("--k_values", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--query_subsample", type=int, default=256)
    parser.add_argument("--pool_size", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--block_idx", type=int, default=23,
                        help="Which block to analyze (default: last block)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--also_block_22", action="store_true",
                        help="Also compute Gramian for block 22 (expected zero)")
    args = parser.parse_args()

    setup_logging()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model: %s", args.model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    teacher_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build student (copy of teacher, then randomize suffix)
    logger.info("Building student (randomized suffix)...")
    student = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )

    # Untie lm_head before randomization
    lm_head = getattr(student, "lm_head", None)
    embed = getattr(getattr(student, "model", None), "embed_tokens", None)
    if lm_head and embed and lm_head.weight.data_ptr() == embed.weight.data_ptr():
        lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
        if hasattr(student, "config"):
            student.config.tie_word_embeddings = False
        logger.info("Untied lm_head from embed_tokens")

    # Randomize block 23 (and 22 if requested)
    blocks_to_randomize = [args.block_idx]
    if args.also_block_22:
        blocks_to_randomize.append(args.block_idx - 1)

    for block_idx in blocks_to_randomize:
        param_names = get_block_param_names(student, block_idx)
        for name, param in student.named_parameters():
            if name in param_names:
                if param.dim() >= 2:
                    torch.nn.init.kaiming_uniform_(param)
                elif "norm" in name.lower():
                    torch.nn.init.ones_(param)
                else:
                    torch.nn.init.zeros_(param)

    # Randomize lm_head
    for name, param in student.named_parameters():
        if "lm_head" in name and param.dim() >= 2:
            torch.nn.init.kaiming_uniform_(param)

    # Build query pool
    logger.info("Building query pool: %d inputs, max_len=%d",
                args.pool_size, args.max_seq_len)
    query_ids = build_query_pool(tokenizer, args.pool_size, args.max_seq_len, args.seed)

    # Build gauge basis for target blocks
    target_blocks = [args.block_idx]
    if args.also_block_22:
        target_blocks.append(args.block_idx - 1)

    all_results = {}
    for block_idx in target_blocks:
        logger.info("\n" + "=" * 60)
        logger.info("=== Analyzing Block %d ===", block_idx)
        logger.info("=" * 60)

        param_names = get_block_param_names(student, block_idx)
        logger.info("Block %d: %d parameters across %d tensors",
                     block_idx, sum(p.numel() for n, p in student.named_parameters() if n in param_names),
                     len(param_names))

        # Build gauge basis
        spec_for_gauge = sg_build_flat_param_spec(student, param_names)
        gauge_basis = build_suffix_gauge_basis(
            student, spec_for_gauge, [block_idx],
            include_rmsnorm=True, include_mlp=True,
        )
        logger.info("Gauge basis: %d directions", gauge_basis.num_directions)

        # Build minimal cache
        cache = MinimalCache(teacher_model, query_ids, device=device)

        block_results = []
        for K in args.K_values:
            if K > args.max_seq_len:
                logger.warning("K=%d > max_seq_len=%d, skipping", K, args.max_seq_len)
                continue

            for k in args.k_values:
                logger.info("\n--- Block %d: K=%d positions, k=%d probes ---", block_idx, K, k)
                t0 = time.time()
                try:
                    result = compute_gramian_for_config(
                        student=student,
                        cache=cache,
                        param_names=param_names,
                        K=K,
                        k=k,
                        query_subsample=args.query_subsample,
                        gauge_basis=gauge_basis,
                        seed=args.seed,
                        block_idx=block_idx,
                    )
                    result["elapsed_seconds"] = time.time() - t0
                    block_results.append(result)

                    # Save incrementally
                    with open(output_dir / f"block_{block_idx}_incremental.json", "w") as f:
                        json.dump(block_results, f, indent=2)

                except Exception as e:
                    logger.error("Failed for K=%d, k=%d: %s", K, k, e)
                    import traceback
                    traceback.print_exc()
                    block_results.append({
                        "K": K, "k": k, "error": str(e),
                        "elapsed_seconds": time.time() - t0,
                    })

                # Clear GPU cache between configs
                torch.cuda.empty_cache()

        all_results[f"block_{block_idx}"] = block_results

    # Save final results
    with open(output_dir / "gramian_rank_diagnostic.json", "w") as f:
        json.dump({
            "model": args.model_name,
            "block_indices": target_blocks,
            "K_values": args.K_values,
            "k_values": args.k_values,
            "query_subsample": args.query_subsample,
            "pool_size": args.pool_size,
            "max_seq_len": args.max_seq_len,
            "results": all_results,
        }, f, indent=2)

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Gramian Rank vs Observation Size")
    logger.info("=" * 80)
    for block_key, results in all_results.items():
        logger.info("\n%s:", block_key)
        logger.info("%-6s %-6s %-10s %-10s %-10s %-10s %-8s",
                     "K", "k", "σ_max", "eff_rank", "rank>1e-4", "rank>1e-2", "time(s)")
        logger.info("-" * 70)
        for r in results:
            if "error" in r:
                logger.info("%-6d %-6d ERROR: %s", r["K"], r["k"], r.get("error", ""))
            else:
                logger.info("%-6d %-6d %-10.3e %-10.1f %-10d %-10d %-8.1f",
                           r["K"], r.get("actual_k", r["k"]),
                           r["sigma_max"], r["effective_rank"],
                           r["rank_above_1e-04"], r["rank_above_1e-02"],
                           r.get("elapsed_seconds", 0))

    logger.info("\nDiagnostic complete. Results saved to %s", output_dir)

    # Key interpretation
    logger.info("\n=== INTERPRETATION ===")
    b23_results = all_results.get(f"block_{args.block_idx}", [])
    if b23_results:
        baseline = next((r for r in b23_results if r.get("K") == 8 and r.get("k") == 64), None)
        expanded = [r for r in b23_results if r.get("K", 0) > 8 and "error" not in r]
        if baseline and expanded:
            base_rank = baseline.get("rank_above_1e-04", 0)
            max_rank = max(r.get("rank_above_1e-04", 0) for r in expanded)
            if max_rank > base_rank * 1.5:
                logger.info("POSITIVE: Rank increased from %d to %d with more positions!",
                           base_rank, max_rank)
                logger.info("→ Observation bottleneck confirmed. Proceed with expanded-K training.")
            else:
                logger.info("STRUCTURAL LIMIT: Rank stayed at ~%d despite more positions.",
                           base_rank)
                logger.info("→ Bottleneck is in model structure, not observation setup.")
                logger.info("→ Pivot to active query optimization or functional approach.")


if __name__ == "__main__":
    main()
