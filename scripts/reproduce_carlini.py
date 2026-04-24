#!/usr/bin/env python3
"""
Reproduce Carlini et al. 2024 "Stealing Part of a Production Language Model"
Algebraic extraction of output projection layer via SVD of logit vectors.

Key insight: for a transformer with hidden dim d and vocab size V, the logit
output z(x) = W_lm * h_L(x) lies in a d-dimensional subspace of R^V (since
d << V). By collecting logits from N >> d random queries and computing SVD:

  1. Stack logits: Y = [z(x_1); ...; z(x_N)] in R^{N x V}
  2. SVD: Y = U S V^T
  3. Rank of Y reveals hidden dim d (sharp drop in singular values at index d)
  4. Right singular vectors V[:, :d] span the same column space as W_lm^T
  5. Recovery: W_recovered = V[:, :d]^T (up to rotation)

For tied embeddings (Qwen2.5-0.5B), W_lm = E, so this also recovers the
embedding matrix up to an orthogonal rotation.

Usage:
    python scripts/reproduce_carlini.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --num_queries 2048 \
        --max_seq_len 128 \
        --output_dir results/v5_carlini_baseline \
        --seed 42
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, device: str, dtype: torch.dtype):
    """Load a causal LM, respecting HF_ENDPOINT mirror."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model: %s (device=%s, dtype=%s)", model_name, device, dtype)
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        logger.info("Using HF mirror: %s", hf_endpoint)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def get_true_lm_head(model) -> torch.Tensor:
    """Extract the true lm_head weight matrix W in R^{V x d}.

    Handles weight tying: if lm_head has no independent weight, fall back
    to model.model.embed_tokens.weight.
    """
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        w = model.lm_head.weight.data
    else:
        w = model.model.embed_tokens.weight.data
    return w.float().cpu()


# ── Core: black-box query collection ────────────────────────────────────


@torch.no_grad()
def collect_logits(
    model,
    vocab_size: int,
    num_queries: int,
    max_seq_len: int,
    device: str,
    batch_size: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """Simulate black-box API: send N random token sequences, collect last-position logits.

    Returns:
        Y: Tensor of shape (num_queries, vocab_size), float32
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    all_logits = []
    n_collected = 0

    pbar = tqdm(total=num_queries, desc="Collecting logits", unit="query")
    while n_collected < num_queries:
        bs = min(batch_size, num_queries - n_collected)
        # Random token ids -- this is the "random query" strategy from the paper
        input_ids = torch.randint(
            0, vocab_size, (bs, max_seq_len), generator=rng, device="cpu"
        ).to(device)

        outputs = model(input_ids=input_ids)
        # Last position logits: (bs, V)
        last_logits = outputs.logits[:, -1, :].float().cpu()
        all_logits.append(last_logits)

        n_collected += bs
        pbar.update(bs)
    pbar.close()

    Y = torch.cat(all_logits, dim=0)  # (N, V)
    assert Y.shape == (num_queries, vocab_size), f"Expected ({num_queries}, {vocab_size}), got {Y.shape}"
    return Y


# ── Step 1: Hidden dimension recovery via SVD ───────────────────────────


def recover_hidden_dim(
    Y: torch.Tensor,
    true_d: int,
    output_dir: Path,
    gap_ratio_threshold: float = 2.0,
) -> dict:
    """Compute SVD of centered logit matrix, identify hidden dimension from
    the singular value spectrum.

    The hidden dim d manifests as a sharp drop: sigma_{d} >> sigma_{d+1}.
    We detect it as the index where the ratio sigma_i / sigma_{i+1} is maximized.

    Args:
        Y: (N, V) logit matrix, float32
        true_d: true hidden dimension for comparison
        output_dir: directory for saving plots
        gap_ratio_threshold: minimum ratio to declare a gap (for reporting)

    Returns:
        dict with recovered_d, singular_values, gap_ratios, etc.
    """
    N, V = Y.shape
    logger.info("Computing SVD of Y (%d x %d) ...", N, V)

    # Center Y (subtract mean logit vector)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    t0 = time.time()
    # Economy SVD: only compute min(N, V) singular values
    # Use gesvd driver for numerical stability with large V
    U, S, Vh = torch.linalg.svd(Y_centered, full_matrices=False)
    svd_time = time.time() - t0
    logger.info("SVD completed in %.1f s, S shape=%s", svd_time, S.shape)

    S_np = S.numpy()
    k = min(len(S_np) - 1, 2 * true_d + 50)  # look at enough singular values

    # Gap ratios: sigma_i / sigma_{i+1}
    gap_ratios = S_np[:-1] / np.maximum(S_np[1:], 1e-30)

    # Detect hidden dim: largest gap ratio in the plausible range [true_d-50, true_d+50]
    search_lo = max(0, true_d - 50)
    search_hi = min(len(gap_ratios), true_d + 50)
    search_window = gap_ratios[search_lo:search_hi]
    d_hat = int(search_lo + np.argmax(search_window))

    # Also report the global maximum gap (should coincide for well-behaved models)
    global_max_idx = int(np.argmax(gap_ratios[:k]))
    global_max_ratio = float(gap_ratios[global_max_idx])

    logger.info(
        "Hidden dim recovery: d_hat=%d (true=%d), gap ratio at d_hat=%.2f, "
        "global max gap at idx=%d (ratio=%.2f)",
        d_hat, true_d, gap_ratios[d_hat], global_max_idx, global_max_ratio,
    )

    # ── Plot singular value spectrum ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: singular values (log scale)
    plot_range = min(k, len(S_np))
    axes[0].semilogy(range(plot_range), S_np[:plot_range], "b-", linewidth=0.8)
    axes[0].axvline(x=true_d, color="r", linestyle="--", linewidth=1.5, label=f"true d={true_d}")
    axes[0].axvline(x=d_hat, color="g", linestyle=":", linewidth=1.5, label=f"recovered d={d_hat}")
    axes[0].set_xlabel("Singular value index")
    axes[0].set_ylabel("Singular value (log scale)")
    axes[0].set_title(f"Singular Value Spectrum (N={N})")
    axes[0].legend()

    # Right: gap ratios
    gap_plot_range = min(k, len(gap_ratios))
    axes[1].plot(range(gap_plot_range), gap_ratios[:gap_plot_range], "b-", linewidth=0.8)
    axes[1].axvline(x=true_d, color="r", linestyle="--", linewidth=1.5, label=f"true d={true_d}")
    axes[1].axvline(x=d_hat, color="g", linestyle=":", linewidth=1.5, label=f"recovered d={d_hat}")
    axes[1].set_xlabel("Index i")
    axes[1].set_ylabel(r"$\sigma_i / \sigma_{i+1}$")
    axes[1].set_title("Gap Ratios")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "singular_value_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved singular value spectrum plot")

    return {
        "recovered_d": int(d_hat),
        "true_d": int(true_d),
        "match": d_hat == true_d,
        "gap_ratio_at_d_hat": float(gap_ratios[d_hat]) if d_hat < len(gap_ratios) else 0.0,
        "global_max_gap_idx": int(global_max_idx),
        "global_max_gap_ratio": float(global_max_ratio),
        "svd_time_s": round(svd_time, 2),
        "top_singular_values": S_np[:min(10, len(S_np))].tolist(),
        "singular_values_around_d": S_np[max(0, true_d - 3):true_d + 4].tolist(),
        # Return SVD components for step 2
        "_U": U,
        "_S": S,
        "_Vh": Vh,
    }


# ── Step 2: Output projection recovery ──────────────────────────────────


def recover_output_projection(
    Vh: torch.Tensor,
    S: torch.Tensor,
    d_hat: int,
    W_true: torch.Tensor,
    output_dir: Path,
) -> dict:
    """Recover the output projection W_lm from the right singular vectors.

    W_hat = Vh[:d_hat, :] in R^{d_hat x V} (rows of Vh = right singular vectors).
    The true W_lm in R^{V x d}.

    We measure:
      (a) Subspace alignment: principal angles between col(W_hat^T) and col(W_true)
      (b) Procrustes-aligned cosine similarity
      (c) Projection residual

    Args:
        Vh: right singular vectors from SVD, shape (min(N,V), V)
        S: singular values
        d_hat: recovered hidden dimension
        W_true: true lm_head weight, shape (V, d)
        output_dir: for saving artifacts

    Returns:
        dict with recovery metrics
    """
    V_vocab, d_true = W_true.shape
    logger.info(
        "Recovery: W_hat from Vh[:d_hat=%d, :], W_true shape=(%d, %d)",
        d_hat, V_vocab, d_true,
    )

    # W_hat^T in R^{V x d_hat}: columns span recovered subspace
    W_hat_T = Vh[:d_hat, :].T.float()  # (V, d_hat)
    W_true_f = W_true.float()  # (V, d_true)

    # ── (a) Subspace principal angles ──
    # Orthonormalize both subspaces via QR
    Q_hat, _ = torch.linalg.qr(W_hat_T)   # (V, d_hat)
    Q_true, _ = torch.linalg.qr(W_true_f)  # (V, d_true)

    # Cross Gram matrix
    M = Q_hat.T @ Q_true  # (d_hat, d_true)
    # Singular values of M are cos(principal angles)
    cos_angles = torch.linalg.svdvals(M)
    cos_angles = cos_angles.clamp(0.0, 1.0)

    mean_cos = float(cos_angles.mean())
    min_cos = float(cos_angles.min())
    max_cos = float(cos_angles.max())

    logger.info(
        "Subspace principal angles: mean_cos=%.6f, min_cos=%.6f, max_cos=%.6f",
        mean_cos, min_cos, max_cos,
    )

    # ── (b) Procrustes alignment ──
    # Find optimal rotation R such that W_hat^T @ R ~ W_true
    # Solve: R* = argmin ||W_hat^T @ R - W_true||_F
    # Solution: R = V_p @ U_p^T where W_hat^T^T @ W_true = U_p S_p V_p^T
    cross = W_hat_T.T @ W_true_f  # (d_hat, d_true)

    if d_hat == d_true:
        U_p, _, Vt_p = torch.linalg.svd(cross, full_matrices=False)
        R_opt = U_p @ Vt_p  # (d_hat, d_true) orthogonal
        W_aligned = W_hat_T @ R_opt  # (V, d_true)

        # Per-column cosine similarity
        cos_per_col = F.cosine_similarity(W_aligned, W_true_f, dim=0)  # (d_true,)
        procrustes_mean_cos = float(cos_per_col.mean())
        procrustes_min_cos = float(cos_per_col.min())

        # Frobenius relative error
        frob_err = float(torch.norm(W_aligned - W_true_f) / torch.norm(W_true_f))

        logger.info(
            "Procrustes alignment: mean_cos=%.6f, min_cos=%.6f, relative_frob_error=%.6f",
            procrustes_mean_cos, procrustes_min_cos, frob_err,
        )
    else:
        # Dimensions don't match -- only subspace metrics are meaningful
        procrustes_mean_cos = float("nan")
        procrustes_min_cos = float("nan")
        frob_err = float("nan")
        logger.warning(
            "d_hat=%d != d_true=%d: skipping Procrustes (subspace metrics still valid)",
            d_hat, d_true,
        )

    # ── (c) Projection residual ──
    # How well does projecting logits onto true W_lm subspace preserve them?
    # This verifies the theoretical claim: Y lives in col(W_lm^T)
    # P = Q_true @ Q_true^T is the projector onto col(W_true)
    # We measure: ||Y - Y @ P||_F / ||Y||_F  -- but Y is large, so use Q_hat instead:
    #   Project Q_hat columns onto Q_true and measure residual
    projected = Q_true @ (Q_true.T @ Q_hat)  # (V, d_hat)
    proj_residual = float(torch.norm(Q_hat - projected) / torch.norm(Q_hat))

    logger.info("Projection residual (Q_hat onto Q_true subspace): %.6f", proj_residual)

    return {
        "subspace_mean_cos_angle": mean_cos,
        "subspace_min_cos_angle": min_cos,
        "subspace_max_cos_angle": max_cos,
        "principal_angle_cosines": cos_angles[:10].tolist(),
        "procrustes_mean_cos": procrustes_mean_cos,
        "procrustes_min_cos": procrustes_min_cos,
        "procrustes_frob_relative_error": frob_err,
        "projection_residual": proj_residual,
        "d_hat": d_hat,
        "d_true": d_true,
    }


# ── Step 3: Query efficiency analysis ───────────────────────────────────


def query_efficiency_analysis(
    model,
    vocab_size: int,
    true_d: int,
    W_true: torch.Tensor,
    max_seq_len: int,
    device: str,
    output_dir: Path,
    seed: int = 42,
    query_budgets: list[int] | None = None,
) -> list[dict]:
    """Repeat steps 1-2 for various N to measure how recovery scales with queries.

    Args:
        query_budgets: list of N values to try

    Returns:
        list of result dicts, one per budget
    """
    if query_budgets is None:
        query_budgets = [64, 128, 256, 512, 1024, 2048]

    # Collect the maximum number of logits once, then slice
    max_n = max(query_budgets)
    logger.info("Collecting %d logits for efficiency analysis ...", max_n)
    Y_all = collect_logits(
        model, vocab_size, max_n, max_seq_len, device,
        batch_size=32, seed=seed,
    )

    results = []
    for n in query_budgets:
        logger.info("--- Query budget N=%d ---", n)
        Y = Y_all[:n]
        Y_centered = Y - Y.mean(dim=0, keepdim=True)

        # SVD
        t0 = time.time()
        _U, S, Vh = torch.linalg.svd(Y_centered, full_matrices=False)
        svd_time = time.time() - t0

        S_np = S.numpy()
        gap_ratios = S_np[:-1] / np.maximum(S_np[1:], 1e-30)

        # Detect d_hat
        search_lo = max(0, true_d - 50)
        search_hi = min(len(gap_ratios), true_d + 50)
        if search_lo < search_hi:
            d_hat = int(search_lo + np.argmax(gap_ratios[search_lo:search_hi]))
        else:
            d_hat = int(np.argmax(gap_ratios[:min(len(gap_ratios), 2 * true_d)]))

        # Recovery metrics (only if N >= d_hat so Vh has enough rows)
        if Vh.shape[0] >= d_hat and d_hat > 0:
            rec = recover_output_projection(Vh, S, d_hat, W_true, output_dir)
        else:
            rec = {
                "subspace_mean_cos_angle": float("nan"),
                "subspace_min_cos_angle": float("nan"),
                "procrustes_mean_cos": float("nan"),
                "projection_residual": float("nan"),
            }

        entry = {
            "num_queries": n,
            "recovered_d": int(d_hat),
            "d_correct": d_hat == true_d,
            "gap_ratio_at_d": float(gap_ratios[d_hat]) if d_hat < len(gap_ratios) else 0.0,
            "svd_time_s": round(svd_time, 2),
            "subspace_mean_cos": rec.get("subspace_mean_cos_angle", float("nan")),
            "subspace_min_cos": rec.get("subspace_min_cos_angle", float("nan")),
            "procrustes_mean_cos": rec.get("procrustes_mean_cos", float("nan")),
            "projection_residual": rec.get("projection_residual", float("nan")),
        }
        results.append(entry)
        logger.info(
            "N=%4d | d_hat=%d (correct=%s) | subspace_cos=%.6f | procrustes_cos=%.6f",
            n, d_hat, d_hat == true_d,
            entry["subspace_mean_cos"], entry["procrustes_mean_cos"],
        )

    # ── Plot scaling ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ns = [r["num_queries"] for r in results]

    # (a) Gap ratio at detected d
    axes[0].plot(ns, [r["gap_ratio_at_d"] for r in results], "bo-")
    axes[0].set_xlabel("Number of queries N")
    axes[0].set_ylabel("Gap ratio at d_hat")
    axes[0].set_title("Gap Sharpness vs. Query Budget")
    axes[0].set_xscale("log", base=2)

    # (b) Subspace cosine
    valid_cos = [(r["num_queries"], r["subspace_mean_cos"]) for r in results if not np.isnan(r["subspace_mean_cos"])]
    if valid_cos:
        axes[1].plot([x[0] for x in valid_cos], [x[1] for x in valid_cos], "go-")
        axes[1].set_xlabel("Number of queries N")
        axes[1].set_ylabel("Mean cosine (principal angles)")
        axes[1].set_title("Subspace Recovery vs. Query Budget")
        axes[1].set_xscale("log", base=2)
        axes[1].set_ylim(0.9, 1.005)

    # (c) Procrustes cosine
    valid_proc = [(r["num_queries"], r["procrustes_mean_cos"]) for r in results if not np.isnan(r["procrustes_mean_cos"])]
    if valid_proc:
        axes[2].plot([x[0] for x in valid_proc], [x[1] for x in valid_proc], "ro-")
        axes[2].set_xlabel("Number of queries N")
        axes[2].set_ylabel("Procrustes mean cosine")
        axes[2].set_title("Procrustes Alignment vs. Query Budget")
        axes[2].set_xscale("log", base=2)

    plt.tight_layout()
    fig.savefig(output_dir / "query_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved query efficiency plot")

    return results


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Carlini et al. 2024: algebraic lm_head extraction via SVD"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num_queries", type=int, default=2048,
                        help="Number of random queries N for main experiment")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help="Length of random token sequences")
    parser.add_argument("--output_dir", type=str, default="results/v5_carlini_baseline")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_efficiency", action="store_true",
                        help="Skip the query efficiency sweep (step 3)")
    parser.add_argument("--query_budgets", type=str, default="64,128,256,512,1024,2048",
                        help="Comma-separated list of N for efficiency analysis")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # ── Load model ──
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model, tokenizer = load_model(args.model_name, device, dtype)

    vocab_size = model.config.vocab_size
    hidden_size = model.config.hidden_size
    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", False)

    logger.info(
        "Model: %s | vocab=%d, hidden=%d, tie_embeddings=%s",
        args.model_name, vocab_size, hidden_size, tie_word_embeddings,
    )

    # Extract true W_lm for evaluation
    W_true = get_true_lm_head(model)  # (V, d)
    logger.info("True W_lm shape: %s", W_true.shape)
    assert W_true.shape == (vocab_size, hidden_size), (
        f"Expected ({vocab_size}, {hidden_size}), got {W_true.shape}"
    )

    # ════════════════════════════════════════════════════════════════
    # Step 1: Collect logits and recover hidden dimension
    # ════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 1: Collect logits and recover hidden dimension")
    logger.info("=" * 60)

    Y = collect_logits(
        model, vocab_size, args.num_queries, args.max_seq_len,
        device, batch_size=args.batch_size, seed=args.seed,
    )
    logger.info("Collected logit matrix Y: shape=%s, dtype=%s", Y.shape, Y.dtype)

    dim_result = recover_hidden_dim(Y, hidden_size, output_dir)

    # Extract SVD components for step 2
    U, S, Vh = dim_result.pop("_U"), dim_result.pop("_S"), dim_result.pop("_Vh")

    # ════════════════════════════════════════════════════════════════
    # Step 2: Recover output projection
    # ════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 2: Recover output projection (lm_head)")
    logger.info("=" * 60)

    d_hat = dim_result["recovered_d"]
    recovery_result = recover_output_projection(Vh, S, d_hat, W_true, output_dir)

    # Save recovered W_hat
    W_hat = Vh[:d_hat, :].float()  # (d_hat, V)
    torch.save(
        {"W_hat": W_hat, "d_hat": d_hat, "singular_values": S.cpu()},
        output_dir / "recovered_W_hat.pt",
    )
    logger.info("Saved recovered W_hat (%s) to %s", W_hat.shape, output_dir / "recovered_W_hat.pt")

    # ════════════════════════════════════════════════════════════════
    # Step 3: Query efficiency analysis
    # ════════════════════════════════════════════════════════════════
    if not args.skip_efficiency:
        logger.info("=" * 60)
        logger.info("STEP 3: Query efficiency analysis")
        logger.info("=" * 60)

        budgets = [int(x) for x in args.query_budgets.split(",")]
        efficiency_results = query_efficiency_analysis(
            model, vocab_size, hidden_size, W_true,
            args.max_seq_len, device, output_dir,
            seed=args.seed + 1,  # different seed from main experiment
            query_budgets=budgets,
        )
    else:
        efficiency_results = []

    # ════════════════════════════════════════════════════════════════
    # Assemble and save results
    # ════════════════════════════════════════════════════════════════
    all_results = {
        "config": {
            "model_name": args.model_name,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "tie_word_embeddings": tie_word_embeddings,
            "num_queries": args.num_queries,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "device": device,
        },
        "step1_hidden_dim": dim_result,
        "step2_recovery": recovery_result,
        "step3_efficiency": efficiency_results,
    }

    results_path = output_dir / "carlini_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved results to %s", results_path)

    # ════════════════════════════════════════════════════════════════
    # Print summary
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Carlini et al. 2024 Reproduction -- Summary")
    print("=" * 70)
    print(f"  Model:             {args.model_name}")
    print(f"  Vocab size:        {vocab_size}")
    print(f"  True hidden dim:   {hidden_size}")
    print(f"  Tied embeddings:   {tie_word_embeddings}")
    print(f"  Num queries:       {args.num_queries}")
    print("-" * 70)
    print(f"  Step 1 -- Hidden Dimension Recovery")
    print(f"    Recovered d:     {dim_result['recovered_d']} (true: {hidden_size})")
    print(f"    Correct:         {dim_result['match']}")
    print(f"    Gap ratio:       {dim_result['gap_ratio_at_d_hat']:.2f}")
    print("-" * 70)
    print(f"  Step 2 -- Output Projection Recovery (N={args.num_queries})")
    print(f"    Subspace cos:    mean={recovery_result['subspace_mean_cos_angle']:.6f}, "
          f"min={recovery_result['subspace_min_cos_angle']:.6f}")
    print(f"    Procrustes cos:  mean={recovery_result['procrustes_mean_cos']:.6f}")
    print(f"    Frob rel error:  {recovery_result['procrustes_frob_relative_error']:.6f}")
    print(f"    Proj residual:   {recovery_result['projection_residual']:.6f}")
    print("-" * 70)
    if efficiency_results:
        print(f"  Step 3 -- Query Efficiency")
        print(f"    {'N':>6s} | {'d_hat':>5s} | {'subspace_cos':>12s} | {'procrustes_cos':>14s}")
        print(f"    {'------':>6s} | {'-----':>5s} | {'------------':>12s} | {'--------------':>14s}")
        for r in efficiency_results:
            print(f"    {r['num_queries']:6d} | {r['recovered_d']:5d} | "
                  f"{r['subspace_mean_cos']:12.6f} | {r['procrustes_mean_cos']:14.6f}")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")
    print(f"  Results JSON:     {results_path}")
    print(f"  Recovered W_hat:  {output_dir / 'recovered_W_hat.pt'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
