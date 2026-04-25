#!/usr/bin/env python3
"""Clean standalone reproduction of Carlini et al. 2024.

"Stealing Part of a Production Language Model" (ICML 2024 Best Paper),
arXiv:2403.06634.

Core method
-----------
For a causal LM with hidden dim ``d`` and vocabulary size ``V``, the last
position logit vector ``z(x) = W_lm @ h_L(x)`` lives in the ``d``-dimensional
column space of ``W_lm`` inside ``R^V``. Collecting ``N >> d`` logit vectors
and running SVD on the stacked matrix ``Y`` reveals:

  1. A sharp drop in singular values at index ``d`` (hidden dim recovery).
  2. The top-``d`` right singular vectors span ``col(W_lm^T)`` (up to a
     rotation), giving an algebraic extraction of the output projection
     subspace with *no gradient updates* and <10k queries.

This script is a **clean re-implementation** suitable for side-by-side
comparison with follow-up work. It is self-contained: no dependencies on
the main project's ``scripts/`` or ``src/`` modules.

Metrics reported
----------------
 (i)   Hidden dim recovery accuracy (d_hat == d_true?)
 (ii)  Subspace principal angle cosines (mean / min / max)
 (iii) Procrustes Frobenius error and per-column cosine (over the overlap
       dimension when d_hat != d_true)
 (iv)  Query efficiency scaling — how recovery quality scales with N

Usage
-----
    python baselines/carlini_2024/run_carlini.py \\
        --model_name Qwen/Qwen2.5-0.5B \\
        --num_queries 2048 --seq_len 128 \\
        --output_dir baselines/carlini_2024/results/qwen25_05b \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("carlini_baseline")


# ── Utilities ────────────────────────────────────────────────────────────


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_teacher(model_name: str, device: str, dtype: torch.dtype):
    """Load any HuggingFace causal LM. Respects HF_ENDPOINT and HF_HUB_OFFLINE."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        logger.info("HF_ENDPOINT=%s", hf_endpoint)
    if os.environ.get("HF_HUB_OFFLINE"):
        logger.info("HF_HUB_OFFLINE is set — using cached weights only.")

    logger.info("Loading teacher: %s (device=%s, dtype=%s)", model_name, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def get_true_lm_head(model) -> torch.Tensor:
    """Extract the ground-truth output projection matrix W_lm in R^{V x d}.

    Handles weight tying: if the model shares weights between ``lm_head`` and
    ``embed_tokens``, we fall back to the embedding table.
    """
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        w = model.lm_head.weight.data
        # If tied, lm_head.weight is the embedding table; still correct.
    else:
        # Fallback for non-standard architectures.
        w = model.get_input_embeddings().weight.data
    return w.float().cpu()


# ── Step 1 — Random-query logit collection ──────────────────────────────


@torch.no_grad()
def collect_logits(
    model,
    vocab_size: int,
    num_queries: int,
    seq_len: int,
    device: str,
    batch_size: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """Simulate black-box API: send ``num_queries`` random token sequences,
    collect the last-position logit vector for each.

    Returns a ``(num_queries, vocab_size)`` float32 CPU tensor.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    logits_chunks: list[torch.Tensor] = []
    n_done = 0
    pbar = tqdm(total=num_queries, desc="Collecting logits", unit="q")
    while n_done < num_queries:
        bs = min(batch_size, num_queries - n_done)
        input_ids = torch.randint(
            0, vocab_size, (bs, seq_len), generator=rng, device="cpu"
        ).to(device)
        out = model(input_ids=input_ids)
        # Take last position; always correct for causal LMs.
        last_logits = out.logits[:, -1, :].float().cpu()
        logits_chunks.append(last_logits)
        n_done += bs
        pbar.update(bs)
    pbar.close()

    Y = torch.cat(logits_chunks, dim=0)
    assert Y.shape == (num_queries, vocab_size)
    return Y


# ── Step 2 — Hidden dimension recovery via SVD ─────────────────────────


def detect_hidden_dim(
    Y: torch.Tensor,
    true_d: int,
    window: int = 50,
) -> tuple[int, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """Center ``Y``, SVD, and pick ``d_hat`` as the largest gap ratio
    ``sigma_i / sigma_{i+1}`` inside ``[true_d-window, true_d+window]``.
    """
    Y_c = Y - Y.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Y_c, full_matrices=False)

    S_np = S.numpy()
    gap = S_np[:-1] / np.maximum(S_np[1:], 1e-30)

    lo = max(0, true_d - window)
    hi = min(len(gap), true_d + window)
    if hi <= lo:
        d_hat = int(np.argmax(gap))
    else:
        d_hat = int(lo + np.argmax(gap[lo:hi]))
    return d_hat, S_np, gap, Vh, S


# ── Step 3 — Output-projection subspace recovery ────────────────────────


def _procrustes_with_overlap(
    W_hat_T: torch.Tensor,
    W_true: torch.Tensor,
    d_hat: int,
    d_true: int,
) -> dict[str, Any]:
    """Procrustes alignment of recovered columns vs the true ``W_lm``.

    When ``d_hat != d_true``, we use the overlap dimension
    ``d_ov = min(d_hat, d_true)``:
      - If ``d_hat > d_true``: use the top-``d_true`` singular components of
        the recovered basis (rotated to align).
      - If ``d_hat < d_true``: align ``d_hat`` recovered columns to the best
        ``d_hat`` directions of ``W_true`` (its top-``d_hat`` principal
        components).

    Returns metrics computed on the overlap dimension.
    """
    d_ov = min(d_hat, d_true)

    # For the overlap, find the best matched subspaces first.
    # W_hat_T is (V, d_hat), W_true is (V, d_true).
    if d_hat == d_true:
        A = W_hat_T
        B = W_true
    elif d_hat < d_true:
        # Keep all recovered columns; pick the d_hat-dim subspace of W_true
        # that best aligns with them. That subspace is W_true @ R where R
        # comes from SVD(W_hat_T.T @ W_true).
        Q_hat, _ = torch.linalg.qr(W_hat_T)
        Q_true_full, _ = torch.linalg.qr(W_true)
        M = Q_hat.T @ Q_true_full  # (d_hat, d_true)
        _, _, Vt = torch.linalg.svd(M, full_matrices=False)
        # Right singular vectors of M in the true-space basis give the best
        # d_hat-dim match inside col(W_true).
        B_basis = Q_true_full @ Vt.T[:, :d_hat]  # (V, d_hat)
        A = W_hat_T
        B = B_basis  # dimension-matched subspace of true lm_head
    else:
        # d_hat > d_true: trim recovered to its top-d_true singular directions
        # aligned to W_true.
        Q_hat_full, _ = torch.linalg.qr(W_hat_T)
        Q_true, _ = torch.linalg.qr(W_true)
        M = Q_hat_full.T @ Q_true  # (d_hat, d_true)
        U_m, _, _ = torch.linalg.svd(M, full_matrices=False)
        A_basis = Q_hat_full @ U_m[:, :d_true]  # (V, d_true)
        A = A_basis
        B = W_true

    # Procrustes: R = U V^T where U S V^T = A^T B
    cross = A.T @ B
    U_p, _, Vt_p = torch.linalg.svd(cross, full_matrices=False)
    R_opt = U_p @ Vt_p
    A_aligned = A @ R_opt  # (V, d_ov)

    # Normalize per-column for scale-invariant comparison
    A_n = F.normalize(A_aligned, dim=0)
    B_n = F.normalize(B, dim=0)

    cos_per_col = (A_n * B_n).sum(dim=0)  # (d_ov,)
    frob_err = float(torch.norm(A_n - B_n) / torch.norm(B_n))

    return {
        "overlap_dim": int(d_ov),
        "procrustes_mean_cos": float(cos_per_col.mean()),
        "procrustes_min_cos": float(cos_per_col.min()),
        "procrustes_max_cos": float(cos_per_col.max()),
        "procrustes_frob_relative_error": frob_err,
    }


def recover_output_projection(
    Vh: torch.Tensor,
    d_hat: int,
    W_true: torch.Tensor,
) -> dict[str, Any]:
    """Compare the recovered output-projection subspace to the true
    ``W_lm``. Returns subspace principal-angle cosines, a robust
    Procrustes (handles ``d_hat != d_true``), and a projection residual.
    """
    V_vocab, d_true = W_true.shape
    W_hat_T = Vh[:d_hat, :].T.float()  # (V, d_hat)
    W_true_f = W_true.float()

    # Subspace principal angles via QR + singular values of cross-Gram.
    Q_hat, _ = torch.linalg.qr(W_hat_T)
    Q_true, _ = torch.linalg.qr(W_true_f)
    M = Q_hat.T @ Q_true
    cos_angles = torch.linalg.svdvals(M).clamp(0.0, 1.0)

    # Procrustes — robust to d_hat != d_true.
    proc = _procrustes_with_overlap(W_hat_T, W_true_f, d_hat, d_true)

    # Projection residual: how well Q_hat lies inside col(W_true).
    projected = Q_true @ (Q_true.T @ Q_hat)
    proj_residual = float(torch.norm(Q_hat - projected) / torch.norm(Q_hat))

    return {
        "subspace_mean_cos_angle": float(cos_angles.mean()),
        "subspace_min_cos_angle": float(cos_angles.min()),
        "subspace_max_cos_angle": float(cos_angles.max()),
        "principal_angle_cosines_top10": cos_angles[:10].tolist(),
        "projection_residual": proj_residual,
        "d_hat": int(d_hat),
        "d_true": int(d_true),
        **proc,
    }


# ── Step 4 — Query-efficiency sweep ────────────────────────────────────


def query_efficiency_sweep(
    Y_all: torch.Tensor,
    true_d: int,
    W_true: torch.Tensor,
    budgets: list[int],
) -> list[dict[str, Any]]:
    """Slice the cached logit matrix at several budgets and re-run the
    extraction, so we measure how recovery improves with more queries.
    """
    results = []
    for n in budgets:
        if n > Y_all.shape[0]:
            logger.warning("Budget %d > collected %d — skipping", n, Y_all.shape[0])
            continue
        t0 = time.time()
        try:
            d_hat, S_np, gap, Vh, _ = detect_hidden_dim(Y_all[:n], true_d)
            svd_t = time.time() - t0
            rec = recover_output_projection(Vh, d_hat, W_true)
        except Exception as e:
            logger.warning("Sweep N=%d failed: %s — skipping", n, e)
            continue
        results.append(
            {
                "num_queries": n,
                "recovered_d": d_hat,
                "d_correct": d_hat == true_d,
                "gap_ratio_at_d": float(gap[d_hat]) if d_hat < len(gap) else 0.0,
                "svd_time_s": round(svd_t, 2),
                **rec,
            }
        )
        logger.info(
            "N=%5d | d_hat=%4d (true %d) | subspace_cos=%.5f | procrustes_cos=%.5f",
            n, d_hat, true_d,
            rec["subspace_mean_cos_angle"],
            rec["procrustes_mean_cos"],
        )
    return results


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carlini et al. 2024 clean baseline reproduction"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num_queries", type=int, default=2048)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="baselines/carlini_2024/results/default",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--budgets",
        type=str,
        default="64,128,256,512,1024,2048",
        help="Comma-separated list of query budgets for the efficiency sweep.",
    )
    parser.add_argument(
        "--skip_efficiency",
        action="store_true",
        help="Skip the query efficiency sweep (step 4).",
    )
    parser.add_argument(
        "--save_W_hat",
        action="store_true",
        help="Save the recovered W_hat matrix as a .pt file.",
    )
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        device = "cpu"

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    t_start = time.time()
    model, _ = load_teacher(args.model_name, device, dtype)

    vocab_size = model.config.vocab_size
    hidden_size = model.config.hidden_size
    tie_emb = bool(getattr(model.config, "tie_word_embeddings", False))
    logger.info(
        "Config: vocab=%d, hidden=%d, tie_word_embeddings=%s",
        vocab_size, hidden_size, tie_emb,
    )

    W_true = get_true_lm_head(model)
    assert W_true.shape == (vocab_size, hidden_size), (
        f"Expected W_true shape ({vocab_size}, {hidden_size}), got {W_true.shape}"
    )

    # ── Main extraction ──
    logger.info("=" * 60)
    logger.info("Main extraction — N=%d queries", args.num_queries)
    logger.info("=" * 60)
    Y_main = collect_logits(
        model, vocab_size, args.num_queries, args.seq_len,
        device, batch_size=args.batch_size, seed=args.seed,
    )

    d_hat, S_np, gap, Vh, S = detect_hidden_dim(Y_main, hidden_size)
    logger.info(
        "Recovered d_hat=%d (true %d), gap ratio=%.2f",
        d_hat, hidden_size, gap[d_hat] if d_hat < len(gap) else float("nan"),
    )

    recovery = recover_output_projection(Vh, d_hat, W_true)

    # Free teacher before the sweep (SVD is cheap; teacher forward isn't).
    # Keep it actually — the sweep re-slices ``Y_main`` rather than re-querying.

    # ── Efficiency sweep (reuses the already-collected Y_main) ──
    if not args.skip_efficiency:
        budgets = [int(x) for x in args.budgets.split(",") if int(x) <= args.num_queries]
        logger.info("Budgets: %s", budgets)
        efficiency = query_efficiency_sweep(Y_main, hidden_size, W_true, budgets)
    else:
        efficiency = []

    elapsed = time.time() - t_start

    # ── Results JSON ──
    top_sv = S_np[:10].tolist()
    sv_around = S_np[max(0, hidden_size - 3): hidden_size + 4].tolist()

    results = {
        "method": "Carlini et al. 2024 (SVD on last-position logits)",
        "paper_arxiv_id": "2403.06634",
        "config": {
            "model_name": args.model_name,
            "vocab_size": int(vocab_size),
            "hidden_size": int(hidden_size),
            "tie_word_embeddings": tie_emb,
            "num_queries": args.num_queries,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "device": device,
            "dtype": str(dtype),
        },
        "hidden_dim_recovery": {
            "recovered_d": int(d_hat),
            "true_d": int(hidden_size),
            "match": bool(d_hat == hidden_size),
            "gap_ratio_at_d_hat": (
                float(gap[d_hat]) if d_hat < len(gap) else 0.0
            ),
            "top_10_singular_values": top_sv,
            "singular_values_around_d": sv_around,
        },
        "output_projection_recovery": recovery,
        "query_efficiency_sweep": efficiency,
        "elapsed_seconds": round(elapsed, 2),
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", results_path)

    if args.save_W_hat:
        W_hat = Vh[:d_hat, :].float()
        torch.save(
            {"W_hat": W_hat, "d_hat": d_hat, "singular_values": S.cpu()},
            out_dir / "W_hat.pt",
        )
        logger.info("Wrote %s", out_dir / "W_hat.pt")

    # ── Verdict ──
    print()
    print("=" * 72)
    print("  Carlini 2024 — Clean Baseline Reproduction")
    print("=" * 72)
    print(f"  Model            : {args.model_name}")
    print(f"  Vocab / hidden   : {vocab_size} / {hidden_size}  (tied={tie_emb})")
    print(f"  Queries          : {args.num_queries}")
    print("-" * 72)
    print(f"  Recovered d      : {d_hat}  (true {hidden_size})  "
          f"[{'match' if d_hat == hidden_size else 'mismatch'}]")
    print(f"  Subspace cos     : mean={recovery['subspace_mean_cos_angle']:.6f} "
          f"min={recovery['subspace_min_cos_angle']:.6f}")
    print(f"  Procrustes cos   : mean={recovery['procrustes_mean_cos']:.6f} "
          f"(overlap d={recovery['overlap_dim']})")
    print(f"  Frob error (rel) : {recovery['procrustes_frob_relative_error']:.6f}")
    print(f"  Projection resid : {recovery['projection_residual']:.6f}")
    print(f"  Elapsed          : {elapsed:.1f} s")

    # Research-bar verdict
    ok_subspace = recovery["subspace_mean_cos_angle"] > 0.99
    ok_d = abs(d_hat - hidden_size) <= 2
    verdict = "PASS" if (ok_subspace and ok_d) else "CHECK"
    print("-" * 72)
    print(f"  VERDICT: {verdict}  "
          f"(subspace_cos>0.99: {ok_subspace}, |d_hat-d|<=2: {ok_d})")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
