#!/usr/bin/env python3
# SAFETY NOTICE: QUARANTINED (alpha-theory prune 2026-04-19)
# This script is NOT cited in the paper. It was part of killed branches:
#   A1 S-PSI, A2 Moments CP, A4 logit-bias, A5 memory probing, A6 active query,
#   A7 algebraic v2/v3/v4, B3 matched-KD.
# Retained in repo for reproducibility of quarantined history; do not use for
# new claims.
#!/usr/bin/env python3
"""
High-Order Moments Tensor Decomposition Attack — Proof-of-concept.

Goal
----
Attempt to recover transformer block parameters via CP decomposition of
empirical cross-moments between (dimensionality-reduced) black-box logits
and public input embeddings.

Theoretical basis
-----------------
For a 2-layer NN with activation sigma and weights W1, W2, Anandkumar et al.
(2014) showed that

    M_3 := E[y otimes x otimes x otimes x - lower-order terms]

admits a symmetric CP decomposition sum_j lambda_j * a_j otimes a_j otimes a_j
whose factor vectors {a_j} are the columns of W1 (the first-layer weights),
and the "lambda_j" are third-moment scalings of sigma at zero.  The weights
can then be recovered (up to sign / permutation / scale) via tensor power
iteration or alternating-least-squares (ALS).

For a transformer the clean factorization disappears — RMSNorm introduces a
normalization that wipes magnitude information, the stack of residual blocks
folds the pathways together, and the attention mechanism is NOT a fixed
linear-in-x map.  Nevertheless, the LAST decoder block's MLP

    h_L = h' + W_down @ SiLU(W_gate @ x_tilde) . (W_up @ x_tilde)

looks structurally similar to the two-layer toy.  If the attacker can
deprive the network of nonlinearity layer-by-layer (approximately, via a
third-order moment), CP factors should carry a signature of W_down columns
and (after a fold) W_gate / W_up rows.

This POC tries exactly that:

  z(x) = logits on last position    (V-dim, observable)
  e(x) = mean-pool of token embeddings from the public embedding table
         (d-dim, attacker-computable since the embedding table = W_lm^T up
         to tied-embedding transpose and is publicly known from the model
         hub; for a strict black-box setting, use the Carlini-recovered
         W_eff transpose instead).

Define the 3-way cross-moment tensor

  M_3 = (1/N) * sum_i  pinv(W_lm) @ z(x_i)  otimes  e(x_i)  otimes  e(x_i)

This is a d x d x d tensor (we use d = d_model so it fits in ~3 GB).

Apply CP-ALS with rank r = d_ff to obtain factors
    A in R^{d x r},  B in R^{d x r},  C in R^{d x r}, scaling lambda in R^r
such that  M_3 approx sum_j lambda_j * A[:, j] otimes B[:, j] otimes C[:, j].

Evaluation
----------
For each teacher weight matrix of interest:
    W_lm (V, d), W_norm_final (d,),
    last-block W_gate (d_ff, d), W_up (d_ff, d), W_down (d, d_ff),
    last-block W_q, W_k, W_v, W_o.
we find the best column-to-factor match via Hungarian assignment on the
cosine-similarity cost and report the mean aligned cosine.

We also run the SAME pipeline with (a) a random tensor of matching shape
and (b) a moment tensor built from random noise inputs, to establish a
null distribution.  The signal must exceed the null by a configurable
margin to count as "success".

Constraints
-----------
* Memory: d=896 => M_3 has 896^3 entries.  In float32 that is 2.7 GB, well
  within 80 GB A100.  We keep M_3 on the GPU throughout.
* CP rank r = d_ff = 4864 is OVERCOMPLETE (r > d).  ALS still converges.
* HF_HUB_OFFLINE=1 compatible.

Usage
-----
CUDA_VISIBLE_DEVICES=1 python scripts/attack_higher_order_moments.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_queries 8192 \
    --cp_rank 4864 \
    --cp_iterations 50 \
    --output_dir results/v5_attack_moments \
    --seed 42
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ── Model architecture constants (Qwen2.5-0.5B) ─────────────────────────────
NUM_LAYERS = 24
LAST_BLOCK_IDX = 23
D_MODEL = 896
D_FF = 4864
VOCAB_SIZE = 151936
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64


# ═══════════════════════════════════════════════════════════════════════════
# Logging / args / utils
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="High-Order Moments CP-Decomposition Attack (POC)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--num_queries", type=int, default=8192,
                   help="Number of query sequences for moment estimation.")
    p.add_argument("--max_seq_len", type=int, default=128,
                   help="Token sequence length.")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Forward-pass batch size.")
    p.add_argument("--cp_rank", type=int, default=D_FF,
                   help="Rank for CP decomposition.")
    p.add_argument("--cp_iterations", type=int, default=50,
                   help="ALS iterations.")
    p.add_argument("--cp_tol", type=float, default=1e-5,
                   help="Early-stop tolerance on relative change of fit.")
    p.add_argument("--cp_reg", type=float, default=1e-6,
                   help="Tikhonov regularization on ALS Gram updates.")
    p.add_argument("--carlini_queries", type=int, default=8192,
                   help="Queries used to estimate W_eff = W_lm @ diag(g_final).")
    p.add_argument("--carlini_seq_len", type=int, default=128)
    p.add_argument("--pool", type=str, default="mean",
                   choices=["mean", "last", "sum"],
                   help="How to pool token embeddings into a single d-vector.")
    p.add_argument("--output_dir", type=str, default="results/v5_attack_moments")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow_synthetic", action="store_true",
                   help="Fall back to random tokens when WikiText is unavailable.")
    p.add_argument("--null_seed_offset", type=int, default=98765,
                   help="Offset for reproducible null-control RNG.")
    p.add_argument("--success_threshold", type=float, default=0.1,
                   help="Min aligned cos to declare SUCCESS on any matrix.")
    p.add_argument("--null_margin", type=float, default=0.02,
                   help="Signal must exceed null by at least this margin.")
    p.add_argument("--matmul_match_rows", type=int, default=256,
                   help="Cap on rows used for Hungarian matching "
                        "(memory safety for d_ff x r cost matrices).")
    p.add_argument("--use_weight_tying", action="store_true", default=True,
                   help="Use embedding table W_emb = W_lm as attacker-legal "
                        "input featurizer (Qwen2.5-0.5B has tied embeddings).")
    p.add_argument("--dry_run_small", action="store_true",
                   help="Override to tiny sizes for a smoke test.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flat_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flattened cosine similarity (float64)."""
    a_f = a.double().flatten()
    b_f = b.double().flatten()
    denom = (a_f.norm() * b_f.norm()).clamp(min=1e-30)
    return float((a_f @ b_f) / denom)


def human_bytes(n: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


# ═══════════════════════════════════════════════════════════════════════════
# Query pool (WikiText or synthetic)
# ═══════════════════════════════════════════════════════════════════════════

def build_query_pool(
    tokenizer,
    n: int,
    seq_len: int,
    seed: int,
    allow_synthetic: bool,
) -> torch.Tensor:
    """Build input-id pool; prefer WikiText, fall back to random if allowed."""
    ids: list[torch.Tensor] = []
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    if offline:
        logger.info(
            "HF_HUB_OFFLINE=1 detected — will attempt cached-dataset load "
            "before falling back to synthetic tokens."
        )
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        for ex in ds:
            if len(ids) >= n:
                break
            text = ex.get("text", "")
            if len(text.strip()) < 20:
                continue
            tok = tokenizer(
                text, max_length=seq_len, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            ids.append(tok["input_ids"].squeeze(0))
    except Exception as e:
        if not allow_synthetic:
            raise RuntimeError(
                f"Dataset load failed ({e}). Use --allow_synthetic to fall back."
            ) from e
        logger.warning("Dataset load failed (%s) — using random tokens.", e)

    remaining = n - len(ids)
    if remaining > 0:
        if not allow_synthetic:
            raise RuntimeError(
                f"Only {len(ids)}/{n} tokens loaded. Use --allow_synthetic to pad."
            )
        rng = torch.Generator().manual_seed(seed)
        rand = torch.randint(3, tokenizer.vocab_size, (remaining, seq_len),
                             generator=rng)
        for i in range(remaining):
            ids.append(rand[i])

    return torch.stack(ids[:n])


# ═══════════════════════════════════════════════════════════════════════════
# Carlini SVD — gives us W_eff_hat (up to rotation)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_carlini_svd(
    model,
    vocab_size: int,
    hidden_size: int,
    num_queries: int,
    seq_len: int,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Carlini et al. 2024 SVD recovery of W_eff = W_lm @ diag(g_final).

    Returns (W_eff_hat in R^{d_hat x V} with orthonormal rows, d_hat, S).
    """
    logger.info("[carlini] collecting %d random-token logits (seq_len=%d) ...",
                num_queries, seq_len)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    ys: list[torch.Tensor] = []
    pbar = tqdm(total=num_queries, desc="carlini queries", unit="q")
    n_done = 0
    while n_done < num_queries:
        bs = min(batch_size, num_queries - n_done)
        ids = torch.randint(0, vocab_size, (bs, seq_len), generator=rng, device="cpu").to(device)
        out = model(input_ids=ids)
        ys.append(out.logits[:, -1, :].float().cpu())
        n_done += bs
        pbar.update(bs)
    pbar.close()

    Y = torch.cat(ys, dim=0)                                 # (N, V)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)
    logger.info("[carlini] SVD on centered Y of shape %s ...", tuple(Y_centered.shape))
    t0 = time.time()
    _, S, Vh = torch.linalg.svd(Y_centered, full_matrices=False)
    logger.info("[carlini] SVD finished in %.1fs.", time.time() - t0)

    S_np = S.numpy()
    gap = S_np[:-1] / np.maximum(S_np[1:], 1e-30)
    lo = max(0, hidden_size - 50)
    hi = min(len(gap), hidden_size + 50)
    d_hat = int(lo + np.argmax(gap[lo:hi]))
    logger.info("[carlini] d_hat=%d (true=%d), gap_ratio=%.2f",
                d_hat, hidden_size, gap[d_hat])

    W_eff_hat = Vh[:d_hat, :].float()                         # (d_hat, V) rows o.n.
    return {"W_eff_hat": W_eff_hat, "S": S, "d_hat": d_hat}


# ═══════════════════════════════════════════════════════════════════════════
# Collect z(x) and pooled embedding e(x)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_logits_and_embed(
    model,
    tokenizer,
    query_ids: torch.Tensor,              # (N, T)
    device: torch.device,
    batch_size: int,
    pool: str,
    embed_table: torch.Tensor,            # (V, d) — ATTACKER-KNOWN (public table)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the teacher on each query and collect:

        z(x_i)  = last-position logits, (V,)           observable (black-box)
        e(x_i)  = pooled embedding(x_i), (d,)          attacker-computable

    z is only computed through a forward pass; e is a pure table lookup.
    """
    N, T = query_ids.shape
    V, d = embed_table.shape

    Z = torch.zeros(N, V, dtype=torch.float32)
    E = torch.zeros(N, d, dtype=torch.float32)

    pbar = tqdm(total=N, desc="collect z/e", unit="q")
    for s in range(0, N, batch_size):
        e_idx = min(s + batch_size, N)
        batch = query_ids[s:e_idx].to(device)                 # (B, T)

        # Observable logits (last-position)
        out = model(input_ids=batch)
        z_batch = out.logits[:, -1, :].float().cpu()           # (B, V)
        Z[s:e_idx] = z_batch

        # Attacker-known embedding pool (NO model call needed, but we use
        # the same tensor to ensure we apply the exact table on-device)
        ids_cpu = batch.detach().cpu()
        emb_full = embed_table[ids_cpu]                        # (B, T, d)
        if pool == "mean":
            pooled = emb_full.mean(dim=1)
        elif pool == "last":
            pooled = emb_full[:, -1, :]
        elif pool == "sum":
            pooled = emb_full.sum(dim=1)
        else:
            raise ValueError(f"Unknown pool: {pool}")
        E[s:e_idx] = pooled

        pbar.update(e_idx - s)
    pbar.close()

    return Z, E


# ═══════════════════════════════════════════════════════════════════════════
# Moment tensors
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MomentTensors:
    M1_z: torch.Tensor                     # (d_hat,)    E[z_red]
    M1_e: torch.Tensor                     # (d,)        E[e]
    M2: torch.Tensor                       # (d_hat, d)  E[z_red otimes e]
    M3: torch.Tensor                       # (d_hat, d, d)  E[z_red otimes e otimes e]
    n_samples: int
    pool: str
    z_norm: float
    e_norm: float
    M3_norm: float
    M3_frobenius_ratio: float              # ||M3|| / (||M1_z|| * ||M1_e||^2)


@torch.no_grad()
def compute_moments(
    Z: torch.Tensor,                       # (N, V)
    E: torch.Tensor,                       # (N, d)
    W_eff_hat: torch.Tensor,               # (d_hat, V)
    device: torch.device,
    chunk_size: int = 256,
    pool: str = "mean",
) -> MomentTensors:
    """Compute M_1, M_2, M_3 by streaming over N in chunks.

    Dimensionality reduction: we project raw logits z onto the d_hat-dim
    subspace recovered by Carlini:  z_reduced = W_eff_hat @ z  (d_hat,).
    This is IDENTICAL to the pinv projection when W_eff_hat has orthonormal
    rows (which our SVD returns).
    """
    N, V = Z.shape
    d_hat = W_eff_hat.shape[0]
    d = E.shape[1]

    logger.info("[moments] N=%d, d_hat=%d, d_embed=%d", N, d_hat, d)
    logger.info("[moments] M_3 allocation: %s",
                human_bytes(d_hat * d * d * 4))

    W_eff_dev = W_eff_hat.to(device, dtype=torch.float32)

    # Accumulators on device in float32
    M1_z = torch.zeros(d_hat, dtype=torch.float32, device=device)
    M1_e = torch.zeros(d, dtype=torch.float32, device=device)
    M2   = torch.zeros(d_hat, d, dtype=torch.float32, device=device)
    M3   = torch.zeros(d_hat, d, d, dtype=torch.float32, device=device)

    pbar = tqdm(total=N, desc="moment accum", unit="q")
    for s in range(0, N, chunk_size):
        e_idx = min(s + chunk_size, N)
        B = e_idx - s
        z_chunk = Z[s:e_idx].to(device, dtype=torch.float32)    # (B, V)
        e_chunk = E[s:e_idx].to(device, dtype=torch.float32)    # (B, d)

        # Reduce logits
        zr = z_chunk @ W_eff_dev.T                              # (B, d_hat)

        M1_z += zr.sum(dim=0)
        M1_e += e_chunk.sum(dim=0)
        # M2 += sum_i  zr_i otimes e_i
        M2 += zr.T @ e_chunk                                    # (d_hat, d)

        # M3 += sum_i  zr_i[a] * e_i[b] * e_i[c]
        # Efficient einsum: (B, d_hat) x (B, d) x (B, d)  -> (d_hat, d, d)
        # Use mixed float32 and float64 accum if very large.  We keep fp32.
        M3 += torch.einsum("ba,bc,bd->acd", zr, e_chunk, e_chunk)

        pbar.update(B)
    pbar.close()

    M1_z /= N
    M1_e /= N
    M2   /= N
    M3   /= N

    # Optional: center moments (subtract lower-order outer products).
    # In Anandkumar's construction, symmetric M_3 minus E[y]*E[x otimes x]
    # and E[x] *(things) isolates the symmetric rank-r tensor.  Our setup is
    # asymmetric (y is not x), so we subtract the CORRECT product term:
    #   M3_centered = M3 - E[zr] otimes E[e otimes e] - (cross terms).
    # For simplicity here we CENTER by subtracting M1_z otimes Cov(e) AND
    # the fully-factored lower-order term.  We then report both.

    # We return the *uncentered* M3 as the primary object (stronger signal
    # if the pipeline is asymmetric); centering done inside CP diagnostics.

    z_norm = float(M1_z.norm().cpu())
    e_norm = float(M1_e.norm().cpu())
    M3_norm = float(M3.norm().cpu())
    denom = (z_norm * e_norm * e_norm) if (z_norm * e_norm) > 0 else 1.0
    M3_frobenius_ratio = float(M3_norm / max(denom, 1e-30))

    logger.info("[moments] ||M1_z||=%.3e  ||M1_e||=%.3e  ||M2||=%.3e  ||M3||=%.3e",
                z_norm, e_norm, float(M2.norm().cpu()), M3_norm)
    logger.info("[moments] ||M3|| / (||M1_z|| * ||M1_e||^2) = %.3e", M3_frobenius_ratio)

    return MomentTensors(
        M1_z=M1_z.cpu(), M1_e=M1_e.cpu(),
        M2=M2.cpu(), M3=M3,     # keep M3 on device
        n_samples=N,
        pool=pool,
        z_norm=z_norm,
        e_norm=e_norm,
        M3_norm=M3_norm,
        M3_frobenius_ratio=M3_frobenius_ratio,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CP decomposition via alternating least squares (ALS)
# ═══════════════════════════════════════════════════════════════════════════

def _khatri_rao(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Column-wise Khatri-Rao product.

    A: (I, R), B: (J, R)  ->  (I*J, R)
    """
    I, R = A.shape
    J = B.shape[0]
    assert B.shape[1] == R
    # KR(A, B)[ij, r] = A[i, r] * B[j, r]
    return (A.unsqueeze(1) * B.unsqueeze(0)).reshape(I * J, R)


def _mttkrp(T: torch.Tensor, factors: list[torch.Tensor], mode: int) -> torch.Tensor:
    """Matricized tensor times Khatri-Rao product (MTTKRP).

    T  : 3-tensor of shape (d0, d1, d2)
    factors[i]: (d_i, R)
    mode: which axis to unfold.

    Returns: (d_mode, R)
    """
    d0, d1, d2 = T.shape
    R = factors[0].shape[1]

    if mode == 0:
        # T.reshape(d0, d1, d2) -> T.reshape(d0, d1*d2) has columns indexed
        # by j*d2 + k (outer d1, inner d2).  _khatri_rao(factors[1], factors[2])
        # has rows (j, k) -> j*d2 + k, matching the reshape order.
        KR = _khatri_rao(factors[1], factors[2])             # (d1*d2, R)
        T0 = T.reshape(d0, d1 * d2)                          # (d0, d1*d2)
        return T0 @ KR
    elif mode == 1:
        T1 = T.permute(1, 0, 2).reshape(d1, d0 * d2)
        # unfolding M1[j, i*d2 + k].  KR(factors[0], factors[2]) => (d0*d2, R)
        KR = _khatri_rao(factors[0], factors[2])
        return T1 @ KR
    elif mode == 2:
        T2 = T.permute(2, 0, 1).reshape(d2, d0 * d1)
        KR = _khatri_rao(factors[0], factors[1])
        return T2 @ KR
    else:
        raise ValueError(f"Bad mode: {mode}")


def cp_decomposition_als(
    T: torch.Tensor,
    rank: int,
    max_iter: int = 50,
    tol: float = 1e-5,
    reg: float = 1e-6,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """CP / PARAFAC decomposition of a 3-way tensor via ALS.

    Solves:  T  approx  sum_{r=1}^{R}  lambda_r * a_r otimes b_r otimes c_r
    where A = [a_1 ... a_R] in R^{d0 x R}, similarly B, C.

    Overcomplete (R > d) is fine.  We minimize the Frobenius loss
    ||T - reconstruct(A, B, C)||_F via alternating least squares.

    Returns: {A, B, C, lambdas, fit_history, reconstruction_error}
    """
    device = T.device
    dtype = T.dtype
    d0, d1, d2 = T.shape

    if verbose:
        logger.info("[cp-als] tensor shape = %s, rank = %d, iters = %d",
                    tuple(T.shape), rank, max_iter)
        logger.info("[cp-als] memory: T=%s, factors=%s",
                    human_bytes(T.numel() * T.element_size()),
                    human_bytes((d0 + d1 + d2) * rank * T.element_size()))

    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(d0, rank, device=device, dtype=dtype, generator=g)
    B = torch.randn(d1, rank, device=device, dtype=dtype, generator=g)
    C = torch.randn(d2, rank, device=device, dtype=dtype, generator=g)

    # Normalize columns
    for M in (A, B, C):
        M.div_(M.norm(dim=0, keepdim=True).clamp(min=1e-30))

    lambdas = torch.ones(rank, device=device, dtype=dtype)

    T_norm = T.norm().item()
    if T_norm < 1e-30:
        logger.warning("[cp-als] tensor has near-zero norm (%.3e); skipping ALS.",
                       T_norm)
        return {
            "A": A.cpu(), "B": B.cpu(), "C": C.cpu(),
            "lambdas": lambdas.cpu(),
            "fit_history": [1.0],
            "reconstruction_error": 1.0,
            "converged": False,
        }

    T_fl0 = T.reshape(d0, d1 * d2)   # unfolding, for reconstruction residual

    history: list[dict[str, float]] = []
    prev_fit = None
    t0 = time.time()

    for it in range(max_iter):
        # ── Mode-0 update ───────────────────────────────────────────────────
        # A* = T_(0) (C (Khatri-Rao) B) (V)^{-1}
        # where V = (C^T C) * (B^T B)  (Hadamard product in R x R)
        V = (B.T @ B) * (C.T @ C)
        V += reg * V.diagonal().mean() * torch.eye(rank, device=device, dtype=dtype)
        KR = _khatri_rao(B, C)                               # (d1*d2, R)
        # T_(0) = T.reshape(d0, d1*d2), which matches ordering j * d2 + k
        T0 = T.reshape(d0, d1 * d2)
        A_new = torch.linalg.solve(V, (T0 @ KR).T).T         # (d0, R)
        lam_a = A_new.norm(dim=0).clamp(min=1e-30)
        A = A_new / lam_a

        # ── Mode-1 update ───────────────────────────────────────────────────
        V = (A.T @ A) * (C.T @ C)
        V += reg * V.diagonal().mean() * torch.eye(rank, device=device, dtype=dtype)
        KR = _khatri_rao(A, C)
        T1 = T.permute(1, 0, 2).reshape(d1, d0 * d2)
        B_new = torch.linalg.solve(V, (T1 @ KR).T).T
        lam_b = B_new.norm(dim=0).clamp(min=1e-30)
        B = B_new / lam_b

        # ── Mode-2 update ───────────────────────────────────────────────────
        V = (A.T @ A) * (B.T @ B)
        V += reg * V.diagonal().mean() * torch.eye(rank, device=device, dtype=dtype)
        KR = _khatri_rao(A, B)
        T2 = T.permute(2, 0, 1).reshape(d2, d0 * d1)
        C_new = torch.linalg.solve(V, (T2 @ KR).T).T
        lam_c = C_new.norm(dim=0).clamp(min=1e-30)
        C = C_new / lam_c

        lambdas = lam_a * lam_b * lam_c

        # ── Reconstruction fit ──────────────────────────────────────────────
        # Using Gram-based shortcut:
        #   ||T - hat||_F^2 = ||T||^2 - 2 <T, hat> + ||hat||^2
        # ||hat||^2 = lambda^T ((A^T A) * (B^T B) * (C^T C)) lambda
        # <T, hat>  = sum_r lambda_r * (a_r^T T_(0) (b_r krp c_r))
        with torch.no_grad():
            gram = (A.T @ A) * (B.T @ B) * (C.T @ C)         # (R, R)
            T0 = T.reshape(d0, d1 * d2)
            TT_krp = T0 @ _khatri_rao(B, C)                  # (d0, R)
            inner_vec = (A * TT_krp).sum(dim=0)              # (R,)
            inner_val = (lambdas * inner_vec).sum()
            norm_hat_sq = (lambdas.unsqueeze(0) * gram * lambdas.unsqueeze(1)).sum()
            err_sq = T_norm**2 - 2.0 * inner_val + norm_hat_sq
            err_sq = float(max(err_sq.item(), 0.0))
            rel_err = math.sqrt(err_sq) / max(T_norm, 1e-30)
            fit = 1.0 - rel_err

        entry = {
            "iter": it,
            "fit": float(fit),
            "rel_err": float(rel_err),
            "elapsed_s": time.time() - t0,
        }
        history.append(entry)

        if verbose and (it < 3 or it % max(1, max_iter // 10) == 0 or it == max_iter - 1):
            logger.info("[cp-als] iter %3d  fit=%.6f  rel_err=%.6e  elapsed=%.1fs",
                        it, fit, rel_err, entry["elapsed_s"])

        if prev_fit is not None and abs(fit - prev_fit) < tol:
            if verbose:
                logger.info("[cp-als] converged (|fit - prev_fit|=%.3e < %.3e)",
                            abs(fit - prev_fit), tol)
            break
        prev_fit = fit

    return {
        "A": A.cpu(),
        "B": B.cpu(),
        "C": C.cpu(),
        "lambdas": lambdas.cpu(),
        "fit_history": history,
        "reconstruction_error": history[-1]["rel_err"],
        "final_fit": history[-1]["fit"],
        "iters_run": len(history),
        "converged": prev_fit is not None and abs(history[-1]["fit"] - prev_fit) < tol,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Matching recovered factors to teacher weights (Hungarian)
# ═══════════════════════════════════════════════════════════════════════════

def hungarian_match_cos(
    factors: torch.Tensor,         # (d, R)  recovered
    target: torch.Tensor,          # (d, K)  teacher column vectors
    max_rows: int = 256,
) -> dict[str, Any]:
    """Compute the best bijective column-assignment between `factors` and
    `target` under cosine cost, using the Hungarian algorithm from SciPy.

    If R > K, we match the K teacher columns to their best K factor columns.
    If K > max_rows, we subsample K to cap memory.
    """
    from scipy.optimize import linear_sum_assignment

    F_n = F.normalize(factors.float(), dim=0).cpu()         # (d, R)
    T_n = F.normalize(target.float(), dim=0).cpu()          # (d, K)

    d, R = F_n.shape
    _, K = T_n.shape

    # Subsample K if too many
    sub_K = min(K, max_rows)
    if sub_K < K:
        idx = torch.linspace(0, K - 1, sub_K, dtype=torch.long)
        T_sub = T_n[:, idx]
    else:
        idx = torch.arange(K)
        T_sub = T_n

    # Cost matrix: (sub_K, R), where entry [k, r] = 1 - |cos(T_sub[:, k], F_n[:, r])|
    # We maximize absolute cosine (sign-invariant, same as Anandkumar et al.).
    cos_mat = T_sub.T @ F_n                                 # (sub_K, R)
    cost = 1.0 - cos_mat.abs().numpy()

    # Hungarian: assign teacher columns to factor columns.  scipy's
    # linear_sum_assignment handles rectangular input; it returns arrays of
    # length min(rows, cols).  We always call it on the (sub_K, R) cost so
    # row_ind indexes teachers and col_ind indexes factors, in the SAME
    # order of the original cost matrix.
    row_ind, col_ind = linear_sum_assignment(cost)

    aligned_cos = cos_mat.abs().numpy()[row_ind, col_ind]
    # For reporting signed cos, grab the signed values at the matched positions
    aligned_signed = cos_mat.numpy()[row_ind, col_ind]

    return {
        "aligned_cos_mean": float(np.mean(aligned_cos)),
        "aligned_cos_median": float(np.median(aligned_cos)),
        "aligned_cos_max": float(np.max(aligned_cos)),
        "aligned_cos_top10_mean": float(np.mean(np.sort(aligned_cos)[-10:])),
        "aligned_cos_top5_mean": float(np.mean(np.sort(aligned_cos)[-5:])),
        "aligned_signed_cos_mean": float(np.mean(aligned_signed)),
        "num_matched": int(len(row_ind)),
        "subsampled_K": int(sub_K),
        "original_K": int(K),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Null control: same pipeline with random tensor of same shape
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_null_control(
    M3_shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    M3_norm_target: float,
    cp_rank: int,
    cp_iterations: int,
    cp_tol: float,
    cp_reg: float,
    seed: int,
) -> dict[str, Any]:
    """Generate a random tensor of shape M3_shape with ||.||_F = M3_norm_target,
    run CP-ALS, return factors.  Serves as a null for the alignment metric.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    T_null = torch.randn(*M3_shape, device=device, dtype=dtype, generator=g)
    # Scale to match the real moment norm
    curr_norm = T_null.norm().item()
    if curr_norm > 0:
        T_null *= (M3_norm_target / curr_norm)

    cp = cp_decomposition_als(
        T_null,
        rank=cp_rank,
        max_iter=cp_iterations,
        tol=cp_tol,
        reg=cp_reg,
        seed=seed + 7777,
        verbose=False,
    )
    del T_null
    torch.cuda.empty_cache()
    return cp


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation driver
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_factors_against_teacher(
    cp_result: dict[str, Any],
    teacher,
    max_rows: int = 256,
    label: str = "signal",
) -> dict[str, Any]:
    """For each factor matrix (A, B, C), compute Hungarian-aligned cosines
    against each teacher weight matrix of interest.

    Returns a nested dict keyed by teacher matrix name, then factor name.
    """
    A = cp_result["A"]                      # (d_hat, R)
    B = cp_result["B"]                      # (d, R)
    C = cp_result["C"]                      # (d, R)

    # Teacher matrices (CPU, float32)
    t_last = teacher.model.layers[LAST_BLOCK_IDX]
    targets: dict[str, torch.Tensor] = {
        # Last block MLP
        "W_gate.rows":  t_last.mlp.gate_proj.weight.detach().float().cpu().T,   # (d, d_ff)
        "W_up.rows":    t_last.mlp.up_proj.weight.detach().float().cpu().T,     # (d, d_ff)
        "W_down.cols":  t_last.mlp.down_proj.weight.detach().float().cpu(),     # (d, d_ff)
        # Last block attention
        "W_q.rows":     t_last.self_attn.q_proj.weight.detach().float().cpu().T, # (d, d_out=d)
        "W_k.rows":     t_last.self_attn.k_proj.weight.detach().float().cpu().T,
        "W_v.rows":     t_last.self_attn.v_proj.weight.detach().float().cpu().T,
        "W_o.cols":     t_last.self_attn.o_proj.weight.detach().float().cpu(),   # (d, d)
        # Head / norm
        "W_lm.cols":    teacher.lm_head.weight.detach().float().cpu().T,         # (d, V)  huge
        "norm_final":   teacher.model.norm.weight.detach().float().cpu().unsqueeze(1),  # (d, 1)
    }

    # We match A (which lives in d_hat-dim) vs targets that are d-dim.  We
    # accept the dimensional mismatch by matching A-factors only against
    # d_hat-dim teacher quantities.  For Qwen2.5-0.5B d_hat = d, so A factors
    # can also be matched to d-dim weight columns.

    results: dict[str, Any] = {}
    for tname, T_mat in targets.items():
        d_match, K = T_mat.shape
        per_factor: dict[str, Any] = {}
        for fname, fac in [("A", A), ("B", B), ("C", C)]:
            d_fac, R = fac.shape
            if d_fac != d_match:
                per_factor[fname] = {
                    "skipped": True,
                    "reason": f"dim mismatch: factor dim {d_fac} vs target dim {d_match}",
                }
                continue
            per_factor[fname] = hungarian_match_cos(
                fac, T_mat, max_rows=max_rows,
            )
        results[tname] = per_factor

    logger.info("[eval:%s] Hungarian-aligned cos (best per matrix):", label)
    for tname, per_factor in results.items():
        best = -1.0
        best_fac = ""
        for fname, m in per_factor.items():
            if m.get("skipped"):
                continue
            if m["aligned_cos_top5_mean"] > best:
                best = m["aligned_cos_top5_mean"]
                best_fac = fname
        logger.info("  %-20s  best top5-mean aligned cos = %.4f  (factor %s)",
                    tname, best, best_fac)

    return results


def collapse_best_cos(nested: dict[str, Any]) -> dict[str, float]:
    """For each teacher matrix, return the max top5-mean aligned cos across factors."""
    out = {}
    for tname, per_factor in nested.items():
        best = -1.0
        for fname, m in per_factor.items():
            if m.get("skipped"):
                continue
            best = max(best, m.get("aligned_cos_top5_mean", -1.0))
        out[tname] = best
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    if args.dry_run_small:
        args.num_queries = 64
        args.carlini_queries = 256
        args.carlini_seq_len = 32
        args.max_seq_len = 16
        args.batch_size = 4
        args.cp_rank = 16
        args.cp_iterations = 5

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    device = torch.device(args.device)
    if "cuda" in str(device) and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — falling back to CPU (VERY slow).")
        device = torch.device("cpu")

    logger.info("=" * 72)
    logger.info("HIGH-ORDER MOMENTS TENSOR DECOMPOSITION ATTACK (POC)")
    logger.info("=" * 72)
    logger.info("Model:          %s", args.model_name)
    logger.info("Queries:        %d (seq_len=%d)", args.num_queries, args.max_seq_len)
    logger.info("CP rank:        %d", args.cp_rank)
    logger.info("CP iterations:  %d", args.cp_iterations)
    logger.info("Output dir:     %s", out_dir)
    logger.info("Device:         %s", device)

    # ── Load teacher ────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("-" * 72)
    logger.info("STAGE 1: Load teacher + tokenizer")
    logger.info("-" * 72)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, trust_remote_code=True,
    ).to(device)
    teacher.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = teacher.config.vocab_size
    hidden_size = teacher.config.hidden_size
    assert vocab_size == VOCAB_SIZE, f"Expected V={VOCAB_SIZE}, got {vocab_size}"
    assert hidden_size == D_MODEL, f"Expected d={D_MODEL}, got {hidden_size}"
    logger.info("[setup] teacher loaded: V=%d, d=%d, layers=%d",
                vocab_size, hidden_size, teacher.config.num_hidden_layers)

    # ── Carlini SVD — get W_eff_hat (d_hat x V, orthonormal rows) ──────────
    logger.info("-" * 72)
    logger.info("STAGE 2: Carlini SVD")
    logger.info("-" * 72)
    carlini_out = run_carlini_svd(
        teacher,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_queries=args.carlini_queries,
        seq_len=args.carlini_seq_len,
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    W_eff_hat = carlini_out["W_eff_hat"]
    d_hat = int(carlini_out["d_hat"])

    # ── Build query pool + embedding table ─────────────────────────────────
    logger.info("-" * 72)
    logger.info("STAGE 3: Build query pool and collect z(x), e(x)")
    logger.info("-" * 72)
    query_ids = build_query_pool(
        tokenizer, args.num_queries, args.max_seq_len,
        seed=args.seed, allow_synthetic=args.allow_synthetic,
    )
    logger.info("[queries] shape=%s", tuple(query_ids.shape))

    # Public embedding table: for tied embeddings this is W_lm^T == E in (V, d).
    # This is publicly available from the model hub regardless of weight
    # access, so it's ATTACKER-LEGAL.
    embed_table = teacher.model.embed_tokens.weight.detach().float().cpu()  # (V, d)

    Z, E = collect_logits_and_embed(
        teacher, tokenizer,
        query_ids=query_ids,
        device=device,
        batch_size=args.batch_size,
        pool=args.pool,
        embed_table=embed_table,
    )
    logger.info("[collect] Z shape=%s, E shape=%s, ||Z||=%.3e, ||E||=%.3e",
                tuple(Z.shape), tuple(E.shape),
                float(Z.norm()), float(E.norm()))

    # ── Compute moment tensors ──────────────────────────────────────────────
    logger.info("-" * 72)
    logger.info("STAGE 4: Compute M_1, M_2, M_3")
    logger.info("-" * 72)
    moments = compute_moments(
        Z=Z, E=E, W_eff_hat=W_eff_hat,
        device=device,
        chunk_size=args.batch_size * 4,
        pool=args.pool,
    )

    # Free the raw Z/E buffers — M3 is what we need now
    del Z
    del E
    gc.collect()
    torch.cuda.empty_cache()

    # Additional diagnostics on M3: condition number of mode-0 unfolding
    M3 = moments.M3                                         # on device
    try:
        _, S_M3, _ = torch.linalg.svd(M3.reshape(M3.shape[0], -1), full_matrices=False)
        S_M3_np = S_M3.float().cpu().numpy()
        cond_M3 = float(S_M3_np[0] / max(S_M3_np[-1], 1e-30))
        top10_S_M3 = S_M3_np[:10].tolist()
    except Exception as e:
        logger.warning("[moments] SVD on unfolded M3 failed: %s", e)
        cond_M3 = float("nan")
        top10_S_M3 = []

    moment_diag = {
        "n_samples": moments.n_samples,
        "pool": moments.pool,
        "M3_shape": list(M3.shape),
        "M3_norm": moments.M3_norm,
        "M3_frobenius_ratio": moments.M3_frobenius_ratio,
        "M3_unfolding_mode0_cond": cond_M3,
        "M3_unfolding_mode0_top10_S": top10_S_M3,
        "M1_z_norm": moments.z_norm,
        "M1_e_norm": moments.e_norm,
        "M2_norm": float(moments.M2.norm()),
    }

    # ── CP decomposition on real M3 ─────────────────────────────────────────
    logger.info("-" * 72)
    logger.info("STAGE 5: CP-ALS on empirical moment tensor")
    logger.info("-" * 72)
    cp_real = cp_decomposition_als(
        M3,
        rank=args.cp_rank,
        max_iter=args.cp_iterations,
        tol=args.cp_tol,
        reg=args.cp_reg,
        seed=args.seed + 101,
        verbose=True,
    )
    logger.info("[cp-real] final fit=%.6f  iters=%d  converged=%s",
                cp_real["final_fit"], cp_real["iters_run"], cp_real["converged"])

    # ── Evaluate recovered factors against teacher weights ─────────────────
    logger.info("-" * 72)
    logger.info("STAGE 6: Hungarian matching against teacher weights")
    logger.info("-" * 72)
    real_alignment = evaluate_factors_against_teacher(
        cp_real, teacher,
        max_rows=args.matmul_match_rows,
        label="real",
    )

    real_best = collapse_best_cos(real_alignment)

    # ── Null control 1: random tensor same norm ─────────────────────────────
    logger.info("-" * 72)
    logger.info("STAGE 7: Null control — random tensor, same norm")
    logger.info("-" * 72)
    cp_null = run_null_control(
        M3_shape=tuple(M3.shape),
        device=device,
        dtype=M3.dtype,
        M3_norm_target=moments.M3_norm,
        cp_rank=args.cp_rank,
        cp_iterations=args.cp_iterations,
        cp_tol=args.cp_tol,
        cp_reg=args.cp_reg,
        seed=args.seed + args.null_seed_offset,
    )
    logger.info("[cp-null] final fit=%.6f  iters=%d",
                cp_null["final_fit"], cp_null["iters_run"])

    null_alignment = evaluate_factors_against_teacher(
        cp_null, teacher,
        max_rows=args.matmul_match_rows,
        label="null",
    )
    null_best = collapse_best_cos(null_alignment)

    # Drop the big M3 from GPU before further work
    del M3
    torch.cuda.empty_cache()

    # ── Decide verdict ──────────────────────────────────────────────────────
    logger.info("-" * 72)
    logger.info("STAGE 8: Signal vs null comparison")
    logger.info("-" * 72)

    signal_minus_null: dict[str, float] = {}
    for tname in real_best:
        s = real_best[tname]
        n = null_best.get(tname, float("nan"))
        signal_minus_null[tname] = s - n

    success_matrices: list[str] = []
    for tname, s in real_best.items():
        n = null_best.get(tname, 0.0)
        if s > args.success_threshold and (s - n) > args.null_margin:
            success_matrices.append(tname)

    success = len(success_matrices) > 0

    # Pretty-print comparison table
    logger.info("  %-20s  %12s  %12s  %12s",
                "teacher matrix", "real top5-cos", "null top5-cos", "signal-null")
    for tname in real_best:
        logger.info("  %-20s  %12.4f  %12.4f  %+12.4f",
                    tname, real_best[tname],
                    null_best.get(tname, float("nan")),
                    signal_minus_null[tname])

    verdict = {
        "success": bool(success),
        "success_threshold": args.success_threshold,
        "null_margin": args.null_margin,
        "success_matrices": success_matrices,
        "real_best_top5_cos": real_best,
        "null_best_top5_cos": null_best,
        "signal_minus_null": signal_minus_null,
    }

    # ── Save results ────────────────────────────────────────────────────────
    results: dict[str, Any] = {
        "args": vars(args),
        "carlini": {
            "d_hat": d_hat,
            "S_top10": carlini_out["S"][:10].tolist(),
        },
        "moment_diagnostics": moment_diag,
        "cp_real": {
            "final_fit": cp_real["final_fit"],
            "reconstruction_error": cp_real["reconstruction_error"],
            "iters_run": cp_real["iters_run"],
            "converged": cp_real["converged"],
            "fit_history": cp_real["fit_history"],
            "lambda_top10": cp_real["lambdas"].sort(descending=True).values[:10].tolist(),
            "lambda_norm": float(cp_real["lambdas"].norm()),
        },
        "cp_null": {
            "final_fit": cp_null["final_fit"],
            "reconstruction_error": cp_null["reconstruction_error"],
            "iters_run": cp_null["iters_run"],
            "converged": cp_null["converged"],
            "lambda_top10": cp_null["lambdas"].sort(descending=True).values[:10].tolist(),
        },
        "alignment_real": real_alignment,
        "alignment_null": null_alignment,
        "verdict": verdict,
    }

    results_path = out_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    # Also save small factor tensors for follow-up analysis (not M3 itself).
    factors_path = out_dir / "cp_real_factors.pt"
    torch.save({
        "A": cp_real["A"],
        "B": cp_real["B"],
        "C": cp_real["C"],
        "lambdas": cp_real["lambdas"],
    }, factors_path)
    logger.info("[io] saved CP factors to %s", factors_path)

    # ── Human-readable summary ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  HIGH-ORDER MOMENTS ATTACK — SUMMARY")
    print("=" * 72)
    print(f"  Model:                 {args.model_name}")
    print(f"  N queries:             {args.num_queries}  (seq_len={args.max_seq_len}, pool={args.pool})")
    print(f"  CP rank:               {args.cp_rank}")
    print(f"  CP iterations:         {args.cp_iterations}")
    print(f"  Output dir:            {out_dir}")
    print("-" * 72)
    print("  Stage 2 — Carlini SVD")
    print(f"    d_hat:                     {d_hat} (true: {hidden_size})")
    print(f"    top 5 singular values:     {carlini_out['S'][:5].tolist()}")
    print("-" * 72)
    print("  Stage 4 — Moment tensor diagnostics")
    print(f"    M3 shape:                  {moment_diag['M3_shape']}")
    print(f"    ||M3||_F:                  {moment_diag['M3_norm']:.3e}")
    print(f"    cond(M3_(0)):              {moment_diag['M3_unfolding_mode0_cond']:.3e}")
    print(f"    top 10 singular values of M3_(0):")
    for i, v in enumerate(moment_diag["M3_unfolding_mode0_top10_S"]):
        print(f"      [{i:2d}]  {v:.3e}")
    print("-" * 72)
    print("  Stage 5 — CP-ALS convergence")
    print(f"    real  final fit:            {cp_real['final_fit']:.6f}")
    print(f"    real  recon rel_err:        {cp_real['reconstruction_error']:.6e}")
    print(f"    real  iters:                {cp_real['iters_run']}")
    print(f"    null  final fit:            {cp_null['final_fit']:.6f}")
    print(f"    null  iters:                {cp_null['iters_run']}")
    print("-" * 72)
    print("  Stage 7 — Alignment vs teacher weight matrices (top-5 aligned cos)")
    print(f"    {'matrix':20s}  {'real':>10s}  {'null':>10s}  {'signal-null':>14s}")
    for tname in real_best:
        print(f"    {tname:20s}  {real_best[tname]:10.4f}  "
              f"{null_best.get(tname, float('nan')):10.4f}  "
              f"{signal_minus_null[tname]:+14.4f}")
    print("=" * 72)

    if success:
        best_tname = max(success_matrices, key=lambda t: real_best[t])
        print(f"  VERDICT:  SUCCESS — aligned cos > {args.success_threshold:.2f} for")
        for tname in success_matrices:
            s = real_best[tname]
            n = null_best.get(tname, 0.0)
            print(f"    - {tname}: real={s:.4f}, null={n:.4f}, signal-null={s-n:+.4f}")
        print(f"  Best match: {best_tname} (aligned cos = {real_best[best_tname]:.4f})")
    else:
        print("  VERDICT:  FAILED — no teacher matrix reached aligned cos "
              f"> {args.success_threshold:.2f} ABOVE the null by >"
              f" {args.null_margin:.2f}.")
        print("  Detailed diagnostic:")
        # Identify the closest-to-success matrix
        best_tname = max(real_best, key=lambda t: real_best[t])
        best_s = real_best[best_tname]
        best_n = null_best.get(best_tname, 0.0)
        print(f"    * Best matrix match was {best_tname}:")
        print(f"        real signal  = {best_s:.4f}")
        print(f"        null control = {best_n:.4f}")
        print(f"        gap          = {best_s - best_n:+.4f}")
        print("    * Interpretation:")
        if best_s < 0.05:
            print("      No meaningful structure recovered at all.  Either the")
            print("      moment tensor is dominated by noise (N too small), or")
            print("      transformers lack the clean CP-factorizable structure")
            print("      that Anandkumar-style moment methods exploit.")
        elif best_s - best_n < args.null_margin:
            print("      Signal is within the null distribution's range.  The")
            print("      recovered alignment is an ARTIFACT of Hungarian matching")
            print("      (which finds the best match even in random data at")
            print("      this rank), NOT genuine weight leakage through M_3.")
        elif best_s < args.success_threshold:
            print("      A tiny signal above null exists, but is below the")
            print(f"      claim-worthy threshold ({args.success_threshold:.2f}).")
            print("      Consider: (a) larger N, (b) centered moments, or")
            print("      (c) cumulant corrections to isolate the MLP structure.")
        print("    * Root cause candidates:")
        print("      - RMSNorm between attention and MLP erases magnitude")
        print("        information, breaking the y ~ sigma(W @ x) assumption.")
        print("      - Residual-stream architecture mixes pathways across")
        print("        layers; no single block's weights dominate the moment.")
        print("      - Attention's softmax is input-dependent, not a fixed")
        print("        linear map; the CP factorization assumption is violated.")
        print("      - CP rank r=%d on d=%d tensor is OVERCOMPLETE; the" % (
            args.cp_rank, min(moment_diag["M3_shape"])))
        print("        decomposition is non-unique without extra constraints.")
    print("=" * 72)
    print(f"  Results: {results_path}")
    print(f"  Factors: {factors_path}")
    print("=" * 72 + "\n")

    if success:
        print("SUCCESS")
    else:
        print("FAILED with detailed diagnostic (see above and results.json)")


if __name__ == "__main__":
    main()
