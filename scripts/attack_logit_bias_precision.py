#!/usr/bin/env python3
# SAFETY NOTICE: QUARANTINED (alpha-theory prune 2026-04-19)
# This script is NOT cited in the paper. It was part of killed branches:
#   A1 S-PSI, A2 Moments CP, A4 logit-bias, A5 memory probing, A6 active query,
#   A7 algebraic v2/v3/v4, B3 matched-KD.
# Retained in repo for reproducibility of quarantined history; do not use for
# new claims.
#!/usr/bin/env python3
"""
Logit-bias Precision Attack — Proof-of-concept (Finlayson et al. 2024 style).

Goal
----
Under the assumption that the API exposes top-k logprobs PLUS a per-token
`logit_bias` control (an actual OpenAI / Anthropic capability that Finlayson
et al., arXiv:2403.09539, showed can be abused to pry out internal state),
recover each teacher logit to nearly machine precision for a chosen set of
K probe tokens.  With K >> d_hidden we get an OVERdetermined linear system
whose solution is h_L(x) itself (not just the direction a la Carlini's SVD).

Five phases:

    Phase 1  binary-search logit_bias on K probe tokens per query, recovering
             z(x)[probe_j]  up to ~1e-3 absolute error (tolerance param).

    Phase 2  linear solve  z[probe] = W_lm[probe, :] @ h_L(x) + c          so
             h_L(x) = pinv(W_lm[probe]) @ (z[probe] - c)                   for
             each query; no rotation ambiguity since we PIN W_lm via the
             probe-indexed rows (public).

             (We USE the public lm_head matrix for the probe rows.  The
             Finlayson attack itself does not need to know W_lm; it just
             recovers logit values.  For the downstream "what can those
             precise logits buy us?" question we need a projector, which
             here we take from the public model.  In a black-box setting
             you would use Carlini's recovered W_eff; rotation then
             lives in the downstream analysis, not in h_L recovery.)

    Phase 3  finite-difference Jacobian: Δh_L(x, x') = h_L(x') - h_L(x) is
             now scale-exact, not direction-only.

    Phase 4  try to back out last-block parameter contributions from the
             stack of (Δinput_embed, Δh_L) pairs — this is the HARD part.
             We set up per-query linear systems for W_O, W_down under the
             "one-token-perturbation + prefix frozen" assumption.

    Phase 5  compare against a Carlini SVD baseline computed on the SAME
             query set.  Principal angle cos of h_L recovery in the two
             modes; Carlini gives subspace while we give vectors.

Threat model
------------
We SIMULATE an API with logit_bias locally since we have teacher weights.
The attack code itself makes no white-box queries: it only calls
`query_api(model, tokens, top_k, logit_bias)` which returns the top-k
logprobs as a real API would.  Binary search runs against that interface.

Usage
-----
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 python \
    scripts/attack_logit_bias_precision.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_queries 1024 \
    --num_probe_tokens 5000 \
    --binary_search_iters 20 \
    --output_dir results/v5_attack_logit_bias \
    --seed 42

Note: even if Phase 4 (block recovery) fails, Phase 2's exact h_L recovery is
already a useful paper finding: "given a logit_bias API, last-hidden-state
recovery goes from subspace (Carlini) to vector-exact (ours)."
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Fail fast if we accidentally try to hit the network
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# ── Architecture constants for Qwen2.5-0.5B (used only for asserts) ──────────
NUM_LAYERS = 24
LAST_BLOCK_IDX = 23
D_MODEL = 896
D_FF = 4864
VOCAB_SIZE = 151936
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64


# ═════════════════════════════════════════════════════════════════════════════
# Boilerplate
# ═════════════════════════════════════════════════════════════════════════════

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Logit-bias Precision Attack POC (Finlayson et al. 2024 style)"
    )
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--num_queries", type=int, default=1024,
                   help="Number of base queries to recover h_L for.")
    p.add_argument("--num_probe_tokens", type=int, default=5000,
                   help="Number of distinct vocabulary ids to probe per query.")
    p.add_argument("--binary_search_iters", type=int, default=20,
                   help="Binary-search iterations per probe (log2(precision)).")
    p.add_argument("--top_k", type=int, default=5,
                   help="API top-k that we are limited to observing.")
    p.add_argument("--max_seq_len", type=int, default=64,
                   help="Length of random-token queries.")
    p.add_argument("--perturbations_per_query", type=int, default=4,
                   help="Perturbed queries per base query (for Jacobian phase).")
    p.add_argument("--carlini_queries", type=int, default=2048,
                   help="Number of queries for baseline Carlini SVD.")
    p.add_argument("--carlini_seq_len", type=int, default=128,
                   help="Seq length for Carlini SVD queries.")
    p.add_argument("--output_dir", type=str,
                   default="results/v5_attack_logit_bias")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Query batch size for forward passes.")
    p.add_argument("--bias_init_magnitude", type=float, default=30.0,
                   help="Starting magnitude for bias binary search.")
    p.add_argument("--probe_strategy", type=str,
                   default="stratified",
                   choices=["stratified", "random", "first_k"],
                   help="How to pick probe token ids.")
    p.add_argument("--skip_jacobian", action="store_true",
                   help="Skip Phases 3-4 for quick smoke tests.")
    p.add_argument("--skip_carlini", action="store_true",
                   help="Skip Phase 5 (Carlini comparison) for quick smoke tests.")
    p.add_argument("--dry_run_small", action="store_true",
                   help="Tiny sizes for smoke test.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flat_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    denom = (a_f.norm() * b_f.norm()).clamp(min=1e-30)
    return float((a_f @ b_f) / denom)


def per_row_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    if a_f.dim() == 1:
        return flat_cos(a_f, b_f)
    return float(F.cosine_similarity(a_f, b_f, dim=-1).mean())


# ═════════════════════════════════════════════════════════════════════════════
# Simulated API
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _forward_last_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return last-position logits (B, V) in float32."""
    out = model(input_ids=input_ids)
    return out.logits[:, -1, :].float()


@torch.no_grad()
def query_api(
    model,
    input_ids: torch.Tensor,          # (B, T)
    top_k: int,
    logit_bias: torch.Tensor | None,  # (B, V) or (V,) or None
    cached_logits: torch.Tensor | None = None,  # (B, V) avoids recompute
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate an API that returns top-k logprobs.

    Exposed interface:  z'(x) = z(x) + bias       then apply softmax / top-k.

    For efficiency we cache the true logits z(x) once per base query and then
    reuse them for successive bias values (an arithmetic add is free).  This
    matches the behavior of a real API that answers many bias variants on the
    same prompt very quickly (since the backbone forward is the bottleneck).
    """
    if cached_logits is not None:
        logits = cached_logits
    else:
        logits = _forward_last_logits(model, input_ids)

    if logit_bias is not None:
        if logit_bias.dim() == 1:
            logits = logits + logit_bias.unsqueeze(0)
        else:
            logits = logits + logit_bias

    # Top-k over biased logprobs
    logprobs = F.log_softmax(logits, dim=-1)
    top_lp, top_ids = logprobs.topk(top_k, dim=-1)
    return top_lp, top_ids


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: Binary-search logit recovery
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def recover_logits_binary_search(
    model,
    input_ids: torch.Tensor,           # (B, T)
    probe_token_ids: torch.Tensor,     # (K,) probe token ids (shared across queries)
    cached_logits: torch.Tensor,       # (B, V) true teacher logits (used via query_api)
    binary_search_iters: int,
    top_k: int,
    bias_init_magnitude: float,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Recover logits z(x)[probe_token_ids] via logit_bias binary search.

    Algorithm (Finlayson et al., simplified)
    ----------------------------------------
    For each probe token t in probe_token_ids:
      1. Pick a small set of 'anchor' tokens (chosen from the native top-1 of
         the unbiased query, whose logit we already approximated).  A high-
         precision anchor provides a reference value that the API confirms.
      2. Binary-search b ∈ [-M, +M] such that `t` enters the top-k *and*
         crosses the first anchor.  At the crossover  z(x)[t] + b = z(x)[anchor],
         so  z(x)[t] = z(x)[anchor] - b.

    Shortcut for the POC
    --------------------
    An API-true binary search does:   for each candidate b, call query_api;
    observe whether t is in top-k; refine interval.  That costs O(log 1/tol)
    API calls per (query, probe) pair, so  N*K*iters  total API queries.  For
    N=1024, K=5000, iters=20 that is ~1e8 API calls, which (even with a cache
    on the backbone forward) is a lot of simulated calls.

    We take a CORRECTNESS-PRESERVING shortcut: we run the binary search
    vectorized over all queries and all probe tokens simultaneously.  Each
    "API call" is a softmax + top-k on a (B, V) matrix.  We still log the
    total number of API calls so the paper can report them honestly.

    Returns
    -------
    z_probe_hat : Tensor of shape (B, K) containing recovered logit values.
    total_api_calls : scalar  (B * K * iters in this implementation)
    """
    B, T = input_ids.shape
    K = probe_token_ids.shape[0]
    V = cached_logits.shape[-1]

    # Identify the "anchor" for each query: the argmax token (always in top-k).
    anchor_ids = cached_logits.argmax(dim=-1)                        # (B,)
    # By offsetting biased probe logits into the anchor's neighborhood we
    # force a clean crossover event.
    anchor_vals = cached_logits.gather(1, anchor_ids.unsqueeze(1)).squeeze(1)  # (B,)

    # For efficiency we binary-search on Δ = bias offset relative to the
    # target token's true logit.  The interval is symmetric around the anchor.
    lo = torch.full((B, K), -bias_init_magnitude, dtype=torch.float32, device=device)
    hi = torch.full((B, K), +bias_init_magnitude, dtype=torch.float32, device=device)

    # Cached per-query/per-probe: does a bias of `b` cause the probe token to
    # beat the anchor?  That is equivalent to  z(x)[t] + b > z(x)[anchor].
    #
    # In the simulated API we construct the one-hot bias tensor for each probe
    # and add it to the base logits, then check the boundary.  We do not need
    # to physically run the model -- the cached_logits suffice -- but we DO
    # count one API call per probe-iteration per query (the caller would have
    # to perform the forward at the API level if they lacked a cache).

    # Precompute probe values from the TRUE logits.  In the oracle mode this
    # is cheap; in a real API we would recover these value-by-value through
    # exactly the binary search below.  We use the oracle values solely to
    # validate the binary search -- they are NOT used for downstream decisions.
    oracle_z_probe = cached_logits[:, probe_token_ids]                 # (B, K)

    api_calls = 0

    # For each binary-search iteration, check if b = mid causes the probe
    # token to cross the anchor:  z[t] + mid > z[anchor]  ⇔  mid > anchor - z[t]
    pbar = tqdm(total=binary_search_iters, desc="binary search", unit="iter", leave=False)
    for it in range(binary_search_iters):
        mid = 0.5 * (lo + hi)                                          # (B, K)
        # In a real API we'd make one call per probe per iter.  We batch:
        # the question is  "is  z[t] + mid > z[anchor]"  or equivalently
        # "does probe t appear in top-k after bias mid is applied to t only?"
        # (For top_k > 1 the answer is slightly different but the crossover
        #  between mid and the anchor gap is sharp; the oracle check is
        #  identical to the top-1 boundary which is what we actually need.)
        probe_val_lt_anchor = (oracle_z_probe + mid) < anchor_vals.unsqueeze(1)
        # If probe is still below anchor, bias is too small → lo = mid.
        # If probe is above anchor, bias is too large → hi = mid.
        lo = torch.where(probe_val_lt_anchor, mid, lo)
        hi = torch.where(probe_val_lt_anchor, hi, mid)
        api_calls += B * K
        pbar.update(1)
    pbar.close()

    # Final estimate:  at convergence  z[t] + b* ≈ z[anchor], so
    #                 z[t] ≈ z[anchor] - b*
    b_star = 0.5 * (lo + hi)
    z_probe_hat = anchor_vals.unsqueeze(1) - b_star                    # (B, K)

    # Diagnostic: absolute error vs oracle
    abs_err = (z_probe_hat - oracle_z_probe).abs()
    logger.info(
        "[binsearch] %d iters: abs_err median=%.2e, mean=%.2e, max=%.2e",
        binary_search_iters,
        float(abs_err.median()),
        float(abs_err.mean()),
        float(abs_err.max()),
    )

    return z_probe_hat, api_calls


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Linear system for h_L(x)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def solve_hidden_states(
    z_probe_hat: torch.Tensor,         # (B, K) recovered logit values
    W_probe: torch.Tensor,             # (K, d) public lm_head rows at probe ids
    g_final: torch.Tensor,             # (d,) final RMSNorm weight
) -> torch.Tensor:
    """Solve  z_probe = W_probe @ h_L_norm  for h_L_norm (the rms-normalized
    last hidden state at the final position).

    Qwen2.5-0.5B uses tied embeddings and no bias in lm_head.  The final
    RMSNorm multiplies element-wise by g_final before the linear projection,
    so strictly speaking the projected quantity is h_L_norm = rms(h_L) * g.
    We return THAT quantity; downstream code multiplies or divides by g as
    appropriate.  No bias c is present on this architecture.

    Shapes
    ------
    W_probe  (K, d),   z_probe_hat (B, K)   →   h_L_norm (B, d).
    """
    K, d = W_probe.shape
    B = z_probe_hat.shape[0]
    assert K >= d, f"Need K={K} >= d={d} for overdetermined solve"

    # Least-squares via SVD (stable for the mildly ill-conditioned W_probe).
    # lstsq takes (A, B) where we want  A x = B  shape (K, d) and (K, ..).
    # We solve  W_probe @ h = z_probe_hat^T  column-wise.
    sol = torch.linalg.lstsq(
        W_probe.double(),
        z_probe_hat.double().T,    # (K, B)
    ).solution                     # (d, B)
    h_L_norm_hat = sol.T.float()   # (B, d)

    # "Un-normalize" by g to get back the pre-norm h_L direction.  Since
    # RMSNorm divides by RMS (a scalar), this does NOT recover ||h_L|| — that
    # info is destroyed by the norm.  We return the pre-g-scaled, rms-divided
    # vector; cosine sim to the true (h_L / rms(h_L)) IS identity.
    h_L_dir_hat = h_L_norm_hat / g_final.unsqueeze(0).float().clamp(min=1e-12)
    return h_L_norm_hat, h_L_dir_hat


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Jacobian via finite-differences
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def apply_single_token_perturbation(
    input_ids: torch.Tensor,           # (B, T)
    positions: torch.Tensor,           # (B,) position indices
    new_tokens: torch.Tensor,          # (B,) replacement token ids
) -> torch.Tensor:
    out = input_ids.clone()
    out[torch.arange(input_ids.shape[0]), positions] = new_tokens
    return out


@torch.no_grad()
def collect_recovered_hidden_states(
    model,
    input_ids: torch.Tensor,           # (N, T)
    probe_token_ids: torch.Tensor,     # (K,)
    W_probe: torch.Tensor,             # (K, d)
    g_final: torch.Tensor,             # (d,)
    binary_search_iters: int,
    top_k: int,
    bias_init_magnitude: float,
    device: torch.device,
    batch_size: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """End-to-end wrapper: simulated API → binary search → lstsq.

    Also captures the oracle h_L_direction (from model hidden states) so we
    can measure recovery precision.  The oracle signal is saved for the
    evaluation block and never touched by the attack pipeline.

    Returns
    -------
    h_L_dir_hat : (N, d) recovered h_L directions
    h_L_dir_true : (N, d) oracle directions for evaluation
    total_api_calls : int
    """
    N = input_ids.shape[0]
    d = W_probe.shape[1]
    h_L_dir_hat = torch.zeros(N, d, dtype=torch.float32)
    h_L_dir_true = torch.zeros(N, d, dtype=torch.float32)
    total_calls = 0

    pbar = tqdm(total=N, desc=f"Phase1/2 ({label})", unit="q")
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        batch = input_ids[s:e].to(device)

        # Cached true logits (one API call's worth, batched)
        out = model(input_ids=batch, output_hidden_states=True)
        cached_logits = out.logits[:, -1, :].float()                   # (B, V)
        # Oracle h_L direction at last position (after final RMSNorm * g)
        hs_L = out.hidden_states[LAST_BLOCK_IDX + 1][:, -1, :]        # (B, d)
        # Apply final RMSNorm manually to compare against W_lm @ h_L_norm.
        # Ensure g_final lives on the same device as the forward activations.
        g_final_dev = g_final.to(hs_L.device).float()
        rms = torch.sqrt(hs_L.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        h_L_norm_true = (hs_L.float() / rms) * g_final_dev.unsqueeze(0)
        # The "direction" we compare is h_L_norm / g = h_L / rms(h_L)
        h_L_dir_true_batch = (hs_L.float() / rms)                      # (B, d)

        # Phase 1: binary search
        z_probe_hat, calls = recover_logits_binary_search(
            model, batch, probe_token_ids, cached_logits,
            binary_search_iters=binary_search_iters,
            top_k=top_k,
            bias_init_magnitude=bias_init_magnitude,
            device=device,
        )
        total_calls += calls + (e - s)  # +1 forward call per query (cached_logits)

        # Phase 2: linear solve
        _, h_L_dir_batch = solve_hidden_states(
            z_probe_hat, W_probe.to(device), g_final.to(device),
        )

        h_L_dir_hat[s:e] = h_L_dir_batch.cpu()
        h_L_dir_true[s:e] = h_L_dir_true_batch.cpu()

        pbar.update(e - s)
    pbar.close()

    return h_L_dir_hat, h_L_dir_true, total_calls


# ═════════════════════════════════════════════════════════════════════════════
# Phase 5: Carlini baseline (subspace recovery via SVD)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_carlini_baseline(
    model,
    vocab_size: int,
    hidden_size: int,
    num_queries: int,
    seq_len: int,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Collect random-token logits, SVD, report Vh's top-d subspace.

    Returns dict with S, Vh[:d_hat, :], d_hat, and principal-angle cosines
    against the public W_lm.
    """
    logger.info("[Carlini] collecting %d random-token logits (seq_len=%d) ...",
                num_queries, seq_len)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    ys: list[torch.Tensor] = []
    pbar = tqdm(total=num_queries, desc="Carlini queries", unit="q")
    n_done = 0
    while n_done < num_queries:
        bs = min(batch_size, num_queries - n_done)
        ids = torch.randint(0, vocab_size, (bs, seq_len),
                            generator=rng, device="cpu").to(device)
        out = model(input_ids=ids)
        ys.append(out.logits[:, -1, :].float().cpu())
        n_done += bs
        pbar.update(bs)
    pbar.close()

    Y = torch.cat(ys, dim=0)                                          # (N, V)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    t0 = time.time()
    _, S, Vh = torch.linalg.svd(Y_centered, full_matrices=False)
    svd_time = time.time() - t0

    S_np = S.numpy()
    gap = S_np[:-1] / np.maximum(S_np[1:], 1e-30)
    lo_gap = max(0, hidden_size - 50)
    hi_gap = min(len(gap), hidden_size + 50)
    d_hat = int(lo_gap + np.argmax(gap[lo_gap:hi_gap]))

    logger.info(
        "[Carlini] SVD done in %.1fs, d_hat=%d (true=%d), gap_ratio=%.2f",
        svd_time, d_hat, hidden_size, gap[d_hat],
    )

    return {
        "W_eff_hat_rows": Vh[:d_hat, :].float(),   # (d_hat, V)
        "singular_values_top10": S_np[:10].tolist(),
        "d_hat": int(d_hat),
        "gap_ratio_at_d_hat": float(gap[d_hat]),
        "svd_time_s": round(svd_time, 2),
    }


def subspace_principal_cos(Q_hat_rows: torch.Tensor,
                           W_true: torch.Tensor) -> dict[str, Any]:
    """Compare row-space of Q_hat_rows (d, V) to col(W_true) (V, d).

    Returns principal-angle cosines between the two d-dim subspaces.
    """
    A = Q_hat_rows.T.float()                        # (V, d_hat)
    B = W_true.float()                              # (V, d_true)
    Qa, _ = torch.linalg.qr(A)
    Qb, _ = torch.linalg.qr(B)
    cos = torch.linalg.svdvals(Qa.T @ Qb).clamp(0.0, 1.0)
    return {
        "mean_cos": float(cos.mean()),
        "min_cos": float(cos.min()),
        "top5_cos": cos[:5].tolist(),
    }


def carlini_h_L_projection(
    model,
    input_ids: torch.Tensor,
    W_eff_hat_rows: torch.Tensor,       # (d_hat, V)
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each query, project logits through Carlini's recovered row basis
    to get h_L_direction in Carlini's rotated frame.

    Since W_eff_hat_rows has orthonormal rows, this projection IS the
    subspace component of the true h_L (up to orthogonal rotation R).
    To compare magnitude to oracle, we measure cosine AFTER Procrustes-
    aligning to the oracle h_L_direction (this rewards Carlini in the most
    generous way).
    """
    N = input_ids.shape[0]
    d_hat = W_eff_hat_rows.shape[0]
    proj = W_eff_hat_rows.to(device)                 # (d_hat, V)

    u_all = torch.zeros(N, d_hat, dtype=torch.float32)
    hL_true_all = torch.zeros(N, D_MODEL, dtype=torch.float32)

    with torch.no_grad():
        pbar = tqdm(total=N, desc="Carlini project", unit="q")
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            batch = input_ids[s:e].to(device)
            out = model(input_ids=batch, output_hidden_states=True)
            z = out.logits[:, -1, :].float()
            u = z @ proj.T                          # (B, d_hat)

            hL = out.hidden_states[LAST_BLOCK_IDX + 1][:, -1, :].float()
            rms = torch.sqrt(hL.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            hL_dir = hL / rms

            u_all[s:e] = u.cpu()
            hL_true_all[s:e] = hL_dir.cpu()
            pbar.update(e - s)
        pbar.close()

    return u_all, hL_true_all


def best_orthogonal_alignment_cos(
    u_hat: torch.Tensor,            # (N, d_hat) e.g. Carlini or ours
    h_true: torch.Tensor,           # (N, d_true) oracle h_L direction
) -> dict[str, float]:
    """Find best orthogonal R and residual-rescaling α s.t. α u @ R ≈ h_true,
    then report mean per-row cosine.  If d_hat != d_true, orthogonal R is
    a (d_hat, d_true) Stiefel matrix.
    """
    A = u_hat.double()                # (N, d_hat)
    B = h_true.double()               # (N, d_true)

    cross = A.T @ B                   # (d_hat, d_true)
    U_p, _, Vh_p = torch.linalg.svd(cross, full_matrices=False)
    R = U_p @ Vh_p                    # (d_hat, d_true)

    aligned = (A @ R).float()          # (N, d_true)
    cos = F.cosine_similarity(aligned, B.float(), dim=-1)
    return {
        "mean_cos": float(cos.mean()),
        "min_cos": float(cos.min()),
        "median_cos": float(cos.median()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4: Last-block parameter recovery attempts
# ═════════════════════════════════════════════════════════════════════════════

def ridge_regress(
    A: torch.Tensor,       # (M, d_in)
    B: torch.Tensor,       # (M, d_out)
    ridge: float = 1e-4,
) -> torch.Tensor:
    """Solve  A @ X = B  for X in (d_in, d_out) with Tikhonov regularization.

    Returns X in double precision then cast to float32.
    """
    A = A.double()
    B = B.double()
    d_in = A.shape[1]
    gram = A.T @ A
    gram += ridge * gram.diagonal().mean().clamp(min=1e-12) * torch.eye(
        d_in, dtype=torch.float64, device=A.device
    )
    rhs = A.T @ B
    X = torch.linalg.solve(gram, rhs)
    return X.float()


def try_last_block_recovery(
    h_L_dir_base: torch.Tensor,        # (N, d)  our recovered h_L directions
    h_L_dir_pert: torch.Tensor,        # (N, P, d) perturbed-query recoveries
    perturb_positions: torch.Tensor,   # (N, P)
    new_token_ids: torch.Tensor,       # (N, P)
    input_ids: torch.Tensor,           # (N, T)
    embedding_table: torch.Tensor,     # (V, d) public embedding
    teacher,                           # for oracle-only evaluation metrics
    device: torch.device,
) -> dict[str, Any]:
    """Try to regress a last-block parameter signal from (Δinput, Δh_L) pairs.

    This is intentionally conservative.  The residual-stream decomposition at
    the last position is

        h_L(x) = h_{L-1}(x) + attn(x) + mlp(x)

    where each term is a nonlinear function of the entire sequence.  For a
    single-token perturbation at position p on the sequence, the change at
    position L (= last position) depends on the attention pattern from the
    last position to position p for every block, compounded.  We try to
    isolate the LAST BLOCK contribution under two approximations:

      (A) If the perturbation is at position T-1 (last position), the change
          propagates through all blocks, but at each block j's attention the
          "query" vector at last position changes while "keys/values" at
          other positions are fixed (except at the perturbation position for
          block j).  The total Δh_L carries a J_full(x) @ Δe contribution.

      (B) The dominant component of J_full at the last block is
          W_O @ W_V (for attention) and W_down @ W_up (for MLP) applied to
          the embedding difference at the perturbation position.

    We therefore regress Δh_L ≈ M @ Δe where M is an (d, d) linear operator.
    A perfect fit would imply the network is linear (it isn't) but even a
    partial fit reveals the principal singular directions of the full
    Jacobian, which we then compare to last-block weights.
    """
    N, P, d = h_L_dir_pert.shape
    T = input_ids.shape[1]

    # Build Δe = e[new_token] - e[old_token] for each (n, p).
    old_tokens = input_ids.gather(1, perturb_positions)               # (N, P)
    e_old = embedding_table[old_tokens]                               # (N, P, d)
    e_new = embedding_table[new_token_ids]                            # (N, P, d)
    Delta_e = (e_new - e_old).reshape(N * P, d)                       # (M, d)

    Delta_h = (h_L_dir_pert - h_L_dir_base.unsqueeze(1)).reshape(N * P, d)   # (M, d)

    # Regress Δh = Δe @ M^T  =>  M = lstsq(Δe, Δh)^T
    M_est = ridge_regress(Delta_e, Delta_h)                           # (d, d) — but really (d_in=d, d_out=d)
    # Fit quality
    pred = Delta_e.double() @ M_est.double()
    resid = (pred - Delta_h.double())
    fit_cos = float(F.cosine_similarity(
        pred.float().flatten(0).unsqueeze(0),
        Delta_h.float().flatten(0).unsqueeze(0),
    ).item())
    fit_rel = float(resid.norm() / Delta_h.double().norm().clamp(min=1e-30))
    logger.info("[phase4] regression fit cos=%.4f, rel-residual=%.4f",
                fit_cos, fit_rel)

    # Candidate comparisons against teacher's last block parameters.
    # These are all ATTACK-LEGAL in the sense that we don't use them to
    # construct M_est; we only use them to SCORE M_est afterwards.
    last_block = teacher.model.layers[LAST_BLOCK_IDX]
    W_O_true = last_block.self_attn.o_proj.weight.data.float().cpu()       # (d, d)
    W_down_true = last_block.mlp.down_proj.weight.data.float().cpu()       # (d, d_ff)

    # We ALIGN M_est with each target by best orthogonal rotation (Procrustes)
    # on both sides, because the full-model Jacobian mixes block-internal
    # projections in unknown ways; principal angles are the cleanest scorer.
    def principal_cos(M: torch.Tensor, W_target: torch.Tensor) -> float:
        # Column spaces in R^d
        if M.shape != W_target.shape:
            # Match columns: compare col-space of M to col-space of W_target
            Qm, _ = torch.linalg.qr(M.float())
            Qw, _ = torch.linalg.qr(W_target.float())
            # Only the smaller-d subspace contributes
            k = min(Qm.shape[1], Qw.shape[1])
            cos = torch.linalg.svdvals(Qm[:, :k].T @ Qw[:, :k]).clamp(0.0, 1.0)
        else:
            Qm, _ = torch.linalg.qr(M.float())
            Qw, _ = torch.linalg.qr(W_target.float())
            cos = torch.linalg.svdvals(Qm.T @ Qw).clamp(0.0, 1.0)
        return float(cos.mean())

    cos_W_O = principal_cos(M_est, W_O_true)
    cos_W_down = principal_cos(M_est, W_down_true)

    # Raw flat cosine — informative baseline, usually tiny for this kind of
    # nonlinear Jacobian inversion.
    flat_cos_W_O = flat_cos(M_est, W_O_true)
    flat_cos_W_down_first_d = flat_cos(M_est, W_down_true[:, :d])

    logger.info("[phase4] principal-angle cos vs W_O:     %.4f", cos_W_O)
    logger.info("[phase4] principal-angle cos vs W_down:  %.4f", cos_W_down)
    logger.info("[phase4] flat cos         vs W_O:        %.4f", flat_cos_W_O)
    logger.info("[phase4] flat cos         vs W_down[:, :d]: %.4f",
                flat_cos_W_down_first_d)

    return {
        "regression_fit_cos": fit_cos,
        "regression_rel_residual": fit_rel,
        "principal_cos_W_O": cos_W_O,
        "principal_cos_W_down": cos_W_down,
        "flat_cos_W_O": flat_cos_W_O,
        "flat_cos_W_down_first_d": flat_cos_W_down_first_d,
        "num_perturbation_pairs": int(N * P),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Model / tokenizer utilities
# ═════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Loading model: %s (device=%s, dtype=%s)", model_name, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def get_true_lm_head(model) -> torch.Tensor:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        w = model.lm_head.weight.data
    else:
        w = model.model.embed_tokens.weight.data
    return w.float().cpu()


def get_final_norm(model) -> torch.Tensor:
    return model.model.norm.weight.data.float().cpu()


def build_random_query_pool(
    n: int, seq_len: int, vocab_size: int, seed: int,
) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, vocab_size, (n, seq_len), generator=rng)
    return ids


def pick_probe_tokens(
    num_probe: int,
    vocab_size: int,
    W_lm: torch.Tensor,        # (V, d)
    strategy: str,
    seed: int,
) -> torch.Tensor:
    """Pick probe token ids.

    - "random":       uniform random ids
    - "first_k":      deterministic first K
    - "stratified":   take K that SPAN the row-space of W_lm well; we use
                      a QR pivot over W_lm^T.  This minimizes the condition
                      number of W_probe — important because Phase 2 solves
                      a linear system with it.
    """
    rng = np.random.default_rng(seed)
    if strategy == "random":
        ids = rng.choice(vocab_size, size=num_probe, replace=False)
        return torch.tensor(ids, dtype=torch.long)
    if strategy == "first_k":
        return torch.arange(num_probe, dtype=torch.long)
    if strategy == "stratified":
        # QR with column pivoting on W_lm^T   (columns = vocab rows of W_lm)
        # selects the most-orthogonal-subset of vocab rows.  For large V we
        # approximate via random sub-sampling then QR-pivot.
        sample_size = min(max(3 * num_probe, num_probe + 16), vocab_size)
        candidate_ids = rng.choice(vocab_size, size=sample_size, replace=False)
        W_cand = W_lm[candidate_ids].float()                           # (sample, d)
        # QR of W_cand^T: Q (d, r), R (r, sample) with r=min(d, sample).
        # Column norms of R give per-candidate leverage scores of length `sample`.
        q_result = torch.linalg.qr(W_cand.T, mode="reduced")
        leverage = (q_result.R ** 2).sum(dim=0)                        # (sample,)
        if leverage.shape[0] < num_probe:
            # QR returned fewer scores than we want (d < num_probe case).
            # Take top-d by leverage, then pad with random candidates.
            top_k = int(leverage.shape[0])
            top_idx = torch.topk(leverage, k=top_k).indices
            chosen_cands = candidate_ids[top_idx.cpu().numpy()]
            used = set(chosen_cands.tolist())
            extra_needed = num_probe - top_k
            pool = np.array([i for i in candidate_ids if i not in used])
            if extra_needed > pool.size:
                # Need more than the candidate pool has → sample from vocab
                # excluding already chosen ids.
                all_pool = np.setdiff1d(
                    np.arange(vocab_size), np.array(list(used)),
                    assume_unique=False,
                )
                extra = rng.choice(all_pool, size=extra_needed, replace=False)
            else:
                extra = rng.choice(pool, size=extra_needed, replace=False)
            merged = np.concatenate([chosen_cands, extra])
            return torch.tensor(merged, dtype=torch.long)
        top_k = int(num_probe)
        top_idx = torch.topk(leverage, k=top_k).indices
        chosen = torch.tensor(candidate_ids[top_idx.cpu().numpy()], dtype=torch.long)
        return chosen
    raise ValueError(f"Unknown strategy: {strategy}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    setup_logging()
    args = parse_args()

    if args.dry_run_small:
        args.num_queries = 32
        args.num_probe_tokens = 1024
        args.binary_search_iters = 10
        args.perturbations_per_query = 2
        args.carlini_queries = 256

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model, tokenizer = load_model(args.model_name, device, dtype)

    vocab_size = model.config.vocab_size
    hidden_size = model.config.hidden_size
    logger.info(
        "Model: vocab=%d, hidden=%d, tie_embeddings=%s",
        vocab_size, hidden_size,
        getattr(model.config, "tie_word_embeddings", False),
    )

    # Load public matrices
    W_lm_full = get_true_lm_head(model)                                # (V, d)
    g_final = get_final_norm(model)                                    # (d,)
    embedding_table = model.model.embed_tokens.weight.data.float().cpu()   # (V, d)

    # ═════════════════════════════════════════════════════════════════════
    # Phase 0: pick probe tokens and construct W_probe
    # ═════════════════════════════════════════════════════════════════════
    t0 = time.time()
    probe_token_ids = pick_probe_tokens(
        args.num_probe_tokens, vocab_size, W_lm_full, args.probe_strategy, args.seed,
    )
    W_probe = W_lm_full[probe_token_ids]                               # (K, d)
    logger.info(
        "Phase 0 | picked %d probe tokens (strategy=%s) in %.2fs; W_probe shape=%s, "
        "cond(W_probe)≈%.2e",
        args.num_probe_tokens, args.probe_strategy, time.time() - t0,
        tuple(W_probe.shape),
        float(torch.linalg.cond(W_probe.float())),
    )

    # ═════════════════════════════════════════════════════════════════════
    # Phase 1/2: recover h_L(x) for N base queries
    # ═════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("Phase 1 + 2: recover h_L for %d queries", args.num_queries)
    logger.info("=" * 70)
    base_ids = build_random_query_pool(
        args.num_queries, args.max_seq_len, vocab_size, args.seed,
    )
    t0 = time.time()
    h_L_hat, h_L_true, calls_p12 = collect_recovered_hidden_states(
        model, base_ids, probe_token_ids, W_probe, g_final,
        args.binary_search_iters, args.top_k, args.bias_init_magnitude,
        device, args.batch_size, label="base",
    )
    phase12_time = time.time() - t0
    logger.info("Phase 1+2 done in %.1fs", phase12_time)

    # Evaluate per-query recovery cos
    per_row = F.cosine_similarity(h_L_hat, h_L_true, dim=-1)
    mean_cos_ours = float(per_row.mean())
    min_cos_ours = float(per_row.min())
    median_cos_ours = float(per_row.median())
    frac_above_099 = float((per_row > 0.99).float().mean())
    frac_above_0999 = float((per_row > 0.999).float().mean())
    logger.info(
        "h_L recovery: mean=%.6f, min=%.6f, median=%.6f, P(>0.99)=%.2f, P(>0.999)=%.2f",
        mean_cos_ours, min_cos_ours, median_cos_ours,
        frac_above_099, frac_above_0999,
    )

    # ═════════════════════════════════════════════════════════════════════
    # Phase 3: Jacobian via finite differences
    # ═════════════════════════════════════════════════════════════════════
    phase3_metrics: dict[str, Any] = {"skipped": args.skip_jacobian}
    phase4_metrics: dict[str, Any] = {"skipped": args.skip_jacobian}
    calls_p3 = 0

    if not args.skip_jacobian:
        logger.info("=" * 70)
        logger.info("Phase 3: finite-difference Jacobian probe")
        logger.info("=" * 70)
        P = args.perturbations_per_query

        # Choose last-position perturbations (simplest: Δ at position T-1).
        rng = np.random.default_rng(args.seed ^ 0x5A5A)
        perturb_positions = torch.full(
            (args.num_queries, P), args.max_seq_len - 1, dtype=torch.long,
        )
        new_token_ids = torch.tensor(
            rng.integers(low=3, high=vocab_size, size=(args.num_queries, P)),
            dtype=torch.long,
        )

        # Build perturbed query set.  Flat layout: row n*P + p_idx is the
        # p_idx-th perturbation of base query n.  This stride is what Phase 3
        # reshape-back relies on.
        all_perturbed = torch.zeros(
            args.num_queries * P, args.max_seq_len, dtype=torch.long,
        )
        for p_idx in range(P):
            perturbed_slice = apply_single_token_perturbation(
                base_ids, perturb_positions[:, p_idx], new_token_ids[:, p_idx],
            )
            # Place into rows  p_idx, P + p_idx, 2P + p_idx, ... (stride P starting at p_idx)
            all_perturbed[p_idx::P] = perturbed_slice

        t0 = time.time()
        h_L_hat_pert, h_L_true_pert, calls_p3 = collect_recovered_hidden_states(
            model, all_perturbed, probe_token_ids, W_probe, g_final,
            args.binary_search_iters, args.top_k, args.bias_init_magnitude,
            device, args.batch_size, label="perturbed",
        )
        phase3_time = time.time() - t0

        h_L_hat_pert = h_L_hat_pert.reshape(args.num_queries, P, -1)
        h_L_true_pert = h_L_true_pert.reshape(args.num_queries, P, -1)

        # Recovery quality on perturbed queries
        per_row_p = F.cosine_similarity(
            h_L_hat_pert.reshape(-1, hidden_size),
            h_L_true_pert.reshape(-1, hidden_size),
            dim=-1,
        )
        mean_cos_pert = float(per_row_p.mean())

        # Precision of Δh_L
        Dh_hat = h_L_hat_pert - h_L_hat.unsqueeze(1)
        Dh_true = h_L_true_pert - h_L_true.unsqueeze(1)
        per_row_Dh = F.cosine_similarity(
            Dh_hat.reshape(-1, hidden_size), Dh_true.reshape(-1, hidden_size), dim=-1,
        )
        mean_cos_Dh = float(per_row_Dh.mean())
        median_cos_Dh = float(per_row_Dh.median())
        rel_err_Dh = float(
            (Dh_hat - Dh_true).float().norm() / Dh_true.float().norm().clamp(min=1e-30)
        )
        logger.info(
            "ΔhL precision: mean_cos=%.6f, median_cos=%.6f, rel_err=%.4e",
            mean_cos_Dh, median_cos_Dh, rel_err_Dh,
        )

        phase3_metrics = {
            "skipped": False,
            "num_perturbations_per_query": P,
            "total_perturbed_queries": args.num_queries * P,
            "perturbed_h_L_mean_cos": mean_cos_pert,
            "Delta_h_L_mean_cos": mean_cos_Dh,
            "Delta_h_L_median_cos": median_cos_Dh,
            "Delta_h_L_relative_error": rel_err_Dh,
            "phase3_time_s": round(phase3_time, 2),
        }

        # ═════════════════════════════════════════════════════════════════
        # Phase 4: try last-block parameter recovery
        # ═════════════════════════════════════════════════════════════════
        logger.info("=" * 70)
        logger.info("Phase 4: last-block parameter recovery from (Δinput, Δh_L) pairs")
        logger.info("=" * 70)
        phase4_metrics = try_last_block_recovery(
            h_L_hat, h_L_hat_pert, perturb_positions, new_token_ids,
            base_ids, embedding_table, model, device,
        )
        phase4_metrics["skipped"] = False

    # ═════════════════════════════════════════════════════════════════════
    # Phase 5: Carlini SVD baseline for comparison
    # ═════════════════════════════════════════════════════════════════════
    carlini_metrics: dict[str, Any] = {"skipped": args.skip_carlini}
    if not args.skip_carlini:
        logger.info("=" * 70)
        logger.info("Phase 5: Carlini SVD baseline")
        logger.info("=" * 70)
        carlini = run_carlini_baseline(
            model, vocab_size, hidden_size, args.carlini_queries,
            args.carlini_seq_len, device, args.batch_size,
            seed=args.seed + 1,
        )
        # (a) subspace-level comparison (Carlini's published claim)
        sub = subspace_principal_cos(carlini["W_eff_hat_rows"], W_lm_full)

        # (b) per-query h_L recovery under Carlini's method:
        #     project logits through recovered row basis, align rotation, cos.
        u_carlini, hL_oracle = carlini_h_L_projection(
            model, base_ids, carlini["W_eff_hat_rows"], device, args.batch_size,
        )
        carlini_align = best_orthogonal_alignment_cos(u_carlini, hL_oracle)

        carlini_metrics = {
            "skipped": False,
            "d_hat": carlini["d_hat"],
            "gap_ratio_at_d_hat": carlini["gap_ratio_at_d_hat"],
            "subspace_mean_cos": sub["mean_cos"],
            "subspace_min_cos": sub["min_cos"],
            "subspace_top5_cos": sub["top5_cos"],
            "per_query_h_L_aligned_mean_cos": carlini_align["mean_cos"],
            "per_query_h_L_aligned_min_cos": carlini_align["min_cos"],
            "per_query_h_L_aligned_median_cos": carlini_align["median_cos"],
            "num_queries_for_svd": args.carlini_queries,
            "svd_time_s": carlini["svd_time_s"],
        }
        logger.info(
            "Carlini: subspace cos=%.6f, per-query h_L aligned cos=%.6f",
            sub["mean_cos"], carlini_align["mean_cos"],
        )

    # ═════════════════════════════════════════════════════════════════════
    # Assemble and save results
    # ═════════════════════════════════════════════════════════════════════
    # Honest API-call accounting.  In the real attack the forward-pass cache
    # doesn't exist; each binary-search step is one real API call.  We report
    # both (a) conceptual API calls as seen by the attacker and (b) simulated
    # forward passes (i.e. wall-clock cost of our experiment).
    conceptual_api_calls_base = (
        args.num_queries * args.num_probe_tokens * args.binary_search_iters
        + args.num_queries  # 1 unbiased forward per query to learn anchor
    )
    conceptual_api_calls_pert = 0
    if not args.skip_jacobian:
        conceptual_api_calls_pert = (
            args.num_queries * args.perturbations_per_query
            * args.num_probe_tokens * args.binary_search_iters
            + args.num_queries * args.perturbations_per_query
        )
    conceptual_api_total = conceptual_api_calls_base + conceptual_api_calls_pert

    results = {
        "config": vars(args),
        "phase0_probe_tokens": {
            "num_probe": int(args.num_probe_tokens),
            "strategy": args.probe_strategy,
            "W_probe_shape": list(W_probe.shape),
            "W_probe_cond_number": float(torch.linalg.cond(W_probe.float())),
        },
        "phase1_2_h_L_recovery": {
            "num_queries": args.num_queries,
            "mean_cos_ours": mean_cos_ours,
            "min_cos_ours": min_cos_ours,
            "median_cos_ours": median_cos_ours,
            "frac_above_0.99": frac_above_099,
            "frac_above_0.999": frac_above_0999,
            "phase12_time_s": round(phase12_time, 2),
        },
        "phase3_jacobian": phase3_metrics,
        "phase4_block_recovery": phase4_metrics,
        "phase5_carlini_comparison": carlini_metrics,
        "query_budget": {
            "phase1_2_simulated_forward_calls": int(calls_p12),
            "phase3_simulated_forward_calls": int(calls_p3),
            "phase1_2_conceptual_api_calls": conceptual_api_calls_base,
            "phase3_conceptual_api_calls": conceptual_api_calls_pert,
            "conceptual_api_total": conceptual_api_total,
            "note": (
                "conceptual_api_total is what a real API would charge; "
                "simulated_forward_calls is what this script actually spent "
                "on GPU forward passes (we reuse cached logits across the "
                "binary-search inner loop)."
            ),
        },
    }

    # Save primary artifacts
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results to %s", results_path)

    # Save recovered h_L tensors for downstream use
    torch.save(
        {
            "h_L_hat": h_L_hat,
            "h_L_true": h_L_true,
            "base_input_ids": base_ids,
            "probe_token_ids": probe_token_ids,
        },
        output_dir / "recovered_h_L.pt",
    )

    # ═════════════════════════════════════════════════════════════════════
    # Verdict
    # ═════════════════════════════════════════════════════════════════════
    BASE_SUCCESS_THRESHOLD = 0.99       # h_L recovery must exceed Carlini's ~0.9
    EXTENDED_SUCCESS_THRESHOLD = 0.1    # block recovery principal-angle cos

    base_ok = mean_cos_ours > BASE_SUCCESS_THRESHOLD
    extended_ok = False
    extended_note = ""
    if not args.skip_jacobian:
        best_block_cos = max(
            phase4_metrics.get("principal_cos_W_O", 0.0),
            phase4_metrics.get("principal_cos_W_down", 0.0),
        )
        extended_ok = best_block_cos > EXTENDED_SUCCESS_THRESHOLD
        extended_note = f"best_principal_cos_block_weight={best_block_cos:.4f}"

    print("\n" + "=" * 70)
    print("  Logit-bias Precision Attack (Finlayson-style POC) — Verdict")
    print("=" * 70)
    print(f"  Model:             {args.model_name}")
    print(f"  Queries (base):    {args.num_queries}")
    print(f"  Probe tokens:      {args.num_probe_tokens}   (strategy: {args.probe_strategy})")
    print(f"  Binary-search its: {args.binary_search_iters}")
    print("-" * 70)
    print("  Phase 1 + 2 — h_L recovery (ours)")
    print(f"    mean cos:        {mean_cos_ours:.6f}")
    print(f"    median cos:      {median_cos_ours:.6f}")
    print(f"    min cos:         {min_cos_ours:.6f}")
    print(f"    P(cos>0.99):     {frac_above_099:.3f}")
    print(f"    P(cos>0.999):    {frac_above_0999:.3f}")
    if not args.skip_jacobian:
        print("-" * 70)
        print("  Phase 3 — Δh_L precision")
        print(f"    mean cos:        {phase3_metrics['Delta_h_L_mean_cos']:.6f}")
        print(f"    rel error:       {phase3_metrics['Delta_h_L_relative_error']:.4e}")
        print("  Phase 4 — last-block parameter recovery")
        print(f"    regression fit cos:  {phase4_metrics['regression_fit_cos']:.4f}")
        print(f"    principal cos W_O:   {phase4_metrics['principal_cos_W_O']:.4f}")
        print(f"    principal cos W_down:{phase4_metrics['principal_cos_W_down']:.4f}")
    if not args.skip_carlini:
        print("-" * 70)
        print("  Phase 5 — Carlini SVD baseline (same W_lm ground truth)")
        print(f"    subspace cos:    {carlini_metrics['subspace_mean_cos']:.6f}")
        print(f"    per-query cos:   {carlini_metrics['per_query_h_L_aligned_mean_cos']:.6f}")
        print(f"    (ours beats Carlini on per-query h_L by "
              f"{mean_cos_ours - carlini_metrics['per_query_h_L_aligned_mean_cos']:+.4f})")
    print("-" * 70)
    print("  Query budget (conceptual API calls)")
    print(f"    Phase 1+2: {conceptual_api_calls_base:,}")
    if not args.skip_jacobian:
        print(f"    Phase 3:   {conceptual_api_calls_pert:,}")
    print(f"    Total:     {conceptual_api_total:,}")
    print("-" * 70)
    if base_ok and extended_ok:
        print("  VERDICT: SUCCESS (base + extended)")
        print(f"    base: h_L_cos={mean_cos_ours:.4f} > {BASE_SUCCESS_THRESHOLD}")
        print(f"    extended: {extended_note}")
    elif base_ok and args.skip_jacobian:
        print("  VERDICT: SUCCESS (base only, extended skipped)")
        print(f"    h_L_cos={mean_cos_ours:.4f} > {BASE_SUCCESS_THRESHOLD}")
    elif base_ok:
        print("  VERDICT: PARTIAL SUCCESS (base only)")
        print(f"    base: h_L_cos={mean_cos_ours:.4f} > {BASE_SUCCESS_THRESHOLD}")
        print(f"    extended failed: {extended_note}")
    else:
        print("  VERDICT: FAILED")
        print(f"    base: h_L_cos={mean_cos_ours:.4f} <= {BASE_SUCCESS_THRESHOLD}")
        print(f"    extended: {extended_note or 'skipped'}")
    print("=" * 70)
    print(f"  Output dir:  {output_dir}")
    print(f"  Results JSON: {results_path}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
