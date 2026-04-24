#!/usr/bin/env python3
"""
Defense Ablation — test how common LLM API defenses break our algebraic attacks.

Tests Qwen2.5-0.5B under 5 defense families and reports, per (defense, param):
  1. Carlini subspace cos   — can the defense break lm_head subspace extraction?
  2. Oracle W_down cos      — can the defense break internal-layer recovery?
  3. Utility KL (to clean)  — is the model still usable as an API?

Memory-aware design:
  - For Qwen2.5-0.5B, V=151936. A full (N, T, V) fp32 tensor at N=2048, T=128
    is ~152 GB. We therefore only store:
      * last-position logits (N, V)  for Carlini
      * a held-out full-logit batch (N_util, T, V) for KL utility
      * block-(idx) (h_in, h_mid, h_out) for the oracle W_down attack
  - For Step-0 (h_out recovery via pinv(W_lm) @ defended_logits), we apply the
    defense batch-by-batch to the full logits during a second forward pass,
    and project to h_out_est in a streaming manner — never materializing (N,T,V).

For each defense we measure three attack surfaces:
  (a) Carlini subspace from defended last-position logits
  (b) h_out recovery from defended full logits (Step-0 of the oracle attack)
  (c) W_down OLS using h_out_est (from b) as the target — this is what an
      attacker who has oracle (W_gate, W_up, g_mlp) but only defended logits
      would actually achieve.

Defense cost is KL(P_def || P_clean) on a held-out utility batch.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/defense_ablation.py \\
        --model_name Qwen/Qwen2.5-0.5B \\
        --num_queries 2048 --max_seq_len 128 \\
        --output_dir results/v5_defense_ablation \\
        --seed 42
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── logging / seed ──────────────────────────────────────────────────────

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


# ── defense functions ──────────────────────────────────────────────────
# Each returns a callable: defended_logits = fn(clean_logits).
# They operate on any-shape tensors whose LAST dim is vocab.


def defense_topk(k: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return top-k logits; mask others to -inf.

    For algebraic attacks that read raw logit values, we keep the preserved
    entries at their true values. Matches real APIs that surface top-k only.
    """
    def fn(logits: torch.Tensor) -> torch.Tensor:
        V = logits.shape[-1]
        if k >= V:
            return logits
        vals, idx = logits.topk(k, dim=-1)
        out = torch.full_like(logits, float("-inf"))
        out.scatter_(-1, idx, vals)
        return out
    return fn


def defense_quantize(bits: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Quantize logits to q bits using per-row min/max dequantization."""
    levels = 1 << bits

    def fn(logits: torch.Tensor) -> torch.Tensor:
        lf = logits.float()
        finite_mask = torch.isfinite(lf)
        safe = torch.where(finite_mask, lf, torch.zeros_like(lf))
        lo = safe.amin(dim=-1, keepdim=True)
        hi = safe.amax(dim=-1, keepdim=True)
        span = (hi - lo).clamp(min=1e-12)
        step = span / (levels - 1)
        q = torch.round((lf - lo) / step).clamp(0, levels - 1)
        out = q * step + lo
        out = torch.where(finite_mask, out, lf)
        return out.to(logits.dtype)
    return fn


def defense_noise(
    sigma: float,
    logit_scale: float,
    seed: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Add N(0, (sigma * logit_scale)^2) noise to each logit entry.

    ``logit_scale`` is the std of clean logits; sigma is expressed in those units.
    Uses a fresh generator per call so that noise is deterministic but different
    for repeated calls (which matters when we apply the defense at both
    "collect last-logits" and "streaming full logits" stages).
    """
    # Two separate counters — the attack sees independent noise realizations
    # on each logit query, which is what a real API exhibits.
    state = {"counter": 0}

    def fn(logits: torch.Tensor) -> torch.Tensor:
        # Seed a CPU generator per call for reproducibility.
        gen = torch.Generator(device="cpu").manual_seed(
            seed + 7 + state["counter"]
        )
        state["counter"] += 1
        noise_cpu = torch.randn(
            logits.shape, generator=gen, dtype=torch.float32,
        )
        noise = noise_cpu.to(logits.device).to(logits.dtype) * (
            sigma * logit_scale
        )
        return logits + noise
    return fn


def defense_temperature(T: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Multiply logits by T. Monotonic rescaling — included as a control.

    T < 1 flattens (effectively raising softmax temperature); T > 1 sharpens.
    Because this is a pure scalar, it does NOT change the column space of the
    logit matrix, so Carlini subspace cos should stay near 1 and oracle W_down
    OLS should recover the scaled weights exactly (up to a known factor).
    """
    def fn(logits: torch.Tensor) -> torch.Tensor:
        return logits * T
    return fn


def defense_random_projection(
    vocab_size: int, seed: int, device: torch.device,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Apply a structured random orthogonal transform on the vocab axis.

    R z = P (s ⊙ z)  where s ∈ {+1,-1}^V is a sign vector and P is a
    permutation. This is orthogonal, has the same row-norm as z, but
    completely scrambles the column space of W_lm.
    """
    rng = torch.Generator(device="cpu").manual_seed(seed + 131)
    signs = (torch.randint(0, 2, (vocab_size,), generator=rng) * 2 - 1).float()
    perm = torch.randperm(vocab_size, generator=rng)
    signs_dev = signs.to(device)
    perm_dev = perm.to(device)

    def fn(logits: torch.Tensor) -> torch.Tensor:
        dev = logits.device
        s = signs_dev.to(dev) if signs_dev.device != dev else signs_dev
        p = perm_dev.to(dev) if perm_dev.device != dev else perm_dev
        out = logits.float() * s
        out = out.index_select(-1, p)
        return out.to(logits.dtype)
    return fn


def build_defense_registry(vocab_size: int, logit_scale: float, seed: int, device):
    """defense_name -> {param_name, params list, build(param)->fn, label(param)->str}."""
    return {
        "topk": {
            "param_name": "k",
            "params": [10, 50, 200, 1000, vocab_size],  # vocab_size == 'all'
            "build": lambda p: defense_topk(p),
            "label": lambda p: ("all" if p >= vocab_size else str(p)),
        },
        "quantize": {
            "param_name": "bits",
            "params": [4, 8, 16, 32],
            "build": lambda p: defense_quantize(p),
            "label": lambda p: str(p),
        },
        "noise": {
            "param_name": "sigma",
            "params": [0.0, 0.01, 0.1, 1.0],
            "build": lambda p: defense_noise(p, logit_scale, seed),
            "label": lambda p: f"{p:.2f}",
        },
        "temperature": {
            "param_name": "T",
            "params": [0.5, 1.0, 2.0, 5.0],
            "build": lambda p: defense_temperature(p),
            "label": lambda p: f"{p:.1f}",
        },
        "random_projection": {
            "param_name": "seed",
            "params": [seed],
            "build": lambda _: defense_random_projection(vocab_size, seed, device),
            "label": lambda _: "on",
        },
    }


# ── model helpers ───────────────────────────────────────────────────────

def load_teacher(model_name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading teacher: %s", model_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def rms_norm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms * g.float()).to(x.dtype)


# ── Collect bundle (memory-aware) ───────────────────────────────────────
#
# We do TWO forward-pass phases:
#   Phase A: collect last-position logits (N, V)
#            and block-(idx) hidden states (N, T, d) for the oracle attack
#            Rationale: these are affordable. (N, V) at 2048x151936 fp32 ≈ 1.2 GB.
#            (N, T, d) at 2048x128x896 fp32 ≈ 0.9 GB per tensor.
#   Phase B: utility held-out queries (N_util, T, V), stored in full.
#            N_util is typically 64–128 — at 128x128x151936 fp32 ≈ 9.5 GB (OK).
#
# For each defense, we also need full-logits at ALL N queries so we can recover
# h_out via pinv(W_lm) @ defended_logits. We never materialize those — instead,
# we re-run the forward pass batch by batch, apply the defense, project to d,
# and accumulate h_out_est and the W_down normal equations on the fly.


@dataclass
class ForwardBundle:
    logits_last: torch.Tensor     # (N, V)      fp32 CPU
    input_ids: torch.Tensor       # (N, T)      int64 CPU — reused for re-forwards
    h_in_full: torch.Tensor       # (N, T, d)   fp32 CPU
    h_mid_full: torch.Tensor      # (N, T, d)   fp32 CPU
    h_out_full: torch.Tensor      # (N, T, d)   fp32 CPU
    util_input_ids: torch.Tensor  # (N_util, T) held-out utility batch
    util_logits_clean: torch.Tensor  # (N_util, T, V) fp32 CPU
    W_lm: torch.Tensor            # (V, d)  fp32 CPU
    g_final: torch.Tensor         # (d,)
    g_mlp: torch.Tensor           # (d,)
    W_gate: torch.Tensor          # (d_ff, d)
    W_up: torch.Tensor            # (d_ff, d)
    W_down: torch.Tensor          # (d, d_ff)


@torch.no_grad()
def collect_forward_bundle(
    model, vocab_size: int, num_queries: int, max_seq_len: int, batch_size: int,
    block_idx: int, device: torch.device, seed: int, num_util_queries: int,
) -> ForwardBundle:
    """Run forward pass and store only affordable tensors."""
    target_block = model.model.layers[block_idx]
    attn_buf: list[torch.Tensor] = []

    def hook(_mod, _inp, out):
        attn_out = out[0] if isinstance(out, tuple) else out
        attn_buf.append(attn_out.detach().cpu().float())

    handle = target_block.self_attn.register_forward_hook(hook)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    all_last = []
    all_input_ids = []
    all_h_in, all_h_mid, all_h_out = [], [], []

    try:
        for start in tqdm(range(0, num_queries, batch_size),
                          desc="Phase A: attack queries"):
            bs = min(batch_size, num_queries - start)
            input_ids = torch.randint(
                0, vocab_size, (bs, max_seq_len), generator=gen,
            )
            input_ids_dev = input_ids.to(device)

            attn_buf.clear()
            out = model(
                input_ids_dev,
                output_hidden_states=True,
                return_dict=True,
            )

            last_logits = out.logits[:, -1, :].float().cpu()
            all_last.append(last_logits)
            all_input_ids.append(input_ids)

            hs = out.hidden_states
            h_in = hs[block_idx].detach().cpu().float()
            h_out = hs[block_idx + 1].detach().cpu().float()
            attn_out = attn_buf[0]
            h_mid = h_in + attn_out
            all_h_in.append(h_in)
            all_h_mid.append(h_mid)
            all_h_out.append(h_out)

        # Utility held-out batch (separate RNG seed to avoid overlap)
        gen_u = torch.Generator(device="cpu").manual_seed(seed + 9999)
        util_ids_list = []
        util_logits_list = []
        for start in tqdm(range(0, num_util_queries, batch_size),
                          desc="Phase B: utility held-out"):
            bs = min(batch_size, num_util_queries - start)
            input_ids = torch.randint(
                0, vocab_size, (bs, max_seq_len), generator=gen_u,
            )
            input_ids_dev = input_ids.to(device)
            attn_buf.clear()  # hook still attached; throw attn output away
            out = model(input_ids_dev, return_dict=True)
            util_ids_list.append(input_ids)
            util_logits_list.append(out.logits.float().cpu())
    finally:
        handle.remove()

    # Extract params we'll need for attacks.
    W_lm = model.lm_head.weight.data.float().cpu()
    g_final = model.model.norm.weight.data.float().cpu()
    g_mlp = model.model.layers[block_idx].post_attention_layernorm.weight.data.float().cpu()
    W_gate = model.model.layers[block_idx].mlp.gate_proj.weight.data.float().cpu()
    W_up = model.model.layers[block_idx].mlp.up_proj.weight.data.float().cpu()
    W_down = model.model.layers[block_idx].mlp.down_proj.weight.data.float().cpu()

    return ForwardBundle(
        logits_last=torch.cat(all_last, dim=0),
        input_ids=torch.cat(all_input_ids, dim=0),
        h_in_full=torch.cat(all_h_in, dim=0),
        h_mid_full=torch.cat(all_h_mid, dim=0),
        h_out_full=torch.cat(all_h_out, dim=0),
        util_input_ids=torch.cat(util_ids_list, dim=0),
        util_logits_clean=torch.cat(util_logits_list, dim=0),
        W_lm=W_lm, g_final=g_final, g_mlp=g_mlp,
        W_gate=W_gate, W_up=W_up, W_down=W_down,
    )


# ── Carlini subspace on defended last-position logits ──────────────────

def carlini_subspace_cos(
    defended_logits_last: torch.Tensor,   # (N, V)
    W_lm: torch.Tensor,                    # (V, d)
    g_final: torch.Tensor,                 # (d,)
    d_hat: int,
) -> dict[str, float]:
    """SVD the defended (N, V) matrix, compare to W_eff = W_lm * diag(g_final).

    If defense returned -inf entries (top-k), we replace with row-min-finite
    before SVD — this is the sharpest thing the attacker can do given the
    constraint.
    """
    Y = defended_logits_last.clone().float()
    finite = torch.isfinite(Y)
    if not finite.all():
        row_has_finite = finite.any(dim=-1)
        safe = torch.where(finite, Y, torch.full_like(Y, float("inf")))
        row_min = safe.amin(dim=-1, keepdim=True)
        row_min = torch.where(
            torch.isfinite(row_min), row_min, torch.zeros_like(row_min),
        )
        Y = torch.where(finite, Y, row_min.expand_as(Y))
        Y = torch.where(row_has_finite.unsqueeze(-1), Y, torch.zeros_like(Y))

    Y_c = Y - Y.mean(dim=0, keepdim=True)

    try:
        _U, _S, Vh = torch.linalg.svd(Y_c, full_matrices=False)
    except RuntimeError as e:
        logger.warning("  Carlini SVD failed: %s", e)
        return {
            "subspace_mean_cos": 0.0,
            "subspace_min_cos": 0.0,
            "procrustes_mean_cos": 0.0,
        }

    _, d = W_lm.shape
    W_eff = W_lm * g_final.unsqueeze(0)      # (V, d)
    W_hat_T = Vh[:d_hat, :].T.float()        # (V, d_hat)

    Q_hat, _ = torch.linalg.qr(W_hat_T)
    Q_true, _ = torch.linalg.qr(W_eff)
    M = Q_hat.T @ Q_true
    cos_angles = torch.linalg.svdvals(M).clamp(0.0, 1.0)
    mean_cos = float(cos_angles.mean())
    min_cos = float(cos_angles.min())

    if d_hat == d:
        cross = W_hat_T.T @ W_eff
        Up, _, Vtp = torch.linalg.svd(cross, full_matrices=False)
        W_aligned = W_hat_T @ (Up @ Vtp)
        cos_per_col = F.cosine_similarity(W_aligned, W_eff, dim=0)
        proc_mean = float(cos_per_col.mean())
    else:
        proc_mean = float("nan")

    return {
        "subspace_mean_cos": mean_cos,
        "subspace_min_cos": min_cos,
        "procrustes_mean_cos": proc_mean,
    }


# ── Streaming: re-run full forward, apply defense, project to h_out_est ─
#
# Also accumulates OLS normal equations for W_down in the same pass, using
# h_out_est (from defended logits) as target.

@torch.no_grad()
def streaming_step0_and_wdown(
    model,
    bundle: ForwardBundle,
    defense_fn: Callable,
    device: torch.device,
    batch_size: int,
    do_wdown: bool,
) -> dict[str, Any]:
    """Re-forward all queries, apply defense to (B, T, V), project to h_out_est.

    Meanwhile, accumulate:
      * h_out cosine across all (n, t) positions against oracle h_out
      * OLS normal eqs for W_down: AAt += a^T a,  RAt += r_est^T a
        where r_est = h_out_est - h_mid (oracle h_mid)

    Returns dict with h_out_cos and (optionally) W_down recovery stats.
    """
    d = bundle.W_lm.shape[1]
    d_ff = bundle.W_gate.shape[0]

    # pinv(W_eff) on CPU in float64, move to device once.
    W_eff = (bundle.W_lm * bundle.g_final.unsqueeze(0)).double()  # (V, d)
    gram = W_eff.T @ W_eff
    gram += 1e-8 * gram.diagonal().mean() * torch.eye(d, dtype=torch.float64)
    gram_inv = torch.linalg.inv(gram)
    pinv_W_eff = (gram_inv @ W_eff.T).float().to(device)           # (d, V)

    # Oracle params for W_down OLS
    g_mlp_dev = bundle.g_mlp.to(device)
    W_gate_dev = bundle.W_gate.to(device)
    W_up_dev = bundle.W_up.to(device)

    # Running sums for cosine between h_out_est and oracle h_out:
    # cos = <H_est, H_true>_F / (||H_est||_F * ||H_true||_F)
    dot_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    est_sq = torch.tensor(0.0, dtype=torch.float64, device=device)
    true_sq = torch.tensor(0.0, dtype=torch.float64, device=device)

    AAt = torch.zeros(d_ff, d_ff, dtype=torch.float64, device=device)
    RAt = torch.zeros(d, d_ff, dtype=torch.float64, device=device)

    N = bundle.input_ids.shape[0]
    T = bundle.input_ids.shape[1]

    for start in tqdm(range(0, N, batch_size), desc="    streaming eval", leave=False):
        end = min(start + batch_size, N)
        bs = end - start

        ids = bundle.input_ids[start:end].to(device)
        out = model(ids, return_dict=True)
        logits_full = out.logits  # (bs, T, V)  bf16 on CUDA

        # Apply defense on device
        defended = defense_fn(logits_full)
        # Replace any non-finite with a finite substitute for projection.
        finite_mask = torch.isfinite(defended)
        if not finite_mask.all():
            # Row-min-finite substitution (per (b, t)).
            safe = torch.where(
                finite_mask, defended.float(),
                torch.full_like(defended.float(), float("inf")),
            )
            rmn = safe.amin(dim=-1, keepdim=True)
            rmn = torch.where(
                torch.isfinite(rmn), rmn, torch.zeros_like(rmn),
            )
            defended_safe = torch.where(
                finite_mask, defended.float(), rmn.expand_as(defended.float()),
            )
        else:
            defended_safe = defended.float()

        # Project: u = defended @ pinv_W_eff^T -> (bs, T, d)
        u = defended_safe @ pinv_W_eff.T
        # Scale recovery via projection on oracle h_in
        h_in_b = bundle.h_in_full[start:end].to(device)  # (bs, T, d)
        uu = (u * u).sum(dim=-1, keepdim=True)
        uh = (u * h_in_b).sum(dim=-1, keepdim=True)
        alpha = (uh / (uu + 1e-12)).clamp(min=0.1)
        h_out_est = alpha * u                              # (bs, T, d)

        h_out_true = bundle.h_out_full[start:end].to(device)
        # Running Frobenius cosine
        dot_sum += (h_out_est.double() * h_out_true.double()).sum()
        est_sq += (h_out_est.double() ** 2).sum()
        true_sq += (h_out_true.double() ** 2).sum()

        if do_wdown:
            # Oracle activations from oracle h_mid
            h_mid_b = bundle.h_mid_full[start:end].to(device)
            hm = h_mid_b.reshape(-1, d)
            x = rms_norm(hm, g_mlp_dev)
            a = F.silu(x @ W_gate_dev.T) * (x @ W_up_dev.T)      # (bs*T, d_ff)

            # Target: r_est = h_out_est - h_mid  (use DEFENDED h_out target)
            ho_est_flat = h_out_est.reshape(-1, d)
            r_est = (ho_est_flat - hm).float()

            AAt += a.double().T @ a.double()
            RAt += r_est.double().T @ a.double()

        del logits_full, defended, defended_safe, u, h_in_b, h_out_est, h_out_true
        if device.type == "cuda" and (start // batch_size) % 8 == 0:
            torch.cuda.empty_cache()

    # Cosine
    h_out_cos = float(
        dot_sum / (torch.sqrt(est_sq) * torch.sqrt(true_sq) + 1e-30)
    )

    result: dict[str, Any] = {
        "h_out_cos": h_out_cos,
    }

    if do_wdown:
        # Auto-ridge
        diag_mean = AAt.diagonal().mean().item()
        try:
            eigvals = torch.linalg.eigvalsh(AAt)
            cond = (eigvals[-1] / eigvals[0].clamp(min=1e-30)).item()
            ridge = (1e-10 * diag_mean if cond < 1e8
                     else 1e-8 * diag_mean if cond < 1e12
                     else 1e-6 * diag_mean)
        except Exception:
            cond = -1.0
            ridge = 1e-8 * diag_mean

        AAt += ridge * torch.eye(d_ff, dtype=torch.float64, device=device)
        try:
            L = torch.linalg.cholesky(AAt)
            W_down_rec = torch.linalg.solve_triangular(
                L.T,
                torch.linalg.solve_triangular(L, RAt.T, upper=False),
                upper=True,
            ).T.float().cpu()
        except RuntimeError:
            W_down_rec = torch.linalg.solve(AAt, RAt.T).T.float().cpu()

        w_down_cos = float(F.cosine_similarity(
            W_down_rec.flatten().unsqueeze(0),
            bundle.W_down.flatten().unsqueeze(0),
        ).item())
        result["w_down_cos_defended"] = w_down_cos
        result["w_down_cond"] = cond

    return result


# ── Utility KL on held-out batch ───────────────────────────────────────

@torch.no_grad()
def kl_defended_vs_clean(
    defended_full: torch.Tensor,  # (N_util, T, V)  fp32 CPU/device
    clean_full: torch.Tensor,     # (N_util, T, V)  fp32 CPU
    device: torch.device,
    chunk: int = 16,
) -> dict[str, float]:
    """Mean KL(P_def || P_clean) per token, plus top-1 preservation rate.

    Processes in chunks along N_util to keep peak memory bounded.
    """
    N_u = defended_full.shape[0]
    assert clean_full.shape[0] == N_u

    kl_sum = 0.0
    token_count = 0
    top1_match = 0

    for s in range(0, N_u, chunk):
        e = min(s + chunk, N_u)
        d = defended_full[s:e].to(device)
        c = clean_full[s:e].to(device)

        # Subtract row max for stability; handle -inf naturally via log_softmax.
        d_ = d - d.amax(dim=-1, keepdim=True)
        c_ = c - c.amax(dim=-1, keepdim=True)
        log_p_def = F.log_softmax(d_, dim=-1)
        log_p_cln = F.log_softmax(c_, dim=-1)
        p_def = log_p_def.exp()

        # Mask: skip entries where either logprob is -inf or p_def==0
        mask = (
            torch.isfinite(log_p_def)
            & torch.isfinite(log_p_cln)
            & (p_def > 0)
        )
        contrib = torch.where(
            mask, p_def * (log_p_def - log_p_cln), torch.zeros_like(p_def),
        )
        kl_per_token = contrib.sum(dim=-1)  # (bs, T)
        kl_sum += float(kl_per_token.sum().item())
        token_count += int(kl_per_token.numel())

        top1_d = d.argmax(dim=-1)
        top1_c = c.argmax(dim=-1)
        top1_match += int((top1_d == top1_c).sum().item())

        del d, c, d_, c_, log_p_def, log_p_cln, p_def, contrib, kl_per_token

    return {
        "kl_mean": kl_sum / max(1, token_count),
        "top1_preserved": top1_match / max(1, token_count),
    }


# ── Apply defense to held-out clean logits for KL utility ──────────────

def compute_utility(
    defense_fn: Callable,
    bundle: ForwardBundle,
    device: torch.device,
) -> dict[str, float]:
    """Apply defense to held-out clean logits (not re-running forward) and
    measure KL against clean. This cleanly separates the defense's effect
    on the output distribution from forward-pass noise.
    """
    # Apply in chunks on device
    N_u, T, V = bundle.util_logits_clean.shape
    defended_chunks = []
    chunk = 16
    for s in range(0, N_u, chunk):
        e = min(s + chunk, N_u)
        c = bundle.util_logits_clean[s:e].to(device)
        d = defense_fn(c).float().cpu()
        defended_chunks.append(d)
    defended_full = torch.cat(defended_chunks, dim=0)
    return kl_defended_vs_clean(defended_full, bundle.util_logits_clean, device)


# ── One defense run ─────────────────────────────────────────────────────

def run_single_defense(
    model,
    defense_name: str,
    param_value: Any,
    defense_fn: Callable,
    bundle: ForwardBundle,
    device: torch.device,
    d_hat: int,
    batch_size: int,
    do_wdown: bool,
) -> dict[str, Any]:
    t0 = time.time()

    # (a) Carlini subspace on defended last-position logits
    with torch.no_grad():
        ll_dev = bundle.logits_last.to(device)
        defended_last = defense_fn(ll_dev).detach().cpu()
        del ll_dev
        if device.type == "cuda":
            torch.cuda.empty_cache()

    t_c = time.time()
    carlini = carlini_subspace_cos(
        defended_last, bundle.W_lm, bundle.g_final, d_hat,
    )
    logger.info(
        "  Carlini: subspace_cos=%.4f min_cos=%.4f proc_cos=%.4f (%.1fs)",
        carlini["subspace_mean_cos"], carlini["subspace_min_cos"],
        carlini["procrustes_mean_cos"], time.time() - t_c,
    )
    del defended_last

    # (b+c) Streaming Step-0 + W_down OLS
    t_s = time.time()
    stream = streaming_step0_and_wdown(
        model, bundle, defense_fn, device, batch_size, do_wdown,
    )
    logger.info(
        "  h_out_cos=%.4f  |  W_down_cos=%s  (%.1fs)",
        stream["h_out_cos"],
        (f"{stream['w_down_cos_defended']:.4f}"
         if "w_down_cos_defended" in stream else "skipped"),
        time.time() - t_s,
    )

    # (d) Utility KL
    t_u = time.time()
    util = compute_utility(defense_fn, bundle, device)
    logger.info(
        "  Utility: KL=%.4f, top1_preserved=%.4f (%.1fs)",
        util["kl_mean"], util["top1_preserved"], time.time() - t_u,
    )

    total = time.time() - t0

    return {
        "defense": defense_name,
        "param_value": param_value,
        "carlini_subspace_cos": carlini["subspace_mean_cos"],
        "carlini_subspace_min_cos": carlini["subspace_min_cos"],
        "carlini_procrustes_cos": carlini["procrustes_mean_cos"],
        "h_out_recovery_cos": stream["h_out_cos"],
        "w_down_cos_defended": stream.get("w_down_cos_defended", float("nan")),
        "w_down_cond_defended": stream.get("w_down_cond", float("nan")),
        "utility_kl": util["kl_mean"],
        "utility_top1_preserved": util["top1_preserved"],
        "wallclock_s": round(total, 1),
    }


# ── pareto table ────────────────────────────────────────────────────────

def build_summary_table(records: list[dict]) -> str:
    header = (
        f"{'defense':>18s} | {'param':>10s} | "
        f"{'carlini_cos':>11s} | {'h_out_cos':>9s} | "
        f"{'wdown_cos':>9s} | {'KL':>7s} | {'top1':>5s}"
    )
    lines = [header, "-" * len(header)]
    for r in records:
        wd = r.get("w_down_cos_defended", float("nan"))
        wd_str = f"{wd:9.4f}" if (wd == wd) else f"{'  skip':>9s}"  # nan check
        lines.append(
            f"{r['defense']:>18s} | {str(r.get('param_value_label', '')):>10s} | "
            f"{r['carlini_subspace_cos']:11.4f} | "
            f"{r['h_out_recovery_cos']:9.4f} | "
            f"{wd_str} | "
            f"{r['utility_kl']:7.3f} | "
            f"{r['utility_top1_preserved']:5.2f}"
        )
    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Defense ablation for algebraic attacks")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--num_queries", type=int, default=2048)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_util_queries", type=int, default=64,
                   help="Held-out queries for KL utility measurement")
    p.add_argument("--block_idx", type=int, default=23,
                   help="Block used for oracle W_down attack (last block)")
    p.add_argument("--output_dir", type=str, default="results/v5_defense_ablation")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_defenses", type=str, default="",
                   help="Comma-separated defense names to skip")
    p.add_argument("--only_defenses", type=str, default="",
                   help="Comma-separated defense names to run (overrides skip)")
    p.add_argument("--skip_oracle_wdown", action="store_true",
                   help="Skip the (expensive) oracle W_down OLS per defense")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, _tokenizer = load_teacher(args.model_name, device)
    vocab_size = int(model.config.vocab_size)
    hidden_size = int(model.config.hidden_size)
    logger.info(
        "Model: vocab=%d, hidden=%d, num_layers=%d",
        vocab_size, hidden_size, model.config.num_hidden_layers,
    )

    # ── Collect clean forward bundle ───────────────────────────────
    logger.info("=" * 70)
    logger.info(
        "STEP 1: Collect bundle (attack N=%d, util N=%d, T=%d, block=%d)",
        args.num_queries, args.num_util_queries, args.max_seq_len, args.block_idx,
    )
    logger.info("=" * 70)

    bundle = collect_forward_bundle(
        model, vocab_size, args.num_queries, args.max_seq_len, args.batch_size,
        args.block_idx, device, args.seed, args.num_util_queries,
    )
    logger.info(
        "Bundle: logits_last=%s, h_mid=%s, util_logits=%s",
        bundle.logits_last.shape,
        bundle.h_mid_full.shape,
        bundle.util_logits_clean.shape,
    )

    logit_scale = float(bundle.logits_last.std().item())
    logger.info("Clean logit std = %.3f (used as noise baseline scale)", logit_scale)

    # ── Clean baseline ─────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 2: Clean baseline (no defense)")
    logger.info("=" * 70)

    identity_fn = lambda z: z  # noqa: E731
    clean_record = run_single_defense(
        model=model,
        defense_name="none",
        param_value="none",
        defense_fn=identity_fn,
        bundle=bundle,
        device=device,
        d_hat=hidden_size,
        batch_size=args.batch_size,
        do_wdown=not args.skip_oracle_wdown,
    )
    clean_record["param_value_label"] = "none"
    clean_record["param_name"] = "none"

    # ── Sweep defenses ─────────────────────────────────────────────
    registry = build_defense_registry(vocab_size, logit_scale, args.seed, device)
    skip = {s.strip() for s in args.skip_defenses.split(",") if s.strip()}
    only = {s.strip() for s in args.only_defenses.split(",") if s.strip()}
    should_run = (lambda n: (n in only) if only else (n not in skip))

    all_records: list[dict] = [clean_record]

    for def_name, spec in registry.items():
        if not should_run(def_name):
            logger.info("Skipping defense: %s", def_name)
            continue

        logger.info("=" * 70)
        logger.info("DEFENSE: %s (%s in %s)",
                    def_name, spec["param_name"], spec["params"])
        logger.info("=" * 70)

        per_defense = []
        for pv in spec["params"]:
            logger.info("--- %s=%s ---", spec["param_name"], spec["label"](pv))
            defense_fn = spec["build"](pv)
            rec = run_single_defense(
                model=model,
                defense_name=def_name,
                param_value=pv,
                defense_fn=defense_fn,
                bundle=bundle,
                device=device,
                d_hat=hidden_size,
                batch_size=args.batch_size,
                do_wdown=not args.skip_oracle_wdown,
            )
            rec["param_value_label"] = spec["label"](pv)
            rec["param_name"] = spec["param_name"]
            per_defense.append(rec)
            all_records.append(rec)

        with open(out_dir / f"{def_name}.json", "w") as f:
            json.dump({
                "defense": def_name,
                "param_name": spec["param_name"],
                "records": per_defense,
            }, f, indent=2, default=str)

    # ── Save combined summary ─────────────────────────────────────
    summary = {
        "config": {
            "model_name": args.model_name,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_queries": args.num_queries,
            "num_util_queries": args.num_util_queries,
            "max_seq_len": args.max_seq_len,
            "block_idx": args.block_idx,
            "seed": args.seed,
            "device": str(device),
            "logit_scale_baseline": logit_scale,
        },
        "clean_baseline": clean_record,
        "records": all_records,
    }
    with open(out_dir / "defense_ablation.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved summary to %s", out_dir / "defense_ablation.json")

    table = build_summary_table(all_records)
    with open(out_dir / "summary_table.txt", "w") as f:
        f.write(table + "\n")
    print("\n" + "=" * 90)
    print("  DEFENSE ABLATION — SUMMARY (lower cos = stronger defense)")
    print("=" * 90)
    print(table)
    print("=" * 90)
    print(f"  JSON:  {out_dir / 'defense_ablation.json'}")
    print(f"  Table: {out_dir / 'summary_table.txt'}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
