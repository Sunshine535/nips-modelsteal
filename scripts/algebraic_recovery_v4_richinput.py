#!/usr/bin/env python3
# SAFETY NOTICE: QUARANTINED (alpha-theory prune 2026-04-19)
# This script is NOT cited in the paper. It was part of killed branches:
#   A1 S-PSI, A2 Moments CP, A4 logit-bias, A5 memory probing, A6 active query,
#   A7 algebraic v2/v3/v4, B3 matched-KD.
# Retained in repo for reproducibility of quarantined history; do not use for
# new claims.
#!/usr/bin/env python3
"""
Algebraic Block Recovery v4 — RICH-INPUT (random tokens) hypothesis test.

Diagnosis recap (from diagnose_phase2_failure.py)
-------------------------------------------------
For WikiText queries at block 23 of Qwen2.5-0.5B:
    * h_mid effective rank ≈ 9.43 / 896 (= 1.1%)
    * Phase 2 MLP cos plateau: gate=0.088, up=0.196 — matches sqrt(9/896) ≈ 0.1
      identifiability ceiling.
    * Warm-starting from teacher + small noise is PRESERVED: the teacher is a
      local minimum of the joint-opt loss.
    * W_down recovery quality is NOT the bottleneck (oracle W_down does not
      lift gate/up cos meaningfully).

Remaining hypothesis (C): data rank of X = RMSNorm(h_mid) limits recovery.
If we can inflate the rank of X, gate/up cosines should rise correspondingly.

v4 strategy
-----------
Replace WikiText queries with RANDOM TOKEN SEQUENCES (uniform over vocab).
Random tokens are out-of-distribution for the LM and should force the residual
stream to visit more of R^896 than natural text ever does.

Pipeline (reuses v3 infrastructure verbatim)
--------------------------------------------
Step 0: h_out recovery from logits via W_lm @ diag(g_final).
Phase 1: W_down via high-precision float64 OLS (oracle gate/up).
Phase 2: Joint (W_gate, W_up) optimization with W_down FIXED.

Experiments
-----------
A  (primary)  : Full v3 pipeline with random-token queries.
B  (bonus)    : Compare input distributions (wikitext / random / random-no-
                special / mixed) and report eff_rank + MLP cos.
C  (bonus)    : Scaling — N ∈ {512, 1024, 2048, 4096, 8192} with random tokens,
                track eff_rank and recovery cos.

Success criterion (same as v3)
------------------------------
ANY aligned cos > 0.3 on W_gate or W_up is a breakthrough. > 0.5 is
best-paper level.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python scripts/algebraic_recovery_v4_richinput.py \
        --model_name Qwen/Qwen2.5-0.5B --block_idx 23 --num_queries 2048 \
        --output_dir results/v5_algebraic_v4 --seed 42 \
        --phase2_steps 15000 \
        --run_comparison --run_scaling
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # noqa: BLE001
    HAS_MPL = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.permutation_alignment import align_ffn_neurons

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
D_MODEL = 896
D_FF = 4864
VOCAB_SIZE = 151936


# ══════════════════════════════════════════════════════════════════════════════
# Common helpers (copied from v3 to keep v4 self-contained)
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Algebraic Block Recovery v4 — rich input")
    p.add_argument("--model_name", type=str, default=MODEL_NAME)
    p.add_argument("--block_idx", type=int, default=23)
    p.add_argument("--num_queries", type=int, default=2048)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--output_dir", type=str, default="results/v5_algebraic_v4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for data collection and OLS accumulation")

    # Random-token generation
    p.add_argument("--reserved_tokens", type=int, default=100,
                   help="For 'random_no_special': skip the first N token ids")
    p.add_argument("--input_dist", type=str, default="random",
                   choices=["random", "random_no_special", "wikitext", "mixed"],
                   help="Input distribution for EXPERIMENT A")

    # Phase 2: joint optimization (same defaults as v3)
    p.add_argument("--phase2_steps", type=int, default=15000,
                   help="Adam optimization steps (matches v3 hyperparams)")
    p.add_argument("--opt_batch_size", type=int, default=512,
                   help="Minibatch of (x, r) pairs per Adam step")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Peak learning rate for Adam")
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--init_scale", type=float, default=0.02)
    p.add_argument("--num_restarts", type=int, default=1,
                   help="Random restarts in experiment A")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # EXPERIMENT switches
    p.add_argument("--run_comparison", action="store_true",
                   help="EXPERIMENT B: compare 4 input distributions.")
    p.add_argument("--run_scaling", action="store_true",
                   help="EXPERIMENT C: scaling in N for random tokens.")
    p.add_argument("--scaling_ns", type=str, default="512,1024,2048,4096,8192",
                   help="Comma-separated N values for the scaling sweep.")
    p.add_argument("--scaling_phase2_steps", type=int, default=5000,
                   help="Phase 2 steps per scaling point (kept short for budget)")
    p.add_argument("--comparison_phase2_steps", type=int, default=5000,
                   help="Phase 2 steps per comparison point")

    return p.parse_args()


def rms_norm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms * g.float()).to(x.dtype)


def flat_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def per_row_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    if a_f.dim() == 1:
        return flat_cosine(a_f, b_f)
    return F.cosine_similarity(a_f, b_f, dim=-1).mean().item()


def cos_with_alignment(
    W_rec: torch.Tensor,
    W_true: torch.Tensor,
    W_ref_rec: torch.Tensor,
    W_ref_true: torch.Tensor,
) -> tuple[float, float]:
    """Hungarian-align rows of W_rec to W_true using (W_ref_rec, W_ref_true)
    to build the similarity matrix. Returns (flat_cos, per_row_cos) after
    alignment."""
    from scipy.optimize import linear_sum_assignment
    n = min(W_ref_rec.shape[0], W_ref_true.shape[0])
    a = F.normalize(W_ref_rec[:n].float(), dim=-1)
    b = F.normalize(W_ref_true[:n].float(), dim=-1)
    sim = a @ b.T
    cost = 1.0 - sim.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = list(range(n))
    for r, c in zip(row_ind, col_ind):
        perm[r] = c
    W_rec_perm = torch.stack([W_rec[perm[i]] for i in range(n)])
    flat = flat_cosine(W_rec_perm, W_true[:n])
    row = per_row_cosine(W_rec_perm, W_true[:n])
    return flat, row


def lr_schedule(step: int, total: int, warmup: int, peak_lr: float) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * t))


# ══════════════════════════════════════════════════════════════════════════════
# Query building — the central contribution of v4
# ══════════════════════════════════════════════════════════════════════════════

def build_wikitext_queries(tokenizer, pool_size: int, max_seq_len: int,
                           seed: int) -> torch.Tensor:
    """Reference WikiText queries (v3 baseline)."""
    input_ids_list: list[torch.Tensor] = []
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

    remaining = pool_size - len(input_ids_list)
    if remaining > 0:
        # Pad with random tokens rather than failing — makes scaling easier.
        rng = torch.Generator().manual_seed(seed + 99991)
        random_ids = torch.randint(
            3, tokenizer.vocab_size, (remaining, max_seq_len), generator=rng
        )
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    return torch.stack(input_ids_list[:pool_size])


def build_random_token_queries(
    vocab_size: int,
    pool_size: int,
    max_seq_len: int,
    seed: int,
    reserved_tokens: int = 0,
) -> torch.Tensor:
    """Generate uniform-random token IDs.

    Parameters
    ----------
    vocab_size : int
        Size of the tokenizer vocabulary.
    pool_size : int
        Number of sequences.
    max_seq_len : int
        Sequence length.
    seed : int
        RNG seed (independent of the global torch seed).
    reserved_tokens : int, optional
        If > 0, draw from [reserved_tokens, vocab_size) to avoid special
        tokens (BOS, EOS, pad, ...) living at the low end of the vocab.

    Returns
    -------
    torch.Tensor of shape [pool_size, max_seq_len], dtype long.
    """
    rng = torch.Generator().manual_seed(seed)
    lo = int(max(0, reserved_tokens))
    hi = int(vocab_size)
    if hi <= lo:
        raise ValueError(f"Bad vocab range: [{lo}, {hi})")
    ids = torch.randint(lo, hi, (pool_size, max_seq_len), generator=rng)
    return ids


def build_mixed_queries(
    tokenizer, pool_size: int, max_seq_len: int, seed: int,
) -> torch.Tensor:
    """50% WikiText + 50% random tokens."""
    half = pool_size // 2
    other = pool_size - half
    wiki = build_wikitext_queries(tokenizer, half, max_seq_len, seed)
    rand = build_random_token_queries(
        tokenizer.vocab_size, other, max_seq_len, seed + 1, reserved_tokens=0,
    )
    combined = torch.cat([wiki, rand], dim=0)
    perm = torch.randperm(combined.shape[0],
                          generator=torch.Generator().manual_seed(seed + 2))
    return combined[perm]


def build_queries_by_name(
    dist: str, tokenizer, pool_size: int, max_seq_len: int, seed: int,
    reserved_tokens: int,
) -> torch.Tensor:
    """Dispatch to the right query builder."""
    vocab = int(tokenizer.vocab_size)
    if dist == "random":
        return build_random_token_queries(vocab, pool_size, max_seq_len,
                                          seed, reserved_tokens=0)
    if dist == "random_no_special":
        return build_random_token_queries(vocab, pool_size, max_seq_len,
                                          seed, reserved_tokens=reserved_tokens)
    if dist == "wikitext":
        return build_wikitext_queries(tokenizer, pool_size, max_seq_len, seed)
    if dist == "mixed":
        return build_mixed_queries(tokenizer, pool_size, max_seq_len, seed)
    raise ValueError(f"Unknown input distribution: {dist}")


# ══════════════════════════════════════════════════════════════════════════════
# Hidden-state collection (copied from v3, tightened)
# ══════════════════════════════════════════════════════════════════════════════

def collect_hidden_states(
    model, query_ids: torch.Tensor, block_idx: int,
    device: torch.device, batch_size: int,
    collect_logits: bool = True,
) -> dict[str, torch.Tensor]:
    """Run the model on `query_ids` and return h_in, h_mid, h_out, (logits).

    h_mid is reconstructed as h_in + self_attn_output (pre-MLP residual).
    """
    N = query_ids.shape[0]
    all_h_in, all_h_mid, all_h_out, all_logits = [], [], [], []

    target_block = model.model.layers[block_idx]
    attn_buf: list[torch.Tensor] = []

    def attn_hook(module, inputs, output):
        a = output[0] if isinstance(output, tuple) else output
        attn_buf.append(a.detach().cpu().float())

    h_handle = target_block.self_attn.register_forward_hook(attn_hook)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = query_ids[start:end].to(device)
            attn_buf.clear()
            outputs = model(
                batch, output_hidden_states=True, return_dict=True,
            )
            hs = outputs.hidden_states
            h_in_b = hs[block_idx].detach().cpu().float()
            h_out_b = hs[block_idx + 1].detach().cpu().float()
            attn_out_b = attn_buf[0]
            h_mid_b = h_in_b + attn_out_b
            all_h_in.append(h_in_b)
            all_h_mid.append(h_mid_b)
            all_h_out.append(h_out_b)
            if collect_logits:
                all_logits.append(outputs.logits.detach().cpu().float())
            if (start // batch_size) % 10 == 0:
                logger.info("  collected %d / %d", end, N)

    h_handle.remove()

    ret = {
        "h_in": torch.cat(all_h_in, dim=0),
        "h_mid": torch.cat(all_h_mid, dim=0),
        "h_out": torch.cat(all_h_out, dim=0),
    }
    if collect_logits:
        ret["logits"] = torch.cat(all_logits, dim=0)
    return ret


# ══════════════════════════════════════════════════════════════════════════════
# Rank diagnostics on X = RMSNorm(h_mid)
# ══════════════════════════════════════════════════════════════════════════════

def compute_rank_stats(
    X_all: torch.Tensor, device: torch.device,
    topks: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 768, 896),
) -> dict:
    """Return effective rank, energy fractions, and top singular values
    for X (shape [M, d])."""
    M, d = X_all.shape
    XtX = torch.zeros(d, d, dtype=torch.float64, device=device)
    chunk = 16384
    with torch.no_grad():
        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            xb = X_all[s:e].to(device).double()
            XtX += xb.T @ xb

    tr = float(XtX.diagonal().sum().item())
    frob_sq = float((XtX ** 2).sum().item())
    eff_rank = (tr * tr) / max(frob_sq, 1e-30)

    eigvals = torch.linalg.eigvalsh(XtX).cpu()
    eigvals_desc = torch.flip(eigvals, dims=[0]).clamp(min=0.0)
    sigma = torch.sqrt(eigvals_desc).numpy()

    sigma_max = float(sigma[0]) if len(sigma) > 0 and sigma[0] > 0 else 1.0
    rank_thresh = int((sigma > 0.01 * sigma_max).sum())

    total_energy = float((sigma ** 2).sum())
    energy_in_topk = {}
    for k in topks:
        if k > len(sigma):
            continue
        energy_in_topk[int(k)] = float((sigma[:k] ** 2).sum()) / max(total_energy, 1e-30)

    return {
        "M": int(M),
        "d": int(d),
        "effective_rank": float(eff_rank),
        "rank_at_1pct_sigma_max": rank_thresh,
        "sigma_max": sigma_max,
        "sigma_top10": [float(s) for s in sigma[:10]],
        "sigma_bottom10": [float(s) for s in sigma[-10:]],
        "energy_in_topk": energy_in_topk,
        "sigma_full": [float(s) for s in sigma],  # for plotting
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: high-precision OLS for W_down (oracle gate/up activations)
# ══════════════════════════════════════════════════════════════════════════════

def solve_w_down_ols(
    h_mid_all: torch.Tensor,
    h_out_all: torch.Tensor,
    g_mlp: torch.Tensor,
    W_gate_true: torch.Tensor,
    W_up_true: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, dict]:
    """Solve W_down via float64 OLS using oracle gate/up activations."""
    N, T, d = h_mid_all.shape
    d_ff = W_gate_true.shape[0]
    M = N * T

    g_mlp_dev = g_mlp.to(device)
    W_gate_dev = W_gate_true.to(device)
    W_up_dev = W_up_true.to(device)

    AAt = torch.zeros(d_ff, d_ff, dtype=torch.float64, device=device)
    RAt = torch.zeros(d, d_ff, dtype=torch.float64, device=device)

    logger.info("  Accumulating OLS normal equations (float64, M=%d)...", M)
    with torch.no_grad():
        for n_start in range(0, N, batch_size):
            n_end = min(n_start + batch_size, N)
            h_mid_b = h_mid_all[n_start:n_end].to(device)
            h_out_b = h_out_all[n_start:n_end].to(device)

            M_b = h_mid_b.shape[0] * T
            h_mid_flat = h_mid_b.reshape(M_b, d)
            h_out_flat = h_out_b.reshape(M_b, d)

            x = rms_norm(h_mid_flat, g_mlp_dev)
            gate_out = F.silu(x @ W_gate_dev.T)
            up_out = x @ W_up_dev.T
            a = gate_out * up_out
            r = (h_out_flat - h_mid_flat).float()

            a_64 = a.double()
            r_64 = r.double()
            AAt += a_64.T @ a_64
            RAt += r_64.T @ a_64

    diag_mean = AAt.diagonal().mean().item()
    try:
        eigvals = torch.linalg.eigvalsh(AAt)
        cond = (eigvals[-1] / eigvals[0].clamp(min=1e-30)).item()
        if cond < 1e8:
            ridge = 1e-10 * diag_mean
        elif cond < 1e12:
            ridge = 1e-8 * diag_mean
        else:
            ridge = 1e-6 * diag_mean
    except Exception as e:  # noqa: BLE001
        logger.warning("  eigvalsh failed (%s)", e)
        ridge = 1e-8 * diag_mean
        cond = -1.0

    AAt += ridge * torch.eye(d_ff, dtype=torch.float64, device=device)

    try:
        L = torch.linalg.cholesky(AAt)
        W_down_rec = torch.linalg.solve_triangular(
            L.T,
            torch.linalg.solve_triangular(L, RAt.T, upper=False),
            upper=True,
        ).T.float()
    except RuntimeError as e:
        logger.warning("  Cholesky failed (%s); falling back to solve", e)
        W_down_rec = torch.linalg.solve(AAt, RAt.T).T.float()

    metrics = {
        "condition_number": cond,
        "ridge": ridge,
        "diag_mean": diag_mean,
        "num_data_points": M,
    }
    return W_down_rec.cpu(), metrics


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: joint (W_gate, W_up) optimization with W_down fixed
# ══════════════════════════════════════════════════════════════════════════════

def joint_optimize_gate_up(
    X_all: torch.Tensor,            # [M, d] normalized MLP input (CPU ok)
    r_mlp_all: torch.Tensor,        # [M, d] MLP residual (CPU ok)
    W_down_fixed: torch.Tensor,     # [d, d_ff] recovered W_down
    W_gate_true: torch.Tensor,      # [d_ff, d] for eval only
    W_up_true: torch.Tensor,        # [d_ff, d] for eval only
    device: torch.device,
    total_steps: int,
    batch_size: int,
    peak_lr: float,
    warmup_steps: int,
    weight_decay: float,
    init_scale: float,
    grad_clip: float,
    eval_every: int,
    seed: int,
    W_gate_init: torch.Tensor | None = None,
    W_up_init: torch.Tensor | None = None,
    tag: str = "",
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Minimize || W_down (SiLU(W_gate x) * (W_up x)) − r || via Adam.

    W_gate_true / W_up_true are used ONLY for eval logging.
    """
    M, d = X_all.shape
    d_ff, _ = W_gate_true.shape

    torch.manual_seed(seed)

    if W_gate_init is not None:
        W_gate = nn.Parameter(W_gate_init.clone().float().to(device))
    else:
        W_gate = nn.Parameter(torch.randn(d_ff, d, device=device) * init_scale)
    if W_up_init is not None:
        W_up = nn.Parameter(W_up_init.clone().float().to(device))
    else:
        W_up = nn.Parameter(torch.randn(d_ff, d, device=device) * init_scale)

    W_down = W_down_fixed.float().to(device)

    W_gate_true_dev = W_gate_true.float().to(device)
    W_up_true_dev = W_up_true.float().to(device)

    X_cpu = X_all.pin_memory() if X_all.is_cpu else X_all
    r_cpu = r_mlp_all.pin_memory() if r_mlp_all.is_cpu else r_mlp_all

    optimizer = torch.optim.Adam(
        [W_gate, W_up], lr=peak_lr, betas=(0.9, 0.95),
        weight_decay=weight_decay, eps=1e-8,
    )

    log: list[dict] = []
    t_start = time.time()
    rng = torch.Generator()
    rng.manual_seed(seed + 10_000)

    loss_ema = None
    best_sum_cos = -2.0
    best_state = None

    for step in range(total_steps):
        cur_lr = lr_schedule(step, total_steps, warmup_steps, peak_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        idx = torch.randint(0, M, (batch_size,), generator=rng)
        x_b = X_cpu[idx].to(device, non_blocking=True)
        r_b = r_cpu[idx].to(device, non_blocking=True).float()

        g = x_b @ W_gate.T
        s = F.silu(g)
        u = x_b @ W_up.T
        a = s * u
        r_pred = a @ W_down.T
        loss = F.mse_loss(r_pred, r_b)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([W_gate, W_up], grad_clip)
        optimizer.step()

        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else 0.99 * loss_ema + 0.01 * loss_val

        if step % eval_every == 0 or step == total_steps - 1:
            with torch.no_grad():
                Wg = W_gate.detach()
                Wu = W_up.detach()

                raw_g = flat_cosine(Wg, W_gate_true_dev)
                raw_u = flat_cosine(Wu, W_up_true_dev)
                raw_g_row = per_row_cosine(Wg, W_gate_true_dev)
                raw_u_row = per_row_cosine(Wu, W_up_true_dev)

                is_final = (step == total_steps - 1)
                do_align = is_final or (step % (eval_every * 4) == 0)

                cg_flat = cg_row = cu_flat = cu_row = float("nan")
                cj_flat = cj_row = cju_flat = cju_row = float("nan")
                Wg_cpu = Wg.cpu()
                Wu_cpu = Wu.cpu()

                if do_align:
                    cg_flat, cg_row = cos_with_alignment(
                        Wg_cpu, W_gate_true.cpu(),
                        Wg_cpu, W_gate_true.cpu(),
                    )
                    cu_flat, cu_row = cos_with_alignment(
                        Wu_cpu, W_up_true.cpu(),
                        Wu_cpu, W_up_true.cpu(),
                    )
                    W_joint_rec = torch.cat([Wg_cpu, Wu_cpu], dim=1)
                    W_joint_tea = torch.cat([W_gate_true.cpu(), W_up_true.cpu()], dim=1)
                    cj_flat, cj_row = cos_with_alignment(
                        Wg_cpu, W_gate_true.cpu(),
                        W_joint_rec, W_joint_tea,
                    )
                    cju_flat, cju_row = cos_with_alignment(
                        Wu_cpu, W_up_true.cpu(),
                        W_joint_rec, W_joint_tea,
                    )

                elapsed = time.time() - t_start
                if do_align:
                    logger.info(
                        "[%s] step %5d/%d | lr=%.2e | loss=%.4e ema=%.4e | "
                        "raw(g,u)=(%.3f,%.3f) | self-aligned(g,u)=(%.3f,%.3f) | "
                        "joint-aligned(g,u)=(%.3f,%.3f) | %.1fs",
                        tag, step, total_steps, cur_lr, loss_val, loss_ema,
                        raw_g, raw_u, cg_flat, cu_flat, cj_flat, cju_flat, elapsed,
                    )
                else:
                    logger.info(
                        "[%s] step %5d/%d | lr=%.2e | loss=%.4e ema=%.4e | "
                        "raw(g,u)=(%.3f,%.3f) row=(%.3f,%.3f) | %.1fs",
                        tag, step, total_steps, cur_lr, loss_val, loss_ema,
                        raw_g, raw_u, raw_g_row, raw_u_row, elapsed,
                    )

                entry = {
                    "step": step,
                    "lr": cur_lr,
                    "loss": loss_val,
                    "loss_ema": loss_ema,
                    "raw_cos_gate": raw_g,
                    "raw_cos_up": raw_u,
                    "raw_cos_gate_row": raw_g_row,
                    "raw_cos_up_row": raw_u_row,
                    "self_aligned_cos_gate_flat": cg_flat,
                    "self_aligned_cos_gate_row": cg_row,
                    "self_aligned_cos_up_flat": cu_flat,
                    "self_aligned_cos_up_row": cu_row,
                    "joint_aligned_cos_gate_flat": cj_flat,
                    "joint_aligned_cos_gate_row": cj_row,
                    "joint_aligned_cos_up_flat": cju_flat,
                    "joint_aligned_cos_up_row": cju_row,
                }
                log.append(entry)

                if do_align and not math.isnan(cj_flat):
                    sum_cos = cj_flat + cju_flat
                else:
                    sum_cos = raw_g + raw_u
                if sum_cos > best_sum_cos:
                    best_sum_cos = sum_cos
                    best_state = {
                        "W_gate": Wg_cpu.clone(),
                        "W_up": Wu_cpu.clone(),
                        "step": step,
                        "sum_cos": sum_cos,
                    }

    if best_state is not None:
        return best_state["W_gate"], best_state["W_up"], log
    return W_gate.detach().cpu(), W_up.detach().cpu(), log


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end pipeline for a single (distribution, N) point
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    model,
    tokenizer,
    query_ids: torch.Tensor,
    g_mlp: torch.Tensor,
    W_gate_true: torch.Tensor,
    W_up_true: torch.Tensor,
    W_down_true: torch.Tensor,
    W_lm: torch.Tensor,
    g_final: torch.Tensor,
    block_idx: int,
    device: torch.device,
    batch_size: int,
    phase2_steps: int,
    opt_batch_size: int,
    lr: float,
    lr_warmup_steps: int,
    weight_decay: float,
    init_scale: float,
    grad_clip: float,
    eval_every: int,
    seed: int,
    tag: str,
    collect_logits: bool = True,
    prefix: str | None = None,
) -> dict:
    """Runs Step 0 (optional) + Phase 1 + Phase 2 on the given queries.

    Returns a dict with all metrics and (best) recovered weights.
    """
    N = query_ids.shape[0]
    T = query_ids.shape[1]

    if prefix is None:
        prefix = f"model.layers.{block_idx}"

    # ── Collect hidden states ────────────────────────────────────────────────
    logger.info("[%s] Collecting hidden states for N=%d sequences of length %d...",
                tag, N, T)
    hs = collect_hidden_states(
        model, query_ids, block_idx, device, batch_size,
        collect_logits=collect_logits,
    )
    h_mid_all = hs["h_mid"]
    h_out_all = hs["h_out"]
    h_in_all = hs["h_in"]

    out: dict = {"N": int(N), "T": int(T), "M": int(N * T), "tag": tag}

    # ── Step 0: h_out recovery from logits (if logits present) ───────────────
    if collect_logits:
        logits_all = hs["logits"]
        logger.info("[%s] Step 0: h_out recovery from logits", tag)
        W_lm_cpu = W_lm.cpu().float()
        g_final_cpu = g_final.cpu().float()
        W_eff = W_lm_cpu * g_final_cpu.unsqueeze(0)
        gram = W_eff.double().T @ W_eff.double()
        gram += 1e-8 * gram.diagonal().mean() * torch.eye(D_MODEL, dtype=torch.float64)
        gram_inv = torch.linalg.inv(gram)
        pinv_W_eff = (gram_inv @ W_eff.double().T).float()

        h_out_est = torch.zeros_like(h_out_all)
        cos_list: list[float] = []
        for n_start in range(0, N, batch_size):
            n_end = min(n_start + batch_size, N)
            z_batch = logits_all[n_start:n_end]
            h_in_batch = h_in_all[n_start:n_end]
            h_out_batch = h_out_all[n_start:n_end]

            u = z_batch @ pinv_W_eff.T
            uu = (u * u).sum(dim=-1, keepdim=True)
            uh = (u * h_in_batch).sum(dim=-1, keepdim=True)
            alpha = (uh / (uu + 1e-12)).clamp(min=0.1)
            h_out_est_batch = alpha * u
            h_out_est[n_start:n_end] = h_out_est_batch

            for i in range(0, h_out_est_batch.shape[0],
                           max(1, h_out_est_batch.shape[0] // 4)):
                for t in range(0, T, max(1, T // 4)):
                    c = flat_cosine(h_out_est_batch[i, t], h_out_batch[i, t])
                    cos_list.append(c)

        out["step0_h_out_recovery"] = {
            "mean_per_vector_cosine": float(sum(cos_list) / len(cos_list))
                                      if cos_list else 0.0,
            "global_cosine": flat_cosine(h_out_est, h_out_all),
            "num_sampled_vectors": len(cos_list),
        }
        logger.info("[%s]   h_out: mean_cos=%.4f, global=%.4f",
                    tag,
                    out["step0_h_out_recovery"]["mean_per_vector_cosine"],
                    out["step0_h_out_recovery"]["global_cosine"])
        del logits_all, pinv_W_eff, gram, gram_inv, W_eff

    del h_in_all, hs

    # ── Build (X, r_mlp) for Phase 2 ─────────────────────────────────────────
    M_total = N * T
    X_all = torch.zeros(M_total, D_MODEL)
    r_mlp_all = torch.zeros(M_total, D_MODEL)
    g_mlp_dev = g_mlp.to(device)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            h_mid_b = h_mid_all[s:e].to(device)
            h_out_b = h_out_all[s:e].to(device)
            M_b = h_mid_b.shape[0] * T
            h_mid_flat = h_mid_b.reshape(M_b, D_MODEL)
            h_out_flat = h_out_b.reshape(M_b, D_MODEL)
            x = rms_norm(h_mid_flat, g_mlp_dev)
            r = (h_out_flat - h_mid_flat).float()
            off = s * T
            X_all[off:off + M_b] = x.cpu()
            r_mlp_all[off:off + M_b] = r.cpu()

    # ── Rank diagnostics on X (BEFORE training) ──────────────────────────────
    logger.info("[%s] Rank diagnostics on X = RMSNorm(h_mid)", tag)
    t_rank = time.time()
    rank_stats = compute_rank_stats(X_all, device)
    rank_stats["time_seconds"] = round(time.time() - t_rank, 1)
    out["rank_stats"] = rank_stats
    logger.info(
        "[%s]   eff_rank=%.2f / %d | top128=%.3f | top256=%.3f | top512=%.3f",
        tag, rank_stats["effective_rank"], rank_stats["d"],
        rank_stats["energy_in_topk"].get(128, float("nan")),
        rank_stats["energy_in_topk"].get(256, float("nan")),
        rank_stats["energy_in_topk"].get(512, float("nan")),
    )

    # ── Phase 1: W_down OLS (oracle gate/up) ─────────────────────────────────
    logger.info("[%s] Phase 1: W_down OLS (oracle gate/up)", tag)
    t1 = time.time()
    W_down_rec, ols_metrics = solve_w_down_ols(
        h_mid_all, h_out_all, g_mlp,
        W_gate_true, W_up_true,
        device, batch_size,
    )
    cos_wdown = flat_cosine(W_down_rec, W_down_true)
    cos_wdown_row = per_row_cosine(W_down_rec, W_down_true)
    frob_err = (W_down_rec - W_down_true).norm().item() / (
        W_down_true.norm().item() + 1e-12
    )
    out["phase1_w_down_ols"] = {
        "W_down_cosine": cos_wdown,
        "W_down_per_row_cosine": cos_wdown_row,
        "W_down_frob_error": frob_err,
        "time_seconds": round(time.time() - t1, 1),
        **ols_metrics,
    }
    logger.info("[%s]   W_down cos=%.4f (row=%.4f)  frob=%.4f  cond=%.2e",
                tag, cos_wdown, cos_wdown_row, frob_err,
                ols_metrics.get("condition_number", -1.0))

    # Release big hidden-state tensors now that X / r_mlp are computed
    del h_mid_all, h_out_all

    # ── Phase 2: joint opt ───────────────────────────────────────────────────
    logger.info("[%s] Phase 2: joint (W_gate, W_up) optimization (%d steps)",
                tag, phase2_steps)
    t2 = time.time()
    Wg_rec, Wu_rec, log = joint_optimize_gate_up(
        X_all, r_mlp_all, W_down_rec,
        W_gate_true, W_up_true, device,
        total_steps=phase2_steps,
        batch_size=opt_batch_size,
        peak_lr=lr,
        warmup_steps=min(lr_warmup_steps, max(50, phase2_steps // 10)),
        weight_decay=weight_decay,
        init_scale=init_scale,
        grad_clip=grad_clip,
        eval_every=eval_every,
        seed=seed,
        tag=tag,
    )
    p2_time = time.time() - t2

    # ── Final alignment using full FFN permutation ───────────────────────────
    rec_dict = {
        f"{prefix}.mlp.gate_proj.weight": Wg_rec,
        f"{prefix}.mlp.up_proj.weight": Wu_rec,
        f"{prefix}.mlp.down_proj.weight": W_down_rec,
    }
    tea_dict = {
        f"{prefix}.mlp.gate_proj.weight": W_gate_true,
        f"{prefix}.mlp.up_proj.weight": W_up_true,
        f"{prefix}.mlp.down_proj.weight": W_down_true,
    }
    aligned = align_ffn_neurons(rec_dict, tea_dict, prefix)
    cg_aligned = flat_cosine(
        aligned[f"{prefix}.mlp.gate_proj.weight"], W_gate_true
    )
    cu_aligned = flat_cosine(
        aligned[f"{prefix}.mlp.up_proj.weight"], W_up_true
    )
    cd_aligned = flat_cosine(
        aligned[f"{prefix}.mlp.down_proj.weight"], W_down_true
    )
    cg_aligned_row = per_row_cosine(
        aligned[f"{prefix}.mlp.gate_proj.weight"], W_gate_true
    )
    cu_aligned_row = per_row_cosine(
        aligned[f"{prefix}.mlp.up_proj.weight"], W_up_true
    )

    cg_raw = flat_cosine(Wg_rec, W_gate_true)
    cu_raw = flat_cosine(Wu_rec, W_up_true)

    # Functional reconstruction on a held-out slice
    with torch.no_grad():
        hold_start = int(0.9 * M_total)
        hold_end = min(hold_start + 16384, M_total)
        X_hold = X_all[hold_start:hold_end].to(device)
        r_hold = r_mlp_all[hold_start:hold_end].to(device)
        Wg_dev = Wg_rec.to(device)
        Wu_dev = Wu_rec.to(device)
        Wd_dev = W_down_rec.to(device)
        a = F.silu(X_hold @ Wg_dev.T) * (X_hold @ Wu_dev.T)
        r_pred = a @ Wd_dev.T
        func_cos = flat_cosine(r_pred.cpu(), r_hold.cpu())
        func_mse = F.mse_loss(r_pred, r_hold).item()

    out["phase2_joint_opt"] = {
        "raw_cos_gate": cg_raw,
        "raw_cos_up": cu_raw,
        "aligned_cos_gate": cg_aligned,
        "aligned_cos_up": cu_aligned,
        "aligned_cos_gate_row": cg_aligned_row,
        "aligned_cos_up_row": cu_aligned_row,
        "aligned_cos_down": cd_aligned,
        "functional_recon_cos": func_cos,
        "functional_recon_mse": func_mse,
        "final_loss_ema": log[-1]["loss_ema"] if log else None,
        "time_seconds": round(p2_time, 1),
        "log_tail": log[-5:],
    }
    logger.info("[%s] Phase 2 DONE: raw(g,u)=(%.4f,%.4f) aligned(g,u)=(%.4f,%.4f)  "
                "row(%.4f,%.4f)  func_cos=%.4f  time=%.1fs",
                tag, cg_raw, cu_raw, cg_aligned, cu_aligned,
                cg_aligned_row, cu_aligned_row, func_cos, p2_time)

    out["weights"] = {
        "W_gate_rec": Wg_rec,
        "W_up_rec": Wu_rec,
        "W_down_rec": W_down_rec,
        "W_gate_rec_aligned": aligned[f"{prefix}.mlp.gate_proj.weight"],
        "W_up_rec_aligned": aligned[f"{prefix}.mlp.up_proj.weight"],
    }

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison_spectra(spectra: dict[str, list[float]], out_path: Path):
    """Overlay singular-value spectra of X for several input distributions."""
    if not HAS_MPL:
        logger.warning("matplotlib unavailable; skipping spectra plot")
        return
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for name, sigma in spectra.items():
            sigma = list(sigma)
            axes[0].plot(sigma, linewidth=1.2, label=name)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("rank index i")
        axes[0].set_ylabel(r"$\sigma_i(X)$")
        axes[0].set_title("Singular values of X = RMSNorm(h_mid)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        ks = [8, 16, 32, 64, 128, 256, 512, 768, 896]
        for name, sigma in spectra.items():
            sigma_arr = torch.tensor(sigma, dtype=torch.float64)
            total = float((sigma_arr ** 2).sum())
            energy = [float((sigma_arr[:k] ** 2).sum()) / max(total, 1e-30)
                      for k in ks if k <= len(sigma_arr)]
            kk = [k for k in ks if k <= len(sigma_arr)]
            axes[1].plot(kk, energy, marker="o", label=name)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("top-k")
        axes[1].set_ylabel("cumulative energy fraction")
        axes[1].set_title(r"Energy $\sum_{i \le k} \sigma_i^2 / \|X\|_F^2$")
        axes[1].set_ylim(0.0, 1.02)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=140)
        plt.close(fig)
        logger.info("  spectra plot saved: %s", out_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("spectra plot failed: %s", e)


def plot_scaling(
    ns: list[int],
    eff_rank: list[float],
    cos_gate: list[float],
    cos_up: list[float],
    out_path: Path,
):
    if not HAS_MPL:
        return
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(ns, eff_rank, "o-", color="C2", label="effective rank")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("N (queries)")
        axes[0].set_ylabel("eff rank of X")
        axes[0].set_title(f"Effective rank vs N (random tokens, d={D_MODEL})")
        axes[0].axhline(D_MODEL, color="k", linestyle=":", alpha=0.3,
                        label="d_model")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ns, cos_gate, "o-", color="C0", label="gate (aligned)")
        axes[1].plot(ns, cos_up, "s-", color="C1", label="up (aligned)")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("N (queries)")
        axes[1].set_ylabel("MLP cosine")
        axes[1].set_title("MLP recovery cos vs N (random tokens)")
        axes[1].set_ylim(0.0, 1.02)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=140)
        plt.close(fig)
        logger.info("  scaling plot saved: %s", out_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("scaling plot failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# JSON helper (strip non-serialisable tensors)
# ══════════════════════════════════════════════════════════════════════════════

def _jsonify(o):
    if isinstance(o, (bool, int, float, str, type(None))):
        return o
    if isinstance(o, torch.Tensor):
        if o.numel() == 1:
            return float(o.item())
        # long-ish lists become lists; tensors in the output dict should be
        # stored via torch.save, not JSON.
        return o.tolist()
    if hasattr(o, "item"):
        try:
            return float(o.item())
        except Exception:  # noqa: BLE001
            return str(o)
    if isinstance(o, (list, tuple)):
        return [_jsonify(x) for x in o]
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    return str(o)


def strip_weights(d: dict) -> dict:
    """Remove 'weights' sub-dicts so the remaining structure can be JSON-dumped."""
    def recurse(node):
        if isinstance(node, dict):
            return {k: recurse(v) for k, v in node.items() if k != "weights"}
        if isinstance(node, (list, tuple)):
            return [recurse(x) for x in node]
        return node
    return recurse(d)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    setup_logging()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    block_idx = args.block_idx
    N = args.num_queries
    T = args.max_seq_len
    BS = args.batch_size

    # Prefer offline loading where caches exist (matches diagnose script)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    logger.info("=" * 72)
    logger.info("Algebraic Block Recovery v4 — RICH INPUT (random tokens)")
    logger.info("=" * 72)
    logger.info("Model        : %s", args.model_name)
    logger.info("Block        : %d", block_idx)
    logger.info("Queries (exp A): %d x %d tokens (dist=%s) -> %d points",
                N, T, args.input_dist, N * T)
    logger.info("Device       : %s", device)
    logger.info("Phase 2      : %d steps, batch=%d, lr=%.2e",
                args.phase2_steps, args.opt_batch_size, args.lr)
    logger.info("Experiments  : primary A%s%s",
                ", +B (comparison)" if args.run_comparison else "",
                ", +C (scaling)"    if args.run_scaling    else "")

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Load teacher model and tokenizer ─────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading teacher model...")
    try:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float32,
            device_map={"": device}, trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Offline load failed (%s); retrying with network.", e)
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float32,
            device_map={"": device}, trust_remote_code=True,
        )
    teacher.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, local_files_only=True,
        )
    except Exception:  # noqa: BLE001
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Extract teacher params ───────────────────────────────────────────────
    prefix = f"model.layers.{block_idx}"
    true_params: dict[str, torch.Tensor] = {}
    for name, param in teacher.named_parameters():
        if name.startswith(prefix) or name in ("lm_head.weight", "model.norm.weight"):
            true_params[name] = param.data.cpu().clone()

    W_lm = teacher.lm_head.weight.data.float().to(device)
    g_final = teacher.model.norm.weight.data.float().to(device)
    g_mlp = true_params[f"{prefix}.post_attention_layernorm.weight"].float()
    W_gate_true = true_params[f"{prefix}.mlp.gate_proj.weight"].float()
    W_up_true = true_params[f"{prefix}.mlp.up_proj.weight"].float()
    W_down_true = true_params[f"{prefix}.mlp.down_proj.weight"].float()

    logger.info("Teacher parameters (ORACLE — eval-only):")
    logger.info("  W_gate: %s  W_up: %s  W_down: %s",
                W_gate_true.shape, W_up_true.shape, W_down_true.shape)

    # Dict of top-level results
    results: dict = {
        "config": vars(args),
        "model_info": {
            "vocab_size": int(tokenizer.vocab_size),
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "block_idx": block_idx,
        },
    }
    t_global = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT A — primary: full pipeline with random-token queries
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("EXPERIMENT A — full pipeline with dist=%s", args.input_dist)
    logger.info("=" * 72)

    queryA = build_queries_by_name(
        args.input_dist, tokenizer, N, T, args.seed,
        reserved_tokens=args.reserved_tokens,
    )
    logger.info("  query ids: %s  (min=%d, max=%d, mean=%.1f)",
                queryA.shape,
                int(queryA.min().item()), int(queryA.max().item()),
                float(queryA.float().mean().item()))

    experimentA = run_full_pipeline(
        teacher, tokenizer, queryA,
        g_mlp, W_gate_true, W_up_true, W_down_true,
        W_lm, g_final,
        block_idx, device, BS,
        phase2_steps=args.phase2_steps,
        opt_batch_size=args.opt_batch_size,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        weight_decay=args.weight_decay,
        init_scale=args.init_scale,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        seed=args.seed,
        tag=f"A-{args.input_dist}",
        collect_logits=True,
        prefix=prefix,
    )

    results["experiment_A"] = {
        "input_distribution": args.input_dist,
        "reserved_tokens": args.reserved_tokens,
        **experimentA,
    }

    # Persist experiment-A weights
    torch.save({
        "distribution": args.input_dist,
        **experimentA.get("weights", {}),
    }, out_dir / "expA_weights.pt")

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT B — compare input distributions
    # ══════════════════════════════════════════════════════════════════════════
    comparison_plot_path = None
    if args.run_comparison:
        logger.info("=" * 72)
        logger.info("EXPERIMENT B — comparing input distributions")
        logger.info("=" * 72)

        # We always run experimentA's distribution so it appears here too,
        # but reuse its results to save time.
        DIST_ORDER = ["wikitext", "random", "random_no_special", "mixed"]
        comparison: list[dict] = []
        spectra: dict[str, list[float]] = {}

        # Seed the comparison entry for experiment A's distribution
        if args.input_dist in DIST_ORDER:
            entry = {
                "distribution": args.input_dist,
                "note": "(reuse of EXPERIMENT A)",
                "rank_stats": experimentA["rank_stats"],
                "phase1_w_down_ols": experimentA["phase1_w_down_ols"],
                "phase2_joint_opt": experimentA["phase2_joint_opt"],
            }
            comparison.append(entry)
            spectra[args.input_dist] = experimentA["rank_stats"]["sigma_full"]

        # Run remaining distributions
        already = {args.input_dist}
        for dist in DIST_ORDER:
            if dist in already:
                continue
            logger.info("-" * 72)
            logger.info("  B / dist = %s", dist)
            logger.info("-" * 72)
            try:
                qids = build_queries_by_name(
                    dist, tokenizer, N, T, args.seed + 1_000,
                    reserved_tokens=args.reserved_tokens,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("  skip dist=%s: %s", dist, e)
                continue

            expB = run_full_pipeline(
                teacher, tokenizer, qids,
                g_mlp, W_gate_true, W_up_true, W_down_true,
                W_lm, g_final,
                block_idx, device, BS,
                phase2_steps=args.comparison_phase2_steps,
                opt_batch_size=args.opt_batch_size,
                lr=args.lr,
                lr_warmup_steps=args.lr_warmup_steps,
                weight_decay=args.weight_decay,
                init_scale=args.init_scale,
                grad_clip=args.grad_clip,
                eval_every=args.eval_every,
                seed=args.seed + 2_000,
                tag=f"B-{dist}",
                collect_logits=False,
                prefix=prefix,
            )
            entry = {
                "distribution": dist,
                "rank_stats": expB["rank_stats"],
                "phase1_w_down_ols": expB["phase1_w_down_ols"],
                "phase2_joint_opt": expB["phase2_joint_opt"],
            }
            comparison.append(entry)
            spectra[dist] = expB["rank_stats"]["sigma_full"]

        results["experiment_B"] = {
            "phase2_steps": args.comparison_phase2_steps,
            "results": comparison,
        }

        # Plot overlay spectra
        comparison_plot_path = out_dir / "expB_spectra.png"
        plot_comparison_spectra(spectra, comparison_plot_path)
        results["experiment_B"]["spectra_plot"] = str(comparison_plot_path)

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT C — scaling in N (random tokens)
    # ══════════════════════════════════════════════════════════════════════════
    scaling_plot_path = None
    if args.run_scaling:
        logger.info("=" * 72)
        logger.info("EXPERIMENT C — scaling in N (random tokens)")
        logger.info("=" * 72)

        ns_list = [int(n) for n in args.scaling_ns.split(",") if n.strip()]
        scaling: list[dict] = []
        eff_rank_list: list[float] = []
        cos_gate_list: list[float] = []
        cos_up_list: list[float] = []

        for n_val in ns_list:
            logger.info("-" * 72)
            logger.info("  C / N = %d (random tokens)", n_val)
            logger.info("-" * 72)

            qids = build_random_token_queries(
                tokenizer.vocab_size, n_val, T,
                args.seed + 5_000 + n_val,
                reserved_tokens=0,
            )
            expC = run_full_pipeline(
                teacher, tokenizer, qids,
                g_mlp, W_gate_true, W_up_true, W_down_true,
                W_lm, g_final,
                block_idx, device, BS,
                phase2_steps=args.scaling_phase2_steps,
                opt_batch_size=args.opt_batch_size,
                lr=args.lr,
                lr_warmup_steps=args.lr_warmup_steps,
                weight_decay=args.weight_decay,
                init_scale=args.init_scale,
                grad_clip=args.grad_clip,
                eval_every=args.eval_every,
                seed=args.seed + 7_000 + n_val,
                tag=f"C-N{n_val}",
                collect_logits=False,
                prefix=prefix,
            )

            entry = {
                "N": n_val,
                "rank_stats": expC["rank_stats"],
                "phase1_w_down_ols": expC["phase1_w_down_ols"],
                "phase2_joint_opt": expC["phase2_joint_opt"],
            }
            scaling.append(entry)
            eff_rank_list.append(expC["rank_stats"]["effective_rank"])
            cos_gate_list.append(expC["phase2_joint_opt"]["aligned_cos_gate"])
            cos_up_list.append(expC["phase2_joint_opt"]["aligned_cos_up"])

        results["experiment_C"] = {
            "ns": ns_list,
            "phase2_steps": args.scaling_phase2_steps,
            "results": scaling,
        }

        scaling_plot_path = out_dir / "expC_scaling.png"
        plot_scaling(ns_list, eff_rank_list, cos_gate_list, cos_up_list,
                     scaling_plot_path)
        results["experiment_C"]["plot"] = str(scaling_plot_path)

    # ── Save JSON (strip tensors + weights) ──────────────────────────────────
    results["elapsed_seconds"] = round(time.time() - t_global, 1)
    cleaned = strip_weights(results)
    with open(out_dir / "results.json", "w") as f:
        json.dump(_jsonify(cleaned), f, indent=2)
    logger.info("Results JSON saved: %s", out_dir / "results.json")

    # ══════════════════════════════════════════════════════════════════════════
    # Final summary (print banner)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ALGEBRAIC BLOCK RECOVERY v4 (RICH INPUT) — SUMMARY")
    print("=" * 80)

    expA = results["experiment_A"]
    rs = expA["rank_stats"]
    print(f"\nEXPERIMENT A — dist={expA['input_distribution']} "
          f"N={expA['N']}  T={expA['T']}  M={expA['M']}")
    print(f"  eff_rank = {rs['effective_rank']:.2f} / {rs['d']}  "
          f"(= {rs['effective_rank']/rs['d']:.3f})")
    print(f"  rank@1%σ_max = {rs['rank_at_1pct_sigma_max']}")
    for k in (128, 256, 512):
        if k in rs["energy_in_topk"]:
            print(f"  top-{k} energy = {rs['energy_in_topk'][k]:.4f}")

    p1 = expA["phase1_w_down_ols"]
    print(f"\n  Phase 1 W_down cos       = {p1['W_down_cosine']:.4f}  "
          f"(row {p1['W_down_per_row_cosine']:.4f})")
    p2 = expA["phase2_joint_opt"]
    print(f"  Phase 2 aligned gate cos = {p2['aligned_cos_gate']:.4f}  "
          f"(row {p2['aligned_cos_gate_row']:.4f})")
    print(f"  Phase 2 aligned up   cos = {p2['aligned_cos_up']:.4f}  "
          f"(row {p2['aligned_cos_up_row']:.4f})")
    print(f"  Phase 2 functional cos   = {p2['functional_recon_cos']:.4f}  "
          f"(mse {p2['functional_recon_mse']:.4e})")

    if "step0_h_out_recovery" in expA:
        s0 = expA["step0_h_out_recovery"]
        print(f"\n  Step 0 h_out recovery:")
        print(f"    mean per-vector cos = {s0['mean_per_vector_cosine']:.4f}")
        print(f"    global cos          = {s0['global_cosine']:.4f}")

    # Verdict for experiment A
    g_best = p2["aligned_cos_gate"]
    u_best = p2["aligned_cos_up"]
    any_best = max(g_best, u_best, p2["raw_cos_gate"], p2["raw_cos_up"])
    print("\n  ── Verdict (experiment A) ──")
    if any_best > 0.5:
        print("    *** BEST-PAPER LEVEL: cos > 0.5 ***")
    elif any_best > 0.3:
        print("    *** BREAKTHROUGH: cos > 0.3 ***")
    elif any_best > 0.1:
        print("    Partial signal (cos > 0.1)")
    else:
        print("    No signal (cos < 0.1) — rich-input hypothesis REJECTED "
              "at this scale.")

    if "experiment_B" in results:
        print("\nEXPERIMENT B — input distribution sweep")
        for e in results["experiment_B"]["results"]:
            rs = e["rank_stats"]
            p2 = e["phase2_joint_opt"]
            p1 = e["phase1_w_down_ols"]
            print(f"  dist={e['distribution']:>18s}  "
                  f"eff_rank={rs['effective_rank']:7.2f}  "
                  f"Wdown={p1['W_down_cosine']:.3f}  "
                  f"gate={p2['aligned_cos_gate']:.3f}  "
                  f"up={p2['aligned_cos_up']:.3f}  "
                  f"recon={p2['functional_recon_cos']:.3f}")
        if comparison_plot_path is not None:
            print(f"  spectra plot: {comparison_plot_path}")

    if "experiment_C" in results:
        print("\nEXPERIMENT C — scaling in N (random tokens)")
        for e in results["experiment_C"]["results"]:
            rs = e["rank_stats"]
            p2 = e["phase2_joint_opt"]
            p1 = e["phase1_w_down_ols"]
            print(f"  N={e['N']:>5d}  eff_rank={rs['effective_rank']:7.2f}  "
                  f"Wdown={p1['W_down_cosine']:.3f}  "
                  f"gate={p2['aligned_cos_gate']:.3f}  "
                  f"up={p2['aligned_cos_up']:.3f}")
        if scaling_plot_path is not None:
            print(f"  scaling plot: {scaling_plot_path}")

    print(f"\nTotal elapsed: {results['elapsed_seconds']:.1f}s")
    print(f"Results JSON : {out_dir / 'results.json'}")
    print(f"Weights (A)  : {out_dir / 'expA_weights.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
