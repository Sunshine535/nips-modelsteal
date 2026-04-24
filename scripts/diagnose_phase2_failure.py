#!/usr/bin/env python3
"""
Diagnose Phase 2 MLP (W_gate, W_up) Recovery Failure.

Context
-------
In algebraic_recovery_v3_breakthrough.py, Phase 2 jointly optimizes
(W_gate, W_up) against a fixed W_down (recovered via Phase 1 OLS).
The functional residual (r_pred - r_true) reconstruction converges to
cos ~ 1.0, but parameter-level cosine plateaus at (gate=0.09, up=0.20).
Three random restarts converge to the same values.

This script runs THREE experiments to determine WHICH of three
hypotheses explains the plateau:

    A) W_down imperfection corrupts the gate/up learning target.
       Test: replace W_down_rec with W_down_true (oracle) and see if
       cos jumps above 0.5.

    B) Teacher (W_gate, W_up) is NOT a local minimum of the loss.
       Test: warm-start from teacher + alpha * noise for alpha in
       {0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0} with W_down_true
       and observe whether training PRESERVES, IMPROVES, or DEGRADES
       the initial cosine.

    C) Data rank limitation. The MLP input X = RMSNorm(h_mid) has
       shape [M, 896] but may only span a low-dim subspace of R^896.
       Test: SVD of X, report effective rank, energy spectrum, and
       per-neuron data coverage.

Critical: teacher weights are used ONLY for diagnostic evaluation and
explicitly labeled "ORACLE". The main v3 recovery path remains clean.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_phase2_failure.py \
        --model_name Qwen/Qwen2.5-0.5B --block_idx 23 --num_queries 2048 \
        --output_dir results/v5_diagnose_phase2 --seed 42 \
        --phase_a_steps 10000 --phase_b_steps 5000
"""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
D_MODEL = 896
D_FF = 4864
VOCAB_SIZE = 151936


# ══════════════════════════════════════════════════════════════════════════════
# Helpers (copied / adapted from algebraic_recovery_v3_breakthrough.py)
# ══════════════════════════════════════════════════════════════════════════════


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 2 failure mode diagnostic (A / B / C hypotheses)."
    )
    p.add_argument("--model_name", type=str, default=MODEL_NAME)
    p.add_argument("--block_idx", type=int, default=23)
    p.add_argument("--num_queries", type=int, default=2048)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument(
        "--output_dir", type=str, default="results/v5_diagnose_phase2"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--allow_synthetic", action="store_true",
        help="Fall back to random tokens if dataset load fails.",
    )

    # Phase 2 joint-opt hyperparams
    p.add_argument("--phase_a_steps", type=int, default=10000,
                   help="Steps for Hypothesis A cold-start runs.")
    p.add_argument("--phase_b_steps", type=int, default=5000,
                   help="Steps for Hypothesis B warm-start sweep.")
    p.add_argument("--opt_batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--init_scale", type=float, default=0.02)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Hypothesis B noise levels
    p.add_argument(
        "--warm_alphas", type=str,
        default="0.0,0.01,0.05,0.1,0.3,0.5,1.0,2.0",
        help="Comma-separated alphas for warm-start sweep.",
    )

    return p.parse_args()


def build_query_pool(tokenizer, pool_size: int, max_seq_len: int, seed: int,
                     allow_synthetic: bool = False) -> torch.Tensor:
    """Build query input_ids from WikiText (fallback: random tokens)."""
    input_ids_list: list[torch.Tensor] = []
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
    except Exception as e:  # noqa: BLE001
        if not allow_synthetic:
            raise RuntimeError(
                f"Dataset load failed: {e}. Use --allow_synthetic to fall back."
            ) from e
        logger.warning("Dataset load failed: %s. Using random tokens.", e)

    remaining = pool_size - len(input_ids_list)
    if remaining > 0:
        if not allow_synthetic:
            raise RuntimeError(
                f"Only {len(input_ids_list)}/{pool_size} from dataset. "
                "Use --allow_synthetic to pad with random tokens."
            )
        rng = torch.Generator().manual_seed(seed)
        random_ids = torch.randint(
            3, tokenizer.vocab_size, (remaining, max_seq_len), generator=rng
        )
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    return torch.stack(input_ids_list[:pool_size])


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


def lr_schedule(step: int, total: int, warmup: int, peak_lr: float) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * t))


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 OLS for W_down (cold-start, "recovered" W_down)
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
    """Float64 OLS W_down given oracle gate/up activations."""
    N, T, d = h_mid_all.shape
    d_ff = W_gate_true.shape[0]
    M = N * T

    g_mlp_dev = g_mlp.to(device)
    W_gate_dev = W_gate_true.to(device)
    W_up_dev = W_up_true.to(device)

    AAt = torch.zeros(d_ff, d_ff, dtype=torch.float64, device=device)
    RAt = torch.zeros(d, d_ff, dtype=torch.float64, device=device)

    logger.info("  Accumulating normal equations (float64, M=%d points)...", M)
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
        logger.warning("  Eigenvalue computation failed (%s)", e)
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
    except RuntimeError:
        W_down_rec = torch.linalg.solve(AAt, RAt.T).T.float()

    metrics = {
        "condition_number": cond,
        "ridge": ridge,
        "diag_mean": diag_mean,
        "num_data_points": M,
    }
    return W_down_rec.cpu(), metrics


# ══════════════════════════════════════════════════════════════════════════════
# Joint (W_gate, W_up) optimization (copied from v3, trimmed for diagnostics)
# ══════════════════════════════════════════════════════════════════════════════


def joint_optimize_gate_up(
    X_all: torch.Tensor,
    r_mlp_all: torch.Tensor,
    W_down_fixed: torch.Tensor,
    W_gate_true: torch.Tensor,
    W_up_true: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, list[dict], dict]:
    """Minimize || W_down (SiLU(W_gate X) * (W_up X)) - r || via Adam.

    W_gate_true / W_up_true are used ONLY for evaluation logging (never
    enter the loss).
    """
    M, d = X_all.shape
    d_ff, _ = W_gate_true.shape

    torch.manual_seed(seed)

    if W_gate_init is not None:
        W_gate = nn.Parameter(W_gate_init.clone().float().to(device))
    else:
        W_gate = nn.Parameter(
            torch.randn(d_ff, d, device=device) * init_scale
        )
    if W_up_init is not None:
        W_up = nn.Parameter(W_up_init.clone().float().to(device))
    else:
        W_up = nn.Parameter(
            torch.randn(d_ff, d, device=device) * init_scale
        )

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

    # Record init-time cosine
    with torch.no_grad():
        init_cos_g = flat_cosine(W_gate.detach(), W_gate_true_dev)
        init_cos_u = flat_cosine(W_up.detach(), W_up_true_dev)
        init_cos_g_row = per_row_cosine(W_gate.detach(), W_gate_true_dev)
        init_cos_u_row = per_row_cosine(W_up.detach(), W_up_true_dev)

    loss_ema = None
    best_sum = -2.0
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
        if loss_ema is None:
            loss_ema = loss_val
        else:
            loss_ema = 0.99 * loss_ema + 0.01 * loss_val

        if step % eval_every == 0 or step == total_steps - 1:
            with torch.no_grad():
                Wg = W_gate.detach()
                Wu = W_up.detach()
                cos_g = flat_cosine(Wg, W_gate_true_dev)
                cos_u = flat_cosine(Wu, W_up_true_dev)
                cos_g_row = per_row_cosine(Wg, W_gate_true_dev)
                cos_u_row = per_row_cosine(Wu, W_up_true_dev)
                elapsed = time.time() - t_start
                logger.info(
                    "[%s] step %5d/%d | lr=%.2e | loss=%.4e ema=%.4e | "
                    "cos(g,u)=(%.3f,%.3f) row=(%.3f,%.3f) | %.1fs",
                    tag, step, total_steps, cur_lr, loss_val, loss_ema,
                    cos_g, cos_u, cos_g_row, cos_u_row, elapsed,
                )
                log.append({
                    "step": step,
                    "lr": cur_lr,
                    "loss": loss_val,
                    "loss_ema": loss_ema,
                    "cos_gate": cos_g,
                    "cos_up": cos_u,
                    "cos_gate_row": cos_g_row,
                    "cos_up_row": cos_u_row,
                })
                sum_cos = cos_g + cos_u
                if sum_cos > best_sum:
                    best_sum = sum_cos
                    best_state = {
                        "W_gate": Wg.cpu().clone(),
                        "W_up": Wu.cpu().clone(),
                        "step": step,
                        "cos_gate": cos_g,
                        "cos_up": cos_u,
                        "cos_gate_row": cos_g_row,
                        "cos_up_row": cos_u_row,
                    }

    init_info = {
        "init_cos_gate": init_cos_g,
        "init_cos_up": init_cos_u,
        "init_cos_gate_row": init_cos_g_row,
        "init_cos_up_row": init_cos_u_row,
    }

    if best_state is not None:
        return best_state["W_gate"], best_state["W_up"], log, init_info
    return W_gate.detach().cpu(), W_up.detach().cpu(), log, init_info


# ══════════════════════════════════════════════════════════════════════════════
# Hypothesis C: Data rank analysis
# ══════════════════════════════════════════════════════════════════════════════


def analyze_data_rank(
    X_all: torch.Tensor,           # [M, d] (CPU)
    W_gate_true: torch.Tensor,     # [d_ff, d]
    W_up_true: torch.Tensor,       # [d_ff, d]
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Compute eigen-spectrum and per-neuron coverage of X = RMSNorm(h_mid)."""
    logger.info("  Computing X^T X in float64 (d=%d)...", X_all.shape[1])
    M, d = X_all.shape

    # X^T X (d x d) in float64 — streaming over rows if M is large
    XtX = torch.zeros(d, d, dtype=torch.float64, device=device)
    chunk = 16384
    with torch.no_grad():
        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            xb = X_all[s:e].to(device).double()
            XtX += xb.T @ xb

    # Divide by M for empirical covariance (but keep the uncentred matrix too)
    XtX_full = XtX.clone()
    cov = XtX_full / float(M)

    # Eigen-decompose symmetric PSD
    logger.info("  Eigendecomposing X^T X (d=%d)...", d)
    eigvals, eigvecs = torch.linalg.eigh(XtX_full)  # ascending

    # Convert to singular values of X (sqrt of eigenvalues of X^T X)
    eigvals_cpu = eigvals.cpu()
    eigvals_desc = torch.flip(eigvals_cpu, dims=[0])
    eigvals_desc = eigvals_desc.clamp(min=0.0)
    sigma = torch.sqrt(eigvals_desc).numpy()  # [d]
    sigma_max = float(sigma[0]) if sigma[0] > 0 else 1.0

    # Effective rank: (trace(A))^2 / ||A||_F^2 where A = XX^T (or X^T X)
    tr = float(XtX_full.diagonal().sum().item())
    frob_sq = float((XtX_full ** 2).sum().item())
    effective_rank = (tr * tr) / max(frob_sq, 1e-30)

    # Rank thresholded at 0.01 * sigma_max
    sigma_thresh = 0.01 * sigma_max
    rank_threshold = int((sigma > sigma_thresh).sum())

    # Energy fractions in top-k
    total_energy = float((sigma ** 2).sum())
    energy_in_topk = {}
    for k in (8, 16, 32, 64, 128, 256, 512, 768, 896):
        if k > len(sigma):
            continue
        top_energy = float((sigma[:k] ** 2).sum())
        energy_in_topk[k] = top_energy / max(total_energy, 1e-30)

    # Plot singular-value spectrum
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(sigma, marker=".", linewidth=0.8)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("rank index i")
        axes[0].set_ylabel(r"$\sigma_i(X)$")
        axes[0].set_title(
            f"Singular values of X (M={M}, d={d})\n"
            f"eff_rank={effective_rank:.1f}, thresh_rank={rank_threshold}"
        )
        axes[0].axvline(effective_rank, color="red", linestyle="--", alpha=0.5,
                        label=f"eff rank {effective_rank:.0f}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        energy_k = sorted(energy_in_topk.keys())
        energy_v = [energy_in_topk[k] for k in energy_k]
        axes[1].plot(energy_k, energy_v, marker="o")
        axes[1].set_xlabel("top-k")
        axes[1].set_ylabel("cumulative energy fraction")
        axes[1].set_title(r"Cumulative energy $\sum_{i\le k} \sigma_i^2 / \|X\|_F^2$")
        axes[1].set_xscale("log")
        axes[1].set_ylim(0.0, 1.02)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        spectrum_path = out_dir / "hypC_spectrum.png"
        plt.savefig(spectrum_path, dpi=140)
        plt.close(fig)
        logger.info("  Spectrum plot saved: %s", spectrum_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("  Spectrum plot failed: %s", e)
        spectrum_path = None

    # Per-neuron data coverage
    # coverage_n = || W_n^T Sigma_X W_n || / (sigma_max(X)^2 * || W_n ||^2)
    #            = v_n^T Sigma_X v_n where v_n = W_n / ||W_n||
    # = fraction of the probe direction covered by data, scaled to max eigval
    # Normalize by top eigenvalue so fully-aligned = 1.
    logger.info("  Computing per-neuron data coverage (gate, up)...")
    top_eig = float(eigvals_desc[0])
    if top_eig <= 0:
        top_eig = 1.0

    def coverage(W: torch.Tensor) -> torch.Tensor:
        # W: [d_ff, d]. Return [d_ff]: (W_n^T XtX W_n) / (||W_n||^2 * top_eig * M)
        W_dev = W.to(device).float()
        # coverage = row-wise quadratic form normalized to [0, 1]
        Wd = W_dev.double()
        # (W XtX_norm) row-wise
        XtX_norm = XtX_full / (top_eig * float(M))
        qform = (Wd @ XtX_norm @ Wd.T).diagonal()
        w_sq = (Wd * Wd).sum(dim=-1)
        cov = qform / (w_sq + 1e-30)
        return cov.cpu()

    cov_gate = coverage(W_gate_true)
    cov_up = coverage(W_up_true)

    neuron_stats = {
        "gate_mean": float(cov_gate.mean()),
        "gate_median": float(cov_gate.median()),
        "gate_min": float(cov_gate.min()),
        "gate_max": float(cov_gate.max()),
        "gate_p10": float(torch.quantile(cov_gate, 0.10)),
        "gate_p90": float(torch.quantile(cov_gate, 0.90)),
        "up_mean": float(cov_up.mean()),
        "up_median": float(cov_up.median()),
        "up_min": float(cov_up.min()),
        "up_max": float(cov_up.max()),
        "up_p10": float(torch.quantile(cov_up, 0.10)),
        "up_p90": float(torch.quantile(cov_up, 0.90)),
    }

    # Plot neuron coverage histogram
    try:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        ax.hist(cov_gate.numpy(), bins=80, alpha=0.55, label="gate", color="C0")
        ax.hist(cov_up.numpy(), bins=80, alpha=0.55, label="up", color="C1")
        ax.set_xlabel(r"per-neuron coverage $v_n^\top \Sigma_X v_n \,/\, \sigma_{\max}^2$")
        ax.set_ylabel("count")
        ax.set_title(
            "Per-neuron data coverage (teacher gate/up vs X spectrum)\n"
            f"gate median={neuron_stats['gate_median']:.3f}, "
            f"up median={neuron_stats['up_median']:.3f}"
        )
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        cov_plot_path = out_dir / "hypC_neuron_coverage.png"
        plt.tight_layout()
        plt.savefig(cov_plot_path, dpi=140)
        plt.close(fig)
        logger.info("  Coverage histogram saved: %s", cov_plot_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("  Coverage plot failed: %s", e)
        cov_plot_path = None

    return {
        "M": M,
        "d": d,
        "effective_rank": effective_rank,
        "rank_at_1pct_sigma_max": rank_threshold,
        "sigma_max": sigma_max,
        "sigma_min_nonzero": float(sigma[sigma > 0].min()) if (sigma > 0).any() else 0.0,
        "sigma_top10": [float(s) for s in sigma[:10]],
        "sigma_bottom10": [float(s) for s in sigma[-10:]],
        "energy_in_topk": energy_in_topk,
        "neuron_coverage_stats": neuron_stats,
        "spectrum_plot": str(spectrum_path) if spectrum_path else None,
        "coverage_plot": str(cov_plot_path) if cov_plot_path else None,
    }


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

    logger.info("=" * 72)
    logger.info("Phase 2 Failure Mode Diagnostic (Hypotheses A / B / C)")
    logger.info("=" * 72)
    logger.info("Model  : %s", args.model_name)
    logger.info("Block  : %d", block_idx)
    logger.info("Queries: %d x %d tokens = %d data points", N, T, N * T)
    logger.info("Device : %s", device)
    logger.info("Output : %s", out_dir)
    logger.info("Phase A: %d steps | Phase B: %d steps",
                args.phase_a_steps, args.phase_b_steps)

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Prefer offline loading where caches exist
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # ── Load teacher ─────────────────────────────────────────────────────────
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

    # ── Extract teacher parameters (for diagnostic use only, explicitly flagged) ──
    prefix = f"model.layers.{block_idx}"
    true_params: dict[str, torch.Tensor] = {}
    for name, param in teacher.named_parameters():
        if name.startswith(prefix) or name in ("lm_head.weight", "model.norm.weight"):
            true_params[name] = param.data.cpu().clone()

    g_mlp = true_params[f"{prefix}.post_attention_layernorm.weight"].float()
    W_gate_true = true_params[f"{prefix}.mlp.gate_proj.weight"].float()
    W_up_true = true_params[f"{prefix}.mlp.up_proj.weight"].float()
    W_down_true = true_params[f"{prefix}.mlp.down_proj.weight"].float()

    logger.info("Teacher params (ORACLE — diagnostic use only):")
    logger.info("  W_gate: %s  W_up: %s  W_down: %s",
                W_gate_true.shape, W_up_true.shape, W_down_true.shape)

    # ── Build query pool ─────────────────────────────────────────────────────
    logger.info("Building query pool (%d queries)...", N)
    query_ids = build_query_pool(
        tokenizer, N, T, args.seed, allow_synthetic=args.allow_synthetic,
    )
    logger.info("Query pool: %s", query_ids.shape)

    # ── Collect hidden states ────────────────────────────────────────────────
    logger.info("Collecting hidden states (h_in, h_mid, h_out)...")

    all_h_mid = []
    all_h_out = []

    target_block = teacher.model.layers[block_idx]
    attn_buf: list[torch.Tensor] = []

    def attn_hook(module, inputs, output):
        a = output[0] if isinstance(output, tuple) else output
        attn_buf.append(a.detach().cpu().float())

    h_handle = target_block.self_attn.register_forward_hook(attn_hook)

    with torch.no_grad():
        for start in range(0, N, BS):
            end = min(start + BS, N)
            batch = query_ids[start:end].to(device)
            attn_buf.clear()
            outputs = teacher(batch, output_hidden_states=True, return_dict=True)
            hs = outputs.hidden_states
            h_in_batch = hs[block_idx].detach().cpu().float()
            h_out_batch = hs[block_idx + 1].detach().cpu().float()
            attn_out_batch = attn_buf[0]
            h_mid_batch = h_in_batch + attn_out_batch
            all_h_mid.append(h_mid_batch)
            all_h_out.append(h_out_batch)
            if (start // BS) % 10 == 0:
                logger.info("  Collected %d / %d", end, N)

    h_handle.remove()
    del teacher
    torch.cuda.empty_cache()

    h_mid_all = torch.cat(all_h_mid, dim=0)
    h_out_all = torch.cat(all_h_out, dim=0)
    del all_h_mid, all_h_out
    logger.info("  h_mid: %s  h_out: %s", h_mid_all.shape, h_out_all.shape)

    # Precompute X = RMSNorm(h_mid) and r_mlp = h_out - h_mid
    logger.info("Building training data (X, r_mlp)...")
    M_total = N * T
    X_all = torch.zeros(M_total, D_MODEL)
    r_mlp_all = torch.zeros(M_total, D_MODEL)

    g_mlp_dev = g_mlp.to(device)
    with torch.no_grad():
        for s in range(0, N, BS):
            e = min(s + BS, N)
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

    logger.info("  X: %s  r_mlp: %s", X_all.shape, r_mlp_all.shape)

    results: dict = {"config": vars(args)}

    # ══════════════════════════════════════════════════════════════════════════
    # HYPOTHESIS C — Data rank analysis (run first; cheap, no optimization)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("HYPOTHESIS C: data rank of X = RMSNorm(h_mid)")
    logger.info("=" * 72)
    t_c = time.time()
    hypC = analyze_data_rank(
        X_all, W_gate_true, W_up_true, device, out_dir,
    )
    hypC["time_seconds"] = round(time.time() - t_c, 1)
    results["hypothesis_C_data_rank"] = hypC

    logger.info("  Effective rank (trace^2 / Frobenius^2): %.2f / %d",
                hypC["effective_rank"], hypC["d"])
    logger.info("  Rank at 0.01 * sigma_max: %d", hypC["rank_at_1pct_sigma_max"])
    for k, v in sorted(hypC["energy_in_topk"].items()):
        logger.info("  top-%4d energy fraction: %.4f", k, v)
    ns = hypC["neuron_coverage_stats"]
    logger.info("  Gate coverage: median=%.4f, p10=%.4f, p90=%.4f",
                ns["gate_median"], ns["gate_p10"], ns["gate_p90"])
    logger.info("  Up   coverage: median=%.4f, p10=%.4f, p90=%.4f",
                ns["up_median"], ns["up_p10"], ns["up_p90"])

    # ══════════════════════════════════════════════════════════════════════════
    # HYPOTHESIS A — W_down imperfection vs. oracle W_down
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("HYPOTHESIS A: W_down imperfection corrupts gate/up recovery")
    logger.info("=" * 72)

    # A.1 — recover W_down via Phase 1 OLS (as in v3)
    logger.info("  A.1: Phase 1 OLS for W_down (cold-start, as v3)...")
    t_a1 = time.time()
    W_down_rec, ols_metrics = solve_w_down_ols(
        h_mid_all, h_out_all, g_mlp, W_gate_true, W_up_true, device, BS,
    )
    w_down_rec_cos = flat_cosine(W_down_rec, W_down_true)
    w_down_rec_row = per_row_cosine(W_down_rec, W_down_true)
    phase1_time = time.time() - t_a1
    logger.info("    W_down_rec cos=%.4f (row=%.4f) time=%.1fs",
                w_down_rec_cos, w_down_rec_row, phase1_time)

    # A.2 — joint opt with W_down_rec
    logger.info("  A.2: joint (W_gate, W_up) opt w/ W_down_rec (cold start)...")
    t_a2 = time.time()
    Wg_rec_A, Wu_rec_A, log_A, init_A = joint_optimize_gate_up(
        X_all, r_mlp_all, W_down_rec,
        W_gate_true, W_up_true, device,
        total_steps=args.phase_a_steps,
        batch_size=args.opt_batch_size,
        peak_lr=args.lr,
        warmup_steps=args.lr_warmup_steps,
        weight_decay=args.weight_decay,
        init_scale=args.init_scale,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        seed=args.seed,
        tag="A-rec",
    )
    time_a2 = time.time() - t_a2

    cos_A_g = flat_cosine(Wg_rec_A, W_gate_true)
    cos_A_u = flat_cosine(Wu_rec_A, W_up_true)
    cos_A_g_row = per_row_cosine(Wg_rec_A, W_gate_true)
    cos_A_u_row = per_row_cosine(Wu_rec_A, W_up_true)

    # A.3 — joint opt with W_down_TRUE (ORACLE)
    logger.info("  A.3: joint (W_gate, W_up) opt w/ W_down_TRUE (ORACLE)...")
    t_a3 = time.time()
    Wg_rec_At, Wu_rec_At, log_At, init_At = joint_optimize_gate_up(
        X_all, r_mlp_all, W_down_true,
        W_gate_true, W_up_true, device,
        total_steps=args.phase_a_steps,
        batch_size=args.opt_batch_size,
        peak_lr=args.lr,
        warmup_steps=args.lr_warmup_steps,
        weight_decay=args.weight_decay,
        init_scale=args.init_scale,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        seed=args.seed,
        tag="A-oracle",
    )
    time_a3 = time.time() - t_a3

    cos_At_g = flat_cosine(Wg_rec_At, W_gate_true)
    cos_At_u = flat_cosine(Wu_rec_At, W_up_true)
    cos_At_g_row = per_row_cosine(Wg_rec_At, W_gate_true)
    cos_At_u_row = per_row_cosine(Wu_rec_At, W_up_true)

    results["hypothesis_A_w_down_imperfection"] = {
        "w_down_rec_cos_flat": w_down_rec_cos,
        "w_down_rec_cos_row": w_down_rec_row,
        "w_down_rec_frob_err": (W_down_rec - W_down_true).norm().item() /
                                (W_down_true.norm().item() + 1e-12),
        "phase1_time_seconds": round(phase1_time, 1),
        "ols_metrics": {
            k: (float(v) if hasattr(v, "item") or isinstance(v, (int, float)) else str(v))
            for k, v in ols_metrics.items()
        },
        "with_w_down_rec": {
            "cos_gate": cos_A_g,
            "cos_up": cos_A_u,
            "cos_gate_row": cos_A_g_row,
            "cos_up_row": cos_A_u_row,
            "final_loss_ema": log_A[-1]["loss_ema"] if log_A else None,
            "time_seconds": round(time_a2, 1),
            "init": init_A,
            "log_tail": log_A[-5:],
        },
        "with_w_down_TRUE_oracle": {
            "cos_gate": cos_At_g,
            "cos_up": cos_At_u,
            "cos_gate_row": cos_At_g_row,
            "cos_up_row": cos_At_u_row,
            "final_loss_ema": log_At[-1]["loss_ema"] if log_At else None,
            "time_seconds": round(time_a3, 1),
            "init": init_At,
            "log_tail": log_At[-5:],
        },
        "delta_cos_gate_oracle_vs_rec": cos_At_g - cos_A_g,
        "delta_cos_up_oracle_vs_rec": cos_At_u - cos_A_u,
    }

    logger.info("  A — Summary:")
    logger.info("    W_down_rec    → (g,u) = (%.3f, %.3f) row=(%.3f,%.3f)",
                cos_A_g, cos_A_u, cos_A_g_row, cos_A_u_row)
    logger.info("    W_down_ORACLE → (g,u) = (%.3f, %.3f) row=(%.3f,%.3f)",
                cos_At_g, cos_At_u, cos_At_g_row, cos_At_u_row)
    logger.info("    delta         =       (%+.3f, %+.3f)",
                cos_At_g - cos_A_g, cos_At_u - cos_A_u)

    # Save checkpoint weights
    torch.save({
        "W_down_rec": W_down_rec,
        "W_gate_A_rec_wdown": Wg_rec_A,
        "W_up_A_rec_wdown": Wu_rec_A,
        "W_gate_A_oracle_wdown": Wg_rec_At,
        "W_up_A_oracle_wdown": Wu_rec_At,
    }, out_dir / "hypA_weights.pt")

    # ══════════════════════════════════════════════════════════════════════════
    # HYPOTHESIS B — Warm-start basin sweep (with oracle W_down)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("HYPOTHESIS B: teacher basin — warm-start sweep (oracle W_down)")
    logger.info("=" * 72)

    alphas = [float(a) for a in args.warm_alphas.split(",") if a.strip()]
    logger.info("  alphas: %s", alphas)

    # Normalization constant for noise: ||W||_F / sqrt(numel) = per-entry RMS
    gate_rms = (W_gate_true.float().norm().item() /
                math.sqrt(W_gate_true.numel()))
    up_rms = (W_up_true.float().norm().item() /
              math.sqrt(W_up_true.numel()))
    logger.info("  noise scale: gate_rms=%.4e, up_rms=%.4e",
                gate_rms, up_rms)

    warm_results: list[dict] = []
    t_b = time.time()

    for alpha in alphas:
        logger.info("-" * 72)
        logger.info("  B.α = %.3f", alpha)
        logger.info("-" * 72)

        torch.manual_seed(args.seed + int(alpha * 1000) + 7)
        noise_g = torch.randn_like(W_gate_true) * gate_rms
        noise_u = torch.randn_like(W_up_true) * up_rms
        Wg0 = W_gate_true + alpha * noise_g
        Wu0 = W_up_true + alpha * noise_u

        # Pre-training cosine and loss
        with torch.no_grad():
            pre_cos_g = flat_cosine(Wg0, W_gate_true)
            pre_cos_u = flat_cosine(Wu0, W_up_true)
            pre_cos_g_row = per_row_cosine(Wg0, W_gate_true)
            pre_cos_u_row = per_row_cosine(Wu0, W_up_true)

            # loss on big probe batch
            probe_M = min(4096, M_total)
            idx = torch.arange(probe_M)
            x_probe = X_all[idx].to(device)
            r_probe = r_mlp_all[idx].to(device)
            Wg0d = Wg0.to(device)
            Wu0d = Wu0.to(device)
            Wd_dev = W_down_true.to(device)
            a_probe = F.silu(x_probe @ Wg0d.T) * (x_probe @ Wu0d.T)
            r_pred = a_probe @ Wd_dev.T
            pre_loss = F.mse_loss(r_pred, r_probe).item()
            pre_cos_recon = flat_cosine(r_pred.cpu(), r_probe.cpu())

        logger.info("    pre-training: cos(g,u)=(%.3f,%.3f) row=(%.3f,%.3f) "
                    "loss=%.4e recon_cos=%.4f",
                    pre_cos_g, pre_cos_u, pre_cos_g_row, pre_cos_u_row,
                    pre_loss, pre_cos_recon)

        # Optimize (using ORACLE W_down to isolate gate/up question)
        Wg_rec_B, Wu_rec_B, log_B, init_B = joint_optimize_gate_up(
            X_all, r_mlp_all, W_down_true,
            W_gate_true, W_up_true, device,
            total_steps=args.phase_b_steps,
            batch_size=args.opt_batch_size,
            peak_lr=args.lr,
            warmup_steps=min(args.lr_warmup_steps, max(50, args.phase_b_steps // 10)),
            weight_decay=args.weight_decay,
            init_scale=args.init_scale,
            grad_clip=args.grad_clip,
            eval_every=args.eval_every,
            seed=args.seed + int(alpha * 10_000) + 7,
            W_gate_init=Wg0,
            W_up_init=Wu0,
            tag=f"B-α{alpha:.3f}",
        )

        post_cos_g = flat_cosine(Wg_rec_B, W_gate_true)
        post_cos_u = flat_cosine(Wu_rec_B, W_up_true)
        post_cos_g_row = per_row_cosine(Wg_rec_B, W_gate_true)
        post_cos_u_row = per_row_cosine(Wu_rec_B, W_up_true)

        # Post-training recon check on held-out probe
        with torch.no_grad():
            hold_start = int(0.9 * M_total)
            X_hold = X_all[hold_start:hold_start + 4096].to(device)
            r_hold = r_mlp_all[hold_start:hold_start + 4096].to(device)
            a_post = (F.silu(X_hold @ Wg_rec_B.to(device).T) *
                      (X_hold @ Wu_rec_B.to(device).T))
            r_post = a_post @ W_down_true.to(device).T
            post_loss = F.mse_loss(r_post, r_hold).item()
            post_cos_recon = flat_cosine(r_post.cpu(), r_hold.cpu())

        warm_results.append({
            "alpha": alpha,
            "pre_cos_gate": pre_cos_g,
            "pre_cos_up": pre_cos_u,
            "pre_cos_gate_row": pre_cos_g_row,
            "pre_cos_up_row": pre_cos_u_row,
            "pre_loss": pre_loss,
            "pre_recon_cos": pre_cos_recon,
            "post_cos_gate": post_cos_g,
            "post_cos_up": post_cos_u,
            "post_cos_gate_row": post_cos_g_row,
            "post_cos_up_row": post_cos_u_row,
            "post_loss_ema": log_B[-1]["loss_ema"] if log_B else None,
            "post_recon_cos": post_cos_recon,
            "post_loss_heldout": post_loss,
            "delta_cos_gate": post_cos_g - pre_cos_g,
            "delta_cos_up": post_cos_u - pre_cos_u,
            "log_head": log_B[:3],
            "log_tail": log_B[-5:],
        })

        logger.info("    post: cos(g,u)=(%.3f,%.3f) row=(%.3f,%.3f)  "
                    "delta=(%+.3f,%+.3f)  loss_ema=%.4e",
                    post_cos_g, post_cos_u, post_cos_g_row, post_cos_u_row,
                    post_cos_g - pre_cos_g, post_cos_u - pre_cos_u,
                    log_B[-1]["loss_ema"] if log_B else 0.0)

    results["hypothesis_B_teacher_basin"] = {
        "alphas": alphas,
        "gate_noise_rms": gate_rms,
        "up_noise_rms": up_rms,
        "results": warm_results,
        "time_seconds": round(time.time() - t_b, 1),
    }

    # Plot warm-start sweep
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        xs = [w["alpha"] for w in warm_results]
        pre_g = [w["pre_cos_gate"] for w in warm_results]
        pre_u = [w["pre_cos_up"] for w in warm_results]
        post_g = [w["post_cos_gate"] for w in warm_results]
        post_u = [w["post_cos_up"] for w in warm_results]

        axes[0].plot(xs, pre_g, "o--", color="C0", alpha=0.6, label="pre gate")
        axes[0].plot(xs, post_g, "o-", color="C0", label="post gate")
        axes[0].plot(xs, pre_u, "s--", color="C1", alpha=0.6, label="pre up")
        axes[0].plot(xs, post_u, "s-", color="C1", label="post up")
        axes[0].axhline(1.0, color="k", linestyle=":", alpha=0.3)
        axes[0].set_xscale("symlog", linthresh=0.01)
        axes[0].set_xlabel(r"perturbation $\alpha$")
        axes[0].set_ylabel("cosine vs teacher")
        axes[0].set_title("Warm-start basin sweep (W_down = ORACLE)")
        axes[0].legend(loc="lower left", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        delta_g = [w["delta_cos_gate"] for w in warm_results]
        delta_u = [w["delta_cos_up"] for w in warm_results]
        axes[1].plot(xs, delta_g, "o-", color="C0", label="delta gate")
        axes[1].plot(xs, delta_u, "s-", color="C1", label="delta up")
        axes[1].axhline(0, color="k", linestyle=":", alpha=0.3)
        axes[1].set_xscale("symlog", linthresh=0.01)
        axes[1].set_xlabel(r"perturbation $\alpha$")
        axes[1].set_ylabel("post-cos − pre-cos")
        axes[1].set_title(r"Training effect: $\Delta$cos (>0 improves, <0 degrades)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        warm_plot_path = out_dir / "hypB_warm_start_sweep.png"
        plt.savefig(warm_plot_path, dpi=140)
        plt.close(fig)
        results["hypothesis_B_teacher_basin"]["sweep_plot"] = str(warm_plot_path)
        logger.info("  Warm-start plot saved: %s", warm_plot_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("  Warm-start plot failed: %s", e)

    # ══════════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("SYNTHESIS")
    logger.info("=" * 72)

    hypA = results["hypothesis_A_w_down_imperfection"]
    hypB = results["hypothesis_B_teacher_basin"]
    hypC = results["hypothesis_C_data_rank"]

    hypA_oracle_best = max(
        hypA["with_w_down_TRUE_oracle"]["cos_gate"],
        hypA["with_w_down_TRUE_oracle"]["cos_up"],
    )
    hypA_rec_best = max(
        hypA["with_w_down_rec"]["cos_gate"],
        hypA["with_w_down_rec"]["cos_up"],
    )
    hypA_delta = hypA_oracle_best - hypA_rec_best

    # Verdict logic
    verdict: dict = {
        "hypothesis_A_confirmed": hypA_oracle_best > 0.5 and hypA_delta > 0.2,
        "hypothesis_A_weak_support": 0.05 < hypA_delta <= 0.2,
        "hypothesis_A_rejected": hypA_delta <= 0.05,
    }

    # Hypothesis B confirmed if training DEGRADES warm-start cosines
    # (i.e. teacher is not a basin).
    basin_preserved = True
    basin_degrades_small_noise = False
    basin_finds_basin_from_cold = False
    for w in hypB["results"]:
        a = w["alpha"]
        # Small-noise case (alpha <= 0.1): expect near-perfect recovery
        if a <= 0.1:
            if w["post_cos_gate"] < 0.8 or w["post_cos_up"] < 0.8:
                basin_preserved = False
            if w["pre_cos_gate"] > w["post_cos_gate"] + 0.05:
                basin_degrades_small_noise = True
            if w["pre_cos_up"] > w["post_cos_up"] + 0.05:
                basin_degrades_small_noise = True
        # Cold-start case (alpha ~ 0): compare to hypA cold-start
        if a == 0.0:
            # teacher init: should stay at cos=1.0
            if w["post_cos_gate"] < 0.98 or w["post_cos_up"] < 0.98:
                basin_degrades_small_noise = True

    verdict["hypothesis_B_basin_preserved_small_noise"] = basin_preserved
    verdict["hypothesis_B_basin_degrades_small_noise"] = (
        basin_degrades_small_noise
    )
    verdict["hypothesis_B_confirmed"] = basin_degrades_small_noise
    verdict["hypothesis_B_rejected"] = basin_preserved

    # Hypothesis C: data rank limited if eff_rank << d
    eff_rank_frac = hypC["effective_rank"] / hypC["d"]
    energy_top128 = hypC["energy_in_topk"].get(128, 1.0)
    verdict["hypothesis_C_effective_rank_fraction"] = eff_rank_frac
    verdict["hypothesis_C_energy_top128"] = energy_top128
    verdict["hypothesis_C_confirmed"] = (
        eff_rank_frac < 0.5 or energy_top128 > 0.95
    )
    verdict["hypothesis_C_weak_support"] = (
        0.5 <= eff_rank_frac < 0.8 or 0.9 <= energy_top128 <= 0.95
    )

    results["verdict"] = verdict

    logger.info("  HypA metrics:")
    logger.info("    oracle W_down → max(g,u) cos = %.3f", hypA_oracle_best)
    logger.info("    rec W_down    → max(g,u) cos = %.3f", hypA_rec_best)
    logger.info("    delta         = %+.3f", hypA_delta)
    logger.info("  HypB basin preserved (small α): %s", basin_preserved)
    logger.info("  HypB degrades small-noise warm: %s",
                basin_degrades_small_noise)
    logger.info("  HypC effective_rank / d = %.3f", eff_rank_frac)
    logger.info("  HypC top-128 energy     = %.3f", energy_top128)

    # ── Print final synthesis banner ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Phase 2 Failure Mode Diagnostic — SYNTHESIS")
    print("=" * 80)
    print()
    print("Hypothesis A — W_down imperfection corrupts gate/up targets")
    print(f"  with W_down_rec (Phase 1 OLS, cos={hypA['w_down_rec_cos_flat']:.3f}):")
    print(f"    cos(gate) = {hypA['with_w_down_rec']['cos_gate']:.4f}  "
          f"(row {hypA['with_w_down_rec']['cos_gate_row']:.4f})")
    print(f"    cos(up)   = {hypA['with_w_down_rec']['cos_up']:.4f}  "
          f"(row {hypA['with_w_down_rec']['cos_up_row']:.4f})")
    print(f"  with W_down_TRUE (ORACLE, cos=1.000):")
    print(f"    cos(gate) = {hypA['with_w_down_TRUE_oracle']['cos_gate']:.4f}  "
          f"(row {hypA['with_w_down_TRUE_oracle']['cos_gate_row']:.4f})")
    print(f"    cos(up)   = {hypA['with_w_down_TRUE_oracle']['cos_up']:.4f}  "
          f"(row {hypA['with_w_down_TRUE_oracle']['cos_up_row']:.4f})")
    print(f"  delta(oracle - rec) = "
          f"({hypA['delta_cos_gate_oracle_vs_rec']:+.3f}, "
          f"{hypA['delta_cos_up_oracle_vs_rec']:+.3f})")
    print()
    print("Hypothesis B — Teacher basin (with ORACLE W_down)")
    for w in hypB["results"]:
        print(f"  α={w['alpha']:6.3f}: "
              f"pre=({w['pre_cos_gate']:+.3f},{w['pre_cos_up']:+.3f}) "
              f"post=({w['post_cos_gate']:+.3f},{w['post_cos_up']:+.3f}) "
              f"Δ=({w['delta_cos_gate']:+.3f},{w['delta_cos_up']:+.3f}) "
              f"loss_ema={w['post_loss_ema']:.2e}")
    print()
    print("Hypothesis C — Data rank of X = RMSNorm(h_mid)")
    print(f"  Shape: M={hypC['M']}  d={hypC['d']}")
    print(f"  Effective rank (tr^2 / ||·||_F^2): "
          f"{hypC['effective_rank']:.2f} / {hypC['d']} "
          f"(= {eff_rank_frac:.3f})")
    print(f"  Rank at 0.01 * sigma_max:         "
          f"{hypC['rank_at_1pct_sigma_max']}")
    for k in sorted(hypC["energy_in_topk"].keys()):
        print(f"  top-{k:4d} energy fraction:         "
              f"{hypC['energy_in_topk'][k]:.4f}")
    ns = hypC["neuron_coverage_stats"]
    print(f"  per-neuron coverage (gate): "
          f"median={ns['gate_median']:.3f} "
          f"p10={ns['gate_p10']:.3f} p90={ns['gate_p90']:.3f}")
    print(f"  per-neuron coverage (up):   "
          f"median={ns['up_median']:.3f} "
          f"p10={ns['up_p10']:.3f} p90={ns['up_p90']:.3f}")
    print()
    print("-" * 80)
    print("Verdict")
    print("-" * 80)
    if verdict["hypothesis_A_confirmed"]:
        print("  [A] CONFIRMED: oracle W_down lifts cos by ≥0.2 and to >0.5.")
        print("      Fix: invest in higher-quality W_down (richer data or more iter.).")
    elif verdict["hypothesis_A_weak_support"]:
        print("  [A] PARTIAL: oracle W_down helps (Δ ∈ (0.05, 0.2]) but not decisive.")
        print("      Fix: W_down is part of the story — combine with B/C fixes.")
    else:
        print("  [A] REJECTED: oracle W_down barely changes cos.")
        print("      W_down quality is NOT the bottleneck.")

    if verdict["hypothesis_B_confirmed"]:
        print("  [B] CONFIRMED: training degrades small-noise warm-start.")
        print("      Teacher is NOT a local minimum of this loss.")
        print("      Fix: add regularization, change loss (e.g. featurewise CCA),")
        print("            or exploit gauge symmetry explicitly.")
    elif verdict["hypothesis_B_rejected"]:
        print("  [B] REJECTED: training preserves small-noise warm-start.")
        print("      Teacher IS a basin; the failure is in reaching it from cold start.")
        print("      Fix: better initialization (e.g., algebraic init) or curriculum.")
    else:
        print("  [B] INCONCLUSIVE: warm-start results are mixed.")

    if verdict["hypothesis_C_confirmed"]:
        print("  [C] CONFIRMED: data rank << d_model OR top-128 energy > 95%.")
        print("      Fix: richer queries (diverse topics, longer contexts, active)")
        print("            or accept recovery only on data subspace.")
    elif verdict["hypothesis_C_weak_support"]:
        print("  [C] PARTIAL: data subspace is somewhat concentrated.")
    else:
        print("  [C] REJECTED: X spans the full 896-dim space.")
        print("      Data coverage is sufficient.")

    print()
    print(f"Results JSON : {out_dir / 'diagnose_results.json'}")
    print(f"Weights      : {out_dir / 'hypA_weights.pt'}")
    print(f"Plots        : {out_dir}/hypC_spectrum.png, "
          f"hypC_neuron_coverage.png, hypB_warm_start_sweep.png")
    print("=" * 80)

    # Save JSON
    def _jsonify(o):
        if isinstance(o, (bool, int, float, str, type(None))):
            return o
        if isinstance(o, torch.Tensor):
            return float(o.item()) if o.numel() == 1 else o.tolist()
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

    with open(out_dir / "diagnose_results.json", "w") as f:
        json.dump(_jsonify(results), f, indent=2)

    logger.info("Diagnostic complete. Artifacts in %s", out_dir)


if __name__ == "__main__":
    main()
