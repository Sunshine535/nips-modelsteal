#!/usr/bin/env python3
# SAFETY NOTICE: QUARANTINED (alpha-theory prune 2026-04-19)
# This script is NOT cited in the paper. It was part of killed branches:
#   A1 S-PSI, A2 Moments CP, A4 logit-bias, A5 memory probing, A6 active query,
#   A7 algebraic v2/v3/v4, B3 matched-KD.
# Retained in repo for reproducibility of quarantined history; do not use for
# new claims.
#!/usr/bin/env python3
"""
Algebraic Block Recovery v2 — Improved structured decomposition.

Fixes two key issues from v1:
  1. W_down cos=0.618 should be ~1.0: float64 accumulation, minimal ridge,
     all data points (no subsampling).
  2. ALS for gate/up diverged (loss=7e11): replace with per-neuron target
     decomposition after recovering W_down.

Pipeline:
  Step 0: Recover h_out from logits via pinv(W_lm) and RMS re-scaling.
  Phase 1: W_down via high-precision OLS (oracle activations, float64, all data).
  Phase 2: W_gate/W_up via per-neuron target activation decomposition.
  Phase 3: Query scaling ablation (sweep N).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/algebraic_recovery_v2.py \
        --model_name Qwen/Qwen2.5-0.5B --block_idx 23 \
        --output_dir results/v5_algebraic_v2 --seed 42
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.permutation_alignment import align_ffn_neurons

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
D_MODEL = 896
D_FF = 4864
VOCAB_SIZE = 151936
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64


# ── helpers ──────────────────────────────────────────────────────────────────

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Algebraic Block Recovery v2")
    p.add_argument("--model_name", type=str, default=MODEL_NAME)
    p.add_argument("--block_idx", type=int, default=23)
    p.add_argument("--num_queries", type=int, default=2048)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--output_dir", type=str, default="results/v5_algebraic_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for data collection and OLS accumulation")
    p.add_argument("--allow_synthetic", action="store_true",
                   help="Fall back to random tokens if dataset load fails")
    # Phase 2 parameters
    p.add_argument("--neuron_als_iters", type=int, default=30,
                   help="Per-neuron ALS iterations")
    p.add_argument("--neuron_lbfgs_steps", type=int, default=20,
                   help="L-BFGS steps for gate optimization per neuron ALS iter")
    p.add_argument("--neuron_batch_size", type=int, default=64,
                   help="Number of neurons to process in parallel")
    # Phase 3 ablation
    p.add_argument("--ablation_sizes", type=str, default="256,512,1024,2048",
                   help="Comma-separated query counts for scaling ablation")
    p.add_argument("--skip_ablation", action="store_true",
                   help="Skip Phase 3 ablation")
    return p.parse_args()


# ── data loading ─────────────────────────────────────────────────────────────

def build_query_pool(tokenizer, pool_size: int, max_seq_len: int, seed: int,
                     allow_synthetic: bool = False) -> torch.Tensor:
    """Build query input_ids from WikiText (or random tokens as fallback)."""
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
    except Exception as e:
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


# ── RMSNorm (matches Qwen2RMSNorm) ──────────────────────────────────────────

def rms_norm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: x / rms(x) * g,  rms(x) = sqrt(mean(x^2) + eps)."""
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms * g.float()).to(x.dtype)


# ── cosine helpers ───────────────────────────────────────────────────────────

def flat_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened)."""
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def per_row_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-row cosine similarity."""
    a_f = a.float()
    b_f = b.float()
    if a_f.dim() == 1:
        return flat_cosine(a_f, b_f)
    cos = F.cosine_similarity(a_f, b_f, dim=-1)
    return cos.mean().item()


# ── Phase 1: High-precision OLS for W_down ──────────────────────────────────

def solve_w_down_ols(
    h_mid_all: torch.Tensor,    # [N, T, d]
    h_out_all: torch.Tensor,    # [N, T, d]
    g_mlp: torch.Tensor,        # [d]
    W_gate_true: torch.Tensor,  # [d_ff, d]
    W_up_true: torch.Tensor,    # [d_ff, d]
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, dict]:
    """Solve W_down via OLS using true gate/up activations.

    Accumulates normal equations in float64 for maximum precision.
    Uses ALL data points (no subsampling).
    Ridge is auto-tuned from condition number.

    Returns: (W_down_recovered [d, d_ff] on CPU, metrics dict)
    """
    N, T, d = h_mid_all.shape
    d_ff = W_gate_true.shape[0]
    M = N * T

    g_mlp_dev = g_mlp.to(device)
    W_gate_dev = W_gate_true.to(device)
    W_up_dev = W_up_true.to(device)

    # Accumulate normal equations in float64
    # System: W_down @ A = R  where A=[a_1,...,a_M] each [d_ff], R=[r_1,...,r_M] each [d]
    # Normal equations: W_down @ (A A^T) = R A^T
    AAt = torch.zeros(d_ff, d_ff, dtype=torch.float64, device=device)
    RAt = torch.zeros(d, d_ff, dtype=torch.float64, device=device)

    logger.info("  Accumulating normal equations (float64, M=%d points)...", M)
    with torch.no_grad():
        for n_start in range(0, N, batch_size):
            n_end = min(n_start + batch_size, N)
            h_mid_b = h_mid_all[n_start:n_end].to(device)  # [bs, T, d]
            h_out_b = h_out_all[n_start:n_end].to(device)  # [bs, T, d]

            M_b = h_mid_b.shape[0] * T
            h_mid_flat = h_mid_b.reshape(M_b, d)
            h_out_flat = h_out_b.reshape(M_b, d)

            # x = RMSNorm(h_mid, g_mlp)
            x = rms_norm(h_mid_flat, g_mlp_dev)  # [M_b, d]

            # True activations
            gate_out = F.silu(x @ W_gate_dev.T)  # [M_b, d_ff]
            up_out = x @ W_up_dev.T               # [M_b, d_ff]
            a = gate_out * up_out                  # [M_b, d_ff]

            # MLP residual
            r = (h_out_flat - h_mid_flat).float()  # [M_b, d]

            # Accumulate in float64
            a_64 = a.double()
            r_64 = r.double()
            AAt += a_64.T @ a_64   # [d_ff, d_ff]
            RAt += r_64.T @ a_64   # [d, d_ff]

            if (n_start // batch_size) % 10 == 0:
                logger.info("    Batch %d/%d", n_start // batch_size + 1,
                            (N + batch_size - 1) // batch_size)

    # Compute condition number for auto-ridge
    logger.info("  Computing condition number...")
    diag_mean = AAt.diagonal().mean().item()
    try:
        eigvals = torch.linalg.eigvalsh(AAt)
        cond = (eigvals[-1] / eigvals[0].clamp(min=1e-30)).item()
        logger.info("  Condition number: %.2e (min_eig=%.2e, max_eig=%.2e)",
                    cond, eigvals[0].item(), eigvals[-1].item())
        # Auto ridge: if well-conditioned, use very small; if ill-conditioned, use more
        if cond < 1e8:
            ridge = 1e-10 * diag_mean
            logger.info("  Well-conditioned: ridge=%.2e (1e-10 * diag_mean)", ridge)
        elif cond < 1e12:
            ridge = 1e-8 * diag_mean
            logger.info("  Moderate conditioning: ridge=%.2e (1e-8 * diag_mean)", ridge)
        else:
            ridge = 1e-6 * diag_mean
            logger.info("  Ill-conditioned: ridge=%.2e (1e-6 * diag_mean)", ridge)
    except Exception as e:
        logger.warning("  Eigenvalue computation failed (%s), using default ridge", e)
        ridge = 1e-8 * diag_mean
        cond = -1.0

    AAt += ridge * torch.eye(d_ff, dtype=torch.float64, device=device)

    # Solve via Cholesky (most stable for SPD systems)
    logger.info("  Solving OLS (%dx%d) via Cholesky...", d, d_ff)
    try:
        L = torch.linalg.cholesky(AAt)
        # W_down = RAt @ inv(AAt) = solve(AAt, RAt^T)^T
        W_down_rec = torch.linalg.solve_triangular(
            L.T,
            torch.linalg.solve_triangular(L, RAt.T, upper=False),
            upper=True,
        ).T.float()
    except RuntimeError as e:
        logger.warning("  Cholesky failed (%s), using torch.linalg.solve", e)
        W_down_rec = torch.linalg.solve(AAt, RAt.T).T.float()

    metrics = {
        "condition_number": cond,
        "ridge": ridge,
        "diag_mean": diag_mean,
        "num_data_points": M,
    }

    return W_down_rec.cpu(), metrics


# ── Phase 2: Per-neuron decomposition for gate/up ───────────────────────────

def compute_target_activations(
    W_down_rec: torch.Tensor,  # [d, d_ff] recovered W_down
    r_mlp: torch.Tensor,       # [M, d] MLP residuals
    device: torch.device,
) -> torch.Tensor:
    """Compute target activations via pseudo-inverse of W_down.

    For fat matrix W_down [d, d_ff] (d < d_ff):
      pinv(W_down) = W_down^T @ inv(W_down @ W_down^T)  [d_ff, d]
      a_target = pinv(W_down) @ r  (minimum-norm solution)

    Returns: a_target [M, d_ff] on CPU
    """
    d, d_ff = W_down_rec.shape
    W = W_down_rec.double().to(device)

    # Gram matrix G = W @ W^T [d, d], small and easy to invert
    G = W @ W.T  # [d, d]
    # Add small ridge for numerical stability
    G += 1e-10 * G.diagonal().mean() * torch.eye(d, dtype=torch.float64, device=device)
    G_inv = torch.linalg.inv(G)  # [d, d]

    # pinv(W) = W^T @ G_inv  [d_ff, d]
    pinv_W = W.T @ G_inv  # [d_ff, d]

    logger.info("  Computing target activations (M=%d)...", r_mlp.shape[0])

    # Process in chunks to manage memory
    M = r_mlp.shape[0]
    chunk_size = 8192
    a_target_list = []

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        r_chunk = r_mlp[start:end].double().to(device)  # [chunk, d]
        a_chunk = r_chunk @ pinv_W.T  # [chunk, d_ff]
        a_target_list.append(a_chunk.float().cpu())

    return torch.cat(a_target_list, dim=0)  # [M, d_ff]


def recover_gate_up_per_neuron(
    a_target: torch.Tensor,     # [M, d_ff] target activations
    X: torch.Tensor,            # [M, d] normalized MLP inputs
    W_gate_true: torch.Tensor,  # [d_ff, d] for init/eval
    W_up_true: torch.Tensor,    # [d_ff, d] for init/eval
    device: torch.device,
    als_iters: int = 30,
    lbfgs_steps: int = 20,
    neuron_batch_size: int = 64,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Recover W_gate and W_up via per-neuron ALS.

    For each neuron n:
      a_target[n, i] = SiLU(w_gate_n^T @ x_i) * (w_up_n^T @ x_i)

    ALS alternates:
      1. Fix w_gate -> solve w_up via linear regression (after dividing by SiLU)
      2. Fix w_up -> optimize w_gate via L-BFGS (nonlinear, 896 params, fast)

    Returns: (W_gate_rec [d_ff, d], W_up_rec [d_ff, d], metrics)
    """
    M, d_ff = a_target.shape
    _, d = X.shape
    torch.manual_seed(seed + 500)

    W_gate_rec = torch.zeros(d_ff, d)
    W_up_rec = torch.zeros(d_ff, d)

    # Precompute X^T X for linear regression (shared across all neurons)
    X_dev = X.float().to(device)
    XtX = X_dev.double().T @ X_dev.double()  # [d, d]
    XtX_ridge = XtX + 1e-8 * XtX.diagonal().mean() * torch.eye(
        d, dtype=torch.float64, device=device
    )
    XtX_inv = torch.linalg.inv(XtX_ridge)  # [d, d]

    total_loss_initial = 0.0
    total_loss_final = 0.0
    per_neuron_cos_gate = []
    per_neuron_cos_up = []

    num_batches = (d_ff + neuron_batch_size - 1) // neuron_batch_size
    logger.info("  Per-neuron decomposition: %d neurons, %d ALS iters, "
                "%d L-BFGS steps, batch=%d",
                d_ff, als_iters, lbfgs_steps, neuron_batch_size)

    t0 = time.time()

    for batch_start in range(0, d_ff, neuron_batch_size):
        batch_end = min(batch_start + neuron_batch_size, d_ff)
        bn = batch_end - batch_start
        batch_idx = batch_start // neuron_batch_size

        # Target activations for this batch of neurons: [M, bn]
        a_batch = a_target[:, batch_start:batch_end].float().to(device)

        # Initialize w_gate from small random (could also try warm-start from OLS)
        w_gate_batch = torch.randn(bn, d, device=device) * 0.02
        w_up_batch = torch.randn(bn, d, device=device) * 0.02

        # Better init: least-squares fit ignoring the nonlinearity
        # a_target[n,i] ~ (w_gate_n^T x_i) * (w_up_n^T x_i)
        # Linearize: if we assume gate ~ up, then a ~ (w^T x)^2
        # Or just use a linear fit: a_target[n,:] ~ w_n^T X^T => w_n = (X^T X)^-1 X^T a
        with torch.no_grad():
            # Initial linear approximation for w_up (ignoring SiLU gating)
            # a ≈ w_up^T x  =>  w_up = (X^T X)^-1 X^T a
            w_up_init = (XtX_inv @ X_dev.double().T @ a_batch.double()).T.float()  # [bn, d]
            w_up_batch = w_up_init.clone()

            # Then init gate from sign pattern
            u_init = X_dev @ w_up_batch.T  # [M, bn]
            # Where u > 0, gate should be positive (SiLU(g) > 0 => g > ~-1)
            # Where a_target / u is large, gate should allow it through
            safe_u = u_init.clone()
            safe_u[safe_u.abs() < 1e-6] = 1e-6
            silu_target = a_batch / safe_u  # approx SiLU(g) target
            # Invert SiLU approximately: for SiLU(g) = g*sigma(g), when g >> 0: SiLU(g)~g
            # Use the linear fit: g ~ X @ w_gate => solve w_gate
            g_approx = silu_target.clone()  # rough approximation
            # Clamp to avoid extreme values
            g_approx = g_approx.clamp(-10, 10)
            w_gate_init = (XtX_inv @ X_dev.double().T @ g_approx.double()).T.float()
            w_gate_batch = w_gate_init.clone()

        for als_iter in range(als_iters):
            # === Step 1: Fix w_gate, solve w_up (linear) ===
            with torch.no_grad():
                g = X_dev @ w_gate_batch.T  # [M, bn]
                s = F.silu(g)               # [M, bn]

                # u_target[i] = a_target[i] / SiLU(g[i])  where SiLU(g) != 0
                # Then: w_up^T x = u_target => w_up = (X^T X)^-1 X^T u_target
                # Mask out small SiLU values to avoid division instability
                s_abs = s.abs()
                mask = s_abs > 1e-4  # [M, bn] boolean

                # For masked-out points, use interpolated target
                u_target = torch.zeros_like(a_batch)
                u_target[mask] = a_batch[mask] / s[mask]

                # For points where SiLU is too small, use the linear approx
                # (these contribute little anyway)
                not_mask = ~mask
                if not_mask.any():
                    u_target[not_mask] = 0.0  # zero contribution

                # Weighted least squares: weight by s_abs to downweight bad points
                # w_up_n = (X^T diag(w) X)^-1 X^T diag(w) u_target
                # For simplicity, use uniform weights but only on masked points
                # Solve column by column, or use the precomputed inverse
                # Simple: w_up = (X^T X)^-1 X^T u_target  (since mask is per-neuron,
                # the per-neuron weighted version is expensive; uniform approx is OK
                # given M >> d)
                w_up_batch = (XtX_inv @ X_dev.double().T @ u_target.double()).T.float()

            # === Step 2: Fix w_up, optimize w_gate via L-BFGS ===
            with torch.no_grad():
                u = X_dev @ w_up_batch.T  # [M, bn] — fixed

            # Optimize w_gate to minimize:
            #   sum_i (SiLU(x_i^T w_gate_n) * u[n,i] - a_target[n,i])^2
            # This is 896 params per neuron, massively overdetermined => L-BFGS

            # Use Adam (simpler than LBFGS, handles non-contiguous grads)
            w_gate_param = w_gate_batch.clone().detach().contiguous().requires_grad_(True)
            optimizer = torch.optim.Adam([w_gate_param], lr=0.05)

            # Subsample for L-BFGS if M is very large (L-BFGS is full-batch anyway)
            M_sub = min(M, 65536)
            if M_sub < M:
                sub_idx = torch.randperm(M, device=device)[:M_sub]
                X_sub = X_dev[sub_idx]
                u_sub = u[sub_idx]
                a_sub = a_batch[sub_idx]
            else:
                X_sub = X_dev
                u_sub = u
                a_sub = a_batch

            # Adam optimization loop for W_gate
            for step_i in range(lbfgs_steps * 10):  # more steps than LBFGS since simpler optimizer
                optimizer.zero_grad()
                g_opt = X_sub @ w_gate_param.T  # [M_sub, bn]
                s_opt = F.silu(g_opt)           # [M_sub, bn]
                pred = s_opt * u_sub            # [M_sub, bn]
                loss = ((pred - a_sub) ** 2).mean()
                loss.backward()
                optimizer.step()
            w_gate_batch = w_gate_param.detach().contiguous()

        # Record per-neuron results
        with torch.no_grad():
            g_final = X_dev @ w_gate_batch.T
            s_final = F.silu(g_final)
            u_final = X_dev @ w_up_batch.T
            pred_final = s_final * u_final
            batch_loss = ((pred_final - a_batch) ** 2).mean().item()
            total_loss_final += batch_loss * bn

            # Per-neuron cosine with true weights
            for j in range(bn):
                n = batch_start + j
                cg = F.cosine_similarity(
                    w_gate_batch[j:j+1], W_gate_true[n:n+1].to(device)
                ).item()
                cu = F.cosine_similarity(
                    w_up_batch[j:j+1], W_up_true[n:n+1].to(device)
                ).item()
                per_neuron_cos_gate.append(cg)
                per_neuron_cos_up.append(cu)

        W_gate_rec[batch_start:batch_end] = w_gate_batch.cpu()
        W_up_rec[batch_start:batch_end] = w_up_batch.cpu()

        if (batch_idx + 1) % max(1, num_batches // 10) == 0 or batch_idx == 0:
            elapsed = time.time() - t0
            eta = elapsed / (batch_end / d_ff) - elapsed if batch_end < d_ff else 0
            mean_cg = sum(per_neuron_cos_gate[-bn:]) / bn
            mean_cu = sum(per_neuron_cos_up[-bn:]) / bn
            logger.info(
                "    Neurons %d-%d/%d | batch_loss=%.6f | "
                "mean_cos gate=%.4f up=%.4f | %.1fs (ETA %.0fs)",
                batch_start, batch_end, d_ff, batch_loss,
                mean_cg, mean_cu, elapsed, eta,
            )

    total_loss_final /= d_ff

    # Summary statistics
    cos_gate_arr = torch.tensor(per_neuron_cos_gate)
    cos_up_arr = torch.tensor(per_neuron_cos_up)

    # Some neurons may have flipped sign (cos ~ -1). Check for that.
    # If a neuron has cos_gate < 0, it might mean the sign is absorbed into w_up
    abs_cos_gate = cos_gate_arr.abs()
    abs_cos_up = cos_up_arr.abs()

    metrics = {
        "total_loss_final": total_loss_final,
        "per_neuron_cos_gate_mean": cos_gate_arr.mean().item(),
        "per_neuron_cos_gate_median": cos_gate_arr.median().item(),
        "per_neuron_cos_gate_abs_mean": abs_cos_gate.mean().item(),
        "per_neuron_cos_up_mean": cos_up_arr.mean().item(),
        "per_neuron_cos_up_median": cos_up_arr.median().item(),
        "per_neuron_cos_up_abs_mean": abs_cos_up.mean().item(),
        "per_neuron_cos_gate_gt09": (abs_cos_gate > 0.9).float().mean().item(),
        "per_neuron_cos_up_gt09": (abs_cos_up > 0.9).float().mean().item(),
    }

    return W_gate_rec, W_up_rec, metrics


# ── Phase 1 with variable N (for ablation) ──────────────────────────────────

def run_phase1_ablation(
    h_mid_all: torch.Tensor,
    h_out_all: torch.Tensor,
    g_mlp: torch.Tensor,
    W_gate_true: torch.Tensor,
    W_up_true: torch.Tensor,
    W_down_true: torch.Tensor,
    device: torch.device,
    N_values: list[int],
    T: int,
    batch_size: int,
    seed: int,
) -> list[dict]:
    """Run Phase 1 (W_down OLS) for different numbers of queries."""
    results = []
    N_total = h_mid_all.shape[0]

    for N_sub in N_values:
        if N_sub > N_total:
            logger.warning("  Skipping N=%d (only %d available)", N_sub, N_total)
            continue

        logger.info("  --- Ablation: N=%d (M=%d data points) ---", N_sub, N_sub * T)

        # Subsample queries (use first N_sub for deterministic subset)
        torch.manual_seed(seed + 999)
        idx = torch.randperm(N_total)[:N_sub]
        idx = idx.sort().values  # keep order for reproducibility

        h_mid_sub = h_mid_all[idx]
        h_out_sub = h_out_all[idx]

        W_down_rec, ols_metrics = solve_w_down_ols(
            h_mid_sub, h_out_sub, g_mlp,
            W_gate_true, W_up_true,
            device, batch_size,
        )

        cos_flat = flat_cosine(W_down_rec, W_down_true)
        cos_row = per_row_cosine(W_down_rec, W_down_true)
        frob_err = (W_down_rec - W_down_true).norm().item() / (
            W_down_true.norm().item() + 1e-12
        )

        result = {
            "N": N_sub,
            "M": N_sub * T,
            "W_down_cosine": cos_flat,
            "W_down_per_row_cosine": cos_row,
            "W_down_frob_error": frob_err,
            **ols_metrics,
        }
        results.append(result)

        logger.info("    N=%d: cos=%.6f, per_row_cos=%.6f, frob=%.6f, cond=%.2e",
                    N_sub, cos_flat, cos_row, frob_err,
                    ols_metrics.get("condition_number", -1))

    return results


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

    logger.info("=" * 70)
    logger.info("Algebraic Block Recovery v2")
    logger.info("=" * 70)
    logger.info("Model : %s", args.model_name)
    logger.info("Block : %d", block_idx)
    logger.info("Queries: %d x %d tokens = %d data points", N, T, N * T)
    logger.info("Device : %s", device)

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Load teacher ─────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32,
        device_map={"": device}, trust_remote_code=True,
    )
    teacher.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Extract teacher's true parameters ────────────────────────────────────
    prefix = f"model.layers.{block_idx}"
    true_params: dict[str, torch.Tensor] = {}
    for name, param in teacher.named_parameters():
        if name.startswith(prefix) or name in ("lm_head.weight", "model.norm.weight"):
            true_params[name] = param.data.cpu().clone()

    W_lm = teacher.lm_head.weight.data.float().to(device)         # [V, d]
    g_final = teacher.model.norm.weight.data.float().to(device)    # [d]
    g_mlp = true_params[f"{prefix}.post_attention_layernorm.weight"].float()  # [d]
    W_gate_true = true_params[f"{prefix}.mlp.gate_proj.weight"].float()  # [d_ff, d]
    W_up_true = true_params[f"{prefix}.mlp.up_proj.weight"].float()      # [d_ff, d]
    W_down_true = true_params[f"{prefix}.mlp.down_proj.weight"].float()  # [d, d_ff]

    logger.info("Teacher parameters extracted.")
    logger.info("  W_gate: %s, W_up: %s, W_down: %s",
                W_gate_true.shape, W_up_true.shape, W_down_true.shape)

    # ── Build query pool ─────────────────────────────────────────────────────
    logger.info("Building query pool (%d queries)...", N)
    query_ids = build_query_pool(
        tokenizer, N, T, args.seed, allow_synthetic=args.allow_synthetic,
    )
    logger.info("Query pool ready: %s", query_ids.shape)

    # ── Collect hidden states via hooks ──────────────────────────────────────
    logger.info("Collecting hidden states and logits...")

    all_h_in = []
    all_h_mid = []
    all_h_out = []
    all_logits = []

    target_block = teacher.model.layers[block_idx]
    attn_outputs_buffer: list[torch.Tensor] = []

    def attn_output_hook(module, inputs, output):
        attn_out = output[0] if isinstance(output, tuple) else output
        attn_outputs_buffer.append(attn_out.detach().cpu().float())

    hook_handle = target_block.self_attn.register_forward_hook(attn_output_hook)

    with torch.no_grad():
        for start in range(0, N, BS):
            end = min(start + BS, N)
            batch = query_ids[start:end].to(device)

            attn_outputs_buffer.clear()
            outputs = teacher(batch, output_hidden_states=True, return_dict=True)

            hs = outputs.hidden_states
            h_in_batch = hs[block_idx].detach().cpu().float()
            h_out_batch = hs[block_idx + 1].detach().cpu().float()
            attn_out_batch = attn_outputs_buffer[0]
            h_mid_batch = h_in_batch + attn_out_batch

            logits_batch = outputs.logits.detach().cpu().float()

            all_h_in.append(h_in_batch)
            all_h_mid.append(h_mid_batch)
            all_h_out.append(h_out_batch)
            all_logits.append(logits_batch)

            if (start // BS) % 10 == 0:
                logger.info("  Collected %d / %d queries", end, N)

    hook_handle.remove()

    h_in_all = torch.cat(all_h_in, dim=0)     # [N, T, d]
    h_mid_all = torch.cat(all_h_mid, dim=0)   # [N, T, d]
    h_out_all = torch.cat(all_h_out, dim=0)   # [N, T, d]
    logits_all = torch.cat(all_logits, dim=0)  # [N, T, V]

    logger.info("Hidden states collected:")
    logger.info("  h_in:   %s", h_in_all.shape)
    logger.info("  h_mid:  %s", h_mid_all.shape)
    logger.info("  h_out:  %s", h_out_all.shape)
    logger.info("  logits: %s", logits_all.shape)

    del all_h_in, all_h_mid, all_h_out, all_logits
    torch.cuda.empty_cache()

    results: dict = {}
    t_global = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 0: Recover h_out from logits (same as v1)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 0: Recover h_out from logits via pinv(W_lm)")
    logger.info("=" * 70)

    # NO DATA LEAKAGE: Carlini's attack recovers W_lm @ diag(g_final) JOINTLY,
    # not W_lm and g_final separately. We use the joint form W_eff = W_lm @ diag(g_final).
    # Then z = W_eff @ (h_out / rms(h_out)), so u = pinv(W_eff) @ z directly.
    W_lm_cpu = W_lm.cpu().float()
    g_final_cpu = g_final.cpu().float()
    # Joint matrix (what Carlini actually recovers):
    W_eff = W_lm_cpu * g_final_cpu.unsqueeze(0)  # [V, d], column i scaled by g_final[i]

    # pinv(W_eff) via Gram formulation (float64 for stability)
    gram = W_eff.double().T @ W_eff.double()   # [d, d]
    gram += 1e-8 * gram.diagonal().mean() * torch.eye(D_MODEL, dtype=torch.float64)
    gram_inv = torch.linalg.inv(gram)           # [d, d]
    pinv_W_eff = (gram_inv @ W_eff.double().T).float()  # [d, V]

    logger.info("  pinv(W_lm @ diag(g_final)) computed: %s (NO g_final leak)", pinv_W_eff.shape)

    h_out_est = torch.zeros_like(h_out_all)
    cos_list = []

    for n_start in range(0, N, BS):
        n_end = min(n_start + BS, N)
        z_batch = logits_all[n_start:n_end]    # [bs, T, V]
        h_in_batch = h_in_all[n_start:n_end]   # [bs, T, d]
        h_out_batch = h_out_all[n_start:n_end]  # [bs, T, d]

        # Direct: u = pinv(W_eff) @ z = h_out / rms(h_out)
        u = z_batch @ pinv_W_eff.T             # [bs, T, d]

        # Recover scale via projection
        uu = (u * u).sum(dim=-1, keepdim=True)
        uh = (u * h_in_batch).sum(dim=-1, keepdim=True)
        alpha = uh / (uu + 1e-12)
        alpha = alpha.clamp(min=0.1)

        h_out_est_batch = alpha * u
        h_out_est[n_start:n_end] = h_out_est_batch

        # Sample cosines (not all NxT to save time)
        for i in range(0, h_out_est_batch.shape[0], max(1, h_out_est_batch.shape[0] // 4)):
            for t in range(0, T, max(1, T // 4)):
                c = flat_cosine(h_out_est_batch[i, t], h_out_batch[i, t])
                cos_list.append(c)

    mean_cos_h = sum(cos_list) / len(cos_list) if cos_list else 0.0
    global_cos_h = flat_cosine(h_out_est, h_out_all)

    results["step0_h_out_recovery"] = {
        "mean_per_vector_cosine": mean_cos_h,
        "global_cosine": global_cos_h,
        "num_sampled_vectors": len(cos_list),
    }
    logger.info("  h_out recovery: mean_cos=%.4f, global_cos=%.4f",
                mean_cos_h, global_cos_h)

    del logits_all, pinv_W_eff, gram, gram_inv, W_lm_cpu, W_eff
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: W_down via high-precision OLS (oracle activations)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("PHASE 1: W_down recovery via high-precision OLS")
    logger.info("  (oracle activations, float64, all %d data points)", N * T)
    logger.info("=" * 70)

    t1 = time.time()
    W_down_rec, ols_metrics = solve_w_down_ols(
        h_mid_all, h_out_all, g_mlp,
        W_gate_true, W_up_true,
        device, BS,
    )

    cos_wdown = flat_cosine(W_down_rec, W_down_true)
    cos_wdown_row = per_row_cosine(W_down_rec, W_down_true)
    frob_err = (W_down_rec - W_down_true).norm().item() / (
        W_down_true.norm().item() + 1e-12
    )
    # Max absolute error
    max_abs_err = (W_down_rec - W_down_true).abs().max().item()
    # Relative entry-wise error
    rel_entry_err = (W_down_rec - W_down_true).abs().mean().item() / (
        W_down_true.abs().mean().item() + 1e-12
    )

    phase1_time = time.time() - t1
    results["phase1_w_down_ols"] = {
        "W_down_cosine": cos_wdown,
        "W_down_per_row_cosine": cos_wdown_row,
        "W_down_frob_error": frob_err,
        "W_down_max_abs_error": max_abs_err,
        "W_down_rel_entry_error": rel_entry_err,
        "time_seconds": round(phase1_time, 1),
        **ols_metrics,
    }

    logger.info("  Phase 1 results:")
    logger.info("    W_down cosine:        %.6f", cos_wdown)
    logger.info("    W_down per-row cos:   %.6f", cos_wdown_row)
    logger.info("    W_down frob error:    %.6f", frob_err)
    logger.info("    W_down max abs error: %.6f", max_abs_err)
    logger.info("    W_down rel entry err: %.6f", rel_entry_err)
    logger.info("    Condition number:     %.2e", ols_metrics.get("condition_number", -1))
    logger.info("    Time: %.1fs", phase1_time)

    # ── Sanity check: verify on a few data points ────────────────────────────
    with torch.no_grad():
        g_mlp_dev = g_mlp.to(device)
        W_gate_dev = W_gate_true.to(device)
        W_up_dev = W_up_true.to(device)

        # Take first batch
        h_mid_check = h_mid_all[:BS].to(device).reshape(BS * T, D_MODEL)
        h_out_check = h_out_all[:BS].to(device).reshape(BS * T, D_MODEL)
        x_check = rms_norm(h_mid_check, g_mlp_dev)
        a_check = F.silu(x_check @ W_gate_dev.T) * (x_check @ W_up_dev.T)
        r_true = (h_out_check - h_mid_check).float()

        r_rec = (a_check @ W_down_rec.to(device).T).float()
        recon_cos = flat_cosine(r_rec.cpu(), r_true.cpu())
        recon_mse = F.mse_loss(r_rec, r_true).item()
        logger.info("    Sanity check (first %d points): recon_cos=%.6f, mse=%.6e",
                    BS * T, recon_cos, recon_mse)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: W_gate / W_up via per-neuron decomposition
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("PHASE 2: W_gate / W_up via per-neuron target decomposition")
    logger.info("=" * 70)

    t2 = time.time()

    # Step 2a: Compute target activations from recovered W_down
    logger.info("  Step 2a: Computing target activations via pinv(W_down_rec)...")

    M_total = N * T
    g_mlp_dev = g_mlp.to(device)

    # Compute MLP residual and normalized input for all data
    # r_mlp = h_out - h_mid, X = RMSNorm(h_mid, g_mlp)
    r_mlp_all = torch.zeros(M_total, D_MODEL)
    X_all = torch.zeros(M_total, D_MODEL)

    with torch.no_grad():
        for n_start in range(0, N, BS):
            n_end = min(n_start + BS, N)
            h_mid_b = h_mid_all[n_start:n_end].to(device)
            h_out_b = h_out_all[n_start:n_end].to(device)
            M_b = h_mid_b.shape[0] * T
            h_mid_flat = h_mid_b.reshape(M_b, D_MODEL)
            h_out_flat = h_out_b.reshape(M_b, D_MODEL)

            x_b = rms_norm(h_mid_flat, g_mlp_dev)
            r_b = (h_out_flat - h_mid_flat).float()

            offset = n_start * T
            r_mlp_all[offset:offset + M_b] = r_b.cpu()
            X_all[offset:offset + M_b] = x_b.cpu()

    logger.info("  MLP residuals and normalized inputs computed: r_mlp %s, X %s",
                r_mlp_all.shape, X_all.shape)

    # Compute target activations
    a_target = compute_target_activations(W_down_rec, r_mlp_all, device)
    logger.info("  Target activations: %s", a_target.shape)

    # Validate target activations against true activations
    with torch.no_grad():
        X_check_dev = X_all[:BS * T].to(device)
        a_true_check = (F.silu(X_check_dev @ W_gate_dev.T) *
                        (X_check_dev @ W_up_dev.T)).cpu()
        a_target_check = a_target[:BS * T]
        target_cos = flat_cosine(a_target_check, a_true_check)
        target_mse = F.mse_loss(a_target_check, a_true_check).item()
        logger.info("  Target activation quality: cos=%.6f, mse=%.6e", target_cos, target_mse)
        # Per-neuron quality
        per_neuron_target_cos = F.cosine_similarity(
            a_target_check.T, a_true_check.T, dim=1
        )
        logger.info("  Per-neuron target cos: mean=%.4f, median=%.4f, min=%.4f",
                    per_neuron_target_cos.mean().item(),
                    per_neuron_target_cos.median().item(),
                    per_neuron_target_cos.min().item())

    results["phase2_target_quality"] = {
        "target_activation_cosine": target_cos,
        "target_activation_mse": target_mse,
        "per_neuron_target_cos_mean": per_neuron_target_cos.mean().item(),
        "per_neuron_target_cos_median": per_neuron_target_cos.median().item(),
    }

    # Step 2b: Per-neuron ALS
    logger.info("  Step 2b: Per-neuron ALS decomposition...")

    W_gate_rec, W_up_rec, neuron_metrics = recover_gate_up_per_neuron(
        a_target, X_all,
        W_gate_true, W_up_true,
        device,
        als_iters=args.neuron_als_iters,
        lbfgs_steps=args.neuron_lbfgs_steps,
        neuron_batch_size=args.neuron_batch_size,
        seed=args.seed,
    )

    phase2_time = time.time() - t2

    # Evaluate recovered gate/up
    cos_gate_flat = flat_cosine(W_gate_rec, W_gate_true)
    cos_up_flat = flat_cosine(W_up_rec, W_up_true)
    cos_gate_row = per_row_cosine(W_gate_rec, W_gate_true)
    cos_up_row = per_row_cosine(W_up_rec, W_up_true)

    logger.info("  Phase 2 raw cosines:")
    logger.info("    W_gate flat: %.6f, per-row: %.6f", cos_gate_flat, cos_gate_row)
    logger.info("    W_up   flat: %.6f, per-row: %.6f", cos_up_flat, cos_up_row)

    # Hungarian alignment
    rec_dict = {
        f"{prefix}.mlp.gate_proj.weight": W_gate_rec,
        f"{prefix}.mlp.up_proj.weight": W_up_rec,
        f"{prefix}.mlp.down_proj.weight": W_down_rec,
    }
    tea_dict = {
        f"{prefix}.mlp.gate_proj.weight": W_gate_true,
        f"{prefix}.mlp.up_proj.weight": W_up_true,
        f"{prefix}.mlp.down_proj.weight": W_down_true,
    }

    logger.info("  Running Hungarian neuron alignment (may take a moment for d_ff=%d)...",
                D_FF)
    aligned_dict = align_ffn_neurons(rec_dict, tea_dict, f"{prefix}")

    cos_gate_aligned = flat_cosine(
        aligned_dict[f"{prefix}.mlp.gate_proj.weight"], W_gate_true
    )
    cos_up_aligned = flat_cosine(
        aligned_dict[f"{prefix}.mlp.up_proj.weight"], W_up_true
    )
    cos_down_aligned = flat_cosine(
        aligned_dict[f"{prefix}.mlp.down_proj.weight"], W_down_true
    )

    cos_gate_aligned_row = per_row_cosine(
        aligned_dict[f"{prefix}.mlp.gate_proj.weight"], W_gate_true
    )
    cos_up_aligned_row = per_row_cosine(
        aligned_dict[f"{prefix}.mlp.up_proj.weight"], W_up_true
    )
    cos_down_aligned_row = per_row_cosine(
        aligned_dict[f"{prefix}.mlp.down_proj.weight"], W_down_true
    )

    logger.info("  Phase 2 aligned cosines:")
    logger.info("    W_gate: flat=%.6f, per-row=%.6f", cos_gate_aligned, cos_gate_aligned_row)
    logger.info("    W_up  : flat=%.6f, per-row=%.6f", cos_up_aligned, cos_up_aligned_row)
    logger.info("    W_down: flat=%.6f, per-row=%.6f", cos_down_aligned, cos_down_aligned_row)

    # Functional test: reconstruct MLP output with recovered weights
    with torch.no_grad():
        X_func = X_all[:BS * T].to(device)
        r_true_func = r_mlp_all[:BS * T].to(device)
        a_rec = (F.silu(X_func @ W_gate_rec.to(device).T) *
                 (X_func @ W_up_rec.to(device).T))
        r_rec_func = a_rec @ W_down_rec.to(device).T
        func_cos = flat_cosine(r_rec_func.cpu(), r_true_func.cpu())
        func_mse = F.mse_loss(r_rec_func, r_true_func).item()
        logger.info("  Functional test: recon_cos=%.6f, mse=%.6e", func_cos, func_mse)

    results["phase2_gate_up"] = {
        "raw_cosine": {
            "gate_flat": cos_gate_flat,
            "gate_per_row": cos_gate_row,
            "up_flat": cos_up_flat,
            "up_per_row": cos_up_row,
        },
        "aligned_cosine": {
            "gate_flat": cos_gate_aligned,
            "gate_per_row": cos_gate_aligned_row,
            "up_flat": cos_up_aligned,
            "up_per_row": cos_up_aligned_row,
            "down_flat": cos_down_aligned,
            "down_per_row": cos_down_aligned_row,
        },
        "functional": {
            "recon_cosine": func_cos,
            "recon_mse": func_mse,
        },
        "neuron_metrics": neuron_metrics,
        "time_seconds": round(phase2_time, 1),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: Query scaling ablation
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_ablation:
        logger.info("=" * 70)
        logger.info("PHASE 3: Query scaling ablation")
        logger.info("=" * 70)

        ablation_sizes = [int(x) for x in args.ablation_sizes.split(",")]
        ablation_results = run_phase1_ablation(
            h_mid_all, h_out_all, g_mlp,
            W_gate_true, W_up_true, W_down_true,
            device, ablation_sizes, T, BS, args.seed,
        )
        results["phase3_ablation"] = ablation_results

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_global
    results["elapsed_seconds"] = round(elapsed, 1)
    results["config"] = vars(args)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save recovered weights
    torch.save({
        "W_down_rec": W_down_rec,
        "W_gate_rec": W_gate_rec,
        "W_up_rec": W_up_rec,
    }, out_dir / "recovered_weights.pt")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ALGEBRAIC BLOCK RECOVERY v2 — SUMMARY")
    print("=" * 80)

    print(f"\nStep 0 — h_out recovery from logits:")
    s0 = results["step0_h_out_recovery"]
    print(f"  Mean per-vector cosine: {s0['mean_per_vector_cosine']:.4f}")
    print(f"  Global cosine: {s0['global_cosine']:.4f}")

    print(f"\nPhase 1 — W_down OLS (float64, all {N*T} points):")
    p1 = results["phase1_w_down_ols"]
    print(f"  W_down cosine:        {p1['W_down_cosine']:.6f}")
    print(f"  W_down per-row cos:   {p1['W_down_per_row_cosine']:.6f}")
    print(f"  W_down frob error:    {p1['W_down_frob_error']:.6f}")
    print(f"  Condition number:     {p1.get('condition_number', -1):.2e}")

    print(f"\nPhase 2 — W_gate/W_up per-neuron decomposition:")
    p2 = results["phase2_gate_up"]
    print(f"  Target activation cos: {results['phase2_target_quality']['target_activation_cosine']:.6f}")
    print(f"  Aligned cosines:")
    ac = p2["aligned_cosine"]
    print(f"    W_gate: flat={ac['gate_flat']:.6f}, per-row={ac['gate_per_row']:.6f}")
    print(f"    W_up  : flat={ac['up_flat']:.6f}, per-row={ac['up_per_row']:.6f}")
    print(f"    W_down: flat={ac['down_flat']:.6f}, per-row={ac['down_per_row']:.6f}")
    print(f"  Functional: recon_cos={p2['functional']['recon_cosine']:.6f}, "
          f"mse={p2['functional']['recon_mse']:.6e}")
    nm = p2["neuron_metrics"]
    print(f"  Per-neuron gate cos: mean={nm['per_neuron_cos_gate_mean']:.4f}, "
          f"abs_mean={nm['per_neuron_cos_gate_abs_mean']:.4f}, "
          f"gt0.9={nm['per_neuron_cos_gate_gt09']:.1%}")
    print(f"  Per-neuron up cos:   mean={nm['per_neuron_cos_up_mean']:.4f}, "
          f"abs_mean={nm['per_neuron_cos_up_abs_mean']:.4f}, "
          f"gt0.9={nm['per_neuron_cos_up_gt09']:.1%}")

    if not args.skip_ablation and "phase3_ablation" in results:
        print(f"\nPhase 3 — Query scaling ablation (W_down OLS):")
        for entry in results["phase3_ablation"]:
            print(f"  N={entry['N']:5d} (M={entry['M']:7d}): "
                  f"cos={entry['W_down_cosine']:.6f}, frob={entry['W_down_frob_error']:.6f}")

    # Verdict
    print("\n" + "-" * 40)
    best_down = p1["W_down_cosine"]
    best_gate = ac["gate_flat"]
    best_up = ac["up_flat"]
    print(f"Best: W_down={best_down:.4f}, W_gate={best_gate:.4f}, W_up={best_up:.4f}")
    if best_down > 0.95:
        print("  Phase 1 SUCCESS: W_down cos > 0.95")
    elif best_down > 0.8:
        print("  Phase 1 GOOD: W_down cos > 0.8 (v1 was 0.618)")
    else:
        print(f"  Phase 1 PARTIAL: W_down cos = {best_down:.4f}")

    if best_gate > 0.5 and best_up > 0.5:
        print("  Phase 2 SUCCESS: gate/up cos > 0.5 (v1 ALS diverged)")
    elif best_gate > 0.1 or best_up > 0.1:
        print("  Phase 2 PARTIAL: some signal recovered")
    else:
        print("  Phase 2 FAILED: gate/up near random")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to: {out_dir / 'results.json'}")
    print(f"Weights saved to: {out_dir / 'recovered_weights.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
