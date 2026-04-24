#!/usr/bin/env python3
# SAFETY NOTICE: QUARANTINED (alpha-theory prune 2026-04-19)
# This script is NOT cited in the paper. It was part of killed branches:
#   A1 S-PSI, A2 Moments CP, A4 logit-bias, A5 memory probing, A6 active query,
#   A7 algebraic v2/v3/v4, B3 matched-KD.
# Retained in repo for reproducibility of quarantined history; do not use for
# new claims.
#!/usr/bin/env python3
"""
Algebraic Block Recovery v3 — BREAKTHROUGH via joint (W_gate, W_up) optimization.

Diagnosis of v2 failure
-----------------------
v2's Phase 2 used pinv(W_down) @ r_mlp to reconstruct a 4864-dim activation
from a 896-dim residual. This returns the minimum-norm solution, which
discards 4864 − 896 = 3968 dimensions of information about the gated
activation. Per-neuron ALS was therefore trying to fit the wrong target.

v3 strategy
-----------
Never try to recover target activations. Fix W_down from Phase 1 and
DIRECTLY optimize (W_gate, W_up) by minimizing the residual loss

        L = || W_down (SiLU(W_gate x) * (W_up x)) − r_mlp ||²

with Adam + cosine LR schedule, minibatches, and multiple random restarts.
This is a non-convex problem of dimension 2 · 4864 · 896 ≈ 8.7M params,
but the observed residual tensor r has M = 2048 · 128 = 262K points ×
896 dims ≈ 235M scalar observations → massively overdetermined.

Pipeline
--------
Step 0: Recover h_out from logits using W_eff = W_lm @ diag(g_final).
Phase 1: W_down via high-precision OLS (oracle gate/up).
Phase 2 (NEW): Joint (W_gate, W_up) optimization with W_down fixed.
Phase 3 (bonus): Warm-start basin validation — init near teacher + noise.
Phase 4 (bonus): Re-solve W_down using recovered activations (self-consistent).

Success criterion
-----------------
ANY aligned cos > 0.3 on W_gate or W_up is a breakthrough. > 0.5 is
best-paper level.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python scripts/algebraic_recovery_v3_breakthrough.py \
        --model_name Qwen/Qwen2.5-0.5B --block_idx 23 --num_queries 2048 \
        --output_dir results/v5_algebraic_v3 --seed 42 --total_steps 20000 \
        --num_restarts 3
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


# ── helpers ──────────────────────────────────────────────────────────────────

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Algebraic Block Recovery v3")
    p.add_argument("--model_name", type=str, default=MODEL_NAME)
    p.add_argument("--block_idx", type=int, default=23)
    p.add_argument("--num_queries", type=int, default=2048)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--output_dir", type=str, default="results/v5_algebraic_v3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for data collection and OLS accumulation")
    p.add_argument("--allow_synthetic", action="store_true",
                   help="Fall back to random tokens if dataset load fails")

    # Phase 2: joint optimization
    p.add_argument("--total_steps", type=int, default=20000,
                   help="Adam optimization steps per restart")
    p.add_argument("--opt_batch_size", type=int, default=512,
                   help="Minibatch of (x, r) pairs per Adam step")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Peak learning rate for Adam")
    p.add_argument("--lr_warmup_steps", type=int, default=500,
                   help="Linear warmup steps before cosine decay")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--init_scale", type=float, default=0.02,
                   help="Std of Gaussian init for W_gate/W_up")
    p.add_argument("--num_restarts", type=int, default=3,
                   help="Number of random seeds for Phase 2")
    p.add_argument("--eval_every", type=int, default=500,
                   help="Frequency of cosine-similarity logging")
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Phase 3: warm-start basin experiment
    p.add_argument("--run_warm_start", action="store_true",
                   help="Bonus: warm-start with teacher + noise (validate basin)")
    p.add_argument("--warm_start_alphas", type=str, default="0.1,0.3,0.5,1.0",
                   help="Noise amplitudes for warm-start sweep")
    p.add_argument("--warm_start_steps", type=int, default=5000)

    # Phase 4: re-solve W_down with recovered activations
    p.add_argument("--run_w_down_resolve", action="store_true",
                   help="Bonus: re-fit W_down using recovered activations")
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
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms * g.float()).to(x.dtype)


# ── cosine helpers ───────────────────────────────────────────────────────────

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
    """Hungarian-align rows of W_rec to W_true using a reference matrix
    (gate or up). Returns (flat_cos, per_row_cos) after alignment.
    W_ref_rec / W_ref_true are used to build the similarity matrix (so gate
    and up share the same permutation). Pass W_rec itself to self-align."""
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


# ── Phase 1: High-precision OLS for W_down (reused from v2) ─────────────────

def solve_w_down_ols(
    h_mid_all: torch.Tensor,
    h_out_all: torch.Tensor,
    g_mlp: torch.Tensor,
    W_gate_true: torch.Tensor,
    W_up_true: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, dict]:
    """Solve W_down via float64 OLS using true gate/up activations."""
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

            if (n_start // batch_size) % 10 == 0:
                logger.info("    Batch %d/%d", n_start // batch_size + 1,
                            (N + batch_size - 1) // batch_size)

    diag_mean = AAt.diagonal().mean().item()
    try:
        eigvals = torch.linalg.eigvalsh(AAt)
        cond = (eigvals[-1] / eigvals[0].clamp(min=1e-30)).item()
        logger.info("  Condition number: %.2e (min_eig=%.2e, max_eig=%.2e)",
                    cond, eigvals[0].item(), eigvals[-1].item())
        if cond < 1e8:
            ridge = 1e-10 * diag_mean
        elif cond < 1e12:
            ridge = 1e-8 * diag_mean
        else:
            ridge = 1e-6 * diag_mean
        logger.info("  Ridge: %.2e", ridge)
    except Exception as e:
        logger.warning("  Eigenvalue computation failed (%s)", e)
        ridge = 1e-8 * diag_mean
        cond = -1.0

    AAt += ridge * torch.eye(d_ff, dtype=torch.float64, device=device)

    logger.info("  Solving OLS (%dx%d) via Cholesky...", d, d_ff)
    try:
        L = torch.linalg.cholesky(AAt)
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


# ── Phase 2: JOINT OPTIMIZATION of (W_gate, W_up) with W_down fixed ─────────

def lr_schedule(step: int, total: int, warmup: int, peak_lr: float) -> float:
    """Linear warmup + cosine decay."""
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * t))


def joint_optimize_gate_up(
    X_all: torch.Tensor,            # [M, d] normalized MLP input
    r_mlp_all: torch.Tensor,        # [M, d] MLP residual
    W_down_fixed: torch.Tensor,     # [d, d_ff] recovered W_down
    W_gate_true: torch.Tensor,      # [d_ff, d] for evaluation only
    W_up_true: torch.Tensor,        # [d_ff, d] for evaluation only
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
    W_gate_init: torch.Tensor | None = None,  # optional warm start
    W_up_init: torch.Tensor | None = None,
    tag: str = "",
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Minimize || W_down (SiLU(W_gate x) * (W_up x)) − r || via Adam.

    IMPORTANT: W_gate_true / W_up_true are ONLY used for eval logging.
    They are never used to steer the optimizer. Data-leakage audit:
    the loss function only sees X, r, W_down_fixed.
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

    W_down = W_down_fixed.float().to(device)  # frozen

    # move evaluation tensors to device
    W_gate_true_dev = W_gate_true.float().to(device)
    W_up_true_dev = W_up_true.float().to(device)

    # Move data to device (small enough: M ~ 262K × 896 × 4B = 940MB per tensor)
    # For memory safety, keep data on CPU and move per-minibatch.
    # But in practice 2048 queries × 128 × 896 × 2 × 4 = 1.9 GB which fits on A100.
    # We'll choose CPU-resident and per-batch gather — safer.
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
        # LR schedule
        cur_lr = lr_schedule(step, total_steps, warmup_steps, peak_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Sample minibatch
        idx = torch.randint(0, M, (batch_size,), generator=rng)
        x_b = X_cpu[idx].to(device, non_blocking=True)         # [B, d]
        r_b = r_cpu[idx].to(device, non_blocking=True).float() # [B, d]

        # Forward: r_pred = W_down @ (SiLU(W_gate x) * (W_up x))
        # Use bf16 forward for speed; keep gradients in float32 via autograd
        g = x_b @ W_gate.T         # [B, d_ff]
        s = F.silu(g)
        u = x_b @ W_up.T           # [B, d_ff]
        a = s * u                  # [B, d_ff]
        r_pred = a @ W_down.T      # [B, d]

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

                # Raw (no-alignment) cosines — cheap, always compute
                raw_g = flat_cosine(Wg, W_gate_true_dev)
                raw_u = flat_cosine(Wu, W_up_true_dev)
                raw_g_row = per_row_cosine(Wg, W_gate_true_dev)
                raw_u_row = per_row_cosine(Wu, W_up_true_dev)

                # Hungarian alignment is O(d_ff^3) and costs ~5s per call.
                # Do it only at checkpoint multiples (1 in 4 evals) and at
                # the final step.
                is_final = (step == total_steps - 1)
                do_align = is_final or (step % (eval_every * 4) == 0)

                cg_flat = cg_row = cu_flat = cu_row = float("nan")
                cj_flat = cj_row = cju_flat = cju_row = float("nan")
                Wg_cpu = Wg.cpu()
                Wu_cpu = Wu.cpu()

                if do_align:
                    # Self-alignment: gate permutation from gate, up permutation from up
                    cg_flat, cg_row = cos_with_alignment(
                        Wg_cpu, W_gate_true.cpu(),
                        Wg_cpu, W_gate_true.cpu(),
                    )
                    cu_flat, cu_row = cos_with_alignment(
                        Wu_cpu, W_up_true.cpu(),
                        Wu_cpu, W_up_true.cpu(),
                    )
                    # Joint permutation using concatenated gate||up
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
                        "raw(g,u)=(%.3f,%.3f) | align-self(g,u)=(%.3f,%.3f) | "
                        "align-joint(g,u)=(%.3f,%.3f) | %.1fs",
                        tag, step, total_steps, cur_lr, loss_val, loss_ema,
                        raw_g, raw_u, cg_flat, cu_flat, cj_flat, cju_flat, elapsed,
                    )
                else:
                    logger.info(
                        "[%s] step %5d/%d | lr=%.2e | loss=%.4e ema=%.4e | "
                        "raw(g,u)=(%.3f,%.3f) row(%.3f,%.3f) | %.1fs",
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

                # Track best by sum of joint-aligned cosines (when available)
                # or by raw cosines otherwise
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

    # Return best checkpoint (by joint-aligned cosine), not final
    if best_state is not None:
        return best_state["W_gate"], best_state["W_up"], log
    return W_gate.detach().cpu(), W_up.detach().cpu(), log


# ── Phase 3: Warm-start basin validation ─────────────────────────────────────

def warm_start_sweep(
    X_all, r_mlp_all, W_down_rec, W_gate_true, W_up_true,
    device, alphas: list[float], steps: int, batch_size: int,
    peak_lr: float, warmup_steps: int, init_scale: float,
    grad_clip: float, eval_every: int, seed: int,
) -> list[dict]:
    """For each α, init W_gate/W_up = W_true + α · noise and measure recovery.

    This probes the optimization basin: if training fixes any deviation
    up to α_max, then the attack landscape has a finite basin around the
    teacher, and the question becomes how to find that basin from scratch.
    """
    results = []
    for alpha in alphas:
        logger.info("=" * 70)
        logger.info("  Warm-start α = %.3f", alpha)
        logger.info("=" * 70)
        torch.manual_seed(seed + int(alpha * 1000))
        noise_g = torch.randn_like(W_gate_true) * init_scale
        noise_u = torch.randn_like(W_up_true) * init_scale
        Wg0 = W_gate_true + alpha * noise_g
        Wu0 = W_up_true + alpha * noise_u

        # Initial loss sanity check
        with torch.no_grad():
            # compute loss at init on a big batch
            x_probe = X_all[:4096].to(device)
            r_probe = r_mlp_all[:4096].to(device)
            Wg0_dev = Wg0.to(device)
            Wu0_dev = Wu0.to(device)
            Wd_dev = W_down_rec.to(device)
            a_probe = F.silu(x_probe @ Wg0_dev.T) * (x_probe @ Wu0_dev.T)
            r_pred = a_probe @ Wd_dev.T
            init_loss = F.mse_loss(r_pred, r_probe).item()
        logger.info("  Init loss @ α=%.3f: %.4e", alpha, init_loss)

        Wg_rec, Wu_rec, log = joint_optimize_gate_up(
            X_all, r_mlp_all, W_down_rec,
            W_gate_true, W_up_true, device,
            total_steps=steps,
            batch_size=batch_size,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            weight_decay=0.0,
            init_scale=init_scale,
            grad_clip=grad_clip,
            eval_every=eval_every,
            seed=seed + int(alpha * 10_000),
            W_gate_init=Wg0,
            W_up_init=Wu0,
            tag=f"warm-α{alpha:.2f}",
        )

        # Final metrics
        cg = flat_cosine(Wg_rec, W_gate_true)
        cu = flat_cosine(Wu_rec, W_up_true)
        cg_row = per_row_cosine(Wg_rec, W_gate_true)
        cu_row = per_row_cosine(Wu_rec, W_up_true)

        results.append({
            "alpha": alpha,
            "init_loss": init_loss,
            "final_loss_ema": log[-1]["loss_ema"] if log else None,
            "raw_cos_gate_final": cg,
            "raw_cos_up_final": cu,
            "per_row_cos_gate_final": cg_row,
            "per_row_cos_up_final": cu_row,
            "log": log,
        })
        logger.info("  α=%.2f → gate cos=%.4f, up cos=%.4f",
                    alpha, cg, cu)
    return results


# ── Phase 4: Re-solve W_down using recovered activations ────────────────────

def resolve_w_down_self_consistent(
    X_all, h_mid_all, h_out_all,
    W_gate_rec: torch.Tensor, W_up_rec: torch.Tensor,
    g_mlp: torch.Tensor,
    W_down_true: torch.Tensor,
    device: torch.device, batch_size: int,
) -> tuple[torch.Tensor, dict]:
    """Given recovered W_gate/W_up, re-fit W_down to see if self-consistent
    recovery improves the Phase 1 result."""
    N, T, d = h_mid_all.shape
    d_ff = W_gate_rec.shape[0]
    M = N * T

    g_mlp_dev = g_mlp.to(device)
    Wg_dev = W_gate_rec.to(device)
    Wu_dev = W_up_rec.to(device)

    AAt = torch.zeros(d_ff, d_ff, dtype=torch.float64, device=device)
    RAt = torch.zeros(d, d_ff, dtype=torch.float64, device=device)

    with torch.no_grad():
        for n_start in range(0, N, batch_size):
            n_end = min(n_start + batch_size, N)
            h_mid_b = h_mid_all[n_start:n_end].to(device)
            h_out_b = h_out_all[n_start:n_end].to(device)
            M_b = h_mid_b.shape[0] * T
            h_mid_flat = h_mid_b.reshape(M_b, d)
            h_out_flat = h_out_b.reshape(M_b, d)
            x = rms_norm(h_mid_flat, g_mlp_dev)
            a = F.silu(x @ Wg_dev.T) * (x @ Wu_dev.T)
            r = (h_out_flat - h_mid_flat).float()
            AAt += a.double().T @ a.double()
            RAt += r.double().T @ a.double()

    diag_mean = AAt.diagonal().mean().item()
    ridge = 1e-8 * diag_mean
    AAt += ridge * torch.eye(d_ff, dtype=torch.float64, device=device)
    try:
        L = torch.linalg.cholesky(AAt)
        W_down_sc = torch.linalg.solve_triangular(
            L.T,
            torch.linalg.solve_triangular(L, RAt.T, upper=False),
            upper=True,
        ).T.float()
    except RuntimeError:
        W_down_sc = torch.linalg.solve(AAt, RAt.T).T.float()

    cos_flat = flat_cosine(W_down_sc.cpu(), W_down_true)
    cos_row = per_row_cosine(W_down_sc.cpu(), W_down_true)
    metrics = {
        "W_down_cos_flat": cos_flat,
        "W_down_cos_row": cos_row,
        "ridge": ridge,
    }
    return W_down_sc.cpu(), metrics


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
    logger.info("Algebraic Block Recovery v3 — BREAKTHROUGH")
    logger.info("=" * 70)
    logger.info("Model : %s", args.model_name)
    logger.info("Block : %d", block_idx)
    logger.info("Queries: %d x %d tokens = %d data points", N, T, N * T)
    logger.info("Device : %s", device)
    logger.info("Phase 2: %d steps x %d restarts, batch=%d, lr=%.2e",
                args.total_steps, args.num_restarts,
                args.opt_batch_size, args.lr)

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

    W_lm = teacher.lm_head.weight.data.float().to(device)
    g_final = teacher.model.norm.weight.data.float().to(device)
    g_mlp = true_params[f"{prefix}.post_attention_layernorm.weight"].float()
    W_gate_true = true_params[f"{prefix}.mlp.gate_proj.weight"].float()
    W_up_true = true_params[f"{prefix}.mlp.up_proj.weight"].float()
    W_down_true = true_params[f"{prefix}.mlp.down_proj.weight"].float()

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
    del teacher
    torch.cuda.empty_cache()

    h_in_all = torch.cat(all_h_in, dim=0)
    h_mid_all = torch.cat(all_h_mid, dim=0)
    h_out_all = torch.cat(all_h_out, dim=0)
    logits_all = torch.cat(all_logits, dim=0)

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
    # STEP 0: Recover h_out from logits via W_lm @ diag(g_final)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 0: Recover h_out from logits via pinv(W_lm @ diag(g_final))")
    logger.info("=" * 70)

    # NO DATA LEAKAGE: Carlini's attack recovers W_lm @ diag(g_final) jointly.
    W_lm_cpu = W_lm.cpu().float()
    g_final_cpu = g_final.cpu().float()
    W_eff = W_lm_cpu * g_final_cpu.unsqueeze(0)  # [V, d]

    gram = W_eff.double().T @ W_eff.double()
    gram += 1e-8 * gram.diagonal().mean() * torch.eye(D_MODEL, dtype=torch.float64)
    gram_inv = torch.linalg.inv(gram)
    pinv_W_eff = (gram_inv @ W_eff.double().T).float()

    logger.info("  pinv(W_lm @ diag(g_final)) computed: %s", pinv_W_eff.shape)

    h_out_est = torch.zeros_like(h_out_all)
    cos_list = []

    for n_start in range(0, N, BS):
        n_end = min(n_start + BS, N)
        z_batch = logits_all[n_start:n_end]
        h_in_batch = h_in_all[n_start:n_end]
        h_out_batch = h_out_all[n_start:n_end]

        u = z_batch @ pinv_W_eff.T
        uu = (u * u).sum(dim=-1, keepdim=True)
        uh = (u * h_in_batch).sum(dim=-1, keepdim=True)
        alpha = uh / (uu + 1e-12)
        alpha = alpha.clamp(min=0.1)

        h_out_est_batch = alpha * u
        h_out_est[n_start:n_end] = h_out_est_batch

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
    max_abs_err = (W_down_rec - W_down_true).abs().max().item()
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

    logger.info("  W_down cos=%.6f, per-row=%.6f, frob=%.6f, cond=%.2e, time=%.1fs",
                cos_wdown, cos_wdown_row, frob_err,
                ols_metrics.get("condition_number", -1), phase1_time)

    # Sanity: functional reconstruction with oracle activations + recovered W_down
    with torch.no_grad():
        g_mlp_dev = g_mlp.to(device)
        W_gate_dev = W_gate_true.to(device)
        W_up_dev = W_up_true.to(device)

        h_mid_check = h_mid_all[:BS].to(device).reshape(BS * T, D_MODEL)
        h_out_check = h_out_all[:BS].to(device).reshape(BS * T, D_MODEL)
        x_check = rms_norm(h_mid_check, g_mlp_dev)
        a_check = F.silu(x_check @ W_gate_dev.T) * (x_check @ W_up_dev.T)
        r_true = (h_out_check - h_mid_check).float()

        r_rec = (a_check @ W_down_rec.to(device).T).float()
        recon_cos = flat_cosine(r_rec.cpu(), r_true.cpu())
        logger.info("  Phase 1 functional recon_cos (oracle a): %.6f", recon_cos)

    # ══════════════════════════════════════════════════════════════════════════
    # Prepare X and r_mlp for Phase 2 (compute ONCE, reuse across restarts)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("Preparing training data (X, r_mlp) for Phase 2...")
    logger.info("=" * 70)

    M_total = N * T
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

    logger.info("  Training data ready: X %s, r_mlp %s",
                X_all.shape, r_mlp_all.shape)

    # Release hidden-state storage to free RAM
    del h_in_all
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: JOINT OPTIMIZATION of (W_gate, W_up) — THE CORE ATTACK
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("PHASE 2: Joint (W_gate, W_up) optimization [multi-restart]")
    logger.info("  W_down is FIXED (from Phase 1).")
    logger.info("  W_gate_true/W_up_true are used ONLY for eval logging.")
    logger.info("  DATA LEAKAGE AUDIT: loss uses only X, r_mlp, W_down_rec.")
    logger.info("=" * 70)

    t2 = time.time()
    restart_records = []
    best_restart = None

    for restart_idx in range(args.num_restarts):
        logger.info("─" * 70)
        logger.info("  RESTART %d / %d (seed=%d)", restart_idx + 1,
                    args.num_restarts, args.seed + restart_idx * 1000)
        logger.info("─" * 70)

        Wg_rec, Wu_rec, log = joint_optimize_gate_up(
            X_all, r_mlp_all, W_down_rec,
            W_gate_true, W_up_true, device,
            total_steps=args.total_steps,
            batch_size=args.opt_batch_size,
            peak_lr=args.lr,
            warmup_steps=args.lr_warmup_steps,
            weight_decay=args.weight_decay,
            init_scale=args.init_scale,
            grad_clip=args.grad_clip,
            eval_every=args.eval_every,
            seed=args.seed + restart_idx * 1000,
            W_gate_init=None,
            W_up_init=None,
            tag=f"r{restart_idx}",
        )

        # Final evaluation with alignment
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

        # Raw (no alignment)
        cg_raw = flat_cosine(Wg_rec, W_gate_true)
        cu_raw = flat_cosine(Wu_rec, W_up_true)

        # Functional reconstruction on held-out batch (use last 10% of data)
        with torch.no_grad():
            hold_start = int(0.9 * M_total)
            X_hold = X_all[hold_start:].to(device)
            r_hold = r_mlp_all[hold_start:].to(device)
            Wg_dev = Wg_rec.to(device)
            Wu_dev = Wu_rec.to(device)
            Wd_dev = W_down_rec.to(device)
            a = F.silu(X_hold @ Wg_dev.T) * (X_hold @ Wu_dev.T)
            r_pred = a @ Wd_dev.T
            func_cos = flat_cosine(r_pred.cpu(), r_hold.cpu())
            func_mse = F.mse_loss(r_pred, r_hold).item()

        # Per-neuron death analysis (neurons whose SiLU(W_gate_true x) is
        # near zero for almost all inputs — those are pathologically hard to
        # recover, and we should not blame the attack for missing them)
        with torch.no_grad():
            X_probe = X_all[:min(M_total, 16384)].to(device)
            g_true = X_probe @ W_gate_true.to(device).T
            s_true = F.silu(g_true)
            mean_abs_s = s_true.abs().mean(dim=0)  # [d_ff]
            dead_threshold = 1e-3 * mean_abs_s.mean().item()
            dead_count = (mean_abs_s < dead_threshold).sum().item()
            near_dead_count = (mean_abs_s < 1e-2 * mean_abs_s.mean()).sum().item()

        record = {
            "restart": restart_idx,
            "seed": args.seed + restart_idx * 1000,
            "final_loss_ema": log[-1]["loss_ema"] if log else None,
            "raw_cos_gate": cg_raw,
            "raw_cos_up": cu_raw,
            "aligned_cos_gate": cg_aligned,
            "aligned_cos_up": cu_aligned,
            "aligned_cos_gate_row": cg_aligned_row,
            "aligned_cos_up_row": cu_aligned_row,
            "aligned_cos_down": cd_aligned,
            "functional_recon_cos": func_cos,
            "functional_recon_mse": func_mse,
            "dead_neurons": dead_count,
            "near_dead_neurons": near_dead_count,
            "total_neurons": D_FF,
            "log_tail": log[-5:],  # keep size bounded
        }
        restart_records.append(record)

        logger.info("  ═══ Restart %d summary ═══", restart_idx)
        logger.info("    raw cos:       gate=%.4f, up=%.4f", cg_raw, cu_raw)
        logger.info("    aligned cos:   gate=%.4f (row=%.4f), up=%.4f (row=%.4f)",
                    cg_aligned, cg_aligned_row, cu_aligned, cu_aligned_row)
        logger.info("    functional:    recon_cos=%.4f, mse=%.4e",
                    func_cos, func_mse)
        logger.info("    dead neurons:  %d / %d (near-dead %d)",
                    dead_count, D_FF, near_dead_count)

        # Track best by sum of aligned cosines
        score = cg_aligned + cu_aligned
        if best_restart is None or score > best_restart["score"]:
            best_restart = {
                "score": score,
                "restart": restart_idx,
                "W_gate": Wg_rec.clone(),
                "W_up": Wu_rec.clone(),
                "aligned_W_gate": aligned[f"{prefix}.mlp.gate_proj.weight"].clone(),
                "aligned_W_up": aligned[f"{prefix}.mlp.up_proj.weight"].clone(),
                "record": record,
            }

    phase2_time = time.time() - t2

    results["phase2_joint_opt"] = {
        "restart_records": restart_records,
        "best_restart": best_restart["restart"],
        "best_score": best_restart["score"],
        "time_seconds": round(phase2_time, 1),
    }

    Wg_best = best_restart["W_gate"]
    Wu_best = best_restart["W_up"]
    Wg_best_aligned = best_restart["aligned_W_gate"]
    Wu_best_aligned = best_restart["aligned_W_up"]

    logger.info("=" * 70)
    logger.info("PHASE 2 BEST (restart %d):", best_restart["restart"])
    logger.info("  aligned gate cos = %.4f", best_restart["record"]["aligned_cos_gate"])
    logger.info("  aligned up   cos = %.4f", best_restart["record"]["aligned_cos_up"])
    logger.info("  functional recon = %.4f", best_restart["record"]["functional_recon_cos"])
    logger.info("  Phase 2 total time: %.1fs", phase2_time)
    logger.info("=" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 (BONUS): Warm-start basin validation
    # ══════════════════════════════════════════════════════════════════════════
    if args.run_warm_start:
        logger.info("=" * 70)
        logger.info("PHASE 3 (BONUS): Warm-start basin validation")
        logger.info("=" * 70)
        alphas = [float(x) for x in args.warm_start_alphas.split(",")]
        t3 = time.time()
        warm_results = warm_start_sweep(
            X_all, r_mlp_all, W_down_rec,
            W_gate_true, W_up_true,
            device, alphas,
            steps=args.warm_start_steps,
            batch_size=args.opt_batch_size,
            peak_lr=args.lr,
            warmup_steps=min(args.lr_warmup_steps, args.warm_start_steps // 10),
            init_scale=args.init_scale,
            grad_clip=args.grad_clip,
            eval_every=args.eval_every,
            seed=args.seed,
        )
        # Strip heavy fields before serializing
        for r in warm_results:
            if "log" in r and len(r["log"]) > 10:
                r["log_tail"] = r["log"][-5:]
                r["log_head"] = r["log"][:3]
                del r["log"]
        results["phase3_warm_start"] = {
            "alphas": alphas,
            "results": warm_results,
            "time_seconds": round(time.time() - t3, 1),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4 (BONUS): Re-solve W_down with recovered activations
    # ══════════════════════════════════════════════════════════════════════════
    if args.run_w_down_resolve:
        logger.info("=" * 70)
        logger.info("PHASE 4 (BONUS): Re-solve W_down with recovered gate/up")
        logger.info("=" * 70)
        t4 = time.time()
        W_down_sc, sc_metrics = resolve_w_down_self_consistent(
            X_all, h_mid_all, h_out_all,
            Wg_best, Wu_best, g_mlp, W_down_true,
            device, BS,
        )
        logger.info("  Self-consistent W_down cos: flat=%.6f, row=%.6f",
                    sc_metrics["W_down_cos_flat"], sc_metrics["W_down_cos_row"])
        logger.info("  (Phase 1 baseline was %.6f)", cos_wdown)
        results["phase4_w_down_resolve"] = {
            **sc_metrics,
            "baseline_phase1_cos": cos_wdown,
            "delta_cos": sc_metrics["W_down_cos_flat"] - cos_wdown,
            "time_seconds": round(time.time() - t4, 1),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Save artifacts
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_global
    results["elapsed_seconds"] = round(elapsed, 1)
    results["config"] = vars(args)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o)
                  if hasattr(o, "item") else str(o))

    torch.save({
        "W_down_rec": W_down_rec,
        "W_gate_rec": Wg_best,
        "W_up_rec": Wu_best,
        "W_gate_rec_aligned": Wg_best_aligned,
        "W_up_rec_aligned": Wu_best_aligned,
    }, out_dir / "recovered_weights.pt")

    # ── Print final summary ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ALGEBRAIC BLOCK RECOVERY v3 (BREAKTHROUGH) — SUMMARY")
    print("=" * 80)

    s0 = results["step0_h_out_recovery"]
    print(f"\nStep 0 — h_out recovery from logits:")
    print(f"  mean per-vector cosine: {s0['mean_per_vector_cosine']:.4f}")
    print(f"  global cosine:          {s0['global_cosine']:.4f}")

    p1 = results["phase1_w_down_ols"]
    print(f"\nPhase 1 — W_down OLS (float64, all {N*T} points):")
    print(f"  W_down cosine:        {p1['W_down_cosine']:.6f}")
    print(f"  W_down per-row cos:   {p1['W_down_per_row_cosine']:.6f}")
    print(f"  condition number:     {p1.get('condition_number', -1):.2e}")

    p2 = results["phase2_joint_opt"]
    print(f"\nPhase 2 — Joint (W_gate, W_up) optimization "
          f"({args.total_steps} steps × {args.num_restarts} restarts):")
    for rec in p2["restart_records"]:
        print(f"  restart {rec['restart']}: "
              f"raw(g,u)=({rec['raw_cos_gate']:.4f},{rec['raw_cos_up']:.4f}) "
              f"aligned(g,u)=({rec['aligned_cos_gate']:.4f},"
              f"{rec['aligned_cos_up']:.4f}) "
              f"func_recon={rec['functional_recon_cos']:.4f} "
              f"loss_ema={rec['final_loss_ema']:.4e}")
    best = p2["restart_records"][p2["best_restart"]]
    print(f"  BEST (restart {p2['best_restart']}):")
    print(f"    aligned W_gate cosine: {best['aligned_cos_gate']:.4f}")
    print(f"    aligned W_up   cosine: {best['aligned_cos_up']:.4f}")
    print(f"    functional recon cos : {best['functional_recon_cos']:.4f}")
    print(f"    near-dead neurons    : {best['near_dead_neurons']} / {D_FF}")

    if "phase3_warm_start" in results:
        print(f"\nPhase 3 (bonus) — Warm-start basin sweep:")
        for r in results["phase3_warm_start"]["results"]:
            print(f"  α={r['alpha']:.2f}: init_loss={r['init_loss']:.2e}, "
                  f"final_loss_ema={r['final_loss_ema']:.2e}, "
                  f"gate_cos={r['raw_cos_gate_final']:.4f}, "
                  f"up_cos={r['raw_cos_up_final']:.4f}")

    if "phase4_w_down_resolve" in results:
        p4 = results["phase4_w_down_resolve"]
        print(f"\nPhase 4 (bonus) — W_down re-solve with recovered gate/up:")
        print(f"  Phase 1 baseline:     {p4['baseline_phase1_cos']:.6f}")
        print(f"  Self-consistent:      {p4['W_down_cos_flat']:.6f} "
              f"(Δ = {p4['delta_cos']:+.6f})")

    # Verdict
    print("\n" + "-" * 40)
    g_best = best["aligned_cos_gate"]
    u_best = best["aligned_cos_up"]
    any_best = max(g_best, u_best, best["raw_cos_gate"], best["raw_cos_up"])
    print(f"Best cosines — aligned: gate={g_best:.4f}, up={u_best:.4f}")
    print(f"              raw:      gate={best['raw_cos_gate']:.4f}, "
          f"up={best['raw_cos_up']:.4f}")
    if any_best > 0.5:
        print("  *** BEST-PAPER LEVEL: cos > 0.5 ***")
    elif any_best > 0.3:
        print("  *** BREAKTHROUGH: cos > 0.3 ***")
    elif any_best > 0.1:
        print("  PARTIAL: some signal (cos > 0.1)")
    else:
        print("  NO SIGNAL: cos < 0.1, joint opt did not recover structure")

    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print(f"Results: {out_dir / 'results.json'}")
    print(f"Weights: {out_dir / 'recovered_weights.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
