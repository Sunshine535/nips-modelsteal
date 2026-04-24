#!/usr/bin/env python3
# SAFETY NOTICE: QUARANTINED (alpha-theory prune 2026-04-19)
# This script is NOT cited in the paper. It was part of killed branches:
#   A1 S-PSI, A2 Moments CP, A4 logit-bias, A5 memory probing, A6 active query,
#   A7 algebraic v2/v3/v4, B3 matched-KD.
# Retained in repo for reproducibility of quarantined history; do not use for
# new claims.
#!/usr/bin/env python3
"""
Finite-Difference Jacobian Attack — Proof-of-concept probe.

============================================================================
SAFETY NOTICE (2026-04-19)
----------------------------------------------------------------------------
As currently implemented, this script SEEDS the student from the teacher
on STAGE 2 (see `attack_jacobian_fd.py` stage labelled "Instantiate random
student, copy lm_head + final norm + last block"): it copies the teacher's
`lm_head`, final RMSNorm, and the last block into the student before
perturbing by 1%.  Any positive finite-difference result produced by this
script is therefore contaminated and MUST NOT be cited as a black-box
attack outcome.  The script self-documents this choice as "POC
convenience" in the STAGE 2 comments.

No paper-body result in main.tex currently cites this script's outputs.
Before any such citation, the STAGE 2 copy must be removed (e.g., replace
the last-block copy with Kaiming-random initialization) and the attack
rerun.
============================================================================

Goal
----
Attempt to recover ANY internal transformer parameter (not just lm_head)
from PURE black-box logit queries, leveraging Carlini's W_lm @ diag(g_final)
(what their SVD actually recovers) as a starting point.

Core idea
---------
1. Run Carlini SVD to extract W_eff = W_lm @ diag(g_final).  Known to work.
2. For each query x, recover the rms-normalized last hidden direction
        u(x) = pinv(W_eff) @ z(x)
   This is the *direction* of h_L with ambiguous scalar magnitude (since
   RMSNorm destroys norm information).
3. Perturb x at a single token position -> x'.  Collect finite-difference
   estimates  Du(x, x') = u(x') - u(x).  These encode (up to RMS scale)
   directional derivatives of the teacher's full-model Jacobian.
4. Do the same with a student (same architecture, mostly random weights).
   Student's internal hidden states are fully observable, so we have a
   reference Jacobian J_student(x) computable exactly.
5. The residual Du_teacher - Du_student encodes the *difference* in the last
   block's parameters — *if* the student's prefix hidden states at block 22
   are close enough to the teacher's.
6. Enforce (5) by a short KD pre-training stage on last-token logit MSE.
   That nudges h_22_student toward h_22_teacher on the query set.

Threat model
------------
Pure black-box API access to teacher.  NO hidden states, NO oracle boundary,
NO fine-tuning access.  Attacker only sees teacher(x) -> z(x).  Attacker
KNOWS the architecture (Qwen2.5-0.5B).

Verdict
-------
Prints ATTACK SUCCEEDED (cos > 0.2 for any recovered block matrix) vs
ATTACK FAILED (documenting the barrier) at the end, plus a detailed JSON
breakdown of what the finite-difference Jacobian actually sees.

Usage
-----
CUDA_VISIBLE_DEVICES=0 python scripts/attack_jacobian_fd.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_queries 2048 \
    --perturbations_per_query 8 \
    --pretrain_steps 500 \
    --output_dir results/v5_attack_jacobian_fd \
    --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ── Model architecture constants (Qwen2.5-0.5B) ──────────────────────────────
NUM_LAYERS = 24
LAST_BLOCK_IDX = 23           # block we target
SECOND_LAST_IDX = 22          # boundary block
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
    p = argparse.ArgumentParser(description="Finite-Difference Jacobian Attack (POC)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--num_queries", type=int, default=2048,
                   help="Number of base queries")
    p.add_argument("--perturbations_per_query", type=int, default=8,
                   help="How many single-token perturbations per base query")
    p.add_argument("--perturb_position", type=str, default="last",
                   choices=["last", "random"],
                   help="Which token position to perturb")
    p.add_argument("--max_seq_len", type=int, default=64,
                   help="Sequence length for queries (shorter = faster).")
    p.add_argument("--pretrain_steps", type=int, default=500,
                   help="KD pre-training steps for student prefix (logit-MSE)")
    p.add_argument("--pretrain_lr", type=float, default=1e-4)
    p.add_argument("--pretrain_batch_size", type=int, default=8)
    p.add_argument("--collect_batch_size", type=int, default=16)
    p.add_argument("--carlini_queries", type=int, default=4096,
                   help="Number of queries for Carlini SVD (usually more than base queries)")
    p.add_argument("--carlini_seq_len", type=int, default=128,
                   help="Seq length for Carlini SVD queries")
    p.add_argument("--output_dir", type=str, default="results/v5_attack_jacobian_fd")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow_synthetic", action="store_true",
                   help="Fall back to random tokens when WikiText is unavailable")
    p.add_argument("--dry_run_small", action="store_true",
                   help="Override to tiny sizes for smoke test")
    p.add_argument("--force_d_hat_to_hidden", action="store_true",
                   help="Force Carlini d_hat = hidden_size (skip gap detection). "
                        "Use when detected d_hat is off-by-one and the attack's "
                        "regression pipeline requires dim match.")
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
    cos = F.cosine_similarity(a_f, b_f, dim=-1)
    return float(cos.mean())


def rms_norm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm matching Qwen2RMSNorm (cast to float for stability)."""
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms * g.float()).to(x.dtype)


# ═════════════════════════════════════════════════════════════════════════════
# Query pool
# ═════════════════════════════════════════════════════════════════════════════

def build_query_pool(
    tokenizer,
    n: int,
    seq_len: int,
    seed: int,
    allow_synthetic: bool,
) -> torch.Tensor:
    """Build query token pool, preferring WikiText."""
    input_ids_list: list[torch.Tensor] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        for ex in ds:
            if len(input_ids_list) >= n:
                break
            text = ex.get("text", "")
            if len(text.strip()) < 20:
                continue
            tokens = tokenizer(
                text, max_length=seq_len, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            input_ids_list.append(tokens["input_ids"].squeeze(0))
    except Exception as e:
        if not allow_synthetic:
            raise RuntimeError(
                f"Dataset load failed: {e}. Use --allow_synthetic to fall back."
            ) from e
        logger.warning("Dataset load failed (%s) — using random tokens", e)

    remaining = n - len(input_ids_list)
    if remaining > 0:
        if not allow_synthetic:
            raise RuntimeError(
                f"Only {len(input_ids_list)}/{n} tokens loaded. "
                "Use --allow_synthetic to pad with random ids."
            )
        rng = torch.Generator().manual_seed(seed)
        random_ids = torch.randint(
            3, tokenizer.vocab_size, (remaining, seq_len), generator=rng
        )
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    return torch.stack(input_ids_list[:n])


# ═════════════════════════════════════════════════════════════════════════════
# Carlini SVD: recover W_eff = W_lm @ diag(g_final) up to orthogonal rotation
# ═════════════════════════════════════════════════════════════════════════════

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
    force_d_hat_to_hidden: bool = False,
) -> dict[str, torch.Tensor]:
    """Collect random-token logits, SVD, return top-d right singular vectors.

    Returns a dict with:
      * W_eff_hat : (d, V) recovered row-basis of W_eff (= Vh[:d, :])
      * S         : (min(N,V),) singular values (for diagnostic)
      * d_hat     : recovered hidden dim
    """
    logger.info("[Carlini] collecting %d random-token logits (seq_len=%d) ...",
                num_queries, seq_len)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    ys: list[torch.Tensor] = []
    pbar = tqdm(total=num_queries, desc="Carlini SVD queries", unit="q")
    n_done = 0
    while n_done < num_queries:
        bs = min(batch_size, num_queries - n_done)
        ids = torch.randint(0, vocab_size, (bs, seq_len), generator=rng, device="cpu").to(device)
        out = model(input_ids=ids)
        # Last-position logits (Carlini's simplest formulation)
        ys.append(out.logits[:, -1, :].float().cpu())
        n_done += bs
        pbar.update(bs)
    pbar.close()

    Y = torch.cat(ys, dim=0)                 # (N, V)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)
    logger.info("[Carlini] running SVD on Y_centered (%d x %d) ...", *Y_centered.shape)

    t0 = time.time()
    _, S, Vh = torch.linalg.svd(Y_centered, full_matrices=False)
    logger.info("[Carlini] SVD finished in %.1fs (shape S=%s, Vh=%s)",
                time.time() - t0, tuple(S.shape), tuple(Vh.shape))

    # Detect hidden dim via max gap ratio around true d
    S_np = S.numpy()
    gap = S_np[:-1] / np.maximum(S_np[1:], 1e-30)
    lo = max(0, hidden_size - 50)
    hi = min(len(gap), hidden_size + 50)
    d_hat_detected = int(lo + np.argmax(gap[lo:hi]))
    logger.info("[Carlini] d_hat_detected=%d (true=%d), gap_ratio=%.2f",
                d_hat_detected, hidden_size, gap[d_hat_detected])

    if force_d_hat_to_hidden:
        d_hat = int(hidden_size)
        logger.info("[Carlini] forcing d_hat=%d (= hidden_size) for dim-matched attack.", d_hat)
    else:
        d_hat = d_hat_detected

    W_eff_hat = Vh[:d_hat, :].float()         # (d, V)
    return {"W_eff_hat": W_eff_hat, "S": S, "d_hat": d_hat, "d_hat_detected": d_hat_detected}


def compare_w_eff(
    W_eff_hat: torch.Tensor,              # (d, V)
    W_lm_true: torch.Tensor,              # (V, d)
    g_final_true: torch.Tensor,           # (d,)
) -> dict[str, float]:
    """Compare recovered W_eff_hat (rotation-ambiguous) against ground truth
    W_lm @ diag(g_final) in subspace (principal-angle) sense."""
    W_eff_true = W_lm_true.float() * g_final_true.float().unsqueeze(0)   # (V, d)

    Q_hat, _ = torch.linalg.qr(W_eff_hat.T.float())          # (V, d)
    Q_true, _ = torch.linalg.qr(W_eff_true.float())          # (V, d)
    cos_ang = torch.linalg.svdvals(Q_hat.T @ Q_true).clamp(0.0, 1.0)

    return {
        "principal_angle_mean_cos": float(cos_ang.mean()),
        "principal_angle_min_cos": float(cos_ang.min()),
        "top_principal_angle_cosines": cos_ang[:5].tolist(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Recover h_L direction from logits: u(x) = pinv(W_eff) @ z(x)
# ═════════════════════════════════════════════════════════════════════════════

# NOTE: Carlini's SVD returns V_h[:d_hat, :] whose rows are orthonormal right
# singular vectors, so pinv(W_eff_hat) is just W_eff_hat itself.  The mapping
# z -> u = W_eff_hat @ z is therefore an ORTHOGONAL projection, and u only
# recovers h_L up to an unknown orthogonal rotation R (rotation ambiguity of
# Carlini's SVD).  The attack compensates for R by Procrustes-aligning
# teacher's u into the student's frame (see `rotate_teacher_to_student_basis`).


# ═════════════════════════════════════════════════════════════════════════════
# Student: same architecture, random init, trainable prefix via KD
# ═════════════════════════════════════════════════════════════════════════════

def make_student_from_config(teacher, device: torch.device):
    """Clone the teacher's config, instantiate fresh random weights."""
    from transformers import AutoModelForCausalLM
    cfg = copy.deepcopy(teacher.config)
    # Build from config -> random init
    student = AutoModelForCausalLM.from_config(cfg).to(device)
    student.train()
    return student


def copy_lm_head_and_final_norm(student, teacher) -> None:
    """Initialize student's lm_head and final norm to teacher values.

    This is ATTACKER-LEGAL knowledge: Carlini already gives us W_eff up to
    rotation, and in the paper we are asking: can we go BEYOND W_eff?  So we
    grant ourselves only W_eff-equivalent weights, not g_final separately.

    For POC simplicity we copy the exact values; in a real attack we would
    inject the rotation-ambiguous reconstruction and the analysis would apply
    in the rotated frame.  The cos comparisons we do later are NOT invariant
    to this, so note: copying the true W_lm and g_final is a STRONG help —
    failure under this help is extra-strong evidence of a pure-logits barrier.
    """
    with torch.no_grad():
        student.model.norm.weight.data.copy_(teacher.model.norm.weight.data)
        if teacher.config.tie_word_embeddings:
            # Tied: shared with embed_tokens
            student.model.embed_tokens.weight.data.copy_(
                teacher.model.embed_tokens.weight.data
            )
            # lm_head is tied — already shares buffer
        else:
            student.lm_head.weight.data.copy_(teacher.lm_head.weight.data)


def freeze_suffix_and_head(student, last_trainable_block: int = SECOND_LAST_IDX) -> None:
    """Freeze the LAST block and lm_head + final norm.  Train only prefix (0..22)."""
    # Freeze final norm + lm_head (and embeddings since tied)
    for p in student.model.norm.parameters():
        p.requires_grad_(False)
    if hasattr(student, "lm_head"):
        for p in student.lm_head.parameters():
            p.requires_grad_(False)
    # Freeze the last block (this is what we'll "attack" later)
    for p in student.model.layers[LAST_BLOCK_IDX].parameters():
        p.requires_grad_(False)
    # Leave embeddings trainable ONLY if not tied; tied ones share with head
    # but since lm_head weight is frozen, the embedding grad is also frozen.

    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    logger.info(
        "[student] trainable params: %d / %d  (%.1f%%)",
        n_train, n_total, 100.0 * n_train / max(n_total, 1),
    )


# ═════════════════════════════════════════════════════════════════════════════
# KD pretraining: align h_22_student to h_22_teacher via logit MSE
# ═════════════════════════════════════════════════════════════════════════════

def pretrain_student_kd(
    student,
    teacher,
    query_ids: torch.Tensor,       # (N, T)
    device: torch.device,
    steps: int,
    lr: float,
    batch_size: int,
    log_interval: int = 25,
) -> list[dict[str, float]]:
    """Pretrain student on teacher's last-position logit MSE.

    Because the last block and lm_head of the student are FROZEN to teacher
    values, driving the logits of the student toward the teacher's forces
    h_L_student -> h_L_teacher, and since the last block and final norm are
    shared, that in turn forces h_22_student -> h_22_teacher.  In the limit of
    perfect MSE=0, the finite-difference Jacobian we compute through the
    student becomes equivalent to the teacher's, at which point the residual
    encodes only mismatch that must come from the last block — which is zero.
    So we'll NEVER beat a perfect KD setup.

    The POC point: in PRACTICE MSE stays bounded away from zero, so a residual
    signal exists, and we test whether it correlates with last-block params.
    """
    if steps <= 0:
        logger.info("[pretrain] steps=0, skipping KD pretraining.")
        return []

    student.train()
    teacher.eval()

    # Only pass trainable params to the optimizer
    trainable = [p for p in student.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps, eta_min=lr * 0.1)

    N = query_ids.shape[0]
    rng = np.random.default_rng(seed=0xC0FFEE)

    history: list[dict[str, float]] = []
    t0 = time.time()

    for step in range(steps):
        idx = rng.choice(N, size=batch_size, replace=False)
        batch = query_ids[idx].to(device)

        with torch.no_grad():
            t_out = teacher(input_ids=batch)
            t_logits = t_out.logits.float()         # (B, T, V)

        s_out = student(input_ids=batch)
        s_logits = s_out.logits.float()

        # Focus on last-position MSE (simpler signal, faster)
        loss_last = F.mse_loss(s_logits[:, -1, :], t_logits[:, -1, :])
        # Light full-seq MSE to regularize
        loss_full = F.mse_loss(s_logits, t_logits)
        loss = loss_last + 0.1 * loss_full

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        sched.step()

        if step % log_interval == 0 or step == steps - 1:
            entry = {
                "step": int(step),
                "loss_last": float(loss_last.detach()),
                "loss_full": float(loss_full.detach()),
                "loss_total": float(loss.detach()),
                "lr": float(optim.param_groups[0]["lr"]),
                "elapsed": time.time() - t0,
            }
            history.append(entry)
            logger.info(
                "[pretrain] step %4d/%d  loss_last=%.4e  loss_full=%.4e  lr=%.2e",
                step, steps, entry["loss_last"], entry["loss_full"], entry["lr"],
            )

    student.eval()
    return history


# ═════════════════════════════════════════════════════════════════════════════
# Collection: run teacher + student on base & perturbed queries, capture
# logits AND (for the student) full hidden states + last-block oracle signals.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CollectedState:
    """Everything we gather from running teacher & student on queries.

    To keep memory bounded, we project teacher/student last-position logits
    through pinv(W_eff) on-the-fly and store only the resulting u vectors
    (hidden dim).  Full logits are NEVER materialized for N*P combinations.
    """
    # Teacher (black-box-observable only) projected through pinv(W_eff)
    u_teacher_base: torch.Tensor           # (N, d_hat)       u = pinv @ z
    u_teacher_pert: torch.Tensor           # (N, P, d_hat)

    # Student (white-box) projected through pinv(W_eff)
    u_student_base: torch.Tensor           # (N, d_hat)
    u_student_pert: torch.Tensor           # (N, P, d_hat)

    # h_22_student (block-22 output = last-block input) at base & perturbed
    h22_student_base: torch.Tensor         # (N, D)
    h22_student_pert: torch.Tensor         # (N, P, D)
    # h_L_student (pre final-norm, i.e. block-23 output) at last position
    hL_student_base: torch.Tensor          # (N, D)
    hL_student_pert: torch.Tensor          # (N, P, D)

    # Metadata
    perturb_positions: torch.Tensor        # (N, P)   which token was changed
    new_token_ids: torch.Tensor            # (N, P)   id we replaced with

    # Oracle-only (for evaluation; NEVER used in attack logic)
    h22_teacher_base: torch.Tensor         # (N, D)
    h22_teacher_pert: torch.Tensor         # (N, P, D)
    hL_teacher_base: torch.Tensor          # (N, D)
    hL_teacher_pert: torch.Tensor          # (N, P, D)


def _register_block_output_hook(layer, buffer: list):
    """Register a forward hook capturing a block's (last-pos) hidden output."""
    def hook(_module, _inputs, outputs):
        # Qwen decoder layer returns (hidden_states, ...)
        hs = outputs[0] if isinstance(outputs, tuple) else outputs
        # Keep last position only to save memory
        buffer.append(hs[:, -1, :].detach().float().cpu())
    return layer.register_forward_hook(hook)


@torch.no_grad()
def run_one_batch(model, input_ids, hooks_flag: bool):
    """Forward pass returning outputs (.logits, .hidden_states) if hooks_flag.

    Actually we rely on output_hidden_states=True to collect block boundaries
    and pair it with a hook on block 22 for the AFTER-block-22 output.
    """
    out = model(input_ids=input_ids, output_hidden_states=hooks_flag, return_dict=True)
    return out


@torch.no_grad()
def collect_states(
    teacher,
    student,
    query_ids: torch.Tensor,                    # (N, T)
    perturb_positions: torch.Tensor,            # (N, P)
    new_token_ids: torch.Tensor,                # (N, P)
    W_eff_hat: torch.Tensor,                    # (d_hat, V)  for on-the-fly projection
    device: torch.device,
    batch_size: int,
) -> CollectedState:
    """Run teacher & student on every base and perturbed query.

    Project logits to u-space on-GPU (saves ~170x memory vs storing raw logits).
    Stores only last-position hidden states at block 22 and block 23 to keep
    memory bounded at O(N*P*D_MODEL) instead of O(N*P*VOCAB_SIZE).

    Perturbation: for each base x_n, for each p in [0..P-1], we build x'_np
    by copying x_n and replacing token at perturb_positions[n, p] with
    new_token_ids[n, p].
    """
    N, T = query_ids.shape
    _, P = new_token_ids.shape
    d_hat = W_eff_hat.shape[0]
    pinv_map = W_eff_hat.to(device).float()                 # (d_hat, V)

    u_t_base = torch.zeros(N, d_hat)
    u_s_base = torch.zeros(N, d_hat)
    h22_t_base = torch.zeros(N, D_MODEL)
    h22_s_base = torch.zeros(N, D_MODEL)
    hL_t_base = torch.zeros(N, D_MODEL)
    hL_s_base = torch.zeros(N, D_MODEL)

    u_t_pert = torch.zeros(N, P, d_hat)
    u_s_pert = torch.zeros(N, P, d_hat)
    h22_t_pert = torch.zeros(N, P, D_MODEL)
    h22_s_pert = torch.zeros(N, P, D_MODEL)
    hL_t_pert = torch.zeros(N, P, D_MODEL)
    hL_s_pert = torch.zeros(N, P, D_MODEL)

    teacher.eval()
    student.eval()

    def _extract(model_out, pinv_map_dev: torch.Tensor):
        """Return (h22, hL, u) for last-pos tokens given model forward output."""
        hs = model_out.hidden_states
        h22 = hs[SECOND_LAST_IDX + 1][:, -1, :]
        hL = hs[LAST_BLOCK_IDX + 1][:, -1, :]
        z = model_out.logits[:, -1, :].float()           # (B, V)
        u = z @ pinv_map_dev.T                            # (B, d_hat)
        return h22.float().cpu(), hL.float().cpu(), u.float().cpu()

    # ── base queries ────────────────────────────────────────────────────────
    logger.info("[collect] base queries (N=%d) ...", N)
    pbar = tqdm(total=N, desc="base")
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        batch = query_ids[s:e].to(device)

        t_out = teacher(input_ids=batch, output_hidden_states=True)
        h22_t, hL_t, u_t = _extract(t_out, pinv_map)

        s_out = student(input_ids=batch, output_hidden_states=True)
        h22_s, hL_s, u_s = _extract(s_out, pinv_map)

        u_t_base[s:e] = u_t
        u_s_base[s:e] = u_s
        h22_t_base[s:e] = h22_t
        h22_s_base[s:e] = h22_s
        hL_t_base[s:e] = hL_t
        hL_s_base[s:e] = hL_s

        pbar.update(e - s)
    pbar.close()

    # ── perturbed queries ───────────────────────────────────────────────────
    logger.info("[collect] perturbed queries (N*P=%d) ...", N * P)
    pbar = tqdm(total=N * P, desc="pert")
    for p_idx in range(P):
        perturbed = query_ids.clone()                       # (N, T)
        pos = perturb_positions[:, p_idx]                   # (N,)
        tid = new_token_ids[:, p_idx]                       # (N,)
        perturbed[torch.arange(N), pos] = tid

        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            batch = perturbed[s:e].to(device)

            t_out = teacher(input_ids=batch, output_hidden_states=True)
            h22_t, hL_t, u_t = _extract(t_out, pinv_map)

            s_out = student(input_ids=batch, output_hidden_states=True)
            h22_s, hL_s, u_s = _extract(s_out, pinv_map)

            u_t_pert[s:e, p_idx] = u_t
            u_s_pert[s:e, p_idx] = u_s
            h22_t_pert[s:e, p_idx] = h22_t
            h22_s_pert[s:e, p_idx] = h22_s
            hL_t_pert[s:e, p_idx] = hL_t
            hL_s_pert[s:e, p_idx] = hL_s

            pbar.update(e - s)
    pbar.close()

    return CollectedState(
        u_teacher_base=u_t_base,
        u_teacher_pert=u_t_pert,
        u_student_base=u_s_base,
        u_student_pert=u_s_pert,
        h22_student_base=h22_s_base,
        h22_student_pert=h22_s_pert,
        hL_student_base=hL_s_base,
        hL_student_pert=hL_s_pert,
        perturb_positions=perturb_positions,
        new_token_ids=new_token_ids,
        h22_teacher_base=h22_t_base,
        h22_teacher_pert=h22_t_pert,
        hL_teacher_base=hL_t_base,
        hL_teacher_pert=hL_t_pert,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Attack: accumulate finite-difference Jacobian samples and try to recover
# last-block params from residuals.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AttackDiagnostics:
    """Intermediate numbers we report regardless of attack success."""
    # Step 1: how good is u (direction of h_L) from logits?
    u_cosine_to_h_L_teacher: float        # cos(u_t, h_L_t / ||h_L_t||)
    u_cosine_to_h_L_student: float

    # Step 2: how good are finite-difference directions?
    Du_teacher_cos_to_DhL_teacher: float  # cos(Du_t, D(h_L_t))
    Du_student_cos_to_DhL_student: float

    # Step 3: HOW DIFFERENT are h_22 between teacher and student?
    # If close -> prefix alignment worked.  If far -> prefix barrier dominates.
    h22_diff_relative_norm: float         # mean ||h22_t - h22_s|| / ||h22_t||
    h22_cos_teacher_student: float        # mean cos(h22_t, h22_s)

    # Step 4: residual signal — Du_teacher - Du_student *after* rotating
    # the teacher's direction into the student's basis (via best orthogonal
    # alignment).
    residual_norm: float
    residual_to_signal_ratio: float       # ||residual|| / ||Du_teacher||


def rotate_teacher_to_student_basis(
    u_teacher: torch.Tensor,            # (..., d) in rotated Carlini frame
    u_teacher_ref: torch.Tensor,        # (..., d) vectors we also have for student
    u_student_ref: torch.Tensor,        # (..., d) student's version of the same
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the best orthogonal R such that u_student_ref ≈ u_teacher_ref @ R.
    Apply R to u_teacher so we can compare to student quantities directly.

    Returns (u_teacher_aligned, R).
    """
    # Flatten to (M, d)
    orig_shape = u_teacher.shape
    d = orig_shape[-1]

    A = u_teacher_ref.reshape(-1, d).double()
    B = u_student_ref.reshape(-1, d).double()

    # Solve procrustes: min ||A R - B||_F  =>  R = V W^T where
    # A^T B = V S W^T
    cross = A.T @ B                                # (d, d)
    U_p, _, Vh_p = torch.linalg.svd(cross, full_matrices=False)
    R = (U_p @ Vh_p).float()                        # (d, d), orthogonal

    u_aligned = (u_teacher.reshape(-1, d).float() @ R).reshape(orig_shape)
    return u_aligned, R


def run_attack(
    state: CollectedState,
    W_eff_hat: torch.Tensor,            # (d, V) Carlini-recovered row basis
    teacher,
    student,
    device: torch.device,
) -> tuple[AttackDiagnostics, dict[str, float]]:
    """Main attack: attempt to recover the last block's W_O (simplest target).

    The algebra
    -----------
    For a Qwen block, the output at last position obeys
        h_L = h_22 + attn_out(h_22) + mlp_out(h_22 + attn_out(h_22))
    where attn_out involves the block's q/k/v/o projections, and mlp_out
    involves gate/up/down.  At a fixed x, the block computes a deterministic
    function of h_22.  So
        h_L_teacher(x)  = Block_teacher(h22_teacher(x))
        h_L_student(x)  = Block_student(h22_student(x))

    After KD pretraining we HOPE h22_student ≈ h22_teacher.  Let
        delta_h22(x) = h22_teacher(x) - h22_student(x)     (small)
        delta_block  = Block_teacher(·) - Block_student(·) (unknown)
    Then
        h_L_teacher - h_L_student
          = (Block_t - Block_s)(h22_t)
            + Block_s(h22_t) - Block_s(h22_s)
          = delta_block(h22_t) + J_block_s(h22_s) @ delta_h22  +  O(||delta_h22||^2)

    In a pure-logits attack we have u_teacher (=h_L_teacher / ||h_L_teacher||
    up to rotation R).  We first align R via orthogonal Procrustes on u_base
    pairs.  Then if ||delta_h22|| is small and we can ESTIMATE h22_teacher
    (via student's h22 as a proxy), we can probe delta_block by solving
        u_teacher(x) - u_student(x) - J_block_s(h22_s) @ (h22_t_est - h22_s) \approx delta_block(h22_s).

    The cleanest test: do the residual vectors {u_t_rot(x) - u_s(x) - J_s(x)(h22_t - h22_s)}
    have SIGNIFICANT variance in a direction aligned with (Block_teacher -
    Block_student)(h22_s)?  If yes, regressing on h22_s should recover a
    fraction of delta_block, which can be added to student's weights to yield
    a teacher-weight estimate.

    This function: (a) computes all the diagnostics, (b) performs a simple
    linear regression h_L_teacher ~ Block(h_22_student; weights) targeting
    only W_O (the output proj of the last block's attention) to test
    whether any cos > 0.2 emerges.  Other matrices listed in the result dict.
    """
    logger.info("=" * 70)
    logger.info("ATTACK: Finite-Difference Jacobian")
    logger.info("=" * 70)

    # ── Step 1: read u = pinv(W_eff) @ z (already computed on-the-fly) ─────
    u_t_base = state.u_teacher_base
    u_s_base = state.u_student_base
    u_t_pert = state.u_teacher_pert
    u_s_pert = state.u_student_pert

    # Compare to the TRUE hL direction (oracle only, for diagnostics)
    # Guard for Carlini-detected dim d_u != true residual dim d_h
    d_u_diag = u_t_base.shape[-1]
    d_h_diag = state.hL_teacher_base.shape[-1]
    if d_u_diag == d_h_diag:
        hL_t_norm = F.normalize(state.hL_teacher_base, dim=-1)
        u_t_norm = F.normalize(u_t_base, dim=-1)
        u_cos_t = float(F.cosine_similarity(u_t_norm, hL_t_norm, dim=-1).mean())
        hL_s_norm = F.normalize(state.hL_student_base, dim=-1)
        u_s_norm = F.normalize(u_s_base, dim=-1)
        u_cos_s = float(F.cosine_similarity(u_s_norm, hL_s_norm, dim=-1).mean())
    else:
        logger.warning(
            "Carlini d_hat=%d != true residual dim=%d; "
            "oracle u-vs-hL cosine check skipped.", d_u_diag, d_h_diag,
        )
        u_cos_t = float("nan")
        u_cos_s = float("nan")

    logger.info("[attack] u_teacher aligned to h_L_teacher direction: mean cos = %.4f", u_cos_t)
    logger.info("[attack] u_student aligned to h_L_student direction: mean cos = %.4f", u_cos_s)

    # ── Step 2: align teacher's rotated frame to student via Procrustes ────
    # We compute R s.t. u_t_base @ R ≈ u_s_base (in expectation when
    # teacher≈student).  Of course if teacher and student diverge this is an
    # approximation; it isolates the rotation amb. of the Carlini recovery.
    u_t_base_rot, R_carlini = rotate_teacher_to_student_basis(
        u_t_base, u_t_base, u_s_base
    )
    u_t_pert_rot = (u_t_pert.reshape(-1, u_t_pert.shape[-1]).float() @ R_carlini).reshape(
        u_t_pert.shape
    )

    # ── Step 3: finite differences ──────────────────────────────────────────
    # Du: (N, P, d) = u_pert - u_base[:, None, :]
    Du_t_rot = u_t_pert_rot - u_t_base_rot.unsqueeze(1)      # rotated teacher
    Du_s = u_s_pert - u_s_base.unsqueeze(1)

    # Oracle Du for teacher/student (true hidden-state differences)
    DhL_t = state.hL_teacher_pert - state.hL_teacher_base.unsqueeze(1)
    DhL_s = state.hL_student_pert - state.hL_student_base.unsqueeze(1)
    Dh22_t = state.h22_teacher_pert - state.h22_teacher_base.unsqueeze(1)
    Dh22_s = state.h22_student_pert - state.h22_student_base.unsqueeze(1)

    # Per-sample cosines (flatten N*P).  d_u = Carlini's recovered dim
    # (should equal D_MODEL in practice); d_h = true hidden dim from model.
    N, P, d_u = Du_t_rot.shape
    d_h = state.hL_teacher_base.shape[-1]
    if d_u != d_h:
        logger.warning(
            "Carlini d_hat=%d != true hidden dim=%d; oracle Du-cosine checks skipped.",
            d_u, d_h,
        )
    Du_t_flat = Du_t_rot.reshape(N * P, d_u)
    Du_s_flat = Du_s.reshape(N * P, d_u)

    if d_u == d_h:
        DhL_t_flat = DhL_t.reshape(N * P, d_h)
        DhL_s_flat = DhL_s.reshape(N * P, d_h)
        cos_Du_t = float(F.cosine_similarity(
            Du_t_flat, F.normalize(DhL_t_flat, dim=-1), dim=-1
        ).mean())
        cos_Du_s = float(F.cosine_similarity(
            Du_s_flat, F.normalize(DhL_s_flat, dim=-1), dim=-1
        ).mean())
    else:
        cos_Du_t = float("nan")
        cos_Du_s = float("nan")

    d = d_u  # used below for jacobian regression; we assume d_u == d_h from here

    logger.info("[attack] Du_teacher vs D(h_L_teacher) dir cos: %.4f (oracle check)", cos_Du_t)
    logger.info("[attack] Du_student vs D(h_L_student) dir cos: %.4f (oracle check)", cos_Du_s)

    # ── Step 4: how bad is the h22 mismatch after KD? ──────────────────────
    h22_diff = state.h22_teacher_base - state.h22_student_base
    rel = (h22_diff.norm(dim=-1) / state.h22_teacher_base.norm(dim=-1).clamp(min=1e-12))
    h22_rel = float(rel.mean())
    h22_cos = float(F.cosine_similarity(
        state.h22_teacher_base, state.h22_student_base, dim=-1
    ).mean())
    logger.info("[attack] h22 teacher vs student: mean cos=%.4f, mean rel-norm-diff=%.4f",
                h22_cos, h22_rel)

    # ── Step 5: residual signal ─────────────────────────────────────────────
    # Residual at last position:  u_t_rot(x) - u_s(x), compared to its
    # predictable-from-h22 part.  If prefix mismatch dominated, the residual
    # aligns with Dh22 (via J_block_s).  If the block mismatch contributes,
    # the residual should have components in h22_s-aligned directions that
    # CANNOT be explained by Dh22 alone.
    residual = u_t_base_rot - u_s_base                   # (N, d)
    # Ratio
    residual_ratio = float(
        residual.norm(dim=-1).mean() / u_t_base_rot.norm(dim=-1).clamp(min=1e-12).mean()
    )
    logger.info("[attack] residual ||u_t_rot - u_s|| / ||u_t_rot|| (mean): %.4f",
                residual_ratio)

    diag = AttackDiagnostics(
        u_cosine_to_h_L_teacher=u_cos_t,
        u_cosine_to_h_L_student=u_cos_s,
        Du_teacher_cos_to_DhL_teacher=cos_Du_t,
        Du_student_cos_to_DhL_student=cos_Du_s,
        h22_diff_relative_norm=h22_rel,
        h22_cos_teacher_student=h22_cos,
        residual_norm=float(residual.norm(dim=-1).mean()),
        residual_to_signal_ratio=residual_ratio,
    )

    # ── Step 6: Actual parameter-recovery attack ────────────────────────────
    # We attack the LAST block's down_proj (W_down, shape (d, d_ff)) and
    # output proj (W_O, shape (d, d)) — the two output-side matrices.
    # Strategy:
    #   Given rotated  u_t_rot(x) as a proxy for h_L_teacher / ||h_L_teacher||,
    #   we have the observational model
    #       h_L_teacher = a(x) * u_t_rot_hat(x)              (a unknown scalar)
    #   Student's block takes h22_s(x) and outputs h_L_s(x).  We want to
    #   fit a linear map   h_L_teacher ≈ BlockApprox(h22_s)   where BlockApprox
    #   is LINEAR in the last-block-output weights we want to recover.
    #   Since the Block function is highly non-linear in its params, we
    #   instead probe via *finite differences*.  The differential of the block
    #   w.r.t. h22 at the student's h22 is J_block_s(h22_s).  Teacher's is
    #   J_block_t(h22_t).  From finite diffs:
    #       Du_t_rot(x, x')  =  J_block_t(h22_t(x)) @ Dh22_t(x, x') + O(||Dh22||^2)
    #   If we assume h22_t ≈ h22_s (good KD), and we HAVE Dh22_s (computable),
    #   then Du_t_rot ≈ J_block_t(h22_s) @ Dh22_s.
    #   Similarly Du_s ≈ J_block_s(h22_s) @ Dh22_s.
    #   The DIFFERENCE DJ = J_block_t - J_block_s satisfies
    #       Du_t_rot - Du_s ≈ DJ(h22_s) @ Dh22_s.
    #   For many (x, x') pairs we can regress DJ as an unknown (d x d) linear
    #   operator.  This gives us the DIFFERENCE between teacher and student
    #   block jacobians averaged across the input distribution.  From DJ we
    #   can, IN PRINCIPLE, back out at least leading-order deltas in block
    #   weights.  POC: we solve the linear system and compare to the true
    #   jacobian difference as a sanity check.

    logger.info("[attack] Regressing DJ ≈ (Du_t_rot - Du_s) (Dh22_s)+  ...")
    if d_u != d_h:
        logger.error(
            "Jacobian regression requires d_u == d_h (Carlini recovered hidden "
            "dim = true hidden dim).  Got d_u=%d, d_h=%d.  Skipping regression.",
            d_u, d_h,
        )
        # Early-out with default / NaN metrics so the script still runs
        # through the verdict step.
        diag = AttackDiagnostics(
            u_cosine_to_h_L_teacher=u_cos_t,
            u_cosine_to_h_L_student=u_cos_s,
            Du_teacher_cos_to_DhL_teacher=cos_Du_t,
            Du_student_cos_to_DhL_student=cos_Du_s,
            h22_diff_relative_norm=h22_rel,
            h22_cos_teacher_student=h22_cos,
            residual_norm=float(residual.norm(dim=-1).mean()),
            residual_to_signal_ratio=residual_ratio,
        )
        attack_metrics = {
            "u_cosine_to_h_L_teacher": u_cos_t,
            "u_cosine_to_h_L_student": u_cos_s,
            "Du_teacher_cos_to_DhL_teacher": cos_Du_t,
            "Du_student_cos_to_DhL_student": cos_Du_s,
            "h22_cos_teacher_student": h22_cos,
            "h22_rel_norm_diff": h22_rel,
            "residual_ratio": residual_ratio,
            "J_t_fit_cos": float("nan"),
            "J_s_fit_cos": float("nan"),
            "DJ_fit_cos": float("nan"),
            "cos_DJt_DJs": float("nan"),
            "cos_DJest_vs_DJt": float("nan"),
            "principal_angles_Wo_teacher_mean_cos": float("nan"),
            "principal_angles_Wo_student_mean_cos": float("nan"),
            "principal_angles_Wdown_teacher_mean_cos": float("nan"),
            "principal_angles_Wdown_student_mean_cos": float("nan"),
            "best_naive_W_O_cos": float("nan"),
            "best_naive_W_O_alpha": float("nan"),
            "baseline_student_W_O_cos": float("nan"),
            "baseline_student_W_down_cos": float("nan"),
            "dim_mismatch": True,
        }
        return diag, attack_metrics

    Du_t_all = Du_t_rot.reshape(N * P, d).double()
    Du_s_all = Du_s.reshape(N * P, d).double()
    Dh22_s_all = Dh22_s.reshape(N * P, d).double()
    Diff_Du = Du_t_all - Du_s_all                         # (M, d)

    # Also regress Du_t_rot directly, giving us an estimate of J_block_t.
    # We solve  J_t @ Dh22^T ≈ Du_t_rot^T  => J_t = (Dh22^T Dh22)^-1 Dh22^T Du
    gram = Dh22_s_all.T @ Dh22_s_all                      # (d, d)
    gram += 1e-6 * gram.diagonal().mean() * torch.eye(d, dtype=torch.float64)
    gram_inv = torch.linalg.inv(gram)
    # note: we want J_t s.t. Du ≈ Dh22 @ J_t^T  (since each row is a sample)
    J_t_est = (Dh22_s_all.T @ Du_t_all)                   # (d, d)
    J_t_est = (gram_inv @ J_t_est).T                      # (d, d)  row-vec mapping
    # i.e. for row-vectors:  du = dh22 @ J_t_est^T

    J_s_est = (gram_inv @ (Dh22_s_all.T @ Du_s_all)).T     # (d, d)

    DJ_est = J_t_est - J_s_est                             # recovered Δ-jacobian
    # Residual fit quality
    pred_Du_t = (Dh22_s_all @ J_t_est.T)                  # (M, d)
    pred_Du_s = (Dh22_s_all @ J_s_est.T)
    fit_t = float(F.cosine_similarity(
        pred_Du_t.float().flatten(0).unsqueeze(0),
        Du_t_all.float().flatten(0).unsqueeze(0),
    ).item()) if pred_Du_t.numel() > 0 else float("nan")
    fit_s = float(F.cosine_similarity(
        pred_Du_s.float().flatten(0).unsqueeze(0),
        Du_s_all.float().flatten(0).unsqueeze(0),
    ).item()) if pred_Du_s.numel() > 0 else float("nan")
    fit_diff = float(F.cosine_similarity(
        (pred_Du_t - pred_Du_s).float().flatten(0).unsqueeze(0),
        Diff_Du.float().flatten(0).unsqueeze(0),
    ).item()) if pred_Du_t.numel() > 0 else float("nan")
    logger.info("[attack] J_t_est fit-cos (pred Du_t vs Du_t):   %.4f", fit_t)
    logger.info("[attack] J_s_est fit-cos (pred Du_s vs Du_s):   %.4f", fit_s)
    logger.info("[attack] DJ_est   fit-cos (diff pred vs diff):  %.4f", fit_diff)

    # Compute TRUE jacobians J_t, J_s at a single reference point via
    # torch.func (oracle) for comparison.  Only when d is small; here d=896
    # we can afford it batched at a few points.
    logger.info("[attack] computing ORACLE last-block Jacobians at sample points ...")
    try:
        from torch.func import jacrev
    except ImportError:
        jacrev = None

    oracle_metrics: dict[str, Any] = {}
    if jacrev is not None:
        # Build a callable that runs the last block + final norm for a single
        # (1, d) input  h22.  For teacher/student separately.
        teacher.eval(); student.eval()
        with torch.no_grad():
            last_t = teacher.model.layers[LAST_BLOCK_IDX]
            last_s = student.model.layers[LAST_BLOCK_IDX]

            # We approximate block as f(h22) = last_layer(h22)[0] at the final
            # position, assuming KV cache / positional effects are constant.
            # This is rough for a POC — the block is position-aware, but at
            # the LAST token position, with the rest of the sequence fixed,
            # it behaves as a nonlinear map in h22[-1].
            # To keep the POC tractable we estimate J_t by central differences
            # on small random directions at a reference point.

            n_probe = min(16, N)
            ref_idx = torch.randperm(N)[:n_probe]
            h22_refs = state.h22_teacher_base[ref_idx].to(device).float()  # (nr, d)

            # We do a finite-difference estimate of J at each h22_ref using
            # the FORWARD PROCEDURE on the real sequence (the block output at
            # last position responds non-trivially to changes in the last-pos
            # residual).  However we don't have the full sequence hidden
            # states stored here; so we skip this exact oracle and rely on
            # the regressed J_t_est quality as a proxy.
            # (This is fine for the POC — we document that J_t_est fit-cos
            # approaches 1.0 when the linearization is accurate.)
            oracle_metrics["oracle_jacobian_note"] = (
                "Skipped full-oracle last-block Jacobian: requires full-"
                "sequence rollout per probe.  J_t_est fit-cos is the "
                "linearization-quality proxy."
            )

    # ── Step 7: From J_t_est, try to read off last-block W_O / W_down ──────
    # The last block maps h22 -> h_L via residual pattern:
    #   y1 = h22 + attn_out(ln(h22)); y2 = y1 + mlp_out(ln(y1)); h_L = y2
    # The Jacobian at h22 for fixed context is:
    #   J = I + dAttn/dh22 + dMLP/dy1 @ (I + dAttn/dh22)
    #     = I + J_attn + J_mlp @ (I + J_attn)
    # where J_attn involves W_q,W_k,W_v,W_O and J_mlp involves W_gate,W_up,W_down.
    # We can't easily disentangle these from J alone, BUT the trace of J - I
    # tells us sum over blocks of the attention+mlp contributions at that
    # point.
    # A crude test: compute cos between (J_t_est - I)  and (J_s_est - I).
    I_d = torch.eye(d, dtype=torch.float64)
    DJ_t = (J_t_est - I_d).float()
    DJ_s = (J_s_est - I_d).float()
    cos_DJt_DJs = flat_cos(DJ_t, DJ_s)
    logger.info("[attack] cos(J_t - I, J_s - I):   %.4f", cos_DJt_DJs)
    logger.info("[attack] cos(DJ, DJ_t):           %.4f", flat_cos(DJ_est.float(), DJ_t))

    # Now extract an ESTIMATE of the last block's W_O via a structured probe.
    # For the attention part, the map  x -> W_O @ (attn_mixed_values)  means
    # the LEFT singular directions of J_attn - part are column-space of W_O.
    # We compute the SVD of (J_t_est - I) and compare its top directions with
    # the true W_O's column space.
    teacher_last = teacher.model.layers[LAST_BLOCK_IDX]
    student_last = student.model.layers[LAST_BLOCK_IDX]
    W_O_true = teacher_last.self_attn.o_proj.weight.detach().float().cpu()    # (d, d)
    W_O_student = student_last.self_attn.o_proj.weight.detach().float().cpu()
    W_down_true = teacher_last.mlp.down_proj.weight.detach().float().cpu()    # (d, d_ff)
    W_down_student = student_last.mlp.down_proj.weight.detach().float().cpu()

    # Subspace alignment of (J_t_est - I)'s column space to W_O columns.
    U_jt, _, _ = torch.linalg.svd(DJ_t, full_matrices=False)
    U_jt = U_jt[:, :64]                                # (d, 64)
    Q_Wo_true, _ = torch.linalg.qr(W_O_true)            # (d, d)
    Q_Wo_student, _ = torch.linalg.qr(W_O_student)
    # principal angles
    cos_angles_t = torch.linalg.svdvals(U_jt.T @ Q_Wo_true).clamp(0, 1)
    cos_angles_s = torch.linalg.svdvals(U_jt.T @ Q_Wo_student).clamp(0, 1)

    logger.info("[attack] Top-64 directions of J_t_est-I vs col(W_O_teacher): mean cos = %.4f",
                cos_angles_t.mean().item())
    logger.info("[attack] Top-64 directions of J_t_est-I vs col(W_O_student): mean cos = %.4f",
                cos_angles_s.mean().item())

    # Similarly W_down subspace check
    Q_Wd_true, _ = torch.linalg.qr(W_down_true)         # (d, d_ff) -> Q is (d, d)
    Q_Wd_student, _ = torch.linalg.qr(W_down_student)
    cos_angles_down_t = torch.linalg.svdvals(U_jt.T @ Q_Wd_true).clamp(0, 1)
    cos_angles_down_s = torch.linalg.svdvals(U_jt.T @ Q_Wd_student).clamp(0, 1)

    # Also: the NAIVE attack — pretend we can read off W_O by replacing
    # student's W_O with something derived from DJ_est.  A simple rule:
    #   W_O_recovered = W_O_student + alpha * ProjectOntoColSpace(DJ_est)
    # then compare cos(W_O_recovered, W_O_true).  Try multiple alphas.
    best_W_O_cos = -1.0
    best_alpha = 0.0
    for alpha in [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        W_O_guess = W_O_student + alpha * DJ_est.float()        # (d, d)
        c = flat_cos(W_O_guess, W_O_true)
        if c > best_W_O_cos:
            best_W_O_cos = c
            best_alpha = alpha

    # And W_down (d x d_ff): reduce DJ_est to fit the d_ff dimension.
    # We can't straight add DJ_est (d x d) to W_down (d x d_ff), so we only
    # use the subspace-alignment diagnostic for W_down.

    logger.info("[attack] Naive W_O recovery:  best_cos=%.4f at alpha=%.2f  "
                "(student baseline cos vs true: %.4f)",
                best_W_O_cos, best_alpha, flat_cos(W_O_student, W_O_true))

    attack_metrics = {
        "u_cosine_to_h_L_teacher": u_cos_t,
        "u_cosine_to_h_L_student": u_cos_s,
        "Du_teacher_cos_to_DhL_teacher": cos_Du_t,
        "Du_student_cos_to_DhL_student": cos_Du_s,
        "h22_cos_teacher_student": h22_cos,
        "h22_rel_norm_diff": h22_rel,
        "residual_ratio": residual_ratio,
        "J_t_fit_cos": fit_t,
        "J_s_fit_cos": fit_s,
        "DJ_fit_cos": fit_diff,
        "cos_DJt_DJs": cos_DJt_DJs,
        "cos_DJest_vs_DJt": flat_cos(DJ_est.float(), DJ_t),
        "principal_angles_Wo_teacher_mean_cos": float(cos_angles_t.mean()),
        "principal_angles_Wo_student_mean_cos": float(cos_angles_s.mean()),
        "principal_angles_Wdown_teacher_mean_cos": float(cos_angles_down_t.mean()),
        "principal_angles_Wdown_student_mean_cos": float(cos_angles_down_s.mean()),
        "best_naive_W_O_cos": best_W_O_cos,
        "best_naive_W_O_alpha": best_alpha,
        "baseline_student_W_O_cos": flat_cos(W_O_student, W_O_true),
        "baseline_student_W_down_cos": flat_cos(W_down_student, W_down_true),
    }
    return diag, attack_metrics


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    if args.dry_run_small:
        args.num_queries = 16
        args.perturbations_per_query = 2
        args.pretrain_steps = 4
        args.carlini_queries = 256
        args.carlini_seq_len = 32
        args.max_seq_len = 16
        args.collect_batch_size = 4
        args.pretrain_batch_size = 2

    # Respect HF offline / mirror settings from env
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    device = torch.device(args.device)
    if "cuda" in str(device) and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — falling back to CPU (this will be slow)")
        device = torch.device("cpu")

    logger.info("=" * 70)
    logger.info("FINITE-DIFFERENCE JACOBIAN ATTACK (POC)")
    logger.info("=" * 70)
    logger.info("Model:               %s", args.model_name)
    logger.info("Device:              %s", device)
    logger.info("Num queries:         %d", args.num_queries)
    logger.info("Perturbations/query: %d", args.perturbations_per_query)
    logger.info("KD pretrain steps:   %d", args.pretrain_steps)
    logger.info("Seed:                %d", args.seed)
    logger.info("Output dir:          %s", out_dir)

    # ── Load teacher ────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32
    logger.info("[setup] loading teacher ...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    teacher.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = teacher.config.vocab_size
    hidden_size = teacher.config.hidden_size
    assert vocab_size == VOCAB_SIZE, f"Expected V={VOCAB_SIZE}, got {vocab_size}"
    assert hidden_size == D_MODEL, f"Expected d={D_MODEL}, got {hidden_size}"
    logger.info("[setup] teacher loaded: vocab=%d, d=%d, layers=%d",
                vocab_size, hidden_size, teacher.config.num_hidden_layers)

    # ── Carlini SVD ─────────────────────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("STAGE 1: Carlini SVD to recover W_eff = W_lm @ diag(g_final)")
    logger.info("-" * 70)
    carlini_out = run_carlini_svd(
        teacher, vocab_size, hidden_size,
        num_queries=args.carlini_queries,
        seq_len=args.carlini_seq_len,
        device=device,
        batch_size=args.collect_batch_size,
        seed=args.seed,
        force_d_hat_to_hidden=args.force_d_hat_to_hidden,
    )
    W_eff_hat = carlini_out["W_eff_hat"]            # (d, V), cpu
    d_hat = carlini_out["d_hat"]

    # Oracle diagnostic: how well did SVD recover W_eff?
    W_lm_true = teacher.lm_head.weight.detach().float().cpu()                    # (V, d)
    g_final_true = teacher.model.norm.weight.detach().float().cpu()               # (d,)
    carlini_compare = compare_w_eff(W_eff_hat, W_lm_true, g_final_true)
    logger.info("[carlini] principal-angle mean cos = %.4f (vs true W_eff)",
                carlini_compare["principal_angle_mean_cos"])

    # ── Build student ───────────────────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("STAGE 2: Instantiate random student, copy lm_head + final norm + last block")
    logger.info("-" * 70)
    student = make_student_from_config(teacher, device=device)

    # Copy lm_head + final norm (attacker-legal equivalent of Carlini knowledge)
    copy_lm_head_and_final_norm(student, teacher)

    # COPY THE LAST BLOCK FROM TEACHER to student, then perturb it slightly.
    # This is the POC convenience: we pretend the attacker has freed the
    # last block to train on its own, but we seed it with the teacher
    # values *perturbed* so the student does NOT perfectly match.  Then we
    # check if the finite-difference attack can *distinguish* the correct
    # block from the perturbed one.  If even this SUCCEEDS, we know the
    # attack geometry works.  If not, the barrier is structural.
    #
    # For a harder test, also try with FULLY RANDOM last block (commented
    # below).
    with torch.no_grad():
        # Copy teacher's last block into the student
        t_block_state = teacher.model.layers[LAST_BLOCK_IDX].state_dict()
        student.model.layers[LAST_BLOCK_IDX].load_state_dict(t_block_state)
        # Apply a small structured perturbation so we can detect delta
        # Use a ~1% multiplicative noise
        perturb_scale = 0.01
        rng = torch.Generator(device=device).manual_seed(args.seed + 12345)
        for name, p in student.model.layers[LAST_BLOCK_IDX].named_parameters():
            if "weight" in name and p.dim() >= 2:
                noise = torch.randn(p.shape, generator=rng, device=device) * perturb_scale
                p.data.add_(noise * p.data.abs().mean())

    # Freeze suffix, head, final norm
    freeze_suffix_and_head(student, last_trainable_block=SECOND_LAST_IDX)

    # ── Query pool ──────────────────────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("STAGE 3: Build query pool (%d queries @ len %d)",
                args.num_queries, args.max_seq_len)
    logger.info("-" * 70)
    query_ids = build_query_pool(
        tokenizer, args.num_queries, args.max_seq_len,
        seed=args.seed, allow_synthetic=args.allow_synthetic,
    )

    # Pick perturbation positions (default: last token) and new token ids
    if args.perturb_position == "last":
        pert_pos = torch.full(
            (args.num_queries, args.perturbations_per_query),
            args.max_seq_len - 1, dtype=torch.long,
        )
    else:
        rng = torch.Generator().manual_seed(args.seed + 1)
        pert_pos = torch.randint(
            0, args.max_seq_len,
            (args.num_queries, args.perturbations_per_query),
            generator=rng,
        )
    rng = torch.Generator().manual_seed(args.seed + 2)
    new_tok = torch.randint(
        3, vocab_size,
        (args.num_queries, args.perturbations_per_query),
        generator=rng,
    )

    # ── KD pretraining ──────────────────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("STAGE 4: KD pretraining (steps=%d, lr=%.2e)",
                args.pretrain_steps, args.pretrain_lr)
    logger.info("-" * 70)
    pretrain_hist = pretrain_student_kd(
        student, teacher,
        query_ids=query_ids,
        device=device,
        steps=args.pretrain_steps,
        lr=args.pretrain_lr,
        batch_size=args.pretrain_batch_size,
    )

    # ── Collect states ──────────────────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("STAGE 5: Collect teacher/student states on base+perturbed queries")
    logger.info("-" * 70)
    state = collect_states(
        teacher, student,
        query_ids=query_ids,
        perturb_positions=pert_pos,
        new_token_ids=new_tok,
        W_eff_hat=W_eff_hat,
        device=device,
        batch_size=args.collect_batch_size,
    )

    # ── Attack ──────────────────────────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("STAGE 6: Finite-difference Jacobian attack")
    logger.info("-" * 70)
    diag, attack_metrics = run_attack(
        state=state,
        W_eff_hat=W_eff_hat,
        teacher=teacher,
        student=student,
        device=device,
    )

    # ── Verdict ─────────────────────────────────────────────────────────────
    # Primary criterion: can the attack do better than "copy student" on the
    # last-block W_O?  If best_naive_W_O_cos > baseline_student_W_O_cos + 0.2
    # and best_naive_W_O_cos > 0.2 in absolute terms AND we did not use oracle
    # h22 in the regression.  (We used student h22, not teacher h22 — that is
    # attacker-legal.)
    baseline = attack_metrics["baseline_student_W_O_cos"]
    attacked = attack_metrics["best_naive_W_O_cos"]
    improvement = attacked - baseline

    # Secondary criterion: are the top directions of J_t_est-I closer to
    # col(W_O_teacher) than to col(W_O_student)?  If so we aligned in the
    # right direction.
    ang_t = attack_metrics["principal_angles_Wo_teacher_mean_cos"]
    ang_s = attack_metrics["principal_angles_Wo_student_mean_cos"]
    angles_distinguish = ang_t - ang_s

    success = (
        (attacked > 0.2 and improvement > 0.05)
        or (angles_distinguish > 0.05 and ang_t > 0.2)
    )

    verdict = {
        "attacked_W_O_cos": float(attacked),
        "baseline_student_W_O_cos": float(baseline),
        "improvement_over_student": float(improvement),
        "top_dir_to_teacher_Wo_cos": float(ang_t),
        "top_dir_to_student_Wo_cos": float(ang_s),
        "direction_distinguishes_teacher": float(angles_distinguish),
        "success": bool(success),
    }

    # ── Save everything ─────────────────────────────────────────────────────
    results = {
        "args": vars(args),
        "carlini": {
            "d_hat": int(d_hat),
            **{k: v for k, v in carlini_compare.items()},
        },
        "pretrain_history": pretrain_hist,
        "attack_diagnostics": {
            "u_cosine_to_h_L_teacher": diag.u_cosine_to_h_L_teacher,
            "u_cosine_to_h_L_student": diag.u_cosine_to_h_L_student,
            "Du_teacher_cos_to_DhL_teacher": diag.Du_teacher_cos_to_DhL_teacher,
            "Du_student_cos_to_DhL_student": diag.Du_student_cos_to_DhL_student,
            "h22_cos_teacher_student": diag.h22_cos_teacher_student,
            "h22_diff_relative_norm": diag.h22_diff_relative_norm,
            "residual_norm": diag.residual_norm,
            "residual_to_signal_ratio": diag.residual_to_signal_ratio,
        },
        "attack_metrics": attack_metrics,
        "verdict": verdict,
    }

    results_path = out_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Human summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FINITE-DIFFERENCE JACOBIAN ATTACK — SUMMARY")
    print("=" * 72)
    print(f"  Model:                 {args.model_name}")
    print(f"  Queries:               N={args.num_queries}, P={args.perturbations_per_query}")
    print(f"  KD pretrain steps:     {args.pretrain_steps}")
    print(f"  Output dir:            {out_dir}")
    print("-" * 72)
    print("  Stage 1 — Carlini SVD")
    print(f"    d_hat:                     {d_hat} (true: {hidden_size})")
    print(f"    principal-angle mean cos:  {carlini_compare['principal_angle_mean_cos']:.4f}")
    print(f"    principal-angle min cos:   {carlini_compare['principal_angle_min_cos']:.4f}")
    print("-" * 72)
    print("  Stage 4 — KD pretraining (aligning h22_student to h22_teacher)")
    if pretrain_hist:
        last_entry = pretrain_hist[-1]
        print(f"    final step:                {last_entry['step']}")
        print(f"    final loss_last:           {last_entry['loss_last']:.4e}")
        print(f"    final loss_full:           {last_entry['loss_full']:.4e}")
    else:
        print("    (skipped)")
    print("-" * 72)
    print("  Stage 6 — Attack diagnostics")
    print(f"    u_teacher cos to h_L_teacher (oracle):      {diag.u_cosine_to_h_L_teacher:.4f}")
    print(f"    u_student cos to h_L_student (oracle):      {diag.u_cosine_to_h_L_student:.4f}")
    print(f"    Du_teacher cos to D(h_L_t) (oracle):        {diag.Du_teacher_cos_to_DhL_teacher:.4f}")
    print(f"    Du_student cos to D(h_L_s) (oracle):        {diag.Du_student_cos_to_DhL_student:.4f}")
    print(f"    h22 teacher vs student cos:                 {diag.h22_cos_teacher_student:.4f}")
    print(f"    h22 rel-norm diff:                          {diag.h22_diff_relative_norm:.4f}")
    print(f"    residual ||u_t_rot - u_s|| / ||u_t_rot||:   {diag.residual_to_signal_ratio:.4f}")
    print("-" * 72)
    print("  Stage 6 — Jacobian linearization quality")
    print(f"    J_t fit cos (Du_t vs Dh22 @ J_t^T):         {attack_metrics['J_t_fit_cos']:.4f}")
    print(f"    J_s fit cos:                                {attack_metrics['J_s_fit_cos']:.4f}")
    print(f"    DJ fit cos (pred(Du_t)-pred(Du_s)):         {attack_metrics['DJ_fit_cos']:.4f}")
    print(f"    cos(J_t - I, J_s - I):                      {attack_metrics['cos_DJt_DJs']:.4f}")
    print(f"    cos(DJ_est, J_t - I):                       {attack_metrics['cos_DJest_vs_DJt']:.4f}")
    print("-" * 72)
    print("  Stage 6 — Subspace alignment (last-block W_O)")
    print(f"    top-64 dirs of J_t-I vs W_O_teacher cols:   {ang_t:.4f}")
    print(f"    top-64 dirs of J_t-I vs W_O_student cols:   {ang_s:.4f}")
    print(f"    distinguishing power (teacher-student):     {angles_distinguish:.4f}")
    print("-" * 72)
    print("  Stage 6 — Naive W_O recovery")
    print(f"    baseline cos(W_O_student, W_O_teacher):     {baseline:.4f}")
    print(f"    best attacked cos(W_O_recovered, teacher):  {attacked:.4f}")
    print(f"    improvement over student baseline:          {improvement:+.4f}")
    print(f"    best alpha:                                 {attack_metrics['best_naive_W_O_alpha']:.2f}")
    print("=" * 72)
    if success:
        print("  VERDICT:  ATTACK SUCCEEDED (cos > 0.2 for recovered block matrix)")
    else:
        print("  VERDICT:  ATTACK FAILED (documenting the barrier)")
        print("  Interpretation:")
        if diag.h22_cos_teacher_student < 0.9:
            print("    - h22 mismatch dominates: KD pretraining did not align prefix.")
            print("      With h22_cos=%.3f, the residual Du_t_rot - Du_s is mostly from"
                  % diag.h22_cos_teacher_student)
            print("      prefix weight differences, not the last block.")
        if attack_metrics["J_t_fit_cos"] < 0.3:
            print("    - Linearization poor: finite-difference assumption violated.")
            print("      Nonlinearity at this perturbation magnitude swamps first-order.")
        if attacked < 0.2 and baseline < 0.2:
            print("    - Carlini's W_eff + single-step perturbations are insufficient to")
            print("      penetrate beyond the rotation ambiguity of h_L.")
            print("      This is evidence for a *structural* pure-logits barrier: the")
            print("      logit observation z = W_eff @ h_L collapses all rotational")
            print("      information about h_L, which finite differences cannot unmix")
            print("      without additional external structure (e.g., logit-bias probing,")
            print("      Carlini's rank-d attack, or oracle boundary access).")
    print("=" * 72)
    print(f"  Results: {results_path}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
