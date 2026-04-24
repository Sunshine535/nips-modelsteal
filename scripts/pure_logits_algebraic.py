#!/usr/bin/env python3
"""Pure-Logits Algebraic Attack on Internal Transformer Layers.

This script definitively answers the following question: can an attacker that
observes ONLY black-box logits (no oracle access to the teacher's internal
boundary hidden states) still recover internal block parameters algebraically?

The paper's current "oracle regime" assumes the attacker has access to
``h_{L-1}(x)`` (the teacher's block-input hidden state) for every query ``x``.
That is unrealistic — a real black-box attacker has only ``(x, z(x))`` pairs.

Pipeline
--------
Step A (PURE-LOGITS Carlini extraction):
    Run the teacher on many queries, collect last-position logits, SVD of
    centred logit matrix recovers the span of ``W_lm @ diag(g_final)`` up to
    an orthogonal rotation. This is pure algebraic attack — no hidden state
    access.

Step B (PURE-LOGITS h_L recovery):
    For each query ``x``, the attacker estimates
    ``u(x) = h_L(x) / rms(h_L(x)) = pinv(W_lm_eff) @ z(x)``.
    We validate this against the teacher's true ``h_L`` (used for evaluation
    only, NEVER in the recovery algorithm).

Step C (Pure-logits internal-block recovery attempt):
    The student model has the SAME architecture as the teacher, but its prefix
    blocks 0..(L-K) are randomly initialised and FROZEN. Only the trainable
    suffix blocks are being attacked. The boundary hidden state the student
    sees at the input of block ``L-K+1`` is ``student_h_{L-K}(x) != h_{L-K}``.
    The attacker tries algebraic W_down OLS as if ``student_h_{L-K}`` were the
    teacher's value.

Step D (Ablations):
    D1. Pure-logits algebraic — uses ``student_h_{L-K}`` and Carlini-recovered
        ``u(x)`` only. This is the "real" pure-logits attack.
    D2. Pure-logits + oracle activations — same but uses teacher's TRUE gate/up
        activations (control: isolates whether the activation-side constraint
        or the input-side mismatch is the bottleneck).
    D3. Upper-bound control (oracle everything) — reruns the teacher-internal
        oracle baseline for direct comparison.
    D4. Carlini lm_head subspace cos — how well ``W_lm_eff`` is recovered.
    D5. u(x) fidelity — cos of recovered ``u(x)`` vs teacher's true normalized
        h_L. (This is the "best case" input the pure-logits attack can have.)
    D6. Functional KL — compare the student's block-L output under recovered
        parameters against the teacher's true block-L output.

Strict data-leakage discipline
------------------------------
* Recovery algorithm gets only:
    - The student model (architecture + random weights).
    - ``(x, z(x))`` pairs from the teacher.
* Teacher's ``W_gate / W_up / W_down`` and internal hidden states
  ``h_{L-K}, h_{mid}, h_L`` are ONLY used for evaluation, never for recovery.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python scripts/pure_logits_algebraic.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --block_idx 23 \
        --num_queries 2048 \
        --max_seq_len 128 \
        --output_dir results/v5_pure_logits_algebraic \
        --seed 42
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.permutation_alignment import align_ffn_neurons  # noqa: E402

logger = logging.getLogger(__name__)

# ── Constants (Qwen2.5-0.5B) ────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
D_MODEL = 896
D_FF = 4864
VOCAB_SIZE = 151936
NUM_LAYERS = 24
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64


# ════════════════════════════════════════════════════════════════════════════
# Logging and argument parsing
# ════════════════════════════════════════════════════════════════════════════


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Pure-logits algebraic attack on internal transformer layers"
    )
    p.add_argument("--model_name", type=str, default=MODEL_NAME)
    p.add_argument("--block_idx", type=int, default=23,
                   help="Target block index (the one we try to recover).")
    p.add_argument("--num_queries", type=int, default=2048)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for teacher/student forward passes.")
    p.add_argument("--output_dir", type=str,
                   default="results/v5_pure_logits_algebraic")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--allow_synthetic", action="store_true",
                   help="Allow padding with random tokens if WikiText load fails")
    p.add_argument("--query_dist", type=str, default="random",
                   choices=["random", "wikitext", "mixed"],
                   help="Distribution of query tokens")
    p.add_argument("--reserved_tokens", type=int, default=0,
                   help="Skip the first N token ids when generating random tokens")
    p.add_argument("--student_init_std", type=float, default=0.02,
                   help="Std of the normal distribution used to randomise the student")
    p.add_argument("--skip_ablation_d2", action="store_true",
                   help="Skip the oracle-activation control (Ablation D2).")
    p.add_argument("--skip_ablation_d3", action="store_true",
                   help="Skip the full-oracle upper-bound control (Ablation D3).")
    p.add_argument("--functional_batch_size", type=int, default=16,
                   help="Batch size for functional KL eval (smaller if OOM).")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Model loading (with HF_HUB_OFFLINE fallback)
# ════════════════════════════════════════════════════════════════════════════


def load_model(model_name: str, device: torch.device, dtype: torch.dtype = torch.float32):
    """Load a causal LM. Tries online first; falls back to HF_HUB_OFFLINE=1."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model: %s (device=%s, dtype=%s)", model_name, device, dtype)
    load_kwargs = dict(torch_dtype=dtype, trust_remote_code=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        logger.warning("Online load failed (%s). Retrying with HF_HUB_OFFLINE=1...", e)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def make_student_from_teacher(
    teacher: nn.Module,
    student_init_std: float,
    seed: int,
    device: torch.device,
) -> nn.Module:
    """Build a structurally identical student, randomize all weights, keep buffers."""
    import copy

    logger.info("Cloning teacher architecture for student ...")
    student = copy.deepcopy(teacher)

    # Re-seed *before* randomising so the student's init is deterministic.
    torch.manual_seed(seed + 7919)

    with torch.no_grad():
        for name, param in student.named_parameters():
            # RMSNorm gains should stay ~1 (otherwise the student is pathological)
            if "norm" in name.lower() and param.dim() == 1:
                param.data.fill_(1.0)
                continue
            # embed_tokens and lm_head get small normal init; this matches
            # standard transformer initialization and avoids explosion.
            if param.dim() >= 2:
                param.data.normal_(mean=0.0, std=student_init_std)
            else:
                param.data.zero_()

        # Retie embeddings if teacher has tied weights.
        tied = getattr(student.config, "tie_word_embeddings", False)
        if tied and hasattr(student, "lm_head"):
            # Share storage so student.lm_head.weight IS student.embed_tokens.weight.
            student.lm_head.weight = student.model.embed_tokens.weight

    student = student.to(device)
    student.eval()
    return student


def freeze_student_prefix(student: nn.Module, num_frozen_layers: int):
    """Freeze (requires_grad=False) all layers 0..num_frozen_layers-1.

    In pure-logits setup we don't actually train the student, but this makes
    the "trainable vs frozen" contract explicit and matches paper wording.
    """
    frozen = 0
    for i in range(num_frozen_layers):
        for p in student.model.layers[i].parameters():
            p.requires_grad = False
            frozen += 1
    logger.info("Froze %d parameter tensors in student layers 0..%d",
                frozen, num_frozen_layers - 1)


# ════════════════════════════════════════════════════════════════════════════
# Query pool construction
# ════════════════════════════════════════════════════════════════════════════


def build_wikitext_queries(tokenizer, pool_size: int, max_seq_len: int, seed: int,
                           allow_synthetic: bool) -> torch.Tensor:
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
        logger.warning("WikiText load failed (%s). Padding with random tokens.", e)

    remaining = pool_size - len(input_ids_list)
    if remaining > 0:
        if not allow_synthetic:
            raise RuntimeError(
                f"Only {len(input_ids_list)}/{pool_size} from dataset. "
                "Use --allow_synthetic to pad with random tokens."
            )
        rng = torch.Generator().manual_seed(seed + 11)
        random_ids = torch.randint(
            3, tokenizer.vocab_size, (remaining, max_seq_len), generator=rng
        )
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    return torch.stack(input_ids_list[:pool_size])


def build_random_queries(vocab_size: int, pool_size: int, max_seq_len: int,
                         seed: int, reserved: int = 0) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    lo = int(max(0, reserved))
    hi = int(vocab_size)
    if hi <= lo:
        raise ValueError(f"Bad vocab range [{lo}, {hi})")
    return torch.randint(lo, hi, (pool_size, max_seq_len), generator=rng)


def build_queries(dist: str, tokenizer, pool_size: int, max_seq_len: int,
                  seed: int, reserved: int, allow_synthetic: bool) -> torch.Tensor:
    vocab = int(tokenizer.vocab_size)
    if dist == "random":
        return build_random_queries(vocab, pool_size, max_seq_len, seed, reserved)
    if dist == "wikitext":
        return build_wikitext_queries(tokenizer, pool_size, max_seq_len,
                                      seed, allow_synthetic)
    if dist == "mixed":
        half = pool_size // 2
        other = pool_size - half
        wiki = build_wikitext_queries(tokenizer, half, max_seq_len, seed,
                                      allow_synthetic=True)
        rand = build_random_queries(vocab, other, max_seq_len, seed + 1, 0)
        combined = torch.cat([wiki, rand], dim=0)
        perm = torch.randperm(combined.shape[0],
                              generator=torch.Generator().manual_seed(seed + 2))
        return combined[perm]
    raise ValueError(f"Unknown query distribution: {dist}")


# ════════════════════════════════════════════════════════════════════════════
# RMSNorm helper (matches Qwen2RMSNorm)
# ════════════════════════════════════════════════════════════════════════════


def rms_norm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: x / rms(x) * g, rms(x) = sqrt(mean(x^2) + eps)."""
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms * g.float()).to(x.dtype)


def rms_scale(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Returns rms(x) with trailing dim kept: shape (..., 1)."""
    return torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)


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


# ════════════════════════════════════════════════════════════════════════════
# Hidden-state collection (teacher and student)
# ════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def collect_states(
    model: nn.Module,
    query_ids: torch.Tensor,
    block_idx: int,
    device: torch.device,
    batch_size: int,
    collect_logits: bool = True,
) -> dict[str, torch.Tensor]:
    """Collect h_in, h_mid, h_out for the target block (and optionally logits).

    h_in = hidden_states[block_idx]   (input to block block_idx)
    h_mid = h_in + self_attn(h_in)    (pre-MLP residual)
    h_out = hidden_states[block_idx+1] (output of block block_idx)
    """
    N, T = query_ids.shape
    all_h_in: list[torch.Tensor] = []
    all_h_mid: list[torch.Tensor] = []
    all_h_out: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []

    target_block = model.model.layers[block_idx]
    attn_buf: list[torch.Tensor] = []

    def attn_hook(module, inputs, output):
        a = output[0] if isinstance(output, tuple) else output
        attn_buf.append(a.detach().cpu().float())

    handle = target_block.self_attn.register_forward_hook(attn_hook)

    try:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = query_ids[start:end].to(device)
            attn_buf.clear()
            outputs = model(batch, output_hidden_states=True, return_dict=True)
            hs = outputs.hidden_states
            h_in_b = hs[block_idx].detach().cpu().float()
            h_out_b = hs[block_idx + 1].detach().cpu().float()
            attn_b = attn_buf[0]
            h_mid_b = h_in_b + attn_b

            all_h_in.append(h_in_b)
            all_h_mid.append(h_mid_b)
            all_h_out.append(h_out_b)
            if collect_logits:
                all_logits.append(outputs.logits.detach().cpu().float())

            if (start // batch_size) % 10 == 0:
                logger.info("    collected %d / %d queries", end, N)
    finally:
        handle.remove()

    out = {
        "h_in": torch.cat(all_h_in, dim=0),
        "h_mid": torch.cat(all_h_mid, dim=0),
        "h_out": torch.cat(all_h_out, dim=0),
    }
    if collect_logits:
        out["logits"] = torch.cat(all_logits, dim=0)
    return out


# ════════════════════════════════════════════════════════════════════════════
# STEP A — Carlini extraction of W_lm_eff = W_lm @ diag(g_final)
# ════════════════════════════════════════════════════════════════════════════


def carlini_extract_lm_head(
    logits_last: torch.Tensor,   # [N, V]
    d: int,
    output_dir: Path,
) -> dict:
    """SVD-based recovery of the span of W_lm @ diag(g_final).

    Returns W_hat (shape [d, V]) whose rows span the same d-dimensional
    subspace of R^V as the columns of ``W_lm @ diag(g_final)``.
    """
    N, V = logits_last.shape
    logger.info("  Carlini extraction: SVD of centred logit matrix (N=%d, V=%d)", N, V)

    Y_centered = (logits_last - logits_last.mean(dim=0, keepdim=True)).float()
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(Y_centered, full_matrices=False)
    svd_time = time.time() - t0

    S_np = S.numpy()

    # Top-d singular values form the "signal subspace"; the sharp drop at
    # index d identifies the hidden dimension. We assume d is known.
    gap_ratios = S_np[:-1] / np.maximum(S_np[1:], 1e-30)
    search_lo = max(0, d - 50)
    search_hi = min(len(gap_ratios), d + 50)
    if search_hi <= search_lo:
        # N is too small to observe the gap at index d (need N >= d).
        # Fall back: use the smallest available index.
        logger.warning(
            "    Gap search window empty (search_lo=%d >= search_hi=%d); "
            "likely N=%d < d=%d. Falling back to d_hat = min(d, N-1).",
            search_lo, search_hi, N, d,
        )
        d_hat = min(d, len(gap_ratios) - 1)
    else:
        d_hat = int(search_lo + np.argmax(gap_ratios[search_lo:search_hi]))

    logger.info(
        "    SVD done in %.2fs, d_hat=%d (true=%d, gap_ratio=%.2f), "
        "top5_singular_vals=%s",
        svd_time, d_hat, d, gap_ratios[d_hat] if d_hat < len(gap_ratios) else 0.0,
        [f"{s:.3f}" for s in S_np[:5]],
    )

    # W_hat: rows are the top-d right singular vectors (shape [d, V]).
    # The column span of W_hat.T equals the column span of W_lm.
    W_hat = Vh[:d, :].float()  # [d, V]

    return {
        "W_hat": W_hat,
        "d_hat": d_hat,
        "singular_values_top20": [float(s) for s in S_np[:20]],
        "gap_ratio_at_d": float(gap_ratios[d_hat]) if d_hat < len(gap_ratios) else 0.0,
        "svd_time_s": svd_time,
    }


def procrustes_align(W_rec: torch.Tensor, W_true: torch.Tensor):
    """Find rotation R such that W_rec @ R ~ W_true (Frobenius-optimal)."""
    # W_rec: [V, d],  W_true: [V, d]
    cross = W_rec.T @ W_true                                    # [d, d]
    U_p, _, Vt_p = torch.linalg.svd(cross, full_matrices=False)
    R = U_p @ Vt_p                                              # [d, d]
    aligned = W_rec @ R
    cos_cols = F.cosine_similarity(aligned, W_true, dim=0)
    frob_err = float(torch.norm(aligned - W_true) / max(torch.norm(W_true).item(),
                                                         1e-12))
    return aligned, R, cos_cols, frob_err


# ════════════════════════════════════════════════════════════════════════════
# STEP B — pure-logits u(x) recovery via pinv(W_lm_eff)
# ════════════════════════════════════════════════════════════════════════════


def recover_u_pure_logits(
    logits_all: torch.Tensor,   # [N, T, V]
    W_eff_full: torch.Tensor,   # [V, d] the Carlini-recovered effective matrix
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Compute u(x) = pinv(W_eff) @ z(x) for each query position.

    Returns u_all of shape [N, T, d], where each slice is a length-d vector
    in the row space of W_eff.
    """
    N, T, V = logits_all.shape
    d = W_eff_full.shape[1]

    # Gram pinv in float64 for stability
    W_dev = W_eff_full.double().to(device)           # [V, d]
    gram = W_dev.T @ W_dev                            # [d, d]
    gram += 1e-8 * gram.diagonal().mean() * torch.eye(d, dtype=torch.float64,
                                                       device=device)
    gram_inv = torch.linalg.inv(gram)                 # [d, d]
    pinv_W = (gram_inv @ W_dev.T).float()             # [d, V]

    u_all = torch.zeros(N, T, d, dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            z = logits_all[s:e].to(device).float()    # [bs, T, V]
            # pinv(W_eff) @ z  == z @ pinv_W.T
            u = z @ pinv_W.T                           # [bs, T, d]
            u_all[s:e] = u.cpu()

    del W_dev, gram, gram_inv, pinv_W
    torch.cuda.empty_cache()
    return u_all


# ════════════════════════════════════════════════════════════════════════════
# High-precision OLS for W_down (identical solver as v2/v4, repeated here
# so this script is self-contained and can be audited easily)
# ════════════════════════════════════════════════════════════════════════════


def solve_w_down_ols(
    h_mid_all: torch.Tensor,      # [N, T, d] INPUT to MLP (pre-RMSNorm residual)
    r_mlp_all: torch.Tensor,      # [N, T, d] MLP residual (h_out - h_mid)
    g_mlp: torch.Tensor,          # [d] RMSNorm gain used by the MLP
    W_gate: torch.Tensor,         # [d_ff, d] gate weight used to form activations
    W_up: torch.Tensor,           # [d_ff, d] up weight used to form activations
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, dict]:
    """OLS in float64 for W_down given:
        a(x) = SiLU(W_gate @ RMSNorm(h_mid, g_mlp)) * (W_up @ RMSNorm(h_mid, g_mlp))
        r(x) = W_down @ a(x)

    Returns W_down_rec [d, d_ff] on CPU, plus conditioning metrics.
    """
    N, T, d = h_mid_all.shape
    d_ff = W_gate.shape[0]
    M = N * T

    g_mlp_dev = g_mlp.to(device)
    W_gate_dev = W_gate.to(device).float()
    W_up_dev = W_up.to(device).float()

    AAt = torch.zeros(d_ff, d_ff, dtype=torch.float64, device=device)
    RAt = torch.zeros(d, d_ff, dtype=torch.float64, device=device)

    logger.info("    Accumulating normal equations (float64, M=%d)...", M)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            hm = h_mid_all[s:e].to(device).reshape(-1, d)
            rm = r_mlp_all[s:e].to(device).reshape(-1, d).float()

            x = rms_norm(hm, g_mlp_dev)
            a = F.silu(x @ W_gate_dev.T) * (x @ W_up_dev.T)   # [M_b, d_ff]

            a64 = a.double()
            r64 = rm.double()
            AAt += a64.T @ a64
            RAt += r64.T @ a64

    diag_mean = AAt.diagonal().mean().item()
    try:
        eigvals = torch.linalg.eigvalsh(AAt)
        cond = (eigvals[-1] / eigvals[0].clamp(min=1e-30)).item()
    except Exception:
        cond = -1.0

    if cond > 0 and cond < 1e8:
        ridge = 1e-10 * diag_mean
    elif cond > 0 and cond < 1e12:
        ridge = 1e-8 * diag_mean
    else:
        ridge = 1e-6 * diag_mean

    AAt += ridge * torch.eye(d_ff, dtype=torch.float64, device=device)

    try:
        L = torch.linalg.cholesky(AAt)
        W_down_rec = torch.linalg.solve_triangular(
            L.T,
            torch.linalg.solve_triangular(L, RAt.T, upper=False),
            upper=True,
        ).T.float()
    except RuntimeError:
        logger.warning("    Cholesky failed, falling back to torch.linalg.solve")
        W_down_rec = torch.linalg.solve(AAt, RAt.T).T.float()

    metrics = {
        "condition_number": float(cond),
        "ridge": float(ridge),
        "diag_mean": float(diag_mean),
        "M": int(M),
    }
    return W_down_rec.cpu(), metrics


# ════════════════════════════════════════════════════════════════════════════
# Functional KL eval: inject recovered params into student, compare to teacher
# ════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def functional_kl_eval(
    teacher: nn.Module,
    student: nn.Module,
    query_ids: torch.Tensor,
    block_idx: int,
    device: torch.device,
    batch_size: int,
    max_queries: int = 128,
) -> dict:
    """Compare teacher's block-L output (logits) to the student's block-L output
    under the *student's current MLP weights* (which we will have replaced with
    recovered weights).

    We feed the same ``h_in`` (teacher boundary state) to *both* teacher and
    student at block_idx and run only that one block forward. The KL is
    measured on the resulting logits (after running the remaining layers and
    lm_head). Both teacher and student here use their own LN + lm_head, so
    this is a *joint* functional recovery metric.

    Args:
        max_queries : limit to keep this cheap (default 128 sequences).
    """
    N = min(query_ids.shape[0], max_queries)
    ids = query_ids[:N].to(device)

    # Teacher logits (ground-truth)
    t_logits_all: list[torch.Tensor] = []
    s_logits_all: list[torch.Tensor] = []

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        batch = ids[s:e]
        t_logits = teacher(batch, return_dict=True).logits.float().cpu()
        s_logits = student(batch, return_dict=True).logits.float().cpu()
        t_logits_all.append(t_logits)
        s_logits_all.append(s_logits)

    t_logits = torch.cat(t_logits_all, dim=0)
    s_logits = torch.cat(s_logits_all, dim=0)

    # KL(teacher || student) on the last position to keep signal/noise high
    tp = F.log_softmax(t_logits[:, -1, :], dim=-1)
    sp = F.log_softmax(s_logits[:, -1, :], dim=-1)
    # KL(p||q) = sum p (log p - log q)
    p = tp.exp()
    kl = (p * (tp - sp)).sum(dim=-1).mean().item()

    # Top-1 agreement
    t_top = t_logits[:, -1, :].argmax(dim=-1)
    s_top = s_logits[:, -1, :].argmax(dim=-1)
    top1_agree = (t_top == s_top).float().mean().item()

    # Spearman-like: overlap in top-10
    k = 10
    t_topk = torch.topk(t_logits[:, -1, :], k=k).indices
    s_topk = torch.topk(s_logits[:, -1, :], k=k).indices
    overlap = torch.tensor([
        len(set(t_topk[i].tolist()) & set(s_topk[i].tolist())) / k
        for i in range(N)
    ]).mean().item()

    return {
        "kl_teacher_student": kl,
        "top1_agreement": top1_agree,
        "top10_overlap": overlap,
        "num_queries": N,
    }


# ════════════════════════════════════════════════════════════════════════════
# Recovery evaluation — cosine / Frobenius / functional-residual MSE
# ════════════════════════════════════════════════════════════════════════════


def evaluate_w_down(
    W_down_rec: torch.Tensor,        # [d, d_ff]
    W_down_true: torch.Tensor,       # [d, d_ff]
    W_gate_true: torch.Tensor,       # [d_ff, d]   used only to build sanity
    W_up_true: torch.Tensor,         # [d_ff, d]
    h_mid_ref: torch.Tensor,         # [N, T, d]  for residual MSE probe
    r_mlp_ref: torch.Tensor,         # [N, T, d]
    g_mlp: torch.Tensor,             # [d]
    device: torch.device,
    batch_size: int,
) -> dict:
    """Return cosine (flat / per-row) and Frobenius metrics between recovered
    and true W_down, plus a "sanity" functional residual MSE (how well the
    recovered W_down + oracle gate/up reproduce the MLP residual).
    """
    cos_flat = flat_cosine(W_down_rec, W_down_true)
    cos_row = per_row_cosine(W_down_rec, W_down_true)
    frob = (W_down_rec - W_down_true).norm().item() / max(
        W_down_true.norm().item(), 1e-12)
    max_abs = (W_down_rec - W_down_true).abs().max().item()
    rel_entry = (W_down_rec - W_down_true).abs().mean().item() / max(
        W_down_true.abs().mean().item(), 1e-12)

    # Functional residual MSE with oracle gate/up (first batch only, for speed)
    with torch.no_grad():
        g_mlp_dev = g_mlp.to(device)
        Wg_dev = W_gate_true.to(device).float()
        Wu_dev = W_up_true.to(device).float()
        Wd_rec_dev = W_down_rec.to(device).float()

        hm = h_mid_ref[:batch_size].to(device).reshape(-1, hm_d := h_mid_ref.shape[-1])
        rm = r_mlp_ref[:batch_size].to(device).reshape(-1, hm_d).float()
        x = rms_norm(hm, g_mlp_dev)
        a = F.silu(x @ Wg_dev.T) * (x @ Wu_dev.T)
        r_pred = a @ Wd_rec_dev.T
        func_cos = flat_cosine(r_pred.cpu(), rm.cpu())
        func_mse = F.mse_loss(r_pred, rm).item()

    return {
        "W_down_cosine": cos_flat,
        "W_down_per_row_cosine": cos_row,
        "W_down_frob_error": frob,
        "W_down_max_abs_error": max_abs,
        "W_down_rel_entry_error": rel_entry,
        "sanity_functional_residual_cos": func_cos,
        "sanity_functional_residual_mse": func_mse,
    }


# ════════════════════════════════════════════════════════════════════════════
# Build pure-logits (h_mid_for_attack, r_mlp_for_attack) pairs
# ════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def build_attacker_data_pure_logits(
    teacher_h_L: torch.Tensor,       # [N, T, d]   only used to measure rms
    student_h_in: torch.Tensor,      # [N, T, d]   student boundary state
    student_h_mid: torch.Tensor,     # [N, T, d]   student pre-MLP residual
    student_h_out: torch.Tensor,     # [N, T, d]   student post-MLP residual (= h_L at target block)
    u_pure: torch.Tensor,            # [N, T, d]   Carlini-recovered direction
    g_mlp_student: torch.Tensor,     # [d]
    device: torch.device,
    batch_size: int,
) -> dict:
    """Construct ``(h_mid, h_out_est)`` pairs as seen by the attacker.

    h_mid_attack := student_h_mid (the attacker's best guess at the MLP input).
    h_out_attack := student_h_mid + student_rms_scale * u_pure
                       where student_rms_scale is the RMS of student's own
                       block-L output (estimate of the true h_L's magnitude
                       in the student's reference frame).

    Returns a dict with h_mid_attack, h_out_attack, r_mlp_attack, X_attack,
    plus diagnostic metrics.
    """
    N, T, d = student_h_mid.shape

    # Use the already-collected student h_out (from full forward pass)
    # as the scale reference. This avoids manually invoking the block,
    # which can fail on models that require pre-computed rotary embeddings.
    student_h_L_all = student_h_out.float()

    # Scale per (sample, position) using student's own rms
    student_rms = rms_scale(student_h_L_all)                # [N, T, 1]

    # Reconstruct estimate of teacher's h_L in student's scaling:
    # h_L_est = student_rms * u_pure   (direction from Carlini, magnitude
    # from student's own activation scale)
    h_L_est = student_rms * u_pure

    # h_mid_attack: student's own boundary mid (this is NOT the teacher's!)
    h_mid_attack = student_h_mid.clone()
    # MLP residual attacker pretends to have
    r_mlp_attack = h_L_est - h_mid_attack                    # [N, T, d]

    # Diagnostics
    teacher_rms = rms_scale(teacher_h_L)
    rms_ratio = (student_rms.squeeze(-1) / teacher_rms.squeeze(-1).clamp(min=1e-9))
    direction_cos_values: list[float] = []
    # sample a few (sample, token) positions
    for i in range(0, N, max(1, N // 8)):
        for t in range(0, T, max(1, T // 8)):
            true_u = teacher_h_L[i, t] / rms_scale(teacher_h_L[i, t]).squeeze(-1)
            direction_cos_values.append(flat_cosine(u_pure[i, t], true_u))

    info = {
        "h_mid_attack": h_mid_attack,
        "r_mlp_attack": r_mlp_attack,
        "h_L_est": h_L_est,
        "student_rms_mean": float(student_rms.mean()),
        "teacher_rms_mean": float(teacher_rms.mean()),
        "rms_ratio_mean": float(rms_ratio.mean()),
        "rms_ratio_median": float(rms_ratio.median()),
        "u_direction_mean_cos": float(sum(direction_cos_values) /
                                      max(len(direction_cos_values), 1)),
        "u_direction_num_samples": len(direction_cos_values),
    }
    return info


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════


def main():
    args = parse_args()
    setup_logging()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    block_idx = args.block_idx
    N = args.num_queries
    T = args.max_seq_len
    BS = args.batch_size

    logger.info("=" * 72)
    logger.info("PURE-LOGITS ALGEBRAIC ATTACK ON INTERNAL TRANSFORMER LAYER")
    logger.info("=" * 72)
    logger.info("Model         : %s", args.model_name)
    logger.info("Target block  : %d", block_idx)
    logger.info("Queries       : %d x %d tokens (%d data points)", N, T, N * T)
    logger.info("Device        : %s", device)
    logger.info("Query dist    : %s", args.query_dist)
    logger.info("Output dir    : %s", out_dir)

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Load teacher ─────────────────────────────────────────────────────────
    teacher, tokenizer = load_model(args.model_name, device, torch.float32)

    # ── Extract teacher's TRUE parameters (for EVALUATION ONLY) ──────────────
    prefix = f"model.layers.{block_idx}"
    true_params: dict[str, torch.Tensor] = {}
    for name, param in teacher.named_parameters():
        if name.startswith(prefix) or name in ("lm_head.weight",
                                                "model.norm.weight",
                                                "model.embed_tokens.weight"):
            true_params[name] = param.data.cpu().clone()

    W_lm_teacher = teacher.lm_head.weight.data.float().cpu()            # [V, d]
    g_final_teacher = teacher.model.norm.weight.data.float().cpu()      # [d]
    g_mlp_teacher = true_params[f"{prefix}.post_attention_layernorm.weight"].float()
    W_gate_teacher = true_params[f"{prefix}.mlp.gate_proj.weight"].float()   # [d_ff, d]
    W_up_teacher = true_params[f"{prefix}.mlp.up_proj.weight"].float()       # [d_ff, d]
    W_down_teacher = true_params[f"{prefix}.mlp.down_proj.weight"].float()   # [d, d_ff]

    d = W_down_teacher.shape[0]
    d_ff = W_down_teacher.shape[1]
    V = W_lm_teacher.shape[0]
    logger.info("Teacher shapes: W_lm=%s g_final=%s W_gate=%s W_up=%s W_down=%s",
                W_lm_teacher.shape, g_final_teacher.shape,
                W_gate_teacher.shape, W_up_teacher.shape, W_down_teacher.shape)

    # ── Build query pool ─────────────────────────────────────────────────────
    logger.info("Building query pool (%d queries, dist=%s) ...", N, args.query_dist)
    query_ids = build_queries(
        args.query_dist, tokenizer, N, T,
        args.seed, args.reserved_tokens, args.allow_synthetic,
    )
    logger.info("Query pool ready: %s", tuple(query_ids.shape))

    # ── Teacher forward pass: collect h_in / h_mid / h_out / logits ──────────
    logger.info("Collecting teacher hidden states and logits ...")
    teacher_states = collect_states(
        teacher, query_ids, block_idx, device, BS, collect_logits=True,
    )
    h_in_T = teacher_states["h_in"]
    h_mid_T = teacher_states["h_mid"]
    h_out_T = teacher_states["h_out"]
    logits_all = teacher_states["logits"]
    logger.info("  h_in: %s, h_mid: %s, h_out: %s, logits: %s",
                tuple(h_in_T.shape), tuple(h_mid_T.shape),
                tuple(h_out_T.shape), tuple(logits_all.shape))
    del teacher_states

    results: dict = {
        "config": {
            "model_name": args.model_name,
            "block_idx": block_idx,
            "num_queries": N,
            "max_seq_len": T,
            "query_dist": args.query_dist,
            "seed": args.seed,
            "d_model": d,
            "d_ff": d_ff,
            "vocab_size": V,
            "student_init_std": args.student_init_std,
        },
    }

    # ════════════════════════════════════════════════════════════════════════
    # STEP A — Carlini extraction of W_lm_eff = W_lm @ diag(g_final)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("STEP A — Carlini extraction of W_lm_eff (PURE-LOGITS)")
    logger.info("=" * 72)

    t_stepA = time.time()

    # Use last-position logits for SVD (matches reproduce_carlini.py).
    logits_last = logits_all[:, -1, :].contiguous()   # [N, V]
    carlini_out = carlini_extract_lm_head(logits_last, d, out_dir)
    W_hat = carlini_out["W_hat"]                      # [d, V]

    # Build the "true" effective matrix for evaluation only:
    # teacher's lm_head absorbs the final RMSNorm gain g_final, so the true
    # effective matrix is W_lm @ diag(g_final)  (shape [V, d]).
    W_eff_true = (W_lm_teacher * g_final_teacher.unsqueeze(0)).float()     # [V, d]

    # Subspace evaluation: principal angles
    Q_hat, _ = torch.linalg.qr(W_hat.T)               # [V, d]
    Q_true, _ = torch.linalg.qr(W_eff_true)           # [V, d]
    M_align = Q_hat.T @ Q_true                         # [d, d]
    cos_angles = torch.linalg.svdvals(M_align).clamp(0.0, 1.0)

    # Procrustes alignment: find rotation R such that W_hat.T @ R ~ W_eff_true
    W_hat_T = W_hat.T.float()                          # [V, d]
    aligned, R_opt, cos_cols, frob_err = procrustes_align(W_hat_T, W_eff_true)

    # W_lm_eff in the *attacker's* frame (what they use for pinv):
    # We use W_hat.T directly; the rotation is absorbed into subsequent steps.
    W_lm_eff_attacker = W_hat.T.float()                # [V, d]

    stepA_results = {
        "svd_time_s": carlini_out["svd_time_s"],
        "d_hat": carlini_out["d_hat"],
        "singular_values_top20": carlini_out["singular_values_top20"],
        "gap_ratio_at_d": carlini_out["gap_ratio_at_d"],
        "subspace_mean_cos": float(cos_angles.mean()),
        "subspace_min_cos": float(cos_angles.min()),
        "subspace_max_cos": float(cos_angles.max()),
        "procrustes_mean_col_cos": float(cos_cols.mean()),
        "procrustes_min_col_cos": float(cos_cols.min()),
        "procrustes_frob_rel_error": frob_err,
        "time_s": round(time.time() - t_stepA, 2),
    }
    results["step_a_carlini_extraction"] = stepA_results
    logger.info("  Step A done: subspace_cos=%.6f, procrustes_cos=%.6f",
                stepA_results["subspace_mean_cos"],
                stepA_results["procrustes_mean_col_cos"])

    # ════════════════════════════════════════════════════════════════════════
    # STEP B — Recover u(x) = h_L(x)/rms(h_L(x)) via pinv(W_lm_eff)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("STEP B — u(x) = h_L/rms(h_L) recovery via pinv(W_lm_eff)")
    logger.info("=" * 72)

    t_stepB = time.time()

    # Use attacker's W_lm_eff (W_hat.T, rotated to the Carlini frame).
    # For evaluation we also compute u using the true W_eff to measure the
    # "best-case" ceiling — but the actual attack only has access to the
    # attacker-frame matrix.
    u_pure_attacker = recover_u_pure_logits(
        logits_all, W_lm_eff_attacker, device, BS,
    )

    # Teacher's true u(x) = h_L / rms(h_L)
    h_L_rms = rms_scale(h_out_T)                       # [N, T, 1]
    u_true = (h_out_T.float() / h_L_rms.clamp(min=1e-9))

    # Evaluation: cosine(u_attacker vs u_true) — the attacker's u lives in
    # a rotated frame, so direct cosine is misleading. Instead, compare the
    # *rotated* attacker u:  u_rot = u_attacker @ R  (R is the Procrustes
    # rotation solving the Carlini ambiguity).
    # Equivalent: project teacher's u onto the Carlini subspace.
    # Both R and Q_true are [d, d]; careful with shapes.

    # NB: W_lm_eff_attacker = Vh[:d, :].T, so pinv(W_lm_eff_attacker) @ z
    # lives in the *Vh[:d]* row-basis, which is a rotation of the teacher
    # basis. The rotation is the same one Procrustes recovers: ``R_opt``.
    # Derivation (column vectors):
    #   z = W_eff_true @ u_true_col
    #   W_eff_true ≈ W_hat.T @ R_opt     (procrustes definition)
    #   so z ≈ W_hat.T @ (R_opt @ u_true_col)
    #   u_attacker_col = pinv(W_hat.T) @ z ≈ R_opt @ u_true_col
    #   => u_true_col ≈ R_opt.T @ u_attacker_col
    # For row vectors (which is how our tensors are laid out):
    #   u_true_row ≈ u_attacker_row @ R_opt
    u_attacker_rotated = u_pure_attacker @ R_opt.float()   # back to teacher frame

    # Per-vector cosine statistics
    u_flat_true = u_true.reshape(-1, d)
    u_flat_att = u_attacker_rotated.reshape(-1, d)
    per_vec_cos = F.cosine_similarity(u_flat_att, u_flat_true, dim=-1)
    stepB_results = {
        "u_recovery_mean_cos": float(per_vec_cos.mean()),
        "u_recovery_median_cos": float(per_vec_cos.median()),
        "u_recovery_min_cos": float(per_vec_cos.min()),
        "u_recovery_p05_cos": float(torch.quantile(per_vec_cos, 0.05)),
        "u_recovery_p95_cos": float(torch.quantile(per_vec_cos, 0.95)),
        "time_s": round(time.time() - t_stepB, 2),
    }
    results["step_b_u_recovery"] = stepB_results
    logger.info("  u(x) recovery: mean_cos=%.4f  median=%.4f  p05=%.4f  p95=%.4f",
                stepB_results["u_recovery_mean_cos"],
                stepB_results["u_recovery_median_cos"],
                stepB_results["u_recovery_p05_cos"],
                stepB_results["u_recovery_p95_cos"])

    # Keep *rotated* u for downstream use: this places Carlini-recovered
    # directions in the same basis as the student's hidden states.
    u_pure_for_attack = u_attacker_rotated                 # [N, T, d]

    # ════════════════════════════════════════════════════════════════════════
    # Build the student (random prefix, frozen prefix, random lm_head)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("STUDENT — random prefix, frozen 0..%d, suffix [%d..%d] attacked",
                block_idx - 1, block_idx, NUM_LAYERS - 1)
    logger.info("=" * 72)

    student = make_student_from_teacher(
        teacher, args.student_init_std, args.seed, device,
    )
    freeze_student_prefix(student, num_frozen_layers=block_idx)

    # Collect student's hidden states on the same queries.
    # Student's h_in at block block_idx is what the attacker will see as
    # the "boundary state".
    logger.info("Collecting STUDENT hidden states ...")
    student_states = collect_states(
        student, query_ids, block_idx, device, BS, collect_logits=False,
    )
    h_in_S = student_states["h_in"]
    h_mid_S = student_states["h_mid"]
    h_out_S = student_states["h_out"]
    del student_states

    # Sanity: how different are student's boundary states from teacher's?
    per_vec_boundary_cos = F.cosine_similarity(
        h_in_S.reshape(-1, d), h_in_T.reshape(-1, d), dim=-1,
    )
    per_vec_hmid_cos = F.cosine_similarity(
        h_mid_S.reshape(-1, d), h_mid_T.reshape(-1, d), dim=-1,
    )
    per_vec_hout_cos = F.cosine_similarity(
        h_out_S.reshape(-1, d), h_out_T.reshape(-1, d), dim=-1,
    )
    logger.info("  h_in (teacher vs student) cos:  mean=%.4f  median=%.4f",
                per_vec_boundary_cos.mean().item(),
                per_vec_boundary_cos.median().item())
    logger.info("  h_mid cos (T vs S):             mean=%.4f  median=%.4f",
                per_vec_hmid_cos.mean().item(),
                per_vec_hmid_cos.median().item())
    logger.info("  h_out cos (T vs S):             mean=%.4f  median=%.4f",
                per_vec_hout_cos.mean().item(),
                per_vec_hout_cos.median().item())

    results["boundary_diagnostics"] = {
        "h_in_TS_mean_cos": float(per_vec_boundary_cos.mean()),
        "h_in_TS_median_cos": float(per_vec_boundary_cos.median()),
        "h_mid_TS_mean_cos": float(per_vec_hmid_cos.mean()),
        "h_mid_TS_median_cos": float(per_vec_hmid_cos.median()),
        "h_out_TS_mean_cos": float(per_vec_hout_cos.mean()),
        "h_out_TS_median_cos": float(per_vec_hout_cos.median()),
    }

    # ── Student's g_mlp / W_gate / W_up (for ablation D1 "student side") ────
    s_prefix = f"model.layers.{block_idx}"
    s_params = {n: p.data.cpu().clone()
                for n, p in student.named_parameters() if n.startswith(s_prefix)}
    g_mlp_student = s_params[
        f"{s_prefix}.post_attention_layernorm.weight"].float()
    W_gate_student = s_params[f"{s_prefix}.mlp.gate_proj.weight"].float()
    W_up_student = s_params[f"{s_prefix}.mlp.up_proj.weight"].float()
    W_down_student = s_params[f"{s_prefix}.mlp.down_proj.weight"].float()

    # Student block reference (for its MLP/attn forward in the scale estimate)
    student_block = student.model.layers[block_idx]

    # ════════════════════════════════════════════════════════════════════════
    # Attacker data construction (pure-logits regime)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("Building attacker data (pure-logits regime)")
    logger.info("=" * 72)

    attacker_data = build_attacker_data_pure_logits(
        teacher_h_L=h_out_T,
        student_h_in=h_in_S,
        student_h_mid=h_mid_S,
        student_h_out=h_out_S,
        u_pure=u_pure_for_attack,
        g_mlp_student=g_mlp_student,
        device=device,
        batch_size=BS,
    )
    h_mid_attack = attacker_data["h_mid_attack"]     # [N, T, d] = student_h_mid
    r_mlp_attack = attacker_data["r_mlp_attack"]     # [N, T, d] = h_L_est - h_mid_attack

    logger.info("  student_rms mean=%.4f, teacher_rms mean=%.4f, ratio=%.4f",
                attacker_data["student_rms_mean"],
                attacker_data["teacher_rms_mean"],
                attacker_data["rms_ratio_mean"])
    logger.info("  u-direction mean cos (sampled %d): %.4f",
                attacker_data["u_direction_num_samples"],
                attacker_data["u_direction_mean_cos"])

    results["attacker_data_diagnostics"] = {
        "student_rms_mean": attacker_data["student_rms_mean"],
        "teacher_rms_mean": attacker_data["teacher_rms_mean"],
        "rms_ratio_mean": attacker_data["rms_ratio_mean"],
        "rms_ratio_median": attacker_data["rms_ratio_median"],
        "u_direction_mean_cos": attacker_data["u_direction_mean_cos"],
        "u_direction_num_samples": attacker_data["u_direction_num_samples"],
    }

    # ════════════════════════════════════════════════════════════════════════
    # STEP C / ABLATION D1 — Pure-logits algebraic W_down OLS
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("STEP C / D1 — Pure-logits W_down OLS")
    logger.info("   h_mid: student's own,  r_mlp: from Carlini u(x) + student_rms")
    logger.info("   gate/up: STUDENT's random (no oracle)")
    logger.info("=" * 72)

    t_d1 = time.time()
    W_down_rec_D1, ols_metrics_D1 = solve_w_down_ols(
        h_mid_attack, r_mlp_attack,
        g_mlp_student,
        W_gate_student, W_up_student,
        device, BS,
    )

    eval_D1 = evaluate_w_down(
        W_down_rec_D1, W_down_teacher,
        W_gate_teacher, W_up_teacher,
        h_mid_T, (h_out_T - h_mid_T),
        g_mlp_teacher, device, BS,
    )
    d1_time = time.time() - t_d1

    results["step_c_d1_pure_logits"] = {
        **eval_D1,
        **ols_metrics_D1,
        "time_s": round(d1_time, 1),
        "description": ("Pure-logits algebraic: student h_mid + Carlini u(x) for h_L "
                        "estimate + STUDENT gate/up activations in OLS. "
                        "No oracle access to teacher's internal states beyond logits."),
    }
    logger.info("  [D1] W_down cos=%.4f  row_cos=%.4f  frob=%.4f  cond=%.2e",
                eval_D1["W_down_cosine"], eval_D1["W_down_per_row_cosine"],
                eval_D1["W_down_frob_error"],
                ols_metrics_D1["condition_number"])

    # ════════════════════════════════════════════════════════════════════════
    # ABLATION D2 — Pure-logits but with oracle gate/up (control)
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_ablation_d2:
        logger.info("=" * 72)
        logger.info("ABLATION D2 — Pure-logits + oracle gate/up activations")
        logger.info("   h_mid: student's own (MLP *input* is still mismatched!)")
        logger.info("   gate/up: TEACHER's (oracle)")
        logger.info("   This isolates whether the h_mid mismatch alone kills recovery.")
        logger.info("=" * 72)

        t_d2 = time.time()
        # Use the student's h_mid as input to OLS, but build activations
        # from TEACHER's gate/up. The RMSNorm uses TEACHER's g_mlp (otherwise
        # the activation statistics are incoherent).
        W_down_rec_D2, ols_metrics_D2 = solve_w_down_ols(
            h_mid_attack, r_mlp_attack,
            g_mlp_teacher,              # oracle g_mlp
            W_gate_teacher, W_up_teacher,  # oracle gate/up
            device, BS,
        )

        eval_D2 = evaluate_w_down(
            W_down_rec_D2, W_down_teacher,
            W_gate_teacher, W_up_teacher,
            h_mid_T, (h_out_T - h_mid_T),
            g_mlp_teacher, device, BS,
        )
        d2_time = time.time() - t_d2
        results["ablation_d2_oracle_gate_up"] = {
            **eval_D2,
            **ols_metrics_D2,
            "time_s": round(d2_time, 1),
            "description": ("Pure-logits + oracle gate/up: student's h_mid still "
                            "mismatched, but we cheat by using teacher gate/up "
                            "and teacher g_mlp. If this fails, the bottleneck is "
                            "h_mid mismatch, not activation constraint."),
        }
        logger.info("  [D2] W_down cos=%.4f  row_cos=%.4f  frob=%.4f",
                    eval_D2["W_down_cosine"], eval_D2["W_down_per_row_cosine"],
                    eval_D2["W_down_frob_error"])
    else:
        logger.info("Skipping Ablation D2 (--skip_ablation_d2)")

    # ════════════════════════════════════════════════════════════════════════
    # ABLATION D3 — Upper-bound control (FULL ORACLE, like v2/v4 baseline)
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_ablation_d3:
        logger.info("=" * 72)
        logger.info("ABLATION D3 — Full-oracle upper bound (teacher h_mid + teacher r_mlp)")
        logger.info("   This is the paper's v2 oracle regime; should give cos ~ 1.0.")
        logger.info("=" * 72)

        t_d3 = time.time()
        W_down_rec_D3, ols_metrics_D3 = solve_w_down_ols(
            h_mid_T, (h_out_T - h_mid_T),
            g_mlp_teacher,
            W_gate_teacher, W_up_teacher,
            device, BS,
        )
        eval_D3 = evaluate_w_down(
            W_down_rec_D3, W_down_teacher,
            W_gate_teacher, W_up_teacher,
            h_mid_T, (h_out_T - h_mid_T),
            g_mlp_teacher, device, BS,
        )
        d3_time = time.time() - t_d3
        results["ablation_d3_full_oracle"] = {
            **eval_D3,
            **ols_metrics_D3,
            "time_s": round(d3_time, 1),
            "description": ("Full oracle upper bound (teacher h_mid, teacher "
                            "gate/up): should reproduce v2's near-perfect "
                            "W_down recovery."),
        }
        logger.info("  [D3] W_down cos=%.4f  row_cos=%.4f  frob=%.4f",
                    eval_D3["W_down_cosine"], eval_D3["W_down_per_row_cosine"],
                    eval_D3["W_down_frob_error"])
    else:
        logger.info("Skipping Ablation D3 (--skip_ablation_d3)")

    # ════════════════════════════════════════════════════════════════════════
    # ABLATION D4 — Hungarian-aligned cosine (is there permutation gauge?)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("ABLATION D4 — FFN Hungarian alignment on D1 result")
    logger.info("=" * 72)
    rec_dict = {
        f"{prefix}.mlp.gate_proj.weight": W_gate_student,    # student's random
        f"{prefix}.mlp.up_proj.weight":   W_up_student,      # student's random
        f"{prefix}.mlp.down_proj.weight": W_down_rec_D1,
    }
    tea_dict = {
        f"{prefix}.mlp.gate_proj.weight": W_gate_teacher,
        f"{prefix}.mlp.up_proj.weight":   W_up_teacher,
        f"{prefix}.mlp.down_proj.weight": W_down_teacher,
    }
    try:
        aligned_dict = align_ffn_neurons(rec_dict, tea_dict, prefix)
        cos_down_aligned = flat_cosine(
            aligned_dict[f"{prefix}.mlp.down_proj.weight"], W_down_teacher,
        )
        cos_down_row_aligned = per_row_cosine(
            aligned_dict[f"{prefix}.mlp.down_proj.weight"], W_down_teacher,
        )
    except Exception as e:
        logger.warning("Hungarian alignment failed (%s)", e)
        cos_down_aligned = float("nan")
        cos_down_row_aligned = float("nan")

    results["ablation_d4_hungarian_aligned_d1"] = {
        "W_down_aligned_cosine": cos_down_aligned,
        "W_down_aligned_per_row_cosine": cos_down_row_aligned,
        "description": ("Hungarian-aligned W_down cosine for the D1 recovered "
                        "weights. If pure-logits attack is working at all, we'd "
                        "expect alignment to help; if it's noise, alignment "
                        "won't save it."),
    }
    logger.info("  [D4] Aligned W_down cos=%.4f (raw D1 cos=%.4f)",
                cos_down_aligned, results["step_c_d1_pure_logits"]["W_down_cosine"])

    # ════════════════════════════════════════════════════════════════════════
    # ABLATION D5 — Comparison to published S-PSI pure-logits numbers
    # ════════════════════════════════════════════════════════════════════════
    # We don't re-run S-PSI here; we just record the reference point.
    results["ablation_d5_spsi_reference"] = {
        "spsi_pure_logits_w_down_cos_low": 0.12,
        "spsi_pure_logits_w_down_cos_high": 0.14,
        "source": "paper Table X (v2_pure_logits block 23 row)",
        "comment": ("Gradient-based S-PSI in pure-logits recovers "
                    "W_down cos ~ 0.12-0.14. Our algebraic pure-logits D1 "
                    "should be compared to this. If D1 > 0.14, algebra beats "
                    "gradient in pure-logits."),
    }

    # ════════════════════════════════════════════════════════════════════════
    # ABLATION D6 — Functional KL test
    #   Inject W_down_rec_D1 into the student and measure end-to-end KL.
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 72)
    logger.info("ABLATION D6 — Functional KL of student-with-recovered-W_down")
    logger.info("=" * 72)

    t_d6 = time.time()
    functional_metrics = {}

    # Save the student's original W_down so we can restore it
    orig_student_down = student.model.layers[block_idx].mlp.down_proj.weight.data.clone()

    try:
        # (a) Baseline: student with its own random W_down
        functional_metrics["baseline_student_random"] = functional_kl_eval(
            teacher, student, query_ids, block_idx,
            device, args.functional_batch_size,
            max_queries=min(128, N),
        )

        # (b) Inject recovered W_down (pure-logits D1) and re-measure
        with torch.no_grad():
            student.model.layers[block_idx].mlp.down_proj.weight.data.copy_(
                W_down_rec_D1.to(orig_student_down.device).to(orig_student_down.dtype)
            )
        functional_metrics["with_recovered_w_down_d1"] = functional_kl_eval(
            teacher, student, query_ids, block_idx,
            device, args.functional_batch_size,
            max_queries=min(128, N),
        )

        # (c) Oracle: replace ALL MLP weights of block with teacher's
        with torch.no_grad():
            lay = student.model.layers[block_idx]
            lay.mlp.down_proj.weight.data.copy_(
                W_down_teacher.to(orig_student_down.device).to(orig_student_down.dtype)
            )
            lay.mlp.gate_proj.weight.data.copy_(
                W_gate_teacher.to(orig_student_down.device).to(orig_student_down.dtype)
            )
            lay.mlp.up_proj.weight.data.copy_(
                W_up_teacher.to(orig_student_down.device).to(orig_student_down.dtype)
            )
        functional_metrics["with_oracle_mlp_all"] = functional_kl_eval(
            teacher, student, query_ids, block_idx,
            device, args.functional_batch_size,
            max_queries=min(128, N),
        )
    finally:
        # Restore student's original weights so nothing leaks to later tests.
        with torch.no_grad():
            lay = student.model.layers[block_idx]
            lay.mlp.down_proj.weight.data.copy_(orig_student_down)
            lay.mlp.gate_proj.weight.data.copy_(
                W_gate_student.to(orig_student_down.device).to(orig_student_down.dtype)
            )
            lay.mlp.up_proj.weight.data.copy_(
                W_up_student.to(orig_student_down.device).to(orig_student_down.dtype)
            )

    functional_metrics["time_s"] = round(time.time() - t_d6, 1)
    results["ablation_d6_functional_kl"] = functional_metrics
    logger.info(
        "  [D6] KL (teacher||student) baseline=%.4e  with_recovered_W_down=%.4e  "
        "oracle=%.4e",
        functional_metrics["baseline_student_random"]["kl_teacher_student"],
        functional_metrics["with_recovered_w_down_d1"]["kl_teacher_student"],
        functional_metrics["with_oracle_mlp_all"]["kl_teacher_student"],
    )

    # ════════════════════════════════════════════════════════════════════════
    # Save results + recovered weights
    # ════════════════════════════════════════════════════════════════════════
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    weights_path = out_dir / "recovered_weights.pt"
    save_dict = {
        "W_lm_eff_attacker": W_lm_eff_attacker.cpu(),
        "W_hat_carlini": W_hat.cpu(),
        "R_procrustes": R_opt.cpu(),
        "W_down_rec_D1_pure_logits": W_down_rec_D1,
        "u_pure_attacker_rotated": u_pure_for_attack.cpu(),
    }
    if not args.skip_ablation_d2:
        save_dict["W_down_rec_D2_oracle_gate_up"] = W_down_rec_D2
    if not args.skip_ablation_d3:
        save_dict["W_down_rec_D3_full_oracle"] = W_down_rec_D3
    torch.save(save_dict, weights_path)

    # ════════════════════════════════════════════════════════════════════════
    # Print summary
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PURE-LOGITS ALGEBRAIC ATTACK — SUMMARY")
    print("=" * 80)
    print(f"  Model        : {args.model_name}")
    print(f"  Target block : {block_idx}  (of {NUM_LAYERS} layers)")
    print(f"  Queries      : {N} x {T} tokens = {N*T} data points")
    print(f"  Query dist   : {args.query_dist}")
    print("-" * 80)

    print("\n  Step A — Carlini extraction (PURE-LOGITS):")
    sa = results["step_a_carlini_extraction"]
    print(f"    subspace_mean_cos           = {sa['subspace_mean_cos']:.6f}")
    print(f"    procrustes_mean_col_cos     = {sa['procrustes_mean_col_cos']:.6f}")
    print(f"    procrustes_frob_rel_error   = {sa['procrustes_frob_rel_error']:.6f}")
    print(f"    gap ratio at d              = {sa['gap_ratio_at_d']:.2f}")

    print("\n  Step B — u(x) recovery (PURE-LOGITS, rotated to teacher frame):")
    sb = results["step_b_u_recovery"]
    print(f"    u_recovery_mean_cos         = {sb['u_recovery_mean_cos']:.6f}")
    print(f"    u_recovery_median_cos       = {sb['u_recovery_median_cos']:.6f}")
    print(f"    u_recovery_p05_cos          = {sb['u_recovery_p05_cos']:.6f}")
    print(f"    u_recovery_p95_cos          = {sb['u_recovery_p95_cos']:.6f}")

    print("\n  Boundary diagnostics (how different is student's frame?):")
    bd = results["boundary_diagnostics"]
    print(f"    h_in  teacher-vs-student mean cos = {bd['h_in_TS_mean_cos']:.6f}")
    print(f"    h_mid teacher-vs-student mean cos = {bd['h_mid_TS_mean_cos']:.6f}")
    print(f"    h_out teacher-vs-student mean cos = {bd['h_out_TS_mean_cos']:.6f}")

    print("\n  Step C / D1 — Pure-logits W_down OLS (the main experiment):")
    sc = results["step_c_d1_pure_logits"]
    print(f"    W_down_cosine               = {sc['W_down_cosine']:.6f}")
    print(f"    W_down_per_row_cosine       = {sc['W_down_per_row_cosine']:.6f}")
    print(f"    W_down_frob_error           = {sc['W_down_frob_error']:.6f}")
    print(f"    condition_number            = {sc['condition_number']:.2e}")

    if not args.skip_ablation_d2:
        print("\n  D2 — Pure-logits + oracle gate/up:")
        d2 = results["ablation_d2_oracle_gate_up"]
        print(f"    W_down_cosine               = {d2['W_down_cosine']:.6f}")
        print(f"    W_down_per_row_cosine       = {d2['W_down_per_row_cosine']:.6f}")

    if not args.skip_ablation_d3:
        print("\n  D3 — Full oracle upper bound:")
        d3 = results["ablation_d3_full_oracle"]
        print(f"    W_down_cosine               = {d3['W_down_cosine']:.6f}  (expected ~1.0)")
        print(f"    W_down_frob_error           = {d3['W_down_frob_error']:.6f}")

    d4 = results["ablation_d4_hungarian_aligned_d1"]
    print("\n  D4 — Hungarian-aligned D1 W_down:")
    print(f"    aligned cos                 = {d4['W_down_aligned_cosine']}")
    print(f"    aligned per-row cos         = {d4['W_down_aligned_per_row_cosine']}")

    d5 = results["ablation_d5_spsi_reference"]
    print("\n  D5 — S-PSI reference (paper, for comparison):")
    print(f"    S-PSI pure-logits W_down cos ~ "
          f"[{d5['spsi_pure_logits_w_down_cos_low']}, "
          f"{d5['spsi_pure_logits_w_down_cos_high']}]")
    if sc['W_down_cosine'] > d5['spsi_pure_logits_w_down_cos_high']:
        print("    -> D1 BEATS S-PSI pure-logits (algebra > gradient in pure-logits).")
    elif sc['W_down_cosine'] > 0.05:
        print("    -> D1 shows some signal but below S-PSI pure-logits.")
    else:
        print("    -> D1 at noise floor; pure-logits algebraic attack fails.")

    d6 = results["ablation_d6_functional_kl"]
    print("\n  D6 — Functional KL (lower = better match to teacher):")
    print(f"    baseline (student random)    KL = {d6['baseline_student_random']['kl_teacher_student']:.4e}")
    print(f"    with D1 recovered W_down     KL = {d6['with_recovered_w_down_d1']['kl_teacher_student']:.4e}")
    print(f"    oracle MLP                   KL = {d6['with_oracle_mlp_all']['kl_teacher_student']:.4e}")

    print("\n" + "-" * 80)
    print("  Paper verdict:")
    d1_cos = sc["W_down_cosine"]
    if d1_cos < 0.05:
        print(f"    CLEAR RESULT: pure-logits algebraic W_down recovery = {d1_cos:.4f} ≈ 0.")
        print("    CONFIRMS that algebraic internal-layer attack REQUIRES oracle boundary")
        print("    states. This is a strong paper contribution — it delineates what's")
        print("    possible in the pure-logits regime vs oracle regime.")
    elif d1_cos > 0.2:
        print(f"    BREAKTHROUGH: pure-logits algebraic W_down recovery = {d1_cos:.4f} > 0.2.")
        print("    This would be a major finding — algebra works without oracle access.")
    else:
        print(f"    PARTIAL: pure-logits algebraic W_down = {d1_cos:.4f}. Not nothing, not")
        print("    everything. Needs deeper investigation; compare to S-PSI reference.")

    print(f"\n  Results JSON      : {results_path}")
    print(f"  Recovered weights : {weights_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
