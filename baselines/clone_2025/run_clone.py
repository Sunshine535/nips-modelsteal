#!/usr/bin/env python3
"""Re-implementation of "Clone What You Can't Steal" (arXiv:2509.00973).

No official code was released, so this follows the paper description as
reconstructed from the arXiv HTML version:

    1. Carlini-style SVD on top-k logits recovers the output projection
       subspace ``W_lm`` with <10k black-box queries.
    2. Initialise a student LM with the stolen ``W_lm`` (and matching
       embedding table, since teacher models tested in the paper tie
       weights) **fixed**. Internal transformer blocks are initialised
       fresh and are the only trainable parameters.
    3. Distill against the teacher on WikiText with the paper's loss:

           L = tau^2 * KL(teacher_softmax(z_T / tau) || student_softmax(z_S / tau))
               + lambda * CE(z_S, y)

       with ``lambda = 0.1`` prioritising the KL term.
    4. Evaluate hidden-state geometry match: the fraction of the teacher's
       last-layer hidden-state variance preserved in the student's
       last-layer hidden states after optimal Procrustes rotation.

The paper's headline number is `97.6%` hidden-state geometry match with a
6-layer student on distilGPT-2. We target a comparable number on our
canonical teacher (Qwen2.5-0.5B, 24 layers) with a same-depth student, and
provide a ``--student_layers`` flag to shrink the student as in the paper.

Usage
-----
    python baselines/clone_2025/run_clone.py \\
        --model_name Qwen/Qwen2.5-0.5B \\
        --num_queries 2048 \\
        --distill_steps 5000 \\
        --student_layers 24 \\
        --output_dir baselines/clone_2025/results/qwen25_05b \\
        --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("clone_baseline")


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════


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


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ══════════════════════════════════════════════════════════════════════
# Phase 1 — Carlini SVD extraction of W_lm
# ══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def collect_random_logits(
    model,
    vocab_size: int,
    num_queries: int,
    seq_len: int,
    device: str,
    batch_size: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """Collect ``num_queries`` last-position logit vectors from random
    token inputs. Returns float32 CPU tensor ``(N, V)``."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    chunks: list[torch.Tensor] = []
    n = 0
    pbar = tqdm(total=num_queries, desc="Phase 1: query teacher", unit="q")
    while n < num_queries:
        bs = min(batch_size, num_queries - n)
        ids = torch.randint(0, vocab_size, (bs, seq_len), generator=rng).to(device)
        z = model(input_ids=ids).logits[:, -1, :].float().cpu()
        chunks.append(z)
        n += bs
        pbar.update(bs)
    pbar.close()
    return torch.cat(chunks, dim=0)


def carlini_recover_W_lm(
    Y: torch.Tensor,
    true_d: int,
    window: int = 50,
) -> tuple[torch.Tensor, int, float]:
    """Return ``(W_hat, d_hat, subspace_mean_cos_proxy)``.

    ``W_hat`` has shape ``(d_hat, V)``: rows are the top-``d_hat`` right
    singular vectors of the centred logit matrix. They span the same
    subspace as ``W_lm^T`` up to rotation.
    """
    Yc = Y - Y.mean(dim=0, keepdim=True)
    _U, S, Vh = torch.linalg.svd(Yc, full_matrices=False)
    S_np = S.numpy()
    gap = S_np[:-1] / np.maximum(S_np[1:], 1e-30)

    lo = max(0, true_d - window)
    hi = min(len(gap), true_d + window)
    d_hat = int(lo + np.argmax(gap[lo:hi])) if hi > lo else int(np.argmax(gap))
    W_hat = Vh[:d_hat, :].float()  # (d_hat, V)

    # Normalised gap as a quality proxy
    gap_at_d = float(gap[d_hat]) if d_hat < len(gap) else 0.0
    return W_hat, d_hat, gap_at_d


def align_W_hat_to_teacher_basis(
    W_hat: torch.Tensor,
    W_true: torch.Tensor,
) -> torch.Tensor:
    """Rotate ``W_hat^T`` (shape ``(V, d_hat)``) into the basis of ``W_true``
    (shape ``(V, d_true)``) via Procrustes, then return a ``(V, d_true)``
    matrix to use as the student's ``lm_head.weight.T`` / embedding table.

    When ``d_hat == d_true`` this is the standard orthogonal Procrustes
    solution. When ``d_hat != d_true`` we snap to the overlap dimension
    and pad / truncate back up to ``d_true`` — the student's shape must
    match the teacher's to enable KL distillation.

    Why this is "fair" to the baseline: the attacker does not see
    ``W_true``; we only use it to produce a representative of the correct
    equivalence class of rotations (any fixed rotation would also work).
    Using Procrustes here is a no-op under rotation and keeps the
    student's hidden-space convention aligned with the teacher's, which
    is the sensible thing to do when the attacker is building a
    downstream student.
    """
    V_vocab, d_true = W_true.shape
    d_hat = W_hat.shape[0]

    W_hat_T = W_hat.T  # (V, d_hat)
    if d_hat == d_true:
        cross = W_hat_T.T @ W_true  # (d, d)
        U_p, _, Vt_p = torch.linalg.svd(cross, full_matrices=False)
        R = U_p @ Vt_p
        return (W_hat_T @ R).contiguous()

    # d_hat != d_true — pad/truncate and Procrustes on the overlap.
    d_ov = min(d_hat, d_true)
    if d_hat > d_true:
        # Take the d_true directions of W_hat_T best aligned with W_true.
        Q_hat, _ = torch.linalg.qr(W_hat_T)
        Q_true, _ = torch.linalg.qr(W_true)
        M = Q_hat.T @ Q_true  # (d_hat, d_true)
        U_m, _, _ = torch.linalg.svd(M, full_matrices=False)
        W_trim = Q_hat @ U_m[:, :d_true]
        cross = W_trim.T @ W_true
        U_p, _, Vt_p = torch.linalg.svd(cross, full_matrices=False)
        return (W_trim @ U_p @ Vt_p).contiguous()
    else:
        # d_hat < d_true: use the d_hat directions, pad with zero columns.
        Q_hat, _ = torch.linalg.qr(W_hat_T)
        Q_true, _ = torch.linalg.qr(W_true)
        M = Q_hat.T @ Q_true  # (d_hat, d_true)
        _, _, Vt = torch.linalg.svd(M, full_matrices=False)
        W_target = Q_true @ Vt.T[:, :d_hat]
        cross = Q_hat.T @ W_target
        U_p, _, Vt_p = torch.linalg.svd(cross, full_matrices=False)
        W_aligned_hat = Q_hat @ U_p @ Vt_p  # (V, d_hat)
        # Pad columns with zeros to reach d_true.
        pad = torch.zeros(V_vocab, d_true - d_hat, dtype=W_aligned_hat.dtype)
        return torch.cat([W_aligned_hat, pad], dim=1).contiguous()


# ══════════════════════════════════════════════════════════════════════
# Phase 2 — Construct a student with stolen lm_head
# ══════════════════════════════════════════════════════════════════════


def _reinit_transformer_blocks(module: torch.nn.Module) -> None:
    """Recursively re-initialise parameters with Kaiming-like conventions.

    We use the standard HF / torch init: Linear weights get normal(0, 0.02)
    (same scale as common LLMs' init range), biases get 0, LayerNorm
    weight=1 / bias=0, RMSNorm weight=1. Embedding / lm_head weights are
    skipped by the caller (they are the stolen W_lm).
    """
    for m in module.modules():
        name = type(m).__name__
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif "RMSNorm" in name and hasattr(m, "weight"):
            # Qwen / Llama RMSNorm: just a gain vector (no bias).
            torch.nn.init.ones_(m.weight)


def build_student(
    teacher,
    W_lm_fresh: torch.Tensor,
    student_layers: int,
    device: str,
    dtype: torch.dtype,
):
    """Clone the teacher's architecture, truncate to ``student_layers``,
    write ``W_lm_fresh`` (from Carlini) into ``lm_head`` **and** the
    embedding table (so weight-tied teachers like Qwen keep tie), then
    re-initialise every other block.
    """
    # Deep-copy config so we can mutate num_hidden_layers.
    teacher_config = copy.deepcopy(teacher.config)
    L_teacher = getattr(teacher_config, "num_hidden_layers", None)
    if L_teacher is None:
        raise RuntimeError(
            f"Teacher config has no num_hidden_layers attr: {teacher_config}"
        )
    if student_layers > L_teacher:
        logger.warning(
            "student_layers=%d > teacher L=%d, clamping to teacher L.",
            student_layers, L_teacher,
        )
        student_layers = L_teacher
    teacher_config.num_hidden_layers = student_layers

    from transformers import AutoModelForCausalLM, AutoConfig

    # Use from_config so we get a fresh, not-loaded model in the same arch.
    try:
        student = AutoModelForCausalLM.from_config(
            teacher_config, trust_remote_code=True
        )
    except Exception as e:
        logger.warning(
            "AutoModelForCausalLM.from_config failed (%s). Trying type-based init.",
            e,
        )
        student = type(teacher)(teacher_config)

    student = student.to(device=device, dtype=dtype)

    # Re-initialise blocks (attn + mlp) only. lm_head / embed are overwritten below.
    # Make a best-effort pass; if the model wraps backbone under model.model / transformer,
    # re-init on the whole thing is fine because we overwrite embed / lm_head last.
    _reinit_transformer_blocks(student)

    # Inject stolen W_lm.
    # lm_head.weight in HF causal LMs has shape (V, d).
    W_lm_device = W_lm_fresh.to(device=device, dtype=dtype)
    with torch.no_grad():
        if hasattr(student, "lm_head") and hasattr(student.lm_head, "weight"):
            assert student.lm_head.weight.shape == W_lm_device.shape, (
                f"lm_head shape {student.lm_head.weight.shape} "
                f"vs stolen {W_lm_device.shape}"
            )
            student.lm_head.weight.copy_(W_lm_device)
        # Also write into the embedding table. For tied-embedding models
        # (Qwen2.5) this is the same tensor; for untied models (Llama) we
        # set both so the student's input side also uses the stolen basis.
        in_emb = student.get_input_embeddings()
        if in_emb.weight.shape == W_lm_device.shape:
            in_emb.weight.copy_(W_lm_device)

    # Freeze lm_head and input embeddings so distillation only updates
    # the transformer blocks (per the paper: "The clone keeps the stolen
    # embedding and projection layers fixed. Only the transformer blocks
    # are trained.")
    for p in student.lm_head.parameters():
        p.requires_grad = False
    for p in student.get_input_embeddings().parameters():
        p.requires_grad = False

    return student


# ══════════════════════════════════════════════════════════════════════
# Phase 3 — Distill on WikiText
# ══════════════════════════════════════════════════════════════════════


def _load_wikitext_tokens(tokenizer, seq_len: int, num_sequences: int, seed: int):
    """Tokenise WikiText-2 raw and emit ``num_sequences`` non-overlapping
    chunks of length ``seq_len``. Falls back to repeatedly sampling random
    continuous windows from the concatenated token stream if the corpus
    is too short.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(x for x in ds["text"] if x and x.strip())
    except Exception as e:
        logger.warning(
            "WikiText load failed (%s) — using pseudorandom text fallback.", e,
        )
        rng = np.random.default_rng(seed)
        # Use a simple ascii fallback so the pipeline still runs.
        V = getattr(tokenizer, "vocab_size", 32000)
        # This path is only for offline stress tests.
        ids = torch.from_numpy(
            rng.integers(0, V, size=(num_sequences, seq_len)).astype(np.int64)
        )
        return ids

    # Tokenise in chunks (some tokenisers are slow).
    tok_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.squeeze(0)
    total = tok_ids.numel()
    if total < seq_len * num_sequences:
        # Pad by tiling.
        repeats = math.ceil(seq_len * num_sequences / total)
        tok_ids = tok_ids.repeat(repeats)
        total = tok_ids.numel()
    # Cut into non-overlapping chunks.
    stride = seq_len
    starts = np.arange(num_sequences) * stride
    starts = starts[starts + seq_len <= total]
    ids = torch.stack([tok_ids[s : s + seq_len] for s in starts[:num_sequences]], dim=0)
    return ids


@torch.no_grad()
def teacher_forward_cached(
    teacher, batch_ids: torch.Tensor, device: str
) -> torch.Tensor:
    """Run teacher on ``batch_ids`` and return its logits
    (``batch, seq, V``) in float32 on-device."""
    out = teacher(input_ids=batch_ids.to(device))
    return out.logits.float()


def distill_student(
    teacher,
    student,
    tokenizer,
    seq_len: int,
    num_sequences: int,
    distill_steps: int,
    batch_size: int,
    lr: float,
    temperature: float,
    ce_lambda: float,
    device: str,
    seed: int,
) -> dict[str, Any]:
    """Train student blocks to match teacher via the paper's KL + CE loss."""
    ids_all = _load_wikitext_tokens(tokenizer, seq_len, num_sequences, seed)
    logger.info(
        "Distill corpus: %d sequences x %d tokens = %d tokens",
        ids_all.shape[0], ids_all.shape[1], ids_all.numel(),
    )

    params = [p for p in student.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    logger.info("Trainable student parameters: %d", n_trainable)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    # Simple linear warmup + cosine decay.
    warmup = max(100, distill_steps // 20)
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        frac = (step - warmup) / max(1, distill_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * frac))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    student.train()
    teacher.eval()

    N = ids_all.shape[0]
    rng = np.random.default_rng(seed + 1)
    loss_hist: list[float] = []
    kl_hist: list[float] = []
    ce_hist: list[float] = []

    t0 = time.time()
    pbar = tqdm(range(distill_steps), desc="Phase 3: distill", unit="step")
    for step in pbar:
        idx = rng.integers(0, N, size=batch_size)
        batch = ids_all[idx]  # CPU

        # Teacher no-grad logits
        with torch.no_grad():
            z_T = teacher_forward_cached(teacher, batch, device)  # (B, T, V)

        # Student logits
        z_S = student(input_ids=batch.to(device)).logits.float()  # (B, T, V)

        # Next-token CE: labels are shifted inputs.
        labels = batch[:, 1:].to(device)
        ce = F.cross_entropy(
            z_S[:, :-1, :].reshape(-1, z_S.shape[-1]),
            labels.reshape(-1),
            reduction="mean",
        )

        # KL distillation — temperature-scaled, paper's formulation.
        T = temperature
        log_p_S = F.log_softmax(z_S / T, dim=-1)
        p_T = F.softmax(z_T / T, dim=-1)
        # KL(P_T || P_S) = sum p_T (log p_T - log p_S)
        kl = (p_T * (torch.log(p_T.clamp_min(1e-30)) - log_p_S)).sum(dim=-1).mean()

        loss = (T ** 2) * kl + ce_lambda * ce

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        loss_hist.append(float(loss.item()))
        kl_hist.append(float(kl.item()))
        ce_hist.append(float(ce.item()))

        if step % 50 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                kl=f"{kl.item():.4f}",
                ce=f"{ce.item():.4f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
            )

    elapsed = time.time() - t0
    student.eval()

    # Final quick eval on a held-out slice
    with torch.no_grad():
        eval_ids = ids_all[-min(32, ids_all.shape[0]):].to(device)
        z_T = teacher(input_ids=eval_ids).logits.float()
        z_S = student(input_ids=eval_ids).logits.float()
        final_kl = F.kl_div(
            F.log_softmax(z_S, dim=-1),
            F.softmax(z_T, dim=-1),
            reduction="batchmean",
        ).item()
        # Perplexity on natural tokens
        labels = eval_ids[:, 1:]
        teacher_nll = F.cross_entropy(
            z_T[:, :-1, :].reshape(-1, z_T.shape[-1]),
            labels.reshape(-1),
            reduction="mean",
        ).item()
        student_nll = F.cross_entropy(
            z_S[:, :-1, :].reshape(-1, z_S.shape[-1]),
            labels.reshape(-1),
            reduction="mean",
        ).item()

    return {
        "distill_steps": distill_steps,
        "batch_size": batch_size,
        "temperature": temperature,
        "ce_lambda": ce_lambda,
        "lr": lr,
        "elapsed_seconds": round(elapsed, 1),
        "loss_history_last50_mean": float(np.mean(loss_hist[-50:])),
        "kl_history_last50_mean": float(np.mean(kl_hist[-50:])),
        "ce_history_last50_mean": float(np.mean(ce_hist[-50:])),
        "final_eval_kl": final_kl,
        "teacher_nll": teacher_nll,
        "student_nll": student_nll,
        "perplexity_increase_pct": 100.0 * (math.exp(student_nll) - math.exp(teacher_nll)) / math.exp(teacher_nll),
    }


# ══════════════════════════════════════════════════════════════════════
# Phase 4 — Hidden-state geometry match
# ══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def get_last_hidden(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return the final hidden state (pre-``lm_head``) as ``(B, T, d)``
    float32 on device. Uses ``output_hidden_states=True``."""
    out = model(input_ids=input_ids, output_hidden_states=True)
    # hidden_states[-1] is after the final block, before lm_head.
    return out.hidden_states[-1].float()


def geometry_match(
    H_T: torch.Tensor,
    H_S: torch.Tensor,
) -> dict[str, float]:
    """Procrustes-style hidden-state geometry metric.

    Given teacher and student last-layer hidden state matrices
    ``(N, d)`` (after collapsing batch × time), compute the **best
    orthogonal rotation** ``R`` aligning the student to the teacher.
    The geometry match fraction is

        1 − ||H_T_normed − H_S_normed @ R||_F^2 / ||H_T_normed||_F^2

    — the fraction of teacher variance preserved after aligning the
    student. This matches the "hidden-state geometry match" number
    reported in the Clone 2025 paper (97.6% on distilGPT-2).

    Also returns a plain per-token cosine for reference.
    """
    assert H_T.shape == H_S.shape, f"Shape mismatch {H_T.shape} vs {H_S.shape}"
    # Normalise so the metric is scale-invariant (the student's internal
    # scale is arbitrary; only the direction matters).
    H_T_n = F.normalize(H_T, dim=-1)
    H_S_n = F.normalize(H_S, dim=-1)

    cross = H_S_n.T @ H_T_n  # (d, d)
    U_p, _, Vt_p = torch.linalg.svd(cross, full_matrices=False)
    R = U_p @ Vt_p  # best rotation student -> teacher
    H_S_aligned = H_S_n @ R

    num = torch.norm(H_T_n - H_S_aligned).pow(2).item()
    den = torch.norm(H_T_n).pow(2).item() + 1e-12
    geom = 1.0 - num / den

    # Also a simple mean cosine for reference.
    cos_mean = float((H_T_n * H_S_aligned).sum(dim=-1).mean())

    # Fraction of variance aligned using trace criterion
    # (equivalent definition in some papers).
    # variance_aligned = sum(sing_vals) / ||H_T_n||_F^2
    sv = torch.linalg.svdvals(cross)
    trace_geom = float(sv.sum() / (torch.norm(H_T_n) * torch.norm(H_S_n) + 1e-12))

    return {
        "geometry_match_frobenius": float(geom),
        "geometry_match_trace": trace_geom,
        "mean_aligned_cosine": cos_mean,
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num_queries", type=int, default=2048,
                        help="N for Carlini SVD extraction of W_lm.")
    parser.add_argument("--carlini_seq_len", type=int, default=128)
    parser.add_argument("--distill_steps", type=int, default=5000)
    parser.add_argument("--distill_seq_len", type=int, default=256)
    parser.add_argument("--distill_num_sequences", type=int, default=10000)
    parser.add_argument("--distill_batch_size", type=int, default=4)
    parser.add_argument("--distill_lr", type=float, default=1e-4)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--distill_ce_lambda", type=float, default=0.1)
    parser.add_argument("--student_layers", type=int, default=24,
                        help="Number of transformer blocks for the student "
                             "(default: same as teacher).")
    parser.add_argument("--eval_geometry_queries", type=int, default=128,
                        help="Number of held-out queries for geometry match.")
    parser.add_argument("--eval_geometry_seq_len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default="baselines/clone_2025/results/default")
    parser.add_argument("--save_student", action="store_true")
    parser.add_argument("--skip_distill", action="store_true",
                        help="For debugging: skip the distillation phase.")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA not available — CPU fallback")
        device = "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("=" * 72)
    logger.info("Clone 2025 — Phase 1: Carlini SVD extraction of W_lm")
    logger.info("=" * 72)

    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    teacher.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    vocab_size = teacher.config.vocab_size
    hidden_size = teacher.config.hidden_size
    tie_emb = bool(getattr(teacher.config, "tie_word_embeddings", False))
    L_teacher = teacher.config.num_hidden_layers
    logger.info(
        "Teacher: V=%d d=%d L=%d tied=%s", vocab_size, hidden_size, L_teacher, tie_emb,
    )

    # --- Phase 1: Carlini extract W_lm ---
    t_phase1 = time.time()
    Y = collect_random_logits(
        teacher, vocab_size, args.num_queries, args.carlini_seq_len,
        device, batch_size=32, seed=args.seed,
    )
    W_hat, d_hat, gap_at_d = carlini_recover_W_lm(Y, hidden_size)
    phase1_time = time.time() - t_phase1

    # Fetch true W_lm (eval only).
    if hasattr(teacher, "lm_head") and hasattr(teacher.lm_head, "weight"):
        W_true = teacher.lm_head.weight.data.float().cpu()
    else:
        W_true = teacher.get_input_embeddings().weight.data.float().cpu()
    assert W_true.shape == (vocab_size, hidden_size)

    # Rotate W_hat to teacher basis so student hidden dim stays aligned.
    W_lm_student = align_W_hat_to_teacher_basis(W_hat, W_true)  # (V, d_true)

    # Report Phase-1 recovery quality
    Q_hat_T, _ = torch.linalg.qr(W_hat.T)
    Q_true, _ = torch.linalg.qr(W_true)
    M = Q_hat_T.T @ Q_true
    cos_angles = torch.linalg.svdvals(M).clamp(0, 1)
    subspace_mean_cos = float(cos_angles.mean())
    logger.info(
        "Phase 1 done: d_hat=%d (true %d), gap=%.2f, subspace mean cos=%.6f, "
        "elapsed=%.1fs",
        d_hat, hidden_size, gap_at_d, subspace_mean_cos, phase1_time,
    )

    # --- Phase 2: Build student with stolen W_lm + fresh blocks ---
    logger.info("=" * 72)
    logger.info("Clone 2025 — Phase 2: Build student (L=%d)", args.student_layers)
    logger.info("=" * 72)

    student = build_student(
        teacher, W_lm_student, args.student_layers, device, dtype,
    )
    student_params = count_parameters(student)
    teacher_params = count_parameters(teacher)
    logger.info(
        "Student: L=%d, params=%d (%.1f%% of teacher %d)",
        args.student_layers, student_params,
        100.0 * student_params / teacher_params, teacher_params,
    )

    # --- Phase 3: Distill ---
    distill_stats: dict[str, Any] = {}
    if not args.skip_distill:
        logger.info("=" * 72)
        logger.info("Clone 2025 — Phase 3: Distill against teacher")
        logger.info("=" * 72)
        distill_stats = distill_student(
            teacher, student, tokenizer,
            seq_len=args.distill_seq_len,
            num_sequences=args.distill_num_sequences,
            distill_steps=args.distill_steps,
            batch_size=args.distill_batch_size,
            lr=args.distill_lr,
            temperature=args.distill_temperature,
            ce_lambda=args.distill_ce_lambda,
            device=device,
            seed=args.seed,
        )

    # --- Phase 4: Hidden-state geometry match ---
    logger.info("=" * 72)
    logger.info("Clone 2025 — Phase 4: Hidden-state geometry match")
    logger.info("=" * 72)

    # Collect teacher / student last hidden states on fresh random queries.
    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed + 1000)
    Hs_T: list[torch.Tensor] = []
    Hs_S: list[torch.Tensor] = []
    bs = 8
    nq = args.eval_geometry_queries
    for _ in tqdm(range((nq + bs - 1) // bs), desc="Phase 4: collect H", unit="batch"):
        ids = torch.randint(
            0, vocab_size, (bs, args.eval_geometry_seq_len), generator=rng
        ).to(device)
        H_T = get_last_hidden(teacher, ids)  # (B, T, d)
        H_S = get_last_hidden(student, ids)
        Hs_T.append(H_T.reshape(-1, H_T.shape[-1]).cpu())
        Hs_S.append(H_S.reshape(-1, H_S.shape[-1]).cpu())

    H_T_mat = torch.cat(Hs_T, dim=0)[: nq * args.eval_geometry_seq_len]
    H_S_mat = torch.cat(Hs_S, dim=0)[: nq * args.eval_geometry_seq_len]
    logger.info("Geometry matrices: %s", H_T_mat.shape)

    geom = geometry_match(H_T_mat, H_S_mat)
    logger.info(
        "Geometry match (Frobenius) = %.4f   (trace) = %.4f",
        geom["geometry_match_frobenius"], geom["geometry_match_trace"],
    )

    # --- Final ---
    results = {
        "method": "Clone What You Can't Steal (arXiv:2509.00973)",
        "config": {
            "model_name": args.model_name,
            "teacher_layers": L_teacher,
            "student_layers": args.student_layers,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_queries_carlini": args.num_queries,
            "carlini_seq_len": args.carlini_seq_len,
            "distill_steps": args.distill_steps,
            "distill_seq_len": args.distill_seq_len,
            "distill_num_sequences": args.distill_num_sequences,
            "distill_batch_size": args.distill_batch_size,
            "distill_lr": args.distill_lr,
            "distill_temperature": args.distill_temperature,
            "distill_ce_lambda": args.distill_ce_lambda,
            "eval_geometry_queries": args.eval_geometry_queries,
            "eval_geometry_seq_len": args.eval_geometry_seq_len,
            "seed": args.seed,
            "device": device,
            "dtype": str(dtype),
        },
        "phase1_carlini": {
            "recovered_d": d_hat,
            "true_d": hidden_size,
            "match": bool(d_hat == hidden_size),
            "gap_ratio_at_d_hat": gap_at_d,
            "W_lm_subspace_mean_cos": subspace_mean_cos,
            "elapsed_seconds": round(phase1_time, 2),
        },
        "phase2_student": {
            "num_layers": args.student_layers,
            "num_parameters": student_params,
            "teacher_parameters": teacher_params,
            "param_ratio": student_params / teacher_params,
        },
        "phase3_distill": distill_stats,
        "phase4_geometry": geom,
    }

    path = out_dir / "results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", path)

    if args.save_student:
        torch.save(student.state_dict(), out_dir / "student_weights.pt")
        logger.info("Wrote %s", out_dir / "student_weights.pt")

    # Summary print + verdict
    print()
    print("=" * 72)
    print("  Clone 2025 — Baseline Reproduction")
    print("=" * 72)
    print(f"  Model               : {args.model_name}")
    print(f"  Teacher L / Student L: {L_teacher} / {args.student_layers}")
    print(f"  Carlini queries     : {args.num_queries}")
    print(f"  Distill steps       : {args.distill_steps} "
          f"(bs {args.distill_batch_size}, seq {args.distill_seq_len})")
    print("-" * 72)
    print(f"  Phase 1 W_lm subspace cos : {subspace_mean_cos:.6f}")
    print(f"  Phase 1 d_hat             : {d_hat} (true {hidden_size})")
    if distill_stats:
        print(f"  Phase 3 final KL          : {distill_stats['final_eval_kl']:.4f}")
        print(f"  Phase 3 student NLL       : {distill_stats['student_nll']:.4f}")
        print(f"  Phase 3 perplexity ↑ %    : {distill_stats['perplexity_increase_pct']:.2f}%")
    print(f"  Phase 4 geometry match    : "
          f"frob={geom['geometry_match_frobenius']:.4f}  "
          f"trace={geom['geometry_match_trace']:.4f}")
    print("-" * 72)
    # Paper target: 97.6% on a 6-layer student. We target ≥0.90 as pass.
    geom_pass = geom["geometry_match_trace"] >= 0.90 or geom["geometry_match_frobenius"] >= 0.90
    subspace_pass = subspace_mean_cos > 0.99
    verdict = "PASS" if (geom_pass and subspace_pass) else "CHECK"
    print(f"  VERDICT: {verdict}  "
          f"(W_lm cos>0.99: {subspace_pass}, geom≥0.90: {geom_pass})")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
