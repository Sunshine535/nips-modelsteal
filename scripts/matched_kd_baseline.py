#!/usr/bin/env python3
# SAFETY NOTICE: QUARANTINED (alpha-theory prune 2026-04-19)
# This script is NOT cited in the paper. It was part of killed branches:
#   A1 S-PSI, A2 Moments CP, A4 logit-bias, A5 memory probing, A6 active query,
#   A7 algebraic v2/v3/v4, B3 matched-KD.
# Retained in repo for reproducibility of quarantined history; do not use for
# new claims.
#!/usr/bin/env python3
"""Matched-Budget KD Baseline — head-to-head against S-PSI.

============================================================================
SAFETY NOTICE (2026-04-19)
----------------------------------------------------------------------------
The `kd_pure_logits` variant in this script IS NOT a valid black-box
baseline as currently implemented.  `make_student(...)` loads the student
via `AutoModelForCausalLM.from_pretrained(model_name, ...)` (i.e., the
teacher checkpoint) and `randomize_suffix(...)` only reinitializes the
suffix blocks + lm_head.  The non-suffix (prefix) blocks therefore INHERIT
the teacher's weights — this is a teacher leak and the pure-logits variant
cannot be cited as a leak-free KD baseline.

The `kd_joint_oracle` and `kd_progressive_oracle` variants explicitly
inject teacher boundary states, so they are oracle-conditional by design.
They should be labelled as such wherever their numbers are reported.

No paper-body result in main.tex currently cites results from this
script.  This file is retained for development purposes only.  Before
citing any output of this script in a paper table, the `kd_pure_logits`
student must be re-implemented via `make_student_from_config(...)` (see
`scripts/attack_jacobian_fd.py:make_student_from_config`) to avoid the
prefix teacher leak.
============================================================================

Responds to the paper's own §8 Limitations admission:

    "No KD baseline. The functional recovery we observe (Table 8) may be
     entirely explainable by standard knowledge distillation dynamics."

This script runs a matched-budget comparison between S-PSI and *pure*
knowledge distillation, holding every confounder fixed:

    * same suffix (lm_head + last N blocks, Kaiming random init, lm_head untied)
    * same query pool (2048 WikiText-103 sequences, T=128, 80/20 split)
    * same query budget (500K)
    * same optimizer (Adam, lr=1e-4 blocks / lr=1e-3 lm_head, grad_clip=1.0)
    * same precision (bf16 forward, fp32 optimizer state)
    * same last-K logit positions for matching
    * same Hungarian / Procrustes evaluation pipeline

The only thing that differs is the LOSS and the use of S-PSI-specific
tricks (which are explicitly DISABLED here):

    NO algebraic / warm-start initialization (pure Kaiming random)
    NO sensitivity / perturbation-delta matching loss
    NO gauge projection during training
    NO per-block progressive curriculum (or an explicit "progressive"
      variant with KD loss only, for head-to-head comparison)

Variants
--------
kd_joint_oracle
    All suffix params optimized jointly with Hinton KD loss and the same
    oracle boundary (teacher h_{L-N-1}) injected at the first suffix
    block's input.

kd_progressive_oracle
    Same oracle injection, but blocks are trained one-at-a-time
    (last→first), exactly matching S-PSI's curriculum — except the loss
    is pure KD.  Teases apart "curriculum" vs "loss" effects.

kd_pure_logits
    No oracle, no boundary injection — pure logit-matching KD from
    random prefix.  This is the classical black-box baseline.

Output matches S-PSI's schema so the comparison builder picks up numbers
automatically:

    results/v5_matched_kd/{variant}_{seed}/kd_summary.json
    results/v5_matched_kd/{variant}_{seed}/regime_summary.json

Command::

    CUDA_VISIBLE_DEVICES=0 python scripts/matched_kd_baseline.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --num_queries 2048 --max_seq_len 128 \
        --num_suffix_blocks 2 \
        --output_dir results/v5_matched_kd \
        --seeds 42 123 777 \
        --variants kd_joint_oracle kd_progressive_oracle
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.permutation_alignment import (  # noqa: E402
    compute_aligned_cosine,
    compute_lm_head_aligned_cosine,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Setup helpers
# ============================================================================


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matched-budget KD baseline")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--num_queries", type=int, default=2048,
                   help="Total query sequences (before 80/20 split)")
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--num_suffix_blocks", type=int, default=2)
    p.add_argument("--query_budget", type=int, default=500_000)
    p.add_argument("--logit_suffix_positions", type=int, default=8,
                   help="Last K positions used for logit matching (S-PSI: 8)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4,
                   help="LR for transformer blocks")
    p.add_argument("--lm_head_lr", type=float, default=1e-3,
                   help="LR for lm_head")
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Hinton KD loss
    p.add_argument("--kd_temperature", type=float, default=2.0)
    p.add_argument("--kd_lambda_ce", type=float, default=0.1,
                   help="Weight on next-token cross-entropy")

    # Steps per phase
    p.add_argument("--joint_max_steps", type=int, default=5500,
                   help="Max steps for kd_joint_oracle / kd_pure_logits")
    p.add_argument("--block_steps", type=int, default=2000,
                   help="Per-block steps for kd_progressive_oracle")
    p.add_argument("--lm_head_steps", type=int, default=1500,
                   help="Final lm_head steps for kd_progressive_oracle")
    p.add_argument("--convergence_threshold", type=float, default=1e-7)
    p.add_argument("--patience", type=int, default=500)

    p.add_argument("--heldout_fraction", type=float, default=0.2)
    p.add_argument("--functional_eval_queries", type=int, default=256,
                   help="Num held-out queries used for functional eval")

    p.add_argument("--output_dir", type=str,
                   default="results/v5_matched_kd")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    p.add_argument("--variants", type=str, nargs="+",
                   default=["kd_joint_oracle",
                            "kd_progressive_oracle",
                            "kd_pure_logits"],
                   choices=["kd_joint_oracle",
                            "kd_progressive_oracle",
                            "kd_pure_logits"])
    p.add_argument("--allow_synthetic", action="store_true",
                   help="Fall back to random tokens if wikitext is unavailable")
    p.add_argument("--force_rerun", action="store_true")
    return p.parse_args()


# ============================================================================
# Query pool (identical to run_spsi.py)
# ============================================================================


def build_query_pool(
    tokenizer,
    pool_size: int,
    max_seq_len: int,
    seed: int,
    allow_synthetic: bool = False,
) -> torch.Tensor:
    input_ids_list: list[torch.Tensor] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                          split="validation")
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
    except Exception as exc:  # noqa: BLE001
        if not allow_synthetic:
            raise RuntimeError(
                f"Dataset load failed: {exc}. "
                "Use --allow_synthetic to fall back to random tokens."
            ) from exc
        logger.warning("Dataset load failed: %s. Falling back to random.",
                       exc)

    remaining = pool_size - len(input_ids_list)
    if remaining > 0:
        if not allow_synthetic:
            raise RuntimeError(
                f"Only {len(input_ids_list)}/{pool_size} from dataset."
            )
        rng = torch.Generator().manual_seed(seed + 137)
        random_ids = torch.randint(
            3, tokenizer.vocab_size, (remaining, max_seq_len), generator=rng)
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    return torch.stack(input_ids_list[:pool_size])


# ============================================================================
# Model surgery: untie lm_head, randomize suffix
# ============================================================================


def untie_lm_head(student: nn.Module) -> bool:
    lm_head = getattr(student, "lm_head", None)
    embed = None
    if hasattr(student, "model") and hasattr(student.model, "embed_tokens"):
        embed = student.model.embed_tokens
    elif hasattr(student, "transformer") and hasattr(student.transformer,
                                                     "wte"):
        embed = student.transformer.wte
    if lm_head is None or embed is None:
        return False
    if lm_head.weight.data_ptr() == embed.weight.data_ptr():
        lm_head.weight = nn.Parameter(
            lm_head.weight.data.clone(),
            requires_grad=lm_head.weight.requires_grad,
        )
        if hasattr(student, "config"):
            student.config.tie_word_embeddings = False
        logger.info("Untied lm_head from embed_tokens.")
        return True
    return False


def get_num_blocks(model: nn.Module) -> int:
    max_idx = -1
    for name, _ in model.named_parameters():
        for part in name.split("."):
            if part.isdigit():
                max_idx = max(max_idx, int(part))
                break
    return max_idx + 1


def _param_in_blocks(name: str, target_blocks: set[int]) -> bool:
    for part in name.split("."):
        if part.isdigit() and int(part) in target_blocks:
            return True
    return False


def randomize_suffix(
    student: nn.Module,
    num_blocks: int,
    num_suffix_blocks: int,
    include_lm_head: bool = True,
) -> list[str]:
    """Kaiming-randomize suffix block weights + lm_head.

    Returns the list of parameter names that were re-initialized.
    """
    if include_lm_head:
        untie_lm_head(student)

    last_block = num_blocks - 1
    target_blocks = set(range(last_block - num_suffix_blocks + 1,
                              last_block + 1))
    reset_names: list[str] = []

    for name, param in student.named_parameters():
        reset = False
        if include_lm_head and "lm_head" in name:
            reset = True
        elif _param_in_blocks(name, target_blocks):
            reset = True
        if not reset:
            continue

        if param.dim() >= 2:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        else:
            if ("layernorm" in name.lower()
                    or "rmsnorm" in name.lower()
                    or "norm" in name.lower()):
                nn.init.ones_(param)
            else:
                nn.init.zeros_(param)
        reset_names.append(name)

    return reset_names


def get_block_param_names(model: nn.Module, block_idx: int) -> list[str]:
    names: list[str] = []
    target = str(block_idx)
    for name, _ in model.named_parameters():
        for part in name.split("."):
            if part == target:
                names.append(name)
                break
    return names


def get_lm_head_param_names(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, _ in model.named_parameters():
        if "lm_head" in name:
            names.append(name)
    return names


# ============================================================================
# Oracle boundary injection via forward pre-hook
# ============================================================================


def _get_block_module(model: nn.Module, block_idx: int) -> Optional[nn.Module]:
    target = str(block_idx)
    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-1] == target:
            return module
    return None


class OracleBoundaryInjector:
    """Forward pre-hook that replaces the input of a student block with
    a cached teacher hidden state from *exactly* that position.

    Mirrors ``_BoundaryInjectionHook`` from ``src.parameter_inverter`` so
    this script is self-contained.
    """

    def __init__(
        self,
        student: nn.Module,
        boundary_hidden_states: torch.Tensor,  # [N, T, d], cached on CPU
        block_idx: int,
    ):
        self.student = student
        self.boundary = boundary_hidden_states
        self.block_idx = block_idx
        self._current: Optional[torch.Tensor] = None
        self._handle = None

    def set_batch(self, indices: torch.Tensor) -> None:
        device = next(self.student.parameters()).device
        self._current = self.boundary[indices].to(
            device=device, dtype=torch.bfloat16)

    def clear_batch(self) -> None:
        self._current = None

    def __enter__(self):
        block = _get_block_module(self.student, self.block_idx)
        if block is None:
            raise RuntimeError(
                f"Could not locate block module for idx={self.block_idx}")

        def hook_fn(module, args):
            if self._current is None:
                return args
            if isinstance(args, tuple) and len(args) > 0:
                return (self._current,) + args[1:]
            return args

        self._handle = block.register_forward_pre_hook(hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._current = None


# ============================================================================
# Teacher caching: logits, boundary state, next-token targets
# ============================================================================


@torch.no_grad()
def precompute_teacher_artifacts(
    teacher: nn.Module,
    query_ids: torch.Tensor,
    boundary_block_idx: int,
    logit_suffix_positions: int,
    device: str,
    batch_size: int = 16,
) -> dict[str, torch.Tensor]:
    """Pre-compute teacher logits, next-token targets, and boundary states.

    Returns dict with:
        clean_logits        [N, K, V]   bf16 on CPU
        next_token_targets  [N, K]      long on CPU (labels for CE)
        boundary_states     [N, T, d]   bf16 on CPU  (input to block boundary_block_idx+1)
    """
    teacher.eval()
    N = query_ids.shape[0]
    K = logit_suffix_positions

    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    boundary_chunks: list[torch.Tensor] = []

    total_queries = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = query_ids[start:end].to(device)

        out = teacher(batch, output_hidden_states=True, return_dict=True)
        logits_full = out.logits  # [b, T, V]
        if K > 0:
            logits_kept = logits_full[:, -K:, :]
        else:
            logits_kept = logits_full

        # Next-token targets: for position t we want x_{t+1}. For the last
        # position there is no teacher forcing target — we drop it by using
        # the clean teacher top-1 prediction (matches the KD practice of
        # pseudo-labels when no supervision is available).
        if K > 0:
            start_pos = logits_full.size(1) - K
            end_pos = logits_full.size(1)
            # real labels for positions [start_pos, end_pos - 2]
            true_next = batch[:, start_pos + 1: end_pos]  # [b, K-1]
            # pseudo label for final position = top-1 of teacher
            pseudo = logits_full[:, -1, :].argmax(dim=-1, keepdim=True)
            if true_next.size(1) == K - 1:
                nxt = torch.cat([true_next, pseudo], dim=1)
            else:
                nxt = pseudo.repeat(1, K)
            target_chunks.append(nxt.detach().cpu())
        else:
            shifted = torch.cat([batch[:, 1:],
                                 logits_full.argmax(dim=-1)[:, -1:]], dim=1)
            target_chunks.append(shifted.detach().cpu())

        logits_chunks.append(logits_kept.detach().to(torch.bfloat16).cpu())

        # Boundary state: hidden_states[i] is the INPUT to block i
        # (index 0 = embedding output, index L = final residual).
        # We want the input to the first SUFFIX block. If the first suffix
        # block index is b, we store hidden_states[b].
        #
        # Convention in this script: boundary_block_idx is the *pre-suffix*
        # hidden-state index. We grab hidden_states[boundary_block_idx + 1].
        bnd = out.hidden_states[boundary_block_idx + 1]
        boundary_chunks.append(bnd.detach().to(torch.bfloat16).cpu())

        total_queries += batch.size(0)

    return {
        "clean_logits": torch.cat(logits_chunks, dim=0),
        "next_token_targets": torch.cat(target_chunks, dim=0),
        "boundary_states": torch.cat(boundary_chunks, dim=0),
        "queries_issued": total_queries,
    }


# ============================================================================
# Hinton KD loss
# ============================================================================


def hinton_kd_loss(
    student_logits: torch.Tensor,   # [b, K, V]  float
    teacher_logits: torch.Tensor,   # [b, K, V]  float
    targets: torch.Tensor,          # [b, K]     long
    temperature: float,
    lambda_ce: float,
) -> tuple[torch.Tensor, dict]:
    """Classic Hinton distillation:

        L = KL(softmax(z_t / T) || softmax(z_s / T)) * T^2
          + λ * CrossEntropy(z_s, target)

    KL is computed in the conventional forward direction
    (student log-prob || teacher prob) via ``F.kl_div``.
    """
    T = temperature
    B, K, V = student_logits.shape

    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits / T, dim=-1)

    # batchmean reduction divides by (B*K), not by elements, matching Hinton
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T ** 2)

    ce = F.cross_entropy(
        student_logits.reshape(B * K, V),
        targets.reshape(B * K).to(student_logits.device),
    )

    loss = kl + lambda_ce * ce
    return loss, {"kl": kl.item(), "ce": ce.item(), "loss": loss.item()}


# ============================================================================
# Trainers
# ============================================================================


def _build_optimizer(
    student: nn.Module,
    trainable_param_names: list[str],
    block_lr: float,
    lm_head_lr: float,
) -> tuple[torch.optim.Optimizer, int]:
    """Per-parameter-group Adam: separate LR for lm_head vs blocks."""
    lm_head_params, block_params = [], []
    for name, p in student.named_parameters():
        if name not in trainable_param_names:
            p.requires_grad = False
            continue
        p.requires_grad = True
        if "lm_head" in name:
            lm_head_params.append(p)
        else:
            block_params.append(p)

    groups = []
    if block_params:
        groups.append({"params": block_params, "lr": block_lr})
    if lm_head_params:
        groups.append({"params": lm_head_params, "lr": lm_head_lr})

    optimizer = torch.optim.Adam(groups, weight_decay=0.0)
    total = sum(p.numel() for p in block_params) + sum(
        p.numel() for p in lm_head_params)
    return optimizer, total


def _sample_indices(n: int, batch_size: int, rng: torch.Generator) -> torch.Tensor:
    return torch.randint(0, n, (batch_size,), generator=rng)


def train_kd(
    student: nn.Module,
    trainable_param_names: list[str],
    query_ids: torch.Tensor,
    artifacts: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: str,
    oracle_block_idx: Optional[int],
    max_steps: int,
    rng_seed: int,
    query_budget_remaining: int,
    progress_prefix: str = "",
) -> dict:
    """Core KD training loop used by all three variants.

    ``oracle_block_idx`` is the block whose input should be replaced with
    the cached teacher hidden state (i.e., the first suffix block). If
    ``None``, no boundary injection happens (pure-logits variant).
    """
    optimizer, total_trainable = _build_optimizer(
        student, trainable_param_names, args.learning_rate, args.lm_head_lr)
    logger.info("%sTrainable parameters: %d (%.2fM)",
                progress_prefix, total_trainable, total_trainable / 1e6)

    K = args.logit_suffix_positions
    clean_logits = artifacts["clean_logits"]       # [N, K, V]
    targets = artifacts["next_token_targets"]      # [N, K]
    boundary = artifacts.get("boundary_states")

    rng = torch.Generator().manual_seed(rng_seed)

    injector: Optional[OracleBoundaryInjector] = None
    if oracle_block_idx is not None and boundary is not None:
        injector = OracleBoundaryInjector(
            student, boundary_hidden_states=boundary,
            block_idx=oracle_block_idx,
        )

    best_loss = float("inf")
    patience_counter = 0
    total_queries_used = 0
    loss_trace: list[float] = []
    start_time = time.time()

    # Disable gradient checkpointing (breaks our pre-hook).
    if hasattr(student, "gradient_checkpointing_disable"):
        student.gradient_checkpointing_disable()

    n_pool = query_ids.shape[0]

    def _step_body(step: int) -> dict:
        nonlocal best_loss, patience_counter, total_queries_used

        indices = _sample_indices(n_pool, args.batch_size, rng)

        batch_ids = query_ids[indices].to(device)
        batch_cl = clean_logits[indices].to(device=device,
                                            dtype=torch.float32)
        batch_tgt = targets[indices].to(device)

        if injector is not None:
            injector.set_batch(indices)

        student.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_s = student(batch_ids).logits  # [b, T, V]
        if K > 0:
            z_s = z_s[:, -K:, :]
        z_s = z_s.float()

        if injector is not None:
            injector.clear_batch()

        loss, comps = hinton_kd_loss(
            z_s, batch_cl, batch_tgt,
            temperature=args.kd_temperature,
            lambda_ce=args.kd_lambda_ce,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad],
            max_norm=args.grad_clip,
        )
        optimizer.step()

        total_queries_used += batch_ids.size(0)

        lv = comps["loss"]
        loss_trace.append(lv)
        if lv < best_loss - args.convergence_threshold:
            best_loss = lv
            patience_counter = 0
        else:
            patience_counter += 1
        return comps

    # Main loop
    def run_loop():
        return None

    if injector is not None:
        with injector:
            for step in range(max_steps):
                if query_budget_remaining is not None and \
                        total_queries_used >= query_budget_remaining:
                    logger.info("%sbudget exhausted at step %d",
                                progress_prefix, step)
                    break
                comps = _step_body(step)
                if step % 100 == 0:
                    logger.info(
                        "%sstep %d/%d | loss=%.4f kl=%.4f ce=%.4f "
                        "best=%.4f pat=%d/%d q=%d",
                        progress_prefix, step, max_steps,
                        comps["loss"], comps["kl"], comps["ce"],
                        best_loss, patience_counter, args.patience,
                        total_queries_used,
                    )
                if patience_counter >= args.patience:
                    logger.info("%sconverged (patience=%d) at step %d",
                                progress_prefix, args.patience, step)
                    break
    else:
        for step in range(max_steps):
            if query_budget_remaining is not None and \
                    total_queries_used >= query_budget_remaining:
                logger.info("%sbudget exhausted at step %d",
                            progress_prefix, step)
                break
            comps = _step_body(step)
            if step % 100 == 0:
                logger.info(
                    "%sstep %d/%d | loss=%.4f kl=%.4f ce=%.4f "
                    "best=%.4f pat=%d/%d q=%d",
                    progress_prefix, step, max_steps,
                    comps["loss"], comps["kl"], comps["ce"],
                    best_loss, patience_counter, args.patience,
                    total_queries_used,
                )
            if patience_counter >= args.patience:
                logger.info("%sconverged (patience=%d) at step %d",
                            progress_prefix, args.patience, step)
                break

    elapsed = time.time() - start_time

    return {
        "best_loss": float(best_loss),
        "final_loss": float(loss_trace[-1]) if loss_trace else float("inf"),
        "num_steps": len(loss_trace),
        "converged": patience_counter >= args.patience,
        "elapsed_seconds": elapsed,
        "queries_used": total_queries_used,
        "loss_trace_tail": [float(x) for x in loss_trace[-50:]],
    }


# ============================================================================
# Evaluation
# ============================================================================


@torch.no_grad()
def functional_eval(
    student: nn.Module,
    teacher: nn.Module,
    eval_ids: torch.Tensor,
    device: str,
    batch_size: int = 8,
) -> dict:
    """Functional recovery: KL(teacher || student), top-1, top-5 on held-out."""
    student.eval()
    teacher.eval()

    total_kl = 0.0
    total_ce = 0.0
    total_top1 = 0
    total_top5 = 0
    total_n = 0

    for start in range(0, len(eval_ids), batch_size):
        end = min(start + batch_size, len(eval_ids))
        batch = eval_ids[start:end].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_logits = student(batch).logits.float()
            t_logits = teacher(batch).logits.float()

        # KL(teacher || student) — direction that matches Hinton convention
        # of "student approximates teacher distribution".
        log_p_s = F.log_softmax(s_logits, dim=-1)
        p_t = F.softmax(t_logits, dim=-1)
        log_p_t = F.log_softmax(t_logits, dim=-1)
        kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1).mean().item()

        # Cross-entropy between argmax_t and student
        t_top1 = t_logits.argmax(dim=-1)
        s_top1 = s_logits.argmax(dim=-1)
        top1_match = (t_top1 == s_top1).float().mean().item()

        # Top-5 overlap
        t_top5 = t_logits.topk(5, dim=-1).indices  # [b, T, 5]
        s_top5 = s_logits.topk(5, dim=-1).indices
        top5_overlap = 0.0
        for b in range(batch.size(0)):
            for t in range(batch.size(1)):
                top5_overlap += len(
                    set(t_top5[b, t].tolist())
                    & set(s_top5[b, t].tolist())
                ) / 5.0
        top5_overlap /= (batch.size(0) * batch.size(1))

        # CE from student logits against teacher top-1 (label)
        ce = F.cross_entropy(
            s_logits.reshape(-1, s_logits.size(-1)),
            t_top1.reshape(-1),
        ).item()

        n = batch.size(0)
        total_kl += kl * n
        total_ce += ce * n
        total_top1 += top1_match * n
        total_top5 += top5_overlap * n
        total_n += n

    return {
        "kl_teacher_student": total_kl / max(total_n, 1),
        "ce_teacher_label": total_ce / max(total_n, 1),
        "top1_agreement": total_top1 / max(total_n, 1),
        "top5_overlap": total_top5 / max(total_n, 1),
        "n_eval": total_n,
    }


def block_prefix(model: nn.Module, block_idx: int) -> Optional[str]:
    target = str(block_idx)
    for name, _ in model.named_parameters():
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == target:
                return ".".join(parts[: i + 1])
    return None


def evaluate_recovery(
    student: nn.Module,
    ground_truth: dict[str, torch.Tensor],
    target_blocks: list[int],
    num_attention_heads: int,
    head_dim: int,
) -> dict:
    """Per-matrix aligned cosine for every suffix block + Procrustes
    aligned cosine for lm_head, matching run_spsi.py's evaluation."""
    params = {n: p.data.cpu().clone() for n, p in student.named_parameters()}

    block_results = []
    for bidx in target_blocks:
        prefix = block_prefix(student, bidx)
        if prefix is None:
            block_results.append({
                "block_idx": bidx,
                "name": f"block_{bidx}",
                "per_matrix_cosine": {},
                "mean_cosine": -1.0,
                "prefix": None,
            })
            continue
        try:
            unaligned, aligned = compute_aligned_cosine(
                params, ground_truth, prefix,
                num_attention_heads, head_dim,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Alignment failed for %s: %s", prefix, exc)
            unaligned, aligned = {}, {}
        if aligned:
            mean_cos = sum(aligned.values()) / len(aligned)
        elif unaligned:
            mean_cos = sum(unaligned.values()) / len(unaligned)
        else:
            mean_cos = -1.0
        block_results.append({
            "block_idx": bidx,
            "name": f"block_{bidx}",
            "prefix": prefix,
            "per_matrix_cosine": {
                "unaligned": {k: float(v) for k, v in unaligned.items()},
                "aligned": {k: float(v) for k, v in aligned.items()},
            },
            "mean_cosine": float(mean_cos),
        })

    lm_head_res = compute_lm_head_aligned_cosine(params, ground_truth)
    lm_head_entry = {
        "name": "lm_head",
        "per_matrix_cosine": {
            "procrustes": {k: (float(v) if isinstance(v, (int, float))
                               else v)
                           for k, v in lm_head_res.items()
                           if k != "top_singular_values"},
        },
        "mean_cosine": float(lm_head_res.get("aligned_cosine",
                                             lm_head_res.get("raw_cosine",
                                                             -1.0))),
    }

    return {
        "blocks": block_results,
        "lm_head": lm_head_entry,
        "raw_lm_head": {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in lm_head_res.items()
                        if k != "top_singular_values"},
    }


# ============================================================================
# Variant orchestration
# ============================================================================


def make_student(model_name: str, device: str, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    student = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    student.eval()
    if hasattr(student, "gradient_checkpointing_disable"):
        student.gradient_checkpointing_disable()
    return student


def run_kd_joint_oracle(
    student: nn.Module,
    trainable_names: list[str],
    query_ids: torch.Tensor,
    artifacts: dict,
    args: argparse.Namespace,
    device: str,
    oracle_block_idx: int,
    rng_seed: int,
    query_budget: int,
) -> dict:
    logger.info("=== kd_joint_oracle: %d suffix blocks + lm_head, joint ===",
                args.num_suffix_blocks)
    return train_kd(
        student=student,
        trainable_param_names=trainable_names,
        query_ids=query_ids,
        artifacts=artifacts,
        args=args,
        device=device,
        oracle_block_idx=oracle_block_idx,
        max_steps=args.joint_max_steps,
        rng_seed=rng_seed,
        query_budget_remaining=query_budget,
        progress_prefix="[joint] ",
    )


def run_kd_progressive_oracle(
    student: nn.Module,
    query_ids: torch.Tensor,
    artifacts: dict,
    args: argparse.Namespace,
    device: str,
    oracle_block_idx: int,
    num_blocks: int,
    rng_seed: int,
    query_budget: int,
) -> dict:
    """S-PSI-style curriculum: block L-1, then L-2, ..., then lm_head.

    The only thing that differs from S-PSI is the loss (pure Hinton KD)
    and the absence of sensitivity + algebraic init.
    """
    logger.info(
        "=== kd_progressive_oracle: last block first, then earlier ===")

    target_blocks = list(range(num_blocks - args.num_suffix_blocks, num_blocks))

    # Phase 0: lm_head recovery first (same as S-PSI ordering in run_spsi.py).
    lm_head_names = get_lm_head_param_names(student)
    logger.info(
        "Phase 0: lm_head only (%d params)", len(lm_head_names))
    budget_remaining = query_budget
    phase_results: list[dict] = []

    r0 = train_kd(
        student=student,
        trainable_param_names=lm_head_names,
        query_ids=query_ids,
        artifacts=artifacts,
        args=args,
        device=device,
        oracle_block_idx=None,  # lm_head needs no boundary injection
        max_steps=args.lm_head_steps,
        rng_seed=rng_seed,
        query_budget_remaining=budget_remaining,
        progress_prefix="[lm_head] ",
    )
    phase_results.append({"phase": "lm_head", **r0})
    budget_remaining -= r0["queries_used"]

    # Now sequentially train each suffix block from LAST to FIRST.
    for k, bidx in enumerate(reversed(target_blocks)):
        if budget_remaining <= 0:
            break
        block_names = get_block_param_names(student, bidx)
        # During per-block training, we fix lm_head (already trained) and
        # all other blocks. The oracle boundary is ALWAYS injected at the
        # very first suffix block — mirroring S-PSI's behavior.
        logger.info(
            "Phase %d: block %d only (%d params)",
            k + 1, bidx, len(block_names))
        r = train_kd(
            student=student,
            trainable_param_names=block_names,
            query_ids=query_ids,
            artifacts=artifacts,
            args=args,
            device=device,
            oracle_block_idx=oracle_block_idx,
            max_steps=args.block_steps,
            rng_seed=rng_seed + 7 * (k + 1),
            query_budget_remaining=budget_remaining,
            progress_prefix=f"[block_{bidx}] ",
        )
        phase_results.append({"phase": f"block_{bidx}", **r})
        budget_remaining -= r["queries_used"]

    return {
        "phases": phase_results,
        "queries_used": query_budget - max(budget_remaining, 0),
        "budget_remaining": budget_remaining,
    }


def run_kd_pure_logits(
    student: nn.Module,
    trainable_names: list[str],
    query_ids: torch.Tensor,
    artifacts: dict,
    args: argparse.Namespace,
    device: str,
    rng_seed: int,
    query_budget: int,
) -> dict:
    logger.info("=== kd_pure_logits: no oracle, random prefix ===")
    return train_kd(
        student=student,
        trainable_param_names=trainable_names,
        query_ids=query_ids,
        artifacts=artifacts,
        args=args,
        device=device,
        oracle_block_idx=None,
        max_steps=args.joint_max_steps,
        rng_seed=rng_seed,
        query_budget_remaining=query_budget,
        progress_prefix="[pure] ",
    )


# ============================================================================
# Per-seed driver
# ============================================================================


def run_variant_for_seed(
    variant: str,
    seed: int,
    args: argparse.Namespace,
    teacher: nn.Module,
    tokenizer,
    ground_truth: dict[str, torch.Tensor],
    num_blocks: int,
    num_attention_heads: int,
    head_dim: int,
    device: str,
) -> dict:
    """Build a fresh student, run the chosen variant, evaluate, and write
    summary JSON."""

    output_dir = Path(args.output_dir) / f"{variant}_{seed}"
    summary_path = output_dir / "kd_summary.json"
    if summary_path.exists() and not args.force_rerun:
        logger.info("[%s/%d] already complete, skipping (%s).",
                    variant, seed, summary_path)
        with open(summary_path) as f:
            return json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n========== %s | seed=%d ==========", variant, seed)

    # --- Build query pool (data seed is deterministic but independent
    # of the model-init seed so we compare students across seeds on the
    # SAME data).
    pool = build_query_pool(
        tokenizer, args.num_queries, args.max_seq_len,
        seed=42, allow_synthetic=args.allow_synthetic,
    )
    n_total = pool.shape[0]
    n_heldout = int(n_total * args.heldout_fraction)
    n_train = n_total - n_heldout
    perm = torch.randperm(
        n_total, generator=torch.Generator().manual_seed(42 + 1))
    train_ids = pool[perm[:n_train]]
    heldout_ids = pool[perm[n_train:]]

    # --- Precompute teacher artifacts ONCE per variant
    #
    # The first suffix block index is (num_blocks - args.num_suffix_blocks).
    # Its INPUT hidden state is hidden_states[first_suffix]. We pass
    # boundary_block_idx = first_suffix - 1 so that the precompute fn
    # grabs hidden_states[boundary_block_idx + 1].
    first_suffix = num_blocks - args.num_suffix_blocks
    boundary_block_idx = first_suffix - 1

    logger.info(
        "Precomputing teacher artifacts (N=%d, K=%d, boundary_before_block=%d)",
        n_train, args.logit_suffix_positions, first_suffix)
    artifacts = precompute_teacher_artifacts(
        teacher=teacher,
        query_ids=train_ids,
        boundary_block_idx=boundary_block_idx,
        logit_suffix_positions=args.logit_suffix_positions,
        device=device,
        batch_size=16,
    )
    logger.info(
        "Teacher artifacts: logits=%s, targets=%s, boundary=%s",
        tuple(artifacts["clean_logits"].shape),
        tuple(artifacts["next_token_targets"].shape),
        tuple(artifacts["boundary_states"].shape),
    )

    precompute_queries_used = artifacts["queries_issued"]
    query_budget_remaining = args.query_budget - precompute_queries_used
    logger.info(
        "Query budget: %d total, %d used for precompute, %d remaining",
        args.query_budget, precompute_queries_used, query_budget_remaining)

    # --- Build student + randomize suffix
    student = make_student(args.model_name, device=device, seed=seed)
    reset_names = randomize_suffix(
        student, num_blocks=num_blocks,
        num_suffix_blocks=args.num_suffix_blocks,
        include_lm_head=True,
    )
    logger.info("Randomized %d suffix parameter tensors.", len(reset_names))

    trainable_names = reset_names  # suffix-only training, rest frozen
    target_blocks = list(range(num_blocks - args.num_suffix_blocks, num_blocks))

    # --- Run chosen variant
    start_time = time.time()
    if variant == "kd_joint_oracle":
        train_stats = run_kd_joint_oracle(
            student=student,
            trainable_names=trainable_names,
            query_ids=train_ids,
            artifacts=artifacts,
            args=args,
            device=device,
            oracle_block_idx=first_suffix,
            rng_seed=seed,
            query_budget=query_budget_remaining,
        )
    elif variant == "kd_progressive_oracle":
        train_stats = run_kd_progressive_oracle(
            student=student,
            query_ids=train_ids,
            artifacts=artifacts,
            args=args,
            device=device,
            oracle_block_idx=first_suffix,
            num_blocks=num_blocks,
            rng_seed=seed,
            query_budget=query_budget_remaining,
        )
    elif variant == "kd_pure_logits":
        train_stats = run_kd_pure_logits(
            student=student,
            trainable_names=trainable_names,
            query_ids=train_ids,
            artifacts=artifacts,
            args=args,
            device=device,
            rng_seed=seed,
            query_budget=query_budget_remaining,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    total_elapsed = time.time() - start_time

    # --- Evaluate: parameter recovery
    logger.info("Evaluating parameter recovery ...")
    # Move student to CPU for eval to free GPU memory
    student.cpu()
    torch.cuda.empty_cache()
    recovery_eval = evaluate_recovery(
        student, ground_truth, target_blocks,
        num_attention_heads, head_dim,
    )
    # Move back to GPU for functional eval
    student.to(device)
    student.eval()

    # --- Functional eval on held-out
    logger.info("Evaluating functional recovery on %d held-out ...",
                min(args.functional_eval_queries, len(heldout_ids)))
    eval_ids = heldout_ids[: args.functional_eval_queries]
    func_eval = functional_eval(student, teacher, eval_ids,
                                device=device, batch_size=8)

    logger.info(
        "Functional: KL=%.4f, top1=%.4f, top5_overlap=%.4f",
        func_eval["kl_teacher_student"],
        func_eval["top1_agreement"],
        func_eval["top5_overlap"],
    )

    # --- Build summary (matching S-PSI schema so comparison builder
    # picks it up automatically)
    recovered_blocks_list = recovery_eval["blocks"] + [recovery_eval["lm_head"]]
    spsi_style_blocks = []
    total_queries = precompute_queries_used
    if isinstance(train_stats.get("queries_used"), int):
        total_queries += train_stats["queries_used"]
    elif "phases" in train_stats:
        total_queries += sum(ph.get("queries_used", 0)
                             for ph in train_stats["phases"])

    for entry in recovered_blocks_list:
        name = entry.get("name", "unknown")
        per_matrix = entry.get("per_matrix_cosine", {})
        if isinstance(per_matrix, dict) and "aligned" in per_matrix:
            flat_pm = per_matrix["aligned"]
        elif isinstance(per_matrix, dict) and "procrustes" in per_matrix:
            # Emit raw/aligned cosines under standardised name for tables
            proc = per_matrix["procrustes"]
            if "aligned_cosine" in proc and "raw_cosine" in proc:
                flat_pm = {
                    "lm_head.weight": proc["aligned_cosine"],
                    "lm_head.raw": proc["raw_cosine"],
                }
            else:
                flat_pm = proc
        else:
            flat_pm = per_matrix

        spsi_style_blocks.append({
            "name": name,
            "per_matrix_cosine": flat_pm,
            "mean_cosine": entry.get("mean_cosine", -1.0),
            "final_loss": train_stats.get("final_loss",
                                          train_stats.get("best_loss",
                                                          -1.0)),
            "num_steps": train_stats.get("num_steps", 0),
            "num_queries": total_queries,
            "converged": train_stats.get("converged", False),
        })

    kd_summary = {
        "variant": variant,
        "seed": seed,
        "model": args.model_name,
        "num_suffix_blocks": args.num_suffix_blocks,
        "num_queries": args.num_queries,
        "max_seq_len": args.max_seq_len,
        "query_budget": args.query_budget,
        "logit_suffix_positions": args.logit_suffix_positions,
        "kd_temperature": args.kd_temperature,
        "kd_lambda_ce": args.kd_lambda_ce,
        "optimizer": "Adam",
        "block_lr": args.learning_rate,
        "lm_head_lr": args.lm_head_lr,
        "grad_clip": args.grad_clip,
        "precision": "bf16 forward / fp32 adam",

        # Training
        "training_stats": train_stats,
        "total_elapsed_seconds": total_elapsed,
        "total_queries_used": total_queries,

        # Evaluation (two views)
        "recovery": recovery_eval,
        "functional": func_eval,
    }

    with open(summary_path, "w") as f:
        json.dump(kd_summary, f, indent=2, default=str)

    # S-PSI-compatible regime summary (so downstream aggregators work)
    regime_payload = {
        "per_init": [{
            "init_seed": seed,
            "blocks": spsi_style_blocks,
        }],
        "cross_init_stats": {
            b["name"]: {
                "mean": b["mean_cosine"],
                "std": 0.0,
                "min": b["mean_cosine"],
                "max": b["mean_cosine"],
                "values": [b["mean_cosine"]],
            }
            for b in spsi_style_blocks
        },
    }
    with open(output_dir / "regime_summary.json", "w") as f:
        json.dump(regime_payload, f, indent=2)

    logger.info("Wrote %s and regime_summary.json", summary_path)

    del student
    del artifacts
    torch.cuda.empty_cache()

    return kd_summary


# ============================================================================
# Cross-seed aggregation + side-by-side print
# ============================================================================


def aggregate_summaries(
    per_variant: dict[str, list[dict]],
    output_dir: Path,
) -> dict:
    """Build the cross-seed aggregate and emit a side-by-side table."""
    def _mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        n = len(values)
        m = sum(values) / n
        v = sum((x - m) ** 2 for x in values) / max(n - 1, 1)
        return m, v ** 0.5

    agg = {}
    for variant, runs in per_variant.items():
        # Collect per-block mean cosines across seeds
        block_rows: dict[str, list[float]] = {}
        func_rows: dict[str, list[float]] = {}
        for r in runs:
            for entry in r["recovery"]["blocks"] + [r["recovery"]["lm_head"]]:
                name = entry["name"]
                block_rows.setdefault(name, []).append(entry["mean_cosine"])
            for k, v in r["functional"].items():
                if isinstance(v, (int, float)):
                    func_rows.setdefault(k, []).append(float(v))

        agg[variant] = {
            "num_seeds": len(runs),
            "parameter_recovery": {
                name: {
                    "mean": _mean_std(vals)[0],
                    "std": _mean_std(vals)[1],
                    "values": vals,
                }
                for name, vals in block_rows.items()
            },
            "functional": {
                name: {
                    "mean": _mean_std(vals)[0],
                    "std": _mean_std(vals)[1],
                    "values": vals,
                }
                for name, vals in func_rows.items()
            },
        }

    out_path = output_dir / "matched_kd_aggregate.json"
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    logger.info("Aggregate written to %s", out_path)
    return agg


def print_comparison_table(
    agg: dict,
    spsi_reference: Optional[dict] = None,
) -> None:
    """Side-by-side table comparing KD variants (and optional S-PSI ref).

    Prints to logger.info for capture by subprocess logs.
    """
    variants = list(agg.keys())
    if spsi_reference:
        header_cols = ["S-PSI (ref)"] + variants
    else:
        header_cols = variants

    def _fmt(stat: dict) -> str:
        if stat is None:
            return "     n/a"
        m = stat.get("mean", float("nan"))
        s = stat.get("std", 0.0)
        return f"{m:.3f}+-{s:.3f}"

    # Collect block names (superset)
    block_names: list[str] = []
    for variant in variants:
        for name in agg[variant]["parameter_recovery"].keys():
            if name not in block_names:
                block_names.append(name)

    col_widths = [22] + [14] * len(header_cols)

    def _row(cols: list[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cols, col_widths))

    logger.info("=" * (sum(col_widths) + 3 * (len(col_widths) - 1) + 1))
    logger.info(_row(["Metric", *header_cols]))
    logger.info("-" * (sum(col_widths) + 3 * (len(col_widths) - 1) + 1))

    for bname in block_names:
        row = [bname]
        if spsi_reference:
            row.append(_fmt(spsi_reference.get(bname)))
        for variant in variants:
            row.append(_fmt(
                agg[variant]["parameter_recovery"].get(bname)))
        logger.info(_row(row))

    logger.info("-" * (sum(col_widths) + 3 * (len(col_widths) - 1) + 1))
    func_keys = ["kl_teacher_student", "top1_agreement", "top5_overlap"]
    for fk in func_keys:
        row = [fk]
        if spsi_reference:
            row.append("     n/a")
        for variant in variants:
            row.append(_fmt(agg[variant]["functional"].get(fk)))
        logger.info(_row(row))
    logger.info("=" * (sum(col_widths) + 3 * (len(col_widths) - 1) + 1))


def maybe_load_spsi_reference() -> Optional[dict]:
    """Load S-PSI cross-init stats from the canonical location if present.

    Returns a dict {block_name: {mean, std}}.
    """
    candidates = [
        Path("results/v2_alg_clean_s42") / "regime_oracle" /
        "regime_summary.json",
    ]
    for cand in candidates:
        if cand.exists():
            try:
                with open(cand) as f:
                    data = json.load(f)
                stats = data.get("cross_init_stats", {})
                if stats:
                    return stats
            except Exception:  # noqa: BLE001
                continue
    return None


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()
    setup_logging()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "resolved_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("=== Matched-Budget KD Baseline ===")
    logger.info("Model: %s", args.model_name)
    logger.info("Variants: %s", args.variants)
    logger.info("Seeds: %s", args.seeds)

    logger.info("Loading teacher ...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ground_truth = {n: p.data.cpu().clone()
                    for n, p in teacher.named_parameters()}
    if getattr(teacher.config, "tie_word_embeddings", False):
        lm_head = getattr(teacher, "lm_head", None)
        if lm_head is not None and "lm_head.weight" not in ground_truth:
            ground_truth["lm_head.weight"] = lm_head.weight.data.cpu().clone()

    num_blocks = get_num_blocks(teacher)
    num_heads = teacher.config.num_attention_heads
    head_dim = teacher.config.hidden_size // num_heads

    logger.info("Model has %d blocks, %d heads, head_dim=%d",
                num_blocks, num_heads, head_dim)
    logger.info("Suffix blocks: %s + lm_head",
                list(range(num_blocks - args.num_suffix_blocks, num_blocks)))

    per_variant: dict[str, list[dict]] = {v: [] for v in args.variants}
    for variant in args.variants:
        for seed in args.seeds:
            summary = run_variant_for_seed(
                variant=variant,
                seed=seed,
                args=args,
                teacher=teacher,
                tokenizer=tokenizer,
                ground_truth=ground_truth,
                num_blocks=num_blocks,
                num_attention_heads=num_heads,
                head_dim=head_dim,
                device=device,
            )
            per_variant[variant].append(summary)

    agg = aggregate_summaries(per_variant, output_dir)
    spsi_ref = maybe_load_spsi_reference()
    print_comparison_table(agg, spsi_ref)

    logger.info("\nResults written under %s", output_dir)


if __name__ == "__main__":
    main()
