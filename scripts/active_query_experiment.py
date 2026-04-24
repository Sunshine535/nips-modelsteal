#!/usr/bin/env python3
# SAFETY NOTICE: QUARANTINED (alpha-theory prune 2026-04-19)
# This script is NOT cited in the paper. It was part of killed branches:
#   A1 S-PSI, A2 Moments CP, A4 logit-bias, A5 memory probing, A6 active query,
#   A7 algebraic v2/v3/v4, B3 matched-KD.
# Retained in repo for reproducibility of quarantined history; do not use for
# new claims.
#!/usr/bin/env python3
"""Active Query Experiment — embedding-level adversarial queries.

============================================================================
SAFETY NOTICE (2026-04-19)
----------------------------------------------------------------------------
The `W_down` recovery numbers produced by this script are ORACLE-
CONDITIONAL, not black-box.  `wdown_recovery_cosine(...)` (see function
defined below) receives the teacher's true `W_gate` and `W_up`
matrices and solves a linear system for `W_down` via ridge-OLS.  The
reported cosine quantifies recovery of ONE MLP matrix given exact
knowledge of the OTHER TWO — this is an identifiability / observability
diagnostic, not a black-box parameter-recovery attack.

No paper-body result in main.tex currently cites this script's
`W_down` cosine as an attack number.  If a future revision wants to
cite these numbers, they must be framed explicitly as oracle-conditional
diagnostics in both tables and discussion.
============================================================================

Responds to the paper's §8 Limitations admission:

    "Passive queries only. Adversarially optimized queries could
     potentially expand the observable subspace. Developing and
     evaluating active query strategies is an important direction
     for future work."

Reviewer question this addresses: "If the attacker is allowed to
optimize its queries, can parameter recovery exceed the 0.618 cosine
ceiling that your paper reports?"

Approach (light-weight but principled)
-----------------------------------------
We do NOT need to train the teacher; we operate at the INPUT
EMBEDDING layer and use torch.optim to find embedding sequences
E in R^{T x d} that maximize some observability metric of the
teacher's middle-layer hidden state h_mid = RMSNorm(h_mid_raw).

Observability metrics (we test all of them):

    (a) min eigenvalue of X^T X (X = RMSNorm(h_mid)) — raw rank floor
    (b) effective rank of X (tr(X^T X)^2 / ||X^T X||_F^2)
    (c) nuclear norm of X
    (d) (negative) condition number of X^T X — maximize -kappa

Variants run in this script
-----------------------------------------
baseline_wikitext            passive WikiText (v4 reference)
baseline_random_tokens       uniform-random token IDs (v4 random)
baseline_random_embeddings   gaussian noise directly in embedding space
active_maxrank_projected     grad-optimized E then projected to nearest
                             embedding-table row each step (discrete
                             attacker)
active_maxrank_continuous    grad-optimized E WITHOUT projection
                             (continuous attacker — upper bound)

For each variant we report:

    * effective rank of h_mid across the N queries
    * top-k energy fractions at k = 128, 256, 512
    * W_down recovery cosine via oracle (gate, up) OLS against h_mid
    * top-1 / top-5 agreement on reconstructed logits

The core question: does the W_down cosine exceed the 0.618 ceiling
when queries are optimized? If yes, the negative result in the paper
is weaker than claimed. If no, the residual-stream low-rank is a
fundamental limit regardless of query design.

Command::

    CUDA_VISIBLE_DEVICES=2 python scripts/active_query_experiment.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --num_queries 1024 --max_seq_len 128 \
        --opt_steps 300 --opt_lr 0.01 \
        --output_dir results/v5_active_query \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.permutation_alignment import align_ffn_neurons  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================================
# Setup
# ============================================================================


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Active query experiment — embedding-level optimization")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--num_queries", type=int, default=1024)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--block_idx", type=int, default=23,
                   help="Middle-layer block whose pre-MLP state we analyse")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Queries per optimization batch")
    p.add_argument("--forward_batch_size", type=int, default=16,
                   help="Batch size for passive / eval forwards")

    # Optimizer
    p.add_argument("--opt_steps", type=int, default=300)
    p.add_argument("--opt_lr", type=float, default=0.01)
    p.add_argument("--objective", type=str, default="effective_rank",
                   choices=["effective_rank", "min_eigenvalue",
                            "nuclear_norm", "neg_condition"])
    p.add_argument("--project_every", type=int, default=1,
                   help="For projected variant: project E back to nearest "
                        "embedding-table row every N steps")
    p.add_argument("--projection_softness", type=float, default=0.0,
                   help="If > 0, mix projected+continuous: E <- (1-a)*E + a*P(E)")

    p.add_argument("--wdown_ols_ridge", type=float, default=1e-4)
    p.add_argument("--rank_topks", type=str, default="128,256,512,768")
    p.add_argument("--passive_seed_offset", type=int, default=137)

    # Control
    p.add_argument("--variants", type=str, nargs="+",
                   default=["baseline_wikitext",
                            "baseline_random_tokens",
                            "baseline_random_embeddings",
                            "active_maxrank_projected",
                            "active_maxrank_continuous"],
                   choices=["baseline_wikitext",
                            "baseline_random_tokens",
                            "baseline_random_embeddings",
                            "active_maxrank_projected",
                            "active_maxrank_continuous"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str,
                   default="results/v5_active_query")
    p.add_argument("--allow_synthetic", action="store_true")
    p.add_argument("--force_rerun", action="store_true")
    return p.parse_args()


# ============================================================================
# Query builders (token-space)
# ============================================================================


def build_wikitext_tokens(
    tokenizer, pool_size: int, max_seq_len: int, seed: int,
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
        logger.warning("Dataset load failed: %s. Filling with random.", exc)

    remaining = pool_size - len(input_ids_list)
    if remaining > 0:
        if not allow_synthetic:
            raise RuntimeError(
                f"Only {len(input_ids_list)}/{pool_size} from dataset.")
        rng = torch.Generator().manual_seed(seed + 9991)
        rand = torch.randint(3, tokenizer.vocab_size,
                             (remaining, max_seq_len), generator=rng)
        for i in range(remaining):
            input_ids_list.append(rand[i])

    return torch.stack(input_ids_list[:pool_size])


def build_random_tokens(
    vocab_size: int, pool_size: int, max_seq_len: int, seed: int,
) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(3, vocab_size, (pool_size, max_seq_len),
                         generator=rng)


# ============================================================================
# Teacher introspection helpers
# ============================================================================


def get_embedding_table(model: nn.Module) -> torch.Tensor:
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight.detach()
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte.weight.detach()
    raise RuntimeError("Could not locate embedding table on model.")


def forward_from_embeddings(
    model: nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_hidden_states: bool = True,
) -> dict:
    """Forward the HF CausalLM from pre-embedded inputs."""
    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    return out


def collect_h_mid_from_tokens(
    model: nn.Module,
    input_ids: torch.Tensor,
    block_idx: int,
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Run teacher on token inputs; collect h_mid at block_idx.

    h_mid := h_in + self_attn_output (the pre-MLP residual stream).
    """
    model.eval()
    N = input_ids.shape[0]
    h_mids: list[torch.Tensor] = []
    h_ins: list[torch.Tensor] = []
    h_outs: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []

    target_block = model.model.layers[block_idx]
    attn_buf: list[torch.Tensor] = []

    def attn_hook(module, inputs, output):
        a = output[0] if isinstance(output, tuple) else output
        attn_buf.append(a.detach().cpu().float())

    handle = target_block.self_attn.register_forward_hook(attn_hook)
    try:
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = input_ids[start:end].to(device)
                attn_buf.clear()
                out = model(batch, output_hidden_states=True,
                            return_dict=True)
                hs = out.hidden_states
                h_in = hs[block_idx].detach().cpu().float()
                h_out = hs[block_idx + 1].detach().cpu().float()
                attn_out = attn_buf[0]
                h_mid = h_in + attn_out
                h_ins.append(h_in)
                h_mids.append(h_mid)
                h_outs.append(h_out)
                logits_list.append(out.logits.detach().cpu().float())
    finally:
        handle.remove()

    return {
        "h_in": torch.cat(h_ins, 0),
        "h_mid": torch.cat(h_mids, 0),
        "h_out": torch.cat(h_outs, 0),
        "logits": torch.cat(logits_list, 0),
    }


# ============================================================================
# RMSNorm applied with the teacher's own gamma
# ============================================================================


def post_attn_rmsnorm(
    model: nn.Module, block_idx: int, h_mid: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """Apply the teacher's post-attention RMSNorm (the one that precedes
    the MLP) to h_mid. Critical because the MLP input is this, not h_mid."""
    block = model.model.layers[block_idx]
    if hasattr(block, "post_attention_layernorm"):
        g = block.post_attention_layernorm.weight.detach()
        eps_real = getattr(block.post_attention_layernorm, "variance_epsilon",
                           eps)
    else:
        g = torch.ones(h_mid.shape[-1], device=h_mid.device, dtype=h_mid.dtype)
        eps_real = eps
    # h_mid: [..., d]
    x = h_mid.float()
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps_real)
    return (x / rms) * g.float().to(x.device)


# ============================================================================
# Observability metrics
# ============================================================================


def compute_XtX(X: torch.Tensor) -> torch.Tensor:
    """X: [M, d] -> [d, d]. Computed in float64 on GPU (fallback to CPU)."""
    if X.device.type == "cuda":
        return (X.double().T @ X.double())
    return (X.double().T @ X.double())


def effective_rank(XtX: torch.Tensor) -> float:
    tr = XtX.diagonal().sum().item()
    frob_sq = (XtX ** 2).sum().item()
    return (tr * tr) / max(frob_sq, 1e-30)


def top_eigvals(XtX: torch.Tensor) -> torch.Tensor:
    ev = torch.linalg.eigvalsh(XtX).clamp(min=0.0)
    return torch.flip(ev, dims=[0])  # descending


def energy_in_topk(ev_desc: torch.Tensor, k: int) -> float:
    k = min(k, ev_desc.numel())
    total = ev_desc.sum().item()
    if total <= 0:
        return 0.0
    return ev_desc[:k].sum().item() / total


def observability_loss(
    X: torch.Tensor, objective: str, ridge: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Differentiable objective. Caller MINIMIZES.

    * effective_rank  -> -eff_rank
    * min_eigenvalue  -> -min eigenvalue (smoothed via logdet)
    * nuclear_norm    -> -nuclear_norm
    * neg_condition   -> +cond (we want cond small, so minimize + cond)

    All are computed in float32 for stability within autograd.
    """
    # X: [M, d]
    M, d = X.shape
    XtX = (X.T @ X) / max(M, 1) + ridge * torch.eye(
        d, device=X.device, dtype=X.dtype)

    stats: dict = {}
    if objective == "effective_rank":
        tr = XtX.diagonal().sum()
        frob = (XtX ** 2).sum()
        er = (tr * tr) / (frob + 1e-12)
        loss = -er
        stats["eff_rank"] = er.detach().item()
    elif objective == "min_eigenvalue":
        # Smooth surrogate via logdet: log det(X^T X + eps I) is a
        # concave lower bound for sum log eigenvalues; maximizing it
        # pushes all eigenvalues up (including the smallest one).
        try:
            ld = torch.linalg.slogdet(XtX)[1]
        except Exception:  # noqa: BLE001
            ld = torch.log(XtX.diagonal().clamp(min=1e-12)).sum()
        loss = -ld
        stats["logdet"] = ld.detach().item()
    elif objective == "nuclear_norm":
        # ||X||_* = tr(sqrt(X^T X)) = sum sqrt(eigvals)
        ev = torch.linalg.eigvalsh(XtX).clamp(min=1e-12)
        nuc = ev.sqrt().sum()
        loss = -nuc
        stats["nuclear_norm"] = nuc.detach().item()
    elif objective == "neg_condition":
        ev = torch.linalg.eigvalsh(XtX).clamp(min=1e-12)
        cond = ev[-1] / ev[0]
        loss = cond  # minimize
        stats["cond"] = cond.detach().item()
    else:
        raise ValueError(f"Unknown objective: {objective}")

    return loss, stats


# ============================================================================
# Token-space projection (for discrete-attacker variant)
# ============================================================================


def project_to_nearest_embedding(
    inputs_embeds: torch.Tensor, embed_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each position, find the vocabulary row closest in L2.

    Returns (projected_embeds, nearest_token_ids).
    """
    B, T, d = inputs_embeds.shape
    flat = inputs_embeds.reshape(B * T, d)
    # Chunked to avoid OOM with vocab~150k
    V = embed_table.shape[0]
    nearest_ids = torch.empty(B * T, dtype=torch.long, device=flat.device)
    chunk = 512
    for s in range(0, B * T, chunk):
        e = min(s + chunk, B * T)
        dots = flat[s:e] @ embed_table.T                           # [c, V]
        vn = (embed_table ** 2).sum(dim=-1)                        # [V]
        fn = (flat[s:e] ** 2).sum(dim=-1, keepdim=True)            # [c, 1]
        dist_sq = fn - 2 * dots + vn.unsqueeze(0)                  # [c, V]
        nearest_ids[s:e] = dist_sq.argmin(dim=-1)
    projected = embed_table[nearest_ids].reshape(B, T, d)
    return projected, nearest_ids.reshape(B, T)


# ============================================================================
# Active optimization
# ============================================================================


def run_active_optimization(
    model: nn.Module,
    initial_ids: torch.Tensor,         # [N, T]
    block_idx: int,
    args: argparse.Namespace,
    device: str,
    variant: str,
) -> dict:
    """Optimize input embeddings to maximize observability on h_mid."""
    assert variant in ("active_maxrank_projected",
                       "active_maxrank_continuous")
    model.eval()

    embed_table = get_embedding_table(model).to(device=device,
                                                dtype=torch.float32)
    N, T = initial_ids.shape
    d = embed_table.shape[1]

    # Start from the token-embedding of ``initial_ids`` + small gaussian noise.
    with torch.no_grad():
        base = embed_table[initial_ids.to(device)].to(torch.float32)
        noise = 0.01 * torch.randn_like(base)
        E = (base + noise).detach().clone()
    E.requires_grad_(True)

    optimizer = torch.optim.Adam([E], lr=args.opt_lr)

    traj = []
    start_time = time.time()
    for step in range(args.opt_steps):
        # Mini-batch over queries
        batch_losses = []
        batch_stats: list[dict] = []

        perm = torch.randperm(N, device=device)
        for bs in range(0, N, args.batch_size):
            be = min(bs + args.batch_size, N)
            idx = perm[bs:be]
            E_batch = E[idx]

            # Forward teacher (autocast bf16) with grad through E
            # Attention hook captures self_attn output. We need to be
            # careful: attn_buf grows across batches, so reset per call.
            target_block = model.model.layers[block_idx]
            attn_out: list[torch.Tensor] = []

            def attn_hook(module, inputs, output):
                a = output[0] if isinstance(output, tuple) else output
                attn_out.append(a)

            handle = target_block.self_attn.register_forward_hook(attn_hook)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = forward_from_embeddings(
                        model, E_batch.to(torch.bfloat16))
                hs = out.hidden_states
                h_in = hs[block_idx]
                h_mid = h_in + attn_out[0]
                X = post_attn_rmsnorm(model, block_idx, h_mid).float()
                # [b, T, d] -> [b*T, d]
                X_flat = X.reshape(-1, X.shape[-1])

                loss, stats = observability_loss(
                    X_flat, args.objective, ridge=1e-6)
            finally:
                handle.remove()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # clip per-sequence (keeps well-scaled across N)
            torch.nn.utils.clip_grad_norm_([E], max_norm=1.0)
            optimizer.step()

            batch_losses.append(loss.detach().item())
            batch_stats.append(stats)

        avg_loss = sum(batch_losses) / max(len(batch_losses), 1)

        # Optional projection to nearest token (discrete attacker)
        if variant == "active_maxrank_projected" and (
                step % args.project_every == 0) and step > 0:
            with torch.no_grad():
                projected, token_ids = project_to_nearest_embedding(
                    E.detach(), embed_table)
                a = args.projection_softness
                if a <= 0.0:
                    E.data.copy_(projected)
                else:
                    E.data.copy_((1 - a) * E.data + a * projected)

        if step % 20 == 0 or step == args.opt_steps - 1:
            logger.info("[%s] step %d/%d | loss=%.5f",
                        variant, step, args.opt_steps, avg_loss)
            traj.append({"step": step, "loss": avg_loss,
                         "objective_stats": batch_stats[-1]})

    elapsed = time.time() - start_time

    # Freeze E for downstream eval
    E_final = E.detach()
    if variant == "active_maxrank_projected":
        projected, token_ids = project_to_nearest_embedding(
            E_final, embed_table)
        E_final = projected.detach()
        result_ids = token_ids.cpu()
    else:
        result_ids = None  # no exact tokens — continuous attacker

    return {
        "final_embeds": E_final,
        "final_ids": result_ids,
        "traj": traj,
        "elapsed_seconds": elapsed,
    }


# ============================================================================
# W_down OLS reconstruction (oracle gate/up)
# ============================================================================


def teacher_mlp_weights(
    model: nn.Module, block_idx: int,
) -> dict[str, torch.Tensor]:
    block = model.model.layers[block_idx]
    mlp = block.mlp
    return {
        "W_gate": mlp.gate_proj.weight.detach().cpu().float(),   # [d_ff, d]
        "W_up":   mlp.up_proj.weight.detach().cpu().float(),
        "W_down": mlp.down_proj.weight.detach().cpu().float(),   # [d, d_ff]
    }


def wdown_recovery_cosine(
    X: torch.Tensor,                 # [M, d]  RMSNorm-normalised h_mid
    h_out_minus_res: torch.Tensor,   # [M, d]  y = h_out - (h_in + attn_out)
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    teacher_W_down: torch.Tensor,
    ridge: float,
) -> dict:
    """Oracle W_down solve.

    The MLP implements y = W_down (SiLU(W_gate x) * (W_up x)). Given
    oracle (W_gate, W_up) we solve the linear system

            y = W_down s       where s = SiLU(x @ W_gate^T) * (x @ W_up^T)

    via ridge-regularised least squares in float64, then report Hungarian-
    aligned cosine against the teacher.
    """
    M, d = X.shape
    d_ff = W_gate.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_ = X.to(device=device, dtype=torch.float64)
    Wg_ = W_gate.to(device=device, dtype=torch.float64)
    Wu_ = W_up.to(device=device, dtype=torch.float64)

    gate_pre = X_ @ Wg_.T                     # [M, d_ff]
    up_pre = X_ @ Wu_.T                       # [M, d_ff]
    gate_act = gate_pre * torch.sigmoid(gate_pre)  # SiLU
    S = gate_act * up_pre                     # [M, d_ff]

    y = h_out_minus_res.to(device=device, dtype=torch.float64)  # [M, d]

    # W_down.T solves S @ W_down.T = y  ==> W_down = (S^T S + λI)^-1 S^T y
    StS = S.T @ S                            # [d_ff, d_ff]
    Sty = S.T @ y                            # [d_ff, d]
    I = torch.eye(d_ff, device=device, dtype=torch.float64)
    lam = ridge * StS.diagonal().mean().item()
    W_down_T_hat = torch.linalg.solve(StS + lam * I, Sty)
    W_down_hat = W_down_T_hat.T              # [d, d_ff]

    # Raw cosine (flat)
    W_down_hat_cpu = W_down_hat.cpu().float()
    raw_cos = F.cosine_similarity(
        W_down_hat_cpu.flatten().unsqueeze(0),
        teacher_W_down.flatten().unsqueeze(0),
    ).item()

    # Hungarian alignment over FFN neurons (columns of W_down, rows of W_gate/W_up)
    params_rec = {
        "dummy.mlp.gate_proj.weight": W_gate,
        "dummy.mlp.up_proj.weight":   W_up,
        "dummy.mlp.down_proj.weight": W_down_hat_cpu,
    }
    params_tea = {
        "dummy.mlp.gate_proj.weight": W_gate,
        "dummy.mlp.up_proj.weight":   W_up,
        "dummy.mlp.down_proj.weight": teacher_W_down,
    }
    aligned = align_ffn_neurons(params_rec, params_tea, "dummy")
    aligned_down = aligned["dummy.mlp.down_proj.weight"]
    aligned_cos = F.cosine_similarity(
        aligned_down.flatten().unsqueeze(0),
        teacher_W_down.flatten().unsqueeze(0),
    ).item()

    # Residual diagnostic
    resid = (S @ W_down_T_hat - y).norm().item() / (y.norm().item() + 1e-12)

    return {
        "raw_cosine": raw_cos,
        "aligned_cosine": aligned_cos,
        "relative_residual": resid,
        "n_samples": M,
        "d_ff": d_ff,
    }


# ============================================================================
# Variant runners
# ============================================================================


def run_baseline_wikitext(
    args, tokenizer, device,
) -> dict:
    ids = build_wikitext_tokens(
        tokenizer, args.num_queries, args.max_seq_len, args.seed,
        allow_synthetic=args.allow_synthetic)
    return {"token_ids": ids, "source": "wikitext", "inputs_embeds": None}


def run_baseline_random_tokens(
    args, tokenizer, device,
) -> dict:
    ids = build_random_tokens(
        tokenizer.vocab_size, args.num_queries, args.max_seq_len,
        args.seed + args.passive_seed_offset)
    return {"token_ids": ids, "source": "random_tokens",
            "inputs_embeds": None}


def run_baseline_random_embeddings(
    args, teacher, device,
) -> dict:
    """Sample embedding sequences from N(0, sigma^2 I) with sigma matched
    to the per-row RMS of the teacher's embedding matrix.

    This is the strongest NON-adversarial attacker in continuous space:
    it tests whether smooth random noise already saturates observability.
    """
    embed_table = get_embedding_table(teacher).to(device).float()
    per_row_rms = embed_table.pow(2).mean(dim=-1).sqrt()
    sigma = per_row_rms.mean().item()

    rng = torch.Generator(device=device).manual_seed(args.seed + 211)
    E = sigma * torch.randn(
        args.num_queries, args.max_seq_len, embed_table.shape[1],
        generator=rng, device=device, dtype=torch.float32,
    )
    return {
        "token_ids": None,
        "source": "random_embeddings",
        "inputs_embeds": E,
        "scale_sigma": sigma,
    }


def run_active(
    variant: str, args, teacher, tokenizer, device,
) -> dict:
    # initial tokens: uniform random (wider exploration than wikitext)
    init_ids = build_random_tokens(
        tokenizer.vocab_size, args.num_queries, args.max_seq_len,
        args.seed + 421)
    opt = run_active_optimization(
        model=teacher, initial_ids=init_ids,
        block_idx=args.block_idx, args=args, device=device, variant=variant,
    )
    return {
        "token_ids": opt["final_ids"],
        "source": variant,
        "inputs_embeds": opt["final_embeds"],  # may be projected rows
        "opt_traj": opt["traj"],
        "elapsed_seconds": opt["elapsed_seconds"],
    }


# ============================================================================
# Evaluation
# ============================================================================


def compute_h_mid(
    teacher: nn.Module, prepared: dict, args, device,
) -> dict[str, torch.Tensor]:
    """Dispatch to correct forward path depending on input source."""
    if prepared.get("inputs_embeds") is not None:
        E = prepared["inputs_embeds"]
        # Collect in detached mode (we already have the final E)
        acc = {"h_in": [], "h_mid": [], "h_out": [], "logits": []}
        target_block = teacher.model.layers[args.block_idx]
        attn_buf: list[torch.Tensor] = []

        def attn_hook(module, inputs, output):
            a = output[0] if isinstance(output, tuple) else output
            attn_buf.append(a.detach().cpu().float())

        handle = target_block.self_attn.register_forward_hook(attn_hook)
        try:
            with torch.no_grad():
                for start in range(0, E.shape[0], args.forward_batch_size):
                    end = min(start + args.forward_batch_size, E.shape[0])
                    E_b = E[start:end].to(device=device, dtype=torch.bfloat16)
                    attn_buf.clear()
                    out = forward_from_embeddings(teacher, E_b)
                    hs = out.hidden_states
                    h_in = hs[args.block_idx].detach().cpu().float()
                    h_out = hs[args.block_idx + 1].detach().cpu().float()
                    attn_out = attn_buf[0]
                    h_mid = h_in + attn_out
                    acc["h_in"].append(h_in)
                    acc["h_mid"].append(h_mid)
                    acc["h_out"].append(h_out)
                    acc["logits"].append(out.logits.detach().cpu().float())
        finally:
            handle.remove()

        return {k: torch.cat(v, 0) for k, v in acc.items()}
    else:
        ids = prepared["token_ids"]
        return collect_h_mid_from_tokens(
            teacher, ids, args.block_idx, args.forward_batch_size, device)


def summarise_observability(
    teacher: nn.Module,
    h: dict[str, torch.Tensor],
    args,
    device,
    teacher_mlp: dict[str, torch.Tensor],
) -> dict:
    """Given h_in/h_mid/h_out from a forward pass, compute:
      * effective rank + top-k energies of X = RMSNorm(h_mid)
      * W_down recovery via oracle (gate, up) OLS
      * top-1 / top-5 agreement vs random reference (sanity)
    """
    block_idx = args.block_idx
    h_mid = h["h_mid"]         # [N, T, d]
    h_in = h["h_in"]
    h_out = h["h_out"]

    N, T, d = h_mid.shape

    # X = RMSNorm(h_mid) (teacher's own post-attn RMSNorm)
    block = teacher.model.layers[block_idx]
    g = block.post_attention_layernorm.weight.detach().cpu().float()
    eps = getattr(block.post_attention_layernorm, "variance_epsilon", 1e-6)
    x = h_mid.float()
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    X = (x / rms) * g.view(1, 1, -1)
    X_flat = X.reshape(N * T, d)

    # Rank stats via float64 accumulation
    XtX = X_flat.double().T @ X_flat.double() / max(N * T, 1)
    eff_rank = effective_rank(XtX)
    ev_desc = top_eigvals(XtX)
    sigma = ev_desc.sqrt()

    topks = [int(k) for k in args.rank_topks.split(",")]
    energy_frac = {f"top{k}": energy_in_topk(ev_desc, k) for k in topks}

    # ----- W_down recovery via oracle (gate, up) OLS -----
    # y = h_out - (h_in + attn_out) = h_out - h_mid
    y_flat = (h_out.float() - h_mid.float()).reshape(N * T, d)

    W_gate = teacher_mlp["W_gate"]
    W_up = teacher_mlp["W_up"]
    W_down_teacher = teacher_mlp["W_down"]

    wdown = wdown_recovery_cosine(
        X=X_flat.float(),
        h_out_minus_res=y_flat,
        W_gate=W_gate, W_up=W_up,
        teacher_W_down=W_down_teacher,
        ridge=args.wdown_ols_ridge,
    )

    return {
        "num_queries": N,
        "seq_len": T,
        "d_model": d,
        "num_samples_for_rank": N * T,

        "effective_rank": float(eff_rank),
        "top_singular_values": sigma[:20].cpu().tolist(),
        "energy_fraction": energy_frac,
        "min_eigenvalue": float(ev_desc[-1].item()),
        "max_eigenvalue": float(ev_desc[0].item()),
        "condition_number": float(
            ev_desc[0].item() / max(ev_desc[-1].item(), 1e-30)),

        "wdown_recovery": wdown,
    }


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()
    setup_logging()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=== Active Query Experiment ===")
    logger.info("Model: %s, block_idx=%d, N=%d, T=%d, objective=%s",
                args.model_name, args.block_idx, args.num_queries,
                args.max_seq_len, args.objective)
    logger.info("Variants: %s", args.variants)

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

    teacher_mlp = teacher_mlp_weights(teacher, args.block_idx)

    # ---- Main loop over variants ----
    per_variant_results = {}
    for variant in args.variants:
        var_dir = output_dir / variant
        var_dir.mkdir(parents=True, exist_ok=True)
        outfile = var_dir / "results.json"
        if outfile.exists() and not args.force_rerun:
            logger.info("[%s] exists, skipping", variant)
            with open(outfile) as f:
                per_variant_results[variant] = json.load(f)
            continue

        logger.info("\n========== %s ==========", variant)
        start_time = time.time()

        if variant == "baseline_wikitext":
            prepared = run_baseline_wikitext(args, tokenizer, device)
        elif variant == "baseline_random_tokens":
            prepared = run_baseline_random_tokens(args, tokenizer, device)
        elif variant == "baseline_random_embeddings":
            prepared = run_baseline_random_embeddings(args, teacher, device)
        elif variant in ("active_maxrank_projected",
                         "active_maxrank_continuous"):
            prepared = run_active(variant, args, teacher, tokenizer, device)
        else:
            raise ValueError(variant)

        # Collect hidden states
        h = compute_h_mid(teacher, prepared, args, device)

        # Summarise observability + W_down recovery
        metrics = summarise_observability(teacher, h, args, device, teacher_mlp)

        elapsed = time.time() - start_time
        payload = {
            "variant": variant,
            "source": prepared.get("source", variant),
            "block_idx": args.block_idx,
            "num_queries": args.num_queries,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "objective": args.objective,
            "opt_steps": args.opt_steps,
            "opt_lr": args.opt_lr,
            "elapsed_seconds": elapsed,
            "metrics": metrics,
        }

        # Persist optimization trajectory separately if present
        if "opt_traj" in prepared:
            payload["opt_traj_tail"] = prepared["opt_traj"][-20:]
            payload["opt_traj_head"] = prepared["opt_traj"][:5]
            payload["opt_elapsed_seconds"] = prepared.get(
                "elapsed_seconds", 0.0)

        if "scale_sigma" in prepared:
            payload["scale_sigma"] = prepared["scale_sigma"]

        with open(outfile, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        per_variant_results[variant] = payload

        logger.info(
            "[%s] eff_rank=%.2f | min_ev=%.2e | max_ev=%.2e | cond=%.2e | "
            "W_down aligned cos=%.4f (raw=%.4f, resid=%.4f)",
            variant,
            metrics["effective_rank"],
            metrics["min_eigenvalue"],
            metrics["max_eigenvalue"],
            metrics["condition_number"],
            metrics["wdown_recovery"]["aligned_cosine"],
            metrics["wdown_recovery"]["raw_cosine"],
            metrics["wdown_recovery"]["relative_residual"],
        )

        # Free memory
        if hasattr(prepared, "clear"):
            prepared.clear()
        del prepared, h, metrics
        torch.cuda.empty_cache()

    # ---- Cross-variant table ----
    logger.info("\n============ SUMMARY ============")
    row_fmt = ("%-32s | eff_rank=%7.2f | top128=%5.3f | top512=%5.3f | "
               "W_down_aligned=%6.4f (raw=%6.4f)")
    for variant, payload in per_variant_results.items():
        m = payload["metrics"]
        logger.info(
            row_fmt, variant,
            m["effective_rank"],
            m["energy_fraction"].get("top128", float("nan")),
            m["energy_fraction"].get("top512", float("nan")),
            m["wdown_recovery"]["aligned_cosine"],
            m["wdown_recovery"]["raw_cosine"],
        )

    # ---- Aggregate file ----
    agg = {
        "model": args.model_name,
        "block_idx": args.block_idx,
        "num_queries": args.num_queries,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "wdown_ceiling_reference": 0.618,
        "variants": per_variant_results,
    }

    # Extract head-to-head arrays
    variants = list(per_variant_results.keys())
    agg["table"] = {
        "effective_rank": {v: per_variant_results[v]["metrics"]
                           ["effective_rank"] for v in variants},
        "wdown_aligned_cosine": {
            v: per_variant_results[v]["metrics"]["wdown_recovery"]
            ["aligned_cosine"] for v in variants},
        "wdown_raw_cosine": {
            v: per_variant_results[v]["metrics"]["wdown_recovery"]
            ["raw_cosine"] for v in variants},
    }

    # Key question answer: does any active variant exceed the 0.618 ceiling?
    baseline_wd = max(
        (per_variant_results[v]["metrics"]["wdown_recovery"]["aligned_cosine"]
         for v in variants
         if v.startswith("baseline_")),
        default=float("nan"),
    )
    best_active_wd = max(
        (per_variant_results[v]["metrics"]["wdown_recovery"]["aligned_cosine"]
         for v in variants
         if v.startswith("active_")),
        default=float("nan"),
    )
    exceeded_ceiling = (
        best_active_wd == best_active_wd  # not NaN
        and best_active_wd > 0.618
    )
    improved_over_baseline = (
        best_active_wd == best_active_wd
        and baseline_wd == baseline_wd
        and best_active_wd > baseline_wd + 0.02
    )

    agg["headline"] = {
        "best_baseline_wdown_cos": baseline_wd,
        "best_active_wdown_cos": best_active_wd,
        "delta_over_baseline": best_active_wd - baseline_wd
        if (baseline_wd == baseline_wd and best_active_wd == best_active_wd)
        else float("nan"),
        "exceeds_0618_ceiling": bool(exceeded_ceiling),
        "meaningfully_improves_over_passive": bool(improved_over_baseline),
    }

    with open(output_dir / "active_query_aggregate.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)

    logger.info("\nAggregate: best_baseline=%.4f, best_active=%.4f, "
                "delta=%+.4f, exceeds_0.618=%s",
                baseline_wd, best_active_wd,
                agg["headline"]["delta_over_baseline"],
                agg["headline"]["exceeds_0618_ceiling"])
    logger.info("Results under %s", output_dir)


if __name__ == "__main__":
    main()
