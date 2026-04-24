#!/usr/bin/env python3
"""
Multi-model algebraic recovery sweep.

Runs the same algebraic recovery pipeline (Carlini logit-SVD extraction +
algebraic W_down OLS + joint (W_gate, W_up) optimization with fixed W_down)
on several transformer LMs, to validate that the key findings from
algebraic_recovery_v4_richinput.py generalize across architectures and scales.

Key findings we want to test the generality of:
    1.  Identifiability ceiling governed by effective rank of the residual
        stream at the MLP input (X = RMSNorm(h_mid)).
    2.  Residual stream at deep blocks is low-rank (rank << d_model) for
        natural text and gets inflated by random-token queries.
    3.  Algebraic oracle-W_down OLS recovers W_down exactly (up to numerical
        noise) in high precision.
    4.  Joint (W_gate, W_up) recovery saturates at a cosine level dictated
        by the effective rank of X, independent of W_down quality.

Per-model pipeline (≤ ~20 min on A100 for ≤ 9B models; 27B is slower):
    0.  Load teacher (bf16 by default) with trust_remote_code, auto-detect the
        architecture (hidden_size, intermediate_size, GQA, activation, RoPE,
        tie_word_embeddings, MoE status), find the MLP module of the last
        dense block.
    1.  Collect hidden states for N queries × T positions, with hooks for
        h_in (pre-block), h_mid (post-attn, pre-MLP residual), h_out
        (post-block).  Random tokens are used as inputs (v4 finding: they
        inflate rank).
    2.  Compute effective rank and energy concentration of X.
    3.  Run Carlini SVD extraction on *last-position* logits to recover W_lm
        up to orthogonal rotation, report subspace principal-angle cosine.
    4.  Solve algebraic W_down via float64 OLS using ORACLE gate/up
        activations.  Report cos, per-row cos, Frobenius error, residual cos.
    5.  Joint-optimize (W_gate, W_up) with fixed recovered W_down by Adam
        (default 5000 steps).  Report raw, self-aligned, permutation-aligned
        cosines plus functional reconstruction cos on a held-out slice.
    6.  Write per-model JSON and a global summary.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python scripts/multi_model_sweep.py \\
        --models Qwen/Qwen2.5-0.5B Qwen/Qwen3.5-0.8B meta-llama/Llama-3.2-1B \\
        --num_queries 2048 --max_seq_len 128 \\
        --output_dir results/v5_multi_model_sweep \\
        --seed 42 --joint_opt_steps 5000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.permutation_alignment import align_ffn_neurons  # noqa: F401
    HAS_ALIGN = True
except Exception:  # noqa: BLE001
    HAS_ALIGN = False

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen3.5-0.8B",
    "meta-llama/Llama-3.2-1B",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Space-separated list of HF model IDs.")
    p.add_argument("--models_json", type=str, default=None,
                   help="JSON file: list of {model_name, ...} dicts. Wins over --models.")
    p.add_argument("--num_queries", type=int, default=2048)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--output_dir", type=str, default="results/v5_multi_model_sweep")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for data collection and OLS accumulation.")
    p.add_argument("--model_dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"],
                   help="Weights dtype. Default: bfloat16.")
    p.add_argument("--device_map", type=str, default="auto",
                   choices=["auto", "single", "balanced_low_0"],
                   help="'single' pins to --device; 'auto' lets HF spread big models.")

    # Joint-opt hyperparameters (kept short for throughput)
    p.add_argument("--joint_opt_steps", type=int, default=5000)
    p.add_argument("--opt_batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_warmup_steps", type=int, default=300)
    p.add_argument("--init_scale", type=float, default=0.02)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=500)

    # Carlini step
    p.add_argument("--carlini_num_queries", type=int, default=2048,
                   help="Separate random-query budget for Carlini SVD extraction.")

    # Per-model safeguards
    p.add_argument("--per_model_timeout_s", type=float, default=3600.0,
                   help="Soft timeout per model. Only applied between phases.")
    p.add_argument("--max_data_points", type=int, default=262144,
                   help="Cap on M = N*T to control memory on big-d models.")
    p.add_argument("--skip_on_error", action="store_true", default=True,
                   help="Log and skip model on error (default).")
    p.add_argument("--no_skip_on_error", dest="skip_on_error",
                   action="store_false",
                   help="Re-raise errors instead of skipping.")
    p.add_argument("--skip_carlini", action="store_true",
                   help="Skip Carlini SVD step.")
    p.add_argument("--skip_joint_opt", action="store_true",
                   help="Skip Phase 2 joint optimization (rank + Carlini + "
                        "W_down OLS only).")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip models that already have results.json.")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--block_offset_from_end", type=int, default=1,
                   help="Target block = num_hidden_layers - block_offset_from_end.")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def safe_name(model_name: str) -> str:
    """Slug-safe directory name for a HF model id."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.strip()).strip("_")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flat_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def per_row_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    if a_f.dim() == 1:
        return flat_cosine(a_f, b_f)
    return F.cosine_similarity(a_f, b_f, dim=-1).mean().item()


def rms_norm_tensor(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Llama/Qwen style RMSNorm: x / sqrt(mean(x^2) + eps) * g."""
    x_f = x.float()
    rms = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f / rms * g.float()).to(x.dtype)


def dtype_from_str(s: str) -> torch.dtype:
    return {"float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16}[s]


def _jsonify(o):
    """Recursively strip tensors / numpy scalars / Paths for JSON dumping."""
    if isinstance(o, (bool, int, float, str, type(None))):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o
    if isinstance(o, torch.Tensor):
        if o.numel() == 1:
            return float(o.item())
        return o.tolist()
    if isinstance(o, (np.ndarray, np.generic)):
        try:
            return o.tolist()
        except Exception:  # noqa: BLE001
            return str(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item") and not isinstance(o, (list, tuple, dict)):
        try:
            return float(o.item())
        except Exception:  # noqa: BLE001
            return str(o)
    if isinstance(o, (list, tuple)):
        return [_jsonify(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _jsonify(v) for k, v in o.items()}
    return str(o)


def cos_with_self_alignment(
    W_rec: torch.Tensor, W_true: torch.Tensor,
) -> tuple[float, float]:
    """Hungarian-align rows of W_rec to W_true using themselves as the ref."""
    n = min(W_rec.shape[0], W_true.shape[0])
    a = F.normalize(W_rec[:n].float(), dim=-1)
    b = F.normalize(W_true[:n].float(), dim=-1)
    sim = a @ b.T
    cost = 1.0 - sim.cpu().numpy()
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except Exception as e:  # noqa: BLE001
        logger.warning("Hungarian alignment failed: %s", e)
        return flat_cosine(W_rec[:n], W_true[:n]), per_row_cosine(W_rec[:n], W_true[:n])
    perm = list(range(n))
    for r, c in zip(row_ind, col_ind):
        perm[r] = c
    W_rec_perm = torch.stack([W_rec[perm[i]] for i in range(n)])
    return flat_cosine(W_rec_perm, W_true[:n]), per_row_cosine(W_rec_perm, W_true[:n])


def cos_with_joint_alignment(
    Wg_rec: torch.Tensor, Wu_rec: torch.Tensor,
    Wg_true: torch.Tensor, Wu_true: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Align neurons using joint (Wg||Wu) features (like v4 joint alignment)."""
    n = min(Wg_rec.shape[0], Wg_true.shape[0])
    rec = torch.cat([Wg_rec[:n], Wu_rec[:n]], dim=-1).float()
    tea = torch.cat([Wg_true[:n], Wu_true[:n]], dim=-1).float()
    sim = F.normalize(rec, dim=-1) @ F.normalize(tea, dim=-1).T
    cost = 1.0 - sim.cpu().numpy()
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except Exception as e:  # noqa: BLE001
        logger.warning("Joint-alignment Hungarian failed: %s", e)
        return (flat_cosine(Wg_rec[:n], Wg_true[:n]),
                per_row_cosine(Wg_rec[:n], Wg_true[:n]),
                flat_cosine(Wu_rec[:n], Wu_true[:n]),
                per_row_cosine(Wu_rec[:n], Wu_true[:n]))
    perm = list(range(n))
    for r, c in zip(row_ind, col_ind):
        perm[r] = c
    Wg_p = torch.stack([Wg_rec[perm[i]] for i in range(n)])
    Wu_p = torch.stack([Wu_rec[perm[i]] for i in range(n)])
    return (flat_cosine(Wg_p, Wg_true[:n]),
            per_row_cosine(Wg_p, Wg_true[:n]),
            flat_cosine(Wu_p, Wu_true[:n]),
            per_row_cosine(Wu_p, Wu_true[:n]))


def lr_schedule(step: int, total: int, warmup: int, peak_lr: float) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * t))


# ──────────────────────────────────────────────────────────────────────────────
# Architecture introspection
# ──────────────────────────────────────────────────────────────────────────────

def introspect_architecture(model) -> dict:
    """Pull key arch attributes off the config, with graceful fallbacks.

    The return dict is also used to decide whether a model is supported.
    """
    cfg = model.config
    info: dict[str, Any] = {}

    def _get(attr, default=None):
        return getattr(cfg, attr, default)

    info["model_type"] = _get("model_type", None)
    info["architecture"] = (getattr(cfg, "architectures", None) or [None])[0]
    info["hidden_size"] = int(_get("hidden_size") or _get("n_embd") or 0)
    info["intermediate_size"] = int(
        _get("intermediate_size") or _get("ffn_hidden_size") or
        _get("moe_intermediate_size") or _get("inter_dim") or 0
    )
    info["num_hidden_layers"] = int(
        _get("num_hidden_layers") or _get("n_layer") or _get("num_layers") or 0
    )
    info["num_attention_heads"] = int(
        _get("num_attention_heads") or _get("n_head") or 0
    )
    info["num_key_value_heads"] = int(
        _get("num_key_value_heads") or info["num_attention_heads"]
    )
    head_dim = _get("head_dim", None)
    if head_dim is None and info["num_attention_heads"] > 0 and info["hidden_size"] > 0:
        head_dim = info["hidden_size"] // info["num_attention_heads"]
    info["head_dim"] = int(head_dim or 0)
    info["vocab_size"] = int(_get("vocab_size") or 0)
    info["tie_word_embeddings"] = bool(_get("tie_word_embeddings", False))
    info["rope_theta"] = float(_get("rope_theta") or _get("rotary_emb_base") or 0) or None
    rope_scaling = _get("rope_scaling", None)
    info["rope_scaling"] = rope_scaling if isinstance(rope_scaling, dict) else None
    info["hidden_act"] = _get("hidden_act") or _get("activation_function")
    info["rms_norm_eps"] = float(_get("rms_norm_eps") or _get("layer_norm_epsilon") or 1e-6)

    # MoE detection
    moe_markers = [
        "num_experts", "n_routed_experts", "num_local_experts",
        "moe_layer_freq", "first_k_dense_replace",
    ]
    is_moe = False
    for m in moe_markers:
        if _get(m, None) is not None:
            is_moe = True
            break
    info["is_moe"] = is_moe
    if is_moe:
        info["num_experts"] = int(
            _get("num_experts") or _get("n_routed_experts")
            or _get("num_local_experts") or 0
        )
        info["moe_intermediate_size"] = _get("moe_intermediate_size")
        info["first_k_dense_replace"] = _get("first_k_dense_replace")

    return info


def find_block_list(model) -> tuple[list[nn.Module], str]:
    """Find the transformer block list and a short prefix for tagging.

    Returns (list_of_blocks, prefix_string). Prefix is e.g. 'model.layers'.
    """
    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("model.decoder.layers", lambda m: m.model.decoder.layers),
        ("transformer.h", lambda m: m.transformer.h),
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
    ]
    for prefix, getter in candidates:
        try:
            blocks = getter(model)
            if blocks is not None and len(blocks) > 0:
                return list(blocks), prefix
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(
        "Could not locate transformer block list on model; "
        "unsupported architecture."
    )


def find_final_norm(model) -> torch.Tensor | None:
    """Return the final RMSNorm/LayerNorm weight (post-last-block), or None."""
    candidates = [
        lambda m: m.model.norm.weight,
        lambda m: m.model.final_layernorm.weight,
        lambda m: m.transformer.ln_f.weight,
        lambda m: m.gpt_neox.final_layer_norm.weight,
    ]
    for fn in candidates:
        try:
            w = fn(model)
            if w is not None:
                return w.data.float().detach().cpu().clone()
        except Exception:  # noqa: BLE001
            continue
    return None


def get_lm_head_weight(model) -> torch.Tensor:
    """Return the effective lm_head weight (V, d). Handles weight tying."""
    try:
        return model.lm_head.weight.data.float().detach().cpu().clone()
    except Exception:  # noqa: BLE001
        return model.get_output_embeddings().weight.data.float().detach().cpu().clone()


def find_dense_mlp(block: nn.Module, is_moe: bool) -> tuple[nn.Module | None, dict]:
    """For dense blocks return the mlp module; for MoE return a dense shared
    MLP if available (DeepSeek-V2 / DeepSeek-MoE style `shared_experts`
    or `shared_expert`). Returns (module_or_None, info_dict).
    """
    info: dict[str, Any] = {}
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        mlp = getattr(block, "feed_forward", None)
    info["block_has_mlp"] = mlp is not None
    if mlp is None:
        return None, info

    def _has_gate_up_down(m):
        return (
            hasattr(m, "gate_proj") and hasattr(m, "up_proj")
            and hasattr(m, "down_proj")
        )

    # Case 1: standard SwiGLU dense MLP
    if _has_gate_up_down(mlp):
        info["path"] = "mlp"
        return mlp, info

    # Case 2: MoE with shared experts (DeepSeek-V2 / Qwen2-MoE)
    for shared_attr in ("shared_experts", "shared_expert"):
        se = getattr(mlp, shared_attr, None)
        if se is not None and _has_gate_up_down(se):
            info["path"] = f"mlp.{shared_attr}"
            info["is_shared_expert"] = True
            return se, info

    # Case 3: DeepSeek-MoE dense first-k layers: the block itself is dense
    # but wrapped in a Moe module. `mlp.experts[0]` is one dense expert.
    experts = getattr(mlp, "experts", None)
    if experts is not None and len(experts) > 0 and _has_gate_up_down(experts[0]):
        info["path"] = "mlp.experts[0]"
        info["is_single_expert"] = True
        info["num_experts"] = len(experts)
        return experts[0], info

    info["path"] = None
    return None, info


def find_post_attn_norm(block: nn.Module) -> torch.Tensor | None:
    """Find the pre-MLP layernorm on a block (post_attention_layernorm)."""
    for name in ("post_attention_layernorm", "post_ffn_layernorm",
                 "ln_2", "post_attention_norm"):
        n = getattr(block, name, None)
        if n is not None and hasattr(n, "weight"):
            return n.weight.data.float().detach().cpu().clone()
    return None


def find_pre_attn_norm(block: nn.Module) -> torch.Tensor | None:
    """Find the pre-attention layernorm (input_layernorm), used for sanity only."""
    for name in ("input_layernorm", "pre_attention_layernorm",
                 "ln_1", "pre_attention_norm"):
        n = getattr(block, name, None)
        if n is not None and hasattr(n, "weight"):
            return n.weight.data.float().detach().cpu().clone()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Query building
# ──────────────────────────────────────────────────────────────────────────────

def build_random_token_queries(
    vocab_size: int, n: int, T: int, seed: int, reserved: int = 0,
) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    lo = max(0, int(reserved))
    hi = int(vocab_size)
    return torch.randint(lo, hi, (n, T), generator=rng)


# ──────────────────────────────────────────────────────────────────────────────
# Hidden-state collection (architecture-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_hidden_states(
    model,
    block,                        # the target transformer block
    query_ids: torch.Tensor,
    block_idx: int,
    device: torch.device,
    batch_size: int,
    d_model: int,
    collect_logits: bool = True,
) -> dict[str, torch.Tensor]:
    """Run model on `query_ids` (N, T) and return h_in, h_mid, h_out (and logits).

    h_mid is reconstructed as h_in + self_attn(out). We hook the self_attn
    module of the target block.
    """
    N, T = query_ids.shape
    attn_buf: list[torch.Tensor] = []

    def attn_hook(module, inputs, output):
        a = output[0] if isinstance(output, tuple) else output
        attn_buf.append(a.detach().to(dtype=torch.float32, device="cpu"))

    attn_mod = getattr(block, "self_attn", None) or getattr(block, "attention", None)
    if attn_mod is None:
        raise RuntimeError("Target block has no self_attn/attention submodule.")

    h = attn_mod.register_forward_hook(attn_hook)

    all_h_in, all_h_mid, all_h_out, all_logits = [], [], [], []
    try:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = query_ids[start:end].to(device)
            attn_buf.clear()
            outputs = model(
                batch, output_hidden_states=True, return_dict=True,
            )
            hs = outputs.hidden_states
            h_in_b = hs[block_idx].detach().to(dtype=torch.float32, device="cpu")
            h_out_b = hs[block_idx + 1].detach().to(dtype=torch.float32, device="cpu")
            if not attn_buf:
                raise RuntimeError(
                    "self_attn forward hook captured nothing — "
                    "unsupported attention wrapper."
                )
            attn_out_b = attn_buf[0]
            if attn_out_b.shape != h_in_b.shape:
                raise RuntimeError(
                    f"self_attn output shape {tuple(attn_out_b.shape)} != "
                    f"h_in shape {tuple(h_in_b.shape)}; unsupported attention wrapper."
                )
            h_mid_b = h_in_b + attn_out_b
            all_h_in.append(h_in_b)
            all_h_mid.append(h_mid_b)
            all_h_out.append(h_out_b)
            if collect_logits:
                logits_b = outputs.logits.detach().to(dtype=torch.float32, device="cpu")
                all_logits.append(logits_b)
            if (start // max(1, batch_size)) % 8 == 0:
                logger.info("    collected %d / %d", end, N)
    finally:
        h.remove()

    ret = {
        "h_in": torch.cat(all_h_in, dim=0),
        "h_mid": torch.cat(all_h_mid, dim=0),
        "h_out": torch.cat(all_h_out, dim=0),
    }
    if collect_logits:
        ret["logits"] = torch.cat(all_logits, dim=0)
    assert ret["h_in"].shape[-1] == d_model, (
        f"h_in dim {ret['h_in'].shape[-1]} != d_model {d_model}"
    )
    return ret


# ──────────────────────────────────────────────────────────────────────────────
# Rank diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def compute_rank_stats(
    X_all: torch.Tensor, device: torch.device,
) -> dict:
    """Effective rank, energy-in-top-k, top-10/bottom-10 singular values."""
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

    sigma_max = float(sigma[0]) if len(sigma) and sigma[0] > 0 else 1.0
    rank_thresh = int((sigma > 0.01 * sigma_max).sum())

    total_energy = float((sigma ** 2).sum())
    topk_list = [k for k in (8, 16, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096) if k <= d]
    # Always include d itself
    if d not in topk_list:
        topk_list.append(d)
    energy_in_topk = {
        int(k): float((sigma[:k] ** 2).sum()) / max(total_energy, 1e-30)
        for k in topk_list
    }

    return {
        "M": int(M),
        "d": int(d),
        "effective_rank": float(eff_rank),
        "rank_at_1pct_sigma_max": rank_thresh,
        "sigma_max": sigma_max,
        "sigma_top10": [float(s) for s in sigma[:10]],
        "sigma_bottom10": [float(s) for s in sigma[-10:]],
        "energy_in_topk": energy_in_topk,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Carlini logit-SVD extraction (last-position)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def carlini_extraction(
    model,
    vocab_size: int,
    hidden_size: int,
    W_lm_true: torch.Tensor,     # (V, d)
    device: torch.device,
    num_queries: int,
    max_seq_len: int,
    seed: int,
    batch_size: int = 32,
) -> dict:
    """Classic Carlini 2024 algebraic extraction.

    Collect last-token logits for N random-token queries, SVD, detect the
    hidden-dim gap, compute principal-angle cosines between recovered and
    true column space of W_lm.
    """
    rng = torch.Generator().manual_seed(seed)
    N = int(num_queries)
    all_logits = []

    t0 = time.time()
    for start in range(0, N, batch_size):
        bs = min(batch_size, N - start)
        ids = torch.randint(0, vocab_size, (bs, max_seq_len), generator=rng)
        ids = ids.to(device)
        out = model(ids)
        all_logits.append(out.logits[:, -1, :].float().cpu())
    Y = torch.cat(all_logits, dim=0)
    collect_time = time.time() - t0

    Yc = Y - Y.mean(dim=0, keepdim=True)
    try:
        U, S, Vh = torch.linalg.svd(Yc, full_matrices=False)
    except Exception as e:  # noqa: BLE001
        logger.warning("  Carlini SVD failed: %s", e)
        return {"error": f"svd_failed: {e}"}

    S_np = S.numpy()
    k = min(len(S_np) - 1, 2 * hidden_size + 50)
    gap_ratios = S_np[:-1] / np.maximum(S_np[1:], 1e-30)
    lo = max(0, hidden_size - 50)
    hi = min(len(gap_ratios), hidden_size + 50)
    if hi <= lo:
        d_hat = int(np.argmax(gap_ratios[:k]))
    else:
        d_hat = int(lo + np.argmax(gap_ratios[lo:hi]))

    # Principal-angle cos between col(Vh[:d_hat].T) and col(W_lm_true)
    try:
        W_hat_T = Vh[:d_hat, :].T.float()
        Q_hat, _ = torch.linalg.qr(W_hat_T)
        Q_true, _ = torch.linalg.qr(W_lm_true.float())
        sigma_angles = torch.linalg.svdvals(Q_hat.T @ Q_true).clamp(0.0, 1.0)
        mean_cos = float(sigma_angles.mean())
        min_cos = float(sigma_angles.min())
        max_cos = float(sigma_angles.max())
        n_angles = int(len(sigma_angles))
    except Exception as e:  # noqa: BLE001
        logger.warning("  principal angles failed: %s", e)
        mean_cos = float("nan")
        min_cos = float("nan")
        max_cos = float("nan")
        n_angles = 0
        sigma_angles = torch.zeros(0)

    return {
        "num_queries": N,
        "d_true": int(hidden_size),
        "d_hat": int(d_hat),
        "d_match": bool(d_hat == hidden_size),
        "gap_ratio_at_d_hat": float(gap_ratios[d_hat]) if d_hat < len(gap_ratios) else 0.0,
        "top_singular_values": [float(s) for s in S_np[:10]],
        "singular_values_around_d": [float(s) for s in
                                     S_np[max(0, hidden_size - 3):hidden_size + 4]],
        "subspace_mean_cos": mean_cos,
        "subspace_min_cos": min_cos,
        "subspace_max_cos": max_cos,
        "num_principal_angles": n_angles,
        "collect_seconds": round(collect_time, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Algebraic W_down OLS with oracle gate/up activations
# ──────────────────────────────────────────────────────────────────────────────

def _activation_fn(name: str | None):
    """Return the nonlinearity used by the MLP gate.

    Most SwiGLU LLMs use SiLU; some use GELU.
    """
    if name is None:
        return F.silu
    name = name.lower()
    if "silu" in name or "swi" in name:
        return F.silu
    if "gelu" in name:
        if "new" in name or "pytorch_tanh" in name:
            return lambda x: F.gelu(x, approximate="tanh")
        return F.gelu
    # Default
    return F.silu


@torch.no_grad()
def solve_w_down_ols(
    h_mid_all: torch.Tensor,    # (N, T, d) float32 CPU
    h_out_all: torch.Tensor,    # (N, T, d) float32 CPU
    g_mlp: torch.Tensor,        # (d,) float32 CPU — post_attention_layernorm.weight
    W_gate_true: torch.Tensor,  # (d_ff, d) — ORACLE
    W_up_true: torch.Tensor,    # (d_ff, d) — ORACLE
    device: torch.device,
    batch_size: int,
    rms_eps: float,
    act_fn,
) -> tuple[torch.Tensor, dict]:
    """Solve W_down from r = h_out - h_mid = a W_down^T with a = act(xWg) * (xWu)
    via float64 normal-equation OLS."""
    N, T, d = h_mid_all.shape
    d_ff = W_gate_true.shape[0]
    M = N * T

    g_dev = g_mlp.to(device)
    Wg_dev = W_gate_true.to(device)
    Wu_dev = W_up_true.to(device)

    AAt = torch.zeros(d_ff, d_ff, dtype=torch.float64, device=device)
    RAt = torch.zeros(d, d_ff, dtype=torch.float64, device=device)

    for n_start in range(0, N, batch_size):
        n_end = min(n_start + batch_size, N)
        h_mid_b = h_mid_all[n_start:n_end].to(device)
        h_out_b = h_out_all[n_start:n_end].to(device)
        mb = h_mid_b.shape[0] * T
        h_mid_flat = h_mid_b.reshape(mb, d)
        h_out_flat = h_out_b.reshape(mb, d)

        x = rms_norm_tensor(h_mid_flat, g_dev, eps=rms_eps)
        g = act_fn(x @ Wg_dev.T)
        u = x @ Wu_dev.T
        a = g * u
        r = (h_out_flat - h_mid_flat).float()

        AAt += a.double().T @ a.double()
        RAt += r.double().T @ a.double()

    diag_mean = float(AAt.diagonal().mean().item())
    try:
        eigvals = torch.linalg.eigvalsh(AAt)
        cond = (eigvals[-1] / eigvals[0].clamp(min=1e-30)).item()
    except Exception:  # noqa: BLE001
        cond = -1.0
    if cond < 0:
        ridge = 1e-8 * diag_mean
    elif cond < 1e8:
        ridge = 1e-10 * diag_mean
    elif cond < 1e12:
        ridge = 1e-8 * diag_mean
    else:
        ridge = 1e-6 * diag_mean
    AAt += ridge * torch.eye(d_ff, dtype=torch.float64, device=device)

    try:
        L = torch.linalg.cholesky(AAt)
        W_down_rec = torch.linalg.solve_triangular(
            L.T, torch.linalg.solve_triangular(L, RAt.T, upper=False), upper=True
        ).T.float()
    except Exception as e:  # noqa: BLE001
        logger.warning("    Cholesky failed (%s); falling back to solve", e)
        W_down_rec = torch.linalg.solve(AAt, RAt.T).T.float()

    return W_down_rec.cpu(), {
        "condition_number": cond,
        "ridge": ridge,
        "diag_mean": diag_mean,
        "num_data_points": M,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Phase-2 joint optimization for (W_gate, W_up) with fixed W_down
# ──────────────────────────────────────────────────────────────────────────────

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
    init_scale: float,
    grad_clip: float,
    eval_every: int,
    seed: int,
    act_fn,
    tag: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Minimize || W_down (act(Wg x) * (Wu x)) - r ||^2 via Adam."""
    M, d = X_all.shape
    d_ff, _ = W_gate_true.shape
    torch.manual_seed(seed)

    W_gate = nn.Parameter(torch.randn(d_ff, d, device=device) * init_scale)
    W_up = nn.Parameter(torch.randn(d_ff, d, device=device) * init_scale)
    W_down = W_down_fixed.float().to(device)

    Wg_true_dev = W_gate_true.float().to(device)
    Wu_true_dev = W_up_true.float().to(device)

    # Leave X and r on CPU, move minibatches as needed
    optimizer = torch.optim.Adam(
        [W_gate, W_up], lr=peak_lr, betas=(0.9, 0.95), eps=1e-8,
    )

    log: list[dict] = []
    rng = torch.Generator().manual_seed(seed + 31337)
    t0 = time.time()
    loss_ema = None
    best_sum = -2.0
    best_state: dict | None = None

    for step in range(total_steps):
        cur_lr = lr_schedule(step, total_steps, warmup_steps, peak_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        idx = torch.randint(0, M, (batch_size,), generator=rng)
        x_b = X_all[idx].to(device, non_blocking=True)
        r_b = r_mlp_all[idx].to(device, non_blocking=True)

        g_pre = x_b @ W_gate.T
        s = act_fn(g_pre)
        u = x_b @ W_up.T
        a = s * u
        r_pred = a @ W_down.T
        loss = F.mse_loss(r_pred, r_b)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([W_gate, W_up], grad_clip)
        optimizer.step()

        lv = loss.item()
        loss_ema = lv if loss_ema is None else 0.99 * loss_ema + 0.01 * lv

        if step % eval_every == 0 or step == total_steps - 1:
            with torch.no_grad():
                Wg = W_gate.detach()
                Wu = W_up.detach()
                raw_g = flat_cosine(Wg, Wg_true_dev)
                raw_u = flat_cosine(Wu, Wu_true_dev)
                raw_g_row = per_row_cosine(Wg, Wg_true_dev)
                raw_u_row = per_row_cosine(Wu, Wu_true_dev)

                do_align = (step == total_steps - 1) or (step % (eval_every * 4) == 0)
                sg_flat = sg_row = su_flat = su_row = float("nan")
                jg_flat = jg_row = ju_flat = ju_row = float("nan")
                if do_align:
                    Wg_cpu = Wg.cpu()
                    Wu_cpu = Wu.cpu()
                    sg_flat, sg_row = cos_with_self_alignment(
                        Wg_cpu, W_gate_true.cpu()
                    )
                    su_flat, su_row = cos_with_self_alignment(
                        Wu_cpu, W_up_true.cpu()
                    )
                    jg_flat, jg_row, ju_flat, ju_row = cos_with_joint_alignment(
                        Wg_cpu, Wu_cpu,
                        W_gate_true.cpu(), W_up_true.cpu(),
                    )

                elapsed = time.time() - t0
                if do_align:
                    logger.info(
                        "    [%s] step %5d/%d lr=%.2e loss=%.4e ema=%.4e | "
                        "raw(g,u)=(%.3f,%.3f) self(g,u)=(%.3f,%.3f) "
                        "joint(g,u)=(%.3f,%.3f) | %.1fs",
                        tag, step, total_steps, cur_lr, lv, loss_ema,
                        raw_g, raw_u, sg_flat, su_flat, jg_flat, ju_flat, elapsed,
                    )
                else:
                    logger.info(
                        "    [%s] step %5d/%d lr=%.2e loss=%.4e ema=%.4e | "
                        "raw(g,u)=(%.3f,%.3f) row=(%.3f,%.3f) | %.1fs",
                        tag, step, total_steps, cur_lr, lv, loss_ema,
                        raw_g, raw_u, raw_g_row, raw_u_row, elapsed,
                    )

                entry = {
                    "step": step,
                    "lr": cur_lr,
                    "loss": lv,
                    "loss_ema": loss_ema,
                    "raw_cos_gate": raw_g,
                    "raw_cos_up": raw_u,
                    "raw_cos_gate_row": raw_g_row,
                    "raw_cos_up_row": raw_u_row,
                    "self_aligned_cos_gate_flat": sg_flat,
                    "self_aligned_cos_gate_row": sg_row,
                    "self_aligned_cos_up_flat": su_flat,
                    "self_aligned_cos_up_row": su_row,
                    "joint_aligned_cos_gate_flat": jg_flat,
                    "joint_aligned_cos_gate_row": jg_row,
                    "joint_aligned_cos_up_flat": ju_flat,
                    "joint_aligned_cos_up_row": ju_row,
                }
                log.append(entry)

                if do_align and not math.isnan(jg_flat):
                    sum_cos = jg_flat + ju_flat
                else:
                    sum_cos = raw_g + raw_u
                if sum_cos > best_sum:
                    best_sum = sum_cos
                    best_state = {
                        "W_gate": Wg.detach().cpu().clone(),
                        "W_up": Wu.detach().cpu().clone(),
                        "step": step,
                        "sum_cos": sum_cos,
                    }

    if best_state is not None:
        return best_state["W_gate"], best_state["W_up"], log
    return W_gate.detach().cpu(), W_up.detach().cpu(), log


# ──────────────────────────────────────────────────────────────────────────────
# Per-model pipeline
# ──────────────────────────────────────────────────────────────────────────────

def pick_target_block(num_hidden_layers: int, offset: int) -> int:
    return max(0, num_hidden_layers - max(1, int(offset)))


def run_model(args, model_name: str, out_dir: Path) -> dict:
    """Run the full pipeline on a single model.

    Returns a lightweight summary dict (for the global summary) and writes
    `out_dir/results.json` with per-model metrics.
    """
    t_model_start = time.time()
    logger.info("=" * 80)
    logger.info("MODEL: %s  ->  %s", model_name, out_dir)
    logger.info("=" * 80)

    # HF offline first, network fallback
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = torch.device(args.device)
    model_dtype = dtype_from_str(args.model_dtype)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Load teacher with graceful online fallback ───────────────────────────
    def _load_model(local_only: bool):
        kwargs = dict(
            torch_dtype=model_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if args.device_map == "single":
            kwargs["device_map"] = {"": device}
        elif args.device_map == "balanced_low_0":
            kwargs["device_map"] = "balanced_low_0"
        else:
            kwargs["device_map"] = "auto"
        if local_only:
            kwargs["local_files_only"] = True
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    try:
        teacher = _load_model(local_only=True)
        load_mode = "offline"
    except Exception as e1:
        logger.warning("  Offline load failed (%s); retrying with network.", e1)
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        teacher = _load_model(local_only=False)
        load_mode = "online"
    teacher.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True,
        )
    except Exception:  # noqa: BLE001
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("  Tokenizer load failed: %s", e)
            tokenizer = None
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Introspect architecture ──────────────────────────────────────────────
    arch = introspect_architecture(teacher)
    arch["load_mode"] = load_mode
    arch["model_dtype"] = args.model_dtype
    logger.info("  Arch: hidden=%d  inter=%d  layers=%d  heads=%d/%d (GQA=%s)  "
                "vocab=%d  tie=%s  act=%s  MoE=%s",
                arch["hidden_size"], arch["intermediate_size"],
                arch["num_hidden_layers"], arch["num_attention_heads"],
                arch["num_key_value_heads"],
                arch["num_attention_heads"] != arch["num_key_value_heads"],
                arch["vocab_size"], arch["tie_word_embeddings"],
                arch["hidden_act"], arch["is_moe"])

    d_model = arch["hidden_size"]
    d_ff = arch["intermediate_size"]
    n_layers = arch["num_hidden_layers"]
    vocab = arch["vocab_size"]
    rms_eps = arch["rms_norm_eps"]
    act_fn = _activation_fn(arch.get("hidden_act"))

    if d_model <= 0 or d_ff <= 0 or n_layers <= 0:
        raise RuntimeError(
            f"Unsupported architecture (d={d_model}, dff={d_ff}, layers={n_layers}).",
        )

    # ── Find block list and target block ─────────────────────────────────────
    blocks, _blocks_prefix = find_block_list(teacher)
    block_idx = pick_target_block(n_layers, args.block_offset_from_end)
    logger.info("  Target block: %d / %d", block_idx, n_layers)
    block = blocks[block_idx]
    attn_mod = getattr(block, "self_attn", None) or getattr(block, "attention", None)
    if attn_mod is None:
        raise RuntimeError("No self_attn/attention on the target block.")

    # ── Find MLP module (dense) ──────────────────────────────────────────────
    mlp_mod, mlp_info = find_dense_mlp(block, arch["is_moe"])
    arch["mlp_info"] = mlp_info
    if mlp_mod is None:
        # MoE block with routed-only experts — try a shallower dense layer.
        if arch["is_moe"]:
            fkdr = arch.get("first_k_dense_replace") or 1
            fallback_idx = max(0, min(n_layers - 1, int(fkdr) - 1))
            logger.warning("  Block %d has no dense MLP; falling back to block %d",
                           block_idx, fallback_idx)
            block_idx = fallback_idx
            block = blocks[block_idx]
            mlp_mod, mlp_info = find_dense_mlp(block, arch["is_moe"])
            arch["mlp_info_fallback"] = mlp_info
            attn_mod = getattr(block, "self_attn", None) or getattr(block, "attention", None)
        if mlp_mod is None:
            raise RuntimeError(
                "No dense SwiGLU MLP found on any block — cannot run pipeline."
            )

    # Teacher tensors (ORACLE for eval only)
    W_gate_true = mlp_mod.gate_proj.weight.data.float().detach().cpu().clone()
    W_up_true = mlp_mod.up_proj.weight.data.float().detach().cpu().clone()
    W_down_true = mlp_mod.down_proj.weight.data.float().detach().cpu().clone()

    g_mlp = find_post_attn_norm(block)
    if g_mlp is None:
        logger.warning("  No post_attention_layernorm weight found; using 1s.")
        g_mlp = torch.ones(d_model)
    g_final = find_final_norm(teacher)
    if g_final is None:
        logger.warning("  No final norm weight; using 1s.")
        g_final = torch.ones(d_model)

    W_lm_true = get_lm_head_weight(teacher)  # (V, d)
    if W_lm_true.shape != (vocab, d_model):
        logger.warning("  Unexpected lm_head shape %s (expected %s); "
                       "Carlini step will still run, shape might not match exactly.",
                       tuple(W_lm_true.shape), (vocab, d_model))

    logger.info("  Oracle shapes: Wg=%s Wu=%s Wd=%s",
                tuple(W_gate_true.shape), tuple(W_up_true.shape),
                tuple(W_down_true.shape))
    assert W_gate_true.shape == (d_ff, d_model), W_gate_true.shape
    assert W_up_true.shape == (d_ff, d_model), W_up_true.shape
    assert W_down_true.shape == (d_model, d_ff), W_down_true.shape

    # ── Budget: maybe shrink N given d_model ─────────────────────────────────
    N = args.num_queries
    T = args.max_seq_len
    M_max = int(args.max_data_points)
    if N * T > M_max:
        N = max(1, M_max // max(1, T))
        logger.warning("  Capping N from %d -> %d to honour max_data_points=%d.",
                       args.num_queries, N, M_max)

    # ── Build random-token queries ───────────────────────────────────────────
    query_ids = build_random_token_queries(vocab, N, T, args.seed)
    logger.info("  Query ids %s (min=%d max=%d mean=%.1f)",
                tuple(query_ids.shape),
                int(query_ids.min().item()), int(query_ids.max().item()),
                float(query_ids.float().mean().item()))

    per_model: dict[str, Any] = {
        "model_name": model_name,
        "architecture": arch,
        "block_idx": block_idx,
        "num_hidden_layers": n_layers,
        "num_queries": N,
        "max_seq_len": T,
        "carlini_num_queries": args.carlini_num_queries,
        "config": vars(args),
    }

    # ════════════════════════════════════════════════════════════════════════
    # Step A: collect hidden states
    # ════════════════════════════════════════════════════════════════════════
    logger.info("  [A] Collect hidden states (N=%d, T=%d)", N, T)
    tA = time.time()
    hs = collect_hidden_states(
        teacher, block, query_ids, block_idx, device, args.batch_size,
        d_model=d_model, collect_logits=False,
    )
    h_mid = hs["h_mid"]
    h_out = hs["h_out"]
    del hs["h_in"], hs
    per_model["collect_seconds"] = round(time.time() - tA, 2)
    logger.info("    done in %.1fs  | h_mid %s, h_out %s",
                per_model["collect_seconds"],
                tuple(h_mid.shape), tuple(h_out.shape))

    # Build X = RMSNorm(h_mid) and r_mlp = h_out - h_mid, both (M, d)
    X_all = torch.empty(N * T, d_model, dtype=torch.float32)
    r_mlp_all = torch.empty(N * T, d_model, dtype=torch.float32)
    g_dev = g_mlp.to(device)
    with torch.no_grad():
        for s in range(0, N, args.batch_size):
            e = min(s + args.batch_size, N)
            h_mid_b = h_mid[s:e].to(device)
            h_out_b = h_out[s:e].to(device)
            mb = h_mid_b.shape[0] * T
            x = rms_norm_tensor(h_mid_b.reshape(mb, d_model), g_dev, eps=rms_eps)
            r = (h_out_b.reshape(mb, d_model) - h_mid_b.reshape(mb, d_model)).float()
            off = s * T
            X_all[off:off + mb] = x.cpu()
            r_mlp_all[off:off + mb] = r.cpu()

    # ── Rank diagnostics on X ────────────────────────────────────────────────
    logger.info("  [B] Rank diagnostics on X = RMSNorm(h_mid)")
    tB = time.time()
    rank_stats = compute_rank_stats(X_all, device)
    rank_stats["time_seconds"] = round(time.time() - tB, 2)
    per_model["h_mid_rank"] = {
        "effective_rank": rank_stats["effective_rank"],
        "rank_at_1pct_sigma_max": rank_stats["rank_at_1pct_sigma_max"],
        "d": rank_stats["d"],
        "sigma_max": rank_stats["sigma_max"],
        "top_128_energy": rank_stats["energy_in_topk"].get(128),
        "top_256_energy": rank_stats["energy_in_topk"].get(256),
        "top_512_energy": rank_stats["energy_in_topk"].get(512),
        "energy_in_topk": rank_stats["energy_in_topk"],
        "sigma_top10": rank_stats["sigma_top10"],
        "M": rank_stats["M"],
        "time_seconds": rank_stats["time_seconds"],
    }
    logger.info("    eff_rank=%.2f / %d  (frac=%.4f)  top128=%.3f  top256=%.3f",
                rank_stats["effective_rank"], rank_stats["d"],
                rank_stats["effective_rank"] / max(rank_stats["d"], 1),
                per_model["h_mid_rank"]["top_128_energy"] or float("nan"),
                per_model["h_mid_rank"]["top_256_energy"] or float("nan"))

    # ════════════════════════════════════════════════════════════════════════
    # Step C: Carlini algebraic extraction
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_carlini:
        logger.info("  [C] Carlini logit-SVD extraction (N=%d last-token)",
                    args.carlini_num_queries)
        tC = time.time()
        try:
            carlini_res = carlini_extraction(
                teacher, vocab, d_model, W_lm_true, device,
                num_queries=args.carlini_num_queries,
                max_seq_len=T, seed=args.seed + 17,
                batch_size=args.batch_size,
            )
            carlini_res["time_seconds"] = round(time.time() - tC, 2)
            per_model["carlini"] = carlini_res
            logger.info("    d_hat=%d (true %d, match=%s)  "
                        "subspace mean_cos=%.4f  min_cos=%.4f  time=%.1fs",
                        carlini_res.get("d_hat", -1), d_model,
                        carlini_res.get("d_match"),
                        carlini_res.get("subspace_mean_cos", float("nan")),
                        carlini_res.get("subspace_min_cos", float("nan")),
                        carlini_res["time_seconds"])
        except Exception as e:  # noqa: BLE001
            logger.warning("  Carlini step failed: %s", e)
            per_model["carlini"] = {"error": str(e), "traceback": traceback.format_exc()}
    else:
        per_model["carlini"] = {"skipped": True}

    # ════════════════════════════════════════════════════════════════════════
    # Step D: algebraic W_down OLS (oracle gate/up activations)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("  [D] Algebraic W_down OLS (oracle gate/up, float64, M=%d)", N * T)
    tD = time.time()
    try:
        W_down_rec, ols_metrics = solve_w_down_ols(
            h_mid, h_out, g_mlp,
            W_gate_true, W_up_true, device, args.batch_size,
            rms_eps=rms_eps, act_fn=act_fn,
        )
        cos_wd = flat_cosine(W_down_rec, W_down_true)
        cos_wd_row = per_row_cosine(W_down_rec.T, W_down_true.T)
        frob = (W_down_rec - W_down_true).norm().item() / (
            W_down_true.norm().item() + 1e-12
        )
        # Functional reconstruction of the MLP residual on a held-out slice
        with torch.no_grad():
            hold_start = int(0.9 * X_all.shape[0])
            hold_end = min(hold_start + 16384, X_all.shape[0])
            X_hold = X_all[hold_start:hold_end].to(device)
            r_hold = r_mlp_all[hold_start:hold_end].to(device)
            Wg_d = W_gate_true.to(device)
            Wu_d = W_up_true.to(device)
            Wd_d = W_down_rec.to(device)
            a = act_fn(X_hold @ Wg_d.T) * (X_hold @ Wu_d.T)
            r_pred = a @ Wd_d.T
            recon_cos = flat_cosine(r_pred.cpu(), r_hold.cpu())
            recon_mse = F.mse_loss(r_pred, r_hold).item()

        per_model["algebraic_w_down"] = {
            "cos": cos_wd,
            "cos_per_column_row": cos_wd_row,
            "frob_relative_error": frob,
            "recon_cos": recon_cos,
            "recon_mse": recon_mse,
            "time_s": round(time.time() - tD, 2),
            **ols_metrics,
        }
        logger.info("    W_down cos=%.4f  frob=%.4f  recon_cos=%.4f  cond=%.2e  time=%.1fs",
                    cos_wd, frob, recon_cos, ols_metrics.get("condition_number", -1.0),
                    per_model["algebraic_w_down"]["time_s"])
    except Exception as e:  # noqa: BLE001
        logger.warning("  W_down OLS failed: %s", e)
        per_model["algebraic_w_down"] = {"error": str(e), "traceback": traceback.format_exc()}
        W_down_rec = None

    # Free h_mid / h_out now (they're big for long contexts)
    del h_mid, h_out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════════
    # Step E: joint (W_gate, W_up) optimization with fixed W_down
    # ════════════════════════════════════════════════════════════════════════
    if not args.skip_joint_opt and W_down_rec is not None:
        logger.info("  [E] Joint (W_gate, W_up) optimization "
                    "(%d steps, batch=%d, lr=%.1e)",
                    args.joint_opt_steps, args.opt_batch_size, args.lr)
        tE = time.time()
        try:
            Wg_rec, Wu_rec, log = joint_optimize_gate_up(
                X_all, r_mlp_all, W_down_rec,
                W_gate_true, W_up_true, device,
                total_steps=args.joint_opt_steps,
                batch_size=args.opt_batch_size,
                peak_lr=args.lr,
                warmup_steps=min(args.lr_warmup_steps,
                                 max(50, args.joint_opt_steps // 10)),
                init_scale=args.init_scale,
                grad_clip=args.grad_clip,
                eval_every=args.eval_every,
                seed=args.seed + 71,
                act_fn=act_fn,
                tag=safe_name(model_name)[:20],
            )
            p2_time = time.time() - tE

            # Metrics
            raw_g = flat_cosine(Wg_rec, W_gate_true)
            raw_u = flat_cosine(Wu_rec, W_up_true)
            raw_g_row = per_row_cosine(Wg_rec, W_gate_true)
            raw_u_row = per_row_cosine(Wu_rec, W_up_true)
            sg_flat, sg_row = cos_with_self_alignment(Wg_rec, W_gate_true)
            su_flat, su_row = cos_with_self_alignment(Wu_rec, W_up_true)
            jg_flat, jg_row, ju_flat, ju_row = cos_with_joint_alignment(
                Wg_rec, Wu_rec, W_gate_true, W_up_true,
            )

            # Functional reconstruction on held-out slice
            with torch.no_grad():
                hold_start = int(0.9 * X_all.shape[0])
                hold_end = min(hold_start + 16384, X_all.shape[0])
                X_hold = X_all[hold_start:hold_end].to(device)
                r_hold = r_mlp_all[hold_start:hold_end].to(device)
                Wg_d = Wg_rec.to(device)
                Wu_d = Wu_rec.to(device)
                Wd_d = W_down_rec.to(device)
                a = act_fn(X_hold @ Wg_d.T) * (X_hold @ Wu_d.T)
                r_pred = a @ Wd_d.T
                func_cos = flat_cosine(r_pred.cpu(), r_hold.cpu())
                func_mse = F.mse_loss(r_pred, r_hold).item()

            per_model["joint_opt"] = {
                "raw_cos_gate": raw_g,
                "raw_cos_up": raw_u,
                "raw_cos_gate_row": raw_g_row,
                "raw_cos_up_row": raw_u_row,
                "self_aligned_cos_gate_flat": sg_flat,
                "self_aligned_cos_gate_row": sg_row,
                "self_aligned_cos_up_flat": su_flat,
                "self_aligned_cos_up_row": su_row,
                "joint_aligned_cos_gate_flat": jg_flat,
                "joint_aligned_cos_gate_row": jg_row,
                "joint_aligned_cos_up_flat": ju_flat,
                "joint_aligned_cos_up_row": ju_row,
                "functional_recon_cos": func_cos,
                "functional_recon_mse": func_mse,
                "final_loss_ema": log[-1]["loss_ema"] if log else None,
                "steps": args.joint_opt_steps,
                "time_s": round(p2_time, 2),
                "log_tail": log[-5:],
                # Alias fields asked for in the spec
                "gate_cos": jg_flat if not math.isnan(jg_flat) else raw_g,
                "up_cos": ju_flat if not math.isnan(ju_flat) else raw_u,
                "functional_recon": func_cos,
            }
            logger.info("    raw(g,u)=(%.3f,%.3f) self(g,u)=(%.3f,%.3f) "
                        "joint(g,u)=(%.3f,%.3f) func=%.3f time=%.1fs",
                        raw_g, raw_u, sg_flat, su_flat,
                        jg_flat, ju_flat, func_cos, p2_time)
        except Exception as e:  # noqa: BLE001
            logger.warning("  Joint opt failed: %s", e)
            per_model["joint_opt"] = {"error": str(e),
                                      "traceback": traceback.format_exc()}
    else:
        per_model["joint_opt"] = {"skipped": True}

    # Clean up
    del X_all, r_mlp_all
    if W_down_rec is not None:
        del W_down_rec
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    per_model["total_time_s"] = round(time.time() - t_model_start, 2)
    logger.info("  MODEL DONE in %.1fs", per_model["total_time_s"])

    # Persist per-model JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(_jsonify(per_model), f, indent=2)
    logger.info("  Wrote %s", out_dir / "results.json")

    # Release model
    del teacher
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return per_model


# ──────────────────────────────────────────────────────────────────────────────
# Summary rendering
# ──────────────────────────────────────────────────────────────────────────────

def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_summary_row(per_model: dict) -> dict:
    arch = per_model.get("architecture", {})
    rank = per_model.get("h_mid_rank", {})
    carlini = per_model.get("carlini", {})
    wd = per_model.get("algebraic_w_down", {})
    jo = per_model.get("joint_opt", {})
    d = rank.get("d")
    eff = rank.get("effective_rank")
    eff_frac = (eff / d) if (eff and d) else None

    return {
        "model_name": per_model.get("model_name"),
        "hidden_size": arch.get("hidden_size"),
        "intermediate_size": arch.get("intermediate_size"),
        "num_hidden_layers": arch.get("num_hidden_layers"),
        "num_attention_heads": arch.get("num_attention_heads"),
        "num_key_value_heads": arch.get("num_key_value_heads"),
        "tie_word_embeddings": arch.get("tie_word_embeddings"),
        "hidden_act": arch.get("hidden_act"),
        "is_moe": arch.get("is_moe"),
        "block_idx": per_model.get("block_idx"),
        "h_mid_eff_rank": eff,
        "h_mid_eff_rank_frac": eff_frac,
        "h_mid_top128_energy": rank.get("top_128_energy"),
        "h_mid_top256_energy": rank.get("top_256_energy"),
        "carlini_d_hat": carlini.get("d_hat"),
        "carlini_d_match": carlini.get("d_match"),
        "carlini_subspace_cos": carlini.get("subspace_mean_cos"),
        "carlini_subspace_min_cos": carlini.get("subspace_min_cos"),
        "w_down_cos": wd.get("cos"),
        "w_down_recon_cos": wd.get("recon_cos"),
        "w_down_frob": wd.get("frob_relative_error"),
        "w_down_time_s": wd.get("time_s"),
        "joint_gate_cos": jo.get("gate_cos"),
        "joint_up_cos": jo.get("up_cos"),
        "joint_functional_recon": jo.get("functional_recon"),
        "joint_raw_gate_cos": jo.get("raw_cos_gate"),
        "joint_raw_up_cos": jo.get("raw_cos_up"),
        "joint_time_s": jo.get("time_s"),
        "total_time_s": per_model.get("total_time_s"),
        "error": per_model.get("error"),
    }


def render_summary_table(rows: list[dict]) -> str:
    """ASCII summary table for console output."""
    headers = [
        ("model", 28),
        ("d", 5),
        ("dff", 6),
        ("L", 4),
        ("MoE", 5),
        ("eff_r", 7),
        ("eff/d", 7),
        ("carlini", 7),
        ("Wdown", 7),
        ("recon", 7),
        ("gate", 6),
        ("up", 6),
        ("func", 6),
        ("time", 6),
    ]

    def fmt(row):
        def _f(v, width, kind="f3"):
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return "-".rjust(width)
            if kind == "f3":
                return f"{v:>{width}.3f}"
            if kind == "f2":
                return f"{v:>{width}.2f}"
            if kind == "f1":
                return f"{v:>{width}.1f}"
            if kind == "int":
                return f"{int(v):>{width}d}"
            return str(v).rjust(width)
        parts = [
            str(row.get("model_name") or "?")[-28:].ljust(28),
            _f(row.get("hidden_size"), 5, "int"),
            _f(row.get("intermediate_size"), 6, "int"),
            _f(row.get("num_hidden_layers"), 4, "int"),
            ("Y" if row.get("is_moe") else "N").rjust(5),
            _f(row.get("h_mid_eff_rank"), 7, "f2"),
            _f(row.get("h_mid_eff_rank_frac"), 7, "f3"),
            _f(row.get("carlini_subspace_cos"), 7, "f3"),
            _f(row.get("w_down_cos"), 7, "f3"),
            _f(row.get("w_down_recon_cos"), 7, "f3"),
            _f(row.get("joint_gate_cos"), 6, "f3"),
            _f(row.get("joint_up_cos"), 6, "f3"),
            _f(row.get("joint_functional_recon"), 6, "f3"),
            _f(row.get("total_time_s"), 6, "f1"),
        ]
        return "  ".join(parts)

    header = "  ".join(h.ljust(w) if i == 0 else h.rjust(w)
                       for i, (h, w) in enumerate(headers))
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        lines.append(fmt(row))
    return "\n".join(lines)


def write_summary_csv(rows: list[dict], path: Path):
    """Write a CSV mirror of the summary rows (optional convenience)."""
    if not rows:
        return
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in keys})


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model list
    if args.models_json:
        with open(args.models_json) as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "models" in raw:
            raw = raw["models"]
        models = []
        for item in raw:
            if isinstance(item, str):
                models.append(item)
            elif isinstance(item, dict) and "model_name" in item:
                models.append(item["model_name"])
    else:
        models = args.models or list(DEFAULT_MODELS)

    if not models:
        raise SystemExit("No models specified. Pass --models or --models_json.")

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(out_dir / "models.json", "w") as f:
        json.dump({"models": models}, f, indent=2)

    logger.info("=" * 80)
    logger.info("MULTI-MODEL ALGEBRAIC RECOVERY SWEEP")
    logger.info("=" * 80)
    logger.info("Models       : %s", models)
    logger.info("Output dir   : %s", out_dir)
    logger.info("N x T        : %d x %d", args.num_queries, args.max_seq_len)
    logger.info("Joint steps  : %d", args.joint_opt_steps)
    logger.info("Device       : %s  (dtype=%s, device_map=%s)",
                args.device, args.model_dtype, args.device_map)

    t_global = time.time()
    all_per_model: list[dict] = []
    summary_rows: list[dict] = []

    for i, mname in enumerate(models):
        tag = safe_name(mname)
        model_out = out_dir / tag
        model_out.mkdir(parents=True, exist_ok=True)
        res_path = model_out / "results.json"

        if args.resume and res_path.exists():
            logger.info("[%d/%d] SKIP (exists): %s", i + 1, len(models), mname)
            try:
                with open(res_path) as f:
                    pm = json.load(f)
                all_per_model.append(pm)
                summary_rows.append(build_summary_row(pm))
            except Exception as e:  # noqa: BLE001
                logger.warning("  Could not re-read existing results: %s", e)
            continue

        logger.info("[%d/%d] START: %s -> %s",
                    i + 1, len(models), mname, model_out)
        try:
            pm = run_model(args, mname, model_out)
            all_per_model.append(pm)
            summary_rows.append(build_summary_row(pm))
        except Exception as e:  # noqa: BLE001
            logger.error("[%d/%d] FAILED: %s -- %s",
                         i + 1, len(models), mname, e)
            logger.error("  %s", traceback.format_exc())
            err = {
                "model_name": mname,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            all_per_model.append(err)
            summary_rows.append(build_summary_row(err))
            with open(model_out / "results.json", "w") as f:
                json.dump(_jsonify(err), f, indent=2)
            if not args.skip_on_error:
                raise

            # After error, actively clean up GPU memory so the next model has room.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Write incremental summary (so an interrupted sweep still has one)
        summary = {
            "args": vars(args),
            "models": models,
            "per_model_summary": summary_rows,
            "full_results": all_per_model,
            "elapsed_seconds": round(time.time() - t_global, 1),
            "completed_count": i + 1,
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(_jsonify(summary), f, indent=2)
        write_summary_csv(summary_rows, out_dir / "summary.csv")

    # Final rendering
    total = time.time() - t_global
    print()
    print("=" * 110)
    print("MULTI-MODEL ALGEBRAIC RECOVERY SWEEP — SUMMARY")
    print("=" * 110)
    print(render_summary_table(summary_rows))
    print("-" * 110)
    print(f"Total elapsed: {total:.1f}s   ({total/60:.1f} min)")
    print(f"Output dir   : {out_dir}")
    print(f"Summary JSON : {out_dir / 'summary.json'}")
    print(f"Summary CSV  : {out_dir / 'summary.csv'}")
    print("=" * 110)


if __name__ == "__main__":
    main()
