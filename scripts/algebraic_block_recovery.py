#!/usr/bin/env python3
"""Algebraic Block Recovery: Structured regression for transformer block parameter recovery.

Recovers Block 23 parameters of Qwen2.5-0.5B by exploiting the algebraic
structure of the residual stream, RMSNorm, and SiLU-gated MLP — no gradient
descent on the block itself.

Pipeline:
  Step 0: Recover h_out from logits via pinv(W_lm) and RMS re-scaling.
  Step 1: Oracle MLP recovery — given TRUE h_mid and TRUE activations,
          solve W_down by OLS.  Validates the linear algebra.
  Step 2: Semi-oracle MLP — given TRUE h_mid, recover W_gate/W_up/W_down
          via Alternating Least Squares (ALS).
  Step 3: Blind MLP bootstrap — estimate h_mid from h_out_est, iterate.

Usage:
    python scripts/algebraic_block_recovery.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --block_idx 23 \
        --num_queries 2048 \
        --max_seq_len 128 \
        --output_dir results/v5_algebraic_recovery \
        --seed 42 \
        --device cuda:0 \
        --als_iters 20 \
        --als_gate_lr 0.01 \
        --als_gate_steps 100
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.permutation_alignment import align_ffn_neurons, compute_aligned_cosine

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
    p = argparse.ArgumentParser(description="Algebraic Block Recovery")
    p.add_argument("--model_name", type=str, default=MODEL_NAME)
    p.add_argument("--block_idx", type=int, default=23)
    p.add_argument("--num_queries", type=int, default=2048)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--output_dir", type=str, default="results/v5_algebraic_recovery")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--als_iters", type=int, default=20)
    p.add_argument("--als_gate_lr", type=float, default=0.01)
    p.add_argument("--als_gate_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for data collection and OLS accumulation")
    p.add_argument("--allow_synthetic", action="store_true",
                   help="Fall back to random tokens if dataset load fails")
    p.add_argument("--ols_ridge", type=float, default=1e-4,
                   help="Tikhonov regularisation for OLS solves")
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


# ── cosine helper ────────────────────────────────────────────────────────────

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

    logger.info("=== Algebraic Block Recovery ===")
    logger.info("Model : %s", args.model_name)
    logger.info("Block : %d", block_idx)
    logger.info("Queries: %d x %d tokens", N, T)
    logger.info("Device : %s", device)

    # Save args
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

    # Convenience aliases
    W_lm = teacher.lm_head.weight.data.float().to(device)         # [V, d]
    g_final = teacher.model.norm.weight.data.float().to(device)    # [d]
    g_attn = true_params[f"{prefix}.input_layernorm.weight"].float()   # [d]
    g_mlp = true_params[f"{prefix}.post_attention_layernorm.weight"].float()  # [d]
    W_gate_true = true_params[f"{prefix}.mlp.gate_proj.weight"].float()  # [d_ff, d]
    W_up_true = true_params[f"{prefix}.mlp.up_proj.weight"].float()      # [d_ff, d]
    W_down_true = true_params[f"{prefix}.mlp.down_proj.weight"].float()  # [d, d_ff]

    logger.info("Teacher parameters extracted.")
    logger.info("  W_lm: %s, g_final: %s", W_lm.shape, g_final.shape)
    logger.info("  W_gate: %s, W_up: %s, W_down: %s",
                W_gate_true.shape, W_up_true.shape, W_down_true.shape)

    # ── Build query pool ─────────────────────────────────────────────────────
    logger.info("Building query pool (%d queries)...", N)
    query_ids = build_query_pool(
        tokenizer, N, T, args.seed, allow_synthetic=args.allow_synthetic,
    )
    logger.info("Query pool ready: %s", query_ids.shape)

    # ── Collect hidden states via hooks ──────────────────────────────────────
    # We need:
    #   h_in[n,t]  = output of block (block_idx - 1) = input to block block_idx
    #   h_mid[n,t] = h_in + Attn(RMSNorm(h_in))  (input to MLP sub-layer)
    #   h_out[n,t] = output of block block_idx

    logger.info("Collecting hidden states and logits...")

    # hidden_states[i] = output after block i (hidden_states[0] = embeddings)
    # So h_in  = hidden_states[block_idx]      (output of block block_idx-1)
    #    h_out = hidden_states[block_idx + 1]   (output of block block_idx)

    # For h_mid we hook into the MLP sub-layer.
    # In Qwen2: block.mlp receives input AFTER post_attention_layernorm,
    # but h_mid is the RESIDUAL STATE before the MLP sub-layer.
    # h_mid = h_in + attn_output.
    # We can get h_mid = h_out - mlp_output, or via the self_attn output.

    # Strategy: use output_hidden_states for h_in and h_out;
    # hook into block.self_attn output for attn_output -> h_mid = h_in + attn_output

    all_h_in = []
    all_h_mid = []
    all_h_out = []
    all_logits = []

    # Register hook to capture attention residual output
    target_block = teacher.model.layers[block_idx]
    attn_outputs_buffer: list[torch.Tensor] = []

    def attn_output_hook(module, inputs, output):
        # Qwen2Attention.forward returns (attn_output, attn_weights, past_kv)
        # attn_output shape: [batch, seq_len, d]
        attn_out = output[0] if isinstance(output, tuple) else output
        attn_outputs_buffer.append(attn_out.detach().cpu().float())

    hook_handle = target_block.self_attn.register_forward_hook(attn_output_hook)

    with torch.no_grad():
        for start in range(0, N, BS):
            end = min(start + BS, N)
            batch = query_ids[start:end].to(device)

            attn_outputs_buffer.clear()
            outputs = teacher(batch, output_hidden_states=True, return_dict=True)

            # hidden_states: tuple of (num_layers+1) tensors, each [batch, T, d]
            hs = outputs.hidden_states
            h_in_batch = hs[block_idx].detach().cpu().float()      # input to block
            h_out_batch = hs[block_idx + 1].detach().cpu().float() # output of block

            # attn_output was captured by hook
            attn_out_batch = attn_outputs_buffer[0]  # [batch, T, d]
            h_mid_batch = h_in_batch + attn_out_batch  # residual after attention

            logits_batch = outputs.logits.detach().cpu().float()   # [batch, T, V]

            all_h_in.append(h_in_batch)
            all_h_mid.append(h_mid_batch)
            all_h_out.append(h_out_batch)
            all_logits.append(logits_batch)

            if (start // BS) % 10 == 0:
                logger.info("  Collected %d / %d queries", end, N)

    hook_handle.remove()

    h_in_all = torch.cat(all_h_in, dim=0)    # [N, T, d]
    h_mid_all = torch.cat(all_h_mid, dim=0)  # [N, T, d]
    h_out_all = torch.cat(all_h_out, dim=0)  # [N, T, d]
    logits_all = torch.cat(all_logits, dim=0) # [N, T, V]

    logger.info("Hidden states collected:")
    logger.info("  h_in:   %s", h_in_all.shape)
    logger.info("  h_mid:  %s", h_mid_all.shape)
    logger.info("  h_out:  %s", h_out_all.shape)
    logger.info("  logits: %s", logits_all.shape)

    # Free the full logits from GPU, we work on CPU
    del all_h_in, all_h_mid, all_h_out, all_logits
    torch.cuda.empty_cache()

    results: dict = {}
    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 0: Recover h_out from logits
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 0: Recover h_out from logits via pinv(W_lm)")
    logger.info("=" * 70)

    # pinv(W_lm) via Gram matrix: pinv = (W^T W)^{-1} W^T
    # W_lm is [V, d], so W_lm^T @ W_lm is [d, d]
    W_lm_cpu = W_lm.cpu().float()
    gram = W_lm_cpu.T @ W_lm_cpu  # [d, d]
    gram += args.ols_ridge * torch.eye(D_MODEL)  # regularize
    gram_inv = torch.linalg.inv(gram)  # [d, d]
    pinv_Wlm = gram_inv @ W_lm_cpu.T  # [d, V]
    logger.info("  pinv(W_lm) computed: %s", pinv_Wlm.shape)

    g_final_cpu = g_final.cpu().float()

    # Recover h_out for all (n, t) pairs
    # z[n,t] = W_lm @ (g_final * h_out[n,t] / rms(h_out[n,t]))
    # v = pinv(W_lm) @ z[n,t] = g_final * h_out[n,t] / rms(h_out[n,t])
    # u = v / g_final = h_out[n,t] / rms(h_out[n,t])  — unit-RMS direction
    # We need to recover the scale rms(h_out).
    # Estimate: alpha = dot(u, h_in) / dot(u, u)  (project h_in onto u direction)

    h_out_est = torch.zeros_like(h_out_all)  # [N, T, d]

    cos_list = []
    l2_list = []

    # Process in batches to avoid OOM on logits @ pinv
    for n_start in range(0, N, BS):
        n_end = min(n_start + BS, N)
        z_batch = logits_all[n_start:n_end]  # [bs, T, V]
        h_in_batch = h_in_all[n_start:n_end]  # [bs, T, d]
        h_out_batch = h_out_all[n_start:n_end]  # [bs, T, d]

        # v = z @ pinv_Wlm^T = z @ (pinv_Wlm.T)  — since pinv is [d, V]
        # z is [bs, T, V], pinv_Wlm.T is [V, d]
        v = z_batch @ pinv_Wlm.T  # [bs, T, d]

        # u = v / g_final (element-wise)
        u = v / (g_final_cpu.unsqueeze(0).unsqueeze(0) + 1e-12)  # [bs, T, d]

        # Recover scale: alpha = dot(u, h_in) / dot(u, u)
        # More robust: use the known relationship rms(h_out) and h_out = h_in + delta
        uu = (u * u).sum(dim=-1, keepdim=True)  # [bs, T, 1]
        uh = (u * h_in_batch).sum(dim=-1, keepdim=True)  # [bs, T, 1]
        alpha = uh / (uu + 1e-12)  # [bs, T, 1]
        alpha = alpha.clamp(min=0.1)  # safety floor

        h_out_est_batch = alpha * u  # [bs, T, d]
        h_out_est[n_start:n_end] = h_out_est_batch

        # Per-sample cosine with true h_out
        for i in range(h_out_est_batch.shape[0]):
            for t in range(h_out_est_batch.shape[1]):
                c = flat_cosine(h_out_est_batch[i, t], h_out_batch[i, t])
                cos_list.append(c)
                l2 = (h_out_est_batch[i, t] - h_out_batch[i, t]).norm().item()
                l2_list.append(l2)

    mean_cos = sum(cos_list) / len(cos_list)
    median_cos = sorted(cos_list)[len(cos_list) // 2]
    mean_l2 = sum(l2_list) / len(l2_list)
    # Also compute global cosine (flatten all)
    global_cos = flat_cosine(h_out_est, h_out_all)

    results["step0_h_out_recovery"] = {
        "mean_per_vector_cosine": mean_cos,
        "median_per_vector_cosine": median_cos,
        "global_cosine": global_cos,
        "mean_l2_error": mean_l2,
        "num_vectors": len(cos_list),
    }
    logger.info("  h_out recovery: mean_cos=%.4f, median_cos=%.4f, "
                "global_cos=%.4f, mean_l2=%.4f",
                mean_cos, median_cos, global_cos, mean_l2)

    # Free logits — no longer needed
    del logits_all, pinv_Wlm, gram, gram_inv, W_lm_cpu
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Oracle MLP recovery (validate the math)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 1: Oracle MLP recovery (known h_mid, known gate/up -> solve W_down)")
    logger.info("=" * 70)

    # Flatten data: M = N * T data points
    # x = RMSNorm(h_mid, g_mlp)  — input to MLP
    # r_mlp = h_out_true - h_mid_true  — MLP output residual
    # a = SiLU(W_gate @ x) * (W_up @ x)  — intermediate activations
    # W_down @ a = r_mlp  — linear system

    g_mlp_dev = g_mlp.to(device)
    W_gate_dev = W_gate_true.to(device)
    W_up_dev = W_up_true.to(device)
    W_down_dev = W_down_true.to(device)

    # Accumulate normal equations: AAt [d_ff, d_ff] and RAt [d, d_ff]
    # A is [d_ff, M] (activations), R is [d, M] (residuals)
    # W_down @ A = R  =>  W_down = R A^T (A A^T)^{-1}
    AAt = torch.zeros(D_FF, D_FF, dtype=torch.float64, device=device)
    RAt = torch.zeros(D_MODEL, D_FF, dtype=torch.float64, device=device)

    logger.info("  Accumulating normal equations for W_down OLS...")
    with torch.no_grad():
        for n_start in range(0, N, BS):
            n_end = min(n_start + BS, N)
            h_mid_b = h_mid_all[n_start:n_end].to(device)  # [bs, T, d]
            h_out_b = h_out_all[n_start:n_end].to(device)  # [bs, T, d]

            # Flatten to [bs*T, d]
            M_b = h_mid_b.shape[0] * h_mid_b.shape[1]
            h_mid_flat = h_mid_b.reshape(M_b, D_MODEL)
            h_out_flat = h_out_b.reshape(M_b, D_MODEL)

            # x = RMSNorm(h_mid, g_mlp)
            x = rms_norm(h_mid_flat, g_mlp_dev)  # [M_b, d]

            # activations
            gate_out = F.silu(x @ W_gate_dev.T)  # [M_b, d_ff]
            up_out = x @ W_up_dev.T              # [M_b, d_ff]
            a = gate_out * up_out                 # [M_b, d_ff]

            # residual
            r = (h_out_flat - h_mid_flat).float()  # [M_b, d]

            # Accumulate in float64
            a_64 = a.double()
            r_64 = r.double()
            AAt += a_64.T @ a_64  # [d_ff, d_ff]
            RAt += r_64.T @ a_64  # [d, d_ff]

            if (n_start // BS) % 20 == 0:
                logger.info("    Batch %d/%d", n_start // BS, (N + BS - 1) // BS)

    # Tikhonov regularization
    AAt += args.ols_ridge * torch.eye(D_FF, dtype=torch.float64, device=device)

    # Solve: W_down_rec = RAt @ inv(AAt) = solve(AAt^T, RAt^T)^T
    logger.info("  Solving OLS for W_down (%dx%d)...", D_MODEL, D_FF)
    # AAt is symmetric, use Cholesky
    try:
        L = torch.linalg.cholesky(AAt)
        W_down_rec = torch.linalg.solve_triangular(
            L.T,
            torch.linalg.solve_triangular(L, RAt.T, upper=False),
            upper=True,
        ).T.float()
    except RuntimeError:
        logger.warning("  Cholesky failed, falling back to torch.linalg.solve")
        W_down_rec = torch.linalg.solve(AAt.T, RAt.T).T.float()

    W_down_rec_cpu = W_down_rec.cpu()

    cos_wdown = flat_cosine(W_down_rec_cpu, W_down_true)
    cos_wdown_perrow = per_row_cosine(W_down_rec_cpu, W_down_true)
    frob_err = (W_down_rec_cpu - W_down_true).norm().item() / (W_down_true.norm().item() + 1e-12)

    results["step1_oracle_mlp"] = {
        "W_down_cosine": cos_wdown,
        "W_down_per_row_cosine": cos_wdown_perrow,
        "W_down_frob_error": frob_err,
    }
    logger.info("  Oracle W_down recovery: cos=%.6f, per_row_cos=%.6f, frob_err=%.6f",
                cos_wdown, cos_wdown_perrow, frob_err)

    del AAt, RAt, W_down_rec
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Semi-oracle MLP (known h_mid, blind W_gate/W_up/W_down via ALS)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 2: Semi-oracle MLP via ALS (known h_mid, recover gate/up/down)")
    logger.info("=" * 70)

    # Collect all data points on device
    # Flatten: M = N * T, x [M, d], r [M, d]
    # We subsample for ALS to fit in memory
    M_total = N * T
    M_als = min(M_total, 32768)  # cap at ~32k data points for ALS
    logger.info("  Total data points: %d, using %d for ALS", M_total, M_als)

    rng_als = torch.Generator().manual_seed(args.seed + 100)
    als_indices = torch.randperm(M_total, generator=rng_als)[:M_als]

    # Flatten h_mid and h_out
    h_mid_flat_all = h_mid_all.reshape(M_total, D_MODEL)
    h_out_flat_all = h_out_all.reshape(M_total, D_MODEL)

    X_als = h_mid_flat_all[als_indices].to(device)  # [M_als, d]
    R_als = (h_out_flat_all[als_indices] - h_mid_flat_all[als_indices]).to(device)  # [M_als, d]

    # x = RMSNorm(X_als, g_mlp)
    X_norm = rms_norm(X_als, g_mlp_dev)  # [M_als, d]

    # Initialize ALS
    torch.manual_seed(args.seed + 200)
    W_gate_als = torch.randn(D_FF, D_MODEL, device=device) * 0.01
    W_up_als = torch.randn(D_FF, D_MODEL, device=device) * 0.01
    W_down_als = torch.randn(D_MODEL, D_FF, device=device) * 0.01

    als_lr = args.als_gate_lr
    als_gate_steps = args.als_gate_steps
    ridge = args.ols_ridge

    logger.info("  ALS: %d iters, gate_lr=%.4f, gate_steps=%d, ridge=%.1e",
                args.als_iters, als_lr, als_gate_steps, ridge)

    als_history = []

    for als_iter in range(args.als_iters):
        iter_t0 = time.time()

        # ── Fix W_gate: compute SiLU activations ──
        with torch.no_grad():
            s = F.silu(X_norm @ W_gate_als.T)   # [M, d_ff]

        # ── Fix W_gate + s: solve for W_up via OLS ──
        # r = W_down @ (s * (W_up @ x)) = W_down @ diag(s_i) @ W_up @ x_i
        # This is bilinear in (W_down, W_up).
        # Strategy: first solve W_up assuming W_down = I (or current W_down),
        # then solve W_down given W_up.

        # Phase A: Solve W_up.
        # Target for W_up: b_i = pinv(W_down) @ r_i / s_i  (per-neuron division)
        # b = pinv(W_down) @ R^T  ->  b is [d_ff, M]
        # Then W_up @ x_i = b_i / s_i element-wise
        with torch.no_grad():
            # pinv(W_down) via Gram
            WdTWd = W_down_als.double().T @ W_down_als.double()  # [d_ff, d_ff]
            WdTWd += ridge * torch.eye(D_FF, dtype=torch.float64, device=device)
            WdTWd_inv = torch.linalg.inv(WdTWd)
            pinv_Wd = WdTWd_inv @ W_down_als.double().T  # [d_ff, d]

            b = (pinv_Wd @ R_als.double().T).T.float()  # [M, d_ff]

            # target for W_up: t_i = b_i / s_i
            s_safe = s.clone()
            s_safe[s_safe.abs() < 1e-6] = 1e-6  # avoid div by zero
            target_up = b / s_safe  # [M, d_ff]

            # Solve: W_up @ X_norm^T = target_up^T
            # W_up = target_up^T @ pinv(X_norm^T) = target_up^T @ X_norm @ inv(X_norm^T X_norm)
            XtX = X_norm.double().T @ X_norm.double()  # [d, d]
            XtX += ridge * torch.eye(D_MODEL, dtype=torch.float64, device=device)
            XtX_inv = torch.linalg.inv(XtX)

            W_up_als = (target_up.double().T @ X_norm.double() @ XtX_inv).float()  # [d_ff, d]

        # Phase B: Fix s and W_up, solve W_down via OLS (exact same as Step 1)
        with torch.no_grad():
            up_out = X_norm @ W_up_als.T  # [M, d_ff]
            a_als = s * up_out             # [M, d_ff]

            AAt2 = a_als.double().T @ a_als.double()  # [d_ff, d_ff]
            AAt2 += ridge * torch.eye(D_FF, dtype=torch.float64, device=device)
            RAt2 = R_als.double().T @ a_als.double()  # [d, d_ff]

            try:
                L2 = torch.linalg.cholesky(AAt2)
                W_down_als = torch.linalg.solve_triangular(
                    L2.T,
                    torch.linalg.solve_triangular(L2, RAt2.T, upper=False),
                    upper=True,
                ).T.float()
            except RuntimeError:
                W_down_als = torch.linalg.solve(AAt2.T, RAt2.T).T.float()

        # Phase C: Gradient descent on W_gate (the only nonlinear parameter)
        W_gate_param = W_gate_als.clone().detach().requires_grad_(True)
        gate_optimizer = torch.optim.Adam([W_gate_param], lr=als_lr)

        # Use a subset for gradient steps to speed things up
        M_grad = min(M_als, 8192)
        grad_idx = torch.randperm(M_als, generator=rng_als)[:M_grad]
        X_grad = X_norm[grad_idx]
        R_grad = R_als[grad_idx]

        for gs in range(als_gate_steps):
            s_g = F.silu(X_grad @ W_gate_param.T)
            up_g = X_grad @ W_up_als.T.detach()
            a_g = s_g * up_g
            r_pred = a_g @ W_down_als.T.detach()
            loss = F.mse_loss(r_pred, R_grad)
            gate_optimizer.zero_grad()
            loss.backward()
            gate_optimizer.step()

        W_gate_als = W_gate_param.detach()

        # ── Evaluate this iteration ──
        with torch.no_grad():
            s_eval = F.silu(X_norm @ W_gate_als.T)
            up_eval = X_norm @ W_up_als.T
            a_eval = s_eval * up_eval
            r_pred_eval = a_eval @ W_down_als.T
            recon_loss = F.mse_loss(r_pred_eval, R_als).item()

        cos_g = flat_cosine(W_gate_als.cpu(), W_gate_true)
        cos_u = flat_cosine(W_up_als.cpu(), W_up_true)
        cos_d = flat_cosine(W_down_als.cpu(), W_down_true)

        # Also compute aligned cosines (Hungarian neuron matching)
        # Build param dicts for alignment
        rec_dict = {
            f"{prefix}.mlp.gate_proj.weight": W_gate_als.cpu(),
            f"{prefix}.mlp.up_proj.weight": W_up_als.cpu(),
            f"{prefix}.mlp.down_proj.weight": W_down_als.cpu(),
        }
        tea_dict = {
            f"{prefix}.mlp.gate_proj.weight": W_gate_true,
            f"{prefix}.mlp.up_proj.weight": W_up_true,
            f"{prefix}.mlp.down_proj.weight": W_down_true,
        }
        aligned_dict = align_ffn_neurons(rec_dict, tea_dict, f"{prefix}")
        cos_g_a = flat_cosine(aligned_dict[f"{prefix}.mlp.gate_proj.weight"], W_gate_true)
        cos_u_a = flat_cosine(aligned_dict[f"{prefix}.mlp.up_proj.weight"], W_up_true)
        cos_d_a = flat_cosine(aligned_dict[f"{prefix}.mlp.down_proj.weight"], W_down_true)

        iter_dt = time.time() - iter_t0
        logger.info(
            "  ALS iter %2d/%d | loss=%.6f | raw cos: gate=%.4f up=%.4f down=%.4f "
            "| aligned: gate=%.4f up=%.4f down=%.4f | %.1fs",
            als_iter + 1, args.als_iters, recon_loss,
            cos_g, cos_u, cos_d,
            cos_g_a, cos_u_a, cos_d_a,
            iter_dt,
        )

        als_history.append({
            "iter": als_iter + 1,
            "recon_loss": recon_loss,
            "raw_cosine": {"gate": cos_g, "up": cos_u, "down": cos_d},
            "aligned_cosine": {"gate": cos_g_a, "up": cos_u_a, "down": cos_d_a},
        })

    results["step2_semi_oracle_mlp"] = {
        "final_recon_loss": als_history[-1]["recon_loss"] if als_history else -1,
        "final_raw_cosine": als_history[-1]["raw_cosine"] if als_history else {},
        "final_aligned_cosine": als_history[-1]["aligned_cosine"] if als_history else {},
        "als_history": als_history,
    }

    # Keep best ALS weights for Step 3
    W_gate_semi = W_gate_als.clone()
    W_up_semi = W_up_als.clone()
    W_down_semi = W_down_als.clone()

    del W_gate_als, W_up_als, W_down_als, X_norm, R_als, X_als
    del AAt2, RAt2, WdTWd, WdTWd_inv, pinv_Wd
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Blind MLP bootstrap (use estimated h_out, iterate h_mid estimate)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("STEP 3: Blind MLP bootstrap (h_out from logits, iterate h_mid)")
    logger.info("=" * 70)

    # Flatten estimated h_out and h_in
    h_in_flat = h_in_all.reshape(M_total, D_MODEL)
    h_out_est_flat = h_out_est.reshape(M_total, D_MODEL)

    # Subsample for blind ALS
    M_blind = min(M_total, 16384)
    blind_idx = torch.randperm(M_total, generator=rng_als)[:M_blind]

    X_blind_h_in = h_in_flat[blind_idx].to(device)       # [M_blind, d]
    H_out_est_bl = h_out_est_flat[blind_idx].to(device)   # [M_blind, d]

    # True values for evaluation
    H_mid_true_bl = h_mid_flat_all[blind_idx]
    H_out_true_bl = h_out_flat_all[blind_idx]

    # Initial guess: h_mid ~= h_in (assume attention delta is zero)
    H_mid_est = X_blind_h_in.clone()  # [M_blind, d]

    # Initialize from semi-oracle results (warm start)
    W_gate_bl = W_gate_semi.clone()
    W_up_bl = W_up_semi.clone()
    W_down_bl = W_down_semi.clone()

    blind_iters = max(args.als_iters // 2, 5)
    blind_history = []

    logger.info("  Blind bootstrap: %d outer iters, %d ALS gate steps each",
                blind_iters, als_gate_steps // 2)

    for outer in range(blind_iters):
        outer_t0 = time.time()

        # R_est = h_out_est - h_mid_est
        R_est = H_out_est_bl - H_mid_est  # [M_blind, d]

        # x = RMSNorm(h_mid_est, g_mlp)
        X_norm_bl = rms_norm(H_mid_est, g_mlp_dev)  # [M_blind, d]

        # Single ALS pass: fix gate -> solve up -> solve down -> grad gate
        with torch.no_grad():
            s_bl = F.silu(X_norm_bl @ W_gate_bl.T)  # [M_blind, d_ff]

            # Solve W_up
            WdTWd = W_down_bl.double().T @ W_down_bl.double()
            WdTWd += ridge * torch.eye(D_FF, dtype=torch.float64, device=device)
            pinv_Wd_bl = torch.linalg.inv(WdTWd) @ W_down_bl.double().T

            b_bl = (pinv_Wd_bl @ R_est.double().T).T.float()
            s_safe_bl = s_bl.clone()
            s_safe_bl[s_safe_bl.abs() < 1e-6] = 1e-6
            target_up_bl = b_bl / s_safe_bl

            XtX_bl = X_norm_bl.double().T @ X_norm_bl.double()
            XtX_bl += ridge * torch.eye(D_MODEL, dtype=torch.float64, device=device)
            W_up_bl = (target_up_bl.double().T @ X_norm_bl.double()
                       @ torch.linalg.inv(XtX_bl)).float()

            # Solve W_down
            up_bl = X_norm_bl @ W_up_bl.T
            a_bl = s_bl * up_bl
            AAt_bl = a_bl.double().T @ a_bl.double()
            AAt_bl += ridge * torch.eye(D_FF, dtype=torch.float64, device=device)
            RAt_bl = R_est.double().T @ a_bl.double()
            try:
                L_bl = torch.linalg.cholesky(AAt_bl)
                W_down_bl = torch.linalg.solve_triangular(
                    L_bl.T,
                    torch.linalg.solve_triangular(L_bl, RAt_bl.T, upper=False),
                    upper=True,
                ).T.float()
            except RuntimeError:
                W_down_bl = torch.linalg.solve(AAt_bl.T, RAt_bl.T).T.float()

        # Gradient steps on W_gate
        W_gate_p = W_gate_bl.clone().detach().requires_grad_(True)
        gate_opt = torch.optim.Adam([W_gate_p], lr=als_lr)
        M_grad_bl = min(M_blind, 4096)
        gi = torch.randperm(M_blind, generator=rng_als)[:M_grad_bl]
        X_g = X_norm_bl[gi]
        R_g = R_est[gi]
        for _ in range(als_gate_steps // 2):
            s_g = F.silu(X_g @ W_gate_p.T)
            up_g = X_g @ W_up_bl.T.detach()
            r_p = (s_g * up_g) @ W_down_bl.T.detach()
            loss = F.mse_loss(r_p, R_g)
            gate_opt.zero_grad()
            loss.backward()
            gate_opt.step()
        W_gate_bl = W_gate_p.detach()

        # Update h_mid estimate: h_mid = h_out_est - MLP(RMSNorm(h_mid, g_mlp))
        # Actually: h_out = h_mid + MLP(RMSNorm(h_mid)), so
        # h_mid = h_out - MLP(RMSNorm(h_mid))  — fixed-point iteration
        # Better: h_mid = h_in + attn_residual
        # We don't know attn_residual, but we can estimate it as:
        # attn_residual = h_out_est - h_in - MLP(RMSNorm(h_mid_est, g_mlp))
        with torch.no_grad():
            s_upd = F.silu(X_norm_bl @ W_gate_bl.T)
            up_upd = X_norm_bl @ W_up_bl.T
            mlp_out = (s_upd * up_upd) @ W_down_bl.T  # [M_blind, d]
            # h_out_est = h_in + attn_res + mlp_out
            # attn_res = h_out_est - h_in - mlp_out
            attn_res_est = H_out_est_bl - X_blind_h_in - mlp_out
            H_mid_est = X_blind_h_in + attn_res_est  # = h_out_est - mlp_out

        # Evaluate
        with torch.no_grad():
            recon_loss_bl = F.mse_loss(
                mlp_out, H_out_est_bl - H_mid_est
            ).item()

        cos_g_bl = flat_cosine(W_gate_bl.cpu(), W_gate_true)
        cos_u_bl = flat_cosine(W_up_bl.cpu(), W_up_true)
        cos_d_bl = flat_cosine(W_down_bl.cpu(), W_down_true)

        # Aligned
        rec_bl = {
            f"{prefix}.mlp.gate_proj.weight": W_gate_bl.cpu(),
            f"{prefix}.mlp.up_proj.weight": W_up_bl.cpu(),
            f"{prefix}.mlp.down_proj.weight": W_down_bl.cpu(),
        }
        aligned_bl = align_ffn_neurons(rec_bl, tea_dict, f"{prefix}")
        cos_g_ba = flat_cosine(aligned_bl[f"{prefix}.mlp.gate_proj.weight"], W_gate_true)
        cos_u_ba = flat_cosine(aligned_bl[f"{prefix}.mlp.up_proj.weight"], W_up_true)
        cos_d_ba = flat_cosine(aligned_bl[f"{prefix}.mlp.down_proj.weight"], W_down_true)

        # h_mid estimate quality
        h_mid_cos = flat_cosine(H_mid_est.cpu(), H_mid_true_bl)

        outer_dt = time.time() - outer_t0
        logger.info(
            "  Blind iter %2d/%d | loss=%.6f | h_mid_cos=%.4f | "
            "raw: g=%.4f u=%.4f d=%.4f | aligned: g=%.4f u=%.4f d=%.4f | %.1fs",
            outer + 1, blind_iters, recon_loss_bl, h_mid_cos,
            cos_g_bl, cos_u_bl, cos_d_bl,
            cos_g_ba, cos_u_ba, cos_d_ba,
            outer_dt,
        )

        blind_history.append({
            "iter": outer + 1,
            "recon_loss": recon_loss_bl,
            "h_mid_cosine": h_mid_cos,
            "raw_cosine": {"gate": cos_g_bl, "up": cos_u_bl, "down": cos_d_bl},
            "aligned_cosine": {"gate": cos_g_ba, "up": cos_u_ba, "down": cos_d_ba},
        })

    results["step3_blind_mlp"] = {
        "final_recon_loss": blind_history[-1]["recon_loss"] if blind_history else -1,
        "final_h_mid_cosine": blind_history[-1]["h_mid_cosine"] if blind_history else -1,
        "final_raw_cosine": blind_history[-1]["raw_cosine"] if blind_history else {},
        "final_aligned_cosine": blind_history[-1]["aligned_cosine"] if blind_history else {},
        "blind_history": blind_history,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    results["elapsed_seconds"] = round(elapsed, 1)
    results["config"] = vars(args)

    # Save results
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ALGEBRAIC BLOCK RECOVERY SUMMARY")
    print("=" * 80)

    print(f"\nStep 0 — h_out recovery from logits:")
    s0 = results["step0_h_out_recovery"]
    print(f"  Mean per-vector cosine: {s0['mean_per_vector_cosine']:.4f}")
    print(f"  Median per-vector cosine: {s0['median_per_vector_cosine']:.4f}")
    print(f"  Global cosine: {s0['global_cosine']:.4f}")

    print(f"\nStep 1 — Oracle MLP (W_down OLS with true activations):")
    s1 = results["step1_oracle_mlp"]
    print(f"  W_down cosine: {s1['W_down_cosine']:.6f}")
    print(f"  W_down per-row cosine: {s1['W_down_per_row_cosine']:.6f}")

    print(f"\nStep 2 — Semi-oracle MLP (ALS, true h_mid):")
    s2 = results["step2_semi_oracle_mlp"]
    if s2["final_aligned_cosine"]:
        print(f"  Aligned cosines: gate={s2['final_aligned_cosine']['gate']:.4f}, "
              f"up={s2['final_aligned_cosine']['up']:.4f}, "
              f"down={s2['final_aligned_cosine']['down']:.4f}")
    if s2["final_raw_cosine"]:
        print(f"  Raw cosines:     gate={s2['final_raw_cosine']['gate']:.4f}, "
              f"up={s2['final_raw_cosine']['up']:.4f}, "
              f"down={s2['final_raw_cosine']['down']:.4f}")

    print(f"\nStep 3 — Blind MLP bootstrap:")
    s3 = results["step3_blind_mlp"]
    if s3["final_aligned_cosine"]:
        print(f"  Aligned cosines: gate={s3['final_aligned_cosine']['gate']:.4f}, "
              f"up={s3['final_aligned_cosine']['up']:.4f}, "
              f"down={s3['final_aligned_cosine']['down']:.4f}")
    if s3.get("final_h_mid_cosine"):
        print(f"  h_mid estimate cosine: {s3['final_h_mid_cosine']:.4f}")

    # Best result across all steps
    best_cos = -1.0
    best_label = ""
    for key, val in [
        ("Step 1 W_down", s1["W_down_cosine"]),
    ]:
        if val > best_cos:
            best_cos = val
            best_label = key
    for step_key, step_data in [("Step 2", s2), ("Step 3", s3)]:
        for mat, cos_val in step_data.get("final_aligned_cosine", {}).items():
            label = f"{step_key} {mat}"
            if cos_val > best_cos:
                best_cos = cos_val
                best_label = label

    print(f"\nBest single-matrix cosine: {best_cos:.4f} ({best_label})")
    if best_cos > 0.5:
        print("  ** BREAKTHROUGH: cos > 0.5 achieved! **")
    elif best_cos > 0.1:
        print("  Above random (0.0), but below breakthrough threshold (0.5).")
    else:
        print("  Near random — structured regression did not converge.")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to: {out_dir / 'results.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
