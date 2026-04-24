#!/usr/bin/env python3
"""Gauge-invariant evaluation of parameter recovery quality.

Computes gauge-aligned cosine similarity for attention matrices, accounting
for the V/O GL(d_head) symmetry and Q/K RoPE (C*)^{d_head/2} symmetry.

The key insight: raw cosine similarity can be misleadingly low if the
recovered model found an equivalent parameterization in a different gauge.
By solving the Procrustes problem (or full GL alignment), we measure the
true recovery quality on the quotient manifold.

For V/O alignment per KV group:
  1. Compute M = V_student @ V_teacher^T   (shape [d_head, d_head])
  2. SVD: M = U Sigma V^T
  3. Procrustes S* = U V^T (orthogonal), or full GL: S = V_student @ pinv(V_teacher)
  4. Aligned V: S* @ V_teacher, aligned O: O_teacher @ S*^{-1}
  5. Measure cosine after alignment

For Q/K RoPE alignment per KV group and frequency pair j:
  1. Extract 2x2 blocks from Q and K at positions (2j, 2j+1)
  2. Find optimal C* element (a*I + b*J) that aligns Q and K jointly
  3. Measure cosine after alignment

Also computes MLP cosines as a sanity check (should be unchanged).

Usage:
    python scripts/eval_gauge_invariant.py \
        --results_dir results/v2_random_s42 \
        --model_name Qwen/Qwen2.5-0.5B \
        --block_indices 22,23
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors (flattened)."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    if a_flat.shape != b_flat.shape:
        return float("nan")
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def procrustes_align_vo(
    V_student: torch.Tensor,  # [d_head, hidden_size]
    V_teacher: torch.Tensor,  # [d_head, hidden_size]
    O_student_heads: list,    # list of [hidden_size, d_head] per head in group
    O_teacher_heads: list,    # list of [hidden_size, d_head] per head in group
) -> dict:
    """Solve Procrustes problem for V/O gauge alignment.

    Finds S* = argmin_S ||V_student - S @ V_teacher||_F + ||O_student - O_teacher @ S^{-1}||_F
    using SVD-based orthogonal Procrustes.

    Also computes full GL(d_head) alignment: S = V_student @ pinv(V_teacher).

    Returns dict with raw and aligned cosines for V and O.
    """
    d_head = V_student.shape[0]

    # Raw cosines
    raw_v_cos = cosine_sim(V_student, V_teacher)
    raw_o_cos_list = [cosine_sim(Os, Ot) for Os, Ot in zip(O_student_heads, O_teacher_heads)]
    raw_o_cos = float(np.mean(raw_o_cos_list)) if raw_o_cos_list else float("nan")

    # --- Orthogonal Procrustes (S in O(d_head)) ---
    # Find S* = argmin ||V_student - S V_teacher||_F
    # M = V_student @ V_teacher^T = U Sigma V^T, then S* = U V^T
    M = V_student @ V_teacher.T  # [d_head, d_head]
    try:
        U, S_vals, Vh = torch.linalg.svd(M)
        S_orth = U @ Vh  # [d_head, d_head]

        # Fix reflection
        if torch.det(S_orth) < 0:
            U[:, -1] *= -1
            S_orth = U @ Vh

        V_teacher_aligned = S_orth @ V_teacher
        orth_v_cos = cosine_sim(V_student, V_teacher_aligned)

        # O alignment: O_aligned = O_teacher @ S_orth^T (since S^{-1} = S^T for orthogonal)
        S_orth_inv = S_orth.T
        orth_o_cos_list = []
        for Os, Ot in zip(O_student_heads, O_teacher_heads):
            Ot_aligned = Ot @ S_orth_inv
            orth_o_cos_list.append(cosine_sim(Os, Ot_aligned))
        orth_o_cos = float(np.mean(orth_o_cos_list)) if orth_o_cos_list else float("nan")

    except Exception as e:
        logger.warning("Orthogonal Procrustes failed: %s", e)
        orth_v_cos = raw_v_cos
        orth_o_cos = raw_o_cos
        S_vals = torch.zeros(d_head)

    # --- Full GL(d_head) alignment ---
    # S_gl = V_student @ pinv(V_teacher)
    try:
        V_teacher_pinv = torch.linalg.pinv(V_teacher)  # [hidden, d_head]
        S_gl = V_student @ V_teacher_pinv  # [d_head, d_head]

        V_teacher_gl = S_gl @ V_teacher
        gl_v_cos = cosine_sim(V_student, V_teacher_gl)

        # O alignment: O_aligned = O_teacher @ S_gl^{-1}
        try:
            S_gl_inv = torch.linalg.inv(S_gl)
        except Exception:
            S_gl_inv = torch.linalg.pinv(S_gl)

        gl_o_cos_list = []
        for Os, Ot in zip(O_student_heads, O_teacher_heads):
            Ot_aligned = Ot @ S_gl_inv
            gl_o_cos_list.append(cosine_sim(Os, Ot_aligned))
        gl_o_cos = float(np.mean(gl_o_cos_list)) if gl_o_cos_list else float("nan")

    except Exception as e:
        logger.warning("GL alignment failed: %s", e)
        gl_v_cos = raw_v_cos
        gl_o_cos = raw_o_cos

    return {
        "raw_v_cosine": round(raw_v_cos, 6),
        "raw_o_cosine": round(raw_o_cos, 6),
        "orth_aligned_v_cosine": round(orth_v_cos, 6),
        "orth_aligned_o_cosine": round(orth_o_cos, 6),
        "gl_aligned_v_cosine": round(gl_v_cos, 6),
        "gl_aligned_o_cosine": round(gl_o_cos, 6),
        "top_singular_values": S_vals[:5].tolist(),
    }


def align_qk_rope(
    Q_student_block: torch.Tensor,  # [2, hidden_size] (rows 2j, 2j+1)
    Q_teacher_block: torch.Tensor,
    K_student_block: torch.Tensor,  # [2, hidden_size]
    K_teacher_block: torch.Tensor,
) -> dict:
    """Align Q/K blocks using RoPE-commuting (C*) transformation.

    Find optimal (a, b) such that R = aI_2 + bJ aligns:
      Q_student ~ R @ Q_teacher and K_student ~ R^{-1} @ K_teacher

    For the Q part only (simpler and more numerically stable):
      min_{a,b} ||Q_s - (aI + bJ) Q_t||^2
      This is a linear least squares problem in (a, b).
    """
    # Raw cosine
    raw_q_cos = cosine_sim(Q_student_block, Q_teacher_block)
    raw_k_cos = cosine_sim(K_student_block, K_teacher_block)

    # J matrix: [[0, -1], [1, 0]]
    J = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device=Q_student_block.device)

    try:
        # For Q alignment: Q_s = a * Q_t + b * J @ Q_t
        # Flatten: Q_s_flat = a * Q_t_flat + b * (J @ Q_t)_flat
        Q_t = Q_teacher_block.float()  # [2, hidden]
        Q_s = Q_student_block.float()  # [2, hidden]
        JQ_t = J @ Q_t  # [2, hidden]

        # Stack into regression: [Q_t_flat, JQ_t_flat] @ [a, b]^T = Q_s_flat
        A = torch.stack([Q_t.flatten(), JQ_t.flatten()], dim=1)  # [2*hidden, 2]
        rhs = Q_s.flatten()  # [2*hidden]

        # Solve least squares
        result = torch.linalg.lstsq(A, rhs.unsqueeze(1))
        ab = result.solution.squeeze()  # [2]
        a_val, b_val = ab[0].item(), ab[1].item()

        # R = aI + bJ
        R = a_val * torch.eye(2, device=Q_s.device) + b_val * J

        Q_t_aligned = R @ Q_t
        aligned_q_cos = cosine_sim(Q_s, Q_t_aligned)

        # R^{-1} = (a I - b J) / (a^2 + b^2)
        det_R = a_val**2 + b_val**2
        if det_R > 1e-12:
            R_inv = (a_val * torch.eye(2, device=Q_s.device) - b_val * J) / det_R
            K_t = K_teacher_block.float()
            K_s = K_student_block.float()
            # K uses inverse: K_student ~ R^{-T} K_teacher (since Q R K^T = Q (R K^T))
            # Actually for Q/K gauge: Q -> R Q, K -> R^{-T} K
            K_t_aligned = R_inv.T @ K_t
            aligned_k_cos = cosine_sim(K_s, K_t_aligned)
        else:
            aligned_k_cos = raw_k_cos

    except Exception as e:
        logger.warning("Q/K RoPE alignment failed: %s", e)
        aligned_q_cos = raw_q_cos
        aligned_k_cos = raw_k_cos
        a_val, b_val = 1.0, 0.0

    return {
        "raw_q_cosine": round(raw_q_cos, 6),
        "raw_k_cosine": round(raw_k_cos, 6),
        "aligned_q_cosine": round(aligned_q_cos, 6),
        "aligned_k_cosine": round(aligned_k_cos, 6),
        "a": round(a_val, 6),
        "b": round(b_val, 6),
    }


def evaluate_block(
    student_params: dict,
    teacher_params: dict,
    block_prefix: str,
    num_heads: int,
    num_kv_heads: int,
    d_head: int,
) -> dict:
    """Evaluate gauge-invariant distance for one block.

    Returns dict with raw and aligned cosines for Q, K, V, O, and MLP matrices.
    """
    hidden_size = num_heads * d_head
    heads_per_group = num_heads // num_kv_heads

    q_key = block_prefix + "self_attn.q_proj.weight"
    k_key = block_prefix + "self_attn.k_proj.weight"
    v_key = block_prefix + "self_attn.v_proj.weight"
    o_key = block_prefix + "self_attn.o_proj.weight"

    result = {"block_prefix": block_prefix}

    # Check all keys exist
    for key_name, key in [("q_proj", q_key), ("k_proj", k_key),
                          ("v_proj", v_key), ("o_proj", o_key)]:
        if key not in student_params or key not in teacher_params:
            logger.warning("Missing %s in params for %s", key_name, block_prefix)
            return result

    W_Q_s = student_params[q_key].float()  # [num_heads * d_head, hidden]
    W_K_s = student_params[k_key].float()  # [kv_heads * d_head, hidden]
    W_V_s = student_params[v_key].float()  # [kv_heads * d_head, hidden]
    W_O_s = student_params[o_key].float()  # [hidden, num_heads * d_head]

    W_Q_t = teacher_params[q_key].float()
    W_K_t = teacher_params[k_key].float()
    W_V_t = teacher_params[v_key].float()
    W_O_t = teacher_params[o_key].float()

    # --- V/O gauge alignment per KV group ---
    vo_results = []
    for g in range(num_kv_heads):
        v_start = g * d_head
        v_end = v_start + d_head

        V_s_g = W_V_s[v_start:v_end, :]  # [d_head, hidden]
        V_t_g = W_V_t[v_start:v_end, :]

        head_start = g * heads_per_group
        head_end = head_start + heads_per_group

        O_s_heads = []
        O_t_heads = []
        for h in range(head_start, head_end):
            o_col_start = h * d_head
            o_col_end = o_col_start + d_head
            O_s_heads.append(W_O_s[:, o_col_start:o_col_end])  # [hidden, d_head]
            O_t_heads.append(W_O_t[:, o_col_start:o_col_end])

        vo_res = procrustes_align_vo(V_s_g, V_t_g, O_s_heads, O_t_heads)
        vo_res["kv_group"] = g
        vo_results.append(vo_res)

    result["vo_alignment"] = vo_results

    # Aggregate V/O results
    raw_v = float(np.mean([r["raw_v_cosine"] for r in vo_results]))
    raw_o = float(np.mean([r["raw_o_cosine"] for r in vo_results]))
    orth_v = float(np.mean([r["orth_aligned_v_cosine"] for r in vo_results]))
    orth_o = float(np.mean([r["orth_aligned_o_cosine"] for r in vo_results]))
    gl_v = float(np.mean([r["gl_aligned_v_cosine"] for r in vo_results]))
    gl_o = float(np.mean([r["gl_aligned_o_cosine"] for r in vo_results]))

    result["v_proj"] = {
        "raw_cosine": round(raw_v, 6),
        "orth_aligned_cosine": round(orth_v, 6),
        "gl_aligned_cosine": round(gl_v, 6),
    }
    result["o_proj"] = {
        "raw_cosine": round(raw_o, 6),
        "orth_aligned_cosine": round(orth_o, 6),
        "gl_aligned_cosine": round(gl_o, 6),
    }

    # --- Q/K RoPE alignment per KV group ---
    half_d = d_head // 2
    qk_results = []
    for g in range(num_kv_heads):
        k_row_base = g * d_head
        head_start = g * heads_per_group

        group_q_raws = []
        group_k_raws = []
        group_q_aligned = []
        group_k_aligned = []

        for j in range(half_d):
            # Average over Q heads in the group
            h_q_raws = []
            h_q_aligned = []
            for h in range(head_start, head_start + heads_per_group):
                q_row_2j = h * d_head + 2 * j
                Q_s_block = W_Q_s[q_row_2j:q_row_2j + 2, :]
                Q_t_block = W_Q_t[q_row_2j:q_row_2j + 2, :]

                K_s_block = W_K_s[k_row_base + 2 * j:k_row_base + 2 * j + 2, :]
                K_t_block = W_K_t[k_row_base + 2 * j:k_row_base + 2 * j + 2, :]

                qk_res = align_qk_rope(Q_s_block, Q_t_block, K_s_block, K_t_block)
                h_q_raws.append(qk_res["raw_q_cosine"])
                h_q_aligned.append(qk_res["aligned_q_cosine"])

            group_q_raws.extend(h_q_raws)
            group_q_aligned.extend(h_q_aligned)

            # K is shared across the group
            K_s_block = W_K_s[k_row_base + 2 * j:k_row_base + 2 * j + 2, :]
            K_t_block = W_K_t[k_row_base + 2 * j:k_row_base + 2 * j + 2, :]
            group_k_raws.append(cosine_sim(K_s_block, K_t_block))
            # Use the first head's alignment for K (they share the same K)
            Q_s_0 = W_Q_s[head_start * d_head + 2 * j:head_start * d_head + 2 * j + 2, :]
            Q_t_0 = W_Q_t[head_start * d_head + 2 * j:head_start * d_head + 2 * j + 2, :]
            qk_res_0 = align_qk_rope(Q_s_0, Q_t_0, K_s_block, K_t_block)
            group_k_aligned.append(qk_res_0["aligned_k_cosine"])

        qk_results.append({
            "kv_group": g,
            "mean_raw_q_cosine": round(float(np.mean(group_q_raws)), 6),
            "mean_aligned_q_cosine": round(float(np.mean(group_q_aligned)), 6),
            "mean_raw_k_cosine": round(float(np.mean(group_k_raws)), 6),
            "mean_aligned_k_cosine": round(float(np.mean(group_k_aligned)), 6),
        })

    result["qk_rope_alignment"] = qk_results

    # Aggregate Q/K results
    raw_q = float(np.mean([r["mean_raw_q_cosine"] for r in qk_results]))
    aligned_q = float(np.mean([r["mean_aligned_q_cosine"] for r in qk_results]))
    raw_k = float(np.mean([r["mean_raw_k_cosine"] for r in qk_results]))
    aligned_k = float(np.mean([r["mean_aligned_k_cosine"] for r in qk_results]))

    result["q_proj"] = {
        "raw_cosine": round(raw_q, 6),
        "rope_aligned_cosine": round(aligned_q, 6),
    }
    result["k_proj"] = {
        "raw_cosine": round(raw_k, 6),
        "rope_aligned_cosine": round(aligned_k, 6),
    }

    # --- MLP cosines (sanity check, should be unchanged) ---
    mlp_keys = [
        ("gate_proj.weight", "mlp.gate_proj.weight"),
        ("up_proj.weight", "mlp.up_proj.weight"),
        ("down_proj.weight", "mlp.down_proj.weight"),
    ]
    mlp_cosines = {}
    for short_name, suffix in mlp_keys:
        full_key = block_prefix + suffix
        if full_key in student_params and full_key in teacher_params:
            mlp_cosines[short_name] = round(
                cosine_sim(student_params[full_key], teacher_params[full_key]), 6
            )
    result["mlp"] = mlp_cosines

    # --- RMSNorm cosines (also sanity check) ---
    norm_keys = [
        ("input_layernorm.weight", "input_layernorm.weight"),
        ("post_attention_layernorm.weight", "post_attention_layernorm.weight"),
    ]
    norm_cosines = {}
    for short_name, suffix in norm_keys:
        full_key = block_prefix + suffix
        if full_key in student_params and full_key in teacher_params:
            norm_cosines[short_name] = round(
                cosine_sim(student_params[full_key], teacher_params[full_key]), 6
            )
    result["norms"] = norm_cosines

    return result


def find_recovered_model(results_dir: Path) -> Path:
    """Find the recovered model directory within a results directory.

    Searches common patterns:
      - {results_dir}/oracle/init_0/recovered_model/
      - {results_dir}/recovered_model/
      - {results_dir}/ (if it contains a config.json)
    """
    # Pattern 1: SPSI output (regime_oracle or oracle)
    for regime in ["regime_oracle", "oracle", "regime_pure_logits", "pure_logits"]:
        for init_idx in range(5):
            candidate = results_dir / regime / f"init_{init_idx}" / "recovered_model"
            if candidate.exists() and (candidate / "config.json").exists():
                return candidate

    # Pattern 2: direct recovered model
    candidate = results_dir / "recovered_model"
    if candidate.exists() and (candidate / "config.json").exists():
        return candidate

    # Pattern 3: results_dir itself is the model
    if (results_dir / "config.json").exists():
        return results_dir

    raise FileNotFoundError(
        f"Could not find recovered model in {results_dir}. "
        f"Expected oracle/init_0/recovered_model/ or similar."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gauge-invariant evaluation of parameter recovery",
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to a v2_* or v4_* results directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Teacher model name (default: Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--block_indices", type=str, default="22,23",
                        help="Comma-separated block indices to evaluate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results_dir/gauge_invariant/)")
    args = parser.parse_args()

    setup_logging()

    results_dir = Path(args.results_dir)
    block_indices = [int(x.strip()) for x in args.block_indices.split(",")]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "gauge_invariant"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load teacher model
    logger.info("Loading teacher model: %s", args.model_name)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    teacher_params = {n: p.data.cpu().clone() for n, p in teacher.named_parameters()}

    num_heads = teacher.config.num_attention_heads
    num_kv_heads = getattr(teacher.config, "num_key_value_heads", num_heads)
    hidden_size = teacher.config.hidden_size
    d_head = hidden_size // num_heads
    num_layers = teacher.config.num_hidden_layers

    logger.info("Architecture: H=%d, H_kv=%d, d_head=%d, hidden=%d, layers=%d",
                num_heads, num_kv_heads, d_head, hidden_size, num_layers)

    del teacher
    torch.cuda.empty_cache()

    # Load recovered (student) model
    logger.info("Finding recovered model in: %s", results_dir)
    recovered_path = find_recovered_model(results_dir)
    logger.info("Loading recovered model from: %s", recovered_path)

    student = AutoModelForCausalLM.from_pretrained(
        str(recovered_path), torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    student_params = {n: p.data.cpu().clone() for n, p in student.named_parameters()}
    del student
    torch.cuda.empty_cache()

    # Evaluate each block
    all_results = {
        "model_name": args.model_name,
        "results_dir": str(results_dir),
        "recovered_model_path": str(recovered_path),
        "architecture": {
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "d_head": d_head,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "vo_gauge_dims_per_block": num_kv_heads * d_head * d_head,
            "qk_rope_gauge_dims_per_block": num_kv_heads * d_head,
        },
        "blocks": {},
    }

    for block_idx in block_indices:
        logger.info("\n=== Evaluating Block %d ===", block_idx)

        # Find block prefix
        block_prefix = f"model.layers.{block_idx}."

        block_result = evaluate_block(
            student_params, teacher_params,
            block_prefix, num_heads, num_kv_heads, d_head,
        )

        all_results["blocks"][f"block_{block_idx}"] = block_result

        # Log summary
        logger.info("Block %d summary:", block_idx)
        if "v_proj" in block_result:
            vp = block_result["v_proj"]
            logger.info("  V: raw=%.4f, orth=%.4f, GL=%.4f",
                        vp["raw_cosine"], vp["orth_aligned_cosine"], vp["gl_aligned_cosine"])
        if "o_proj" in block_result:
            op = block_result["o_proj"]
            logger.info("  O: raw=%.4f, orth=%.4f, GL=%.4f",
                        op["raw_cosine"], op["orth_aligned_cosine"], op["gl_aligned_cosine"])
        if "q_proj" in block_result:
            qp = block_result["q_proj"]
            logger.info("  Q: raw=%.4f, rope_aligned=%.4f",
                        qp["raw_cosine"], qp["rope_aligned_cosine"])
        if "k_proj" in block_result:
            kp = block_result["k_proj"]
            logger.info("  K: raw=%.4f, rope_aligned=%.4f",
                        kp["raw_cosine"], kp["rope_aligned_cosine"])
        if "mlp" in block_result:
            logger.info("  MLP: %s", block_result["mlp"])

    # Interpretation
    logger.info("\n=== INTERPRETATION ===")
    for block_key, block_data in all_results["blocks"].items():
        if "v_proj" not in block_data:
            continue
        raw_v = block_data["v_proj"]["raw_cosine"]
        aligned_v = block_data["v_proj"]["gl_aligned_cosine"]
        raw_o = block_data["o_proj"]["raw_cosine"]
        aligned_o = block_data["o_proj"]["gl_aligned_cosine"]

        if aligned_v > raw_v + 0.05 or aligned_o > raw_o + 0.05:
            logger.info(
                "%s: GAUGE MASKING DETECTED. V: raw=%.4f -> aligned=%.4f (+%.4f), "
                "O: raw=%.4f -> aligned=%.4f (+%.4f). "
                "Recovery is REAL but masked by gauge freedom.",
                block_key, raw_v, aligned_v, aligned_v - raw_v,
                raw_o, aligned_o, aligned_o - raw_o,
            )
        else:
            logger.info(
                "%s: No significant gauge masking. V: raw=%.4f -> aligned=%.4f, "
                "O: raw=%.4f -> aligned=%.4f. "
                "Raw cosine already reflects true recovery quality.",
                block_key, raw_v, aligned_v, raw_o, aligned_o,
            )

    # Save results
    output_file = output_dir / "gauge_invariant_eval.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nResults saved to %s", output_file)

    # Print compact summary table
    print("\n" + "=" * 90)
    print("GAUGE-INVARIANT EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Block':<10} {'Matrix':<10} {'Raw Cos':<10} {'Aligned Cos':<14} {'Delta':<10} {'Method':<12}")
    print("-" * 90)
    for block_key, block_data in all_results["blocks"].items():
        for matrix in ["v_proj", "o_proj", "q_proj", "k_proj"]:
            if matrix not in block_data:
                continue
            md = block_data[matrix]
            raw = md.get("raw_cosine", float("nan"))
            if matrix in ["v_proj", "o_proj"]:
                aligned = md.get("gl_aligned_cosine", raw)
                method = "GL(d_head)"
            else:
                aligned = md.get("rope_aligned_cosine", raw)
                method = "RoPE C*"
            delta = aligned - raw
            print(f"{block_key:<10} {matrix:<10} {raw:<10.4f} {aligned:<14.4f} {delta:<+10.4f} {method:<12}")

        # MLP
        if "mlp" in block_data:
            for mlp_name, mlp_cos in block_data["mlp"].items():
                print(f"{block_key:<10} {mlp_name:<10} {mlp_cos:<10.4f} {'(no gauge)':<14} {'---':<10} {'---':<12}")
    print("=" * 90)


if __name__ == "__main__":
    main()
