#!/usr/bin/env python3
"""
Functional equivalence test: does the recovered W_lm actually work?

Setup
-----
Teacher: Qwen2.5-0.5B (pretrained).
Student: same architecture, prefix copied from teacher (ORACLE PREFIX so we
isolate the W_lm recovery question from prefix-mismatch error), but with
lm_head replaced by one of the following:

  1. ORACLE   : teacher.lm_head        (sanity: KL ≈ 0)
  2. RANDOM   : freshly random-init    (worst case: high KL)
  3. CARLINI  : Carlini SVD recovered  (subspace 0.9999, but rotated)
  4. CP_A_LIFT: CP factor A lifted via W_eff^T @ A  (moments attack naive)
  5. COMBINED : Carlini W_eff + CP-based rotation (path A proposed)

We compute KL(teacher || student) and token-level top-1 match on a held-out
WikiText-103 slice. If the COMBINED variant delivers low KL, we have a
functionally equivalent lm_head recovery --- the real black-box parameter
theft test.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python scripts/functional_kl_eval.py \
        --cp_factors_pt results/v5_attack_moments_v2/cp_real_factors.pt \
        --output_dir results/v6_functional_kl_qwen \
        --num_queries 1024 \
        --seq_len 128
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--cp_factors_pt", required=True,
                   help="Path to cp_real_factors.pt (dict with A, B, C, lambdas).")
    p.add_argument("--num_queries", type=int, default=1024)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--carlini_queries", type=int, default=8192,
                   help="Queries for Carlini SVD (to recover W_eff).")
    p.add_argument("--carlini_seq_len", type=int, default=128)
    p.add_argument("--output_dir", default="results/v6_functional_kl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool", default="mean",
                   choices=["mean", "last", "sum"])
    p.add_argument("--eval_positions", type=int, default=16,
                   help="Last-N positions to evaluate KL on per query.")
    return p.parse_args()


@torch.no_grad()
def run_carlini_svd(model, vocab_size, hidden_size, num_queries,
                    seq_len, device, batch_size, seed):
    """Collect random-token logits and extract W_eff via SVD."""
    logger.info("Carlini SVD: collecting %d queries @ seq_len=%d",
                num_queries, seq_len)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    ys = []
    done = 0
    pbar = tqdm(total=num_queries, desc="Carlini", unit="q")
    while done < num_queries:
        bs = min(batch_size, num_queries - done)
        ids = torch.randint(0, vocab_size, (bs, seq_len), generator=rng).to(device)
        out = model(input_ids=ids)
        ys.append(out.logits[:, -1, :].float().cpu())
        done += bs
        pbar.update(bs)
    pbar.close()
    Y = torch.cat(ys, dim=0)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    logger.info("Running SVD on (%d, %d) matrix ...", *Yc.shape)
    _, S, Vh = torch.linalg.svd(Yc, full_matrices=False)
    # Detect rank via gap
    S_np = S.numpy()
    gap = S_np[:-1] / np.maximum(S_np[1:], 1e-30)
    lo = max(0, hidden_size - 50)
    hi = min(len(gap), hidden_size + 50)
    d_hat = int(lo + np.argmax(gap[lo:hi]))
    logger.info("d_hat = %d (vs true d=%d), gap=%.2f",
                d_hat, hidden_size, gap[d_hat])
    W_eff = Vh[:d_hat, :].float()  # (d_hat, V)
    return {"W_eff": W_eff, "S": S, "d_hat": d_hat}


@torch.no_grad()
def _build_student_from_teacher(teacher, device):
    """Deep-copy teacher → student so we can swap lm_head only."""
    from transformers import AutoModelForCausalLM
    student = AutoModelForCausalLM.from_pretrained(
        teacher.config._name_or_path, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    # explicit untie the lm_head from embed_tokens
    lm_shape = student.lm_head.weight.shape
    student.lm_head.weight = torch.nn.Parameter(
        student.lm_head.weight.clone().detach().to(device, dtype=torch.bfloat16)
    )
    return student


@torch.no_grad()
def _install_lm_head(student, W_lm_V_d, device):
    """Install a (V, d_w) float weight into student.lm_head.

    If d_w < true hidden dim d, zero-pad the trailing dims (for Carlini d_hat=894
    case where the last 1-2 singular components fell below gap).  If d_w > d,
    truncate.  This preserves the functional meaning: the extra teacher dims
    contribute negligibly anyway (that's why the gap detector cut them).
    """
    V, d_w = W_lm_V_d.shape
    lm = student.lm_head
    V_true, d_true = lm.weight.shape
    assert V == V_true, f"vocab mismatch {V} vs {V_true}"
    if d_w == d_true:
        new_W = W_lm_V_d
    elif d_w < d_true:
        pad = torch.zeros(V, d_true - d_w, device=W_lm_V_d.device, dtype=W_lm_V_d.dtype)
        new_W = torch.cat([W_lm_V_d, pad], dim=1)
    else:
        new_W = W_lm_V_d[:, :d_true]
    lm.weight.data = new_W.to(device=device, dtype=lm.weight.dtype)


@torch.no_grad()
def _kl_on_pool(teacher, student, query_ids, eval_positions, device, batch_size):
    """KL(teacher || student) averaged over the last `eval_positions` tokens."""
    kls = []
    top1 = []
    N = query_ids.shape[0]
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        ids = query_ids[s:e].to(device)
        zt = teacher(input_ids=ids).logits[:, -eval_positions:, :].float()
        zs = student(input_ids=ids).logits[:, -eval_positions:, :].float()
        pt = F.log_softmax(zt, dim=-1)
        ps = F.log_softmax(zs, dim=-1)
        kl = (pt.exp() * (pt - ps)).sum(dim=-1).mean().item()
        kls.append(kl)
        top_t = zt.argmax(dim=-1)
        top_s = zs.argmax(dim=-1)
        top1.append((top_t == top_s).float().mean().item())
    return float(np.mean(kls)), float(np.mean(top1))


def build_query_pool(tokenizer, num_queries, seq_len, seed=42):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        ids = []
        for ex in ds:
            text = ex.get("text", "").strip()
            if len(text) < 40:
                continue
            enc = tokenizer(text, max_length=seq_len, truncation=True,
                            padding="max_length", return_tensors="pt")
            ids.append(enc["input_ids"].squeeze(0))
            if len(ids) >= num_queries:
                break
        return torch.stack(ids[:num_queries])
    except Exception as e:
        logger.warning("wikitext unavail (%s); using random tokens", e)
        rng = torch.Generator().manual_seed(seed)
        return torch.randint(3, tokenizer.vocab_size,
                              (num_queries, seq_len), generator=rng)


def compute_combined_W_lm(W_eff, cp_A, cp_lambdas, V, d):
    """Combined Carlini + CP: use CP factors in Carlini basis to align rotation.

    Carlini gives W_eff (d_hat, V) — rotated teacher W_lm.
    CP mode-0 factor A (d_hat, r) lives in the same Carlini basis.
    Each CP factor j, if the attack worked, should approximately match a
    Carlini-rotated teacher W_lm column (up to sign/scale).

    Simplest combined estimator:
      1. Normalise both W_eff columns and CP factors
      2. For each teacher vocab index v, find the CP factor that best matches
         W_eff[:, v] by absolute cosine — call it j*(v)
      3. Blend: W_lm_recovered[v, :] = W_eff[:, v]  (trivially, as starting
         point), then adjust via the aligned CP factor for permutation correction.

    Without oracle W_lm, the best we can do is to use W_eff directly (it IS the
    full teacher projection in Carlini basis), rotated back into hidden-dim d
    via the d_hat × d canonical identity (since we ensured d_hat ≥ d).

    Pragmatic choice (per the proposed Path A):
      W_lm_recovered = W_eff[:d, :].T  (V, d)    ← naive Carlini only
      W_lm_combined  = W_lm_recovered  (+ permutation alignment from CP)

    We additionally return CP_lift as a sanity baseline:
      CP_lift[v, :] = best CP factor for teacher column v, lifted to V-space.

    Returns
    -------
    W_carlini     : (V, d)  Carlini W_eff^T rotated to hidden dim (baseline)
    W_combined    : (V, d)  Carlini + CP permutation-fixed (this is the test)
    """
    d_hat, V_chk = W_eff.shape
    assert V == V_chk
    r = cp_A.shape[1]

    # --- Baseline Carlini in V×d format ---
    # W_eff rows are orthonormal (from SVD). Take top-d rows.
    # This recovers teacher W_lm up to a d×d rotation (the gauge).
    # As student.lm_head, it preserves subspace but NOT per-column identity.
    W_carlini_d = W_eff[:d, :]          # (d, V), take top-d rows
    W_carlini = W_carlini_d.T           # (V, d)

    # --- CP-based rotation fix ---
    # For each top-d CP factor (by |λ|), project into Carlini subspace and
    # align with the closest matching W_eff column. Build a d×d rotation R
    # such that W_carlini @ R approximates the true teacher W_lm.
    # Simpler version: use CP factors directly as d column-directions
    # weighted by λ.

    # Rank-r candidate basis from CP (in Carlini subspace): A (d_hat, r)
    # We want d directions out of r factors.
    # Select top-d by |λ|:
    top_d_idx = torch.argsort(cp_lambdas.abs(), descending=True)[:d]
    B_cp = cp_A[:, top_d_idx]           # (d_hat, d)

    # Build orthonormal B_ortho spanning those directions
    U_cp, _, _ = torch.linalg.svd(B_cp, full_matrices=False)
    U_cp = U_cp[:, :d]                  # (d_hat, d)

    # Align U_cp with first d standard basis (to recover a rotation)
    # Simpler: use it directly — W_combined = (U_cp.T @ W_eff).T ∈ (V, d)
    W_combined_d = U_cp.T @ W_eff       # (d, V)
    W_combined = W_combined_d.T         # (V, d)

    return W_carlini, W_combined


def compute_cp_lift(cp_A, W_eff, V, d):
    """Lift CP factors (d_hat × r) to V-space via W_eff^T @ A (V × r),
    then pick top-d factors by norm to form a (V, d) lm_head candidate."""
    r = cp_A.shape[1]
    A_vspace = W_eff.T @ cp_A           # (V, r)
    # pick top-d factors by norm
    norms = A_vspace.norm(dim=0)
    top_d_idx = torch.argsort(norms, descending=True)[:d]
    W_cp_lift = A_vspace[:, top_d_idx]  # (V, d)
    return W_cp_lift


def main():
    setup_logging()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Loading teacher: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    V = teacher.config.vocab_size
    d = teacher.config.hidden_size

    logger.info("Running Carlini SVD ...")
    car = run_carlini_svd(
        teacher, V, d,
        num_queries=args.carlini_queries,
        seq_len=args.carlini_seq_len,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    W_eff = car["W_eff"].to(args.device)    # (d_hat, V)
    d_hat = car["d_hat"]
    logger.info("W_eff shape: %s (d_hat=%d, V=%d)", tuple(W_eff.shape), d_hat, V)

    logger.info("Loading CP factors from %s", args.cp_factors_pt)
    cp = torch.load(args.cp_factors_pt, map_location=args.device, weights_only=False)
    cp_A = cp["A"].to(args.device)          # (894, 4864)
    cp_lambdas = cp["lambdas"].to(args.device)

    # Align the Carlini basis dims — may need to pad/truncate if d_hat differs
    if W_eff.shape[0] != cp_A.shape[0]:
        min_dh = min(W_eff.shape[0], cp_A.shape[0])
        logger.warning("Carlini d_hat=%d vs CP A rows=%d; truncating to %d",
                       W_eff.shape[0], cp_A.shape[0], min_dh)
        W_eff = W_eff[:min_dh, :]
        cp_A = cp_A[:min_dh, :]

    # --- Compute recovered lm_head variants ---
    W_carlini, W_combined = compute_combined_W_lm(
        W_eff, cp_A, cp_lambdas, V, d,
    )
    W_cp_lift = compute_cp_lift(cp_A, W_eff, V, d)

    logger.info("Recovered matrices:")
    logger.info("  W_carlini  shape %s", tuple(W_carlini.shape))
    logger.info("  W_combined shape %s", tuple(W_combined.shape))
    logger.info("  W_cp_lift  shape %s", tuple(W_cp_lift.shape))

    # --- Build oracle reference & random baseline ---
    W_oracle = teacher.lm_head.weight.detach().float().clone()  # (V, d)
    rng = torch.Generator(device=args.device).manual_seed(args.seed)
    W_random = torch.randn(V, d, device=args.device,
                           generator=rng, dtype=torch.float32) * 0.02

    # --- Build query pool for KL evaluation ---
    logger.info("Building query pool: %d queries @ seq_len %d",
                args.num_queries, args.seq_len)
    query_ids = build_query_pool(tokenizer, args.num_queries, args.seq_len,
                                  seed=args.seed + 100)

    # --- Build student (copy of teacher for oracle prefix) ---
    logger.info("Building student (deep copy of teacher)...")
    student = _build_student_from_teacher(teacher, args.device)

    # --- Evaluate each variant ---
    results = {}
    variants = [
        ("oracle_teacher_lm_head", W_oracle),
        ("random_init", W_random),
        ("carlini_subspace_only", W_carlini),
        ("cp_lift_naive", W_cp_lift),
        ("carlini_plus_cp_combined", W_combined),
    ]
    for name, W in variants:
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating: %s", name)
        _install_lm_head(student, W, args.device)
        t0 = time.time()
        kl, top1 = _kl_on_pool(
            teacher, student, query_ids,
            args.eval_positions, args.device, args.batch_size,
        )
        elapsed = time.time() - t0
        logger.info("  KL = %.4f  top1_agree = %.4f  (%.1fs)", kl, top1, elapsed)
        results[name] = {"kl": kl, "top1_agree": top1, "elapsed_s": elapsed}

    # --- Summarize & verdict ---
    kl_oracle = results["oracle_teacher_lm_head"]["kl"]
    kl_random = results["random_init"]["kl"]
    kl_combined = results["carlini_plus_cp_combined"]["kl"]
    kl_carlini = results["carlini_subspace_only"]["kl"]

    logger.info("\n" + "=" * 60)
    logger.info("FUNCTIONAL KL SUMMARY")
    logger.info("=" * 60)
    for name, r in results.items():
        logger.info("  %-32s  KL=%.4f  top1=%.4f",
                    name, r["kl"], r["top1_agree"])

    # --- Report stolen-parameter verdict ---
    rel_improvement = (kl_random - kl_combined) / max(kl_random - kl_oracle, 1e-9)
    logger.info("\nRelative improvement of combined vs random: %.2f%%",
                100 * rel_improvement)
    stolen = kl_combined < 1.0 and kl_combined < 0.5 * kl_random
    logger.info("VERDICT: %s parameter theft",
                "CONFIRMED FUNCTIONAL" if stolen else "NOT CONFIRMED")

    out = {
        "args": vars(args),
        "V": V, "d": d, "d_hat": d_hat,
        "results": results,
        "kl_random_baseline": kl_random,
        "kl_oracle_upper_bound": kl_oracle,
        "kl_combined": kl_combined,
        "relative_improvement_pct": 100 * rel_improvement,
        "functional_theft_confirmed": bool(stolen),
    }
    out_path = Path(args.output_dir) / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
