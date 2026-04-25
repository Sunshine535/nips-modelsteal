#!/usr/bin/env python3
"""Post-hoc evaluation of a finished ``run_clone.py`` run.

Reads ``results.json`` produced by ``run_clone.py`` and prints the key
hidden-state geometry numbers, perplexity increase, and W_lm recovery
quality side by side.

If ``student_weights.pt`` was saved (``--save_student`` in the run), this
also re-runs the geometry match metric on a larger sample for a more
stable estimate.

Usage
-----
    python baselines/clone_2025/eval_clone.py \\
        --results_dir baselines/clone_2025/results/qwen25_05b

    # More stable geometry estimate on held-out queries
    python baselines/clone_2025/eval_clone.py \\
        --results_dir baselines/clone_2025/results/qwen25_05b \\
        --rerun_geometry --num_eval_queries 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--rerun_geometry", action="store_true",
                        help="Reload student + teacher and recompute geometry "
                             "on a fresh query set (requires student_weights.pt).")
    parser.add_argument("--num_eval_queries", type=int, default=512)
    parser.add_argument("--eval_seq_len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    rd = Path(args.results_dir)
    res_path = rd / "results.json"
    if not res_path.exists():
        raise FileNotFoundError(f"{res_path} not found. Run run_clone.py first.")
    with open(res_path) as f:
        res = json.load(f)

    cfg = res["config"]
    p1 = res["phase1_carlini"]
    p2 = res["phase2_student"]
    p3 = res.get("phase3_distill", {})
    p4 = res["phase4_geometry"]

    print("=" * 72)
    print(f"Clone 2025 — {cfg['model_name']}")
    print("=" * 72)
    print(f"Teacher L / Student L: {cfg['teacher_layers']} / {cfg['student_layers']}")
    print(f"Carlini queries      : {cfg['num_queries_carlini']}")
    print(f"Distill steps        : {cfg['distill_steps']} "
          f"(bs={cfg['distill_batch_size']}, lr={cfg['distill_lr']}, "
          f"T={cfg['distill_temperature']}, λCE={cfg['distill_ce_lambda']})")
    print("-" * 72)
    print("Phase 1 — Carlini extraction of W_lm:")
    print(f"  d_hat              = {p1['recovered_d']} (true {p1['true_d']})  "
          f"[{'match' if p1['match'] else 'mismatch'}]")
    print(f"  gap ratio @ d_hat  = {p1['gap_ratio_at_d_hat']:.3f}")
    print(f"  subspace mean cos  = {p1['W_lm_subspace_mean_cos']:.6f}")
    print("-" * 72)
    print("Phase 2 — Student architecture:")
    print(f"  student params     = {p2['num_parameters']:>12,}")
    print(f"  teacher params     = {p2['teacher_parameters']:>12,}")
    print(f"  ratio              = {p2['param_ratio'] * 100:.1f}%")
    if p3:
        print("-" * 72)
        print("Phase 3 — Distillation:")
        print(f"  final eval KL      = {p3['final_eval_kl']:.4f}")
        print(f"  teacher NLL        = {p3['teacher_nll']:.4f}")
        print(f"  student NLL        = {p3['student_nll']:.4f}")
        print(f"  perplexity ↑ %     = {p3['perplexity_increase_pct']:.2f}%")
        print(f"  distill elapsed    = {p3['elapsed_seconds']:.1f}s")
    print("-" * 72)
    print("Phase 4 — Hidden-state geometry match:")
    print(f"  Frobenius form     = {p4['geometry_match_frobenius']:.4f}")
    print(f"  Trace form         = {p4['geometry_match_trace']:.4f}  "
          f"(paper-style; target 0.976 @ L=6 on distilGPT-2)")
    print(f"  Mean aligned cos   = {p4['mean_aligned_cosine']:.4f}")
    print("=" * 72)

    # Optional: rerun geometry on a large held-out sample
    if args.rerun_geometry:
        print(f"[rerun_geometry] Loading student weights from {rd / 'student_weights.pt'}")
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM
        from run_clone import geometry_match, get_last_hidden, build_student  # type: ignore

        device = args.device
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        teacher = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()

        w_path = rd / "student_weights.pt"
        if not w_path.exists():
            print(f"[rerun_geometry] {w_path} not found — skipping")
            return

        # Re-build student with same architecture and load weights.
        import copy
        teacher_config = copy.deepcopy(teacher.config)
        teacher_config.num_hidden_layers = cfg["student_layers"]
        student = AutoModelForCausalLM.from_config(
            teacher_config, trust_remote_code=True
        ).to(device=device, dtype=dtype)
        state = torch.load(w_path, map_location=device)
        student.load_state_dict(state)
        student.eval()

        rng = torch.Generator(device="cpu")
        rng.manual_seed(cfg["seed"] + 9999)
        bs = 8
        V = cfg["vocab_size"]
        T_ = args.eval_seq_len
        Hs_T, Hs_S = [], []
        for _ in range((args.num_eval_queries + bs - 1) // bs):
            ids = torch.randint(0, V, (bs, T_), generator=rng).to(device)
            with torch.no_grad():
                Hs_T.append(get_last_hidden(teacher, ids).reshape(-1, teacher.config.hidden_size).cpu())
                Hs_S.append(get_last_hidden(student, ids).reshape(-1, teacher.config.hidden_size).cpu())
        H_T = torch.cat(Hs_T, dim=0)[: args.num_eval_queries * T_]
        H_S = torch.cat(Hs_S, dim=0)[: args.num_eval_queries * T_]

        geom = geometry_match(H_T, H_S)
        print("-" * 72)
        print(f"[rerun_geometry] N={args.num_eval_queries} queries x T={T_}")
        print(f"  Frobenius form     = {geom['geometry_match_frobenius']:.4f}")
        print(f"  Trace form         = {geom['geometry_match_trace']:.4f}")
        print(f"  Mean aligned cos   = {geom['mean_aligned_cosine']:.4f}")
        out = rd / "eval_geometry_rerun.json"
        with open(out, "w") as f:
            json.dump(geom, f, indent=2)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
