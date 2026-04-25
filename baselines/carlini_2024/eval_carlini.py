#!/usr/bin/env python3
"""Evaluate a saved Carlini extraction.

Reads the ``results.json`` produced by ``run_carlini.py`` (and optionally a
``W_hat.pt``) and prints the key comparison numbers in a form that lines up
with the main paper's comparison table.

Usage
-----
    python baselines/carlini_2024/eval_carlini.py \\
        --results_dir baselines/carlini_2024/results/qwen25_05b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing results.json (and optionally W_hat.pt).",
    )
    args = parser.parse_args()

    rd = Path(args.results_dir)
    res_path = rd / "results.json"
    if not res_path.exists():
        raise FileNotFoundError(f"{res_path} not found. Run run_carlini.py first.")
    with open(res_path) as f:
        res = json.load(f)

    cfg = res["config"]
    hd = res["hidden_dim_recovery"]
    rec = res["output_projection_recovery"]

    print("=" * 72)
    print(f"Carlini 2024 extraction — {cfg['model_name']}")
    print("=" * 72)
    print(f"Queries             : {cfg['num_queries']}  seq_len={cfg['seq_len']}")
    print(f"Hidden (true/recov) : {hd['true_d']} / {hd['recovered_d']} "
          f"[{'match' if hd['match'] else 'mismatch'}]")
    print(f"Gap ratio @ d_hat   : {hd['gap_ratio_at_d_hat']:.3f}")
    print("-" * 72)
    print("Subspace principal-angle cosines (top right singular vectors"
          " vs W_lm columns):")
    print(f"  mean={rec['subspace_mean_cos_angle']:.6f}   "
          f"min={rec['subspace_min_cos_angle']:.6f}   "
          f"max={rec['subspace_max_cos_angle']:.6f}")
    print(f"  top 10: {[round(x, 5) for x in rec['principal_angle_cosines_top10']]}")
    print("-" * 72)
    print("Procrustes alignment (overlap dim "
          f"= {rec['overlap_dim']}):")
    print(f"  mean cos   = {rec['procrustes_mean_cos']:.6f}")
    print(f"  min cos    = {rec['procrustes_min_cos']:.6f}")
    print(f"  rel Frob   = {rec['procrustes_frob_relative_error']:.6f}")
    print(f"  proj resid = {rec['projection_residual']:.6f}")
    print("-" * 72)
    print(f"Elapsed: {res['elapsed_seconds']:.1f}s")

    sweep = res.get("query_efficiency_sweep", [])
    if sweep:
        print("-" * 72)
        print("Query-efficiency sweep:")
        print(f"  {'N':>6s}  {'d_hat':>5s}  {'subspace':>9s}  {'procrustes':>11s}  {'rel Frob':>9s}")
        for row in sweep:
            print(
                f"  {row['num_queries']:6d}  {row['recovered_d']:5d}  "
                f"{row['subspace_mean_cos_angle']:9.5f}  "
                f"{row['procrustes_mean_cos']:11.5f}  "
                f"{row['procrustes_frob_relative_error']:9.5f}"
            )
    print("=" * 72)

    # A concise machine-readable summary
    summary = {
        "model_name": cfg["model_name"],
        "num_queries": cfg["num_queries"],
        "hidden_dim_match": hd["match"],
        "recovered_d": hd["recovered_d"],
        "true_d": hd["true_d"],
        "W_lm_subspace_mean_cos": rec["subspace_mean_cos_angle"],
        "W_lm_procrustes_mean_cos": rec["procrustes_mean_cos"],
        "W_lm_procrustes_rel_frob_error": rec["procrustes_frob_relative_error"],
        "elapsed_seconds": res["elapsed_seconds"],
    }
    out_path = rd / "eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
