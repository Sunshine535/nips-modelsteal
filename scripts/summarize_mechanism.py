#!/usr/bin/env python3
"""Summarize mechanism diagnostics across seeds and variants."""
import argparse, json
import os
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = Path(args.results_dir)
    seed_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not seed_dirs:
        print(f"No seed_* directories found in {base}")
        return

    by_variant = {}
    teacher_ppl = None
    basis_info = None
    for sd in seed_dirs:
        rj = sd / "results.json"
        mj = sd / "manifest.json"
        if not rj.exists():
            continue
        with open(rj) as f:
            r = json.load(f)
        teacher_ppl = r["teacher"]["ppl"]
        basis_info = r.get("basis", {})
        for vname, vd in r["variants"].items():
            by_variant.setdefault(vname, []).append({
                "seed": int(sd.name.split("_")[-1]),
                "ppl": vd["final"]["ppl"],
                "kl": vd["final"]["kl"],
                "top1": vd["final"]["top1"],
                "kd_ce_ratio_early": np.mean(vd["mechanism_log"]["kd_ce_ratio"][:5])
                    if vd["mechanism_log"]["kd_ce_ratio"] else float("nan"),
                "kd_ce_ratio_late": np.mean(vd["mechanism_log"]["kd_ce_ratio"][-5:])
                    if vd["mechanism_log"]["kd_ce_ratio"] else float("nan"),
                "tail_w_late": np.mean(vd["mechanism_log"]["weight_mean_on_tail"][-5:])
                    if vd["mechanism_log"].get("weight_mean_on_tail") else float("nan"),
                "variant_budget": vd.get("variant_budget", {}),
            })

    lines = []
    lines.append(f"# Mechanism Summary — {base.name}\n")
    lines.append(f"Teacher PPL: {teacher_ppl:.2f}")
    if basis_info:
        lines.append(f"Basis: {basis_info.get('source')} "
                      f"(access={basis_info.get('access_level')}, "
                      f"queries={basis_info.get('n_queries_used_in_recovery')})")
    lines.append("")
    lines.append("| Variant | PPL m±s | KL m±s | Top-1 m±s | kd/ce early | kd/ce late | tail_w late | budget topk/probe |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for vname, runs in by_variant.items():
        ppls = np.array([r["ppl"] for r in runs])
        kls = np.array([r["kl"] for r in runs])
        t1s = np.array([r["top1"] for r in runs])
        kc_e = np.array([r["kd_ce_ratio_early"] for r in runs])
        kc_l = np.array([r["kd_ce_ratio_late"] for r in runs])
        tw_l = np.array([r["tail_w_late"] for r in runs])
        b = runs[0].get("variant_budget", {})
        topk = b.get("topk_queries", 0)
        probe = b.get("probe_queries", 0)
        tw_str = "n/a" if np.all(np.isnan(tw_l)) else f"{np.nanmean(tw_l):.4f}"
        lines.append(
            f"| {vname} | {ppls.mean():.2f} ± {ppls.std():.2f} "
            f"| {kls.mean():.4f} ± {kls.std():.4f} "
            f"| {t1s.mean():.4f} ± {t1s.std():.4f} "
            f"| {np.nanmean(kc_e):.4f} | {np.nanmean(kc_l):.4f} "
            f"| {tw_str} "
            f"| {topk}/{probe:,} |"
        )

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
