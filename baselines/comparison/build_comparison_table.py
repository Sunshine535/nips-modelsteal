#!/usr/bin/env python3
"""Build the side-by-side comparison table across all methods.

Aggregates result JSONs from:

  - ``baselines/carlini_2024/results/*/results.json``   (Carlini 2024)
  - ``baselines/clone_2025/results/*/results.json``     (Clone 2025)
  - ``results/v5_multi_sweep/*/summary.json``           (Ours — multi-model sweep)
  - ``results/v5_multi_model_sweep/*/summary.json``     (alt path — same)
  - ``results/v5_pure_logits_algebraic/results.json``   (Ours — pure-logits)
  - ``results/v5_algebraic_v2_clean/results.json``      (Ours — oracle / algebraic W_down)
  - ``results/v4_llama_spsi/experiment_summary.json``   (Ours — Llama oracle)

And writes:

  - ``baselines/comparison/comparison.json`` — machine-readable table
  - ``baselines/comparison/comparison.md``   — human-readable markdown

The schema tolerates missing files (we just emit "-" in the corresponding
cells and print a warning so you know what's stale).

Usage
-----
    python baselines/comparison/build_comparison_table.py \\
        --project_root /home/tarkoy/nips/nips-modelsteal
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("comparison")

# ── Data model ──────────────────────────────────────────────────────────


@dataclass
class Row:
    method: str
    model: str
    attack_target: str
    pure_logits: bool
    w_lm_cos: Optional[float] = None
    w_down_cos: Optional[float] = None
    hidden_geom_match: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def time_str(self) -> str:
        if self.elapsed_seconds is None:
            return "—"
        s = self.elapsed_seconds
        if s < 60:
            return f"{s:.1f}s"
        if s < 3600:
            return f"{s / 60:.1f}m"
        return f"{s / 3600:.1f}h"

    def fmt(self, val: Optional[float], digits: int = 4) -> str:
        if val is None:
            return "—"
        return f"{val:.{digits}f}"


# ── Readers ─────────────────────────────────────────────────────────────


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not parse %s: %s", path, e)
        return None


def load_carlini(path: Path) -> Optional[Row]:
    j = _load_json(path)
    if not j:
        return None
    rec = j["output_projection_recovery"]
    # Prefer procrustes mean cos if finite, else fall back to subspace mean cos.
    proc = rec.get("procrustes_mean_cos")
    if proc is None or (isinstance(proc, float) and proc != proc):  # NaN check
        proc = None
    w_lm = proc if proc is not None else rec.get("subspace_mean_cos_angle")
    return Row(
        method="Carlini 2024",
        model=j["config"]["model_name"].split("/")[-1],
        attack_target="lm_head",
        pure_logits=True,
        w_lm_cos=w_lm,
        elapsed_seconds=j.get("elapsed_seconds"),
        extra={
            "subspace_cos": rec.get("subspace_mean_cos_angle"),
            "d_hat": rec.get("d_hat"),
            "d_true": rec.get("d_true"),
        },
    )


def load_clone(path: Path) -> Optional[Row]:
    j = _load_json(path)
    if not j:
        return None
    cfg = j["config"]
    p1 = j.get("phase1_carlini", {})
    p4 = j.get("phase4_geometry", {})
    p3 = j.get("phase3_distill", {})
    w_lm = p1.get("W_lm_subspace_mean_cos")
    # Use the "trace" form for geometry as it matches the paper's figure.
    geom = p4.get("geometry_match_trace") or p4.get("geometry_match_frobenius")
    elapsed = (
        p1.get("elapsed_seconds", 0.0) + p3.get("elapsed_seconds", 0.0)
    ) or None
    return Row(
        method=f"Clone 2025 (L={cfg['student_layers']})",
        model=cfg["model_name"].split("/")[-1],
        attack_target="lm_head + distill",
        pure_logits=True,
        w_lm_cos=w_lm,
        hidden_geom_match=geom,
        elapsed_seconds=elapsed,
        extra={
            "student_layers": cfg["student_layers"],
            "teacher_layers": cfg["teacher_layers"],
            "perplexity_increase_pct": p3.get("perplexity_increase_pct"),
        },
    )


def load_ours_algebraic_v2_clean(path: Path) -> Optional[Row]:
    """Parses ``results/v5_algebraic_v2_clean/results.json`` — our main
    algebraic W_down oracle run on Qwen2.5-0.5B.
    """
    j = _load_json(path)
    if not j:
        return None
    cfg = j.get("config", {})
    p1 = j.get("phase1_w_down_ols", {})
    p2 = j.get("phase2_gate_up", {}).get("aligned_cosine", {})
    step0 = j.get("step0_h_out_recovery", {})
    w_down = p1.get("W_down_cosine")
    # 'hidden_geom_match' proxy = h_out global cosine (how well the block
    # reproduces teacher's boundary hidden state).
    geom = step0.get("global_cosine")
    elapsed = j.get("elapsed_seconds")
    return Row(
        method="Ours (oracle algebraic W_down)",
        model=cfg.get("model_name", "").split("/")[-1] or "unknown",
        attack_target="lm_head + W_down",
        pure_logits=False,
        w_lm_cos=None,  # separate lm_head metric not in this file
        w_down_cos=w_down,
        hidden_geom_match=geom,
        elapsed_seconds=elapsed,
        extra={
            "W_down_per_row_cos": p1.get("W_down_per_row_cosine"),
            "W_gate_aligned_cos": p2.get("gate_flat"),
            "W_up_aligned_cos": p2.get("up_flat"),
        },
    )


def load_ours_pure_logits(root: Path) -> list[Row]:
    """Read ``results/v2_pure_logits_s*`` (original pure-logits runs).
    These store ``spsi_summary.json`` with per-matrix cosines; we surface
    the lm_head cosine (usually near-zero) and the most informative
    block matrix cosine (the max over block matrices to highlight that
    no structural recovery happens).
    """
    rows: list[Row] = []
    for sdir in sorted(root.glob("v2_pure_logits_s*")):
        summary_path = sdir / "regime_pure_logits" / "init_0" / "spsi_summary.json"
        j = _load_json(summary_path)
        if not j:
            continue
        # lm_head block
        lm_block = next((b for b in j["blocks"] if b["name"] == "lm_head"), None)
        if lm_block is None:
            continue
        lm_cos = lm_block["per_matrix_cosine"].get("lm_head.weight")
        # Aggregate W_down: max over any block.down_proj.weight cosines
        down_cos_vals = []
        for b in j["blocks"]:
            if b["name"].startswith("block_"):
                v = b["per_matrix_cosine"].get("down_proj.weight")
                if v is not None:
                    down_cos_vals.append(v)
        w_down_cos = max(down_cos_vals, default=None)
        rows.append(
            Row(
                method=f"Ours (pure-logits SPSI, {sdir.name.split('_')[-1]})",
                model="Qwen2.5-0.5B",
                attack_target="lm_head + h_L",
                pure_logits=True,
                w_lm_cos=lm_cos,
                w_down_cos=w_down_cos,
                elapsed_seconds=None,
                extra={"total_queries": j.get("total_queries")},
            )
        )
    return rows


def load_ours_pure_logits_algebraic(path: Path) -> Optional[Row]:
    j = _load_json(path)
    if not j:
        return None
    # Be tolerant of schema variants; try a few plausible keys.
    cfg = j.get("config", {})
    d1 = j.get("pure_logits_algebraic") or j.get("experiment_D1") or {}
    u_fid = j.get("u_fidelity") or d1.get("u_fidelity") or {}
    w_lm_cos = (
        j.get("carlini_subspace_cos")
        or j.get("lm_head_subspace_cos")
        or (j.get("step_A_carlini") or {}).get("subspace_mean_cos_angle")
    )
    w_down = d1.get("W_down_cosine") or d1.get("w_down_cos")
    return Row(
        method="Ours (pure-logits algebraic)",
        model=(cfg.get("model_name") or "Qwen2.5-0.5B").split("/")[-1],
        attack_target="lm_head + h_L (algebraic)",
        pure_logits=True,
        w_lm_cos=w_lm_cos,
        w_down_cos=w_down,
        elapsed_seconds=j.get("elapsed_seconds"),
        extra=u_fid,
    )


def load_ours_multi_sweep(root: Path) -> list[Row]:
    """Aggregate per-model JSONs from the multi-model sweep."""
    rows: list[Row] = []
    # Tolerate both plural / singular directory names.
    candidates = [
        root / "v5_multi_sweep",
        root / "v5_multi_model_sweep",
    ]
    for base in candidates:
        if not base.exists():
            continue
        for sdir in sorted(base.iterdir()):
            if not sdir.is_dir():
                continue
            sum_path = sdir / "summary.json"
            j = _load_json(sum_path)
            if not j:
                continue
            # Extract the per-model metrics we need
            carlini = j.get("carlini_svd", {})
            w_down_ols = j.get("algebraic_w_down", {})
            joint = j.get("joint_w_gate_w_up", {})
            w_lm_cos = carlini.get("subspace_mean_cos_angle")
            w_down_cos = w_down_ols.get("W_down_cosine")
            elapsed = j.get("elapsed_seconds") or j.get("total_seconds")
            rows.append(
                Row(
                    method="Ours (oracle algebraic W_down)",
                    model=j.get("model_name", sdir.name).split("/")[-1],
                    attack_target="lm_head + W_down",
                    pure_logits=False,
                    w_lm_cos=w_lm_cos,
                    w_down_cos=w_down_cos,
                    hidden_geom_match=(joint.get("functional") or {}).get("recon_cosine"),
                    elapsed_seconds=elapsed,
                )
            )
    return rows


def load_ours_llama_spsi(path: Path) -> Optional[Row]:
    j = _load_json(path)
    if not j:
        return None
    # Parse SPSI summary for Llama
    # v4_llama_spsi/experiment_summary.json schema varies — be tolerant
    cfg = j.get("config", {}) or j.get("args", {})
    model = cfg.get("model_name") or "meta-llama/Llama-3.2-1B"
    rec = j.get("recovery", {}) or j.get("final_metrics", {}) or j
    w_lm = (
        rec.get("W_lm_cos")
        or rec.get("lm_head_cos")
        or rec.get("subspace_mean_cos_angle")
    )
    w_down = rec.get("W_down_cosine") or rec.get("down_proj_cosine")
    geom = rec.get("hidden_geom_match") or rec.get("h_out_cosine")
    return Row(
        method="Ours (oracle algebraic W_down)",
        model=model.split("/")[-1],
        attack_target="lm_head + W_down",
        pure_logits=False,
        w_lm_cos=w_lm,
        w_down_cos=w_down,
        hidden_geom_match=geom,
        elapsed_seconds=j.get("elapsed_seconds"),
    )


# ── Main aggregation ───────────────────────────────────────────────────


def gather_rows(project_root: Path) -> list[Row]:
    rows: list[Row] = []

    # Carlini: one per model subdir
    carlini_root = project_root / "baselines" / "carlini_2024" / "results"
    if carlini_root.exists():
        for sdir in sorted(carlini_root.iterdir()):
            if not sdir.is_dir():
                continue
            row = load_carlini(sdir / "results.json")
            if row is not None:
                rows.append(row)

    # Clone: one per model subdir
    clone_root = project_root / "baselines" / "clone_2025" / "results"
    if clone_root.exists():
        for sdir in sorted(clone_root.iterdir()):
            if not sdir.is_dir():
                continue
            row = load_clone(sdir / "results.json")
            if row is not None:
                rows.append(row)

    # Ours — pure-logits algebraic
    pla_path = project_root / "results" / "v5_pure_logits_algebraic" / "results.json"
    r = load_ours_pure_logits_algebraic(pla_path)
    if r is not None:
        rows.append(r)

    # Ours — v2 pure-logits SPSI runs (multi-seed)
    rows.extend(load_ours_pure_logits(project_root / "results"))

    # Ours — algebraic v2 clean (main oracle recovery)
    alg_path = project_root / "results" / "v5_algebraic_v2_clean" / "results.json"
    r = load_ours_algebraic_v2_clean(alg_path)
    if r is not None:
        rows.append(r)

    # Ours — multi-model sweep
    rows.extend(load_ours_multi_sweep(project_root / "results"))

    # Ours — Llama SPSI
    llama_path = project_root / "results" / "v4_llama_spsi" / "experiment_summary.json"
    r = load_ours_llama_spsi(llama_path)
    if r is not None:
        rows.append(r)

    return rows


def build_markdown(rows: list[Row]) -> str:
    header = (
        "| Method | Model | Attack target | W_lm cos | W_down cos | "
        "Hidden geom match | Pure-logits? | Time |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        lines.append(
            "| {method} | {model} | {target} | {wlm} | {wdn} | {geom} | "
            "{pl} | {tm} |".format(
                method=r.method,
                model=r.model,
                target=r.attack_target,
                wlm=r.fmt(r.w_lm_cos),
                wdn=r.fmt(r.w_down_cos),
                geom=r.fmt(r.hidden_geom_match),
                pl="✓" if r.pure_logits else "✗",
                tm=r.time_str(),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Root of the nips-modelsteal checkout.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write comparison.md / comparison.json "
             "(default: <project_root>/baselines/comparison/).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.output_dir) if args.output_dir else (
        project_root / "baselines" / "comparison"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = gather_rows(project_root)
    if not rows:
        logger.warning(
            "No results found under %s — did you run any experiments yet?",
            project_root,
        )

    # JSON
    js = {
        "rows": [asdict(r) for r in rows],
        "n_rows": len(rows),
    }
    json_path = out_dir / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(js, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    # Markdown
    md = "# Baselines vs Ours — Side-by-side comparison\n\n"
    md += (
        "Auto-generated by `build_comparison_table.py`. Rebuild with:\n\n"
        "```\npython baselines/comparison/build_comparison_table.py\n```\n\n"
    )
    md += build_markdown(rows) + "\n\n"
    md += "## Notes on metrics\n\n"
    md += (
        "- **W_lm cos** — mean cosine of recovered output projection columns vs "
        "true `lm_head.weight` after rotation / Procrustes. Subspace-level"
        " equality only — anything above ~0.999 is effectively identical.\n"
        "- **W_down cos** — cosine of the recovered last-block `down_proj.weight`"
        " vs true. Requires access to the post-attention residual stream "
        "(`h_mid`), so Carlini / Clone cannot produce this.\n"
        "- **Hidden geom match** — fraction of teacher variance preserved "
        "after optimal Procrustes rotation of student last-layer hidden "
        "states. Paper-style trace form used when available.\n"
        "- **Pure-logits?** — ✓ means only `(x, z(x))` pairs are used; ✗ means "
        "the method uses teacher-internal hidden states (oracle regime).\n"
    )
    md_path = out_dir / "comparison.md"
    with open(md_path, "w") as f:
        f.write(md)
    logger.info("Wrote %s", md_path)

    # Console preview
    print()
    print(build_markdown(rows))
    print()


if __name__ == "__main__":
    main()
