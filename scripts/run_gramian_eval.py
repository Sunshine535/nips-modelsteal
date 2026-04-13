#!/usr/bin/env python3
"""Offline Gramian eigenspectrum analysis for S-PSI suffix recovery.

Loads teacher and student models, computes the projected Gramian
eigenspectrum for each suffix block, and correlates Gramian metrics
(λ_min, effective rank, log-det) with recovery quality from
spsi_summary.json.

Outputs:
  - gramian_summary.json    per-block Gramian eigenspectra & metrics
  - gramian_vs_recovery.json  correlation between λ_min and cosine recovery

Usage:
    python scripts/run_gramian_eval.py \
        --config configs/inversion_config.yaml \
        --results_dir results/spsi \
        --output_dir results/gramian_eval \
        --num_probes 64 \
        --query_subsample 256

    # without gauge projection (ablation)
    python scripts/run_gramian_eval.py \
        --config configs/inversion_config.yaml \
        --results_dir results/spsi \
        --output_dir results/gramian_eval_no_gauge \
        --no_project_gauge
"""

import argparse
import glob as glob_mod
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gramian import (
    GramianConfig,
    GramianResult,
    build_probe_matrix,
    compute_gramian_for_block,
    make_flat_param_spec,
)
from src.symmetry_gauge import (
    build_flat_param_spec as sg_build_flat_param_spec,
    build_suffix_gauge_basis,
    gauge_summary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging(log_dir: str | None = None):
    """Configure root logger with console + optional file handler."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(Path(log_dir) / "gramian_eval.log", mode="a")
        )
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def get_num_blocks(model: torch.nn.Module) -> int:
    """Detect total number of transformer blocks."""
    max_idx = -1
    for name, _ in model.named_parameters():
        for part in name.split("."):
            if part.isdigit():
                max_idx = max(max_idx, int(part))
                break
    return max_idx + 1


def get_block_param_names(model: torch.nn.Module, block_idx: int) -> list[str]:
    """Return parameter names belonging to a specific block."""
    names = []
    target = str(block_idx)
    for name, _ in model.named_parameters():
        parts = name.split(".")
        for p in parts:
            if p == target:
                names.append(name)
                break
    return names


def get_lm_head_param_names(model: torch.nn.Module) -> list[str]:
    return [n for n, _ in model.named_parameters() if "lm_head" in n]


def load_spsi_summary(results_dir: Path) -> dict:
    """Load spsi_summary.json files from all regime/init subdirectories.

    Returns:
        Nested dict:  {regime: {init_idx: {block_name: mean_cosine, ...}}}
    """
    summaries: dict[str, dict[int, dict]] = {}

    for regime_dir in sorted(results_dir.glob("regime_*")):
        regime = regime_dir.name.replace("regime_", "")
        summaries[regime] = {}

        for init_dir in sorted(regime_dir.glob("init_*")):
            summary_file = init_dir / "spsi_summary.json"
            if not summary_file.exists():
                logger.warning("Missing %s, skipping.", summary_file)
                continue

            try:
                with open(summary_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Cannot read %s: %s", summary_file, e)
                continue

            init_idx_str = init_dir.name.replace("init_", "")
            init_idx = int(init_idx_str) if init_idx_str.isdigit() else 0

            block_cosines = {}
            for block in data.get("blocks", []):
                block_name = block.get("name", "unknown")
                block_cosines[block_name] = {
                    "mean_cosine": block.get("mean_cosine", -1.0),
                    "per_matrix_cosine": block.get("per_matrix_cosine", {}),
                    "final_loss": block.get("final_loss", 0.0),
                    "num_steps": block.get("num_steps", 0),
                    "converged": block.get("converged", False),
                }
            summaries[regime][init_idx] = block_cosines

    return summaries


def build_query_pool(tokenizer, pool_size: int, max_seq_len: int, seed: int = 42):
    """Build query pool from WikiText, falling back to random tokens."""
    input_ids_list = []
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
                text,
                max_length=max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids_list.append(tokens["input_ids"].squeeze(0))
    except Exception as e:
        logger.warning("Dataset load failed (%s), using random tokens.", e)

    remaining = pool_size - len(input_ids_list)
    if remaining > 0:
        rng = torch.Generator().manual_seed(seed)
        random_ids = torch.randint(
            3, tokenizer.vocab_size, (remaining, max_seq_len), generator=rng
        )
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    return torch.stack(input_ids_list[:pool_size])


def gramian_result_to_dict(result: GramianResult) -> dict:
    """Serialise a GramianResult to a JSON-friendly dict."""
    return {
        "eigenvalues": result.eigenvalues.tolist(),
        "projected_min_eig": result.projected_min_eig,
        "logdet": result.logdet,
        "trace": result.trace,
        "effective_rank": result.effective_rank,
        "num_queries_used": result.num_queries_used,
    }


def compute_correlation(xs: list[float], ys: list[float]) -> dict:
    """Compute Pearson and Spearman correlation between two lists."""
    if len(xs) < 3 or len(ys) < 3:
        return {
            "pearson": float("nan"),
            "spearman": float("nan"),
            "spearman_pvalue": float("nan"),
            "n": len(xs),
        }

    xs_arr = np.array(xs, dtype=np.float64)
    ys_arr = np.array(ys, dtype=np.float64)

    # Pearson
    if xs_arr.std() < 1e-15 or ys_arr.std() < 1e-15:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(xs_arr, ys_arr)[0, 1])

    # Spearman (rank-based)
    from scipy.stats import spearmanr

    spearman_r, spearman_p = spearmanr(xs_arr, ys_arr)

    return {
        "pearson": pearson,
        "spearman": float(spearman_r),
        "spearman_pvalue": float(spearman_p),
        "n": len(xs),
    }


# ---------------------------------------------------------------------------
# Gramian computation for one model (teacher or recovered student)
# ---------------------------------------------------------------------------


def compute_block_gramians(
    model: torch.nn.Module,
    query_ids: torch.Tensor,
    block_indices: list[int],
    gramian_cfg: GramianConfig,
    project_gauge: bool,
    device: str,
    label: str = "model",
) -> dict[str, dict]:
    """Compute Gramian eigenspectra for each block in *block_indices*.

    Uses a simple TeacherCache-like wrapper so compute_gramian_for_block
    can access input_ids and logits in the expected format.

    Returns:
        {block_name: gramian_result_dict}
    """
    from types import SimpleNamespace

    model.eval()
    model.to(device)

    # Build a minimal cache holding input_ids + clean logits
    logger.info("[%s] Forward pass for cache (%d queries)...", label, len(query_ids))
    input_ids = query_ids.to(device)

    all_logits = []
    bs = 8
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(input_ids), bs):
            batch = input_ids[start : start + bs]
            logits = model(batch).logits
            all_logits.append(logits.cpu())
    clean_logits = torch.cat(all_logits, dim=0)

    cache = SimpleNamespace(
        input_ids=query_ids.cpu(),
        clean_logits=clean_logits,
        perturbed_input_ids=None,
        boundary_states={},
    )

    results: dict[str, dict] = {}
    for block_idx in block_indices:
        block_name = f"block_{block_idx}"
        param_names = get_block_param_names(model, block_idx)
        if not param_names:
            logger.warning("[%s] Block %d has no parameters, skipping.", label, block_idx)
            continue

        logger.info(
            "[%s] Computing Gramian for %s (%d params, %d probes)...",
            label, block_name, len(param_names), gramian_cfg.num_probes,
        )

        # Build gauge basis if requested
        gauge_basis = None
        if project_gauge:
            spec_sg = sg_build_flat_param_spec(model, param_names)
            gb = build_suffix_gauge_basis(
                model, spec_sg, [block_idx],
                include_rmsnorm=True, include_mlp=True,
                device="cpu",
            )
            if gb.num_directions > 0:
                gauge_basis = gb  # pass GaugeBasis object directly (sparse)
                logger.info(
                    "[%s] Gauge basis for %s: %d directions.",
                    label, block_name, gb.num_directions,
                )

        gramian_result, spec, probe_matrix = compute_gramian_for_block(
            student=model,
            cache=cache,
            param_names=param_names,
            config=gramian_cfg,
            gauge_basis=gauge_basis,
        )

        results[block_name] = gramian_result_to_dict(gramian_result)
        results[block_name]["num_params"] = spec.num_params
        results[block_name]["num_gauge_directions"] = (
            gauge_basis.num_directions if gauge_basis is not None else 0
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Gramian eigenspectrum evaluation for S-PSI"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inversion_config.yaml",
        help="YAML config (teacher/student model_name, num_suffix_blocks)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/spsi",
        help="Root results directory containing regime_*/init_*/spsi_summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/gramian_eval",
        help="Output directory for gramian_summary.json and gramian_vs_recovery.json",
    )
    parser.add_argument(
        "--num_probes", type=int, default=64, help="Number of random probe directions k"
    )
    parser.add_argument(
        "--query_subsample",
        type=int,
        default=256,
        help="Number of queries subsampled for Gramian estimation",
    )
    parser.add_argument(
        "--project_gauge",
        action="store_true",
        default=True,
        help="Project out gauge symmetries (default: True)",
    )
    parser.add_argument(
        "--no_project_gauge",
        dest="project_gauge",
        action="store_false",
        help="Disable gauge projection (ablation)",
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=2000,
        help="Number of query inputs for Gramian computation",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=256, help="Maximum sequence length"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=str(output_dir))
    torch.manual_seed(args.seed)

    # ── Load config ──────────────────────────────────────────────────────
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning("Config %s not found, using defaults.", config_path)
        cfg = {}

    teacher_name = cfg.get("teacher", {}).get("model_name", "Qwen/Qwen3.5-0.8B")
    student_name = cfg.get("student", {}).get("model_name", teacher_name)
    num_suffix_blocks = cfg.get("inversion", {}).get("num_suffix_blocks", 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path(args.results_dir)

    logger.info("=== Gramian Eigenspectrum Evaluation ===")
    logger.info("Teacher: %s  | Student: %s", teacher_name, student_name)
    logger.info("Suffix blocks: %d | Probes: %d | Queries: %d",
                num_suffix_blocks, args.num_probes, args.query_subsample)
    logger.info("Gauge projection: %s", args.project_gauge)

    # ── Load recovery summaries ──────────────────────────────────────────
    recovery_data = load_spsi_summary(results_dir)
    if not recovery_data:
        logger.error("No spsi_summary.json found under %s. Exiting.", results_dir)
        sys.exit(1)
    logger.info(
        "Loaded recovery data: %s",
        {r: len(inits) for r, inits in recovery_data.items()},
    )

    # ── Load tokenizer + build query pool ────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    query_ids = build_query_pool(
        tokenizer, args.pool_size, args.max_seq_len, args.seed
    )
    logger.info("Query pool ready: %d inputs.", len(query_ids))

    # ── Load teacher model ───────────────────────────────────────────────
    logger.info("Loading teacher model: %s ...", teacher_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    teacher_model.eval()
    num_blocks = get_num_blocks(teacher_model)
    suffix_indices = list(range(num_blocks - num_suffix_blocks, num_blocks))
    logger.info("Total blocks: %d  |  Suffix blocks: %s", num_blocks, suffix_indices)

    # ── Gramian config ───────────────────────────────────────────────────
    gramian_cfg = GramianConfig(
        num_probes=args.num_probes,
        query_subsample=args.query_subsample,
        project_gauge=args.project_gauge,
        seed=args.seed,
        include_sensitivity=False,  # offline eval — no perturbed data needed
        query_batch_size=4,
    )

    # ── Teacher Gramian (reference) ──────────────────────────────────────
    logger.info("Computing teacher Gramian eigenspectra...")
    teacher_gramians = compute_block_gramians(
        teacher_model, query_ids, suffix_indices,
        gramian_cfg, args.project_gauge, device,
        label="teacher",
    )

    # ── Per-regime, per-init: load recovered student → compute Gramian ───
    all_gramian_data: dict = {"teacher": teacher_gramians, "regimes": {}}
    correlation_data: dict = {}

    for regime, init_dict in recovery_data.items():
        logger.info("\n=== Regime: %s (%d inits) ===", regime, len(init_dict))
        regime_gramians: dict = {}
        # Accumulate (λ_min, cosine) pairs per block for correlation
        corr_pairs: dict[str, list[tuple[float, float]]] = {}

        for init_idx, block_cosines in sorted(init_dict.items()):
            init_label = f"regime_{regime}/init_{init_idx}"
            init_dir = results_dir / f"regime_{regime}" / f"init_{init_idx}"

            # Try to load the recovered student from checkpoint
            ckpt_dir = init_dir / "checkpoints"
            if not ckpt_dir.exists():
                logger.warning("No checkpoints in %s, skipping.", init_dir)
                continue

            # Load a fresh student and apply checkpoint weights
            logger.info("Loading student for %s ...", init_label)
            student = AutoModelForCausalLM.from_pretrained(
                student_name,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                trust_remote_code=True,
            )

            # Load the latest checkpoint for each block
            loaded_any = False
            for block_idx in suffix_indices:
                block_label = f"block_{block_idx}"
                pattern = str(ckpt_dir / f"ckpt_{block_label}_step*.pt")
                ckpts = sorted(
                    glob_mod.glob(pattern),
                    key=lambda x: int(x.split("step")[1].split(".")[0]),
                )
                if not ckpts:
                    # Also try lm_head checkpoint
                    if block_label == "block_" + str(suffix_indices[0]):
                        pass  # lm_head handled below
                    continue
                latest = ckpts[-1]
                logger.info("  Loading %s", latest)
                state = torch.load(latest, map_location=device, weights_only=True)
                if "model_state" in state:
                    student.load_state_dict(state["model_state"], strict=False)
                    loaded_any = True
                elif "state_dict" in state:
                    student.load_state_dict(state["state_dict"], strict=False)
                    loaded_any = True

            # Also try lm_head checkpoint
            lm_pattern = str(ckpt_dir / "ckpt_lm_head_step*.pt")
            lm_ckpts = sorted(
                glob_mod.glob(lm_pattern),
                key=lambda x: int(x.split("step")[1].split(".")[0]),
            )
            if lm_ckpts:
                latest_lm = lm_ckpts[-1]
                logger.info("  Loading lm_head: %s", latest_lm)
                lm_state = torch.load(latest_lm, map_location=device, weights_only=True)
                if "model_state" in lm_state:
                    student.load_state_dict(lm_state["model_state"], strict=False)
                    loaded_any = True

            if not loaded_any:
                logger.warning("No checkpoint loaded for %s, skipping.", init_label)
                del student
                torch.cuda.empty_cache()
                continue

            student.eval()

            # Compute Gramian for each suffix block using recovered student
            logger.info("Computing recovered-student Gramian for %s ...", init_label)
            student_gramians = compute_block_gramians(
                student, query_ids, suffix_indices,
                gramian_cfg, args.project_gauge, device,
                label=init_label,
            )

            # Store per-init results
            init_result = {"gramians": student_gramians, "recovery": block_cosines}
            regime_gramians[f"init_{init_idx}"] = init_result

            # Collect correlation pairs: λ_min vs mean_cosine per block
            for block_name, gram_dict in student_gramians.items():
                if block_name not in block_cosines:
                    continue
                cosine_val = block_cosines[block_name].get("mean_cosine", -1.0)
                if cosine_val < 0:
                    continue
                lam_min = gram_dict["projected_min_eig"]
                corr_pairs.setdefault(block_name, []).append((lam_min, cosine_val))

            del student
            torch.cuda.empty_cache()

        all_gramian_data["regimes"][regime] = regime_gramians

        # ── Compute correlation for this regime ──────────────────────────
        regime_corr: dict = {}
        for block_name, pairs in corr_pairs.items():
            lam_mins = [p[0] for p in pairs]
            cosines = [p[1] for p in pairs]
            corr = compute_correlation(lam_mins, cosines)
            regime_corr[block_name] = {
                "correlation": corr,
                "lambda_min_values": lam_mins,
                "cosine_values": cosines,
                "lambda_min_mean": float(np.mean(lam_mins)),
                "lambda_min_std": float(np.std(lam_mins)),
                "cosine_mean": float(np.mean(cosines)),
                "cosine_std": float(np.std(cosines)),
            }
            logger.info(
                "  %s: λ_min=%.4e ± %.4e, cos=%.4f ± %.4f, "
                "Pearson=%.3f, Spearman=%.3f (p=%.4f, n=%d)",
                block_name,
                float(np.mean(lam_mins)),
                float(np.std(lam_mins)),
                float(np.mean(cosines)),
                float(np.std(cosines)),
                corr["pearson"],
                corr["spearman"],
                corr["spearman_pvalue"],
                corr["n"],
            )

        correlation_data[regime] = regime_corr

    # ── Also correlate teacher Gramian λ_min against average recovery ────
    teacher_vs_recovery: dict = {}
    for regime, init_dict in recovery_data.items():
        for block_name in teacher_gramians:
            cosines = []
            for init_idx, block_cosines in init_dict.items():
                if block_name in block_cosines:
                    c = block_cosines[block_name].get("mean_cosine", -1.0)
                    if c >= 0:
                        cosines.append(c)
            if cosines:
                teacher_vs_recovery.setdefault(regime, {})[block_name] = {
                    "teacher_lambda_min": teacher_gramians[block_name]["projected_min_eig"],
                    "teacher_effective_rank": teacher_gramians[block_name]["effective_rank"],
                    "teacher_logdet": teacher_gramians[block_name]["logdet"],
                    "recovery_cosine_mean": float(np.mean(cosines)),
                    "recovery_cosine_std": float(np.std(cosines)),
                    "num_inits": len(cosines),
                }
    correlation_data["teacher_vs_recovery"] = teacher_vs_recovery

    # ── Save outputs ─────────────────────────────────────────────────────
    gramian_summary = {
        "config": {
            "teacher_model": teacher_name,
            "student_model": student_name,
            "num_suffix_blocks": num_suffix_blocks,
            "suffix_block_indices": suffix_indices,
            "num_probes": args.num_probes,
            "query_subsample": args.query_subsample,
            "project_gauge": args.project_gauge,
            "pool_size": args.pool_size,
            "seed": args.seed,
        },
        "teacher_gramians": teacher_gramians,
        "per_regime": {
            regime: {
                init_key: {
                    "gramians": init_data["gramians"],
                    "recovery_cosines": {
                        bn: bc.get("mean_cosine", -1.0)
                        for bn, bc in init_data["recovery"].items()
                    },
                }
                for init_key, init_data in regime_data.items()
            }
            for regime, regime_data in all_gramian_data.get("regimes", {}).items()
        },
    }

    summary_path = output_dir / "gramian_summary.json"
    with open(summary_path, "w") as f:
        json.dump(gramian_summary, f, indent=2)
    logger.info("Saved → %s", summary_path)

    corr_path = output_dir / "gramian_vs_recovery.json"
    with open(corr_path, "w") as f:
        json.dump(correlation_data, f, indent=2)
    logger.info("Saved → %s", corr_path)

    # ── Print summary table ──────────────────────────────────────────────
    logger.info("\n=== Gramian vs Recovery Summary ===")
    for regime, regime_corr in correlation_data.items():
        if regime == "teacher_vs_recovery":
            continue
        logger.info("Regime: %s", regime)
        for block_name, data in regime_corr.items():
            c = data["correlation"]
            logger.info(
                "  %s  |  λ_min %.4e ± %.4e  |  cos %.4f ± %.4f  |  "
                "r=%.3f  ρ=%.3f (p=%.4f)  n=%d",
                block_name,
                data["lambda_min_mean"],
                data["lambda_min_std"],
                data["cosine_mean"],
                data["cosine_std"],
                c["pearson"],
                c["spearman"],
                c["spearman_pvalue"],
                c["n"],
            )

    if "teacher_vs_recovery" in correlation_data:
        logger.info("\nTeacher Gramian vs Average Recovery:")
        for regime, blocks in correlation_data["teacher_vs_recovery"].items():
            for block_name, tvr in blocks.items():
                logger.info(
                    "  [%s] %s  |  teacher λ_min=%.4e  eff_rank=%.2f  |  "
                    "cos %.4f ± %.4f  (n=%d)",
                    regime, block_name,
                    tvr["teacher_lambda_min"],
                    tvr["teacher_effective_rank"],
                    tvr["recovery_cosine_mean"],
                    tvr["recovery_cosine_std"],
                    tvr["num_inits"],
                )

    # Free teacher model
    del teacher_model
    torch.cuda.empty_cache()

    logger.info("\n=== Gramian Evaluation Complete ===")
    logger.info("Outputs: %s", output_dir)


if __name__ == "__main__":
    main()
