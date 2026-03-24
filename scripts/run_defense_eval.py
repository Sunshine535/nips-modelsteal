#!/usr/bin/env python3
"""Defense evaluation: test progressive inversion robustness against API defenses.

Defenses tested:
  - Logit rounding:          1, 2, 3 decimal places
  - Additive Gaussian noise: σ = 0.01, 0.1, 1.0
  - Temperature perturbation: δ = 0.1, 0.3, 0.5
  - Watermarking:            green_ratio = 0.3, 0.5, 0.7

For each defense configuration, runs a shortened progressive inversion and measures
recovery quality degradation relative to the no-defense baseline.

Usage:
    python scripts/run_defense_eval.py \
        --teacher_model Qwen/Qwen3.5-4B \
        --output_dir results/defense_eval

    python scripts/run_defense_eval.py --defenses logit_rounding gaussian_noise
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.active_query import QueryPool
from src.parameter_inverter import (
    BlackBoxTeacher,
    InversionConfig,
    LayerWiseInverter,
)

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── Defense Implementations ──────────────────────────────────────────

def make_logit_rounding(decimal_places: int) -> Callable:
    factor = 10 ** decimal_places
    def defense(logits: torch.Tensor) -> torch.Tensor:
        return torch.round(logits * factor) / factor
    return defense


def make_gaussian_noise(sigma: float) -> Callable:
    def defense(logits: torch.Tensor) -> torch.Tensor:
        return logits + torch.randn_like(logits) * sigma
    return defense


def make_temperature_perturbation(delta: float) -> Callable:
    def defense(logits: torch.Tensor) -> torch.Tensor:
        temp = 1.0 + (torch.rand(1, device=logits.device).item() * 2 - 1) * delta
        return logits / max(temp, 0.1)
    return defense


def make_watermarking(vocab_size: int, green_ratio: float, bias: float = 2.0) -> Callable:
    rng = torch.Generator().manual_seed(42)
    green_size = int(vocab_size * green_ratio)
    perm = torch.randperm(vocab_size, generator=rng)
    mask = torch.zeros(vocab_size)
    mask[perm[:green_size]] = bias

    def defense(logits: torch.Tensor) -> torch.Tensor:
        return logits + mask.to(logits.device).unsqueeze(0).unsqueeze(0).expand_as(logits)
    return defense


DEFENSE_CONFIGS = {
    "logit_rounding": {
        "param_name": "decimal_places",
        "param_values": [1, 2, 3],
        "factory": make_logit_rounding,
    },
    "gaussian_noise": {
        "param_name": "sigma",
        "param_values": [0.01, 0.1, 1.0],
        "factory": make_gaussian_noise,
    },
    "temperature_perturbation": {
        "param_name": "delta",
        "param_values": [0.1, 0.3, 0.5],
        "factory": make_temperature_perturbation,
    },
    "watermarking": {
        "param_name": "green_ratio",
        "param_values": [0.3, 0.5, 0.7],
        "factory": None,  # needs vocab_size
    },
}


# ── Utility Measurement ──────────────────────────────────────────────

@torch.no_grad()
def measure_defense_utility(teacher_model, defense_fn, tokenizer, device, num_samples=500):
    """Measure API utility degradation from defense."""
    teacher_model.eval()
    vocab_size = tokenizer.vocab_size
    rng = torch.Generator().manual_seed(456)
    batch_size, seq_len = 32, 128

    total_kl = 0.0
    total_top1_preserved = 0
    total_tokens = 0

    for _ in range(0, num_samples, batch_size):
        bsz = min(batch_size, num_samples)
        input_ids = torch.randint(3, vocab_size, (bsz, seq_len), generator=rng).to(device)

        clean_logits = teacher_model(input_ids).logits
        defended_logits = defense_fn(clean_logits) if defense_fn else clean_logits

        kl = F.kl_div(
            F.log_softmax(defended_logits, dim=-1),
            F.softmax(clean_logits, dim=-1),
            reduction="sum",
        ).item()
        total_kl += kl

        total_top1_preserved += (
            clean_logits.argmax(-1) == defended_logits.argmax(-1)
        ).sum().item()
        total_tokens += bsz * seq_len

    return {
        "kl_divergence": total_kl / total_tokens,
        "top1_preserved_rate": total_top1_preserved / total_tokens,
    }


# ── Run Inversion Under Defense ──────────────────────────────────────

def run_inversion_trial(
    teacher_raw, student_model_name, defense_fn, ground_truth,
    tokenizer, query_budget, device, output_dir,
):
    """Run shortened progressive inversion under a single defense."""
    teacher = BlackBoxTeacher(model=teacher_raw, device=device, defense_fn=defense_fn)

    student = AutoModelForCausalLM.from_pretrained(
        student_model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    student.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    pool = QueryPool(tokenizer, pool_size=5000, max_seq_len=256, device=device)
    pool.build_random(tokenizer.vocab_size)

    inv_config = InversionConfig(
        query_budget=query_budget,
        batch_size=32,
        learning_rate=1e-4,
        max_steps_per_layer=3000,
        convergence_threshold=1e-6,
        active_query_strategy="gradient_magnitude",
    )

    inverter = LayerWiseInverter(student, teacher, inv_config, device)
    inverter.set_query_pool(pool)

    result = inverter.run_progressive_inversion(
        teacher_ground_truth=ground_truth,
        output_dir=output_dir,
    )

    valid_sims = [lr.cosine_similarity for lr in result.layer_results if lr.cosine_similarity >= 0]
    valid_losses = [lr.final_loss for lr in result.layer_results]

    metrics = {
        "mean_cosine_similarity": float(np.mean(valid_sims)) if valid_sims else 0.0,
        "std_cosine_similarity": float(np.std(valid_sims)) if valid_sims else 0.0,
        "mean_final_loss": float(np.mean(valid_losses)) if valid_losses else 0.0,
        "total_queries": result.total_queries,
        "num_layers_inverted": len(result.layer_results),
        "per_layer": [
            {
                "name": lr.layer_name,
                "cosine_similarity": lr.cosine_similarity,
                "l2_distance": lr.l2_distance,
                "final_loss": lr.final_loss,
                "converged": lr.converged,
            }
            for lr in result.layer_results
        ],
    }

    del student, inverter
    torch.cuda.empty_cache()
    return metrics


# ── Plotting ─────────────────────────────────────────────────────────

def plot_defense_impact(all_results, output_dir):
    """Plot recovery degradation across all defenses."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    defense_names = list(all_results.keys())
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    for idx, (defense_name, defense_data) in enumerate(all_results.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        trials = defense_data["trials"]
        baseline_sim = defense_data["baseline"]["mean_cosine_similarity"]

        param_values = [t["param_value"] for t in trials]
        cos_sims = [t["inversion"]["mean_cosine_similarity"] for t in trials]
        utility_kl = [t["utility"]["kl_divergence"] for t in trials]

        ax.plot(param_values, cos_sims, "o-", color=colors[idx % 4],
                linewidth=2, markersize=8, label="Recovery")
        ax.axhline(y=baseline_sim, color="gray", linestyle="--", alpha=0.5, label="No defense")

        ax2 = ax.twinx()
        ax2.plot(param_values, utility_kl, "s--", color="orange", alpha=0.6, label="Utility KL")
        ax2.set_ylabel("API Utility Loss (KL)", color="orange")

        param_name = DEFENSE_CONFIGS.get(defense_name, {}).get("param_name", "param")
        ax.set_xlabel(f"{param_name}")
        ax.set_ylabel("Recovery (Cosine Sim)")
        ax.set_title(defense_name.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "defense_impact_grid.png"), dpi=150)
    plt.close()


def plot_pareto_frontier(all_results, output_dir):
    """Scatter plot: defense utility loss vs attack recovery."""
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {"logit_rounding": "o", "gaussian_noise": "s",
               "temperature_perturbation": "^", "watermarking": "D"}
    colors = {"logit_rounding": "#FF6B6B", "gaussian_noise": "#4ECDC4",
              "temperature_perturbation": "#45B7D1", "watermarking": "#96CEB4"}

    for defense_name, defense_data in all_results.items():
        for trial in defense_data["trials"]:
            util_kl = trial["utility"]["kl_divergence"]
            recovery = trial["inversion"]["mean_cosine_similarity"]
            ax.scatter(util_kl, recovery,
                       marker=markers.get(defense_name, "o"),
                       color=colors.get(defense_name, "gray"),
                       s=100, alpha=0.8, label=defense_name, zorder=5)
            ax.annotate(
                f"{trial['param_value']}",
                (util_kl, recovery),
                textcoords="offset points", xytext=(5, 5), fontsize=7,
            )

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
    ax.legend(*zip(*unique), loc="lower left")

    ax.set_xlabel("API Utility Loss (KL Divergence)")
    ax.set_ylabel("Attack Recovery (Cosine Similarity)")
    ax.set_title("Defense-Utility Pareto Frontier")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "pareto_frontier.png"), dpi=150)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Defense Evaluation")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--query_budget", type=int, default=100000,
                        help="Per-trial query budget (shortened for defense sweep)")
    parser.add_argument("--output_dir", type=str, default="results/defense_eval")
    parser.add_argument("--config", type=str, default="configs/inversion_config.yaml")
    parser.add_argument("--defenses", type=str, nargs="+",
                        default=list(DEFENSE_CONFIGS.keys()),
                        help="Defenses to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    import traceback as _tb
    args = parse_args()
    setup_logging()
    torch.manual_seed(args.seed)

    try:
        _run_main(args)
    except Exception:
        logger.error("FATAL: %s", _tb.format_exc())
        raise


def _run_main(args):
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        args.teacher_model = config.get("teacher", {}).get("model_name", args.teacher_model)
        args.student_model = config.get("student", {}).get("model_name", args.student_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Defense Evaluation ===")
    logger.info("Teacher: %s, Student: %s", args.teacher_model, args.student_model)
    logger.info("Defenses: %s", args.defenses)

    logger.info("Loading teacher model...")
    teacher_raw = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ground_truth = {
        name: param.data.clone().cpu()
        for name, param in teacher_raw.named_parameters()
    }

    # Run no-defense baseline
    logger.info("Running no-defense baseline...")
    baseline = run_inversion_trial(
        teacher_raw, args.student_model, None, ground_truth,
        tokenizer, args.query_budget, device,
        str(output_dir / "baseline"),
    )
    logger.info("Baseline: cos_sim=%.4f, queries=%d",
                baseline["mean_cosine_similarity"], baseline["total_queries"])

    all_defense_results = {}

    for defense_name in args.defenses:
        if defense_name not in DEFENSE_CONFIGS:
            logger.warning("Unknown defense: %s", defense_name)
            continue

        dcfg = DEFENSE_CONFIGS[defense_name]
        param_name = dcfg["param_name"]
        param_values = dcfg["param_values"]

        logger.info("\n=== Defense: %s ===", defense_name)
        trials = []

        for pv in param_values:
            logger.info("  Testing %s=%s...", param_name, pv)

            if defense_name == "watermarking":
                defense_fn = make_watermarking(tokenizer.vocab_size, green_ratio=pv)
            else:
                defense_fn = dcfg["factory"](pv)

            utility = measure_defense_utility(
                teacher_raw, defense_fn, tokenizer, device,
            )
            logger.info("    Utility: KL=%.4f, top1_preserved=%.4f",
                         utility["kl_divergence"], utility["top1_preserved_rate"])

            trial_dir = str(output_dir / defense_name / f"{param_name}_{pv}")
            inv_metrics = run_inversion_trial(
                teacher_raw, args.student_model, defense_fn, ground_truth,
                tokenizer, args.query_budget, device, trial_dir,
            )
            logger.info("    Recovery: cos_sim=%.4f, queries=%d",
                         inv_metrics["mean_cosine_similarity"], inv_metrics["total_queries"])

            trials.append({
                "param_name": param_name,
                "param_value": pv,
                "utility": utility,
                "inversion": inv_metrics,
            })

        all_defense_results[defense_name] = {
            "baseline": baseline,
            "trials": trials,
        }

    defense_impact = {
        "baseline_cos_sim": baseline["mean_cosine_similarity"],
        "defenses": {},
    }
    for dname, ddata in all_defense_results.items():
        defense_impact["defenses"][dname] = []
        for trial in ddata["trials"]:
            degradation = baseline["mean_cosine_similarity"] - trial["inversion"]["mean_cosine_similarity"]
            defense_impact["defenses"][dname].append({
                "param_value": trial["param_value"],
                "recovery_cos_sim": trial["inversion"]["mean_cosine_similarity"],
                "degradation": degradation,
                "utility_kl": trial["utility"]["kl_divergence"],
                "utility_top1_preserved": trial["utility"]["top1_preserved_rate"],
            })

    with open(output_dir / "defense_impact.json", "w") as f:
        json.dump(defense_impact, f, indent=2, default=str)

    with open(output_dir / "full_defense_results.json", "w") as f:
        json.dump(all_defense_results, f, indent=2, default=str)

    if all_defense_results:
        plot_defense_impact(all_defense_results, str(output_dir))
        plot_pareto_frontier(all_defense_results, str(output_dir))

    logger.info("\n=== Defense Evaluation Summary ===")
    logger.info("Baseline (no defense): cos_sim=%.4f", baseline["mean_cosine_similarity"])
    for dname, ddata in all_defense_results.items():
        logger.info("  %s:", dname)
        for trial in ddata["trials"]:
            pv = trial["param_value"]
            cs = trial["inversion"]["mean_cosine_similarity"]
            deg = baseline["mean_cosine_similarity"] - cs
            logger.info("    %s=%s: cos_sim=%.4f (degradation=%.4f)", dname, pv, cs, deg)

    logger.info("Results: %s", output_dir)
    logger.info("Defense impact: %s", output_dir / "defense_impact.json")


if __name__ == "__main__":
    main()
