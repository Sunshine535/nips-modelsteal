#!/usr/bin/env python3
"""
Stage 4 — Defense Evaluation.

Evaluate how different API-side defenses degrade the parameter inversion
attack. For each defense mechanism:
  1. Wrap the teacher with the defense
  2. Run the inversion pipeline
  3. Measure recovery quality degradation

Defenses tested:
  - logit_rounding: Round output logits to k decimal places
  - gaussian_noise: Add N(0, σ²) noise to logits
  - temperature_perturbation: Randomly perturb softmax temperature
  - topk_masking: Return only top-K logits (rest zeroed)
  - watermarking: Kirchenbauer-style green/red token lists

Usage:
    python scripts/defense_evaluation.py \
        --teacher_model Qwen/Qwen3.5-9B \
        --student_model Qwen/Qwen3.5-0.8B \
        --defense_type logit_rounding \
        --output_dir results/defense/logit_rounding
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


def load_config(config_path: str) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# ── Defense Functions ────────────────────────────────────────────────

def make_logit_rounding_defense(decimal_places: int = 4) -> Callable:
    """Round logits to specified decimal places."""
    factor = 10 ** decimal_places
    def defense(logits: torch.Tensor) -> torch.Tensor:
        return torch.round(logits * factor) / factor
    return defense


def make_gaussian_noise_defense(sigma: float = 0.1) -> Callable:
    """Add Gaussian noise to logits."""
    def defense(logits: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(logits) * sigma
        return logits + noise
    return defense


def make_temperature_perturbation_defense(delta: float = 0.3) -> Callable:
    """Randomly perturb softmax temperature before returning logits."""
    def defense(logits: torch.Tensor) -> torch.Tensor:
        temp = 1.0 + (torch.rand(1, device=logits.device).item() * 2 - 1) * delta
        return logits / temp
    return defense


def make_topk_masking_defense(k: int = 100) -> Callable:
    """Return only top-K logit values, zero out the rest."""
    def defense(logits: torch.Tensor) -> torch.Tensor:
        topk_values, topk_indices = logits.topk(k, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(-1, topk_indices, topk_values)
        return masked
    return defense


def make_watermarking_defense(
    vocab_size: int, green_ratio: float = 0.5, bias: float = 2.0, seed: int = 42
) -> Callable:
    """Kirchenbauer-style watermarking: bias green-listed tokens."""
    rng = torch.Generator().manual_seed(seed)
    green_size = int(vocab_size * green_ratio)
    perm = torch.randperm(vocab_size, generator=rng)
    green_list = set(perm[:green_size].tolist())
    green_mask = torch.zeros(vocab_size)
    green_mask[perm[:green_size]] = bias

    def defense(logits: torch.Tensor) -> torch.Tensor:
        device = logits.device
        mask = green_mask.to(device)
        return logits + mask.unsqueeze(0).unsqueeze(0).expand_as(logits)
    return defense


DEFENSE_REGISTRY = {
    "logit_rounding": {
        "factory": make_logit_rounding_defense,
        "param_name": "decimal_places",
        "param_values": [2, 4, 6],
    },
    "gaussian_noise": {
        "factory": make_gaussian_noise_defense,
        "param_name": "sigma",
        "param_values": [0.01, 0.1, 1.0],
    },
    "temperature_perturbation": {
        "factory": make_temperature_perturbation_defense,
        "param_name": "delta",
        "param_values": [0.1, 0.3, 0.5],
    },
    "topk_masking": {
        "factory": make_topk_masking_defense,
        "param_name": "k",
        "param_values": [50, 100, 1000],
    },
    "watermarking": {
        "factory": make_watermarking_defense,
        "param_name": "green_ratio",
        "param_values": [0.3, 0.5, 0.7],
    },
}


# ── API Utility Measurement ──────────────────────────────────────────

@torch.no_grad()
def measure_api_utility(
    teacher_model,
    defense_fn: Optional[Callable],
    tokenizer,
    num_samples: int = 1000,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Measure how much the defense degrades normal API utility.
    Compare clean vs defended outputs on the same inputs.
    """
    teacher_model.eval()
    vocab_size = tokenizer.vocab_size
    rng = torch.Generator().manual_seed(123)

    total_kl = 0.0
    total_top1_preserved = 0
    total_tokens = 0

    batch_size = 32
    seq_len = 128

    for _ in range(0, num_samples, batch_size):
        bsz = min(batch_size, num_samples)
        input_ids = torch.randint(3, vocab_size, (bsz, seq_len), generator=rng).to(device)

        clean_logits = teacher_model(input_ids).logits
        defended_logits = defense_fn(clean_logits) if defense_fn else clean_logits

        clean_probs = F.softmax(clean_logits, dim=-1)
        defended_log_probs = F.log_softmax(defended_logits, dim=-1)

        kl = F.kl_div(defended_log_probs, clean_probs, reduction="sum").item()
        total_kl += kl

        clean_preds = clean_logits.argmax(dim=-1)
        defended_preds = defended_logits.argmax(dim=-1)
        total_top1_preserved += (clean_preds == defended_preds).sum().item()
        total_tokens += bsz * seq_len

    return {
        "kl_divergence": total_kl / total_tokens,
        "top1_preserved_rate": total_top1_preserved / total_tokens,
    }


# ── Run Single Defense Evaluation ────────────────────────────────────

def run_defense_trial(
    teacher_raw,
    student_model_name: str,
    defense_fn: Optional[Callable],
    defense_name: str,
    param_value,
    query_budget: int,
    device: str,
    tokenizer,
    output_dir: str,
) -> dict:
    """Run inversion under a single defense configuration."""
    teacher = BlackBoxTeacher(model=teacher_raw, device=device, defense_fn=defense_fn)

    student = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    ground_truth = {
        name: param.data.clone().cpu()
        for name, param in teacher_raw.named_parameters()
    }

    inv_config = InversionConfig(
        query_budget=query_budget,
        batch_size=32,
        learning_rate=1e-4,
        max_steps_per_layer=2000,
        active_query_strategy="gradient_magnitude",
    )

    inverter = LayerWiseInverter(student, teacher, inv_config, device)

    pool = QueryPool(tokenizer, pool_size=5000, max_seq_len=256, device=device)
    pool.build_random(tokenizer.vocab_size)
    inverter.set_query_pool(pool)

    result = inverter.run_progressive_inversion(
        teacher_ground_truth=ground_truth,
        output_dir=output_dir,
        regime="pure_logits",
    )

    mean_cos_sim = np.mean([
        lr.cosine_similarity for lr in result.layer_results
        if lr.cosine_similarity >= 0
    ]) if result.layer_results else 0.0

    return {
        "defense": defense_name,
        "param_value": param_value,
        "mean_cosine_similarity": float(mean_cos_sim),
        "total_queries": result.total_queries,
        "num_layers_inverted": len(result.layer_results),
        "layer_details": [
            {
                "name": lr.layer_name,
                "cosine_similarity": lr.cosine_similarity,
                "final_loss": lr.final_loss,
            }
            for lr in result.layer_results
        ],
    }


# ── Plotting ─────────────────────────────────────────────────────────

def plot_defense_tradeoff(
    results: list[dict],
    utility_results: list[dict],
    defense_name: str,
    output_path: str,
):
    """Plot defense strength (attack degradation) vs API utility."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    param_values = [r["param_value"] for r in results]
    cos_sims = [r["mean_cosine_similarity"] for r in results]

    ax1.plot(param_values, cos_sims, "o-", color="red", linewidth=2, markersize=8)
    ax1.set_xlabel(f"Defense Parameter")
    ax1.set_ylabel("Attack Recovery (Cosine Sim)")
    ax1.set_title(f"{defense_name}: Attack Success vs. Defense Strength")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    if utility_results:
        utils_kl = [u["kl_divergence"] for u in utility_results]
        ax2.plot(utils_kl, cos_sims, "s-", color="purple", linewidth=2, markersize=8)
        for i, pv in enumerate(param_values):
            ax2.annotate(f"{pv}", (utils_kl[i], cos_sims[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax2.set_xlabel("API Utility Degradation (KL)")
        ax2.set_ylabel("Attack Recovery (Cosine Sim)")
        ax2.set_title("Pareto Frontier: Defense vs. Utility")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved defense tradeoff plot to %s", output_path)


# ── Main ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4: Defense Evaluation")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--defense_type", type=str, default="logit_rounding",
                        choices=list(DEFENSE_REGISTRY.keys()))
    parser.add_argument("--query_budget", type=int, default=100000)
    parser.add_argument("--output_dir", type=str, default="results/defense")
    parser.add_argument("--config", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading teacher: %s", args.teacher_model)
    teacher_raw = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    defense_spec = DEFENSE_REGISTRY[args.defense_type]
    factory = defense_spec["factory"]
    param_name = defense_spec["param_name"]
    param_values = defense_spec["param_values"]

    # --- Run no-defense baseline ---
    logger.info("Running no-defense baseline...")
    baseline = run_defense_trial(
        teacher_raw, args.student_model, None, "no_defense", 0,
        args.query_budget, device, tokenizer,
        str(output_dir / "baseline"),
    )
    logger.info("Baseline cos_sim: %.4f", baseline["mean_cosine_similarity"])

    # --- Run each defense configuration ---
    all_results = [baseline]
    utility_results = [{"kl_divergence": 0.0, "top1_preserved_rate": 1.0}]

    for pv in param_values:
        logger.info("Testing %s with %s=%s...", args.defense_type, param_name, pv)

        if args.defense_type == "watermarking":
            defense_fn = factory(vocab_size=tokenizer.vocab_size, **{param_name: pv})
        else:
            defense_fn = factory(**{param_name: pv})

        utility = measure_api_utility(
            teacher_raw, defense_fn, tokenizer, num_samples=500, device=device
        )
        utility_results.append(utility)
        logger.info("  API utility: KL=%.4f, top1_preserved=%.4f",
                    utility["kl_divergence"], utility["top1_preserved_rate"])

        trial_result = run_defense_trial(
            teacher_raw, args.student_model, defense_fn, args.defense_type, pv,
            args.query_budget, device, tokenizer,
            str(output_dir / f"{param_name}_{pv}"),
        )
        all_results.append(trial_result)
        logger.info("  Recovery cos_sim: %.4f", trial_result["mean_cosine_similarity"])

    # --- Save results ---
    summary = {
        "defense_type": args.defense_type,
        "results": all_results,
        "utility": utility_results,
    }
    with open(output_dir / "defense_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # --- Plot ---
    defense_only = [r for r in all_results if r["defense"] != "no_defense"]
    if defense_only:
        plot_defense_tradeoff(
            defense_only,
            utility_results[1:],
            args.defense_type,
            str(output_dir / "defense_tradeoff.png"),
        )

    logger.info("=== Defense Evaluation Summary ===")
    for r in all_results:
        logger.info(
            "  %s (%s=%s): cos_sim=%.4f, queries=%d",
            r["defense"], param_name, r["param_value"],
            r["mean_cosine_similarity"], r["total_queries"],
        )

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
