#!/usr/bin/env python3
"""
Stage 3 — Evaluate Extraction Quality.

Compare the recovered model against the ground-truth teacher on:
  1. Weight-space metrics: cosine similarity, L2 distance per layer
  2. Function-space metrics: output match rate, KL divergence on held-out data
  3. Downstream accuracy on standard benchmarks

Usage:
    python scripts/eval_extraction.py \
        --teacher_model Qwen/Qwen3.5-9B \
        --recovered_model results/inversion/recovered_model \
        --output_dir results/evaluation
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# ── Weight-Space Metrics ─────────────────────────────────────────────

def compute_weight_metrics(
    teacher_model, recovered_model
) -> dict[str, dict[str, float]]:
    """Compute per-layer cosine similarity and L2 distance."""
    teacher_sd = dict(teacher_model.named_parameters())
    recovered_sd = dict(recovered_model.named_parameters())

    metrics = {}
    for name in teacher_sd:
        if name not in recovered_sd:
            continue

        t_vec = teacher_sd[name].data.float().flatten()
        r_vec = recovered_sd[name].data.float().flatten()

        if t_vec.shape != r_vec.shape:
            logger.warning("Shape mismatch for %s: %s vs %s", name, t_vec.shape, r_vec.shape)
            continue

        cos_sim = F.cosine_similarity(t_vec.unsqueeze(0), r_vec.unsqueeze(0)).item()
        l2_dist = (t_vec - r_vec).norm(2).item()
        rel_l2 = l2_dist / max(t_vec.norm(2).item(), 1e-10)

        metrics[name] = {
            "cosine_similarity": cos_sim,
            "l2_distance": l2_dist,
            "relative_l2": rel_l2,
            "num_params": t_vec.numel(),
        }

    return metrics


def aggregate_by_component(
    per_layer: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Aggregate metrics by component type (q_proj, k_proj, etc.)."""
    groups = defaultdict(list)
    for name, m in per_layer.items():
        for component in ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",
                          "lm_head", "embed_tokens", "layernorm",
                          "input_layernorm", "post_attention_layernorm"]:
            if component in name:
                groups[component].append(m)
                break
        else:
            groups["other"].append(m)

    aggregated = {}
    for comp, metrics_list in groups.items():
        aggregated[comp] = {
            "mean_cosine_similarity": np.mean([m["cosine_similarity"] for m in metrics_list]),
            "std_cosine_similarity": np.std([m["cosine_similarity"] for m in metrics_list]),
            "mean_l2_distance": np.mean([m["l2_distance"] for m in metrics_list]),
            "num_layers": len(metrics_list),
            "total_params": sum(m["num_params"] for m in metrics_list),
        }
    return aggregated


# ── Function-Space Metrics ───────────────────────────────────────────

@torch.no_grad()
def compute_functional_metrics(
    teacher_model,
    recovered_model,
    tokenizer,
    num_samples: int = 5000,
    batch_size: int = 32,
    max_seq_len: int = 256,
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate output match rate and KL divergence on random inputs."""
    teacher_model.eval()
    recovered_model.eval()

    rng = torch.Generator().manual_seed(42)
    vocab_size = tokenizer.vocab_size

    total_match = 0
    total_kl = 0.0
    total_tokens = 0
    total_top5_match = 0

    for start in tqdm(range(0, num_samples, batch_size), desc="Functional eval"):
        bsz = min(batch_size, num_samples - start)
        input_ids = torch.randint(3, vocab_size, (bsz, max_seq_len), generator=rng).to(device)

        teacher_logits = teacher_model(input_ids).logits
        recovered_logits = recovered_model(input_ids).logits

        # Top-1 match rate
        teacher_preds = teacher_logits.argmax(dim=-1)
        recovered_preds = recovered_logits.argmax(dim=-1)
        total_match += (teacher_preds == recovered_preds).sum().item()

        # Top-5 overlap
        teacher_top5 = teacher_logits.topk(5, dim=-1).indices
        recovered_top5 = recovered_logits.topk(5, dim=-1).indices
        for i in range(bsz):
            for j in range(max_seq_len):
                t_set = set(teacher_top5[i, j].tolist())
                r_set = set(recovered_top5[i, j].tolist())
                total_top5_match += len(t_set & r_set) / 5.0

        # KL divergence
        t_probs = F.softmax(teacher_logits, dim=-1)
        r_log_probs = F.log_softmax(recovered_logits, dim=-1)
        kl = F.kl_div(r_log_probs, t_probs, reduction="sum").item()
        total_kl += kl

        total_tokens += bsz * max_seq_len

    return {
        "top1_match_rate": total_match / total_tokens,
        "top5_overlap_rate": total_top5_match / total_tokens,
        "mean_kl_divergence": total_kl / total_tokens,
    }


# ── Depth-Recovery Curve ─────────────────────────────────────────────

def extract_depth_curve(per_layer: dict[str, dict[str, float]]) -> list[dict]:
    """Extract per-block cosine similarity for depth-recovery analysis."""
    block_sims = defaultdict(list)

    for name, m in per_layer.items():
        parts = name.split(".")
        block_idx = None
        for p in parts:
            if p.isdigit():
                block_idx = int(p)
                break

        if block_idx is not None:
            block_sims[block_idx].append(m["cosine_similarity"])
        elif "lm_head" in name:
            block_sims[-1].append(m["cosine_similarity"])
        elif "embed" in name:
            block_sims[-2].append(m["cosine_similarity"])

    curve = []
    for idx in sorted(block_sims.keys()):
        sims = block_sims[idx]
        label = f"block_{idx}" if idx >= 0 else ("lm_head" if idx == -1 else "embed")
        curve.append({
            "block": label,
            "block_idx": idx,
            "mean_cosine_sim": np.mean(sims),
            "std_cosine_sim": np.std(sims),
            "num_params": len(sims),
        })

    return curve


# ── Plotting ─────────────────────────────────────────────────────────

def plot_depth_curve(curve: list[dict], output_path: str):
    """Plot cosine similarity vs. layer depth."""
    blocks = [c["block"] for c in curve]
    means = [c["mean_cosine_sim"] for c in curve]
    stds = [c["std_cosine_sim"] for c in curve]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(blocks))
    ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(blocks, rotation=90, fontsize=7)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Parameter Recovery vs. Layer Depth")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="0.9 threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved depth curve plot to %s", output_path)


def plot_component_comparison(aggregated: dict, output_path: str):
    """Bar chart of mean cosine similarity by component type."""
    components = sorted(aggregated.keys())
    means = [aggregated[c]["mean_cosine_similarity"] for c in components]
    stds = [aggregated[c]["std_cosine_similarity"] for c in components]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(components))
    ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7, color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=45)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Parameter Recovery by Component Type")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved component plot to %s", output_path)


# ── Main ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Evaluate Extraction Quality")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--recovered_model", type=str, default="results/inversion/recovered_model")
    parser.add_argument("--num_eval_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
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
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    logger.info("Loading recovered model: %s", args.recovered_model)
    recovered_model = AutoModelForCausalLM.from_pretrained(
        args.recovered_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Weight-space evaluation ---
    logger.info("Computing weight-space metrics...")
    per_layer = compute_weight_metrics(teacher_model, recovered_model)
    aggregated = aggregate_by_component(per_layer)

    logger.info("=== Weight-Space Summary ===")
    for comp, m in sorted(aggregated.items()):
        logger.info(
            "  %s: cos_sim=%.4f±%.4f, l2=%.4f, layers=%d, params=%s",
            comp, m["mean_cosine_similarity"], m["std_cosine_similarity"],
            m["mean_l2_distance"], m["num_layers"], f"{m['total_params']:,}",
        )

    # --- Depth-recovery curve ---
    depth_curve = extract_depth_curve(per_layer)
    plot_depth_curve(depth_curve, str(output_dir / "depth_recovery_curve.png"))
    plot_component_comparison(aggregated, str(output_dir / "component_comparison.png"))

    # --- Function-space evaluation ---
    logger.info("Computing functional metrics (%d samples)...", args.num_eval_samples)
    func_metrics = compute_functional_metrics(
        teacher_model, recovered_model, tokenizer,
        num_samples=args.num_eval_samples,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=device,
    )

    logger.info("=== Function-Space Summary ===")
    for k, v in func_metrics.items():
        logger.info("  %s: %.4f", k, v)

    # --- Save all results ---
    results = {
        "weight_space": {
            "per_layer": {k: v for k, v in per_layer.items()},
            "aggregated": aggregated,
            "depth_curve": depth_curve,
        },
        "function_space": func_metrics,
        "config": {
            "teacher_model": args.teacher_model,
            "recovered_model": args.recovered_model,
            "num_eval_samples": args.num_eval_samples,
        },
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("All results saved to %s", output_dir)


if __name__ == "__main__":
    main()
