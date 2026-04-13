#!/usr/bin/env python3
"""Retroactively compute correct per-matrix cosine similarity for Wave 1 results.

Wave 1 experiments used a buggy compute_per_matrix_cosine() that collapsed
all weight keys to "weight" and all bias keys to "bias".  This script loads
the recovered models from disk, loads the teacher (ground truth), and
recomputes the per-matrix cosines correctly.

Usage:
    python scripts/eval_wave1_cosine.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --results_dir results \
        --output eval_wave1_results.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_per_matrix_cosine(
    recovered: dict[str, torch.Tensor],
    ground_truth: dict[str, torch.Tensor],
    param_names: list[str],
) -> dict[str, float]:
    """Correct per-matrix cosine similarity."""
    metrics = {}
    for name in param_names:
        if name not in ground_truth or name not in recovered:
            continue
        gt = ground_truth[name].float().flatten()
        pred = recovered[name].float().flatten()
        if gt.shape != pred.shape:
            continue
        sim = F.cosine_similarity(pred.unsqueeze(0), gt.unsqueeze(0)).item()
        # Use abbreviated key: e.g. "self_attn.q_proj.weight" -> "q_proj.weight"
        parts = name.split(".")
        short = ".".join(parts[-2:]) if len(parts) >= 2 else name
        metrics[short] = round(sim, 6)
    return metrics


def get_block_param_names(model, block_idx: int) -> list[str]:
    """Get all parameter names belonging to a specific transformer block."""
    names = []
    for name, _ in model.named_parameters():
        for part in name.split("."):
            if part.isdigit() and int(part) == block_idx:
                names.append(name)
                break
    return names


def get_lm_head_param_names(model) -> list[str]:
    """Get lm_head parameter names.

    When tie_word_embeddings=True, lm_head.weight is an alias for
    embed_tokens.weight and does NOT appear in named_parameters().
    In this case we use embed_tokens.weight as the lm_head proxy.
    """
    names = [n for n, _ in model.named_parameters() if "lm_head" in n]
    if not names:
        # Tied embeddings — use embed_tokens as proxy
        names = [n for n, _ in model.named_parameters() if "embed_tokens" in n]
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="eval_wave1_results.json")
    parser.add_argument("--suffix_blocks", type=int, default=2,
                        help="Number of suffix blocks that were recovered")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_base = Path(args.results_dir) / "exp1_algebraic_init"

    # Load teacher (ground truth)
    logger.info("Loading teacher model: %s", args.model_name)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    ground_truth = {
        n: p.data.cpu().clone() for n, p in teacher.named_parameters()
    }
    num_blocks = teacher.config.num_hidden_layers
    del teacher
    torch.cuda.empty_cache()

    all_results = {}

    # Build param name lists once (using a meta-device model to avoid GPU memory)
    meta_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map="meta", trust_remote_code=True,
    )
    lm_names = get_lm_head_param_names(meta_model)
    block_name_map = {}  # block_idx -> list[str]
    for offset in range(args.suffix_blocks):
        block_idx = num_blocks - 1 - offset
        block_name_map[block_idx] = get_block_param_names(meta_model, block_idx)
    del meta_model

    # Scan for completed experiment directories
    for init_method in ["random", "alg_clean", "alg_aug"]:
        method_dir = results_base / init_method
        if not method_dir.exists():
            continue

        for seed_dir in sorted(method_dir.glob("seed_*")):
            seed = seed_dir.name  # e.g. "seed_42"
            recovered_dir = seed_dir / "regime_oracle" / "init_0" / "recovered_model"
            summary_file = seed_dir / "regime_oracle" / "init_0" / "spsi_summary.json"

            if not recovered_dir.exists():
                logger.info("Skipping %s/%s — no recovered model", init_method, seed)
                continue

            logger.info("Evaluating %s/%s ...", init_method, seed)

            # Load recovered model
            recovered_model = AutoModelForCausalLM.from_pretrained(
                str(recovered_dir), torch_dtype=torch.bfloat16,
                device_map={"": device}, trust_remote_code=True,
            )
            recovered = {
                n: p.data.cpu().clone() for n, p in recovered_model.named_parameters()
            }
            del recovered_model
            torch.cuda.empty_cache()

            exp_key = f"{init_method}/{seed}"
            all_results[exp_key] = {"blocks": {}}

            # Evaluate lm_head
            lm_cos = compute_per_matrix_cosine(recovered, ground_truth, lm_names)
            lm_mean = sum(lm_cos.values()) / max(len(lm_cos), 1)
            all_results[exp_key]["lm_head"] = {
                "per_matrix": lm_cos,
                "mean_cosine": round(lm_mean, 6),
            }

            # Evaluate suffix blocks
            for block_idx, block_names in sorted(block_name_map.items(), reverse=True):
                block_cos = compute_per_matrix_cosine(
                    recovered, ground_truth, block_names
                )
                block_mean = sum(block_cos.values()) / max(len(block_cos), 1)
                all_results[exp_key]["blocks"][f"block_{block_idx}"] = {
                    "per_matrix": block_cos,
                    "mean_cosine": round(block_mean, 6),
                }

            # Read original summary for context
            if summary_file.exists():
                with open(summary_file) as f:
                    orig = json.load(f)
                all_results[exp_key]["total_queries"] = orig.get("total_queries", 0)
                all_results[exp_key]["original_summary"] = orig

            logger.info(
                "  %s: lm_head=%.4f, blocks=%s",
                exp_key, lm_mean,
                {k: v["mean_cosine"] for k, v in all_results[exp_key]["blocks"].items()}
            )

    # Write results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("WAVE 1 PER-MATRIX COSINE SIMILARITY (CORRECTED)")
    print("=" * 80)
    for exp_key, data in sorted(all_results.items()):
        print(f"\n{exp_key}:")
        print(f"  lm_head mean: {data['lm_head']['mean_cosine']:.4f}")
        for bk, bv in data["blocks"].items():
            print(f"  {bk} mean: {bv['mean_cosine']:.4f}")
            for k, v in sorted(bv["per_matrix"].items()):
                print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
