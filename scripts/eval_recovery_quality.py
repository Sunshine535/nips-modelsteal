#!/usr/bin/env python3
"""Comprehensive evaluation of model recovery quality.

Three evaluation levels:
  1. Weight-level: cosine similarity, MSE, relative L2 error per layer
  2. Output-level: KL divergence, top-k token agreement, exact match rate
  3. Downstream:   GSM8K, MMLU, HumanEval accuracy

Compares KD baseline vs progressive inversion at the same query budget.

Usage:
    python scripts/eval_recovery_quality.py \
        --teacher_model Qwen/Qwen3.5-4B \
        --kd_model results/kd_baseline/best_student \
        --inversion_model results/progressive_inversion/strategy_gradient_magnitude/recovered_model \
        --output_dir results/recovery_evaluation
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def setup_distributed():
    if "RANK" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        for backend in ("nccl", "gloo"):
            try:
                dist.init_process_group(backend)
                break
            except Exception:
                if backend == "gloo":
                    raise
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size, local_rank
    return 0, 1, 0


def setup_logging(rank: int = 0):
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[%(asctime)s][Rank {rank}] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── Weight-Level Metrics ─────────────────────────────────────────────

def compute_weight_metrics(teacher_model, recovered_model):
    """Per-layer cosine similarity, MSE, relative L2 error."""
    teacher_sd = dict(teacher_model.named_parameters())
    recovered_sd = dict(recovered_model.named_parameters())
    metrics = {}

    for name in teacher_sd:
        if name not in recovered_sd:
            continue
        t_vec = teacher_sd[name].data.float().flatten()
        r_vec = recovered_sd[name].data.float().flatten()
        if t_vec.shape != r_vec.shape:
            continue

        cos_sim = F.cosine_similarity(t_vec.unsqueeze(0), r_vec.unsqueeze(0)).item()
        mse = F.mse_loss(t_vec, r_vec).item()
        l2 = (t_vec - r_vec).norm(2).item()
        rel_err = l2 / max(t_vec.norm(2).item(), 1e-10)

        metrics[name] = {
            "cosine_similarity": cos_sim,
            "mse": mse,
            "l2_distance": l2,
            "relative_error": rel_err,
            "num_params": t_vec.numel(),
        }

    return metrics


def aggregate_weight_metrics(per_layer):
    """Aggregate by component type and block depth."""
    component_groups = defaultdict(list)
    block_groups = defaultdict(list)

    for name, m in per_layer.items():
        for comp in ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "lm_head", "embed_tokens", "layernorm"]:
            if comp in name:
                component_groups[comp].append(m)
                break
        else:
            component_groups["other"].append(m)

        block_idx = None
        for part in name.split("."):
            if part.isdigit():
                block_idx = int(part)
                break
        if block_idx is not None:
            block_groups[block_idx].append(m)
        elif "lm_head" in name:
            block_groups[-1].append(m)

    by_component = {}
    for comp, mlist in component_groups.items():
        by_component[comp] = {
            "mean_cos_sim": float(np.mean([m["cosine_similarity"] for m in mlist])),
            "std_cos_sim": float(np.std([m["cosine_similarity"] for m in mlist])),
            "mean_mse": float(np.mean([m["mse"] for m in mlist])),
            "mean_rel_error": float(np.mean([m["relative_error"] for m in mlist])),
            "count": len(mlist),
        }

    by_block = {}
    for idx in sorted(block_groups.keys()):
        mlist = block_groups[idx]
        label = f"block_{idx}" if idx >= 0 else "lm_head"
        by_block[label] = {
            "mean_cos_sim": float(np.mean([m["cosine_similarity"] for m in mlist])),
            "std_cos_sim": float(np.std([m["cosine_similarity"] for m in mlist])),
            "mean_mse": float(np.mean([m["mse"] for m in mlist])),
        }

    return by_component, by_block


# ── Output-Level Metrics ─────────────────────────────────────────────

@torch.no_grad()
def compute_output_metrics(teacher, recovered, tokenizer, device,
                           num_samples=2000, batch_size=16, seq_len=256,
                           rank=0, world_size=1):
    """KL divergence, top-k agreement, exact match rate (DDP-aware)."""
    teacher.eval()
    recovered.eval()
    vocab_size = tokenizer.vocab_size
    rng = torch.Generator().manual_seed(42)

    total_tokens = 0
    total_top1_match = 0
    total_top5_overlap = 0.0
    total_top10_overlap = 0.0
    total_kl = 0.0
    total_reverse_kl = 0.0
    total_exact_seq_match = 0
    total_seqs = 0

    batch_idx = 0
    for start in tqdm(range(0, num_samples, batch_size), desc="Output metrics", disable=(rank != 0)):
        bsz = min(batch_size, num_samples - start)
        input_ids = torch.randint(3, vocab_size, (bsz, seq_len), generator=rng)

        if batch_idx % world_size != rank:
            batch_idx += 1
            continue
        batch_idx += 1

        input_ids = input_ids.to(device)
        t_logits = teacher(input_ids).logits
        r_logits = recovered(input_ids).logits

        t_preds = t_logits.argmax(-1)
        r_preds = r_logits.argmax(-1)
        total_top1_match += (t_preds == r_preds).sum().item()

        t_top5 = t_logits.topk(5, dim=-1).indices
        r_top5 = r_logits.topk(5, dim=-1).indices
        t_top10 = t_logits.topk(10, dim=-1).indices
        r_top10 = r_logits.topk(10, dim=-1).indices

        for i in range(bsz):
            total_exact_seq_match += (t_preds[i] == r_preds[i]).all().item()
            total_seqs += 1
            for j in range(seq_len):
                t5 = set(t_top5[i, j].tolist())
                r5 = set(r_top5[i, j].tolist())
                total_top5_overlap += len(t5 & r5) / 5.0
                t10 = set(t_top10[i, j].tolist())
                r10 = set(r_top10[i, j].tolist())
                total_top10_overlap += len(t10 & r10) / 10.0

        t_probs = F.softmax(t_logits, dim=-1)
        r_log_probs = F.log_softmax(r_logits, dim=-1)
        total_kl += F.kl_div(r_log_probs, t_probs, reduction="sum").item()

        r_probs = F.softmax(r_logits, dim=-1)
        t_log_probs = F.log_softmax(t_logits, dim=-1)
        total_reverse_kl += F.kl_div(t_log_probs, r_probs, reduction="sum").item()

        total_tokens += bsz * seq_len

    if world_size > 1:
        counters = torch.tensor(
            [total_top1_match, total_top5_overlap, total_top10_overlap,
             total_kl, total_reverse_kl, total_exact_seq_match, total_seqs, total_tokens],
            dtype=torch.float64, device=device,
        )
        dist.all_reduce(counters, op=dist.ReduceOp.SUM)
        (total_top1_match, total_top5_overlap, total_top10_overlap,
         total_kl, total_reverse_kl, total_exact_seq_match, total_seqs, total_tokens) = counters.tolist()

    return {
        "top1_match_rate": total_top1_match / max(total_tokens, 1),
        "top5_overlap_rate": total_top5_overlap / max(total_tokens, 1),
        "top10_overlap_rate": total_top10_overlap / max(total_tokens, 1),
        "mean_kl_divergence": total_kl / max(total_tokens, 1),
        "mean_reverse_kl": total_reverse_kl / max(total_tokens, 1),
        "exact_sequence_match_rate": total_exact_seq_match / max(total_seqs, 1),
    }


# ── Downstream Benchmarks ────────────────────────────────────────────

@torch.no_grad()
def evaluate_gsm8k(model, tokenizer, device, max_samples=200, rank=0, world_size=1):
    """Evaluate on GSM8K (math reasoning) — DDP-aware sample splitting."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        logger.warning("Could not load GSM8K: %s", e)
        return {"accuracy": -1, "note": str(e)}

    model.eval()
    correct, total = 0, 0

    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        if i % world_size != rank:
            continue
        question = ex.get("question", "")
        answer_str = ex.get("answer", "")
        match = re.search(r"####\s*(-?\d[\d,]*)", answer_str)
        if not match:
            continue
        gold = match.group(1).replace(",", "")

        prompt = (
            f"<|im_start|>system\nSolve this math problem step by step.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        numbers = re.findall(r"-?\d[\d,]*", response)
        pred = numbers[-1].replace(",", "") if numbers else ""
        if pred == gold:
            correct += 1
        total += 1

    if world_size > 1:
        counters = torch.tensor([correct, total], dtype=torch.float64, device=device)
        dist.all_reduce(counters, op=dist.ReduceOp.SUM)
        correct, total = int(counters[0].item()), int(counters[1].item())

    acc = correct / max(total, 1)
    logger.info("GSM8K: %d/%d = %.4f", correct, total, acc)
    return {"accuracy": acc, "correct": correct, "total": total}


@torch.no_grad()
def evaluate_mmlu(model, tokenizer, device, max_samples=200, rank=0, world_size=1):
    """Evaluate on MMLU (multiple choice) — DDP-aware sample splitting."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception as e:
        logger.warning("Could not load MMLU: %s", e)
        return {"accuracy": -1, "note": str(e)}

    model.eval()
    correct, total = 0, 0
    choices_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        if i % world_size != rank:
            continue
        question = ex.get("question", "")
        choices = ex.get("choices", [])
        answer_idx = ex.get("answer", -1)
        if not question or not choices or answer_idx < 0:
            continue

        gold = choices_map.get(answer_idx, "")
        choice_text = "\n".join(f"  {choices_map[j]}: {c}" for j, c in enumerate(choices))
        prompt = (
            f"<|im_start|>user\n{question}\n\n{choice_text}\n\nAnswer:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        pred = response[0].upper() if response else ""
        if pred == gold:
            correct += 1
        total += 1

    if world_size > 1:
        counters = torch.tensor([correct, total], dtype=torch.float64, device=device)
        dist.all_reduce(counters, op=dist.ReduceOp.SUM)
        correct, total = int(counters[0].item()), int(counters[1].item())

    acc = correct / max(total, 1)
    logger.info("MMLU: %d/%d = %.4f", correct, total, acc)
    return {"accuracy": acc, "correct": correct, "total": total}


@torch.no_grad()
def evaluate_humaneval_proxy(model, tokenizer, device, max_samples=50):
    """Proxy code completion evaluation (function completion)."""
    prompts = [
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        "def is_palindrome(s):\n    \"\"\"Check if string s is a palindrome.\"\"\"\n",
        "def binary_search(arr, target):\n    \"\"\"Return index of target in sorted array, or -1.\"\"\"\n",
        "def merge_sort(arr):\n    \"\"\"Sort array using merge sort.\"\"\"\n",
        "def gcd(a, b):\n    \"\"\"Return greatest common divisor of a and b.\"\"\"\n",
    ]

    model.eval()
    completions = []
    for prompt_text in prompts[:max_samples]:
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        has_return = "return" in response
        has_logic = len(response.strip()) > 20
        completions.append({
            "prompt": prompt_text.strip(),
            "completion": response[:200],
            "has_return": has_return,
            "has_logic": has_logic,
        })

    quality_rate = sum(1 for c in completions if c["has_return"] and c["has_logic"]) / max(len(completions), 1)
    return {"quality_rate": quality_rate, "completions": completions}


# ── Plotting ─────────────────────────────────────────────────────────

def plot_comparison(kd_output, inv_output, output_dir):
    """Plot KD vs Inversion comparison charts."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = ["top1_match_rate", "top5_overlap_rate", "mean_kl_divergence"]
    labels = ["Top-1 Match Rate", "Top-5 Overlap Rate", "Mean KL Divergence"]

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        kd_val = kd_output.get(metric, 0)
        inv_val = inv_output.get(metric, 0)

        x = [0, 1]
        vals = [kd_val, inv_val]
        colors = ["#4ECDC4", "#FF6B6B"]
        axes[i].bar(x, vals, color=colors, width=0.5)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(["KD Baseline", "Progressive\nInversion"])
        axes[i].set_ylabel(label)
        axes[i].set_title(label)

        for j, v in enumerate(vals):
            axes[i].text(j, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "kd_vs_inversion_output.png"), dpi=150)
    plt.close()


def plot_layer_recovery(per_layer_metrics, output_dir, label=""):
    """Plot per-block recovery quality."""
    block_sims = defaultdict(list)
    for name, m in per_layer_metrics.items():
        for part in name.split("."):
            if part.isdigit():
                block_sims[int(part)].append(m["cosine_similarity"])
                break
        else:
            if "lm_head" in name:
                block_sims[-1].append(m["cosine_similarity"])

    blocks = sorted(block_sims.keys())
    means = [np.mean(block_sims[b]) for b in blocks]
    stds = [np.std(block_sims[b]) for b in blocks]
    labels_x = [f"lm_head" if b == -1 else f"block_{b}" for b in blocks]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(blocks)), means, yerr=stds, capsize=2, alpha=0.7, color="steelblue")
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels(labels_x, rotation=90, fontsize=7)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Weight Recovery by Layer Depth{' - ' + label if label else ''}")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / f"layer_recovery_{label or 'all'}.png"), dpi=150)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Recovery Quality")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--kd_model", type=str, default="results/kd_baseline/best_student")
    parser.add_argument("--inversion_model", type=str,
                        default="results/progressive_inversion/strategy_gradient_magnitude/recovered_model")
    parser.add_argument("--output_dir", type=str, default="results/recovery_evaluation")
    parser.add_argument("--num_output_samples", type=int, default=2000)
    parser.add_argument("--downstream_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--config", type=str, default="configs/inversion_config.yaml")
    return parser.parse_args()


def load_model(path, device):
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True,
    )


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    setup_logging(rank)

    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        args.teacher_model = config.get("teacher", {}).get("model_name", args.teacher_model)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading teacher: %s (rank %d/%d)", args.teacher_model, rank, world_size)
    teacher = load_model(args.teacher_model, device)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {"teacher_model": args.teacher_model}

    model_paths = {}
    if Path(args.kd_model).exists():
        model_paths["kd_baseline"] = args.kd_model
    else:
        logger.warning("KD model not found: %s", args.kd_model)
    if Path(args.inversion_model).exists():
        model_paths["progressive_inversion"] = args.inversion_model
    else:
        logger.warning("Inversion model not found: %s", args.inversion_model)

    for model_name, model_path in model_paths.items():
        logger.info("\n=== Evaluating: %s ===", model_name)
        model = load_model(model_path, device)
        model_results = {}

        if rank == 0:
            logger.info("Computing weight-level metrics...")
            per_layer = compute_weight_metrics(teacher, model)
            by_component, by_block = aggregate_weight_metrics(per_layer)
            model_results["weight_level"] = {
                "by_component": by_component,
                "by_block": by_block,
            }
            overall_cos = np.mean([m["cosine_similarity"] for m in per_layer.values()])
            overall_mse = np.mean([m["mse"] for m in per_layer.values()])
            logger.info("  Weight: mean_cos=%.4f, mean_mse=%.6f", overall_cos, overall_mse)
            plot_layer_recovery(per_layer, str(output_dir), label=model_name)

        if world_size > 1:
            dist.barrier()

        logger.info("Computing output-level metrics...")
        output_metrics = compute_output_metrics(
            teacher, model, tokenizer, device,
            num_samples=args.num_output_samples, batch_size=args.batch_size,
            rank=rank, world_size=world_size,
        )
        model_results["output_level"] = output_metrics
        for k, v in output_metrics.items():
            logger.info("  %s: %.4f", k, v)

        logger.info("Running downstream benchmarks...")
        downstream = {}
        downstream["gsm8k"] = evaluate_gsm8k(
            model, tokenizer, device, args.downstream_samples, rank, world_size)
        downstream["mmlu"] = evaluate_mmlu(
            model, tokenizer, device, args.downstream_samples, rank, world_size)
        if rank == 0:
            downstream["humaneval_proxy"] = evaluate_humaneval_proxy(model, tokenizer, device)
        else:
            downstream["humaneval_proxy"] = {}
        if world_size > 1:
            dist.barrier()
        model_results["downstream"] = downstream

        results[model_name] = model_results
        del model
        torch.cuda.empty_cache()

    if rank == 0:
        if "kd_baseline" in results and "progressive_inversion" in results:
            plot_comparison(
                results["kd_baseline"]["output_level"],
                results["progressive_inversion"]["output_level"],
                str(output_dir),
            )

        with open(output_dir / "recovery_evaluation.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("\n=== Recovery Evaluation Summary ===")
        for model_name in model_paths:
            r = results[model_name]
            out = r.get("output_level", {})
            ds = r.get("downstream", {})
            logger.info("%s:", model_name)
            logger.info("  Output: top1=%.4f, kl=%.6f",
                         out.get("top1_match_rate", 0), out.get("mean_kl_divergence", 0))
            logger.info("  GSM8K: %.4f, MMLU: %.4f",
                         ds.get("gsm8k", {}).get("accuracy", 0),
                         ds.get("mmlu", {}).get("accuracy", 0))
        logger.info("All results saved to %s", output_dir)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
