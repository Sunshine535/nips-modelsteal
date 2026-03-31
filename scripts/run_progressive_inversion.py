#!/usr/bin/env python3
"""Full progressive parameter inversion pipeline with phased execution.

Phase 1: Invert lm_head (output projection) — most directly observable
Phase 2: Invert last transformer layer
Phase 3: Progressively invert deeper layers (output → input)

Compares three query selection strategies: random, gradient-based, Fisher-based.
Tracks weight recovery MSE per layer and downstream accuracy per phase.

Usage:
    python scripts/run_progressive_inversion.py \
        --teacher_model Qwen/Qwen3.5-4B \
        --output_dir results/progressive_inversion

    torchrun --nproc_per_node=4 scripts/run_progressive_inversion.py ...

DEPRECATED: This script uses the legacy InversionConfig/LayerWiseInverter API.
For new experiments, use run_spsi.py with the SPSIConfig-based pipeline.
"""

import argparse
import copy
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.active_query import ActiveQuerySelector, QueryPool
from src.parameter_inverter import (
    BlackBoxTeacher,
    InversionConfig,
    LayerWiseInverter,
)

logger = logging.getLogger(__name__)


def setup_logging(rank: int = 0):
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[%(asctime)s][Rank {rank}] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def extract_ground_truth(model):
    gt = {}
    for name, param in model.named_parameters():
        gt[name] = param.data.cpu().clone()
    torch.cuda.empty_cache()
    return gt


def build_query_pool(tokenizer, pool_size, max_seq_len, device, seed=42, dataset_cache=None):
    pool = QueryPool(
        tokenizer=tokenizer, pool_size=pool_size,
        max_seq_len=max_seq_len, device=device, seed=seed,
    )
    try:
        if dataset_cache is not None:
            pool.build_from_dataset(dataset_cache, text_column="text")
        else:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
            pool.build_from_dataset(ds, text_column="text")
    except Exception as e:
        logger.warning("Dataset load failed: %s. Using random pool.", e)
        pool.build_random(tokenizer.vocab_size)
    return pool


def classify_layers(student_model):
    """Classify model parameters into inversion phases.

    Handles weight-tied models where lm_head.weight shares its tensor with
    the embedding layer (tie_word_embeddings=True).  In that case
    named_parameters() only lists the shared tensor once under the embedding
    name, so a plain string match on "lm_head" finds nothing.  We detect
    this by comparing data_ptr() of the lm_head module's weight.
    """
    lm_head_params = []
    last_block_params = []
    deeper_params = []

    lm_head_ptrs: set[int] = set()
    for mod_name, module in student_model.named_modules():
        if "lm_head" in mod_name:
            for p in module.parameters():
                lm_head_ptrs.add(p.data_ptr())

    max_block_idx = -1
    for name, _ in student_model.named_parameters():
        for part in name.split("."):
            if part.isdigit():
                max_block_idx = max(max_block_idx, int(part))
                break

    for name, param in student_model.named_parameters():
        if "lm_head" in name or param.data_ptr() in lm_head_ptrs:
            lm_head_params.append((name, param))
        elif any(part == str(max_block_idx) for part in name.split(".")):
            last_block_params.append((name, param))
        elif any(part.isdigit() for part in name.split(".")):
            deeper_params.append((name, param))

    logger.info(
        "Layer classification: lm_head=%d, last_block=%d, deeper=%d (max_block=%d)",
        len(lm_head_params), len(last_block_params), len(deeper_params), max_block_idx,
    )
    return lm_head_params, last_block_params, deeper_params, max_block_idx


@torch.no_grad()
def compute_phase_metrics(student_model, teacher_model, tokenizer, device, num_samples=200):
    """Quick downstream metrics: top-1 match, KL divergence."""
    student_model.eval()
    teacher_model.eval()
    vocab_size = tokenizer.vocab_size
    rng = torch.Generator().manual_seed(123)

    total_match = 0
    total_kl = 0.0
    total_tokens = 0
    seq_len = 64
    batch_size = 4

    for _ in range(0, num_samples, batch_size):
        bsz = min(batch_size, num_samples)
        input_ids = torch.randint(3, vocab_size, (bsz, seq_len), generator=rng).to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            t_logits = teacher_model(input_ids).logits
            s_logits = student_model(input_ids).logits

        total_match += (t_logits.argmax(-1) == s_logits.argmax(-1)).sum().item()
        kl = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction="sum",
        ).item()
        total_kl += kl
        total_tokens += bsz * seq_len

    return {
        "top1_match_rate": total_match / total_tokens,
        "mean_kl_divergence": total_kl / total_tokens,
    }


def compute_weight_metrics(student_model, ground_truth, param_names):
    """Compute per-layer weight recovery metrics."""
    metrics = {}
    for name in param_names:
        if name not in ground_truth:
            continue
        for n, p in student_model.named_parameters():
            if n == name:
                gt = ground_truth[name].to(p.device).float()
                pred = p.data.float()
                cos_sim = F.cosine_similarity(
                    pred.flatten().unsqueeze(0), gt.flatten().unsqueeze(0)
                ).item()
                mse = F.mse_loss(pred, gt).item()
                l2 = (pred - gt).norm(2).item()
                rel_err = l2 / max(gt.norm(2).item(), 1e-10)
                metrics[name] = {
                    "cosine_similarity": cos_sim,
                    "mse": mse,
                    "l2_distance": l2,
                    "relative_error": rel_err,
                }
                break
    return metrics


def run_phase(
    phase_name, param_list, student_model, teacher, ground_truth,
    inv_config, query_pool, device, output_dir,
):
    """Run inversion for a single phase (group of parameters)."""
    logger.info("=== Phase: %s (%d parameter tensors) ===", phase_name, len(param_list))

    if not param_list:
        logger.info("  No parameters in this phase — skipping")
        return {
            "phase": phase_name,
            "layer_result": {
                "layer_name": "(empty)", "cosine_similarity": -1.0,
                "l2_distance": -1.0, "final_loss": 0.0,
                "num_steps": 0, "num_queries_used": 0, "converged": True,
            },
            "weight_metrics": {},
            "total_params": 0,
        }

    layer_names = [name for name, _ in param_list]
    target_params = [param for _, param in param_list]
    total_params = sum(p.numel() for p in target_params)
    logger.info("  Parameters to invert: %d", total_params)

    inverter = LayerWiseInverter(
        student_model=student_model, teacher=teacher,
        config=inv_config, device=device,
    )
    inverter.set_query_pool(query_pool)

    result = inverter.invert_layer(layer_names, target_params, ground_truth)

    weight_metrics = compute_weight_metrics(student_model, ground_truth, layer_names)

    phase_dir = Path(output_dir) / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    phase_result = {
        "phase": phase_name,
        "layer_result": {
            "layer_name": result.layer_name,
            "cosine_similarity": result.cosine_similarity,
            "final_loss": result.final_loss,
            "num_steps": result.num_steps,
            "num_queries_used": result.num_queries_used,
            "converged": result.converged,
        },
        "weight_metrics": weight_metrics,
        "total_params": total_params,
    }

    with open(phase_dir / "phase_result.json", "w") as f:
        json.dump(phase_result, f, indent=2, default=str)

    return phase_result


def run_full_inversion(
    strategy, teacher_model, student_model_name, ground_truth,
    tokenizer, args, device, output_dir,
):
    """Run complete progressive inversion with a given query strategy."""
    logger.info("=== Strategy: %s ===", strategy)
    strategy_dir = Path(output_dir) / f"strategy_{strategy}"
    strategy_dir.mkdir(parents=True, exist_ok=True)

    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    student_model.gradient_checkpointing_enable()
    for m in student_model.modules():
        if hasattr(m, "reset_parameters") and isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            m.reset_parameters()

    teacher = BlackBoxTeacher(model=teacher_model, device=device)

    query_pool = build_query_pool(
        tokenizer, args.query_pool_size, args.max_seq_len, device, args.seed,
        dataset_cache=getattr(args, '_dataset_cache', None),
    )

    per_phase_budget = args.query_budget // 3
    inv_config = InversionConfig(
        query_budget=per_phase_budget,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps_per_layer=args.max_steps_per_layer,
        convergence_threshold=1e-6,
        optimizer_type="adam",
        regularization_lambda=1e-4,
        active_query_strategy=strategy,
        active_query_pool_size=args.query_pool_size,
        seed=args.seed,
    )

    lm_head_params, last_block_params, deeper_params, max_block_idx = classify_layers(student_model)

    all_phase_results = []
    total_queries = 0

    # Phase 1: lm_head
    student_model.train()
    p1 = run_phase(
        "phase1_lm_head", lm_head_params, student_model, teacher,
        ground_truth, inv_config, query_pool, device, str(strategy_dir),
    )
    total_queries += p1["layer_result"]["num_queries_used"]
    p1_downstream = compute_phase_metrics(student_model, teacher_model, tokenizer, device)
    p1["downstream"] = p1_downstream
    all_phase_results.append(p1)
    logger.info("Phase 1 done: cos_sim=%.4f, top1_match=%.4f",
                p1["layer_result"]["cosine_similarity"], p1_downstream["top1_match_rate"])

    # Phase 2: last transformer block
    student_model.train()
    p2 = run_phase(
        "phase2_last_block", last_block_params, student_model, teacher,
        ground_truth, inv_config, query_pool, device, str(strategy_dir),
    )
    total_queries += p2["layer_result"]["num_queries_used"]
    p2_downstream = compute_phase_metrics(student_model, teacher_model, tokenizer, device)
    p2["downstream"] = p2_downstream
    all_phase_results.append(p2)
    logger.info("Phase 2 done: cos_sim=%.4f, top1_match=%.4f",
                p2["layer_result"]["cosine_similarity"], p2_downstream["top1_match_rate"])

    # Phase 3: deeper layers (grouped by block, output → input)
    block_groups = defaultdict(list)
    for name, param in deeper_params:
        block_idx = None
        for part in name.split("."):
            if part.isdigit():
                block_idx = int(part)
                break
        if block_idx is not None:
            block_groups[block_idx].append((name, param))

    remaining_budget = args.query_budget - total_queries
    sorted_blocks = sorted(block_groups.keys(), reverse=True)
    if sorted_blocks:
        budget_per_block = max(remaining_budget // len(sorted_blocks), 1000)

    for block_idx in sorted_blocks:
        if total_queries >= args.query_budget:
            logger.info("Query budget exhausted at block %d", block_idx)
            break

        block_config = InversionConfig(
            query_budget=budget_per_block,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps_per_layer=min(args.max_steps_per_layer, 5000),
            convergence_threshold=1e-6,
            optimizer_type="adam",
            regularization_lambda=1e-4,
            active_query_strategy=strategy,
            active_query_pool_size=args.query_pool_size,
            seed=args.seed,
        )

        student_model.train()
        block_result = run_phase(
            f"phase3_block_{block_idx}", block_groups[block_idx],
            student_model, teacher, ground_truth, block_config,
            query_pool, device, str(strategy_dir),
        )
        total_queries += block_result["layer_result"]["num_queries_used"]
        all_phase_results.append(block_result)

    final_downstream = compute_phase_metrics(student_model, teacher_model, tokenizer, device)

    summary = {
        "strategy": strategy,
        "total_queries": total_queries,
        "query_budget": args.query_budget,
        "phases": all_phase_results,
        "final_downstream": final_downstream,
    }

    with open(strategy_dir / "inversion_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    student_model.save_pretrained(str(strategy_dir / "recovered_model"))
    tokenizer.save_pretrained(str(strategy_dir / "recovered_model"))

    logger.info("Strategy %s complete: queries=%d, final_top1=%.4f, final_kl=%.6f",
                strategy, total_queries, final_downstream["top1_match_rate"],
                final_downstream["mean_kl_divergence"])

    del student_model
    torch.cuda.empty_cache()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Progressive Parameter Inversion")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--query_budget", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps_per_layer", type=int, default=10000)
    parser.add_argument("--query_pool_size", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="results/progressive_inversion")
    parser.add_argument("--config", type=str, default="configs/inversion_config.yaml")
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["random", "gradient_magnitude", "fisher_information"],
                        help="Query strategies to compare")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    import traceback as _tb
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    setup_logging(rank)
    torch.manual_seed(args.seed)

    try:
        _run_main(args, rank, world_size, local_rank)
    except Exception:
        logger.error("FATAL: %s", _tb.format_exc())
        raise
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def _run_main(args, rank, world_size, local_rank):
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        args.teacher_model = config.get("teacher", {}).get("model_name", args.teacher_model)
        args.student_model = config.get("student", {}).get("model_name", args.student_model)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Progressive Parameter Inversion ===")
    logger.info("Teacher: %s, Student init: %s", args.teacher_model, args.student_model)
    logger.info("Query budget: %d, Strategies: %s", args.query_budget, args.strategies)

    logger.info("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    teacher_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Pre-loading dataset for query pool...")
    try:
        from datasets import load_dataset
        args._dataset_cache = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="validation",
        )
    except Exception as e:
        logger.warning("Dataset pre-load failed: %s (will use random pool)", e)
        args._dataset_cache = None

    logger.info("Extracting ground-truth weights...")
    ground_truth = extract_ground_truth(teacher_model)

    all_strategy_results = {}
    for strategy in args.strategies:
        result = run_full_inversion(
            strategy, teacher_model, args.student_model, ground_truth,
            tokenizer, args, device, str(output_dir),
        )
        all_strategy_results[strategy] = result

    comparison = {
        "strategies": {},
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "query_budget": args.query_budget,
    }
    for strategy, result in all_strategy_results.items():
        comparison["strategies"][strategy] = {
            "total_queries": result["total_queries"],
            "final_top1_match": result["final_downstream"]["top1_match_rate"],
            "final_kl_divergence": result["final_downstream"]["mean_kl_divergence"],
            "num_phases": len(result["phases"]),
        }

    with open(output_dir / "strategy_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    logger.info("\n=== Strategy Comparison ===")
    for s, r in comparison["strategies"].items():
        logger.info("  %s: top1=%.4f, kl=%.6f, queries=%d",
                     s, r["final_top1_match"], r["final_kl_divergence"], r["total_queries"])


if __name__ == "__main__":
    main()
