#!/usr/bin/env python3
"""S-PSI: Sensitivity-Guided Progressive Suffix Inversion.

Main experiment script.  Runs suffix inversion under two regimes
(oracle-prefix and pure-logits) with multiple random initializations.

Usage:
    python scripts/run_spsi.py \
        --model_name Qwen/Qwen3.5-0.6B \
        --regime oracle \
        --num_inits 5 \
        --num_suffix_blocks 2 \
        --output_dir results/spsi_oracle

    python scripts/run_spsi.py --regime pure_logits ...
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parameter_inverter import (
    BlackBoxTeacher,
    BlockResult,
    SPSIConfig,
    SPSIResult,
    TeacherCache,
    run_spsi,
)
from src.permutation_alignment import compute_aligned_cosine

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_query_pool(tokenizer, pool_size: int, max_seq_len: int, seed: int = 42):
    """Build query input_ids from WikiText or random tokens."""
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
                text, max_length=max_seq_len, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            input_ids_list.append(tokens["input_ids"].squeeze(0))
    except Exception as e:
        logger.warning("Dataset load failed: %s. Using random tokens.", e)

    remaining = pool_size - len(input_ids_list)
    if remaining > 0:
        rng = torch.Generator().manual_seed(seed)
        random_ids = torch.randint(
            3, tokenizer.vocab_size, (remaining, max_seq_len), generator=rng
        )
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    return torch.stack(input_ids_list[:pool_size])


def randomize_suffix(
    student: torch.nn.Module,
    num_blocks: int,
    num_suffix_blocks: int,
    include_lm_head: bool = True,
):
    """Re-initialize suffix block parameters (and optionally lm_head)."""
    last_block = num_blocks - 1
    target_blocks = set(range(last_block - num_suffix_blocks + 1, last_block + 1))

    for name, param in student.named_parameters():
        reset = False
        if include_lm_head and "lm_head" in name:
            reset = True
        for part in name.split("."):
            if part.isdigit() and int(part) in target_blocks:
                reset = True
                break
        if reset:
            if param.dim() >= 2:
                torch.nn.init.kaiming_uniform_(param)
            else:
                torch.nn.init.zeros_(param)


def parse_args():
    parser = argparse.ArgumentParser(description="S-PSI Experiment")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--config", type=str, default="configs/inversion_config.yaml")
    parser.add_argument("--regime", type=str, default="oracle",
                        choices=["oracle", "pure_logits", "both"])
    parser.add_argument("--num_inits", type=int, default=5)
    parser.add_argument("--num_suffix_blocks", type=int, default=2)
    parser.add_argument("--query_budget", type=int, default=500_000)
    parser.add_argument("--pool_size", type=int, default=10_000)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Weight for sensitivity loss (0 = ablation)")
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--output_dir", type=str, default="results/spsi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_offset", type=int, default=0,
                        help="Offset for init index (for multi-GPU parallel runs)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from a checkpoint directory")
    parser.add_argument("--wrong_teacher", action="store_true",
                        help="Run wrong-teacher falsification control")
    parser.add_argument("--wrong_teacher_seed", type=int, default=9999,
                        help="Seed for wrong teacher random init")
    parser.add_argument("--heldout_fraction", type=float, default=0.2,
                        help="Fraction of query pool to hold out for validation")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    torch.manual_seed(args.seed)

    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config_yaml = yaml.safe_load(f)
        args.model_name = config_yaml.get("teacher", {}).get(
            "model_name", args.model_name
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== S-PSI: Sensitivity-Guided Progressive Suffix Inversion ===")
    logger.info("Model: %s", args.model_name)
    logger.info("Regime: %s | Inits: %d | Suffix blocks: %d",
                args.regime, args.num_inits, args.num_suffix_blocks)

    logger.info("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    teacher_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ground_truth = {
        name: param.data.cpu().clone()
        for name, param in teacher_model.named_parameters()
    }

    logger.info("Building query pool (%d inputs, max_len=%d)...",
                args.pool_size, args.max_seq_len)
    full_query_ids = build_query_pool(tokenizer, args.pool_size, args.max_seq_len, args.seed)

    n_total = len(full_query_ids)
    n_heldout = int(n_total * args.heldout_fraction)
    n_train = n_total - n_heldout
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
    query_ids = full_query_ids[perm[:n_train]]
    heldout_ids = full_query_ids[perm[n_train:]]
    logger.info("Train pool: %d, Held-out pool: %d", n_train, n_heldout)

    num_blocks = 0
    for name, _ in teacher_model.named_parameters():
        for part in name.split("."):
            if part.isdigit():
                num_blocks = max(num_blocks, int(part) + 1)
                break

    boundary_layers = list(range(
        num_blocks - args.num_suffix_blocks - 1,
        num_blocks
    ))

    num_attention_heads = teacher_model.config.num_attention_heads
    head_dim = teacher_model.config.hidden_size // num_attention_heads

    config = SPSIConfig(
        query_budget=args.query_budget,
        batch_size=args.batch_size,
        max_steps_per_block=args.max_steps,
        max_seq_len=args.max_seq_len,
        beta=args.beta,
        seed=args.seed,
        num_perturbation_positions=4,
        num_replacement_tokens=2,
    )

    teacher_bb = BlackBoxTeacher(model=teacher_model, device=device)
    cache = TeacherCache(device=device)

    logger.info("Pre-computing teacher cache...")
    cache.build(
        teacher_bb, query_ids, config,
        cache_boundary_layers=boundary_layers,
    )
    logger.info("Teacher cache ready. Total queries: %d", teacher_bb.query_count)

    regimes = []
    if args.regime in ("oracle", "both"):
        regimes.append("oracle")
    if args.regime in ("pure_logits", "both"):
        regimes.append("pure_logits")

    all_results = {}
    for regime in regimes:
        regime_dir = output_dir / f"regime_{regime}"
        regime_dir.mkdir(parents=True, exist_ok=True)
        regime_results = []

        for raw_idx in range(args.num_inits):
            init_idx = raw_idx + args.init_offset
            init_seed = args.seed + init_idx * 1000
            torch.manual_seed(init_seed)

            logger.info(
                "\n=== Regime: %s | Init %d/%d (seed=%d) ===",
                regime, raw_idx + 1, args.num_inits, init_seed,
            )

            init_dir = str(regime_dir / f"init_{init_idx}")
            summary_file = Path(init_dir) / "spsi_summary.json"
            if summary_file.exists() and not os.environ.get("FORCE_RERUN"):
                logger.info("Init %d already complete (%s exists), skipping.", init_idx, summary_file)
                try:
                    with open(summary_file) as _f:
                        _prev = json.load(_f)
                    regime_results.append(SPSIResult(
                        regime=regime, init_seed=init_seed,
                        block_results=[BlockResult(
                            block_name=b["name"],
                            per_matrix_cosine=b.get("per_matrix_cosine", {}),
                            mean_cosine=b.get("mean_cosine", -1.0),
                            final_loss=b.get("final_loss", 0),
                            num_steps=b.get("num_steps", 0),
                            num_queries=b.get("num_queries", 0),
                            converged=b.get("converged", False),
                        ) for b in _prev.get("blocks", [])],
                        total_queries=_prev.get("total_queries", 0),
                    ))
                except Exception:
                    pass
                continue

            student = AutoModelForCausalLM.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16,
                device_map={"": device}, trust_remote_code=True,
            )

            if regime == "oracle":
                randomize_suffix(
                    student, num_blocks, args.num_suffix_blocks,
                    include_lm_head=True,
                )
            else:
                for name, m in student.named_modules():
                    if hasattr(m, "reset_parameters") and isinstance(
                        m, (torch.nn.Linear, torch.nn.Embedding)
                    ):
                        m.reset_parameters()
                for name, param in student.named_parameters():
                    if "layernorm" in name.lower() or "rmsnorm" in name.lower():
                        if "weight" in name:
                            torch.nn.init.ones_(param)
                        elif "bias" in name:
                            torch.nn.init.zeros_(param)

            if hasattr(student, "gradient_checkpointing_enable"):
                student.gradient_checkpointing_enable()

            result = run_spsi(
                student=student,
                teacher=teacher_bb,
                cache=cache,
                config=config,
                regime=regime,
                num_suffix_blocks=args.num_suffix_blocks,
                ground_truth=ground_truth,
                output_dir=init_dir,
                init_seed=init_seed,
            )

            recovered_params = {
                n: p.data.cpu().clone() for n, p in student.named_parameters()
            }
            for br in result.block_results:
                block_idx_str = br.block_name.replace("block_", "")
                if block_idx_str.isdigit():
                    prefix = _find_block_prefix(student, int(block_idx_str))
                    if prefix:
                        unaligned, aligned = compute_aligned_cosine(
                            recovered_params, ground_truth,
                            prefix, num_attention_heads, head_dim,
                        )
                        br.per_matrix_cosine = {
                            "unaligned": unaligned,
                            "aligned": aligned,
                        }
                        if aligned:
                            br.mean_cosine = sum(aligned.values()) / len(aligned)

            train_metrics = _compute_heldout_loss(
                student, teacher_bb,
                query_ids[:min(len(heldout_ids), len(query_ids))],
                device,
                num_pert_positions=config.num_perturbation_positions,
                num_pert_tokens=config.num_replacement_tokens,
            )
            heldout_metrics = _compute_heldout_loss(
                student, teacher_bb, heldout_ids, device,
                num_pert_positions=config.num_perturbation_positions,
                num_pert_tokens=config.num_replacement_tokens,
            )

            generalization_report = {
                "train_logit_loss": train_metrics["heldout_logit_loss"],
                "train_sensitivity_loss": train_metrics["heldout_sensitivity_loss"],
                "heldout_logit_loss": heldout_metrics["heldout_logit_loss"],
                "heldout_sensitivity_loss": heldout_metrics["heldout_sensitivity_loss"],
                "logit_overfit_ratio": (
                    heldout_metrics["heldout_logit_loss"]
                    / max(train_metrics["heldout_logit_loss"], 1e-10)
                ),
                "sensitivity_overfit_ratio": (
                    heldout_metrics["heldout_sensitivity_loss"]
                    / max(train_metrics["heldout_sensitivity_loss"], 1e-10)
                ),
            }

            regime_results.append(result)

            with open(Path(init_dir) / "generalization_report.json", "w") as f:
                json.dump(generalization_report, f, indent=2)
            logger.info(
                "Generalization: logit overfit=%.2f, sens overfit=%.2f",
                generalization_report["logit_overfit_ratio"],
                generalization_report["sensitivity_overfit_ratio"],
            )

            del student
            torch.cuda.empty_cache()

        cross_init = _compute_cross_init_stats(regime_results)
        all_results[regime] = {
            "per_init": [
                {
                    "init_seed": r.init_seed,
                    "blocks": [
                        {
                            "name": br.block_name,
                            "per_matrix_cosine": br.per_matrix_cosine,
                            "mean_cosine": br.mean_cosine,
                        }
                        for br in r.block_results
                    ],
                }
                for r in regime_results
            ],
            "cross_init_stats": cross_init,
        }

        with open(regime_dir / "regime_summary.json", "w") as f:
            json.dump(all_results[regime], f, indent=2)

    if args.wrong_teacher:
        logger.info("\n=== Wrong-Teacher Falsification Control ===")
        wrong_teacher_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=True,
        )
        torch.manual_seed(args.wrong_teacher_seed)
        for m in wrong_teacher_model.modules():
            if hasattr(m, "reset_parameters") and isinstance(
                m, (torch.nn.Linear, torch.nn.Embedding)
            ):
                m.reset_parameters()

        wrong_teacher_params = {
            n: p.data.cpu().clone() for n, p in wrong_teacher_model.named_parameters()
        }
        del wrong_teacher_model
        torch.cuda.empty_cache()

        wrong_results = {}
        oracle_dir = output_dir / "regime_oracle"
        for block_idx in range(num_blocks - args.num_suffix_blocks, num_blocks):
            prefix = _find_block_prefix(teacher_model, block_idx)
            if not prefix:
                continue

            per_init_comparisons = []
            for init_idx in range(args.num_inits):
                init_dir = oracle_dir / f"init_{init_idx}"
                ckpt_files = list(init_dir.glob("checkpoints/ckpt_*.pt")) if init_dir.exists() else []

                init_seed = args.seed + init_idx * 1000
                torch.manual_seed(init_seed)
                student_check = AutoModelForCausalLM.from_pretrained(
                    args.model_name, torch_dtype=torch.bfloat16,
                    device_map={"": device}, trust_remote_code=True,
                )
                randomize_suffix(student_check, num_blocks, args.num_suffix_blocks, True)

                result_check = run_spsi(
                    student=student_check, teacher=teacher_bb, cache=cache,
                    config=config, regime="oracle",
                    num_suffix_blocks=args.num_suffix_blocks,
                    ground_truth=ground_truth,
                    output_dir=str(init_dir) + "_wt",
                    init_seed=init_seed,
                )

                recovered_params = {
                    n: p.data.cpu().clone() for n, p in student_check.named_parameters()
                }

                _, cos_vs_true = compute_aligned_cosine(
                    recovered_params, ground_truth,
                    prefix, num_attention_heads, head_dim,
                )
                _, cos_vs_wrong = compute_aligned_cosine(
                    recovered_params, wrong_teacher_params,
                    prefix, num_attention_heads, head_dim,
                )

                true_mean = sum(cos_vs_true.values()) / max(len(cos_vs_true), 1)
                wrong_mean = sum(cos_vs_wrong.values()) / max(len(cos_vs_wrong), 1)
                per_init_comparisons.append({
                    "init_seed": init_seed,
                    "recovered_vs_true": true_mean,
                    "recovered_vs_wrong": wrong_mean,
                    "gap": true_mean - wrong_mean,
                })
                logger.info(
                    "  Init %d Block %d: recovered_vs_true=%.4f, recovered_vs_wrong=%.4f, gap=%.4f",
                    init_idx, block_idx, true_mean, wrong_mean, true_mean - wrong_mean,
                )

                del student_check, recovered_params
                torch.cuda.empty_cache()

            wrong_results[f"block_{block_idx}"] = {
                "per_init": per_init_comparisons,
                "mean_gap": sum(c["gap"] for c in per_init_comparisons) / max(len(per_init_comparisons), 1),
            }

        all_results["wrong_teacher_control"] = wrong_results
        with open(output_dir / "wrong_teacher_control.json", "w") as f:
            json.dump(wrong_results, f, indent=2)

    with open(output_dir / "experiment_summary.json", "w") as f:
        json.dump(
            {
                "model": args.model_name,
                "regimes": list(k for k in all_results if k != "wrong_teacher_control"),
                "num_inits": args.num_inits,
                "num_suffix_blocks": args.num_suffix_blocks,
                "beta": args.beta,
                "heldout_fraction": args.heldout_fraction,
                "results": all_results,
            },
            f, indent=2,
        )

    logger.info("\n=== S-PSI Experiment Complete ===")
    for regime, data in all_results.items():
        if regime == "wrong_teacher_control":
            continue
        stats = data["cross_init_stats"]
        logger.info("Regime: %s", regime)
        for block_name, s in stats.items():
            logger.info(
                "  %s: mean_cos=%.4f +/- %.4f",
                block_name, s["mean"], s["std"],
            )

    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    logger.info("Results saved to %s", output_dir)


def _find_block_prefix(model: torch.nn.Module, block_idx: int) -> str:
    """Find the parameter name prefix for a given block index."""
    target = str(block_idx)
    for name, _ in model.named_parameters():
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == target:
                return ".".join(parts[: i + 1])
    return ""


@torch.no_grad()
def _compute_heldout_loss(
    student: torch.nn.Module,
    teacher: "BlackBoxTeacher",
    heldout_ids: torch.Tensor,
    device: str,
    num_pert_positions: int = 4,
    num_pert_tokens: int = 2,
) -> dict:
    """Compute train-comparable losses on held-out queries.

    Returns dict with:
        heldout_logit_loss: MSE on clean logits.
        heldout_sensitivity_loss: MSE on logit deltas from perturbations.
    """
    import torch.nn.functional as F

    student.eval()
    total_logit_loss = 0.0
    total_sens_loss = 0.0
    n = 0
    rng = torch.Generator().manual_seed(12345)

    for start in range(0, len(heldout_ids), 16):
        batch = heldout_ids[start : start + 16].to(device)
        bsz = batch.size(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_s = student(batch).logits
        z_t = teacher.query(batch)
        total_logit_loss += F.mse_loss(z_s.float(), z_t.float()).item() * bsz

        seq_len = batch.size(1)
        positions = torch.randint(0, seq_len, (bsz, num_pert_positions), generator=rng)
        replacements = torch.randint(3, 100, (bsz, num_pert_positions, num_pert_tokens), generator=rng)

        pert_list = []
        for b in range(bsz):
            for p_idx in range(num_pert_positions):
                pos = positions[b, p_idx].item()
                for r_idx in range(num_pert_tokens):
                    x_prime = batch[b].clone()
                    x_prime[pos] = replacements[b, p_idx, r_idx]
                    pert_list.append(x_prime)

        if pert_list:
            pert_batch = torch.stack(pert_list).to(device)
            pert_per = num_pert_positions * num_pert_tokens
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_s_pert = student(pert_batch).logits
            z_t_pert = teacher.query(pert_batch)

            delta_s = z_s_pert.float() - z_s.float().repeat_interleave(pert_per, dim=0)
            delta_t = z_t_pert.float() - z_t.float().repeat_interleave(pert_per, dim=0)
            total_sens_loss += F.mse_loss(delta_s, delta_t).item() * bsz

        n += bsz

    return {
        "heldout_logit_loss": total_logit_loss / max(n, 1),
        "heldout_sensitivity_loss": total_sens_loss / max(n, 1),
    }


def _compute_cross_init_stats(results: list) -> dict:
    """Compute mean and variance of cosine similarity across initializations."""
    import numpy as np

    block_sims: dict[str, list[float]] = {}
    for r in results:
        for br in r.block_results:
            if br.block_name not in block_sims:
                block_sims[br.block_name] = []
            block_sims[br.block_name].append(br.mean_cosine)

    stats = {}
    for name, sims in block_sims.items():
        stats[name] = {
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims)),
            "min": float(np.min(sims)),
            "max": float(np.max(sims)),
            "values": sims,
        }
    return stats


if __name__ == "__main__":
    main()
