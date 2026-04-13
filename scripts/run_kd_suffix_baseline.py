#!/usr/bin/env python3
"""Suffix-matched KD baseline: fair comparison against S-PSI.

Only trains the SAME suffix blocks as S-PSI (lm_head + last N blocks),
using the SAME query pool, budget, and evaluation metrics.

This answers: "Does S-PSI's Gramian-guided approach outperform
naive logit-matching KD on the same parameter subset?"

Usage:
    python scripts/run_kd_suffix_baseline.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --num_suffix_blocks 2 \
        --pool_size 512 \
        --max_steps 2000 \
        --output_dir results/kd_suffix_baseline
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.permutation_alignment import compute_aligned_cosine, compute_lm_head_aligned_cosine

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_query_pool(tokenizer, pool_size, max_seq_len, seed=42):
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
        rng = torch.Generator().manual_seed(seed + 1)
        random_ids = torch.randint(3, tokenizer.vocab_size, (remaining, max_seq_len), generator=rng)
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    return torch.stack(input_ids_list[:pool_size])


def get_suffix_param_names(model, num_suffix_blocks, include_lm_head=True):
    """Get parameter names for suffix blocks + lm_head (same as S-PSI target)."""
    num_blocks = 0
    for name, _ in model.named_parameters():
        for part in name.split("."):
            if part.isdigit():
                num_blocks = max(num_blocks, int(part) + 1)
                break

    target_blocks = set(range(num_blocks - num_suffix_blocks, num_blocks))
    suffix_names = []
    for name, _ in model.named_parameters():
        is_suffix = False
        if include_lm_head and "lm_head" in name:
            is_suffix = True
        for part in name.split("."):
            if part.isdigit() and int(part) in target_blocks:
                is_suffix = True
                break
        if is_suffix:
            suffix_names.append(name)

    return suffix_names, target_blocks, num_blocks


def reset_suffix_params(student, target_blocks, include_lm_head=True):
    """Randomize only suffix block parameters (same init as S-PSI random baseline)."""
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
                if "norm" in name.lower():
                    torch.nn.init.ones_(param)
                else:
                    torch.nn.init.zeros_(param)


@torch.no_grad()
def collect_teacher_logits(teacher_model, query_ids, batch_size, device,
                           suffix_positions=32):
    """Collect teacher logits for last K suffix positions (matching S-PSI)."""
    dataset = TensorDataset(query_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits = []
    total_queries = 0

    for (batch_ids,) in loader:
        batch_ids = batch_ids.to(device)
        logits = teacher_model(batch_ids).logits
        # Only keep last K positions (matching S-PSI's logit_suffix_positions)
        if suffix_positions > 0 and logits.size(1) > suffix_positions:
            logits = logits[:, -suffix_positions:, :]
        all_logits.append(logits.cpu())
        total_queries += batch_ids.size(0)

    logger.info("Collected teacher logits: %d queries, suffix_positions=%d",
                total_queries, suffix_positions)
    return torch.cat(all_logits, dim=0), total_queries


def train_suffix_kd(student, teacher_logits, query_ids, suffix_param_names,
                    args, device):
    """Train suffix parameters via logit-matching KD."""
    # Freeze non-suffix parameters
    for name, param in student.named_parameters():
        param.requires_grad = name in suffix_param_names

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    logger.info("Trainable: %d / %d (%.2f%%)", trainable, total,
                100.0 * trainable / total)

    suffix_positions = args.logit_suffix_positions

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-5,
    )

    dataset = TensorDataset(query_ids, teacher_logits)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True)

    student.train()
    if hasattr(student, "gradient_checkpointing_disable"):
        student.gradient_checkpointing_disable()

    best_loss = float("inf")
    patience_counter = 0
    step = 0
    losses = []

    start_time = time.time()

    while step < args.max_steps:
        for batch_ids, batch_teacher_logits in loader:
            if step >= args.max_steps:
                break

            batch_ids = batch_ids.to(device)
            batch_teacher_logits = batch_teacher_logits.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_logits = student(batch_ids).logits

            if suffix_positions > 0 and student_logits.size(1) > suffix_positions:
                student_logits = student_logits[:, -suffix_positions:, :]

            # MSE loss on logits (same as S-PSI's logit loss)
            loss = F.mse_loss(student_logits.float(), batch_teacher_logits.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)
            step += 1

            if loss_val < best_loss - args.convergence_threshold:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logger.info("Early stopping at step %d (patience=%d)",
                            step, args.patience)
                elapsed = time.time() - start_time
                return step, best_loss, True, elapsed, losses

            if step % 200 == 0:
                logger.info("Step %d/%d | loss=%.6f | best=%.6f | patience=%d/%d",
                            step, args.max_steps, loss_val, best_loss,
                            patience_counter, args.patience)

    elapsed = time.time() - start_time
    return step, best_loss, False, elapsed, losses


def main():
    parser = argparse.ArgumentParser(description="Suffix-matched KD baseline")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num_suffix_blocks", type=int, default=2)
    parser.add_argument("--pool_size", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--convergence_threshold", type=float, default=1e-7)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--logit_suffix_positions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inits", type=int, default=3,
                        help="Number of random initializations")
    parser.add_argument("--output_dir", type=str,
                        default="results/kd_suffix_baseline")
    args = parser.parse_args()

    setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("=== Suffix-Matched KD Baseline ===")
    logger.info("Model: %s, suffix_blocks=%d, K=%d",
                args.model_name, args.num_suffix_blocks,
                args.logit_suffix_positions)

    # Load teacher
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    teacher_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ground_truth = {n: p.data.cpu().clone() for n, p in teacher_model.named_parameters()}
    if getattr(teacher_model.config, "tie_word_embeddings", False):
        lm_head = getattr(teacher_model, "lm_head", None)
        if lm_head is not None and "lm_head.weight" not in ground_truth:
            ground_truth["lm_head.weight"] = lm_head.weight.data.cpu().clone()

    # Build query pool
    query_ids = build_query_pool(tokenizer, args.pool_size, args.max_seq_len,
                                 args.seed)
    logger.info("Query pool: %s", query_ids.shape)

    # Collect teacher logits once
    teacher_logits, num_queries = collect_teacher_logits(
        teacher_model, query_ids, batch_size=32, device=device,
        suffix_positions=args.logit_suffix_positions,
    )
    logger.info("Teacher logits shape: %s (queries used: %d)",
                teacher_logits.shape, num_queries)

    # Get suffix structure
    suffix_names_set = set()
    _, target_blocks, num_blocks = get_suffix_param_names(
        teacher_model, args.num_suffix_blocks)
    for name, _ in teacher_model.named_parameters():
        is_suffix = False
        if "lm_head" in name:
            is_suffix = True
        for part in name.split("."):
            if part.isdigit() and int(part) in target_blocks:
                is_suffix = True
                break
        if is_suffix:
            suffix_names_set.add(name)

    suffix_param_names = list(suffix_names_set)
    logger.info("Suffix blocks: %s (from %d total)", sorted(target_blocks),
                num_blocks)
    logger.info("Suffix params: %d names", len(suffix_param_names))

    num_attention_heads = teacher_model.config.num_attention_heads
    head_dim = teacher_model.config.hidden_size // num_attention_heads

    # Free teacher GPU memory
    del teacher_model
    torch.cuda.empty_cache()

    all_results = []

    for init_idx in range(args.num_inits):
        init_seed = args.seed + init_idx * 1000
        torch.manual_seed(init_seed)

        logger.info("\n=== Init %d/%d (seed=%d) ===", init_idx + 1,
                    args.num_inits, init_seed)

        # Fresh student
        student = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=True,
        )

        # Untie lm_head
        lm_head = getattr(student, "lm_head", None)
        embed = getattr(getattr(student, "model", None), "embed_tokens", None)
        if lm_head and embed and lm_head.weight.data_ptr() == embed.weight.data_ptr():
            lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
            if hasattr(student, "config"):
                student.config.tie_word_embeddings = False

        # Randomize suffix blocks only
        reset_suffix_params(student, target_blocks, include_lm_head=True)

        # Train
        num_steps, final_loss, converged, elapsed, losses = train_suffix_kd(
            student, teacher_logits, query_ids, suffix_param_names,
            args, device,
        )

        # Evaluate: per-matrix aligned cosine for each suffix block
        post_params = {n: p.data.cpu().clone() for n, p in student.named_parameters()}

        block_results = []
        for block_idx in sorted(target_blocks):
            prefix = None
            for name in post_params:
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == str(block_idx):
                        prefix = ".".join(parts[:i + 1])
                        break
                if prefix:
                    break

            if prefix:
                _, aligned = compute_aligned_cosine(
                    post_params, ground_truth, prefix,
                    num_attention_heads, head_dim,
                )
                mean_cos = sum(aligned.values()) / len(aligned) if aligned else 0
            else:
                aligned, mean_cos = {}, 0

            block_results.append({
                "block_idx": block_idx,
                "prefix": prefix,
                "mean_cosine": float(mean_cos),
                "per_matrix": {k: float(v) for k, v in aligned.items()},
            })
            logger.info("  Block %d: mean_cos=%.4f", block_idx, mean_cos)

        # lm_head evaluation
        lm_head_cos = compute_lm_head_aligned_cosine(post_params, ground_truth)
        logger.info("  lm_head: raw_cos=%.4f, aligned_cos=%.4f",
                    lm_head_cos.get("raw_cosine", 0),
                    lm_head_cos.get("aligned_cosine", 0))

        result = {
            "init_idx": init_idx,
            "init_seed": init_seed,
            "num_steps": num_steps,
            "final_loss": float(final_loss),
            "converged": converged,
            "elapsed_seconds": elapsed,
            "num_queries": num_queries,
            "blocks": block_results,
            "lm_head": {k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in lm_head_cos.items()
                        if k != "top_singular_values"},
            "lm_head_aligned_cosine": float(lm_head_cos.get("aligned_cosine", 0)),
        }
        all_results.append(result)

        # Save per-init
        init_dir = output_dir / f"init_{init_idx}"
        init_dir.mkdir(parents=True, exist_ok=True)
        with open(init_dir / "kd_summary.json", "w") as f:
            json.dump(result, f, indent=2)

        # Save model
        model_to_save = student.module if hasattr(student, "module") else student
        model_to_save.save_pretrained(init_dir / "recovered_model")

        del student
        torch.cuda.empty_cache()

    # Aggregate summary
    summary = {
        "method": "suffix_matched_kd",
        "model_name": args.model_name,
        "num_suffix_blocks": args.num_suffix_blocks,
        "logit_suffix_positions": args.logit_suffix_positions,
        "pool_size": args.pool_size,
        "num_queries_used": num_queries,
        "results": all_results,
    }

    # Compute cross-init stats
    for block_idx in sorted(target_blocks):
        cosines = [r["blocks"][list(sorted(target_blocks)).index(block_idx)]["mean_cosine"]
                   for r in all_results]
        summary[f"block_{block_idx}_mean"] = sum(cosines) / len(cosines)
        summary[f"block_{block_idx}_std"] = (
            sum((c - summary[f"block_{block_idx}_mean"])**2 for c in cosines)
            / max(1, len(cosines) - 1)
        ) ** 0.5

    lm_cosines = [r["lm_head_aligned_cosine"] for r in all_results]
    summary["lm_head_mean"] = sum(lm_cosines) / len(lm_cosines)

    with open(output_dir / "kd_suffix_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    logger.info("\n=== KD Suffix Baseline Summary ===")
    logger.info("%-20s | %-10s", "Component", "Mean cos")
    logger.info("-" * 35)
    logger.info("%-20s | %-10.4f", "lm_head", summary["lm_head_mean"])
    for block_idx in sorted(target_blocks):
        logger.info("%-20s | %-10.4f ± %.4f", f"Block {block_idx}",
                    summary[f"block_{block_idx}_mean"],
                    summary[f"block_{block_idx}_std"])

    logger.info("\nResults saved to %s", output_dir)


if __name__ == "__main__":
    main()
