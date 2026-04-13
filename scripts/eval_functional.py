#!/usr/bin/env python3
"""Held-out Functional Evaluation: KL divergence, top-k accuracy, perplexity.

Compares recovered student models (from S-PSI) against the teacher on
a HELD-OUT set of WikiText sequences NOT used during inversion training.

Metrics per model:
  - KL(teacher || student)  — forward KL divergence (averaged over positions)
  - Top-1 accuracy match    — fraction where argmax agrees
  - Top-5 overlap           — fraction of teacher top-5 in student top-5
  - Perplexity (teacher)    — teacher's own PPL on held-out data
  - Perplexity (student)    — student's PPL on the same data

Models evaluated:
  1. Random baseline (freshly initialized Qwen2.5-0.5B with untied lm_head)
  2. All recovered models found under results/exp1_algebraic_init/

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_functional.py \
        --teacher_model Qwen/Qwen2.5-0.5B \
        --results_root results/exp1_algebraic_init \
        --output_dir results/functional_eval \
        --num_samples 500 --batch_size 4 --seq_len 256
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_heldout_dataset(tokenizer, num_samples: int, seq_len: int,
                          skip_first: int = 2048, seed: int = 99):
    """Load WikiText-103 validation split, skip `skip_first` usable sequences,
    then take the next `num_samples`.  This guarantees no overlap with the
    training pool (which uses the first ~2048 sequences from the same split).
    """
    from datasets import load_dataset

    logger.info("Loading WikiText-103 validation split...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

    input_ids_list = []
    usable_count = 0
    for ex in ds:
        text = ex.get("text", "")
        if len(text.strip()) < 30:
            continue
        usable_count += 1
        if usable_count <= skip_first:
            continue
        tokens = tokenizer(
            text, max_length=seq_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        input_ids_list.append(tokens["input_ids"].squeeze(0))
        if len(input_ids_list) >= num_samples:
            break

    if len(input_ids_list) < num_samples:
        logger.warning(
            "Only found %d held-out sequences (wanted %d). "
            "Padding with random tokens.",
            len(input_ids_list), num_samples,
        )
        rng = torch.Generator().manual_seed(seed)
        remaining = num_samples - len(input_ids_list)
        random_ids = torch.randint(
            3, tokenizer.vocab_size, (remaining, seq_len), generator=rng
        )
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    logger.info(
        "Held-out dataset: %d sequences (skipped first %d usable from pool)",
        len(input_ids_list), skip_first,
    )
    return torch.stack(input_ids_list)


@torch.no_grad()
def compute_functional_metrics(teacher, student, eval_data, device,
                               batch_size: int = 4, suffix_positions: int = 8):
    """Compute KL, top-1/5 match, and perplexities on eval_data.

    Only the last `suffix_positions` token positions are used for logit
    comparison (matching the inversion training setup).

    Returns dict of aggregated metrics.
    """
    teacher.eval()
    student.eval()

    total_positions = 0
    total_kl = 0.0
    total_top1_match = 0
    total_top5_overlap = 0.0

    # For perplexity: accumulate cross-entropy loss
    total_teacher_ce = 0.0
    total_student_ce = 0.0
    total_ppl_tokens = 0

    n = eval_data.size(0)
    K = suffix_positions

    for start in tqdm(range(0, n, batch_size), desc="  eval", leave=False):
        end = min(start + batch_size, n)
        input_ids = eval_data[start:end].to(device)
        bsz, seq_len = input_ids.shape

        # Forward pass
        t_out = teacher(input_ids)
        s_out = student(input_ids)

        t_logits = t_out.logits  # (bsz, seq_len, vocab)
        s_logits = s_out.logits

        # ── Logit comparison on last K positions ──
        if K > 0 and K < seq_len:
            t_logits_k = t_logits[:, -K:, :]
            s_logits_k = s_logits[:, -K:, :]
        else:
            t_logits_k = t_logits
            s_logits_k = s_logits
            K = seq_len

        bsz_k = t_logits_k.size(0)
        num_pos = t_logits_k.size(1)

        # KL(teacher || student) = sum_x p_T(x) * log(p_T(x) / p_S(x))
        t_log_probs = F.log_softmax(t_logits_k.float(), dim=-1)
        s_log_probs = F.log_softmax(s_logits_k.float(), dim=-1)
        t_probs = t_log_probs.exp()

        # F.kl_div expects (log Q, P) and computes sum P * (log P - log Q)
        kl = F.kl_div(s_log_probs, t_probs, reduction="sum").item()
        total_kl += kl
        total_positions += bsz_k * num_pos

        # Top-1 match
        t_pred = t_logits_k.argmax(dim=-1)  # (bsz, K)
        s_pred = s_logits_k.argmax(dim=-1)
        total_top1_match += (t_pred == s_pred).sum().item()

        # Top-5 overlap
        t_top5 = t_logits_k.topk(5, dim=-1).indices  # (bsz, K, 5)
        s_top5 = s_logits_k.topk(5, dim=-1).indices
        for b in range(bsz_k):
            for p in range(num_pos):
                t5 = set(t_top5[b, p].tolist())
                s5 = set(s_top5[b, p].tolist())
                total_top5_overlap += len(t5 & s5) / 5.0

        # ── Perplexity (full sequence, standard next-token prediction) ──
        # Shift: predict token t+1 from logits at position t
        shift_t = t_logits[:, :-1, :].contiguous()
        shift_s = s_logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()

        # Mask out padding tokens
        pad_mask = (targets != 0)  # 0 is typically pad for padded seqs
        num_valid = pad_mask.sum().item()

        if num_valid > 0:
            t_ce = F.cross_entropy(
                shift_t.view(-1, shift_t.size(-1)),
                targets.view(-1),
                reduction="none",
            ).view(targets.shape)
            s_ce = F.cross_entropy(
                shift_s.view(-1, shift_s.size(-1)),
                targets.view(-1),
                reduction="none",
            ).view(targets.shape)

            total_teacher_ce += (t_ce * pad_mask).sum().item()
            total_student_ce += (s_ce * pad_mask).sum().item()
            total_ppl_tokens += num_valid

    # Aggregate
    mean_kl = total_kl / max(total_positions, 1)
    top1_rate = total_top1_match / max(total_positions, 1)
    top5_rate = total_top5_overlap / max(total_positions, 1)

    teacher_ppl = float("inf")
    student_ppl = float("inf")
    if total_ppl_tokens > 0:
        teacher_ppl = torch.exp(
            torch.tensor(total_teacher_ce / total_ppl_tokens)
        ).item()
        student_ppl = torch.exp(
            torch.tensor(total_student_ce / total_ppl_tokens)
        ).item()

    return {
        "mean_kl_divergence": mean_kl,
        "top1_accuracy_match": top1_rate,
        "top5_overlap_rate": top5_rate,
        "teacher_perplexity": teacher_ppl,
        "student_perplexity": student_ppl,
        "num_eval_positions": total_positions,
        "num_ppl_tokens": total_ppl_tokens,
    }


def create_random_student(model_name: str, device: str, dtype=torch.bfloat16):
    """Create a randomly-initialized student of the same architecture."""
    logger.info("Creating random baseline student...")
    student = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype,
        device_map={"": device}, trust_remote_code=True,
    )

    # Untie lm_head from embed_tokens
    lm_head = getattr(student, "lm_head", None)
    embed = None
    if hasattr(student, "model") and hasattr(student.model, "embed_tokens"):
        embed = student.model.embed_tokens
    if lm_head is not None and embed is not None:
        if lm_head.weight.data_ptr() == embed.weight.data_ptr():
            lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
            if hasattr(student, "config"):
                student.config.tie_word_embeddings = False

    # Randomize ALL parameters
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

    student.eval()
    return student


def load_recovered_model(model_dir: str, model_name: str, device: str,
                         dtype=torch.bfloat16):
    """Load a recovered student model from a saved HuggingFace directory."""
    logger.info("Loading recovered model from %s", model_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype,
            device_map={"": device}, trust_remote_code=True,
        )
        model.eval()
        return model
    except Exception as e:
        logger.warning("Failed to load model from %s: %s", model_dir, e)
        return None


def reconstruct_from_checkpoint(ckpt_dir: str, model_name: str, device: str,
                                dtype=torch.bfloat16):
    """Try to reconstruct a student from the latest checkpoint .pt files.

    Checkpoints contain model_state_dict which is the FULL student state_dict.
    """
    ckpt_path = Path(ckpt_dir)
    pt_files = sorted(ckpt_path.glob("ckpt_*.pt"))
    if not pt_files:
        return None

    logger.info("Found %d checkpoint files in %s", len(pt_files), ckpt_dir)

    # Load the student architecture
    student = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype,
        device_map={"": device}, trust_remote_code=True,
    )

    # Untie lm_head
    lm_head = getattr(student, "lm_head", None)
    embed = None
    if hasattr(student, "model") and hasattr(student.model, "embed_tokens"):
        embed = student.model.embed_tokens
    if lm_head is not None and embed is not None:
        if lm_head.weight.data_ptr() == embed.weight.data_ptr():
            lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
            if hasattr(student, "config"):
                student.config.tie_word_embeddings = False

    # Load state dict from the last checkpoint (which has the latest weights)
    # Checkpoints are per-block; we need to load all of them sequentially
    # to reconstruct the full model.  Each checkpoint has the FULL model
    # state_dict, so the last one is the final recovered state.
    last_ckpt = pt_files[-1]
    logger.info("Loading checkpoint: %s", last_ckpt.name)
    ckpt = torch.load(str(last_ckpt), map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        # The checkpoint contains complete student state_dict
        student.load_state_dict(ckpt["model_state_dict"], strict=False)
        student.eval()
        logger.info("Reconstructed student from checkpoint %s", last_ckpt.name)
        return student
    else:
        logger.warning("Checkpoint %s has no model_state_dict key", last_ckpt)
        return None


def discover_result_dirs(results_root: str):
    """Discover all init directories under results_root.

    Expected structure:
      results_root/{method}/seed_{S}/regime_{R}/init_{I}/
    or:
      results_root/{method}/seed_{S}/  (flat, used as output_dir directly)
    """
    root = Path(results_root)
    if not root.exists():
        logger.warning("Results root does not exist: %s", root)
        return []

    discovered = []

    # Pattern 1: method/seed_*/regime_*/init_*
    for init_dir in sorted(root.glob("*/seed_*/regime_*/init_*")):
        if init_dir.is_dir():
            parts = init_dir.relative_to(root).parts
            method = parts[0]
            seed = parts[1]
            regime = parts[2]
            init = parts[3]
            discovered.append({
                "path": str(init_dir),
                "method": method,
                "seed": seed,
                "regime": regime,
                "init": init,
                "label": f"{method}/{seed}/{regime}/{init}",
            })

    # Pattern 2: method/seed_*/init_* (no regime subdir — direct output_dir)
    for init_dir in sorted(root.glob("*/seed_*/init_*")):
        if init_dir.is_dir():
            # Skip if already matched by pattern 1
            parts = init_dir.relative_to(root).parts
            if len(parts) == 3:
                method = parts[0]
                seed = parts[1]
                init = parts[2]
                label = f"{method}/{seed}/{init}"
                if not any(d["label"].startswith(label) for d in discovered):
                    discovered.append({
                        "path": str(init_dir),
                        "method": method,
                        "seed": seed,
                        "regime": "unknown",
                        "init": init,
                        "label": label,
                    })

    # Pattern 3: direct regime_*/init_* under root (single method)
    for init_dir in sorted(root.glob("regime_*/init_*")):
        if init_dir.is_dir():
            parts = init_dir.relative_to(root).parts
            regime = parts[0]
            init = parts[1]
            discovered.append({
                "path": str(init_dir),
                "method": "default",
                "seed": "unknown",
                "regime": regime,
                "init": init,
                "label": f"{regime}/{init}",
            })

    # Pattern 4: v2 layout — {experiment_dir}/regime_*/init_*
    # e.g. results/v2_random_s42/regime_oracle/init_0/
    for init_dir in sorted(root.glob("*/regime_*/init_*")):
        if init_dir.is_dir():
            parts = init_dir.relative_to(root).parts
            exp_name = parts[0]  # e.g. v2_random_s42
            regime = parts[1]    # e.g. regime_oracle
            init = parts[2]      # e.g. init_0
            label = f"{exp_name}/{regime}/{init}"
            if not any(d["label"] == label for d in discovered):
                discovered.append({
                    "path": str(init_dir),
                    "method": exp_name,
                    "seed": "unknown",
                    "regime": regime,
                    "init": init,
                    "label": label,
                })

    logger.info("Discovered %d result directories under %s", len(discovered), root)
    return discovered


def find_loadable_model(init_dir: str, model_name: str, device: str,
                        dtype=torch.bfloat16):
    """Try to find and load a model from an init directory.

    Priority:
      1. recovered_model/ directory (full HF model)
      2. checkpoints/ directory (reconstruct from .pt)
      3. None (no loadable model found)
    """
    base = Path(init_dir)

    # Priority 1: saved HF model
    recovered_dir = base / "recovered_model"
    if recovered_dir.exists() and (recovered_dir / "config.json").exists():
        model = load_recovered_model(str(recovered_dir), model_name, device, dtype)
        if model is not None:
            return model, "recovered_model"

    # Priority 2: checkpoint reconstruction
    ckpt_dir = base / "checkpoints"
    if ckpt_dir.exists():
        model = reconstruct_from_checkpoint(str(ckpt_dir), model_name, device, dtype)
        if model is not None:
            return model, "checkpoint"

    return None, "none"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Held-out functional evaluation of recovered models"
    )
    parser.add_argument(
        "--teacher_model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="Teacher model name or path",
    )
    parser.add_argument(
        "--results_root", type=str, default="results/exp1_algebraic_init",
        help="Root directory containing experiment results",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/functional_eval",
        help="Output directory for metrics",
    )
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of held-out sequences")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--suffix_positions", type=int, default=8,
                        help="Number of last positions for logit comparison")
    parser.add_argument("--skip_first", type=int, default=2048,
                        help="Skip first N usable sequences (training pool)")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--skip_random_baseline", action="store_true",
                        help="Skip random baseline evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Held-out Functional Evaluation")
    logger.info("=" * 60)
    logger.info("Teacher: %s", args.teacher_model)
    logger.info("Device: %s | dtype: %s", device, dtype)
    logger.info("Eval samples: %d | seq_len: %d | suffix_pos: %d",
                args.num_samples, args.seq_len, args.suffix_positions)

    # ── Load teacher ──
    logger.info("Loading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=dtype,
        device_map={"": device}, trust_remote_code=True,
    )
    teacher.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build held-out data ──
    eval_data = build_heldout_dataset(
        tokenizer, args.num_samples, args.seq_len,
        skip_first=args.skip_first, seed=args.seed,
    )
    logger.info("Eval data shape: %s", eval_data.shape)

    all_results = {}
    model_count = 0

    # ── 1. Teacher self-evaluation (reference) ──
    logger.info("\n--- Evaluating: Teacher (self-reference) ---")
    t_start = time.time()
    teacher_metrics = compute_functional_metrics(
        teacher, teacher, eval_data, device,
        batch_size=args.batch_size, suffix_positions=args.suffix_positions,
    )
    teacher_metrics["eval_time_s"] = time.time() - t_start
    teacher_metrics["model_source"] = "teacher_self"
    all_results["teacher_reference"] = teacher_metrics
    logger.info("  KL=%.6f  Top1=%.4f  Top5=%.4f  PPL_T=%.2f  PPL_S=%.2f",
                teacher_metrics["mean_kl_divergence"],
                teacher_metrics["top1_accuracy_match"],
                teacher_metrics["top5_overlap_rate"],
                teacher_metrics["teacher_perplexity"],
                teacher_metrics["student_perplexity"])

    # ── 2. Random baseline ──
    if not args.skip_random_baseline:
        logger.info("\n--- Evaluating: Random Baseline ---")
        random_student = create_random_student(args.teacher_model, device, dtype)
        t_start = time.time()
        random_metrics = compute_functional_metrics(
            teacher, random_student, eval_data, device,
            batch_size=args.batch_size, suffix_positions=args.suffix_positions,
        )
        random_metrics["eval_time_s"] = time.time() - t_start
        random_metrics["model_source"] = "random_init"
        all_results["random_baseline"] = random_metrics
        logger.info("  KL=%.6f  Top1=%.4f  Top5=%.4f  PPL_T=%.2f  PPL_S=%.2f",
                    random_metrics["mean_kl_divergence"],
                    random_metrics["top1_accuracy_match"],
                    random_metrics["top5_overlap_rate"],
                    random_metrics["teacher_perplexity"],
                    random_metrics["student_perplexity"])
        del random_student
        torch.cuda.empty_cache()

    # ── 3. Recovered models from experiments ──
    discovered = discover_result_dirs(args.results_root)
    loaded_count = 0
    skipped_count = 0

    for info in discovered:
        label = info["label"]
        init_path = info["path"]

        model, source = find_loadable_model(
            init_path, args.teacher_model, device, dtype
        )
        if model is None:
            logger.info("  [SKIP] %s — no loadable model found", label)
            skipped_count += 1
            # Still record that we found the dir but couldn't load
            summary_file = Path(init_path) / "spsi_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        summary = json.load(f)
                    all_results[f"summary_only:{label}"] = {
                        "model_source": "summary_only_no_weights",
                        "spsi_summary": summary,
                        "note": "No saved model weights; only cosine metrics available",
                    }
                except Exception:
                    pass
            continue

        logger.info("\n--- Evaluating: %s (source: %s) ---", label, source)
        # Check vocab size compatibility
        t_vocab = teacher.config.vocab_size
        s_vocab = model.config.vocab_size
        if t_vocab != s_vocab:
            logger.warning("  [SKIP] vocab mismatch: teacher=%d, student=%d", t_vocab, s_vocab)
            del model
            torch.cuda.empty_cache()
            skipped_count += 1
            continue
        t_start = time.time()
        metrics = compute_functional_metrics(
            teacher, model, eval_data, device,
            batch_size=args.batch_size, suffix_positions=args.suffix_positions,
        )
        metrics["eval_time_s"] = time.time() - t_start
        metrics["model_source"] = source
        metrics["init_dir"] = init_path
        metrics["method"] = info["method"]
        metrics["seed"] = info["seed"]
        metrics["regime"] = info["regime"]

        safe_label = label.replace("/", "__")
        all_results[safe_label] = metrics
        loaded_count += 1

        logger.info("  KL=%.6f  Top1=%.4f  Top5=%.4f  PPL_T=%.2f  PPL_S=%.2f",
                    metrics["mean_kl_divergence"],
                    metrics["top1_accuracy_match"],
                    metrics["top5_overlap_rate"],
                    metrics["teacher_perplexity"],
                    metrics["student_perplexity"])

        del model
        torch.cuda.empty_cache()

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Result dirs discovered: %d", len(discovered))
    logger.info("Models loaded & evaluated: %d", loaded_count)
    logger.info("Skipped (no weights): %d", skipped_count)

    # ── Summary table ──
    logger.info("\n%-40s  %8s  %8s  %8s  %10s  %10s",
                "Model", "KL", "Top1", "Top5", "PPL_T", "PPL_S")
    logger.info("-" * 100)
    for name, m in all_results.items():
        if "mean_kl_divergence" in m:
            logger.info("%-40s  %8.4f  %8.4f  %8.4f  %10.2f  %10.2f",
                        name[:40],
                        m["mean_kl_divergence"],
                        m["top1_accuracy_match"],
                        m["top5_overlap_rate"],
                        m["teacher_perplexity"],
                        m["student_perplexity"])

    # ── Save results ──
    out_file = output_dir / "functional_metrics.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", out_file)

    # Also save a compact table
    table = []
    for name, m in all_results.items():
        if "mean_kl_divergence" in m:
            table.append({
                "model": name,
                "kl_divergence": round(m["mean_kl_divergence"], 6),
                "top1_match": round(m["top1_accuracy_match"], 4),
                "top5_overlap": round(m["top5_overlap_rate"], 4),
                "teacher_ppl": round(m["teacher_perplexity"], 2),
                "student_ppl": round(m["student_perplexity"], 2),
                "source": m.get("model_source", "unknown"),
            })
    table_file = output_dir / "functional_table.json"
    with open(table_file, "w") as f:
        json.dump(table, f, indent=2)
    logger.info("Summary table saved to %s", table_file)


if __name__ == "__main__":
    main()
