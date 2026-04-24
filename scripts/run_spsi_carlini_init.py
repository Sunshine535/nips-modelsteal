#!/usr/bin/env python3
"""S-PSI with Carlini-extracted lm_head initialization.

Tests whether initializing the student's lm_head from Carlini's algebraically
extracted W_hat (recovered from black-box logit access) improves block-level
parameter recovery compared to random lm_head initialization.

Three configurations are compared on Block 23 (oracle regime):
  1. carlini_exact : lm_head = teacher lm_head  (upper bound)
  2. carlini_svd   : lm_head = W_hat^T padded   (actual Carlini output)
  3. random         : lm_head = Kaiming random   (baseline)

Usage:
    python scripts/run_spsi_carlini_init.py \
        --carlini_W_hat results/v5_carlini_baseline/recovered_W_hat.pt \
        --output_dir results/v5_carlini_spsi \
        --device cuda:0
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parameter_inverter import (
    BlackBoxTeacher,
    SPSIConfig,
    TeacherCache,
    get_block_param_names,
    get_lm_head_param_names,
    get_num_blocks,
    invert_block,
    compute_per_matrix_cosine,
)
from src.permutation_alignment import compute_aligned_cosine, compute_lm_head_aligned_cosine

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
TARGET_BLOCK = 23          # last block
D_MODEL = 896
VOCAB_SIZE = 151936


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="S-PSI with Carlini lm_head initialization"
    )
    parser.add_argument(
        "--carlini_W_hat", type=str,
        default="results/v5_carlini_baseline/recovered_W_hat.pt",
        help="Path to recovered_W_hat.pt (shape [d_hat, V])",
    )
    parser.add_argument("--output_dir", type=str, default="results/v5_carlini_spsi")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--pool_size", type=int, default=2000,
                        help="Query pool size (smaller than full experiment for speed)")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Sensitivity loss weight")
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--allow_synthetic", action="store_true",
                        help="Fall back to random tokens if dataset load fails")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Query pool
# ---------------------------------------------------------------------------

def build_query_pool(tokenizer, pool_size: int, max_seq_len: int, seed: int,
                     allow_synthetic: bool = False) -> torch.Tensor:
    """Build query input_ids from WikiText (or random tokens as fallback)."""
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
        if not allow_synthetic:
            raise RuntimeError(
                f"Dataset load failed: {e}. Use --allow_synthetic to fall back."
            ) from e
        logger.warning("Dataset load failed: %s. Using random tokens.", e)

    remaining = pool_size - len(input_ids_list)
    if remaining > 0:
        if not allow_synthetic:
            raise RuntimeError(
                f"Only {len(input_ids_list)}/{pool_size} from dataset. "
                "Use --allow_synthetic to pad with random tokens."
            )
        rng = torch.Generator().manual_seed(seed)
        random_ids = torch.randint(
            3, tokenizer.vocab_size, (remaining, max_seq_len), generator=rng
        )
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    return torch.stack(input_ids_list[:pool_size])


# ---------------------------------------------------------------------------
# lm_head initialization strategies
# ---------------------------------------------------------------------------

def untie_lm_head(student: torch.nn.Module):
    """Untie lm_head from embed_tokens so it can be set independently."""
    lm_head = getattr(student, "lm_head", None)
    embed = None
    if hasattr(student, "model") and hasattr(student.model, "embed_tokens"):
        embed = student.model.embed_tokens

    if lm_head is None or embed is None:
        return False

    if lm_head.weight.data_ptr() == embed.weight.data_ptr():
        lm_head.weight = torch.nn.Parameter(
            lm_head.weight.data.clone(),
            requires_grad=lm_head.weight.requires_grad,
        )
        if hasattr(student, "config"):
            student.config.tie_word_embeddings = False
        logger.info("Untied lm_head from embed_tokens.")
        return True
    return False


def init_lm_head_carlini_exact(student, teacher_model):
    """Set student lm_head = teacher lm_head (upper bound for Carlini)."""
    untie_lm_head(student)
    teacher_lm_head_w = teacher_model.lm_head.weight.data.clone()
    student.lm_head.weight.data.copy_(teacher_lm_head_w)
    logger.info("lm_head init: carlini_exact (teacher lm_head copied)")


def init_lm_head_carlini_svd(student, W_hat_path, teacher_model):
    """Set student lm_head from Carlini's W_hat with Procrustes alignment.

    W_hat is [d_hat, V] where d_hat <= d.  We:
      1. Pad to [d, V] with zeros for missing dimensions
      2. Transpose to [V, d] (lm_head shape)
      3. Procrustes-align against teacher lm_head to fix rotation
    """
    untie_lm_head(student)

    loaded = torch.load(W_hat_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict):
        W_hat = loaded["W_hat"]
    else:
        W_hat = loaded
    d_hat, V = W_hat.shape
    logger.info("Loaded W_hat: shape %s (d_hat=%d, V=%d)", W_hat.shape, d_hat, V)

    assert V == VOCAB_SIZE, f"Expected V={VOCAB_SIZE}, got {V}"

    # Pad to [d, V] if d_hat < d
    if d_hat < D_MODEL:
        W_padded = torch.zeros(D_MODEL, V, dtype=W_hat.dtype)
        W_padded[:d_hat] = W_hat
        logger.info("Padded W_hat from [%d, %d] to [%d, %d]", d_hat, V, D_MODEL, V)
    elif d_hat == D_MODEL:
        W_padded = W_hat
    else:
        # d_hat > d: truncate (shouldn't happen for 894 < 896)
        W_padded = W_hat[:D_MODEL]
        logger.warning("Truncated W_hat from d_hat=%d to d=%d", d_hat, D_MODEL)

    # W_padded is [d, V], lm_head expects [V, d]
    W_init = W_padded.T.float()  # [V, d]

    # Procrustes alignment: find R* = argmin ||W_init @ R - W_teacher||_F
    # W_teacher^T @ W_init = U S V^T  =>  R* = V U^T
    W_teacher = teacher_model.lm_head.weight.data.float().cpu()  # [V, d]
    M = W_init.T @ W_teacher  # [d, d]
    U, S, Vh = torch.linalg.svd(M)
    R = U @ Vh  # [d, d]
    # Fix reflection
    if torch.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vh

    W_aligned = W_init @ R  # [V, d]

    # Diagnostic: cosine similarity before and after alignment
    raw_cos = F.cosine_similarity(
        W_init.flatten().unsqueeze(0),
        W_teacher.flatten().unsqueeze(0),
    ).item()
    aligned_cos = F.cosine_similarity(
        W_aligned.flatten().unsqueeze(0),
        W_teacher.flatten().unsqueeze(0),
    ).item()
    frob_err = (W_aligned - W_teacher).norm().item() / (W_teacher.norm().item() + 1e-12)
    logger.info(
        "lm_head init: carlini_svd | raw_cos=%.6f, aligned_cos=%.6f, "
        "frob_err=%.6f, top singular values: %s",
        raw_cos, aligned_cos, frob_err, S[:5].tolist(),
    )

    student.lm_head.weight.data.copy_(W_aligned.to(student.lm_head.weight.dtype))
    return {
        "raw_cosine": raw_cos,
        "aligned_cosine": aligned_cos,
        "procrustes_frob_error": frob_err,
        "d_hat": d_hat,
        "top_singular_values": S[:10].tolist(),
    }


def init_lm_head_random(student):
    """Randomize student lm_head (Kaiming uniform)."""
    untie_lm_head(student)
    torch.nn.init.kaiming_uniform_(student.lm_head.weight)
    logger.info("lm_head init: random (Kaiming uniform)")


# ---------------------------------------------------------------------------
# Randomize a single block
# ---------------------------------------------------------------------------

def randomize_block(student, block_idx: int):
    """Re-initialize all parameters in a single block."""
    target = str(block_idx)
    for name, param in student.named_parameters():
        parts = name.split(".")
        if target not in parts:
            continue
        if param.dim() >= 2:
            torch.nn.init.kaiming_uniform_(param)
        else:
            if "norm" in name.lower():
                torch.nn.init.ones_(param)
            else:
                torch.nn.init.zeros_(param)


# ---------------------------------------------------------------------------
# Find block prefix for alignment
# ---------------------------------------------------------------------------

def find_block_prefix(model, block_idx: int) -> str:
    target = str(block_idx)
    for name, _ in model.named_parameters():
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == target:
                return ".".join(parts[: i + 1])
    return ""


# ---------------------------------------------------------------------------
# Run one configuration
# ---------------------------------------------------------------------------

def run_single_config(
    config_name: str,
    teacher_model,
    teacher_bb: BlackBoxTeacher,
    cache: TeacherCache,
    ground_truth: dict,
    spsi_config: SPSIConfig,
    args,
    init_lm_head_fn,
    output_subdir: Path,
) -> dict:
    """Run block-23 recovery with a specific lm_head initialization."""
    logger.info("\n" + "=" * 70)
    logger.info("Configuration: %s", config_name)
    logger.info("=" * 70)

    device = args.device
    torch.manual_seed(args.seed)

    # Fresh student for this config
    student = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )

    # 1. Initialize lm_head according to strategy
    init_meta = init_lm_head_fn(student)

    # 2. Freeze lm_head
    lm_head_names = get_lm_head_param_names(student)
    for name, param in student.named_parameters():
        if name in lm_head_names or "lm_head" in name:
            param.requires_grad = False

    # Verify lm_head cosine before block training
    lm_head_cos_pre = {}
    for name, param in student.named_parameters():
        if "lm_head" in name and name in ground_truth:
            gt = ground_truth[name].to(param.device).float().flatten()
            pred = param.data.float().flatten()
            if gt.shape == pred.shape:
                lm_head_cos_pre[name] = F.cosine_similarity(
                    pred.unsqueeze(0), gt.unsqueeze(0)
                ).item()
    logger.info("lm_head cosine before block training: %s", lm_head_cos_pre)

    # 3. Randomize block 23 parameters
    randomize_block(student, TARGET_BLOCK)

    if hasattr(student, "gradient_checkpointing_disable"):
        student.gradient_checkpointing_disable()

    # 4. Get block param names
    block_names = get_block_param_names(student, TARGET_BLOCK)
    if not block_names:
        raise RuntimeError(f"No parameters found for block {TARGET_BLOCK}")
    logger.info("Block %d has %d parameter tensors", TARGET_BLOCK, len(block_names))

    # 5. Run S-PSI inversion on block 23
    t0 = time.time()
    result = invert_block(
        student=student,
        cache=cache,
        config=spsi_config,
        param_names=block_names,
        ground_truth=ground_truth,
        boundary_layer_idx=TARGET_BLOCK,
        use_oracle_boundary=True,
        checkpoint_dir=str(output_subdir / "checkpoints"),
    )
    elapsed = time.time() - t0

    # 6. Compute aligned cosine similarity (Hungarian alignment)
    recovered_params = {
        n: p.data.cpu().clone() for n, p in student.named_parameters()
    }
    block_prefix = find_block_prefix(student, TARGET_BLOCK)
    num_attention_heads = teacher_model.config.num_attention_heads
    head_dim = teacher_model.config.hidden_size // num_attention_heads

    unaligned, aligned = {}, {}
    if block_prefix:
        unaligned, aligned = compute_aligned_cosine(
            recovered_params, ground_truth,
            block_prefix, num_attention_heads, head_dim,
        )

    # Also get lm_head alignment metrics
    lm_head_alignment = compute_lm_head_aligned_cosine(
        recovered_params, ground_truth, lm_head_key="lm_head.weight",
    )

    # Summary
    aligned_mean = sum(aligned.values()) / max(len(aligned), 1) if aligned else -1.0
    unaligned_mean = sum(unaligned.values()) / max(len(unaligned), 1) if unaligned else -1.0

    summary = {
        "config_name": config_name,
        "target_block": TARGET_BLOCK,
        "regime": "oracle",
        "num_steps": result.num_steps,
        "final_loss": result.final_loss,
        "converged": result.converged,
        "elapsed_seconds": round(elapsed, 1),
        "lm_head_cosine_pre_training": lm_head_cos_pre,
        "lm_head_alignment": lm_head_alignment,
        "block_unaligned_cosine": unaligned,
        "block_aligned_cosine": aligned,
        "block_unaligned_mean": unaligned_mean,
        "block_aligned_mean": aligned_mean,
        "raw_per_matrix_cosine": result.per_matrix_cosine,
        "raw_mean_cosine": result.mean_cosine,
    }
    if init_meta:
        summary["init_meta"] = init_meta

    output_subdir.mkdir(parents=True, exist_ok=True)
    with open(output_subdir / "result.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "%s: aligned_mean=%.4f, unaligned_mean=%.4f, raw_mean=%.4f, "
        "loss=%.6f, steps=%d, time=%.1fs",
        config_name, aligned_mean, unaligned_mean, result.mean_cosine,
        result.final_loss, result.num_steps, elapsed,
    )

    del student, recovered_params
    torch.cuda.empty_cache()

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logging()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device

    # Save resolved args
    with open(output_dir / "resolved_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("=== S-PSI Carlini lm_head Initialization Experiment ===")
    logger.info("Model: %s", args.model_name)
    logger.info("Target block: %d", TARGET_BLOCK)
    logger.info("Steps: %d", args.num_steps)
    logger.info("Device: %s", device)

    # --- Load teacher ---
    logger.info("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    teacher_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ground truth (for evaluation)
    ground_truth = {
        name: param.data.cpu().clone()
        for name, param in teacher_model.named_parameters()
    }
    if getattr(teacher_model.config, "tie_word_embeddings", False):
        lm_head = getattr(teacher_model, "lm_head", None)
        if lm_head is not None and "lm_head.weight" not in ground_truth:
            ground_truth["lm_head.weight"] = lm_head.weight.data.cpu().clone()

    # --- Build query pool and teacher cache ---
    logger.info("Building query pool (%d inputs)...", args.pool_size)
    query_ids = build_query_pool(
        tokenizer, args.pool_size, args.max_seq_len, args.seed,
        allow_synthetic=args.allow_synthetic,
    )

    num_blocks = get_num_blocks(teacher_model)
    logger.info("Model has %d blocks. Targeting block %d.", num_blocks, TARGET_BLOCK)
    assert TARGET_BLOCK < num_blocks, (
        f"TARGET_BLOCK={TARGET_BLOCK} >= num_blocks={num_blocks}"
    )

    # Cache boundary states for block 23 (need layer 22 boundary = input to block 23)
    # boundary_layers should include the block before target so oracle injection works
    boundary_layers = [TARGET_BLOCK - 1, TARGET_BLOCK]

    spsi_config = SPSIConfig(
        query_budget=10_000_000,  # effectively unlimited for this experiment
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lm_head_lr=1e-3,   # not used (lm_head is frozen)
        lm_head_steps=0,    # not used
        max_steps_per_block=args.num_steps,
        convergence_threshold=1e-7,
        patience=args.patience,
        alpha=1.0,
        beta=args.beta,
        gamma=1e-5,
        num_perturbation_positions=4,
        num_replacement_tokens=2,
        max_seq_len=args.max_seq_len,
        logit_suffix_positions=8,
        seed=args.seed,
    )

    teacher_bb = BlackBoxTeacher(model=teacher_model, device=device)
    cache = TeacherCache(device=device)

    logger.info("Pre-computing teacher cache (boundary layers: %s)...", boundary_layers)
    cache.build(teacher_bb, query_ids, spsi_config, cache_boundary_layers=boundary_layers)
    logger.info("Teacher cache ready. Queries used: %d", teacher_bb.query_count)

    # --- Define configurations ---
    W_hat_path = Path(args.carlini_W_hat)
    has_W_hat = W_hat_path.exists()

    configs = []

    # 1. carlini_exact: teacher lm_head (upper bound)
    def make_carlini_exact_init(teacher_model_ref):
        def fn(student):
            init_lm_head_carlini_exact(student, teacher_model_ref)
            return {"method": "carlini_exact", "description": "teacher lm_head copied"}
        return fn
    configs.append(("carlini_exact", make_carlini_exact_init(teacher_model)))

    # 2. carlini_svd: from W_hat
    if has_W_hat:
        def make_carlini_svd_init(path, teacher_model_ref):
            def fn(student):
                meta = init_lm_head_carlini_svd(student, path, teacher_model_ref)
                return meta
            return fn
        configs.append(("carlini_svd", make_carlini_svd_init(str(W_hat_path), teacher_model)))
    else:
        logger.warning(
            "W_hat not found at %s — skipping carlini_svd config. "
            "Run the Carlini baseline first.", W_hat_path,
        )

    # 3. random: baseline
    def make_random_init():
        def fn(student):
            init_lm_head_random(student)
            return {"method": "random", "description": "Kaiming uniform"}
        return fn
    configs.append(("random", make_random_init()))

    # --- Run all configurations ---
    all_summaries = {}
    for config_name, init_fn in configs:
        summary = run_single_config(
            config_name=config_name,
            teacher_model=teacher_model,
            teacher_bb=teacher_bb,
            cache=cache,
            ground_truth=ground_truth,
            spsi_config=spsi_config,
            args=args,
            init_lm_head_fn=init_fn,
            output_subdir=output_dir / config_name,
        )
        all_summaries[config_name] = summary

    # --- Save combined results ---
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    # --- Print comparison table ---
    print("\n" + "=" * 90)
    print("COMPARISON: Block 23 Recovery with Different lm_head Initializations")
    print("=" * 90)
    print(f"{'Config':<20} {'Aligned Mean':>14} {'Unaligned Mean':>16} {'Raw Mean':>12} {'Loss':>12} {'Steps':>8}")
    print("-" * 90)

    for name, s in all_summaries.items():
        print(
            f"{name:<20} "
            f"{s['block_aligned_mean']:>14.4f} "
            f"{s['block_unaligned_mean']:>16.4f} "
            f"{s['raw_mean_cosine']:>12.4f} "
            f"{s['final_loss']:>12.6f} "
            f"{s['num_steps']:>8d}"
        )

    print("-" * 90)

    # lm_head initial cosines
    print("\nlm_head Cosine (before block training):")
    print(f"{'Config':<20} {'lm_head cos':>14}")
    print("-" * 40)
    for name, s in all_summaries.items():
        lm_cos = s.get("lm_head_cosine_pre_training", {})
        cos_val = next(iter(lm_cos.values()), -1.0) if lm_cos else -1.0
        print(f"{name:<20} {cos_val:>14.6f}")

    # Per-matrix breakdown for aligned cosines
    print("\nPer-matrix aligned cosine (Block 23):")
    all_matrix_names = set()
    for s in all_summaries.values():
        all_matrix_names.update(s.get("block_aligned_cosine", {}).keys())
    sorted_names = sorted(all_matrix_names)

    if sorted_names:
        header = f"{'Matrix':<35}" + "".join(f"{n:>18}" for n in all_summaries.keys())
        print(header)
        print("-" * len(header))
        for mat_name in sorted_names:
            row = f"{mat_name:<35}"
            for config_name, s in all_summaries.items():
                val = s.get("block_aligned_cosine", {}).get(mat_name, float("nan"))
                row += f"{val:>18.4f}"
            print(row)

    print("\n" + "=" * 90)
    print(f"Results saved to: {output_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
