#!/usr/bin/env python3
"""Knowledge Distillation baseline for fair comparison against progressive parameter inversion.

Standard KD: teacher (Qwen3.5-4B frozen) → student (Qwen3.5-4B random init).
Uses KL divergence on logits with the same query budget as parameter inversion so the
comparison is apples-to-apples on total information extracted from the teacher.

Usage:
    python scripts/run_kd_baseline.py \
        --teacher_model Qwen/Qwen3.5-4B \
        --student_model Qwen/Qwen3.5-4B \
        --query_budget 500000 \
        --output_dir results/kd_baseline

    torchrun --nproc_per_node=4 scripts/run_kd_baseline.py ...
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def build_query_dataset(tokenizer, num_queries: int, max_seq_len: int, seed: int = 42):
    """Build queries from wikitext, falling back to random tokens."""
    rng = torch.Generator().manual_seed(seed)
    input_ids_list = []

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", trust_remote_code=True)
        for example in ds:
            if len(input_ids_list) >= num_queries:
                break
            text = example.get("text", "")
            if len(text.strip()) < 20:
                continue
            tokens = tokenizer(
                text, max_length=max_seq_len, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            input_ids_list.append(tokens["input_ids"].squeeze(0))
    except Exception as e:
        logger.warning("Dataset load failed: %s. Using random tokens.", e)

    remaining = num_queries - len(input_ids_list)
    if remaining > 0:
        random_ids = torch.randint(3, tokenizer.vocab_size, (remaining, max_seq_len), generator=rng)
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    all_ids = torch.stack(input_ids_list[:num_queries])
    logger.info("Query dataset: %s", all_ids.shape)
    return TensorDataset(all_ids)


@torch.no_grad()
def collect_teacher_logits(teacher_model, dataset, batch_size, device, temperature=1.0):
    """Query teacher and collect full logits (sparse top-K for memory)."""
    teacher_model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_input_ids, all_targets = [], []

    for (batch_ids,) in tqdm(loader, desc="Collecting teacher logits"):
        batch_ids = batch_ids.to(device)
        logits = teacher_model(batch_ids).logits

        soft = F.softmax(logits / temperature, dim=-1)
        topk = 128
        values, indices = soft.topk(topk, dim=-1)
        sparse = torch.zeros_like(soft)
        sparse.scatter_(-1, indices, values)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True)

        all_input_ids.append(batch_ids.cpu())
        all_targets.append(sparse.cpu().half())

    return TensorDataset(torch.cat(all_input_ids), torch.cat(all_targets))


def train_kd_student(
    student_model, train_dataset, args, rank=0, world_size=1,
):
    """Train student via KL divergence distillation from teacher logits."""
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    student_model = student_model.to(device)

    if world_size > 1:
        student_model = DDP(student_model, device_ids=[rank])

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None), drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    total_steps = len(loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    global_step = 0
    metrics_log = []

    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        student_model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_ids, batch_targets in tqdm(
            loader, desc=f"KD Epoch {epoch+1}/{args.num_epochs}", disable=(rank != 0)
        ):
            batch_ids = batch_ids.to(device)
            batch_targets = batch_targets.to(device).float()

            student_logits = student_model(batch_ids).logits
            student_log_probs = F.log_softmax(student_logits / args.temperature, dim=-1)

            kl_loss = F.kl_div(student_log_probs, batch_targets, reduction="batchmean")
            loss = kl_loss * (args.temperature ** 2)

            # Hard-label cross-entropy as auxiliary
            hard_labels = batch_targets.argmax(dim=-1)
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                hard_labels.view(-1),
            )
            total_loss = args.alpha * loss + (1 - args.alpha) * ce_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            num_batches += 1
            global_step += 1

            if rank == 0 and global_step % 100 == 0:
                logger.info(
                    "Step %d | kl=%.4f | ce=%.4f | total=%.4f | lr=%.2e",
                    global_step, loss.item(), ce_loss.item(),
                    total_loss.item(), scheduler.get_last_lr()[0],
                )

        avg_loss = epoch_loss / max(1, num_batches)
        metrics_log.append({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "global_step": global_step,
        })

        if rank == 0:
            logger.info("Epoch %d | avg_loss=%.4f", epoch + 1, avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                model_to_save = student_model.module if hasattr(student_model, "module") else student_model
                model_to_save.save_pretrained(output_dir / "best_student")
                logger.info("Saved best student (loss=%.4f)", best_loss)

    if rank == 0:
        model_to_save = student_model.module if hasattr(student_model, "module") else student_model
        model_to_save.save_pretrained(output_dir / "final_student")

        summary = {
            "best_loss": best_loss,
            "num_epochs": args.num_epochs,
            "query_budget": args.query_budget,
            "teacher_model": args.teacher_model,
            "student_model": args.student_model,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "metrics": metrics_log,
        }
        with open(output_dir / "kd_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("KD baseline complete. Results in %s", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Baseline")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--query_budget", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for KL loss vs CE loss")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="results/kd_baseline")
    parser.add_argument("--config", type=str, default="configs/inversion_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    setup_logging(rank)
    torch.manual_seed(args.seed)

    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        args.teacher_model = config.get("teacher", {}).get("model_name", args.teacher_model)
        args.student_model = config.get("student", {}).get("model_name", args.student_model)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    logger.info("=== Knowledge Distillation Baseline ===")
    logger.info("Teacher: %s → Student: %s", args.teacher_model, args.student_model)
    logger.info("Query budget: %d, Temperature: %.1f, Alpha: %.2f",
                args.query_budget, args.temperature, args.alpha)

    logger.info("Loading teacher: %s", args.teacher_model)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_queries = min(args.query_budget, args.query_budget // args.max_seq_len)
    logger.info("Building query dataset (%d queries)...", num_queries)
    query_dataset = build_query_dataset(tokenizer, num_queries, args.max_seq_len, args.seed)

    if rank == 0:
        logger.info("Collecting teacher logits...")
        distill_dataset = collect_teacher_logits(
            teacher_model, query_dataset, args.batch_size, device, args.temperature,
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(distill_dataset, Path(args.output_dir) / "distill_dataset.pt")
    else:
        distill_dataset = None

    if world_size > 1:
        dist.barrier()
        if rank != 0:
            distill_dataset = torch.load(Path(args.output_dir) / "distill_dataset.pt")

    del teacher_model
    torch.cuda.empty_cache()

    logger.info("Loading student: %s (random init)", args.student_model)
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    # Re-initialize weights to simulate random init
    student_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    train_kd_student(student_model, distill_dataset, args, rank, world_size)

    if rank == 0:
        tokenizer.save_pretrained(Path(args.output_dir) / "best_student")
        tokenizer.save_pretrained(Path(args.output_dir) / "final_student")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
