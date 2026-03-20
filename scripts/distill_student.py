#!/usr/bin/env python3
"""
Stage 1 — Behavioral Distillation.

Query the black-box teacher model and train a student to match its logit
distribution. This produces a warm-start initialization for the subsequent
parameter inversion stage.

Usage:
    python scripts/distill_student.py \
        --teacher_model Qwen/Qwen3.5-9B \
        --student_model Qwen/Qwen3.5-0.8B \
        --num_queries 100000 \
        --output_dir results/distillation

    # Multi-GPU:
    torchrun --nproc_per_node=8 scripts/distill_student.py ...
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
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def load_config(config_path: str) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def build_query_dataset(
    tokenizer,
    num_queries: int,
    max_seq_len: int,
    seed: int = 42,
    dataset_name: str = "wikitext",
) -> TensorDataset:
    """Build a dataset of input sequences to query the teacher."""
    rng = torch.Generator().manual_seed(seed)

    try:
        from datasets import load_dataset

        logger.info("Loading dataset '%s' for query generation...", dataset_name)
        ds = load_dataset(dataset_name, "wikitext-103-raw-v1", split="train", trust_remote_code=True)

        input_ids_list = []
        for example in ds:
            if len(input_ids_list) >= num_queries:
                break
            text = example.get("text", "")
            if len(text.strip()) < 20:
                continue
            tokens = tokenizer(
                text,
                max_length=max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids_list.append(tokens["input_ids"].squeeze(0))

        if len(input_ids_list) < num_queries:
            logger.warning(
                "Dataset only yielded %d samples; padding with random tokens",
                len(input_ids_list),
            )

    except Exception as e:
        logger.warning("Could not load dataset: %s. Using random tokens.", e)
        input_ids_list = []

    remaining = num_queries - len(input_ids_list)
    if remaining > 0:
        vocab_size = tokenizer.vocab_size
        random_ids = torch.randint(
            3, vocab_size, (remaining, max_seq_len), generator=rng
        )
        for i in range(remaining):
            input_ids_list.append(random_ids[i])

    all_ids = torch.stack(input_ids_list[:num_queries])
    logger.info("Built query dataset: %s", all_ids.shape)
    return TensorDataset(all_ids)


@torch.no_grad()
def collect_teacher_logits(
    teacher_model,
    dataset: TensorDataset,
    batch_size: int,
    device: str,
    temperature: float = 2.0,
) -> TensorDataset:
    """Query teacher model and collect soft targets."""
    teacher_model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_input_ids = []
    all_soft_targets = []

    logger.info("Collecting teacher logits over %d samples...", len(dataset))
    for (batch_ids,) in tqdm(loader, desc="Teacher queries", disable=False):
        batch_ids = batch_ids.to(device)
        logits = teacher_model(batch_ids).logits
        soft_targets = F.softmax(logits / temperature, dim=-1)

        # Keep only top-K probs to save memory
        topk = 100
        values, indices = soft_targets.topk(topk, dim=-1)
        sparse_targets = torch.zeros_like(soft_targets)
        sparse_targets.scatter_(-1, indices, values)
        sparse_targets = sparse_targets / sparse_targets.sum(dim=-1, keepdim=True)

        all_input_ids.append(batch_ids.cpu())
        all_soft_targets.append(sparse_targets.cpu().half())

    input_ids = torch.cat(all_input_ids)
    soft_targets = torch.cat(all_soft_targets)
    logger.info("Collected teacher logits: inputs=%s, targets=%s", input_ids.shape, soft_targets.shape)
    return TensorDataset(input_ids, soft_targets)


def train_student(
    student_model,
    train_dataset: TensorDataset,
    args,
    rank: int = 0,
    world_size: int = 1,
):
    """Train student to match teacher's soft targets via KL divergence."""
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    student_model = student_model.to(device)

    if world_size > 1:
        student_model = DDP(student_model, device_ids=[rank])

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_training_steps = len(loader) * args.num_epochs
    warmup_steps = int(num_training_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return max(0.01, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training student for %d epochs (%d steps)", args.num_epochs, num_training_steps)

    global_step = 0
    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        student_model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_ids, batch_targets in tqdm(
            loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=(rank != 0)
        ):
            batch_ids = batch_ids.to(device)
            batch_targets = batch_targets.to(device).float()

            student_logits = student_model(batch_ids).logits
            student_log_probs = F.log_softmax(student_logits / args.temperature, dim=-1)

            loss = F.kl_div(student_log_probs, batch_targets, reduction="batchmean")
            loss = loss * (args.temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if rank == 0 and global_step % 100 == 0:
                logger.info(
                    "Step %d | loss=%.4f | lr=%.2e",
                    global_step, loss.item(), scheduler.get_last_lr()[0],
                )

        avg_loss = epoch_loss / max(1, num_batches)
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

        metrics = {
            "best_loss": best_loss,
            "num_epochs": args.num_epochs,
            "num_queries": args.num_queries,
            "teacher_model": args.teacher_model,
            "student_model": args.student_model,
            "temperature": args.temperature,
        }
        with open(output_dir / "distillation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Distillation complete. Results in %s", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Behavioral Distillation")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--num_queries", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="results/distillation")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="wikitext")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    setup_logging(rank)

    config = load_config(args.config)
    torch.manual_seed(args.seed)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    logger.info("Loading teacher model: %s", args.teacher_model)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    logger.info("Loading student model: %s", args.student_model)
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Building query dataset (%d queries)...", args.num_queries)
    query_dataset = build_query_dataset(
        tokenizer, args.num_queries, args.max_seq_len, args.seed, args.dataset
    )

    if rank == 0:
        logger.info("Collecting teacher logits...")
        distill_dataset = collect_teacher_logits(
            teacher_model, query_dataset, args.batch_size, device, args.temperature
        )
        torch.save(distill_dataset, Path(args.output_dir) / "distill_dataset.pt")
    else:
        distill_dataset = None

    if world_size > 1:
        dist.barrier()
        if rank != 0:
            distill_dataset = torch.load(Path(args.output_dir) / "distill_dataset.pt")

    del teacher_model
    torch.cuda.empty_cache()

    train_student(student_model, distill_dataset, args, rank, world_size)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
