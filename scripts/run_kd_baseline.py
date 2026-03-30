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
import glob
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


def save_training_checkpoint(path, model, optimizer, epoch, step, **extra):
    model_sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({"epoch": epoch, "step": step,
                "model_state_dict": model_sd,
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                **extra}, path)


def load_training_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model_state_dict"])
    if optimizer and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0)


def find_latest_checkpoint(output_dir, pattern="checkpoint_*.pt"):
    ckpts = sorted(glob.glob(os.path.join(output_dir, pattern)),
                   key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


def setup_logging(rank: int = 0):
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[%(asctime)s][Rank {rank}] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_distributed():
    if "RANK" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        for backend in ("nccl", "gloo"):
            try:
                dist.init_process_group(backend)
                logger.info("Distributed init OK: backend=%s", backend)
                break
            except Exception as e:
                logger.warning("Backend %s failed: %s", backend, e)
                if backend == "gloo":
                    raise
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size, local_rank
    return 0, 1, 0


def build_query_dataset(tokenizer, num_queries: int, max_seq_len: int, seed: int = 42):
    """Build queries from wikitext, falling back to random tokens."""
    rng = torch.Generator().manual_seed(seed)
    input_ids_list = []

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
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
def collect_teacher_logits(teacher_model, dataset, batch_size, device, temperature=1.0, topk=128):
    """Query teacher and collect top-K logits only (sparse storage to save memory)."""
    teacher_model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_input_ids, all_topk_probs, all_topk_indices = [], [], []

    for (batch_ids,) in tqdm(loader, desc="Collecting teacher logits"):
        batch_ids = batch_ids.to(device)
        logits = teacher_model(batch_ids).logits
        soft = F.softmax(logits / temperature, dim=-1)
        del logits

        values, indices = soft.topk(topk, dim=-1)
        del soft
        values = values / values.sum(dim=-1, keepdim=True)

        all_input_ids.append(batch_ids.cpu())
        all_topk_probs.append(values.cpu().half())
        all_topk_indices.append(indices.cpu())
        torch.cuda.empty_cache()

    return TensorDataset(
        torch.cat(all_input_ids),
        torch.cat(all_topk_probs),
        torch.cat(all_topk_indices),
    )


def train_kd_student(
    student_model, train_dataset, args, rank=0, world_size=1, local_rank=0,
    resume_from_checkpoint=False,
):
    """Train student via KL divergence distillation from sparse top-K teacher logits."""
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    student_model = student_model.to(device)

    if hasattr(student_model, "gradient_checkpointing_enable"):
        student_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    if world_size > 1:
        student_model = DDP(student_model, device_ids=[local_rank])

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None), drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    grad_accum = args.gradient_accumulation_steps
    total_opt_steps = (len(loader) // grad_accum) * args.num_epochs
    warmup_steps = int(total_opt_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return max(0.01, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    global_step = 0
    metrics_log = []
    start_epoch = 0

    if resume_from_checkpoint:
        ckpt_path = find_latest_checkpoint(str(output_dir), "checkpoint_epoch*.pt")
        if ckpt_path:
            logger.info("Resuming from %s", ckpt_path)
            start_epoch, global_step = load_training_checkpoint(ckpt_path, student_model, optimizer)
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            best_loss = ckpt_data.get("best_loss", float("inf"))
            metrics_log = ckpt_data.get("metrics_log", [])
            if ckpt_data.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
            logger.info("  Resuming from epoch %d, global_step %d", start_epoch, global_step)

    for epoch in range(start_epoch, args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        student_model.train()
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for micro_step, (batch_ids, topk_probs, topk_indices) in enumerate(tqdm(
            loader, desc=f"KD Epoch {epoch+1}/{args.num_epochs}", disable=(rank != 0)
        )):
            batch_ids = batch_ids.to(device)
            topk_probs = topk_probs.to(device).float()
            topk_indices = topk_indices.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_logits = student_model(batch_ids).logits

            scaled = student_logits.float() / args.temperature
            log_z = scaled.logsumexp(dim=-1, keepdim=True)
            student_log_probs_topk = scaled.gather(-1, topk_indices) - log_z
            del scaled, log_z

            kl_loss = F.kl_div(student_log_probs_topk, topk_probs, reduction="batchmean")
            loss = kl_loss * (args.temperature ** 2)
            del student_log_probs_topk

            hard_labels = topk_indices[..., 0]
            ce_loss = F.cross_entropy(
                student_logits.float().view(-1, student_logits.size(-1)),
                hard_labels.view(-1),
            )
            del student_logits, topk_probs, topk_indices

            total_loss = (args.alpha * loss + (1 - args.alpha) * ce_loss) / grad_accum
            total_loss.backward()

            epoch_loss += total_loss.item() * grad_accum
            num_batches += 1

            if (micro_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                torch.cuda.empty_cache()

                if rank == 0 and global_step % 100 == 0:
                    logger.info(
                        "Step %d | kl=%.4f | ce=%.4f | total=%.4f | lr=%.2e",
                        global_step, kl_loss.item(), ce_loss.item(),
                        total_loss.item() * grad_accum, scheduler.get_last_lr()[0],
                    )

        if (micro_step + 1) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

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

            save_training_checkpoint(
                str(output_dir / f"checkpoint_epoch{epoch + 1}.pt"),
                student_model, optimizer, epoch + 1, global_step,
                best_loss=best_loss, metrics_log=metrics_log,
                scheduler_state_dict=scheduler.state_dict(),
            )

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
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--query_budget", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
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
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="'auto' to resume from latest checkpoint, or path")
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
    logger.info("Teacher: %s → Student: %s (rank %d/%d)", args.teacher_model, args.student_model, rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        num_queries = args.query_budget
        logger.info("Loading teacher: %s", args.teacher_model)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=True,
        )
        logger.info("Building query dataset (%d queries)...", num_queries)
        query_dataset = build_query_dataset(tokenizer, num_queries, args.max_seq_len, args.seed)
        logger.info("Collecting teacher logits (batch_size=32)...")
        distill_dataset = collect_teacher_logits(
            teacher_model, query_dataset, batch_size=32, device=device, temperature=args.temperature,
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(distill_dataset, Path(args.output_dir) / "distill_dataset.pt")
        del teacher_model, query_dataset
        torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()
        if rank != 0:
            distill_dataset = torch.load(
                Path(args.output_dir) / "distill_dataset.pt", weights_only=False,
            )

    logger.info("Loading student: %s (random init)", args.student_model)
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    student_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    train_kd_student(student_model, distill_dataset, args, rank, world_size, local_rank,
                     resume_from_checkpoint=bool(args.resume_from_checkpoint))

    if rank == 0:
        tokenizer.save_pretrained(Path(args.output_dir) / "best_student")
        tokenizer.save_pretrained(Path(args.output_dir) / "final_student")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
