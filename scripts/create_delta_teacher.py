#!/usr/bin/env python3
"""Create a delta teacher: fine-tune base model on a held-out domain for a few steps.

This creates a teacher whose behavior differs from the public base model,
so CE-only fine-tuning on public data CANNOT recover the teacher's private behavior.

No LoRA needed — just full fine-tune for N steps on a domain-specific dataset.
The key is that the fine-tuning data is PRIVATE (not available to the attacker).

Usage:
    python scripts/create_delta_teacher.py --config configs/tdart_smoke.yaml --dry_run
    python scripts/create_delta_teacher.py --base_model Qwen/Qwen2.5-0.5B \
        --finetune_steps 500 --output_dir teachers/delta_teacher_v1
"""
from __future__ import annotations
import argparse, json, logging, os, sys, hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--finetune_steps", type=int, default=500)
    p.add_argument("--finetune_lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--output_dir", default="teachers/delta_teacher_v1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--config", type=str, default=None)
    return p.parse_args()


class PrivateDomainDataset(Dataset):
    """Simulates a private fine-tuning dataset by using WikiText TEST split.

    The attacker only has access to TRAIN split for CE-only. TEST is private.
    This creates a realistic teacher-student gap: teacher knows test-domain
    patterns that the attacker's CE-only on train cannot recover.
    """
    def __init__(self, tokenizer, seq_len, max_samples=10000):
        self.samples = []
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            buf = []
            for ex in ds:
                text = ex.get("text", "").strip()
                if len(text) < 20:
                    continue
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                buf.extend(ids)
                while len(buf) >= seq_len:
                    self.samples.append(torch.tensor(buf[:seq_len], dtype=torch.long))
                    buf = buf[seq_len:]
                    if len(self.samples) >= max_samples:
                        break
                if len(self.samples) >= max_samples:
                    break
        except Exception as e:
            logger.warning("Could not load WikiText test: %s", e)
            rng = torch.Generator().manual_seed(99)
            V = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 32000
            for _ in range(min(max_samples, 1000)):
                self.samples.append(torch.randint(3, V, (seq_len,), generator=rng))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def main():
    setup_logging()
    args = parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k in ["base_model", "finetune_steps", "finetune_lr", "batch_size",
                   "seq_len", "output_dir", "device", "seed"]:
            if k in cfg:
                setattr(args, k, cfg[k])
        if isinstance(args.finetune_lr, str):
            args.finetune_lr = float(args.finetune_lr)

    if args.dry_run:
        logger.info("DRY RUN — delta teacher creation plan:")
        logger.info("  Base model: %s", args.base_model)
        logger.info("  Fine-tune steps: %d", args.finetune_steps)
        logger.info("  Fine-tune data: WikiText-103 TEST split (private)")
        logger.info("  Output: %s", args.output_dir)
        logger.info("  Reference (attacker has): WikiText-103 TRAIN split")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16
    ).to(args.device)

    logger.info("Loading private fine-tune data (WikiText TEST)...")
    ft_ds = PrivateDomainDataset(tokenizer, args.seq_len)
    ft_loader = DataLoader(ft_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=0, drop_last=True)
    logger.info("Fine-tune samples: %d", len(ft_ds))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr,
                                   weight_decay=0.01)

    model.train()
    step = 0
    ft_iter = iter(ft_loader)
    while step < args.finetune_steps:
        try:
            batch = next(ft_iter)
        except StopIteration:
            ft_iter = iter(ft_loader)
            batch = next(ft_iter)

        ids = batch.to(args.device)
        out = model(input_ids=ids, labels=ids)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        if step % 50 == 0:
            logger.info("Step %d/%d  loss=%.4f", step, args.finetune_steps, loss.item())

    model.eval()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    manifest = {
        "base_model": args.base_model,
        "finetune_steps": args.finetune_steps,
        "finetune_lr": args.finetune_lr,
        "finetune_data": "wikitext-103-raw-v1/test (PRIVATE — attacker does not have this)",
        "attacker_data": "wikitext-103-raw-v1/train (PUBLIC)",
        "seed": args.seed,
        "output_dir": str(out_dir),
    }
    with open(out_dir / "teacher_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Delta teacher saved: %s", out_dir)
    logger.info("Manifest: %s", out_dir / "teacher_manifest.json")


if __name__ == "__main__":
    main()
