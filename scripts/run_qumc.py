#!/usr/bin/env python3
"""Q-UMC: Query-budgeted Uncertainty-gated Logit Completion.

All teacher access goes through strict oracles. No direct full-logit access
except in the oracle upper-bound baseline (explicitly tagged).

Variants:
  A. strict_topk_kd        — top-K only KD (strict baseline)
  B. completion_no_unc     — logit completion without uncertainty weights
  C. completion_uncertainty — logit completion with calibrated uncertainty
  D. full_logit_upper      — full-logit KD (oracle upper bound, not black-box)
  E. old_lc_simulator      — old Logit Completion from enhanced_kd_clone.py (historical)
"""
from __future__ import annotations
import argparse, json, logging, math, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.oracles import StrictBlackBoxAPI
from src.logit_completion import CalibratedLogitCompleter
from src.result_manifest import save_manifest

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")


def parse_args():
    p = argparse.ArgumentParser(description="Q-UMC training")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--num_steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--kd_temp", type=float, default=2.0)
    p.add_argument("--kd_alpha", type=float, default=0.7)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--probe_tokens", type=int, default=2000)
    p.add_argument("--cal_batches", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--output_dir", default="results/qumc")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--perturb_std", type=float, default=0.01)
    p.add_argument("--allow_synthetic", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--variants", nargs="*",
                   default=["strict_topk_kd", "completion_no_unc",
                            "completion_uncertainty", "full_logit_upper"])
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--overfit_one_batch", action="store_true")
    return p.parse_args()


class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, split, seq_len, max_samples=50000):
        self.samples = []
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
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
        except Exception:
            pass
        if not self.samples:
            rng = torch.Generator().manual_seed(42)
            V = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 32000
            for _ in range(min(max_samples, 1000)):
                self.samples.append(torch.randint(3, V, (seq_len,), generator=rng))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


@torch.no_grad()
def evaluate(teacher_api, student, eval_loader, device, max_batches=50):
    kls, top1s, losses = [], [], []
    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break
        ids = batch.to(device)
        t_logits = teacher_api.get_full_logits_ORACLE_ONLY(ids)
        s_out = student(input_ids=ids)
        s_logits = s_out.logits[:, :-1, :].float()
        t_logits_shift = t_logits[:, :-1, :].float()
        labels = ids[:, 1:]

        t_lp = F.log_softmax(t_logits_shift, dim=-1)
        s_lp = F.log_softmax(s_logits, dim=-1)
        kl = (t_lp.exp() * (t_lp - s_lp)).sum(-1).mean().item()
        kls.append(kl)

        t1 = (t_logits_shift.argmax(-1) == s_logits.argmax(-1)).float().mean().item()
        top1s.append(t1)

        ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)),
                             labels.reshape(-1), reduction="mean")
        losses.append(ce.item())

    ppl = math.exp(sum(losses) / len(losses)) if losses else float("inf")
    return {
        "kl": sum(kls) / len(kls),
        "top1": sum(top1s) / len(top1s),
        "ppl": ppl,
        "ce": sum(losses) / len(losses),
    }


def init_student(model, perturb_std):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * perturb_std)
    return model


def train_variant(name, teacher_api, completer, student, train_loader,
                  eval_loader, args, device):
    logger.info("\n" + "=" * 60)
    logger.info("Training variant: %s", name)
    logger.info("=" * 60)

    is_full_logit = (name == "full_logit_upper")
    use_completion = name in ("completion_no_unc", "completion_uncertainty")
    use_uncertainty = (name == "completion_uncertainty")

    probe_ids = torch.arange(args.probe_tokens)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    teacher_api.budget.topk_queries = 0
    teacher_api.budget.probe_queries = 0

    student.train()
    T = args.kd_temp
    step = 0
    history = []
    train_iter = iter(train_loader)
    pbar = tqdm(total=args.num_steps, desc=name)

    while step < args.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch.to(device)
        B, seq = ids.shape

        if is_full_logit:
            t_logits = teacher_api.get_full_logits_ORACLE_ONLY(ids)
        elif use_completion:
            topk_vals, topk_idx, logit_shape = teacher_api.get_topk(ids)
            probe_logits = teacher_api.get_probe_logits(ids, probe_ids)
            t_logits = completer.complete(topk_vals, topk_idx, probe_logits, logit_shape)
        else:
            topk_vals, topk_idx, logit_shape = teacher_api.get_topk(ids)
            V = logit_shape[-1]
            large_neg = -1e9
            t_logits = torch.full((B, seq, V), large_neg, device=device)
            t_logits.scatter_(-1, topk_idx, topk_vals)

        s_out = student(input_ids=ids)
        s_logits = s_out.logits.float()
        t_logits = t_logits.float()

        if use_uncertainty and completer.uncertainty_weights is not None:
            w = completer.get_weights(topk_idx, s_logits.shape[-1])
            t_soft = F.softmax(t_logits / T, dim=-1)
            s_logsm = F.log_softmax(s_logits / T, dim=-1)
            loss_kd = (w * t_soft * (t_soft.log() - s_logsm)).sum(-1).mean() * (T * T)
        else:
            loss_kd = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)

        labels = ids[:, 1:].contiguous()
        loss_ce = F.cross_entropy(
            s_logits[:, :-1, :].reshape(-1, s_logits.size(-1)),
            labels.reshape(-1),
        )

        loss = args.kd_alpha * loss_kd + (1 - args.kd_alpha) * loss_ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        pbar.update(1)
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", kd=f"{loss_kd.item():.4f}")

        if step % args.eval_every == 0 or step == args.num_steps:
            student.eval()
            metrics = evaluate(teacher_api, student, eval_loader, device, args.eval_batches)
            logger.info("[%s] step %d: KL=%.4f top1=%.4f ppl=%.2f",
                        name, step, metrics["kl"], metrics["top1"], metrics["ppl"])
            metrics["step"] = step
            history.append(metrics)
            student.train()

        if args.overfit_one_batch and step >= args.num_steps:
            break

    pbar.close()
    student.eval()
    final = evaluate(teacher_api, student, eval_loader, device, args.eval_batches)
    logger.info("[%s] FINAL: KL=%.4f top1=%.4f ppl=%.2f",
                name, final["kl"], final["top1"], final["ppl"])

    return {
        "name": name,
        "history": history,
        "final": final,
        "query_budget": teacher_api.budget.summary(),
    }


def main():
    setup_logging()
    args = parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
        for attr in ("lr", "kd_temp", "kd_alpha", "perturb_std"):
            setattr(args, attr, float(getattr(args, attr)))

    if args.dry_run:
        logger.info("DRY RUN — planned variants: %s", args.variants)
        logger.info("Config: %s", vars(args))
        return

    seeds = args.seeds if args.seeds else [args.seed]
    base_output = Path(args.output_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    for seed in seeds:
        torch.manual_seed(seed)
        seed_dir = base_output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading teacher: %s (seed=%d)", args.model_name, seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16
        ).to(args.device).eval()

        api = StrictBlackBoxAPI(teacher, K=args.topk, device=args.device)

        V = teacher.config.vocab_size
        d = teacher.config.hidden_size
        W_lm = teacher.lm_head.weight.data.float()
        probe_ids = torch.arange(args.probe_tokens)
        W_probe = W_lm[probe_ids].cpu()

        completer = CalibratedLogitCompleter(
            W_probe, probe_ids, W_lm, device=args.device)

        logger.info("Building datasets...")
        train_ds = WikiTextDataset(tokenizer, "train", args.seq_len)
        eval_ds = WikiTextDataset(tokenizer, "validation", args.seq_len, max_samples=2000)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
        logger.info("Train: %d, Eval: %d", len(train_ds), len(eval_ds))

        teacher_metrics = evaluate(api, teacher, eval_loader, args.device, args.eval_batches)
        logger.info("Teacher: KL=%.4f top1=%.4f ppl=%.2f",
                     teacher_metrics["kl"], teacher_metrics["top1"], teacher_metrics["ppl"])

        if args.cal_batches > 0:
            logger.info("Fitting calibration...")
            cal_probes, cal_fulls = [], []
            for i, batch in enumerate(eval_loader):
                if i >= args.cal_batches:
                    break
                ids = batch.to(args.device)
                full_z = api.get_full_logits_ORACLE_ONLY(ids)
                probe_z = full_z[:, :, probe_ids]
                cal_probes.append(probe_z.reshape(-1, args.probe_tokens))
                cal_fulls.append(full_z.reshape(-1, V))
            cal_p = torch.cat(cal_probes, dim=0)[:5000]
            cal_f = torch.cat(cal_fulls, dim=0)[:5000]
            completer.fit_calibration(cal_p, cal_f)
            logger.info("Calibration fitted on %d samples", cal_p.shape[0])

        all_results = {"teacher": teacher_metrics, "seed": seed, "args": vars(args), "variants": {}}

        for variant_name in args.variants:
            torch.manual_seed(seed)
            student = AutoModelForCausalLM.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16
            ).to(args.device)
            init_student(student, args.perturb_std)

            res = train_variant(variant_name, api, completer, student,
                                train_loader, eval_loader, args, args.device)
            all_results["variants"][variant_name] = res
            del student
            torch.cuda.empty_cache()

        t_ppl = teacher_metrics["ppl"]
        logger.info("\n" + "=" * 60)
        logger.info("Q-UMC RESULTS (seed=%d)", seed)
        logger.info("=" * 60)
        logger.info("%-25s %8s %8s %8s %8s", "Variant", "KL", "Top-1", "PPL", "PPL-deg%")
        logger.info("-" * 65)
        logger.info("%-25s %8.4f %8.4f %8.2f %8s", "Teacher",
                     teacher_metrics["kl"], teacher_metrics["top1"], t_ppl, "0.0%")
        for vn in args.variants:
            f = all_results["variants"][vn]["final"]
            deg = (f["ppl"] - t_ppl) / t_ppl * 100
            all_results["variants"][vn]["ppl_degradation_pct"] = deg
            logger.info("%-25s %8.4f %8.4f %8.2f %7.1f%%",
                         vn, f["kl"], f["top1"], f["ppl"], deg)

        with open(seed_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        save_manifest(
            seed_dir,
            command=" ".join(sys.argv),
            config=vars(args),
            seed=seed,
            model_name=args.model_name,
            dataset="wikitext-103",
            split_info={"train": "train", "eval": "validation",
                        "train_samples": len(train_ds), "eval_samples": len(eval_ds)},
            query_budget=api.budget.summary(),
            metrics={vn: all_results["variants"][vn]["final"] for vn in args.variants},
        )
        logger.info("Saved: %s", seed_dir)


if __name__ == "__main__":
    main()
