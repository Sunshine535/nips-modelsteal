#!/usr/bin/env python3
"""Enhanced model cloning: logit KD + hidden-state representation KD.

Three variants (run sequentially on same GPU):
  A) logit_only    — standard Clone 2025 baseline: KL(teacher, student) on logits
  B) h_final_oracle — logit KD + oracle h_final MSE (upper bound)
  C) h_final_attack — logit KD + RECOVERED h_final MSE (realistic attack via logit_bias lstsq)

Evaluation (on held-out WikiText-103 validation):
  - Perplexity (student standalone)
  - KL(teacher || student)
  - Top-1 agreement rate
  - Cosine similarity of logit distributions

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/enhanced_kd_clone.py \
        --num_steps 5000 --batch_size 8 --eval_every 1000
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--num_steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--kd_temp", type=float, default=2.0)
    p.add_argument("--kd_alpha", type=float, default=0.9,
                   help="Weight on soft KD loss vs hard CE loss")
    p.add_argument("--h_beta", type=float, default=0.3,
                   help="Weight on h_final MSE loss (variants B/C/D)")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--K_probe", type=int, default=2000,
                   help="Number of probe tokens for h_final recovery (variant C)")
    p.add_argument("--output_dir", default="results/v7_enhanced_kd")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, split, seq_len, max_samples=50000):
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        except:
            ds = None
        self.samples = []
        if ds:
            buf = []
            for ex in ds:
                text = ex.get("text", "").strip()
                if len(text) < 20: continue
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                buf.extend(ids)
                while len(buf) >= seq_len:
                    self.samples.append(torch.tensor(buf[:seq_len], dtype=torch.long))
                    buf = buf[seq_len:]
                    if len(self.samples) >= max_samples:
                        break
                if len(self.samples) >= max_samples:
                    break
        if not self.samples:
            rng = torch.Generator().manual_seed(42)
            V = tokenizer.vocab_size
            for _ in range(max_samples):
                self.samples.append(torch.randint(3, V, (seq_len,), generator=rng))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


@torch.no_grad()
def evaluate(teacher, student, eval_loader, device, max_batches=50):
    """Compute KL, top-1 agreement, student perplexity on eval set."""
    kls, top1s, losses = [], [], []
    for i, batch in enumerate(eval_loader):
        if i >= max_batches: break
        ids = batch.to(device)
        t_out = teacher(input_ids=ids)
        s_out = student(input_ids=ids)
        t_logits = t_out.logits[:, :-1, :].float()
        s_logits = s_out.logits[:, :-1, :].float()
        labels = ids[:, 1:]
        # KL
        t_lp = F.log_softmax(t_logits, dim=-1)
        s_lp = F.log_softmax(s_logits, dim=-1)
        kl = (t_lp.exp() * (t_lp - s_lp)).sum(-1).mean().item()
        kls.append(kl)
        # top-1
        t1 = (t_logits.argmax(-1) == s_logits.argmax(-1)).float().mean().item()
        top1s.append(t1)
        # student CE loss → perplexity
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


def recover_h_final_lstsq(z_logits, W_probe, probe_ids, device):
    """Simulate logit_bias h_final recovery: lstsq(W_probe, z_probe).

    Supports both 2D (B, V) and 3D (B, T, V) input for all-position recovery.
    """
    if z_logits.dim() == 3:
        B, T, V = z_logits.shape
        z_flat = z_logits.reshape(B * T, V)
        z_probe = z_flat[:, probe_ids].float()
        sol = torch.linalg.lstsq(W_probe.double().to(device),
                                  z_probe.double().T).solution
        return sol.T.float().reshape(B, T, -1)
    z_probe = z_logits[:, probe_ids].float()
    sol = torch.linalg.lstsq(W_probe.double().to(device),
                              z_probe.double().T).solution
    return sol.T.float()


class GaugeProjection(nn.Module):
    """Learned linear projection to absorb Carlini's gauge ambiguity."""
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


def train_variant(name, teacher, student, train_loader, eval_loader,
                  args, device, use_h=False, use_recovered_h=False,
                  all_pos=False, W_probe=None, probe_ids=None, W_lm=None):
    """Train one KD variant.

    all_pos: if True, recover h_final at ALL positions (SCRD-all).
             if False and use_recovered_h, recover at last position only.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Training variant: %s", name)
    logger.info("=" * 60)

    gauge_proj = None
    extra_params = list(student.parameters())
    if use_recovered_h:
        d = student.config.hidden_size
        gauge_proj = GaugeProjection(d).to(device)
        extra_params = extra_params + list(gauge_proj.parameters())

    optimizer = torch.optim.AdamW(extra_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    teacher.eval()
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
        with torch.no_grad():
            t_out = teacher(input_ids=ids, output_hidden_states=use_h or use_recovered_h)
            t_logits = t_out.logits.float()

        s_out = student(input_ids=ids, output_hidden_states=use_h or use_recovered_h)
        s_logits = s_out.logits.float()

        # Soft KD loss (KL on softened logits)
        loss_kd = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        # Hard CE loss
        labels = ids[:, 1:].contiguous()
        loss_ce = F.cross_entropy(
            s_logits[:, :-1, :].reshape(-1, s_logits.size(-1)),
            labels.reshape(-1),
        )

        loss = args.kd_alpha * loss_kd + (1 - args.kd_alpha) * loss_ce

        # Hidden-state representation loss
        if use_h and not use_recovered_h:
            # Oracle h_final (ALL positions for fair comparison)
            if all_pos:
                t_h = t_out.hidden_states[-1].float()       # (B, T, d)
                s_h = s_out.hidden_states[-1].float()       # (B, T, d)
            else:
                t_h = t_out.hidden_states[-1][:, -1, :].float()
                s_h = s_out.hidden_states[-1][:, -1, :].float()
            loss_h = F.mse_loss(s_h, t_h)
            loss = loss + args.h_beta * loss_h
        elif use_recovered_h:
            # Recovered h_final via lstsq (simulating logit_bias attack)
            with torch.no_grad():
                if all_pos:
                    h_recovered = recover_h_final_lstsq(
                        t_logits, W_probe, probe_ids, device)  # (B, T, d)
                else:
                    t_z_last = t_logits[:, -1, :]  # (B, V)
                    h_recovered = recover_h_final_lstsq(
                        t_z_last, W_probe, probe_ids, device)  # (B, d)
            if all_pos:
                s_h = s_out.hidden_states[-1].float()       # (B, T, d)
                s_h_proj = gauge_proj(s_h)                   # absorb gauge
            else:
                s_h = s_out.hidden_states[-1][:, -1, :].float()
                s_h_proj = gauge_proj(s_h)
            loss_h = F.mse_loss(s_h_proj, h_recovered)
            loss = loss + args.h_beta * loss_h

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
            metrics = evaluate(teacher, student, eval_loader, device, args.eval_batches)
            logger.info("[%s] step %d: KL=%.4f top1=%.4f ppl=%.2f",
                        name, step, metrics["kl"], metrics["top1"], metrics["ppl"])
            metrics["step"] = step
            history.append(metrics)
            student.train()

    pbar.close()
    # Final eval
    student.eval()
    final = evaluate(teacher, student, eval_loader, device, args.eval_batches)
    logger.info("[%s] FINAL: KL=%.4f top1=%.4f ppl=%.2f",
                name, final["kl"], final["top1"], final["ppl"])
    return {"name": name, "history": history, "final": final}


def main():
    setup_logging()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading teacher: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    V = teacher.config.vocab_size
    d = teacher.config.hidden_size

    # Prepare probe matrix for h_final recovery (variant C)
    W_lm = teacher.lm_head.weight.data.float()
    probe_ids = torch.arange(args.K_probe)
    W_probe = W_lm[probe_ids].cpu()
    logger.info("W_probe cond = %.2f", torch.linalg.cond(W_probe).item())

    # Build datasets
    logger.info("Building datasets...")
    train_ds = WikiTextDataset(tokenizer, "train", args.seq_len, max_samples=50000)
    eval_ds = WikiTextDataset(tokenizer, "validation", args.seq_len, max_samples=2000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)
    logger.info("Train: %d samples, Eval: %d samples", len(train_ds), len(eval_ds))

    # Teacher perplexity baseline
    logger.info("Computing teacher perplexity baseline...")
    teacher_metrics = evaluate(teacher, teacher, eval_loader, args.device, args.eval_batches)
    logger.info("Teacher baseline: KL=%.4f top1=%.4f ppl=%.2f",
                teacher_metrics["kl"], teacher_metrics["top1"], teacher_metrics["ppl"])

    all_results = {"teacher_baseline": teacher_metrics, "args": vars(args), "variants": {}}

    # ═══════════════════════════════════════════════════════════════════
    # Variant A: logit-only KD (Clone 2025 baseline)
    # ═══════════════════════════════════════════════════════════════════
    torch.manual_seed(args.seed)
    student_a = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(args.device)
    # Randomize all weights for fair comparison
    with torch.no_grad():
        for name, p in student_a.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.02)
            elif "norm" in name.lower() or "layernorm" in name.lower():
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)

    res_a = train_variant("A_logit_only", teacher, student_a, train_loader,
                          eval_loader, args, args.device)
    all_results["variants"]["A_logit_only"] = res_a
    del student_a
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════
    # Variant B: logit KD + oracle h_final
    # ═══════════════════════════════════════════════════════════════════
    torch.manual_seed(args.seed)
    student_b = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(args.device)
    with torch.no_grad():
        for name, p in student_b.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.02)
            elif "norm" in name.lower() or "layernorm" in name.lower():
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)

    res_b = train_variant("B_h_oracle", teacher, student_b, train_loader,
                          eval_loader, args, args.device, use_h=True, all_pos=False)
    all_results["variants"]["B_h_oracle"] = res_b
    del student_b
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════
    # Variant C: logit KD + recovered h_final (attack-realistic)
    # ═══════════════════════════════════════════════════════════════════
    torch.manual_seed(args.seed)
    student_c = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(args.device)
    with torch.no_grad():
        for name, p in student_c.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.02)
            elif "norm" in name.lower() or "layernorm" in name.lower():
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)

    res_c = train_variant("C_h_attack_last", teacher, student_c, train_loader,
                          eval_loader, args, args.device,
                          use_recovered_h=True, all_pos=False,
                          W_probe=W_probe, probe_ids=probe_ids, W_lm=W_lm)
    all_results["variants"]["C_h_attack_last"] = res_c
    del student_c
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════
    # Variant D: SCRD — logit KD + recovered h_final at ALL positions
    # ═══════════════════════════════════════════════════════════════════
    torch.manual_seed(args.seed)
    student_d = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(args.device)
    with torch.no_grad():
        for pname, p in student_d.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.02)
            elif "norm" in pname.lower() or "layernorm" in pname.lower():
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)

    res_d = train_variant("D_SCRD_allpos", teacher, student_d, train_loader,
                          eval_loader, args, args.device,
                          use_recovered_h=True, all_pos=True,
                          W_probe=W_probe, probe_ids=probe_ids, W_lm=W_lm)
    all_results["variants"]["D_SCRD_allpos"] = res_d
    del student_d
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════
    # Variant B_allpos: oracle h_final at ALL positions (upper bound for D)
    # ═══════════════════════════════════════════════════════════════════
    torch.manual_seed(args.seed)
    student_b2 = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(args.device)
    with torch.no_grad():
        for pname, p in student_b2.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.02)
            elif "norm" in pname.lower() or "layernorm" in pname.lower():
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)

    res_b2 = train_variant("B2_oracle_allpos", teacher, student_b2, train_loader,
                           eval_loader, args, args.device,
                           use_h=True, all_pos=True)
    all_results["variants"]["B2_oracle_allpos"] = res_b2
    del student_b2
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    variant_names = ["A_logit_only", "B_h_oracle", "B2_oracle_allpos",
                     "C_h_attack_last", "D_SCRD_allpos"]

    logger.info("\n" + "=" * 60)
    logger.info("SCRD ENHANCED KD CLONE — FINAL RESULTS")
    logger.info("=" * 60)
    logger.info("%-22s %8s %8s %8s %8s", "Variant", "KL", "Top-1", "PPL", "PPL-deg%")
    logger.info("-" * 60)
    t_ppl = teacher_metrics["ppl"]
    logger.info("%-22s %8.4f %8.4f %8.2f %8s", "Teacher (ref)",
                teacher_metrics["kl"], teacher_metrics["top1"], t_ppl, "0.0%")
    for vname in variant_names:
        f = all_results["variants"][vname]["final"]
        deg = (f["ppl"] - t_ppl) / t_ppl * 100
        all_results["variants"][vname]["ppl_degradation_pct"] = deg
        logger.info("%-22s %8.4f %8.4f %8.2f %7.1f%%",
                     vname, f["kl"], f["top1"], f["ppl"], deg)

    out_path = Path(args.output_dir) / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nSaved: %s", out_path)


if __name__ == "__main__":
    main()
