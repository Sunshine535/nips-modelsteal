#!/usr/bin/env python3
"""Q-UMC training (R2-compliant).

Per GPT-5.5 Pro R2 Tasks 2-7:
  - BasisProvider controls completion basis (no teacher W_lm in strict variants)
  - Calibration uses disjoint calibration split via counted probe oracle only
  - Unified sequence_kl_loss across all variants (no normalization confound)
  - Mechanism logs: tail MSE, weight stats, KD/CE ratio, per-variant budgets
  - Manifest: per-variant + calibration budgets, basis_source, strict_access_audit

Variants:
  A. strict_topk_kd               — top-K only KD (strict, sparse)
  A'. old_lc_simulator            — reproduce old leaked-path completion (historical)
  B. completion_no_unc_strict     — Carlini W_eff completion, no uncertainty
  C. completion_uncertainty_strict — + uncertainty from disjoint-probe calibration
  D. full_logit_upper             — ORACLE: full teacher logits (upper bound)
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
from src.basis_provider import BasisProvider
from src.kd_losses import sequence_kl_loss, sequence_ce_loss
from src.result_manifest import save_manifest

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")


def parse_args():
    p = argparse.ArgumentParser(description="Q-UMC training (R2)")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--num_steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--kd_temp", type=float, default=2.0)
    p.add_argument("--kd_alpha", type=float, default=0.7)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--probe_tokens", type=int, default=2000)
    p.add_argument("--heldout_probe_tokens", type=int, default=500,
                   help="Disjoint probe set for calibration (counted vs budget)")
    p.add_argument("--cal_examples", type=int, default=500,
                   help="Number of prompts in calibration split")
    p.add_argument("--carlini_examples", type=int, default=2000,
                   help="Number of prompts for Carlini SVD basis recovery")
    p.add_argument("--max_probe_budget", type=int, default=-1)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--output_dir", default="results/qumc_v2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--perturb_std", type=float, default=0.01)
    p.add_argument("--basis_source", default="carlini_recovered",
                   choices=["carlini_recovered", "teacher_oracle",
                            "public_pretrained", "null"])
    p.add_argument("--allow_synthetic", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--variants", nargs="*",
                   default=["strict_topk_kd", "completion_no_unc_strict",
                            "completion_uncertainty_strict", "full_logit_upper"])
    p.add_argument("--config", type=str, default=None)
    return p.parse_args()


class WikiTextSplit(Dataset):
    def __init__(self, tokenizer, split, seq_len, max_samples=50000,
                 allow_synthetic=False):
        self.samples = []
        self.source = "none"
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
            self.source = f"wikitext-103-raw-v1/{split}"
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
            if not allow_synthetic:
                raise RuntimeError(
                    f"WikiText load failed for split={split}: {e}. "
                    f"Set --allow_synthetic to fall back.")
            self.source = f"synthetic_fallback/{split}"
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
    """Evaluation uses full teacher logits (explicitly oracle access for metrics only)."""
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


def recover_carlini_basis(api, train_loader, d, V, n_examples, device):
    """Recover W_eff via Carlini SVD from top-K queries on training prompts."""
    batches = []
    seen = 0
    for batch in train_loader:
        batches.append(batch.to(device))
        seen += batch.shape[0]
        if seen >= n_examples:
            break
    basis = BasisProvider.carlini_recover(api, batches, d, V, device=device)
    return basis


def calibrate_via_probes(completer, api, cal_loader, probe_ids,
                         heldout_probe_ids, basis_W, n_examples, device):
    """Run calibration using ONLY probe oracle on disjoint calibration split."""
    cal_probe_in_rows = []
    heldout_target_rows = []
    seen = 0
    all_probe_ids = torch.cat([probe_ids, heldout_probe_ids])
    for batch in cal_loader:
        ids = batch.to(device)
        combined = api.get_probe_logits(ids, all_probe_ids)
        B, T, K_total = combined.shape
        combined_flat = combined.reshape(B * T, K_total)
        cal_probe_in_rows.append(combined_flat[:, :len(probe_ids)].cpu())
        heldout_target_rows.append(
            combined_flat[:, len(probe_ids):].cpu())
        seen += B * T
        if seen >= n_examples:
            break
    cal_probe_in = torch.cat(cal_probe_in_rows, dim=0)[:n_examples]
    heldout_targets = torch.cat(heldout_target_rows, dim=0)[:n_examples]
    heldout_W = basis_W[heldout_probe_ids]
    diag = completer.fit_calibration_from_probes(
        cal_input_probe_logits=cal_probe_in.to(device),
        heldout_probe_ids=heldout_probe_ids,
        heldout_probe_targets=heldout_targets.to(device),
        heldout_W_rows=heldout_W.to(device),
    )
    return diag


def train_variant(name, teacher_api, completer, student, train_loader,
                  eval_loader, args, device):
    logger.info("\n" + "=" * 60)
    logger.info("Training variant: %s", name)
    logger.info("=" * 60)

    is_full_logit = (name == "full_logit_upper")
    use_completion = name in ("completion_no_unc_strict",
                              "completion_uncertainty_strict",
                              "completion_uncertainty_normed_kl")
    use_uncertainty = name in ("completion_uncertainty_strict",
                                "completion_uncertainty_normed_kl")
    use_normalized_kl = (name == "completion_uncertainty_normed_kl")
    is_old_simulator = (name == "old_lc_simulator")
    is_old_simulator_oracle = (name == "old_lc_simulator_teacherW")
    is_ce_only = (name == "ce_only")
    is_topk_tailzero = (name == "strict_topk_weighted_tailzero")

    probe_ids = completer.probe_ids.cpu() if completer is not None else None

    start_topk = teacher_api.budget.topk_queries
    start_probe = teacher_api.budget.probe_queries

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    student.train()
    step = 0
    history = []
    mech_log = {
        "variant": name,
        "kd_mean": [], "ce_mean": [], "kd_ce_ratio": [],
        "weight_mean_on_tail": [],
    }
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

        weights = None
        topk_idx = None
        if is_ce_only:
            t_logits = None
        elif is_full_logit:
            t_logits = teacher_api.get_full_logits_ORACLE_ONLY(ids)
        elif is_old_simulator or is_old_simulator_oracle:
            t_logits_full = teacher_api.get_full_logits_ORACLE_ONLY(ids)
            V = t_logits_full.shape[-1]
            z_probe_leaked = t_logits_full[:, :, probe_ids.to(device)]
            h_hat = completer.recover_h(z_probe_leaked)
            z_hat = completer.reconstruct_logits(h_hat)
            topk_vals, topk_idx = t_logits_full.topk(args.topk, dim=-1)
            t_logits = z_hat.clone()
            t_logits.scatter_(-1, topk_idx, topk_vals)
        elif use_completion:
            topk_vals, topk_idx, logit_shape = teacher_api.get_topk(ids)
            probe_logits = teacher_api.get_probe_logits(ids, probe_ids.to(device))
            t_logits = completer.complete(topk_vals, topk_idx, probe_logits,
                                          logit_shape)
            V = logit_shape[-1]
            if use_uncertainty and completer.uncertainty_weights is not None:
                weights = completer.get_weights(topk_idx, V)
        elif is_topk_tailzero:
            topk_vals, topk_idx, logit_shape = teacher_api.get_topk(ids)
            V = logit_shape[-1]
            t_logits = torch.zeros((B, seq, V), device=device)
            t_logits.scatter_(-1, topk_idx, topk_vals)
            weights = torch.zeros((B, seq, V), device=device)
            weights.scatter_(-1, topk_idx, 1.0)
        else:
            topk_vals, topk_idx, logit_shape = teacher_api.get_topk(ids)
            V = logit_shape[-1]
            large_neg = -1e9
            t_logits = torch.full((B, seq, V), large_neg, device=device)
            t_logits.scatter_(-1, topk_idx, topk_vals)

        s_out = student(input_ids=ids)
        s_logits = s_out.logits.float()

        loss_ce = sequence_ce_loss(s_logits, ids)

        if is_ce_only:
            loss_kd = torch.tensor(0.0, device=device)
            loss = loss_ce
        else:
            t_logits = t_logits.float()
            loss_kd = sequence_kl_loss(s_logits, t_logits,
                                        weights=weights,
                                        temperature=args.kd_temp,
                                        normalize_by_weight_mass=use_normalized_kl)
            loss = args.kd_alpha * loss_kd + (1 - args.kd_alpha) * loss_ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            mech_log["kd_mean"].append(float(loss_kd.item()))
            mech_log["ce_mean"].append(float(loss_ce.item()))
            mech_log["kd_ce_ratio"].append(float(loss_kd.item() / (loss_ce.item() + 1e-8)))
            if weights is not None and topk_idx is not None:
                tail_mask = torch.ones_like(weights)
                tail_mask.scatter_(-1, topk_idx, 0.0)
                tail_w = (weights * tail_mask).sum() / (tail_mask.sum() + 1e-8)
                mech_log["weight_mean_on_tail"].append(float(tail_w.item()))

        step += 1
        pbar.update(1)
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                            kd=f"{loss_kd.item():.4f}",
                            ce=f"{loss_ce.item():.4f}")

        if step % args.eval_every == 0 or step == args.num_steps:
            student.eval()
            metrics = evaluate(teacher_api, student, eval_loader,
                              device, args.eval_batches)
            logger.info("[%s] step %d: KL=%.4f top1=%.4f ppl=%.2f",
                        name, step, metrics["kl"], metrics["top1"], metrics["ppl"])
            metrics["step"] = step
            history.append(metrics)
            student.train()

    pbar.close()
    student.eval()
    final = evaluate(teacher_api, student, eval_loader, device, args.eval_batches)
    logger.info("[%s] FINAL: KL=%.4f top1=%.4f ppl=%.2f",
                name, final["kl"], final["top1"], final["ppl"])

    variant_budget = {
        "topk_queries": teacher_api.budget.topk_queries - start_topk,
        "probe_queries": teacher_api.budget.probe_queries - start_probe,
    }
    return {
        "name": name,
        "history": history,
        "final": final,
        "variant_budget": variant_budget,
        "mechanism_log": mech_log,
        "basis_source": completer.basis_source if completer is not None else "n/a",
        "basis_access_level": completer.basis_access_level if completer is not None else "n/a",
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
        logger.info("Basis source: %s", args.basis_source)
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

        api = StrictBlackBoxAPI(teacher, K=args.topk, device=args.device,
                                max_probe_budget=args.max_probe_budget)

        V = teacher.config.vocab_size
        d = teacher.config.hidden_size

        logger.info("Building disjoint splits (train/cal/val)...")
        train_ds = WikiTextSplit(tokenizer, "train", args.seq_len,
                                  allow_synthetic=args.allow_synthetic)
        eval_ds = WikiTextSplit(tokenizer, "validation", args.seq_len,
                                 max_samples=2000,
                                 allow_synthetic=args.allow_synthetic)
        if len(train_ds) < args.cal_examples + args.carlini_examples:
            raise RuntimeError("Not enough train data to carve cal + carlini splits.")
        n_cal = args.cal_examples
        n_carlini = args.carlini_examples
        cal_samples = train_ds.samples[:n_cal]
        carlini_samples = train_ds.samples[n_cal:n_cal + n_carlini]
        train_main_samples = train_ds.samples[n_cal + n_carlini:]

        class Subset(Dataset):
            def __init__(self, s): self.s = s
            def __len__(self): return len(self.s)
            def __getitem__(self, i): return self.s[i]

        cal_ds = Subset(cal_samples)
        carlini_ds = Subset(carlini_samples)
        train_ds_main = Subset(train_main_samples)

        train_loader = DataLoader(train_ds_main, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
        cal_loader = DataLoader(cal_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
        carlini_loader = DataLoader(carlini_ds, batch_size=args.batch_size,
                                    shuffle=False, num_workers=0)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)

        logger.info("Train %d | Cal %d | Carlini %d | Eval %d",
                     len(train_ds_main), len(cal_ds), len(carlini_ds), len(eval_ds))

        teacher_metrics = evaluate(api, teacher, eval_loader,
                                    args.device, args.eval_batches)
        logger.info("Teacher: KL=%.4f top1=%.4f ppl=%.2f",
                     teacher_metrics["kl"], teacher_metrics["top1"],
                     teacher_metrics["ppl"])

        logger.info("Recovering basis via %s...", args.basis_source)
        if args.basis_source == "carlini_recovered":
            basis = recover_carlini_basis(api, carlini_loader, d, V,
                                           args.carlini_examples, args.device)
        elif args.basis_source == "teacher_oracle":
            basis = BasisProvider.teacher_oracle_unsafe(teacher)
        else:
            basis = BasisProvider.null()
        logger.info("Basis source=%s access=%s n_queries=%d",
                     basis.source, basis.access_level, basis.n_queries_used)

        basis_budget_start = dict(api.budget.summary())

        probe_ids = torch.arange(args.probe_tokens)
        heldout_probe_ids = torch.arange(args.probe_tokens,
                                          args.probe_tokens + args.heldout_probe_tokens)
        W_probe = basis.W[probe_ids].cpu() if basis.W.numel() > 0 else None

        completer = None
        cal_diag = {}
        cal_budget_start = None
        cal_budget_end = None
        if W_probe is not None:
            bp = BasisProvider(args.basis_source)
            if args.basis_source in bp.STRICT_SOURCES:
                bp.assert_allowed_for_strict()
            completer = CalibratedLogitCompleter(
                W_probe, probe_ids, basis, device=args.device)
            logger.info("Fitting calibration via probes only...")
            cal_budget_start = dict(api.budget.summary())
            cal_diag = calibrate_via_probes(
                completer, api, cal_loader,
                probe_ids, heldout_probe_ids, basis.W,
                args.cal_examples, args.device,
            )
            cal_budget_end = dict(api.budget.summary())
            logger.info("Calibration diagnostics: heldout_mse=%.4f weight_mean=%.4f",
                         cal_diag.get("heldout_mse_mean", 0.0),
                         cal_diag.get("weight_mean", 0.0))

        pre_train_budget = dict(api.budget.summary())
        all_results = {
            "teacher": teacher_metrics,
            "seed": seed,
            "args": vars(args),
            "variants": {},
            "basis": {
                "source": basis.source,
                "access_level": basis.access_level,
                "n_queries_used_in_recovery": basis.n_queries_used,
                "notes": basis.notes,
            },
            "calibration_diagnostics": cal_diag,
            "calibration_budget": (None if cal_budget_start is None else {
                "topk_queries": cal_budget_end["topk_queries"] - cal_budget_start["topk_queries"],
                "probe_queries": cal_budget_end["probe_queries"] - cal_budget_start["probe_queries"],
            }),
            "dataset_info": {
                "train_source": train_ds.source,
                "eval_source": eval_ds.source,
                "train_samples_used": len(train_ds_main),
                "cal_samples_used": len(cal_ds),
                "carlini_samples_used": len(carlini_ds),
            },
            "pre_train_budget": pre_train_budget,
        }

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

        all_results["final_budget"] = dict(api.budget.summary())

        t_ppl = teacher_metrics["ppl"]
        logger.info("\n" + "=" * 60)
        logger.info("Q-UMC R2 RESULTS (seed=%d, basis=%s)", seed, basis.source)
        logger.info("=" * 60)
        logger.info("%-30s %8s %8s %8s %8s",
                     "Variant", "KL", "Top-1", "PPL", "PPL-deg%")
        logger.info("-" * 70)
        logger.info("%-30s %8.4f %8.4f %8.2f %8s", "Teacher",
                     teacher_metrics["kl"], teacher_metrics["top1"], t_ppl, "0.0%")
        for vn in args.variants:
            f = all_results["variants"][vn]["final"]
            deg = (f["ppl"] - t_ppl) / t_ppl * 100
            all_results["variants"][vn]["ppl_degradation_pct"] = deg
            logger.info("%-30s %8.4f %8.4f %8.2f %7.1f%%",
                         vn, f["kl"], f["top1"], f["ppl"], deg)

        with open(seed_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        save_manifest(
            seed_dir,
            command=" ".join(sys.argv),
            config=vars(args),
            seed=seed,
            model_name=args.model_name,
            dataset=train_ds.source,
            split_info={
                "train": train_ds.source, "eval": eval_ds.source,
                "train_samples": len(train_ds_main),
                "cal_samples": len(cal_ds),
                "eval_samples": len(eval_ds),
            },
            query_budget=all_results["final_budget"],
            metrics={vn: all_results["variants"][vn]["final"]
                     for vn in args.variants},
            extra={
                "basis": all_results["basis"],
                "calibration_budget": all_results["calibration_budget"],
                "variant_budgets": {vn: all_results["variants"][vn]["variant_budget"]
                                    for vn in args.variants},
                "evaluation_full_logits_used_for_metrics": True,
                "strict_access_audit": {
                    "teacher_wlm_used_as_basis": basis.source == "teacher_oracle",
                    "calibration_uses_probes_only": True,
                    "unified_kl_loss": True,
                },
            },
        )
        logger.info("Saved: %s", seed_dir)


if __name__ == "__main__":
    main()
