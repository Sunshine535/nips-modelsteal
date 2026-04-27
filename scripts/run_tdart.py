#!/usr/bin/env python3
"""C-DART: Censored Teacher-Delta Ranking Transfer.

Steal teacher-specific behavior by learning relative residual logits
over observed candidate tokens, anchored to a public reference model,
with censored constraints from top-K absence.

Variants:
  ce_only                  — CE on public data only (no teacher signal)
  strict_topk_kd           — top-K sparse KL (Q-UMC baseline)
  bild_topk_delta          — top-K logit difference loss (BiLD-style baseline)
  tdart_no_adaptive        — residual ranking, teacher top-K only (T-DART v1 best)
  tdart_full               — T-DART with adaptive probes (v1 negative)
  cdart_no_censor          — residual ranking without censored constraints (ablation)
  cdart_full               — MAIN: residual ranking + censored top-K absence constraints
  full_logit_upper         — oracle full-logit KD (upper bound, not strict)
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
from src.residual_delta import compute_residual, build_pairwise_preferences, compute_student_residual
from src.adaptive_candidates import select_candidates_batch
from src.ranking_losses import pairwise_residual_rank_loss, residual_mse_loss
from src.censored_delta import build_censored_candidates, censored_residual_rank_loss
from src.kd_losses import sequence_kl_loss, sequence_ce_loss
from src.result_manifest import save_manifest

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

def parse_args():
    p = argparse.ArgumentParser(description="T-DART training")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--teacher_checkpoint_path", default="teachers/delta_teacher_v1")
    p.add_argument("--num_steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--kd_temp", type=float, default=2.0)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--max_probe_tokens_per_prompt", type=int, default=64)
    p.add_argument("--K_student", type=int, default=20)
    p.add_argument("--K_reference", type=int, default=20)
    p.add_argument("--lambda_ce", type=float, default=0.3)
    p.add_argument("--lambda_rank", type=float, default=0.5)
    p.add_argument("--lambda_delta", type=float, default=0.2)
    p.add_argument("--min_margin", type=float, default=0.1)
    p.add_argument("--lambda_censor", type=float, default=0.2)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--output_dir", default="results/tdart_gate_v1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--perturb_std", type=float, default=0.0)
    p.add_argument("--finetune_steps", type=int, default=500)
    p.add_argument("--finetune_lr", type=float, default=5e-5)
    p.add_argument("--allow_synthetic", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--variants", nargs="*",
                   default=["ce_only", "strict_topk_kd", "bild_topk_delta",
                            "tdart_no_adaptive", "tdart_full", "full_logit_upper"])
    return p.parse_args()

class WikiTextSplit(Dataset):
    def __init__(self, tokenizer, split, seq_len, max_samples=50000, allow_synthetic=False):
        self.samples = []
        self.source = "none"
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
            self.source = f"wikitext-103/{split}"
            buf = []
            for ex in ds:
                t = ex.get("text","").strip()
                if len(t) < 20: continue
                ids = tokenizer(t, add_special_tokens=False)["input_ids"]
                buf.extend(ids)
                while len(buf) >= seq_len:
                    self.samples.append(torch.tensor(buf[:seq_len], dtype=torch.long))
                    buf = buf[seq_len:]
                    if len(self.samples) >= max_samples: break
                if len(self.samples) >= max_samples: break
        except Exception as e:
            if not allow_synthetic:
                raise RuntimeError(f"WikiText load failed: {e}")
            self.source = f"synthetic/{split}"
            rng = torch.Generator().manual_seed(42)
            V = tokenizer.vocab_size if hasattr(tokenizer,"vocab_size") else 32000
            for _ in range(min(max_samples, 1000)):
                self.samples.append(torch.randint(3, V, (seq_len,), generator=rng))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

@torch.no_grad()
def evaluate(teacher, student, eval_loader, device, max_batches=50):
    kls, top1s, losses = [], [], []
    for i, batch in enumerate(eval_loader):
        if i >= max_batches: break
        ids = batch.to(device)
        t_logits = teacher(input_ids=ids).logits[:, :-1, :].float()
        s_logits = student(input_ids=ids).logits[:, :-1, :].float()
        labels = ids[:, 1:]
        t_lp = F.log_softmax(t_logits, dim=-1)
        s_lp = F.log_softmax(s_logits, dim=-1)
        kl = (t_lp.exp() * (t_lp - s_lp)).sum(-1).mean().item()
        kls.append(kl)
        top1s.append((t_logits.argmax(-1) == s_logits.argmax(-1)).float().mean().item())
        ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), labels.reshape(-1))
        losses.append(ce.item())
    ppl = math.exp(sum(losses)/len(losses)) if losses else float("inf")
    return {"kl": sum(kls)/len(kls), "top1": sum(top1s)/len(top1s),
            "ppl": ppl, "ce": sum(losses)/len(losses)}

def train_variant(name, teacher_api, teacher_model, reference_model,
                  student, train_loader, eval_loader, args, device):
    logger.info("\n" + "="*60)
    logger.info("Training variant: %s", name)
    logger.info("="*60)

    is_ce_only = (name == "ce_only")
    is_full_logit = (name == "full_logit_upper")
    is_topk_kd = (name == "strict_topk_kd")
    is_bild = (name == "bild_topk_delta")
    is_tdart = name.startswith("tdart")
    is_cdart = name.startswith("cdart")
    use_adaptive = (name == "tdart_full")
    use_residual = is_tdart or is_cdart or is_bild
    use_censored = (name == "cdart_full")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    reference_model.eval()
    student.train()
    step = 0
    history = []
    mech = {"variant": name, "rank_loss": [], "delta_mse": [], "ce_loss": [],
            "censor_loss": [], "censored_pairs": [],
            "candidate_size": [], "probe_queries": 0,
            "censored_constraints_active": use_censored}
    train_iter = iter(train_loader)
    pbar = tqdm(total=args.num_steps, desc=name)

    while step < args.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch.to(device)
        B, T_seq = ids.shape

        s_out = student(input_ids=ids)
        s_logits = s_out.logits.float()
        loss_ce = sequence_ce_loss(s_logits, ids)

        if is_ce_only:
            loss = loss_ce
            loss_rank_val = 0.0
            loss_delta_val = 0.0
        elif is_full_logit:
            t_logits = teacher_api.get_full_logits_ORACLE_ONLY(ids)
            loss_kd = sequence_kl_loss(s_logits, t_logits.float(),
                                       temperature=args.kd_temp)
            loss = 0.7 * loss_kd + 0.3 * loss_ce
            loss_rank_val = 0.0
            loss_delta_val = loss_kd.item()
        elif is_topk_kd:
            topk_vals, topk_idx, logit_shape = teacher_api.get_topk(ids)
            V = logit_shape[-1]
            t_logits = torch.full((B, T_seq, V), -1e9, device=device)
            t_logits.scatter_(-1, topk_idx, topk_vals)
            loss_kd = sequence_kl_loss(s_logits, t_logits.float(),
                                       temperature=args.kd_temp)
            loss = 0.7 * loss_kd + 0.3 * loss_ce
            loss_rank_val = 0.0
            loss_delta_val = loss_kd.item()
        else:
            with torch.no_grad():
                ref_logits = reference_model(input_ids=ids).logits.float()

            topk_vals, topk_idx, logit_shape = teacher_api.get_topk(ids)
            V = logit_shape[-1]

            total_rank_loss = torch.tensor(0.0, device=device)
            total_delta_loss = torch.tensor(0.0, device=device)
            total_censor_loss = torch.tensor(0.0, device=device)
            total_censor_pairs = 0
            n_pos = 0

            for pos in range(-8, 0):
                t_topk_ids_pos = topk_idx[:, pos, :]
                s_logits_pos = s_logits[:, pos, :]
                ref_logits_pos = ref_logits[:, pos, :]
                t_topk_vals_pos = topk_vals[:, pos, :]

                for b in range(B):
                    if use_adaptive:
                        cand_ids = select_candidates_batch(
                            t_topk_ids_pos[b],
                            s_logits_pos[b].detach(),
                            ref_logits_pos[b],
                            K_student=args.K_student,
                            K_reference=args.K_reference,
                            max_probe_tokens=args.max_probe_tokens_per_prompt,
                            strategy="disagreement")
                        probe_needed = []
                        for cid in cand_ids:
                            if cid not in t_topk_ids_pos[b]:
                                probe_needed.append(cid.item())
                        if probe_needed:
                            probe_t = torch.tensor(probe_needed, device=device)
                            probed = teacher_api.probe_oracle.query_probe(
                                ids[b:b+1], probe_t)
                            mech["probe_queries"] += len(probe_needed)
                            t_at_cand = torch.zeros(len(cand_ids), device=device)
                            for j, cid in enumerate(cand_ids):
                                tk_match = (t_topk_ids_pos[b] == cid).nonzero(as_tuple=True)
                                if len(tk_match[0]) > 0:
                                    t_at_cand[j] = t_topk_vals_pos[b, tk_match[0][0]]
                                else:
                                    pb_match = (probe_t == cid).nonzero(as_tuple=True)
                                    if len(pb_match[0]) > 0:
                                        t_at_cand[j] = probed[0, pos, pb_match[0][0]]
                        else:
                            t_at_cand = t_topk_vals_pos[b]
                            cand_ids = t_topk_ids_pos[b]
                    elif is_bild:
                        cand_ids = t_topk_ids_pos[b]
                        t_at_cand = t_topk_vals_pos[b]
                    else:
                        cand_ids = t_topk_ids_pos[b]
                        t_at_cand = t_topk_vals_pos[b]

                    delta_t = compute_residual(t_at_cand.unsqueeze(0),
                                               ref_logits_pos[b:b+1], cand_ids)
                    delta_s = compute_student_residual(s_logits_pos[b:b+1],
                                                       ref_logits_pos[b:b+1], cand_ids)

                    if is_bild:
                        total_delta_loss = total_delta_loss + residual_mse_loss(delta_s, delta_t)
                    else:
                        signs, margins, pair_mask = build_pairwise_preferences(
                            delta_t, min_margin=args.min_margin)
                        total_rank_loss = total_rank_loss + pairwise_residual_rank_loss(
                            delta_s, delta_t, mask=pair_mask)
                        total_delta_loss = total_delta_loss + residual_mse_loss(delta_s, delta_t)

                    if use_censored:
                        s_topk_b = s_logits_pos[b].detach().topk(args.K_student).indices
                        r_topk_b = ref_logits_pos[b].topk(args.K_reference).indices
                        obs_ids, cen_ids, tau_T, dt_ub = build_censored_candidates(
                            t_topk_ids_pos[b], t_topk_vals_pos[b],
                            s_topk_b, r_topk_b, ref_logits_pos[b])
                        if len(cen_ids) > 0:
                            delta_s_obs = compute_student_residual(
                                s_logits_pos[b:b+1], ref_logits_pos[b:b+1], obs_ids).squeeze(0)
                            delta_t_obs = compute_residual(
                                t_topk_vals_pos[b].unsqueeze(0),
                                ref_logits_pos[b:b+1], obs_ids).squeeze(0)
                            delta_s_cen = compute_student_residual(
                                s_logits_pos[b:b+1], ref_logits_pos[b:b+1], cen_ids).squeeze(0)
                            total_censor_loss = total_censor_loss + censored_residual_rank_loss(
                                delta_s_obs, delta_t_obs, delta_s_cen, dt_ub,
                                margin=args.min_margin)
                            total_censor_pairs += len(cen_ids)
                    n_pos += 1

            if n_pos > 0:
                total_rank_loss = total_rank_loss / n_pos
                total_delta_loss = total_delta_loss / n_pos
                if use_censored:
                    total_censor_loss = total_censor_loss / n_pos

            loss = (args.lambda_ce * loss_ce +
                    args.lambda_rank * total_rank_loss +
                    args.lambda_delta * total_delta_loss)
            if use_censored:
                loss = loss + args.lambda_censor * total_censor_loss
            loss_rank_val = total_rank_loss.item()
            loss_delta_val = total_delta_loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        pbar.update(1)
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                            rank=f"{loss_rank_val:.4f}",
                            ce=f"{loss_ce.item():.4f}")
            mech["rank_loss"].append(loss_rank_val)
            if use_censored:
                mech["censor_loss"].append(total_censor_loss.item() if isinstance(total_censor_loss, torch.Tensor) else 0.0)
                mech["censored_pairs"].append(total_censor_pairs)
            mech["delta_mse"].append(loss_delta_val)
            mech["ce_loss"].append(loss_ce.item())
            if is_tdart:
                mech["candidate_size"].append(n_pos)

        if step % args.eval_every == 0 or step == args.num_steps:
            student.eval()
            metrics = evaluate(teacher_model, student, eval_loader, device, args.eval_batches)
            logger.info("[%s] step %d: KL=%.4f top1=%.4f ppl=%.2f",
                        name, step, metrics["kl"], metrics["top1"], metrics["ppl"])
            metrics["step"] = step
            history.append(metrics)
            student.train()

    pbar.close()
    student.eval()
    final = evaluate(teacher_model, student, eval_loader, device, args.eval_batches)
    logger.info("[%s] FINAL: KL=%.4f top1=%.4f ppl=%.2f",
                name, final["kl"], final["top1"], final["ppl"])

    return {"name": name, "history": history, "final": final,
            "mechanism_log": mech,
            "query_budget": teacher_api.budget.summary()}

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
        for attr in ("lr", "kd_temp", "lambda_ce", "lambda_rank", "lambda_delta",
                      "lambda_censor", "min_margin", "perturb_std", "finetune_lr"):
            if hasattr(args, attr):
                setattr(args, attr, float(getattr(args, attr)))

    if args.dry_run:
        logger.info("DRY RUN — T-DART")
        logger.info("Variants: %s", args.variants)
        logger.info("Teacher: %s", args.teacher_checkpoint_path)
        logger.info("Reference: %s", args.base_model)
        logger.info("Config: %s", {k:v for k,v in vars(args).items() if k != "config"})
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    seeds = args.seeds if args.seeds else [args.seed]
    base_output = Path(args.output_dir)

    for seed in seeds:
        torch.manual_seed(seed)
        seed_dir = base_output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading reference: %s", args.base_model)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        reference = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16
        ).to(args.device).eval()
        for p in reference.parameters():
            p.requires_grad = False

        teacher_path = Path(args.teacher_checkpoint_path)
        if teacher_path.exists() and (teacher_path / "config.json").exists():
            logger.info("Loading delta teacher from: %s", teacher_path)
            teacher = AutoModelForCausalLM.from_pretrained(
                str(teacher_path), torch_dtype=torch.bfloat16
            ).to(args.device).eval()
        else:
            logger.info("No teacher checkpoint — creating delta teacher on the fly...")
            from scripts.create_delta_teacher import PrivateDomainDataset
            teacher = AutoModelForCausalLM.from_pretrained(
                args.base_model, torch_dtype=torch.bfloat16
            ).to(args.device)
            ft_ds = PrivateDomainDataset(tokenizer, args.seq_len)
            ft_loader = DataLoader(ft_ds, batch_size=4, shuffle=True, drop_last=True)
            ft_opt = torch.optim.AdamW(teacher.parameters(), lr=args.finetune_lr)
            teacher.train()
            ft_iter = iter(ft_loader)
            for st in range(args.finetune_steps):
                try:
                    batch = next(ft_iter)
                except StopIteration:
                    ft_iter = iter(ft_loader)
                    batch = next(ft_iter)
                out = teacher(input_ids=batch.to(args.device), labels=batch.to(args.device))
                ft_opt.zero_grad()
                out.loss.backward()
                ft_opt.step()
                if (st+1) % 100 == 0:
                    logger.info("  Teacher FT step %d/%d loss=%.4f", st+1, args.finetune_steps, out.loss.item())
            teacher.eval()
            teacher_path.mkdir(parents=True, exist_ok=True)
            teacher.save_pretrained(str(teacher_path))
            tokenizer.save_pretrained(str(teacher_path))
            logger.info("Delta teacher saved: %s", teacher_path)

        for p in teacher.parameters():
            p.requires_grad = False

        api = StrictBlackBoxAPI(teacher, K=args.topk, device=args.device)

        logger.info("Building datasets...")
        train_ds = WikiTextSplit(tokenizer, "train", args.seq_len,
                                  allow_synthetic=args.allow_synthetic)
        eval_ds = WikiTextSplit(tokenizer, "validation", args.seq_len,
                                 max_samples=2000, allow_synthetic=args.allow_synthetic)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
        logger.info("Train: %d, Eval: %d", len(train_ds), len(eval_ds))

        teacher_metrics = evaluate(teacher, reference, eval_loader, args.device, args.eval_batches)
        logger.info("Teacher-vs-Reference gap: KL=%.4f top1=%.4f",
                     teacher_metrics["kl"], teacher_metrics["top1"])

        all_results = {"teacher_ref_gap": teacher_metrics, "seed": seed,
                       "args": vars(args), "variants": {}}

        for variant_name in args.variants:
            torch.manual_seed(seed)
            student = AutoModelForCausalLM.from_pretrained(
                args.base_model, torch_dtype=torch.bfloat16
            ).to(args.device)
            if args.perturb_std > 0:
                with torch.no_grad():
                    for p in student.parameters():
                        p.add_(torch.randn_like(p) * args.perturb_std)

            api.budget.topk_queries = 0
            api.budget.probe_queries = 0

            res = train_variant(variant_name, api, teacher, reference,
                                student, train_loader, eval_loader, args, args.device)
            all_results["variants"][variant_name] = res
            del student
            torch.cuda.empty_cache()

        logger.info("\n" + "="*60)
        logger.info("T-DART RESULTS (seed=%d)", seed)
        logger.info("="*60)
        t_ppl = teacher_metrics.get("ppl", 0)
        logger.info("%-25s %8s %8s %8s", "Variant", "KL", "Top-1", "PPL")
        logger.info("-"*55)
        for vn in args.variants:
            f = all_results["variants"][vn]["final"]
            logger.info("%-25s %8.4f %8.4f %8.2f", vn, f["kl"], f["top1"], f["ppl"])

        with open(seed_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        save_manifest(seed_dir, command=" ".join(sys.argv),
                      config=vars(args), seed=seed,
                      model_name=args.base_model, dataset=train_ds.source,
                      metrics={vn: all_results["variants"][vn]["final"] for vn in args.variants})
        logger.info("Saved: %s", seed_dir)

if __name__ == "__main__":
    main()
