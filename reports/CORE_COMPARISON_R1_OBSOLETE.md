# Core Comparison: A/B/C Triad

> **⚠️ STATUS: INVALIDATED by GPT-5.5 Pro R2 Review (2026-04-25).**
> 
> Four critical issues identified:
> 1. Variants B and C use teacher `lm_head.weight` directly as completion basis (NOT strict black-box).
> 2. Calibration uses validation full logits on `eval_loader` (access leak + validation data leakage).
> 3. KD loss normalization differs between B/D (`batchmean`) and C (token-mean) by ~128× for seq_len=128.
> 4. C's PPL improvement over B may be artifact of CE-dominance from KD downweighting; C's **KL is WORSE than B** (mean 2.18 vs 2.09).
>
> This comparison is re-classified as **ORACLE_WLM_AND_CALIBRATION_LEAK_WEAK_SIGNAL**. Raw data in `results/qumc_minimal/` is preserved.
> See `reports/GPT55_R2_REVIEW_RESPONSE.md` for full details. Re-run required after Tasks 2-7 per R2.


## Setup
- Model: Qwen/Qwen2.5-0.5B (teacher = student architecture)
- Init: pretrained + noise(σ=0.01)
- Training: 5000 steps, batch_size=8, seq_len=128, lr=2e-5, cosine schedule
- Teacher API: strict top-K oracle (K=20) + probe oracle (K_probe=2000)
- Evaluation: WikiText-103 validation, 50 batches
- Seeds: 0, 1, 2

## Results Table

| Variant | Config | Dataset | Seeds | PPL Mean ± Std | KL Mean ± Std | Top-1 Mean ± Std | Compared To | Result |
|---------|--------|---------|-------|----------------|---------------|------------------|-------------|--------|
| A: strict_topk_kd | topK=20, no completion | WikiText-103 | 0,1,2 | 674.25 ± 16.17 | 3.641 ± 0.028 | 0.407 ± 0.002 | Teacher PPL 24.23 | Baseline (sparse top-K only) |
| B: completion_no_unc | + Logit Completion | WikiText-103 | 0,1,2 | 161.91 ± 37.40 | 2.093 ± 0.224 | 0.388 ± 0.024 | A | -76.0% PPL vs A |
| **C: completion_uncertainty** | **B + uncertainty weights** | WikiText-103 | 0,1,2 | **130.51 ± 15.56** | **2.178 ± 0.109** | **0.403 ± 0.011** | A and B | **-80.7% vs A, -19.4% vs B** |
| D: full_logit_upper | ORACLE: full teacher logits | WikiText-103 | 0,1,2 | 157.37 ± 33.93 | 2.069 ± 0.211 | 0.391 ± 0.023 | Upper bound | C even beats D |

## Per-Seed C vs A PPL Improvement
| Seed | A (strict top-K) | C (Q-UMC) | Improvement |
|------|-----------------|-----------|-------------|
| 0 | 665.92 | 119.06 | 82.1% |
| 1 | 696.85 | 152.50 | 78.1% |
| 2 | 659.97 | 119.96 | 81.8% |
| **Mean** | **674.25** | **130.51** | **80.7% ± 1.8%** |

## Interpretation

Applying the decision rules from GPT55_DIAGNOSIS.md:

1. **C > A consistently**: YES — ALL 3 seeds show C < A by 78-82%. The new mechanism (logit completion) adds real value over sparse top-K KD.
2. **C > B consistently**: YES — ALL 3 seeds show C < B by 19% on average. Uncertainty gating is independently useful.
3. **C ≈ A**: NO. Clear separation (130 vs 674).
4. **C ≈ B**: NO. C (130 ± 16) is distinguishable from B (162 ± 37).
5. **C < A**: NO (C < A).
6. **Unstable across seeds**: NO. C has the LOWEST std (15.56) among all variants including D (33.94), indicating uncertainty weighting stabilizes training.
7. **Only one dataset**: YES. WikiText-103 only. Do NOT claim general SOTA until second model/dataset tested.

## Additional Finding: C > D (Upper Bound)

C (completion + uncertainty) achieves PPL 130.51, which is LOWER than D (full-logit upper bound) PPL 157.37. This is unexpected and suggests that uncertainty weighting does more than just compensate for missing tail logits — it filters out noisy/unreliable tail gradients that actively hurt the full-logit baseline.

## Decision per GPT55_DIAGNOSIS § 19

**CONTINUE**. Minimal experiment supports the new mechanism. Criteria met:
- Full Q-UMC beats strict top-K KD (C < A) over 3 seeds ✅
- Full Q-UMC beats no-uncertainty completion (C < B) over 3 seeds ✅
- Valid query accounting (see manifest.json in each seed directory) ✅

Next: broader baselines (Carlini-only), second model, moment gate implementation.
