# Honest Final Status — Complete Experiment Chain

## Final Table (v4, 3 seeds, strict black-box, Qwen2.5-0.5B, topK=20)

| Variant | PPL (m±s) | KL (m±s) | Beats ce_only PPL? | Beats ce_only KL? |
|---------|-----------|----------|--------------------|--------------------|
| Teacher | 24.23 | 0.000 | — | — |
| **ce_only** | **125.08 ± 17.99** | **2.373 ± 0.130** | **baseline** | **baseline** |
| strict_topk_kd | 411.12 ± 7.11 | 3.294 ± 0.020 | 0/3 seeds | 0/3 seeds |
| probe_dense_kd | 425.31 ± 86.00 | 3.040 ± 0.196 | 0/3 seeds | 0/3 seeds |
| completion_uncertainty_strict | 123.90 ± 16.11 | 2.366 ± 0.122 | 2/3 seeds | 2/3 seeds |
| full_logit_upper | 115.03 ± 16.88 | 1.961 ± 0.140 | 3/3 seeds | 3/3 seeds |

## Key Findings — 100% Honest

### 1. CE-only is the strongest non-oracle variant
No KD variant consistently beats pure CE fine-tuning on WikiText:
- `strict_topk_kd` (PPL 411) — much worse than ce_only (125)
- `probe_dense_kd` (PPL 425 ± 86) — worse AND unstable (seed 1: PPL 546!)
- `completion_uncertainty_strict` (PPL 124) — matches ce_only (≈125), not better
- Only `full_logit_upper` (PPL 115) beats ce_only — but it uses oracle full logits

### 2. probe_dense_kd DOES NOT work
- Mean PPL 425 ± 86 — WORSE than strict_topk_kd (411 ± 7)
- Seed 1: PPL 546 (catastrophic instability)
- Fixed 2000 probe tokens with subset-softmax KD is unreliable
- Root cause: softmax over 2000 fixed tokens creates an artificial distribution that misleads KD

### 3. completion_uncertainty_strict ≈ ce_only (confirmed across 3 seeds)
- completion_uncertainty: PPL 123.90 ± 16.11, KL 2.366
- ce_only: PPL 125.08 ± 17.99, KL 2.373
- Gap is negligible (1.2 PPL, within 1 std)
- Confirmed: uncertainty gate turns off tail KD → C becomes ce_only

### 4. KD is HARMFUL in this setup
Every KD variant performs WORSE than ce_only:
- strict_topk_kd: +286 PPL penalty over ce_only
- probe_dense_kd: +300 PPL penalty
- completion_uncertainty: ~0 penalty (because KD is effectively disabled)
- Only full-logit KD helps (because it has the full correct distribution)

### 5. Why KD hurts
In pretrained-perturbed setup (σ=0.01), the student already has 99% of teacher knowledge. The remaining 1% difference is recoverable via CE on WikiText alone (PPL 125 → close to teacher 24 with more steps). Sparse KD signals (20 or 2000 tokens out of 151k) inject NOISE that pushes the student away from the correct distribution.

## Root Cause Chain

```
pretrained_perturbed (σ=0.01) student already very close to teacher
                    ↓
CE-only fine-tuning on WikiText = best non-oracle recovery (PPL ~125)
                    ↓
top-K=20 KD signal is too sparse → noise > signal → PPL 411
                    ↓
probe_dense=2000 KD: subset-softmax creates artificial distribution → unstable, PPL 425
                    ↓
Carlini-basis completion: gauge ambiguity → tail logits pure noise → heldout_mse=12464
                    ↓
uncertainty gate correctly kills noise → C ≈ ce_only → null mechanism
                    ↓
only full-logit KD (oracle) has enough signal → PPL 115
```

## What This Means for the Paper

### Cannot claim
- "Q-UMC recovers useful dense logits from strict API" — FALSE
- "Uncertainty gating improves model cloning" — FALSE (it just turns off bad KD)
- "Probe expansion enhances KD" — FALSE (probe_dense worse than ce_only)
- "Logit Completion beats baseline" — only in leaked R1 setting (contaminated)

### Can honestly claim (negative results)
- "In strict black-box with Carlini-recovered basis, dense logit completion provides NO improvement over CE-only fine-tuning under matched compute"
- "Sparse KD (top-K or probe-subset) is actively harmful when student is pretrained-perturbed"
- "Uncertainty gating prevents completion from failing catastrophically, but does not provide positive signal"
- "The gap between CE-only (PPL 125) and full-logit KD (PPL 115) represents the true information inaccessible from top-K/probe API"

### Still viable paper angles
1. **Theory paper on observability/identifiability** (γ-loop paper at 7.5/10 READY):
   The Moment-CP + Gramian theory paper is independently valid if Moment-CP artifacts can be restored
2. **Negative result paper on strict black-box KD limitations**:
   "Why Dense Logit Completion Fails Under Strict Black-Box Access" — publishable at workshops
3. **Different experimental setup**: test with larger gap (teacher = fine-tuned, student = base model) where CE-only can't recover and KD truly needed

## Experiment Chain Summary

| Experiment | Key Finding | Status |
|-----------|-------------|--------|
| v7 (random init, 5k) | Random KD fails completely | Verified negative |
| v8 (pretrained, full logits) | Dense full-logit KD works; hidden MSE doesn't help | Verified positive (oracle) |
| v9 (random init, 50k) | Even 50k steps from random insufficient | Verified negative |
| v10 (σ=0.1) | Large perturbation destroys model | Verified negative |
| v11/v12 (top-K, old code) | Old Logit Completion seems to work | Contaminated (teacher W_lm leak) |
| v13/v14 (old Logit Completion) | E beats A by 23-28% | Contaminated (same leak) |
| qumc_minimal (R1) | C>B>A | Invalid (teacher W_lm + cal leak + loss scale) |
| qumc_minimal_strict_v2 (R2) | C>A on PPL+KL | Weak signal (C≈ce_only not tested) |
| qumc_minimal_strict_v3 (R3) | ce_only ≈ C; normed_kl crashes | C is null mechanism |
| **qumc_probe_dense_v4** | **probe_dense worse than ce_only** | **Root cause fix test: NEGATIVE** |

## Decision

**STOP current method path. Return to GPT-5.5 Pro with honest results.**

The Q-UMC mechanism — in all forms tested (completion with uncertainty, completion normalized, probe-dense KD) — does not beat CE-only fine-tuning under strict black-box access in the pretrained-perturbed experimental setup.

Next steps should be decided by GPT-5.5 Pro based on this evidence:
1. Pivot to theory paper (γ-loop, already at 7.5/10)
2. Change experimental setup (larger teacher-student gap)
3. Explore fundamentally different mechanism
