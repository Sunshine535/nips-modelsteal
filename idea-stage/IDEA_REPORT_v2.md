# Research Pipeline Stage 1: Result Audit & Direction Decision

**Date**: 2026-04-27
**Request**: "从当前结果中找到可以投稿 NeurIPS main 的正面结果，如果没有，则找到正确路径"
**Difficulty**: nightmare | **Effort**: beast | **Reviewer**: oracle-pro

---

## 1. Executive Summary

**Verdict: C-DART 是唯一通过所有 gate 的正面结果，但当前实验规模不足以支撑 NeurIPS main 投稿。需要 2-3 周实验扩展 + 1 周写作。**

C-DART (Censored Delta Adaptive Ranking Teacher) 在 3 seed、7 variant 的严格消融中：
- 比 ce_only 改进 **65.6% KL** (0.608 → 0.209)
- 比 BiLD baseline 改进 **5.9% KL** (0.222 → 0.209)
- Censoring 机制贡献 **+4.2%** 增量收益 (0.218 → 0.209)
- 闭合 ce_only→oracle gap 的 **66.3%** (0.608→0.209 / 0.608→0.006)
- 3 seed 标准差 < 0.003 KL

但仅在 1 个模型 (Qwen2.5-0.5B)、1 种 gap (500 步微调)、1 种 top-K 设置 (K=20) 上验证。NeurIPS main 至少需要 3 模型、3 gap、5 baseline、下游任务评估。

---

## 2. All Results Audit — Honest Status

### 2.1 Positive Results (Clean, Gate-Passing)

| Result | Metric | Status | NeurIPS Main? |
|--------|--------|--------|---------------|
| **C-DART full** | KL=0.209±0.003, 65.6% vs ce_only, 3/3 seeds | **CLEAN, ALL GATES PASS** | Needs scaling |
| **Gramian full-rank** | rank=128, κ=2.4-3.1, well-conditioned | CLEAN | Already submitted (theory paper) |
| **Warm-start barrier** | |Δcos| < 0.01 at all α | CLEAN | Already submitted (theory paper) |
| **Moments CP lm_head** | cos=0.81 vs null 0.16 | CLEAN | Partial (1 layer only) |

### 2.2 Contaminated / Invalid Results

| Result | Issue | Status |
|--------|-------|--------|
| v13 Logit Completion (top-K=20) | teacher W_lm leak → PPL 137 fabricated | **CONTAMINATED** |
| v14 Logit Completion (top-K=5) | Same teacher W_lm leak | **CONTAMINATED** |
| Q-UMC R1 (qumc_minimal) | teacher W_lm + calibration leak + loss scale bug | **INVALID** |

### 2.3 Negative Results (Confirmed, Honest)

| Result | Finding | Root Cause |
|--------|---------|------------|
| Q-UMC strict v2 | completion_uncertainty ≈ ce_only (PPL 124 vs 125) | Uncertainty gate kills noise → null mechanism |
| probe_dense_kd v4 | PPL 425 ± 86, worse than ce_only | Subset-softmax creates artificial distribution |
| SCRD all variants (v7-v12) | Oracle h_final MSE provides NO benefit over logit-only | Representation/behavior gradient conflict |
| strict_topk_kd | KL=1.489, catastrophically worse | Sparse top-K KL on absolute logits is harmful |
| T-DART full (adaptive) | KL=0.281, WORSE than bild 0.222 | Adaptive probing injects noise |
| All gradient-based parameter recovery | cos ≈ 0.12-0.14 | Non-convex optimization barrier |
| K-expansion (32→128) | Zero effect on recovery | Bottleneck is optimization, not information |

### 2.4 Already Submitted

| Result | Venue | Status |
|--------|-------|--------|
| Theory paper (Transformer Tomography) | NeurIPS 2026 | Submitted 2026-04-19, 8.5/10 READY |

---

## 3. C-DART Deep Analysis

### 3.1 Method Summary

C-DART exploits three insights unavailable to standard KD:

1. **Delta Teacher**: Build residual Δ = teacher_logits - reference_logits. The reference model (public base model) anchors the student, and the delta isolates teacher-specific behavior.

2. **Censored Constraints**: When a token appears in reference's top-K but NOT in teacher's top-K, the teacher has *actively suppressed* it. This ABSENCE is a training signal (push student probability down for censored tokens).

3. **Ranking Loss**: Instead of matching absolute logit values (which are noisy under top-K truncation), learn pairwise ordering from teacher's top-K rankings.

### 3.2 Why It Works (Mechanism)

```
Teacher (fine-tuned on private data) produces top-K logits
                    ↓
Reference (public base model) produces top-K logits for same input
                    ↓
Delta = teacher_topK - reference_topK → isolates private behavior
                    ↓
Censored tokens = reference_topK \ teacher_topK → suppression signal
                    ↓
Student learns: (1) match delta residual, (2) respect censored suppressions,
                (3) preserve ranking order, (4) CE on ground truth tokens
```

### 3.3 3-Seed Results Table

| Variant | KL↓ (m±s) | Top-1↑ (m±s) | PPL (m±s) | vs ce_only KL |
|---------|-----------|-------------|-----------|---------------|
| ce_only | 0.608 ± 0.002 | 0.678 ± 0.000 | 15.73 ± 0.02 | baseline |
| strict_topk_kd | 1.489 ± 0.001 | 0.828 ± 0.000 | 118.54 ± 0.31 | -145% worse |
| bild_topk_delta | 0.222 ± 0.001 | 0.768 ± 0.000 | 17.76 ± 0.01 | +63.4% |
| tdart_no_adaptive | 0.218 ± 0.003 | 0.773 ± 0.002 | 18.15 ± 0.03 | +64.2% |
| cdart_no_censor | 0.218 ± 0.003 | 0.773 ± 0.002 | 18.15 ± 0.03 | +64.2% |
| **cdart_full** | **0.209 ± 0.003** | **0.776 ± 0.002** | **18.25 ± 0.05** | **+65.6%** |
| full_logit_upper | 0.006 ± 0.000 | 0.957 ± 0.001 | 25.05 ± 0.01 | oracle |

### 3.4 Gap Closure Analysis

```
ce_only (no teacher signal):     KL = 0.608
cdart_full (top-K=20 only):      KL = 0.209
full_logit_upper (oracle):       KL = 0.006

Gap closed = (0.608 - 0.209) / (0.608 - 0.006) = 66.3%

C-DART recovers 2/3 of the oracle-level performance
using ONLY top-20 logit access.
```

### 3.5 Ablation Decomposition

| Component | KL | Δ vs ce_only | Incremental |
|-----------|-----|-------------|-------------|
| + delta teacher (bild) | 0.222 | +63.4% | +63.4% (base mechanism) |
| + ranking loss (tdart) | 0.218 | +64.2% | +0.8% |
| + censoring (cdart_full) | 0.209 | +65.6% | +1.4% |
| Total incremental over bild | — | — | **+5.9%** |

### 3.6 Critical Honest Observations

**PPL is NOT the right metric here**:
- ce_only PPL=15.73 is LOWER than teacher PPL=24.23
- This is because ce_only fine-tunes on WikiText training data — it becomes a better LM but diverges from teacher
- KL to teacher is the correct metric for model cloning fidelity
- full_logit_upper PPL=25.05 ≈ teacher 24.23 confirms: perfect cloning matches teacher, not beats it

**The setup is narrow**:
- Teacher: 500-step fine-tune on WikiText TEST (minimal private behavior)
- Teacher-Reference gap: KL=0.495 (small gap)
- Only Qwen2.5-0.5B tested
- Only K=20 tested

---

## 4. NeurIPS Main Readiness Assessment

### 4.1 Scorecard (Current State)

| Criterion | Required | Current | Gap |
|-----------|----------|---------|-----|
| Positive result | yes | C-DART +65.6% KL | Exists |
| Multi-model | 3+ models | 1 (Qwen2.5-0.5B) | **CRITICAL** |
| Multi-scale gap | 3+ settings | 1 (500-step FT) | **CRITICAL** |
| Baselines | 5+ methods | 3 (ce_only, topk_kd, bild) | **CRITICAL** |
| Downstream eval | Task accuracy | None | **MAJOR** |
| Top-K sensitivity | K sweep | K=20 only | MODERATE |
| Query budget | Budget curve | Fixed 5K steps | MODERATE |
| Defense eval | 2+ defenses | None | MODERATE |
| Theoretical grounding | Connects to theory | Implicit | MINOR |
| Paper draft | Complete | None (method paper) | Expected |

**Overall: 3/10 — NOT READY for submission, but viable with 3-4 weeks of work.**

### 4.2 What Prevents Immediate Submission

1. **Single model**: NeurIPS reviewer will ask "does this generalize beyond 0.5B Qwen?" — auto-reject risk
2. **No real baselines**: BiLD is the only competitor. Need DKD, DIST, at minimum
3. **No downstream tasks**: KL is informative but not sufficient. Need MMLU/HellaSwag/etc.
4. **Tiny teacher gap**: 500-step FT is barely different from base. Need to test with real fine-tuned models

---

## 5. The Correct Path: C-DART → NeurIPS Main

### 5.1 Recommended Paper Framing

**Title**: "Censored Delta Distillation: Extracting Teacher-Specific Behavior from Black-Box Top-K Logit APIs"

**Contribution Type**: Methods (Empirical + Algorithmic)

**Core Claim**: Under strict black-box access (only top-K logits visible), standard KD fails catastrophically. C-DART introduces three techniques — reference-anchored delta learning, censored absence constraints, and ranking loss — that together recover 66% of the full-logit oracle's performance using only top-20 logits.

**Why NeurIPS-worthy**:
1. Timely: API providers are restricting access (top-K only, logprob limits)
2. Surprising negative result: Standard KD is HARMFUL under top-K constraints
3. Novel mechanism: Censored constraints (exploiting absence of information)
4. Strong ablation structure: Each component contributes measurably
5. Connects to theoretical understanding (observability framework from companion paper)

### 5.2 Required Experiments

#### Phase 1: Multi-Model Validation (Week 1, 4x A100)

| Model | Size | Architecture | GPU Hours |
|-------|------|-------------|-----------|
| Qwen2.5-0.5B | 0.5B | GQA, SiLU-gated | Done |
| Llama-3.2-1B | 1.3B | GQA, SiLU-gated | ~20h |
| Gemma-2-2B | 2.6B | GQA, GeGLU | ~40h |
| Qwen2.5-3B | 3.1B | GQA, SiLU-gated | ~50h |

For each model: 7 variants x 3 seeds x 5000 steps

#### Phase 2: Gap Scaling (Week 1-2, overlap)

| Gap Level | Teacher Setup | Expected Gap |
|-----------|-------------|-------------|
| Small | 500 steps WikiText FT | KL ~ 0.5 (done) |
| Medium | 5000 steps WikiText FT | KL ~ 2-5 |
| Large | 10000 steps on domain-specific data | KL ~ 5-15 |
| Cross-domain | Teacher FT on code → student on text | KL > 10 |

#### Phase 3: Baselines (Week 2)

| Baseline | Source | Implementation |
|----------|--------|---------------|
| CE-only | Ours | Done |
| Top-K KD (naive) | Standard | Done |
| BiLD delta | Li et al. style | Done |
| DKD | CVPR 2022 | Implement from description |
| DIST | NeurIPS 2022 | Implement from description |
| Tramer-style extraction | Tramer et al. | Adapt to top-K setting |

#### Phase 4: Top-K Sensitivity + Query Budget (Week 2)

- K = {5, 10, 20, 50, 100, full}
- Query budget = {1K, 5K, 10K, 50K}
- Combined sweep: K x budget matrix

#### Phase 5: Downstream Evaluation (Week 2-3)

- MMLU (5-shot)
- HellaSwag (10-shot)
- ARC-Challenge (25-shot)
- GSM8K (5-shot CoT)
- TruthfulQA (6-shot)
- Evaluate: teacher, ce_only, cdart_full, full_logit_upper

#### Phase 6: Defense Evaluation (Week 3)

- SPLITS (ICLR 2025) — logit perturbation
- Temperature scaling defense
- Logit rounding defense
- Evaluate C-DART robustness under each defense

### 5.3 Estimated Compute & Timeline

| Phase | GPU-Hours | Calendar | Dependencies |
|-------|-----------|----------|-------------|
| Multi-model (Phase 1) | ~150h | Week 1 | None |
| Gap scaling (Phase 2) | ~100h | Week 1-2 | Overlap Phase 1 |
| Baselines (Phase 3) | ~80h | Week 2 | Phase 1 results for comparison |
| K + Budget sweep (Phase 4) | ~60h | Week 2 | None |
| Downstream eval (Phase 5) | ~40h | Week 2-3 | Phase 1+3 models ready |
| Defense eval (Phase 6) | ~30h | Week 3 | Phase 1 |
| **Total** | **~460h** | **3 weeks** | 4x A100 |

### 5.4 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| C-DART doesn't generalize to larger models | Medium | Fatal | Test early (Phase 1 first) |
| Gap scaling: C-DART fails at large gaps | Medium | Major | Adjust mechanism; may need iterative approach |
| BiLD is actually equivalent at scale | Low-Medium | Major | Ensure censoring mechanism shows consistent gain |
| Reviewer: "marginal improvement" (5.9% over BiLD) | High | Major | Need larger improvements on harder settings |
| Compute shortage | Low | Major | 4x A100 sufficient; prioritize critical experiments |

### 5.5 Kill Criteria

**STOP and pivot if ANY of these hold after Phase 1**:
1. C-DART fails to beat ce_only on Llama-3.2-1B (< 30% KL improvement)
2. Censoring mechanism shows 0% incremental on 2+ models
3. C-DART ≈ BiLD on all models (< 3% margin)

---

## 6. Alternative Paths (If C-DART Fails at Scale)

### Alt 1: Negative Result + Characterization Paper
- "Why Standard KD Fails Under API Constraints: A Systematic Study"
- Combine: KD-is-harmful finding + observability theory + defense implications
- Target: NeurIPS 2026 workshop / NeurIPS 2027 main track
- Risk: Low impact, high reviewer "so what?"

### Alt 2: Moments CP → Separate Method Paper
- Extend cos=0.81 lm_head extraction to multi-layer
- Requires: stabilize CP decomposition, explain pathology, replicate across models
- Target: ICLR 2027 (October deadline)
- Risk: CP decomposition is hard to stabilize; may not extend

### Alt 3: Pivot to Defense Paper
- "Defending Against Delta-Based Model Extraction Under Top-K APIs"
- Use C-DART as the attack, study defenses that neutralize it
- Requires: C-DART works well enough to be a real threat
- Target: Security venue (CCS, S&P)

---

## 7. Decision Point

**Recommended: Proceed with C-DART (Section 5) as primary path.**

Justification:
1. Only clean positive result in the entire project
2. Clear mechanism with ablation support
3. Direction Lock explicitly targets this
4. Sufficient compute available (4x A100)
5. 3-week timeline is tight but feasible

**Gate 1 Checkpoint: Complete Phase 1 (multi-model validation) first.**
If C-DART generalizes → full speed ahead.
If C-DART fails → pivot to Alt 1 or Alt 3.
