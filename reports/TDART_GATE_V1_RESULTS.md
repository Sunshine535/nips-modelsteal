# T-DART Gate v1 — Final 3-Seed Results

**Date**: 2026-04-26
**Setup**: Teacher = Qwen2.5-0.5B fine-tuned 500 steps on WikiText TEST (private)
**Reference/Student**: Qwen2.5-0.5B base (public)
**Teacher-Reference gap**: KL=0.495, Top-1=67.4%
**Training**: 5000 steps, batch 8, seq_len 128, topK=20

## Headline Table (3 seeds, mean ± std)

| Variant | KL↓ (m±s) | Top-1↑ (m±s) | PPL (m±s) | vs ce_only KL |
|---------|-----------|-------------|-----------|---------------|
| ce_only | 0.608 ± 0.002 | 0.678 ± 0.000 | 15.73 ± 0.02 | baseline |
| strict_topk_kd | 1.489 ± 0.001 | 0.828 ± 0.000 | 118.54 ± 0.31 | -145% worse |
| **bild_topk_delta** | **0.222 ± 0.001** | **0.768 ± 0.000** | 17.76 ± 0.01 | **+63.4%** |
| **tdart_no_adaptive** | **0.216 ± 0.003** | **0.773 ± 0.002** | 18.35 ± 0.05 | **+64.4%** |
| **tdart_full** | **0.281 ± 0.004** | 0.729 ± 0.002 | 20.68 ± 0.12 | **+53.8%** |
| full_logit_upper | 0.006 ± 0.000 | 0.957 ± 0.001 | 25.05 ± 0.01 | oracle |

## Core Gate Results

### Gate 1: Delta methods beat CE-only? ✅ YES
- bild_topk_delta: KL **0.222** vs ce_only **0.608** — **63.4% improvement, 3/3 seeds**
- tdart_no_adaptive: KL **0.216** vs ce_only **0.608** — **64.4% improvement, 3/3 seeds**
- tdart_full: KL **0.281** vs ce_only **0.608** — **53.8% improvement, 3/3 seeds**

**All delta/residual methods beat CE-only on KL across ALL 3 seeds.** This is the first positive result in strict black-box that survives all GPT-5.5 R1-R4 gates.

### Gate 2: Teacher-specific signal is real?  ✅ YES
- Teacher-Reference gap KL=0.495 — meaningful private behavior exists
- CE-only achieves KL=0.608 (WORSE than the gap) — CE on public data cannot fully recover teacher
- Delta methods achieve KL=0.22 (better than gap) — they extract teacher-specific information

### Gate 3: Adaptive probing adds value? ❌ NO (in current implementation)
- tdart_no_adaptive KL=0.216 < tdart_full KL=0.281 on ALL 3 seeds
- Adaptive probing introduces noise from per-sample probe queries
- Current disagreement-based candidate selection needs optimization or the per-sample loop introduces training instability

### Gate 4: Multi-seed stability? ✅ YES
- All variants show extremely low std (0.001-0.004 KL)
- bild and tdart_no_adaptive are the most stable

## Honest Observations

### Positive
1. **First positive strict black-box result** that survives CE-only baseline after 4 rounds of GPT-5.5 review
2. Delta/residual methods extract teacher-specific behavior (KL 0.22) better than the teacher-reference gap (KL 0.50)
3. Results are highly stable across seeds (std < 0.004)
4. This works because: fine-tuned teacher has PRIVATE behavior that CE on public data cannot recover; delta methods use top-K API to observe teacher-specific preferences and learn them

### Concerns
1. **bild ≈ tdart_no_adaptive** (0.222 vs 0.216) — simple BiLD-style MSE on top-K deltas is nearly as good as pairwise ranking. Novelty risk vs BiLD/Delta KD
2. **tdart_full is WORSE** than both simpler variants — adaptive probing hurts rather than helps
3. **PPL direction is wrong**: delta methods have PPL 17-20 vs ce_only PPL 15.7. CE-only gives better language modeling because it trains on ground-truth tokens. Delta methods sacrifice PPL to match teacher's distribution
4. **strict_topk_kd catastrophically fails** (KL 1.49) — shows raw top-K KL on absolute logits is harmful, consistent with Q-UMC findings

### What This Means for the Paper
- **CAN claim**: Teacher-specific residual extraction via top-K API beats CE-only on teacher-student KL by 63-64% across 3 seeds
- **CAN claim**: Without reference-anchored residual objective, top-K KD fails
- **CANNOT claim**: Adaptive probing adds value (currently hurts)
- **CANNOT claim**: T-DART beats BiLD-style baseline (bild ≈ tdart_no_adaptive)
- **MUST address**: novelty over BiLD/Delta KD; need to show strict API constraint + reference model anchoring is the real contribution, not the specific loss function

## Decision per GPT-5.5 R4 Gates

Per the gate experiment rules:
- tdart_full > ce_only: ✅ YES (KL 0.281 < 0.608, 3/3 seeds)
- tdart_full > bild_topk_delta: ❌ NO (0.281 > 0.222)
- tdart_no_adaptive > ce_only: ✅ YES (0.216 < 0.608, 3/3 seeds)
- tdart_no_adaptive > bild: ✅ MARGINAL (0.216 < 0.222, 3/3 seeds)

**Decision: CONTINUE with tdart_no_adaptive as primary method variant.** Adaptive probing needs redesign or should be dropped. The core contribution is the residual-delta framework under strict API, not the specific probe selection strategy.
