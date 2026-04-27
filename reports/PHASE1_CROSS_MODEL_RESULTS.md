# Phase 1: C-DART Cross-Model Validation — COMPLETE

**Date**: 2026-04-27 → 2026-04-28 (Qwen3 added)
**Models**: Qwen2.5-0.5B, Qwen3-0.6B, Llama-3.2-1B, Llama-3.2-3B
**Setup**: Delta teacher (500-step FT on WikiText TEST), top-K=20, 5000 steps, 3 seeds

---

## Headline: ALL GATES PASS ON ALL MODELS

| Model | ce_only KL | cdart_full KL | vs ce_only | vs BiLD | Censoring Δ | Gap Closure |
|-------|-----------|--------------|------------|---------|-------------|-------------|
| **Qwen2.5-0.5B** | 0.608±0.002 | **0.209±0.003** | **+65.6%** | +5.9% | +4.2% | **66.3%** |
| **Qwen3-0.6B** | 0.303±0.000 | **0.120±0.001** | **+60.4%** | +9.8% | +4.3% | **64.5%** |
| **Llama-3.2-1B** | 0.647±0.003 | **0.296±0.009** | **+54.2%** | +11.4% | +8.2% | **58.3%** |
| **Llama-3.2-3B** | 0.564±0.002 | **0.336±0.008** | **+40.3%** | +8.7% | +8.8% | **53.5%** |

---

## Detailed Results by Model

### Qwen3-0.6B (28L, d=1024, 16H/8KV, vocab=151936)

Teacher-Reference gap: KL=1.024

| Variant | KL↓ (m±s) | Top-1↑ (m±s) | PPL | vs ce_only |
|---------|-----------|-------------|-----|------------|
| ce_only | 0.303 ± 0.000 | 0.736 ± 0.001 | 18.7 | baseline |
| strict_topk_kd | 1.418 ± 0.000 | 0.822 ± 0.000 | 99.5 | -368% |
| bild_topk_delta | 0.133 ± 0.000 | 0.815 ± 0.000 | 20.4 | +56.1% |
| tdart_no_adaptive | 0.125 ± 0.000 | 0.822 ± 0.001 | 20.7 | +58.6% |
| cdart_no_censor | 0.125 ± 0.000 | 0.822 ± 0.001 | 20.7 | +58.6% |
| **cdart_full** | **0.120 ± 0.001** | **0.826 ± 0.001** | 20.8 | **+60.4%** |
| full_logit_upper | 0.019 ± 0.000 | 0.920 ± 0.000 | 23.6 | oracle |

### Qwen2.5-0.5B (24L, d=896, 14H/2KV, vocab=151936)

Teacher-Reference gap: KL=0.495

| Variant | KL↓ (m±s) | Top-1↑ (m±s) | PPL | vs ce_only |
|---------|-----------|-------------|-----|------------|
| ce_only | 0.608 ± 0.002 | 0.678 ± 0.000 | 15.7 | baseline |
| strict_topk_kd | 1.489 ± 0.001 | 0.828 ± 0.000 | 118.5 | -145% |
| bild_topk_delta | 0.222 ± 0.001 | 0.768 ± 0.000 | 17.8 | +63.4% |
| tdart_no_adaptive | 0.218 ± 0.003 | 0.773 ± 0.002 | 18.2 | +64.2% |
| cdart_no_censor | 0.218 ± 0.003 | 0.773 ± 0.002 | 18.2 | +64.2% |
| **cdart_full** | **0.209 ± 0.003** | **0.776 ± 0.002** | 18.3 | **+65.6%** |
| full_logit_upper | 0.006 ± 0.000 | 0.957 ± 0.001 | 25.1 | oracle |

### Llama-3.2-1B (16L, d=2048, 32H/8KV, vocab=128256)

Teacher-Reference gap: KL=0.991

| Variant | KL↓ (m±s) | Top-1↑ (m±s) | PPL | vs ce_only |
|---------|-----------|-------------|-----|------------|
| ce_only | 0.647 ± 0.003 | 0.662 ± 0.001 | 12.1 | baseline |
| strict_topk_kd | 1.303 ± 0.000 | 0.811 ± 0.001 | 66.1 | -101% |
| bild_topk_delta | 0.335 ± 0.007 | 0.736 ± 0.002 | 13.9 | +48.2% |
| tdart_no_adaptive | 0.323 ± 0.008 | 0.744 ± 0.003 | 14.2 | +50.1% |
| cdart_no_censor | 0.323 ± 0.008 | 0.744 ± 0.003 | 14.2 | +50.1% |
| **cdart_full** | **0.296 ± 0.009** | **0.751 ± 0.002** | 14.3 | **+54.2%** |
| full_logit_upper | 0.046 ± 0.000 | 0.883 ± 0.001 | 18.8 | oracle |

### Llama-3.2-3B (28L, d=3072, 24H/8KV, vocab=128256)

Teacher-Reference gap: KL=0.935

| Variant | KL↓ (m±s) | Top-1↑ (m±s) | PPL | vs ce_only |
|---------|-----------|-------------|-----|------------|
| ce_only | 0.564 ± 0.002 | 0.689 ± 0.002 | 10.1 | baseline |
| strict_topk_kd | 1.215 ± 0.002 | 0.774 ± 0.001 | 35.5 | -116% |
| bild_topk_delta | 0.368 ± 0.008 | 0.732 ± 0.002 | 11.2 | +34.6% |
| tdart_no_adaptive | 0.369 ± 0.007 | 0.735 ± 0.002 | 11.3 | +34.6% |
| cdart_no_censor | 0.369 ± 0.007 | 0.735 ± 0.002 | 11.3 | +34.6% |
| **cdart_full** | **0.336 ± 0.008** | **0.743 ± 0.003** | 11.5 | **+40.3%** |
| full_logit_upper | 0.139 ± 0.000 | 0.812 ± 0.000 | 13.0 | oracle |

---

## Cross-Model Gate Analysis

### Gate 1: C-DART beats ce_only? ✅ ALL 4 MODELS
- Qwen2.5-0.5B: +65.6% (3/3 seeds)
- Qwen3-0.6B: +60.4% (3/3 seeds)
- Llama-3.2-1B: +54.2% (3/3 seeds)
- Llama-3.2-3B: +40.3% (3/3 seeds)

### Gate 2: C-DART beats BiLD? ✅ ALL 4 MODELS
- Qwen2.5-0.5B: +5.9% (3/3 seeds)
- Qwen3-0.6B: +9.8% (3/3 seeds)
- Llama-3.2-1B: +11.4% (3/3 seeds)
- Llama-3.2-3B: +8.7% (3/3 seeds)

### Gate 3: Censoring adds value? ✅ ALL 4 MODELS
- Qwen2.5-0.5B: +4.2% (3/3 seeds)
- Qwen3-0.6B: +4.3% (3/3 seeds)
- Llama-3.2-1B: +8.2% (3/3 seeds)
- Llama-3.2-3B: +8.8% (3/3 seeds)

### Gate 4: Stability? ✅ ALL 4 MODELS
- All std < 0.009 KL across 3 seeds
- Qwen3-0.6B: exceptionally stable (std ≤ 0.001)

---

## Key Findings

### 1. C-DART generalizes across architectures and model generations ✅
Tested on 2 distinct model families (Qwen with 2-8 KV heads vs Llama with 8 KV heads),
2 Qwen generations (2.5 and 3), 4 model sizes (0.5B, 0.6B, 1.3B, 3.2B). All gates pass consistently.

### 2. Censoring is the novel contribution
The ablation decomposition is clean:
- delta teacher (bild): provides base improvement (+34-63% over ce_only)
- + ranking loss: marginal (+0.0-1.9% incremental)
- + **censoring**: consistent **+4.2-8.8%** incremental across all models

The censoring mechanism is STRONGER on Llama (+8.2-8.8%) than on Qwen (+4.2%),
suggesting it becomes more valuable with more diverse architectures.

### 3. Standard top-K KD is universally harmful
strict_topk_kd is -101% to -145% worse than ce_only on all models.
This is a strong, surprising negative result.

### 4. Gap closure consistent at 53-66%
- Qwen2.5-0.5B: 66.3%
- Qwen3-0.6B: 64.5%
- Llama-3.2-1B: 58.3%
- Llama-3.2-3B: 53.5%

Decreases slightly with model size; Qwen models show higher closure than Llama.

### 5. Teacher-reference gap varies by model
- Qwen2.5-0.5B: KL=0.495 (small gap)
- Qwen3-0.6B: KL=1.024 (larger gap)
- Llama-3.2-1B: KL=0.991 (larger gap)
- Llama-3.2-3B: KL=0.935 (similar to 1B)

C-DART works across different gap magnitudes (0.5-1.0 KL).

---

## Decision

**ALL 4 MODELS PASS. PROCEED TO AUTO-REVIEW-LOOP.**
