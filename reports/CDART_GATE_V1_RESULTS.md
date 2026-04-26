# C-DART Gate v1 — Final 3-Seed Results

## Setup
- Teacher: Qwen2.5-0.5B fine-tuned 500 steps on WikiText TEST (private)
- Reference/Student: Qwen2.5-0.5B base (public)
- Teacher-Reference gap: KL=0.495, Top-1=67.4%
- Training: 5000 steps, batch 8, seq_len 128, topK=20
- 3 GPUs parallel (1 seed per GPU)

## Headline Table (3 seeds, mean ± std)

| Variant | KL↓ (m±s) | Top-1↑ (m±s) | PPL (m±s) |
|---------|-----------|-------------|-----------|
| ce_only | 0.608 ± 0.002 | 0.678 ± 0.000 | 15.73 ± 0.02 |
| strict_topk_kd | 1.489 ± 0.001 | 0.828 ± 0.000 | 118.54 ± 0.31 |
| bild_topk_delta | 0.222 ± 0.001 | 0.768 ± 0.000 | 17.76 ± 0.01 |
| tdart_no_adaptive | 0.218 ± 0.003 | 0.773 ± 0.002 | 18.15 ± 0.03 |
| cdart_no_censor | 0.218 ± 0.003 | 0.773 ± 0.002 | 18.15 ± 0.03 |
| **cdart_full** | **0.209 ± 0.003** | **0.776 ± 0.002** | 18.25 ± 0.05 |
| full_logit_upper | 0.006 ± 0.000 | 0.957 ± 0.001 | 25.05 ± 0.01 |

## Gate Results

### Gate 1: CE-only — ALL delta methods beat CE-only? ✅
- cdart_full vs ce_only: **+65.6% KL improvement**, 3/3 seeds

### Gate 2: Censoring — cdart_full beats cdart_no_censor? ✅
- Per-seed: +3.2%, +4.8%, +4.4%
- Mean: **+4.2% ± 0.7%**, ALL 3 seeds
- Censored top-K absence constraints provide consistent positive incremental signal

### Gate 3: Novelty — cdart_full beats BiLD-style baseline? ✅
- Per-seed: +5.0%, +7.9%, +4.8%
- Mean: **+5.9% ± 1.4%**, ALL 3 seeds
- C-DART differentiates from simple top-K delta MSE

### Gate 4: Stability — consistent across seeds? ✅
- cdart_full std = 0.003 KL (very stable)
- All ordering preserved across all 3 seeds

## Method Ranking (by KL, lower is better)
```
1. cdart_full          0.209  ← MAIN METHOD (strict black-box)
2. tdart_no_adaptive   0.218  ← ablation: no censoring
3. cdart_no_censor     0.218  ← ablation: = tdart_no_adaptive
4. bild_topk_delta     0.222  ← closest prior-work baseline
5. ce_only             0.608  ← no teacher signal
6. strict_topk_kd      1.489  ← sparse top-K KL (harmful)
7. full_logit_upper    0.006  ← oracle
```

## Decision
**CONTINUE TO FULL BENCHMARK.** C-DART passes all gates:
- Beats CE-only by 65.6% KL (3/3 seeds)
- Beats BiLD baseline by 5.9% KL (3/3 seeds)
- Censored mechanism provides 4.2% incremental value (3/3 seeds)
- Highly stable (std < 0.003 KL)
