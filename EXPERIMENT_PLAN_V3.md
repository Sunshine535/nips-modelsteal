# V3 Experiment Plan — Observation Space Expansion

## Hypothesis

The rank=32 Gramian observed in v1/v2 is an artifact of insufficient observation:
- Only K=8 suffix positions observed (out of 128)
- Only k=64 random probes (sketched Gramian is 64×64)

**If true**: Expanding K and k will reveal much higher true Gramian rank → better parameter recovery is possible.
**If false**: rank=32 is a structural limit of single-block last-layer observation → need fundamentally different approach.

## Hardware: 4× A100-80GB (same as v1/v2)

## Experiment Matrix

### Phase 1: Diagnostic (GPU 0-1, ~2-4 hours)

| Wave | GPU | Script | Config | Purpose |
|------|-----|--------|--------|---------|
| 1 | 0 | `diagnose_gramian_rank.py` | K=8, k={64,128,256} | Verify true rank with current K |
| 2 | 1 | `diagnose_gramian_rank.py` | K={8,32,64,128}, k=128, also_block_22 | Test position expansion |

Key metric: How does `rank_above_1e-4` scale with K?

### Phase 2: Training (GPU 2-3, ~2-4 hours)

| Wave | GPU | Script | Config | Purpose |
|------|-----|--------|--------|---------|
| 3 | 2 | `run_expanded_observation.py` | K=64, k=128, alg_clean, s42 | Training with 8× more positions |
| 4 | 3 | `run_expanded_observation.py` | K=32, k=128, alg_clean, s42 | Training with 4× more positions |

Key metric: Does mean_cosine improve beyond v2 baseline (0.12-0.14)?

### Phase 3: Conditional (depends on Phase 1-2 results)

| Condition | Action | GPU Budget |
|-----------|--------|-----------|
| rank↑ significantly + cos↑ | More seeds, active queries | 8 GPU-hours |
| rank↑ but cos unchanged | Check init quality, try more steps | 4 GPU-hours |
| rank≈32 regardless of K | Pivot to active queries or functional bootstrap | 4 GPU-hours |

## Decision Tree

```
Phase 1 complete
├── rank(K=128) >> rank(K=8)?
│   ├── YES → Observation bottleneck confirmed
│   │   ├── Phase 2 cos > 0.14?
│   │   │   ├── YES → BREAKTHROUGH. Run 3 seeds, update paper.
│   │   │   └── NO  → rank increased but init/optimization still fails.
│   │   │       → Try: (a) more steps, (b) active query, (c) larger init_truncation_rank
│   │   └── ...
│   └── NO  → Structural limit confirmed
│       ├── rank=32 is fundamental for Block 23
│       ├── Theory: d_model=896 → 896 potentially observable directions
│       │   but Jacobian coupling reduces effective rank to ~32
│       ├── Next: (a) Gramian-aware active queries
│       │         (b) Functional bootstrap (use KL=0.42 clone)
│       │         (c) Accept as true negative, strengthen paper
│       └── ...
```

## Code Changes (from v2)

### New files
- `scripts/diagnose_gramian_rank.py` — Gramian rank diagnostic with variable K, k
- `scripts/run_expanded_observation.py` — S-PSI training with expanded K
- `scripts/dispatch_v3_experiments.sh` — 4-GPU dispatch

### Modified files
- `src/active_query.py` — Added `GramianAwareSelector` class

### Memory considerations
- K=128, k=256: S_clean matrix = ~20 GB → uses CPU-side accumulation path
- K=64, k=128: S_clean = ~5 GB → fits on GPU with batch_size=1-2
- K=32, k=256: S_clean = ~5 GB → fits on GPU

## Estimated Compute

| Phase | GPU-hours | Wall-clock |
|-------|-----------|-----------|
| Phase 1 diagnostic | 8h (4h × 2 GPUs) | ~4h |
| Phase 2 training | 8h (4h × 2 GPUs) | ~4h |
| Phase 3 conditional | 4-8h | ~2-4h |
| **Total** | **20-24h** | **~10h** |

## Status

- [x] Implementation complete
- [x] Deployed to server (4×A100)
- [x] Phase 1 diagnostic results — ALL full rank. v2 rank=32 was K=8 observation bottleneck.
- [x] Phase 2 training results (K=32) — cos≈0.12 (no improvement). Only LayerNorm recovers.
- [ ] Phase 2 training (K=64) — OOM, needs solo run
- [x] Phase 3 decision: **rank↑ confirmed but cos unchanged** → per decision tree: "Check init quality, try more steps"

## V3 Results Summary

### Diagnostic (Phase 1)

| Block | K | k | rank | σ_max | κ | eff_rank |
|-------|--:|---:|-----:|------:|----:|--------:|
| 23 | 8 | 128 | 128 | 11,462 | 2.44 | 122.0 |
| 23 | 32 | 128 | 128 | 44,879 | 2.41 | 122.1 |
| 23 | 64 | 128 | 128 | 89,327 | 2.39 | 122.2 |
| 23 | 128 | 128 | 128 | 178,067 | 2.36 | 122.2 |
| 22 | 8 | 128 | 128 | 25,479 | 3.17 | 116.7 |
| 22 | 32 | 128 | 128 | 99,262 | 3.14 | 116.8 |
| 22 | 64 | 128 | 128 | 196,615 | 3.09 | 116.9 |
| 22 | 128 | 128 | 128 | 394,729 | 3.07 | 117.0 |

Key findings:
- σ_max ∝ K (linear scaling)
- κ constant (~2.4 for Block 23, ~3.1 for Block 22)
- Block 22 signal **2.2× stronger** than Block 23

### Training (Phase 2, K=32)

| Component | cos | Detail |
|-----------|-----|--------|
| lm_head | 0.209 | improved from v2's 0.12 |
| Block 23 | 0.128 | LayerNorm recovers (0.67/0.98), weights ≈ 0 |
| Block 22 | 0.142 | LayerNorm recovers (0.73/0.99), weights ≈ 0 |

### Root Cause Analysis

1. **v2 rank=32**: Real K=8 bottleneck, not structural. Expanding K → full rank.
2. **v2 Block 22 Gramian=0**: `_BoundaryInjectionHook` at Block 23 input truncates Block 22 signal. Correct behavior in oracle mode.
3. **Full-rank Gramian but cos≈0**: Observability (rank) is necessary but not sufficient. Optimization in 14.9M-dim space with 128-dim algebraic init coverage (0.0009%) cannot bridge the gap. Non-convex landscape traps SGD.
