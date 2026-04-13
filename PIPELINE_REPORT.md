# Research Pipeline Report

**Direction**: Progressive Parameter Inversion — Recovering LLM Weights from Black-Box Logit Access
**Chosen Idea**: Transformer Tomography: Observability-Governed Weight Recovery from Black-Box Logit Access
**Date**: 2026-03-28 → 2026-04-13 (3 auto-review loops across ~2 weeks)
**Pipeline**: idea-discovery → implement → run-experiment → auto-review-loop (×3)

## Journey Summary

- **Ideas generated**: 6 → filtered to 3 (novelty confirmed) → piloted 1 → chose "Transformer Tomography"
- **Implementation**: S-PSI framework (Sensitivity-Guided Progressive Suffix Inversion) with Gramian analysis, algebraic initialization, and gauge-quotiented parameter space
- **Experiments**: 15+ GPU experiments across 4×A100-80GB server, ~100 GPU-hours total
- **Review rounds**: 3 auto-review loops, 13 total scored rounds
  - v1 (nightmare): 8 rounds → 7/10 Weak Accept
  - v2 (medium, post-bugfix): 3 rounds → 7/10 Weak Accept
  - v3 (medium, expanded observations): 2 rounds → 7/10 Ready for submission
- **Final score**: 7/10 — "Ready for submission"

## Score Progression

| Loop | Round | Score | Verdict |
|------|-------|-------|---------|
| v1 | 1 | 5 | Reject |
| v1 | 2 | 6 | Not ready |
| v1 | 3 | 4 | Not ready |
| v1 | 5 | 4 | Not ready |
| v1 | 7 | 6 | Not ready |
| v1 | 8 | 5 | Reject |
| v1 | 9 | 7 | Borderline Accept |
| v1 | 10 | 7 | Weak Accept |
| v2 | 1 | 5 | Almost |
| v2 | 2 | 6 | Almost |
| v2 | 3 | 7 | Weak Accept |
| v3 | 1 | 6 | Almost |
| **v3** | **2** | **7** | **Ready for submission** |

## Key Findings

1. **Gramian full-rank**: The suffix observability Gramian G(Q) is full-rank and well-conditioned (κ≈2.4) for Qwen2.5-0.5B Block 23, with σ_max scaling linearly with K (suffix positions). The v2 rank=32 was a K=8 observation bottleneck.

2. **Observability ≠ Recoverability**: Despite full-rank Gramian, parameter recovery yields cos≈0.12 for internal weight matrices. LayerNorm and lm_head partially recover (cos 0.2–0.99), but Q/K/V/O/MLP projections remain near random.

3. **Warm-start smoking gun**: Training contributes |Δcos| < 0.01 at ALL initialization distances (α=0 to α=1). SGD cannot traverse the non-convex landscape regardless of starting point proximity.

4. **K-expansion null result**: K=128 (4× more suffix positions) yields identical recovery to K=32 (Block 23 cos=0.129 vs 0.128), confirming the bottleneck is optimization, not observation.

5. **Functional recovery paradox**: Models with cos≈0.12 (parameter-level failure) achieve KL=0.38 and Top-1=0.26 (functional partial success), explained by RMSNorm/lm_head recovery accounting for output distribution structure.

6. **Negative result as contribution**: Internal transformer weights are protected from extraction via passive logit queries — not because they are unobservable, but because gradient-based optimization cannot navigate the 14.9M-dimensional non-convex landscape.

## Paper Evolution

| Version | Narrative | Trigger |
|---------|-----------|---------|
| v1 | "We can recover weights" | Initial hypothesis |
| v1.5 | "Identifiability analysis reveals partial recovery" | cos≈0 for large matrices |
| v2 | "Negative result + identifiability framework" | Bugfix revealed v1 data contaminated |
| v3 | **"Observability-Recoverability Gap"** | Warm-start + K-expansion evidence |

## Final Status

- [x] Ready for submission (7/10, "Ready for submission")
- [x] AUTO_REVIEW.md complete with full 13-round history
- [x] Paper (main.tex) updated with all results and proper framing
- [x] All experiments completed and results archived
- [x] Functional evaluation completed
- [x] Warm-start sweep (key new evidence) completed

## Remaining TODOs (optional, not blocking)

- [ ] Add 1-2 additional seeds on warm-start endpoints (α=0, α=1)
- [ ] Create optimization-trace figure (update-norm / distance-traveled / loss-curves)
- [ ] Test on a second architecture (Llama-3.2-1B) for generality

## Files Changed

### Core source
| File | Description |
|------|-------------|
| `src/parameter_inverter.py` | S-PSI engine, boundary injection, oracle sensitivity |
| `src/permutation_alignment.py` | Hungarian alignment, aligned cosine similarity |
| `src/symmetry_gauge.py` | Continuous gauge basis, projection (new) |
| `src/gramian.py` | Flat param spec, JVP sketch, Gramian computation (new) |
| `src/algebraic_init.py` | Sketched Gauss-Newton initialization (new) |

### Scripts
| File | Description |
|------|-------------|
| `scripts/run_spsi.py` | Main S-PSI training script |
| `scripts/run_warmstart_sweep.py` | Warm-start initialization sweep (new) |
| `scripts/run_gramian_eval.py` | Offline Gramian analysis (new) |
| `scripts/diagnose_gramian_rank.py` | Gramian rank diagnostic (new) |
| `scripts/eval_functional.py` | Functional evaluation (new) |

### Paper
| File | Description |
|------|-------------|
| `paper/main.tex` | 7 tables, 2 figures, ~40 pages with appendices |

### Results
| Path | Description |
|------|-------------|
| `results/warmstart_sweep/` | Warm-start sweep results (7 alpha values) |
| `results/expanded_obs_K128/` | K=128 expanded observation training |
| `results/v3_remote/` | v3 experiment results from server |

### Documentation
| File | Description |
|------|-------------|
| `AUTO_REVIEW.md` | Full review history (v1+v2+v3, 13 rounds) |
| `EXPERIMENT_PLAN.md` | Original experiment design |
| `EXPERIMENT_PLAN_V3.md` | v3 observation expansion plan |
| `REVIEW_STATE.json` | Loop state (completed, 7/10) |
| `IDEA_REPORT.md` | Idea discovery output |

## Compute Budget

| Phase | GPU-hours | Details |
|-------|-----------|---------|
| v1 experiments | ~40h | 7 configs × 1-4 seeds, smoke tests, failed runs |
| v2 diagnostic | ~24h | Gramian rank analysis, K-sweep, re-training |
| v3 expanded obs | ~20h | K=128 training, warm-start sweep (7 runs) |
| v3 functional eval | ~4h | Held-out functional evaluation |
| Failed/OOM runs | ~12h | K=128 OOM, warmstart OOM (3×) |
| **Total** | **~100h** | 4×A100-80GB server |

## What Worked

1. **Observability theory framing** — casting model stealing as a control-theory observability problem was the key intellectual contribution
2. **Gauge projection** — without it, the Gramian analysis is meaningless due to symmetry degeneracies
3. **Warm-start sweep** — the single most impactful experiment, providing direct causal evidence for the optimization barrier
4. **Iterative narrative pivot** — evolving from "recovery works" to "recovery fails and here's why" was essential for acceptance
5. **Honest framing** — narrowing claims to a case study and softening causal language

## What Could Be Better

1. **Cross-model validation** — running on Llama-3.2-1B would strengthen generality claims
2. **Active query optimization** — adversarial queries remain an open question
3. **Larger probe count** — k=256 or k=1024 for tighter Gramian rank estimates
4. **Multi-seed statistical tests** — only 1-3 seeds per config, no confidence intervals
