# Research Pipeline Report

**Direction**: Transformer Tomography — Characterizing Weight Identifiability Limits from Black-Box Logit Access
**Chosen Idea**: Transformer Tomography (ranked #1, best paper potential 9-10/10)
**Date**: 2026-04-08 -> 2026-04-14
**Pipeline**: idea-discovery -> implement -> run-experiment -> auto-review-loop
**Difficulty**: nightmare | **Effort**: beast

## Journey Summary

- **Ideas generated**: 10 -> filtered to 4 -> piloted 2 -> chose 1 (Transformer Tomography)
- **Implementation**: Built full framework — `src/gramian.py`, `src/symmetry_gauge.py`, `src/algebraic_init.py`, extended `src/parameter_inverter.py` and `src/permutation_alignment.py`. Paper written from scratch in LaTeX (NeurIPS 2026 format).
- **Experiments**: 5 major experiment sets across 4x A100-80GB remote server
  - v3: Gramian diagnostics + position sweep + probe sweep + expanded observation training
  - v4: KD suffix baseline (3 seeds), active query (buggy, excluded), Llama-3.2-1B S-PSI
  - Warm-start interpolation sweep
  - Total: ~100+ GPU-hours
- **Review rounds**: 6 scored rounds (2 medium + 1 proof-checker + 3 nightmare)
- **Proof-checking**: 3 rounds, nightmare difficulty, 7.5/10. Fixed 5 FATAL + 5 CRITICAL + 8 MAJOR + 3 MINOR issues.

## Stage Completion

| Stage | Status | Output |
|-------|--------|--------|
| 1. Idea Discovery | COMPLETE | `IDEA_REPORT.md` |
| 2. Implementation | COMPLETE | `src/`, `scripts/`, `paper/main.tex` |
| 3. Deploy Experiments | COMPLETE | `results/v3_remote/`, `results/v4_*/`, `results/warmstart_sweep/`, `results/expanded_obs_K128/` |
| 4. Auto Review Loop | COMPLETE (7/10) | `AUTO_REVIEW.md`, `REVIEW_STATE.json` |
| 5. Research Summary | COMPLETE | `NARRATIVE_REPORT.md`, this file |
| 6. Paper Writing | NOT STARTED | `AUTO_WRITE=false` |

## Score Progression

| Phase | Round | Score | Difficulty |
|-------|-------|-------|------------|
| Medium review | 1 | 5/10 | medium |
| Medium review | 2 | 7/10 | medium |
| Proof-checker | - | 7.5/10 | nightmare |
| Nightmare review | 1 | 6/10 | nightmare |
| Nightmare review | 2 | 6.5/10 | nightmare |
| Nightmare review | 3 | 7/10 | nightmare |

## Key Findings

1. **Gramian full-rank**: The suffix observability Gramian G(Q) has full rank (= k) for ALL tested (K, k) configurations, with sigma_max scaling linearly in K and kappa ~ 1.9-3.6.

2. **Observability-Recoverability Gap**: Despite full-rank Gramian, parameter recovery yields cos ~ 0.12-0.14 for all weight matrices. Only RMSNorm recovers (cos 0.67-0.99).

3. **Warm-start evidence**: Training contributes |Delta cos| < 0.01 at ALL initialization distances. Optimization cannot traverse the non-convex landscape.

4. **K-expansion null**: K=128 yields identical recovery to K=32, confirming bottleneck is optimization, not observation.

5. **Cross-architecture**: Llama-3.2-1B shows same negative result for blocks (cos ~ 0.22, driven by LayerNorm) with partial lm_head recovery (cos 0.66 Procrustes).

6. **Functional paradox**: Models with cos ~ 0.12 achieve KL 0.42 (24x better than random), explained by RMSNorm/lm_head recovery in low-dimensional subspace.

## Key Fixes Across All Rounds

1. SiLU gate symmetry corrected (gate excluded from MLP scaling)
2. Attention V/O per KV group, not per head
3. Sequence-level Jacobians (self-attention couples positions)
4. Rank bound rewritten with Td factorization
5. Depth screening theorem: query-dependent Jacobians kept inside sum
6. Stale "rank~32" narrative removed (v3 shows full rank for all configs)
7. Llama-3.2-1B cross-architecture validation integrated
8. Warm-start claims scoped to block-local oracle experiment
9. Llama mechanism claim labeled as conjecture
10. Active query bug diagnosed (student = teacher), data excluded

## Writing Handoff

- **NARRATIVE_REPORT.md**: GENERATED
- **Venue**: NeurIPS 2026 (run `/paper-writing` manually)
- **Manual figures needed**:
  - `figures/framework_overview.pdf` — Architecture diagram
  - `figures/gramian_eigenspectrum.pdf` — Eigenspectrum plot

## Remaining TODOs (if time permits)

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| HIGH | Matched-budget KD baseline | 4-8 GPU-hours | Disambiguates functional recovery section |
| MEDIUM | Check in raw v2 JSON artifacts | SSH + download | Table provenance for Exp1/2/3/4 |
| LOW | Fix active query script + re-run | 2 GPU-hours + code fix | Would add active query results |
| LOW | Llama multi-seed | 4 GPU-hours | Strengthen cross-architecture claim |
| TRIVIAL | Delete/quarantine buggy active query data | 1 min | Repo hygiene |

## Compute Budget

| Phase | GPU-hours | Details |
|-------|-----------|---------|
| v1-v2 experiments | ~40h | Multiple configs, seeds, smoke tests |
| v3 diagnostics | ~24h | Gramian rank analysis, K-sweep, re-training |
| v3 expanded obs | ~20h | K=128 training, warm-start sweep |
| v4 experiments | ~12h | KD baseline, Llama, active query |
| Failed/OOM runs | ~8h | Various |
| **Total** | **~104h** | 4x A100-80GB server |

## Reviewer's Final Note (GPT-5.4, nightmare)

> "If you must submit now, it is submit-able. If you want the safer version, the highest-value last fix is not new theory, it is making the older tables auditable."
