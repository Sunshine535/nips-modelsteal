# Evidence Registry

This file classifies all experimental results by reliability and usability for claims.

## INVALID (must not be cited as positive evidence)

| Result | Source | Reason | Date Invalidated |
|--------|--------|--------|-----------------|
| Active-query cos≈0.999 | `findings.md`, old active-query runs | BUG CONFIRMED: student initialized with teacher weights | 2026-04-19 |
| qumc_minimal C>B>A "strict black-box Q-UMC" | `results/qumc_minimal/`, `reports/CORE_COMPARISON.md` v1 | **ORACLE_WLM_AND_CALIBRATION_LEAK_WEAK_SIGNAL** — B/C use teacher `lm_head.weight` as completion basis; calibration uses validation full logits; KD loss normalization inconsistent between B and C (~128× scale diff); C mean KL=2.18 WORSE than B mean KL=2.09. Raw JSON preserved but re-interpreted as weak PPL signal under oracle-teacher-W + oracle-calibration, NOT strict Q-UMC validation. See `reports/GPT55_R2_REVIEW_RESPONSE.md`. | 2026-04-25 |

## SIMULATOR-POSITIVE (signal present but not strict black-box)

| Result | Source | Issue | Usable As |
|--------|--------|-------|-----------|
| v13 Logit Completion E topK=20: PPL 137 vs A 178 | `results/v13_lc_topk20/results.json` | `complete_logits` accesses full teacher logits internally; not strict top-K API | Historical signal showing dense tail helps; NOT final evidence for top-K claim |
| v14 Logit Completion E topK=5: PPL 136 vs A 190 | `results/v14_lc_topk5/results.json` | Same full-logit access issue | Same as above |

## UNDER-AUDITED (raw artifacts incomplete)

| Result | Source | Issue | Required to Restore |
|--------|--------|-------|-------------------|
| Moment-CP W_lm top-5 cos 0.813 | `ATTACK_4WAY_SUMMARY.md`, `paper/main.tex` | Script header says quarantined; raw `results/v5_attack_moments_v2/` present locally but artifacts sparse | Full re-run with unquarantined script, seed control, and manifest |
| S-PSI Gramian full-rank claims | `CLAIMS_FROM_RESULTS.md` | Raw logs absent from public results | Upload original logs or re-run with manifest |

## VERIFIED NEGATIVE (reliable, valuable as mechanism evidence)

| Result | Source | What It Shows |
|--------|--------|--------------|
| v7 random init 5k: PPL≈2800 | `results/v7_scrd/results.json` | Random-init KD under small budget = failure |
| v9 random init 50k: PPL≈2000 | `results/v9_scrd_long/results.json` | Even 10x more steps from random = still failure |
| v10 σ=0.1 perturbation: PPL=10^41 | `results/v10_perturb01/results.json` | Large perturbation destroys model |
| v13/v14 hidden MSE (B): PPL 700-750 | `results/v13_lc_topk20/results.json` | Hidden-state MSE conflicts with logit KD under top-K |
| S-PSI beta=0 ≈ sensitivity | `CLAIMS_FROM_RESULTS.md` | Sensitivity mechanism not useful |

## VERIFIED POSITIVE (reliable, usable as evidence)

| Result | Source | What It Shows |
|--------|--------|--------------|
| v8 pretrained+noise full-logit KD: PPL 135 | `results/v8_scrd_pretrained/results.json` | Dense full-logit supervision is strong (upper bound) |
| Carlini SVD subspace cos 0.9999 | Prior experiments | Output projection subspace recoverable |
| h_final lstsq recovery cos 0.977 | `/tmp/functional_theft_v2.py` results | Per-query hidden state recoverable at high quality |
