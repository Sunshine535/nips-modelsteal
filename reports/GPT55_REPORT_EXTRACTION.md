# GPT-5.5 Pro Report Extraction

## Diagnosis File Used
`GPT55_DIAGNOSIS.md` (repo root, 890 lines, accessed 2026-04-24)

## Recommended MAIN METHOD PATH
**Q-UMC: Query-budgeted Uncertainty-gated Moment-guided Logit Completion**
- Strict top-K + optional logit-bias/probe API with counted query budget
- Recover h_hat via lstsq from counted probe logits
- Reconstruct dense tail logits, merge with exact top-K
- Weight tail by calibration uncertainty and optional moment/null-margin confidence
- Train student with uncertainty-weighted dense KL + CE; NO hidden-state MSE

## Missing Mechanism
Query-accounted strict oracle abstraction + calibrated uncertainty tail weighting. Current v13/v14 use full teacher logits internally (not strict black-box). The mechanism gap is: no strict oracle, no probe query accounting, no uncertainty weighting, no moment confidence gate.

## Evidence From Positive Results
- v13/v14: Logit Completion E beats top-K KD by 23-28% PPL (simulator, not strict)
- v8: Full-logit dense KD achieves PPL 135 (upper bound ceiling)
- Carlini SVD recovers output projection subspace at cos 0.9999
- Moment-CP W_lm top-5 cos 0.813 (under-audited, raw artifacts sparse)

## Evidence From Negative Results
- v7/v9: Random init KD fails (PPL 2000-2800)
- v13/v14 B/C/D: Hidden-state MSE catastrophically hurts (PPL 700-750 vs 178 baseline)
- S-PSI: sensitivity beta=0 no worse; K expansion no recovery; warmstart negligible
- Jacobian FD: sees common architecture subspace, not teacher-specific
- Memory probing: 77x rank expansion but no parameter recovery

## Evidence From Unstable Results
- Active-query cos 0.999: INVALID (student loaded teacher weights)
- Paper claim audits: repeatedly FAIL (number mismatches, evidence gaps)
- Moment-CP script quarantined but paper uses Moment thesis

## Evidence From Failed Ablations
- beta=0 same as sensitivity → sensitivity mechanism useless
- K expansion 32→128 no recovery → more observations != identifiability
- Hidden MSE worse under top-K → representation/behavior conflict

## Why Existing Best Positive Fragment Is Insufficient
v13/v14 E variant uses full `t_logits` from teacher to recover probe logits, then only masks top-K for the scatter-back. This is NOT strict top-K black-box access. If claim is "top-K API attack", results are contaminated.

## Files to Inspect
- `scripts/enhanced_kd_clone.py` (Logit Completion, P0 bug)
- `src/parameter_inverter.py` (S-PSI, P1 bugs)
- `findings.md` (active-query bug)
- `results/v7-v14/results.json`
- `tests/test_modelsteal.py`

## Files to Edit
- NEW: `src/oracles.py`, `src/logit_completion.py`, `src/result_manifest.py`
- NEW: `scripts/run_qumc.py`
- NEW: `configs/qumc_smoke.yaml`, `configs/qumc_minimal.yaml`
- NEW: `tests/test_oracles.py`, `tests/test_logit_completion.py`, `tests/test_manifest.py`
- NEW: `docs/evidence_registry.md`

## Files to Archive
- Active-query old positive results (INVALID)
- v13/v14 reclassified as simulator-positive

## Files to Keep
- All raw result JSONs
- `scripts/enhanced_kd_clone.py` (historical reference)
- `scripts/reproduce_carlini.py` (baseline)

## Files to Keep Only as Baseline
- Sparse top-K KD (variant A)
- Full-logit KD upper bound (v8)
- Random KD (v7/v9, negative baseline)
- Carlini SVD

## Files to Keep Only as Ablation
- Hidden-state MSE/SCRD (variants B/C/D)
- S-PSI sensitivity
- Algebraic init

## Suspected Bugs
- P0: `complete_logits` uses full teacher logits (not strict black-box)
- P0: Active-query student=teacher init (confirmed INVALID)
- P0: Moment-CP script quarantined but paper main claim
- P1: `parameter_inverter.py` silent perturb fallback
- P1: CUDA autocast hardcoded (CPU tests fail)
- P1: S-PSI student.train() during deterministic matching

## Required Logging
- query_count_topk, query_count_probe, effective_budget
- tail_completion_mse, tail_completion_kl
- uncertainty weight distribution, fraction_tail_active
- moment_null_margin, moment_gate_tokens
- student_ppl, teacher_student_KL, top1_agreement
- config, command, git hash, seed, data split hash

## Required Minimal Experiments
1. Smoke test (Q-UMC code runs)
2. Data/metric sanity
3. One-batch overfit
4. Strict oracle activation test
5. Reproduce old positive fragment (old E in simulator)
6. Reproduce old negative (random KD)
7. A/B/C core comparison (strict_topk / completion_no_uncertainty / full_qumc)
8. Multi-seed stability (3-5 seeds)

## Required Core Comparison
- A: Existing Best Positive Fragment Only (old E simulator)
- B: Q-UMC without new mechanism (strict completion, no uncertainty/moment)
- C: Full Q-UMC (strict completion + uncertainty + optional moment)

## Required Baselines
- Sparse top-K KD, full-logit KD upper bound, old E simulator, Carlini-only, matched KD

## Stop / Continue / Pivot Criteria
- **Continue**: Full Q-UMC beats strict top-K KD and no-uncertainty completion over ≥3 seeds
- **Stop**: Strict completion loses all advantage after full-logit leakage removal
- **Pivot**: Uncertainty helps but moment gate does not → drop Moment, pure calibrated completion
