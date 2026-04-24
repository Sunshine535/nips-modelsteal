# Remaining Risks

## High Priority

### R1: Calibration phase itself uses oracle-only full logits
The `fit_calibration` step in `run_qumc.py` uses `get_full_logits_ORACLE_ONLY(ids)` to compute per-token reconstruction MSE. Under the strictest threat model (attacker ONLY has top-K + paid probe queries), this should be done by issuing enough probe queries over a calibration set. Currently this is a small "cheat" in the training protocol.

Mitigation: Reimplement `fit_calibration` to use the probe oracle with a declared budget. This is a few lines of code change but requires a larger probe set to get enough coverage.

### R2: Single model, single dataset
Only Qwen2.5-0.5B on WikiText-103 has been tested. GPT55_DIAGNOSIS §C10 requires a second model gate before broad claims.

Mitigation: Run same `run_qumc.py --config qumc_minimal.yaml` with `--model_name meta-llama/Llama-3.2-1B` and a second dataset.

### R3: Moment gate (src/moment_gate.py) not implemented
GPT55_DIAGNOSIS Task 8 asks for a moment/null-margin confidence gate. Current C variant uses calibration uncertainty only; moment gate is P2 optional.

Mitigation: Implement as follow-up. Currently does not block publication because C already beats baselines without it — moment gate would be an ablation demonstrating it is NOT required.

## Medium Priority

### R4: Full-logit upper bound D uses the same `get_full_logits_ORACLE_ONLY` as evaluation
Both the evaluation (for computing PPL/KL) and the D variant's training fetch use the same method. They are distinguishable by context (eval vs train loop), but a code reader could be confused. Should add a flag or separate method.

### R5: Probe budget not charged for calibration
Currently the QueryBudget object tracks training-time probe queries but the calibration phase queries are not counted. If we count them, fair comparison requires accounting.

### R6: Paper, README, CLAIMS_FROM_RESULTS not yet updated
Per Task 11, we deliberately did not update these until broader baselines are done. Risk is that the current state is internally inconsistent until updated.

## Low Priority / Documented But Not Fixed

### R7: Old S-PSI code has P1 bugs
Silent perturb fallback, train-mode during deterministic matching, unnormalized regularizer. Not touched this pass because S-PSI is kept only as ablation.

### R8: Moment-CP artifacts incomplete in results/
The `results/v5_attack_moments_v2/` directory exists but raw artifacts for the paper's 0.813 claim are incomplete. Paper main claim must be frozen until re-run.

### R9: Active-query old results not yet physically archived
Documented as INVALID in evidence_registry but not moved to archive/. Paper and README must not cite.

## Non-Risk (already addressed)

- Config string-vs-float bug: FIXED in this pass.
- v13/v14 contamination: TAGGED in evidence_registry.
- Strict oracle separation: TESTED (11/11 pass).

## Decision Implication

Per GPT55_DIAGNOSIS §19 "Continue / Stop / Pivot":
- **Continue** ← current state. Full Q-UMC beats baselines over 3 seeds with valid query accounting (at least for training; calibration is a caveat R1).

Risks R1, R2, R3 should be addressed before NeurIPS-strength claims. R6 is sequential (paper update after R2 done).
