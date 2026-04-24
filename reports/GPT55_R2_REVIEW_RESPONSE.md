# GPT-5.5 Pro R2 Review Response

**Verdict accepted: Decision F — current `qumc_minimal` results are INVALID DUE TO BUG for the strict black-box Q-UMC claim.**

## Four Critical Issues Confirmed

### Issue 1 (P0): Teacher `lm_head.weight` leaks into strict B/C path
- `scripts/run_qumc.py` line: `W_lm = teacher.lm_head.weight.data.float()`
- Passed to `CalibratedLogitCompleter(W_probe, probe_ids, W_lm, ...)` for variants B (`completion_no_unc`) and C (`completion_uncertainty`)
- This is NOT strict black-box — attacker in reality would only have Carlini SVD–recovered W_eff (up to gauge) or a no-access null basis
- Impact: B/C improvements cannot be attributed to the strict Q-UMC mechanism

### Issue 2 (P0): Calibration uses validation full logits
- `fit_calibration` call uses `api.get_full_logits_ORACLE_ONLY(ids)` on `eval_loader`
- Two violations: (a) full-logit access outside oracle upper-bound; (b) same split used for calibration AND evaluation → data leakage
- Report (REMAINING_RISKS.md R1) acknowledged the access issue but failed to flag the data leakage

### Issue 3 (P1): KD loss normalization inconsistent between variants
- B/D use `F.kl_div(..., reduction="batchmean")` → divides by batch dimension only
- C uses `(w * t_soft * (t_soft.log() - s_logsm)).sum(-1).mean()` → divides by `B*T` (mean over all positions)
- For seq_len=128, C's effective KD weight is ~128× smaller than B's
- C's "improvement" over B is partially explained by CE term dominating (KD downweighted)

### Issue 4 (P1): C vs B KL comparison was cherry-picked
- Report claimed "C > B" based on PPL only
- Actual KL: seed 0 B=1.94 C=2.10 (B better), seed 1 B=2.41 C=2.33 (C slightly better), seed 2 B=1.93 C=2.10 (B better)
- Mean KL: B=2.09 < C=2.18 — C is WORSE at teacher-student imitation
- This contradicts the paper's central claim about functional cloning

## Consequences

Per GPT55 R2 Task 1, these results are re-classified:

| Previous Label | New Label |
|----------------|-----------|
| `results/qumc_minimal/` as "positive A/B/C validation" | **ORACLE_WLM_AND_CALIBRATION_LEAK_WEAK_SIGNAL** — weak PPL signal only under oracle teacher output layer + oracle calibration; does NOT validate strict Q-UMC |
| CORE_COMPARISON.md "C>B>A validates mechanism" | **DOWNGRADED** — cannot support strict-claim conclusion |
| v13/v14 "simulator-positive signal" | Unchanged (already labeled as contaminated) |

## Raw results preserved
All JSON in `results/qumc_minimal/` remain unchanged. Only the interpretation is downgraded.

## Next Actions (following GPT55 R2 Tasks 2-9)

- [ ] Task 2: BasisProvider, ban teacher.lm_head.weight in strict path
- [ ] Task 3: Disjoint calibration split via counted probe oracle
- [ ] Task 4: Unified sequence_kl_loss function for all variants
- [ ] Task 5: Add old_lc_simulator variant for same-seed A-baseline
- [ ] Task 6: Fix manifest per-variant + per-phase budgets
- [ ] Task 7: Mechanism logs (tail MSE, weight stats, KD/CE ratio)
- [ ] Task 8: Rerun strict minimal v2 (only after 1-7 pass)
- [ ] Task 9: Second model (only after strict_v2 passes)

## Honest Method Naming Until Fixed

The current implementation should be named:
> **"oracle-W + oracle-calibration + weighted-KD variant"** — not Q-UMC, not strict black-box, not query-budgeted.

True Q-UMC requires Tasks 2-4 to be completed before the name is earned.
