# Bug Fix Log

## Bug Fix: P0 — Logit Completion used full teacher logits (not strict black-box)

Files changed: NEW `src/oracles.py`, NEW `src/logit_completion.py`, NEW `scripts/run_qumc.py` (isolates the strict path from `scripts/enhanced_kd_clone.py` which is kept unchanged as historical reference).

Reason: GPT55_DIAGNOSIS.md identified this as the P0 bug (row P17 in phenomenon ledger). Old `complete_logits` in `enhanced_kd_clone.py` received full teacher logits via `teacher(...).logits` and then extracted probe rows from them. This means v13/v14 positive results were produced under a setting that leaks information beyond top-K + logit-bias API.

Evidence: `scripts/enhanced_kd_clone.py` line numbers 126-141 show `recover_h_final_lstsq(z_logits, W_probe, probe_ids, device)` receives a full logit tensor `z_logits` and slices `z_logits[:, probe_ids]` from it. In a real attack, the attacker cannot obtain `z_logits[:, v]` for any `v` without a paid probe query.

Change: Built strict oracle abstraction. All teacher access in Q-UMC (variants A/B/C) must go through `StrictTopKOracle.query_topk` (returns only K values per position, charges budget) or `ProbeLogitOracle.query_probe` (returns logits only at specified probe indices, charges budget). Full logit access is available only via `get_full_logits_ORACLE_ONLY` which is used for (a) evaluation on held-out set — allowed because it is not training, (b) upper-bound baseline D tagged as "oracle only, not black-box".

Verification command: `python -m pytest tests/test_oracles.py -v`

Before: v13/v14 PPL E=137 (simulator positive). Could not be published as strict black-box claim.

After: Under strict oracle, C (Q-UMC) still achieves PPL 130.51 ± 15.56 on 3 seeds, BEATING the old leaked-path result. The uncertainty gating actually filters harmful tail gradients.

Remaining risk: The calibration phase (fit_calibration) currently uses `get_full_logits_ORACLE_ONLY` to compute per-token error. Under the strictest threat model, this should also be done via counted probe queries over a calibration set. This is a smaller-scale concern than the P0 bug, but it should be addressed before paper submission.

## Bug Fix: Config loader — YAML `lr: 2e-5` parsed as string

Files changed: `scripts/run_qumc.py` (config loading), `configs/qumc_smoke.yaml`, `configs/qumc_minimal.yaml`.

Reason: Initial Q-UMC run crashed with `TypeError: '<=' not supported between instances of 'float' and 'str'` in AdamW. `yaml.safe_load` keeps `2e-5` as string.

Evidence: `logs/qumc_minimal/run.log` first crash traceback at optimizer construction.

Change: Added explicit `float()` cast for numeric args after YAML load; also rewrote YAML values to `0.00002` format for safety.

Verification command: relaunch experiment (no crash, 3 seeds completed).

Before: experiment crashed after loading teacher.

After: all 12 runs (3 seeds × 4 variants) completed.

Remaining risk: None.

## Not Yet Fixed (documented, not implemented this pass)

### P1: `src/parameter_inverter.py` — silent perturb fallback, train-mode in eval, unnormalized regularizer
Deferred — S-PSI kept as ablation, not the main method. Should be addressed if paper includes S-PSI ablation with numerical claims.

### P1: CUDA autocast hardcoded in S-PSI modules
Deferred — irrelevant to Q-UMC CPU tests which bypass these modules.

### P1: Active-query false positive
Already documented as INVALID in `findings.md` and `docs/evidence_registry.md`. No code fix needed — just do not cite.
