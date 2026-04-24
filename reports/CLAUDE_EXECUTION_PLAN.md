# Claude Execution Plan

## 1. Diagnosis File Location
`/home/tarkoy/nips/nips-modelsteal/GPT55_DIAGNOSIS.md` (repo root, 890 lines)

## 2. MAIN METHOD PATH
**Q-UMC: Query-budgeted Uncertainty-gated Moment-guided Logit Completion**

Core idea: Under strict top-K + optional logit-bias/probe API, recover h_hat via counted probes, reconstruct calibrated dense logits, merge with exact top-K, weight tail tokens by uncertainty, train student with weighted dense KL. No hidden-state MSE.

## 3. Missing Mechanism
Query-budgeted strict black-box oracle abstraction + calibrated uncertainty weighting on reconstructed tail logits. Current v13/v14 positive results use full teacher logits internally (P0 bug), making the "strict top-K black-box" claim contaminated.

## 4. Evidence Supporting Diagnosis
- v13/v14: Logit Completion E beats top-K KD A by 23-28% PPL → dense tail signal helps
- v8: Full-logit KD achieves PPL 135 → dense supervision is the ceiling
- v13/v14 B/C/D: Hidden-state MSE catastrophically hurts → h-space objective conflicts with behavior
- v7/v9: Random init KD fails → pretrained manifold needed
- S-PSI: sensitivity beta=0 no worse → sensitivity mechanism not useful
- Active query: bug confirmed, cos 0.999 invalid

## 5. Evidence Contradicting/Weakening Diagnosis
- v13/v14 positive result may shrink or disappear once full-logit leakage is removed (the strict oracle is the critical test)
- Moment-CP signal (cos 0.813) lacks raw artifacts in public repo
- Single seed (42) for all positive results

## 6. Files to Inspect
- `scripts/enhanced_kd_clone.py` (current Logit Completion implementation)
- `src/parameter_inverter.py` (S-PSI core, bugs)
- `scripts/reproduce_carlini.py` (Carlini SVD baseline)
- `scripts/attack_higher_order_moments.py` (quarantined Moment-CP)
- `findings.md` (active-query bug record)
- `results/v7-v14/results.json` (all experiment results)
- `tests/test_modelsteal.py` (existing tests)

## 7. Files to Edit
- NEW: `src/oracles.py` (strict top-K + probe oracle)
- NEW: `src/logit_completion.py` (calibrated completer)
- NEW: `src/result_manifest.py` (provenance)
- NEW: `scripts/run_qumc.py` (main training script)
- NEW: `configs/qumc_smoke.yaml`, `configs/qumc_minimal.yaml`
- NEW: `tests/test_oracles.py`, `tests/test_logit_completion.py`, `tests/test_manifest.py`
- NEW: `docs/evidence_registry.md`
- NEW: all `reports/*.md` files

## 8. Files to Archive
- Active-query positive results → `archive/20260424_active_query_invalid/`
- v13/v14 reclassified as "simulator-positive, not strict-black-box"

## 9. Files NOT to Touch
- `results/v7-v14/results.json` (raw data preserved)
- `paper/main.tex` (not updated until experiments pass)
- `scripts/enhanced_kd_clone.py` (kept as historical, not modified)
- `src/parameter_inverter.py` (kept as ablation, bugs documented but not fixed in this pass)

## 10. Tests Before/After
BEFORE: `python -m pytest tests/test_modelsteal.py -q` (existing)
AFTER: full test suite including test_oracles, test_logit_completion, test_manifest, smoke run

## 11. Rollback Conditions
- If strict oracle completion fails to beat strict top-K KD over 3 seeds → STOP, report to GPT-5.5
- If test infrastructure cannot run on CPU (no local GPU) → prepare commands only, flag as NEEDS_GPU
- If Moment-CP artifacts cannot be located → proceed without moment gate, set g_v=1
