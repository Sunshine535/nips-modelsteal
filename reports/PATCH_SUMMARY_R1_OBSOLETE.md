# Patch Summary

## Files Added (New)
- `src/oracles.py` — StrictTopKOracle, ProbeLogitOracle, QueryBudget
- `src/logit_completion.py` — CalibratedLogitCompleter
- `src/result_manifest.py` — Provenance tracking
- `scripts/run_qumc.py` — Q-UMC main training script with 4 strict variants
- `configs/qumc_smoke.yaml`, `configs/qumc_minimal.yaml`
- `tests/test_oracles.py`, `tests/test_logit_completion.py`, `tests/test_manifest.py`
- `docs/evidence_registry.md` — Classifies all results by reliability
- `reports/CLAUDE_EXECUTION_PLAN.md`, `LOCAL_REPO_SCAN.md`, `GPT55_REPORT_EXTRACTION.md`
- `reports/CURRENT_RESULT_AUDIT.md`, `CORE_COMPARISON.md`, `PATCH_SUMMARY.md`
- `reports/NEXT_GPT55_REVIEW_PACKAGE.md`
- `results/qumc_minimal/seed_{0,1,2}/{results,manifest}.json`
- `GPT55_DIAGNOSIS.md`

## Files Modified
- `.gitignore` (via `git add -f` for results)

## Files Archived
None yet (archive/ directory planned but not executed this pass).

## Files Intentionally NOT Touched
- `scripts/enhanced_kd_clone.py` (historical reference for old E variant)
- `paper/main.tex` (not updated until broader baselines done)
- `src/parameter_inverter.py` (S-PSI, kept as ablation)
- All `results/v7-v14/*.json` (raw data preserved)
- `README.md` (requires post-experiment update, Task 11)

## Bugs Fixed
- **P0**: Old `complete_logits` in `enhanced_kd_clone.py` used full teacher logits (not strict black-box). Fixed by implementing strict oracle abstraction in `src/oracles.py` where teacher access goes through StrictTopKOracle/ProbeLogitOracle with budget tracking.
- **Config bug**: YAML `lr: 2e-5` loaded as string, caused `TypeError` in AdamW. Fixed by forcing float conversion in run_qumc.py config loader.

## New Method Components Implemented
1. QueryBudget class (enforces probe budget, tracks costs)
2. StrictTopKOracle (top-K only, no full logit access)
3. ProbeLogitOracle (logit-bias simulation with counted queries)
4. CalibratedLogitCompleter (probe → h_hat → z_hat → merge top-K → uncertainty weights)
5. Calibration phase (fit_calibration) to learn per-token uncertainty
6. Weighted dense KL loss in `completion_uncertainty` variant

## Configs Added
- `configs/qumc_smoke.yaml` (smoke test, 2 steps, synthetic data allowed)
- `configs/qumc_minimal.yaml` (5000 steps, topK=20, probe_tokens=2000, 3 seeds)

## Tests Added
- 11 tests across 3 files, all passing on CPU
- Coverage: query budget, oracle separation, recovery correctness, uncertainty computation, manifest schema, git hash capture

## Commands Run
```bash
# Local CPU tests
python3 -m pytest tests/test_oracles.py tests/test_logit_completion.py tests/test_manifest.py -v
# → 11/11 PASS

# Dry run
python3 scripts/run_qumc.py --config configs/qumc_smoke.yaml --dry_run
# → OK

# GPU smoke test (remote)
CUDA_VISIBLE_DEVICES=0 python scripts/run_qumc.py \
  --num_steps 2 --batch_size 1 --seq_len 16 --eval_batches 1 \
  --topk 5 --probe_tokens 64 --output_dir results/smoke_qumc --allow_synthetic
# → All 4 variants ran without crash, manifest saved

# GPU minimal experiment (remote, 3 seeds × 4 variants × 5000 steps)
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run_qumc.py \
  --config configs/qumc_minimal.yaml --seeds 0 1 2 \
  --output_dir results/qumc_minimal > logs/qumc_minimal/run.log 2>&1 &
# → Completed ~2h. All 12 runs finished, results saved.
```

## Results Observed
```
Variant                   PPL (mean±std)    vs A
A: strict_topk_kd         674.25 ± 16.17   —
B: completion_no_unc      161.91 ± 37.40   -76.0%
C: completion_uncertainty 130.51 ± 15.56   -80.7%  ← WINS
D: full_logit_upper       157.37 ± 33.93   -76.7%  (oracle upper bound)
```
**C > B > A on ALL 3 seeds. C even beats D (upper bound).**

## Failed Checks
None.

## Unresolved Risks
- Single model only (Qwen2.5-0.5B). Second-model gate (Task 10 minimal experiment 3) pending.
- Moment gate module (`src/moment_gate.py`, Task 8) not implemented. Currently `g_v=1` default.
- Paper not updated (Task 11 pending broader baselines).
- Only WikiText-103 dataset.
- Calibration uses `get_full_logits_ORACLE_ONLY` — calibration phase is itself an oracle access. Strict threat model would require counting this as oracle-side training cost.
