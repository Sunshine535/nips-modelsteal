# Next GPT-5.5 Pro Review Package

## Summary of Changes

Per GPT55_DIAGNOSIS.md, implemented Q-UMC (Query-budgeted Uncertainty-gated Logit Completion):

### New Modules (all tested, 11/11 pass on CPU)
1. `src/oracles.py` — StrictTopKOracle, ProbeLogitOracle, QueryBudget
   - Hard prohibition on full teacher logit access in training path
   - Every probe query counted against budget
   - Budget enforcement with RuntimeError on overflow
2. `src/logit_completion.py` — CalibratedLogitCompleter
   - Probe-based h_hat recovery via regularized lstsq
   - Dense logit reconstruction from h_hat
   - Calibration-based per-token uncertainty weights
   - Exact top-K merge preserving API-provided values
3. `src/result_manifest.py` — Provenance tracking
   - Git hash, seed, config, query budget, metrics, timestamps
4. `scripts/run_qumc.py` — Main Q-UMC training script
   - 4 strict variants: strict_topk_kd, completion_no_unc, completion_uncertainty, full_logit_upper
   - All teacher access through strict oracles (except tagged upper-bound)
   - Multi-seed support, config file support, dry-run mode
5. `docs/evidence_registry.md` — Classifies all results by reliability
6. `configs/qumc_smoke.yaml`, `configs/qumc_minimal.yaml`

### P0 Bug Fix
Old `complete_logits` in `enhanced_kd_clone.py` received full teacher logits and extracted probe logits from them — NOT strict top-K black-box. New Q-UMC enforces strict oracle separation.

## Git Diff Summary
15 files added, 2124 insertions. Commit `b558b43`.

## Commands Run
```
python3 -m pytest tests/test_oracles.py tests/test_logit_completion.py tests/test_manifest.py -v  → 11/11 PASS
python3 scripts/run_qumc.py --config configs/qumc_smoke.yaml --dry_run  → OK
```

## Commands Pending (require GPU)
```bash
# On remote server after git pull:
# Smoke test
CUDA_VISIBLE_DEVICES=0 python scripts/run_qumc.py \
  --config configs/qumc_smoke.yaml \
  --output_dir results/smoke_qumc --allow_synthetic

# Minimal A/B/C comparison (3 seeds)
CUDA_VISIBLE_DEVICES=0 python scripts/run_qumc.py \
  --config configs/qumc_minimal.yaml \
  --seeds 0 1 2 \
  --output_dir results/qumc_minimal
```

## Result Tables
No GPU results yet — SSH to remote server timed out.

### Historical results (from old contaminated path):
| Setting | A (topK-only) PPL | E (old LC, contaminated) PPL | Improvement |
|---------|-------------------|------------------------------|-------------|
| topK=20 | 178.09 | 137.26 | -22.9% |
| topK=5  | 190.04 | 136.41 | -28.2% |

These results are SIMULATOR-POSITIVE (full-logit leakage, see evidence_registry.md).
The strict Q-UMC comparison is the critical next experiment.

## Mechanism Logs
Not yet available (require GPU run).

## Failed Tests
None — 11/11 pass.

## Unresolved Questions
1. **Critical**: Does strict oracle completion still beat strict top-K KD after removing full-logit leakage? This is the core falsification test.
2. Does calibrated uncertainty weighting (variant C) beat unweighted completion (variant B)?
3. Single-seed instability: all prior results are seed=42 only.
4. Moment-CP artifacts incomplete — moment gate not activated yet.

## Whether Results Support Original Diagnosis
YES for all mechanism-level predictions:
- Dense tail supervision helps (v13/v14 signal)
- Hidden-state MSE hurts (v11/v12 B catastrophic)
- Random KD insufficient (v7/v9)
- Pretrained manifold needed (v8 vs v7)
- Full-logit leakage in old completion (P0 bug confirmed in code)

NOT YET TESTED: Whether the signal survives strict oracle removal.

## What GPT-5.5 Pro Should Review Next
1. Code review of `src/oracles.py` — is the strict separation actually enforced?
2. Code review of `src/logit_completion.py` — is the uncertainty calibration principled?
3. After GPU results: `results/qumc_minimal/` A/B/C comparison tables
4. If C > A and C > B over 3 seeds AND under strict oracle with unified loss and disjoint calibration split → method validated (NOTE: R1 run failed this test; see R2 response)
5. If C ≤ A → method fails, prepare negative report
