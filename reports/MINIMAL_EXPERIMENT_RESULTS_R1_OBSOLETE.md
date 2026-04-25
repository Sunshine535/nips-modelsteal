# Minimal Experiment Results

> **⚠️ GPT-5.5 Pro R2 RE-VERDICT (2026-04-25): INVALID for strict Q-UMC claim.**
> Tagged as `ORACLE_WLM_AND_CALIBRATION_LEAK_WEAK_SIGNAL`.
> Tasks 2-7 per R2 must be completed before re-running as `qumc_minimal_strict_v2`.


| Experiment | Command | Config | Dataset | Seed | Metric | Result | Expected | Pass/Fail | Interpretation |
|------------|---------|--------|---------|------|--------|--------|----------|-----------|----------------|
| Smoke test | `python scripts/run_qumc.py --num_steps 2 --batch_size 1 --seq_len 16 --topk 5 --probe_tokens 64 --allow_synthetic` | qumc_smoke | synthetic | 42 | crash/complete | No crash, manifest saved | completes | PASS | Code runs end-to-end on GPU |
| Data sanity | N/A (WikiText train/val automatic split) | — | WikiText-103 | — | split names saved | manifest.split_info captures train/validation names | correct | PASS | See `results/qumc_minimal/seed_*/manifest.json` |
| Metric sanity | `pytest tests/test_*.py` | — | toy | — | 11 tests | 11/11 | all pass | PASS | Teacher KL=0 implicit; top-K masks size K; recovery exact on toy |
| Dry run | `run_qumc.py --config configs/qumc_smoke.yaml --dry_run` | smoke | — | 42 | variants printed | 4 variants listed | planned variants shown | PASS | Pre-registration of A/B/C/D |
| Reproduce old positive | `scripts/enhanced_kd_clone.py` (v13/v14) | — | WikiText-103 | 42 | PPL E vs A | 137 vs 178 (v13) | E < A | PASS (historical, contaminated) | Old signal reproducible but leaked |
| Reproduce old negative | v7 random init 5k | — | WikiText-103 | 42 | PPL | 2834 (random) vs 135 (pretrained) | random >> pretrained | PASS | Random KD fails as predicted |
| Strict oracle activation | `pytest tests/test_oracles.py` | — | toy | — | access checks | 5/5 | pass | PASS | Budget enforcement works |
| Completion calibration | Implicit in run_qumc.py cal_batches=20 | minimal | WikiText | 0 | 5000 samples | calibration fit successful | fitted | PASS | Uncertainty weights computed |
| A: strict_topk_kd | `run_qumc.py --config qumc_minimal.yaml --variants strict_topk_kd` | minimal | WikiText-103 | 0,1,2 | PPL mean | 674.25 ± 16.17 | baseline | PASS | Strict top-K baseline established |
| B: completion_no_unc | same, variant=completion_no_unc | minimal | WikiText-103 | 0,1,2 | PPL mean | 161.91 ± 37.40 | > A | PASS | Completion alone gives 76% improvement |
| **C: completion_uncertainty** | same, variant=completion_uncertainty | minimal | WikiText-103 | 0,1,2 | PPL mean | **130.51 ± 15.56** | > B | **PASS** | **Full Q-UMC wins, 80.7% better than A** |
| D: full_logit_upper | same, variant=full_logit_upper | minimal | WikiText-103 | 0,1,2 | PPL mean | 157.37 ± 33.93 | upper bound | PASS (oracle only) | C even surpasses D |
| Multi-seed stability (C) | 3 seeds | minimal | WikiText-103 | 0,1,2 | std / CI | PPL std 15.56, per-seed improvement 78-82% | non-zero improvement | PASS | Stable across seeds |

## Manifests
- `results/qumc_minimal/seed_0/manifest.json` — git hash, command, config, query budget, metrics
- `results/qumc_minimal/seed_1/manifest.json` — same schema
- `results/qumc_minimal/seed_2/manifest.json` — same schema

## Not Yet Run
- Second-model generalization (Llama-3.2-1B or Qwen2.5-1.5B)
- Official baselines (Carlini-only completion, Clone 2025 reproduction)
- Multi-seed 5-seed bootstrap CI
- topK=5 strict comparison (only topK=20 in minimal)
