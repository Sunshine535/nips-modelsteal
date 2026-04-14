# Research Findings Log

## 2026-04-14: V4 Experiment Results Diagnosis

### Active Query Experiment — BUG CONFIRMED
- **Phenomenon**: All 3 conditions (random, active, full_pool) show cos ≈ 0.999 for block 23
- **Root cause**: `run_active_query.py` lines 339, 364, 411 — student initialized via `AutoModelForCausalLM.from_pretrained(args.model_name)`, loading the SAME pretrained Qwen2.5-0.5B weights as teacher. Student starts at cos ≈ 1.0, KD merely fine-tunes around that.
- **Verdict**: Data invalid. Cannot be used in any claims or review submission.
- **Action needed**: Fix script to use random initialization, re-run.

### KD Suffix Baseline (Qwen) — VALID
- Student randomly initialized (3 seeds: 42, 1042, 2042)
- Block 22 cos: 0.129 ± 0.013, Block 23 cos: 0.134 ± 0.018
- lm_head aligned cos: 0.063
- Confirms negative result: suffix KD from random init does NOT recover parameters

### Llama S-PSI — VALID
- meta-llama/Llama-3.2-1B, oracle regime, 2 suffix blocks
- Block 14 cos: 0.218, Block 15 cos: 0.218
- lm_head aligned cos: 0.663 (Procrustes)
- Shows partial lm_head recovery, near-zero block recovery — consistent with Qwen results

### Gramian Statistics (from active query run, still valid)
- σ_max: 122,750, σ_min: 5,036, condition number: 24.4, effective rank: 47.9
- Healthy Gramian spectra despite observability-recoverability gap

## Proof Checker Completion (2026-04-14)
- 3 rounds, nightmare difficulty, score: 7.5/10
- All 5 FATAL + 5 CRITICAL + 8 MAJOR + 3 MINOR issues fixed
- Acceptance gate PASSED
- Key changes: SiLU symmetry corrected, attention V/O per KV group, sequence-level Jacobians, sketched Gramian language
