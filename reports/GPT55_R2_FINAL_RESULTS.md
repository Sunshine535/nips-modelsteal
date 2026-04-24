# GPT-5.5 Pro R2 Final Results — Q-UMC Strict v2

**Date**: 2026-04-25
**Experiment**: `results/qumc_minimal_strict_v2/` (3 seeds × 5 variants × 5000 steps)
**Model**: Qwen/Qwen2.5-0.5B | topK=20 | probe_tokens=2000 | basis=carlini_recovered
**Decision**: **CONTINUE with caveats** — C > A on both PPL AND KL across all 3 seeds, but mechanism is primarily noise suppression, not positive tail signal.

## Headline Table (3 seeds, mean ± std)

| Variant | PPL ↓ | KL ↓ | Top-1 ↑ | Status |
|---------|-------|------|---------|--------|
| Teacher | 24.23 | 0.0000 | 100% | oracle |
| A: strict_topk_kd | 411.12 ± 7.11 | 3.294 ± 0.020 | 0.406 ± 0.003 | sparse baseline |
| B: completion_no_unc_strict | 524,561 ± 32,968 | 10.98 ± 0.07 | 0.002 ± 0.000 | **Fails — noisy Carlini tail poisons KD** |
| **C: completion_uncertainty_strict** | **123.90 ± 16.11** | **2.366 ± 0.122** | 0.396 ± 0.012 | **Q-UMC** |
| D: full_logit_upper | 115.03 ± 16.88 | 1.961 ± 0.140 | 0.404 ± 0.016 | oracle upper bound |
| E: old_lc_simulator | 524,561 ± 32,968 | 10.98 ± 0.07 | 0.002 ± 0.000 | (=B under strict basis, see caveat below) |

## Core R2 Verification: C > A on BOTH PPL AND KL

| Seed | A PPL | C PPL | ΔPPL% | A KL | C KL | ΔKL% | C < A both? |
|------|-------|-------|-------|------|------|------|-------------|
| 0 | 417.38 | 120.18 | **71.2%** | 3.307 | 2.352 | **28.9%** | ✅ YES |
| 1 | 414.80 | 145.23 | **65.0%** | 3.309 | 2.522 | **23.8%** | ✅ YES |
| 2 | 401.19 | 106.31 | **73.5%** | 3.266 | 2.225 | **31.9%** | ✅ YES |

**Mean improvement: ΔPPL 69.9% ± 3.6%, ΔKL 28.2% ± 3.3%. ALL 3 seeds.**

Unlike R1 (where C KL was actually worse than B KL — Issue 4 from R2 review), this time C beats A on BOTH metrics consistently. No cherry-picking.

## C vs D (Upper Bound Gap)

| Seed | C PPL | D PPL | PPL gap | C KL | D KL | KL gap |
|------|-------|-------|---------|------|------|--------|
| 0 | 120.18 | 109.00 | +11.18 | 2.352 | 1.918 | +0.434 |
| 1 | 145.23 | 138.06 | +7.17 | 2.522 | 2.150 | +0.372 |
| 2 | 106.31 | 98.05 | +8.25 | 2.225 | 1.814 | +0.411 |

C stays within ~10% PPL of full-logit oracle upper bound D. No overshoot (unlike R1 where C beat D — that was loss-normalization artifact).

## Strict Access Audit (from manifests)

All 3 seeds verify:
- `teacher_wlm_used_as_basis`: **False** (Carlini SVD from 100 top-K queries)
- `calibration_uses_probes_only`: **True** (calibration on disjoint cal split via probe oracle)
- `unified_kl_loss`: **True** (same `sequence_kl_loss` for A/B/C/D)

Query budgets per seed:
- Basis recovery: 100 top-K queries
- Calibration: 20,000 probe queries (0 topK)
- Training per variant: 40,000 topK + 80M probe queries (C/B need probes)
- Total per seed: 120,100 topK + 160M probe queries

## Mechanism Log: Honest Finding

**Important caveat from mechanism logs:**

| Variant | Seed 0 kd/ce ratio (early/late) | Seed 0 tail weight mean |
|---------|-------------------------------|------------------------|
| A: strict_topk_kd | 1.92 / 1.39 | — |
| B: completion_no_unc | 1.20 / 0.64 | (weights=None) |
| **C: completion_uncertainty** | **0.001 / 0.001** | **0.0011** |

The uncertainty weights suppress C's KD loss by ~1000×. C's effective KD signal is essentially zero on the tail — the gate is acting as a **noise suppressor**, not a positive signal gate.

**What this means honestly:**
1. Carlini-recovered tail logits are too noisy (heldout_mse=12464, large)
2. Without uncertainty weighting (B), the noise POISONS KD → 488k PPL catastrophic failure
3. Uncertainty weights learned from disjoint probes correctly identify tail as unreliable → downweight ~1000×
4. C's training becomes effectively: "exact top-K KD + CE on WikiText + essentially-zero tail signal"
5. C > A because A's KD distribution is artificially peaked (non-topK = -1e9, unrealistic), while C's weight-masked KD is better calibrated

**This is a negative-space mechanism**: Q-UMC wins not by injecting positive information from the tail, but by *refusing to inject noise* when the tail recovery is unreliable. This is still a valid and publishable mechanism (it's what enables training to succeed at all in strict black-box), but the narrative differs from R1.

## Known Caveats (honest reporting)

1. **`old_lc_simulator` is redundant with B under strict basis**: both use Carlini-recovered basis, so E ≡ B (both 488k PPL). The intended R1 leaked-path comparison (teacher W_lm basis) is NOT tested here. To truly test "strict Q-UMC beats R1 leaked fragment", need a separate run with `basis_source=teacher_oracle` for old_lc_simulator only.

2. **Uncertainty weight is nearly degenerate** (mean 0.0011, std tiny). This means uncertainty calibration is acting as a binary "top-K vs tail" filter, not as a nuanced per-token gate. The claim "calibrated uncertainty" is weaker than advertised.

3. **Probe budget is huge**: 80M probes per variant. Real API attack would need extensive logit_bias queries (~$10-100 at OpenAI rates × 80M = $800k-$8M). Not economically realistic at this scale.

4. **Single model, single dataset**: only Qwen2.5-0.5B on WikiText-103. Generalization untested.

5. **C's kd_ce_ratio near zero** means CE dominates training. Some of C's improvement is essentially "fine-tune on WikiText while avoiding bad gradients", not "Q-UMC mechanism creates new teacher imitation signal".

## Decision per R2 stop/continue/pivot criteria

Per GPT55 R2 §19 recommended decision rules:
- Continue if: "Full Q-UMC beats strict top-K KD and no-uncertainty completion over ≥3 seeds with valid query accounting" → **YES, met**
- Stop if: "strict completion loses all old advantage after full-logit leakage removal" → **NOT stopping, but advantage is partially from noise suppression, not from positive tail info**
- Pivot if: "uncertainty helps but moment gate does not" → **N/A, moment gate never tested**

**Recommended decision: CONTINUE with reframed narrative.**

The paper should NOT claim:
- "Q-UMC recovers informative dense logits from strict API"
- "Uncertainty gating selects high-confidence tail tokens"

The paper CAN claim:
- "Q-UMC prevents strict completion from catastrophically failing due to gauge-induced tail noise"
- "Without uncertainty gating, Carlini-basis completion (B variant) FAILS catastrophically (488k PPL) while strict top-K baseline stays at 411 PPL"
- "With calibrated uncertainty gating, C achieves 124 PPL (vs teacher 24) — 70% improvement over strict top-K"
- "C approaches full-logit upper bound D (115 PPL), closing most of the strict→oracle gap"

## Files
- Raw results: `results/qumc_minimal_strict_v2/seed_{0,1,2}/results.json`
- Manifests: `results/qumc_minimal_strict_v2/seed_{0,1,2}/manifest.json`
- Training log: `logs/qumc_minimal_strict_v2/run.log` (on remote)

## Next Steps (R3 Priorities)

1. Reframe narrative: Q-UMC = strict-black-box noise suppressor, not positive tail extractor
2. Run old_lc_simulator with `basis_source=teacher_oracle` separately for R1 leaked-path comparison
3. Investigate why uncertainty weights collapse to near-binary (mean=0.0011 is too extreme)
4. Test second model (Llama-3.2-1B) with same protocol
5. Reduce probe budget — 80M/variant is extreme; test smaller probe sets
