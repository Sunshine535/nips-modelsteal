# Core Comparison: Q-UMC Strict v2 (R2-compliant, R3-honest)

> **Active report.** Supersedes `CORE_COMPARISON_R1_OBSOLETE.md`.
> Per GPT-5.5 R3 verdict: TRUST ONLY AS WEAK SIGNAL.

## Headline (3 seeds, mean ± std)

| Variant | Access | PPL ↓ | KL ↓ | Top-1 ↑ |
|---------|--------|-------|------|---------|
| Teacher | oracle | 24.23 | 0.000 | 1.000 |
| A: strict_topk_kd | strict | 411.12 ± 7.11 | 3.294 ± 0.020 | 0.406 ± 0.003 |
| B: completion_no_unc_strict | strict | 524,561 ± 32,968 | 10.98 ± 0.07 | 0.002 |
| **C: completion_uncertainty_strict** | **strict** | **123.90 ± 16.11** | **2.366 ± 0.122** | **0.396 ± 0.012** |
| D: full_logit_upper | oracle | 115.03 ± 16.88 | 1.961 ± 0.140 | 0.404 ± 0.016 |
| E: old_lc_simulator | strict (=B) | 524,561 ± 32,968 | 10.98 ± 0.07 | 0.002 |

## Verified Claims

1. **C beats A on BOTH PPL AND KL across all 3 seeds** (no cherry-picking, R2 Issue 4 resolved).
   - ΔPPL 69.9% ± 3.6%, ΔKL 28.2% ± 3.3%, ALL seeds.

2. **C remains below D upper bound** (no overshoot — R1 artifact gone after unified KL loss):
   - PPL gap: ~9 (C 123.90 vs D 115.03)
   - KL gap: ~0.4 (C 2.37 vs D 1.96)

3. **B catastrophically fails (488k PPL, KL 11)** — without uncertainty gating, Carlini-basis tail noise poisons KD.

4. **Strict access verified per manifest**:
   - `teacher_wlm_used_as_basis = false`
   - `calibration_uses_probes_only = true`
   - `unified_kl_loss = true`

## Honest Caveats (per R3 review)

### Caveat 1: Mechanism is noise suppression, not positive extraction
Mechanism log shows C's `kd_ce_ratio ≈ 0.001` (vs A's ~1.9). Uncertainty weights have mean 0.0011 on tail. The gate is essentially **a near-binary "top-K vs tail-off" filter**, not a calibrated continuous gate.

C's effective KD signal comes only from top-K (where weight=1.0), and the tail (V-K ≈ 151,916 tokens) gets ~zero KD gradient. Training is dominated by CE on WikiText.

**This means**: C wins not by extracting more tail information, but by **refusing to inject noisy Carlini-tail gradients** that would otherwise destroy training (see B failure).

### Caveat 2: `old_lc_simulator` (E) is redundant with B under strict basis
Both use Carlini-recovered W_eff. The intended R1 leaked-path comparison (with `basis_source=teacher_oracle`) was NOT tested. Cannot claim "strict Q-UMC beats R1 leaked fragment" without it.

### Caveat 3: Missing baselines block strict mechanism attribution
We have NOT yet tested:
- `ce_only` — train with CE only on WikiText, no KD signal
- `ce_plus_strict_topk` — same as A but with explicit CE term
- `strict_topk_weighted_tailzero` — KD on top-K only (no -inf masking artifact)
- `completion_uncertainty_normed_kl` — normalized weighted KL (preserves total KD mass)

**Without these controls, we cannot say C > A is due to Q-UMC mechanism vs CE fine-tuning.**

### Caveat 4: Probe budget is huge
80M probe queries per variant. At OpenAI logit_bias rates this is $800k-$8M per training run. Real-world API attack at this scale is not credible without budget reduction.

### Caveat 5: Single model, single dataset
Only Qwen2.5-0.5B on WikiText-103.

## Decision Gate

| Path | Status |
|------|--------|
| Continue with caveats | ⚠️ |
| Full benchmark | ❌ BLOCKED until Tasks 2-7 from R3 pass |
| SOTA / paper update | ❌ BLOCKED |
| Pivot narrative | ⚠️ Required if v3 shows C ≈ ce_only |

## What This Comparison Supports

- **CAN claim**: "Uncertainty gating prevents Carlini-basis completion from catastrophic failure (488k → 124 PPL)."
- **CAN claim**: "C improves over strict top-K KD on both PPL and KL across 3 seeds."
- **CANNOT claim** (until v3): "Q-UMC recovers useful dense logit information from strict API."
- **CANNOT claim** (until v3): "Calibrated uncertainty selects high-confidence tail tokens."
- **CANNOT claim**: "C beats old leaked-path positive fragment" (E was not properly tested).
