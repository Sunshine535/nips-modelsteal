# Paper Claim Audit Report

**Date**: 2026-04-16
**Auditor**: GPT-5.4 xhigh (fresh zero-context thread, threadId 019d9bd1-17f4-7243-ad89-10cb1e359221)
**Paper**: Transformer Tomography: Characterizing Weight Identifiability Limits from Black-Box Logit Access
**Difficulty**: nightmare, effort: beast

## Overall Verdict: **FAIL**

## Claims Verified: 38 total
- **exact_match**: 2
- **rounding_ok**: 24
- **config_mismatch**: 2  ⚠️
- **scope_overclaim**: 1  ⚠️
- **ambiguous_mapping**: 2  ⚠️
- **number_mismatch**: 3  🔴
- **missing_evidence**: 4  🔴
- **non-quantitative mismatch**: 1  🔴 (tied embeddings)

---

## 🔴 Critical Issues (FAIL level)

### Issue 1: `tie_word_embeddings = false` contradicts paper
- **Location**: §6 Setup — "tied embeddings"
- **Paper says**: "Qwen2.5-0.5B ... tied embeddings"
- **Evidence**: `results/v2_random_s42/regime_oracle/init_0/recovered_model/config.json` has `tie_word_embeddings: false`
- **Fix**: This is actually a **detail about the STUDENT**, not the teacher. The student model was created with untied embeddings (so lm_head can be optimized independently). Clarify wording: "Qwen2.5-0.5B uses tied embeddings in the teacher; the student is instantiated with untied lm_head for independent optimization" — OR confirm teacher config.

### Issue 2 (number_mismatch): Warmstart table α=0.7 row
- **Location**: Table 5 (tab:warmstart)
- **Paper says**: post=0.854, Δ=-0.001
- **Evidence**: `results/warmstart_sweep/sweep_results.json` → post=0.85345, Δ=-0.00048
- **Fix**: Change to post=0.853, Δ=-0.000 (or 0.854 Δ=-0.001 if using sample std, but direct numbers from the JSON round to 0.853)

### Issue 3 (number_mismatch): Warmstart "|Δ| ≤ 0.01" claim
- **Location**: Table 5 caption; §7; §10
- **Paper says**: "|Δ| ≤ 0.01" and "|Δcos| < 0.01 at any interpolation level"
- **Evidence**: `results/warmstart_sweep/sweep_results.json` → max|Δ| = 0.01037 at α=0 (random-init row)
- **Fix**: Change to "|Δ| ≤ 0.011" or "|Δ| ≈ 0.01"

### Issue 4 (number_mismatch): Gauge-invariant Table Block 23
- **Location**: Table 9 (tab:gauge-invariant), Block 23 rows
- **Paper says**: `W_V raw=+0.002`, `W_O GL-aligned=-0.001`
- **Evidence**: `results/v5_gauge_eval_random_s42/gauge_invariant_eval.json` → V raw=0.001479 (rounds to +0.001), O GL=-0.000453 (rounds to -0.000)
- **Fix**: Update table entries — W_V raw: +0.001, W_O GL: -0.000

### Issue 5 (missing_evidence): Table 3 Block 22 +boundary row
- **Location**: Table 3 (tab:gramian), row "Block 22 +boundary"
- **Paper says**: σ_max < 10^-12, κ=∞, eff.rank=0
- **Evidence**: **No raw JSON file** contains this measurement
- **Fix**: Either add the raw result file, or add a footnote explaining this was a manual assertion based on oracle hook behavior

### Issue 6 (missing_evidence): Query budget breakdown "1500×72"
- **Location**: App. query-budget text
- **Paper says**: "1500 × 72 = 108,000 queries" for lm_head recovery
- **Evidence**: Total 108,000 is supported, but `1500 × 72` decomposition not in any JSON
- **Fix**: Cite config or remove the specific decomposition ("~108K queries")

### Issue 7 (missing_evidence): Computational cost (§App. Computational Cost)
- **Location**: App. Computational Cost
- **Paper says**: "45-60 minutes", "~24 GB", "~58 GB", "~40 A100-GPU-hours"
- **Evidence**: No profiling/timing JSON provided
- **Fix**: Either add timing log or soften language ("approximately" is already hedged — acceptable)

### Issue 8 (missing_evidence): v1→v2 headline change "0.54 → 0.12"
- **Location**: App. Code Audit, Headline metric change
- **Paper says**: "0.54 (v1) → 0.12 (v2) ... ~0.02 raw/Procrustes delta"
- **Evidence**: v2 endpoint (0.12) is in results. **No v1 raw result file retained**. No Qwen Procrustes-eval file.
- **Fix**: Archive the v1 result from original logs, or mark as "historical value from deprecated v1 run"

---

## ⚠️ Warning-Level Issues

### Issue 9 (scope_overclaim): K×k grid
- **Location**: Abstract, §1, §7, §8
- **Paper says**: "K ∈ {8, 32, 64, 128} and k ∈ {64, 128, 256} ... always has full rank (=k)"
- **Evidence**: Files cover K-sweep at k=128 AND k-sweep at K=8 — **not the full 4×3=12 grid**
- **Fix**: Clarify the cross-section design. Either: (a) add "across tested K and k sweeps" instead of implying full grid, or (b) actually run full 12-cell grid.

### Issue 10 (config_mismatch): §6 Setup "2048 sequences, 128 tokens, 80/20 split"
- **Location**: §6 Setup
- **Paper says**: "2048 sequences ... maximum length 128 tokens ... 80/20 split"
- **Evidence**: v2 primary runs (which back Tables 1/2/4) used `pool_size=4096, max_seq_len=192`. The "2048×128" matches **v3 diagnostics**, not v2.
- **Fix**: Clarify which experiments used 2048×128 vs 4096×192. OR update Table 1 footnote with actual v2 config.

### Issue 11 (config_mismatch): §6 Setup hyperparameters
- **Location**: §6 Setup, Algebraic initialization
- **Paper says**: "k=128 probes ... truncation rank r=64, ridge λ=10^-4, step scale s=1.0"
- **Evidence**: v3 K32 config uses (128, 64, 1e-4, 1.0). **v2 primary runs used (64, 32, 1e-4, 1.0)** — half the probes and rank.
- **Fix**: Separate setup specs for v2 (Tables 1/2/4) vs v3 (Tables 3/Figure 2). State explicitly that the expanded probe count (k=128) was introduced in v3.

### Issue 12 (ambiguous_mapping): Table 4 κ=2.4 citation
- **Location**: Table 4 caption
- **Paper says**: "full-rank sketched Gramian (κ=2.4)"
- **Evidence**: `v3_expanded_K32_s42/gramian_pre_training.json` gives κ≈2.318; `gramian_diagnostic_positions/gramian_rank_diagnostic.json` gives κ=2.4098 at K=32
- **Fix**: Specify which diagnostic the κ=2.4 comes from (the independent diagnostic, not the training artifact)

### Issue 13 (ambiguous_mapping): §5 teacher cache "120GB → 8GB"
- **Location**: §5 Teacher Cache Construction
- **Paper says**: "K_pos=8 reduces storage from ~120GB to ~8GB"
- **Evidence**: With bf16 and N=2048×T=128: exact numbers are 74.2 GB / 4.6 GB. The 120/8 numbers come from a different N×T config not in the result files.
- **Fix**: Either update to 75/5 GB with current config, or clarify this refers to the larger v2 pool (4096×192).

---

## ✅ Strong Matches

24 claims matched raw evidence at standard rounding precision, including:
- All Block 23/22 per-matrix cosines (Table 2, Table 7)
- All v2 main experiment table means/stds (Table 1, Table 5)
- All Gramian eigenvalues/condition numbers (Table 3)
- All functional evaluation KL/Top-k (Table 8)
- Llama cross-arch values (Table 6)
- 2.2× Block 22 vs 23 ratio
- 24× functional improvement
- 8.6× pure-logits improvement

---

## All Claims (Detailed)

| # | Location | Paper Value | Evidence Value | Status |
|---|----------|-------------|---------------|--------|
| 1 | Abs/§1/§7/§8 | K×k grid | partial sweeps only | scope_overclaim |
| 2 | Abs/Tab3/Fig2 | ~2.2× | 2.223/2.212/2.201/2.217 | rounding_ok |
| 3 | Abs/§1/§7/§10 | 0.12–0.14; LN 0.67–0.99 | 0.1211–0.1456; 0.6713/0.9846 | rounding_ok |
| 4 | Abs/§7/§10 | ≥128; 14.9M | 128; 14,912,384 | rounding_ok |
| 5 | Abs/§7/§10 | KL 0.42; 24× | 0.420258; 24.2612× | rounding_ok |
| 6 | §6 Setup | 24/896/4864/14/2/151936 | exact | exact_match |
| 7 | §6 Setup | 2048; 128; 80/20 | v2 uses 4096; 192 | **config_mismatch** |
| 8 | §6 Setup | α=1.0, β=0.1, γ=1e-5, B=500K | exact | exact_match |
| 9 | §6 Setup | k=128, r=64, λ=1e-4 | v2 uses k=64, r=32 | **config_mismatch** |
| 10 | Tab1 Random | 0.122±.003 | 0.121542±0.002672 | rounding_ok |
| 11 | Tab1 AlgClean | 0.121±.001 | 0.121130±0.001178 | rounding_ok |
| 12 | Tab1 AlgAug | 0.120 | 0.119928 | rounding_ok |
| 13 | Tab2 | ≈0; LN 0.671/0.985 | exact | rounding_ok |
| 14 | Tab5 | oracle/pure rows | matches | rounding_ok |
| 15 | §6 Exp2 | 1.19; 8.6× | 1.186467; 8.5936× | rounding_ok |
| 16 | Tab3 B23 | 11462→178067, κ≈2.4 | exact | rounding_ok |
| 17 | Tab3 B22 | 25479→394729, κ≈3.1 | exact | rounding_ok |
| 18 | Fig2 | k=64/128/256, κ 1.9→3.6 | exact | rounding_ok |
| 19 | Tab3 +boundary | <1e-12; ∞; 0 | **not in any JSON** | **missing_evidence** |
| 20 | Tab4 caption | κ=2.4 | 2.318 (training) vs 2.410 (diag) | ambiguous_mapping |
| 21 | Tab4 K=128 | 0.177/0.129/0.145 | exact | rounding_ok |
| 22 | Tab5 α=0.7 | post=0.854, Δ=-0.001 | 0.853, -0.000 | **number_mismatch** |
| 23 | Tab5 caption | |Δ|≤0.01 | max|Δ|=0.0104 | **number_mismatch** |
| 24 | §10 | 0.13→0.42 | 0.138/0.420 | rounding_ok |
| 25 | Tab6 Llama | 0.663/0.038/0.218 | exact | rounding_ok |
| 26 | Tab7 Controls | 0.119/0.127/0.147 | exact | rounding_ok |
| 27 | Tab8 Functional | all values | exact | rounding_ok |
| 28 | §6 Exp4 | 0.42, 24%, 35%, 1.19 | exact | rounding_ok |
| 29 | Tab9 B23 V/O | +0.002/-0.001 | +0.001/-0.000 | **number_mismatch** |
| 30 | §7 | ~87%; 13.1M/14.9M | 87.67% | rounding_ok |
| 31 | §6 Exp4 | 512 sequences; 8 positions | 4096/8=512 | rounding_ok |
| 32 | App Memory | 75/4.7 GB | 74.19/4.64 GB | rounding_ok |
| 33 | §5 Cache | 120→8 GB | not reproducible | ambiguous_mapping |
| 34 | App Tab2 | bias rows | exact | rounding_ok |
| 35 | App Tab budget | 108K/120K/110K | exact | rounding_ok |
| 36 | App budget | 1500×72=108K | 108000 OK, decomposition missing | missing_evidence |
| 37 | App Compute | 45-60min, 40 GPU-h | **no timing JSON** | missing_evidence |
| 38 | App v1→v2 | 0.54→0.12; ~0.02 | v2 OK, v1 missing | missing_evidence |

---

## Recommended Actions (priority order)

### Must fix before submission (number mismatches)
1. **Claim 22**: Tab5 α=0.7 row → post=0.853, Δ=-0.000
2. **Claim 23**: Tab5 caption → "|Δ| ≤ 0.011" or "|Δcos| < 0.011"
3. **Claim 29**: Tab9 Block 23 → W_V raw=+0.001, W_O GL=-0.000

### Should clarify (config/scope mismatches)
4. **Claim 7+9**: Split §6 Setup into v2 setup (for Tab1/2/4/7) and v3 setup (for Tab3/4/Fig2)
5. **Claim 1**: Clarify whether K×k grid is partial (current) or full (requires new runs)
6. **Claim 20**: State which diagnostic gives κ=2.4 for Tab4 caption
7. **Claim 33**: Update §5 cache estimate to match actual v2 config

### Should document (missing evidence)
8. **Claim 19**: Add raw JSON for Block 22 +boundary measurement
9. **Claim 36**: Cite lm_head config, remove "1500×72" decomposition if not backed
10. **Claim 37**: Add a `compute_log.json` with runtimes, or clarify as estimate
11. **Claim 38**: Archive v1 result or mark as "historical"

### Tied embeddings clarification
12. Clarify the tied-embeddings wording in §6 given the student has `tie_word_embeddings: false`

---

## Summary

Paper data is **mostly honest** (24/38 rounding_ok + 2/38 exact = 68%), but has **5 definite errors/omissions** that would fail a strict reviewer audit:

- 3 **number_mismatch** (wrong rounding or out-of-bounds statements)
- 4 **missing_evidence** (numbers not traceable to raw JSON)
- 1 **tied_embeddings** contradiction

Plus 5 softer issues (2 config, 2 ambiguous, 1 scope) that should be clarified.

**Action**: Fix the 3 number_mismatches immediately. The missing_evidence items should be documented or archived. The softer issues need editorial clarification.
