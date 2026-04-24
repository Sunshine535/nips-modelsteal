# Paper Claim Audit Report — Round 2

**Date**: 2026-04-18
**Auditor**: GPT-5.4 xhigh (fresh zero-context thread, threadId 019d9e85-7065-7591-a938-57ccdeffb074)
**Paper**: Transformer Tomography: Characterizing Weight Identifiability Limits from Black-Box Logit Access
**Difficulty**: nightmare, effort: beast

## Overall Verdict: **FAIL**

## Delta vs Round 1 (2026-04-16)

| Metric | Round 1 | Round 2 |
|--------|---------|---------|
| Claims audited | 38 | 40 |
| exact_match | 2 | 2 |
| rounding_ok | 24 | 22 |
| number_mismatch | 3 | **7** (4 new) |
| missing_evidence | 4 | 3 |
| config_mismatch | 2 | 1 |
| scope_overclaim | 1 | 2 |
| ambiguous_mapping | 2 | 2 |
| unsupported_claim | 0 | 1 |
| non-quantitative (tied) | 1 | **0** ✓ fixed |

### ✅ Fixed since R1
- **Claim 17 (Table 5 α=0.7)**: 0.854/0.854/-0.001 → 0.854/0.853/-0.000 ✓
- **Claim 18 (|Δ| ≤ 0.011 claim)**: updated in paper, now exact_match ✓
- **Claim 29 (Table 9 Block 23 W_V/W_O)**: paper now correctly shows +0.001/-0.000 ✓
- **Tied embeddings**: clarified teacher tied / student untied ✓

---

## 🔴 New number_mismatch Issues (4 found in R2)

### Issue 22 (NEW): Table 8 Random init oracle KL std
- **Location**: Table 8 `Random init (oracle)`
- **Paper says**: KL = 0.420 ± **.015**
- **Evidence**: `results/functional_eval_v2/functional_metrics.json` → sample std = 0.014353, pop std = 0.011719
- **Fix**: Change to `.014` (sample std) or `.012` (pop std). `0.015` rounds neither.

### Issue 23 (NEW): Table 8 Alg init clean KL std
- **Location**: Table 8 `Alg. init clean (oracle)`
- **Paper says**: KL = 0.412 ± **.014**
- **Evidence**: sample std = 0.013266, pop std = 0.010831
- **Fix**: Change to `.013` or `.011`. `0.014` matches neither precision.

### Issue 25 (NEW): Table 8 Pure logits KL std
- **Location**: Table 8 `Pure logits`
- **Paper says**: KL = 1.186 ± **.138**
- **Evidence**: sample std = 0.140406, pop std = 0.121595
- **Fix**: Change to `.140` or `.122`. `0.138` rounds neither.

### Issue 28 (NEW): Table 9 Block 23 W_K Δ
- **Location**: Table 9 (gauge-invariant), Block 23 W_K row
- **Paper says**: Δ = `-0.003`
- **Evidence**: raw=0.001601, aligned=-0.000525 → exact Δ = **-0.002126**
- **Fix**: Change to `-0.002`

### Issue 32 (NEW): Query budget range in App
- **Location**: App. Query Budget prose
- **Paper says**: "approximately 100K--130K queries"
- **Evidence**: Random block23=120,480; Alg clean block23 = **162,960** (above 130K)
- **Fix**: Update to "100K-165K" or "100K-160K" to include Alg clean Block 23

### Issue 34 (NEW): §5 Teacher Cache memory
- **Location**: §5 Teacher Cache Construction
- **Paper says**: "K_pos=8 reduces storage from ~120GB to ~8GB"
- **Evidence**: With saved configs: 2048×192 → 119.5GB / 4.98GB; 4096×192 → 239GB / 9.96GB; 2048×128 → 79.66GB / 4.98GB
- **Fix**: 120GB matches 2048×192 config for V=151936 bf16. But 120→5GB, not 120→8GB. Either say "~120GB → ~5GB" or "~80GB → ~5GB". Current `~8GB` is off.

### Issue 36 (NEW): App per-matrix bias discussion
- **Location**: App. per-matrix cosine discussion
- **Paper says**: "attention biases show slightly elevated absolute cosine (~0.05--0.09), but these are **low-dimensional vectors (896 parameters each)**"
- **Evidence**: q_proj.bias has 896 params, but k_proj.bias has **128**, v_proj.bias has **128** (GQA). Cosines are q≈0.034-0.044, k≈0.001-0.023, v≈0.046-0.054.
- **Fix**: (a) Update "896 parameters each" to "896 for Q, 128 for K and V (GQA)"; (b) cosine range should be "~0.001-0.055" not "0.05-0.09"

---

## ⚠️ Persistent Issues (unchanged from R1)

### Claim 2 (config_mismatch)
**Location**: §6 Setup "2048 sequences × 128 tokens"  
**Evidence**: v2 primary runs used `pool_size=4096, max_seq_len=192`  
**Fix**: Split Setup into v2 (4096×192) and v3 diagnostics (2048×128)

### Claim 4 + 14 (scope_overclaim)
**Location**: Abstract, §1, §7, §8, Table 3 caption  
**Paper says**: "K ∈ {8,32,64,128} and k ∈ {64,128,256}" (12-cell grid)  
**Evidence**: Only K-sweep at k=128 and k-sweep at K=8 saved (7 runs, not 12)  
**Fix**: Say "K-sweep with k=128 and k-sweep with K=8" OR run full 4×3=12 grid

### Claim 13 (missing_evidence)
**Location**: Table 3 Block 22 +boundary row  
**Paper says**: σ<10^-12, κ=∞, rank=0  
**Evidence**: No raw JSON  
**Fix**: Add raw result file or footnote that this is a manual observation

### Claims 38 / 39 / 40 (missing_evidence / unsupported)
- §6 no-gauge ablation eigenspectrum
- §7 sensitivity-augmented Gramian comparison
- App. Computational Cost (runtimes, GPU-hours, memory)
- **Fix**: Add supporting JSON artifacts or mark as narrative estimates

### Claim 30 (ambiguous_mapping)
**Location**: §7 "σ1 ≈ 0.68, σ2 ≈ 0.62"  
**Evidence**: block23 group 0 has σ1=1.070/σ2=0.602; block22 group0 has σ1=0.676/σ2=0.622  
**Fix**: Specify which block/group is being reported, or give all 4 groups' top σ

---

## 📋 Priority Fix Order

### Must fix (trivial — wrong numbers)
1. **Claim 22, 23, 25**: Update Table 8 KL standard deviations to match actual population/sample std
2. **Claim 28**: Table 9 Block 23 W_K Δ → -0.002
3. **Claim 32**: Query budget range → 100K-165K
4. **Claim 34**: Teacher Cache memory → "~120GB → ~5GB" or "~80GB → ~5GB"
5. **Claim 36**: App per-matrix bias discussion — fix param counts (896 Q, 128 K/V) and cosine range

### Should clarify (moderate — scope or config)
6. **Claim 2**: Split §6 Setup between v2 primary and v3 diagnostic configs
7. **Claims 4 + 14**: Narrow K×k grid claim to actual sweep shape (K at k=128 + k at K=8)
8. **Claim 30**: Specify which group's σ values the paper reports

### Should document (soft — missing evidence)
9. **Claim 13, 38, 39, 40**: Either add raw files or reframe as narrative estimates

---

## Summary

Overall the paper is **mostly honest** (24/40 = 60% exact+rounding_ok), but has **7 concrete number mismatches** that would fail rigorous reviewer review. Key issue: standard deviations in Table 8 don't match any of {sample_std, pop_std} for the recorded seed counts.

The paper made clear progress R1→R2: 3 of the 4 R1 number_mismatches were fixed, tied_embeddings clarified. New R2 issues are mostly in areas not previously checked (Table 8 stds, Appendix prose).

**Action**: Fix 5 priority "must fix" items (15-min edits). Document 4 missing-evidence items (decision: add raw files OR soften language).
