# Experiment Progress — Logit Completion Attack

**Date**: 2026-04-24
**Model**: Qwen/Qwen2.5-0.5B (d=896, V=151936, 24 blocks)
**Method**: Logit Completion — recover teacher h_final via logit_bias lstsq, reconstruct full logits, do full-logit KD

## Method Summary

### Threat Model
- Black-box API returns only **top-K logprobs** (K=5 or K=20)
- Attacker has architecture knowledge and public embedding table
- Attacker can use **logit_bias** API parameter to probe individual tokens

### Attack Pipeline
1. **Carlini SVD**: Recover W_lm subspace from ~10k queries (cos 0.9999 subspace)
2. **Per-query h_final recovery**: For each training query, use lstsq(W_probe, z_probe) to recover teacher's final hidden state h_hat (cos 0.977)
3. **Logit Completion**: Reconstruct full teacher logits z_hat = W_lm @ h_hat, merge with exact top-K values
4. **Full-logit KD**: Train student with KL(z_completed, z_student) instead of sparse top-K KD

### Why It Works
- top-K API only provides K logit values out of V=151,936 → sparse supervision
- Logit Completion fills in the remaining V-K logits from h_recovery → dense supervision
- h_recovery quality: cos=0.977, logit reconstruction KL=0.0002 (near-perfect)
- All computation stays in **logit space** — no representation/behavior gradient conflict

## Experiment Results

### v7: Random init, 5k steps, full logits (baseline setup validation)
- **Result**: All variants PPL~2800. Random init + 5k steps = completely insufficient training.
- **Lesson**: Need pretrained init or much more steps.

### v8: Pretrained + noise(σ=0.01), 5k steps, full logits
| Variant | KL | Top-1 | PPL | PPL-deg% |
|---------|-----|-------|-----|----------|
| Teacher | 0.00 | 100% | 24.23 | 0% |
| A: logit-only | 1.92 | 40.7% | 135.40 | 459% |
| B: oracle h MSE (last) | 1.96 | 40.1% | 141.82 | 485% |
| D: SCRD all-pos MSE | 1.91 | 40.7% | 134.75 | 456% |
- **Finding**: With full logit access, MSE on hidden states doesn't help. D≈A.

### v9: Random init, 50k steps, full logits
| Variant | KL | Top-1 | PPL |
|---------|-----|-------|-----|
| A: logit-only | 4.60 | 18.4% | 2035 |
| B: oracle h MSE | 4.58 | 18.2% | 1973 |
- **Finding**: 50k steps still insufficient for 500M model from random init.

### v10: Pretrained + noise(σ=0.1), 5k steps
- **Result**: All variants PPL=10^41. σ=0.1 completely destroys the model.

### v11: Pretrained + noise(σ=0.01), 5k steps, top-K=20
| Variant | KL | Top-1 | PPL | PPL-deg% |
|---------|-----|-------|-----|----------|
| Teacher | 0.00 | 100% | 24.23 | 0% |
| A: top-20 KD only | 2.35 | 40.6% | 178.09 | 635% |
| B: oracle h MSE (last) | 3.69 | 24.6% | 753.15 | 3009% |
| D: SCRD all-pos MSE | 2.33 | 36.7% | 186.59 | 670% |
- **Finding**: MSE on hidden states CATASTROPHICALLY hurts in top-K setting.

### v12: Pretrained + noise(σ=0.01), 5k steps, top-K=5
| Variant | KL | Top-1 | PPL | PPL-deg% |
|---------|-----|-------|-----|----------|
| Teacher | 0.00 | 100% | 24.23 | 0% |
| A: top-5 KD only | 2.42 | 35.4% | 190.04 | 684% |
| B: oracle h MSE (last) | 3.66 | 20.0% | 736.24 | 2938% |
| D: SCRD all-pos MSE | 2.45 | 31.1% | 209.79 | 766% |
- **Finding**: Same pattern. MSE in hidden-state space conflicts with logit KD.

### v13: Pretrained + noise(σ=0.01), 5k steps, top-K=20 + LOGIT COMPLETION
| Variant | KL | Top-1 | PPL | PPL-deg% |
|---------|-----|-------|-----|----------|
| Teacher | 0.00 | 100% | 24.23 | 0% |
| A: top-20 KD only | 2.35 | 40.6% | 178.09 | 635% |
| **E: Logit Completion** | **1.93** | **40.4%** | **137.26** | **467%** |
- **POSITIVE RESULT: E beats A by 22.9% in PPL** (178→137)

### v14: Pretrained + noise(σ=0.01), 5k steps, top-K=5 + LOGIT COMPLETION
| Variant | KL | Top-1 | PPL | PPL-deg% |
|---------|-----|-------|-----|----------|
| Teacher | 0.00 | 100% | 24.23 | 0% |
| A: top-5 KD only | 2.42 | 35.4% | 190.04 | 684% |
| **E: Logit Completion** | **1.92** | **40.6%** | **136.41** | **463%** |
- **POSITIVE RESULT: E beats A by 28.2% in PPL** (190→136)

## Key Insight
Logit Completion converts the logit_bias side-channel information back to **logit space**, avoiding the representation/behavior gradient conflict that killed MSE-based approaches (variants B, C, D). The reconstructed logits have KL=0.0002 quality, making Logit Completion nearly equivalent to having full logit access.

## Root Cause Analysis of MSE Failure
1. **Scale mismatch**: MSE gradient on 896-dim hidden state overwhelms sparse top-K KD gradient
2. **Objective conflict**: MSE optimizes for representation matching; KD optimizes for behavior matching
3. **RMSNorm coupling**: Different hidden state scales can produce identical outputs; MSE penalizes this unnecessarily

## Comparison with SOTA
- Clone 2025: Carlini SVD + KD → 7.31% PPL degradation (6-layer student, different setup)
- Our Logit Completion: 463-467% PPL degradation (same-arch student, pretrained+noise, 5k steps)
- Note: Direct comparison is not apples-to-apples due to different student architectures and training durations. Full baseline comparison pending.

## Next Steps
- [ ] Implement Clone 2025 baseline for fair comparison
- [ ] Multi-model sweep (Llama-3.2-1B, Qwen2.5-1.5B)
- [ ] Longer training (50k steps) with Logit Completion
- [ ] Paper rewrite around Logit Completion as main contribution
