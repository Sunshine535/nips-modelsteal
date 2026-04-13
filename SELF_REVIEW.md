# Self-Review: Transformer Tomography

## Summary
This paper presents "Transformer Tomography," a framework for understanding weight identifiability in transformers from black-box logit access. The main contribution is the suffix observability Gramian on gauge-quotiented parameter space, which precisely characterizes identifiable parameter directions. The key empirical finding is a strong negative identifiability result: internal transformer weights are not recoverable from logit access.

## Strengths

1. **Novel theoretical framework**: The suffix observability Gramian adapted from control theory, combined with the complete continuous symmetry group for modern transformers (RMSNorm + gated MLP), is a genuinely new contribution.

2. **Strong negative result with rigorous evidence**: The zero-Gramian result for Block 22 (confirmed across 3 seeds) is the strongest possible negative identifiability statement. This is more valuable than a weak positive result.

3. **Validated methodology**: Gauss-Newton produces actual loss decreases (ratio > 1), confirming the Gramian framework works as predicted. The negative result is NOT due to methodology failure.

4. **Comprehensive controls**: β=0, wrong teacher, no gauge projection all yield cos ≈ 0, ruling out artifacts.

5. **Sensitivity augmentation analysis**: Showing that sensitivity matching doesn't expand the observable subspace is a clean additional finding.

6. **Practical implications**: Model providers can compute the Gramian to assess API vulnerability.

## Weaknesses

1. **Single model scale**: Only Qwen2.5-0.5B (0.5B params). Reviewers may ask: does the Gramian rank scale with model size?

2. **Oracle regime is idealized**: The boundary-state oracle doesn't exist in practice. While useful as an upper bound, practical attacks must contend with prefix errors.

3. **alg_aug only 1 seed**: Due to computational cost (9x slower Gramian), we only ran 1 seed for the sensitivity-augmented variant.

4. **No learning-based recovery**: We only tried gradient-based and algebraic recovery. A reviewer might ask about meta-learning or RL-based approaches.

5. **Sketch probe count**: k=64 probes. While rank-32 is consistent across seeds, higher k would be more convincing.

6. **No comparison to KD baseline**: We don't explicitly compare to knowledge distillation in functional equivalence metrics (KL divergence, accuracy match).

## Missing Experiments
- [ ] alg_aug Block 22 data (running, expected ~07:31)
- [ ] Larger model validation (even 1.5B would help)
- [ ] KD baseline functional equivalence comparison

## Key Numbers
- Corrected random: lm_head=0.536±0.006, b23=+0.000±0.001, b22=+0.009±0.015
- Alg clean: lm_head=0.540±0.002, b23=-0.000±0.008, b22=+0.001±0.019
- Gramian clean: σ_max=12.2±1.3, rank=32, cond=60.8±5.8
- Gramian aug: σ_max=11.2, rank=32, cond=54.5
- Block 22 Gramian: IDENTICALLY ZERO (all seeds)
