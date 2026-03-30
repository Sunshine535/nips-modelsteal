# Round 1 Review (GPT-5.4)

**Overall Score**: 5.7/10
**Verdict**: RETHINK

## Scores

| Dimension | Score | Assessment |
|-----------|-------|------------|
| Problem Fidelity | 6/10 | Success criterion allows failing on deep-layer recovery |
| Method Specificity | 5/10 | Block-coordinate distillation, not true inversion |
| Contribution Quality | 6/10 | Diluted by query-selection variants, scaling-law, defenses |
| Frontier Leverage | 7/10 | Correct modern setting |
| Feasibility | 4/10 | Full progressive recovery of 4B from random init not credible |
| Validation Focus | 6/10 | Missing identifiability controls |
| Venue Readiness | 5/10 | Too incremental for top venue |

## Key Criticisms

### CRITICAL
1. **Method is staged distillation, not inversion**: Need a block-local inverse operator, not just MSE + freeze
2. **Feasibility**: 4B model full recovery impossible with 570 GPU-hours from random init
3. **Problem Fidelity**: Success condition lets paper "succeed" without solving the core problem

### IMPORTANT
4. **Contribution sprawl**: Too many parallel contributions (inversion, query selection, scaling law, defenses)
5. **Missing identifiability controls**: Need multiple random inits, symmetry-aware alignment
6. **Venue readiness**: Need sharper claim about when/why recovery is possible

## Concrete Fixes Required
- Pre-specify contiguous suffix recovery target (lm_head + last k blocks)
- Use smaller model as proof-of-concept; 4B only for limited stress test
- Delete KD warm-start (hurts identifiability)
- Merge query selection to one method + random baseline
- Replace defenses with identifiability controls
- Use alternating least squares or second-order steps for block recovery

## Simplification Opportunities
- Delete KD warm-start
- Merge query selection to one method
- Remove defenses from main paper

## Modernization Opportunities
- Token-sequence query synthesis instead of static pool
- Multi-position logit traces from rollouts
- Delta-recovery around public base (re-anchoring risk)

<details>
<summary>Raw Review</summary>

[Full verbatim response saved above]

</details>
