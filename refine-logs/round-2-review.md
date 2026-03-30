# Round 2 Review (GPT-5.4)

**Overall Score**: 6.0/10
**Verdict**: REVISE (up from RETHINK)

## Scores

| Dimension | Score | Assessment |
|-----------|-------|------------|
| Problem Fidelity | 6/10 | Not yet isolating true suffix recovery from prefix nuisance |
| Method Specificity | 5/10 | Block-input identifiability undefined — what supplies h_{i-1}? |
| Contribution Quality | 6/10 | Better focus, but novelty still depends on sensitivity > distillation claim |
| Frontier Leverage | 7/10 | Appropriate |
| Feasibility | 6/10 | lm_head + 1-2 blocks plausible; deeper ambitious |
| Validation Focus | 6/10 | Missing oracle-prefix control and known-symmetry null tests |
| Venue Readiness | 6/10 | Claiming slightly more than design supports |

## Key P0 Issues
1. **Block-input identifiability**: When optimizing block B_i, input comes from STUDENT's unrecovered prefix, not teacher's. Need to address this explicitly.
2. **Method data flow**: Must specify what supplies h_{i-1}, whether prefix is frozen/latent/joint, perturbation distribution.
3. **Validation**: Need oracle-prefix vs pure-logits-only control, known-symmetry null tests.

## Key P1 Issues
4. **Contribution**: Make explicit identifiability claim under fixed suffix + stated gauge
5. **Scope**: Narrow headline to lm_head + last 2 blocks
6. **Venue readiness**: Title/abstract must match demonstrable regime

## Simplification
- Remove active query selection from core contribution
- Choose one lm_head recovery path
- Drop 4B stress test from main story
