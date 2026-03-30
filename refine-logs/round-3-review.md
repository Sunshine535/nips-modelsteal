# Round 3 Review (GPT-5.4)

**Overall Score**: 7.1/10
**Verdict**: REVISE (close to READY)

## Scores

| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 8.5 |
| Method Specificity | 8.0 |
| Contribution Quality | 6.0 |
| Frontier Leverage | 7.0 |
| Feasibility | 5.5 |
| Validation Focus | 8.0 |
| Venue Readiness | 6.5 |

## Remaining Issues

### P0: Contribution Quality (6.0)
- Oracle regime STILL confounded: optimizing B_N uses student-produced h_{N-1}, not teacher's
- Fix: Inject exact teacher boundary states h_{i-1} for each block in oracle setting

### P0: Feasibility (5.5)
- Perturbation objective too expensive for budget
- Fix: Precompute teacher clean/perturbed logits, reduce perturbation sampling

### P1: Venue Readiness (6.5)
- Headline claim still overreaches protocol
- Fix: One crisp question — what is recoverable for last 1-2 blocks under exact boundary state
