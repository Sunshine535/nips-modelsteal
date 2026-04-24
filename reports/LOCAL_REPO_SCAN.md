# Local Repository Scan

## Top-Level Directory Map
```
nips-modelsteal/
├── src/           # Core modules (parameter_inverter, gramian, symmetry, algebraic_init, active_query)
├── scripts/       # ~45 experiment/attack/eval scripts
├── configs/       # S-PSI YAML configs (inversion_config.yaml + server variants)
├── results/       # v1-v14 experiment JSON outputs
├── logs/          # v7-v14 training logs
├── paper/         # main.tex, references.bib, main_before_prune.tex
├── tests/         # test_modelsteal.py (minimal)
├── review-stage/  # AUTO_REVIEW.md, REVIEW_STATE.json, Oracle review rounds
├── refine-logs/   # FINAL_PROPOSAL.md, EXPERIMENT_PLAN.md, review rounds
├── idea-stage/    # IDEA_REPORT.md
├── baselines/     # baseline reference scripts
├── research-wiki/ # literature notes
├── reports/       # (NEW) execution reports
├── docs/          # (to be created) evidence registry
├── templates/     # research brief template
├── skills/        # ARIS skill definitions
└── archive/       # (to be created) archived invalid experiments
```

## Component Table

| Component | Path | Purpose | Importance | Notes |
|----------|------|---------|------------|-------|
| S-PSI core | `src/parameter_inverter.py` | Teacher cache, sensitivity loss, progressive block inversion | High (ablation) | P1 bugs: silent perturb fallback, train/eval mode, unnormalized reg |
| Active query | `src/active_query.py` | Query selection for recovery | Low | Bug confirmed in findings.md, results INVALID |
| Gramian | `src/gramian.py` | Observability Gramian computation | Medium | Useful diagnostic, raw logs incomplete |
| Algebraic init | `src/algebraic_init.py` | Linearized initialization | Low | Marginal over random, keep as ablation |
| Symmetry/gauge | `src/symmetry_gauge.py` | RMSNorm/MLP/attention gauge analysis | Medium | Paper theory component |
| Permutation align | `src/permutation_alignment.py` | Hungarian matching for recovered params | Medium | Used by eval scripts |
| Enhanced KD/SCRD/LC | `scripts/enhanced_kd_clone.py` | A/B/C/D/E KD variants, Logit Completion | Very High | P0 bug: complete_logits uses full teacher logits |
| S-PSI launcher | `scripts/run_spsi.py` | S-PSI training entry | Medium (ablation) | Pure-logits regime keeps pretrained embedding |
| Carlini repro | `scripts/reproduce_carlini.py` | SVD output projection recovery | High (baseline) | Required baseline, appears functional |
| Moment-CP attack | `scripts/attack_higher_order_moments.py` | CP tensor decomposition | Medium | QUARANTINED in header, but paper main claim |
| Functional KL eval | `scripts/functional_kl_eval.py` | Functional equivalence test | High (eval) | Tests recovered W_lm functional quality |
| KD baseline | `scripts/run_kd_baseline.py` | Standard KD | Medium (baseline) | |
| Matched KD | `scripts/matched_kd_baseline.py` | Fair KD comparison | Medium (baseline) | |
| v7-v14 results | `results/v7_scrd/` through `results/v14_lc_topk5/` | Experiment JSON | Very High | Single seed, v13/v14 contaminated by full-logit access |
| v5 moments results | `results/v5_attack_moments_v2/` | Moment-CP results | High | Present locally but artifacts incomplete |
| Paper | `paper/main.tex` | NeurIPS submission draft | Very High | Current title: Moment-CP, inconsistent with latest work |
| Existing tests | `tests/test_modelsteal.py` | Basic smoke tests | Low | Minimal coverage |
| Configs | `configs/inversion_config.yaml` | S-PSI config | Low | Defaults to Qwen3.5-0.8B, not matching most experiments |
| findings.md | `findings.md` | Bug records, active-query invalidation | Very High | Confirms active-query bug |
| EXPERIMENT_PROGRESS | `EXPERIMENT_PROGRESS.md` | Full experiment timeline | Very High | Most current summary |
| CLAIMS_FROM_RESULTS | `CLAIMS_FROM_RESULTS.md` | Claim-evidence mapping | High | Shows many unsupported claims |
| PAPER_CLAIM_AUDIT | `PAPER_CLAIM_AUDIT*.md` | Paper vs evidence audit | High | Multiple FAIL entries |

## Historical / Dead Code
| File | Reason | Action |
|------|--------|--------|
| `scripts/active_query_experiment.py` | Bug confirmed | Archive |
| `scripts/algebraic_recovery_v2/v3/v4.py` | Superseded | Archive |
| `scripts/diagnose_phase2_failure.py` | Historical diagnostic | Archive |
| `scripts/attack_jacobian_fd.py` | Negative result, documented | Keep as ablation |
| `scripts/attack_logit_bias_precision.py` | Implementation broke | Keep as reference |
| `scripts/attack_memory_inversion.py` | Negative result | Keep as ablation |
| Multiple `dispatch_*.sh` | Old wave dispatchers | Archive |
