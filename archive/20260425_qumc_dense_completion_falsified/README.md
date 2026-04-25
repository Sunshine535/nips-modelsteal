Q-UMC dense completion / probe-dense KD is archived as a falsified main path.
It remains historical negative evidence and ablation baseline.
Do not cite as positive strict black-box evidence.

Key findings:
- ce_only PPL 125 ≈ completion_uncertainty PPL 124 (Q-UMC = null mechanism)
- probe_dense_kd PPL 425 ± 86 (worse than ce_only, unstable)
- normed_kl variant crashes (PPL 482k) confirming gate is KD off-switch
- Full chain: v7→v14→qumc_minimal→strict_v2→strict_v3→probe_dense_v4 all show
  no strict non-oracle method beats CE-only in pretrained-perturbed setup

Archived files:
- reports/HONEST_FINAL_STATUS.md (copied)
- reports/CORE_COMPARISON_R2_STRICT.md (copied)
- reports/MECHANISM_SUMMARY_R2.md (copied)
- configs/qumc_probe_dense_v4.yaml (copied)

Result directories frozen (not moved, only marked):
- results/qumc_probe_dense_v4/
- results/qumc_minimal_strict_v2/
- results/qumc_minimal/
- results/v13_lc_topk20/
- results/v14_lc_topk5/
