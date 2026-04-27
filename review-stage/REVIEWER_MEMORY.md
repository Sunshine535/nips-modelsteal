# Reviewer Memory

## Round 1 — Score: 2/10
- **Suspicion**: Missing raw results for 3/4 models — might be fabricated
- **Suspicion**: ce_only baseline might be cherry-picked because it's worse than doing nothing on Qwen2.5
- **Suspicion**: KL normalization mismatch could make baselines look artificially bad
- **Suspicion**: tdart_no_adaptive == cdart_no_censor is an ablation inflation pattern
- **Unresolved**: No downstream eval, no K sweep, no gap scaling
- **Patterns**: Author reported "vs ce_only" prominently but buried the frozen reference comparison
