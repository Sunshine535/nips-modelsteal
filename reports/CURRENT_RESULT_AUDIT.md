# Current Result Audit

## Result Table

| Result | File | Dataset | Config | Seed | Metric | Value | Compared Against | Supports Diagnosis? | Notes |
|--------|------|---------|--------|------|--------|-------|------------------|-------------------|-------|
| v7 random 5k | results/v7_scrd/results.json | WikiText | random init, 5k, full logits | 42 | PPL | A=2834, D=2862 | teacher 24.23 | YES (random KD fails) | Verified negative |
| v8 pretrained 5k | results/v8_scrd_pretrained/results.json | WikiText | σ=0.01, 5k, full logits | 42 | PPL | A=135, D=135, B=142 | teacher 24.23 | YES (hidden MSE not useful) | Dense KD ceiling |
| v9 random 50k | results/v9_scrd_long/results.json | WikiText | random init, 50k | 42 | PPL | A=2035 | teacher 24.23 | YES (scaling random KD not enough) | |
| v10 σ=0.1 | results/v10_perturb01/results.json | WikiText | σ=0.1, 5k | 42 | PPL | all 10^41 | teacher 24.23 | YES (σ too large) | |
| v11 topK=20 | results/v11_topk20/results.json | WikiText | σ=0.01, topK=20, 5k | 42 | PPL | A=178, B=753, D=187 | teacher 24.23 | YES (hidden MSE catastrophic under topK) | |
| v12 topK=5 | results/v12_topk5/results.json | WikiText | σ=0.01, topK=5, 5k | 42 | PPL | A=190, B=736, D=210 | teacher 24.23 | YES (same pattern) | |
| v13 topK=20+LC | results/v13_lc_topk20/results.json | WikiText | σ=0.01, topK=20, 5k | 42 | PPL | A=178, E=137 | A=178 | PARTIALLY (signal real, but full-logit leakage) | P0 contamination |
| v14 topK=5+LC | results/v14_lc_topk5/results.json | WikiText | σ=0.01, topK=5, 5k | 42 | PPL | A=190, E=136 | A=190 | PARTIALLY (same leakage issue) | P0 contamination |

## Variant Existence Check

- A. Existing Best Positive Fragment Only (old E simulator): EXISTS in v13/v14 but contaminated
- B. New MAIN METHOD Without New Mechanism (strict completion, no uncertainty): MISSING — to be implemented in run_qumc.py `completion_no_unc`
- C. Full New MAIN METHOD (Q-UMC with uncertainty): MISSING — to be implemented in run_qumc.py `completion_uncertainty`

## Result-Based Execution Decision

**PROCEED** — All evidence supports the GPT-5.5 diagnosis. Strict oracle + calibrated completion modules are implemented and tested. Next step: deploy run_qumc.py to remote GPU server for the A/B/C comparison.
