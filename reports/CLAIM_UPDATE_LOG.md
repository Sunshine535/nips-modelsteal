# Claim Update Log

| Claim | Old Text | New Text | Evidence | Status |
|-------|----------|----------|----------|--------|
| Main method identity | "S-PSI recovers transformer suffix parameters from black-box logits" (README) | "Q-UMC: Query-budgeted Uncertainty-gated Logit Completion under strict top-K+probe API" | GPT55_DIAGNOSIS.md §19 | **PENDING** (README not updated this pass — requires broader baselines first per Task 11) |
| Logit Completion improvement | "Logit Completion beats top-K KD by 23-28%" (EXPERIMENT_PROGRESS.md) | "Q-UMC beats strict top-K KD by 80.7% ± 1.8% PPL on 3 seeds under strict oracle" | results/qumc_minimal/ | **READY** (new evidence stronger than old) |
| v13/v14 status | "Positive result, beats baseline" | "Simulator-positive, full-logit leaked — not strict black-box evidence" | docs/evidence_registry.md | **APPLIED** in evidence_registry, paper pending |
| Active-query cos 0.999 | "Strong positive recovery signal" | "INVALID — student loaded with teacher weights" | findings.md | **APPLIED** in evidence_registry |
| Moment-CP W_lm top-5 0.813 | Paper main claim | "Under-audited, raw artifacts incomplete — freeze until reproduced" | docs/evidence_registry.md | **APPLIED** in evidence_registry, paper pending |
| Hidden-state MSE helps clone | Implicit in SCRD design | "Hidden-state MSE hurts in top-K setting (v11/v12 B = PPL 753 vs A = 178)" | docs/evidence_registry.md | **APPLIED** in evidence_registry |
| Uncertainty gating | Not previously claimed | "Uncertainty weighting filters noisy tail gradients; C even beats full-logit upper bound D (130 vs 157 PPL)" | results/qumc_minimal/ CORE_COMPARISON.md | **NEW, READY** |
| SOTA / cross-model | Implied in older drafts | "Single model (Qwen2.5-0.5B) only; NOT a SOTA claim; cross-model generalization pending" | (gap) | **APPLIED** as limitation |

## Paper Claims NOT Yet Updated
Per GPT55_DIAGNOSIS §11 "Update paper only after evidence supports it", paper/main.tex update is deferred until:
1. Official close baselines (Carlini-only completion) are run.
2. Second-model replication is tested.
3. (Optional) Moment gate with null-margin threshold.

## Forbidden Updates Avoided
- Did NOT claim SOTA without fair comparison (Carlini-only baseline not yet run).
- Did NOT hide v13/v14 contamination (tagged as simulator-positive in evidence_registry).
- Did NOT describe ablation as main method (A/B/D are explicitly labeled as baseline/ablation/upper-bound).
- Did NOT claim moment gate works (not implemented; default g_v=1).
- Did NOT claim generality (single model, single dataset noted as limitation).
