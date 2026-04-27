# C-DART: Censored Delta Distillation for Black-Box Model Extraction under Top-K API Constraints

## Problem Statement
When an API only exposes top-K logits per token, standard KD fails. We propose C-DART to extract teacher-specific behavior under this constraint.

## Threat Model & Scope
- **Attacker has**: (1) top-K logit access to the teacher API, (2) the same public base model used to create the teacher (reference model)
- **Attacker does NOT have**: internal activations, gradients, full logit distribution, or the teacher's fine-tuning data
- **This is NOT arbitrary black-box extraction**: C-DART requires knowing which base model the teacher was fine-tuned from. This is realistic for API providers who fine-tune open-weight base models on proprietary data.

## Method
C-DART combines three techniques:
1. **Delta Teacher**: Computes residual Δ = teacher_logits - reference_logits over shared candidate tokens. The public reference model anchors the student; the delta isolates teacher-specific private behavior.
2. **Censored Constraints**: Tokens in reference top-K but absent from teacher top-K reveal teacher suppressions. C-DART uses this absence as a training signal.
3. **Ranking Loss**: Learns pairwise ordering from teacher's top-K rankings rather than matching absolute logit values.

Loss: λ_ce·L_CE + λ_rank·L_rank + λ_delta·L_delta_MSE + λ_censor·L_censor

## Key Results (Phase 1: 4 models × 3 seeds each)

### Main Table (compared against BOTH frozen reference AND ce_only)

| Model | Ref KL | ce_only KL | cdart KL | oracle KL | vs ref | vs ce_only | Gap closure (ref→ora) |
|-------|--------|-----------|----------|-----------|--------|------------|----------------------|
| Qwen2.5-0.5B | 0.495 | 0.608 | 0.209 | 0.006 | +57.8% | +65.6% | 58.4% |
| Qwen3-0.6B | 1.024 | 0.303 | 0.120 | 0.019 | +88.3% | +60.4% | 90.0% |
| Llama-3.2-1B | 0.991 | 0.647 | 0.296 | 0.046 | +70.1% | +54.2% | 73.5% |
| Llama-3.2-3B | 0.935 | 0.564 | 0.336 | 0.139 | +64.0% | +40.3% | 75.2% |

NOTE: On Qwen2.5-0.5B, ce_only DEGRADES from the frozen reference (0.608 > 0.495), so the "vs ce_only" number is inflated for that model. On all other models, ce_only improves over reference. "vs ref" is the fairer comparison across all models.

### Ablation: Censoring is the key novel contribution

| Model | bild (delta MSE) | +ranking | +censoring (cdart_full) | Censor Δ |
|-------|-----------------|----------|----------------------|----------|
| Qwen2.5-0.5B | 0.222 | 0.218 | 0.209 | +4.2% |
| Qwen3-0.6B | 0.133 | 0.125 | 0.120 | +4.3% |
| Llama-3.2-1B | 0.335 | 0.323 | 0.296 | +8.2% |
| Llama-3.2-3B | 0.368 | 0.369 | 0.336 | +8.8% |

NOTE: tdart_no_adaptive == cdart_no_censor in all experiments (identical code path when max_probe_tokens=0). This is NOT two independent ablations — it is one.

### Negative Finding
Standard top-K KD (direct KL on sparse logits) is catastrophically worse than both ce_only AND frozen reference on all models.

## Known Limitations
1. Only top-K=20 tested — need K sensitivity sweep
2. Only 500-step FT gap — need gap scaling
3. No downstream task evaluation (MMLU, HellaSwag, etc.)
4. Limited baselines — only BiLD-style delta, no DKD/DIST/existing extraction methods
5. KL normalization inconsistency in baselines (full_logit/strict_topk used batchmean, C-DART used token-mean) — NOW FIXED in code
6. Requires knowing the base model — not arbitrary black-box
7. Ranking loss adds marginal value (+0-2%) — main novelty is censoring only

## Code Structure
- src/residual_delta.py — delta teacher computation
- src/censored_delta.py — censored constraints
- src/ranking_losses.py — ranking loss
- src/kd_losses.py — KD losses (unified normalization)
- scripts/run_tdart.py — main training script
- configs/cdart_*.yaml — experiment configs
- results/cdart_*/seed_*/results.json — all 12 result files present
