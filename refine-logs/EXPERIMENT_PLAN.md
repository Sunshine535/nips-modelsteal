# EXPERIMENT PLAN: Side-Channel Representation Distillation (SCRD)

**Date**: 2026-04-24
**Target venue**: NeurIPS 2026
**Compute**: 4× A100-80GB on remote server
**Model**: Qwen/Qwen2.5-0.5B (teacher = student architecture)

## Claim-Driven Design

| Claim | Experiment | Success Criterion |
|-------|-----------|-------------------|
| C1: logit_bias side channel recovers h_final | Already validated | cos 0.977, KL 0.0002 ✅ |
| C2: ALL-position recovery is free | Measure cost + quality at all T positions | cos > 0.95 at all positions |
| C3: SCRD beats logit-only KD | Compare variant A vs D | ppl degradation_D < ppl degradation_A |
| C4: SCRD approaches oracle h_final KD | Compare variant D vs B | gap < 20% |
| C5: SCRD beats Clone 2025 SOTA | Compare variant D/E vs 7.31% | ppl degradation < 7.31% |
| C6: CP init helps | Compare variant E vs D | lm_head cos improvement |

## Experiment Blocks

### Block 0: Validate ALL-position h_final Recovery (1 GPU, ~30 min)
```python
# Modify functional_theft_v2.py to recover h at ALL positions (not just last 8)
# For 512 queries × 128 positions × 2000 probes
# Measure: cos(h_hat, h_true) histogram across ALL positions
# Expected: cos > 0.95 everywhere (slight degradation at early positions due to attention)
```

### Block 1: Main KD Comparison (1 GPU, ~8 hours)
5 variants, sequential on same GPU:

| Variant | Description | Hyperparams |
|---------|-------------|-------------|
| A | Logit-only KD | α=0.9, β=0, τ=2.0 |
| B | Logit + oracle h_final (ALL pos) | α=0.7, β=0.3, τ=2.0 |
| C | Logit + recovered h_final (LAST pos only) | α=0.7, β=0.3, τ=2.0 |
| D | Logit + recovered h_final (ALL pos) — **SCRD** | α=0.7, β=0.3, τ=2.0 |
| E | D + CP-initialized lm_head | same as D + CP warm-start |

Common: 5000 steps, batch 8, seq_len 128, lr 2e-5, AdamW, cosine schedule, eval every 500.

### Block 2: β Sweep (1 GPU, ~4 hours)
Fix variant D, sweep β ∈ {0.01, 0.1, 0.3, 0.5, 1.0}
Determines optimal balance between logit KD and representation distillation.

### Block 3: Training Duration Scaling (1 GPU, ~6 hours)
Fix variant D (best β), sweep num_steps ∈ {1000, 2000, 5000, 10000, 20000}
Shows convergence curve: SCRD vs logit-only KD as function of training budget.
KEY CLAIM: SCRD converges faster under limited query budget.

### Block 4: Multi-Model Sweep (3 GPUs parallel, ~8 hours)
Test on Llama-3.2-1B, Qwen2.5-1.5B, Qwen3.5-0.8B
Shows generalization across architectures.

## Run Order

| Order | Block | GPU | Estimated Time | Dependency |
|-------|-------|-----|---------------|------------|
| 1 | Block 0 | GPU0 | 30 min | None |
| 2 | Block 1 | GPU0 | 8 hours | Block 0 validates recovery |
| 3 | Block 2 | GPU1 | 4 hours | Block 1 confirms C > A |
| 4 | Block 3 | GPU2 | 6 hours | Block 2 determines best β |
| 5 | Block 4 | GPU1-3 | 8 hours | Block 1 confirms method works |

**Total estimated compute**: ~30 GPU-hours (~1 day wall clock on 4 GPUs)

## Evaluation Metrics (per variant)

1. **Perplexity** (student standalone on held-out WikiText)
2. **KL divergence** KL(teacher || student) on eval set
3. **Top-1 agreement** (teacher.argmax == student.argmax)
4. **Perplexity degradation %** = (ppl_student - ppl_teacher) / ppl_teacher × 100
5. **Cosine similarity** of logit distributions
6. **h_final cosine** between student's and teacher's hidden states (tracks representation alignment)

## Code Changes Needed

### Modify `scripts/enhanced_kd_clone.py`:
1. Add ALL-position h_final recovery (currently only last position)
2. Add learned linear projection `proj` for gauge absorption
3. Add variant D (SCRD-all-pos) and variant E (SCRD + CP init)
4. Use Carlini W_eff for probe (realistic attack) instead of oracle W_lm
5. Add β sweep mode
6. Add training duration sweep mode

### New script: `scripts/validate_allpos_recovery.py`
Block 0 validation: measure h_final recovery quality at ALL positions.

## Kill Criteria

- If Block 0: cos < 0.90 at non-last positions → SCRD benefit at early positions is limited; fall back to last-K-positions variant
- If Block 1: variant D ≤ variant A → representation signal doesn't help; abort SCRD
- If Block 1: variant D >> variant B → something wrong with recovery; debug
- If Block 3: SCRD doesn't converge faster → limited practical benefit

## Paper Table Structure

```
Table 1: Main Results (Qwen2.5-0.5B)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Method              PPL↓   KL↓   Top-1↑  PPL-deg%↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Teacher (oracle)    X.XX   0.00   100%    0.0%
Clone 2025†         -      -      -       7.31%
A: Logit-only KD    X.XX   X.XX   XX%     XX.X%
B: Oracle h_final   X.XX   X.XX   XX%     XX.X%
C: SCRD-last        X.XX   X.XX   XX%     XX.X%
D: SCRD-all (ours)  X.XX   X.XX   XX%     XX.X%
E: SCRD+CP (ours)   X.XX   X.XX   XX%     XX.X%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
† Clone 2025 uses 6-layer student; our student is 24-layer (same as teacher)
  for fair comparison on extraction capability, not compression.
```
