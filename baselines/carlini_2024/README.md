# Carlini et al. 2024 — Clean Baseline Reproduction

Reference: Carlini et al., "Stealing Part of a Production Language Model"
(ICML 2024 Best Paper), [arXiv:2403.06634](https://arxiv.org/abs/2403.06634).

## What we reproduce

Given only black-box logit access to a causal LM with hidden dim `d` and
vocab size `V`, the last-position logit vector lies in the `d`-dimensional
column space of the output projection `W_lm`. By collecting `N ≫ d` random
queries, stacking their logit vectors into `Y ∈ R^{N×V}`, centering, and
taking the SVD:

1. A sharp gap between `sigma_d` and `sigma_{d+1}` reveals `d` (the hidden
   dimension of the production model).
2. The top-`d` right singular vectors span `col(W_lm^T)` (up to an
   orthogonal rotation); they are an algebraic extraction of the output
   projection **subspace** with zero gradient updates and `<10k` queries.

This directory contains a **clean, standalone** re-implementation suitable
for side-by-side comparison with follow-up work. It has **no dependencies**
on the main project's `scripts/` or `src/` modules.

## What improves over the original project script

The ancestor `scripts/reproduce_carlini.py` already runs Carlini on
Qwen2.5-0.5B, but it has three limitations:

1. Hard-coded for Qwen2.5 (no `--model_name` path for Qwen3.5 / Llama).
2. Procrustes returns `NaN` whenever `d_hat != d_true`.
3. Metrics are scattered across output dicts.

`run_carlini.py` fixes all three:

- `--model_name` accepts any HuggingFace causal LM loadable with
  `AutoModelForCausalLM(trust_remote_code=True)`.
- Procrustes aligns on the **overlap dimension** `min(d_hat, d_true)` by
  matching the recovered subspace to the best-aligned subspace of `W_lm`
  (or vice versa). Returns valid numbers even when the SVD detects a
  slightly mis-sized hidden dim (common when the top-`V - d` singular
  values aren't exactly zero in bf16).
- Reports (i) hidden-dim recovery accuracy, (ii) subspace principal-angle
  cosines (mean / min / max), (iii) Procrustes Frobenius error, (iv)
  query-efficiency scaling, all in one `results.json`.

## Files

| file              | purpose                                                   |
|-------------------|-----------------------------------------------------------|
| `run_carlini.py`  | Collect logits, SVD, recover `W_lm` subspace.             |
| `eval_carlini.py` | Print / summarize `results.json` and save `eval_summary.json`. |
| `results/`        | Per-model extraction outputs (git-ignored).               |

## Running

```bash
# Qwen2.5-0.5B — canonical model for our paper
python baselines/carlini_2024/run_carlini.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_queries 2048 --seq_len 128 \
    --output_dir baselines/carlini_2024/results/qwen25_05b \
    --seed 42

# Llama-3.2-1B
python baselines/carlini_2024/run_carlini.py \
    --model_name meta-llama/Llama-3.2-1B \
    --num_queries 2048 --seq_len 128 \
    --output_dir baselines/carlini_2024/results/llama32_1b \
    --seed 42

# Post-hoc evaluation
python baselines/carlini_2024/eval_carlini.py \
    --results_dir baselines/carlini_2024/results/qwen25_05b
```

Expected time: `~10-30s` SVD + `~30-60s` model forward passes per model on
an A100 (dominated by the 2048-query forward pass at `seq_len=128`).

## Expected result on Qwen2.5-0.5B

| metric                            | value    |
|-----------------------------------|----------|
| recovered hidden dim              | 894-896  |
| subspace principal-angle mean cos | ≥0.9999  |
| Procrustes mean cosine            | ≥0.9999  |
| Procrustes rel. Frobenius error   | ≤0.01    |

Carlini is pure-logits (no hidden-state oracle needed), and recovers
only the **output projection subspace** — not internal layers.
