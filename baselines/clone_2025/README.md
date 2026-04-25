# Clone What You Can't Steal — Clean Baseline Reproduction

Reference: "Clone What You Can't Steal: Black-Box LLM Replication via
Logit Leakage and Distillation", [arXiv:2509.00973](https://arxiv.org/abs/2509.00973)
(September 2025). **No official code was released.** This directory is a
paper-description-based re-implementation.

## What we reproduce

The paper proposes a two-stage "logit-leakage + distillation" attack:

1. **Output projection theft.** Collect top-k logits from <10k black-box
   random queries, stack them into a matrix, and SVD to recover the
   output projection subspace (same as Carlini et al. 2024 —
   `baselines/carlini_2024/`).
2. **Student distillation.** Build a student LM with the **same
   architecture** as the teacher (or shallower in the paper's 4/6-layer
   experiments), **freeze the stolen embedding and projection layers**,
   and distill only the transformer blocks against the teacher's
   softmaxed logits on WikiText using the loss

   ```
       L = τ² · KL(softmax(z_T / τ) || softmax(z_S / τ))
           + λ · CE(z_S, y)
   ```

   with `λ = 0.1` prioritising the KL term (the value we could verify
   from the paper's arXiv HTML).

The paper's headline claim is **97.6% hidden-state geometry match** with
a 6-layer student on distilGPT-2 (6-layer teacher) after under 24 GPU
hours.

## What "hidden-state geometry match" means

The paper is terse on the exact formula. We implement the most natural
Procrustes-style reading of "how closely their internal representations
align":

```
    H_T, H_S ∈ R^{(N·T) × d}      # teacher / student last-layer hidden states
    normalise each row to unit length
    R = argmin_{R orthogonal} ||H_S_n @ R − H_T_n||_F
    geometry_match = 1 − ||H_T_n − H_S_n @ R||_F² / ||H_T_n||_F²
```

We also report the **trace form**, `trace(Σ) / (||H_T||_F · ||H_S||_F)`
where `Σ` is the cross-covariance singular values — it is the standard
"representation alignment" number used in similar mechanistic
interpretability work and gives a value most comparable to the paper's
0.976.

## Assumptions we had to add

The arXiv HTML does not specify every hyperparameter. Our defaults are
conservative and documented inline:

| item              | paper says | we use                              |
|-------------------|------------|-------------------------------------|
| teacher           | distilGPT-2 | `--model_name` (default Qwen2.5-0.5B) |
| student depth     | 4 / 6 / 8  | `--student_layers` (default = teacher depth) |
| loss              | KL τ² + λ CE (λ=0.1) | same                     |
| temperature τ     | unspecified | 2.0 (standard)                     |
| dataset           | WikiText  | WikiText-2 raw                       |
| steps             | unspecified (<24 GPU-h) | `--distill_steps` default 5000 |
| batch size        | unspecified | 4 × `seq_len 256` (A100-friendly)   |
| learning rate     | unspecified | 1e-4 (AdamW warmup + cosine)        |
| student init      | unspecified | normal(0, 0.02) on Linear, 1 on norms |
| embeddings / W_lm | frozen stolen | frozen stolen (Procrustes-rotated)|

Any of the unspecified entries can be overridden by flags. The
implementation is a conservative mainstream recipe so "CHECK" verdicts
can be traced to a concrete hyperparameter rather than a method
misinterpretation.

## Files

| file            | purpose                                              |
|-----------------|------------------------------------------------------|
| `run_clone.py`  | End-to-end: SVD → build student → distill → eval.    |
| `eval_clone.py` | Post-hoc summary + optional larger-N geometry rerun. |
| `results/`      | Per-model run output (git-ignored).                  |

## Running

```bash
# Qwen2.5-0.5B, full-depth (24-layer) student
python baselines/clone_2025/run_clone.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_queries 2048 \
    --distill_steps 5000 \
    --student_layers 24 \
    --output_dir baselines/clone_2025/results/qwen25_05b \
    --seed 42

# Compressed 6-layer student (paper-style ratio, different teacher)
python baselines/clone_2025/run_clone.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_queries 2048 \
    --distill_steps 5000 \
    --student_layers 6 \
    --output_dir baselines/clone_2025/results/qwen25_05b_L6 \
    --seed 42

# Post-hoc
python baselines/clone_2025/eval_clone.py \
    --results_dir baselines/clone_2025/results/qwen25_05b
```

Expected time on A100: **~25-40 min** per run, dominated by the 5000
distill steps at bs=4 × seq=256 with a teacher + student forward pass.

## Actual numbers (from checked-in `results/`)

| metric                            | measured on Qwen2.5-0.5B | measured on Llama-3.2-1B |
|-----------------------------------|--------------------------|--------------------------|
| Phase 1 `W_lm` subspace mean cos  | 1.0000                   | 0.9857                   |
| Phase 4 hidden-state geom match   | 0.683                    | 0.792                    |

Earlier drafts of this README promised `~0.95-0.99` hidden-state
overlap for the L=24 student; that was aspirational, based on
extrapolation from distilGPT-2 numbers in the Clone 2025 paper, and
did not survive our own run on Qwen.  The checked-in numbers above
are what our re-implementation actually produces after 5000 distill
steps.  The functional gap matters for interpretation: Clone 2025
already achieves only partial hidden-state overlap even as
`W_lm` recovery is essentially perfect, underscoring the
observability--recoverability gap that the main paper studies.
