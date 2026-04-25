# Baselines

Clean, standalone re-implementations of the two LM-stealing baselines our
Transformer Tomography paper compares against:

| dir              | paper                                  | method                        |
|------------------|----------------------------------------|-------------------------------|
| `carlini_2024/`  | Carlini et al. 2024 (ICML 2024 Best Paper), arXiv:2403.06634 | SVD on last-position logits → `W_lm` subspace |
| `clone_2025/`    | "Clone What You Can't Steal", arXiv:2509.00973 (no code released) | Carlini SVD + frozen stolen `W_lm` + knowledge distillation on WikiText |
| `comparison/`    | —                                      | Aggregates all JSONs into one markdown + machine-readable table |

Each subdirectory is a self-contained package:

- `README.md` documents what is reproduced and what hyperparameters are
  paper-specified vs our assumptions.
- `run_*.py` is a single standalone script (no imports from the main
  project's `scripts/` or `src/`).
- `eval_*.py` re-reads the `results.json` and prints / re-computes the
  comparison metrics.
- `results/` holds per-model output JSONs (git-ignored).

## Quick-start

```bash
# 1. Carlini baseline
python baselines/carlini_2024/run_carlini.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_queries 2048 --seq_len 128 \
    --output_dir baselines/carlini_2024/results/qwen25_05b \
    --seed 42

# 2. Clone baseline (requires teacher + training; ~30 min on A100)
python baselines/clone_2025/run_clone.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_queries 2048 \
    --distill_steps 5000 \
    --student_layers 24 \
    --output_dir baselines/clone_2025/results/qwen25_05b \
    --seed 42

# 3. Build comparison table (tolerant of missing runs)
python baselines/comparison/build_comparison_table.py
cat baselines/comparison/comparison.md
```

## Comparison axes (what the table shows)

| axis                 | what it means                                            |
|----------------------|----------------------------------------------------------|
| `W_lm cos`           | recovery of the output projection (all methods)          |
| `W_down cos`         | recovery of the last-block `down_proj.weight` (ours only)|
| Hidden geom match    | Procrustes-aligned hidden-state variance preserved       |
| Pure-logits?         | ✓ = only `(x, z(x))` pairs used; ✗ = oracle hidden states|
| Time                 | wall-clock on A100-80G (approximate)                     |

## Actual numbers (from checked-in results as of 2026-04-19)

Values below are loaded from the result JSONs in each baseline's
`results/` directory.  Earlier drafts of this README quoted
speculative target ballparks; we now only report measured numbers.

| Method                      | W_lm cos | Hidden geom | Pure-logits | Source artifact                                              |
|-----------------------------|---------:|------------:|-----------:|--------------------------------------------------------------|
| Carlini 2024 (Qwen2.5-0.5B) |   1.0000 |           — |          ✓ | `baselines/carlini_2024/results/qwen25_05b/results.json`     |
| Carlini 2024 (Llama-3.2-1B) |   0.9857 |           — |          ✓ | `baselines/carlini_2024/results/llama32_1b/results.json`     |
| Clone 2025 (Qwen2.5-0.5B)   |        — |       0.683 |          ✓ | `baselines/clone_2025/results/.../results.json`              |
| Clone 2025 (Llama-3.2-1B)   |        — |       0.792 |          ✓ | `baselines/clone_2025/results/.../results.json`              |
| S-PSI pure-logits (Qwen)    |  ~0.005  |           — |          ✓ | `results/v2_pure_logits_s42/spsi_summary.json`               |
| S-PSI oracle (Qwen)         |   0.122  |           — |          ✗ | `results/v2_random_s42/spsi_summary.json`                    |
| S-PSI oracle (Llama-3.2-1B) |   0.663  |           — |          ✗ | `results/v4_llama_spsi/experiment_summary.json`              |

The hidden-geometry numbers for Clone 2025 are substantially below the
speculative `~0.95` values in earlier drafts; this reflects the
implementation actually converging to a functionally close but
structurally distinct solution on Qwen2.5-0.5B with L=24 distillation.
The actual `comparison.md` is rebuilt from whatever `results.json`
files are present; this table will become stale if new runs produce
different numbers.
