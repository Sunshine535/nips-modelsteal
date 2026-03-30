# S-PSI: How Identifiable Are Transformer Weights from Logits?

**Sensitivity-Guided Progressive Suffix Inversion** — recovering transformer
suffix parameters from black-box logit access with identifiability analysis.

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-modelsteal.git
cd nips-modelsteal

# 2. Install dependencies
bash setup.sh

# 3. Run S-PSI (oracle-prefix, 5 random inits, last 2 blocks)
python scripts/run_spsi.py \
    --model_name Qwen/Qwen3.5-0.8B \
    --regime oracle \
    --num_inits 5 \
    --num_suffix_blocks 2 \
    --output_dir results/spsi_oracle

# 4. Run S-PSI (pure-logits regime)
python scripts/run_spsi.py \
    --regime pure_logits \
    --num_inits 5 \
    --output_dir results/spsi_pure_logits

# 5. Multi-GPU parallel (one init per GPU)
bash scripts/run_spsi_multigpu.sh --regime oracle --num_inits 8 --gpus 4

# 6. Full pipeline (all experiments)
bash scripts/run_all_experiments.sh --gpus 4

# 7. Wrong-teacher falsification
python scripts/run_spsi.py --regime oracle --wrong_teacher --output_dir results/spsi_wrong_teacher
```

### Ablation: Sensitivity Matching

```bash
# β=0 (no sensitivity matching)
python scripts/run_spsi.py --beta 0.0 --output_dir results/spsi_ablation_beta0
```

## Project Structure

```
nips-modelsteal/
├── README.md
├── setup.sh                           # One-click environment setup
├── requirements.txt
├── PROPOSAL.md                        # Research proposal
├── PAPERS.md                          # Related work
├── PLAN.md                            # Experiment timeline
├── configs/
│   └── inversion_config.yaml          # S-PSI hyperparameters
├── src/
│   ├── __init__.py
│   ├── parameter_inverter.py          # Core S-PSI algorithm
│   ├── permutation_alignment.py       # Hungarian alignment for identifiability
│   └── active_query.py               # Query pool utilities
├── scripts/
│   ├── run_spsi.py                    # Main S-PSI experiment
│   ├── run_spsi_multigpu.sh           # Multi-GPU parallel init launcher
│   ├── run_all_experiments.sh         # Full pipeline launcher
│   ├── run_kd_baseline.py            # KD baseline for comparison
│   ├── eval_recovery_quality.py      # Recovery evaluation
│   ├── run_defense_eval.py           # Defense evaluation (appendix)
│   └── gpu_utils.sh                  # GPU detection utilities
├── refine-logs/                       # ARIS research refinement logs
│   ├── FINAL_PROPOSAL.md
│   ├── score-history.md
│   └── round-*-*.md
├── results/                           # Experiment outputs
└── logs/                              # Training logs
```

## Method Overview

S-PSI studies the **identifiability** of transformer weights from logit observations:

1. **Phase 0**: Recover `lm_head` (output projection) via gradient-based optimization.
2. **Phase 1-K**: Progressively recover transformer blocks from output to input.
3. **Two regimes**:
   - **Oracle-prefix**: Teacher boundary states injected → tests suffix identifiability.
   - **Pure-logits**: Fully black-box → tests realistic attack scenario.
4. **Sensitivity matching**: Match not only logits, but how logits respond to input
   perturbations (token replacements). Stronger constraint on internal parameters.
5. **Identifiability analysis**: 5 random initializations, permutation-aware alignment,
   per-matrix cosine similarity breakdown.

## Experiments

| # | Experiment | Script | Output |
|---|-----------|--------|--------|
| 1 | S-PSI oracle-prefix (5 inits) | `run_spsi.py --regime oracle` | `results/spsi_oracle/` |
| 2 | S-PSI pure-logits (5 inits) | `run_spsi.py --regime pure_logits` | `results/spsi_pure_logits/` |
| 3 | Sensitivity ablation (β=0) | `run_spsi.py --beta 0.0` | `results/spsi_ablation_beta0/` |
| 4 | Wrong-teacher control | `run_spsi.py --wrong_teacher` | `results/spsi_wrong_teacher/` |
| 5 | KD baseline | `run_kd_baseline.py` | `results/kd_baseline/` |
| 6 | Depth boundary analysis | `run_spsi.py --num_suffix_blocks 6` | `results/spsi_depth/` |

### Key Metrics

- **Per-matrix cosine similarity**: Permutation-aligned, per weight matrix (Q/K/V/O/up/gate/down/norm).
- **Cross-init variance**: Low variance = identifiable recovery.
- **Oracle vs pure-logits gap**: Quantifies prefix-mismatch impact.

## Model

| Role | Model | Params | Access |
|------|-------|--------|--------|
| Teacher | Qwen/Qwen3.5-0.8B | 0.8B | Logits only (oracle: + boundary states) |
| Student | Qwen/Qwen3.5-0.8B | 0.8B | Full (suffix recovery target) |

## Compute Budget

| Experiment | GPU-hours |
|-----------|-----------|
| Teacher cache (one-time) | ~5 |
| Oracle-prefix (5 inits × 2 blocks) | ~120 |
| Pure-logits (5 inits × 2 blocks) | ~120 |
| Ablations | ~60 |
| Depth boundary | ~80 |
| **Total** | **~385** |

## Citation

```bibtex
@inproceedings{spsi2026,
  title={S-PSI: How Identifiable Are Transformer Weights from Logits?},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
