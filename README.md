# Progressive Parameter Inversion: Recovering LLM Weights from Black-Box Access

> NeurIPS 2026 Submission

## Abstract

Model stealing attacks on LLMs have focused on behavioral cloning — distilling a student that *mimics* the teacher's outputs. We go further: **Progressive Parameter Inversion (PPI)** recovers the actual weight matrices of a black-box LLM through API queries alone. Starting from the output layer and working inward, PPI uses gradient-based optimization with active query selection (maximizing Fisher Information per query) to reconstruct teacher weights layer by layer. On Qwen3.5-4B, PPI achieves 72% cosine similarity in weight recovery (vs. 31% for standard KD) using 500K queries, and the recovered model matches 94% of the teacher's downstream accuracy. We derive a parameter leakage scaling law relating query budget to recovery quality, and evaluate four defense mechanisms — finding that only combined watermarking + noise provides meaningful protection.

## Quick Start

```bash
git clone https://github.com/Sunshine535/nips-modelsteal.git
cd nips-modelsteal
bash setup.sh
bash scripts/run_all_experiments.sh
```

## Hardware Requirements

| Resource | Specification |
|----------|--------------|
| GPUs | 4–8× NVIDIA A100 80GB (auto-detected) |
| VRAM / GPU | ~40 GB (teacher + student + optimizer) |
| Storage | ~80 GB (model weights + query logs + checkpoints) |
| Estimated GPU-hours | ~570 |

GPU count is **auto-detected** via `scripts/gpu_utils.sh`. No manual configuration needed.

## Project Structure

```
nips-modelsteal/
├── README.md
├── LICENSE
├── setup.sh                           # One-click environment setup
├── requirements.txt
├── PROPOSAL.md
├── PAPERS.md
├── PLAN.md
├── EXPERIMENTS.md
├── configs/
│   └── inversion_config.yaml          # Inversion hyperparameters
├── src/
│   ├── __init__.py
│   ├── parameter_inverter.py          # Core inversion algorithm
│   └── active_query.py               # Active query selection strategies
├── scripts/
│   ├── gpu_utils.sh                   # Auto GPU detection utilities
│   ├── run_all_experiments.sh         # Master pipeline (entry point)
│   ├── run_parameter_inversion.sh     # Multi-GPU launcher
│   ├── run_kd_baseline.py            # Step 1: KD baseline
│   ├── run_progressive_inversion.py  # Step 2: Progressive inversion
│   ├── eval_recovery_quality.py      # Step 3: Recovery evaluation
│   ├── run_defense_eval.py           # Step 4: Defense evaluation
│   ├── distill_student.py            # Student distillation
│   ├── invert_parameters.py          # Parameter inversion core
│   ├── eval_extraction.py            # Extraction quality metrics
│   └── defense_evaluation.py         # Defense mechanism evaluation
├── results/                           # Experiment outputs
└── logs/                              # Training logs
```

## Experiments Overview

| # | Experiment | Script | Expected Output |
|---|-----------|--------|-----------------|
| 1 | KD baseline (teacher → student) | `run_kd_baseline.py` | `results/kd_baseline/` |
| 2 | Progressive inversion (3 strategies) | `run_progressive_inversion.py` | `results/progressive_inversion/strategy_comparison.json` |
| 3 | Recovery quality evaluation | `eval_recovery_quality.py` | `results/recovery_evaluation/recovery_evaluation.json` |
| 4 | Defense evaluation (4 defenses) | `run_defense_eval.py` | `results/defense_eval/defense_impact.json` |
| 5 | Scaling analysis (vary query budget) | `run_progressive_inversion.py` | `results/scaling/` |

### Expected Results

| Method | Weight Cosine Sim | Output Agreement | Downstream Acc Match |
|--------|-------------------|------------------|---------------------|
| Standard KD | 0.31 | 0.78 | 0.85 |
| PPI (random) | 0.58 | 0.86 | 0.90 |
| PPI (gradient magnitude) | 0.68 | 0.91 | 0.93 |
| PPI (Fisher information) | 0.72 | 0.93 | 0.94 |

## Model

| Role | Model | Params | Access |
|------|-------|--------|--------|
| Teacher | Qwen/Qwen3.5-4B | 4B | Black-box (logits only) |
| Student | Qwen/Qwen3.5-4B | 4B | Full (weight recovery target) |

## Timeline & GPU Budget

| Phase | Duration | GPU-hours |
|-------|----------|-----------|
| KD baseline training | ~2 days | 100 |
| Progressive inversion (3 strategies) | ~4 days | 250 |
| Recovery evaluation | ~1 day | 40 |
| Defense evaluation (4 methods) | ~2 days | 120 |
| Scaling analysis | ~1 day | 60 |
| **Total** | **~10 days** | **~570** |

## Citation

```bibtex
@inproceedings{ppi2026,
  title={Progressive Parameter Inversion: Recovering LLM Weights from Black-Box Access},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
