# Progressive Parameter Inversion: Recovering LLM Weights from Black-Box Access

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-modelsteal.git
cd nips-modelsteal

# 2. One-command setup + run all experiments
bash run.sh

# 3. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-modelsteal_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`

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
