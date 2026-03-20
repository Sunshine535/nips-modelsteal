# Progressive Parameter Inversion (PPI)

**Reverse-Engineering LLM Weights via Iterative Distillation**

> NeurIPS 2026 Submission — Model Stealing Beyond Behavioral Cloning

## TL;DR

We propose **Progressive Layer-wise Parameter Inversion (PLPI)**, a method that
goes beyond behavioral cloning to *actually recover* the weight matrices of a
black-box LLM. Starting from the output layer and working inward, we use
gradient-based optimization with active query selection to reconstruct teacher
weights layer by layer.

## Method Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Black-Box Teacher (Qwen3.5-9B)        │
│                   ┌───────────────────┐                 │
│   input x ──────►│  ??? hidden ???    │──────► logits   │
│                   └───────────────────┘                 │
└─────────────────────────┬───────────────────────────────┘
                          │ query API (logits only)
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Progressive Parameter Inversion             │
│                                                         │
│  Stage 1: Behavioral Distillation                       │
│    Student ← KD(Teacher logits)                         │
│                                                         │
│  Stage 2: Layer-wise Weight Recovery (output → deeper)  │
│    For layer L = N, N-1, ..., 1:                        │
│      θ_L* = argmin_θ ||f(x; θ_fixed, θ_L) - teacher(x)||  │
│      + Active Query Selection (max info gain)           │
│                                                         │
│  Stage 3: Parameter Leakage Quantification              │
│    Measure cos_sim(θ_recovered, θ_true) per layer       │
│    Derive scaling law: queries → recovery %             │
└─────────────────────────────────────────────────────────┘
```

## Key Contributions

1. **PLPI Algorithm**: First method to recover actual weight parameters (not just behavior) from a black-box LLM via API queries alone.
2. **Active Query Selection**: Information-theoretic input selection that maximizes parameter leakage per query.
3. **Parameter Leakage Scaling Law**: Empirical characterization of how query budget maps to weight recovery percentage.
4. **Defense Evaluation**: Systematic evaluation of output perturbation, logit rounding, and watermarking as countermeasures.

## Quick Start

### Environment Setup

```bash
# Clone
git clone https://github.com/YOUR_ORG/nips-modelsteal.git
cd nips-modelsteal

# Create env
conda create -n modelsteal python=3.11 -y
conda activate modelsteal
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Stage 1: Distill student from teacher
python scripts/distill_student.py \
    --teacher_model Qwen/Qwen3.5-9B \
    --student_model Qwen/Qwen3.5-0.8B \
    --num_queries 100000 \
    --output_dir results/distillation

# Stage 2: Progressive parameter inversion
python scripts/invert_parameters.py \
    --teacher_model Qwen/Qwen3.5-9B \
    --student_checkpoint results/distillation/best_student \
    --query_budget 500000 \
    --output_dir results/inversion

# Stage 3: Evaluate extraction quality
python scripts/eval_extraction.py \
    --teacher_model Qwen/Qwen3.5-9B \
    --recovered_model results/inversion/recovered_model \
    --output_dir results/evaluation

# Stage 4: Defense evaluation
python scripts/defense_evaluation.py \
    --teacher_model Qwen/Qwen3.5-9B \
    --defense_type logit_rounding \
    --output_dir results/defense
```

### Multi-GPU Launch (8x A100)

```bash
bash scripts/run_parameter_inversion.sh
```

## Models

| Role    | Model          | Params | Access     |
|---------|----------------|--------|------------|
| Teacher | Qwen3.5-9B     | 9B     | Black-box  |
| Student | Qwen3.5-0.8B   | 0.8B   | Full       |
| Student | Qwen3.5-2B     | 2B     | Full       |
| Student | Qwen3.5-4B     | 4B     | Full       |

## Hardware

- 8x NVIDIA A100-80GB
- ~500 GPU-hours estimated for full experiment suite

## Project Structure

```
nips-modelsteal/
├── README.md
├── PROPOSAL.md          # Research proposal with hypotheses
├── PAPERS.md            # Related work analysis
├── PLAN.md              # 8-week experiment plan
├── EXPERIMENTS.md       # Experiment log
├── requirements.txt
├── configs/
│   └── inversion_config.yaml
├── scripts/
│   ├── run_parameter_inversion.sh
│   ├── distill_student.py
│   ├── invert_parameters.py
│   ├── eval_extraction.py
│   └── defense_evaluation.py
└── src/
    ├── __init__.py
    ├── parameter_inverter.py
    └── active_query.py
```

## Citation

```bibtex
@inproceedings{progressive-parameter-inversion-2026,
  title={Progressive Parameter Inversion: Reverse-Engineering LLM Weights via Iterative Distillation},
  author={Anonymous},
  booktitle={NeurIPS},
  year={2026}
}
```

## License

Apache 2.0
