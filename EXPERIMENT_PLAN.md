# Experiment Plan — Transformer Tomography

## Hardware: 4× A100-80GB
## Budget: 100 GPU-hours (~25 wall-clock hours)
## Models: Qwen3.5-0.8B (primary), Llama-3.2-1B (generality)

---

## Shared Settings

- pool_size: 4096
- max_seq_len: 192
- P (perturbation positions): 4
- R (replacement tokens): 2
- lm_head_steps: 1500
- max_steps_per_block: 3000
- Teacher cache reused across all seeds
- 1 run per A100 (parallel dispatch)

---

## Priority-Ordered Experiments

### Exp 1: Algebraic Init vs Random Init (MUST-RUN)
- **GPU-hours**: 36
- **Model**: Qwen3.5-0.8B
- **Regime**: Oracle
- **Target**: lm_head + last 2 suffix blocks
- **Seeds**: 3
- **Init methods**: random, alg_clean (logit Jacobian only), alg_aug (logit + sensitivity)
- **Proves**: Main claim — algebraic init improves convergence speed and query efficiency
- **Key metric**: Aligned cosine vs teacher queries curve
- **Baseline**: Current random-init S-PSI

### Exp 2: Gramian Predicts Recovery (MUST-RUN)
- **GPU-hours**: 24
- **Model**: Qwen3.5-0.8B
- **Regime**: Oracle + Pure-logits
- **Target**: Last 3 blocks (block 21, 22, 23)
- **Computes**: Projected Gramian eigenspectrum per block
- **Correlates with**: Final aligned cosine from Exp 1 + 4 extra pure-logits runs
- **Proves**: Theory claim — λ_min(projected G) predicts recoverability better than naive proxies
- **Baselines**: Unprojected Gramian, initial logit MSE, initial gradient norm, Fisher/divergence proxy
- **Key metric**: R² of λ_min vs cosine scatter plot

### Exp 3: Controls & Ablations (MUST-RUN)
- **GPU-hours**: 10
- **Model**: Qwen3.5-0.8B
- **Sub-experiments**:
  - Wrong-teacher falsification (3 inits)
  - Held-out perturbation family generalization
  - Gauge-projected vs unprojected Gramian comparison
  - β=0 ablation (no sensitivity matching)
- **Proves**: Not overfitting; gauge quotient matters; sensitivity is essential
- **Key metric**: Wrong-teacher gap, overfit ratio, projected vs unprojected R²

### Exp 4: Generality — Llama-3.2-1B (SHOULD-RUN)
- **GPU-hours**: 16
- **Model**: Llama-3.2-1B (same-architecture teacher/student)
- **Regime**: Oracle
- **Target**: lm_head + last block
- **Seeds**: 2
- **Init methods**: random, alg_aug
- **Proves**: Not Qwen-specific; transfers to another GQA/RMSNorm architecture
- **Key metric**: Last-block aligned cosine; algebraic init speedup

### Exp 5: Buffer & Calibration
- **GPU-hours**: 14
- **Purpose**: Cache building, hyperparameter tuning, rerun buffer
- **Includes**: Teacher cache for both models, quick smoke tests, parameter sweeps

---

## Total Budget

| Exp | GPU-hours | Priority |
|-----|-----------|----------|
| 1. Algebraic Init | 36 | MUST-RUN |
| 2. Gramian Prediction | 24 | MUST-RUN |
| 3. Controls & Ablations | 10 | MUST-RUN |
| 4. Llama Generality | 16 | SHOULD-RUN |
| 5. Buffer | 14 | RESERVE |
| **Total** | **100** | |

---

## Expected Paper Figures

| Fig | Type | Data Source | Claim |
|-----|------|------------|-------|
| 1 | Method diagram | N/A | Overview of Transformer Tomography pipeline |
| 2 | Line plot | Exp 1 | Aligned cosine vs queries: alg_aug >> alg_clean >> random |
| 3 | Scatter plot | Exp 2 | λ_min(G) vs final cosine, R² > 0.8 |
| 4 | Heatmap | Exp 2 | Depth × budget phase transition surface |
| 5 | Bar plot | Exp 3 | Wrong-teacher gap: true >> wrong for all blocks |
| 6 | Bar plot | Exp 3 | Held-out generalization: train ≈ heldout |
| 7 | Table | Exp 4 | Qwen vs Llama last-block recovery comparison |

---

## Fallback Plans

| If... | Then... |
|-------|---------|
| Algebraic init ≈ random | Use Gramian eigenspace as preconditioner/regularizer instead |
| Gramian doesn't predict recovery | Downgrade to "local/early-stage diagnostic"; use logdet or effective rank |
| Both fail | Publish S-PSI alone with complete symmetry characterization |
| Llama fails | Narrow claims to "Qwen family empirical study" |
| Budget overrun | Drop Exp 4, reduce Exp 1 to 2 seeds |

---

## Code Changes Required

### New files
1. `src/gramian.py` — Flat param spec, JVP sketch, Gramian computation
2. `src/symmetry_gauge.py` — Continuous gauge basis, projection
3. `src/algebraic_init.py` — Sketched Gauss-Newton initialization
4. `scripts/run_gramian_eval.py` — Offline Gramian analysis

### Modified files
1. `src/parameter_inverter.py` — Add init_method to SPSIConfig, wire algebraic init
2. `scripts/run_spsi.py` — Add CLI flags for init/gramian config
3. `src/permutation_alignment.py` — Add continuous gauge canonicalization

### Implementation Order
1. Factor compute_block_observations() out of invert_block()
2. Implement src/gramian.py (flat param + JVP sketch)
3. Implement src/algebraic_init.py (reduced least-squares)
4. Add simple gauge projection (RMSNorm + gated-MLP)
5. Wire into run_spsi.py
6. Implement scripts/run_gramian_eval.py
7. Extend continuous gauge handling

---

## Run Order on Server

```bash
# Phase 0: Setup (~1 hour)
bash setup.sh
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B')"
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')"

# Phase 1: Exp 1 — Algebraic Init (parallel, 4 GPUs, ~9 wall-hours)
# GPU 0-2: 3 seeds × random init
# GPU 3: cache build + alg_clean seed 0
# Then rotate for alg_aug

# Phase 2: Exp 2 — Gramian computation (parallel, ~6 wall-hours)
# Run gramian_eval.py on cached results

# Phase 3: Exp 3 — Controls (parallel, ~2.5 wall-hours)

# Phase 4: Exp 4 — Llama (parallel, ~4 wall-hours)
```
