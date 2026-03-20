# Experiment Plan — Progressive Parameter Inversion

## Timeline: 8 Weeks (May 19 – July 13, 2026)
## Hardware: 8x A100-80GB
## NeurIPS 2026 Abstract Deadline: Late July

---

## Week 1: Infrastructure & Baselines (May 19–25)

### Tasks
- [ ] Set up environment, verify Qwen3.5-{0.8B, 2B, 4B, 9B} load on 8x A100
- [ ] Implement black-box teacher wrapper (local model, API-style interface returning logits only)
- [ ] Run baseline knowledge distillation: 0.8B student ← 9B teacher, 100K queries
- [ ] Implement evaluation harness: weight cosine similarity, output match rate, KL divergence

### GPU Budget: ~50 GPU-hours
### Deliverable: Working KD baseline + evaluation infrastructure

---

## Week 2: Core Inversion — Output Layer (May 26 – Jun 1)

### Tasks
- [ ] Implement LayerWiseInverter for output projection layer (lm_head)
- [ ] Gradient-based optimization: minimize ||student_logits - teacher_logits||² w.r.t. lm_head weights
- [ ] Ablate: SGD vs Adam vs L-BFGS for weight recovery
- [ ] Evaluate: cosine similarity of recovered lm_head vs ground truth
- [ ] Compare with Carlini et al. algebraic method (reimplemented)

### GPU Budget: ~60 GPU-hours
### Deliverable: Output layer recovery with cos_sim measurement

### Expected Result: cos_sim > 0.95 for output layer (validation of approach)

---

## Week 3: Active Query Selection (Jun 2–8)

### Tasks
- [ ] Implement ActiveQuerySelector: gradient-magnitude-based input selection
- [ ] Implement information-theoretic query selection (Fisher information approximation)
- [ ] Compare: random queries vs gradient-based vs Fisher-based
- [ ] Measure: queries needed to reach 90% cos_sim with each strategy
- [ ] Ablate: query batch sizes {32, 128, 512, 2048}

### GPU Budget: ~80 GPU-hours
### Deliverable: Active query module + ablation results

### Expected Result: 5-10x query efficiency improvement over random

---

## Week 4: Progressive Inversion — Deeper Layers (Jun 9–15)

### Tasks
- [ ] Extend inversion to last transformer block (attention + FFN)
- [ ] Implement layer-freezing: recovered layers frozen during deeper inversion
- [ ] Test depth progression: output → last block → second-to-last → ...
- [ ] Measure recovery curve: cos_sim vs layer depth
- [ ] Identify depth at which recovery degrades below useful threshold

### GPU Budget: ~100 GPU-hours
### Deliverable: Multi-layer recovery results, depth-recovery curve

### Expected Result: Recovery degrades with depth; identify inflection point

---

## Week 5: Scaling Experiments (Jun 16–22)

### Tasks
- [ ] Run full inversion pipeline for 3 student sizes: 0.8B, 2B, 4B
- [ ] Query budget sweep: {10K, 50K, 100K, 500K, 1M}
- [ ] Derive Parameter Leakage Scaling Law: fit curve to (queries, recovery%) data
- [ ] Test: does larger student → better recovery? (capacity hypothesis)
- [ ] Cross-architecture test: attempt inversion with Llama-3.2-1B student → Qwen teacher

### GPU Budget: ~120 GPU-hours
### Deliverable: Scaling law plots, cross-architecture results

---

## Week 6: Defense Evaluation (Jun 23–29)

### Tasks
- [ ] Implement defenses in teacher wrapper:
  - Logit rounding (2, 4, 6 decimal places)
  - Gaussian noise (σ = 0.01, 0.1, 1.0)
  - Temperature perturbation (T ± Uniform(0, 0.5))
  - Top-K logit masking (return only top 50/100/1000)
  - Watermarking (Kirchenbauer-style green/red lists)
- [ ] Re-run full inversion pipeline under each defense
- [ ] Measure: recovery degradation vs usability degradation tradeoff
- [ ] Plot Pareto frontier: defense strength vs API utility

### GPU Budget: ~80 GPU-hours
### Deliverable: Defense evaluation results, Pareto plots

---

## Week 7: Ablations & Analysis (Jun 30 – Jul 6)

### Tasks
- [ ] Ablation: initialization (random vs distilled student vs pretrained)
- [ ] Ablation: optimization (SGD, Adam, L-BFGS, natural gradient)
- [ ] Ablation: loss function (MSE, KL, cosine, Wasserstein)
- [ ] Analysis: which weight matrices are easiest/hardest to recover? (Q, K, V, O, up, gate, down)
- [ ] Analysis: correlation between weight recovery and functional equivalence
- [ ] Negative results: document what doesn't work and why

### GPU Budget: ~60 GPU-hours
### Deliverable: Ablation tables, analysis figures

---

## Week 8: Paper Writing & Polish (Jul 7–13)

### Tasks
- [ ] Write full paper draft (8 pages + appendix)
- [ ] Generate all figures: method diagram, scaling law, depth-recovery curves, defense Pareto
- [ ] Internal review and revision
- [ ] Prepare supplementary materials and code release
- [ ] Final reproducibility check: run key experiments from scratch

### GPU Budget: ~20 GPU-hours (reproducibility check only)
### Deliverable: Complete paper draft

---

## Total GPU Budget

| Week | Focus | GPU-hours |
|------|-------|-----------|
| 1 | Infrastructure & Baselines | 50 |
| 2 | Output Layer Inversion | 60 |
| 3 | Active Query Selection | 80 |
| 4 | Deeper Layers | 100 |
| 5 | Scaling Experiments | 120 |
| 6 | Defense Evaluation | 80 |
| 7 | Ablations & Analysis | 60 |
| 8 | Writing & Polish | 20 |
| **Total** | | **570** |

## Key Milestones

| Date | Milestone | Go/No-Go |
|------|-----------|----------|
| May 25 | KD baseline working, eval harness complete | Must pass |
| Jun 1 | Output layer recovery cos_sim > 0.9 | Go: proceed. No-go: investigate. |
| Jun 15 | Multi-layer recovery curve obtained | Even negative results are publishable |
| Jun 22 | Scaling law derived | Core contribution |
| Jun 29 | Defense evaluation complete | Paper is writable |
| Jul 13 | Paper draft complete | Submit |

## Risk Contingencies

1. **If output layer recovery fails**: Fall back to Carlini-style algebraic extraction + our active query contribution
2. **If deep layer recovery is poor**: Frame as "Parameter Leakage Scaling Law" (negative-ish result is still novel)
3. **If GPU budget runs over**: Reduce query budget sweep; focus on 100K queries only
4. **If cross-architecture fails completely**: Drop it; focus on same-architecture results
