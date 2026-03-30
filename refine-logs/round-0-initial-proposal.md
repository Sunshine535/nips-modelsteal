# Research Proposal: Progressive Layer-wise Parameter Inversion (PLPI)

## Problem Anchor

- **Bottom-line problem**: Can we recover the actual internal weight parameters of a black-box LLM — not just replicate its behavior — using only API query access to output logits?
- **Must-solve bottleneck**: All existing model stealing attacks either (a) extract only specific output-adjacent layers algebraically, (b) produce behavioral clones without weight recovery, or (c) require direct weight access. No method progressively recovers full internal weight matrices from API queries alone.
- **Non-goals**: We do NOT aim to (1) build a better knowledge distillation system, (2) extract proprietary training data, (3) attack real production APIs, or (4) prove weight recovery for arbitrary architectures without structural assumptions.
- **Constraints**: 8x A100-80GB, ~570 GPU-hours total, open-weight models only (Qwen3.5 family), same-architecture teacher/student for main experiments, NeurIPS 2026 submission deadline late July.
- **Success condition**: Demonstrate cosine similarity > 0.9 for output-adjacent layers and a measurable, monotonically decreasing recovery curve for deeper layers. Show that active query selection achieves 5-10x query efficiency over random sampling. Document a "Parameter Leakage Scaling Law" relating query budget to recovery quality.

## Technical Gap

Current model stealing falls into a false dichotomy: algebraic attacks that extract one layer cheaply (Carlini et al., ICML 2024) versus behavioral cloning that matches outputs but has entirely different internal representations. The gap is: **no method iteratively bridges from behavioral similarity toward actual parameter recovery**.

Why naive fixes are insufficient:
- More distillation data improves behavior match but weight cosine similarity plateaus near random
- Larger student models still have different representations
- Simple MSE loss on logits doesn't create gradient signal for deeper layers

The smallest adequate intervention: **Layer-wise inversion with output-to-input progression**, exploiting the observation that output-adjacent parameters (lm_head, last transformer block) have the strongest gradient signal from logit differences, while deeper parameters require recovered outer layers as a "lens" to propagate useful optimization signal inward.

## Method Thesis

- **One-sentence thesis**: Progressive Layer-wise Parameter Inversion (PLPI) recovers LLM weight matrices from black-box logit access by sequentially optimizing parameters from output to input, using previously recovered layers as fixed anchors and information-theoretic active query selection to maximize per-query information gain.
- **Why this is the smallest adequate intervention**: The method adds no new architectural components — it reuses the teacher's architecture and simply changes the optimization objective and procedure from standard distillation.
- **Why this route is timely**: LLM APIs expose full logit distributions (or top-K logits), creating an unprecedented attack surface. Recent algebraic extraction of output layers (Carlini et al.) proves the output end is vulnerable; PLPI asks whether this vulnerability propagates deeper.

## Contribution Focus

- **Dominant contribution**: PLPI algorithm — the first progressive layer-wise weight recovery method from black-box LLM queries, with empirical "Parameter Leakage Scaling Law" characterizing depth-dependent recovery.
- **Supporting contribution**: Active query selection framework (gradient-magnitude and Fisher-information-based) achieving 5-10x query efficiency over random sampling.
- **Explicit non-contributions**: We do NOT propose new model architectures, new training paradigms, or new defenses. Defense evaluation is included for risk assessment, not as a contribution.

## Proposed Method

### Complexity Budget
- **Frozen / reused backbone**: Teacher model (Qwen3.5-4B) loaded read-only as black-box oracle; Student uses identical architecture initialized from random or pretrained weights.
- **New trainable components**: NONE — we optimize existing student parameters, not new modules.
- **Tempting additions intentionally not used**: No auxiliary networks, no meta-learning, no adversarial training, no model compression.

### System Overview

```
Teacher (Black-box)                    Student (Optimizable)
    |                                       |
    |-- logits only --|                     |
    |                 v                     |
    |           [MSE Loss on logits]   <--- |
    |                 |                     |
    |           [Active Query Selection]    |
    |                 |                     |
    |           Phase 1: Optimize lm_head   |
    |           Phase 2: Freeze lm_head,    |
    |                    optimize last block |
    |           Phase 3: Progressive deeper  |
    |                    layer recovery      |
    v                 v                     v
   [Ground Truth]  [Recovery Metrics]  [Recovered Weights]
```

### Core Mechanism: Progressive Layer-wise Inversion

**Input**: Black-box teacher oracle (logits only), student model of same architecture (random init).
**Output**: Recovered weight matrices with per-layer cosine similarity measurements.

**Algorithm**:
1. **Warm-start** (optional): Standard KD for behavioral initialization.
2. **For each layer group L = {lm_head, block_N, block_{N-1}, ..., block_1}**:
   a. **Freeze** all previously recovered layers.
   b. **Optimize** L's parameters to minimize: `L(θ_L) = E_x[||f(x; θ_fixed, θ_L) - teacher(x)||²] + λ·R(θ_L)`
   c. **Active query selection** every K steps: pick inputs from candidate pool that maximize gradient norm w.r.t. θ_L.
   d. **Convergence**: Stop when loss improvement < ε for 500 consecutive steps.
   e. **Record** cosine similarity against ground truth.
3. **Output**: Recovered state dict + per-layer recovery metrics.

**Training signal**: MSE loss between student and teacher logits on selected query inputs.

**Why this is the main novelty**: Unlike KD (which optimizes all parameters jointly for behavior match), PLPI fixes recovered layers and focuses optimization budget on one layer group at a time, creating stronger gradient signal for each successive layer.

### Active Query Selection (Supporting Component)

Three strategies compared:
1. **Random**: Uniform sampling from query pool (baseline).
2. **Gradient Magnitude**: Select inputs where per-sample MSE is highest — proxy for large gradients on target parameters.
3. **Fisher Information**: Select inputs where KL divergence between student and teacher is highest — proxy for Fisher information w.r.t. target parameters.

**Why it does not create contribution sprawl**: Active query selection is a natural sub-component of the inversion algorithm, not a separate contribution. It makes the main method practical within reasonable query budgets.

### Modern Primitive Usage

- **Which primitive**: Open-weight LLMs (Qwen3.5-4B) as both teacher and student.
- **Exact role**: Teacher acts as a black-box oracle simulating API access; Student is the recovery target. No LLM-specific tricks — the method is architecture-agnostic given same-architecture assumption.
- **Why more natural than old-school**: Previous extraction attacks targeted CNNs or small NNs. Operating directly on transformer LLMs with realistic API-style access is the modern threat model.

### Training Plan

- **Stage 1 (KD Warm-start)**: 3 epochs, AdamW, lr=5e-5, cosine schedule, temperature=2.0, WikiText queries.
- **Stage 2 (Progressive Inversion)**: Per-layer, Adam, lr=1e-4, max 10K steps per layer group, L2 regularization λ=1e-4, query budget 500K total split across layers.
- **Active query re-selection** every 10 steps from pool of 10K candidate inputs.

### Failure Modes and Diagnostics

- **Deep layer inversion fails (cos_sim → 0)**: Expected for sufficiently deep layers. Frame as "Parameter Leakage Scaling Law" — the depth-recovery curve itself is a contribution.
- **Optimization divergence**: Gradient clipping (max_norm=1.0), convergence threshold, and automatic early stopping after 500 steps without improvement.
- **Query budget exhaustion**: Budget partitioned per phase; automatic halt at budget limit.
- **Architectural mismatch**: By design, main experiments use same architecture. Cross-architecture is an ablation.

### Novelty and Elegance Argument

**Closest work**: Carlini et al. (ICML 2024) — algebraic extraction of output projection layer.

**Exact difference**: Carlini extracts ONE layer algebraically. PLPI progressively recovers MULTIPLE layers via gradient-based optimization, using recovered outer layers as a fixed "decoder" to propagate signal inward.

**Why focused**: PLPI is a single algorithm (progressive layer-wise optimization), not a collection of modules. The contribution is the insight that recovered layers serve as anchors for deeper recovery, plus the empirical scaling law.

## Claim-Driven Validation Sketch

### Claim 1: PLPI recovers output-adjacent parameters with high fidelity
- **Minimal experiment**: Invert lm_head weights, measure cosine similarity vs ground truth.
- **Baseline**: Standard KD (same query budget, same architecture).
- **Metric**: Weight cosine similarity, output agreement rate.
- **Expected evidence**: cos_sim > 0.9 for lm_head; significantly higher than KD baseline (~0.3).

### Claim 2: Recovery degrades predictably with depth (Scaling Law)
- **Minimal experiment**: Run full progressive inversion across all layers, plot depth-recovery curve.
- **Baseline**: N/A — this is an empirical characterization.
- **Metric**: Per-layer cosine similarity vs layer depth.
- **Expected evidence**: Monotonically decreasing curve with identifiable inflection point.

### Claim 3: Active queries improve query efficiency 5-10x
- **Minimal experiment**: Compare random vs gradient-magnitude vs Fisher selection, measuring queries needed to reach 90% recovery for lm_head.
- **Metric**: Query count at fixed recovery threshold.
- **Expected evidence**: Gradient-magnitude and Fisher require 5-10x fewer queries than random.

### Claim 4: API defenses degrade deep-layer recovery more than shallow
- **Minimal experiment**: Re-run inversion under 4 defenses (logit rounding, noise, temperature perturbation, watermarking).
- **Metric**: Recovery degradation vs defense strength, per layer depth.
- **Expected evidence**: Output layers are robust to defenses; deeper layers are protected even by mild perturbation.

## Experiment Handoff Inputs
- **Must-prove claims**: Claims 1-3 are essential. Claim 4 is important but not blocking.
- **Must-run ablations**: Optimizer type (Adam vs SGD vs L-BFGS), initialization (random vs pretrained vs distilled), loss function (MSE vs KL), query batch size.
- **Critical datasets/metrics**: WikiText for query construction, weight cosine similarity as primary metric, downstream task accuracy (HellaSwag, ARC-Easy, WinoGrande) as secondary.
- **Highest-risk assumptions**: Same-architecture requirement; output layer recovery yielding cos_sim > 0.9; gradient signal propagation through recovered layers.

## Compute & Timeline Estimate
- **Estimated GPU-hours**: ~570 total (50 baseline + 60 output layer + 80 active query + 100 deeper layers + 120 scaling + 80 defense + 60 ablations + 20 writing)
- **Data/annotation cost**: None — uses public models and datasets.
- **Timeline**: 8 weeks (May 19 – July 13, 2026)
