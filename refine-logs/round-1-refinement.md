# Round 1 Refinement

## Problem Anchor
- **Bottom-line problem**: Can we recover the actual internal weight parameters of a black-box LLM — not just replicate its behavior — using only API query access to output logits?
- **Must-solve bottleneck**: All existing model stealing attacks either (a) extract only specific output-adjacent layers algebraically, (b) produce behavioral clones without weight recovery, or (c) require direct weight access. No method progressively recovers full internal weight matrices from API queries alone.
- **Non-goals**: We do NOT aim to (1) build a better knowledge distillation system, (2) extract proprietary training data, (3) attack real production APIs, or (4) prove weight recovery for arbitrary architectures without structural assumptions.
- **Constraints**: 8x A100-80GB, ~570 GPU-hours total, open-weight models only (Qwen3.5 family), NeurIPS 2026 deadline late July.
- **Success condition**: Demonstrate cosine similarity > 0.9 for a contiguous output suffix (lm_head + last k blocks) under logits-only access. Characterize the boundary at which recovery breaks.

## Anchor Check
- **Original bottleneck**: No method recovers internal weights from black-box logits alone.
- **Why the revised method still addresses it**: We now propose a stronger mechanism (Jacobian-guided block inversion with alternating least squares) that goes beyond distillation. We narrow the scope to identifiable suffix recovery — which is the strongest achievable claim under this threat model.
- **Reviewer suggestions rejected as drift**: The suggestion to consider delta-recovery from public base models would change the problem from "black-box extraction" to "fine-tune detection" — explicitly rejected as drift.

## Simplicity Check
- **Dominant contribution after revision**: Jacobian-Guided Progressive Suffix Inversion (J-PSI) — a block-local inversion algorithm that recovers a contiguous output suffix of a transformer from logit-only access, with identifiability analysis.
- **Components removed or merged**: (1) KD warm-start deleted — it hurts identifiability. (2) Three query strategies merged into one (gradient-diversity-based) + random baseline. (3) Defense evaluation moved out of main contributions. (4) "Parameter Leakage Scaling Law" demoted from contribution to analysis section.
- **Reviewer suggestions rejected as unnecessary complexity**: Multi-position autoregressive rollout supervision adds implementation complexity without clear benefit for block-local recovery. Kept as future work.
- **Why the remaining mechanism is still the smallest adequate route**: J-PSI adds only one algorithmic innovation (Jacobian-guided inversion step) to the basic progressive framework, making it genuinely different from staged distillation.

## Changes Made

### 1. Core Mechanism: From MSE Distillation to Jacobian-Guided Block Inversion
- **Reviewer said**: "Freeze recovered layers and optimize one block with logit MSE is not yet an inversion method; it is block-coordinate distillation."
- **Action**: Replaced MSE-only optimization with a two-phase per-block procedure: (a) Jacobian matching — match the input-output Jacobian of the block to create stronger parameter-level signal, (b) Alternating refinement — alternate between updating the current block and fine-tuning the recovered suffix. This breaks the "permanent freeze" failure mode.
- **Reasoning**: Jacobian matching provides direct parameter-level constraints beyond logit-level agreement. For a linear layer y = Wx, matching the Jacobian dy/dx across many inputs is equivalent to recovering W directly. For nonlinear transformer blocks, Jacobian matching constrains weights more tightly than output matching alone.
- **Impact on core method**: Fundamentally different optimization objective — now truly an inversion method.

### 2. Scope: From Full Model to Identifiable Suffix
- **Reviewer said**: "Make the primary claim a pre-specified contiguous suffix recovery target."
- **Action**: Primary claim is now: recover lm_head + last K blocks (K determined empirically, expected K=3-6 for small models) above cos_sim > 0.9. The depth boundary where recovery breaks is an analysis contribution, not the main claim.
- **Reasoning**: This is the strongest credible claim under logits-only access. Deeper layers have exponentially weaker signal — admitting this up front strengthens the paper.
- **Impact on core method**: Cleaner success criterion, more credible experimental plan.

### 3. Model Scale: Proof-of-Concept on Smaller Model
- **Reviewer said**: "Use a smaller model as the main proof-of-concept."
- **Action**: Primary experiments on Qwen3.5-0.6B (28 layers, manageable), with Qwen3.5-4B as stress test on output layer + last 2 blocks only. This makes full suffix recovery genuinely plausible within budget.
- **Reasoning**: A 0.6B model with 28 layers allows us to test recovery across a meaningful depth range. 4B serves as a scaling validation, not the main experiment.
- **Impact on core method**: Much more feasible; allows multiple random init runs for identifiability.

### 4. Identifiability Controls
- **Reviewer said**: "Missing identifiability controls — need multiple random inits, symmetry-aware alignment."
- **Action**: Added three critical controls: (a) Multiple random initializations (5 runs) to test convergence to same solution, (b) Permutation-aware alignment for attention heads and FFN neurons before computing cosine similarity, (c) Scale-invariant metrics accounting for RMSNorm/LayerNorm absorbing scale factors.
- **Reasoning**: Without these controls, high cosine similarity could reflect convergence to a functionally equivalent but differently parameterized solution.
- **Impact on core method**: Strengthens the claim that we recover actual parameters, not just equivalent functions.

### 5. Contribution Focus
- **Reviewer said**: "Contribution sprawl — too many parallel contributions."
- **Action**: Single contribution: J-PSI algorithm with identifiability analysis. Active query selection is a sub-component (one method + random baseline). Scaling law is analysis. Defenses moved to appendix.
- **Reasoning**: One sharp contribution is better than four diluted ones.
- **Impact on core method**: Much sharper paper story.

## Revised Proposal

# Research Proposal: Jacobian-Guided Progressive Suffix Inversion (J-PSI)

## Problem Anchor
[Same as above — verbatim]

## Technical Gap

Current model stealing creates a false dichotomy: algebraic attacks extract one layer cheaply (Carlini et al., ICML 2024) versus behavioral cloning that matches outputs but has entirely different internal representations. The gap: **no method recovers a contiguous block of internal weight matrices from API logits alone, with identifiability guarantees**.

Why naive fixes fail:
- Standard distillation optimizes ALL parameters jointly for logit match, creating an infinite family of functionally equivalent but differently parameterized solutions. Weight cosine similarity stays near random.
- Simple "freeze-and-optimize" (block-coordinate distillation) has the same problem at the block level: MSE on logits does not uniquely determine block parameters due to downstream flexibility.

**Key insight**: For a fixed recovered suffix S = {block_K+1, ..., block_N, lm_head}, the Jacobian of the composite function f_S at the interface point is a much stronger constraint on block_K's parameters than logit-level loss alone. Matching input-output Jacobians across diverse inputs over-determines the block parameters (up to known symmetries).

## Method Thesis

- **One-sentence thesis**: J-PSI recovers a contiguous output suffix of a transformer LLM from black-box logit access by combining progressive layer-wise recovery with per-block Jacobian matching and permutation-aware identifiability alignment.
- **Why smallest adequate intervention**: Adds one algorithmic component (Jacobian matching) to the progressive framework. No new networks, no meta-learning, no adversarial training.
- **Why timely**: LLM APIs expose full logit distributions; Carlini et al. proved output layers are algebraically extractable. J-PSI asks: how deep does this vulnerability extend?

## Contribution Focus

- **Dominant contribution**: J-PSI algorithm — first method to recover a contiguous multi-layer suffix of a transformer from logits-only access, with identifiability analysis showing convergence to the true (modulo known symmetries) parameterization.
- **Supporting analysis**: Empirical characterization of the depth-recovery boundary and query-efficiency gains from gradient-diversity-based query selection.
- **Explicit non-contributions**: No new architectures, no defense proposals, no behavioral cloning improvements.

## Proposed Method

### Complexity Budget
- **Frozen / reused**: Teacher model as read-only black-box oracle, student uses identical architecture.
- **New trainable components**: NONE.
- **Intentionally excluded**: KD warm-start (hurts identifiability), meta-learning, adversarial training, auxiliary networks.

### System Overview

```
Phase 0: Exact lm_head Recovery
  → Algebraic: solve lm_head.weight from logit observations (Carlini-style)
  → Or gradient-based with Jacobian constraint

Phase 1-K: Progressive Block Inversion (output → input)
  For each block B_i (i = N, N-1, ..., N-K+1):
    Step A: Jacobian Matching
      - Sample diverse inputs x_1, ..., x_M
      - Compute teacher logits and numerical Jacobians at block interface
      - Optimize B_i parameters to match Jacobians
    Step B: Logit Refinement
      - Fine-tune B_i parameters with logit MSE loss
    Step C: Suffix Refinement
      - Unfreeze recovered suffix, jointly fine-tune with small lr
    Step D: Symmetry Alignment
      - Align attention head permutations and FFN neuron permutations
      - Compute scale-invariant cosine similarity

Phase K+1: Depth Boundary Analysis
  → Continue inversion deeper, record where recovery breaks
  → Characterize the identifiability boundary
```

### Core Mechanism: Jacobian-Guided Block Inversion

**Why Jacobian matching works**: Consider a transformer block B with parameters θ_B that maps hidden state h_in to h_out = B(h_in; θ_B). The downstream recovered suffix S maps h_out to logits. For a given input x, we observe:
- Teacher logits: z_T(x) = S(B_T(h_in(x); θ_B^*); θ_S)
- Student logits: z_S(x) = S(B_S(h_in(x); θ_B); θ_S)

Standard approach: minimize ||z_T - z_S||² — but S is nonlinear, so many θ_B values give similar z_S.

**J-PSI approach**: Also match the Jacobian ∂z/∂h_in, which constrains how B transforms perturbations of its input. For diverse inputs, this over-determines θ_B up to:
- Attention head permutations (handled by alignment step)
- FFN neuron permutations within gated MLP (handled by alignment step)
- Scale factors absorbed by RMSNorm (handled by scale-invariant metrics)

**Per-block loss**:
```
L(θ_B) = α · E_x[||z_S(x) - z_T(x)||²]                           # logit match
        + β · E_x[||J_S(x) - J_T(x)||_F²]                          # Jacobian match
        + γ · R(θ_B)                                                  # L2 regularization
```

where J_S(x) = ∂z_S/∂h_in and J_T(x) = ∂z_T/∂h_in are computed via autograd (student) and numerical differentiation (teacher, since we only have logit access).

**Numerical Jacobian for teacher**: Since we only observe logits, compute ∂z_T/∂h_in by:
1. Feed input x, get h_in at the block interface (from student's forward pass up to that point — student's earlier layers are NOT recovered, so this uses the student's representation)
2. Perturb h_in → h_in + ε·e_j for each basis direction e_j
3. Re-run the teacher's recovered suffix from h_in + ε·e_j to get perturbed logits
4. J_T[:,j] ≈ (z_T(h_in + ε·e_j) - z_T(h_in)) / ε

**Wait — this requires knowing the teacher's intermediate activations**, which we DON'T have in black-box. We only have final logits.

**Revised approach**: Jacobian matching at the logit level only:
- Perturb INPUT tokens (not hidden states) by replacing token at position p with alternative tokens
- Observe how teacher logits change: finite-difference ∂z_T/∂x_p
- Match this against student's logit response to the same input perturbations
- This constrains the FULL student model (not just one block), but for a fixed recovered suffix, it tightens constraints on the current block

**Per-block loss (revised)**:
```
L(θ_B) = α · E_x[||z_S(x) - z_T(x)||²]                           # logit match
        + β · E_x,p[||Δz_S(x,p) - Δz_T(x,p)||²]                   # logit-sensitivity match
        + γ · R(θ_B)                                                  # L2 regularization
```

where Δz(x,p) = z(x') - z(x) for x' = x with token at position p replaced by a different token.

**Why logit-sensitivity matching strengthens recovery**: Standard MSE only constrains the student to produce the same logits. Logit-sensitivity matching also constrains HOW the student responds to input perturbations — this is a stronger constraint on internal parameters because different parameterizations that produce the same logits may respond differently to perturbations.

### Active Query Selection (Sub-component)

**Single method**: Gradient-diversity-based selection. From candidate pool, select inputs that maximize the diversity of gradients w.r.t. current block parameters. Approximated by selecting inputs with highest per-sample loss variance across recent mini-batches.

**Baseline**: Random selection from WikiText tokenized corpus.

### Permutation-Aware Identifiability Alignment

Before computing cosine similarity between recovered and ground-truth weights:
1. **Attention head alignment**: Find optimal permutation of attention heads (d_model/n_heads groups) that maximizes cosine similarity of Q,K,V,O weight matrices.
2. **FFN neuron alignment**: For gated MLP (up_proj, gate_proj, down_proj), find neuron permutation maximizing cosine similarity.
3. **Scale normalization**: Remove scale factors from RMSNorm parameters before comparison.

This ensures we measure true parameter recovery, not just functional equivalence.

### Training Plan

- **Phase 0 (lm_head)**: Gradient-based optimization, Adam lr=1e-3, 5K steps. Or algebraic recovery if vocabulary logits are full-rank.
- **Phase 1-K (block inversion)**: Per-block, Adam lr=1e-4, max 10K steps. Logit-sensitivity perturbation: 8 token replacements per input per step. Suffix refinement every 2K steps with lr=1e-5 for 500 steps.
- **Query pool**: 10K inputs from WikiText. Gradient-diversity selection every 50 steps.
- **Identifiability**: 5 independent runs from different random initializations.

### Failure Modes and Diagnostics

- **Suffix recovery breaks at depth K**: Record the boundary. This itself is a finding: "under full-logit access, the last K layers are identifiably recoverable."
- **Permutation alignment failure**: Use Hungarian algorithm on head/neuron similarity matrices. If alignment quality is poor, the block is not identifiable.
- **Numerical Jacobian instability**: Use multiple perturbation scales ε ∈ {1e-3, 1e-4, 1e-5} and check consistency.

### Novelty and Elegance Argument

**Closest work**: Carlini et al. (ICML 2024) — algebraic extraction of lm_head.
**Exact difference**: Carlini recovers ONE layer algebraically. J-PSI extends recovery to a contiguous multi-block suffix via Jacobian-guided optimization with identifiability analysis.
**Why focused**: Single algorithm, single contribution. No module zoo.

## Claim-Driven Validation Sketch

### Claim 1 (Primary): J-PSI recovers a contiguous suffix with cos_sim > 0.9
- **Minimal experiment**: Run J-PSI on Qwen3.5-0.6B, recover lm_head + last K blocks.
- **Baseline**: Standard KD (same query budget), block-coordinate distillation (no Jacobian matching).
- **Metric**: Permutation-aligned cosine similarity per layer, aggregated over 5 random inits.
- **Expected evidence**: cos_sim > 0.95 for lm_head, > 0.9 for last 2-4 blocks, with consistent results across inits.

### Claim 2: Jacobian matching is necessary for identifiable recovery
- **Ablation**: J-PSI without logit-sensitivity term (β=0) vs full J-PSI.
- **Metric**: Cross-init cosine similarity variance (low variance = identifiable).
- **Expected evidence**: Without Jacobian matching, different inits converge to different solutions; with it, they converge to the same.

### Claim 3 (Analysis): Recovery boundary and query efficiency
- **Experiment**: Continue inversion past the identifiable suffix, plot recovery vs depth.
- **Query efficiency**: Compare gradient-diversity selection vs random.
- **Expected evidence**: Sharp transition from high to low recovery at identifiable boundary. 3-5x query efficiency gain.

## Experiment Handoff Inputs
- **Must-prove claims**: Claims 1-2 are essential. Claim 3 is supporting analysis.
- **Must-run ablations**: Jacobian matching ablation (critical), suffix refinement ablation, query selection ablation.
- **Critical datasets/metrics**: WikiText for queries, permutation-aligned cosine similarity as primary metric.
- **Highest-risk assumptions**: Logit-sensitivity matching providing sufficient parameter-level constraint; permutation alignment working reliably.

## Compute & Timeline Estimate
- **Primary model**: Qwen3.5-0.6B — ~200 GPU-hours for 5-init full suffix recovery
- **Stress test**: Qwen3.5-4B output suffix only — ~100 GPU-hours
- **Ablations + analysis**: ~150 GPU-hours
- **Total**: ~450 GPU-hours (within budget)
- **Timeline**: 8 weeks
