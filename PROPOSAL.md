# Research Proposal: Progressive Parameter Inversion

## Research Question

**Can we recover the actual weight parameters of a black-box LLM — not just replicate its behavior — using only API query access to output logits?**

## Problem Statement

Current model stealing attacks fall into two categories:
1. **Behavioral cloning**: Train a student to mimic teacher outputs. The student behaves similarly but has entirely different internal representations.
2. **Layer-specific extraction**: Recover specific layers (e.g., embedding projection) through algebraic attacks. Limited to output-adjacent components.

Neither approach recovers the full set of internal weight matrices. We ask: is this fundamental, or can iterative refinement bridge the gap?

## Hypotheses

- **H1 (Feasibility)**: Given sufficient queries, gradient-based optimization on hypothesized weights can converge to the teacher's true parameters for output-adjacent layers (cosine similarity > 0.9).
- **H2 (Depth Decay)**: Recovery accuracy degrades with layer depth following a predictable curve (our "Parameter Leakage Scaling Law").
- **H3 (Active Queries)**: Information-theoretic query selection achieves equivalent recovery with 5-10× fewer queries than random sampling.
- **H4 (Defense Gap)**: Standard API defenses (logit rounding, temperature perturbation) significantly degrade recovery of deeper layers while preserving usability.

## Method: Progressive Layer-wise Parameter Inversion (PLPI)

### Stage 1 — Behavioral Distillation (Warm Start)

Standard knowledge distillation from teacher logits. This gives us:
- A student that approximates teacher behavior
- An initialization point for parameter search
- Calibration data for understanding teacher output distributions

### Stage 2 — Layer-wise Inversion (Core Contribution)

For each layer L, from output (L=N) to input (L=1):

1. **Fix** all previously recovered layers {θ_{L+1}, ..., θ_N}
2. **Parameterize** layer L with free variables θ_L (initialized from distilled student)
3. **Optimize** θ_L to minimize:

   ```
   L(θ_L) = E_x [ ||f(x; θ_fixed, θ_L) - teacher(x)||² ]
            + λ · regularization(θ_L)
   ```

4. **Active Query Selection**: At each optimization step, select next query batch:
   ```
   x* = argmax_x  H(θ_L | x, teacher(x))
   ```
   Approximated via gradient magnitude: inputs where the gradient w.r.t. θ_L is largest are most informative.

5. **Convergence check**: When weight updates < ε, move to layer L-1.

### Stage 3 — Evaluation

- **Weight-space metrics**: Cosine similarity, L2 distance between recovered and true weights per layer
- **Function-space metrics**: Output match rate, KL divergence on held-out data
- **Downstream metrics**: Task accuracy on standard benchmarks

### Stage 4 — Defense Evaluation

Test robustness of PLPI under:
- **Logit rounding**: Round output logits to k decimal places
- **Output perturbation**: Add calibrated Gaussian noise to logits
- **Temperature variation**: Randomly perturb softmax temperature
- **Watermarking**: Embed detectable signatures in output distribution

## Competition Landscape

| Paper | Year/Venue | What They Extract | Limitation |
|-------|-----------|-------------------|------------|
| Carlini et al. "Stealing Part of a Production LLM" | ICML 2024 | Embedding projection layer | Output layer ONLY, <$20 cost |
| "Clone What You Can't Steal" | arXiv 2025 | SVD-based logit leakage, 97.6% hidden geometry | Behavioral clone, NOT weight recovery |
| StolenLoRA | 2025 | LoRA adapter weights, 96.6% extraction | Only targets LoRA adapters, not base model |
| "Aggressive Compression Enables Weight Theft" | 2026 | Full model via 16-100x compression | ASSUMES already having weight access |
| "Beyond Slow Signs" | NeurIPS 2024 | Sign-based extraction, 14.8x efficiency | Sign bits only, not full precision weights |
| Theoretical bounds | Various | O(d²) queries for single-head attention | Theory only, single mechanism |

### Our Differentiation

**ALL** existing attacks either:
- (a) Extract only specific layers (embedding/output)
- (b) Do behavioral cloning without weight recovery
- (c) Require direct access to weights

**We are the first to attempt progressive recovery of internal weight matrices from black-box query access alone.**

## Expected Contributions

1. **PLPI Algorithm**: Novel layer-wise parameter inversion with convergence analysis
2. **Active Query Selection**: Information-theoretic framework for maximally informative queries
3. **Parameter Leakage Scaling Law**: Empirical law relating query budget to parameter recovery
4. **Defense Analysis**: First systematic evaluation of API defenses against parameter inversion
5. **Implications**: Quantified risk assessment for LLM API providers

## Experimental Plan

- **Teacher**: Qwen3.5-9B (we have ground-truth weights for evaluation)
- **Students**: Qwen3.5-0.8B / 2B / 4B
- **Query budgets**: 10K, 50K, 100K, 500K, 1M
- **Baselines**: Standard KD, Carlini-style extraction, behavioral cloning
- **Hardware**: 8x A100-80GB, ~500 GPU-hours total

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Inversion doesn't converge for deep layers | Document depth-recovery curve as negative result; still publishable as scaling law |
| Teacher architecture must match student | Test with architecture mismatch as ablation |
| Query budget too large for practical attack | Compare with existing attacks; frame as theoretical contribution |
| Ethical concerns about enabling model theft | Pair with defense evaluation; responsible disclosure |

## Ethical Considerations

This work studies vulnerabilities in LLM APIs to inform defense design. We:
- Only use open-weight models (Qwen) where we simulate black-box access
- Do not attack any production API
- Propose and evaluate concrete defenses
- Follow responsible disclosure practices
