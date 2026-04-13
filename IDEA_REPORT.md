# Idea Discovery Report

**Direction**: Progressive Parameter Inversion — recovering LLM weights from black-box logit access
**Date**: 2026-04-08 → 2026-04-09
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine

## Executive Summary

After surveying 30+ recent papers, generating 10 ideas via GPT-5.4, deep novelty checking, and two rounds of external review, we converge on:

**🏆 Recommended Idea: "Transformer Tomography: Observability-Governed Weight Recovery from Black-Box Logit Access"**

Core thesis: The suffix observability Gramian G(Q) on the symmetry-quotiented parameter space governs whether, how efficiently, and how completely transformer weights can be recovered from black-box logit access.

Key evidence: 3 of 6 contributions are completely novel (no prior work); remaining 3 are novel for transformers. GPT-5.4 reviewer scored 5/10 initially but identified clear path to strong accept.

## Literature Landscape

### Key Competitors (2024-2026)
| Paper | Venue | What They Do | Gap S-PSI Fills |
|-------|-------|-------------|-----------------|
| Carlini et al. | ICML'24 | Extract output projection for <$20 | Only 1 layer; no theory |
| Liu & Moitra | AAAI'25 | Polynomial-query recovery for low-rank LMs | Requires low-rank assumption |
| "Are Logits All You Need?" | arXiv'25 | Topological model stealing | Theory only, no weight recovery |
| SPLITS | ICLR'25 | Provably robust logit perturbation defense | Must evaluate against |
| IDEalS | arXiv'24 | Bypass perturbation defenses with generative models | Complementary attack |
| Shamir et al. | Crypto'24 | Polynomial-time ReLU extraction | ReLU-only, not transformers |
| "From Deep to Shallow" | arXiv'25 | Deep weight non-identifiability | Binary result; we give quantitative |
| Classes et al. | arXiv'25 | Single-input ReLU recovery | ReLU-only |

### S-PSI's 6 Unique Gaps
1. ✅ Progressive multi-layer recovery (not just output projection)
2. ✅ No low-rank assumption
3. ✅ Sensitivity-guided query strategy
4. ✅ KD → Weight Recovery continuum
5. ✅ Budget vs precision scaling curves
6. ✅ Evaluation against latest defenses (LoDD, SPLITS)

## Ranked Ideas

### 🏆 Idea 1: Transformer Tomography — RECOMMENDED
- **Best Paper Potential**: 9-10/10
- **Novelty**: 7.5/10 (3/6 contributions completely novel)
- **Reviewer Score**: 5/10 → projected 7-8/10 after fixes
- **Risk**: Medium-High (theory must be tight)

**Core Contributions:**
1. (Theory) Suffix observability Gramian G(Q) = (1/N)Σ J_i^T J_i on symmetry-quotiented space
2. (Theory) Local identifiability when projected G full-rank; error ∝ 1/√λ_min
3. (Method) Sketched Gauss-Newton algebraic initialization via JVP probes
4. (Method) Fisher-optimal query design maximizing λ_min(G)
5. (Experiments) Gramian spectrum predicts recovery quality (R²>0.8)
6. (Experiments) 10x query efficiency gain over random-init S-PSI

**Novelty Assessment:**
| Contribution | Novelty | Closest Work | Differentiation |
|-------------|---------|-------------|-----------------|
| Suffix Observability Gramian | Medium-High | Neurocomputing'24 (general NN) | Transformer-specific + symmetry quotient |
| Identifiability theorem | Medium-High | ReLU identifiability line | First for attention+softmax+LayerNorm |
| Deep impossibility boundary | Medium | "Deep to Shallow" (2025) | Quantitative (λ_min) vs binary |
| **Algebraic init via JVP** | **HIGH** | No prior work | Completely novel |
| **Fisher-optimal queries** | **HIGH** | No prior work | Completely novel |
| **Gramian predicts recovery** | **HIGH** | No prior work | Completely novel |

### Idea 2: S-PSI Baseline (current method) — BACKUP
- **Best Paper Potential**: 6/10
- **Novelty**: 6/10
- **Status**: Already implemented, code review score 9/10
- **Risk**: Low (can always run as-is)
- **Use case**: If Transformer Tomography theory doesn't land, publish S-PSI alone

### Eliminated Ideas
| Idea | Reason |
|------|--------|
| Amortized Meta-Inversion | Too high risk, family-specific generalization concern |
| Checkpoint Atlas Sensing | Too benchmark-specific |
| Counterfactual Causal Inversion | Hard to make airtight |
| Pseudo-Oracle Prompt Control | 80-90 GPU-hours, too expensive for 100hr budget |

## Refined Proposal Summary

### Threat Model
- Black-box access to full-vocabulary logits
- Attacker knows architecture (same-family)
- Arbitrary prompts allowed
- Target: suffix block weights up to permutation/scale symmetry

### Complete Symmetry Group (Qwen3.5-0.8B)
- Head permutations: S_H (or wreath product for GQA)
- FFN neuron permutations: S_m with diagonal scaling
- RMSNorm scale absorption: (R×)^d per norm layer
- Value-space basis: GL(d_v) per KV group
- Q/K within-head: signed permutation (if qk_norm present)
- NO global residual-stream GL(d) (broken by skip + RMSNorm + tied embeddings)

### Method (4 phases per block)
1. **Cache**: Precompute teacher logits + perturbation responses
2. **Algebraic Init**: Sketched Gauss-Newton via k JVP probes → reduced least-squares → init along observable directions
3. **Fisher Query Design**: Optimize synthetic prompts to maximize λ_min(projected G)
4. **Gradient Refinement**: Short S-PSI optimization from algebraic init

### Experimental Plan
See EXPERIMENT_PLAN.md

## Reviewer Feedback (GPT-5.4 AC)

### Strengths
1. Core lens (observability/conditioning on quotiented space) is exactly right
2. Method story unusually coherent (algebraic init + optimal queries)
3. Experimental instincts good (wrong-teacher, phase transitions, defense)

### Weaknesses & Fixes
1. **Theory too local** → Add finite-query remainder analysis
2. **Symmetry group incomplete** → Full enumeration done (see above)
3. **Algebraic init not credible** → Replaced tensor decomp with sketched Gauss-Newton
4. **Single model** → Add Llama-3.2-1B as second model
5. **Too much content** → Compress defense section, focus theory+method

### Clear Accept Condition
Recover at least one mid-depth attention+MLP block where Fisher queries causally improve recovery, on two model families, beating strong distillation baselines.

## Next Steps
- [ ] Implement src/gramian.py, src/symmetry_gauge.py, src/algebraic_init.py
- [ ] Extend SPSIConfig with init_method, gramian config
- [ ] Deploy to server (4×A100-80GB)
- [ ] Run experiments per EXPERIMENT_PLAN.md
- [ ] /auto-review-loop to iterate until submission-ready
