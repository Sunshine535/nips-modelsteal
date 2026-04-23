# Idea Discovery Report

**Direction**: 在三阶矩CP + logit_bias h_final + Gramian理论基础上改进，达到/超过 Clone 2025 SOTA
**Date**: 2026-04-24
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline

## Executive Summary

**核心洞察**: cos 0.81 的 W_lm 列恢复无法功能性使用（KL 8.28），但 per-query h_final 恢复几乎完美（cos 0.977, KL 0.0002）。正确策略不是修复 W_lm 恢复精度，而是把 per-query h_final 的强信号输入 KD pipeline 作为额外监督信号。

**Top Idea**: Side-Channel Representation Distillation (SCRD) — 利用 logit_bias 侧信道恢复的 h_final 作为黑盒表征蒸馏目标，在标准 logit KD 之上提供 dense 的内部表征监督。

## Literature Landscape

### Model Stealing / Extraction (2024-2026)
| Paper | Year | Venue | What | Limitation |
|-------|------|-------|------|------------|
| Carlini et al. | 2024 | ICML Best Paper | SVD → W_lm subspace (cos 0.9999) | Gauge ambiguity, subspace only |
| Finlayson et al. | 2024 | COLM | logit_bias → full logit recovery | Proposed but未实现 per-column or KD |
| Clone 2025 | 2025 | arXiv | Carlini SVD + KD → 7.31% ppl↑ | Logit-only KD, no representation signal |
| Golowich et al. | 2025 | arXiv | Low logit rank provable learning | 不恢复权重，线性代理 |
| Cryptanalytic (ePrint) | 2023-26 | ePrint | ReLU NN polynomial extraction | 不适用于 transformer |
| KDD Survey | 2025 | KDD | 综述 | 称"完整参数恢复仍不实用" |

### Knowledge Distillation (2024-2026)
| Paper | Year | Venue | What | Gap |
|-------|------|-------|------|-----|
| Dasgupta & Cohn | 2025 | ICLR | Hidden state matching for KD (CKA) | **白盒**，需要直接访问 teacher hidden states |
| GAD (Ye & Dong) | 2025 | arXiv | GAN-based black-box KD | 无表征匹配，纯行为级 |
| Proxy-KD | 2024 | arXiv | Proxy model bridge | 无侧信道恢复 |
| Sparse Logit Sampling | 2025 | arXiv | Top-K logit KD efficiency | 无表征信号 |
| Flexible Feature Distillation | 2025 | arXiv | Feature-based KD for LLM | 白盒 |

### Transformer Symmetry/Identifiability
| Paper | Year | Venue | What | Relation to us |
|-------|------|-------|------|----------------|
| Maximal Gauge Symmetry | 2025 | ICLR 2026 withdrawn | 完整 gauge group | 与我们 §4.1 对称群推导重叠，他们更完整 |
| Deep Polynomial NN Identifiability | 2025 | NeurIPS 2025 oral | 多项式 NN 可辨识性 | 不同架构 |

### 关键空白 (我们的机会)
1. **无人做过**: 黑盒侧信道恢复 hidden state → 用于 KD 的表征蒸馏
2. **无人做过**: 结合 Carlini SVD + logit_bias h_recovery + KD 的三管齐下攻击
3. **无人做过**: 使用 observability Gramian 理论指导的 query selection for KD
4. **Dasgupta & Cohn 的白盒限制**: 他们的 hidden state matching 需要直接访问 teacher — 我们可以在黑盒下近似实现

---

## Ranked Ideas

### 🏆 Idea 1: SCRD — Side-Channel Representation Distillation [RECOMMENDED]

**One-line**: 利用 logit_bias 侧信道恢复 teacher 的 h_final，作为黑盒表征蒸馏目标，在 logit KD 之上提供 ALL-position dense 的内部表征监督。

**Mechanism**:
```
For each training batch:
  1. Query teacher API → get z(x) ∈ R^{T×V} at ALL T positions (standard)
  2. For each position t: h_hat_t = lstsq(W_eff_probe, z_t[probe_ids])  [LOCAL compute]
     where W_eff = Carlini-recovered output projection
  3. Student loss = α·KL(z_teacher/τ, z_student/τ)·τ²
                  + β·Σ_t MSE(Linear(h_student_t), h_hat_t)
                  + (1-α)·CE(labels)
  4. Linear projection absorbs Carlini's gauge ambiguity A
```

**Why it works**:
- h_final recovery is an overdetermined linear system (K_probe=2000 >> d=896) → cos 0.977
- Recovery at ALL T positions is FREE — autoregressive models return logits at all positions in one forward pass
- h_final carries RICHER information than logits (d=896 dim vector vs V=151,936 dim compressed through softmax)
- Under limited query budget (attack scenario), representation signal accelerates convergence

**Key novelty delta vs existing work**:
- vs Clone 2025: they use ONLY logit KD; we add representation distillation from black-box
- vs Dasgupta & Cohn (ICLR 2025): they need WHITE-BOX access to teacher hidden states; we RECOVER them via side channel
- vs Finlayson 2024: they proposed logit recovery; we use recovered information for DISTILLATION (novel application)

**Theoretical backing** (from our existing assets):
- Gramian: h_final is in the range of W_lm^T → full-rank recovery guaranteed
- Information theory: I(θ_teacher; z, h_final) > I(θ_teacher; z) — strictly more information
- Depth screening: explains why h_final is recoverable but deeper layers are not

**Expected metrics**:
- Variant A (logit-only KD, baseline): X% ppl degradation
- Variant C (logit + ALL-position h_final): should beat variant A significantly
- If ppl degradation < 7.31%: beats Clone 2025

**Risk**: In production APIs, logit_bias may be deprecated (OpenAI did this post-Carlini). Mitigation: frame as security demonstration; works on open-weight APIs (Together, Fireworks, etc.)

**Pilot feasibility**: <2 GPU-hours on single A100 (modify existing enhanced_kd_clone.py)

---

### Idea 2: Carlini-CP Transplant + SCRD [BACKUP/ABLATION]

**One-line**: Initialize student's lm_head with CP-recovered per-column W_lm (cos 0.81), add Carlini subspace regularizer, then train with SCRD.

**Mechanism**:
```
1. Carlini SVD → W_eff (subspace)
2. CP attack → W_cp (per-column cos 0.81)
3. Initialize student.lm_head ← W_cp (warm start)
4. Train with SCRD loss + L2 regularizer pulling lm_head toward Carlini subspace
```

**Novelty**: Clone 2025 freezes Carlini subspace; we initialize with CP per-column and let it fine-tune. Per-column init is new.

**Risk**: cos 0.81 init might not be better than random + KD, because KD is powerful enough to learn lm_head from scratch.

**Role**: Ablation variant for Idea 1. Shows combined value of all three attack channels.

---

### Idea 3: Gramian-Optimal Active Query Selection [THEORY EXTENSION]

**One-line**: Use the suffix observability Gramian to select queries that maximize information extraction, replacing random queries in KD.

**Mechanism**:
```
1. Score candidate queries by their Gramian contribution: score(x) = ||J_x||_F²
2. Select top-B queries from a large pool
3. Run KD with these high-information queries
```

**Novelty**: Nobody uses observability Gramian for query selection in KD/extraction.

**Risk**: Gramian tells about PARAMETER identifiability, not necessarily KD convergence quality. The correlation is indirect.

**Role**: Theoretical contribution that connects observability theory to practical attack efficiency. Could be a secondary contribution.

---

### Idea 4: Progressive Suffix Distillation [SPECULATIVE]

**One-line**: Distill layer-by-layer from output backward, using h_final recovery at each boundary.

**Risk**: Can only recover h_final at the LAST boundary (before lm_head). Cannot recover intermediate hidden states without intermediate logit access. KILLED — the math doesn't work for deeper layers.

---

## Eliminated Ideas

| Idea | Reason |
|------|--------|
| Improve CP to cos > 0.95 | Fundamental limit: tensor moment estimation from finite samples; 0.81 is already near optimal for 4096 queries |
| Use recovered W_lm directly for inference | KL=8.28. Softmax amplification makes this impossible without cos > 0.999 on ALL columns |
| Divide h_hat by g_final to recover h_norm | g_final has near-zero/negative values; destroys signal (diagnosed, cos drops from 0.977 to 0.001) |
| Jacobian-based parameter recovery | Shared architecture subspace problem; cannot distinguish teacher from random student |
| Memory probing for deep layers | 77× Gramian rank expansion but parameter cos still ~0.20 — too weak |
| Progressive suffix (Idea 4) | Can only recover at output boundary; no intermediate logit access |

## Novelty Assessment

### SCRD (Idea 1) — Detailed Novelty Check

| Aspect | Closest Work | Our Delta | Novel? |
|--------|-------------|-----------|--------|
| Black-box h_final recovery | Finlayson 2024 (logit recovery) | We APPLY it for KD, they only proposed logit extraction | ✅ YES |
| Hidden state matching in KD | Dasgupta & Cohn ICLR 2025 | We do BLACK-BOX via side channel; they need WHITE-BOX | ✅ YES |
| Logit KD for model stealing | Clone 2025 | We ADD representation signal; they are logit-only | ✅ YES |
| ALL-position recovery from single forward pass | Nobody | Novel observation: autoregressive logits enable free dense recovery | ✅ YES |
| Gauge-absorbing linear projection | Nobody in KD context | Novel technique for handling Carlini ambiguity in distillation | ✅ YES |

**Novelty score: 8/10 — HIGH.** The individual components exist (Finlayson logit recovery, Dasgupta hidden state matching, Clone 2025 KD), but the COMBINATION and BLACK-BOX application are entirely new.

**Primary reviewer risk**: "This is just standard representation distillation with a side-channel twist."
**Rebuttal**: The side-channel is the ENTIRE point. Nobody has shown that black-box APIs leak enough information to enable representation distillation. This is a qualitative capability gap: logit-only → logit + representation.

## Refined Proposal

See `refine-logs/FINAL_PROPOSAL.md` for detailed method and `refine-logs/EXPERIMENT_PLAN.md` for experiment design.

## Next Steps
- [ ] Modify `enhanced_kd_clone.py` to implement SCRD (ALL-position h_final + linear projection)
- [ ] Deploy on remote server when SSH recovers
- [ ] Run 5 variants: Teacher baseline, Random student, Logit-only KD, SCRD-last-pos, SCRD-all-pos
- [ ] After results: /auto-review-loop to iterate paper until submission-ready
