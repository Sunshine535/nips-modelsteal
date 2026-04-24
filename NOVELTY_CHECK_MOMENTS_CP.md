# Novelty Check Report — Moments-CP Attack

**Date**: 2026-04-19
**Cross-model reviewer**: GPT-5.4 xhigh via Codex MCP (threadId 019da6d6-ae5b-7a43-a516-a756cb8e7c12)
**Target**: Paper positive result only (not the broader observability theorem)

## Proposed Method
Black-box LLM parameter extraction via CP decomposition of the 3rd-order
logit × hidden × hidden moment tensor `M_3 = E[z ⊗ h ⊗ h]`. Rank set to
`d_ff`, ALS iterations. Measured on Qwen2.5-0.5B, last block + `lm_head`.

## Core Claims
1. Per-column `W_lm` (output projection) recovered at cos = 0.81 (null 0.16) — **Novelty: MEDIUM-HIGH**. Closest: Carlini 2024 (subspace) and Finlayson 2024 §7.3 (proposed but not implemented).
2. Weak per-row signal on `W_v`, `W_up`, `W_q`, `W_gate` of last block — **Novelty: HIGH** for internal-layer leakage direction. No prior work claims this.
3. Uses 3rd-order tensor CP (not SVD) on LLM logit×hidden moments — **Novelty: MEDIUM-HIGH**. Anandkumar 2014 is general theory; no prior LLM application.

## Closest Prior Work

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| Carlini et al. *Stealing Part of a Production Language Model* (2403.06634) | 2024 | ICML Best Paper | Same target (output projection) | SVD → subspace only; we recover per-column via 3rd-order moment. |
| Finlayson et al. *Logits of API-Protected LLMs Leak Proprietary Information* (2403.09539) | 2024 | COLM | §7.3 proposes SVD-based softmax-matrix recovery | Proposal only, no empirical per-column evaluation; we implement + evaluate and extend to internal matrices. |
| Clone 2025 *Clone What You Can't Steal* (2509.00973) | 2025 | arXiv | Uses Carlini SVD + KD | Functional clone, not weight recovery. |
| Golowich et al. *Sequences of Logits Reveal the Low Rank Structure* (2510.24966) + *Provably Learning via Low Logit Rank* (2512.09892) | 2025 | arXiv | Spectral learning from logits | Recovers linear-surrogate ISAN with TV-distance guarantee; does NOT recover actual transformer weights; explicitly abstracts away attention + MLP. |
| Anandkumar et al. *Tensor Decompositions for Learning Latent Variable Models* | 2014 | JMLR | Parent theory for CP tensor methods | Targets HMMs / mixtures, not transformers. |
| Foerster et al. *Beyond Slow Signs in High-Fidelity Model Extraction* (2406.10011) | 2024 | NeurIPS | High-fidelity DNN extraction | Feed-forward ReLU (image), cryptanalytic; not LLM logits. |
| Beiser et al. *Data-Augmentation for Reverse-Engineering Weights* (2511.20312) | 2025 | NeurIPS WS | Black-box weight recovery | Not transformer, not tensor moments. |
| Rafi et al. *Revealing Secrets From Pre-trained Models* (2207.09539) | 2022 | arXiv | Transformer parameter inference | Relies on public-checkpoint similarity, not pure query-only logits. |
| Cryptanalytic DNN extraction (IACR 2023/1526, 2026/296, 2026/168) | 2023–26 | ePrint | Polynomial-time NN recovery | Small feed-forward ReLU nets only; not transformers. |
| arXiv 2506.22521 *Survey on LLM Model Extraction* | 2025 | arXiv | Field-wide taxonomy | Explicitly states "complete parameter recovery remains impractical for billion-parameter models"; no tensor-moment attack is surveyed. |
| arXiv 2411.15669 *Implicit High-Order Moment Tensor Estimation* | 2024 | arXiv | General moment-tensor theory | Mixtures / spherical Gaussians / positive-linear ReLU combos; not applied to LLM extraction. |
| arXiv 2506.06975 *Auditing Black-Box LLM APIs* | 2025 | arXiv | Behavioral audit | Rank-based uniformity test; no weight recovery. |

## Overall Novelty Assessment

- **Score**: **6/10**
- **Recommendation**: **PROCEED WITH CAUTION** — proceed and submit, but narrow the claim.
- **Key differentiator**: breaking the second-order gauge ambiguity of Carlini/SVD
  (where `L = W·H = (W·A)(A⁻¹·H)` is identifiable only up to invertible `A`) via CP-uniqueness
  on a 3rd-order moment. This is a **real technical delta**: per-column `W_lm` is not
  obtainable from matrix SVD alone; it requires a higher-order tensor with Kruskal-type
  uniqueness.
- **Primary reviewer risk**: "Finlayson et al. 2024 §7.3 already discussed recovering
  the softmax matrix from outputs via SVD." Needs explicit rebuttal — we must note
  that Finlayson's proposal is an SVD-based suggestion without per-column empirical
  evaluation, and our attack (a) uses a fundamentally different estimator (3rd-order
  moments + CP, not SVD), (b) provides the first empirical per-column success metric,
  (c) extends beyond the output layer.
- **Secondary reviewer risk**: "`cos ≈ 0.20` on internal matrices is too weak to call
  'recovery'." Must be framed as leakage evidence / preliminary signal, not as full
  weight theft.

## Suggested Positioning (Paper Framing)

Claim language for contributions / abstract:

> **First black-box higher-order moment attack on LLM logits that empirically recovers
> output-projection columns beyond subspace identifiability, with preliminary evidence
> of weaker leakage from some last-layer internal matrices.**

**DO**:
- State the Carlini SVD gauge ambiguity `L = W·H = (W·A)(A⁻¹·H)` explicitly and
  show that 3rd-order CP uniqueness breaks it.
- Cite Finlayson 2024 §7.3 as prior *proposal* and state the delta precisely.
- Frame internal-matrix signal as "statistically significant leakage evidence above
  a null baseline" rather than "recovery".
- Cite Golowich 2025 as complementary (behavioral vs structural recovery).

**DO NOT**:
- Claim "first black-box recovery of internal transformer weights". Internal cos ~0.20
  does not support this.
- Frame as "first tensor method for model extraction" without Anandkumar 2014 citation.
- Overstate the W_lm 0.81 without acknowledging Carlini already has 0.9999 *subspace*
  match — our delta is the per-column/per-row breakage of gauge, not raw subspace
  alignment.

## Go/No-Go

**GO**. The contribution is genuine and publishable; the primary failure mode is
narrative overclaim, not absence of novelty. Proceed to integrate the Moments-CP
attack into the paper with the tightened framing above.
