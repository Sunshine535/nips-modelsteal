# Auto-Review Loop — α-Theory Path (nightmare / oracle-pro)

**Start**: 2026-04-19
**Terminal round**: 8
**Score progression**: 4/10 → 6 → 7 → 8 → 8.5 → 8.7 → 9.2 → **9.6**
**Final verdict**: ALMOST (submission-ready pending one human pass)
**Reviewer**: Oracle GPT-5.4 Pro (browser mode, ~10 min/round, each round reads the full repo)

## Mission

Execute Oracle Pro's Round-0 prune directive: convert the paper from a mixed
attack-and-theory submission into a pure theory / characterization paper (route α),
then iterate until Oracle Pro returns a positive verdict.

## Kill list (executed)

A1 S-PSI · A2 Moments CP · A4 logit-bias · A5 memory probing ·
A6 active query · A7 algebraic v2/v3/v4 · B2 Clone 2025 · B3 matched KD ·
E2 multi-model broad sweep · E4 functional KL.

## Retained

T1 Gramian + gauge null · T2 first-order observability criterion ·
T3 Gramian rank upper bound · T4 depth-screening decomposition (qualitative) ·
E1 K×k grid (Qwen, validates Thm 4) · E3 pure-logits cross-arch (Llama, cos 3.9e-5) ·
B1 Carlini SVD baseline · A3 Jacobian-FD (appendix-only).

## Paper shape

- Title: **Observable but Not Identifiable: Weight Non-Identifiability in Transformers from Black-Box Logit Access**
- Length: 595 lines, 266 KB PDF
- Main body: 6 sections (Intro, Related Work, Problem Setup, Symmetry & Observability Theory, Experiments, Discussion)
- Appendix: Notation Table, Proof of Thm 3, Symmetry Group Details, Implementation Details, Per-Matrix Cosine Breakdown

## Round-by-round log

### Round 0 — Oracle Prune Verdict
Oracle GPT-5.4 Pro (browser, 12m19s) issued KILL list and α-route directive.
Full text: `review-stage/ORACLE_PRUNE_VERDICT.md`.

### Round 1 — 4/10 NOT READY (first cut)
Abstract was cleaned but body still smuggled S-PSI via "sensitivity augmentation", W_V=0.27 "previously hidden positive finding", Transformer Tomography branding.
Fixes: deleted Experiment 1 (Alg init vs Random), Experiment 3 (Controls), warm-start, per-matrix dominance, gauge-invariant eval, sensitivity-augmentation.
Full text: `review-stage/ORACLE_ROUND1.md`.

### Round 2 — 6/10 ALMOST
Identity mostly fixed. tab:warmstart still in main; Llama subsection mislabeled oracle; appendix Query Budget / Computational Cost / v1→v2 Audit carried attack-paper DNA.
Fixes: deleted tab:warmstart, replaced Llama subsection with genuine pure-logits cross-arch result (W_down cos = 3.9e-5), scrubbed dead refs.
Full text: `review-stage/ORACLE_ROUND2.md`.

### Round 3 — 7/10 ALMOST
Front-matter overclaim: abstract still said Qwen+Llama jointly validate K×k grid.
Fixes: split contributions cleanly (Qwen = Thm 4 validation; Llama = cross-arch corroboration).
Deleted Query Budget / Compute / v1v2 Audit appendices.
Full text: `review-stage/ORACLE_ROUND3.md`.

### Round 4 — 8/10 ALMOST
Single Discussion section replaces Analysis + Limitations + Broader Impact + Conclusion. Thm 5 softened to qualitative-only with explicit "ρ estimation deferred". Attention-gauge code-gap disclosure added to Limitations.
Full text: `review-stage/ORACLE_ROUND4.md`.

### Round 5 — 8.5/10 ALMOST
Fixes: abstract line 87 cleanly separates Qwen role from Llama role. Notation table stripped of stale `r`, `α,β,γ`, `Δz`. Implementation Details dropped perturbation-strategy paragraph. Per-matrix caption scope-cleaned.
Full text: `review-stage/ORACLE_ROUND5.md`.

### Round 6 — 8.7/10 ALMOST
Fixes: K notation collision partially resolved — `K` now reserved for suffix-block count; `K_pos` for observation positions in notation table, abstract, contributions, Llama caption.
Draft-history residue ("v2", "v3 diagnostic", "in progress") replaced.
Full text: `review-stage/ORACLE_ROUND6.md`.

### Round 7 — 9.2/10 ALMOST
Fixes: abstract formula uses K_pos; limitations K≤128 → K_pos≤128; table column header K → K_pos; discussion K×k → K_pos×k.
Stray space copyedit `comparison} .` → `comparison}.`.
Full text: `review-stage/ORACLE_ROUND7.md`.

### Round 8 — 9.6/10 ALMOST (terminal)
Fixes: last four bare K references (lines 352, 354, 424, 426) migrated to K_pos.
`K_\text{pos}` vs `K_{\text{pos}}` styling normalized. "directions) ." copyedit done.
Full text: `review-stage/ORACLE_ROUND8.md`.

### Round 9 — attempted, browser attachment UI error
Oracle browser automation intermittently fails with "Sent user message did not expose attachment UI after upload". Substantive fix of the 4 remaining K refs was verified via grep (22 K_{\text{pos}} occurrences, 0 bare K in position contexts). Loop closed.

## Outstanding items (flagged non-blocking)

- **Depth-screening experiment for empirical ρ (Thm 5)**: running on remote (blocks 19-23 on Qwen2.5-0.5B, ~4h ETA). Deferred per Oracle R4 decision; Thm 5 currently stated as qualitative decomposition with empirical ρ explicitly deferred.
- **Attention gauge projection implementation**: code gap disclosed in Limitations; R7 concern is honest disclosure, not blocker.
- **Styling normalization pass**: K_\text{pos} ↔ K_{\text{pos}} mostly normalized but one human eye worth doing.

## Method Description

The paper characterizes first-order weight identifiability of transformer
suffix parameters from black-box logit access via the **suffix observability
Gramian** G(Q) = (1/|Q|) Σ J_x^T J_x defined on the gauge-quotiented parameter
space. We derive the continuous symmetry group of modern RMSNorm + gated-MLP
transformers (RMSNorm scaling, gated-MLP neuron scaling, GL(d_head) attention
V/O, RoPE-commuting Q/K), prove the gauge null Proposition 1, and establish
three theorems: a first-order observability criterion (Thm 3), a Gramian rank
upper bound (Thm 4), and a depth-screening decomposition (Thm 5). Empirical
validation of Thm 4 on Qwen2.5-0.5B via a K×k sketched-Gramian sweep confirms
full sketched rank across all tested cells and σ_max ∝ K_pos. Cross-architecture
validation on Llama-3.2-1B via pure-logits algebraic reconstruction yields
cos ≈ 4e-5 on an internal block, consistent with the observability barrier
predicted by the theory. We do not introduce a new weight-recovery method;
the framework is a characterization.

## Next steps (human)

1. Human proof read + one-hour styling polish
2. Incorporate depth-screening experimental ρ (if complete) or leave as future work
3. /paper-claim-audit zero-context final check
4. Compile submission PDF + supplementary materials
