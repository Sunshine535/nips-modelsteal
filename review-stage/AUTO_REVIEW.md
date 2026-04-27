# Auto Review Loop: C-DART

**Target**: NeurIPS 2026 Main Track (Methods)
**Difficulty**: nightmare
**Reviewer**: oracle-pro (GPT-5.4 Pro, repo-level access)
**Started**: 2026-04-28 01:55 CST

---

## Round 1 (2026-04-28 01:55 CST)

### Assessment (Summary)
- Score: 2/10
- Verdict: not ready
- Key criticisms:
  1. Missing raw evidence for 3/4 models (only Qwen2.5 in local repo)
  2. ce_only is worse than frozen reference on Qwen2.5 — unfair baseline
  3. KL normalization inconsistency between baselines and C-DART
  4. Scope overclaiming ("black-box extraction" vs "known-base delta extraction")
  5. Narrow experiments: one verified model, one K, one gap
  6. tdart_no_adaptive == cdart_no_censor (ablation inflation)
  7. No downstream eval, no task suites
  8. Missing manifests/provenance

<details>
<summary>Click to expand full reviewer response</summary>

Score: 2/10

Verdict: not ready

Verified claims:
- Qwen2.5-0.5B cdart_gate_v1 numbers mostly match the report
- On that one model, cdart_full beats ce_only, BiLD, and cdart_no_censor across seeds
- Evaluation KL is computed as token-mean KL(teacher || student) over full teacher logits
- Censoring is actually active in cdart_full; logs show nonzero censor loss

Unverified/false claims:
- "4 models × 3 seeds" not verifiable — Llama and Qwen3 results not in repo
- Cross-model claims are report-only
- "53-66% gap closure" inflated for Qwen2.5 if frozen reference is the fair baseline
- "Standard top-K KD universally harmful" only verified on 1 model, confounded by normalization bug

Weaknesses:
1. Missing raw evidence for 3/4 models
2. Baseline unfair: ce_only worse than doing nothing on Qwen2.5
3. KD baselines not normalized consistently
4. Scope overstated — assumes known base weights
5. Too narrow experiments
6. Incomplete ablations (tdart_no_adaptive == cdart_no_censor)
7. No downstream eval
8. Weak provenance

</details>

### Actions Taken
1. **Pulled all remote results** — 12/12 result files now in local repo (cdart_llama_1b, cdart_llama_3b, cdart_qwen3_0.6b)
2. **Added frozen reference baseline** — computed honest gap closure vs reference: 58-90% (stronger than vs ce_only on 3/4 models)
3. **Fixed KL normalization** — full_logit_upper and strict_topk_kd now use sequence_kl_loss (token-mean) consistent with all other variants
4. **Reframed scope honestly** — narrative report now states "requires knowing the base model, not arbitrary black-box"
5. **Documented tdart_no_adaptive == cdart_no_censor** — acknowledged in narrative as single ablation, not two
6. **Updated narrative report** — dual-baseline comparison, honest limitations section

### Results
- Gap closure vs frozen reference: 58.4% (Qwen2.5), 90.0% (Qwen3), 73.5% (Llama-1B), 75.2% (Llama-3B)
- ce_only DEGRADES from reference only on Qwen2.5 (small gap model)
- On other 3 models, both ce_only and cdart improve over reference, but cdart is much better

### Status
- Continuing to Round 2
- Difficulty: nightmare
