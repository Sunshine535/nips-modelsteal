🧿 oracle 0.9.0 — Cleanup crew for model-made messes—now with citations.
Remote browser host detected: 172.20.208.1:9473
Routing browser automation to remote host 172.20.208.1:9473
Launching browser mode (gpt-5.4-pro) with ~26,051 tokens.
This run can take up to an hour (usually ~10 minutes).
Answer:
I directly read `main(1).tex`, `REVIEWER_MEMORY(1).md`, and `ORACLE_ROUND1.md`. `ORACLE_PRUNE_VERDICT.md` was not mounted in this session, but reviewer memory reproduces its operative directives.

## A. Verification of Round 1 fixes

**F1 — partially complete.**
`Experiment 1` and `Experiment 3` headers are gone. But `tab:warmstart` is still in the main paper at lines 415–433, so this fix is not clean. Also the query-pool paragraph still references deleted `tab:exp1` and `tab:exp3` at line 337. The explicit per-matrix “LayerNorm dominance” paragraph is gone from main, but the same idea still appears in captions/prose (“driven by LayerNorm,” e.g. lines 126, 453, 457).

**F2 — partially complete.**
The main-body sensitivity-augmentation paragraph, augmented rows, and β=0 ablation are gone. But appendix residue remains: the notation table still carries `r` and `\alpha,\beta,\gamma` for algebraic/sensitivity losses (lines 557–559), and the compute appendix still mentions a “sensitivity-augmented variant” (line 691). So the method is no longer in the main story, but it is not fully excised from the document.

**F3 — confirmed clean.**
I do not see `tab:gauge-invariant`, `W_V≈0.27`, or the old “previously hidden positive finding” claim in main.

**F4 — partially complete.**
This is the biggest remaining scope problem. The main paper still leans heavily on oracle material: the threat model still foregrounds the boundary-state oracle (lines 168–176), the Gramian sweep is oracle, `tab:expanded` is oracle, and the Llama subsection is explicitly oracle (lines 453–457). The pure-logit thesis is now the headline, but oracle still carries too much empirical weight.

**F5 — confirmed clean.**
`Transformer Tomography` branding appears gone.

**F6 — confirmed clean.**
The functional-KL / “functional not parametric” paragraph is gone.

**F7 — confirmed clean as disclosure.**
The attention-gauge-projection disclaimer is explicit (lines 232–233 and 489). That fixes the honesty problem, not the implementation gap.

## B. New score

**6/10**

This is a real improvement from Round 1. It now reads like a theory paper with residual attack-paper organs still attached, not like the old attack paper in disguise. The theory core is coherent. What blocks readiness now is mostly pruning discipline and claim/evidence consistency, not the existence of the core thesis.

## C. Remaining weaknesses, ranked, with minimum fix

1. **Cross-arch evidence is internally inconsistent.**
   The abstract/contributions/analysis sell a pure-logit cross-architecture negative result, but the Llama subsection is labeled **oracle regime** (lines 453–457), and line 474 calls it “pure-logits cross-architecture validation.”
   **Minimum fix:** either rerun Llama in pure logits and say so consistently, or move Llama to appendix and stop using it to support the pure-logit headline.

2. **The prune is mechanically incomplete.**
   `tab:warmstart` is still in main, and dead refs remain (`tab:exp1`, `tab:exp3`, `tab:per-matrix-main`).
   **Minimum fix:** delete warm-start, scrub dead refs, and do a final grep for removed labels before freeze.

3. **Experiments are still too wide for the theory route.**
   Main currently has 2 figures and 5 tables before appendix. `tab:exp2`, `tab:expanded`, and the warm-start table are still carrying attack-paper logic.
   **Minimum fix:** in main, keep only:
   (i) the K×k Gramian validation of the rank bound, and
   (ii) one compact pure-logit negative result table.
   Move the rest to appendix.

4. **Theorem 5 is still oversold empirically.**
   The paper claims empirical validation of depth screening/depth contraction (lines 87, 112, 474), but no contraction/ρ experiment is actually included.
   **Minimum fix:** either add the blocks 19–23 contraction result, or stop claiming empirical validation of Thm 5 and present it as theory/qualitative decomposition only.

5. **Appendix contamination remains visible.**
   The notation table, query-budget appendix, compute appendix, and v1→v2 audit still preserve old attack-pipeline DNA.
   **Minimum fix:** quarantine or cut aggressively. For a theory submission, these appendices should be much thinner.

6. **Attention-gauge code gap is disclosed but not neutralized quantitatively.**
   **Minimum fix:** either implement the missing projection, or soften any exact quantitative rhetoric that depends on precise gauge-quotient dimension counting.

## D. Specific comments

**Is Experiments slim enough?**
No. Slimmer than Round 1, yes. Slim enough for the theory thesis, no. I still want it narrower.

**Llama cross-arch: keep or appendix?**
As written: **appendix**. It helps only if it is a clean pure-logit corroboration. In its current oracle/Procrustes form, it muddies the thesis more than it strengthens it.

**Merge Analysis + Limitations + Broader Impact + Conclusion?**
Yes, mostly. Merge **Analysis + Limitations + Conclusion** into a single `Discussion`. Keep `Broader Impact` separate only if you need a standalone venue-compliance section; otherwise fold it in too. The current tail is too fragmented.

**Does the attention-gauge disclaimer suffice?**
It suffices as a disclosure. It does **not** suffice to support strong quantitative “validated gauge-quotient” rhetoric. So: disclosure is enough for honesty; code is not strictly required before submission **if** you soften the quantitative claims. If you keep exact-count language, you need the implementation.

## E. Patterns for Round 3 memory

* Warm-start/orphan tables can survive after section deletion.
* Dead refs are a contamination canary: `tab:exp1`, `tab:exp3`, `tab:per-matrix-main`.
* “Pure-logit” claims can silently drift onto oracle-only subsections.
* Appendix can preserve old-paper DNA through notation, query budget, compute, and audit sections.
* Thm 5 should not be called empirically validated until the contraction/ρ result is actually in the paper.

## F. Verdict

**almost**

Not submit-as-is. But unlike Round 1, this now looks one hard surgical pass away rather than fundamentally mis-scoped.


14m46s · gpt-5.4-pro[browser] · ↑26.05k ↓1.52k ↻0 Δ27.57k
files=3
