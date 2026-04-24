🧿 oracle 0.9.0 — Lint for large language models.
Remote browser host detected: 172.20.208.1:9473
Routing browser automation to remote host 172.20.208.1:9473
Launching browser mode (gpt-5.4-pro) with ~24,249 tokens.
This run can take up to an hour (usually ~10 minutes).
Answer:
I read the TeX source directly.

### A. R2 fix verification

**R2-F1:** clean. `tab:warmstart` is gone.
**R2-F2:** clean. `tab:exp1`, `tab:exp3`, and `tab:per-matrix-main` are gone.
**R2-F3:** clean. The Llama subsection is now genuinely pure-logits (`\subsection{Cross-Architecture Validation: Llama-3.2-1B (pure-logits)}`) and reports the new near-zero result at lines 408–420.
**R2-F4:** clean. `tab:expanded` is gone from main; the story is carried by `tab:gramian` and `tab:exp2`.
**R2-F5:** partial. The abstract, contributions, theorem statement, and discussion now treat Thm 5 qualitatively, which is the right fix. But line 112 still says the paper “empirically validates the predicted rank bound and depth contraction on two architectures,” which is still too strong.
**R2-F6:** clean. The single `Discussion` section is structurally correct and contains exactly the four needed paragraphs.
**R2-F7:** clean. The attention-gauge code-gap disclosure is explicit and sufficient as disclosure.

### B. Score

**7/10**

This is now a coherent theory paper with one real blocker class left: claim hygiene. The big identity problem is fixed.

### C. Round 3 fix list

What still blocks `ready` is not the thesis; it is the remaining mismatch between what the paper now *is* and what a few sentences still *say*.

1. **Fix the last Thm 5 overclaim.**
   Change line 112. Do not say you empirically validate depth contraction. Say you validate the rank bound on Qwen and provide cross-architecture pure-logits evidence consistent with depth screening.

2. **Fix the Llama front-matter overclaim.**
   The abstract and contributions currently say Qwen **and Llama** validate the rank bound across the K×k grid. That is not what the experiments show. Qwen does the K×k Gramian sweep; Llama gives the cross-architecture pure-logits failure result. Rewrite that distinction cleanly.

3. **Delete the main-text historical aside at line 420.**
   “Our earlier oracle-regime Llama experiment (S-PSI variant…) is retained in the appendix for historical context” does not help you. It reintroduces dead-paper smell into the main body.

4. **Clean the appendix hard.**
   The appendix still carries old attack-paper DNA: stale notation (`r`, `\alpha,\beta,\gamma`, `\Delta z`), `sensitivity-augmented variant` in compute, the query-budget table, and the v1→v2 forensic audit. Those are the biggest remaining trust-tax items.

5. **Fix one factual inconsistency in Limitations.**
   The limitations bullet says all experiments use natural text from WikiText-103, but the Llama subsection uses 4096 random-token sequences. That needs one honest sentence.

### D. Specific questions

**Llama pure-logits result as headline cross-arch confirmation:** yes, it is adequate. The pair
`W_down cos = 3.9e-5`, `per-row = 2.3e-5` is emphatically noise-floor, and the oracle upper bound around `0.76` is exactly the sanity check you needed. For this paper’s thesis, one clean second architecture is enough. Do not oversell it into “universal failure,” and do not claim it also validates the Qwen K×k theorem sweep.

**Single Discussion section:** yes, the structure is clean. This is better than the old fragmented tail.

**Thm 5 softened to qualitative-only:** the oversell concern is mostly neutralized, but not fully. I do **not** want the theorem demoted further. I want the remaining prose drift removed. Keep the theorem in main. Just stop saying it is empirically validated.

**Appendix keep/cut:**
Keep: **Proofs**, **Symmetry Details**, **Implementation** (but make it lean).
Keep only if cleaned: **Notation Table**.
Optional: **Per-Matrix cosine table**; it is defensible, but not necessary. If you keep it, keep it because it explains the 0.13–0.14 bias band.
Cut: **Query Budget**, **Compute**, **v1→v2 Audit**. Those belong to the dead attack-paper lineage, not this theory submission.

### E. Verdict

**almost**

This is one more cleanup pass away. The remaining work is surgical: fix the front-matter overclaims, delete the S-PSI historical aside from main, and strip the appendix back to theory-supporting material only. Once that is done, this becomes `ready`.


18m20s · gpt-5.4-pro[browser] · ↑24.25k ↓1.04k ↻0 Δ25.29k
files=3
