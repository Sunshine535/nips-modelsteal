🧿 oracle 0.9.0 — We bring receipts; they bring excuses.
Remote browser host detected: 172.20.208.1:9473
Routing browser automation to remote host 172.20.208.1:9473
Launching browser mode (gpt-5.4-pro) with ~32,406 tokens.
This run can take up to an hour (usually ~10 minutes).
Answer:
I could directly inspect the mounted `main.tex`, the prior prune verdict, and reviewer memory. The backup, proof JSON, and results directories named in the prompt were not mounted here; that does not change the outcome, because `main.tex` alone already fails Round 1.

## A. Verification of Round 0 concerns

1. **§5 S-PSI did not survive literally, but it survived substantively under a rename.**
   There is no literal `S-PSI` string in `main.tex`. But the method is still there as:

* `Sensitivity augmentation` (lines 273–278),
* `Algebraic (augmented)` and `sensitivity-augmented training loop` (365–369),
* the `β=0` no-sensitivity ablation and augmented rows (548–568, 622–623).

So: **the section title disappeared; the method did not.** This is cosmetic cleanup, not real excision.

2. **Abstract: front matter is cleaner, but the file still sells attacks in the body.**
   The abstract itself is mostly α-compliant. It does **not** pitch a new attack, and it does **not** claim positive recovery beyond Carlini’s known `lm_head` subspace result. It only says internal-block algebraic reconstruction stays at noise floor (lines 86–87).
   But this is immediately undercut by the body:

* intro still frames the paper around “weight recovery” and attack surface (98–104),
* experiments are still built around algebraic initialization / recovery tables (359–569),
* the main analysis advertises a “previously hidden positive finding” for `W_V` (579–584),
* conclusion still contains warm-start recovery narrative and functional-KL caveat (667–669).

So the **abstract is cleaner; the paper is not.**

3. **Contaminated scripts are not cited by filename in the paper body, but the quarantine is still cosmetic.**
   I do **not** see exact citations to `active_query_experiment.py`, `algebraic_recovery_v3_*`, or `matched_kd_baseline.py` in `main.tex`.
   But the paper still contains the contaminated experimental narrative: augmented algebraic init, oracle injection, warm-start barrier, v1→v3 forensic appendix, query-budget/compute appendix, and the old recovery framing. So the filenames are gone, but **the contamination remains in prose, tables, captions, and section structure.**

4. **Yes, Table 9 / `tab:gauge-invariant` with `W_V ≈ 0.266` is still in main text, and it is not consistent with the new thesis.**
   It is in the main `Analysis and Discussion` section (579–605). The table gives `W_V = +0.266` for both Blocks 22 and 23 (595, 600), and the text explicitly calls it a **“previously hidden positive finding”** (582).
   That is exactly the kind of narrative Round 0 told them to kill. For route α, this belongs in appendix at most, and probably should be dropped entirely from the main story.

## B. New α-thesis score

**4/10**

Why not lower: the title, abstract, intro scope sentence, and theorems are improved; the theory core is real.
Why not higher: the submitted TeX is **not actually the α paper**. It is still a mixed paper with old attack scaffolding, oracle-heavy experiments, renamed S-PSI, a positive `W_V` recovery claim in main text, old “Transformer Tomography” branding in caption and conclusion, and the functional-KL caveat still alive.

For NeurIPS 2026 Theory, **this is still a reject as written**.

## C. Remaining weaknesses, ranked, with minimum fix

1. **Identity failure: the paper is still mixed, not pure theory.**
   Minimum fix: hard-delete the attack-paper body. Remove Experiment 1, controls/ablations, warm-start, gauge-invariant positive-finding analysis, code-audit appendix, and query-budget/compute appendices. Rebuild main to the Round 0 cut target: theory + clean theorem-validation only.

2. **S-PSI survived by rename.**
   Minimum fix: delete the sensitivity-augmentation paragraph, all `Alg (augmented)` rows, the `β=0` ablation, and every “sensitivity-augmented training loop” reference. Historical audit, if kept at all, belongs in a quarantined appendix sentence.

3. **Main-text positive recovery claim (`W_V≈0.27`) poisons the α thesis.**
   Minimum fix: remove `tab:gauge-invariant` and lines 579–584 from the main paper. At most, append a short appendix note: continuous gauge can affect raw cosine interpretation; we do not center the paper on partial recovery.

4. **Scope contradiction: intro says “local, first-order, passive-query, pure-logit,” but the paper is still dominated by oracle experiments.**
   Minimum fix: move oracle-only material to appendix and keep only pure-logit results in main, or rewrite the scope honestly. For route α, the right move is the former.

5. **Theory/implementation mismatch is admitted in the paper.**
   At lines 232–233 and 780, the paper says the code does not explicitly project the attention gauge.
   Minimum fix: either implement full gauge projection for V/O and Q/K, or soften every quantitative claim that relies on exact gauge-quotiented dimension counting.

6. **Conclusion and captions still leak the old paper.**
   Minimum fix: delete “Transformer Tomography” in the overview caption (126) and conclusion opening (665), and delete the functional-KL paragraph (669) outright.

## D. Patterns to track for Round 2

I added them to reviewer memory: [updated `REVIEWER_MEMORY.md`](sandbox:/mnt/data/REVIEWER_MEMORY.md).

The key suspicions now are:

* front matter cleaned, body not;
* S-PSI reappearing as “sensitivity augmentation”;
* `W_V≈0.27` smuggled back into main;
* old title language surviving in captions/conclusion;
* oracle material still dominating a supposedly pure-logit theory paper;
* theorem/implementation mismatch on gauge projection;
* Thm 5 still being sold as validated before the contraction sweep is actually done.

## E. Verdict

**not ready**

This is not a close call. The supplied `main.tex` is still the old mixed paper wearing an α-style abstract. The cleanup did not reach the body.


17m43s · gpt-5.4-pro[browser] · ↑32.41k ↓1.46k ↻0 Δ33.87k
files=3
