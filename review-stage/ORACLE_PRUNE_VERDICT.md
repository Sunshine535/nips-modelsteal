🧿 oracle 0.9.0 — Detached runs, tethered results.
Remote browser host detected: 172.20.208.1:9473
Routing browser automation to remote host 172.20.208.1:9473
Launching browser mode (gpt-5.4-pro) with ~1,268 tokens.
This run can take up to an hour (usually ~10 minutes).
Answer:
1. **One winning thesis**

**Black-box logit access does not generically identify transformer weights; a suffix observability Gramian characterizes the only locally recoverable directions and predicts when pure-logit extraction must fail.**

2. **KILL LIST**

**Theory**

* **T1 — KEEP**: Foundational definition plus gauge null; without this, there is no paper.
* **T2 — KEEP**: Core local identifiability theorem; this is the mathematical heart.
* **T3 — KEEP**: Quantitative rank ceiling is the strongest reviewer-facing result.
* **T4 — KEEP**: Gives architectural bite and practical intuition via depth screening.

**Attacks**

* **A1 — KILL**: Your “main method” fails in the stated regime; oracle-state success is off-threat-model and narrative poison.
* **A2 — KILL**: Interesting research, bad submission fit. Partial, unstable, poor reconstruction, and it changes the paper into an unfinished attack paper.
* **A3 — DEMOTE_TO_APPENDIX**: Useful corroborating failure mode, but secondary once theory explains the barrier.
* **A4 — KILL**: Broken implementation. Dead.
* **A5 — KILL**: Weak effect, noisy story, no clean theorem link.
* **A6 — KILL**: Contaminated. Non-negotiable.
* **A7 — KILL**: Contaminated / superseded by the honest zero result.

**Baselines**

* **B1 — KEEP**: Excellent contrast: subspace/function recovery can coexist with weight non-identifiability.
* **B2 — KILL**: Incomplete = useless at submission time.
* **B3 — KILL**: Tied to dead S-PSI framing and loses to KD anyway.

**Empirical infrastructure**

* **E1 — KEEP**: Best theorem-validation figure in the whole project.
* **E2 — KILL**: Incomplete sweep only creates reviewer demands you cannot satisfy.
* **E3 — KEEP**: Essential cross-arch confirmation that pure-logit inversion fails where theory says it should.
* **E4 — KILL**: Behavioral KL is not the scientific question.

3. **Paper route**

**Choose (α) Pure theory / characterization paper.** More precisely: **Main Track – Theory**, not Negative Results.

I **agree with Codex**. The current mixed paper is a reject because it sells a method that fails under the stated threat model. NeurIPS 2026 explicitly welcomes in-depth analyses of limitations, has a dedicated **Theory** contribution type whose primary standard is correctness rather than empirical SOTA, and treats **Negative Results** as a separate, high-bar category for surprising negatives. Also, the contribution type is locked after submission, so a confused identity is fatal. Your proof stack fits Theory. Your attack stack does not. ([NeurIPS][1])

4. **Engineering changes needed**

* **Retitle** away from “tomography” and away from any reconstruction vibe. Use “non-identifiability,” “limits,” or “suffix observability.”
* **Rewrite abstract and intro** so the paper contributes exactly three things: framework, theorems, empirical validation.
* **Delete §5 as a method section.** There is no main attack. Replace it with a tight empirical section validating T2–T4.
* **Purge all contaminated and broken branches** from paper, supplement, captions, and released repo branch.
* **Scope claims brutally**: pure logits, black-box, local/first-order. No hidden states. No continuous embeddings. No “weight recovery attack.”
* **Do not show A2 numbers anywhere** in this submission. One sentence in limitations is enough: higher-order leakage is outside scope.
* **Rebuild every retained figure from a clean branch** with script-to-figure provenance.

5. **Cut target**

**6 sections. 4 figures. 2 tables.**

That means:

* main text = setup, framework, theorems, validation, implications, limitations;
* appendix = proofs, A3, extra sanity checks, reproducibility details.

6. **Top 3 reviewer objections and fixes**

* **“This is only local/first-order; higher-order adaptive recovery might still work.”**
  Fix: say exactly that yourself. In title, abstract, theorem statements, and limitations. Do not fake a global impossibility theorem.

* **“Gauge non-identifiability is obvious; what is actually new here?”**
  Fix: do not oversell T1. Sell **T2 + T3 + T4**: the observability criterion, quantitative rank ceiling, and depth screening. That is the novelty.

* **“I don’t trust the experiments because contaminated branches existed.”**
  Fix: total excision. Clean reruns only. One release branch. One manifest. No mention of the fake breakthrough.

7. **Ship now vs follow-up vs discard**

* **Ship now**: **T1–T4, E1, E3, B1**, plus **A3 appendix-only** if it is clean and tight.
* **Follow-up paper**: **A2 only**, after you can stabilize CP, explain what component it recovers, fix the absurd reconstruction pathology, and replicate cleanly across models.
* **Discard**: **A1, A4–A7, B2, B3, E2-as-broad-sweep, E4**.

**Bottom line:** Codex was right. The only accept-shaped object here is a **theory paper on weight non-identifiability from black-box logits**. Moments CP is not the “one correct route” for this submission; it is a separate paper waiting to exist.

[1]: https://neurips.cc/Conferences/2026/CallForPapers "Call for Papers 2026"


12m19s · gpt-5.4-pro[browser] · ↑1.27k ↓1.27k ↻0 Δ2.54k
