# S-PSI Auto Review Log

## Round 1 (2026-03-30)

### Assessment (Summary)
- Score: 5/10
- Verdict: Not ready
- Key criticisms:
  1. **Fatal**: No permutation-aware alignment — per-matrix cosine similarity unreliable without it
  2. **Fatal**: No wrong-teacher falsification — may be function matching, not identifiability
  3. **High**: No held-out generalization — overfitting to cached query pool possible
  4. **High**: Pure-logits may collapse — needs honest framing
  5. **Medium-High**: Query efficiency underdeveloped — no recovery-vs-budget curves
  6. **Medium**: Only one model (0.6B) — narrow external validity

<details>
<summary>Click to expand full reviewer response</summary>

**Score**

`5/10` for NeurIPS/ICML as currently described. The question is strong and the method is interesting, but the evidence package is still below top-venue standard for an identifiability paper.

**Critical Weaknesses**

1. **Fatal: the core weight-recovery metric is not yet symmetry-aware.**
Without permutation-aware alignment over attention heads and FFN neurons, per-matrix cosine similarity is not a reliable identifiability metric.
Minimum fix: implement Hungarian alignment, rerun all 5 seeds, and report aligned vs unaligned recovery for every recovered block.

2. **Fatal: no teacher-specific falsification.**
Right now, you may be finding a suffix that fits the queried logits, not necessarily the true teacher's weights. That is function matching, not yet identifiability.
Minimum fix: add a wrong-teacher/decoy-teacher control using same-architecture checkpoints with different seeds, and show recovered parameters are materially closer to the true teacher than to decoys.

3. **High: no held-out generalization evidence.**
With a fixed cached query pool, reviewers will worry that the optimizer is overfitting finite prompts/perturbations rather than recovering the underlying parameters.
Minimum fix: strict train/val/test splits over prompts and perturbations, then report held-out logit loss, held-out sensitivity loss, and aligned weight similarity.

4. **High: the practical story may collapse if the positive result is mostly oracle-prefix.**
If pure-logits is very weak, then the paper is not really demonstrating a realistic black-box attack; it is demonstrating an upper-bound decomposition study. That can still be publishable, but only if framed honestly.
Minimum fix: either show a nontrivial pure-logits result, or explicitly reframe the paper as an empirical identifiability-boundary study with oracle-prefix as the main lens.

5. **Medium-High: threat-model realism and query efficiency are underdeveloped.**
A reviewer will ask whether ~90K+ teacher queries and oracle boundary states are meaningful in practice.
Minimum fix: add recovery-vs-query-budget curves, exact query counts/cost estimates, and a cleaner split between "upper bound" and "practical attack" in the threat model.

6. **Medium: external validity is narrow.**
One 0.6B Qwen model is enough for a case study, not enough for a broad claim about transformer identifiability.
Minimum fix: add one second model/scale if feasible; otherwise narrow the claims to "a controlled empirical study on one modern decoder-only LM."

**Ready?**

`No.`

This is not submission-ready yet. The project is promising, but the paper's main claim currently outruns the evidence. The minimum bar before submission is: permutation-aware alignment, wrong-teacher falsification, and held-out generalization. If those land cleanly and the oracle/pure-logits story is framed carefully, the paper becomes much more defensible.

</details>

### Actions Taken
- Implemented permutation-aware alignment (Hungarian algorithm on attention heads + FFN neurons + RMSNorm normalization)
- Added wrong-teacher falsification control (--wrong_teacher flag)
- Added held-out train/val split (--heldout_fraction 0.2)
- Updated experiment pipeline with Phase D2 for wrong-teacher control
- Reframed paper as "empirical identifiability-boundary study"

### Status
- Continuing to Round 2

---

## Round 2 (2026-03-30)

### Assessment (Summary)
- Score: 6/10
- Verdict: Still not ready
- Key criticisms:
  1. Wrong-teacher control is invalid (compares ground_truth vs ground_truth)
  2. FFN alignment bug (aligned[key] instead of aligned[down_key])
  3. Held-out generalization incomplete (no sensitivity loss, no train comparison)
  4. Oracle boundary states not actually injected in inversion loop
  5. Query budget not enforced in optimization loop

<details>
<summary>Click to expand full reviewer response</summary>

Score: 6/10. The framing is better, and the held-out split is a real improvement, but the minimum-bar concerns are not yet cleanly closed in the implementation.

Remaining Weaknesses:
1. Fatal: wrong-teacher control is invalid — compares ground_truth vs ground_truth
2. High: FFN alignment bug — aligned[key] instead of aligned[down_key]
3. High: held-out only measures clean logit MSE, not sensitivity loss
4. Medium-High: oracle boundary states never actually used in inversion
5. Medium: query budget never enforced
6. Medium: external validity still one model

</details>

### Actions Taken
- Fixed FFN alignment bug (aligned[down_key])
- Fixed wrong-teacher control: now runs full inversion per init, compares recovered_vs_true and recovered_vs_wrong
- Added held-out sensitivity loss evaluation (perturbation-based)
- Implemented oracle boundary state injection via forward pre-hook (_BoundaryInjectionHook)
- Added query budget enforcement in invert_block
- run_spsi now passes use_oracle_boundary and query_budget_remaining

### Status
- Continuing to Round 3

---

## Round 3 (2026-03-30)

### Assessment (Summary)
- Score: 4/10 (reviewer read wrong repo path initially, re-reviewed inline code)
- Verdict: Not ready
- Key criticisms:
  1. Oracle sensitivity: boundary hook only caches clean boundary, not for perturbed inputs
  2. Wrong-teacher: inversion uses true teacher, only post-hoc distance comparison
  3. Phase F: S-PSI saves JSON only, not loadable HF model → eval pipeline broken
  4. Query budget unfair: KD uses query_budget//max_seq_len, S-PSI precomputes 72K+ queries
  5. Pure-logits: RMSNorm stays at teacher values → teacher leakage

### Actions Taken
- Fixed oracle sensitivity: separated clean forward (with boundary hook) from perturbed forward (no hook needed — perturbations go through full model). Clean z_s used as detached reference for delta computation.
- Added `student.save_pretrained()` in `_save_results()` → loadable HF model directory
- Fixed query budget: `run_spsi` now subtracts precomputation queries from total budget
- Fixed KD budget: removed `query_budget // max_seq_len` → uses raw query count
- Fixed pure-logits: added RMSNorm weight reset to ones, bias reset to zeros
- Added train-set evaluation: `generalization_report.json` with train vs held-out logit/sensitivity loss and overfit ratio

### Status
- Auto Review Loop complete (4 rounds reached)
