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

---

## Round 5 (2026-03-31)

### Score
- 4/10
- Verdict: Not ready to produce trustworthy experimental results

### Findings
1. **Fatal: the advertised multi-GPU path can silently produce incorrect aggregate results.**
   - `scripts/run_spsi_multigpu.sh:62-72` and `scripts/run_all_experiments.sh:96-100` launch multiple `scripts/run_spsi.py` processes into the same output directory.
   - `scripts/run_spsi.py:369-389` writes `regime_summary.json`, and `scripts/run_spsi.py:477-489` writes `experiment_summary.json`; with one-init jobs running concurrently, these shared summaries are overwritten by the last finisher.
   - `scripts/run_all_experiments.sh:114-115` only logs a warning when an init fails and still returns success, after which `scripts/run_all_experiments.sh:160-172` marks the phase as done.
   - Net effect: the code can report a completed multi-GPU experiment while cross-init statistics are incomplete or wrong.

2. **High: checkpoint/resume support is incomplete and the CLI is misleading.**
   - `scripts/run_spsi.py:129-130` exposes `--resume_from`, but the argument is never used.
   - `src/parameter_inverter.py:430-444` auto-loads the latest checkpoint only from the implicit `output_dir/checkpoints` directory; there is no explicit resume target or validation that the resumed run matches the original configuration.
   - `src/parameter_inverter.py:543-546` checkpoints only every `save_every` steps, and `_save_results()` is only called at the end of the full run (`src/parameter_inverter.py:640-641`, `src/parameter_inverter.py:694-700`). A crash between save intervals can therefore lose completed progress.
   - `scripts/run_kd_baseline.py:330-331` advertises a path-valued `--resume_from_checkpoint`, but `scripts/run_kd_baseline.py:389` reduces it to a boolean and `scripts/run_kd_baseline.py:197-206` always resumes from the latest checkpoint in `output_dir`, ignoring any user-supplied path.

3. **High: several repository entry points are stale and do not match the current `src/parameter_inverter.py` API.**
   - `scripts/run_progressive_inversion.py:227` calls `invert_layer`, but `src/parameter_inverter.py:722-759` only exposes `run_progressive_inversion`.
   - `scripts/run_progressive_inversion.py:281-292` and `scripts/run_progressive_inversion.py:346-357` pass `optimizer_type`, `regularization_lambda`, and `active_query_pool_size` into `InversionConfig`, but `src/parameter_inverter.py:710-720` does not define those fields.
   - `scripts/invert_parameters.py:211-223` has the same unsupported `InversionConfig` kwargs, while `scripts/invert_parameters.py:245-248` depends on `result.recovered_state_dict` even though `src/parameter_inverter.py:91-93` always returns `None`.
   - `scripts/run_defense_eval.py:179-187` passes `selection_batch` into `InversionConfig`, which is also unsupported.
   - This is a code completeness problem, not just a cleanup issue: large parts of the advertised artifact are not maintained against the current implementation.

4. **Medium: the repository is not yet in a reproducible-results state.**
   - `EXPERIMENTS.md:28` explicitly says no experiments have been recorded.
   - The checked-in `results/` directory is empty.
   - `scripts/run_spsi.py:145-150` only reads `teacher.model_name` from `configs/inversion_config.yaml`; the inversion and evaluation hyperparameters documented in `configs/inversion_config.yaml:17-67` are not actually loaded into the main run.
   - `scripts/run_spsi.py:55-71` and `scripts/run_kd_baseline.py:95-110` silently fall back to random-token queries when dataset loading fails, which can materially change the experiment without failing fast.

5. **Medium: code quality and operational hygiene are below artifact-ready standard.**
   - `src/parameter_inverter.py:322-333` leaves `_inject_boundary_state()` as dead `pass` code after the refactor to `_BoundaryInjectionHook`.
   - `scripts/run_all_experiments.sh:123-133` force-kills any Python or `torchrun` process that appears to be using a GPU, which is unsafe behavior on a shared server.
   - There are no tests or even smoke checks in the repository, so the broken API boundaries above are not caught automatically.

### Open Questions / Assumptions
- This review is based on static code inspection and parser-level checks. `python -m compileall src scripts` and `bash -n` passed, but I could not run the training code in this environment because PyTorch is not installed here.
- The main single-GPU `scripts/run_spsi.py` path may be salvageable after dependency installation, but the repository as submitted claims a broader, more complete artifact than the code currently supports.

### Actionable Feedback
- Isolate each initialization into its own directory and add a dedicated aggregation step that merges all init-level summaries after all workers finish. Multi-GPU orchestration should fail hard if any init fails.
- Implement real resume semantics: honor the user-provided checkpoint path/directory, save final checkpoints at block boundaries, and write enough metadata to verify restart compatibility.
- Delete or port the stale scripts (`run_progressive_inversion.py`, `invert_parameters.py`, `run_defense_eval.py`) so every documented entry point matches the current `src/` API.
- Make the YAML config authoritative for experiment hyperparameters and write the resolved config into every result directory.
- Remove silent random-query fallback for official runs. If the dataset is unavailable, fail closed unless an explicit fallback flag is set.
- Add smoke tests for one toy S-PSI run, one resume/restart path, and one multi-process aggregation path.

### Verdict
- Score: 4/10
- The core idea is interesting, but the artifact is not ready to support trustworthy NeurIPS experimental results, especially in the advertised multi-GPU setting.
