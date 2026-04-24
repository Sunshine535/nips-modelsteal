# Reviewer Memory

## Pre-Round Context
- Paper: "Transformer Tomography" — NeurIPS 2026
- Medium loop: 2 rounds, 5→7/10
- Proof-checker (nightmare): 3 rounds → 7.5/10, later R5 PASS 9.0/10
- Nightmare review first pass (2026-04-14): 3 rounds, 6→6.5→7/10, closed "completed"
- PAPER_CLAIM_AUDIT_R2: 7 number-mismatches fixed
- Second pass (2026-04-19): reached 8.5 but Oracle Pro later issued prune directive

## Oracle Pro Prune Verdict (2026-04-19, pre-α-loop Round 0)
Oracle Pro (browser mode, gpt-5.4-pro, 12m19s) independently concurred with codex xhigh
that the current paper is schizophrenic: sells S-PSI as a method that fails under its
own threat model, then uses theory to explain the failure. **Directive:** pivot to
(α) pure theory / characterization paper, Main Track — Theory.

### KILL (this submission)
A1 S-PSI · A2 Moments CP · A4 logit-bias precision · A5 memory selective probing ·
A6 active query · A7 algebraic recovery v2/v3/v4 · B2 Clone 2025 · B3 matched KD ·
E2 multi-model broad sweep · E4 functional KL.

### DEMOTE to appendix
A3 Jacobian-FD (documents observable ≠ identifiable).

### KEEP (ship)
T1 Gramian definition + gauge null Prop 1 · T2 Thm 3 first-order observability criterion ·
T3 Thm 4 rank upper bound · T4 Thm 5 depth screening · E1 K×k grid (validates T3) ·
E3 pure-logits algebraic cross-arch (Qwen + Llama confirms pure-logits inversion fails) ·
B1 Carlini SVD (subspace contrast).

### Engineering musts from Oracle Pro
1. Retitle away from "Tomography"; prefer "suffix observability" / "non-identifiability"
2. Delete §5 Method: S-PSI; replace with empirical validation of T2–T4
3. Purge contaminated branches from paper, scripts, released repo
4. Scope: pure logits, black-box, local/first-order ONLY — no hidden states, no continuous embeddings
5. Rebuild all retained figures from clean branch with script-to-figure provenance

### Cut target
6 sections · 4 figures · 2 tables.

### Top 3 objections pre-empted by Oracle Pro
1. Local/first-order scope — state in title/abstract/limitations explicitly
2. Gauge non-identifiability not novel → sell T2 + T3 + T4, not T1
3. Contamination history → total excision

## Persistent suspicions carried from prior reviewers (active to next round)

Oracle Pro nightmare difficulty means the reviewer reads the repo directly, so these are
concrete things it will likely verify on Round 1:

- **Any leftover S-PSI language**: abstract, contributions, §5, captions, §8 conclusion
- **Any Moments-CP numbers in main text**: must be absent (one sentence in limitations OK)
- **Any "recovery" wording selling the paper as attack**: must be replaced with
  "characterization" / "observability" / "identifiability"
- **Contaminated scripts still referenced**: `active_query_experiment.py`,
  `algebraic_recovery_v3_*`, `matched_kd_baseline.py` — if they live in `scripts/`,
  they must carry a SAFETY_NOTICE header and NOT be cited in the paper
- **§5 S-PSI block-local + oracle-boundary language**: must be removed (was sold as
  partial success; Oracle Pro declared off-threat-model and narrative poison)
- **Table 9 W_V ≈ 0.266 gauge-invariant result**: must NOT appear in main text
  (previous reviewers flagged inconsistency with "no weight matrix recovered")
- **Proof state**: `PROOF_CHECK_STATE.json` must match `main.tex` — Thm 5 must have an
  in-paper proof block, not only a claim
- **Multi-model claim**: keep only the Llama cross-arch validation for E3; drop
  `v5_multi_sweep` broad claims
- **Clone 2025 README overclaim**: delete or move to appendix — current README numbers
  don't match artifacts

## Unresolved at start of α-loop Round 1
- Empirical rank bound (Thm 4) vs observed eff_rank: K×k grid has K=128 with k=64 and
  k=256; more cells would strengthen the plot (optional)
- Depth screening (Thm 5) uniform contraction ρ: needs an empirical estimate; so far
  only descriptive
- Cross-arch E3 currently Qwen2.5-0.5B + Llama-3.2-1B; Oracle did not require extra
  arch but adding one (e.g., Llama-3.2-3B) strengthens "pure-logits fails universally"
- Figure/table provenance manifest: needs to exist in `review-stage/` and be cross-linked
  from each surviving figure
- Paper title, abstract, intro, contributions bullets all still sell the old paper

## Patterns to watch (inherited + new)

- Author may partially rename without substantive cuts — verify §5 is gone, not renamed
- Author may claim "contamination excised" but leave dead references in bibliography or
  captions — Oracle will check
- Author may keep figures sourced from contaminated runs — verify the rebuild happened
  with fresh scripts from clean branch
- Author may smuggle "behavioral recovery" claims from Clone 2025 or matched-KD back into
  limitations — these belong in related-work only
- Watch for `PROOF_CHECK_STATE.json` drift: previous rounds used "completed" at 9.0/10,
  new loop must update or delete

## Reviewer threads
- Prior codex nightmare threadId: `019da29f-9aaa-7493-8a59-9863f1cc0c4a` (not reused)
- Oracle Pro prune session: one-shot (saved to `review-stage/ORACLE_PRUNE_VERDICT.md`)
- α-loop Round 1 thread: TBD — oracle-pro browser mode does NOT maintain threadIds
  across CLI invocations; each round is a fresh session with full memory context
  re-injected via `REVIEWER_MEMORY.md`

## Round 1 (2026-04-19) — Oracle Pro verdict: 4/10 NOT READY

Oracle Pro read the cleaned `paper/main.tex` after the first surgical pass and diagnosed:
- Section 5 S-PSI gone literally, but method survived as "sensitivity augmentation" + Alg(aug) rows.
- Abstract clean; body still mixed: W_V=0.27 positive finding advertised as "previously hidden" in main §7.
- `Transformer Tomography` still in overview caption + conclusion.
- Oracle experiments still dominate a supposedly pure-logit theory paper.
- Theory/implementation mismatch: attention gauge not projected in code; several quantitative gauge-quotiented claims depend on projection.

### Round 1 minimum-fix list executed this round
- F1: deleted Experiment 1 (Alg init vs Random) + Experiment 3 (Controls & Ablations) + warm-start table + per-matrix "LayerNorm dominance" paragraph.
- F2: deleted Sensitivity augmentation paragraph; Alg(aug) rows gone with Exp 1.
- F3: deleted `tab:gauge-invariant` table + all prose mentioning V-projection 0.27.
- F5: renamed "Transformer Tomography overview" → "Observability framework overview"; purged remaining instances.
- F6: deleted "Oracle vs. pure-logits gap: functional, not parametric", "Parameter recovery vs. functional equivalence", "Sensitivity augmentation does not improve", "Controls confirm robustness" paragraphs.

Paper 888 → 721 lines after deep prune; PDF 319KB → 281KB.

### Carrying into Round 2
- Depth-screening experiment (blocks 19–23) is LAUNCHED but not yet complete; Oracle previously flagged "Thm 5 sold as validated before contraction sweep done" — expect Oracle to re-ask for empirical ρ.
- Scope claim should match: "first-order, passive-query, pure-logit" — now the body is consistent.
- Implementation-gap disclaimer re: attention gauge projection — NEEDS an explicit note added to §4 or §Limitations.
- Likely remaining reviewer demands: (a) soften any numeric claim that requires exact gauge projection, (b) verify Llama cross-arch numbers point to non-contaminated result files, (c) ensure all references (tab:exp1, tab:per-matrix-main, tab:warmstart, tab:expanded, tab:exp2, tab:exp3) either resolve or are removed.
