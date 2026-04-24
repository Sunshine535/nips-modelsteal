# Keep / Rewrite / Archive Plan

| Item | Path | Current Role | Evidence | Action | Reason | Risk |
|------|------|--------------|----------|--------|--------|------|
| Strict Q-UMC trainer | scripts/run_qumc.py | New main method | 3-seed positive | KEEP | Main method, validated | None |
| Oracle abstraction | src/oracles.py | New core module | Tests pass | KEEP | Required for strict black-box | None |
| Logit completer | src/logit_completion.py | New core module | Tests pass, C>A | KEEP | Required for Q-UMC | None |
| Result manifest | src/result_manifest.py | Provenance | Used in every run | KEEP | Required for reproducibility | None |
| Old enhanced KD | scripts/enhanced_kd_clone.py | Historical reference | v7-v14 results | KEEP AS HISTORICAL (simulator) | Not main method, but reproducible signal | Must tag as "simulator-positive, full-logit leaked" |
| v13/v14 results | results/v13_lc_topk20, v14_lc_topk5 | Positive but contaminated | JSON preserved | KEEP AS HISTORICAL NEGATIVE EVIDENCE | Shows leaked-completion signal; NOT strict-black-box evidence | In evidence_registry |
| Sparse top-K KD | Variant A in run_qumc.py | Baseline | A results | KEEP ONLY AS BASELINE | Required A-control | None |
| Hidden MSE / SCRD variants | B/C/D in enhanced_kd_clone.py | Old proposed improvement | v11/v12 B=753 | KEEP ONLY AS ABLATION | Demonstrates representation conflict | None |
| S-PSI core | src/parameter_inverter.py | Old main (superseded) | CLAIMS_FROM_RESULTS negative | FREEZE / KEEP ONLY AS ABLATION | Diagnostic/negative evidence | P1 bugs documented but not fixed this pass |
| S-PSI sensitivity claim | README/method | Old novelty | beta=0 equal, no recovery | DELETE from active claims | Not supported by evidence | Paper/README update pending |
| Gramian diagnostics | src/gramian.py | Observability analysis | Partial logs | FREEZE | Not main result, raw logs incomplete | None |
| Algebraic init | src/algebraic_init.py | Warmstart | Marginal | KEEP ONLY AS ABLATION | Not main method | None |
| Active query | src/active_query.py, scripts/run_active_query.py | Old experiment | BUG INVALIDATED | ARCHIVE TO archive/ (pending) | Results confirmed invalid | Must not cite |
| Active query old positive results | (if any in results/) | False positive | findings.md BUG CONFIRMED | ARCHIVE (pending) | Must not cite | Paper must exclude |
| Moment-CP script | scripts/attack_higher_order_moments.py | Paper main but quarantined | Script header says quarantined | FREEZE / ARCHIVE until reproduced | Cannot be active main without reproduction | Raw artifacts missing from repo |
| Moment-CP claim (W_lm top5 0.813) | paper/main.tex, ATTACK_4WAY_SUMMARY.md | Current paper thesis | Raw result incomplete | FREEZE in paper | Cannot claim until artifact audited | Paper rewrite pending |
| Carlini reproduction | scripts/reproduce_carlini.py | Subspace baseline | Known to work | KEEP | Required close baseline | None |
| Functional KL eval | scripts/functional_kl_eval.py | Functional metric | Used in evidence | KEEP / may REWRITE | Useful eval, may need Q-UMC-specific variant | None |
| README | README.md | Old S-PSI narrative | Inconsistent with Q-UMC | REWRITE (pending) | Narrative must reflect Q-UMC after broader baselines | None |
| Paper | paper/main.tex | Moment-CP thesis | Artifact gap | REWRITE (pending) | Rewrite to Q-UMC after broader baselines | None |
| Claim audits | PAPER_CLAIM_AUDIT*.md | Self-review | FAIL records | KEEP | Integrity audit trail | None |
| Evidence registry | docs/evidence_registry.md | NEW classification | This pass | KEEP | Result reliability labels | None |
| Missing v5 moments raw | results/v5_attack_moments_v2/ | Claimed but sparse | Artifact gap | LOCATE AND AUDIT (pending) | Cannot restore paper claim without | None |

## Archive Queue (pending)
- Active-query results → `archive/20260424_active_query_invalid/`
- Old Moment-CP quarantined script artifacts → `archive/20260424_moment_cp_pending_audit/`

## NOT Archived
- All raw v7-v14 result JSONs (historical evidence)
- S-PSI code (kept as ablation)
- Carlini reproduction (required baseline)
