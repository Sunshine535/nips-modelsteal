# Local Provenance Audit

## External links / citations / copied-code markers
AUTO_REVIEW.md:53:- The active-query artifact is indeed broken for the reason stated in `findings.md`: `scripts/run_active_query.py` instantiates the "student" from `from_pretrained` in every condition (`scripts/run_active_query.py:339`, `:364`), so the `cos≈0.999` result is invalid. The paper does not report it, which is correct.
AUTO_REVIEW.md:179:- Core table provenance is still the main blocker. The checked-in results/ support the v3/v4 claims well, but Exp1/2/3/4 in the paper still cannot be independently audited from the repo.
AUTO_REVIEW.md:214:  3. Clean up invalid active-query artifact from repo
GPT55_DIAGNOSIS.md:891:[1]: https://github.com/Sunshine535/nips-modelsteal "GitHub - Sunshine535/nips-modelsteal: ModelSteal: Progressive Layer-wise Parameter Inversion of LLMs (NeurIPS 2026) · GitHub"
GPT55_DIAGNOSIS.md:892:[2]: https://github.com/Sunshine535/nips-modelsteal/tree/main/src "nips-modelsteal/src at main · Sunshine535/nips-modelsteal · GitHub"
GPT55_DIAGNOSIS.md:894:[4]: https://github.com/Sunshine535/nips-modelsteal/tree/main/paper "nips-modelsteal/paper at main · Sunshine535/nips-modelsteal · GitHub"
GPT55_DIAGNOSIS.md:895:[5]: https://github.com/Sunshine535/nips-modelsteal/tree/main/scripts "https://github.com/Sunshine535/nips-modelsteal/tree/main/scripts"
GPT55_DIAGNOSIS.md:896:[6]: https://github.com/Sunshine535/nips-modelsteal/tree/main/configs "nips-modelsteal/configs at main · Sunshine535/nips-modelsteal · GitHub"
GPT55_DIAGNOSIS.md:909:[19]: https://github.com/Sunshine535/nips-modelsteal/tree/main/results "nips-modelsteal/results at main · Sunshine535/nips-modelsteal · GitHub"
GPT55_DIAGNOSIS.md:912:[22]: https://github.com/Sunshine535/nips-modelsteal/tree/main/tests "nips-modelsteal/tests at main · Sunshine535/nips-modelsteal · GitHub"
GPT55_DIAGNOSIS.md:918:[28]: https://arxiv.org/abs/2403.06634 "https://arxiv.org/abs/2403.06634"
GPT55_DIAGNOSIS.md:919:[29]: https://arxiv.org/html/2403.09539v3 "https://arxiv.org/html/2403.09539v3"
GPT55_DIAGNOSIS.md:920:[30]: https://arxiv.org/abs/2510.24966 "https://arxiv.org/abs/2510.24966"
GPT55_DIAGNOSIS.md:921:[31]: https://arxiv.org/pdf/2506.17090 "https://arxiv.org/pdf/2506.17090"
GPT55_DIAGNOSIS.md:922:[32]: https://openreview.net/pdf?id=g9vFg3O8YY "https://openreview.net/pdf?id=g9vFg3O8YY"
LITERATURE_SURVEY_2024_2026.md:20:- **arXiv**: [2411.07536](https://arxiv.org/abs/2411.07536) (Nov 2024)
LITERATURE_SURVEY_2024_2026.md:31:- **arXiv**: [2403.06634](https://arxiv.org/abs/2403.06634) (Mar 2024)
LITERATURE_SURVEY_2024_2026.md:39:- **arXiv**: [2403.09539](https://arxiv.org/abs/2403.09539) (Mar 2024)
LITERATURE_SURVEY_2024_2026.md:48:- **arXiv**: [2310.08708](https://arxiv.org/abs/2310.08708)
LITERATURE_SURVEY_2024_2026.md:82:- **Link**: [OpenReview](https://openreview.net/forum?id=bQMejscfJb)
LITERATURE_SURVEY_2024_2026.md:92:- **Link**: [OpenReview](https://openreview.net/forum?id=9xPKxRQKXc)
LITERATURE_SURVEY_2024_2026.md:110:- **arXiv**: [2411.09150](https://arxiv.org/abs/2411.09150)
LITERATURE_SURVEY_2024_2026.md:118:- **arXiv**: [2506.01261](https://arxiv.org/abs/2506.01261)
LITERATURE_SURVEY_2024_2026.md:129:- **arXiv**: [2410.10226](https://arxiv.org/abs/2410.10226)
LITERATURE_SURVEY_2024_2026.md:143:- **arXiv**: [2402.13116](https://arxiv.org/abs/2402.13116)
NARRATIVE_REPORT.md:141:3. **Invalid active-query artifact (LOW)**: `results/v4_active_query/` contains data from a buggy script (student = teacher). Correctly excluded from paper but still in repo.
PAPERS.md:11:- **URL**: https://arxiv.org/abs/2403.06634
PAPER_CLAIM_AUDIT.md:52:- **Fix**: Either add the raw result file, or add a footnote explaining this was a manual assertion based on oracle hook behavior
README.md:12:git clone https://github.com/Sunshine535/nips-modelsteal.git
README.md:144:This project is licensed under the [MIT License](LICENSE).
REVIEWER_MEMORY.md:34:3. Purge contaminated branches from paper, scripts, released repo
SELF_REVIEW.md:8:1. **Novel theoretical framework**: The suffix observability Gramian adapted from control theory, combined with the complete continuous symmetry group for modern transformers (RMSNorm + gated MLP), is a genuinely new contribution.
baselines/carlini_2024/README.md:4:(ICML 2024 Best Paper), [arXiv:2403.06634](https://arxiv.org/abs/2403.06634).
baselines/carlini_2024/run_carlini.py:436:        "paper_arxiv_id": "2403.06634",
baselines/clone_2025/README.md:4:Logit Leakage and Distillation", [arXiv:2509.00973](https://arxiv.org/abs/2509.00973)
baselines/clone_2025/README.md:121:overlap for the L=24 student; that was aspirational, based on
refine-logs/FINAL_PROPOSAL.md:37:- Student S: same architecture. Oracle: teacher prefix copied, suffix randomized. Pure-logits: fully random.
refine-logs/round-2-refinement.md:22:  - **Pure-logits setting**: Student prefix is random-init (not copied). Optimize suffix blocks with frozen random prefix. This is the realistic attack scenario. Recovery quality bounded by prefix mismatch.
refine-logs/round-2-refinement.md:166:- **Setting**: Oracle-prefix (teacher's prefix copied to student).
refine-logs/round-3-refinement.md:79:  - **Oracle**: Teacher prefix copied, suffix randomized.
reports/GPT55_R2_REVIEW_RESPONSE.md:25:- Report claimed "C > B" based on PPL only
reports/HONEST_FINAL_STATUS.md:105:Next steps should be decided by GPT-5.5 Pro based on this evidence:
reports/KEEP_REWRITE_ARCHIVE_PLAN.md:19:| Moment-CP script | scripts/attack_higher_order_moments.py | Paper main but quarantined | Script header says quarantined | FREEZE / ARCHIVE until reproduced | Cannot be active main without reproduction | Raw artifacts missing from repo |
research-wiki/papers/carlini2024_stealing.md:9:  arxiv: "2403.06634"
research-wiki/papers/rawal2025_spsi.md:9:  arxiv: null
research-wiki/papers/tramer2016_stealing.md:9:  arxiv: "1609.02943"
review-stage/ORACLE_PRUNE_VERDICT.md:54:* **Purge all contaminated and broken branches** from paper, supplement, captions, and released repo branch.
scripts/algebraic_recovery_v4_richinput.py:97:# Common helpers (copied from v3 to keep v4 self-contained)
scripts/algebraic_recovery_v4_richinput.py:316:# Hidden-state collection (copied from v3, tightened)
scripts/diagnose_phase2_failure.py:72:# Helpers (copied / adapted from algebraic_recovery_v3_breakthrough.py)
scripts/diagnose_phase2_failure.py:280:# Joint (W_gate, W_up) optimization (copied from v3, trimmed for diagnostics)
scripts/enhanced_kd_clone.py:69:    """Initialize student weights based on init_mode."""
scripts/functional_kl_eval.py:8:Student: same architecture, prefix copied from teacher (ORACLE PREFIX so we
scripts/run_spsi_carlini_init.py:167:    logger.info("lm_head init: carlini_exact (teacher lm_head copied)")
scripts/run_spsi_carlini_init.py:521:            return {"method": "carlini_exact", "description": "teacher lm_head copied"}
skills/arxiv/SKILL.md:2:name: arxiv
skills/arxiv/SKILL.md:3:description: Search, download, and summarize academic papers from arXiv. Use when user says "search arxiv", "download paper", "fetch arxiv", "arxiv search", "get paper pdf", or wants to find and save papers from arXiv to the local paper library.
skills/arxiv/SKILL.md:4:argument-hint: [query-or-arxiv-id]
skills/arxiv/SKILL.md:16:- **FETCH_SCRIPT** - `tools/arxiv_fetch.py` relative to the ARIS install, or the same path relative to the current project. Fall back to inline Python if not found.
skills/arxiv/SKILL.md:19:> - `/arxiv "attention mechanism" - max: 20` - return up to 20 results
skills/arxiv/SKILL.md:20:> - `/arxiv "2301.07041" - download` - download a specific paper by ID
skills/arxiv/SKILL.md:21:> - `/arxiv "query" - dir: literature/` - save PDFs to a custom directory
skills/arxiv/SKILL.md:22:> - `/arxiv "query" - download: all` - download all result PDFs
skills/arxiv/SKILL.md:46:    pathlib.Path('tools/arxiv_fetch.py'),
skills/arxiv/SKILL.md:47:    pathlib.Path.home() / '.claude' / 'skills' / 'arxiv' / 'arxiv_fetch.py',
skills/arxiv/SKILL.md:73:url = (f"http://export.arxiv.org/api/query"
skills/arxiv/SKILL.md:93:        "pdf_url": f"https://arxiv.org/pdf/{aid}.pdf",
skills/arxiv/SKILL.md:94:        "abs_url": f"https://arxiv.org/abs/{aid}",
skills/arxiv/SKILL.md:118:url = 'http://export.arxiv.org/api/query?id_list=ARXIV_ID'
skills/arxiv/SKILL.md:146:    'https://arxiv.org/pdf/ARXIV_ID.pdf',
skills/arxiv/SKILL.md:147:    headers={'User-Agent': 'arxiv-skill/1.0'},
skills/auto-review-loop-minimax/SKILL.md:57:**Why MiniMax instead of Codex MCP?** Codex CLI uses OpenAI's Responses API (`/v1/responses`) which is not supported by third-party providers. See: https://github.com/openai/codex/discussions/7782
skills/dse-loop/SKILL.md:146:1. **Select next design point** based on results so far:
skills/experiment-bridge/SKILL.md:32:> Override: `/experiment-bridge "EXPERIMENT_PLAN.md" — compact: true, base repo: https://github.com/org/project`
skills/feishu-notify/SKILL.md:41:| **Interactive** | `"interactive"` | Full bidirectional. Approve/reject from Feishu, reply to checkpoints | [feishu-claude-code](https://github.com/joewongjc/feishu-claude-code) running |
skills/feishu-notify/SKILL.md:92:Interactive mode uses [feishu-claude-code](https://github.com/joewongjc/feishu-claude-code) as a bridge:
skills/formula-derivation/SKILL.md:181:If the derivation still lacks a coherent object, stable assumptions, or an honest path from premises to result, downgrade the status and write a blocker report instead of forcing a clean story.
skills/grant-proposal/SKILL.md:10:Draft a grant proposal based on: **$ARGUMENTS**
skills/grant-proposal/SKILL.md:578:/grant-proposal "topic — KAKENHI Start-up, sources: zotero, arxiv download: true"
skills/grant-proposal/SKILL.md:589:| `arxiv download` | false | Download arXiv PDFs | → `/research-lit` |

## License / citation / readme files
