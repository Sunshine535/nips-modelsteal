# Claude Next Round Execution Plan — T-DART

## Version
Commit: aaa93c3 (v4 probe-dense results)
Status: Q-UMC falsified, implementing successor T-DART

## Task Order
1. ✅ CONVERSATION_REVIEW_VERSION.md
2. ✅ DIRECTION_LOCK.md
3. ✅ LOCAL_PROVENANCE_AUDIT.md + EXTERNAL_CODE_AND_LICENSE_AUDIT.md + PRIOR_WORK_IMPLEMENTATION_BOUNDARY.md
4. Archive Q-UMC as falsified (Task 1 from GPT R4)
5. Implement src/residual_delta.py + tests (Task 2)
6. Implement src/adaptive_candidates.py + tests (Task 3)
7. Implement src/ranking_losses.py + tests (Task 4)
8. Create delta teacher benchmark (Task 5)
9. Implement scripts/run_tdart.py (Task 6)
10. Add strict access tests (Task 7)
11. Run gate experiments if GPU available (Task 8)

## Key Design Decisions
- Teacher: Qwen2.5-0.5B fine-tuned with LoRA on held-out domain
- Reference: Qwen2.5-0.5B base (public)
- Student: Qwen2.5-0.5B base (same as reference) with perturbation
- Loss: pairwise residual ranking + residual MSE + CE
- Candidates: union of teacher top-K + student top-K + reference top-K + disagreement probes
- All strict variants use only top-K + probe oracle (no teacher weights, no full logits)
