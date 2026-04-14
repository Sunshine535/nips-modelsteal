# Reviewer Memory

## Pre-Round Context
- Paper: "Transformer Tomography" — NeurIPS 2026
- Previous review loop (medium difficulty): 2 rounds, score 5→7/10
- Proof-checker loop (nightmare): 3 rounds, score 7.5/10
- Known issues resolved: SiLU symmetry, attention V/O per KV group, sequence-level Jacobians, sketched Gramian overclaim, code-theory gap documented
- Known remaining concerns:
  - Active query experiment has initialization bug (student loaded from pretrained = teacher), cos ≈ 0.999 invalid
  - No matched-budget KD baseline
  - Single model scale (0.5B) + one Llama generalization point
  - Functional recovery may be standard KD
  - Attention gauge not in code (documented, 0.03% of gauge)

## Round 1 — Score: 6/10
- **Suspicion**: stale "rank≈32" narrative survived in abstract/intro despite v3 showing rank=64 at K=8,k=64
- **Suspicion**: Exp1/2/3/4 tables not backed by checked-in artifacts (trust-based)
- **Suspicion**: Functional table numbers slightly inconsistent with AUTO_REVIEW.md
- **Unresolved**: No matched-budget KD baseline
- **Unresolved**: Llama results existed but were unreported — looks like pipeline drift
- **Unresolved**: PROOF_SKELETON.md was stale
- **Patterns**: Active query Gramian stats might not be trustworthy (script falls back to ungauged probes)
