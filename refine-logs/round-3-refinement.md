# Round 3 Refinement

## Problem Anchor
[Verbatim — unchanged]

## Anchor Check
- Original bottleneck: No method recovers internal weights from black-box logits alone.
- Why revised method still addresses it: Oracle now uses exact teacher boundary states, cleanly isolating block identifiability. Pure-logits regime remains the realistic setting.
- No drift — all changes tighten the oracle protocol.

## Simplicity Check
- Dominant contribution: S-PSI with clean oracle boundary-state protocol for identifiability
- Components removed: Suffix refinement removed from core (ablation only). Active query in appendix.
- Remaining mechanism is minimal: one loss, one progressive schedule, one alignment procedure.

## Changes Made

### 1. Clean Oracle Protocol with Teacher Boundary States
- **Reviewer said**: "Oracle regime still confounded. Optimizing B_N uses student-produced h_{N-1}, not teacher's."
- **Action**: In oracle setting, for each block B_i being optimized:
  1. Run teacher forward up to block i-1, extract h_{i-1}^T(x) — exact teacher boundary state.
  2. Inject h_{i-1}^T(x) as input to student's block B_i.
  3. Run student from B_i through recovered suffix to get z_S(x).
  4. Optimize B_i to match z_T(x).
  This cleanly tests: "Given the TRUE input to block B_i, can we recover B_i's parameters from logit observations?"
- **Impact**: Oracle is now a genuine identifiability test. If this fails, block recovery is not identifiable even under ideal conditions.

### 2. Precomputed Teacher Logit Cache
- **Reviewer said**: "Perturbation objective too expensive."
- **Action**: Precompute ALL teacher outputs upfront:
  - For each input x in query pool: cache z_T(x) and z_T(x'_{p,r}) for all perturbation positions p and replacement tokens r.
  - Total storage: 10K inputs × (1 + 8×4) perturbations × vocab_size × 2 bytes ≈ manageable.
  - During optimization, only student forward+backward passes are needed. No repeated teacher queries.
  - Reduce P from 8 to 4 positions, R from 4 to 2 replacements → 8 perturbations per input (vs 32 before).
- **Impact**: ~4x cost reduction for sensitivity matching. Teacher is queried once during preprocessing.

### 3. Suffix Refinement as Ablation Only
- **Reviewer said**: "Drop suffix refinement unless shown necessary."
- **Action**: Removed from core method. Run as separate ablation to test if periodic unfreezing helps.
- **Impact**: Cleaner attribution — core method is purely progressive with permanent freeze.

### 4. Tighter Scope
- **Reviewer said**: "One crisp question."
- **Action**: Main paper question is: "Under exact boundary-state oracle access and full-logit observation, are the last 1-2 transformer blocks of an LLM identifiably recoverable (up to known symmetries)?"
  - Secondary: "Under pure-logits access (no boundary states), how much of this identifiability is retained?"
  - The paper is primarily an identifiability study with a practical dimension.

## Revised Proposal

# Sensitivity-Guided Progressive Suffix Inversion (S-PSI): How Identifiable Are Transformer Weights from Logits?

## Problem Anchor
- **Bottom-line problem**: Can we recover actual internal weight parameters of a black-box LLM from output logits?
- **Must-solve bottleneck**: No method studies whether internal weight recovery is even IDENTIFIABLE from logits, let alone achieves it.
- **Non-goals**: Not KD, not data extraction, not attacking production APIs.
- **Constraints**: 8x A100-80GB, ~450 GPU-hours, Qwen3.5-0.6B primary, NeurIPS 2026.
- **Success condition**: Demonstrate that under boundary-state oracle, last 1-2 blocks are identifiably recoverable (per-matrix cos_sim > 0.9, consistent across 5 random inits). Characterize degradation under pure-logits.

## Technical Gap

Existing model stealing: Carlini (ICML 2024) showed lm_head is algebraically extractable. But nobody has asked: **how deep does logit-based identifiability extend into the transformer?** This is a fundamental question for LLM security.

Key insight: For a fixed recovered suffix and a known block input, the block's output is fully determined by its parameters. Adding logit-sensitivity matching (how logits respond to input perturbations) provides additional constraints that narrow the set of consistent parameterizations to (near-)unique solutions up to known symmetries (head permutation, neuron permutation, scale absorption).

## Method Thesis

S-PSI is a progressive block-wise inversion algorithm that recovers transformer suffix parameters from logit observations. The paper's main contribution is the identifiability study: under what conditions are transformer weights recoverable from logits?

## Contribution Focus
- **Dominant**: Identifiability decomposition — boundary-state oracle shows clean suffix identifiability; pure-logits shows practical limits.
- **Mechanism**: S-PSI algorithm with sensitivity matching.
- **Non-contributions**: No new architectures, no defenses.

## Method

### Setup
- Teacher T: Qwen3.5-0.6B (28 layers), frozen, can extract intermediate activations for oracle setting.
- Student S: same architecture.
  - **Oracle**: Teacher prefix copied, suffix randomized.
  - **Pure-logits**: Fully random init.
- Query pool Q: 10K WikiText inputs, max_len=256 (reduced for cost).

### Phase 0: Precompute Teacher Cache
- For each x ∈ Q: compute and cache z_T(x).
- For each x, sample P=4 positions, R=2 replacements: compute and cache z_T(x'_{p,r}).
- For oracle: also cache h_i^T(x) for i ∈ {N-2, N-1, N} (boundary states).
- Total teacher forward passes: 10K × (1 + 4×2) = 90K. One-time cost.

### Phase 1: lm_head Recovery
- **Optimization variables**: lm_head.weight.
- **Input**: h_N^T(x) (oracle) or student-forward h_N(x) (pure-logits).
- **Loss**: L = E_x[||lm_head(h_N) - z_T(x)||²].
- **Optimizer**: Adam, lr=1e-3, 5K steps, batch=32.

### Phase 2-3: Block B_N, B_{N-1} Recovery
For each block B_i (i = N, then N-1):
- **Optimization variables**: All parameters of B_i.
- **All else frozen** (recovered suffix blocks + prefix).
- **Oracle input**: h_{i-1}^T(x) — exact teacher boundary state, injected directly.
- **Pure-logits input**: h_{i-1}(x) from student forward through frozen prefix.
- **Forward from B_i**: student computes B_i(h_{i-1}) → ... → z_S(x) through recovered suffix.
- **Loss**:
  ```
  L(θ_Bi) = L_logit + 0.1 · L_sensitivity + 1e-5 · L_reg
  L_logit = E_x[||z_S(x) - z_T(x)||²]
  L_sensitivity = E_{x,p,r}[||Δz_S(x,p,r) - Δz_T(x,p,r)||²]
  L_reg = Σ||w||²
  ```
- **Optimizer**: Adam, lr=1e-4, max 10K steps, batch=16.
- **Stopping**: improvement < 1e-7 over 500 steps.

### Phase 4: Symmetry Alignment
- Hungarian algorithm on attention heads (Q||K||V||O similarity matrix).
- Hungarian algorithm on FFN neurons (up||gate, down similarity).
- RMSNorm scale normalization.
- Per-matrix cosine similarity reported.

### Cost Accounting
| Phase | Computation | GPU-hours (est.) |
|-------|-------------|-----------------|
| Teacher cache (90K forward passes, 0.6B) | 1× teacher | ~5 |
| lm_head recovery (5K steps × 5 inits × 2 regimes) | Student only | ~10 |
| Block N recovery (10K steps × 5 inits × 2 regimes) | Student only | ~60 |
| Block N-1 recovery (10K steps × 5 inits × 2 regimes) | Student only | ~60 |
| Deeper blocks (boundary analysis, 3-4 blocks) | Student only | ~80 |
| Ablations (β=0, suffix refinement) | Student only | ~60 |
| Alignment, evaluation, analysis | CPU-heavy | ~10 |
| Buffer | | ~65 |
| **Total** | | **~350 GPU-hours** |

## Validation

### Claim 1 (Primary): Under boundary-state oracle, S-PSI recovers last 2 blocks with cos_sim > 0.9
- 5 random suffix inits on Qwen3.5-0.6B.
- Per-matrix cosine similarity: Q, K, V, O, up, gate, down, norm (8 matrices × 2 blocks = 16 measurements).
- Baselines: standard KD, block-coordinate distillation (β=0).
- Success: per-matrix cos_sim > 0.9 for majority of matrices, cross-init variance < 0.02.

### Claim 2: Sensitivity matching is necessary for identifiable recovery
- Ablation: β=0.1 vs β=0.
- Metric: cross-init cosine similarity variance.
- Known-symmetry null test: permute teacher heads, verify alignment recovers original.

### Claim 3: Oracle vs pure-logits gap quantifies prefix-mismatch impact
- Same method, both regimes.
- Expected: oracle >> pure-logits for deeper blocks.

### Claim 4 (Analysis): Depth-recovery boundary
- Continue inversion past 2 blocks, plot depth vs recovery.
- Identify transition point.

## Compute: ~350 GPU-hours (safely within 450 budget)
## Timeline: 8 weeks
