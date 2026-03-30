# Round 2 Refinement

## Problem Anchor
[Verbatim from Round 0 — unchanged]

## Anchor Check
- **Original bottleneck**: No method recovers internal weights from black-box logits alone.
- **Why the revised method still addresses it**: We now explicitly model the prefix-mismatch problem and introduce an oracle-prefix upper bound to isolate suffix identifiability from prefix noise. The pure-logits claim is honest about what is recoverable.
- **Reviewer suggestions rejected as drift**: None rejected this round — all P0 feedback is valid and addresses the core mechanism.

## Simplicity Check
- **Dominant contribution after revision**: Sensitivity-Guided Progressive Suffix Inversion (S-PSI) with formal identifiability decomposition: oracle-prefix (upper bound) vs pure-logits (practical regime).
- **Components removed**: Active query selection removed from core contribution (appendix only). Single lm_head path (gradient-based, not algebraic). 4B stress test moved to appendix.
- **Why remaining mechanism is smallest adequate**: One loss function (logit MSE + logit-sensitivity), one progressive schedule, one alignment procedure. No auxiliary networks.

## Changes Made

### 1. Block-Input Problem Explicitly Modeled
- **Reviewer said**: "When optimizing block B_i, input comes from student's unrecovered prefix, not teacher's. The proposal doesn't address this."
- **Action**: Formalized the problem as a decomposition:
  - **Oracle-prefix setting**: Copy teacher's prefix to student, freeze it, then optimize suffix blocks. This isolates suffix identifiability from prefix noise. Serves as UPPER BOUND on recovery quality.
  - **Pure-logits setting**: Student prefix is random-init (not copied). Optimize suffix blocks with frozen random prefix. This is the realistic attack scenario. Recovery quality bounded by prefix mismatch.
  - **Joint-optimization setting**: Jointly optimize prefix and suffix. This finds a functionally equivalent model but may not recover teacher's actual weights (acknowledged limitation).
- **Reasoning**: The oracle-prefix setting cleanly separates two questions: (1) IS suffix recovery identifiable under ideal conditions? (2) How much does prefix mismatch degrade recovery?
- **Impact**: Paper now has two clean regimes with an honest comparison.

### 2. Complete Method Data Flow Specified
- **Reviewer said**: "Must specify what supplies h_{i-1}, whether prefix is frozen/latent/joint."
- **Action**: Full specification:
  - **h_{i-1} comes from**: Forward pass through student model up to block i-1. In oracle-prefix: these blocks have teacher's weights. In pure-logits: these blocks are randomly initialized and frozen during suffix inversion.
  - **Optimization variables per step**: Only parameters of block B_i. Everything else frozen.
  - **Perturbation distribution**: For each input x of length L, sample P=8 positions uniformly, for each position sample 4 replacement tokens from top-50 vocabulary by frequency. Total perturbations per input: 32.
  - **Stopping rule**: Early stop when per-block loss improvement < 1e-7 over 500 steps, or max 10K steps, or query budget exhausted.
  - **Suffix refinement**: Every 2K steps of block optimization, unfreeze all recovered suffix blocks, run 200 steps at lr=1e-5, then re-freeze.
- **Impact**: Method is now fully implementable.

### 3. Validation Controls Added
- **Reviewer said**: "Need oracle-prefix vs pure-logits control, known-symmetry null tests."
- **Action**: Three validation controls:
  1. **Oracle-prefix vs pure-logits**: Main comparison. If oracle-prefix gives cos_sim > 0.9 but pure-logits gives ~0.5, we know suffix recovery IS identifiable but prefix mismatch is the bottleneck.
  2. **Known-symmetry null test**: Take teacher model, apply known permutation to attention heads and FFN neurons. Verify that our alignment procedure correctly recovers the original parameterization from the permuted version.
  3. **Per-matrix recovery**: Report cosine similarity for each individual weight matrix (Q, K, V, O, up, gate, down, norm), not just pooled per-block.
- **Impact**: Identifiability claim is now testable.

### 4. Honest Scope and Claims
- **Reviewer said**: "Title/abstract must match demonstrable regime."
- **Action**:
  - **Headline claim**: Under oracle-prefix conditions, S-PSI recovers lm_head + last 2 transformer blocks of Qwen3.5-0.6B with per-matrix cos_sim > 0.9.
  - **Secondary claim**: Under pure-logits conditions, output-adjacent parameters (lm_head, last block RMSNorm, last block output projection) maintain cos_sim > 0.8, deeper parameters degrade.
  - **Analysis**: Depth-recovery boundary under both conditions. Query efficiency of sensitivity matching vs MSE-only.
  - **Title changed**: From "Jacobian-Guided" to "Sensitivity-Guided" — more accurate for token-perturbation approach.
- **Impact**: Claims are now defensible and honest.

### 5. Active Query Selection Demoted
- **Reviewer said**: "Remove active query selection from core contribution."
- **Action**: Moved to appendix. One comparison (gradient-diversity vs random) mentioned in a single paragraph in main text. Not listed as a contribution.
- **Impact**: Cleaner contribution story.

## Revised Proposal

# Research Proposal: Sensitivity-Guided Progressive Suffix Inversion (S-PSI)

## Problem Anchor
- **Bottom-line problem**: Can we recover actual internal weight parameters of a black-box LLM from API query access to output logits?
- **Must-solve bottleneck**: No method recovers internal weight matrices from API queries alone with identifiability analysis.
- **Non-goals**: Not KD, not data extraction, not attacking production APIs, not arbitrary-architecture recovery.
- **Constraints**: 8x A100-80GB, ~450 GPU-hours, open-weight models, NeurIPS 2026.
- **Success condition**: Under oracle-prefix, recover lm_head + last 2 blocks with per-matrix cos_sim > 0.9. Under pure-logits, characterize what IS recoverable.

## Technical Gap

Model stealing dichotomy: algebraic output-layer extraction (Carlini, ICML 2024) vs behavioral cloning. No work studies:
1. Whether multi-layer suffix recovery is IDENTIFIABLE from logits alone
2. How recovery quality depends on prefix fidelity
3. Whether sensitivity matching (logit response to input perturbations) improves identifiability over logit matching alone

**Key insight**: Logit-sensitivity matching constrains block parameters more tightly than logit agreement. Two models may produce identical logits but respond differently to input perturbations. This creates additional constraints that reduce the set of consistent parameterizations.

**Formalization**: Define the suffix S = {B_{N-K+1}, ..., B_N, lm_head}. The teacher computes z_T(x) = S_T(P_T(x)), where P_T is the unknown prefix. The student's recovery problem splits into:
- **Oracle-prefix regime** (P_S = P_T): Optimize S_S such that S_S(P_T(x)) ≈ z_T(x). Here, suffix parameters are directly constrained by logits.
- **Pure-logits regime** (P_S ≠ P_T): Optimize S_S such that S_S(P_S(x)) ≈ z_T(x). Suffix must compensate for prefix mismatch — recovery of TRUE teacher suffix parameters is harder.

## Method Thesis

S-PSI studies the identifiability of transformer suffix parameters from logit-only access. It combines progressive block-wise inversion with sensitivity matching and permutation-aware alignment. Main claim: under oracle-prefix, a contiguous suffix IS identifiably recoverable; under pure-logits, output-adjacent parameters remain partially recoverable but deeper recovery degrades.

## Contribution Focus

- **Dominant contribution**: S-PSI algorithm with identifiability decomposition (oracle-prefix vs pure-logits), demonstrating that multi-layer suffix recovery from logits is feasible under ideal conditions and characterizing the practical limits.
- **Supporting analysis**: Sensitivity matching as an identifiability amplifier (ablation), depth-recovery boundary.
- **Non-contributions**: No new architectures, no defenses, no active query selection (appendix only).

## Proposed Method

### Complexity Budget
- **Frozen/reused**: Teacher as black-box oracle, student same architecture.
- **New trainable**: NONE.
- **Excluded**: KD warm-start, meta-learning, adversarial training, auxiliary networks, active query selection in main method.

### Complete Data Flow

**Setup**:
- Teacher model T: Qwen3.5-0.6B, loaded, frozen, logits-only access.
- Student model S: Qwen3.5-0.6B, same architecture, parameters initialized:
  - Oracle-prefix: copy T's prefix weights, randomize suffix weights.
  - Pure-logits: fully random init.
- Query pool Q: 10K inputs from WikiText, tokenized to max_len=512.

**Phase 0: lm_head Recovery**
- Optimization variables: lm_head.weight (vocab_size × hidden_dim).
- All other parameters frozen.
- h_N = forward(student, x)[:, -1, :] — last-token hidden state from student.
- Target: z_T(x) = teacher(x).logits[:, -1, :].
- Loss: L = E_x[||lm_head(h_N) - z_T(x)||²].
- Optimizer: Adam, lr=1e-3, 5K steps, batch_size=32.

**Phase 1-K: Block-wise Suffix Inversion**
For block index i = N, N-1, ..., N-K+1:
- **Optimization variables**: All parameters of block B_i: {q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj, down_proj, input_layernorm, post_attention_layernorm}.
- **All other parameters frozen** (recovered suffix + unrecovered prefix).
- **Forward pass**: Full student forward → logits z_S(x).
- **h_{i-1} comes from**: Student's forward pass through blocks 1..i-1 (in oracle-prefix: these are teacher's weights; in pure-logits: random init frozen).

**Per-block loss**:
```
L(θ_Bi) = α · L_logit + β · L_sensitivity + γ · L_reg

L_logit = (1/|B|) Σ_x ||z_S(x) - z_T(x)||²

L_sensitivity = (1/|B|·P·R) Σ_x Σ_p Σ_r ||Δz_S(x,p,r) - Δz_T(x,p,r)||²
  where Δz(x,p,r) = z(x'_{p,r}) - z(x)
  x'_{p,r} = x with token at position p replaced by replacement token r
  p ~ Uniform(1..L), L = sequence length
  r ~ Top-50 tokens by frequency (sample R=4 replacements per position)
  P = 8 positions per input

L_reg = Σ_{w ∈ θ_Bi} ||w||²
```

- **Hyperparameters**: α=1.0, β=0.1, γ=1e-5.
- **Optimizer**: Adam, lr=1e-4, max 10K steps, batch_size=16.
- **Stopping**: Loss improvement < 1e-7 over 500 steps.
- **Suffix refinement**: Every 2K steps, unfreeze all recovered suffix blocks, 200 steps at lr=1e-5, re-freeze.

**Phase K+1: Symmetry Alignment and Evaluation**
- **Attention head alignment**: For each attention layer, compute pairwise cosine similarity between recovered and ground-truth heads (considering Q||K||V||O concatenation). Run Hungarian algorithm to find optimal permutation.
- **FFN neuron alignment**: For gated MLP, compute pairwise similarity between neuron columns of (up_proj||gate_proj) and rows of down_proj. Hungarian alignment.
- **Scale normalization**: For each RMSNorm, normalize weights to unit norm before comparison.
- **Metrics**: Per-matrix cosine similarity (Q, K, V, O, up, gate, down, norm), per-block aggregated, and per-layer-type aggregated.

### Failure Modes and Diagnostics
- **Oracle-prefix fails**: Method itself is flawed → investigate loss landscape, gradient signal strength.
- **Oracle-prefix succeeds but pure-logits fails**: Prefix mismatch is the bottleneck → quantify and report as a finding.
- **Permutation alignment gives low quality**: Block is not identifiable up to known symmetries → report as depth boundary.
- **Sensitivity matching doesn't help (β=0 equally good)**: Sensitivity matching is unnecessary → report honestly.

### Novelty and Elegance
**Closest work**: Carlini et al. (ICML 2024) — algebraic lm_head extraction.
**Exact difference**: S-PSI extends to multi-block suffix with identifiability analysis, introducing sensitivity matching as a constraint amplifier and oracle-prefix/pure-logits decomposition as a diagnostic framework.
**Why focused**: One algorithm, one decomposition framework, one alignment procedure.

## Validation Sketch

### Claim 1 (Primary): Under oracle-prefix, S-PSI recovers lm_head + last 2 blocks with cos_sim > 0.9
- **Model**: Qwen3.5-0.6B (28 layers).
- **Setting**: Oracle-prefix (teacher's prefix copied to student).
- **Baselines**: (a) Standard KD, (b) Block-coordinate distillation (β=0, no sensitivity matching).
- **Metric**: Per-matrix permutation-aligned cosine similarity, averaged over 5 random suffix initializations.
- **Controls**: Per-matrix breakdown (Q/K/V/O/up/gate/down/norm individually).

### Claim 2: Sensitivity matching is necessary for identifiable recovery
- **Ablation**: Full S-PSI (β=0.1) vs β=0 (logit MSE only).
- **Metric**: Cross-initialization variance of recovered parameters. Low variance = identifiable (different inits converge to same solution).
- **Additional control**: Known-symmetry null test — apply known permutation to teacher, verify alignment procedure recovers original.

### Claim 3: Oracle-prefix vs pure-logits decomposition
- **Experiment**: Same method, two regimes. Oracle-prefix and pure-logits (frozen random prefix).
- **Metric**: Per-matrix cos_sim in both regimes.
- **Expected**: Oracle-prefix > 0.9 for last 2 blocks. Pure-logits > 0.8 for lm_head, degrading for deeper blocks.
- **Insight**: Quantifies how much prefix fidelity matters for suffix recovery.

### Claim 4 (Analysis): Depth-recovery boundary
- **Experiment**: Continue inversion past 2 blocks, record per-block recovery.
- **Metric**: Depth vs cos_sim curve.
- **Expected**: Sharp transition from recoverable to non-recoverable.

## Compute & Timeline
- **Oracle-prefix experiments** (5 inits × primary): ~120 GPU-hours
- **Pure-logits experiments** (5 inits × primary): ~120 GPU-hours
- **Ablations** (sensitivity, suffix refinement): ~80 GPU-hours
- **Depth boundary analysis**: ~60 GPU-hours
- **Buffer**: ~70 GPU-hours
- **Total**: ~450 GPU-hours
- **Timeline**: 8 weeks
