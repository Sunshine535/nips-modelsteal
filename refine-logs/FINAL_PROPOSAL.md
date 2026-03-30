# Research Proposal: S-PSI — How Identifiable Are Transformer Weights from Logits?

## Problem Anchor
- **Bottom-line problem**: Can we recover actual internal weight parameters of a black-box LLM from output logits?
- **Must-solve bottleneck**: No method studies whether weight recovery is IDENTIFIABLE from logits, let alone achieves it.
- **Non-goals**: Not KD, not data extraction, not attacking production APIs.
- **Constraints**: 8x A100-80GB, ~450 GPU-hours, Qwen3.5-0.6B primary, NeurIPS 2026.
- **Success condition**: Under boundary-state oracle, last 1-2 blocks recoverable (per-matrix cos_sim > 0.9, consistent across 5 random inits). Characterize degradation under pure-logits.

## Technical Gap

Carlini et al. (ICML 2024) showed the output projection layer (lm_head) is algebraically extractable from API queries. But nobody has asked the deeper question: **how far does logit-based identifiability extend into the transformer?** This is a fundamental question for LLM security — if internal layers are identifiable from logits, the attack surface extends far beyond the output layer.

Key insight: For a fixed recovered suffix and a known block input (boundary state), logit-sensitivity matching — how logits respond to input perturbations — provides additional constraints that narrow the set of consistent parameterizations to near-unique solutions (up to known symmetries: head permutation, neuron permutation, scale absorption by RMSNorm).

## Method Thesis

S-PSI (Sensitivity-Guided Progressive Suffix Inversion) is a progressive block-wise inversion algorithm that recovers transformer suffix parameters from logit observations. The paper's main contribution is an identifiability study: under what conditions are transformer weights recoverable from logits?

## Contribution Focus
- **Dominant**: Identifiability decomposition — boundary-state oracle shows that last 1-2 blocks are identifiably recoverable from logits; pure-logits regime shows practical limits.
- **Mechanism**: S-PSI algorithm with sensitivity matching as the core constraint amplifier.
- **Analysis**: Depth-recovery boundary, oracle vs pure-logits gap quantification.
- **Non-contributions**: No new architectures, no defenses.

## Proposed Method

### Experimental Regimes
1. **Boundary-state oracle** (upper bound): Teacher's exact intermediate activations h_{i-1}^T(x) are injected as input to each block being recovered. Isolates suffix identifiability from prefix noise.
2. **Pure-logits** (realistic): Student prefix is random-init frozen. Tests practical attack scenario.
3. **Comparison**: Gap between oracle and pure-logits quantifies impact of prefix fidelity.

### Setup
- Teacher T: Qwen3.5-0.6B (28 layers), frozen. Oracle setting can extract intermediate activations.
- Student S: same architecture. Oracle: teacher prefix copied, suffix randomized. Pure-logits: fully random.
- Query pool Q: 10K WikiText inputs, max_len=256.

### Phase 0: Precompute Teacher Cache
- Cache z_T(x) and z_T(x'_{p,r}) for P=4 positions, R=2 replacements per input.
- Oracle: also cache h_i^T(x) for relevant layer boundaries.
- 90K teacher forward passes, one-time cost (~5 GPU-hours).

### Phase 1: lm_head Recovery
- **Optimization variables**: lm_head.weight (vocab_size × hidden_dim).
- **Input**: h_N^T(x) (oracle) or student h_N(x) (pure-logits).
- **Loss**: L = E_x[||lm_head(h_N) - z_T(x)||²].
- **Optimizer**: Adam, lr=1e-3, 5K steps, batch=32.

### Phase 2-3: Block B_N, B_{N-1} Recovery
For each block B_i (i = N, then N-1):
- **Optimization variables**: All parameters of B_i (q/k/v/o_proj, up/gate/down_proj, input_layernorm, post_attention_layernorm).
- **All else frozen** (recovered suffix + prefix).
- **Oracle input**: h_{i-1}^T(x) injected directly as block input.
- **Pure-logits input**: h_{i-1} from student forward through frozen prefix.
- **Forward**: B_i(h_{i-1}) → through recovered suffix → z_S(x).
- **Per-block loss**:
  ```
  L(θ_Bi) = L_logit + 0.1 · L_sensitivity + 1e-5 · L_reg
  L_logit = E_x[||z_S(x) - z_T(x)||²]
  L_sensitivity = E_{x,p,r}[||Δz_S(x,p,r) - Δz_T(x,p,r)||²]
  L_reg = Σ||w||²
  ```
  where Δz(x,p,r) = z(x'_{p,r}) - z(x) for token replacement at position p with token r.
- **Optimizer**: Adam, lr=1e-4, max 10K steps, batch=16.
- **Stopping**: Loss improvement < 1e-7 over 500 consecutive steps.

### Phase 4: Symmetry Alignment and Evaluation
- **Attention head alignment**: Hungarian algorithm on Q||K||V||O cosine similarity matrix.
- **FFN neuron alignment**: Hungarian algorithm on (up||gate, down) similarity.
- **RMSNorm normalization**: Remove scale factors before comparison.
- **Metrics**: Per-matrix cosine similarity (Q, K, V, O, up, gate, down, norm — 8 matrices per block).
- **Cross-init consistency**: Variance of cosine similarity across 5 random initializations.

### Phase 5: Depth Boundary Analysis
- Continue inversion past 2 blocks into deeper layers.
- Record per-block recovery quality.
- Identify transition from identifiable to non-identifiable.

### Cost Accounting

| Phase | GPU-hours |
|-------|-----------|
| Teacher cache | ~5 |
| lm_head (5 inits × 2 regimes) | ~10 |
| Block N (5 inits × 2 regimes) | ~60 |
| Block N-1 (5 inits × 2 regimes) | ~60 |
| Depth boundary (3-4 blocks) | ~80 |
| Ablations (β=0, suffix refinement) | ~60 |
| Alignment + evaluation | ~10 |
| Buffer | ~65 |
| **Total** | **~350** |

## Validation

### Claim 1 (Primary): Under boundary-state oracle, S-PSI recovers last 1-2 blocks with cos_sim > 0.9
- 5 random suffix initializations on Qwen3.5-0.6B.
- Per-matrix cosine similarity breakdown (16 measurements for 2 blocks).
- Baselines: standard KD, block-coordinate distillation (β=0, no sensitivity matching).
- Success criteria: per-matrix cos_sim > 0.9, cross-init variance < 0.02.
- **Falsification controls**: wrong-teacher recovery (recover with a different teacher of same architecture — should fail), held-out perturbation families.

### Claim 2: Sensitivity matching is necessary for identifiable recovery
- Ablation: β=0.1 vs β=0.
- Metric: cross-initialization cosine similarity variance (low = identifiable).
- Known-symmetry null test: permute teacher heads, verify alignment recovers original.

### Claim 3: Oracle vs pure-logits gap quantifies prefix-mismatch impact
- Same S-PSI method in both regimes.
- Per-matrix and per-block comparison.
- Expected: oracle >> pure-logits for deeper blocks.

### Claim 4 (Analysis): Depth-recovery boundary
- Continue inversion past identifiable suffix.
- Plot depth vs recovery curve.
- Report local-identifiability diagnostic: empirical rank of block-to-logit Jacobian at recovered solution.

## Ethical Considerations
- Only open-weight models simulating black-box access.
- No production API attacks.
- Results inform defense design (responsible disclosure framing).
- Identifiability study benefits both attackers and defenders.

## Timeline: 8 weeks (May 19 – July 13, 2026)
