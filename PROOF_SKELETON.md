# Proof Skeleton: Transformer Tomography

> **Status (2026-04-14, updated after Round 4):** This document was created during Phase 0.5 of proof checking and contains the original discovery process. All issues flagged here (MC-2, MC-5, MC-6) were resolved in Rounds 1-3. **Round 4 (nightmare/beast)** found 3 additional CRITICAL issues: GL(d_head) V/O symmetry missing, C* Q/K RoPE-commuting symmetry missing, and gauge dimension wrong (14,976/block not 6,660). All resolved. See `PROOF_AUDIT.md` for the full 4-round fix history and final 8.5/10 score.

## 1. Dependency DAG

```
Assumptions
  ├── A1: Architecture knowledge (Qwen2.5-0.5B structure)
  ├── A2: Differentiability of f_T w.r.t. suffix parameters
  ├── A3: RMSNorm + SiLU-gated MLP structure
  └── A4: Gaussian noise model (for Fisher information remark only)

Definitions
  ├── D1 (Def 1): Transformer symmetry group G = (R>0)^d × (R>0)^{d_ff}
  │     Uses: A3
  ├── D2 (Def 2): Suffix observability Gramian G(Q)
  │     Uses: A2
  └── D3 (Def 3): Projected observability Gramian G_perp(Q) = P_perp G P_perp
        Uses: D1, D2

Propositions/Theorems
  ├── P1 (Prop 1): Gauge null-space — G(Q)U = 0
  │     Uses: D1, D2
  │     Used by: D3, T3
  │
  ├── T3 (Thm 3): Local first-order identifiability criterion
  │     Uses: D2, D3, P1
  │     Used by: Discussion (Section 7), T4 interpretation
  │
  ├── T4 (Thm 4): Gramian rank upper bound
  │     Uses: D2, chain rule factorization
  │     Used by: Discussion (rank analysis), Remark 3
  │     Note: Independent of gauge projection
  │
  ├── T5 (Thm 5): Depth screening
  │     Uses: D2, Jacobian chain-rule decomposition
  │     Used by: Discussion (Block 22 vs 23 analysis)
  │
  └── R3 (Remark 3): Fisher information lower bound
        Uses: T4, A4 (Gaussian noise assumption)
        Used by: Discussion (query complexity bound)
```

**Cycle check**: No cycles detected. All dependencies flow downward from assumptions → definitions → propositions → theorems → remarks.

## 2. Assumption Ledger

| ID | Assumption | Where stated | Where used | Verified? |
|----|-----------|-------------|-----------|-----------|
| A1 | Architecture knowledge (identical student/teacher) | §3, line ~161 | Throughout | BY CONSTRUCTION |
| A2 | Differentiability of z(x) w.r.t. θ_suf | Implicit | P1, D2, T3, T4, T5 | IMPLICIT — requires MATH attention backend (noted in App C, line ~844) |
| A3 | RMSNorm + SiLU-gated MLP structure | §4.1, Def 1 | D1, symmetry group | BY CONSTRUCTION (model choice) |
| A4 | Gaussian noise on logits | R3 only (line ~304) | R3 (Fisher bound) | STATED EXPLICITLY as model assumption |
| A5 | Gauge basis U is orthonormal | Remark after Def 1, line ~220 | P1, D3, T3 | CONSTRUCTIVE — built via QR in code |
| A6 | Continuous symmetries are complete (no missing symmetries) | D1, implicit | P1, T3 | UNVERIFIED — paper claims "complete" but no proof of completeness |
| A7 | Fixed parameterization point for linearization | Implicit in all Jacobian-based statements | T3, T4, T5 | IMPLICIT — "locally at current θ" |

## 3. Typed Symbol Table

| Symbol | Type | Depends on | Definition/First use |
|--------|------|-----------|---------------------|
| T, S | Transformer models | — | §3 |
| L | scalar ∈ Z+, L=24 for Qwen | — | §3 |
| K | scalar ∈ Z+, suffix depth | — | §3, also used as #suffix positions in experiments (overloaded!) |
| d | scalar ∈ Z+, d=896 | — | §3 |
| d_ff | scalar ∈ Z+, d_ff=4864 | — | §4.1 |
| V | scalar ∈ Z+, V=151936 | — | §3 |
| H | scalar ∈ Z+, H=14 | — | §6 |
| θ_suf | vector ∈ R^p | L, K | §3, eq implicit |
| Q | set of input sequences | — | §4.2 |
| N | scalar = |Q| | Q | §4.2 |
| B | scalar, query budget | — | §3 |
| z(x) | vector ∈ R^V (or R^{K_pos × V} in practice) | θ_suf, x | §3, eq (2) |
| h_i(x) | vector ∈ R^d | x, block i | §3 |
| J_x | matrix ∈ R^{V × p} | x, θ_suf | Eq (2), Def 2 |
| G(Q) | matrix ∈ R^{p × p}, symmetric PSD | Q, θ_suf | Def 2, eq (2) |
| G | Lie group (R>0)^{2d+d_ff} per block | A3 | Def 1 |
| g (fraktur) | Lie algebra of G, tangent space | G | Remark after Def 1 |
| U | matrix ∈ R^{p × d_g}, orthonormal columns | g | Remark, line ~220 |
| d_g | scalar = 2d + d_ff per block | d, d_ff | line ~217 |
| P_perp | matrix ∈ R^{p × p}, projector = I - UU^T | U | Def 3 |
| G_perp(Q) | matrix ∈ R^{p × p}, symmetric PSD | G(Q), P_perp | Def 3 |
| V (probe) | matrix ∈ R^{p × k}, orthonormal columns | — | §4.3, eq (4) |
| k | scalar ∈ Z+, probe count | — | §4.3 |
| G_tilde | matrix ∈ R^{k × k} = V^T G V | V, G | Eq (4) |
| S_i | matrix ∈ R^{V × k} = [J_i v_1, ..., J_i v_k] | J, V | Eq (4) |
| p_ℓ | scalar, param count of block ℓ | — | T4 |
| d_{g,ℓ} | scalar, gauge dim of block ℓ | — | T4 |
| F_{ℓ+1:L} | function R^d → R^d | blocks ℓ+1..L | T4 |
| A_i | matrix ∈ R^{d × d}, = ∂F_{ℓ+1:L}/∂h_ℓ | — | T4, T5 |
| B_i | matrix ∈ R^{d × p_ℓ}, = ∂h_ℓ/∂θ_ℓ | — | T4 proof |
| M_{ℓ+1:L} | matrix ∈ R^{d × d}, = (1/N)∑A_i^T W_lm^T W_lm A_i | — | T5, eq (6) |
| α, β, γ | scalar loss weights | — | §5.2 |
| r | scalar, truncation rank | — | §5.3 |
| λ | scalar, ridge parameter | — | §5.3, also Tikhonov in eq (3) |

**Flag: K is overloaded** — used as (i) suffix depth in blocks (§3) and (ii) suffix positions (number of token positions observed) in experiments. These are different quantities. §6 uses K_pos for the latter, but the Gramian tables use K for suffix positions.

## 4. Canonical Quantified Statements

### Proposition 1 (Gauge null-space)
```
∀ Q (any query set), ∀ u ∈ g (Lie algebra of G):
  G(Q) u = 0
Scope: exact (holds at any parameterization, for any Q)
Proof type: Direct differentiation of invariance condition
```

### Theorem 3 (Local first-order identifiability)
```
∀ v ⊥ g with ||v||=1:
  v is locally first-order identifiable from Q
  ⟺ v^T G_perp(Q) v > 0

Full identifiability (up to gauge):
  rank(G_perp(Q)) = p - d_g
  ⟺ every non-gauge direction is identifiable

Scope: LOCAL, FIRST-ORDER — paper explicitly notes (line ~257):
  "necessary condition for local identifiability [...] but not sufficient
  for global identifiability"
```

### Theorem 4 (Gramian rank upper bound)
```
∀ block ℓ, ∀ query set Q:
  rank(G_perp^(ℓ)(Q)) ≤ min(p_ℓ - d_{g,ℓ}, d·|Q|, rank(∂F/∂h_ℓ)·rank(∂h_ℓ/∂θ_ℓ))

In particular: rank ≤ d² (hidden dimension bottleneck)
For Qwen2.5-0.5B: rank ≤ 896² = 802,816 ≪ p_ℓ = 14.9M

Scope: POINTWISE at current parameterization (Jacobians depend on θ)
```

### Theorem 5 (Depth screening)
```
For block ℓ < L:
  G_perp^(ℓ)(Q) = P_perp · (∂h_ℓ/∂θ_ℓ)^T · M_{ℓ+1:L} · (∂h_ℓ/∂θ_ℓ) · P_perp

where M_{ℓ+1:L} = (1/N) ∑_i A_i^T W_lm^T W_lm A_i ∈ R^{d×d}

Signal attenuation: ∏_{j=ℓ+1}^{L} σ_min(A_j)
  — "decays exponentially when individual block Jacobians are contractive"

Scope: POINTWISE, qualitative claim about exponential decay
  (no quantitative bound on the product)
```

### Remark 3 (Fisher information lower bound)
```
Under Gaussian noise z^T(x) + N(0, σ²I):
  To recover p_ℓ - d_{g,ℓ} params to precision δ:
    σ^{-2} · N · λ_min(G_tilde/N) ≳ (p_ℓ - d_{g,ℓ})/δ²

  With r_eff ≪ p_ℓ - d_{g,ℓ}:
    Need ≥ (p_ℓ - d_{g,ℓ})/r_eff independent observation directions

  For Block 23: ≥ 465,000 directions (with r_eff = 32)

Scope: INFORMATION-THEORETIC lower bound under A4 (Gaussian noise)
```

## 5. Micro-Claim Inventory

### MC-1: RMSNorm scale absorption is a symmetry
```
Context: [Def 1, A3: RMSNorm + consumer weight structure]
⊢ Goal: W' · RMSNorm_{Dg}(x) = W · RMSNorm_g(x) when W' = W D^{-1}
Rule: Direct algebraic substitution
Side-conditions:
  - D = diag(α_1,...,α_d) with α_i > 0 ✓ (stated)
  - RMSNorm denominator rms(x) does not depend on g ✓ (by definition)
  - Consumer weight W linearly multiplies the normalized output ✓ (by structure)
```

### MC-2: Gated MLP neuron scaling is a symmetry
```
Context: [Def 1, A3: SiLU-gated FFN structure]
⊢ Goal: FFN output invariant under (gate[n,:] → α_n gate[n,:], up[n,:] → α_n up[n,:], down[:,n] → α_n^{-1} down[:,n])
Rule: Direct algebraic substitution in SiLU(gate·x) ⊙ (up·x) · down^T
Side-conditions:
  - SiLU is homogeneous of degree 1? NO! SiLU(αx) ≠ α·SiLU(x)
  ⚠️ CRITICAL: SiLU is NOT homogeneous. SiLU(x) = x·σ(x) where σ is sigmoid.
    SiLU(αx) = αx·σ(αx) ≠ α·x·σ(x) in general.
    The symmetry in eq (5) scales BOTH gate AND up by α_n, but only one goes through SiLU.
    So: SiLU(α_n · gate[n,:]·x) ⊙ (α_n · up[n,:]·x) · (α_n^{-1} · down[:,n])
      = SiLU(α_n · g_n) · α_n · u_n · α_n^{-1} · d_n
      = α_n · SiLU(g_n)/g_n · g_n · u_n · d_n  ... this doesn't simplify cleanly
    WAIT — let me re-check. The output per neuron n is:
      SiLU(w_gate[n,:] · x) * (w_up[n,:] · x) * w_down[:,n]^T
    Under the scaling:
      SiLU(α_n · w_gate[n,:] · x) * (α_n · w_up[n,:] · x) * (α_n^{-1} · w_down[:,n])^T
      = SiLU(α_n · a) * α_n * b * α_n^{-1} * c   where a=gate·x, b=up·x, c=down
      = SiLU(α_n · a) * b * c
    For this to equal the original = SiLU(a) * b * c, we need SiLU(α_n · a) = SiLU(a),
    which is only true when α_n = 1.

    ACTUALLY WAIT — re-reading eq (5) more carefully:
    gate[n,:] → α_n gate[n,:], up[n,:] → α_n up[n,:], down[:,n] → α_n^{-1} down[:,n]
    So both gate AND up are scaled by α_n (not just one of them).
    Output_n = SiLU(α_n · a_n) * (α_n · b_n) * (α_n^{-1}) = SiLU(α_n · a_n) * b_n
    This does NOT equal SiLU(a_n) * b_n unless α_n = 1.

    UNLESS the paper means a DIFFERENT scaling for gate vs up?
    Let me re-read... No, eq (5) clearly scales both gate and up by the same α_n.

    CONCLUSION: The claimed MLP neuron scaling symmetry in Def 1 / eq (5) is INCORRECT
    for SiLU activation, unless the gate and up are scaled differently.

    The CORRECT symmetry for SiLU-gated MLP would be:
    up[n,:] → α_n up[n,:], down[:,n] → α_n^{-1} down[:,n]
    (leaving gate unchanged)
    Then: SiLU(a_n) * (α_n * b_n) * (α_n^{-1}) = SiLU(a_n) * b_n ✓

    Or alternatively: gate[n,:] → α_n gate[n,:] (and absorb via SiLU nonlinearity... but that changes the function)
```

**⚠️ POTENTIAL FATAL ISSUE IDENTIFIED IN MC-2**: The gated MLP neuron scaling claimed in Definition 1 appears incorrect for SiLU activation. SiLU is not positively homogeneous, so scaling gate[n,:] by α_n changes SiLU(α_n · gate·x) ≠ α_n · SiLU(gate·x). The correct symmetry should only scale up and down (not gate), giving d_ff continuous symmetries, or scale gate differently. This affects the gauge dimension count (d_g per block) and therefore the rank bound in T4 and the identifiability analysis in T3.

> **RESOLVED (2026-04-14):** Paper Def 1 corrected to scale only up/down (gate unchanged). Code `symmetry_gauge.py` fixed. Gauge count: 2d + d_ff + 2H_kv = 6660 per block.

### MC-3: Gauge null-space proof (Prop 1)
```
Context: [D1 symmetry, D2 Gramian definition]
⊢ Goal: G(Q)u = 0 for all u ∈ g
Rule: Differentiate z(x; θ + εu) = z(x; θ) at ε=0 → J_x u = 0 → G u = 0
Side-conditions:
  - Symmetry must be EXACT (not approximate) ✓ IF MC-1 holds, ⚠️ IF MC-2 holds
  - Differentiability at current θ — A2 ✓
  - The identity z(x; θ + εu) = z(x; θ) must hold for all x — depends on symmetry being correct
```

### MC-4: Identifiability criterion proof (Thm 3)
```
Context: [D2, D3, P1]
⊢ Goal: (⇒) v^T G_perp v = 0 → v not identifiable
         (⇐) v^T G_perp v > 0 → v identifiable
Rule:
  (⇒): v^T G_perp v = ∑ ||J_i v||² = 0 → J_i v = 0 ∀i → first-order unidentifiable ✓
  (⇐): ∃ i with J_i v ≠ 0 → perturbation θ+εv changes z(x_i) at first order ✓
  (full): rank = p - d_g ⟺ null(G_perp) = g ✓

Side-conditions:
  - v ⊥ g required (stated) ✓
  - "Locally first-order identifiable" only — paper states this ✓
  - Paper correctly notes NOT sufficient for global identifiability (line ~257) ✓
```

### MC-5: Rank bound proof (Thm 4)
```
Context: [D2, chain rule]
⊢ Goal: rank(G_perp^(ℓ)) ≤ min(p_ℓ - d_{g,ℓ}, d·|Q|, rank(∂F/∂h)·rank(∂h/∂θ))
Rule: Factorize J = W_lm · A_i · B_i, then rank(stacked J) ≤ rank bounds
Side-conditions:
  - Chain rule factorization correct: J_{x,ℓ} = W_lm · (∂F_{ℓ+1:L}/∂h_ℓ) · (∂h_ℓ/∂θ_ℓ) ✓
  - rank(AB) ≤ min(rank(A), rank(B)) ✓
  - The claim "rank ≤ d²" from hidden dim bottleneck:
    Each J_{x,ℓ} maps through R^d, so column space of J is at most d-dimensional.
    For |Q| queries, stacked Jacobian has column space ⊂ union of d-dim subspaces.
    ⚠️ The claim rank ≤ d² needs: rank of stacked [A_1 B_1; ...; A_N B_N] ≤ d·d = d²?
    Actually J_{x_i,ℓ} = W_lm A_i B_i ∈ R^{V × p_ℓ}. The key bottleneck:
    A_i ∈ R^{d×d}, B_i ∈ R^{d×p_ℓ}, so A_i B_i ∈ R^{d×p_ℓ} with rank ≤ d.
    Stacking N such: rank([A_1 B_1; ...; A_N B_N]) ≤ N·d... wait that's d·|Q|.
    Where does d² come from? The paper says "bounded by the hidden dimension bottleneck"
    but the direct bound from the proof is d·|Q|, not d².
    ⚠️ The d² bound seems to come from rank(∂F/∂h)·rank(∂h/∂θ) ≤ d·d = d².
    But ∂h_ℓ/∂θ_ℓ ∈ R^{d×p_ℓ} has rank ≤ d, and ∂F/∂h_ℓ ∈ R^{d×d} has rank ≤ d.
    The product rank: rank(AB) ≤ min(rank(A), rank(B)).
    So rank(A_i B_i) ≤ min(d, d) = d for each i.
    Stacking: rank ≤ |Q|·d.
    The term rank(∂F/∂h)·rank(∂h/∂θ) in the min is questionable — when does rank(AB) ≤ rank(A)·rank(B)?
    This is NOT a standard matrix rank inequality. rank(AB) ≤ min(rank(A), rank(B)), NOT rank(A)·rank(B).
    ⚠️ POSSIBLE ERROR: The third term in the min uses a product of ranks, but rank(AB) ≤ min(rank(A), rank(B)), not the product.
```

> **RESOLVED (2026-04-14):** Theorem 4 rewritten to use full sequence state H_ℓ ∈ R^{Td}. Rank bound: min(p_ℓ - d_{g,ℓ}, Td·|Q|). The d² claim removed; factorization through R^{Td} is correct with self-attention coupling positions.

### MC-6: Depth screening derivation (Thm 5)
```
Context: [D2, chain rule decomposition]
⊢ Goal: G_perp^(ℓ) = P_perp · (∂h/∂θ)^T · M · (∂h/∂θ) · P_perp
Rule: Substitute J factorization into G = (1/N)∑J^T J
Side-conditions:
  - J_{x_i,ℓ} = W_lm · A_i · B_i where B_i = ∂h_ℓ/∂θ_ℓ ✓
  - Then J^T J = B_i^T A_i^T W_lm^T W_lm A_i B_i
  - Sum: G = (1/N) ∑ B_i^T (A_i^T W_lm^T W_lm A_i) B_i
  ⚠️ This does NOT factor as B^T M B unless B_i is constant across queries!
  But B_i = ∂h_ℓ/∂θ_ℓ|_{x_i} depends on the query x_i (through the hidden state).
  Eq (6) writes: (∂h_ℓ/∂θ_ℓ)^T · M · (∂h_ℓ/∂θ_ℓ) outside the sum.
  This is only correct if ∂h_ℓ/∂θ_ℓ does not depend on the query.

  For the LAST block (ℓ = L), ∂h_L/∂θ_L depends on the input h_{L-1}(x),
  which varies across queries. So B_i IS query-dependent.

  ⚠️ CRITICAL: Theorem 5's eq (6) appears to factor out ∂h_ℓ/∂θ_ℓ from the sum,
  but this Jacobian is query-dependent. The equation should keep B_i inside the sum.
```

> **RESOLVED (2026-04-14):** Theorem 5 rewritten with A_i, B_i kept inside the sum. Uses sequence-level Jacobians A_i ∈ R^{Td×Td}, B_i ∈ R^{Td×p_ℓ}. σ_max upper bound for suppression (not σ_min).

### MC-7: Exponential decay claim (Thm 5)
```
Context: [T5 depth screening]
⊢ Goal: "signal attenuation scales as ∏_{j=ℓ+1}^{L} σ_min(A_j)"
Rule: If M ≈ ∏ A_j^T A_j, then λ_min(M) ≈ ∏ σ_min(A_j)²
Side-conditions:
  - A_j are the per-block Jacobians of the composition
  - The composition F_{ℓ+1:L} = f_L ∘ ... ∘ f_{ℓ+1}
  - By chain rule: ∂F/∂h_ℓ = A_L · A_{L-1} · ... · A_{ℓ+1}
  - σ_min(∏ A_j) ≥ ∏ σ_min(A_j) — this is actually ≤ for singular values of products!
  ⚠️ Actually σ_min(AB) ≤ σ_min(A)·σ_max(B) and σ_min(AB) ≥ σ_min(A)·σ_min(B)
  The lower bound ∏ σ_min(A_j) is correct as a lower bound on the minimum singular value.
  But the paper says "scales as" ∏ σ_min(A_j), which is qualitatively correct for the
  attenuation scaling behavior. ✓ (qualitative claim, not a precise bound)
```

### MC-8: Fisher information bound (Remark 3)
```
Context: [A4 Gaussian noise, T4 rank]
⊢ Goal: Need ≥ (p_ℓ - d_{g,ℓ})/r_eff independent observation directions
Rule: Cramér-Rao bound under Gaussian model
Side-conditions:
  - Gaussian noise model A4 — only applies under this assumption ✓ (stated)
  - "Independent" observation directions — requires rank accumulation
  - The specific number 465,000: (14.9M)/32 = 465,625 ✓ (arithmetic check)
  ⚠️ Note: uses r_eff = 32 from K=8 setting, but v3 shows r_eff = 128 at K=128.
  With r_eff = 128: 14.9M/128 ≈ 116,000 — still far above budget but different number.
  The remark uses the most pessimistic (K=8) effective rank.
```

## 6. Limit-Order Map

| Statement | Asymptotic claim | Parameter | Uniformity |
|-----------|-----------------|-----------|------------|
| σ_max ∝ K | Linear scaling | K (suffix positions) | Empirical, for K ∈ {8,32,64,128}, fixed model |
| κ ≈ 2.4 (constant) | Independent of K | K | Empirical, for K ∈ {8,32,64,128} |
| Depth screening: ∏ σ_min(A_j) | Exponential decay | L-ℓ (depth) | Qualitative, no rate constant |
| Query complexity: ≥ (p-d_g)/r_eff | Lower bound | p, r_eff | Under Gaussian noise model (A4) |
| cos ≈ 0.12-0.14 | Initialization bias | — | Across all init strategies, 3 seeds |

## 7. Summary of Issues Found During Skeleton Construction

> **All issues below have been RESOLVED.** See `PROOF_AUDIT.md` for fix details.

### ~~POTENTIAL FATAL: MC-2 — Gated MLP symmetry with SiLU~~ **RESOLVED**
The claimed scaling symmetry (gate, up scaled by α_n, down by α_n^{-1}) does NOT preserve the output when SiLU is not positively homogeneous. SiLU(αx) ≠ α·SiLU(x).
→ **Fix:** Paper corrected to scale only up/down. Gate unchanged. Code fixed in symmetry_gauge.py.

### ~~POTENTIAL CRITICAL: MC-5 — Rank bound uses rank product~~ **RESOLVED**
Theorem 4 bounds rank by rank(A)·rank(B), but for matrix products rank(AB) ≤ min(rank(A), rank(B)), not the product.
→ **Fix:** Theorem 4 rewritten with sequence-level factorization through R^{Td}. Bound: min(p_ℓ - d_{g,ℓ}, Td·|Q|).

### ~~POTENTIAL CRITICAL: MC-6 — Depth screening factors out query-dependent Jacobian~~ **RESOLVED**
Theorem 5 eq (6) appears to factor ∂h_ℓ/∂θ_ℓ out of the sum over queries, but this Jacobian is query-dependent (it depends on the input hidden state which varies per query).
→ **Fix:** Theorem 5 rewritten keeping A_i, B_i inside the sum. σ_max upper bound for suppression.

### SYMBOL OVERLOAD: K
K is used for both suffix block depth (§3) and number of suffix token positions observed (experimental tables). These are different quantities.
