# Query Pack — nips-modelsteal

**Generated**: 2026-04-09T01:50:00Z | **Budget**: 8000 chars

## Project Direction (300 chars)
Progressive Parameter Inversion: recover LLM internal weights from black-box logit access. Core thesis: the Observability Gramian G(Q) on symmetry-quotiented parameter space governs suffix-layer weight recovery. Target NeurIPS 2026. Hardware: 4×A100-80GB. Model: Qwen3.5-0.8B.

## Top 5 Gaps (1200 chars)
G1: No theory connects query design to weight recovery guarantees — existing works use heuristic queries
G2: Symmetry-unaware optimization conflates gauge directions with real error — RMSNorm/neuron symmetries ignored
G3: Random init wastes query budget in unobservable subspaces — need algebraic init along observable directions
G4: No per-layer difficulty metric predicts recovery quality a priori — Gramian eigenspectrum may fill this
G5: Scalability beyond 1B unknown — all prior work ≤1B

## Paper Clusters (1600 chars)
**Model Stealing**: Tramer2016, Carlini2024 (cryptanalytic extraction), Rawal2025 (progressive suffix). Extract model weights/behavior from API. Tramer showed linear models are exactly invertable; Carlini showed last-layer extraction for production models; Rawal introduced suffix-progressive strategy.
**Identifiability Theory**: Bona-Pellissier2023 (Gaussian mixtures), Stock2022 (ReLU networks). Theoretical conditions under which parameters are recoverable up to symmetry. Gap: none for transformers specifically.
**Neural Network Symmetries**: Godfrey2022 (permutation symmetries), Ainsworth2023 (Git Re-Basin). Weight-space symmetries form groups; loss landscape has equivalent minima. Relevant for defining the quotient space.
**Active Query Design**: Settles2009 (active learning), Ren2021 (survey). Selecting informative queries to maximize information gain. Gap: not applied to parameter recovery.

## Failed Ideas (1400 chars)
*(none yet — wiki just initialized)*

## Top Papers (1800 chars)
paper:carlini2024_stealing — Stealing production LM weights via cryptanalytic extraction (core)
paper:rawal2025_spsi — S-PSI progressive suffix inversion, our starting point (core)
paper:tramer2016_stealing — Stealing ML models via prediction APIs, foundational (core)
paper:bona2023_identifiability — Identifiability of Gaussian mixtures, theoretical framework (related)
paper:godfrey2022_symmetries — Neural network symmetry groups (related)

## Active Chains (900 chars)
G1 (no query theory) → idea:001 Transformer Tomography (Gramian-based observability)
G2 (symmetry-unaware) → idea:001 gauge projection via tangent space
G3 (random init) → idea:001 algebraic init along observable eigenvectors
G4 (no difficulty metric) → idea:001 Gramian eigenspectrum as predictor

## Open Unknowns (500 chars)
- Does algebraic init actually converge faster, or does gradient descent wash it out?
- How tight is the Gramian eigenvalue → cosine similarity correlation?
- Can Fisher-optimal queries be computed tractably for 1B+ models?
- What is the practical gauge dimension for Qwen3.5 suffix blocks?
