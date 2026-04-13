# Gap Map

Field gaps identified across the research landscape. Stable IDs: G1, G2, ...

---

## G1: No theory connects query design to weight recovery guarantees

- **Status**: unresolved
- **Linked ideas**: idea:001 (Transformer Tomography)
- **Source**: Literature survey 2024-2026
- **Notes**: Existing model stealing works treat query design heuristically. No formal framework links the observability of the query set (in a control-theoretic sense) to the identifiability of internal parameters.

## G2: Symmetry-unaware optimization conflates gauge directions with real error

- **Status**: unresolved
- **Linked ideas**: idea:001
- **Source**: Carlini et al. 2024, Rawal et al. 2025
- **Notes**: Transformer parameters have continuous symmetries (RMSNorm scaling, neuron permutations). Without gauge projection, gradient-based recovery wastes capacity fitting directions that don't change the function.

## G3: Random initialization wastes query budget exploring unobservable subspaces

- **Status**: unresolved
- **Linked ideas**: idea:001
- **Source**: S-PSI codebase analysis
- **Notes**: Current S-PSI uses random init for each suffix block. If the Gramian has a low-rank observable subspace, we should initialize along those directions.

## G4: No per-layer difficulty metric predicts recovery quality a priori

- **Status**: unresolved
- **Linked ideas**: idea:001
- **Source**: empirical observation
- **Notes**: Some layers are harder to recover than others, but there's no predictive metric. The Gramian eigenspectrum (especially lambda_min) may fill this gap.

## G5: Scalability beyond 1B parameters unknown

- **Status**: unresolved
- **Linked ideas**: —
- **Source**: field-wide
- **Notes**: All parameter inversion works target ≤1B models. Whether the approach scales to 7B+ is open.
