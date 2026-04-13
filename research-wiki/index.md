# Research Wiki Index

**Project**: nips-modelsteal — Progressive Parameter Inversion / Transformer Tomography
**Last rebuilt**: 2026-04-09T02:00:00Z

## Papers

| Slug | Title | Relevance | Year |
|------|-------|-----------|------|
| `paper:carlini2024_stealing` | Stealing Part of a Production Language Model | core | 2024 |
| `paper:rawal2025_spsi` | S-PSI: Sensitivity-Guided Progressive Suffix Inversion | core | 2025 |
| `paper:tramer2016_stealing` | Stealing Machine Learning Models via Prediction APIs | core | 2016 |

## Ideas

| ID | Title | Stage | Outcome |
|----|-------|-------|---------|
| `idea:001` | Transformer Tomography: Observability-Governed Progressive Parameter Inversion | active | pending |

## Experiments

| ID | Title | Status |
|----|-------|--------|
| *(pending — smoke test + 5 planned experiments)* | | |

## Claims

| ID | Statement | Status |
|----|-----------|--------|
| `claim:C1` | Logit access sufficient for unembedding extraction | supported |
| `claim:C2` | Progressive suffix inversion recovers multiple blocks | reported |
| `claim:C3` | Linear models recoverable with O(d) queries | supported |
| `claim:C4` | Gramian eigenspectrum predicts recovery quality | reported |
| `claim:C5` | Algebraic init converges faster than random | reported |

## Gaps

| ID | Description | Status |
|----|-------------|--------|
| G1 | No query design → recovery theory | unresolved |
| G2 | Symmetry-unaware optimization | unresolved |
| G3 | Random init wastes query budget | unresolved |
| G4 | No per-layer difficulty predictor | unresolved |
| G5 | Scalability beyond 1B unknown | unresolved |
