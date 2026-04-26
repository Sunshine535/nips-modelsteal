# Prior Work Implementation Boundary

## Main Method (C-DART, strengthened from T-DART)
- All code in `src/residual_delta.py`, `src/censored_delta.py`, `src/ranking_losses.py` must be ORIGINAL implementation
- `src/adaptive_candidates.py` kept for ablation only (adaptive probing not current main mechanism)
- No external code may be copied into these files
- Implementation informed by paper descriptions but written from scratch

## Baselines (under baselines/)
- `baselines/carlini_2024/` — our reimplementation of Carlini SVD, not official code
- `baselines/clone_2025/` — our reimplementation, no official code available
- Future BiLD/PLD/RankDistil baselines — will implement from scratch under baselines/ with attribution

## Existing Q-UMC Code (archived)
- `src/logit_completion.py`, `src/oracles.py` etc. — our original code from Q-UMC path
- Kept as historical negative evidence, not main method
- Oracle infrastructure (`src/oracles.py`) may be reused for T-DART strict access control

## Forbidden
- No external GitHub method code may be placed in `src/` or `scripts/` as main method
- Official baselines must stay under `baselines/` with clear attribution
- If external code is found with unclear provenance, create `reports/POSSIBLE_CODE_PROVENANCE_RISK.md` and stop
