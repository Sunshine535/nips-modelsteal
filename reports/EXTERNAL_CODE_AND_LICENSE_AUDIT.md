# External Code and License Audit

| Paper/Repo | License | Used As | In Code? | Risk |
|-----------|---------|---------|----------|------|
| Carlini et al. 2024 (arXiv:2403.06634) | N/A (paper) | Citation + baseline concept | baselines/carlini_2024/ (our reimplementation) | LOW — our own code, not copied |
| Finlayson et al. 2024 (arXiv:2403.09539) | N/A (paper) | Citation only | Not in code | NONE |
| Clone 2025 (arXiv:2509.00973) | No official code found | Citation + baseline concept | baselines/clone_2025/ (our reimplementation) | LOW |
| RankDistil (AISTATS 2021) | N/A (paper) | Citation, will implement baseline from scratch | Not yet in code | NONE |
| PLD (OpenReview) | N/A (paper) | Citation, will implement baseline from scratch | Not yet in code | NONE |
| BiLD (thesis/COLING) | N/A (paper) | Citation, will implement baseline from scratch | Not yet in code | NONE |
| Delta KD (arXiv:2509.14526) | N/A (paper) | Citation only | Not in code | NONE |
| Proxy-KD (arXiv:2401.07013) | N/A (paper) | Citation only | Not in code | NONE |
| HuggingFace transformers | Apache 2.0 | Dependency | import only | NONE |
| PyTorch | BSD-3 | Dependency | import only | NONE |

## Internal Code Copies
- `scripts/algebraic_recovery_v4_richinput.py:97` — "copied from v3 to keep v4 self-contained" (internal)
- `scripts/diagnose_phase2_failure.py:72` — "copied / adapted from algebraic_recovery_v3" (internal)

## Conclusion
No external code is copied into `src/` or `scripts/` main method. All implementations are original. External papers are cited only. Baselines are our own reimplementations under `baselines/`.

**Provenance status: CLEAR. No blocking risk.**
