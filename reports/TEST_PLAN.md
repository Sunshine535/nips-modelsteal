# Test Plan

| Test | Purpose | Command | Expected Result | Status |
|------|---------|---------|----------------|--------|
| test_query_budget_tracks_counts | QueryBudget records topk/probe | `pytest tests/test_oracles.py::test_query_budget_tracks_counts` | topk=5, probe=100, total=105 | PASS |
| test_query_budget_enforces_limit | Budget raises on overflow | `pytest tests/test_oracles.py::test_query_budget_enforces_limit` | RuntimeError on 2nd call | PASS |
| test_topk_oracle_returns_only_K | Oracle returns K values per position, not full logits | `pytest tests/test_oracles.py::test_topk_oracle_returns_only_K` | (B, T, 5) shape, budget charged | PASS |
| test_probe_oracle_returns_only_probe_ids | Probe returns only at probe indices | `pytest tests/test_oracles.py::test_probe_oracle_returns_only_probe_ids` | (B, T, 4) shape, budget charged by 2*4 | PASS |
| test_strict_api_combined | Combined API shares budget | `pytest tests/test_oracles.py::test_strict_api_combined` | topk_queries=1, probe_queries=20 | PASS |
| test_recovery_exact_on_noiseless | lstsq recovers h when overdetermined | `pytest tests/test_logit_completion.py::test_recovery_exact_on_noiseless` | cos(h_hat, h_true) > 0.95 | PASS |
| test_complete_merges_topk | Exact top-K values preserved after merge | `pytest tests/test_logit_completion.py::test_complete_merges_topk` | gathered top-K equal exact values | PASS |
| test_uncertainty_weights | Calibration produces weights | `pytest tests/test_logit_completion.py::test_uncertainty_weights` | weights shape (V,), non-negative | PASS |
| test_weights_exact_for_topk | Top-K tokens get weight 1.0 | `pytest tests/test_logit_completion.py::test_weights_exact_for_topk` | All top-K positions = 1.0 | PASS |
| test_manifest_saves_required_fields | Manifest captures provenance | `pytest tests/test_manifest.py::test_manifest_saves_required_fields` | timestamp/git_hash/seed/config present | PASS |
| test_git_hash_returns_string | Git hash utility works | `pytest tests/test_manifest.py::test_git_hash_returns_string` | non-empty string | PASS |

## Run All
```bash
python3 -m pytest tests/test_oracles.py tests/test_logit_completion.py tests/test_manifest.py -v
```
Result: **11/11 PASS** on CPU, no GPU needed.

## Missing Tests (Task 6 — deferred, not blocking)
- `tests/test_metrics.py` — teacher-vs-teacher KL=0 on real model, random student KL>0
- `tests/test_data_splits.py` — WikiText train/validation split names and hashes saved in manifest

These are infrastructure tests that could be added but are not required for the Q-UMC mechanism validation. The manifest saves split info already; a test can later assert on its schema.
