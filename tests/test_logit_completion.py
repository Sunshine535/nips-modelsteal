"""Tests for R2-compliant CalibratedLogitCompleter."""
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.logit_completion import CalibratedLogitCompleter
from src.basis_provider import BasisProvider, Basis


@pytest.fixture
def toy_setup():
    torch.manual_seed(0)
    V, d, K_probe = 50, 8, 20
    W_full = torch.randn(V, d)
    probe_ids = torch.arange(K_probe)
    W_probe = W_full[probe_ids]
    basis = Basis(W=W_full, source="carlini_recovered",
                  access_level="strict_black_box", n_queries_used=100)
    return V, d, K_probe, W_full, W_probe, probe_ids, basis


def test_completer_requires_basis_object(toy_setup):
    V, d, K_probe, W_full, W_probe, probe_ids, basis = toy_setup
    with pytest.raises(TypeError, match="Basis object"):
        CalibratedLogitCompleter(W_probe, probe_ids, W_full, device="cpu")


def test_completer_accepts_basis(toy_setup):
    V, d, K_probe, W_full, W_probe, probe_ids, basis = toy_setup
    c = CalibratedLogitCompleter(W_probe, probe_ids, basis, device="cpu")
    assert c.basis_source == "carlini_recovered"
    assert c.basis_access_level == "strict_black_box"


def test_recovery_exact_on_noiseless(toy_setup):
    V, d, K_probe, W_full, W_probe, probe_ids, basis = toy_setup
    c = CalibratedLogitCompleter(W_probe, probe_ids, basis, device="cpu")
    h_true = torch.randn(4, d)
    z_true = h_true @ W_full.T
    z_probe = z_true[:, probe_ids]
    h_hat = c.recover_h(z_probe)
    cos = torch.nn.functional.cosine_similarity(h_hat, h_true, dim=-1)
    assert cos.min() > 0.95


def test_complete_merges_topk(toy_setup):
    V, d, K_probe, W_full, W_probe, probe_ids, basis = toy_setup
    c = CalibratedLogitCompleter(W_probe, probe_ids, basis, device="cpu")
    B, T, K = 2, 3, 5
    h_true = torch.randn(B, T, d)
    z_true = h_true @ W_full.T
    topk_vals, topk_idx = z_true.topk(K, dim=-1)
    probe_logits = z_true[:, :, probe_ids]
    z_complete = c.complete(topk_vals, topk_idx, probe_logits, (B, T, V))
    assert z_complete.shape == (B, T, V)
    gathered = z_complete.gather(-1, topk_idx)
    assert torch.allclose(gathered, topk_vals, atol=1e-5)


def test_calibration_uses_only_probes(toy_setup):
    V, d, K_probe, W_full, W_probe, probe_ids, basis = toy_setup
    c = CalibratedLogitCompleter(W_probe, probe_ids, basis, device="cpu")

    N_cal = 50
    h_cal = torch.randn(N_cal, d)
    z_cal = h_cal @ W_full.T
    cal_probe_in = z_cal[:, probe_ids]

    heldout_ids = torch.arange(K_probe, K_probe + 15)
    heldout_targets = z_cal[:, heldout_ids]
    heldout_W = W_full[heldout_ids]

    diag = c.fit_calibration_from_probes(
        cal_input_probe_logits=cal_probe_in,
        heldout_probe_ids=heldout_ids,
        heldout_probe_targets=heldout_targets,
        heldout_W_rows=heldout_W,
    )
    assert "heldout_mse_mean" in diag
    assert diag["n_heldout_probes"] == 15
    assert c.uncertainty_weights is not None
    assert c.uncertainty_weights.shape == (V,)


def test_weights_exact_for_topk(toy_setup):
    V, d, K_probe, W_full, W_probe, probe_ids, basis = toy_setup
    c = CalibratedLogitCompleter(W_probe, probe_ids, basis, device="cpu")
    B, T, K = 1, 1, 5
    topk_idx = torch.randint(0, V, (B, T, K))
    w = c.get_weights(topk_idx, V)
    topk_w = w.gather(-1, topk_idx)
    assert (topk_w == 1.0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
