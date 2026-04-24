"""Tests for calibrated logit completion."""
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.logit_completion import CalibratedLogitCompleter


@pytest.fixture
def toy_setup():
    torch.manual_seed(0)
    V, d, K_probe = 50, 8, 20
    W_full = torch.randn(V, d)
    probe_ids = torch.arange(K_probe)
    W_probe = W_full[probe_ids]
    return V, d, K_probe, W_full, W_probe, probe_ids


def test_recovery_exact_on_noiseless(toy_setup):
    V, d, K_probe, W_full, W_probe, probe_ids = toy_setup
    completer = CalibratedLogitCompleter(W_probe, probe_ids, W_full, device="cpu")

    h_true = torch.randn(4, d)
    z_true = h_true @ W_full.T
    z_probe = z_true[:, probe_ids]

    h_hat = completer.recover_h(z_probe)
    cos = torch.nn.functional.cosine_similarity(h_hat, h_true, dim=-1)
    assert cos.min() > 0.95, f"Recovery cos too low: {cos.min():.4f}"


def test_complete_merges_topk(toy_setup):
    V, d, K_probe, W_full, W_probe, probe_ids = toy_setup
    completer = CalibratedLogitCompleter(W_probe, probe_ids, W_full, device="cpu")

    B, T, K = 2, 3, 5
    h_true = torch.randn(B, T, d)
    z_true = h_true @ W_full.T

    topk_vals, topk_idx = z_true.topk(K, dim=-1)
    probe_logits = z_true[:, :, probe_ids]

    z_complete = completer.complete(topk_vals, topk_idx, probe_logits, (B, T, V))
    assert z_complete.shape == (B, T, V)

    gathered = z_complete.gather(-1, topk_idx)
    assert torch.allclose(gathered, topk_vals, atol=1e-5), "Top-K values not preserved"


def test_uncertainty_weights(toy_setup):
    V, d, K_probe, W_full, W_probe, probe_ids = toy_setup
    completer = CalibratedLogitCompleter(W_probe, probe_ids, W_full, device="cpu")

    N_cal = 100
    h_cal = torch.randn(N_cal, d)
    z_cal_true = h_cal @ W_full.T
    z_cal_probe = z_cal_true[:, probe_ids]

    completer.fit_calibration(z_cal_probe, z_cal_true)
    assert completer.uncertainty_weights is not None
    assert completer.uncertainty_weights.shape == (V,)
    assert completer.uncertainty_weights.min() >= 0


def test_weights_exact_for_topk(toy_setup):
    V, d, K_probe, W_full, W_probe, probe_ids = toy_setup
    completer = CalibratedLogitCompleter(W_probe, probe_ids, W_full, device="cpu")

    B, T, K = 1, 1, 5
    topk_idx = torch.randint(0, V, (B, T, K))
    w = completer.get_weights(topk_idx, V)
    assert w.shape == (B, T, V)
    topk_w = w.gather(-1, topk_idx)
    assert (topk_w == 1.0).all(), "Top-K tokens must have weight 1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
