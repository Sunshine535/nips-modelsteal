"""Unified KD loss normalization tests (per GPT-5.5 R2 Task 4)."""
import torch
import torch.nn.functional as F
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kd_losses import sequence_kl_loss, sequence_ce_loss


def test_self_kl_is_zero():
    logits = torch.randn(2, 10, 50)
    loss = sequence_kl_loss(logits, logits, weights=None, temperature=2.0)
    assert abs(loss.item()) < 1e-5


def test_weighted_kl_equals_unweighted_when_weights_are_ones():
    """Critical: weights=ones must reproduce unweighted KL exactly."""
    torch.manual_seed(0)
    s = torch.randn(2, 10, 50)
    t = torch.randn(2, 10, 50)
    unweighted = sequence_kl_loss(s, t, weights=None, temperature=2.0)
    ones = torch.ones_like(s)
    weighted = sequence_kl_loss(s, t, weights=ones, temperature=2.0)
    assert torch.allclose(unweighted, weighted, atol=1e-5), (
        f"weights=ones should equal unweighted: {unweighted.item()} vs {weighted.item()}")


def test_kl_matches_manual_3d_formula():
    """Compare against the token-mean formula used in the old C variant."""
    torch.manual_seed(1)
    s = torch.randn(3, 8, 40)
    t = torch.randn(3, 8, 40)
    T = 2.0
    ours = sequence_kl_loss(s, t, weights=None, temperature=T)

    t_soft = F.softmax(t / T, dim=-1)
    s_logsm = F.log_softmax(s / T, dim=-1)
    t_logsm = F.log_softmax(t / T, dim=-1)
    manual = (t_soft * (t_logsm - s_logsm)).sum(-1).mean() * (T * T)
    assert torch.allclose(ours, manual, atol=1e-5)


def test_ce_loss_shape_and_direction():
    s = torch.randn(2, 10, 50)
    labels = torch.randint(0, 50, (2, 10))
    loss = sequence_ce_loss(s, labels)
    assert loss.dim() == 0
    assert loss.item() > 0

    perfect = torch.full_like(s, -1e9)
    for b in range(2):
        for i in range(9):
            perfect[b, i, labels[b, i + 1]] = 1e9
    loss_perfect = sequence_ce_loss(perfect, labels)
    assert loss_perfect.item() < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
