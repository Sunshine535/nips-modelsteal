"""Tests for ranking losses."""
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ranking_losses import pairwise_residual_rank_loss, residual_mse_loss, listwise_plackett_luce_loss


def test_pairwise_zero_when_orders_match():
    """Perfect ordering should have near-zero loss."""
    delta_t = torch.tensor([5.0, 3.0, 1.0])
    delta_s = torch.tensor([50.0, 30.0, 10.0])
    loss = pairwise_residual_rank_loss(delta_s, delta_t, margin_weight=False)
    assert loss.item() < 0.1, f"Matched ordering loss should be small, got {loss.item()}"


def test_pairwise_high_when_orders_reversed():
    delta_t = torch.tensor([5.0, 3.0, 1.0])
    delta_s = torch.tensor([1.0, 3.0, 5.0])
    loss = pairwise_residual_rank_loss(delta_s, delta_t, margin_weight=False)
    assert loss.item() > 1.0, f"Reversed ordering should have high loss, got {loss.item()}"


def test_pairwise_respects_mask():
    delta_t = torch.tensor([5.0, 3.0, 1.0])
    delta_s = torch.tensor([1.0, 3.0, 5.0])
    mask = torch.zeros(3, 3, dtype=torch.bool)
    loss = pairwise_residual_rank_loss(delta_s, delta_t, mask=mask)
    assert loss.item() == 0.0, "Empty mask should give zero loss"


def test_residual_mse_zero_when_equal():
    delta = torch.tensor([1.0, 2.0, 3.0])
    loss = residual_mse_loss(delta, delta)
    assert loss.item() < 1e-6


def test_residual_mse_positive_when_different():
    delta_s = torch.tensor([1.0, 2.0, 3.0])
    delta_t = torch.tensor([4.0, 5.0, 6.0])
    loss = residual_mse_loss(delta_s, delta_t)
    assert loss.item() > 0


def test_pl_loss_decreases_with_better_order():
    delta_t = torch.tensor([5.0, 3.0, 1.0])
    good_s = torch.tensor([5.0, 3.0, 1.0])
    bad_s = torch.tensor([1.0, 3.0, 5.0])
    loss_good = listwise_plackett_luce_loss(good_s, delta_t)
    loss_bad = listwise_plackett_luce_loss(bad_s, delta_t)
    assert loss_good < loss_bad, f"Good order ({loss_good}) should beat bad ({loss_bad})"


def test_pairwise_batch_shape():
    B, T, K = 2, 4, 10
    delta_s = torch.randn(B, T, K)
    delta_t = torch.randn(B, T, K)
    loss = pairwise_residual_rank_loss(delta_s, delta_t)
    assert loss.dim() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
