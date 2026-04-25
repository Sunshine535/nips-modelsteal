"""Tests for residual delta utilities."""
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.residual_delta import compute_residual, build_pairwise_preferences, compute_student_residual


def test_residual_is_difference():
    obs = torch.tensor([5.0, 3.0, 1.0])
    ref = torch.tensor([0.0] * 10)
    ref[2] = 2.0
    ref[5] = 1.0
    ref[7] = 0.5
    cand = torch.tensor([2, 5, 7])
    delta = compute_residual(obs, ref, cand)
    assert torch.allclose(delta, torch.tensor([3.0, 2.0, 0.5]))


def test_pairwise_signs_correct():
    delta_t = torch.tensor([3.0, 1.0, 5.0])
    signs, margins, mask = build_pairwise_preferences(delta_t, min_margin=0.0)
    assert signs[0, 1] == 1.0  # 3 > 1
    assert signs[1, 0] == -1.0
    assert signs[2, 0] == 1.0  # 5 > 3
    assert signs[0, 0] == 0.0  # diagonal


def test_pairwise_margin_filter():
    delta_t = torch.tensor([3.0, 3.1, 5.0])
    signs, margins, mask = build_pairwise_preferences(delta_t, min_margin=0.5)
    assert mask[0, 1] == False  # |3.0 - 3.1| = 0.1 < 0.5
    assert mask[0, 2] == True   # |3.0 - 5.0| = 2.0 > 0.5


def test_student_residual_shape():
    B, T, V = 2, 4, 100
    s = torch.randn(B, T, V)
    r = torch.randn(B, T, V)
    cand = torch.tensor([3, 7, 15, 42, 99])
    delta_s = compute_student_residual(s, r, cand)
    assert delta_s.shape == (B, T, 5)


def test_residual_zero_when_models_same():
    V = 50
    logits = torch.randn(3, V)
    cand = torch.arange(10)
    delta = compute_residual(logits[:, cand], logits, cand)
    assert torch.allclose(delta, torch.zeros(3, 10), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
