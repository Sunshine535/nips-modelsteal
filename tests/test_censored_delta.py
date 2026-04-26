"""Tests for censored residual constraints."""
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.censored_delta import build_censored_candidates, censored_residual_rank_loss


def test_censored_ids_exclude_teacher_topk():
    t_ids = torch.tensor([3, 7, 15])
    t_vals = torch.tensor([5.0, 4.0, 3.0])
    s_ids = torch.tensor([3, 7, 20, 42])
    r_ids = torch.tensor([3, 15, 42, 99])
    r_logits = torch.randn(100)
    obs, cen, tau, ub = build_censored_candidates(t_ids, t_vals, s_ids, r_ids, r_logits)
    assert 3 not in cen.tolist()
    assert 7 not in cen.tolist()
    assert 15 not in cen.tolist()
    assert 20 in cen.tolist()
    assert 42 in cen.tolist()
    assert 99 in cen.tolist()


def test_tau_is_min_teacher_logit():
    t_ids = torch.tensor([0, 1, 2])
    t_vals = torch.tensor([10.0, 5.0, 3.0])
    s_ids = torch.tensor([5])
    r_ids = torch.tensor([6])
    r_logits = torch.zeros(10)
    _, _, tau, _ = build_censored_candidates(t_ids, t_vals, s_ids, r_ids, r_logits)
    assert tau.item() == 3.0


def test_upper_bound_is_tau_minus_reference():
    t_ids = torch.tensor([0, 1])
    t_vals = torch.tensor([10.0, 5.0])
    s_ids = torch.tensor([3])
    r_ids = torch.tensor([3])
    r_logits = torch.zeros(10)
    r_logits[3] = 2.0
    _, cen, tau, ub = build_censored_candidates(t_ids, t_vals, s_ids, r_ids, r_logits)
    assert 3 in cen.tolist()
    idx = (cen == 3).nonzero(as_tuple=True)[0][0]
    assert abs(ub[idx].item() - (5.0 - 2.0)) < 1e-5


def test_hinge_zero_when_student_correctly_ranks():
    delta_s_obs = torch.tensor([5.0, 4.0])
    delta_t_obs = torch.tensor([5.0, 4.0])
    delta_s_cen = torch.tensor([0.0])
    delta_t_ub = torch.tensor([1.0])
    loss = censored_residual_rank_loss(delta_s_obs, delta_t_obs, delta_s_cen, delta_t_ub, margin=0.1)
    assert loss.item() < 0.1


def test_hinge_nonzero_when_student_wrongly_ranks():
    delta_s_obs = torch.tensor([1.0])
    delta_t_obs = torch.tensor([5.0])
    delta_s_cen = torch.tensor([3.0])
    delta_t_ub = torch.tensor([0.5])
    loss = censored_residual_rank_loss(delta_s_obs, delta_t_obs, delta_s_cen, delta_t_ub, margin=0.1)
    assert loss.item() > 0.5


def test_empty_censored_returns_zero():
    delta_s_obs = torch.tensor([5.0])
    delta_t_obs = torch.tensor([5.0])
    delta_s_cen = torch.tensor([])
    delta_t_ub = torch.tensor([])
    loss = censored_residual_rank_loss(delta_s_obs, delta_t_obs, delta_s_cen, delta_t_ub)
    assert loss.item() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
