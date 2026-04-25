"""Tests for adaptive candidate selection."""
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adaptive_candidates import select_candidates, select_candidates_batch


def test_union_includes_all_topk():
    t_ids = torch.tensor([3, 7, 15])
    s_ids = torch.tensor([7, 20, 42])
    r_ids = torch.tensor([3, 42, 99])
    cands = select_candidates(t_ids, s_ids, r_ids, max_probe_tokens=0)
    for v in [3, 7, 15, 20, 42, 99]:
        assert v in cands, f"Expected {v} in candidates"


def test_budget_limits_total():
    t_ids = torch.tensor([3, 7])
    s_ids = torch.tensor([3, 7])
    r_ids = torch.tensor([3, 7])
    s_logits = torch.randn(100)
    r_logits = torch.randn(100)
    cands = select_candidates(t_ids, s_ids, r_ids,
                               max_probe_tokens=5,
                               student_logits=s_logits,
                               reference_logits=r_logits,
                               strategy="disagreement")
    assert len(cands) <= 2 + 5 + 3  # base_unique + probes + margin


def test_disagreement_probes_differ_from_base():
    t_ids = torch.tensor([0, 1])
    s_ids = torch.tensor([0, 1])
    r_ids = torch.tensor([0, 1])
    s_logits = torch.zeros(50)
    r_logits = torch.zeros(50)
    s_logits[40] = 100.0  # big disagreement at 40
    cands = select_candidates(t_ids, s_ids, r_ids,
                               max_probe_tokens=3,
                               student_logits=s_logits,
                               reference_logits=r_logits,
                               strategy="disagreement")
    assert 40 in cands, "High-disagreement token 40 should be probed"


def test_candidates_are_unique():
    t_ids = torch.tensor([3, 3, 7])
    s_ids = torch.tensor([7, 7, 15])
    r_ids = torch.tensor([3, 15])
    cands = select_candidates(t_ids, s_ids, r_ids, max_probe_tokens=0)
    assert len(cands) == len(cands.unique())


def test_batch_version():
    t_ids = torch.tensor([5, 10, 15])
    s_logits = torch.randn(200)
    r_logits = torch.randn(200)
    cands = select_candidates_batch(t_ids, s_logits, r_logits,
                                     K_student=10, K_reference=10,
                                     max_probe_tokens=20)
    assert len(cands) > 3  # at least teacher top-K
    assert len(cands) == len(cands.unique())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
