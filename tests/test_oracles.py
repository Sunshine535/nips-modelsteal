"""Tests for strict black-box oracle abstraction."""
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.oracles import QueryBudget, StrictTopKOracle, ProbeLogitOracle, StrictBlackBoxAPI


class FakeTeacher(torch.nn.Module):
    def __init__(self, V=100, d=16):
        super().__init__()
        self.lm_head = torch.nn.Linear(d, V, bias=False)
        self.d = d
        self.V = V

    def forward(self, input_ids, **kwargs):
        B, T = input_ids.shape
        h = torch.randn(B, T, self.d)
        logits = self.lm_head(h)

        class Out:
            pass

        out = Out()
        out.logits = logits
        return out


def test_query_budget_tracks_counts():
    b = QueryBudget()
    b.charge_topk(5)
    b.charge_probe(100)
    s = b.summary()
    assert s["topk_queries"] == 5
    assert s["probe_queries"] == 100
    assert s["total_queries"] == 105


def test_query_budget_enforces_limit():
    b = QueryBudget(max_probe_budget=50)
    b.charge_probe(30)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        b.charge_probe(30)


def test_topk_oracle_returns_only_K():
    teacher = FakeTeacher(V=100, d=16)
    budget = QueryBudget()
    oracle = StrictTopKOracle(teacher, K=5, budget=budget, device="cpu")
    ids = torch.randint(0, 100, (2, 10))
    vals, idx, shape = oracle.query_topk(ids)
    assert vals.shape == (2, 10, 5)
    assert idx.shape == (2, 10, 5)
    assert shape == (2, 10, 100)
    assert budget.topk_queries == 2


def test_probe_oracle_returns_only_probe_ids():
    teacher = FakeTeacher(V=100, d=16)
    budget = QueryBudget()
    oracle = ProbeLogitOracle(teacher, budget=budget, device="cpu")
    ids = torch.randint(0, 100, (2, 10))
    probe_ids = torch.tensor([3, 7, 15, 42])
    result = oracle.query_probe(ids, probe_ids)
    assert result.shape == (2, 10, 4)
    assert budget.probe_queries == 2 * 4


def test_strict_api_combined():
    teacher = FakeTeacher(V=100, d=16)
    api = StrictBlackBoxAPI(teacher, K=5, device="cpu")
    ids = torch.randint(0, 100, (1, 8))
    topk_vals, topk_idx, shape = api.get_topk(ids)
    assert topk_vals.shape[-1] == 5

    probe_ids = torch.arange(20)
    probe_logits = api.get_probe_logits(ids, probe_ids)
    assert probe_logits.shape[-1] == 20

    s = api.budget.summary()
    assert s["topk_queries"] == 1
    assert s["probe_queries"] == 1 * 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
