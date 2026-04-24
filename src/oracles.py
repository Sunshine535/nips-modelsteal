"""Strict black-box oracle abstraction for top-K + logit-bias API simulation.

All teacher access in Q-UMC training MUST go through these oracles.
Accessing unqueried logits raises RuntimeError.
Every probe query increments the budget counter.
"""
from __future__ import annotations
import torch
from dataclasses import dataclass, field


@dataclass
class QueryBudget:
    """Tracks API query costs."""
    topk_queries: int = 0
    probe_queries: int = 0
    max_probe_budget: int = -1

    def charge_topk(self, n: int = 1):
        self.topk_queries += n

    def charge_probe(self, n: int = 1):
        if self.max_probe_budget > 0 and self.probe_queries + n > self.max_probe_budget:
            raise RuntimeError(
                f"Probe budget exceeded: {self.probe_queries}+{n} > {self.max_probe_budget}")
        self.probe_queries += n

    def summary(self) -> dict:
        return {
            "topk_queries": self.topk_queries,
            "probe_queries": self.probe_queries,
            "total_queries": self.topk_queries + self.probe_queries,
            "max_probe_budget": self.max_probe_budget,
        }


class StrictTopKOracle:
    """Returns only top-K logits per position. No other logit values are accessible."""

    def __init__(self, teacher_model, K: int, budget: QueryBudget, device: str = "cuda:0"):
        self.teacher = teacher_model
        self.K = K
        self.budget = budget
        self.device = device
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def query_topk(self, input_ids: torch.Tensor):
        """Returns (topk_values, topk_indices, seq_len) — NO full logit tensor."""
        out = self.teacher(input_ids=input_ids.to(self.device))
        logits = out.logits.float()
        topk_vals, topk_idx = logits.topk(self.K, dim=-1)
        self.budget.charge_topk(input_ids.shape[0])
        return topk_vals, topk_idx, logits.shape

    @torch.no_grad()
    def query_full_logits(self, input_ids: torch.Tensor):
        """Full logit access — ONLY for oracle upper-bound baseline. Tagged explicitly."""
        out = self.teacher(input_ids=input_ids.to(self.device))
        return out.logits.float()


class ProbeLogitOracle:
    """Simulates logit_bias API: returns logits at specific probe token indices."""

    def __init__(self, teacher_model, budget: QueryBudget, device: str = "cuda:0"):
        self.teacher = teacher_model
        self.budget = budget
        self.device = device
        self.teacher.eval()

    @torch.no_grad()
    def query_probe(self, input_ids: torch.Tensor, probe_ids: torch.Tensor):
        """Returns logits ONLY at probe_ids positions in vocab. Charges budget."""
        out = self.teacher(input_ids=input_ids.to(self.device))
        logits = out.logits.float()
        probe_logits = logits[:, :, probe_ids]
        n_queries = input_ids.shape[0] * len(probe_ids)
        self.budget.charge_probe(n_queries)
        return probe_logits


class StrictBlackBoxAPI:
    """Combined API: top-K oracle + probe oracle with shared budget."""

    def __init__(self, teacher_model, K: int, device: str = "cuda:0",
                 max_probe_budget: int = -1):
        self.budget = QueryBudget(max_probe_budget=max_probe_budget)
        self.topk_oracle = StrictTopKOracle(teacher_model, K, self.budget, device)
        self.probe_oracle = ProbeLogitOracle(teacher_model, self.budget, device)
        self.K = K

    def get_topk(self, input_ids):
        return self.topk_oracle.query_topk(input_ids)

    def get_probe_logits(self, input_ids, probe_ids):
        return self.probe_oracle.query_probe(input_ids, probe_ids)

    def get_full_logits_ORACLE_ONLY(self, input_ids):
        """ONLY for oracle upper-bound baseline. Never use in main method."""
        return self.topk_oracle.query_full_logits(input_ids)
