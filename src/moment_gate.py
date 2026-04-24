"""Optional Moment Confidence Gate (Task 8, P2).

Loads CP factors only if artifact exists; computes null margin per factor;
defaults to gate=1 if no verified moment artifacts are available.

This is deliberately conservative: un-verified CP factors MUST NOT silently
affect results. The gate is available as an ablation but must never be
required for Q-UMC to work.
"""
from __future__ import annotations
import os
import torch


class MomentConfidenceGate:
    def __init__(self, cp_factor_path: str = None, null_margin_threshold: float = 0.02):
        self.cp_factor_path = cp_factor_path
        self.null_margin_threshold = null_margin_threshold
        self.factors = None
        self.null_margins = None
        self.available = False
        if cp_factor_path and os.path.exists(cp_factor_path):
            try:
                data = torch.load(cp_factor_path, map_location="cpu")
                self.factors = data.get("factors", None)
                self.null_margins = data.get("null_margins", None)
                self.available = (self.factors is not None and
                                  self.null_margins is not None)
            except Exception:
                self.available = False

    def compute_gate(self, V: int, device: str = "cpu") -> torch.Tensor:
        """Returns per-token gate values. Default g_v=1 if unavailable."""
        if not self.available:
            return torch.ones(V, device=device)
        gate = torch.zeros(V, device=device)
        mask = self.null_margins > self.null_margin_threshold
        confident_factors = self.factors[mask]
        if confident_factors.numel() == 0:
            return torch.ones(V, device=device)
        scores = confident_factors.norm(dim=-1)
        scores = scores / (scores.max() + 1e-8)
        gate[:len(scores)] = scores
        return gate

    def summary(self) -> dict:
        return {
            "cp_factor_path": self.cp_factor_path,
            "available": self.available,
            "n_factors": 0 if self.factors is None else len(self.factors),
            "null_margin_threshold": self.null_margin_threshold,
        }
