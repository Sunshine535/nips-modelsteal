"""Tests for Moment Confidence Gate (optional module)."""
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.moment_gate import MomentConfidenceGate


def test_gate_defaults_to_one_when_no_artifact():
    """Missing artifact must produce gate=1, never silently affect results."""
    gate = MomentConfidenceGate(cp_factor_path=None)
    g = gate.compute_gate(V=100)
    assert g.shape == (100,)
    assert (g == 1.0).all()
    assert not gate.available


def test_gate_defaults_to_one_for_missing_path():
    """Nonexistent path also defaults to gate=1."""
    gate = MomentConfidenceGate(cp_factor_path="/tmp/nonexistent_cp_factors.pt")
    g = gate.compute_gate(V=50)
    assert (g == 1.0).all()
    assert not gate.available


def test_gate_summary_reports_availability():
    gate = MomentConfidenceGate(cp_factor_path=None)
    s = gate.summary()
    assert s["available"] is False
    assert s["n_factors"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
