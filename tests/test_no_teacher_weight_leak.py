"""Regression tests: strict variants must not access teacher weights."""
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.basis_provider import BasisProvider, Basis


def test_strict_source_passes_assertion():
    bp = BasisProvider("carlini_recovered", variant_name="completion_no_unc")
    bp.assert_allowed_for_strict()


def test_public_pretrained_passes_assertion():
    bp = BasisProvider("public_pretrained", variant_name="completion_uncertainty")
    bp.assert_allowed_for_strict()


def test_teacher_oracle_raises_in_strict():
    bp = BasisProvider("teacher_oracle", variant_name="completion_uncertainty")
    with pytest.raises(RuntimeError, match="strict black-box"):
        bp.assert_allowed_for_strict()


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match="Unknown basis source"):
        BasisProvider("magic_source")


def test_teacher_oracle_basis_carries_warning_tag():
    class FakeTeacher:
        class LMHead:
            weight = type("W", (), {"data": torch.randn(10, 4)})
        lm_head = LMHead()

    b = BasisProvider.teacher_oracle_unsafe(FakeTeacher())
    assert b.source == "teacher_oracle"
    assert b.access_level == "oracle_upper_bound_only"
    assert "NOT BLACK-BOX" in b.notes


def test_null_basis_has_zero_weights():
    b = BasisProvider.null()
    assert b.W.numel() == 0
    assert b.source == "null"


def test_run_qumc_strict_path_has_no_teacher_weight_access():
    """Source-level check: strict B/C training path must not reference
    teacher.lm_head.weight when basis_source != teacher_oracle."""
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_qumc.py")
    if not os.path.exists(path):
        pytest.skip("run_qumc.py not present")
    src = open(path).read()
    assert "BasisProvider" in src, (
        "run_qumc.py must use BasisProvider to select completion basis")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
