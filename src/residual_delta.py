"""Residual delta utilities for T-DART.

Core idea: instead of matching absolute teacher logits (which fail under
partial observation), match teacher-specific RESIDUAL over a public reference:
  Δ_T(v|x) = z_T(v|x) - z_R(v|x)

This isolates what the teacher learned beyond the public base model.
Only computed over observed candidate tokens (no artificial subset-softmax).
"""
from __future__ import annotations
import torch


def compute_residual(observed_logits: torch.Tensor,
                     reference_logits: torch.Tensor,
                     candidate_ids: torch.Tensor) -> torch.Tensor:
    """Compute residual logits: Δ = observed - reference at candidate positions.

    Args:
        observed_logits: (..., K) logit values at candidate_ids
        reference_logits: (..., V) full reference model logits
        candidate_ids: (K,) vocabulary indices

    Returns:
        deltas: (..., K) residual logits
    """
    ref_at_candidates = reference_logits[..., candidate_ids]
    return observed_logits - ref_at_candidates


def build_pairwise_preferences(delta_teacher: torch.Tensor,
                                min_margin: float = 0.0):
    """Build pairwise preference matrix from teacher residual deltas.

    For candidate pair (i, j): preference = sign(Δ_T(i) - Δ_T(j))
    Only pairs with |Δ_T(i) - Δ_T(j)| > min_margin are included.

    Args:
        delta_teacher: (..., K) teacher residual deltas
        min_margin: minimum absolute margin to include pair

    Returns:
        signs: (..., K, K) preference signs (+1, -1, or 0 if below margin)
        margins: (..., K, K) absolute margins
        mask: (..., K, K) bool, True where |margin| > min_margin
    """
    diff = delta_teacher.unsqueeze(-1) - delta_teacher.unsqueeze(-2)
    margins = diff.abs()
    signs = diff.sign()
    mask = margins > min_margin
    signs = signs * mask.float()
    return signs, margins, mask


def compute_student_residual(student_logits: torch.Tensor,
                              reference_logits: torch.Tensor,
                              candidate_ids: torch.Tensor) -> torch.Tensor:
    """Compute student residual at candidate positions.

    Args:
        student_logits: (..., V) full student logits
        reference_logits: (..., V) full reference logits
        candidate_ids: (K,) candidate vocabulary indices

    Returns:
        delta_student: (..., K) student residual at candidates
    """
    s_at_cand = student_logits[..., candidate_ids]
    r_at_cand = reference_logits[..., candidate_ids]
    return s_at_cand - r_at_cand
