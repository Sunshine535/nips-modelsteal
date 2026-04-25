"""Ranking losses for T-DART.

These operate on OBSERVED candidate tokens only — no artificial subset-softmax.
Implemented from scratch per PRIOR_WORK_IMPLEMENTATION_BOUNDARY.md.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def pairwise_residual_rank_loss(delta_student: torch.Tensor,
                                 delta_teacher: torch.Tensor,
                                 mask: torch.Tensor = None,
                                 margin_weight: bool = True) -> torch.Tensor:
    """Pairwise ranking loss: student should preserve teacher's residual ordering.

    L = Σ_{(i,j)} w_ij * log(1 + exp(-sign(Δ_T(i)-Δ_T(j)) * (Δ_S(i)-Δ_S(j))))

    Args:
        delta_student: (..., K) student residual deltas at candidates
        delta_teacher: (..., K) teacher residual deltas
        mask: (..., K, K) bool, which pairs to include (None = all)
        margin_weight: if True, weight pairs by teacher margin |Δ_T(i)-Δ_T(j)|

    Returns:
        Scalar loss averaged over valid pairs.
    """
    diff_t = delta_teacher.unsqueeze(-1) - delta_teacher.unsqueeze(-2)
    diff_s = delta_student.unsqueeze(-1) - delta_student.unsqueeze(-2)

    signs = diff_t.sign()
    logits = -signs * diff_s

    if mask is None:
        K = delta_teacher.shape[-1]
        mask = ~torch.eye(K, dtype=torch.bool, device=delta_teacher.device)
        mask = mask.expand_as(diff_t)

    pair_loss = F.softplus(logits)

    if margin_weight:
        weights = diff_t.abs()
        weights = weights / (weights.sum() / mask.float().sum() + 1e-8)
    else:
        weights = torch.ones_like(pair_loss)

    masked_loss = pair_loss * mask.float() * weights
    n_pairs = mask.float().sum().clamp(min=1.0)
    return masked_loss.sum() / n_pairs


def residual_mse_loss(delta_student: torch.Tensor,
                       delta_teacher: torch.Tensor,
                       confidence: torch.Tensor = None) -> torch.Tensor:
    """MSE on residual deltas at observed candidates.

    Args:
        delta_student: (..., K)
        delta_teacher: (..., K)
        confidence: (..., K) optional per-token confidence weights

    Returns:
        Scalar MSE, optionally weighted.
    """
    sq_err = (delta_student - delta_teacher).pow(2)
    if confidence is not None:
        sq_err = sq_err * confidence
    return sq_err.mean()


def listwise_plackett_luce_loss(delta_student: torch.Tensor,
                                 delta_teacher: torch.Tensor) -> torch.Tensor:
    """Plackett-Luce listwise ranking loss over observed candidates.

    The teacher's ordering defines the "ground truth" permutation.
    Student should assign higher logits to teacher-preferred tokens.

    Args:
        delta_student: (..., K)
        delta_teacher: (..., K)

    Returns:
        Scalar PL loss.
    """
    teacher_order = delta_teacher.argsort(dim=-1, descending=True)
    K = delta_student.shape[-1]
    loss = torch.tensor(0.0, device=delta_student.device)
    for k in range(K - 1):
        remaining = teacher_order[..., k:]
        s_remaining = delta_student.gather(-1, remaining)
        log_softmax = F.log_softmax(s_remaining, dim=-1)
        loss = loss - log_softmax[..., 0].mean()
    return loss / max(K - 1, 1)
