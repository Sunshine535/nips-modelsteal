"""Unified KD loss with consistent normalization across all variants.

Per GPT-5.5 Pro R2 Task 4: B/D used `F.kl_div(..., reduction='batchmean')`
but C used `(w * t_soft * (t_soft.log() - s_logsm)).sum(-1).mean()`.
For 3D logits (B, T, V), the effective KD weight differed by ~T×.

This module defines ONE loss function used by ALL variants.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def sequence_kl_loss(student_logits: torch.Tensor,
                     teacher_logits: torch.Tensor,
                     weights: torch.Tensor = None,
                     temperature: float = 2.0,
                     normalize_by_weight_mass: bool = False) -> torch.Tensor:
    """KL(teacher || student) summed over vocabulary, averaged over (batch, time).

    Args:
        student_logits: (B, T, V)
        teacher_logits: (B, T, V)
        weights: (B, T, V) or None. Per-token weights. If None, uniform 1.
        temperature: softmax temperature (T^2 scaled per Hinton 2015)
        normalize_by_weight_mass: if True, divide per-token KL by sum(weights)
            so that KD scale is preserved when weights are sparse. Default False
            keeps the original behavior (sparse weights → small KD).

    Returns:
        Scalar loss. Normalization: mean over (B, T).
    """
    T = temperature
    t_soft = F.softmax(teacher_logits / T, dim=-1)
    s_logsm = F.log_softmax(student_logits / T, dim=-1)
    t_logsm = F.log_softmax(teacher_logits / T, dim=-1)

    if weights is None:
        per_token_kl = (t_soft * (t_logsm - s_logsm)).sum(dim=-1)
    else:
        weighted_kl = (weights * t_soft * (t_logsm - s_logsm)).sum(dim=-1)
        if normalize_by_weight_mass:
            mass = (weights * t_soft).sum(dim=-1).clamp(min=1e-8)
            per_token_kl = weighted_kl / mass
        else:
            per_token_kl = weighted_kl

    return per_token_kl.mean() * (T * T)


def sequence_ce_loss(student_logits: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
    """CE on next-token prediction.

    Args:
        student_logits: (B, T, V) — student_logits[:, :-1, :] aligns with labels[:, 1:]
        labels: (B, T) ground-truth token ids
    Returns:
        Scalar CE loss, averaged over (B, T-1).
    """
    s_shift = student_logits[:, :-1, :]
    lab_shift = labels[:, 1:]
    return F.cross_entropy(
        s_shift.reshape(-1, s_shift.size(-1)),
        lab_shift.reshape(-1),
    )
