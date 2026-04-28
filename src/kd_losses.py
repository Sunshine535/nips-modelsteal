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


def dkd_loss(student_logits: torch.Tensor,
             teacher_logits: torch.Tensor,
             labels: torch.Tensor,
             temperature: float = 4.0,
             alpha: float = 1.0,
             beta: float = 8.0) -> torch.Tensor:
    """Decoupled Knowledge Distillation (Zhao et al., CVPR 2022).

    Separates KL into target-class (TCKD) and non-target-class (NCKD) terms,
    weighting NCKD more heavily to transfer dark knowledge.

    Args:
        student_logits: (B, T, V)
        teacher_logits: (B, T, V)
        labels: (B, T) ground-truth token ids
        temperature: softmax temperature
        alpha: weight for TCKD (target class)
        beta: weight for NCKD (non-target classes)
    Returns:
        Scalar loss, averaged over (B, T).
    """
    B, T_seq, V = student_logits.shape
    s_flat = student_logits.reshape(-1, V) / temperature
    t_flat = teacher_logits.reshape(-1, V) / temperature
    lab_flat = labels.reshape(-1)

    s_prob = F.softmax(s_flat, dim=-1)
    t_prob = F.softmax(t_flat, dim=-1)

    gt_mask = torch.zeros_like(s_prob).scatter_(1, lab_flat.unsqueeze(1), 1).bool()

    s_tgt = s_prob[gt_mask].unsqueeze(1)
    t_tgt = t_prob[gt_mask].unsqueeze(1)
    s_nontgt = s_prob[~gt_mask].view(-1, V - 1)
    t_nontgt = t_prob[~gt_mask].view(-1, V - 1)

    s_tgt_pair = torch.cat([s_tgt, 1 - s_tgt], dim=1)
    t_tgt_pair = torch.cat([t_tgt, 1 - t_tgt], dim=1)
    tckd = F.kl_div(s_tgt_pair.log().clamp(min=-100), t_tgt_pair,
                    reduction='batchmean') * (temperature ** 2)

    s_nontgt_norm = s_nontgt / s_nontgt.sum(dim=1, keepdim=True).clamp(min=1e-8)
    t_nontgt_norm = t_nontgt / t_nontgt.sum(dim=1, keepdim=True).clamp(min=1e-8)
    nckd = F.kl_div(s_nontgt_norm.log().clamp(min=-100), t_nontgt_norm,
                    reduction='batchmean') * (temperature ** 2)

    return alpha * tckd + beta * nckd


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
