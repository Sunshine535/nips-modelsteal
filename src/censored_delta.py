"""Censored residual constraints for C-DART.

Core novelty: if token u is in student/reference top-K but NOT in teacher top-K,
then z_T(u) <= tau_T where tau_T = min(teacher top-K logits). This gives:
    Δ_T(u) <= tau_T - z_R(u)

This censored upper bound is a training signal from ABSENCE — the API tells
us what the teacher DIDN'T prefer, without needing to probe those tokens.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def build_censored_candidates(teacher_topk_ids: torch.Tensor,
                               teacher_topk_vals: torch.Tensor,
                               student_topk_ids: torch.Tensor,
                               reference_topk_ids: torch.Tensor,
                               reference_logits: torch.Tensor):
    """Identify tokens that are high for student/reference but absent from teacher top-K.

    Args:
        teacher_topk_ids: (K_t,) teacher's returned top-K token IDs
        teacher_topk_vals: (K_t,) teacher's top-K logit values
        student_topk_ids: (K_s,) student's local top-K
        reference_topk_ids: (K_r,) reference's local top-K
        reference_logits: (V,) reference full logits

    Returns:
        observed_ids: teacher top-K token IDs (observed with exact values)
        censored_ids: student/ref candidates NOT in teacher top-K (upper-bounded)
        tau_T: scalar, min teacher top-K logit (censoring threshold)
        delta_T_upper_bound: (len(censored_ids),) upper bound for teacher residual
    """
    teacher_set = set(teacher_topk_ids.tolist())
    sr_union = torch.cat([student_topk_ids, reference_topk_ids]).unique()

    censored_mask = torch.tensor([v.item() not in teacher_set for v in sr_union],
                                  dtype=torch.bool)
    censored_ids = sr_union[censored_mask]

    tau_T = teacher_topk_vals.min()

    ref_at_censored = reference_logits[censored_ids]
    delta_T_upper = tau_T - ref_at_censored

    return teacher_topk_ids, censored_ids, tau_T, delta_T_upper


def censored_residual_rank_loss(delta_student_observed: torch.Tensor,
                                 delta_teacher_observed: torch.Tensor,
                                 delta_student_censored: torch.Tensor,
                                 delta_teacher_upper_bound: torch.Tensor,
                                 margin: float = 0.1) -> torch.Tensor:
    """Hinge loss: observed teacher-preferred tokens should rank above censored tokens.

    For each pair (i ∈ observed, u ∈ censored):
        If Δ_T(i) > upper_bound(u) + margin:
            push Δ_S(i) > Δ_S(u) + margin

    Args:
        delta_student_observed: (N_obs,) student residuals at observed (teacher top-K) tokens
        delta_teacher_observed: (N_obs,) teacher residuals at observed tokens
        delta_student_censored: (N_cen,) student residuals at censored tokens
        delta_teacher_upper_bound: (N_cen,) upper bound on teacher residuals at censored tokens
        margin: minimum required margin for hinge activation

    Returns:
        Scalar loss.
    """
    if len(delta_student_censored) == 0 or len(delta_student_observed) == 0:
        return torch.tensor(0.0, device=delta_student_observed.device)

    diff_obs = delta_student_observed.unsqueeze(-1)
    diff_cen = delta_student_censored.unsqueeze(-2)

    dt_obs = delta_teacher_observed.unsqueeze(-1)
    dt_ub = delta_teacher_upper_bound.unsqueeze(-2)
    safe_mask = (dt_obs > dt_ub + margin)

    hinge = F.relu(margin - (diff_obs - diff_cen))

    masked_hinge = hinge * safe_mask.float()
    n_pairs = safe_mask.float().sum().clamp(min=1.0)
    return masked_hinge.sum() / n_pairs
