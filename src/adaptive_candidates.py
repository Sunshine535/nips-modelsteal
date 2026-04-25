"""Adaptive candidate selection for T-DART.

Instead of fixed probe tokens (which failed in v4), select candidates
per-prompt based on teacher top-K, student top-K, reference top-K,
and optional high-disagreement probes.

Candidate set: C_x = I_T^K ∪ I_S^K ∪ I_R^K ∪ P_adapt
"""
from __future__ import annotations
import torch


def select_candidates(teacher_topk_ids: torch.Tensor,
                      student_topk_ids: torch.Tensor,
                      reference_topk_ids: torch.Tensor,
                      max_probe_tokens: int = 64,
                      student_logits: torch.Tensor = None,
                      reference_logits: torch.Tensor = None,
                      strategy: str = "disagreement") -> torch.Tensor:
    """Select candidate token IDs for residual comparison.

    Args:
        teacher_topk_ids: (K_t,) teacher's top-K token indices (from API)
        student_topk_ids: (K_s,) student's top-K token indices (local)
        reference_topk_ids: (K_r,) reference model's top-K indices (local)
        max_probe_tokens: max additional tokens to probe beyond union
        student_logits: (V,) full student logits for disagreement selection
        reference_logits: (V,) full reference logits
        strategy: "disagreement" or "random"

    Returns:
        candidate_ids: (C,) unique token IDs, budget-limited
    """
    base = torch.cat([teacher_topk_ids.flatten(),
                       student_topk_ids.flatten(),
                       reference_topk_ids.flatten()])
    base_unique = base.unique()

    remaining_budget = max_probe_tokens
    if remaining_budget > 0 and student_logits is not None and reference_logits is not None:
        if strategy == "disagreement":
            diff = (student_logits - reference_logits).abs()
            diff[base_unique] = -float("inf")
            _, probe_ids = diff.topk(min(remaining_budget, len(diff)))
            all_ids = torch.cat([base_unique, probe_ids])
        elif strategy == "random":
            V = student_logits.shape[-1]
            mask = torch.ones(V, dtype=torch.bool)
            mask[base_unique] = False
            pool = mask.nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(pool))[:remaining_budget]
            probe_ids = pool[perm]
            all_ids = torch.cat([base_unique, probe_ids])
        else:
            all_ids = base_unique
    else:
        all_ids = base_unique

    return all_ids.unique()


def select_candidates_batch(teacher_topk_ids: torch.Tensor,
                             student_logits: torch.Tensor,
                             reference_logits: torch.Tensor,
                             K_student: int = 20,
                             K_reference: int = 20,
                             max_probe_tokens: int = 64,
                             strategy: str = "disagreement") -> torch.Tensor:
    """Batch version: select candidates for a single position.

    Args:
        teacher_topk_ids: (K,) from API
        student_logits: (V,) student logits at this position
        reference_logits: (V,) reference logits
        K_student, K_reference: how many student/ref top tokens to include
        max_probe_tokens: additional probes

    Returns:
        candidate_ids: (C,) unique IDs
    """
    s_topk = student_logits.topk(K_student).indices
    r_topk = reference_logits.topk(K_reference).indices
    return select_candidates(
        teacher_topk_ids, s_topk, r_topk,
        max_probe_tokens=max_probe_tokens,
        student_logits=student_logits,
        reference_logits=reference_logits,
        strategy=strategy,
    )
