"""Basis Provider: strict access control for completion basis W_hat.

Per GPT-5.5 Pro R2: variants B and C must NOT use teacher.lm_head.weight
as completion basis. This module enforces that access policy.

Sources:
  - teacher_oracle: full teacher W_lm — ONLY for upper-bound variants
  - carlini_recovered: W_eff reconstructed from API top-K logit queries via SVD
  - public_pretrained: public base-model lm_head (if attacker assumes known
    public checkpoint). Must be declared in threat model.
  - null: no basis available; completion disabled

The access_level is tagged on every basis object so manifest can record it.
"""
from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Basis:
    W: torch.Tensor
    source: str
    access_level: str
    n_queries_used: int = 0
    notes: str = ""


class BasisProvider:
    ALLOWED_SOURCES = {
        "teacher_oracle": "oracle_upper_bound_only",
        "carlini_recovered": "strict_black_box",
        "public_pretrained": "strict_black_box_with_public_weights",
        "null": "none",
    }

    STRICT_SOURCES = {"carlini_recovered", "public_pretrained"}

    def __init__(self, source: str, variant_name: str = ""):
        if source not in self.ALLOWED_SOURCES:
            raise ValueError(
                f"Unknown basis source {source!r}. "
                f"Allowed: {list(self.ALLOWED_SOURCES)}")
        self.source = source
        self.access_level = self.ALLOWED_SOURCES[source]
        self.variant_name = variant_name

    def assert_allowed_for_strict(self):
        """Raises if the source is not allowed in strict black-box variants."""
        if self.source not in self.STRICT_SOURCES:
            raise RuntimeError(
                f"Variant {self.variant_name!r} is strict black-box, "
                f"but basis source is {self.source!r} "
                f"(access_level={self.access_level!r}). "
                f"Strict variants must use one of {self.STRICT_SOURCES}.")

    @staticmethod
    def carlini_recover(api, input_ids_batches, d: int, V: int,
                        device: str = "cuda:0") -> Basis:
        """Recover W_eff from top-K logit observations via SVD (Gram trick).

        Carlini et al. 2024: stack observed logit vectors Z ∈ R^{N×V}, then
        take the top-d right singular vectors as W_eff^T. For large V we
        cannot form V×V. Instead, if N ≤ V, use the Gram trick:
          Z Z^T = U Σ² U^T  (N×N eigendecomposition)
          V_col_i = Z^T U_i / σ_i
        This gives the same top-d right singular vectors without materializing V×V.

        Memory: Z is (N, V) fp32 on CPU; Z Z^T is (N, N) fp32.
        """
        n_queries = 0
        Z_rows_cpu = []
        for batch in input_ids_batches:
            topk_vals, topk_idx, logit_shape = api.get_topk(batch)
            B, T, K = topk_vals.shape
            V_ = logit_shape[-1]
            vals_flat = topk_vals.reshape(B * T, K).float()
            idx_flat = topk_idx.reshape(B * T, K).long()
            mean_val = vals_flat.mean().item()
            z_sparse = torch.full((B * T, V_), mean_val,
                                   device=device, dtype=torch.float32)
            z_sparse.scatter_(-1, idx_flat, vals_flat)
            Z_rows_cpu.append(z_sparse.cpu())
            del z_sparse
            torch.cuda.empty_cache()
            n_queries += B

        Z = torch.cat(Z_rows_cpu, dim=0)
        N = Z.shape[0]
        col_mean = Z.mean(dim=0, keepdim=True)
        Z.sub_(col_mean)

        if N < V:
            gram = Z @ Z.T
            eigvals, U = torch.linalg.eigh(gram.double())
            idx_sort = torch.argsort(eigvals, descending=True)
            d_eff = min(d, N)
            top_idx = idx_sort[:d_eff]
            sigmas = torch.sqrt(eigvals[top_idx].clamp(min=1e-12))
            U_top = U[:, top_idx].float()
            W_eff = (Z.T @ U_top) / sigmas.float().unsqueeze(0)
            if d_eff < d:
                pad = torch.zeros(V, d - d_eff, dtype=W_eff.dtype)
                W_eff = torch.cat([W_eff, pad], dim=1)
        else:
            _, _, Vh = torch.linalg.svd(Z, full_matrices=False)
            W_eff = Vh[:d, :].T

        del Z
        return Basis(
            W=W_eff,
            source="carlini_recovered",
            access_level="strict_black_box",
            n_queries_used=n_queries,
            notes=f"Recovered from {n_queries} top-K queries via Gram-trick SVD "
                  f"(N={N}, V={V}); W_eff ≈ W_lm @ A (gauge ambiguity)."
        )

    @staticmethod
    def teacher_oracle_unsafe(teacher_model) -> Basis:
        """Teacher W_lm — ONLY for oracle upper-bound variants."""
        W = teacher_model.lm_head.weight.data.float()
        return Basis(
            W=W.clone(),
            source="teacher_oracle",
            access_level="oracle_upper_bound_only",
            n_queries_used=0,
            notes="TEACHER WEIGHTS — NOT BLACK-BOX. Use only in oracle upper-bound variant."
        )

    @staticmethod
    def null() -> Basis:
        return Basis(
            W=torch.zeros(0),
            source="null",
            access_level="none",
            n_queries_used=0,
            notes="No basis. Completion disabled."
        )
