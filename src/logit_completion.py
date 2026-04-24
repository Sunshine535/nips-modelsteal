"""Calibrated Logit Completion (R2-compliant).

Per GPT-5.5 Pro R2:
  - Basis W_full must come from BasisProvider (source != teacher_oracle for strict).
  - Calibration uses ONLY probe-oracle queries on a disjoint calibration split.
  - Uncertainty weights estimated from held-out PROBE targets only.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


class CalibratedLogitCompleter:
    """Completes partial top-K logits into dense logit vectors using probe recovery."""

    def __init__(self, W_probe: torch.Tensor, probe_ids: torch.Tensor,
                 basis,
                 device: str = "cuda:0",
                 regularization: float = 1e-6):
        """
        Args:
            W_probe: (K_probe, d) — rows of basis at probe token indices
            probe_ids: (K_probe,) — vocabulary indices used as probes
            basis: Basis object from BasisProvider (source != teacher_oracle for strict)
            regularization: Tikhonov regularization for lstsq
        """
        from src.basis_provider import Basis
        if not isinstance(basis, Basis):
            raise TypeError(
                "CalibratedLogitCompleter requires a Basis object from BasisProvider, "
                "not a raw tensor. This enforces source provenance.")

        self.basis = basis
        self.basis_source = basis.source
        self.basis_access_level = basis.access_level
        self.probe_ids = probe_ids.long()
        self.K_probe = len(probe_ids)
        self.d = W_probe.shape[1]
        self.device = device
        self.reg = regularization

        self.W_probe = W_probe.double().to(device)
        self.W_full = basis.W.float().to(device)

        WtW = self.W_probe.T @ self.W_probe + regularization * torch.eye(
            self.d, dtype=torch.float64, device=device)
        self.WtW_inv = torch.linalg.inv(WtW)
        self.solve_matrix = self.WtW_inv @ self.W_probe.T

        self.uncertainty_weights = None
        self.calibration_diagnostics = {}

    def recover_h(self, probe_logits: torch.Tensor) -> torch.Tensor:
        """Recover h_hat via regularized lstsq."""
        orig_shape = probe_logits.shape[:-1]
        flat = probe_logits.reshape(-1, self.K_probe).double().to(self.device)
        h_hat = (self.solve_matrix @ flat.T).T.float()
        return h_hat.reshape(*orig_shape, self.d)

    def reconstruct_logits(self, h_hat: torch.Tensor) -> torch.Tensor:
        return h_hat @ self.W_full.T

    def complete(self, topk_vals: torch.Tensor, topk_idx: torch.Tensor,
                 probe_logits: torch.Tensor, logit_shape: tuple) -> torch.Tensor:
        h_hat = self.recover_h(probe_logits)
        z_hat = self.reconstruct_logits(h_hat)
        B, T, V = logit_shape
        z_hat = z_hat.reshape(B, T, -1)
        if z_hat.shape[-1] < V:
            z_hat = F.pad(z_hat, (0, V - z_hat.shape[-1]))
        z_complete = z_hat.clone()
        z_complete.scatter_(-1, topk_idx, topk_vals)
        return z_complete

    def fit_calibration_from_probes(self,
                                     cal_input_probe_logits: torch.Tensor,
                                     heldout_probe_ids: torch.Tensor,
                                     heldout_probe_targets: torch.Tensor,
                                     heldout_W_rows: torch.Tensor) -> dict:
        """R2-compliant calibration using ONLY probe queries.

        Uses two disjoint probe sets:
          - `cal_input_probe_logits`: probe logits used as INPUT to recover h_hat
            (same probe_ids as training)
          - `heldout_probe_ids` + `heldout_probe_targets`: a different set of probe
            tokens whose TRUE logits were queried (counted against budget).
            These serve as ground-truth to measure reconstruction error on the tail.

        Args:
            cal_input_probe_logits: (N_cal, K_probe) probe logits at training probe_ids
            heldout_probe_ids: (K_held,) vocab indices disjoint from self.probe_ids
            heldout_probe_targets: (N_cal, K_held) true logits at heldout probes
            heldout_W_rows: (K_held, d) rows of basis at heldout_probe_ids

        Returns:
            Diagnostics dict (tail_mse, weight_stats).
        """
        h_hat = self.recover_h(cal_input_probe_logits)
        W_held = heldout_W_rows.float().to(self.device)
        z_hat_held = h_hat @ W_held.T
        targets = heldout_probe_targets.to(self.device).float()

        per_token_sq_err = (z_hat_held - targets).pow(2).mean(dim=0)

        V = int(self.W_full.shape[0])
        per_vocab_var = torch.full((V,), per_token_sq_err.mean().item(),
                                    device=self.device)
        per_vocab_var[heldout_probe_ids.to(self.device).long()] = per_token_sq_err

        self.uncertainty_weights = 1.0 / (per_vocab_var + 1e-8)
        self.uncertainty_weights = self.uncertainty_weights / self.uncertainty_weights.max()

        diag = {
            "heldout_mse_mean": per_token_sq_err.mean().item(),
            "heldout_mse_std": per_token_sq_err.std().item(),
            "weight_mean": self.uncertainty_weights.mean().item(),
            "weight_std": self.uncertainty_weights.std().item(),
            "weight_min": self.uncertainty_weights.min().item(),
            "weight_max": self.uncertainty_weights.max().item(),
            "n_heldout_probes": int(heldout_probe_ids.numel()),
            "n_calibration_examples": int(cal_input_probe_logits.shape[0]),
        }
        self.calibration_diagnostics = diag
        return diag

    def get_weights(self, topk_idx: torch.Tensor, V: int) -> torch.Tensor:
        B, T, K = topk_idx.shape
        if self.uncertainty_weights is not None:
            w = self.uncertainty_weights.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            w = w[:, :, :V].clone()
        else:
            w = torch.ones(B, T, V, device=self.device)
        w.scatter_(-1, topk_idx, 1.0)
        return w
