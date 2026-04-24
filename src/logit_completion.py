"""Calibrated Logit Completion: reconstruct dense logits from partial API observations.

Flow:
  1. Receive top-K logits (exact) + probe logits (from logit_bias oracle)
  2. Recover h_hat via lstsq from probe logits
  3. Reconstruct full logit vector: z_hat = W_probe_full @ h_hat
  4. Merge: exact top-K where available, z_hat elsewhere
  5. Estimate per-token uncertainty from calibration set
  6. Return completed logits + uncertainty weights
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


class CalibratedLogitCompleter:
    """Completes partial top-K logits into dense logit vectors using probe recovery."""

    def __init__(self, W_probe: torch.Tensor, probe_ids: torch.Tensor,
                 W_full: torch.Tensor, device: str = "cuda:0",
                 regularization: float = 1e-6):
        """
        Args:
            W_probe: (K_probe, d) — rows of W_lm at probe token indices
            probe_ids: (K_probe,) — vocabulary indices used as probes
            W_full: (V, d) — full W_lm (or Carlini-recovered W_eff)
            regularization: Tikhonov regularization for lstsq
        """
        self.probe_ids = probe_ids.long()
        self.K_probe = len(probe_ids)
        self.d = W_probe.shape[1]
        self.device = device
        self.reg = regularization

        self.W_probe = W_probe.double().to(device)
        self.W_full = W_full.float().to(device)

        WtW = self.W_probe.T @ self.W_probe + regularization * torch.eye(
            self.d, dtype=torch.float64, device=device)
        self.WtW_inv = torch.linalg.inv(WtW)
        self.solve_matrix = self.WtW_inv @ self.W_probe.T

        self.uncertainty_weights = None

    def recover_h(self, probe_logits: torch.Tensor) -> torch.Tensor:
        """Recover h_hat from probe logits via regularized lstsq.

        Args:
            probe_logits: (..., K_probe) logit values at probe tokens
        Returns:
            h_hat: (..., d)
        """
        orig_shape = probe_logits.shape[:-1]
        flat = probe_logits.reshape(-1, self.K_probe).double().to(self.device)
        h_hat = (self.solve_matrix @ flat.T).T.float()
        return h_hat.reshape(*orig_shape, self.d)

    def reconstruct_logits(self, h_hat: torch.Tensor) -> torch.Tensor:
        """Reconstruct full logit vector from h_hat.

        Args:
            h_hat: (..., d)
        Returns:
            z_hat: (..., V)
        """
        return h_hat @ self.W_full.T

    def complete(self, topk_vals: torch.Tensor, topk_idx: torch.Tensor,
                 probe_logits: torch.Tensor, logit_shape: tuple) -> torch.Tensor:
        """Full completion pipeline: probe → h_hat → z_hat → merge top-K.

        Args:
            topk_vals: (B, T, K) exact top-K logit values
            topk_idx: (B, T, K) top-K token indices
            probe_logits: (B, T, K_probe) probe logit values
            logit_shape: (B, T, V) shape of full logit tensor
        Returns:
            z_complete: (B, T, V) completed logit tensor
        """
        h_hat = self.recover_h(probe_logits)
        z_hat = self.reconstruct_logits(h_hat)

        B, T, V = logit_shape
        z_hat = z_hat.reshape(B, T, -1)
        if z_hat.shape[-1] < V:
            z_hat = F.pad(z_hat, (0, V - z_hat.shape[-1]))

        z_complete = z_hat.clone()
        z_complete.scatter_(-1, topk_idx, topk_vals)
        return z_complete

    def fit_calibration(self, calibration_probe_logits: torch.Tensor,
                        calibration_true_logits: torch.Tensor):
        """Estimate per-vocabulary-token reconstruction uncertainty from calibration set.

        Args:
            calibration_probe_logits: (N_cal, K_probe)
            calibration_true_logits: (N_cal, V)
        """
        h_hat = self.recover_h(calibration_probe_logits)
        z_hat = self.reconstruct_logits(h_hat)
        errors = (z_hat - calibration_true_logits.to(self.device)).pow(2)
        per_token_mse = errors.mean(dim=0)
        self.uncertainty_weights = 1.0 / (per_token_mse + 1e-8)
        self.uncertainty_weights = self.uncertainty_weights / self.uncertainty_weights.max()

    def get_weights(self, topk_idx: torch.Tensor, V: int) -> torch.Tensor:
        """Get per-token weights: 1.0 for exact top-K, calibrated for tail.

        Args:
            topk_idx: (B, T, K)
            V: vocabulary size
        Returns:
            weights: (B, T, V)
        """
        B, T, K = topk_idx.shape
        if self.uncertainty_weights is not None:
            w = self.uncertainty_weights.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            w = w[:, :, :V].clone()
        else:
            w = torch.ones(B, T, V, device=self.device)

        w.scatter_(-1, topk_idx, 1.0)
        return w
