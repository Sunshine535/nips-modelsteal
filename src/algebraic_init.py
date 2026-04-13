"""
Algebraic Initialization for Transformer Suffix Inversion.

Uses sketched Gauss-Newton (reduced least-squares over JVP probe directions)
to initialize suffix parameters along observable directions, replacing
random initialization with an informed starting point.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from src.gramian import (
    FlatParamSpec,
    make_flat_param_spec,
    build_probe_matrix,
    jvp_logits,
    params_to_flat,
    flat_to_params_,
    GramianConfig,
    GramianResult,
    compute_sketched_gramian,
)
from src.symmetry_gauge import (
    build_suffix_gauge_basis,
    project_probe_matrix,
    GaugeBasis,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AlgebraicInitConfig:
    num_probes: int = 64
    truncation_rank: int = 32  # use top-r singular directions
    ridge: float = 1e-4
    step_scale: float = 1.0  # scale the update: θ_init = θ_0 + scale * V * a*
    query_subsample: int = 256
    include_sensitivity: bool = True
    sensitivity_weight: float = 0.1
    project_gauge: bool = True
    seed: int = 42


@dataclass
class AlgebraicInitResult:
    delta_flat: torch.Tensor  # [p] parameter update in flat space
    coefficients: torch.Tensor  # [k] solution a*
    singular_values: torch.Tensor  # singular values of sketch matrix
    applied_scale: float
    predicted_loss_decrease: float  # predicted quadratic decrease
    gramian_result: object  # GramianResult from gramian.py
    init_loss: float  # loss before init
    post_init_loss: float  # loss after init (if computed)


# ---------------------------------------------------------------------------
# Truncated ridge solve via SVD
# ---------------------------------------------------------------------------


def solve_truncated_ridge(
    sketch_matrix: torch.Tensor,  # [k, k] G̃_aug
    rhs: torch.Tensor,  # [k]
    truncation_rank: int,
    ridge: float,
) -> tuple:
    """Solve (G̃ + λI)a = rhs via truncated SVD.

    Instead of solving the full k×k system, we decompose G̃ = U Σ U^T
    (since G̃ is symmetric PSD from the Gramian), keep only the top-r
    singular components, then solve:

        a* = U_r (Σ_r + λ I_r)^{-1} U_r^T rhs

    This discards unobservable directions (tiny eigenvalues) and
    provides regularisation through both truncation and ridge.

    Returns:
        (a*, singular_values) where a* is [k] and singular_values is [k].
    """
    # G̃ is symmetric PSD → eigendecomposition = SVD
    # eigh returns ascending eigenvalues; we flip to descending
    eigenvalues, eigenvectors = torch.linalg.eigh(sketch_matrix)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    # Clamp rank
    r = min(truncation_rank, len(eigenvalues))

    # Keep only top-r
    sigma_r = eigenvalues[:r]  # [r]
    U_r = eigenvectors[:, :r]  # [k, r]

    # Solve in the truncated subspace: (Σ_r + λI)^{-1} U_r^T rhs
    projected_rhs = U_r.T @ rhs  # [r]
    inv_diag = 1.0 / (sigma_r + ridge)  # [r]
    alpha_r = inv_diag * projected_rhs  # [r]

    # Lift back to full k-space
    a_star = U_r @ alpha_r  # [k]

    return a_star, eigenvalues


# ---------------------------------------------------------------------------
# Loss diagnostic
# ---------------------------------------------------------------------------


def compute_init_loss(
    student: nn.Module,
    cache,  # TeacherCache
    query_indices: torch.Tensor,
    spsi_config,  # SPSIConfig
    boundary_layer_idx: Optional[int] = None,
    use_oracle_boundary: bool = False,
) -> float:
    """Compute current logit + sensitivity loss for diagnostics.

    Evaluates the same loss used in inversion: MSE between teacher and
    student logits (clean term), plus sensitivity matching on perturbed
    inputs. This provides a before/after snapshot for algebraic init.
    """
    device = next(student.parameters()).device
    student.eval()

    N = len(query_indices)
    total_loss = 0.0
    total_samples = 0

    P = spsi_config.num_perturbation_positions
    R = spsi_config.num_replacement_tokens
    pert_per_input = P * R
    beta = spsi_config.beta

    # Whether to use oracle boundary injection
    _use_boundary = (
        use_oracle_boundary
        and boundary_layer_idx is not None
        and hasattr(cache, "boundary_states")
        and boundary_layer_idx in cache.boundary_states
    )

    bs = 8  # small batch for diagnostics
    for start in range(0, N, bs):
        end = min(start + bs, N)
        idx_batch = query_indices[start:end]
        bsz = len(idx_batch)

        input_ids = cache.input_ids[idx_batch].to(device)
        teacher_logits = cache.clean_logits[idx_batch].to(device).float()

        # Boundary hook
        ctx = nullcontext()
        if _use_boundary:
            from src.parameter_inverter import _BoundaryInjectionHook

            ctx = _BoundaryInjectionHook(
                student, cache, idx_batch, boundary_layer_idx
            )

        with ctx, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits = student(input_ids).logits.float()

        # Truncate to match teacher cache suffix positions
        K = getattr(cache, '_logit_suffix_k', 0)
        if K > 0 and student_logits.shape[1] > K:
            student_logits = student_logits[:, -K:, :]

        # Clean loss: MSE on logits
        clean_loss = F.mse_loss(student_logits, teacher_logits, reduction="sum")
        batch_loss = clean_loss.item()

        # Sensitivity loss
        if beta > 0.0 and cache.perturbed_input_ids is not None:
            pert_indices = []
            for idx in idx_batch:
                base = idx.item() * pert_per_input
                pert_indices.extend(range(base, base + pert_per_input))

            if pert_indices and max(pert_indices) < len(cache.perturbed_input_ids):
                pert_ids = cache.perturbed_input_ids[pert_indices].to(device)
                pert_teacher = cache.perturbed_logits[pert_indices].to(device).float()

                # Teacher sensitivity: delta_teacher = pert_logits - clean_logits
                teacher_clean_expanded = teacher_logits.repeat_interleave(
                    pert_per_input, dim=0
                )
                teacher_delta = pert_teacher - teacher_clean_expanded

                if _use_boundary:
                    pert_cache_indices = idx_batch.repeat_interleave(pert_per_input)
                    pert_ctx = _BoundaryInjectionHook(
                        student, cache, pert_cache_indices, boundary_layer_idx
                    )
                else:
                    pert_ctx = nullcontext()

                with pert_ctx, torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    pert_student = student(pert_ids).logits.float()

                # Truncate perturbed logits to match suffix positions
                if K > 0 and pert_student.shape[1] > K:
                    pert_student = pert_student[:, -K:, :]

                student_clean_expanded = student_logits.repeat_interleave(
                    pert_per_input, dim=0
                )
                student_delta = pert_student - student_clean_expanded

                sens_loss = F.mse_loss(
                    student_delta, teacher_delta, reduction="sum"
                )
                batch_loss += beta * sens_loss.item()

        total_loss += batch_loss
        total_samples += bsz

    if total_samples > 0:
        total_loss /= total_samples

    return total_loss


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def algebraic_initialize_block(
    student: nn.Module,
    cache,  # TeacherCache
    param_names: list,
    config: AlgebraicInitConfig,
    spsi_config,  # SPSIConfig
    boundary_layer_idx: Optional[int] = None,
    use_oracle_boundary: bool = False,
    gauge_basis: Optional[object] = None,  # GaugeBasis
) -> AlgebraicInitResult:
    """Main entry point: algebraic initialization for one suffix block.

    Steps:
    1. Build FlatParamSpec for the block's parameters
    2. Build random probe matrix V ∈ R^{p×k}
    3. If gauge_basis provided, project probes out of gauge
    4. Compute sketched Gramian G̃ and rhs via JVP
    5. Solve truncated ridge: a* = (G̃ + λI)^{-1} rhs
    6. Apply: θ_init = θ_0 + scale * V * a*
    7. Return result with diagnostics
    """
    device = next(student.parameters()).device

    # ------------------------------------------------------------------
    # Step 1: Build FlatParamSpec
    # ------------------------------------------------------------------
    spec = make_flat_param_spec(student, param_names)
    p = spec.num_params
    logger.info(
        "Algebraic init: block has %d parameters across %d tensors",
        p,
        len(spec.names),
    )

    # ------------------------------------------------------------------
    # Step 2: Build random probe matrix V ∈ R^{p×k}
    # ------------------------------------------------------------------
    k = min(config.num_probes, p)
    if k < config.num_probes:
        logger.warning(
            "num_probes=%d > num_params=%d; clamping k=%d",
            config.num_probes,
            p,
            k,
        )
    probe_matrix = build_probe_matrix(spec, k, config.seed, device)  # [p, k]

    # ------------------------------------------------------------------
    # Step 3: If gauge_basis provided, project probes out of gauge
    # ------------------------------------------------------------------
    if config.project_gauge and gauge_basis is not None:
        k_before = probe_matrix.shape[1]
        probe_matrix = project_probe_matrix(
            probe_matrix.cpu().float(), gauge_basis
        ).to(device)
        k_after = probe_matrix.shape[1]
        k = k_after
        logger.info(
            "Gauge projection: %d → %d probe directions (lost %d)",
            k_before,
            k_after,
            k_before - k_after,
        )

    if k == 0:
        logger.warning(
            "All probes fell into gauge subspace — skipping algebraic init."
        )
        zero_delta = torch.zeros(p, dtype=torch.float32)
        return AlgebraicInitResult(
            delta_flat=zero_delta,
            coefficients=torch.zeros(0, dtype=torch.float32),
            singular_values=torch.zeros(0, dtype=torch.float32),
            applied_scale=0.0,
            predicted_loss_decrease=0.0,
            gramian_result=None,
            init_loss=0.0,
            post_init_loss=0.0,
        )

    # ------------------------------------------------------------------
    # Step 4: Subsample queries & compute sketched Gramian G̃ and rhs
    # ------------------------------------------------------------------
    N_pool = len(cache.input_ids)
    N = min(config.query_subsample, N_pool)
    rng = torch.Generator(device="cpu").manual_seed(config.seed)
    query_indices = torch.randperm(N_pool, generator=rng)[:N]

    # Compute initial loss for diagnostics
    init_loss = compute_init_loss(
        student,
        cache,
        query_indices,
        spsi_config,
        boundary_layer_idx=boundary_layer_idx,
        use_oracle_boundary=use_oracle_boundary,
    )
    logger.info("Algebraic init: initial loss = %.6e", init_loss)

    # Build GramianConfig from AlgebraicInitConfig
    gramian_config = GramianConfig(
        num_probes=k,
        query_subsample=N,
        ridge=0.0,  # we do ridge in the solve step, not in the Gramian
        include_sensitivity=config.include_sensitivity,
        sensitivity_weight=config.sensitivity_weight,
        project_gauge=False,  # already projected probes above
        seed=config.seed,
    )

    gramian_result = compute_sketched_gramian(
        student=student,
        cache=cache,
        spec=spec,
        query_indices=query_indices,
        probe_matrix=probe_matrix,
        config=gramian_config,
        spsi_config=spsi_config,
        boundary_layer_idx=boundary_layer_idx,
        use_oracle_boundary=use_oracle_boundary,
        gauge_basis=None,  # already handled
    )

    # ------------------------------------------------------------------
    # Step 5: Solve truncated ridge
    # ------------------------------------------------------------------
    G_tilde = gramian_result.sketch_matrix  # [k, k] on CPU
    rhs = gramian_result.rhs  # [k] on CPU

    r = min(config.truncation_rank, k)
    a_star, singular_values = solve_truncated_ridge(
        sketch_matrix=G_tilde,
        rhs=rhs,
        truncation_rank=r,
        ridge=config.ridge,
    )

    # ------------------------------------------------------------------
    # Edge case: if all singular values are tiny, fall back to no init
    # ------------------------------------------------------------------
    sv_max = singular_values[0].item() if len(singular_values) > 0 else 0.0
    SV_FLOOR = 1e-12
    if sv_max < SV_FLOOR:
        logger.warning(
            "All singular values < %.1e (max=%.2e) — "
            "Gramian is nearly zero. Skipping algebraic init (zero delta).",
            SV_FLOOR,
            sv_max,
        )
        zero_delta = torch.zeros(p, dtype=torch.float32)
        return AlgebraicInitResult(
            delta_flat=zero_delta,
            coefficients=a_star,
            singular_values=singular_values,
            applied_scale=0.0,
            predicted_loss_decrease=0.0,
            gramian_result=gramian_result,
            init_loss=init_loss,
            post_init_loss=init_loss,
        )

    # Condition number diagnostics (top / bottom kept singular value)
    sv_min_kept = singular_values[min(r - 1, len(singular_values) - 1)].item()
    cond_number = sv_max / max(sv_min_kept, 1e-30)
    logger.info(
        "Algebraic init solve: rank=%d, σ_max=%.4e, σ_min(kept)=%.4e, cond=%.4e",
        r,
        sv_max,
        sv_min_kept,
        cond_number,
    )

    # Predicted quadratic decrease: Δ = a*^T rhs - 0.5 * a*^T G̃ a*
    # From the normal equations: G̃ a* ≈ rhs (approx due to truncation+ridge)
    # So Δ ≈ 0.5 * a*^T rhs
    predicted_decrease = 0.5 * (a_star @ rhs).item()
    logger.info(
        "Predicted loss decrease: %.6e, |a*|=%.4e",
        predicted_decrease,
        a_star.norm().item(),
    )

    # ------------------------------------------------------------------
    # Step 6: Apply θ_init = θ_0 + scale * V * a*
    # ------------------------------------------------------------------
    V_cpu = probe_matrix.cpu().float()  # [p, k]
    delta_flat = V_cpu @ a_star  # [p]
    delta_flat = config.step_scale * delta_flat

    # Current params
    theta_0 = params_to_flat(student, spec)  # [p], float32, CPU
    theta_init = theta_0 + delta_flat

    # Write back into the model in-place
    flat_to_params_(student, spec, theta_init)

    logger.info(
        "Applied algebraic init: |θ_0|=%.4e, |Δθ|=%.4e, |θ_init|=%.4e, scale=%.3f",
        theta_0.norm().item(),
        delta_flat.norm().item(),
        theta_init.norm().item(),
        config.step_scale,
    )

    # ------------------------------------------------------------------
    # Step 7: Compute post-init loss for diagnostics
    # ------------------------------------------------------------------
    post_init_loss = compute_init_loss(
        student,
        cache,
        query_indices,
        spsi_config,
        boundary_layer_idx=boundary_layer_idx,
        use_oracle_boundary=use_oracle_boundary,
    )
    actual_decrease = init_loss - post_init_loss
    logger.info(
        "Post-init loss: %.6e (actual decrease=%.6e, predicted=%.6e, ratio=%.3f)",
        post_init_loss,
        actual_decrease,
        predicted_decrease,
        actual_decrease / max(abs(predicted_decrease), 1e-30),
    )

    if post_init_loss > init_loss:
        logger.warning(
            "Algebraic init INCREASED loss by %.4e (%.2f%%). "
            "Consider reducing step_scale (currently %.3f) or truncation_rank.",
            post_init_loss - init_loss,
            100.0 * (post_init_loss - init_loss) / max(init_loss, 1e-30),
            config.step_scale,
        )

    return AlgebraicInitResult(
        delta_flat=delta_flat,
        coefficients=a_star,
        singular_values=singular_values,
        applied_scale=config.step_scale,
        predicted_loss_decrease=predicted_decrease,
        gramian_result=gramian_result,
        init_loss=init_loss,
        post_init_loss=post_init_loss,
    )
