"""
Observability Gramian for Transformer Suffix Inversion.

Computes the sketched suffix observability Gramian G(Q) via JVP probes,
without ever forming the full p x p matrix. Supports:
- Clean-logit Gramian: G_clean = (1/N) sum_i J_i^T J_i
- Augmented Gramian: G_aug = G_clean + beta * G_sensitivity
- Gauge-projected Gramian: P_perp G P_perp (project out continuous symmetries)

The key identity: for random orthonormal probes V in R^{p x k},
the k x k sketch  G_tilde = V^T G V  is computed via k JVPs per query,
never forming the full Jacobian.
"""

import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp, vmap
from torch.nn.attention import sdpa_kernel, SDPBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flat parameter bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class FlatParamSpec:
    """Maps between named parameters and a flat vector.

    Attributes:
        names:      list of parameter names (same order as model.named_parameters)
        shapes:     list of shapes for each parameter
        offsets:    list of (start, end) index pairs in the flat vector
        num_params: total number of scalar parameters
    """

    names: list
    shapes: list
    offsets: list
    num_params: int

    # Alias so that symmetry_gauge.py (which reads .total) works with this spec.
    @property
    def total(self) -> int:
        return self.num_params

    def name_to_slice(self, name: str):
        """Return the slice ``[start:end)`` in the flat vector for *name*."""
        for i, n in enumerate(self.names):
            if n == name:
                start, end = self.offsets[i]
                return slice(start, end)
        return None

    def name_to_index(self, name: str):
        """Return the positional index in ``self.names`` for *name*."""
        for i, n in enumerate(self.names):
            if n == name:
                return i
        return None


def make_flat_param_spec(model: nn.Module, param_names: list) -> FlatParamSpec:
    """Create a flat parameter specification for *param_names*.

    Parameters are stored in the order they appear in ``model.named_parameters()``,
    filtered to those whose name is in *param_names*.
    """
    name_set = set(param_names)
    names, shapes, offsets = [], [], []
    cursor = 0
    for name, param in model.named_parameters():
        if name not in name_set:
            continue
        numel = param.numel()
        names.append(name)
        shapes.append(param.shape)
        offsets.append((cursor, cursor + numel))
        cursor += numel
    if cursor == 0:
        raise ValueError(
            f"No matching parameters found. Got {len(param_names)} names; "
            "check that they match model.named_parameters() keys."
        )
    return FlatParamSpec(names=names, shapes=shapes, offsets=offsets, num_params=cursor)


def params_to_flat(model: nn.Module, spec: FlatParamSpec) -> torch.Tensor:
    """Extract suffix parameters into a contiguous flat float32 vector."""
    flat = torch.empty(spec.num_params, dtype=torch.float32, device="cpu")
    param_dict = dict(model.named_parameters())
    for name, (start, end) in zip(spec.names, spec.offsets):
        flat[start:end] = param_dict[name].detach().float().cpu().reshape(-1)
    return flat


def flat_to_params_(
    model: nn.Module, spec: FlatParamSpec, flat: torch.Tensor
) -> None:
    """Write a flat vector back into model parameters **in-place**."""
    param_dict = dict(model.named_parameters())
    for name, shape, (start, end) in zip(spec.names, spec.shapes, spec.offsets):
        param = param_dict[name]
        new_data = flat[start:end].reshape(shape).to(dtype=param.dtype, device=param.device)
        param.data.copy_(new_data)


def _spec_to_param_dict(
    model: nn.Module, spec: FlatParamSpec, flat: torch.Tensor
) -> dict:
    """Convert a flat vector into a {name: tensor} dict matching model dtypes/devices."""
    param_dict = dict(model.named_parameters())
    out = {}
    for name, shape, (start, end) in zip(spec.names, spec.shapes, spec.offsets):
        ref = param_dict[name]
        out[name] = flat[start:end].reshape(shape).to(dtype=ref.dtype, device=ref.device)
    return out


# ---------------------------------------------------------------------------
# Gramian configuration and result
# ---------------------------------------------------------------------------


@dataclass
class GramianConfig:
    """Configuration for sketched Gramian computation."""

    num_probes: int = 64  # k: number of random directions
    query_subsample: int = 256  # N: number of queries to use per estimate
    ridge: float = 1e-6  # Tikhonov regulariser added to diagonal
    include_sensitivity: bool = True  # include perturbation sensitivity term
    sensitivity_weight: float = 0.1  # beta
    project_gauge: bool = True  # project out gauge symmetries
    seed: int = 42
    query_batch_size: int = 4  # mini-batch size for JVP passes (memory)


@dataclass
class GramianResult:
    """Output of ``compute_sketched_gramian``."""

    sketch_matrix: torch.Tensor  # k x k sketched Gramian  (float32)
    eigenvalues: torch.Tensor  # k eigenvalues (descending)
    eigenvectors: torch.Tensor  # k x k eigenvectors (columns)
    rhs: torch.Tensor  # k-dim algebraic init right-hand side
    projected_min_eig: float  # smallest eigenvalue (after gauge projection)
    logdet: float  # log-determinant of regularised sketch
    trace: float  # trace of sketch matrix
    effective_rank: float  # trace^2 / frobenius_norm^2
    num_queries_used: int


# ---------------------------------------------------------------------------
# Probe matrix construction
# ---------------------------------------------------------------------------


def build_probe_matrix(
    spec: FlatParamSpec,
    num_probes: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build random orthonormal probe matrix V in R^{p x k}.

    Generates a p x k Gaussian random matrix and QR-orthonormalises it.
    When p >> k the columns of Q are uniformly distributed on the Stiefel
    manifold, which is ideal for sketching.
    """
    p = spec.num_params
    k = num_probes
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Generate on CPU (large), then move to target device for QR
    raw = torch.randn(p, k, generator=gen, dtype=dtype)
    # Economy QR: Q is p x k with orthonormal columns
    # For large p (>1M), QR is much faster on GPU
    if device.type == "cuda" and p > 100_000:
        raw_dev = raw.to(device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(raw_dev, mode="reduced")
        return Q  # already on device
    else:
        Q, _ = torch.linalg.qr(raw, mode="reduced")
        return Q.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# JVP computation
# ---------------------------------------------------------------------------


def _make_functional_forward(model: nn.Module, spec: FlatParamSpec):
    """Return a function  f(flat_suffix, input_ids) -> logits
    that replaces only the suffix parameters with values from *flat_suffix*.
    """
    # Snapshot the full parameter dict once (detached, no grad)
    base_params = {n: p.detach() for n, p in model.named_parameters()}
    suffix_set = set(spec.names)

    def forward_fn(flat_suffix: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Build the override dict from the flat suffix vector
        override = {}
        for name, shape, (start, end) in zip(
            spec.names, spec.shapes, spec.offsets
        ):
            ref = base_params[name]
            override[name] = flat_suffix[start:end].reshape(shape).to(
                dtype=ref.dtype, device=ref.device
            )
        # Merge: base params + suffix overrides
        full_params = {**base_params, **override}
        return functional_call(model, full_params, (input_ids,)).logits

    return forward_fn


def jvp_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    spec: FlatParamSpec,
    direction: torch.Tensor,
    boundary_hook_factory: Optional[Callable] = None,
) -> torch.Tensor:
    """Compute J * v via forward-mode AD (``torch.func.jvp``).

    Args:
        model: student model (parameters used as linearisation point).
        input_ids: ``[B, T]`` input token IDs.
        spec: flat parameter spec for the suffix parameters.
        direction: ``[p]`` tangent direction in flat parameter space.
        boundary_hook_factory: optional callable returning a context manager
            that injects oracle boundary states.  Applied around the forward.

    Returns:
        ``[B, T, V]`` Jacobian-vector product in logit space (float32).
    """
    device = next(model.parameters()).device

    # Current suffix parameters as flat float32
    primals = params_to_flat(model, spec).to(device)
    tangents = direction.to(device=device, dtype=torch.float32)
    input_ids = input_ids.to(device)

    forward_fn = _make_functional_forward(model, spec)

    ctx = boundary_hook_factory() if boundary_hook_factory is not None else nullcontext()
    with ctx, sdpa_kernel(SDPBackend.MATH):
        # jvp returns (primals_out, tangents_out)
        # NOTE: sdpa_kernel(MATH) is required because forward-mode AD
        # does not support _scaled_dot_product_flash_attention.
        _, jvp_out = jvp(
            lambda flat: forward_fn(flat, input_ids),
            (primals,),
            (tangents,),
        )
    return jvp_out.float()


# ---------------------------------------------------------------------------
# Gauge projection helpers
# ---------------------------------------------------------------------------


def _project_out_gauge(
    probe_matrix: torch.Tensor,
    gauge_basis,  # GaugeBasis or torch.Tensor [p, g]
) -> torch.Tensor:
    """Project probe columns orthogonal to the gauge subspace.

    Supports both the new sparse ``GaugeBasis`` object and the legacy dense
    ``[p, g]`` tensor form for backward compatibility.

    Args:
        probe_matrix: ``[p, k]`` orthonormal probes.
        gauge_basis:  ``GaugeBasis`` (sparse) or ``[p, g]`` dense tensor.

    Returns:
        ``[p, k]`` probes with gauge components removed, re-orthonormalised.
    """
    # Import here to avoid circular import at module level
    from src.symmetry_gauge import GaugeBasis as _GaugeBasis

    if isinstance(gauge_basis, _GaugeBasis):
        # Use the memory-efficient streaming projection
        from src.symmetry_gauge import project_out_gauge as _sg_project
        projected = _sg_project(probe_matrix, gauge_basis)
    else:
        # Legacy path: dense [p, g] tensor
        # P_perp = I - U U^T, applied as: v - U (U^T v)
        coeffs = gauge_basis.T @ probe_matrix  # [g, k]
        projected = probe_matrix - gauge_basis @ coeffs  # [p, k]

    # Re-orthonormalise (columns may have become dependent)
    Q, R = torch.linalg.qr(projected, mode="reduced")
    # Handle numerically zero columns (if gauge ate a probe direction)
    diag = R.diag().abs()
    valid = diag > 1e-8
    if not valid.all():
        n_lost = (~valid).sum().item()
        num_dirs = (
            gauge_basis.num_directions
            if isinstance(gauge_basis, _GaugeBasis)
            else gauge_basis.shape[1]
        )
        logger.warning(
            "Gauge projection removed %d probe directions (dim(gauge)=%d)",
            n_lost,
            num_dirs,
        )
    return Q


# ---------------------------------------------------------------------------
# Core: sketched Gramian computation
# ---------------------------------------------------------------------------


def compute_sketched_gramian(
    student: nn.Module,
    cache,  # TeacherCache
    spec: FlatParamSpec,
    query_indices: torch.Tensor,
    probe_matrix: torch.Tensor,
    config: GramianConfig,
    spsi_config=None,  # SPSIConfig – used for perturbation geometry
    boundary_layer_idx: Optional[int] = None,
    use_oracle_boundary: bool = False,
    gauge_basis=None,  # GaugeBasis or torch.Tensor [p, g] or None
) -> GramianResult:
    """Compute the sketched augmented Gramian and return eigendecomposition.

    Algorithm
    ---------
    For each query x_i (i = 1..N) and each probe v_j (j = 1..k):
        s_{i,j} = J(x_i) * v_j          (via forward-mode JVP)

    Sketch (clean):
        G_tilde_clean = (1/N) sum_i  S_i^T S_i
    where S_i = [s_{i,1} | ... | s_{i,k}] reshaped to (BTV, k).

    If ``config.include_sensitivity`` and perturbed data available:
        For each perturbed variant x_i^delta:
            s^delta_{i,j} = J(x_i^delta) * v_j
        D_{i,j} = s^delta_{i,j} - s_{i,j}
        G_tilde_sens = (1/N) sum_i D_i^T D_i

        G_tilde_aug = G_tilde_clean + beta * G_tilde_sens

    Algebraic init RHS:
        rhs_j = (1/N) sum_i  flatten(J_i v_j)^T  flatten(r_i)
    where r_i = teacher_logits_i - student_logits_i.

    If ``gauge_basis`` is provided, probes are first projected out of the
    gauge subspace.
    """
    device = next(student.parameters()).device
    k = probe_matrix.shape[1]
    N = len(query_indices)

    # --- Gauge projection ---------------------------------------------------
    V = probe_matrix  # [p, k]
    if config.project_gauge and gauge_basis is not None:
        # GaugeBasis objects don't need .to(); dense tensors do.
        from src.symmetry_gauge import GaugeBasis as _GaugeBasis
        if isinstance(gauge_basis, _GaugeBasis):
            V = _project_out_gauge(V, gauge_basis)
        else:
            V = _project_out_gauge(V, gauge_basis.to(V.device, V.dtype))
        k = V.shape[1]  # may have shrunk
        logger.info("Probes after gauge projection: k=%d", k)

    # --- Accumulators -------------------------------------------------------
    G_clean = torch.zeros(k, k, dtype=torch.float32, device=device)
    G_sens = torch.zeros(k, k, dtype=torch.float32, device=device)
    rhs_acc = torch.zeros(k, dtype=torch.float32, device=device)

    student.eval()
    bs = config.query_batch_size

    # Perturbation geometry from spsi_config
    if spsi_config is not None:
        P = spsi_config.num_perturbation_positions
        R = spsi_config.num_replacement_tokens
        pert_per_input = P * R
    else:
        P, R, pert_per_input = 4, 2, 8

    # Whether to use oracle boundary injection
    _use_boundary = (
        use_oracle_boundary
        and boundary_layer_idx is not None
        and hasattr(cache, "boundary_states")
        and boundary_layer_idx in cache.boundary_states
    )

    def _make_boundary_hook(indices_batch):
        """Return a factory for the boundary context manager, or None."""
        if not _use_boundary:
            return None
        from src.parameter_inverter import _BoundaryInjectionHook

        def factory():
            return _BoundaryInjectionHook(
                student, cache, indices_batch, boundary_layer_idx
            )

        return factory

    num_batches = math.ceil(N / bs)
    logger.info(
        "Computing sketched Gramian: N=%d queries, k=%d probes, bs=%d (%d batches)",
        N, k, bs, num_batches,
    )

    for batch_idx in range(num_batches):
        start = batch_idx * bs
        end = min(start + bs, N)
        idx_batch = query_indices[start:end]

        # Fetch clean data from cache
        input_ids = cache.input_ids[idx_batch].to(device)
        teacher_logits = cache.clean_logits[idx_batch].to(device).float()

        # Compute student logits at current parameters (no grad needed)
        boundary_factory = _make_boundary_hook(idx_batch)
        ctx = boundary_factory() if boundary_factory is not None else nullcontext()
        with ctx, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits = student(input_ids).logits.float()

        # Truncate to match teacher cache suffix positions
        _K = getattr(cache, '_logit_suffix_k', 0)
        if _K > 0 and student_logits.shape[1] > _K:
            student_logits = student_logits[:, -_K:, :]

        residual = teacher_logits - student_logits  # [B, K_pos, V_vocab]
        residual_flat = residual.reshape(-1)  # [B*K_pos*V_vocab]

        # --- JVP for each probe direction -----------------------------------
        # S_clean[:, j] = flatten(J_i * v_j)  for current batch
        S_clean = torch.zeros(
            residual_flat.shape[0], k, dtype=torch.float32, device=device
        )

        for j in range(k):
            v_j = V[:, j]  # [p]
            jvp_out = jvp_logits(
                student,
                input_ids,
                spec,
                v_j,
                boundary_hook_factory=boundary_factory,
            )  # [B, T, V_vocab]
            # Truncate JVP output to match suffix positions
            if _K > 0 and jvp_out.shape[1] > _K:
                jvp_out = jvp_out[:, -_K:, :]
            S_clean[:, j] = jvp_out.reshape(-1)

        # Accumulate clean Gramian: S^T S
        G_clean.addmm_(S_clean.T, S_clean)

        # Accumulate RHS: rhs_j += S_clean[:, j]^T @ residual_flat
        rhs_acc += S_clean.T @ residual_flat

        # --- Sensitivity term (optional) ------------------------------------
        if config.include_sensitivity and cache.perturbed_input_ids is not None:
            # Gather perturbed data for this batch
            pert_indices_list = []
            for idx in idx_batch:
                base = idx.item() * pert_per_input
                pert_indices_list.extend(range(base, base + pert_per_input))

            max_pert_idx = max(pert_indices_list) if pert_indices_list else -1
            if pert_indices_list and max_pert_idx < len(cache.perturbed_input_ids):
                pert_ids = cache.perturbed_input_ids[pert_indices_list].to(device)

                # Boundary for perturbed: replicate indices
                if _use_boundary:
                    pert_cache_indices = idx_batch.repeat_interleave(pert_per_input)
                    pert_boundary_factory = _make_boundary_hook(pert_cache_indices)
                else:
                    pert_boundary_factory = None

                # JVP on perturbed inputs, compute D = S_pert - S_clean_expanded
                # We process perturbations in sub-batches to save memory
                n_pert = len(pert_ids)
                # Use suffix-truncated sequence length for sizing
                _eff_seq_len = _K if _K > 0 else input_ids.shape[1]
                vocab_dim = student_logits.shape[-1]
                S_pert = torch.zeros(
                    n_pert * _eff_seq_len * vocab_dim,
                    k,
                    dtype=torch.float32,
                    device=device,
                )

                # We also need the expanded S_clean for subtraction
                # Each original sample has pert_per_input perturbations
                S_clean_expanded = S_clean.reshape(
                    len(idx_batch), -1, k
                ).repeat_interleave(pert_per_input, dim=0).reshape(-1, k)

                pert_sub_bs = max(1, bs)
                for psub_start in range(0, n_pert, pert_sub_bs):
                    psub_end = min(psub_start + pert_sub_bs, n_pert)
                    pert_batch = pert_ids[psub_start:psub_end]
                    flat_len = (psub_end - psub_start) * _eff_seq_len * vocab_dim

                    pfactory = None
                    if pert_boundary_factory is not None:
                        # Slice the perturbed boundary indices for this sub-batch
                        psub_indices = pert_cache_indices[psub_start:psub_end]

                        def _pfactory_fn(_indices=psub_indices):
                            from src.parameter_inverter import _BoundaryInjectionHook
                            return _BoundaryInjectionHook(
                                student, cache, _indices, boundary_layer_idx
                            )

                        pfactory = _pfactory_fn

                    row_start = psub_start * _eff_seq_len * vocab_dim
                    row_end = row_start + flat_len

                    for j in range(k):
                        v_j = V[:, j]
                        jvp_pert = jvp_logits(
                            student, pert_batch, spec, v_j,
                            boundary_hook_factory=pfactory,
                        )
                        # Truncate JVP to match suffix positions
                        if _K > 0 and jvp_pert.shape[1] > _K:
                            jvp_pert = jvp_pert[:, -_K:, :]
                        S_pert[row_start:row_end, j] = jvp_pert.reshape(-1)

                # D = S_pert - S_clean_expanded
                # Truncate S_clean_expanded to match S_pert if shapes differ
                actual_rows = S_pert.shape[0]
                S_clean_exp = S_clean_expanded[:actual_rows]
                D = S_pert - S_clean_exp

                G_sens.addmm_(D.T, D)

    # --- Normalise by N -----------------------------------------------------
    G_clean.div_(N)
    G_sens.div_(N)
    rhs_acc.div_(N)

    # --- Assemble augmented Gramian -----------------------------------------
    beta = config.sensitivity_weight if config.include_sensitivity else 0.0
    G_aug = G_clean + beta * G_sens

    # Ridge regularisation
    G_aug.diagonal().add_(config.ridge)

    # --- Eigendecomposition of k x k matrix ---------------------------------
    eigenvalues, eigenvectors = torch.linalg.eigh(G_aug)
    # eigh returns ascending; flip to descending
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    # --- Summary statistics -------------------------------------------------
    trace_val = eigenvalues.sum().item()
    # log-det: sum of log of eigenvalues (all should be positive thanks to ridge)
    log_eigs = torch.log(eigenvalues.clamp(min=1e-30))
    logdet_val = log_eigs.sum().item()
    min_eig = eigenvalues[-1].item()
    frob_sq = (eigenvalues ** 2).sum().item()
    effective_rank = (trace_val ** 2 / frob_sq) if frob_sq > 0 else 0.0

    logger.info(
        "Gramian stats: trace=%.4e, logdet=%.4f, min_eig=%.4e, eff_rank=%.2f",
        trace_val, logdet_val, min_eig, effective_rank,
    )

    return GramianResult(
        sketch_matrix=G_aug.cpu(),
        eigenvalues=eigenvalues.cpu(),
        eigenvectors=eigenvectors.cpu(),
        rhs=rhs_acc.cpu(),
        projected_min_eig=min_eig,
        logdet=logdet_val,
        trace=trace_val,
        effective_rank=effective_rank,
        num_queries_used=N,
    )


# ---------------------------------------------------------------------------
# Convenience: algebraic initialisation from Gramian
# ---------------------------------------------------------------------------


def algebraic_init_from_gramian(
    result: GramianResult,
    probe_matrix: torch.Tensor,
    spec: FlatParamSpec,
    model: nn.Module,
    regularised: bool = True,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Compute an algebraic initial guess for suffix parameters.

    Solves  G_tilde * alpha = rhs  in the k-dim sketch space, then
    lifts back to the full p-dim parameter space:

        delta_theta = V @ alpha

    and adds the result to the current model parameters.

    Args:
        result: output of ``compute_sketched_gramian``.
        probe_matrix: ``[p, k]`` probe matrix used during Gramian computation.
        spec: flat parameter specification.
        model: student model (current parameters are the linearisation point).
        regularised: if True, use (G + ridge I) for the solve.
        ridge: Tikhonov ridge for the solve.

    Returns:
        ``[p]`` flat parameter vector with the algebraic correction applied.
    """
    G = result.sketch_matrix.clone()
    if regularised:
        G.diagonal().add_(ridge)
    rhs = result.rhs

    # Solve k x k system
    alpha = torch.linalg.solve(G, rhs)  # [k]

    # Lift to full parameter space
    V = probe_matrix.cpu().float()
    delta = V @ alpha  # [p]

    # Current parameters + correction
    theta_current = params_to_flat(model, spec)
    theta_new = theta_current + delta

    logger.info(
        "Algebraic init: |delta|=%.4e, |theta_cur|=%.4e, |alpha|=%.4e",
        delta.norm().item(),
        theta_current.norm().item(),
        alpha.norm().item(),
    )

    return theta_new


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def compute_gramian_for_block(
    student: nn.Module,
    cache,  # TeacherCache
    param_names: list,
    config: GramianConfig,
    spsi_config=None,
    boundary_layer_idx: Optional[int] = None,
    use_oracle_boundary: bool = False,
    gauge_basis=None,  # GaugeBasis or torch.Tensor [p, g] or None
) -> tuple:
    """High-level helper: build spec + probes, compute Gramian, return everything.

    Returns:
        (result, spec, probe_matrix) tuple for downstream use.
    """
    device = next(student.parameters()).device

    # 1. Build flat param spec
    spec = make_flat_param_spec(student, param_names)
    logger.info(
        "Block has %d parameters across %d tensors",
        spec.num_params, len(spec.names),
    )

    # 2. Build probes
    k = min(config.num_probes, spec.num_params)
    if k < config.num_probes:
        logger.warning(
            "num_probes=%d > num_params=%d; clamping k=%d",
            config.num_probes, spec.num_params, k,
        )
    probe_matrix = build_probe_matrix(spec, k, config.seed, device)

    # 3. Subsample query indices
    N_pool = len(cache.input_ids)
    N = min(config.query_subsample, N_pool)
    rng = torch.Generator(device="cpu").manual_seed(config.seed)
    query_indices = torch.randperm(N_pool, generator=rng)[:N]

    # 4. Compute
    result = compute_sketched_gramian(
        student=student,
        cache=cache,
        spec=spec,
        query_indices=query_indices,
        probe_matrix=probe_matrix,
        config=config,
        spsi_config=spsi_config,
        boundary_layer_idx=boundary_layer_idx,
        use_oracle_boundary=use_oracle_boundary,
        gauge_basis=gauge_basis,
    )

    return result, spec, probe_matrix
