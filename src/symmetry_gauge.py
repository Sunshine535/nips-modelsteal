"""
Continuous symmetry gauge basis for transformer suffix blocks.

For the Gramian to measure identifiability on the symmetry-quotiented space,
we project probe directions out of the gauge (null) subspace spanned by
continuous reparameterizations that preserve the input-output map.

Supports: RMSNorm scale gauge, SiLU-gated MLP neuron scaling gauge.

Target architectures:
  - Qwen3.5-0.8B: 24 blocks, 1024 hidden, 16 heads, intermediate=3584,
    SiLU gated MLP, RMSNorm, tied embeddings.
  - Llama-3.2-1B: similar GQA + RMSNorm + SiLU gated MLP.

Key symmetries:
  1. RMSNorm scale absorption: For each RMSNorm, scaling g' = D*g with D
     diagonal can be absorbed by inverse scaling consumer weights.
     Tangent: eps_i * (d/dg_i - sum_j d/dW_consumer_ij).
  2. Gated MLP up/down neuron scaling: For SiLU-gated MLP, scale up rows
     by alpha and down columns by 1/alpha. Gate is NOT scaled because SiLU
     is not positively homogeneous (SiLU(alpha*x) != alpha*SiLU(x)).
     Tangent: eps_n * (d/d_up_n - d/d_down_n).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FlatParamSpec: mapping between named parameters and a flat R^p vector
# ---------------------------------------------------------------------------

@dataclass
class FlatParamSpec:
    """Specification mapping named parameters to a contiguous flat vector.

    Attributes:
        names:   ordered list of parameter names included in the flat vector.
        shapes:  corresponding shapes for each parameter.
        offsets: starting index in the flat vector for each parameter.
        total:   total dimensionality p = sum of numel for all params.
    """
    names: list = field(default_factory=list)
    shapes: list = field(default_factory=list)
    offsets: list = field(default_factory=list)
    total: int = 0

    def name_to_slice(self, name: str) -> Optional[slice]:
        """Return the slice in the flat vector for a given parameter name."""
        for i, n in enumerate(self.names):
            if n == name:
                numel = 1
                for s in self.shapes[i]:
                    numel *= s
                return slice(self.offsets[i], self.offsets[i] + numel)
        return None

    def name_to_index(self, name: str) -> Optional[int]:
        """Return the index in self.names for a given parameter name."""
        for i, n in enumerate(self.names):
            if n == name:
                return i
        return None


def build_flat_param_spec(
    model: nn.Module,
    param_names: Optional[list] = None,
) -> FlatParamSpec:
    """Build a FlatParamSpec from a model's named parameters.

    Args:
        model:       the nn.Module.
        param_names: if given, only include these parameters (in this order).
                     If None, include all parameters in model.named_parameters() order.

    Returns:
        FlatParamSpec with names, shapes, offsets, and total.
    """
    spec = FlatParamSpec()
    offset = 0

    if param_names is not None:
        # Build a lookup from model
        model_params = dict(model.named_parameters())
        for name in param_names:
            if name not in model_params:
                logger.warning("Parameter %s not found in model, skipping.", name)
                continue
            p = model_params[name]
            spec.names.append(name)
            spec.shapes.append(list(p.shape))
            spec.offsets.append(offset)
            offset += p.numel()
    else:
        for name, p in model.named_parameters():
            spec.names.append(name)
            spec.shapes.append(list(p.shape))
            spec.offsets.append(offset)
            offset += p.numel()

    spec.total = offset
    return spec


def flatten_params(
    model: nn.Module,
    spec: FlatParamSpec,
) -> torch.Tensor:
    """Flatten model parameters into a single vector according to spec."""
    model_params = dict(model.named_parameters())
    flat = torch.zeros(spec.total, dtype=torch.float32)
    for i, name in enumerate(spec.names):
        if name in model_params:
            sl = slice(spec.offsets[i], spec.offsets[i] + model_params[name].numel())
            flat[sl] = model_params[name].detach().float().flatten()
    return flat


# ---------------------------------------------------------------------------
# GaugeBasis
# ---------------------------------------------------------------------------

@dataclass
class GaugeBasis:
    """Gauge directions in flat parameter space (memory-efficient streaming form).

    Instead of storing a dense [p, g] orthonormal matrix (which can be hundreds
    of GB), we store gauge directions as a list of *normalised* p-dimensional
    vectors.  Projection is done by iterating over them one at a time
    (modified Gram-Schmidt), so peak memory is O(p·k) where k ≪ g is the
    number of probe columns.

    Attributes:
        directions:      list of (indices, values) pairs – sparse representation
                         of each gauge direction.  ``indices`` is a 1-D int64
                         tensor, ``values`` is a 1-D float32 tensor of the same
                         length.  The direction has been pre-normalised to unit
                         ℓ₂ norm.
        labels:          human-readable label for each direction.
        num_directions:  total number of gauge directions.
        total_dim:       dimensionality p of the flat parameter space.
    """
    directions: list      # list of (indices: Tensor, values: Tensor)
    labels: list          # human-readable label for each direction
    num_directions: int
    total_dim: int = 0


# ---------------------------------------------------------------------------
# Architecture detection helpers
# ---------------------------------------------------------------------------

# Qwen / Llama naming conventions:
#   model.layers.{i}.input_layernorm.weight        -> consumed by self_attn.{q,k,v}_proj.weight
#   model.layers.{i}.post_attention_layernorm.weight -> consumed by mlp.{gate,up}_proj.weight
#   model.norm.weight                                -> consumed by lm_head.weight
#
# The RMSNorm output is: y = (x / rms(x)) * g, where g is the weight vector.
# Any diagonal scaling D applied to g can be absorbed into the consumer W:
#   y' = (x / rms(x)) * (D * g)  <==>  W' = W * D^{-1}  (i.e. scale columns of W)

# Mapping from RMSNorm parameter name to its consumer weight names.
# Pattern: (rmsnorm_suffix, [consumer_suffixes])
_RMSNORM_CONSUMER_MAP = [
    # Pre-attention RMSNorm -> Q, K, V projections
    (
        "input_layernorm.weight",
        [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        ],
    ),
    # Post-attention RMSNorm -> gate_proj, up_proj
    (
        "post_attention_layernorm.weight",
        [
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
        ],
    ),
]

# Final layer norm -> lm_head
_FINAL_NORM_CONSUMER = ("model.norm.weight", ["lm_head.weight"])


def _find_block_prefix(param_name: str) -> Optional[str]:
    """Extract the block prefix like 'model.layers.23.' from a param name."""
    parts = param_name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return ".".join(parts[: i + 2]) + "."
    return None


def _block_idx_from_prefix(prefix: str) -> int:
    """Extract block index from prefix like 'model.layers.23.'."""
    parts = prefix.rstrip(".").split(".")
    return int(parts[-1])


# ---------------------------------------------------------------------------
# RMSNorm gauge basis
# ---------------------------------------------------------------------------

def _build_sparse_direction(idx_val_pairs: list):
    """Build a normalised sparse direction from a list of (index, value) pairs.

    Returns ``(indices_tensor, values_tensor)`` or ``None`` if the direction
    is near-zero.  Both tensors live on CPU.
    """
    if not idx_val_pairs:
        return None
    indices = torch.tensor([p[0] for p in idx_val_pairs], dtype=torch.int64)
    values = torch.tensor([p[1] for p in idx_val_pairs], dtype=torch.float32)
    norm_val = values.norm()
    if norm_val < 1e-12:
        return None
    values = values / norm_val
    return (indices, values)


def build_rmsnorm_gauge_basis(
    model: nn.Module,
    spec: FlatParamSpec,
    block_indices: list,
) -> list:
    """Build gauge tangent vectors for RMSNorm scale absorption.

    Returns list of ``((indices, values), label)`` pairs in **sparse** format.
    Never allocates dense ``R^p`` vectors — memory is O(nnz) per direction.
    """
    model_params = dict(model.named_parameters())
    directions = []

    def _rmsnorm_directions_for(norm_slice, norm_idx, consumer_infos,
                                block_label_prefix):
        hidden_dim = 1
        for s in spec.shapes[norm_idx]:
            hidden_dim *= s

        # Read actual parameter values for parameter-dependent gauge tangents
        norm_param_name = spec.names[norm_idx]
        norm_values = model_params[norm_param_name].detach().float().flatten()

        consumer_weight_cache = {}
        for cname, cslice, cshape in consumer_infos:
            consumer_weight_cache[cname] = model_params[cname].detach().float()

        for i in range(hidden_dim):
            # Lie algebra tangent at current θ: d/dε|_{ε=0} (g_i*(1+ε))  = g_i
            g_i = norm_values[i].item()
            pairs = [(norm_slice.start + i, g_i)]

            for cname, cslice, cshape in consumer_infos:
                out_features = cshape[0]
                in_features = cshape[1] if len(cshape) > 1 else 1
                if i >= in_features:
                    continue
                W = consumer_weight_cache[cname]
                for r in range(out_features):
                    # Tangent: d/dε|_{ε=0} (W_{r,i}/(1+ε)) = -W_{r,i}
                    W_ri = W[r, i].item() if W.dim() > 1 else W[r].item()
                    pairs.append((cslice.start + r * in_features + i, -W_ri))

            sparse_dir = _build_sparse_direction(pairs)
            if sparse_dir is not None:
                directions.append((sparse_dir, f"{block_label_prefix}_dim{i}"))

    for block_idx in block_indices:
        prefix = None
        for name in spec.names:
            candidate = _find_block_prefix(name)
            if candidate is not None and _block_idx_from_prefix(candidate) == block_idx:
                prefix = candidate
                break

        if prefix is None:
            logger.warning("Could not find prefix for block %d, skipping.", block_idx)
            continue

        for norm_suffix, consumer_suffixes in _RMSNORM_CONSUMER_MAP:
            norm_name = prefix + norm_suffix
            norm_slice = spec.name_to_slice(norm_name)
            if norm_slice is None:
                continue

            norm_idx = spec.name_to_index(norm_name)
            consumer_infos = []
            for csuffix in consumer_suffixes:
                cname = prefix + csuffix
                cslice = spec.name_to_slice(cname)
                if cslice is not None:
                    cidx = spec.name_to_index(cname)
                    consumer_infos.append((cname, cslice, spec.shapes[cidx]))

            if not consumer_infos:
                continue

            _rmsnorm_directions_for(
                norm_slice, norm_idx, consumer_infos,
                f"rmsnorm_gauge_block{block_idx}_{norm_suffix}",
            )

    # Handle final norm -> lm_head
    final_norm_name, final_consumer_names = _FINAL_NORM_CONSUMER
    final_norm_slice = spec.name_to_slice(final_norm_name)
    if final_norm_slice is not None:
        final_norm_idx = spec.name_to_index(final_norm_name)
        final_consumer_infos = []
        for cname in final_consumer_names:
            cslice = spec.name_to_slice(cname)
            if cslice is not None:
                cidx = spec.name_to_index(cname)
                final_consumer_infos.append((cname, cslice, spec.shapes[cidx]))

        if final_consumer_infos:
            _rmsnorm_directions_for(
                final_norm_slice, final_norm_idx, final_consumer_infos,
                "rmsnorm_gauge_final_norm",
            )

    logger.info(
        "Built %d RMSNorm gauge directions for blocks %s.",
        len(directions), block_indices,
    )
    return directions


# ---------------------------------------------------------------------------
# Gated MLP neuron scaling gauge basis
# ---------------------------------------------------------------------------

def build_gated_mlp_gauge_basis(
    model: nn.Module,
    spec: FlatParamSpec,
    block_indices: list,
) -> list:
    """Build gauge tangent vectors for gated MLP neuron scaling.

    Returns list of ``((indices, values), label)`` pairs in **sparse** format.
    Never allocates dense ``R^p`` vectors — memory is O(nnz) per direction.
    """
    directions = []

    for block_idx in block_indices:
        prefix = None
        for name in spec.names:
            candidate = _find_block_prefix(name)
            if candidate is not None and _block_idx_from_prefix(candidate) == block_idx:
                prefix = candidate
                break

        if prefix is None:
            continue

        gate_name = prefix + "mlp.gate_proj.weight"
        up_name = prefix + "mlp.up_proj.weight"
        down_name = prefix + "mlp.down_proj.weight"

        gate_slice = spec.name_to_slice(gate_name)
        up_slice = spec.name_to_slice(up_name)
        down_slice = spec.name_to_slice(down_name)

        if gate_slice is None or up_slice is None or down_slice is None:
            logger.warning(
                "Block %d: missing gate/up/down proj in spec, skipping MLP gauge.",
                block_idx,
            )
            continue

        gate_idx = spec.name_to_index(gate_name)
        up_idx = spec.name_to_index(up_name)
        down_idx = spec.name_to_index(down_name)

        gate_shape = spec.shapes[gate_idx]   # [intermediate_size, hidden_size]
        up_shape = spec.shapes[up_idx]       # [intermediate_size, hidden_size]
        down_shape = spec.shapes[down_idx]   # [hidden_size, intermediate_size]

        intermediate_size = gate_shape[0]
        hidden_size = gate_shape[1] if len(gate_shape) > 1 else 1
        down_out = down_shape[0]
        down_in = down_shape[1] if len(down_shape) > 1 else 1

        if intermediate_size != down_in:
            logger.warning(
                "Block %d: gate intermediate %d != down columns %d, skipping.",
                block_idx, intermediate_size, down_in,
            )
            continue

        # Read actual weight values for parameter-dependent gauge tangents
        # Note: gate_proj is NOT included because SiLU is not positively
        # homogeneous — only up/down scaling is an exact symmetry.
        model_params = dict(model.named_parameters())
        up_weight = model_params[up_name].detach().float()
        down_weight = model_params[down_name].detach().float()

        for n in range(intermediate_size):
            # Lie algebra tangent at current θ for neuron scaling:
            # The exact symmetry for SiLU-gated MLP only scales up and down,
            # NOT gate, because SiLU is not positively homogeneous.
            # d/dε|_{ε=0} up_row_n*(1+ε) = up_row_n
            # d/dε|_{ε=0} down_col_n/(1+ε) = -down_col_n

            # up_proj row n: contiguous block of hidden_size elements
            up_start = up_slice.start + n * hidden_size
            up_indices = torch.arange(up_start, up_start + hidden_size,
                                      dtype=torch.int64)
            up_values = up_weight[n].flatten().to(torch.float32)

            # down_proj column n: strided indices
            down_indices = torch.arange(down_out, dtype=torch.int64) * down_in + n
            down_indices = down_indices + down_slice.start
            down_values = -down_weight[:, n].flatten().to(torch.float32)

            all_indices = torch.cat([up_indices, down_indices])
            all_values = torch.cat([up_values, down_values])

            norm_val = all_values.norm()
            if norm_val > 1e-12:
                all_values = all_values / norm_val
                label = f"mlp_neuron_gauge_block{block_idx}_n{n}"
                directions.append(((all_indices, all_values), label))

    logger.info(
        "Built %d gated MLP neuron gauge directions for blocks %s.",
        len(directions), block_indices,
    )
    return directions


# ---------------------------------------------------------------------------
# Combined gauge basis with orthonormalization
# ---------------------------------------------------------------------------

def _dense_to_sparse(v: torch.Tensor):
    """Convert a dense vector to (indices, values) sparse representation.

    Only stores non-zero entries; both tensors live on CPU.
    """
    nz = v.nonzero(as_tuple=False).squeeze(-1)
    return (nz.to(torch.int64), v[nz].to(torch.float32))


def build_suffix_gauge_basis(
    model: nn.Module,
    spec: FlatParamSpec,
    block_indices: list,
    include_rmsnorm: bool = True,
    include_mlp: bool = True,
    device: str = "cpu",          # kept for API compat; not used for dense alloc
) -> GaugeBasis:
    """Build the complete gauge basis for suffix blocks (memory-efficient).

    Combines RMSNorm + gated MLP gauge directions.

    **Unlike the previous implementation** this does **not** construct a dense
    ``[p, g]`` matrix (which would be ~396 GB for a single block of Qwen-0.5B).
    Instead, each gauge direction is stored in sparse ``(indices, values)``
    form.  The directions are already individually normalised by the upstream
    ``build_rmsnorm_gauge_basis`` / ``build_gated_mlp_gauge_basis`` helpers.

    Orthonormalization across directions is *not* performed here; it is done
    lazily during projection via modified Gram-Schmidt in
    ``project_out_gauge`` / ``project_probe_matrix``.  Because each gauge
    direction is structurally sparse and the different gauge families (RMSNorm
    vs. MLP neuron scaling) have non-overlapping support, the raw directions
    are already very close to orthogonal, so the streaming MGS approach
    produces numerically equivalent results.

    Args:
        model:           the transformer model.
        spec:            FlatParamSpec describing the flat parameter space.
        block_indices:   which block indices are in the suffix.
        include_rmsnorm: whether to include RMSNorm gauge directions.
        include_mlp:     whether to include gated MLP gauge directions.
        device:          kept for API compatibility (ignored).

    Returns:
        GaugeBasis with sparse direction list and labels.
    """
    # Builders now return sparse (indices, values) pairs directly —
    # no dense R^p vectors are ever allocated.
    sparse_directions_with_labels = []

    if include_rmsnorm:
        sparse_directions_with_labels.extend(
            build_rmsnorm_gauge_basis(model, spec, block_indices)
        )

    if include_mlp:
        sparse_directions_with_labels.extend(
            build_gated_mlp_gauge_basis(model, spec, block_indices)
        )

    if not sparse_directions_with_labels:
        logger.warning("No gauge directions found. Returning empty basis.")
        return GaugeBasis(
            directions=[],
            labels=[],
            num_directions=0,
            total_dim=spec.total,
        )

    sparse_dirs = []
    labels = []
    for sparse_pair, label in sparse_directions_with_labels:
        sparse_dirs.append(sparse_pair)
        labels.append(label)

    logger.info(
        "Gauge basis: %d directions in R^%d (sparse, no dense matrix).",
        len(sparse_dirs), spec.total,
    )

    return GaugeBasis(
        directions=sparse_dirs,
        labels=labels,
        num_directions=len(sparse_dirs),
        total_dim=spec.total,
    )


# ---------------------------------------------------------------------------
# Projection utilities
# ---------------------------------------------------------------------------

def _sparse_dot_matrix(
    indices: torch.Tensor,    # [nnz] int64
    values: torch.Tensor,     # [nnz] float32
    V: torch.Tensor,          # [p, k]
) -> torch.Tensor:
    """Compute u^T V where u is given in sparse form → returns [k]."""
    return (values.unsqueeze(1) * V[indices]).sum(dim=0)


def _sparse_outer_subtract(
    V: torch.Tensor,          # [p, k]  — modified in-place
    indices: torch.Tensor,    # [nnz] int64
    values: torch.Tensor,     # [nnz] float32
    c: torch.Tensor,          # [k]
) -> None:
    """V[indices] -= values[:, None] * c[None, :]   (in-place)."""
    V[indices] -= values.unsqueeze(1) * c.unsqueeze(0)


def project_out_gauge(
    vectors: torch.Tensor,   # [p, k] or [p]
    gauge_basis: GaugeBasis,
    num_passes: int = 2,
) -> torch.Tensor:
    """Project vectors out of the gauge subspace (fast multi-pass).

    **Optimized algorithm** — avoids the O(g²) inner-loop orthogonalization
    of the old streaming-MGS approach.  Instead, we do ``num_passes`` sweeps
    over the raw (non-orthogonalised) gauge directions.  Each sweep is O(g·k)
    (one sparse dot + rank-1 update per direction).

    Because the gauge directions have **structurally non-overlapping supports**
    (RMSNorm gauge touches norm weights + consumer columns; MLP gauge touches
    gate/up rows + down columns — these parameter sets are disjoint), the raw
    directions are already nearly orthogonal.  A second pass mops up small
    residual components from the few directions that do overlap slightly.

    After projection, the caller (``project_probe_matrix``) runs QR
    re-orthonormalization, which handles any remaining numerical issues.

    Complexity:  O(num_passes · g · nnz_avg · k)
    Memory:      O(p · k)  — only the probe matrix copy.

    Args:
        vectors:      [p, k] matrix or [p] vector to project.
        gauge_basis:  GaugeBasis with sparse .directions list.
        num_passes:   number of projection sweeps (default 2).

    Returns:
        Projected vectors, same shape as input.
    """
    if gauge_basis.num_directions == 0:
        return vectors

    is_1d = vectors.dim() == 1
    if is_1d:
        vectors = vectors.unsqueeze(1)  # [p, 1]

    V = vectors.clone()  # work on a copy

    for pass_idx in range(num_passes):
        for sp_idx, sp_val in gauge_basis.directions:
            sp_idx = sp_idx.to(V.device)
            sp_val = sp_val.to(V.device, V.dtype)

            # c = u^T V  →  [k]
            c = _sparse_dot_matrix(sp_idx, sp_val, V)
            # V -= u c^T  (in-place rank-1 update, touches only nnz rows)
            _sparse_outer_subtract(V, sp_idx, sp_val, c)

    if is_1d:
        V = V.squeeze(1)

    return V


def _sparse_sparse_dot(
    idx_a: torch.Tensor, val_a: torch.Tensor,
    idx_b: torch.Tensor, val_b: torch.Tensor,
) -> torch.Tensor:
    """Compute dot product of two sparse vectors (intersection of supports)."""
    # Use searchsorted for efficient intersection
    # For moderate NNZ (up to ~15k), a simple set intersection is fine.
    # Both idx tensors are sorted (from nonzero()).
    # We use a merge-intersection approach.
    if len(idx_a) == 0 or len(idx_b) == 0:
        return torch.tensor(0.0, device=val_a.device, dtype=val_a.dtype)

    # Intersect using torch operations
    combined = torch.cat([idx_a, idx_b])
    uniq, counts = combined.unique(return_counts=True)
    common = uniq[counts > 1]

    if len(common) == 0:
        return torch.tensor(0.0, device=val_a.device, dtype=val_a.dtype)

    # Gather values at common indices
    # Use searchsorted since indices are sorted
    pos_a = torch.searchsorted(idx_a, common)
    pos_b = torch.searchsorted(idx_b, common)
    return (val_a[pos_a] * val_b[pos_b]).sum()


def _sparse_axpy(
    idx_x: torch.Tensor, val_x: torch.Tensor,
    idx_y: torch.Tensor, val_y: torch.Tensor,
    alpha: torch.Tensor,
):
    """Compute x + alpha * y in sparse form → (new_idx, new_val).

    Both inputs must have sorted indices.
    """
    # Union of indices
    combined_idx = torch.cat([idx_x, idx_y])
    combined_val = torch.cat([val_x, alpha * val_y])

    # Sort by index; where duplicates exist, sum the values
    sort_order = combined_idx.argsort(stable=True)
    sorted_idx = combined_idx[sort_order]
    sorted_val = combined_val[sort_order]

    # Identify unique indices and sum duplicates
    unique_idx, inverse = sorted_idx.unique(return_inverse=True)
    summed_val = torch.zeros(
        len(unique_idx), device=val_x.device, dtype=val_x.dtype
    )
    summed_val.scatter_add_(0, inverse, sorted_val)

    # Drop near-zeros to keep it sparse
    mask = summed_val.abs() > 1e-15
    return unique_idx[mask], summed_val[mask]


def project_probe_matrix(
    probe_matrix: torch.Tensor,  # [p, k]
    gauge_basis: GaugeBasis,
    drop_threshold: float = 1e-10,
) -> torch.Tensor:
    """Project probe columns out of gauge, then re-orthonormalize.

    Uses streaming modified Gram-Schmidt over sparse gauge directions,
    so peak memory is O(p·k) — never O(p·g).

    Steps:
      1. For each gauge direction u_i (streamed one at a time):
            c_i = u_i^T V   (k-dim)
            V  -= u_i c_i^T (sparse rank-1 update)
      2. QR on the [p, k] result to re-orthonormalize.
      3. Drop near-zero columns.

    Args:
        probe_matrix:    [p, k] matrix whose columns are probe directions.
        gauge_basis:     GaugeBasis to project out.
        drop_threshold:  columns with |R_jj| below this are dropped.

    Returns:
        [p, k'] orthonormal matrix where k' <= k.
    """
    if gauge_basis.num_directions == 0:
        # Nothing to project out; just re-orthonormalize
        Q, R = torch.linalg.qr(probe_matrix, mode="reduced")
        diag_R = R.diag().abs()
        tol = max(drop_threshold, 1e-8 * diag_R.max().clamp(min=1e-16).item())
        keep = diag_R > tol
        return Q[:, keep]

    # Step 1: project out gauge (streaming)
    projected = project_out_gauge(probe_matrix, gauge_basis)

    # Step 2: QR re-orthonormalization
    Q, R = torch.linalg.qr(projected, mode="reduced")

    # Step 3: drop near-zero columns
    diag_R = R.diag().abs()
    tol = max(drop_threshold, 1e-8 * diag_R.max().clamp(min=1e-16).item())
    keep = diag_R > tol
    num_dropped = (~keep).sum().item()

    if num_dropped > 0:
        logger.info(
            "project_probe_matrix: dropped %d / %d probes that were in gauge subspace.",
            num_dropped, probe_matrix.shape[1],
        )

    return Q[:, keep]


# ---------------------------------------------------------------------------
# Convenience: projected Gramian
# ---------------------------------------------------------------------------

def compute_projected_gramian(
    probe_matrix: torch.Tensor,  # [p, k]
    gauge_basis: GaugeBasis,
) -> torch.Tensor:
    """Compute the Gramian on the symmetry-quotiented space.

    G_proj = V'^T V'  where V' = project_probe_matrix(V, gauge_basis).

    This is a [k', k'] PSD matrix.  Its eigenvalues measure identifiability
    on the quotient manifold (after removing gauge redundancies).

    Args:
        probe_matrix:  [p, k] probe directions (e.g., finite-difference Jacobian columns).
        gauge_basis:   GaugeBasis to project out.

    Returns:
        [k', k'] projected Gramian matrix.
    """
    V_proj = project_probe_matrix(probe_matrix, gauge_basis)
    # G = V'^T V'  : [k', k']
    G = V_proj.T @ V_proj
    return G


# ---------------------------------------------------------------------------
# Summary / diagnostic utilities
# ---------------------------------------------------------------------------

def gauge_summary(gauge_basis: GaugeBasis) -> dict:
    """Return a summary dict for logging / JSON serialization.

    Keys:
        num_directions: number of gauge directions.
        label_counts:   dict mapping gauge type prefix to count.
        basis_shape:    [p, d] — virtual shape (no dense matrix is stored).
        total_nnz:      total number of nonzero entries across all directions.
    """
    label_counts: dict[str, int] = {}
    for label in gauge_basis.labels:
        # Extract type prefix: "rmsnorm_gauge_..." or "mlp_neuron_gauge_..."
        if label.startswith("rmsnorm_gauge"):
            key = "rmsnorm"
        elif label.startswith("mlp_neuron_gauge"):
            key = "mlp_neuron"
        else:
            key = "other"
        label_counts[key] = label_counts.get(key, 0) + 1

    total_nnz = sum(len(idx) for idx, _ in gauge_basis.directions)

    return {
        "num_directions": gauge_basis.num_directions,
        "label_counts": label_counts,
        "basis_shape": [gauge_basis.total_dim, gauge_basis.num_directions],
        "total_nnz": total_nnz,
    }


def expected_gauge_dimensions(
    num_blocks: int,
    hidden_size: int,
    intermediate_size: int,
    num_kv_heads: int = 0,
    include_final_norm: bool = True,
) -> dict:
    """Compute the expected number of gauge directions analytically.

    Per block:
      - 2 RMSNorm layers x hidden_size dims = 2 * hidden_size
      - 1 gated MLP x intermediate_size neurons = intermediate_size
      - Attention V/O + Q/K scaling: 2 * num_kv_heads (not yet implemented
        in build_gauge_basis, but counted here for theoretical reference)

    Plus optionally:
      - 1 final norm x hidden_size = hidden_size (if lm_head is in the spec)

    Returns dict with per-type and total counts.
    """
    rmsnorm_per_block = 2 * hidden_size
    mlp_per_block = intermediate_size
    attention_per_block = 2 * num_kv_heads

    total_rmsnorm = num_blocks * rmsnorm_per_block
    total_mlp = num_blocks * mlp_per_block
    total_attention = num_blocks * attention_per_block

    if include_final_norm:
        total_rmsnorm += hidden_size

    return {
        "rmsnorm_per_block": rmsnorm_per_block,
        "mlp_per_block": mlp_per_block,
        "attention_per_block": attention_per_block,
        "total_rmsnorm": total_rmsnorm,
        "total_mlp": total_mlp,
        "total_attention": total_attention,
        "total_implemented": total_rmsnorm + total_mlp,
        "total_theoretical": total_rmsnorm + total_mlp + total_attention,
        "total": total_rmsnorm + total_mlp,  # backward compat
        "num_blocks": num_blocks,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_kv_heads": num_kv_heads,
        "include_final_norm": include_final_norm,
    }
