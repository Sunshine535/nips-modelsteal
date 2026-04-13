"""Permutation-aware alignment for identifiability evaluation.

Transformer weights have internal symmetries — attention heads can be permuted,
FFN neurons can be permuted, and RMSNorm can absorb/emit scale factors. These
symmetries mean that two parameterizations can implement the exact same function
while having very different weight vectors.

This module implements Hungarian-algorithm-based alignment to canonicalize
these symmetries before computing cosine similarity.
"""

import logging

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def align_attention_heads(
    recovered_params: dict[str, torch.Tensor],
    teacher_params: dict[str, torch.Tensor],
    block_prefix: str,
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Align attention heads using Hungarian algorithm on Q||K||V||O similarity.

    For each head h, concatenate [Q_h, K_h, V_h, O_h^T] into a single vector,
    compute pairwise cosine similarity, and find optimal permutation.

    Returns: aligned copy of recovered_params (only attention weights modified).
    """
    proj_names = ["q_proj", "k_proj", "v_proj"]
    o_proj_name = "o_proj"

    def find_key(params, keyword):
        for k in params:
            if k.startswith(block_prefix) and keyword in k and "weight" in k:
                return k
        return None

    q_key = find_key(recovered_params, "q_proj")
    if q_key is None:
        return recovered_params

    k_key = find_key(recovered_params, "k_proj")
    v_key = find_key(recovered_params, "v_proj")
    o_key = find_key(recovered_params, "o_proj")

    if any(k is None for k in [k_key, v_key, o_key]):
        logger.warning("Missing attention proj keys for %s", block_prefix)
        return recovered_params

    def get_head_vectors(params, q_k, k_k, v_k, o_k, n_h, h_d):
        Q = params[q_k].float().view(n_h, h_d, -1)
        K = params[k_k].float()
        V = params[v_k].float()
        O = params[o_k].float()

        n_kv = K.shape[0] // h_d
        if n_kv < n_h:
            rep = n_h // n_kv
            K = K.unsqueeze(0).expand(rep, -1, -1).reshape(n_h * h_d, -1)
            V = V.unsqueeze(0).expand(rep, -1, -1).reshape(n_h * h_d, -1)
            n_kv = n_h

        K = K.view(n_kv, h_d, -1)
        V = V.view(n_kv, h_d, -1)
        O = O.view(-1, n_h, h_d)

        vectors = []
        for h in range(min(n_h, n_kv)):
            cat = torch.cat([
                Q[h].flatten(),
                K[h].flatten(),
                V[h].flatten(),
                O[:, h, :].flatten(),
            ])
            vectors.append(cat)
        return torch.stack(vectors)

    try:
        rec_vecs = get_head_vectors(
            recovered_params, q_key, k_key, v_key, o_key, num_heads, head_dim
        )
        tea_vecs = get_head_vectors(
            teacher_params, q_key, k_key, v_key, o_key, num_heads, head_dim
        )
    except Exception as e:
        logger.warning("Head vector extraction failed: %s", e)
        return recovered_params

    n = min(rec_vecs.shape[0], tea_vecs.shape[0])
    rec_vecs = rec_vecs[:n]
    tea_vecs = tea_vecs[:n]

    sim = F.cosine_similarity(
        rec_vecs.unsqueeze(1), tea_vecs.unsqueeze(0), dim=-1
    )
    cost = 1.0 - sim.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    perm = list(range(n))
    for r, c in zip(row_ind, col_ind):
        perm[r] = c

    aligned = dict(recovered_params)

    for proj_name in proj_names:
        key = find_key(recovered_params, proj_name)
        if key is None:
            continue
        w = recovered_params[key].float()
        n_actual = w.shape[0] // head_dim
        if n_actual != n:
            continue
        chunks = w.view(n, head_dim, -1)
        permuted = torch.stack([chunks[perm[i]] for i in range(n)])
        aligned[key] = permuted.view_as(w).to(recovered_params[key].dtype)

    if o_key:
        w = recovered_params[o_key].float()
        reshaped = w.view(-1, n, head_dim)
        permuted = torch.stack([reshaped[:, perm[i], :] for i in range(n)], dim=1)
        aligned[o_key] = permuted.view_as(w).to(recovered_params[o_key].dtype)

    return aligned


def align_ffn_neurons(
    recovered_params: dict[str, torch.Tensor],
    teacher_params: dict[str, torch.Tensor],
    block_prefix: str,
) -> dict[str, torch.Tensor]:
    """Align FFN neurons using Hungarian algorithm on (up||gate, down) similarity.

    For SwiGLU FFN: each neuron i is defined by (up_proj[i], gate_proj[i], down_proj[:, i]).
    """
    def find_key(params, keyword):
        for k in params:
            if k.startswith(block_prefix) and keyword in k and "weight" in k:
                return k
        return None

    up_key = find_key(recovered_params, "up_proj")
    gate_key = find_key(recovered_params, "gate_proj")
    down_key = find_key(recovered_params, "down_proj")

    if any(k is None for k in [up_key, gate_key, down_key]):
        return recovered_params

    def get_neuron_vectors(params, u_k, g_k, d_k):
        up = params[u_k].float()
        gate = params[g_k].float()
        down = params[d_k].float()
        n = up.shape[0]
        vecs = []
        for i in range(n):
            cat = torch.cat([up[i].flatten(), gate[i].flatten(), down[:, i].flatten()])
            vecs.append(cat)
        return torch.stack(vecs)

    try:
        rec_vecs = get_neuron_vectors(recovered_params, up_key, gate_key, down_key)
        tea_vecs = get_neuron_vectors(teacher_params, up_key, gate_key, down_key)
    except Exception as e:
        logger.warning("FFN neuron vector extraction failed: %s", e)
        return recovered_params

    n = min(rec_vecs.shape[0], tea_vecs.shape[0])
    rec_vecs = rec_vecs[:n]
    tea_vecs = tea_vecs[:n]

    sim = F.cosine_similarity(
        rec_vecs.unsqueeze(1), tea_vecs.unsqueeze(0), dim=-1
    )
    cost = 1.0 - sim.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    perm = list(range(n))
    for r, c in zip(row_ind, col_ind):
        perm[r] = c

    aligned = dict(recovered_params)

    for key in [up_key, gate_key]:
        w = recovered_params[key].float()
        if w.shape[0] == n:
            permuted = torch.stack([w[perm[i]] for i in range(n)])
            aligned[key] = permuted.to(recovered_params[key].dtype)

    if down_key:
        w = recovered_params[down_key].float()
        if w.shape[1] == n:
            permuted = torch.stack([w[:, perm[i]] for i in range(n)], dim=1)
            aligned[down_key] = permuted.to(recovered_params[down_key].dtype)

    return aligned


def remove_rmsnorm_scale(
    params: dict[str, torch.Tensor],
    block_prefix: str,
) -> dict[str, torch.Tensor]:
    """Normalize RMSNorm weights to unit scale for fair comparison."""
    normalized = dict(params)
    for key in params:
        if block_prefix in key and "layernorm" in key and "weight" in key:
            w = params[key].float()
            scale = w.norm() / (w.numel() ** 0.5)
            if scale > 1e-8:
                normalized[key] = (w / scale).to(params[key].dtype)
    return normalized


def full_block_alignment(
    recovered_params: dict[str, torch.Tensor],
    teacher_params: dict[str, torch.Tensor],
    block_prefix: str,
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Full permutation-aware alignment: heads + FFN neurons + RMSNorm."""
    aligned = align_attention_heads(
        recovered_params, teacher_params, block_prefix, num_heads, head_dim,
    )
    aligned = align_ffn_neurons(aligned, teacher_params, block_prefix)
    aligned = remove_rmsnorm_scale(aligned, block_prefix)
    teacher_norm = remove_rmsnorm_scale(teacher_params, block_prefix)
    return aligned


def compute_lm_head_aligned_cosine(
    recovered: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    lm_head_key: str = "lm_head.weight",
) -> dict[str, float]:
    """Compute lm_head cosine similarity with Procrustes rotation alignment.

    The lm_head weight W ∈ R^{V×d} has a GL(d) rotation gauge symmetry:
    if the last hidden state h is rotated by Q ∈ O(d), then W → W Q^T
    preserves the output logits z = W h.

    We find the optimal orthogonal Q* = argmin ||W_rec Q - W_tea||_F via
    Procrustes: Q* = V U^T where W_tea^T W_rec = U Σ V^T (SVD).

    Returns dict with keys: 'raw_cosine', 'aligned_cosine', 'procrustes_error'.
    """
    if lm_head_key not in recovered or lm_head_key not in teacher:
        logger.warning("lm_head key '%s' not found in params.", lm_head_key)
        return {}

    W_rec = recovered[lm_head_key].float()  # [V, d]
    W_tea = teacher[lm_head_key].float()    # [V, d]

    if W_rec.shape != W_tea.shape:
        logger.warning("lm_head shape mismatch: %s vs %s", W_rec.shape, W_tea.shape)
        return {}

    # Raw cosine (no alignment)
    raw_cos = F.cosine_similarity(
        W_rec.flatten().unsqueeze(0),
        W_tea.flatten().unsqueeze(0),
    ).item()

    # Procrustes rotation alignment: find Q* = argmin ||W_rec Q - W_tea||_F
    # Solution: Q* = V U^T where W_rec^T W_tea = U Σ V^T
    M = W_rec.T @ W_tea  # [d, d]
    try:
        U, S, Vh = torch.linalg.svd(M)
        Q_star = U @ Vh  # [d, d] orthogonal matrix

        # Check for reflection (det = -1) → flip sign of last column
        if torch.det(Q_star) < 0:
            U[:, -1] *= -1
            Q_star = U @ Vh

        W_rec_aligned = W_rec @ Q_star  # [V, d]

        aligned_cos = F.cosine_similarity(
            W_rec_aligned.flatten().unsqueeze(0),
            W_tea.flatten().unsqueeze(0),
        ).item()

        # Relative Frobenius error
        proc_err = (W_rec_aligned - W_tea).norm().item() / (W_tea.norm().item() + 1e-12)

        logger.info(
            "lm_head Procrustes alignment: raw_cos=%.4f, aligned_cos=%.4f, "
            "proc_err=%.4f, top singular values: %s",
            raw_cos, aligned_cos, proc_err,
            S[:5].tolist(),
        )

        return {
            "raw_cosine": raw_cos,
            "aligned_cosine": aligned_cos,
            "procrustes_error": proc_err,
            "top_singular_values": S[:10].tolist(),
        }
    except Exception as e:
        logger.warning("Procrustes SVD failed: %s", e)
        return {"raw_cosine": raw_cos}


def compute_aligned_cosine(
    recovered: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    block_prefix: str,
    num_heads: int,
    head_dim: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute cosine similarity before and after alignment.

    Returns:
        (unaligned_cosines, aligned_cosines) — both are dicts mapping param short name to cosine sim.
    """
    unaligned = {}
    for key in recovered:
        if not key.startswith(block_prefix) or key not in teacher:
            continue
        r = recovered[key].float().flatten()
        t = teacher[key].float().flatten()
        if r.shape != t.shape:
            continue
        short = key.replace(block_prefix, "").lstrip(".")
        unaligned[short] = F.cosine_similarity(
            r.unsqueeze(0), t.unsqueeze(0)
        ).item()

    aligned_params = full_block_alignment(
        recovered, teacher, block_prefix, num_heads, head_dim
    )

    aligned = {}
    for key in aligned_params:
        if not key.startswith(block_prefix) or key not in teacher:
            continue
        r = aligned_params[key].float().flatten()
        t = teacher[key].float().flatten()
        if r.shape != t.shape:
            continue
        short = key.replace(block_prefix, "").lstrip(".")
        aligned[short] = F.cosine_similarity(
            r.unsqueeze(0), t.unsqueeze(0)
        ).item()

    return unaligned, aligned
