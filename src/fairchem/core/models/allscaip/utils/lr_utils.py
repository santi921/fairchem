"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import torch

from fairchem.core.models.escaip.utils.graph_utils import compilable_scatter

# Coulomb constant in eV*Angstrom (14.3996 eV*A / (4*pi*epsilon_0) -> 90.0474)
COULOMB_CONSTANT = 90.0474


def compilable_scatter_on_dictionary(
    src: dict[str, torch.Tensor],
    index: torch.Tensor,
    dim_size: int,
    dim: int = 0,
    reduce: str = "sum",
) -> dict[str, torch.Tensor]:
    """
    Scatter function for dictionary of tensors with compile support.
    """
    out = {}
    for key in src:
        out[key] = compilable_scatter(src[key], index, dim_size, dim=dim, reduce=reduce)
    return out


def coulomb_energy_from_src_index(
    q: torch.Tensor,
    src_index: torch.Tensor,
    dist_pairwise: torch.Tensor,
    eps: float = 1e-8,
    sigma: float = 1.0,
    epsilon: float = 1e-6,
    twopi: float = 2.0 * np.pi,
    use_convergence: bool = False,
) -> torch.Tensor:
    """
    Compute Coulomb energy per atom using src_index and dist_pairwise.

    Args:
        q: charges, shape (N,) or (N, 1)
        src_index: (2, N, max_neighbors), [0] is source, [1] is neighbor
        dist_pairwise: (N, N) pairwise distance matrix
        eps: threshold for masking zero-distance pairs
        sigma: width parameter for optional convergence function
        epsilon: shift for denominator to avoid singularity
        twopi: 2*pi scaling factor
        use_convergence: whether to apply erf convergence function

    Returns:
        Per-atom Coulomb energy, shape (N,)
    """
    q = q.squeeze(-1) if q.dim() > 1 else q

    src, nbr = src_index[0], src_index[1]
    rij = dist_pairwise[src, nbr]
    qi = q[src]
    qj = q[nbr]

    mask = (src != nbr) & (rij > eps)

    if use_convergence:
        convergence_func = torch.special.erf(rij / (sigma * 1.4142135623730951))
    else:
        convergence_func = torch.ones_like(rij)

    coulomb_term = (qi * qj) / (rij + epsilon) / twopi / 2.0 * convergence_func
    e_ij = torch.where(mask, coulomb_term, torch.zeros_like(coulomb_term))

    energy = e_ij.sum(dim=-1) * COULOMB_CONSTANT
    return energy


def heisenberg_energy_from_src_index(
    q: torch.Tensor,
    src_index: torch.Tensor,
    j_coupling_nn: torch.nn.Module,
    dist_pairwise: torch.Tensor,
    eps: float = 1e-8,
    exchange_type: str = "heisenberg",
) -> torch.Tensor:
    """
    Compute spin exchange energy per atom using a learned coupling J(r).

    Args:
        q: spin charges (N, 2) with alpha and beta channels
        src_index: (2, N, max_neighbors) source and neighbor indices
        j_coupling_nn: NN mapping distance -> coupling strength
        dist_pairwise: (N, N) pairwise distance matrix
        eps: threshold for masking self-interactions
        exchange_type: one of "heisenberg", "ising", "xy"

    Returns:
        Per-atom spin coupling energy, shape (N,)
    """
    src, nbr = src_index[0], src_index[1]
    rij = dist_pairwise[src, nbr]

    N, max_neighbors = rij.shape
    rij_flat = rij.contiguous().view(N * max_neighbors, 1)
    j_coupling_vals = j_coupling_nn(rij_flat).reshape(N, max_neighbors)

    qi = q[src]
    qj = q[nbr]

    qi_alpha = qi[:, :, 0]
    qi_beta = qi[:, :, 1]
    qj_alpha = qj[:, :, 0]
    qj_beta = qj[:, :, 1]

    mask = (src != nbr) & (rij > eps)

    if exchange_type == "heisenberg":
        spin_interaction = (qi_alpha * qj_alpha + qi_beta * qj_beta) * j_coupling_vals
    elif exchange_type == "ising":
        spin_interaction = (
            (qi_alpha - qi_beta) * (qj_alpha - qj_beta)
        ) * j_coupling_vals
    elif exchange_type == "xy":
        spin_interaction = (qi_alpha * qj_beta + qi_beta * qj_alpha) * j_coupling_vals
    else:
        raise ValueError(
            f"Unknown exchange_type '{exchange_type}'. "
            "Must be 'heisenberg', 'ising', or 'xy'."
        )

    e_ij = torch.where(mask, spin_interaction, torch.zeros_like(spin_interaction))
    energy = e_ij.sum(dim=1)
    return energy


def charge_renormalization(
    q: torch.Tensor,
    emb: dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Rescale predicted charges to match target total charge per graph.

    Args:
        q: predicted charges, shape (N,)
        emb: dictionary with "data" key containing GraphAttentionData
        eps: small value to avoid division by zero

    Returns:
        Rescaled charges, shape (N,)
    """
    num_nodes = emb["data"].num_nodes
    num_graphs = emb["data"].num_graphs
    node_batch = emb["data"].node_batch

    valid_charges = q[:num_nodes]
    valid_node_batch = node_batch[:num_nodes]
    target_charges = emb["data"].charge[:num_graphs]

    global_charges = compilable_scatter(
        valid_charges,
        index=valid_node_batch,
        dim_size=num_graphs,
        dim=0,
        reduce="sum",
    )

    rescale_factor = torch.where(
        torch.abs(global_charges) < eps,
        torch.ones_like(global_charges),
        target_charges / global_charges,
    )

    q[:num_nodes] = q[:num_nodes] * rescale_factor[valid_node_batch]
    return q


def charge_spin_renormalization(
    q: torch.Tensor,
    emb: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Rescale predicted charges to match both total charge and spin per graph.

    Uses additive correction distributed uniformly across atoms.

    Args:
        q: predicted charges, shape (N, 2) with alpha and beta channels
        emb: dictionary with "data" key containing GraphAttentionData
            (must have charge, spin, node_batch, num_nodes, num_graphs)

    Returns:
        Rescaled charges, shape (N, 2)
    """
    num_nodes = emb["data"].num_nodes
    num_graphs = emb["data"].num_graphs
    node_batch = emb["data"].node_batch

    results_tensor = torch.zeros_like(q)

    valid_charges = q[:num_nodes]
    valid_node_batch = node_batch[:num_nodes]
    target_charges = emb["data"].charge[:num_graphs]
    target_spins = emb["data"].spin[:num_graphs]

    alpha = valid_charges[:, 0]
    beta = valid_charges[:, 1]

    weights = torch.ones_like(valid_charges)
    w_alpha = weights[:, 0]
    w_beta = weights[:, 1]
    ones_arr = torch.ones_like(w_alpha)

    scatter_dict = compilable_scatter_on_dictionary(
        {
            "alpha": alpha,
            "beta": beta,
            "w_alpha": w_alpha,
            "w_beta": w_beta,
            "ones": ones_arr,
        },
        index=valid_node_batch,
        dim_size=num_graphs,
        dim=0,
        reduce="sum",
    )

    q_sum = scatter_dict["alpha"] + scatter_dict["beta"]
    s_sum = scatter_dict["alpha"] - scatter_dict["beta"]

    dq = target_charges - q_sum
    ds = target_spins - s_sum

    delta_alpha = 0.5 * (dq + ds)
    delta_beta = 0.5 * (dq - ds)

    delta_alpha_expanded = delta_alpha[valid_node_batch]
    delta_beta_expanded = delta_beta[valid_node_batch]

    delta_alpha_expanded = delta_alpha_expanded / (
        scatter_dict["w_alpha"][valid_node_batch] + 1e-8
    )
    delta_beta_expanded = delta_beta_expanded / (
        scatter_dict["w_beta"][valid_node_batch] + 1e-8
    )

    alpha_corr = alpha + delta_alpha_expanded
    beta_corr = beta + delta_beta_expanded

    results_tensor[:num_nodes, 0] = alpha_corr.squeeze(-1)
    results_tensor[:num_nodes, 1] = beta_corr.squeeze(-1)

    return results_tensor


def compute_pairwise_distances(
    pos: torch.Tensor,
    batch: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Compute pairwise distance matrix from positions.

    Only computes distances for the valid (non-padded) nodes.

    Args:
        pos: atom positions, shape (N_total, 3)
        batch: batch indices, shape (N_total,)
        num_nodes: number of real (non-padded) nodes

    Returns:
        Pairwise distance matrix, shape (N_total, N_total)
    """
    valid_pos = pos[:num_nodes]
    diff = valid_pos.unsqueeze(0) - valid_pos.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)

    n_total = pos.shape[0]
    if n_total > num_nodes:
        dist_padded = torch.zeros(n_total, n_total, device=pos.device, dtype=pos.dtype)
        dist_padded[:num_nodes, :num_nodes] = dist
        return dist_padded
    return dist
