"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import torch

from fairchem.core.models.les.util import grad

twopi = 2.0 * np.pi


def potential_full_from_edge_inds(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    q: torch.Tensor,
    sigma: float = 1.0,
    epsilon: float = 1e-6,
    epsilon_factor_les: float = 1.0,  # \epsilon_infty
    twopi: float = 2.0 * np.pi,
    return_bec: bool = False,
    batch: torch.Tensor | None = None,
    conv_function_tf: bool = False,
):
    """
    Get the potential energy for each atom in the batch.
    Takes:
        pos: position matrix of shape (n_atoms, 3)
        edge_index: edge index of shape (2, n_edges)
        q: charge vector of shape (n_atoms, 1)
        radius_lr: cutoff radius for long-range interactions
        sigma: sigma parameter for the error function
        epsilon: epsilon parameter for the error function
        twopi: 2 * pi
        max_num_neighbors: maximum number of neighbors for each atom
        batch: batch vector of shape (n_atoms,)
    Returns:
        potential_dict: dictionary of potential energy for each atom
    """

    # yields list of interactions [source, target]
    results = {}
    n, d = pos.shape
    assert d == 3, "r dimension error"
    assert n == q.size(0), "q dimension error"

    if batch is None:
        batch = torch.zeros(n, dtype=torch.int64, device=pos.device)

    unique_batches = torch.unique(batch)  # Get unique batch indices

    if return_bec:
        # Ensure pos has requires_grad=True
        if not pos.requires_grad:
            pos.requires_grad_(True)

        if not q.requires_grad:
            q.requires_grad_(True)

        normalization_factor = epsilon_factor_les**0.5
        n, d = pos.shape
        assert d == 3, "r dimension error"
        assert n == q.size(0), "q dimension error"
        all_P = []
        all_phases = []
        unique_batches = torch.unique(batch)  # Get unique batch indices

        for i in unique_batches:
            mask = batch == i  # Create a mask for the i-th configuration

            r_now, q_now = pos[mask], q[mask].reshape(-1, 1)  # [n_atoms, 1]

            q_now = q_now - torch.mean(q_now, dim=0, keepdim=True)
            polarization = torch.sum(q_now * r_now, dim=0)
            phase = torch.ones_like(r_now, dtype=torch.complex64)

            all_P.append(polarization * normalization_factor)
            all_phases.append(phase)

        P = torch.stack(all_P, dim=0)
        phases = torch.cat(all_phases, dim=0)

        # Ensure P has requires_grad=True
        if not P.requires_grad:
            P.requires_grad_(True)

        # grad() returns [n_nodes, 3_dr, dim_y]; BEC needs P on first index
        bec_complex = grad(y=P, x=pos).transpose(1, 2).contiguous()

        # dephase — phases aligned with P direction (dim=1)
        result = bec_complex * phases.unsqueeze(2).conj()
        result_bec = result.real
        results["bec"] = result_bec

    j, i = edge_index
    distance_vec = pos[j] - pos[i]
    edge_dist = distance_vec.norm(dim=-1)
    edge_dist_transformed = (1.0 / (edge_dist + epsilon)) / twopi / 2.0

    q_source = q[i].view(-1)
    q_target = q[j].view(-1)
    pairwise_potential = q_source * q_target * edge_dist_transformed

    if conv_function_tf:
        convergence_func = torch.special.erf(edge_dist / sigma / (2.0**0.5))
        pairwise_potential *= convergence_func

    # remove diagonal elements
    pairwise_potential = pairwise_potential * (i != j).float()
    norm_factor = 90.0474

    potential = torch.zeros(
        q.size(0), device=pairwise_potential.device, dtype=pairwise_potential.dtype
    )
    potential.scatter_add_(0, i, pairwise_potential)
    results["potential"] = potential * norm_factor

    return results


def _generate_ewald_kvectors(
    cell_b: torch.Tensor,
    G_b: torch.Tensor,
    dl: float,
    k_sq_max: float,
    device: torch.device,
):
    """
    Generate filtered k-vectors for one batch element.

    Returns kvec_final, k_sq_final, factors (symmetry weights).
    """
    norms = torch.norm(cell_b, dim=1)
    Nk_b = torch.clamp(torch.floor(norms / dl).int(), min=1)

    n1_range = torch.arange(-Nk_b[0], Nk_b[0] + 1, device=device, dtype=G_b.dtype)
    n2_range = torch.arange(-Nk_b[1], Nk_b[1] + 1, device=device, dtype=G_b.dtype)
    n3_range = torch.arange(-Nk_b[2], Nk_b[2] + 1, device=device, dtype=G_b.dtype)

    n1_grid, n2_grid, n3_grid = torch.meshgrid(
        n1_range, n2_range, n3_range, indexing="ij"
    )
    nvec = torch.stack([n1_grid.flatten(), n2_grid.flatten(), n3_grid.flatten()], dim=1)
    kvec = nvec @ G_b
    k_sq = (kvec**2).sum(dim=1)

    # Filter: nonzero and within cutoff
    valid_mask = (k_sq > 0) & (k_sq <= k_sq_max)
    nvec = nvec[valid_mask]
    kvec = kvec[valid_mask]
    k_sq = k_sq[valid_mask]

    if kvec.numel() == 0:
        return None, None, None

    # Hemisphere masking: keep only k-vectors with positive first nonzero index
    non_zero_mask = nvec != 0
    has_non_zero = non_zero_mask.any(dim=1)
    first_non_zero_idx = torch.argmax(non_zero_mask.float(), dim=1)
    sign = torch.gather(nvec, 1, first_non_zero_idx.unsqueeze(1)).squeeze(1)
    hemisphere_mask = (sign > 0) | ~has_non_zero

    nvec = nvec[hemisphere_mask]
    kvec = kvec[hemisphere_mask]
    k_sq = k_sq[hemisphere_mask]

    if kvec.numel() == 0:
        return None, None, None

    # Symmetry factors: origin gets 1.0, all others 2.0
    is_origin = (nvec == 0).all(dim=1)
    factors = torch.where(is_origin, 1.0, 2.0)

    return kvec, k_sq, factors


def potential_full_ewald_batched(
    pos: torch.Tensor,
    q: torch.Tensor,
    cell: torch.Tensor,
    dl: float = 2.0,
    sigma: float = 1.0,
    epsilon: float = 1e-10,
    twopi: float = 2.0 * np.pi,
    return_bec: bool = False,
    batch: torch.Tensor | None = None,
    k_chunk_size: int = 50000,
):
    """
    Get the potential energy for each atom in the batch using Ewald summation.

    Uses chunked k-vector processing to limit peak memory usage.

    Args:
        pos: position matrix of shape (n_atoms, 3)
        q: charge vector of shape (n_atoms, 1)
        cell: cell matrix of shape (batch_size, 3, 3)
        dl: grid resolution for k-space sampling
        sigma: Gaussian width for Ewald splitting
        epsilon: small value to avoid division by zero
        twopi: 2 * pi
        return_bec: whether to return Born effective charges (unused)
        batch: batch vector of shape (n_atoms,)
        k_chunk_size: max k-vectors processed at once (controls memory)

    Returns:
        potential_dict: dictionary with "potential" key
    """
    device = pos.device
    sigma_sq_half = sigma**2 / 2.0
    k_sq_max = (twopi / dl) ** 2
    norm_factor = 90.0474

    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.int64, device=device)

    cell_inv = torch.linalg.inv(cell)
    G = 2 * torch.pi * cell_inv.transpose(-2, -1)

    unique_batches = torch.unique(batch)
    result_potentials = torch.zeros(pos.shape[0], device=device)

    for b_idx in unique_batches:
        atom_mask = batch == b_idx
        pos_b = pos[atom_mask]
        q_b = q[atom_mask]

        if q_b.dim() == 3:
            q_b = q_b.squeeze(-1)

        kvec_final, k_sq_final, factors = _generate_ewald_kvectors(
            cell[b_idx], G[b_idx], dl, k_sq_max, device
        )

        if kvec_final is None:
            continue

        # Precompute kfac = exp(-sigma^2/2 * k^2) / k^2
        kfac = torch.exp(-sigma_sq_half * k_sq_final)
        kfac.div_(k_sq_final + epsilon)
        weighted = factors * kfac  # [n_k]

        # Chunked structure factor: accumulate reciprocal energy over k-chunks
        # Peak memory per chunk: [n_atoms_b, chunk_size] * 2 tensors
        n_k = kvec_final.shape[0]
        recip_energy = torch.tensor(0.0, device=device)

        for start in range(0, n_k, k_chunk_size):
            end = min(start + k_chunk_size, n_k)
            k_dot_r = pos_b @ kvec_final[start:end].T  # [n_atoms, chunk]

            cos_kr = torch.cos(k_dot_r)
            sin_kr = torch.sin(k_dot_r)
            cos_kr *= q_b
            sin_kr *= q_b

            S_real = cos_kr.sum(dim=0)
            S_imag = sin_kr.sum(dim=0)
            S_sq = S_real.pow_(2) + S_imag.pow_(2)

            recip_energy = recip_energy + (weighted[start:end] * S_sq).sum()

        volume = torch.det(cell[b_idx])
        recip_energy = recip_energy / volume

        self_energy = (q_b.view(-1) ** 2).sum() / (sigma * (twopi) ** 1.5)
        batch_energy = (recip_energy - self_energy) * norm_factor

        n_atoms_b = atom_mask.sum()
        result_potentials[atom_mask] = batch_energy / n_atoms_b

    results = {"potential": result_potentials}
    return results


def heisenberg_potential_full_from_edge_inds(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    q: torch.Tensor,
    nn: torch.nn.Module,
    exchange_type: str = "xy",
):
    """
    Compute spin-spin coupling energy per atom using a learned coupling J(r).

    Three exchange types are supported, all using collinear alpha/beta
    spin channels where S_z = (alpha - beta) / 2:

    - ``"heisenberg"``: Full Heisenberg S_i . S_j, proportional to
      ``(alpha_i*alpha_j + beta_i*beta_j - alpha_i*beta_j - beta_i*alpha_j)``
    - ``"ising"``: Longitudinal S_zi * S_zj, proportional to
      ``(alpha_i - beta_i) * (alpha_j - beta_j)``
    - ``"xy"``: Transverse (flip-flop) S_ix*S_jx + S_iy*S_jy, proportional to
      ``(alpha_i*beta_j + beta_i*alpha_j)``

    Note: Since J(r) is learned by the neural network, sign and overall
    scale are absorbed into the NN weights.

    Args:
        pos: position matrix [n_atoms, 3]
        edge_index: edge index [2, n_edges]
        q: spin-channel charges [n_atoms, 2] (alpha, beta)
        nn: neural network mapping distance -> coupling strength
        exchange_type: one of "heisenberg", "ising", "xy"

    Returns:
        Per-atom spin coupling energy [n_atoms]
    """
    j, i = edge_index
    distance_vec = pos[j] - pos[i]
    edge_dist = distance_vec.norm(dim=-1).reshape(-1, 1)
    edge_dist.requires_grad_(True)

    coupling = nn(edge_dist)
    alpha_i = q[i][:, 0]
    beta_i = q[i][:, 1]
    alpha_j = q[j][:, 0]
    beta_j = q[j][:, 1]

    if exchange_type == "xy":
        # Transverse (flip-flop): alpha_i*beta_j + beta_i*alpha_j
        pairwise_potential = (beta_i * alpha_j + alpha_i * beta_j) * coupling
    elif exchange_type == "ising":
        # Longitudinal: (alpha_i - beta_i) * (alpha_j - beta_j)
        pairwise_potential = ((alpha_i - beta_i) * (alpha_j - beta_j)) * coupling
    elif exchange_type == "heisenberg":
        # Full Heisenberg: S_i . S_j = Ising + XY
        # Ising: (a_i - b_i)(a_j - b_j) = aa + bb - ab - ba
        # XY:    a_i*b_j + b_i*a_j
        # Sum:   aa + bb (cross terms cancel)
        pairwise_potential = (alpha_i * alpha_j + beta_i * beta_j) * coupling
    else:
        raise ValueError(
            f"Unknown exchange_type '{exchange_type}'. "
            "Must be 'heisenberg', 'ising', or 'xy'."
        )

    out = torch.zeros(
        q.size(0),
        pairwise_potential.size(1),
        device=pairwise_potential.device,
        dtype=pairwise_potential.dtype,
    )
    out.scatter_add_(
        0, i.unsqueeze(1).expand_as(pairwise_potential), pairwise_potential
    )
    results = out.sum(dim=1)

    return results


def batch_spin_charge_renormalization(
    charges_raw: torch.Tensor,  # [n_atoms, 2],
    q_total: torch.Tensor,  # [n_batches]
    s_total: torch.Tensor,  # [n_batches]
    epsilon: float = 1e-6,
    batch: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
):
    """
    Enforce per-batch charge and spin conservation via additive correction.

    Distributes the residual (target - predicted) uniformly across atoms
    in each batch using optional per-atom weights.

    Note on locality: this correction is global within each batch -- a
    charge error on one atom is corrected by adjusting all atoms. This is
    physically sound since charge conservation is a global constraint, and
    the correction should be small when the NN is well-trained.

    Args:
        charges_raw: raw charges [n_atoms, 2] (alpha, beta channels)
        q_total: target total charge per batch [n_batches]
        s_total: target total spin per batch [n_batches]
        epsilon: regularization for weight normalization
        batch: batch indices [n_atoms]
        weights: optional per-atom weights [n_atoms, 2]

    Returns:
        Renormalized charges [n_atoms, 2] satisfying both constraints.
    """
    device = charges_raw.device
    num_batches = q_total.shape[0]

    alpha = charges_raw[:, 0]
    beta = charges_raw[:, 1]
    # Default weights: uniform per channel
    if weights is None:
        weights = torch.ones_like(charges_raw)
    w_alpha = weights[:, 0]
    w_beta = weights[:, 1]

    # Compute sums per batch for predicted alpha+beta and alpha-beta
    alpha_sum = torch.zeros(num_batches, device=device).scatter_add_(0, batch, alpha)
    beta_sum = torch.zeros(num_batches, device=device).scatter_add_(0, batch, beta)

    q_sum = alpha_sum + beta_sum  # total charge
    s_sum = alpha_sum - beta_sum  # total spin

    # Compute per-batch residuals
    dq = q_total - q_sum
    ds = s_total - s_sum

    # Normalize weights per batch
    w_alpha_sum = torch.zeros(num_batches, device=device).scatter_add_(
        0, batch, w_alpha
    )
    w_beta_sum = torch.zeros(num_batches, device=device).scatter_add_(0, batch, w_beta)
    w_alpha_norm = w_alpha / (w_alpha_sum[batch] + epsilon)
    w_beta_norm = w_beta / (w_beta_sum[batch] + epsilon)

    # Residuals per batch for each constraint
    dq_atom = dq[batch]
    ds_atom = ds[batch]

    delta_alpha = 0.5 * (dq_atom * w_alpha_norm + ds_atom * w_alpha_norm)
    delta_beta = 0.5 * (dq_atom * w_beta_norm - ds_atom * w_beta_norm)

    alpha_corr = alpha + delta_alpha
    beta_corr = beta + delta_beta

    return torch.stack([alpha_corr, beta_corr], dim=-1)
