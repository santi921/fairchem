from __future__ import annotations

import numpy as np
import torch
from torch_scatter import scatter


def potential_full_from_edge_inds(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    q: torch.Tensor,
    sigma: float = 1.0,
    epsilon: float = 1e-6,
    twopi: float = 2.0 * np.pi,
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
    Returns:
        potential_dict: dictionary of potential energy for each atom
    """

    # yields list of interactions [source, target]
    j, i = edge_index
    distance_vec = pos[j] - pos[i]
    # red to [n_interactions, 1]
    edge_dist = distance_vec.norm(dim=-1)

    edge_dist_transformed = (1.0 / (edge_dist + epsilon)) / twopi / 2.0
    convergence_func = torch.special.erf(edge_dist / sigma / (2.0**0.5))

    q_source = q[i].view(-1)
    q_target = q[j].view(-1)
    pairwise_potential = q_source * q_target * edge_dist_transformed * convergence_func

    results = scatter(pairwise_potential, i, dim=0, dim_size=q.size(0), reduce="sum")

    return results


def heisenberg_potential_full_from_edge_inds(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    q: torch.Tensor,
    nn: torch.nn.Module,
    sigma: float = 1.0,
):
    """
    Get the potential energy for each atom in the batch.
    Takes:
        pos: position matrix of shape (n_atoms, 3)
        edge_index: edge index of shape (2, n_edges)
        q: charge vector of shape (n_atoms, 2)
        nn: neural network to calculate the coupling term
        sigma: sigma parameter for the error function
        epsilon: epsilon parameter for the error function
    Returns:
        potential_dict: dictionary of potential energy for each atom
    """

    j, i = edge_index
    distance_vec = pos[j] - pos[i]
    edge_dist = distance_vec.norm(dim=-1).reshape(-1, 1)
    edge_dist.requires_grad_(True)

    convergence_func = torch.special.erf(edge_dist / sigma / (2.0**0.5)).reshape(-1, 1)
    coupling = nn(edge_dist)

    q_source = q[i]
    q_target = q[j]
    pairwise_potential = q_source * q_target * coupling * convergence_func

    results = scatter(
        pairwise_potential, i, dim=0, dim_size=q.size(0), reduce="sum"
    ).sum(dim=1)

    return results
