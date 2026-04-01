"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.allscaip.utils.lr_utils import (
    charge_renormalization,
    charge_spin_renormalization,
    compilable_scatter_on_dictionary,
    compute_pairwise_distances,
    coulomb_energy_from_src_index,
    heisenberg_energy_from_src_index,
)

# ============================================================
# Helper fixtures
# ============================================================


def _make_src_index_and_dist(n_atoms=4, max_neighbors=3):
    """
    Create a simple padded src_index and pairwise distance matrix.

    Builds a linear chain with unit spacing.
    """
    pos = torch.zeros(n_atoms, 3)
    for i in range(n_atoms):
        pos[i, 0] = float(i)

    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    dist_pairwise = torch.norm(diff, dim=-1)

    src = torch.zeros(n_atoms, max_neighbors, dtype=torch.long)
    nbr = torch.zeros(n_atoms, max_neighbors, dtype=torch.long)

    for i in range(n_atoms):
        neighbors = [j for j in range(n_atoms) if j != i]
        neighbors = neighbors[:max_neighbors]
        for k, j in enumerate(neighbors):
            src[i, k] = i
            nbr[i, k] = j
        # pad remaining with self-loops (masked out by src != nbr)
        for k in range(len(neighbors), max_neighbors):
            src[i, k] = i
            nbr[i, k] = i

    src_index = torch.stack([src, nbr], dim=0)
    return src_index, dist_pairwise, pos


def _make_coupling_nn(hidden=16):
    """
    Simple coupling NN with ones initialization for deterministic tests.
    """
    nn_coupling = torch.nn.Sequential(
        torch.nn.Linear(1, hidden),
        torch.nn.Linear(hidden, 1),
    )
    nn_coupling[0].weight.data.fill_(1)
    nn_coupling[0].bias.data.fill_(0)
    nn_coupling[1].weight.data.fill_(1)
    nn_coupling[1].bias.data.fill_(0)
    return nn_coupling


class _MockGraphAttentionData:
    """
    Minimal mock for GraphAttentionData used by renormalization functions.
    """

    def __init__(self, num_nodes, num_graphs, node_batch, charge, spin=None):
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.node_batch = node_batch
        self.charge = charge
        self.spin = spin if spin is not None else torch.zeros(num_graphs)


# ============================================================
# Coulomb energy tests
# ============================================================


def test_coulomb_basic_shape():
    """
    Coulomb energy should return per-atom values with correct shape.
    """
    src_index, dist_pairwise, _ = _make_src_index_and_dist(n_atoms=4)
    q = torch.tensor([1.0, -1.0, 1.0, -1.0])

    energy = coulomb_energy_from_src_index(q, src_index, dist_pairwise)
    assert energy.shape == (4,), f"Expected shape (4,), got {energy.shape}"


def test_coulomb_zero_charges():
    """
    Zero charges should produce zero energy.
    """
    src_index, dist_pairwise, _ = _make_src_index_and_dist(n_atoms=3)
    q = torch.zeros(3)

    energy = coulomb_energy_from_src_index(q, src_index, dist_pairwise)
    assert torch.allclose(energy, torch.zeros(3), atol=1e-8)


def test_coulomb_opposite_charges_negative():
    """
    Opposite charges should produce negative Coulomb energy.
    """
    src_index, dist_pairwise, _ = _make_src_index_and_dist(n_atoms=2)
    q = torch.tensor([1.0, -1.0])

    energy = coulomb_energy_from_src_index(q, src_index, dist_pairwise)
    assert energy[0].item() < 0.0, "Expected negative for +/- pair"


def test_coulomb_same_charges_positive():
    """
    Same-sign charges should produce positive Coulomb energy.
    """
    src_index, dist_pairwise, _ = _make_src_index_and_dist(n_atoms=2)
    q = torch.tensor([1.0, 1.0])

    energy = coulomb_energy_from_src_index(q, src_index, dist_pairwise)
    assert energy[0].item() > 0.0, "Expected positive for +/+ pair"


def test_coulomb_charge_scaling():
    """
    Doubling charges should quadruple the energy (E ~ q1*q2).
    """
    src_index, dist_pairwise, _ = _make_src_index_and_dist(n_atoms=2)

    e1 = coulomb_energy_from_src_index(
        torch.tensor([1.0, 1.0]), src_index, dist_pairwise
    )
    e2 = coulomb_energy_from_src_index(
        torch.tensor([2.0, 2.0]), src_index, dist_pairwise
    )

    ratio = e2[0] / e1[0]
    assert torch.allclose(ratio, torch.tensor(4.0), atol=0.01)


def test_coulomb_convergence_reduces_magnitude():
    """
    The erf convergence function should reduce potential magnitude.
    """
    src_index, dist_pairwise, _ = _make_src_index_and_dist(n_atoms=2)
    q = torch.tensor([1.0, 1.0])

    e_no_conv = coulomb_energy_from_src_index(
        q, src_index, dist_pairwise, use_convergence=False
    )
    e_conv = coulomb_energy_from_src_index(
        q, src_index, dist_pairwise, use_convergence=True
    )

    assert torch.abs(e_conv[0]) <= torch.abs(e_no_conv[0])


# ============================================================
# Heisenberg energy tests
# ============================================================


def test_heisenberg_basic_shape():
    """
    Heisenberg energy should return per-atom values.
    """
    src_index, dist_pairwise, _ = _make_src_index_and_dist(n_atoms=3)
    q = torch.tensor([[0.8, 0.3], [0.4, 0.6], [0.5, 0.5]])
    nn_coupling = _make_coupling_nn()

    energy = heisenberg_energy_from_src_index(q, src_index, nn_coupling, dist_pairwise)
    assert energy.shape == (3,), f"Expected shape (3,), got {energy.shape}"


def test_heisenberg_nonzero():
    """
    Non-zero spins should produce non-zero Heisenberg energy.
    """
    src_index, dist_pairwise, _ = _make_src_index_and_dist(n_atoms=2)
    q = torch.tensor([[1.0, 0.5], [0.3, -0.7]])
    nn_coupling = _make_coupling_nn()

    energy = heisenberg_energy_from_src_index(q, src_index, nn_coupling, dist_pairwise)
    assert not torch.allclose(energy, torch.zeros(2))


def test_heisenberg_exchange_types_differ():
    """
    Different exchange types should produce different results.
    """
    src_index, dist_pairwise, _ = _make_src_index_and_dist(n_atoms=2)
    q = torch.tensor([[0.8, 0.3], [0.4, 0.6]])
    nn_coupling = _make_coupling_nn()

    results = {}
    for ex_type in ("heisenberg", "ising", "xy"):
        results[ex_type] = heisenberg_energy_from_src_index(
            q, src_index, nn_coupling, dist_pairwise, exchange_type=ex_type
        )

    assert not torch.allclose(results["heisenberg"], results["ising"], atol=1e-6)
    assert not torch.allclose(results["heisenberg"], results["xy"], atol=1e-6)
    assert not torch.allclose(results["ising"], results["xy"], atol=1e-6)


def test_heisenberg_invalid_exchange_type():
    """
    Invalid exchange type should raise ValueError.
    """
    src_index, dist_pairwise, _ = _make_src_index_and_dist(n_atoms=2)
    q = torch.tensor([[1.0, 0.5], [0.3, -0.7]])

    with pytest.raises(ValueError, match="invalid"):
        heisenberg_energy_from_src_index(
            q,
            src_index,
            _make_coupling_nn(),
            dist_pairwise,
            exchange_type="invalid",
        )


# ============================================================
# Charge renormalization tests
# ============================================================


def test_charge_renormalization_conserves():
    """
    After renormalization, total charge per graph should match target.
    """
    n_nodes = 4
    n_graphs = 2
    node_batch = torch.tensor([0, 0, 1, 1])
    target_charges = torch.tensor([2.0, 3.0])

    q = torch.tensor([0.5, 0.7, 1.0, 1.5])

    mock_data = _MockGraphAttentionData(
        num_nodes=n_nodes,
        num_graphs=n_graphs,
        node_batch=node_batch,
        charge=target_charges,
    )
    emb = {"data": mock_data}

    q_renorm = charge_renormalization(q.clone(), emb, eps=1e-8)

    for b in range(n_graphs):
        mask = node_batch == b
        total = q_renorm[:n_nodes][mask].sum()
        assert torch.allclose(
            total, target_charges[b], atol=1e-5
        ), f"Batch {b}: expected {target_charges[b]}, got {total}"


def test_charge_renormalization_zero_target():
    """
    Zero target charge should not produce NaN or Inf.
    """
    node_batch = torch.tensor([0, 0])
    target_charges = torch.tensor([0.0])
    q = torch.tensor([0.5, 0.5])

    mock_data = _MockGraphAttentionData(
        num_nodes=2, num_graphs=1, node_batch=node_batch, charge=target_charges
    )
    emb = {"data": mock_data}

    q_renorm = charge_renormalization(q.clone(), emb)
    assert not torch.isnan(q_renorm).any()
    assert not torch.isinf(q_renorm).any()


# ============================================================
# Spin+charge renormalization tests
# ============================================================


def test_spin_charge_renormalization_conserves():
    """
    After renormalization, both charge and spin should match targets.
    """
    n_nodes = 3
    n_graphs = 1
    node_batch = torch.tensor([0, 0, 0])
    target_charges = torch.tensor([2.0])
    target_spins = torch.tensor([1.0])

    q = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])

    mock_data = _MockGraphAttentionData(
        num_nodes=n_nodes,
        num_graphs=n_graphs,
        node_batch=node_batch,
        charge=target_charges,
        spin=target_spins,
    )
    emb = {"data": mock_data}

    q_renorm = charge_spin_renormalization(q, emb)

    valid = q_renorm[:n_nodes]
    total_charge = valid.sum()
    total_spin = (valid[:, 0] - valid[:, 1]).sum()

    assert torch.allclose(
        total_charge, target_charges[0], atol=1e-5
    ), f"Charge: expected {target_charges[0]}, got {total_charge}"
    assert torch.allclose(
        total_spin, target_spins[0], atol=1e-5
    ), f"Spin: expected {target_spins[0]}, got {total_spin}"


def test_spin_charge_renormalization_multi_batch():
    """
    Test spin+charge renormalization with multiple batches.
    """
    n_nodes = 4
    n_graphs = 2
    node_batch = torch.tensor([0, 0, 1, 1])
    target_charges = torch.tensor([1.0, 2.0])
    target_spins = torch.tensor([0.5, -0.5])

    q = torch.tensor([[0.3, 0.2], [0.4, 0.1], [0.5, 0.6], [0.7, 0.3]])

    mock_data = _MockGraphAttentionData(
        num_nodes=n_nodes,
        num_graphs=n_graphs,
        node_batch=node_batch,
        charge=target_charges,
        spin=target_spins,
    )
    emb = {"data": mock_data}

    q_renorm = charge_spin_renormalization(q, emb)

    for b in range(n_graphs):
        mask = node_batch == b
        valid = q_renorm[:n_nodes][mask]
        total_charge = valid.sum()
        total_spin = (valid[:, 0] - valid[:, 1]).sum()
        assert torch.allclose(total_charge, target_charges[b], atol=1e-5)
        assert torch.allclose(total_spin, target_spins[b], atol=1e-5)


# ============================================================
# Pairwise distance tests
# ============================================================


def test_compute_pairwise_distances_basic():
    """
    Test pairwise distances for a simple linear chain.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    batch = torch.tensor([0, 0, 0])

    dist = compute_pairwise_distances(pos, batch, num_nodes=3)
    assert dist.shape == (3, 3)
    assert torch.allclose(dist[0, 1], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(dist[0, 2], torch.tensor(3.0), atol=1e-6)
    assert torch.allclose(dist[1, 2], torch.tensor(2.0), atol=1e-6)
    # Diagonal should be zero
    assert torch.allclose(torch.diag(dist), torch.zeros(3), atol=1e-6)
    # Symmetric
    assert torch.allclose(dist, dist.T, atol=1e-6)


def test_compute_pairwise_distances_with_padding():
    """
    Test that padding is handled correctly (padded rows/cols are zero).
    """
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # padding
            [0.0, 0.0, 0.0],  # padding
        ]
    )
    batch = torch.tensor([0, 0, 0, -1, -1])

    dist = compute_pairwise_distances(pos, batch, num_nodes=3)
    assert dist.shape == (5, 5)
    assert torch.allclose(dist[0, 1], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(dist[3:, :], torch.zeros(2, 5), atol=1e-6)
    assert torch.allclose(dist[:, 3:], torch.zeros(5, 2), atol=1e-6)


# ============================================================
# compilable_scatter_on_dictionary test
# ============================================================


def test_compilable_scatter_on_dictionary():
    """
    Test dictionary scatter sums values correctly per group.
    """
    src = {
        "a": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "b": torch.tensor([10.0, 20.0, 30.0, 40.0]),
    }
    index = torch.tensor([0, 0, 1, 1])

    result = compilable_scatter_on_dictionary(src, index, dim_size=2)

    assert torch.allclose(result["a"], torch.tensor([3.0, 7.0]))
    assert torch.allclose(result["b"], torch.tensor([30.0, 70.0]))
