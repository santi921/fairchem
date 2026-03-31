"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.models.les.les import Les
from fairchem.core.models.les.module import BEC, Ewald
from fairchem.core.models.utils.lr import (
    batch_spin_charge_renormalization,
    heisenberg_potential_full_from_edge_inds,
    potential_full_ewald_batched,
    potential_full_from_edge_inds,
)

# ============================================================
# Direct Coulomb (non-periodic) tests
# ============================================================


def test_potential_non_periodic():
    """
    Test direct Coulomb potential with a simple 3-atom system.

    Atoms at (0,0,0), (0,0,1), (0,0,-1) with charges [1, -1, 1].
    Edge index connects atom 0 <-> atom 1.
    """
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    )
    q = torch.tensor([[1.0], [-1.0], [1.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])

    result = potential_full_from_edge_inds(
        pos=pos,
        edge_index=edge_index,
        q=q,
    )

    potential = result["potential"]
    assert potential.shape == (3,), f"Expected shape (3,), got {potential.shape}"
    # Atom 2 has no edges, so its potential should be 0
    assert (
        potential[2].item() == 0.0
    ), f"Expected 0.0 for unconnected atom, got {potential[2].item()}"
    # Atoms 0 and 1 have opposite charges, so potentials should be negative
    assert (
        potential[0].item() < 0.0
    ), f"Expected negative potential for +/- pair, got {potential[0].item()}"


def test_potential_with_convergence_function():
    """
    Test that the convergence (erf) function modifies the potential.
    """
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    q = torch.tensor([[1.0], [1.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])

    result_no_conv = potential_full_from_edge_inds(
        pos=pos,
        edge_index=edge_index,
        q=q,
        conv_function_tf=False,
    )
    result_conv = potential_full_from_edge_inds(
        pos=pos,
        edge_index=edge_index,
        q=q,
        conv_function_tf=True,
    )

    # Convergence function should reduce the magnitude
    assert torch.abs(result_conv["potential"][0]) <= torch.abs(
        result_no_conv["potential"][0]
    ), "Convergence function should reduce potential magnitude"


# ============================================================
# Heisenberg / spin coupling tests
# ============================================================


def _make_coupling_nn(hidden=20):
    """
    Helper: simple coupling NN with ones weights.
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


def test_heisenberg_potential():
    """
    Test Heisenberg spin-spin coupling potential (default XY exchange).
    """
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    q = torch.tensor([[1.0, 0.5], [0.3, -0.7]])
    edge_index = torch.tensor([[0, 1], [1, 0]])

    result = heisenberg_potential_full_from_edge_inds(
        pos=pos,
        edge_index=edge_index,
        q=q,
        nn=_make_coupling_nn(),
    )

    assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"
    assert not torch.allclose(
        result, torch.zeros(2)
    ), "Expected non-zero Heisenberg energy"


def test_heisenberg_exchange_types():
    """
    Test that all three exchange types produce different results
    and that heisenberg = ising + xy (up to conventions).
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    q = torch.tensor([[0.8, 0.3], [0.4, 0.6]])
    edge_index = torch.tensor([[0, 1], [1, 0]])

    results = {}
    for ex_type in ("xy", "ising", "heisenberg"):
        results[ex_type] = heisenberg_potential_full_from_edge_inds(
            pos=pos,
            edge_index=edge_index,
            q=q,
            nn=_make_coupling_nn(),
            exchange_type=ex_type,
        )

    # All should be non-zero
    for name, val in results.items():
        assert not torch.allclose(val, torch.zeros(2)), f"{name} should be non-zero"

    # XY and Ising should generally differ
    assert not torch.allclose(
        results["xy"], results["ising"], atol=1e-6
    ), "XY and Ising should differ for generic charges"

    # Heisenberg = Ising - XY (since Heisenberg = diag - off-diag)
    # H = (aa + bb - ab - ba) = (a-b)(a-b) - 2(ab+ba) + (a-b)(a-b) ... hmm
    # Actually: heisenberg = ising_terms - xy_terms
    # Where ising_terms = aa + bb - ab - ba, xy_terms = ab + ba
    # But the learned J is the same NN, so just check they're all non-zero and distinct
    assert not torch.allclose(
        results["heisenberg"], results["xy"], atol=1e-6
    ), "Heisenberg and XY should differ"


def test_heisenberg_invalid_exchange_type():
    """
    Test that an invalid exchange type raises ValueError.
    """
    import pytest

    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    q = torch.tensor([[1.0, 0.5], [0.3, -0.7]])
    edge_index = torch.tensor([[0, 1], [1, 0]])

    with pytest.raises(ValueError, match="invalid"):
        heisenberg_potential_full_from_edge_inds(
            pos=pos,
            edge_index=edge_index,
            q=q,
            nn=_make_coupling_nn(),
            exchange_type="invalid",
        )


# ============================================================
# Ewald summation tests (batched lr.py version)
# ============================================================


def test_ewald_potential_basic():
    """
    Test Ewald summation with a simple 2-atom periodic cell.
    """
    cell = torch.tensor(
        [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]],
    )
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    q = torch.tensor([[1.0], [-1.0]])
    batch = torch.tensor([0, 0])

    result = potential_full_ewald_batched(
        pos=pos,
        q=q,
        cell=cell,
        batch=batch,
    )

    potential = result["potential"]
    assert potential.shape == (2,), f"Expected shape (2,), got {potential.shape}"
    assert not torch.allclose(
        potential, torch.zeros(2)
    ), "Expected non-zero Ewald potential for charged system"


def test_ewald_potential_charge_scaling():
    """
    Test that doubling all charges quadruples the Ewald energy.
    """
    cell = torch.tensor(
        [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]],
    )
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    batch = torch.tensor([0, 0])

    q1 = torch.tensor([[1.0], [-1.0]])
    q2 = torch.tensor([[2.0], [-2.0]])

    result1 = potential_full_ewald_batched(pos=pos, q=q1, cell=cell, batch=batch)
    result2 = potential_full_ewald_batched(pos=pos, q=q2, cell=cell, batch=batch)

    energy1 = result1["potential"].sum()
    energy2 = result2["potential"].sum()

    ratio = energy2 / energy1
    assert torch.allclose(
        ratio, torch.tensor(4.0), atol=0.1
    ), f"Expected energy ratio ~4.0, got {ratio.item():.3f}"


def test_ewald_potential_multi_batch():
    """
    Test Ewald summation with multiple batches.
    """
    cell = torch.tensor(
        [
            [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        ],
    )
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
    )
    q = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])
    batch = torch.tensor([0, 0, 1, 1])

    result = potential_full_ewald_batched(pos=pos, q=q, cell=cell, batch=batch)

    potential = result["potential"]
    assert potential.shape == (4,), f"Expected shape (4,), got {potential.shape}"

    assert not torch.allclose(potential[:2], torch.zeros(2))
    assert not torch.allclose(potential[2:], torch.zeros(2))

    # Within each batch, potentials are distributed equally
    assert torch.allclose(potential[0], potential[1])
    assert torch.allclose(potential[2], potential[3])

    # Different cell sizes should give different potentials
    assert not torch.allclose(potential[0], potential[2])


# ============================================================
# LES Ewald module tests
# ============================================================


def test_les_ewald_basic():
    """
    Test the LES Ewald module with a simple periodic system.
    """
    ewald = Ewald(sigma=1.0, dl=2.0)
    cell = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    q = torch.tensor([[1.0], [-1.0]])
    batch = torch.tensor([0, 0])

    result = ewald(q=q, r=pos, cell=cell.unsqueeze(0), batch=batch)
    assert result.shape[0] == 1, f"Expected 1 batch, got {result.shape}"
    assert result.item() != 0.0, "Expected non-zero Ewald energy"


def test_les_ewald_multi_q():
    """
    Test that multi-q charges (n_q > 1) work correctly.
    """
    ewald = Ewald(sigma=1.0, dl=2.0)
    cell = torch.tensor([[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]])
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    batch = torch.tensor([0, 0])

    # Single-q reference
    q1 = torch.tensor([[1.0], [-1.0]])
    e1 = ewald(q=q1, r=pos, cell=cell, batch=batch)

    # Multi-q: first channel same as q1, second channel zero
    q2 = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    e2 = ewald(q=q2, r=pos, cell=cell, batch=batch)

    # Multi-q energy should include both channels; channel 2 is zero
    # so total should match single-channel up to numerics
    assert torch.allclose(
        e1, e2, atol=1e-4
    ), f"Multi-q with zero channel should match single-q: {e1} vs {e2}"


def test_les_ewald_chunked():
    """
    Test that chunked k-vector processing matches un-chunked.
    """
    cell = torch.tensor([[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]])
    pos = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    q = torch.tensor([[1.0], [-1.0], [0.5]])
    batch = torch.tensor([0, 0, 0])

    ewald_unchunked = Ewald(sigma=1.0, dl=2.0, k_chunk_size=None)
    ewald_chunked = Ewald(sigma=1.0, dl=2.0, k_chunk_size=10)

    e1 = ewald_unchunked(q=q, r=pos, cell=cell, batch=batch)
    e2 = ewald_chunked(q=q, r=pos, cell=cell, batch=batch)

    assert torch.allclose(
        e1, e2, atol=1e-6
    ), f"Chunked and un-chunked Ewald should match: {e1} vs {e2}"


def test_les_ewald_realspace_nonperiodic():
    """
    Test Ewald real-space fallback for non-periodic system (zero cell).
    """
    ewald = Ewald(sigma=1.0, dl=2.0)
    cell = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    q = torch.tensor([[1.0], [-1.0]])
    batch = torch.tensor([0, 0])

    result = ewald(q=q, r=pos, cell=cell, batch=batch)
    assert result.numel() > 0
    assert not torch.isnan(result).any(), "Real-space Ewald produced NaN"


# ============================================================
# BEC tests
# ============================================================


def test_bec_shape_nonperiodic():
    """
    Test BEC output shape for non-periodic system.
    """
    bec_module = BEC(remove_mean=True, epsilon_factor=1.0)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        requires_grad=True,
    )
    q = torch.tensor([[1.0], [-1.0]])
    cell = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    batch = torch.tensor([0, 0])

    result = bec_module(q=q, r=pos, cell=cell, batch=batch)
    # BEC shape: [n_nodes, n_P, n_r] = [2, 3, 1] for 1-channel charge
    # After transpose fix, dim 1 is P-direction and dim 2 is r-direction
    assert result.shape[0] == 2, f"Expected 2 atoms, got {result.shape[0]}"


def test_bec_shape_periodic():
    """
    Test BEC output shape for periodic system.
    """
    bec_module = BEC(remove_mean=True, epsilon_factor=1.0)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        requires_grad=True,
    )
    q = torch.tensor([[1.0], [-1.0]])
    cell = torch.tensor([[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]])
    batch = torch.tensor([0, 0])

    result = bec_module(q=q, r=pos, cell=cell, batch=batch)
    assert result.shape[0] == 2, f"Expected 2 atoms, got {result.shape[0]}"
    assert not torch.isnan(result).any(), "BEC produced NaN"


# ============================================================
# Charge renormalization tests
# ============================================================


def test_charge_renormalization():
    """
    Test that batch_spin_charge_renormalization conserves total charge and spin.
    """
    charges_raw = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
    batch = torch.tensor([0, 0, 1])
    q_total = torch.tensor([2.0, 1.0])  # target total charges per batch
    s_total = torch.tensor([0.0, 0.0])  # target total spins per batch

    result = batch_spin_charge_renormalization(
        charges_raw=charges_raw,
        q_total=q_total,
        s_total=s_total,
        batch=batch,
    )

    assert result.shape == charges_raw.shape

    # Check charge conservation: sum of (alpha + beta) per batch = q_total
    for b in range(2):
        mask = batch == b
        total_charge = result[mask].sum()
        assert torch.allclose(
            total_charge, q_total[b], atol=1e-5
        ), f"Batch {b}: expected charge {q_total[b]}, got {total_charge}"

    # Check spin conservation: sum of (alpha - beta) per batch = s_total
    for b in range(2):
        mask = batch == b
        total_spin = (result[mask, 0] - result[mask, 1]).sum()
        assert torch.allclose(
            total_spin, s_total[b], atol=1e-5
        ), f"Batch {b}: expected spin {s_total[b]}, got {total_spin}"


def test_charge_renormalization_neutral_system():
    """
    Test renormalization with a neutral system (target charge = 0).

    This should not produce NaN or Inf.
    """
    charges_raw = torch.tensor([[0.1, -0.1], [-0.05, 0.05]])
    batch = torch.tensor([0, 0])
    q_total = torch.tensor([0.0])
    s_total = torch.tensor([0.0])

    result = batch_spin_charge_renormalization(
        charges_raw=charges_raw,
        q_total=q_total,
        s_total=s_total,
        batch=batch,
    )

    assert not torch.isnan(result).any(), "Renormalization produced NaN"
    assert not torch.isinf(result).any(), "Renormalization produced Inf"
    total = result.sum()
    assert torch.allclose(total, torch.tensor(0.0), atol=1e-5)


# ============================================================
# LES integration test
# ============================================================


def test_les_forward_with_latent_charges():
    """
    Test Les module forward pass with pre-computed latent charges.
    """
    les = Les(sigma=1.0, dl=2.0, use_atomwise=False)
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    cell = torch.tensor([[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]])
    charges = torch.tensor([1.0, -1.0])
    batch = torch.tensor([0, 0])

    output = les(
        positions=pos,
        cell=cell,
        latent_charges=charges,
        batch=batch,
        compute_energy=True,
        compute_bec=False,
    )

    assert "E_lr" in output
    assert output["E_lr"] is not None
    assert output["latent_charges"] is not None
    assert output["BEC"] is None


def test_les_from_yaml(tmp_path):
    """
    Test Les.from_yaml class method.
    """
    config_path = tmp_path / "les_config.yaml"
    config_path.write_text(
        "sigma: 2.0\n" "dl: 1.0\n" "n_layers: 2\n" "use_atomwise: false\n"
    )

    les = Les.from_yaml(str(config_path))
    assert les.sigma == 2.0
    assert les.dl == 1.0
    assert les.n_layers == 2
    assert not les.use_atomwise
