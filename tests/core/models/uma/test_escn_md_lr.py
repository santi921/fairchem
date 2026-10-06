"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch
from ase.build import molecule as get_molecule

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.escn_md_lr import (
    MLP_EFS_Head_LR,
    MLP_Energy_Head_LR,
    eSCNMDBackboneLR,
    intra_system_lr_edges,
)
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackboneLR

LMAX = 2
SPHERE_CHANNELS = 8

BACKBONE_KWARGS = dict(
    max_num_elements=100,
    sphere_channels=SPHERE_CHANNELS,
    lmax=LMAX,
    mmax=2,
    otf_graph=True,
    edge_channels=8,
    num_distance_basis=16,
    hidden_channels=16,
    hidden_channels_lr=8,
    num_layers=2,
    use_dataset_embedding=False,
    always_use_pbc=False,
    direct_forces=False,
)


def _get_water_data() -> AtomicData:
    return AtomicData.from_ase(
        input_atoms=get_molecule("H2O"),
        max_neigh=25,
        radius=6,
        task_name="lr_test",
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )


def test_lr_backbone_forward():
    torch.manual_seed(42)
    backbone = eSCNMDBackboneLR(**BACKBONE_KWARGS)
    data = _get_water_data()

    out = backbone(data)

    num_atoms = data["atomic_numbers"].shape[0]
    assert out["node_embedding"].shape == (
        num_atoms,
        (LMAX + 1) ** 2,
        SPHERE_CHANNELS,
    )
    assert torch.isfinite(out["node_embedding"]).all()
    assert "edge_index_lr" in out
    assert out["batch"].shape == (num_atoms,)


def test_lr_backbone_with_energy_head():
    torch.manual_seed(42)
    backbone = eSCNMDBackboneLR(**BACKBONE_KWARGS)
    head = MLP_Energy_Head_LR(backbone)
    data = _get_water_data()

    emb = backbone(data)
    out = head(data, emb)

    assert out["energy"].shape == (1,)
    assert torch.isfinite(out["energy"]).all()


@pytest.mark.parametrize("num_experts", [0, 2])
def test_moe_lr_backbone_forward(num_experts):
    torch.manual_seed(42)
    backbone = eSCNMDMoeBackboneLR(
        num_experts=num_experts,
        moe_layer_type="pytorch",
        **BACKBONE_KWARGS,
    )
    data = _get_water_data()

    out = backbone(data)

    num_atoms = data["atomic_numbers"].shape[0]
    assert out["node_embedding"].shape == (
        num_atoms,
        (LMAX + 1) ** 2,
        SPHERE_CHANNELS,
    )
    assert torch.isfinite(out["node_embedding"]).all()


def _get_two_molecule_batch() -> AtomicData:
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

    datas = []
    for name, charge in (("C6H6", 1), ("CH3CH2OH", -1)):
        atoms = get_molecule(name)
        atoms.info["charge"] = charge
        atoms.info["spin"] = 2
        datas.append(
            AtomicData.from_ase(
                input_atoms=atoms,
                max_neigh=25,
                radius=6,
                task_name="lr_test",
                r_edges=False,
                r_data_keys=["spin", "charge"],
            )
        )
    return atomicdata_list_to_batch(datas)


def test_lr_graph_is_not_capped_by_short_range_neighbors():
    data = _get_two_molecule_batch()
    kwargs = {**BACKBONE_KWARGS, "cutoff": 2.0, "max_neighbors": 3}
    backbone = eSCNMDBackboneLR(cutoff_lr=None, **kwargs)

    out = backbone(data)

    src, dst = out["edge_index_lr"]
    natoms = data["natoms"].tolist()
    assert src.shape[0] == sum(n * (n - 1) for n in natoms)
    assert (data["batch"][src] == data["batch"][dst]).all()
    assert (src != dst).all()
    dist = (data["pos"][src] - data["pos"][dst]).norm(dim=-1)
    assert dist.max() > 2.0


def test_intra_system_lr_edges_cutoff_and_cap():
    data = _get_two_molecule_batch()
    pos, batch = data["pos"], data["batch"]

    src, dst = intra_system_lr_edges(pos, batch, cutoff=2.5, max_neighbors=None)
    assert ((pos[src] - pos[dst]).norm(dim=-1) < 2.5).all()

    src, dst = intra_system_lr_edges(pos, batch, cutoff=None, max_neighbors=4)
    assert torch.bincount(dst, minlength=pos.shape[0]).max() == 4
    assert (batch[src] == batch[dst]).all()


@pytest.mark.parametrize("heisenberg_tf", [False, True])
def test_normalized_charges_match_system_charge(heisenberg_tf):
    torch.manual_seed(42)
    data = _get_two_molecule_batch()
    backbone = eSCNMDBackboneLR(
        cutoff_lr=None,
        heisenberg_tf=heisenberg_tf,
        lr_output_scaling_factor=0.1,
        **BACKBONE_KWARGS,
    )
    head = MLP_EFS_Head_LR(backbone)

    emb = backbone(data)
    lr = head.lr_predictor.get_lr_energies(emb, data, return_charges=True)

    totals = torch.zeros(2).index_add_(0, data["batch"], lr["charges"].view(-1))
    assert torch.allclose(totals, data["charge"].float(), atol=1e-5)


def test_moe_lr_backbone_with_uma_1p2_options():
    torch.manual_seed(42)
    kwargs = {
        k: v
        for k, v in BACKBONE_KWARGS.items()
        if k not in ("use_dataset_embedding", "num_distance_basis")
    }
    backbone = eSCNMDMoeBackboneLR(
        num_experts=4,
        moe_layer_type="pytorch",
        use_composition_embedding=True,
        composition_dropout=0.1,
        dataset_mapping={"omol": "omol"},
        dataset_emb_grad=True,
        charge_balanced_channels=[0, 1, 2],
        num_distance_basis=32,
        cutoff_lr=None,
        heisenberg_tf=True,
        **kwargs,
    )
    head = MLP_EFS_Head_LR(backbone)
    data = _get_two_molecule_batch()
    data.dataset = ["omol", "omol"]

    out = head(data, backbone(data))

    assert out["energy"]["energy"].shape == (2,)
    assert out["forces"]["forces"].shape == (data["pos"].shape[0], 3)
    assert torch.isfinite(out["forces"]["forces"]).all()


def _get_elongated_boxed_molecule() -> AtomicData:
    # two waters 150 A apart in an OMol-style 120 A vacuum box (pbc=True); the
    # system extent exceeds half the cell height
    atoms = get_molecule("H2O")
    far = get_molecule("H2O")
    far.translate([150.0, 0.0, 0.0])
    atoms += far
    atoms.info.update(charge=0, spin=1)
    return AtomicData.from_ase(
        input_atoms=atoms,
        task_name="lr_test",
        r_edges=False,
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120.0,
    )


def test_all_pairs_lr_graph_handles_elongated_boxed_molecule():
    data = _get_elongated_boxed_molecule()
    assert data["pbc"].all()
    backbone = eSCNMDBackboneLR(cutoff_lr=None, **BACKBONE_KWARGS)

    out = backbone(data)

    n = data["pos"].shape[0]
    assert out["edge_index_lr"].shape[1] == n * (n - 1)


def test_finite_lr_cutoff_uses_intra_system_pairs_in_vacuum_box():
    data = _get_elongated_boxed_molecule()
    backbone = eSCNMDBackboneLR(cutoff_lr=12.0, **BACKBONE_KWARGS)

    src, dst = backbone(data)["edge_index_lr"]

    assert ((data["pos"][src] - data["pos"][dst]).norm(dim=-1) < 12.0).all()


def test_lr_graph_skipped_when_lr_disabled():
    backbone = eSCNMDBackboneLR(latent_charge_tf=False, **BACKBONE_KWARGS)

    out = backbone(_get_elongated_boxed_molecule())

    assert "edge_index_lr" not in out
