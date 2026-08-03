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
    MLP_Energy_Head_LR,
    eSCNMDBackboneLR,
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
