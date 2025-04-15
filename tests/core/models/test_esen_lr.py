import torch

from fairchem.core.models.esen.nn.lr import (
    potential_full_from_edge_inds,
    heisenberg_potential_full_from_edge_inds,
)

def test_charge():
    q = torch.tensor([1.0, -1.0, 1.0])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    pos = torch.tensor([[0.0,  0.0, 0.0], [0.0,  0.0,  1.0], [0.0,  0.0,  -1.0]])

    energy_spin_raw = potential_full_from_edge_inds(
        pos=pos,
        edge_index=edge_index,
        q=q,
    )

    # assert all are zero expect for the first
    benchmark = torch.tensor([-0.05432664975523949])
    assert torch.allclose(
        input=energy_spin_raw[0], other=benchmark, atol=1e-1
    ), f"Expected -0.05432664975523949, got {energy_spin_raw[0]} "


def test_spin():
    q = torch.tensor([[1.0, 1.0, 1.0],[1.0, -1.0, 1.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    pos = torch.tensor([[0.0,  0.0, 0.0], [0.0,  0.0,  1.0], [0.0,  0.0,  -1.0]])

    layers = [torch.nn.Linear(in_features=1, out_features=20)]
    layers += [torch.nn.Linear(in_features=20, out_features=1)]
    nn_coupling = torch.nn.Sequential(*layers)
    # set nn_charge to all ones
    nn_coupling[0].weight.data.fill_(1)
    nn_coupling[0].bias.data.fill_(0)
    nn_coupling[1].weight.data.fill_(1)
    nn_coupling[1].bias.data.fill_(0)

    #nn_coupling.to(self.batch.pos.device)

    energy_spin_raw = heisenberg_potential_full_from_edge_inds(
        pos=pos,
        edge_index=edge_index,
        q=q,
        nn=nn_coupling,
    )

    # assert all are zero expect for the first
    benchmark = torch.tensor([13.653789520263672])
    assert torch.allclose(
        input=energy_spin_raw[0], other=benchmark, atol=1e-1
    ), f"Expected 13.653789520263672, got {energy_spin_raw[0]} "

