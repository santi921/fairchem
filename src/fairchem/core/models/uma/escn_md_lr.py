"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from fairchem.core.common import gp_utils
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.graph.compute import generate_graph
from fairchem.core.models.base import HeadInterface
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from fairchem.core.models.uma.outputs import (
    compute_forces,
    compute_forces_and_stress,
    get_l_component_range,
    reduce_node_to_system,
)
from fairchem.core.models.utils.lr_charges import LRChargePredictor

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData


def intra_system_lr_edges(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float | None,
    max_neighbors: int | None,
) -> torch.Tensor:
    """
    Build a long-range edge index from all intra-system atom pairs.

    Uses a dense pairwise search, which is exact and cheap for the batch sizes
    used in training (a few hundred atoms per rank). Only valid for
    non-periodic systems since periodic images are not considered.

    Args:
        pos: Atomic positions, shape [N, 3].
        batch: System index of each atom, shape [N].
        cutoff: Pair distance cutoff in Angstrom, or None to keep every pair.
        max_neighbors: Optional cap on neighbors per target atom; the nearest
            are kept. None keeps all pairs within the cutoff.

    Returns:
        Edge index of shape [2, E] as (source, target), matching the
        convention of the short-range graph.
    """
    pos = pos.detach()
    valid = batch.unsqueeze(0) == batch.unsqueeze(1)
    valid.fill_diagonal_(False)
    if cutoff is None and max_neighbors is None:
        target, source = valid.nonzero(as_tuple=True)
        return torch.stack([source, target])

    dist = torch.cdist(pos, pos)
    if cutoff is not None:
        valid &= dist < cutoff
    if max_neighbors is not None and max_neighbors < pos.shape[0]:
        dist = dist.masked_fill(~valid, torch.inf)
        nearest = dist.topk(max_neighbors, dim=1, largest=False).indices
        keep = torch.zeros_like(valid).scatter_(1, nearest, True)
        valid &= keep
    target, source = valid.nonzero(as_tuple=True)
    return torch.stack([source, target])


@registry.register_model("escnmd_backbone_lr")
class eSCNMDBackboneLR(eSCNMDBackbone):
    """
    eSCNMD backbone that also emits a long-range edge index for the LR heads.

    The short-range model is exactly eSCNMDBackbone, so every option it
    supports (dataset mappings, charge/spin balanced channels, execution
    backends) works here. The long-range physics lives in the LR heads, which
    read the LR settings below off the backbone and consume
    ``emb["edge_index_lr"]``.

    Args:
        hidden_channels_lr: Hidden width of the latent charge MLP.
        heisenberg_tf: Predict alpha/beta spin channels and add a learned
            spin-exchange energy.
        exchange_type: Spin exchange form: "heisenberg", "ising", or "xy".
        latent_charge_tf: Enable the latent-charge Coulomb energy.
        return_bec: Compute Born effective charges.
        conv_function_tf: Damp the Coulomb kernel with erf(r / sqrt(2) sigma).
        lr_output_scaling_factor: Scale applied to the predicted charges.
        cutoff_lr: Long-range pair cutoff in Angstrom. A negative value reuses
            the short-range cutoff and None keeps every intra-system pair.
        max_neighbors_lr: Optional per-atom cap on long-range neighbors,
            independent of the short-range max_neighbors. None is uncapped.
        normalize_charges_tf: Constrain charges (and spins) to the system totals.
        equil_charges_tf: Add electronegativity/hardness charge-equilibration
            energy terms.
        use_ewald_tf: Use Ewald summation for periodic systems.
        **kwargs: Passed through to eSCNMDBackbone.
    """

    def __init__(
        self,
        hidden_channels_lr: int = 64,
        heisenberg_tf: bool = False,
        exchange_type: str = "heisenberg",
        latent_charge_tf: bool = True,
        return_bec: bool = False,
        conv_function_tf: bool = True,
        lr_output_scaling_factor: float = 1.0,
        cutoff_lr: float | None = -1.0,
        max_neighbors_lr: int | None = None,
        normalize_charges_tf: bool = True,
        equil_charges_tf: bool = False,
        use_ewald_tf: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_channels_lr = hidden_channels_lr
        self.heisenberg_tf = heisenberg_tf
        self.exchange_type = exchange_type
        self.latent_charge_tf = latent_charge_tf
        self.return_bec = return_bec
        self.conv_function_tf = conv_function_tf
        self.lr_output_scaling_factor = lr_output_scaling_factor
        self.normalize_charges_tf = normalize_charges_tf
        self.equil_charges_tf = equil_charges_tf
        self.use_ewald_tf = use_ewald_tf
        self.max_neighbors_lr = max_neighbors_lr
        if cutoff_lr is not None and cutoff_lr < 0.0:
            cutoff_lr = self.cutoff
        self.cutoff_lr = cutoff_lr
        self._warned_periodic_all_pairs = False

    def _images_out_of_reach(self, data_dict: AtomicData, cutoff: float) -> bool:
        """
        Whether no periodic image can fall within cutoff of any atom.

        Molecular datasets such as OMol box each molecule in a large vacuum
        cell with pbc=True, so the pbc flags alone cannot identify isolated
        systems. An image of atom j is at least (cell height - extent) away,
        where extent bounds intra-system distances.
        """
        if not data_dict["pbc"].any():
            return True
        cell = data_dict["cell"]
        volume = torch.linalg.det(cell).abs()
        cross_norms = torch.stack(
            [
                torch.linalg.cross(cell[:, (i + 1) % 3], cell[:, (i + 2) % 3]).norm(
                    dim=-1
                )
                for i in range(3)
            ],
            dim=1,
        )
        min_height = (volume.unsqueeze(1) / cross_norms).amin(dim=1)

        pos, batch = data_dict["pos"], data_dict["batch"]
        num_systems = len(data_dict["natoms"])
        lo = pos.new_full((num_systems, 3), torch.inf).scatter_reduce(
            0, batch.unsqueeze(1).expand(-1, 3), pos, reduce="amin"
        )
        hi = pos.new_full((num_systems, 3), -torch.inf).scatter_reduce(
            0, batch.unsqueeze(1).expand(-1, 3), pos, reduce="amax"
        )
        extent = (hi - lo).norm(dim=1)
        return bool((min_height > extent + cutoff).all())

    @torch.no_grad()
    def _generate_lr_graph(self, data_dict: AtomicData) -> torch.Tensor:
        if self.cutoff_lr is None:
            # all intra-system pairs ignore periodic images by definition; warn
            # once if a system is not isolated even at the short-range cutoff
            if not self._warned_periodic_all_pairs and not self._images_out_of_reach(
                data_dict, self.cutoff
            ):
                logging.warning(
                    "cutoff_lr=None ignores periodic images, but a batch has "
                    f"images within {self.cutoff} A; use a finite cutoff_lr or "
                    "use_ewald_tf for periodic systems"
                )
                self._warned_periodic_all_pairs = True
            return intra_system_lr_edges(
                data_dict["pos"], data_dict["batch"], None, self.max_neighbors_lr
            )

        if self._images_out_of_reach(data_dict, self.cutoff_lr):
            return intra_system_lr_edges(
                data_dict["pos"],
                data_dict["batch"],
                self.cutoff_lr,
                self.max_neighbors_lr,
            )

        # Periodic systems: the direct Coulomb sum ignores image offsets, so
        # this graph is only exact when use_ewald_tf handles the electrostatics.
        graph_dict = generate_graph(
            data_dict,
            cutoff=self.cutoff_lr,
            max_neighbors=self.max_neighbors_lr or self.max_neighbors,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            radius_pbc_version=self.radius_pbc_version,
            pbc=data_dict["pbc"],
        )
        return graph_dict["edge_index"]

    def forward(self, data_dict: AtomicData) -> dict[str, torch.Tensor]:
        if gp_utils.initialized():
            raise NotImplementedError("eSCNMDBackboneLR does not support GP")
        out = super().forward(data_dict)
        if self.latent_charge_tf:
            out["edge_index_lr"] = self._generate_lr_graph(data_dict)
        return out


def _build_lr_predictor(
    backbone: eSCNMDBackboneLR,
    return_bec: bool | None = None,
) -> LRChargePredictor:
    """
    Build an LRChargePredictor from backbone config.
    """
    lr_comp_size = 2 if backbone.heisenberg_tf else 1
    bec = backbone.return_bec if return_bec is None else return_bec
    return LRChargePredictor(
        sphere_channels=backbone.sphere_channels,
        hidden_channels_lr=backbone.hidden_channels_lr,
        lr_comp_size=lr_comp_size,
        lr_output_scaling_factor=backbone.lr_output_scaling_factor,
        normalize_charges_tf=backbone.normalize_charges_tf,
        equil_charges_tf=backbone.equil_charges_tf,
        heisenberg_tf=backbone.heisenberg_tf,
        exchange_type=backbone.exchange_type,
        use_ewald_tf=backbone.use_ewald_tf,
        conv_function_tf=backbone.conv_function_tf,
        return_bec=bec,
    )


def _build_energy_block(sphere_channels: int, hidden_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(sphere_channels, hidden_channels, bias=True),
        nn.SiLU(),
        nn.Linear(hidden_channels, hidden_channels, bias=True),
        nn.SiLU(),
        nn.Linear(hidden_channels, 1, bias=True),
    )


def _compute_energy_with_lr(
    energy_block: nn.Module,
    lr_predictor: LRChargePredictor | None,
    emb: dict[str, torch.Tensor],
    data: AtomicData,
    latent_charge_tf: bool,
    heisenberg_tf: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    """
    Compute short-range + long-range energy. Returns (energy_per_system, lr_dict).
    """
    scalar_embedding = get_l_component_range(
        emb["node_embedding"], l_min=0, l_max=0
    ).squeeze(1)
    node_energy = energy_block(scalar_embedding).view(-1)

    lr_energy_dict = None
    if latent_charge_tf and lr_predictor is not None:
        lr_energy_dict = lr_predictor.get_lr_energies(emb, data)
        node_energy = node_energy + lr_energy_dict["energy"]
        if heisenberg_tf:
            node_energy = node_energy + lr_energy_dict["energy_spin"]

    _, energy_part = reduce_node_to_system(
        node_energy, data["batch"], len(data["natoms"])
    )
    return energy_part, lr_energy_dict


@registry.register_model("esen_efs_head_lr")
class MLP_EFS_Head_LR(nn.Module, HeadInterface):
    """
    Gradient-based energy/force/stress head with long-range electrostatics.
    """

    def __init__(
        self,
        backbone: eSCNMDBackboneLR,
        prefix: str | None = None,
        wrap_property: bool = True,
    ) -> None:
        super().__init__()
        backbone.energy_block = None
        backbone.force_block = None
        self.regress_config = backbone.regress_config
        self.prefix = prefix
        self.wrap_property = wrap_property
        self.latent_charge_tf = backbone.latent_charge_tf
        self.heisenberg_tf = backbone.heisenberg_tf

        self.energy_block = _build_energy_block(
            backbone.sphere_channels, backbone.hidden_channels
        )
        self.lr_predictor = (
            _build_lr_predictor(backbone) if self.latent_charge_tf else None
        )

        assert (
            not self.regress_config.direct_forces
        ), "EFS head is only used for gradient-based forces/stress."

    @conditional_grad(torch.enable_grad())
    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        energy_key = f"{self.prefix}_energy" if self.prefix else "energy"
        forces_key = f"{self.prefix}_forces" if self.prefix else "forces"
        stress_key = f"{self.prefix}_stress" if self.prefix else "stress"

        outputs = {}
        energy, _ = _compute_energy_with_lr(
            self.energy_block,
            self.lr_predictor,
            emb,
            data,
            self.latent_charge_tf,
            self.heisenberg_tf,
        )
        outputs[energy_key] = {"energy": energy} if self.wrap_property else energy

        embeddings = emb["node_embedding"].detach()
        outputs["embeddings"] = (
            {"embeddings": embeddings} if self.wrap_property else embeddings
        )

        if self.regress_config.stress:
            forces, stress = compute_forces_and_stress(
                energy,
                data["pos"],
                data["cell"],
                batch=data["batch"],
                training=self.training,
            )
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
            outputs[stress_key] = {"stress": stress} if self.wrap_property else stress
        elif self.regress_config.forces:
            forces = compute_forces(energy, data["pos"], training=self.training)
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
        return outputs


@registry.register_model("esen_mlp_energy_head_lr")
class MLP_Energy_Head_LR(nn.Module, HeadInterface):
    """
    Energy-only head with long-range electrostatics.
    """

    def __init__(
        self,
        backbone: eSCNMDBackboneLR,
        reduce: str = "sum",
    ) -> None:
        super().__init__()
        self.reduce = reduce
        self.latent_charge_tf = backbone.latent_charge_tf
        self.heisenberg_tf = backbone.heisenberg_tf

        self.energy_block = _build_energy_block(
            backbone.sphere_channels, backbone.hidden_channels
        )
        self.lr_predictor = (
            _build_lr_predictor(backbone, return_bec=False)
            if self.latent_charge_tf
            else None
        )

    def forward(
        self,
        data_dict: AtomicData,
        emb: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        energy, _ = _compute_energy_with_lr(
            self.energy_block,
            self.lr_predictor,
            emb,
            data_dict,
            self.latent_charge_tf,
            self.heisenberg_tf,
        )

        if self.reduce == "mean":
            energy = energy / data_dict["natoms"]
        elif self.reduce != "sum":
            raise ValueError(f"reduce must be 'sum' or 'mean', got: {self.reduce}")
        return {"energy": energy}


@registry.register_model("esen_linear_energy_head_lr")
class Linear_Energy_Head_LR(MLP_Energy_Head_LR):
    """
    Energy-only head with long-range electrostatics (alias of the MLP head).
    """
