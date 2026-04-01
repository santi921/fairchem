"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.profiler import record_function

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.allscaip.AllScAIP import (
    AllScAIPEnergyHead,
    AllScAIPHeadBase,
)
from fairchem.core.models.allscaip.utils.data_preprocess import unpad_results
from fairchem.core.models.allscaip.utils.lr_utils import (
    charge_renormalization,
    charge_spin_renormalization,
    compute_pairwise_distances,
    coulomb_energy_from_src_index,
    heisenberg_energy_from_src_index,
)
from fairchem.core.models.allscaip.utils.nn_utils import get_feedforward
from fairchem.core.models.escaip.utils.graph_utils import compilable_scatter
from fairchem.core.models.les.module import Ewald

if TYPE_CHECKING:
    from fairchem.core.models.allscaip.AllScAIP import AllScAIPBackbone


class AllScAIPLRChargeModule(nn.Module):
    """
    Shared module for AllScAIP long-range charge prediction and energy.

    Owns the charge MLP, optional equilibration networks (hardness,
    electronegativity), and optional Heisenberg spin coupling NN.
    Used by both the direct energy head and gradient force/stress head
    to avoid code duplication.

    Args:
        hidden_size: backbone hidden dimension (input to charge MLP)
        hidden_size_lr: hidden dim for LR-specific MLPs
        activation: activation function name for the backbone FFN
        heisenberg_tf: enable Heisenberg spin coupling
        equil_charges_tf: enable electronegativity equilibration
        constrain_charge: enforce charge/spin conservation
        charge_scale: scaling factor for raw predicted charges
        exchange_type: spin exchange type ("heisenberg", "ising", "xy")
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_size_lr: int = 128,
        activation: str | None = None,
        heisenberg_tf: bool = False,
        equil_charges_tf: bool = False,
        constrain_charge: bool = False,
        charge_scale: float = 1.0,
        exchange_type: str = "heisenberg",
        use_ewald_tf: bool = False,
        conv_function_tf: bool = False,
        ewald_sigma: float = 1.0,
        ewald_dl: float = 2.0,
        ewald_k_chunk_size: int = 50000,
    ):
        super().__init__()
        self.heisenberg_tf = heisenberg_tf
        self.equil_charges_tf = equil_charges_tf
        self.constrain_charge = constrain_charge
        self.charge_scale = charge_scale
        self.exchange_type = exchange_type
        self.use_ewald_tf = use_ewald_tf
        self.conv_function_tf = conv_function_tf

        # 2 output channels (alpha/beta) if spin coupling, else 1
        self.latent_dim_out = 2 if heisenberg_tf else 1

        self.charge_ffn = get_feedforward(
            hidden_dim=hidden_size_lr,
            input_dim=hidden_size,
            hidden_layer_multiplier=1,
            output_dim=self.latent_dim_out,
            bias=True,
            activation=activation,
        )

        if equil_charges_tf:
            self.hardness_ffn = get_feedforward(
                hidden_dim=hidden_size_lr,
                input_dim=hidden_size,
                hidden_layer_multiplier=1,
                output_dim=1,
                bias=True,
                activation=activation,
            )
            self.electronegativity_ffn = get_feedforward(
                hidden_dim=hidden_size_lr,
                input_dim=hidden_size,
                hidden_layer_multiplier=1,
                output_dim=1,
                bias=True,
                activation=activation,
            )

        if heisenberg_tf:
            self.coupling_ffn = get_feedforward(
                input_dim=1,
                hidden_dim=hidden_size_lr,
                hidden_layer_multiplier=1,
                output_dim=1,
                bias=True,
                activation=activation,
            )

        if use_ewald_tf:
            self.ewald = Ewald(
                sigma=ewald_sigma,
                dl=ewald_dl,
                k_chunk_size=ewald_k_chunk_size,
            )

    def _is_periodic(self, data: dict) -> bool:
        """
        Check if the system has a non-degenerate unit cell.
        """
        if data.get("cell") is None:
            return False
        det_cells = torch.linalg.det(data["cell"])
        return not torch.any(det_cells.abs() < 1e-6).item()

    def _compute_ewald_energy(
        self,
        charges_1d: torch.Tensor,
        data: dict,
        graph_data,
    ) -> torch.Tensor:
        """
        Compute per-atom Ewald energy for periodic systems.

        The Ewald module returns per-batch scalar; this distributes
        the energy uniformly across atoms in each batch.
        """
        num_nodes = graph_data.num_nodes

        # Ewald needs unpadded inputs
        valid_charges = charges_1d[:num_nodes].view(-1, 1)
        valid_pos = data["pos"][:num_nodes]
        valid_batch = graph_data.node_batch[:num_nodes]

        ewald_energy = self.ewald(
            q=valid_charges,
            r=valid_pos,
            cell=data["cell"],
            batch=valid_batch,
        )

        # Distribute per-batch energy uniformly to per-atom
        device = data["pos"].device
        dtype = data["pos"].dtype
        n_batches = graph_data.num_graphs
        ones = torch.ones(num_nodes, device=device, dtype=dtype)
        natoms_per_batch = torch.zeros(n_batches, device=device, dtype=dtype)
        natoms_per_batch.scatter_add_(0, valid_batch, ones)
        per_atom_energy = ewald_energy.view(-1) / natoms_per_batch

        # Map back to padded tensor
        max_nodes = graph_data.max_num_nodes
        result = torch.zeros(max_nodes, device=device, dtype=dtype)
        result[:num_nodes] = per_atom_energy[valid_batch]
        return result

    def forward(
        self,
        node_reps: torch.Tensor,
        emb: dict[str, torch.Tensor],
        data: dict,
    ) -> torch.Tensor:
        """
        Compute per-atom LR energy from node representations.

        Args:
            node_reps: backbone output node features, shape (max_N, hidden)
            emb: backbone embedding dict with "data" key
            data: raw AtomicData dict with "pos", "batch" keys

        Returns:
            Per-atom LR energy, shape (max_N,)
        """
        graph_data = emb["data"]

        # Compute pairwise distances (not stored in GraphAttentionData)
        dist_pairwise = compute_pairwise_distances(
            data["pos"], data["batch"], graph_data.num_nodes
        )

        if self.latent_dim_out == 2:
            charges_raw_2d = self.charge_ffn(node_reps) * self.charge_scale

            if self.constrain_charge:
                charges_raw_2d = charge_spin_renormalization(charges_raw_2d, emb)

            charges_raw_1d = charges_raw_2d.sum(dim=1, keepdim=True)

            energy_spin = heisenberg_energy_from_src_index(
                q=charges_raw_2d,
                src_index=graph_data.src_index,
                dist_pairwise=dist_pairwise,
                j_coupling_nn=self.coupling_ffn,
                exchange_type=self.exchange_type,
            )
        else:
            charges_raw_1d = self.charge_ffn(node_reps) * self.charge_scale

            if self.constrain_charge:
                flattened = charge_renormalization(
                    charges_raw_1d.squeeze(-1), emb, eps=1e-8
                )
                charges_raw_1d = flattened.unsqueeze(-1)

        # Periodic systems: Ewald summation; non-periodic: direct Coulomb
        if self.use_ewald_tf and self._is_periodic(data):
            e_charge_single = self._compute_ewald_energy(
                charges_raw_1d, data, graph_data
            )
        else:
            e_charge_single = coulomb_energy_from_src_index(
                charges_raw_1d,
                graph_data.src_index,
                dist_pairwise,
                use_convergence=self.conv_function_tf,
            )

        if self.equil_charges_tf:
            hardness = self.hardness_ffn(node_reps)
            electronegativity = self.electronegativity_ffn(node_reps)
            en_electrostatic = (electronegativity * charges_raw_1d).view(-1)
            en_hardness = 0.5 * (hardness * charges_raw_1d**2).view(-1)
            e_charge_single = e_charge_single + en_electrostatic + en_hardness

        if self.latent_dim_out == 2:
            e_charge_single = e_charge_single + energy_spin

        # Mask out padded nodes (same pattern as upstream energy head)
        e_charge_single = e_charge_single * graph_data.node_padding_mask

        return e_charge_single


@registry.register_model("AllScAIP_energy_head_lr")
class AllScAIPEnergyHeadLR(AllScAIPHeadBase):
    """
    AllScAIP energy head with long-range Coulomb and optional spin coupling.

    Combines short-range energy from backbone features with long-range
    electrostatic energy computed from predicted latent charges.
    Supports torch.compile for the SR and LR forward passes.

    Config kwargs (passed via head_kwargs in YAML):
        hidden_size_lr: hidden dim for LR MLPs (default: 128)
        heisenberg_tf: enable Heisenberg spin coupling (default: False)
        equil_charges_tf: enable equilibration terms (default: False)
        constrain_charge: enforce charge conservation (default: False)
        charge_scale: charge scaling factor (default: 1.0)
        exchange_type: spin exchange type (default: "heisenberg")
        use_ewald_tf: use Ewald summation for periodic systems (default: False)
        conv_function_tf: apply erf convergence to direct Coulomb (default: False)
        ewald_sigma: Gaussian width for Ewald splitting (default: 1.0)
        ewald_dl: grid resolution for Ewald k-space (default: 2.0)
        ewald_k_chunk_size: max k-vectors per chunk (default: 50000)
    """

    def __init__(
        self,
        backbone: AllScAIPBackbone,
        hidden_size_lr: int = 128,
        heisenberg_tf: bool = False,
        equil_charges_tf: bool = False,
        constrain_charge: bool = False,
        charge_scale: float = 1.0,
        exchange_type: str = "heisenberg",
        use_ewald_tf: bool = False,
        conv_function_tf: bool = False,
        ewald_sigma: float = 1.0,
        ewald_dl: float = 2.0,
        ewald_k_chunk_size: int = 50000,
    ):
        super().__init__(backbone)

        # Short-range energy MLP
        self.energy_ffn = get_feedforward(
            hidden_dim=self.global_cfg.hidden_size,
            hidden_layer_multiplier=self.gnn_cfg.output_hidden_layer_multiplier,
            output_dim=1,
            bias=True,
            activation=self.global_cfg.activation,
        )
        self.energy_reduce = self.gnn_cfg.energy_reduce

        # Long-range charge module
        self.lr_module = AllScAIPLRChargeModule(
            hidden_size=self.global_cfg.hidden_size,
            hidden_size_lr=hidden_size_lr,
            heisenberg_tf=heisenberg_tf,
            equil_charges_tf=equil_charges_tf,
            constrain_charge=constrain_charge,
            charge_scale=charge_scale,
            exchange_type=exchange_type,
            use_ewald_tf=use_ewald_tf,
            conv_function_tf=conv_function_tf,
            ewald_sigma=ewald_sigma,
            ewald_dl=ewald_dl,
            ewald_k_chunk_size=ewald_k_chunk_size,
        )

        self.post_init()

    def compiled_forward_sr(self, emb: dict[str, torch.Tensor]):
        """
        Short-range energy: backbone features -> energy MLP -> scatter.
        """
        node_reps = self.get_node_reps(emb)
        energy_output = self.energy_ffn(node_reps)

        # Mask padded nodes
        energy_output = energy_output * emb["data"].node_padding_mask.unsqueeze(-1)

        energy_output = compilable_scatter(
            src=energy_output,
            index=emb["data"].node_batch,
            dim_size=emb["data"].max_batch_size,
            dim=0,
            reduce=self.energy_reduce,
        )
        return energy_output.squeeze()

    def compiled_forward_lr(
        self,
        emb: dict[str, torch.Tensor],
        data: dict,
        node_reps: torch.Tensor,
    ):
        """
        Long-range energy: node features -> charges -> Coulomb scatter.
        """
        e_charge_single = self.lr_module(node_reps, emb, data)

        e_charge = compilable_scatter(
            e_charge_single,
            index=emb["data"].node_batch,
            dim_size=emb["data"].max_batch_size,
            dim=0,
            reduce=self.energy_reduce,
        )
        return e_charge

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        forward_fn_sr = (
            torch.compile(self.compiled_forward_sr)
            if self.global_cfg.use_compile
            else self.compiled_forward_sr
        )

        node_reps = self.get_node_reps(emb)

        with record_function("energy_head_lr_sr"):
            energy_sr = forward_fn_sr(emb)

        with record_function("energy_head_lr_charges"):
            energy_lr = self.compiled_forward_lr(emb, data, node_reps)

        if len(energy_sr.shape) == 0:
            energy_sr = energy_sr.unsqueeze(0)
        if len(energy_lr.shape) == 0:
            energy_lr = energy_lr.unsqueeze(0)

        results_to_unpad = {
            "energy": energy_sr,
            "energy_coul": energy_lr,
        }
        res_unpad = unpad_results(results=results_to_unpad, data=emb["data"])

        return {"energy": res_unpad["energy"] + res_unpad["energy_coul"]}


@registry.register_model("AllScAIP_grad_energy_force_stress_head_lr")
class AllScAIPGradEnergyForceStressHeadLR(AllScAIPEnergyHead):
    """
    AllScAIP gradient-based force/stress head with long-range energy.

    Does NOT support torch.compile (autograd graph breaks compilation).
    Computes combined SR+LR energy then differentiates for forces/stress.

    Config kwargs (passed via head_kwargs in YAML):
        hidden_size_lr: hidden dim for LR MLPs (default: 128)
        heisenberg_tf: enable Heisenberg spin coupling (default: False)
        equil_charges_tf: enable equilibration terms (default: False)
        constrain_charge: enforce charge conservation (default: False)
        charge_scale: charge scaling factor (default: 1.0)
        exchange_type: spin exchange type (default: "heisenberg")
        use_ewald_tf: use Ewald summation for periodic systems (default: False)
        conv_function_tf: apply erf convergence to direct Coulomb (default: False)
        ewald_sigma: Gaussian width for Ewald splitting (default: 1.0)
        ewald_dl: grid resolution for Ewald k-space (default: 2.0)
        ewald_k_chunk_size: max k-vectors per chunk (default: 50000)
    """

    def __init__(
        self,
        backbone: AllScAIPBackbone,
        prefix: str | None = None,
        wrap_property: bool = True,
        hidden_size_lr: int = 128,
        heisenberg_tf: bool = False,
        equil_charges_tf: bool = False,
        constrain_charge: bool = False,
        charge_scale: float = 1.0,
        exchange_type: str = "heisenberg",
        use_ewald_tf: bool = False,
        conv_function_tf: bool = False,
        ewald_sigma: float = 1.0,
        ewald_dl: float = 2.0,
        ewald_k_chunk_size: int = 50000,
    ):
        super().__init__(backbone)
        self.prefix = prefix
        self.wrap_property = wrap_property

        # Long-range charge module
        self.lr_module = AllScAIPLRChargeModule(
            hidden_size=self.global_cfg.hidden_size,
            hidden_size_lr=hidden_size_lr,
            heisenberg_tf=heisenberg_tf,
            equil_charges_tf=equil_charges_tf,
            constrain_charge=constrain_charge,
            charge_scale=charge_scale,
            exchange_type=exchange_type,
            use_ewald_tf=use_ewald_tf,
            conv_function_tf=conv_function_tf,
            ewald_sigma=ewald_sigma,
            ewald_dl=ewald_dl,
            ewald_k_chunk_size=ewald_k_chunk_size,
        )

        self.post_init()

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.prefix:
            energy_key = f"{self.prefix}_energy"
            forces_key = f"{self.prefix}_forces"
            stress_key = f"{self.prefix}_stress"
        else:
            energy_key = "energy"
            forces_key = "forces"
            stress_key = "stress"

        outputs = {}

        # SR energy (from parent AllScAIPEnergyHead)
        with record_function("grad_head_lr_sr_energy"):
            energy_sr = self.compiled_forward(emb)

        # LR energy
        with record_function("grad_head_lr_charges"):
            node_reps = self.get_node_reps(emb)
            e_charge_single = self.lr_module(node_reps, emb, data)

            e_charge = compilable_scatter(
                e_charge_single,
                index=emb["data"].node_batch,
                dim_size=emb["data"].max_batch_size,
                dim=0,
                reduce=self.energy_reduce,
            )

        if len(energy_sr.shape) == 0:
            energy_sr = energy_sr.unsqueeze(0)
        if len(e_charge.shape) == 0:
            e_charge = e_charge.unsqueeze(0)

        results_to_unpad = {
            "energy": energy_sr,
            "energy_coul": e_charge,
        }
        res_unpad = unpad_results(results=results_to_unpad, data=emb["data"])

        energy_output = res_unpad["energy"] + res_unpad["energy_coul"]

        outputs[energy_key] = (
            {"energy": energy_output} if self.wrap_property else energy_output
        )

        if self.regress_stress:
            with record_function("grad_head_lr_stress_forces"):
                grads = torch.autograd.grad(
                    [energy_output.sum()],
                    [data["pos_original"], emb["displacement"]],
                    create_graph=self.training,
                )
                forces = torch.neg(grads[0])
                virial = grads[1].view(-1, 3, 3)
                volume = torch.det(data["cell"]).abs().unsqueeze(-1)
                stress = virial / volume.view(-1, 1, 1)
                stress = stress.view(-1, 9)
                outputs[forces_key] = (
                    {"forces": forces} if self.wrap_property else forces
                )
                outputs[stress_key] = (
                    {"stress": stress} if self.wrap_property else stress
                )
                data["cell"] = emb["orig_cell"]
        elif self.regress_forces:
            with record_function("grad_head_lr_forces"):
                if data["pos"].requires_grad is False:
                    data["pos"].requires_grad = True
                forces = (
                    -1
                    * torch.autograd.grad(
                        energy_output.sum(),
                        data["pos"],
                        create_graph=self.training,
                    )[0]
                )
                outputs[forces_key] = (
                    {"forces": forces} if self.wrap_property else forces
                )

        return unpad_results(results=outputs, data=emb["data"])
