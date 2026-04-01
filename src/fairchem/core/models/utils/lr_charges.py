"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from fairchem.core.models.les.module import Ewald
from fairchem.core.models.utils.lr import (
    batch_spin_charge_renormalization,
    heisenberg_potential_full_from_edge_inds,
    potential_full_from_edge_inds,
)

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData


class LRChargePredictor(nn.Module):
    """
    Shared module for long-range charge prediction and energy computation.

    Owns the charge MLP, optional equilibration networks, optional
    Heisenberg coupling NN, and the LES Ewald module for periodic systems.
    Provides ``get_charges()`` and ``get_lr_energies()`` used by all
    LR head classes.
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels_lr: int,
        lr_comp_size: int = 1,
        lr_output_scaling_factor: float = 1.0,
        normalize_charges_tf: bool = True,
        equil_charges_tf: bool = False,
        heisenberg_tf: bool = False,
        exchange_type: str = "heisenberg",
        use_ewald_tf: bool = False,
        conv_function_tf: bool = True,
        return_bec: bool = False,
        ewald_sigma: float = 1.0,
        ewald_dl: float = 2.0,
        ewald_k_chunk_size: int = 50000,
    ):
        super().__init__()
        self.lr_comp_size = lr_comp_size
        self.lr_output_scaling_factor = lr_output_scaling_factor
        self.normalize_charges_tf = normalize_charges_tf
        self.equil_charges_tf = equil_charges_tf
        self.heisenberg_tf = heisenberg_tf
        self.exchange_type = exchange_type
        self.use_ewald_tf = use_ewald_tf
        self.conv_function_tf = conv_function_tf
        self.return_bec = return_bec

        if self.heisenberg_tf and self.lr_comp_size != 2:
            raise ValueError(
                "heisenberg_tf requires lr_comp_size=2 (alpha/beta spin channels)"
            )

        # Charge prediction MLP
        self.q_output_lr = nn.Sequential(
            nn.Linear(sphere_channels, hidden_channels_lr, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_channels_lr, hidden_channels_lr, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_channels_lr, lr_comp_size, bias=True),
        )

        # Electronegativity equilibration networks
        if self.equil_charges_tf:
            self.hardness_output_lr = nn.Sequential(
                nn.Linear(sphere_channels, hidden_channels_lr, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_channels_lr, hidden_channels_lr, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_channels_lr, 1, bias=True),
            )
            self.electroneg_output_lr = nn.Sequential(
                nn.Linear(sphere_channels, hidden_channels_lr, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_channels_lr, hidden_channels_lr, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_channels_lr, 1, bias=True),
            )

        # Heisenberg spin coupling NN
        if self.heisenberg_tf:
            self.coupling_nn = nn.Sequential(
                nn.Linear(1, hidden_channels_lr, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_channels_lr, hidden_channels_lr, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_channels_lr, 1, bias=True),
            )

        # LES Ewald for periodic systems
        if self.use_ewald_tf:
            self.ewald = Ewald(
                sigma=ewald_sigma,
                dl=ewald_dl,
                k_chunk_size=ewald_k_chunk_size,
            )

    @staticmethod
    def _normalize_single_channel(
        charges_raw: torch.Tensor,
        batch: torch.Tensor,
        charge_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Additive charge normalization: distribute residual uniformly.
        """
        flat = charges_raw.view(-1)
        num_batches = batch.max() + 1

        predicted_total = torch.zeros(num_batches, device=flat.device, dtype=flat.dtype)
        predicted_total.scatter_add_(0, batch, flat)

        ones = torch.ones_like(flat)
        natoms_per_batch = torch.zeros(
            num_batches, device=flat.device, dtype=flat.dtype
        )
        natoms_per_batch.scatter_add_(0, batch, ones)

        target_total = charge_targets.view(-1)[:num_batches]
        residual = target_total - predicted_total
        shift_per_atom = residual / natoms_per_batch
        corrected = flat + shift_per_atom[batch]

        return corrected.view_as(charges_raw)

    def get_charges(
        self,
        node_features: torch.Tensor,
        data: AtomicData,
    ) -> dict[str, torch.Tensor]:
        """
        Predict per-atom charges, with optional spin channels and normalization.

        Returns dict with keys: "charges", and optionally "charges_raw",
        "net_partial_spin", "hardness", "electroneg".
        """
        results = {}
        with torch.enable_grad():
            charges_raw = self.q_output_lr(node_features)

            if self.equil_charges_tf:
                hardness = self.hardness_output_lr(node_features)
                electroneg = self.electroneg_output_lr(node_features)
                results["hardness"] = hardness.view(-1)
                results["electroneg"] = electroneg.view(-1)

        if self.lr_comp_size == 1:
            if self.normalize_charges_tf:
                charges_raw = self._normalize_single_channel(
                    charges_raw,
                    data["batch"],
                    data["charge"],
                )
            results["charges"] = (
                charges_raw.view(-1, 1, 1) * self.lr_output_scaling_factor
            )

        if self.lr_comp_size == 2:
            results["charges"] = (
                charges_raw.sum(dim=1).view(-1, 1, 1) * self.lr_output_scaling_factor
            )
            results["charges_raw"] = charges_raw * self.lr_output_scaling_factor
            alpha = results["charges_raw"][:, 0]
            beta = results["charges_raw"][:, 1]
            results["net_partial_spin"] = (alpha - beta).view(-1, 1, 1)

            if self.normalize_charges_tf:
                charges_renorm = batch_spin_charge_renormalization(
                    charges_raw=results["charges_raw"],
                    batch=data["batch"],
                    s_total=data["spin"],
                    q_total=data["charge"],
                )
                results["charges_raw"] = charges_renorm
                results["charges"] = charges_renorm.sum(dim=1).view(-1, 1, 1)
                results["net_partial_spin"] = (
                    charges_renorm[:, 0] - charges_renorm[:, 1]
                ).view(-1, 1, 1)

        return results

    def get_lr_energies(
        self,
        emb: dict[str, torch.Tensor],
        data: AtomicData,
        return_charges: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Compute long-range electrostatic (and optionally spin) energies.
        """
        results: dict[str, torch.Tensor] = {}

        charge_dict = self.get_charges(
            emb["node_embedding"].narrow(1, 0, 1).squeeze(),
            data,
        )

        if "edge_index_lr" in emb:
            edges_lr = emb["edge_index_lr"]
        else:
            edges_lr = emb["edge_index"]

        # Determine whether to use direct sum or Ewald
        use_direct_sum = True
        if data["cell"] is not None and self.use_ewald_tf:
            det_cells = torch.linalg.det(data["cell"])
            if not torch.any(det_cells < 1e-6):
                use_direct_sum = False

        if use_direct_sum:
            energy_output_lr_dict = potential_full_from_edge_inds(
                edge_index=edges_lr,
                pos=data["pos"],
                q=charge_dict["charges"],
                sigma=1.0,
                epsilon=1e-6,
                return_bec=self.return_bec,
                batch=data["batch"],
                conv_function_tf=self.conv_function_tf,
            )
            results["energy"] = energy_output_lr_dict["potential"]
        else:
            # Use LES Ewald module for periodic systems
            ewald_energy = self.ewald(
                q=charge_dict["charges"].view(-1, 1),
                r=data["pos"],
                cell=data["cell"],
                batch=data["batch"],
            )
            # Ewald returns per-batch scalar; distribute to per-atom
            device = data["pos"].device
            dtype = data["pos"].dtype
            n_atoms = data["pos"].shape[0]
            n_batches = data["cell"].shape[0]
            ones = torch.ones(n_atoms, device=device, dtype=dtype)
            natoms_per_batch = torch.zeros(n_batches, device=device, dtype=dtype)
            natoms_per_batch.scatter_add_(0, data["batch"], ones)
            per_batch_energy = ewald_energy.view(-1)
            per_atom_energy = per_batch_energy / natoms_per_batch
            results["energy"] = per_atom_energy[data["batch"]]

        # Equilibration energy terms
        if self.equil_charges_tf:
            en_electrostatic = charge_dict["electroneg"].view(-1) * charge_dict[
                "charges"
            ].view(-1)
            en_hardness = 0.5 * (
                charge_dict["hardness"].view(-1) * charge_dict["charges"].view(-1) ** 2
            )
            results["energy"] = results["energy"] + en_electrostatic + en_hardness

        # Heisenberg spin coupling
        if self.heisenberg_tf:
            energy_spin = heisenberg_potential_full_from_edge_inds(
                edge_index=edges_lr,
                q=charge_dict["charges_raw"],
                pos=data["pos"],
                nn=self.coupling_nn,
                exchange_type=self.exchange_type,
            )
            results["energy_spin"] = energy_spin

        if return_charges:
            results["charges"] = charge_dict["charges"]
            if self.lr_comp_size == 2:
                results["spin"] = charge_dict.get("net_partial_spin")

        return results
