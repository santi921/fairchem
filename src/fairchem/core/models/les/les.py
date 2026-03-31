"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .module import BEC, Atomwise, Ewald

__all__ = ["Les"]


class Les(nn.Module):
    def __init__(
        self,
        n_in=None,  # input dimension of representation
        n_layers: int = 3,
        n_hidden: int | list | None = None,
        add_linear_nn: bool = True,
        output_scaling_factor: float = 0.1,
        sigma: float = 1.0,
        dl: float = 2.0,
        remove_mean: bool = True,
        epsilon_factor: float = 1.0,
        use_atomwise: bool = True,
        remove_self_interaction: bool = True,
        k_chunk_size: int | None = None,
        les_arguments: dict[str, Any] | None = None,
    ):
        """
        LES model for long-range interactions.

        Args:
            n_in: input dimension of representation (None for lazy init).
            n_layers: number of MLP layers for charge prediction.
            n_hidden: hidden layer sizes.
            add_linear_nn: add linear skip connection.
            output_scaling_factor: scale factor for predicted charges.
            sigma: Gaussian width for Ewald splitting.
            dl: grid resolution for k-space.
            remove_mean: subtract mean charges before BEC.
            epsilon_factor: relative permittivity for BEC.
            use_atomwise: use Atomwise MLP for charge prediction.
            remove_self_interaction: subtract Ewald self-energy.
            k_chunk_size: chunk size for k-vector processing (memory opt).
            les_arguments: dict to override all parameters.
        """
        super().__init__()
        if n_hidden is None:
            n_hidden = [32, 16]

        if les_arguments is not None:
            self._parse_arguments(les_arguments)
        else:
            self.n_in = n_in
            self.les_arguments = {}
            self.n_layers = n_layers
            self.n_hidden = n_hidden
            self.add_linear_nn = add_linear_nn
            self.output_scaling_factor = output_scaling_factor
            self.sigma = sigma
            self.dl = dl
            self.remove_mean = remove_mean
            self.epsilon_factor = epsilon_factor
            self.use_atomwise = use_atomwise
            self.remove_self_interaction = remove_self_interaction
            self.k_chunk_size = k_chunk_size

        self.atomwise: nn.Module = (
            Atomwise(
                n_in=self.n_in,
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                add_linear_nn=self.add_linear_nn,
                output_scaling_factor=self.output_scaling_factor,
            )
            if self.use_atomwise
            else _DummyAtomwise()
        )

        self.ewald = Ewald(
            sigma=self.sigma,
            dl=self.dl,
            remove_self_interaction=self.remove_self_interaction,
            k_chunk_size=self.k_chunk_size,
        )

        self.bec = BEC(
            remove_mean=self.remove_mean,
            epsilon_factor=self.epsilon_factor,
        )

    def _parse_arguments(self, les_arguments: dict[str, Any]):
        """
        Parse arguments for LES model.
        """
        self.n_in = les_arguments.get("n_in")
        self.n_layers = les_arguments.get("n_layers", 3)
        self.n_hidden = les_arguments.get("n_hidden", [32, 16])
        self.add_linear_nn = les_arguments.get("add_linear_nn", True)
        self.output_scaling_factor = les_arguments.get("output_scaling_factor", 0.1)

        self.sigma = les_arguments.get("sigma", 1.0)
        self.dl = les_arguments.get("dl", 2.0)

        self.remove_mean = les_arguments.get("remove_mean", True)
        self.epsilon_factor = les_arguments.get("epsilon_factor", 1.0)
        self.use_atomwise = les_arguments.get("use_atomwise", True)
        self.remove_self_interaction = les_arguments.get(
            "remove_self_interaction", True
        )
        self.k_chunk_size = les_arguments.get("k_chunk_size")

    @classmethod
    def from_yaml(cls, path: str) -> Les:
        """
        Create a Les instance from a YAML configuration file.
        """
        import yaml

        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(les_arguments=config or {})

    def forward(
        self,
        positions: torch.Tensor,  # [n_atoms, 3]
        cell: torch.Tensor,  # [batch_size, 3, 3]
        desc: torch.Tensor | None = None,  # [n_atoms, n_features]
        latent_charges: torch.Tensor | None = None,  # [n_atoms, ]
        batch: torch.Tensor | None = None,
        compute_energy: bool = True,
        compute_bec: bool = False,
        bec_output_index: int | None = None,
        sid: str | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """
        Forward pass.

        Args:
            positions: atom positions [n_atoms, 3].
            cell: unit cells [batch_size, 3, 3].
            desc: atom descriptors [n_atoms, n_features].
            latent_charges: pre-computed charges [n_atoms].
            batch: batch indices [n_atoms].
            compute_energy: compute Ewald energy.
            compute_bec: compute Born effective charges.
            bec_output_index: restrict BEC to one Cartesian component.
            sid: optional system identifier.
        """
        if batch is None:
            batch = torch.zeros(
                positions.shape[0], dtype=torch.int64, device=positions.device
            )

        if latent_charges is not None:
            assert latent_charges.shape[0] == positions.shape[0]
        elif desc is not None and latent_charges is None:
            if not self.use_atomwise:
                raise ValueError(
                    "desc must be provided and use_atomwise must be True "
                    "if latent_charges is not provided"
                )
            assert desc.shape[0] == positions.shape[0]
            latent_charges = self.atomwise(desc, batch)
        else:
            raise ValueError("Either desc or latent_charges must be provided")

        if compute_energy:
            E_lr = self.ewald(
                q=latent_charges,
                r=positions,
                cell=cell,
                batch=batch,
            )
        else:
            E_lr = None

        if compute_bec:
            bec = self.bec(
                q=latent_charges,
                r=positions,
                cell=cell,
                batch=batch,
                output_index=bec_output_index,
            )
        else:
            bec = None

        output = {
            "E_lr": E_lr,
            "latent_charges": latent_charges,
            "BEC": bec,
        }
        return output


class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        raise ValueError("set use_atomwise to True to use Atomwise module")
