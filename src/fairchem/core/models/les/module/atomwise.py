"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from .blocks import Dense, build_mlp

__all__ = ["Atomwise"]


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    Supports lazy initialization: if ``n_in`` is not provided at construction,
    the MLP is built on the first ``forward()`` call using the input feature
    dimension.
    """

    def __init__(
        self,
        n_in: int | None = None,  # input dimension of representation
        n_out: int = 1,
        n_hidden: int | Sequence[int] | None = None,
        n_layers: int = 2,
        bias: bool = True,
        activation: Callable = F.silu,
        add_linear_nn: bool = False,
        output_scaling_factor: float = 1.0,
    ):
        """
        Args:
            n_in: input dimension of representation. If None, inferred on first
                forward call (lazy initialization).
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers
                resulting in a rectangular network.
                If None, the number of neurons is divided by two after each layer
                starting n_in resulting in a pyramidal network.
            n_layers: number of layers.
            add_linear_nn: whether to add a linear NN to the output of the MLP
        """
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.add_linear_nn = add_linear_nn
        self.bias = bias
        self.output_scaling_factor = output_scaling_factor

        # Build networks eagerly if n_in is known, otherwise defer
        if self.n_in is not None:
            self._build_networks(self.n_in)
        else:
            self.outnet = None
            self.linear_nn = None

    def _build_networks(self, n_in: int) -> None:
        """
        Build the MLP and optional linear skip connection.
        """
        self.n_in = n_in
        self.outnet = build_mlp(
            n_in=n_in,
            n_out=self.n_out,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            activation=self.activation,
            bias=self.bias,
        )

        if self.add_linear_nn:
            self.linear_nn = Dense(
                n_in,
                self.n_out,
                bias=self.bias,
                activation=None,
            )
        else:
            self.linear_nn = None

    def forward(
        self,
        desc: torch.Tensor,  # [n_atoms, n_features]
        batch: torch.Tensor,  # [n_atoms]
        training: bool | None = None,
    ) -> torch.Tensor:
        # Lazy initialization on first call
        if self.outnet is None:
            self._build_networks(desc.shape[-1])
            # Move newly created submodules to the input device
            self.outnet = self.outnet.to(desc.device)
            if self.linear_nn is not None:
                self.linear_nn = self.linear_nn.to(desc.device)

        # predict atomwise contributions
        y = self.outnet(desc)
        if self.add_linear_nn:
            y += self.linear_nn(desc)

        return y * self.output_scaling_factor

    def __repr__(self):
        return (
            f"Atomwise(n_in={self.n_in}, n_out={self.n_out}, "
            f"n_hidden={self.n_hidden}, n_layers={self.n_layers}, "
            f"bias={self.bias})"
        )
