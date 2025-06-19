"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from .uma.escn_md import (
    MLP_EFS_Head,
    MLP_EFS_Head_LR,
    MLP_Energy_Head,
    MLP_Energy_Head_LR,
    MLP_Stress_Head,
    eSCNMDBackbone,
    eSCNMDBackboneLR,
)
from .uma.escn_moe import eSCNMDMoeBackbone, eSCNMDMoeBackboneLR

torch.set_float32_matmul_precision("high")

__all__ = [
    "MLP_EFS_Head_LR",
    "MLP_Energy_Head",
    "MLP_Energy_Head_LR",
    "MLP_Stress_Head",
    "MLP_Stress_Head_LR",
    "eSCNMDBackbone",
    "eSCNMDBackboneLRMLP_EFS_Head",
    "eSCNMDMoeBackbone",
    "eSCNMDMoeBackboneLR",
]
