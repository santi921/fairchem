"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from torch.profiler import record_function

from fairchem.core.common import gp_utils
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.graph.compute import generate_graph
from fairchem.core.models.base import HeadInterface
from fairchem.core.models.uma.common.rotation import (
    eulers_to_wigner,
    init_edge_rot_euler_angles,
)
from fairchem.core.models.uma.common.so3 import CoefficientMapping, SO3_Grid
from fairchem.core.models.uma.nn.embedding import (
    ChgSpinEmbedding,
    DatasetEmbedding,
    EdgeDegreeEmbedding,
)
from fairchem.core.models.uma.nn.execution_backends import get_execution_backend
from fairchem.core.models.uma.nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from fairchem.core.models.uma.nn.mole_utils import MOLEInterface
from fairchem.core.models.uma.nn.radial import GaussianSmearing, PolynomialEnvelope
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear
from fairchem.core.models.utils.lr_charges import LRChargePredictor

from .escn_md import (
    ESCNMD_DEFAULT_EDGE_ACTIVATION_CHECKPOINT_CHUNK_SIZE,
    GradRegressConfig,
    resolve_dataset_mapping,
)
from .escn_md_block import eSCNMD_Block

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData


@registry.register_model("escnmd_backbone_lr")
class eSCNMDBackboneLR(nn.Module, MOLEInterface):
    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        num_sphere_samples: int = 128,
        otf_graph: bool = False,
        max_neighbors: int = 300,
        use_pbc: bool = True,
        use_pbc_single: bool = True,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        distance_function: Literal["gaussian"] = "gaussian",
        num_distance_basis: int = 512,
        direct_forces: bool = True,
        regress_forces: bool = True,
        regress_stress: bool = False,
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate",
        ff_type: str = "grid",
        activation_checkpointing: bool = False,
        chg_spin_emb_type: Literal["pos_emb", "lin_emb", "rand_emb"] = "pos_emb",
        cs_emb_grad: bool = False,
        dataset_emb_grad: bool = False,
        dataset_list: list[str] | None = None,
        use_dataset_embedding: bool = True,
        radius_pbc_version: int = 1,
        always_use_pbc: bool = True,
        edge_chunk_size: int | None = None,
        # LR-specific parameters
        hidden_channels_lr: int = 64,
        heisenberg_tf: bool = False,
        exchange_type: str = "heisenberg",
        latent_charge_tf: bool = True,
        return_bec: bool = False,
        conv_function_tf: bool = True,
        lr_output_scaling_factor: float = 1.0,
        cutoff_lr: float = -1.0,
        normalize_charges_tf: bool = True,
        equil_charges_tf: bool = False,
        use_ewald_tf: bool = False,
    ) -> None:
        super().__init__()
        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.grid_resolution = grid_resolution
        self.num_sphere_samples = num_sphere_samples
        self.always_use_pbc = always_use_pbc
        self.backend = get_execution_backend("general")

        # energy conservation related
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.regress_stress = regress_stress
        self.regress_config = GradRegressConfig(
            direct_forces=direct_forces,
            forces=regress_forces,
            stress=regress_stress,
        )

        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.radius_pbc_version = radius_pbc_version
        self.enforce_max_neighbors_strictly = False

        activation_checkpoint_chunk_size = None
        if activation_checkpointing:
            activation_checkpoint_chunk_size = (
                ESCNMD_DEFAULT_EDGE_ACTIVATION_CHECKPOINT_CHUNK_SIZE
            )
        self.edge_chunk_size = edge_chunk_size

        self.chg_spin_emb_type = chg_spin_emb_type
        self.cs_emb_grad = cs_emb_grad
        self.dataset_emb_grad = dataset_emb_grad
        self.dataset_list = dataset_list
        self.use_dataset_embedding = use_dataset_embedding
        if self.use_dataset_embedding:
            self.dataset_mapping = resolve_dataset_mapping(
                self.dataset_list, None, "dataset_list"
            )

        # rotation utils
        Jd_list = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])
        self.sph_feature_size = int((self.lmax + 1) ** 2)
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)

        self.SO3_grid = nn.ModuleDict()
        self.SO3_grid["lmax_lmax"] = SO3_Grid(
            self.lmax, self.lmax, resolution=grid_resolution, rescale=True
        )
        self.SO3_grid["lmax_mmax"] = SO3_Grid(
            self.lmax, self.mmax, resolution=grid_resolution, rescale=True
        )

        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels
        )

        self.charge_embedding = ChgSpinEmbedding(
            self.chg_spin_emb_type,
            "charge",
            self.sphere_channels,
            grad=self.cs_emb_grad,
        )
        self.spin_embedding = ChgSpinEmbedding(
            self.chg_spin_emb_type,
            "spin",
            self.sphere_channels,
            grad=self.cs_emb_grad,
        )

        if self.use_dataset_embedding:
            self.dataset_embedding = DatasetEmbedding(
                self.sphere_channels,
                enable_grad=self.dataset_emb_grad,
                dataset_mapping=self.dataset_mapping,
            )
            self.mix_csd = nn.Linear(3 * self.sphere_channels, self.sphere_channels)
        else:
            self.mix_csd = nn.Linear(2 * self.sphere_channels, self.sphere_channels)

        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis,
                2.0,
            )
        else:
            raise ValueError("Unknown distance function")

        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.edge_channels_list = [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,
            mappingReduced=self.mappingReduced,
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
            backend=self.backend,
        )

        self.envelope = PolynomialEnvelope(exponent=5)

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.act_type = act_type
        self.ff_type = ff_type

        # LR-specific attributes
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

        if cutoff_lr < 0.0:
            self.cutoff_lr = cutoff
        else:
            self.cutoff_lr = cutoff_lr

        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = eSCNMD_Block(
                self.sphere_channels,
                self.hidden_channels,
                self.lmax,
                self.mmax,
                self.mappingReduced,
                self.SO3_grid,
                self.edge_channels_list,
                self.cutoff,
                self.norm_type,
                self.act_type,
                self.ff_type,
                activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
                backend=self.backend,
            )
            self.blocks.append(block)

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels,
        )

        coefficient_index = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )
        self.register_buffer("coefficient_index", coefficient_index, persistent=False)

    def _get_rotmat_and_wigner(
        self, edge_distance_vecs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Jd_buffers = [
            getattr(self, f"Jd_{l}").type(edge_distance_vecs.dtype)
            for l in range(self.lmax + 1)
        ]

        with record_function("obtain rotmat wigner original"):
            euler_angles = init_edge_rot_euler_angles(edge_distance_vecs)
            wigner = eulers_to_wigner(
                euler_angles,
                0,
                self.lmax,
                Jd_buffers,
            )
            wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        if self.mmax != self.lmax:
            wigner = wigner.index_select(1, self.coefficient_index)
            wigner_inv = wigner_inv.index_select(2, self.coefficient_index)

        wigner_and_M_mapping = torch.einsum(
            "mk,nkj->nmj", self.mappingReduced.to_m.to(wigner.dtype), wigner
        )
        wigner_and_M_mapping_inv = torch.einsum(
            "njk,mk->njm", wigner_inv, self.mappingReduced.to_m.to(wigner_inv.dtype)
        )
        return wigner_and_M_mapping, wigner_and_M_mapping_inv

    def _get_displacement_and_cell(
        self, data_dict: AtomicData
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        displacement = None
        orig_cell = None
        if self.regress_stress and not self.direct_forces:
            displacement = torch.zeros(
                (3, 3),
                dtype=data_dict["pos"].dtype,
                device=data_dict["pos"].device,
            )
            num_batch = len(data_dict["natoms"])
            displacement = displacement.view(-1, 3, 3).expand(num_batch, 3, 3)
            displacement.requires_grad = True
            symmetric_displacement = 0.5 * (
                displacement + displacement.transpose(-1, -2)
            )
            if data_dict["pos"].requires_grad is False:
                data_dict["pos"].requires_grad = True
            data_dict["pos_original"] = data_dict["pos"]
            data_dict["pos"] = data_dict["pos"] + torch.bmm(
                data_dict["pos"].unsqueeze(-2),
                torch.index_select(symmetric_displacement, 0, data_dict["batch"]),
            ).squeeze(-2)

            orig_cell = data_dict["cell"]
            data_dict["cell"] = data_dict["cell"] + torch.bmm(
                data_dict["cell"], symmetric_displacement
            )

        if (
            not self.regress_stress
            and self.regress_forces
            and not self.direct_forces
            and data_dict["pos"].requires_grad is False
        ):
            data_dict["pos"].requires_grad = True
        return displacement, orig_cell

    def csd_embedding(self, charge, spin, dataset):
        with record_function("charge spin dataset embeddings"):
            chg_emb = self.charge_embedding(charge)
            spin_emb = self.spin_embedding(spin)
            if self.use_dataset_embedding:
                assert dataset is not None
                dataset_emb = self.dataset_embedding(dataset)
                return torch.nn.SiLU()(
                    self.mix_csd(torch.cat((chg_emb, spin_emb, dataset_emb), dim=1))
                )
            return torch.nn.SiLU()(self.mix_csd(torch.cat((chg_emb, spin_emb), dim=1)))

    def _generate_graph(self, data_dict):
        if self.otf_graph:
            pbc = None
            if self.always_use_pbc:
                pbc = torch.ones(len(data_dict), 3, dtype=torch.bool)
            else:
                assert (
                    "pbc" in data_dict
                ), "Since always_use_pbc is False, pbc conditions must be supplied by the input data"
                pbc = data_dict["pbc"]
            assert (
                pbc.all() or (~pbc).all()
            ), "We can only accept pbc that is all true or all false"
            logging.debug(f"Using radius graph gen version {self.radius_pbc_version}")

            graph_dict = generate_graph(
                data_dict,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
                radius_pbc_version=self.radius_pbc_version,
                pbc=pbc,
                cutoff_lr=self.cutoff_lr,
            )
        else:
            assert (
                "edge_index" in data_dict
            ), "otf_graph is false, need to provide edge_index as input!"
            cell_per_edge = data_dict["cell"].repeat_interleave(
                data_dict["nedges"], dim=0
            )
            shifts = torch.einsum(
                "ij,ijk->ik",
                data_dict["cell_offsets"].to(cell_per_edge.dtype),
                cell_per_edge,
            )
            edge_distance_vec = (
                data_dict["pos"][data_dict["edge_index"][0]]
                - data_dict["pos"][data_dict["edge_index"][1]]
                + shifts
            )
            edge_distance = torch.linalg.norm(edge_distance_vec, dim=-1, keepdim=False)

            graph_dict = {
                "edge_index": data_dict["edge_index"],
                "edge_distance": edge_distance,
                "edge_distance_vec": edge_distance_vec,
            }
        graph_dict["node_offset"] = 0

        if gp_utils.initialized():
            graph_dict = self._init_gp_partitions(
                graph_dict, data_dict["atomic_numbers_full"]
            )
            data_dict["atomic_numbers"] = data_dict["atomic_numbers_full"][
                graph_dict["node_partition"]
            ]
            data_dict["batch"] = data_dict["batch_full"][graph_dict["node_partition"]]

        return graph_dict

    @conditional_grad(torch.enable_grad())
    def forward(self, data_dict: AtomicData) -> dict[str, torch.Tensor]:
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()
        data_dict["atomic_numbers_full"] = data_dict["atomic_numbers"]
        data_dict["batch_full"] = data_dict["batch"]

        csd_mixed_emb = self.csd_embedding(
            charge=data_dict["charge"],
            spin=data_dict["spin"],
            dataset=data_dict.get("dataset", default=None),
        )

        self.set_MOLE_coefficients(
            atomic_numbers_full=data_dict["atomic_numbers_full"],
            batch_full=data_dict["batch_full"],
            csd_mixed_emb=csd_mixed_emb,
        )

        with record_function("get_displacement_and_cell"):
            displacement, orig_cell = self._get_displacement_and_cell(data_dict)

        with record_function("generate_graph"):
            graph_dict = self._generate_graph(data_dict)

        if graph_dict["edge_index"].numel() == 0:
            raise ValueError(
                f"No edges found in input system, this means either you have a single "
                f"atom in the system or the atoms are farther apart than the radius "
                f"cutoff of the model of {self.cutoff} Angstroms."
            )

        with record_function("obtain wigner"):
            (wigner_and_M_mapping, wigner_and_M_mapping_inv) = (
                self._get_rotmat_and_wigner(
                    graph_dict["edge_distance_vec"],
                )
            )

        with record_function("atom embedding"):
            x_message = torch.zeros(
                data_dict["atomic_numbers"].shape[0],
                self.sph_feature_size,
                self.sphere_channels,
                device=data_dict["pos"].device,
                dtype=data_dict["pos"].dtype,
            )
            x_message[:, 0, :] = self.sphere_embedding(data_dict["atomic_numbers"])

        sys_node_embedding = csd_mixed_emb[data_dict["batch"]]
        x_message[:, 0, :] = x_message[:, 0, :] + sys_node_embedding

        self.set_MOLE_sizes(
            nsystems=csd_mixed_emb.shape[0],
            batch_full=data_dict["batch_full"],
            edge_index=graph_dict["edge_index"],
        )
        self.log_MOLE_stats()

        with record_function("edge embedding"):
            dist_scaled = graph_dict["edge_distance"] / self.cutoff
            edge_envelope = self.envelope(dist_scaled).reshape(-1, 1, 1)
            edge_distance_embedding = self.distance_expansion(
                graph_dict["edge_distance"]
            )
            source_embedding = self.source_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][0]]
            )
            target_embedding = self.target_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][1]]
            )
            x_edge = torch.cat(
                (edge_distance_embedding, source_embedding, target_embedding), dim=1
            )
            # Pre-fuse envelope into wigner_inv
            wigner_inv_envelope = wigner_and_M_mapping_inv * edge_envelope
            # The LR backbone does not support graph parallelism, so the
            # scatter target is always the raw edge target (edge_index[1])
            scatter_target = graph_dict["edge_index"][1]
            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                scatter_target,
                wigner_inv_envelope,
            )

        for i in range(self.num_layers):
            with record_function(f"message passing {i}"):
                x_message = self.blocks[i](
                    x_message,
                    x_edge,
                    graph_dict["edge_index"],
                    wigner_and_M_mapping,
                    wigner_inv_envelope,
                    total_atoms_across_gp_ranks=data_dict["atomic_numbers_full"].shape[
                        0
                    ],
                    sys_node_embedding=sys_node_embedding,
                    scatter_target=scatter_target,
                )

        x_message = self.norm(x_message)
        out = {
            "node_embedding": x_message,
            "displacement": displacement,
            "orig_cell": orig_cell,
            "batch": data_dict["batch"],
        }

        if self.cutoff_lr is not None:
            out["edge_index_lr"] = graph_dict.get("edge_index_lr")

        return out

    def _init_gp_partitions(self, graph_dict, atomic_numbers_full):
        edge_index = graph_dict["edge_index"]

        node_partition = torch.tensor_split(
            torch.arange(len(atomic_numbers_full)).to(atomic_numbers_full.device),
            gp_utils.get_gp_world_size(),
        )[gp_utils.get_gp_rank()]
        assert node_partition.numel() > 0, "No atoms in this graph parallel partition."

        edge_partition = torch.where(
            torch.logical_and(
                edge_index[1] >= node_partition.min(),
                edge_index[1] <= node_partition.max(),
            )
        )[0]
        graph_dict["node_offset"] = node_partition.min().item()
        graph_dict["node_partition"] = node_partition
        graph_dict["edge_index"] = edge_index[:, edge_partition]
        graph_dict["edge_distance"] = graph_dict["edge_distance"][edge_partition]
        graph_dict["edge_distance_vec"] = graph_dict["edge_distance_vec"][
            edge_partition
        ]
        return graph_dict

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_Linear,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, SO3_Linear))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)


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
    node_energy = energy_block(emb["node_embedding"].narrow(1, 0, 1).squeeze()).view(-1)

    energy = torch.zeros(
        len(data["natoms"]),
        device=node_energy.device,
        dtype=node_energy.dtype,
    )
    energy.index_add_(0, data["batch"], node_energy)

    lr_energy_dict = None
    if latent_charge_tf and lr_predictor is not None:
        lr_energy_dict = lr_predictor.get_lr_energies(emb, data)
        energy.index_add_(0, data["batch"], lr_energy_dict["energy"])

    if heisenberg_tf and lr_energy_dict is not None:
        energy.index_add_(0, data["batch"], lr_energy_dict["energy_spin"])

    return energy, lr_energy_dict


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
        self.regress_stress = backbone.regress_stress
        self.regress_forces = backbone.regress_forces
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

        backbone.direct_forces = False
        assert (
            not backbone.direct_forces
        ), "EFS head is only used for gradient-based forces/stress."

    @conditional_grad(torch.enable_grad())
    def forward(
        self, data: AtomicData, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.prefix:
            energy_key = f"{self.prefix}_energy"
            forces_key = f"{self.prefix}_forces"
            stress_key = f"{self.prefix}_stress"
        else:
            energy_key = "energy"
            forces_key = "forces"
            stress_key = "stress"

        outputs = {}
        energy_part, _ = _compute_energy_with_lr(
            self.energy_block,
            self.lr_predictor,
            emb,
            data,
            self.latent_charge_tf,
            self.heisenberg_tf,
        )

        if gp_utils.initialized():
            energy = gp_utils.reduce_from_model_parallel_region(energy_part)
        else:
            energy = energy_part

        outputs[energy_key] = {"energy": energy} if self.wrap_property else energy

        if not gp_utils.initialized():
            embeddings = emb["node_embedding"].detach()
            outputs["embeddings"] = (
                {"embeddings": embeddings} if self.wrap_property else embeddings
            )

        if self.regress_stress:
            grads = torch.autograd.grad(
                [energy_part.sum()],
                [data["pos_original"], emb["displacement"]],
                create_graph=self.training,
            )
            if gp_utils.initialized():
                grads = (
                    gp_utils.reduce_from_model_parallel_region(grads[0]),
                    gp_utils.reduce_from_model_parallel_region(grads[1]),
                )

            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(data["cell"]).abs().unsqueeze(-1)
            stress = virial / volume.view(-1, 1, 1)
            stress = stress.view(-1, 9)
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
            outputs[stress_key] = {"stress": stress} if self.wrap_property else stress
            data["cell"] = emb["orig_cell"]

        elif self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    energy_part.sum(), data["pos"], create_graph=self.training
                )[0]
            )
            if gp_utils.initialized():
                forces = gp_utils.reduce_from_model_parallel_region(forces)
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
class Linear_Energy_Head_LR(nn.Module, HeadInterface):
    """
    Energy-only head with linear energy block and long-range electrostatics.
    """

    def __init__(self, backbone: eSCNMDBackboneLR, reduce: str = "sum") -> None:
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
