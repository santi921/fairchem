"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Ewald"]


class Ewald(nn.Module):
    """
    Ewald summation for long-range electrostatics.

    Supports multi-channel charges (n_q > 1) and optional chunked k-vector
    processing to reduce peak memory.

    Args:
        dl: grid resolution for k-space sampling.
        sigma: Gaussian width on each atom.
        remove_self_interaction: subtract self-energy term.
        norm_factor: 1/(2*epsilon_0) in eV*Angstrom units.
        k_chunk_size: if set, process k-vectors in chunks of this size
            to limit peak memory. None means no chunking.
    """

    def __init__(
        self,
        dl: float = 2.0,
        sigma: float = 1.0,
        remove_self_interaction: bool = True,
        norm_factor: float = 90.0474,
        k_chunk_size: int | None = None,
    ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.sigma_sq_half = sigma**2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi**2
        self.remove_self_interaction = remove_self_interaction
        # 1/2\epsilon_0, where \epsilon_0 = 5.55263e-3 e^2 eV^{-1} A^{-1}
        self.norm_factor = norm_factor
        self.k_sq_max = (self.twopi / self.dl) ** 2
        self.k_chunk_size = k_chunk_size

    def forward(
        self,
        q: torch.Tensor,  # [n_atoms, n_q] or [n_atoms]
        r: torch.Tensor,  # [n_atoms, 3]
        cell: torch.Tensor,  # [batch_size, 3, 3]
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q.dim() == 1:
            q = q.unsqueeze(1)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, "r dimension error"
        assert n == q.size(0), "q dimension error"
        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)

        unique_batches = torch.unique(batch)

        results = []
        for i in unique_batches:
            mask = batch == i
            r_raw_now, q_now = r[mask], q[mask]

            if cell is not None:
                box_now = cell[i]

            # check if the box is periodic or not
            if cell is None or torch.linalg.det(box_now) < 1e-6:
                pot = self.compute_potential_realspace(r_raw_now, q_now)
            else:
                pot = self.compute_potential_triclinic(r_raw_now, q_now, box_now)
            results.append(pot)

        return torch.stack(results, dim=0).sum(dim=1)

    def compute_potential_realspace(self, r_raw, q):
        """
        Direct-space Coulomb sum with error function convergence.
        Supports multi-q charges: q shape [n_atoms, n_q].
        """
        epsilon = 1e-6
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        torch.diagonal(r_ij).add_(epsilon)
        r_ij_norm = torch.norm(r_ij, dim=-1)

        convergence_func_ij = torch.special.erf(r_ij_norm / self.sigma / (2.0**0.5))
        r_p_ij = 1.0 / r_ij_norm

        if q.dim() == 1:
            q = q.unsqueeze(1)

        # [1, n, n_q] * [n, 1, n_q] * [n, n, 1] * [n, n, 1]
        pot = (
            q.unsqueeze(0)
            * q.unsqueeze(1)
            * r_p_ij.unsqueeze(2)
            * convergence_func_ij.unsqueeze(2)
        )

        # Exclude diagonal terms
        mask = ~torch.eye(pot.shape[0], device=pot.device).to(torch.bool).unsqueeze(-1)
        mask = torch.vstack([mask.transpose(0, -1)] * pot.shape[-1]).transpose(0, -1)
        pot = pot[mask].sum().view(-1) / self.twopi / 2.0

        if not self.remove_self_interaction:
            pot += torch.sum(q**2) / (self.sigma * self.twopi ** (3.0 / 2.0))

        return pot * self.norm_factor

    @torch.compiler.disable
    def _generate_kvectors(self, cell_now, device):
        """
        Generate filtered k-vectors for one batch element.

        Returns (kvec, k_sq, factors) or (None, None, None) if no valid k-vectors.
        """
        cell_inv = torch.linalg.inv(cell_now)
        G = 2 * torch.pi * cell_inv.T

        norms = torch.norm(cell_now, dim=1)
        Nk = [max(1, int(n.item() / self.dl)) for n in norms]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)

        nvec = (
            torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1)
            .reshape(-1, 3)
            .to(G.dtype)
        )
        kvec = nvec @ G
        k_sq = torch.sum(kvec**2, dim=1)

        # Filter: nonzero and within cutoff
        valid_mask = (k_sq > 0) & (k_sq <= self.k_sq_max)
        kvec = kvec[valid_mask]
        k_sq = k_sq[valid_mask]
        nvec = nvec[valid_mask]

        if kvec.numel() == 0:
            return None, None, None

        # Hemisphere masking: keep only k-vectors with positive first nonzero index
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze(1)
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        nvec_hemi = nvec[hemisphere_mask]

        if kvec.numel() == 0:
            return None, None, None

        factors = torch.where((nvec_hemi == 0).all(dim=1), 1.0, 2.0)
        return kvec, k_sq, factors

    def compute_potential_triclinic(self, r_raw, q, cell_now):
        """
        Reciprocal-space Ewald sum for a triclinic (or orthorhombic) cell.
        Supports multi-q charges: q shape [n_atoms, n_q].
        Optionally uses chunked k-vector processing to limit peak memory.
        """
        device = r_raw.device

        kvec, k_sq, factors = self._generate_kvectors(cell_now, device)

        if kvec is None:
            # No valid k-vectors — return zero energy
            n_q = q.shape[1] if q.dim() > 1 else 1
            return torch.zeros(n_q, device=device) * self.norm_factor

        if q.dim() == 1:
            q = q.unsqueeze(1)

        # Compute kfac: exp(-sigma^2/2 * k^2) / k^2
        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        weighted = factors * kfac  # [n_k]

        volume = torch.det(cell_now)
        n_k = kvec.shape[0]
        chunk = self.k_chunk_size or n_k  # no chunking if k_chunk_size is None

        # Accumulate structure factor energy
        pot = torch.tensor(0.0, device=device)

        for start in range(0, n_k, chunk):
            end = min(start + chunk, n_k)
            k_dot_r = torch.matmul(r_raw, kvec[start:end].T)  # [n_atoms, chunk]

            cos_kr = torch.cos(k_dot_r)
            sin_kr = torch.sin(k_dot_r)

            # Multi-q structure factor: q [n_atoms, n_q], trig [n_atoms, chunk]
            # -> S [n_q, chunk]
            S_k_real = (q.unsqueeze(2) * cos_kr.unsqueeze(1)).sum(dim=0)
            S_k_imag = (q.unsqueeze(2) * sin_kr.unsqueeze(1)).sum(dim=0)
            S_k_sq = S_k_real**2 + S_k_imag**2  # [n_q, chunk]

            pot = pot + (weighted[start:end] * S_k_sq).sum()

        pot = pot / volume

        # Remove self-interaction if applicable
        if self.remove_self_interaction:
            pot = pot - torch.sum(q**2) / (self.sigma * (2 * torch.pi) ** 1.5)

        return pot.unsqueeze(0) * self.norm_factor

    def __repr__(self):
        return (
            f"Ewald(dl={self.dl}, sigma={self.sigma}, "
            f"remove_self_interaction={self.remove_self_interaction}, "
            f"k_chunk_size={self.k_chunk_size})"
        )
