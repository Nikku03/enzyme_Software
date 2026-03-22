from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class QuantumObservableBundle:
    f_plus: torch.Tensor
    f_minus: torch.Tensor
    f_dual: torch.Tensor
    mesp: torch.Tensor
    density_gradient: torch.Tensor
    density_laplacian: torch.Tensor
    homo_proxy: torch.Tensor
    lumo_proxy: torch.Tensor
    homo_lumo_gap: torch.Tensor
    pseudo_hamiltonian: torch.Tensor


@dataclass
class TSDARResult:
    hyperspherical_embedding: torch.Tensor
    dispersion_score: torch.Tensor
    candidate_indices: torch.Tensor


class QuantumDescriptorExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _coords_with_grad(self, r: torch.Tensor) -> torch.Tensor:
        if r.requires_grad:
            return r
        return r.clone().detach().requires_grad_(True)

    def compute_fukui(
        self,
        field,
        r: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        z_atoms: Optional[torch.Tensor] = None,
        current_N: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del pos, z_atoms
        electrons = field.total_electrons if current_N is None else torch.as_tensor(
            current_N,
            dtype=field.total_electrons.dtype,
            device=field.total_electrons.device,
        )
        rho_n = field.query_density(r, total_electrons=electrons)
        rho_np1 = field.query_density(r, total_electrons=electrons + 1.0)
        rho_nm1 = field.query_density(r, total_electrons=(electrons - 1.0).clamp_min(1.0))
        f_plus = rho_np1 - rho_n
        f_minus = rho_n - rho_nm1
        return {
            "f_plus": f_plus,
            "f_minus": f_minus,
            "f_dual": f_plus - f_minus,
        }

    def compute_mesp(
        self,
        field,
        r: torch.Tensor,
        density: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        coords = self._coords_with_grad(r)
        rho = field.query_density(coords) if density is None else density
        grad_rho = torch.autograd.grad(
            rho.sum(),
            coords,
            create_graph=True,
            retain_graph=True,
        )[0]
        laplacian_terms = []
        for axis in range(coords.size(-1)):
            second = torch.autograd.grad(
                grad_rho[..., axis].sum(),
                coords,
                create_graph=True,
                retain_graph=True,
            )[0][..., axis]
            laplacian_terms.append(second)
        laplacian = torch.stack(laplacian_terms, dim=-1).sum(dim=-1)
        nerd_scale = torch.as_tensor(
            field.quantum_norm_factor,
            dtype=rho.dtype,
            device=rho.device,
        )
        mesp = nerd_scale * rho - 0.25 * laplacian
        return {
            "mesp": mesp,
            "density_gradient": grad_rho,
            "density_laplacian": laplacian,
        }

    def compute_frontier_orbitals(
        self,
        field,
        r: torch.Tensor,
        latent_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        coords = self._coords_with_grad(r)
        if latent_features is None:
            _, latent = field.raw_query(coords, return_latent=True)
        else:
            latent = latent_features
        latent_mean = latent.mean(dim=-2)
        grad_components = []
        for idx in range(latent_mean.size(-1)):
            grad_i = torch.autograd.grad(
                latent_mean[..., idx].sum(),
                coords,
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_components.append(grad_i)
        gradient_stack = torch.stack(grad_components, dim=-2)
        pseudo_hamiltonian = torch.einsum("nia,nja->nij", gradient_stack, gradient_stack)
        eigvals = torch.linalg.eigvalsh(pseudo_hamiltonian)
        homo_proxy = eigvals[..., -1]
        lumo_proxy = eigvals[..., 0]
        return {
            "homo_proxy": homo_proxy,
            "lumo_proxy": lumo_proxy,
            "homo_lumo_gap": homo_proxy - lumo_proxy,
            "pseudo_hamiltonian": pseudo_hamiltonian,
        }

    def identify_ts_candidates(
        self,
        trajectory_latent: torch.Tensor,
        top_k: int = 3,
    ) -> TSDARResult:
        if trajectory_latent.ndim < 3:
            raise ValueError("trajectory_latent must have shape [T, N, 8] or [T, ...]")
        flattened = trajectory_latent.reshape(trajectory_latent.size(0), -1)
        embedding = flattened / flattened.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        reactant = embedding[0]
        product = embedding[-1]
        if embedding.size(0) > 2:
            center = embedding.mean(dim=0)
            reactant_disp = (embedding - reactant.unsqueeze(0)).norm(dim=-1)
            product_disp = (embedding - product.unsqueeze(0)).norm(dim=-1)
            center_disp = (embedding - center.unsqueeze(0)).norm(dim=-1)
            score = reactant_disp + product_disp + center_disp
            score[0] = -torch.inf
            score[-1] = -torch.inf
        else:
            score = torch.zeros(embedding.size(0), dtype=embedding.dtype, device=embedding.device)
        k = min(max(int(top_k), 1), max(int(embedding.size(0) - 2), 1))
        candidate_indices = torch.topk(score, k=k, largest=True).indices
        return TSDARResult(
            hyperspherical_embedding=embedding,
            dispersion_score=score,
            candidate_indices=candidate_indices,
        )

    def forward(
        self,
        field,
        r: torch.Tensor,
        latent_features: Optional[torch.Tensor] = None,
    ) -> QuantumObservableBundle:
        fukui = self.compute_fukui(field, r)
        mesp = self.compute_mesp(field, r)
        frontier = self.compute_frontier_orbitals(field, r, latent_features=latent_features)
        return QuantumObservableBundle(
            f_plus=fukui["f_plus"],
            f_minus=fukui["f_minus"],
            f_dual=fukui["f_dual"],
            mesp=mesp["mesp"],
            density_gradient=mesp["density_gradient"],
            density_laplacian=mesp["density_laplacian"],
            homo_proxy=frontier["homo_proxy"],
            lumo_proxy=frontier["lumo_proxy"],
            homo_lumo_gap=frontier["homo_lumo_gap"],
            pseudo_hamiltonian=frontier["pseudo_hamiltonian"],
        )


__all__ = ["QuantumDescriptorExtractor", "QuantumObservableBundle", "TSDARResult"]
