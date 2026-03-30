from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

try:
    from diffusers import DDPMScheduler

    _DIFFUSERS_OK = True
except Exception:  # pragma: no cover - optional dependency
    DDPMScheduler = None
    _DIFFUSERS_OK = False

from .mol_llama_wrapper import LatentBlueprint


@dataclass
class Jacobian_Hook:
    initial_R: torch.Tensor
    latent: torch.Tensor

    def vector_jacobian(self, grad_outputs: torch.Tensor) -> torch.Tensor:
        grad = torch.autograd.grad(
            outputs=self.initial_R,
            inputs=self.latent,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        return grad

    def materialize(self) -> torch.Tensor:
        flat_R = self.initial_R.reshape(-1)
        rows: List[torch.Tensor] = []
        for idx in range(int(flat_R.numel())):
            grad = torch.autograd.grad(
                outputs=flat_R[idx],
                inputs=self.latent,
                retain_graph=True,
                create_graph=True,
                allow_unused=False,
            )[0]
            rows.append(grad.reshape(-1))
        jacobian = torch.stack(rows, dim=0)
        return jacobian.reshape(*self.initial_R.shape, *self.latent.shape)


class _FastPathScheduler:
    def __init__(self, num_steps: int) -> None:
        self.num_steps = max(1, int(num_steps))
        if _DIFFUSERS_OK:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=64,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=False,
                prediction_type="epsilon",
            )
            self.timesteps = torch.linspace(63, 0, steps=self.num_steps, dtype=torch.long)
        else:
            self.scheduler = None
            self.timesteps = torch.linspace(1.0, 0.0, steps=self.num_steps)

    def step_size(self, step_index: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.scheduler is not None:
            timestep = self.timesteps[step_index].to(device=device)
            alpha_bar = self.scheduler.alphas_cumprod[timestep].to(device=device, dtype=dtype)
            return (1.0 - alpha_bar).clamp_min(1.0e-4).sqrt()
        return self.timesteps[step_index].to(device=device, dtype=dtype).clamp_min(1.0e-3)


class _SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, step_value: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=step_value.device, dtype=step_value.dtype)
            * (-math.log(10000.0) / max(half_dim - 1, 1))
        )
        angles = step_value.reshape(1, 1) * freq.reshape(1, -1)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.dim:
            pad = torch.zeros((1, self.dim - embedding.shape[-1]), device=embedding.device, dtype=embedding.dtype)
            embedding = torch.cat([embedding, pad], dim=-1)
        return embedding


def e3_canonical_alignment(coords: torch.Tensor) -> torch.Tensor:
    """
    Canonicalize generated coordinates with the principal moments of inertia.

    This removes arbitrary global translation/rotation from the diffusion output
    so downstream equivariant modules see molecules in a consistent frame.
    """
    squeezed = False
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
        squeezed = True
    if coords.ndim != 3 or coords.size(-1) != 3:
        raise ValueError(f"coords must have shape [N, 3] or [B, N, 3], got {tuple(coords.shape)}")

    centroid = coords.mean(dim=-2, keepdim=True)
    centered = coords - centroid

    x, y, z = centered.unbind(dim=-1)
    r2 = centered.square().sum(dim=-1)
    inertia = torch.zeros(coords.size(0), 3, 3, dtype=coords.dtype, device=coords.device)
    inertia[:, 0, 0] = (r2 - x.square()).sum(dim=-1)
    inertia[:, 1, 1] = (r2 - y.square()).sum(dim=-1)
    inertia[:, 2, 2] = (r2 - z.square()).sum(dim=-1)
    inertia[:, 0, 1] = inertia[:, 1, 0] = -(x * y).sum(dim=-1)
    inertia[:, 0, 2] = inertia[:, 2, 0] = -(x * z).sum(dim=-1)
    inertia[:, 1, 2] = inertia[:, 2, 1] = -(y * z).sum(dim=-1)

    _, eigenvectors = torch.linalg.eigh(inertia.to(dtype=torch.float64))
    eigenvectors = eigenvectors.detach().to(dtype=coords.dtype)
    canonical = centered @ eigenvectors

    skewness = canonical.pow(3).sum(dim=-2)
    signs = torch.where(skewness < 0, -torch.ones_like(skewness), torch.ones_like(skewness))
    canonical = canonical * signs.unsqueeze(-2)

    det = torch.linalg.det(eigenvectors.to(dtype=torch.float64)).to(dtype=coords.dtype)
    flip_mask = det < 0
    if flip_mask.any():
        canonical = canonical.clone()
        canonical[flip_mask, :, 0] = -canonical[flip_mask, :, 0]

    return canonical.squeeze(0) if squeezed else canonical


class _DiffusionRefinementBlock(nn.Module):
    def __init__(self, hidden_dim: int, radial_dim: int) -> None:
        super().__init__()
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + radial_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.coord_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + radial_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.coord_bias = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        centers = torch.linspace(0.25, 6.0, steps=radial_dim)
        self.register_buffer("rbf_centers", centers)
        self.rbf_gamma = 1.5

    def _rbf(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.rbf_gamma * (distance.unsqueeze(-1) - self.rbf_centers) ** 2)

    def forward(
        self,
        node_state: torch.Tensor,
        pos: torch.Tensor,
        latent_tokens: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_atoms = int(node_state.shape[0])
        normed = self.node_norm(node_state)
        cross_context, _ = self.cross_attn(
            query=normed.unsqueeze(0),
            key=latent_tokens.unsqueeze(0),
            value=latent_tokens.unsqueeze(0),
            need_weights=False,
        )
        cross_context = cross_context.squeeze(0)

        rel = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = rel.square().sum(dim=-1).clamp_min(1.0e-8).sqrt()
        direction = rel / dist.unsqueeze(-1)
        radial = self._rbf(dist)
        time_ctx = time_embedding.unsqueeze(0).expand(n_atoms, -1)

        source = normed.unsqueeze(1).expand(n_atoms, n_atoms, -1)
        target = normed.unsqueeze(0).expand(n_atoms, n_atoms, -1)
        time_pair = time_embedding.view(1, 1, -1).expand(n_atoms, n_atoms, -1)
        edge_input = torch.cat([source, target, radial, time_pair], dim=-1)
        edge_logits = self.edge_mlp(edge_input).squeeze(-1)
        edge_logits = edge_logits.masked_fill(torch.eye(n_atoms, device=pos.device, dtype=torch.bool), -1.0e9)
        attention = torch.softmax(edge_logits, dim=-1)

        node_context = torch.matmul(attention, cross_context)
        neighbor_context = torch.matmul(attention, normed)
        node_input = torch.cat([normed, node_context, neighbor_context], dim=-1)
        updated_state = node_state + self.node_update(node_input)

        coord_gate = self.coord_gate(
            torch.cat([source, target, radial], dim=-1)
        ).squeeze(-1)
        coord_update = (attention * coord_gate).unsqueeze(-1) * direction
        coord_update = coord_update.sum(dim=1)
        coord_update = coord_update + 0.1 * self.coord_bias(torch.cat([updated_state, time_ctx], dim=-1))
        updated_pos = pos + 0.2 * torch.tanh(coord_update)
        return updated_state, updated_pos


class StructuralDiffusion3D(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 256,
        hidden_dim: int = 192,
        diffusion_steps: int = 6,
        radial_dim: int = 16,
        supported_atomic_numbers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        supported = list(supported_atomic_numbers or [1, 6, 7, 8, 9, 15, 16, 17, 35, 53])
        self.supported_atomic_numbers = supported
        self.z_to_index = {int(z): idx for idx, z in enumerate(supported)}
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.scheduler = _FastPathScheduler(diffusion_steps)

        self.atom_embedding = nn.Embedding(len(supported) + 1, hidden_dim)
        self.chirality_embedding = nn.Embedding(5, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.token_proj = nn.Linear(latent_dim, hidden_dim)
        self.initial_state = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.initial_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.chirality_axis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.time_embed = _SinusoidalTimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            [_DiffusionRefinementBlock(hidden_dim=hidden_dim, radial_dim=radial_dim) for _ in range(self.scheduler.num_steps)]
        )

    def _map_atomic_numbers(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        mapped = torch.zeros_like(atomic_numbers, dtype=torch.long)
        for idx, value in enumerate(atomic_numbers.tolist()):
            mapped[idx] = int(self.z_to_index.get(int(value), 0))
        return mapped

    def forward(
        self,
        latent_blueprint: LatentBlueprint,
        atomic_numbers: torch.Tensor,
        chirality_codes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Jacobian_Hook]:
        device = atomic_numbers.device
        dtype = latent_blueprint.sequence.dtype
        token_latent = self.token_proj(latent_blueprint.sequence.to(device=device, dtype=dtype))
        pooled_latent = self.latent_proj(latent_blueprint.pooled.to(device=device, dtype=dtype))

        z_index = self._map_atomic_numbers(atomic_numbers)
        atom_state = self.atom_embedding(z_index)
        if chirality_codes is None:
            chirality_codes = torch.zeros_like(atomic_numbers)
        chirality_ids = chirality_codes.to(device=device, dtype=torch.long).clamp(min=-2, max=2) + 2
        stereo_state = self.chirality_embedding(chirality_ids)

        pooled_expanded = pooled_latent.unsqueeze(0).expand(atom_state.shape[0], -1)
        node_state = self.initial_state(torch.cat([atom_state, stereo_state, pooled_expanded], dim=-1))
        pos = self.initial_pos(node_state)
        signed_chirality = chirality_codes.to(device=device, dtype=pos.dtype).unsqueeze(-1)
        pos = pos + 0.15 * torch.tanh(self.chirality_axis(node_state)) * signed_chirality

        for step_idx, block in enumerate(self.blocks):
            step_value = self.scheduler.step_size(step_idx, device=device, dtype=pos.dtype)
            time_embedding = self.time_embed(step_value).to(device=device, dtype=pos.dtype).squeeze(0)
            node_state, pos = block(node_state, pos, token_latent, time_embedding)

        pos = e3_canonical_alignment(pos)
        pos = pos.requires_grad_(True)
        jacobian_hook = Jacobian_Hook(initial_R=pos, latent=latent_blueprint.sequence)
        return pos, jacobian_hook
