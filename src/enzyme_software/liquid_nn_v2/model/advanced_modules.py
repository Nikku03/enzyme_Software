from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.model.liquid_branch import scatter_mean
from enzyme_software.liquid_nn_v2.model.pooling import segment_softmax, segment_sum


if TORCH_AVAILABLE:
    def clamp_hidden_norm(x, max_norm: Optional[float]) -> torch.Tensor:
        if max_norm is None or max_norm <= 0.0 or x.numel() == 0:
            return x
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
        scale = torch.clamp(float(max_norm) / norms, max=1.0)
        return x * scale


    class EnergyLandscape(nn.Module):
        """Small differentiable energy model over atom, group, and molecule states."""

        def __init__(
            self,
            hidden_dim: int,
            energy_hidden_dim: int,
            dropout: float = 0.1,
            energy_value_clip: float = 6.0,
        ):
            super().__init__()
            self.energy_value_clip = max(0.5, float(energy_value_clip))
            self.node_head = nn.Sequential(
                nn.Linear(hidden_dim, energy_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(energy_hidden_dim, 1),
            )
            self.group_head = nn.Sequential(
                nn.Linear(hidden_dim, energy_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(energy_hidden_dim, 1),
            )
            self.mol_head = nn.Sequential(
                nn.Linear(hidden_dim, energy_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(energy_hidden_dim, 1),
            )

        def forward(self, atom_hidden, batch, mol_hidden=None, group_hidden=None, group_mask=None):
            node_energy_raw = self.node_head(atom_hidden)
            num_molecules = int(batch.max().item()) + 1 if batch.numel() else 0
            pooled = scatter_mean(atom_hidden, batch, num_molecules) if num_molecules else atom_hidden.new_zeros((0, atom_hidden.size(-1)))
            mol_base = mol_hidden if mol_hidden is not None else pooled
            mol_energy_raw = self.mol_head(mol_base) if mol_base.numel() else node_energy_raw.new_zeros((0, 1))
            group_energy = None
            if group_hidden is not None:
                group_energy = self.group_head(group_hidden)
                if group_mask is not None:
                    group_energy = group_energy.masked_fill(~group_mask.bool().unsqueeze(-1), 0.0)
            node_energy = torch.tanh(node_energy_raw / self.energy_value_clip) * self.energy_value_clip
            mol_energy = torch.tanh(mol_energy_raw / self.energy_value_clip) * self.energy_value_clip
            if group_energy is not None:
                group_energy = torch.tanh(group_energy / self.energy_value_clip) * self.energy_value_clip
            stats = {
                "mean": float(node_energy.detach().mean().item()),
                "min": float(node_energy.detach().min().item()),
                "max": float(node_energy.detach().max().item()),
                "raw_mean": float(node_energy_raw.detach().mean().item()),
                "raw_abs_max": float(node_energy_raw.detach().abs().max().item()),
            }
            return {
                "node_energy": node_energy,
                "node_energy_raw": node_energy_raw,
                "group_energy": group_energy,
                "mol_energy": mol_energy,
                "mol_energy_raw": mol_energy_raw,
                "stats": stats,
            }


    class BarrierCrossingModule(nn.Module):
        """Predict barrier heights and stable tunneling probabilities."""

        def __init__(
            self,
            hidden_dim: int,
            tunneling_hidden_dim: int,
            alpha_init: float = 1.0,
            barrier_min: float = 0.0,
            barrier_max: float = 12.0,
            probability_floor: float = 1.0e-4,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.barrier_min = float(barrier_min)
            self.barrier_max = float(barrier_max)
            self.probability_floor = float(probability_floor)
            self.barrier_head = nn.Sequential(
                nn.Linear(hidden_dim, tunneling_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(tunneling_hidden_dim, 1),
            )
            self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(alpha_init))))

        def forward(self, hidden):
            raw_barrier = self.barrier_head(hidden)
            barrier = F.softplus(raw_barrier) + self.barrier_min
            barrier = barrier.clamp(max=self.barrier_max)
            alpha = self.log_alpha.exp().clamp(min=1.0e-3, max=10.0)
            log_prob = -(alpha * barrier).clamp(min=0.0, max=20.0)
            tunnel_prob = torch.exp(log_prob).clamp(min=self.probability_floor, max=1.0)
            stats = {
                "barrier_mean": float(barrier.detach().mean().item()),
                "barrier_max": float(barrier.detach().max().item()),
                "tunnel_prob_mean": float(tunnel_prob.detach().mean().item()),
                "tunnel_prob_max": float(tunnel_prob.detach().max().item()),
                "alpha": float(alpha.detach().item()),
            }
            return {
                "barrier": barrier,
                "tunnel_prob": tunnel_prob,
                "stats": stats,
            }


    class GraphTunneling(nn.Module):
        """Sparse non-local atom coupling gated by affinity and tunneling probability."""

        def __init__(
            self,
            hidden_dim: int,
            projection_dim: int = 48,
            max_edges_per_node: int = 4,
            dropout: float = 0.1,
            residual_scale: float = 0.05,
            residual_scale_max: float = 0.25,
        ):
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.max_edges_per_node = max(1, int(max_edges_per_node))
            self.projection_dim = max(8, int(projection_dim))
            self.residual_scale_max = max(1.0e-3, float(residual_scale_max))
            self.query = nn.Linear(hidden_dim, self.projection_dim)
            self.key = nn.Linear(hidden_dim, self.projection_dim)
            self.value = nn.Linear(hidden_dim, self.projection_dim)
            self.pre_norm = nn.LayerNorm(self.projection_dim)
            self.out_proj = nn.Sequential(
                nn.Linear(self.projection_dim, self.projection_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.projection_dim, hidden_dim),
            )
            self.out_norm = nn.LayerNorm(hidden_dim)
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, self.projection_dim),
                nn.SiLU(),
                nn.Linear(self.projection_dim, 1),
                nn.Sigmoid(),
            )
            init_ratio = min(max(float(residual_scale) / self.residual_scale_max, 1.0e-4), 1.0 - 1.0e-4)
            self.residual_logit = nn.Parameter(torch.logit(torch.tensor(init_ratio)))

        def _build_edges(self, hidden, batch) -> Tuple[torch.Tensor, torch.Tensor]:
            device = hidden.device
            edge_pairs: List[torch.Tensor] = []
            edge_scores: List[torch.Tensor] = []
            q = F.normalize(self.query(hidden), dim=-1)
            k = F.normalize(self.key(hidden), dim=-1)
            num_molecules = int(batch.max().item()) + 1 if batch.numel() else 0
            for mol_idx in range(num_molecules):
                mol_atoms = torch.where(batch == mol_idx)[0]
                if mol_atoms.numel() <= 1:
                    continue
                mol_q = q[mol_atoms]
                mol_k = k[mol_atoms]
                affinity = torch.matmul(mol_q, mol_k.transpose(0, 1))
                affinity.fill_diagonal_(-1.0e4)
                topk = min(self.max_edges_per_node, max(1, mol_atoms.numel() - 1))
                scores, nbr_idx = torch.topk(affinity, k=topk, dim=-1)
                src = mol_atoms.unsqueeze(-1).expand_as(nbr_idx).reshape(-1)
                dst = mol_atoms[nbr_idx.reshape(-1)]
                edge_pairs.append(torch.stack([src, dst], dim=0))
                edge_scores.append(scores.reshape(-1))
            if not edge_pairs:
                return (
                    torch.zeros((2, 0), dtype=torch.long, device=device),
                    torch.zeros((0,), dtype=hidden.dtype, device=device),
                )
            return torch.cat(edge_pairs, dim=1), torch.cat(edge_scores, dim=0)

        def forward(self, hidden, batch, tunnel_prob=None):
            edge_index, affinity_scores = self._build_edges(hidden, batch)
            if edge_index.numel() == 0:
                return {
                    "message": torch.zeros_like(hidden),
                    "edge_index": edge_index,
                    "edge_prob": affinity_scores.unsqueeze(-1),
                    "stats": {
                        "num_edges": 0.0,
                        "active_fraction": 0.0,
                        "mean_edge_prob": 0.0,
                        "tunnel_msg_norm_mean": 0.0,
                        "tunnel_gate_mean": 0.0,
                        "tunneling_edge_count": 0.0,
                    },
                }
            src, dst = edge_index[0], edge_index[1]
            affinity = torch.sigmoid(affinity_scores).unsqueeze(-1)
            if tunnel_prob is None:
                pair_prob = affinity
            else:
                pair_prob = affinity * torch.sqrt((tunnel_prob[src] * tunnel_prob[dst]).clamp(min=1.0e-8))
            pair_prob = torch.nan_to_num(pair_prob, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
            values = self.value(hidden[src])
            weighted = values * pair_prob
            aggregated = hidden.new_zeros((hidden.size(0), self.projection_dim))
            aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted), weighted)
            weight_sum = hidden.new_zeros((hidden.size(0), 1))
            weight_sum.scatter_add_(0, dst.unsqueeze(-1), pair_prob)
            normalized = self.pre_norm(aggregated / weight_sum.clamp(min=1.0e-6))
            message_core = self.out_norm(self.out_proj(normalized))
            gate = self.gate_net(torch.cat([hidden, message_core], dim=-1))
            residual_scale = torch.sigmoid(self.residual_logit) * self.residual_scale_max
            message = gate * residual_scale * message_core
            active_fraction = float((pair_prob.detach() > 0.1).float().mean().item())
            return {
                "message": message,
                "edge_index": edge_index,
                "edge_prob": pair_prob,
                "stats": {
                    "num_edges": float(edge_index.size(1)),
                    "tunneling_edge_count": float(edge_index.size(1)),
                    "active_fraction": active_fraction,
                    "mean_edge_prob": float(pair_prob.detach().mean().item()),
                    "tunnel_msg_norm_mean": float(message.detach().norm(dim=-1).mean().item()),
                    "tunnel_gate_mean": float(gate.detach().mean().item()),
                    "tunnel_scale": float(residual_scale.detach().item()),
                },
            }


    class PhaseAugmentedState(nn.Module):
        """Phase-augmented real hidden state for resonance-like modulation."""

        def __init__(self, hidden_dim: int, phase_hidden_dim: int, phase_scale: float = 0.25, dropout: float = 0.1):
            super().__init__()
            self.phase_scale = float(phase_scale)
            self.phase_head = nn.Sequential(
                nn.Linear(hidden_dim, phase_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(phase_hidden_dim, hidden_dim),
            )
            self.mix = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )

        def forward(self, hidden):
            phase = torch.tanh(self.phase_head(hidden)) * 3.14159265
            amplitude = hidden.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
            modulated = hidden * (1.0 + self.phase_scale * torch.cos(phase))
            quadrature = hidden * torch.sin(phase)
            output = self.mix(torch.cat([modulated, quadrature], dim=-1))
            stats = {
                "phase_mean": float(phase.detach().mean().item()),
                "phase_var": float(phase.detach().var().item()),
                "amplitude_norm_mean": float(amplitude.detach().mean().item()),
            }
            return output, phase, stats


    class HigherOrderCoupling(nn.Module):
        """Sparse higher-order coupling among salient atoms within each molecule."""

        def __init__(self, hidden_dim: int, coupling_hidden_dim: int, topk: int = 8, heads: int = 2, dropout: float = 0.1):
            super().__init__()
            self.topk = max(2, int(topk))
            self.salience = nn.Linear(hidden_dim, 1)
            self.query = nn.Linear(hidden_dim, coupling_hidden_dim)
            self.key = nn.Linear(hidden_dim, coupling_hidden_dim)
            self.value = nn.Linear(hidden_dim, coupling_hidden_dim)
            self.value_out = nn.Linear(coupling_hidden_dim, hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, coupling_hidden_dim),
                nn.SiLU(),
                nn.Linear(coupling_hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.out_proj = nn.Sequential(
                nn.Linear(hidden_dim, coupling_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(coupling_hidden_dim, hidden_dim),
            )
            self.heads = max(1, int(heads))

        def forward(self, hidden, batch, priority=None):
            device = hidden.device
            updated = torch.zeros_like(hidden)
            selected_indices: List[torch.Tensor] = []
            strengths = []
            num_molecules = int(batch.max().item()) + 1 if batch.numel() else 0
            base_salience = self.salience(hidden).squeeze(-1)
            if priority is not None:
                base_salience = base_salience + priority.squeeze(-1)
            for mol_idx in range(num_molecules):
                mol_atoms = torch.where(batch == mol_idx)[0]
                if mol_atoms.numel() < 2:
                    continue
                k = min(self.topk, mol_atoms.numel())
                scores = base_salience[mol_atoms]
                _, local_idx = torch.topk(scores, k=k, dim=0)
                idx = mol_atoms[local_idx]
                selected_indices.append(idx)
                q = self.query(hidden[idx])
                k_proj = self.key(hidden[idx])
                attn = torch.matmul(q, k_proj.transpose(0, 1)) / max(1.0, float(q.size(-1)) ** 0.5)
                attn = torch.softmax(attn, dim=-1)
                values = self.value(hidden[idx])
                coupled = self.value_out(torch.matmul(attn, values))
                gate = self.gate(torch.cat([hidden[idx], coupled], dim=-1))
                delta = gate * torch.tanh(self.out_proj(coupled))
                updated[idx] = updated[idx] + delta
                strengths.append(attn.detach().mean())
            if selected_indices:
                indices = torch.cat(selected_indices, dim=0)
            else:
                indices = torch.zeros((0,), dtype=torch.long, device=device)
            avg_strength = float(torch.stack(strengths).mean().item()) if strengths else 0.0
            return {
                "update": updated,
                "selected_indices": indices,
                "stats": {
                    "active_interactions": float(indices.numel()),
                    "average_coupling_strength": avg_strength,
                },
            }


    class PhysicsResidualBranch(nn.Module):
        """Compact deterministic-physics residual branch with gated fusion."""

        def __init__(self, input_dim: int, hidden_dim: int, residual_hidden_dim: int):
            super().__init__()
            self.physics_proj = nn.Sequential(
                nn.Linear(input_dim, residual_hidden_dim),
                nn.SiLU(),
                nn.Linear(residual_hidden_dim, hidden_dim),
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )

        def forward(self, liquid_hidden, physics_hidden):
            projected = self.physics_proj(physics_hidden)
            gate = self.gate(torch.cat([liquid_hidden, projected], dim=-1))
            fused = gate * liquid_hidden + (1.0 - gate) * projected
            stats = {
                "gate_mean": float(gate.detach().mean().item()),
                "liquid_norm": float(liquid_hidden.detach().norm(dim=-1).mean().item()),
                "physics_norm": float(projected.detach().norm(dim=-1).mean().item()),
            }
            return fused, gate, stats


    class DeliberationLoop(nn.Module):
        """Small proposer-critic refinement loop for atom and molecule predictions."""

        def __init__(
            self,
            atom_dim: int,
            mol_dim: int,
            num_cyp_classes: int,
            hidden_dim: int,
            num_steps: int,
            dropout: float = 0.1,
            step_scale: float = 0.1,
            max_state_norm: float = 10.0,
        ):
            super().__init__()
            self.num_steps = max(0, int(num_steps))
            self.step_scale = max(0.0, float(step_scale))
            self.max_state_norm = max(0.0, float(max_state_norm))
            self.atom_proposer = nn.Sequential(
                nn.Linear(atom_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.mol_proposer = nn.Sequential(
                nn.Linear(mol_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_cyp_classes),
            )
            self.critic = nn.Sequential(
                nn.Linear(atom_dim + mol_dim + 4, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.atom_refiner = nn.Sequential(
                nn.Linear(atom_dim + 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, atom_dim),
            )
            self.atom_update_norm = nn.LayerNorm(atom_dim)
            self.atom_gate_head = nn.Sequential(
                nn.Linear(atom_dim * 2 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.mol_refiner = nn.Sequential(
                nn.Linear(mol_dim + num_cyp_classes + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, mol_dim),
            )
            self.mol_update_norm = nn.LayerNorm(mol_dim)
            self.mol_gate_head = nn.Sequential(
                nn.Linear(mol_dim * 2 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.atom_norm = nn.LayerNorm(atom_dim)
            self.mol_norm = nn.LayerNorm(mol_dim)

        def forward(self, atom_hidden, mol_hidden, batch, *, node_energy=None, tunnel_prob=None):
            if self.num_steps == 0:
                return {
                    "atom_hidden": atom_hidden,
                    "mol_hidden": mol_hidden,
                    "site_logits": [],
                    "cyp_logits": [],
                    "critic_scores": [],
                    "stats": {
                        "num_steps": 0.0,
                        "critic_mean": 0.0,
                        "site_delta_abs_mean": 0.0,
                    },
                }
            atom_state = atom_hidden
            mol_state = mol_hidden
            site_logits: List[torch.Tensor] = []
            cyp_logits: List[torch.Tensor] = []
            critic_scores: List[torch.Tensor] = []
            atom_gate_means: List[torch.Tensor] = []
            mol_gate_means: List[torch.Tensor] = []
            atom_state_norms: List[torch.Tensor] = []
            mol_state_norms: List[torch.Tensor] = []
            for _ in range(self.num_steps):
                site_logit = self.atom_proposer(atom_state)
                cyp_logit = self.mol_proposer(mol_state)
                mol_context = mol_state[batch] if mol_state.numel() else atom_state.new_zeros((atom_state.size(0), 0))
                energy_feature = node_energy if node_energy is not None else atom_state.new_zeros((atom_state.size(0), 1))
                tunnel_feature = tunnel_prob if tunnel_prob is not None else atom_state.new_zeros((atom_state.size(0), 1))
                critic_input = torch.cat(
                    [
                        atom_state,
                        mol_context,
                        site_logit,
                        energy_feature,
                        tunnel_feature,
                        cyp_logit[batch].mean(dim=-1, keepdim=True),
                    ],
                    dim=-1,
                )
                critic = torch.tanh(self.critic(critic_input))
                atom_update = self.atom_refiner(torch.cat([atom_state, site_logit, critic], dim=-1))
                atom_update = self.atom_update_norm(atom_update)
                atom_gate = self.atom_gate_head(torch.cat([atom_state, atom_update, critic], dim=-1))
                mol_feedback = scatter_mean(critic, batch, mol_state.size(0)) if mol_state.size(0) else mol_state.new_zeros((0, 1))
                mol_update = self.mol_refiner(torch.cat([mol_state, cyp_logit, mol_feedback], dim=-1))
                mol_update = self.mol_update_norm(mol_update)
                mol_gate = self.mol_gate_head(torch.cat([mol_state, mol_update, mol_feedback], dim=-1))
                atom_state = clamp_hidden_norm(
                    self.atom_norm(
                        clamp_hidden_norm(
                            atom_state + atom_gate * self.step_scale * torch.tanh(atom_update),
                            self.max_state_norm,
                        )
                    ),
                    self.max_state_norm,
                )
                mol_state = clamp_hidden_norm(
                    self.mol_norm(
                        clamp_hidden_norm(
                            mol_state + mol_gate * self.step_scale * torch.tanh(mol_update),
                            self.max_state_norm,
                        )
                    ),
                    self.max_state_norm,
                )
                site_logits.append(site_logit)
                cyp_logits.append(cyp_logit)
                critic_scores.append(critic)
                atom_gate_means.append(atom_gate.detach().mean())
                mol_gate_means.append(mol_gate.detach().mean())
                atom_state_norms.append(atom_state.detach().norm(dim=-1).mean())
                mol_state_norms.append(mol_state.detach().norm(dim=-1).mean())
            site_delta = (site_logits[-1] - site_logits[0]).abs().mean() if len(site_logits) > 1 else atom_hidden.new_tensor(0.0)
            critic_mean = torch.cat(critic_scores, dim=0).mean() if critic_scores else atom_hidden.new_tensor(0.0)
            return {
                "atom_hidden": atom_state,
                "mol_hidden": mol_state,
                "site_logits": site_logits,
                "cyp_logits": cyp_logits,
                "critic_scores": critic_scores,
                "stats": {
                    "num_steps": float(self.num_steps),
                    "critic_mean": float(critic_mean.detach().item()),
                    "site_delta_abs_mean": float(site_delta.detach().item()),
                    "atom_gate_mean": float(torch.stack(atom_gate_means).mean().item()) if atom_gate_means else 0.0,
                    "mol_gate_mean": float(torch.stack(mol_gate_means).mean().item()) if mol_gate_means else 0.0,
                    "atom_hidden_norm_mean": float(torch.stack(atom_state_norms).mean().item()) if atom_state_norms else 0.0,
                    "mol_hidden_norm_mean": float(torch.stack(mol_state_norms).mean().item()) if mol_state_norms else 0.0,
                    "step_scale": float(self.step_scale),
                },
            }
else:  # pragma: no cover
    class EnergyLandscape:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class BarrierCrossingModule:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class GraphTunneling:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class PhaseAugmentedState:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class HigherOrderCoupling:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class PhysicsResidualBranch:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class DeliberationLoop:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
