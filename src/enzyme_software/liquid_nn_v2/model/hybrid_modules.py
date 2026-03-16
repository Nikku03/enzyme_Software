from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class LocalTunnelingBias(nn.Module):
        """Applies a bounded local tunneling-derived bias to per-atom site logits."""

        def __init__(self, scale: float = 0.1, clamp_value: float = 5.0):
            super().__init__()
            self.scale = max(0.0, float(scale))
            self.clamp_value = max(0.5, float(clamp_value))

        def forward(self, site_logits, tunnel_prob):
            if tunnel_prob is None:
                zero = torch.zeros_like(site_logits)
                return site_logits, zero, {
                    "tunnel_prob_mean": 0.0,
                    "tunnel_bias_mean": 0.0,
                    "tunnel_bias_max": 0.0,
                }
            safe_prob = torch.nan_to_num(
                tunnel_prob,
                nan=1.0e-6,
                posinf=1.0,
                neginf=1.0e-6,
            ).clamp(min=1.0e-6, max=1.0)
            tunnel_bias = torch.log(safe_prob).clamp(min=-self.clamp_value, max=0.0) * self.scale
            refined_logits = site_logits + tunnel_bias
            stats = {
                "tunnel_prob_mean": float(safe_prob.detach().mean().item()),
                "tunnel_bias_mean": float(tunnel_bias.detach().mean().item()),
                "tunnel_bias_max": float(tunnel_bias.detach().abs().max().item()),
            }
            return refined_logits, tunnel_bias, stats


    class OutputRefinementHead(nn.Module):
        """Single-step, bounded output refiner for site logits."""

        def __init__(self, atom_dim: int, hidden_dim: int, scale: float = 0.1, dropout: float = 0.1):
            super().__init__()
            self.scale = max(0.0, float(scale))
            self.delta_head = nn.Sequential(
                nn.Linear(atom_dim + 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.gate_head = nn.Sequential(
                nn.Linear(atom_dim + 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, atom_features, site_logits, mol_context: Optional[torch.Tensor] = None):
            if mol_context is None:
                mol_summary = atom_features.new_zeros((atom_features.size(0), 1))
            else:
                mol_summary = mol_context
                if mol_summary.ndim == 1:
                    mol_summary = mol_summary.unsqueeze(-1)
            refinement_input = torch.cat([atom_features, site_logits, mol_summary], dim=-1)
            delta = torch.tanh(self.delta_head(refinement_input))
            gate = self.gate_head(refinement_input)
            refined_logits = site_logits + gate * self.scale * delta
            stats = {
                "refine_gate_mean": float(gate.detach().mean().item()),
                "refine_delta_mean": float(delta.detach().mean().item()),
                "refine_delta_max": float(delta.detach().abs().max().item()),
            }
            return refined_logits, delta, gate, stats
else:  # pragma: no cover
    class LocalTunnelingBias:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class OutputRefinementHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
