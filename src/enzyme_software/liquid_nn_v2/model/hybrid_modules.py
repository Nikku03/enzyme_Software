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


    class TopKCrossAtomReranker(nn.Module):
        """Cross-atom reranker over the top candidate atoms in each molecule.

        The base model remains the proposal stage. This module only revisits the
        highest-scoring atoms inside each molecule and learns a winner decision
        from their joint context.
        """

        def __init__(
            self,
            *,
            atom_dim: int,
            mol_dim: int,
            extra_dim: int = 0,
            hidden_dim: int = 128,
            top_k: int = 8,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
            residual_scale: float = 0.75,
            gate_bias: float = -2.0,
        ):
            super().__init__()
            self.top_k = max(2, int(top_k))
            self.hidden_dim = max(32, int(hidden_dim))
            self.mol_dim = max(0, int(mol_dim))
            self.extra_dim = max(0, int(extra_dim))
            self.residual_scale = max(0.0, float(residual_scale))
            input_dim = int(atom_dim) + self.mol_dim + self.extra_dim + 3
            heads = max(1, min(int(num_heads), self.hidden_dim))
            while self.hidden_dim % heads != 0 and heads > 1:
                heads -= 1
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.layers = nn.ModuleList()
            for _ in range(max(1, int(num_layers))):
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "attn": nn.MultiheadAttention(
                                embed_dim=self.hidden_dim,
                                num_heads=heads,
                                dropout=float(dropout),
                                batch_first=True,
                            ),
                            "norm1": nn.LayerNorm(self.hidden_dim),
                            "ff": nn.Sequential(
                                nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                                nn.SiLU(),
                                nn.Dropout(dropout),
                                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                            ),
                            "norm2": nn.LayerNorm(self.hidden_dim),
                        }
                    )
                )
            self.delta_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, 1),
            )
            self.context_score = nn.Linear(self.hidden_dim, 1)
            self.context_refine = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            )
            self.gate_head = nn.Sequential(
                nn.Linear(self.hidden_dim, max(16, self.hidden_dim // 2)),
                nn.SiLU(),
                nn.Linear(max(16, self.hidden_dim // 2), 1),
            )
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.constant_(self.gate_head[-1].bias, float(gate_bias))

        def forward(
            self,
            *,
            atom_features,
            site_logits,
            batch_index,
            mol_features=None,
            extra_features=None,
        ):
            if site_logits is None or site_logits.numel() == 0 or batch_index.numel() == 0:
                return site_logits, {
                    "selected_mask": torch.zeros_like(site_logits),
                    "stats": {
                        "enabled": 0.0,
                        "selected_fraction": 0.0,
                        "gate_mean": 0.0,
                        "delta_mean": 0.0,
                        "delta_max": 0.0,
                    },
                }
            reranked = site_logits.clone()
            selected_mask = torch.zeros_like(site_logits)
            gate_values = []
            delta_values = []
            num_molecules = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
            for mol_idx in range(num_molecules):
                mol_mask = batch_index == mol_idx
                if not bool(mol_mask.any()):
                    continue
                global_indices = torch.nonzero(mol_mask, as_tuple=False).view(-1)
                mol_logits = site_logits[global_indices].view(-1)
                k = min(self.top_k, int(global_indices.numel()))
                if k < 2:
                    continue
                top_order = torch.topk(mol_logits, k=k, largest=True).indices
                top_global = global_indices[top_order]
                top_logits = site_logits[top_global].view(-1)
                rank_feature = torch.linspace(1.0, 0.0, steps=k, device=site_logits.device, dtype=site_logits.dtype).unsqueeze(-1)
                if k > 1:
                    next_logits = torch.cat([top_logits[1:], top_logits[-1:]], dim=0)
                    gap_feature = (top_logits - next_logits).unsqueeze(-1)
                    gap_feature[-1] = 0.0
                else:
                    gap_feature = top_logits.new_zeros((k, 1))
                pieces = [
                    atom_features[top_global],
                    top_logits.unsqueeze(-1),
                    rank_feature,
                    gap_feature,
                ]
                if mol_features is not None and int(mol_features.shape[0]) > mol_idx:
                    mol_context = mol_features[mol_idx].unsqueeze(0).expand(k, -1)
                else:
                    mol_context = atom_features.new_zeros((k, self.mol_dim))
                pieces.append(mol_context)
                if extra_features is not None and extra_features.numel():
                    pieces.append(extra_features[top_global])
                elif self.extra_dim > 0:
                    pieces.append(atom_features.new_zeros((k, self.extra_dim)))
                candidate_input = torch.cat(pieces, dim=-1)
                hidden = self.input_proj(candidate_input).unsqueeze(0)
                for layer in self.layers:
                    attn_out, _ = layer["attn"](hidden, hidden, hidden, need_weights=False)
                    hidden = layer["norm1"](hidden + attn_out)
                    ff_out = layer["ff"](hidden)
                    hidden = layer["norm2"](hidden + ff_out)
                hidden = hidden.squeeze(0)
                context_alpha = torch.softmax(self.context_score(hidden).view(-1), dim=0).unsqueeze(-1)
                context = torch.sum(context_alpha * hidden, dim=0, keepdim=True).expand_as(hidden)
                refined_hidden = self.context_refine(torch.cat([hidden, context, hidden - context], dim=-1))
                delta = torch.tanh(self.delta_head(refined_hidden))
                gate = torch.sigmoid(self.gate_head(refined_hidden))
                refined = site_logits[top_global] + (self.residual_scale * gate * delta)
                reranked[top_global] = refined
                selected_mask[top_global] = 1.0
                gate_values.append(gate.view(-1))
                delta_values.append(delta.view(-1))
            if gate_values:
                gate_cat = torch.cat(gate_values, dim=0)
                delta_cat = torch.cat(delta_values, dim=0)
                stats = {
                    "enabled": 1.0,
                    "selected_fraction": float(selected_mask.detach().mean().item()),
                    "gate_mean": float(gate_cat.detach().mean().item()),
                    "delta_mean": float(delta_cat.detach().mean().item()),
                    "delta_max": float(delta_cat.detach().abs().max().item()),
                }
            else:
                stats = {
                    "enabled": 1.0,
                    "selected_fraction": 0.0,
                    "gate_mean": 0.0,
                    "delta_mean": 0.0,
                    "delta_max": 0.0,
                }
            return reranked, {"selected_mask": selected_mask, "stats": stats}
else:  # pragma: no cover
    class LocalTunnelingBias:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class OutputRefinementHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class TopKCrossAtomReranker:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
