from __future__ import annotations

from typing import Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.model.fusion import FusionGate
from enzyme_software.liquid_nn_v2.model.pooling import segment_sum


if TORCH_AVAILABLE:
    def scatter_mean(values, batch, num_segments: int):
        if values.numel() == 0:
            return values.new_zeros((num_segments, values.size(-1)))
        counts = torch.bincount(batch, minlength=num_segments).clamp(min=1).unsqueeze(-1)
        return segment_sum(values, batch, num_segments) / counts


    class EdgeAwareMessagePassing(nn.Module):
        """Message passing that conditions on edge descriptors when available."""

        def __init__(self, hidden_dim: int, edge_feature_dim: int, dropout: float = 0.1):
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.edge_feature_dim = max(1, int(edge_feature_dim))
            self.edge_proj = nn.Linear(self.edge_feature_dim, self.hidden_dim)
            self.message_net = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        def forward(self, h, edge_index, edge_attr=None):
            if edge_index.numel() == 0:
                return torch.zeros_like(h)
            src, dst = edge_index[0], edge_index[1]
            if edge_attr is None or edge_attr.numel() == 0:
                edge_attr = torch.zeros((src.numel(), self.edge_feature_dim), device=h.device, dtype=h.dtype)
            elif edge_attr.ndim == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_msg = self.edge_proj(edge_attr.to(dtype=h.dtype))
            messages = self.message_net(torch.cat([h[src], h[dst], edge_msg], dim=-1))
            aggregated = torch.zeros_like(h)
            aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
            degree = torch.bincount(dst, minlength=h.size(0)).to(h.device).clamp(min=1).unsqueeze(-1)
            return aggregated / degree


    class ContextAwareTauPredictor(nn.Module):
        """Predict positive tau corrections conditioned on local and global context."""

        def __init__(
            self,
            hidden_dim: int,
            tau_min: float = 0.1,
            tau_max: float = 1.5,
            use_contextual_tau: bool = True,
        ):
            super().__init__()
            self.tau_min = float(tau_min)
            self.tau_max = float(tau_max)
            self.use_contextual_tau = bool(use_contextual_tau)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim * 4 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2),
            )

        def forward(self, prior_tau, atom_state, messages, mol_context, steric_context):
            prior_tau = prior_tau.clamp(min=self.tau_min, max=self.tau_max)
            if not self.use_contextual_tau:
                tau = prior_tau
                return tau, {
                    "mean": float(tau.detach().mean().item()),
                    "std": float(tau.detach().std().item()),
                    "adjustment_abs_mean": 0.0,
                    "prior_correlation": 1.0,
                }
            if mol_context is None:
                mol_context = torch.zeros_like(atom_state)
            if steric_context is None:
                steric_context = torch.zeros_like(atom_state)
            params = self.net(
                torch.cat(
                    [
                        prior_tau.unsqueeze(-1),
                        atom_state,
                        messages,
                        mol_context,
                        steric_context,
                    ],
                    dim=-1,
                )
            )
            delta = 0.5 * torch.tanh(params[:, 0])
            blend = torch.sigmoid(params[:, 1])
            tau_candidate = torch.exp(torch.log(prior_tau) + delta)
            tau = (1.0 - blend) * prior_tau + blend * tau_candidate
            tau = tau.clamp(min=self.tau_min, max=self.tau_max)
            prior_centered = prior_tau - prior_tau.mean()
            tau_centered = tau - tau.mean()
            denom = torch.sqrt((prior_centered.square().sum() * tau_centered.square().sum()).clamp(min=1.0e-8))
            corr = float((prior_centered * tau_centered).sum().detach().item() / denom.detach().item())
            return tau, {
                "mean": float(tau.detach().mean().item()),
                "std": float(tau.detach().std().item()),
                "adjustment_abs_mean": float((tau - prior_tau).detach().abs().mean().item()),
                "prior_correlation": corr,
            }


    class ContextualLTCLayer(nn.Module):
        """Edge-aware liquid layer with context-aware tau blending."""

        def __init__(
            self,
            hidden_dim: int,
            edge_feature_dim: int,
            ode_steps: int = 6,
            dropout: float = 0.1,
            tau_min: float = 0.1,
            tau_max: float = 1.5,
            use_contextual_tau: bool = True,
        ):
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.ode_steps = max(1, int(ode_steps))
            self.message_passing = EdgeAwareMessagePassing(hidden_dim, edge_feature_dim, dropout=dropout)
            self.target_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.tau_predictor = ContextAwareTauPredictor(
                hidden_dim=hidden_dim,
                tau_min=tau_min,
                tau_max=tau_max,
                use_contextual_tau=use_contextual_tau,
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, h, edge_index, *, batch, edge_attr=None, prior_tau=None, mol_context=None, steric_context=None):
            if prior_tau is None:
                prior_tau = torch.full((h.size(0),), 0.5, device=h.device, dtype=h.dtype)
            if mol_context is None:
                num_molecules = int(batch.max().item()) + 1 if batch.numel() else 0
                mol_context = scatter_mean(h, batch, num_molecules)[batch] if num_molecules else torch.zeros_like(h)
            messages = self.message_passing(h, edge_index, edge_attr=edge_attr)
            tau, tau_stats = self.tau_predictor(prior_tau, h, messages, mol_context, steric_context)
            dt = 1.0 / float(self.ode_steps)

            def dynamics(state):
                target = self.target_net(torch.cat([state, messages], dim=-1))
                return (-state + target) / tau.unsqueeze(-1)

            state = h
            for _ in range(self.ode_steps):
                k1 = dynamics(state)
                k2 = dynamics(state + 0.5 * dt * k1)
                k3 = dynamics(state + 0.5 * dt * k2)
                k4 = dynamics(state + dt * k3)
                state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            updated = self.norm(state + self.dropout(messages) + h)
            return updated, tau, tau_stats


    class SharedMetabolismEncoder(nn.Module):
        """Shared encoder before task split."""

        def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: int,
            physics_dim: int,
            edge_feature_dim: int,
            num_layers: int,
            ode_steps: int,
            dropout: float,
            tau_min: float,
            tau_max: float,
            use_contextual_tau: bool,
            manual_dim: int = 0,
            steric_dim: int = 0,
        ):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.manual_adapter = nn.Linear(manual_dim, hidden_dim) if manual_dim > 0 else None
            self.steric_adapter = nn.Linear(steric_dim, hidden_dim) if steric_dim > 0 else None
            self.layers = nn.ModuleList(
                [
                    ContextualLTCLayer(
                        hidden_dim=hidden_dim,
                        edge_feature_dim=edge_feature_dim,
                        ode_steps=ode_steps,
                        dropout=dropout,
                        tau_min=tau_min,
                        tau_max=tau_max,
                        use_contextual_tau=use_contextual_tau,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.fusion = FusionGate(physics_dim, hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            x,
            edge_index,
            *,
            batch,
            tau_init=None,
            edge_attr=None,
            physics_out=None,
            manual_atom_features=None,
            steric_atom_features=None,
        ):
            h = self.dropout(self.input_proj(x))
            if manual_atom_features is not None and self.manual_adapter is not None:
                h = h + self.manual_adapter(manual_atom_features)
            steric_context = None
            if steric_atom_features is not None and self.steric_adapter is not None:
                steric_context = self.steric_adapter(steric_atom_features)
                h = h + steric_context
            tau_history = []
            tau_stats = []
            for layer in self.layers:
                num_molecules = int(batch.max().item()) + 1 if batch.numel() else 0
                mol_context = scatter_mean(h, batch, num_molecules)[batch] if num_molecules else torch.zeros_like(h)
                h, tau, stats = layer(
                    h,
                    edge_index,
                    batch=batch,
                    edge_attr=edge_attr,
                    prior_tau=tau_init,
                    mol_context=mol_context,
                    steric_context=steric_context,
                )
                tau_history.append(tau)
                tau_stats.append(stats)
            if physics_out is None:
                gate_values = torch.ones_like(h)
                fused = h
            else:
                fused, gate_values = self.fusion(physics_out, h)
            return fused, gate_values, tau_history, tau_stats


    class AtomLiquidLayer(nn.Module):
        """Backward-compatible wrapper for the old atom liquid layer API."""

        def __init__(self, in_dim: int, hidden_dim: int, ode_steps: int = 6):
            super().__init__()
            self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
            self.layer = ContextualLTCLayer(hidden_dim, edge_feature_dim=1, ode_steps=ode_steps)

        def forward(self, x, edge_index, h=None, tau_init=None):
            h0 = self.input_proj(x if h is None else h)
            batch = torch.zeros(h0.size(0), dtype=torch.long, device=h0.device)
            return self.layer(h0, edge_index, batch=batch, prior_tau=tau_init)[:2]


    class LiquidBranch(nn.Module):
        """Backward-compatible branch with the new contextual layers."""

        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, ode_steps: int = 6, dropout: float = 0.1):
            super().__init__()
            self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
            self.layers = nn.ModuleList(
                [ContextualLTCLayer(hidden_dim, edge_feature_dim=1, ode_steps=ode_steps, dropout=dropout) for _ in range(num_layers)]
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index, tau_init=None):
            h = self.dropout(self.input_proj(x))
            tau_history = []
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            for layer in self.layers:
                h, tau, _ = layer(h, edge_index, batch=batch, prior_tau=tau_init)
                h = self.dropout(h)
                tau_history.append(tau)
            return h, tau_history
else:  # pragma: no cover
    def scatter_mean(*args, **kwargs):
        require_torch()

    class EdgeAwareMessagePassing:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class ContextAwareTauPredictor:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class ContextualLTCLayer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class SharedMetabolismEncoder:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class AtomLiquidLayer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class LiquidBranch:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
