from __future__ import annotations

from typing import Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.model.liquid_branch import ContextualLTCLayer, scatter_mean
from enzyme_software.liquid_nn_v2.model.pooling import (
    ChemistryHierarchicalPooling,
    build_group_membership_tensor,
    pack_atom_features,
    segment_softmax,
    segment_sum,
)


if TORCH_AVAILABLE:
    class SoMBranch(nn.Module):
        """Atom-focused branch with within-molecule competition refinement."""

        def __init__(
            self,
            *,
            input_dim: int,
            branch_dim: int,
            num_layers: int,
            edge_feature_dim: int,
            ode_steps: int,
            tau_min: float,
            tau_max: float,
            use_contextual_tau: bool,
            dropout: float,
        ):
            super().__init__()
            self.atom_proj = nn.Sequential(nn.Linear(input_dim, branch_dim), nn.LayerNorm(branch_dim), nn.SiLU())
            self.steric_adapter = nn.Linear(branch_dim, branch_dim)
            self.layers = nn.ModuleList(
                [
                    ContextualLTCLayer(
                        hidden_dim=branch_dim,
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
            self.competition_score = nn.Linear(branch_dim, 1)
            self.competition_proj = nn.Sequential(
                nn.Linear(branch_dim * 2, branch_dim),
                nn.SiLU(),
                nn.LayerNorm(branch_dim),
            )

        def forward(self, shared_atoms, batch, *, edge_index, edge_attr=None, tau_init=None, steric_atom=None):
            h = self.atom_proj(shared_atoms)
            steric_context = self.steric_adapter(steric_atom) if steric_atom is not None else None
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
            num_molecules = int(batch.max().item()) + 1 if batch.numel() else 0
            competition_weights = segment_softmax(self.competition_score(h), batch, num_molecules)
            competition_summary = segment_sum(h * competition_weights, batch, num_molecules)[batch]
            refined = self.competition_proj(torch.cat([h, competition_summary], dim=-1))
            diagnostics = {
                "competition_weight_mean": float(competition_weights.detach().mean().item()),
            }
            return {
                "atom_features": refined,
                "mol_summary": scatter_mean(refined, batch, num_molecules) if num_molecules else refined.new_zeros((0, refined.size(-1))),
                "tau_history": tau_history,
                "tau_stats": tau_stats,
                "diagnostics": diagnostics,
            }


    class CYPBranch(nn.Module):
        """Global branch with chemistry-aware hierarchical pooling."""

        def __init__(
            self,
            *,
            input_dim: int,
            branch_dim: int,
            num_layers: int,
            edge_feature_dim: int,
            ode_steps: int,
            tau_min: float,
            tau_max: float,
            use_contextual_tau: bool,
            dropout: float,
            pooling_hidden_dim: int,
        ):
            super().__init__()
            self.atom_proj = nn.Sequential(nn.Linear(input_dim, branch_dim), nn.LayerNorm(branch_dim), nn.SiLU())
            self.manual_adapter = nn.Linear(branch_dim, branch_dim)
            self.steric_adapter = nn.Linear(branch_dim, branch_dim)
            self.som_adapter = nn.Linear(branch_dim, branch_dim)
            self.layers = nn.ModuleList(
                [
                    ContextualLTCLayer(
                        hidden_dim=branch_dim,
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
            self.pooling = ChemistryHierarchicalPooling(branch_dim, pooling_hidden_dim)
            self.context_fuse = nn.Sequential(
                nn.Linear(branch_dim * 4, branch_dim),
                nn.SiLU(),
                nn.LayerNorm(branch_dim),
            )

        def forward(
            self,
            shared_atoms,
            batch,
            *,
            edge_index,
            edge_attr=None,
            tau_init=None,
            group_membership=None,
            group_assignments=None,
            manual_mol=None,
            steric_atom=None,
            steric_mol=None,
            som_summary=None,
        ):
            h = self.atom_proj(shared_atoms)
            steric_context = self.steric_adapter(steric_atom) if steric_atom is not None else None
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
            padded_atoms, atom_mask = pack_atom_features(h, batch)
            if group_membership is None and group_assignments is not None:
                group_membership = build_group_membership_tensor(group_assignments, batch, max_atoms=padded_atoms.size(1))
            pooled = self.pooling(padded_atoms, group_membership, atom_mask)
            num_molecules = padded_atoms.size(0)
            branch_dim = h.size(-1)
            manual_mol = self.manual_adapter(manual_mol) if manual_mol is not None else h.new_zeros((num_molecules, branch_dim))
            steric_mol = self.steric_adapter(steric_mol) if steric_mol is not None else h.new_zeros((num_molecules, branch_dim))
            som_summary = self.som_adapter(som_summary) if som_summary is not None else h.new_zeros((num_molecules, branch_dim))
            mol_features = self.context_fuse(
                torch.cat(
                    [
                        pooled["molecule_embedding"],
                        manual_mol,
                        steric_mol,
                        som_summary,
                    ],
                    dim=-1,
                )
            )
            return {
                "mol_features": mol_features,
                "group_embeddings": pooled["group_embeddings"],
                "group_mask": pooled["group_mask"],
                "attention_weights": pooled["attention_weights"],
                "tau_history": tau_history,
                "tau_stats": tau_stats,
            }
else:  # pragma: no cover
    class SoMBranch:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class CYPBranch:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
