from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.config import ModelConfig
from enzyme_software.liquid_nn_v2.model.advanced_modules import (
    BarrierCrossingModule,
    DeliberationLoop,
    EnergyLandscape,
    GraphTunneling,
    HigherOrderCoupling,
    PhaseAugmentedState,
    PhysicsResidualBranch,
)
from enzyme_software.liquid_nn_v2.model.branches import CYPBranch, SoMBranch
from enzyme_software.liquid_nn_v2.model.hybrid_modules import LocalTunnelingBias, OutputRefinementHead
from enzyme_software.liquid_nn_v2.model.liquid_branch import SharedMetabolismEncoder, scatter_mean
from enzyme_software.liquid_nn_v2.model.physics_branch import PhysicsBranch
from enzyme_software.liquid_nn_v2.model.priors import ManualEnginePriorEncoder, ResidualFusionHead
from enzyme_software.liquid_nn_v2.model.steric_branch import Steric3DBranch


if TORCH_AVAILABLE:
    class _BaseMetabolismPredictor(nn.Module):
        """Shared construction helpers for baseline and advanced predictors."""

        def __init__(self, config: ModelConfig):
            super().__init__()
            self.config = config
            self.physics_branch = PhysicsBranch(output_dim=config.physics_dim)
            self.manual_priors = ManualEnginePriorEncoder(
                atom_dim=config.som_branch_dim,
                mol_dim=config.cyp_branch_dim,
                num_cyp_classes=config.num_cyp_classes,
            )
            self.steric_branch = Steric3DBranch(config.shared_hidden_dim, config.cyp_branch_dim)
            self.shared_encoder = SharedMetabolismEncoder(
                input_dim=config.atom_input_dim,
                hidden_dim=config.shared_hidden_dim,
                physics_dim=config.physics_dim,
                edge_feature_dim=config.edge_feature_dim,
                num_layers=config.shared_encoder_layers,
                ode_steps=config.ode_steps,
                dropout=config.dropout,
                tau_min=config.tau_min,
                tau_max=config.tau_max,
                use_contextual_tau=config.use_contextual_tau,
                manual_dim=config.som_branch_dim if config.use_manual_engine_priors else 0,
                steric_dim=config.shared_hidden_dim if config.use_3d_branch else 0,
            )
            self.som_branch = SoMBranch(
                input_dim=config.shared_hidden_dim,
                branch_dim=config.som_branch_dim,
                num_layers=config.som_branch_layers,
                edge_feature_dim=config.edge_feature_dim,
                ode_steps=config.ode_steps,
                tau_min=config.tau_min,
                tau_max=config.tau_max,
                use_contextual_tau=config.use_contextual_tau,
                dropout=config.dropout,
                use_cross_atom_attention=config.use_cross_atom_attention,
            )
            self.cyp_branch = CYPBranch(
                input_dim=config.shared_hidden_dim,
                branch_dim=config.cyp_branch_dim,
                num_layers=config.cyp_branch_layers,
                edge_feature_dim=config.edge_feature_dim,
                ode_steps=config.ode_steps,
                tau_min=config.tau_min,
                tau_max=config.tau_max,
                use_contextual_tau=config.use_contextual_tau,
                dropout=config.dropout,
                pooling_hidden_dim=config.group_pooling_hidden_dim,
            )
            self.site_head = ResidualFusionHead(
                input_dim=config.som_branch_dim,
                output_dim=1,
                prior_feature_dim=config.som_branch_dim if config.use_manual_engine_priors else 0,
                hidden_dim=config.som_head_hidden_dim,
                fusion_mode=config.manual_prior_fusion_mode,
                dropout=config.dropout,
            )
            self.cyp_head = ResidualFusionHead(
                input_dim=config.cyp_branch_dim,
                output_dim=config.num_cyp_classes,
                prior_feature_dim=config.cyp_branch_dim if config.use_manual_engine_priors else 0,
                hidden_dim=config.cyp_head_hidden_dim,
                fusion_mode=config.manual_prior_fusion_mode,
                dropout=config.dropout,
            )
            # CYP-to-site conditioning: broadcast CYP logits as per-atom bias before site head
            self.cyp_site_conditioner = nn.Linear(config.num_cyp_classes, 1, bias=False) if config.use_cyp_site_conditioning else None
            # BDE learned prior: residual correction on site logit from named BDE feature
            self.bde_prior = nn.Linear(1, 1, bias=True) if config.use_bde_prior else None
            self.last_gate_values = None
            self.last_tau_history = None

        def _encode_inputs(self, batch):
            x = batch["x"]
            # Append XTB features (6D) to atom feature matrix; zero-fill when unavailable
            xtb = batch.get("xtb_atom_features")
            if xtb is not None:
                x = torch.cat([x, xtb.to(dtype=x.dtype, device=x.device)], dim=-1)
            elif x.shape[-1] < self.config.atom_input_dim:
                pad = x.new_zeros(x.shape[0], self.config.atom_input_dim - x.shape[-1])
                x = torch.cat([x, pad], dim=-1)
            edge_index = batch["edge_index"]
            edge_attr = batch.get("edge_attr")
            mol_batch = batch["batch"]
            tau_init = batch.get("tau_init")
            physics_features = batch["physics_features"]
            group_assignments = batch.get("group_assignments")
            group_membership = batch.get("group_membership")
            num_atoms = int(x.size(0))
            num_molecules = int(mol_batch.max().item()) + 1 if mol_batch.numel() else 0

            physics_out = self.physics_branch(physics_features)
            prior_payload = self.manual_priors(batch, num_atoms=num_atoms, num_molecules=num_molecules, device=x.device)
            steric_payload = self.steric_branch(
                batch.get("atom_3d_features") if self.config.use_3d_branch else None,
                mol_batch,
                num_molecules,
            )
            shared_atoms, gate_values, shared_tau_history, shared_tau_stats = self.shared_encoder(
                x,
                edge_index,
                batch=mol_batch,
                tau_init=tau_init,
                edge_attr=edge_attr,
                physics_out=physics_out,
                manual_atom_features=prior_payload.get("atom_prior_embedding") if self.config.use_manual_engine_priors else None,
                steric_atom_features=steric_payload.get("atom_embedding") if self.config.use_3d_branch else None,
            )
            return {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "batch": mol_batch,
                "tau_init": tau_init,
                "physics_out": physics_out,
                "bde_values": physics_features.get("bde_values"),
                "prior_payload": prior_payload,
                "steric_payload": steric_payload,
                "shared_atoms": shared_atoms,
                "group_assignments": group_assignments,
                "group_membership": group_membership,
                "shared_tau_history": shared_tau_history,
                "shared_tau_stats": shared_tau_stats,
                "gate_values": gate_values,
                "num_molecules": num_molecules,
            }

        def _apply_cyp_conditioning(self, som_features, cyp_logits, mol_batch):
            """Broadcast CYP logit projection as a per-atom bias."""
            if self.cyp_site_conditioner is None:
                return som_features
            cyp_bias = self.cyp_site_conditioner(cyp_logits.detach())  # (B_mol, 1)
            return som_features + cyp_bias[mol_batch]                   # (N_atoms, branch_dim) broadcast

        def _apply_bde_prior(self, site_logits, bde_values):
            """Add learnable BDE→logit residual from named physics features."""
            if self.bde_prior is None:
                return site_logits
            if bde_values is None:
                return site_logits
            if bde_values.ndim == 1:
                bde_values = bde_values.unsqueeze(-1)
            bde_values = torch.nan_to_num(bde_values.to(dtype=site_logits.dtype, device=site_logits.device), nan=0.0, posinf=500.0, neginf=0.0)
            # Normalize raw BDE values to a stable [0,1] range expected by the prior.
            if float(bde_values.detach().abs().max().item()) > 2.0:
                bde_values = ((bde_values - 250.0) / 250.0).clamp(0.0, 1.0)
            return site_logits + self.bde_prior(bde_values)

        def _build_common_outputs(
            self,
            encoded,
            som_payload,
            cyp_payload,
            site_logits,
            cyp_logits,
            site_residual,
            cyp_residual,
            *,
            final_atom_features=None,
            extra=None,
        ):
            tau_history = [*encoded["shared_tau_history"], *som_payload["tau_history"], *cyp_payload["tau_history"]]
            tau_stats = {
                "shared": encoded["shared_tau_stats"],
                "som": som_payload["tau_stats"],
                "cyp": cyp_payload["tau_stats"],
            }
            residual_stats = {
                "som": site_residual["diagnostics"],
                "cyp": cyp_residual["diagnostics"],
            }
            diagnostics = {
                "manual_priors": encoded["prior_payload"]["diagnostics"],
                "steric": encoded["steric_payload"]["diagnostics"],
                "som": som_payload["diagnostics"],
                "tau": tau_stats,
                "residual": residual_stats,
            }
            if extra:
                diagnostics.update(extra)
            self.last_gate_values = encoded["gate_values"].detach()
            self.last_tau_history = [tau.detach() for tau in tau_history]
            return {
                "atom_logits": site_logits,
                "site_logits": site_logits,
                "site_scores": torch.sigmoid(site_logits),
                "cyp_logits": cyp_logits,
                "atom_features": final_atom_features if final_atom_features is not None else som_payload["atom_features"],
                "som_atom_features": som_payload["atom_features"],
                "shared_atom_features": encoded["shared_atoms"],
                "gate_values": encoded["gate_values"],
                "tau_history": tau_history,
                "tau_stats": tau_stats,
                "group_embeddings": cyp_payload["group_embeddings"],
                "group_features": cyp_payload["group_embeddings"],
                "group_mask": cyp_payload["group_mask"],
                "mol_features": cyp_payload["mol_features"],
                "attention_weights": cyp_payload["attention_weights"],
                "residual_stats": residual_stats,
                "prior_logits": {
                    "atom": encoded["prior_payload"].get("atom_prior_logits"),
                    "cyp": encoded["prior_payload"].get("cyp_prior_logits"),
                },
                "residual_outputs": {
                    "atom": site_residual["residual_logits"],
                    "cyp": cyp_residual["residual_logits"],
                },
                "diagnostics": diagnostics if self.config.return_intermediate_stats else {},
            }


    class BaselineLiquidMetabolismPredictor(_BaseMetabolismPredictor):
        """Current stable multitask predictor preserved as the baseline model."""

        def forward(self, batch):
            encoded = self._encode_inputs(batch)
            som_payload = self.som_branch(
                encoded["shared_atoms"],
                encoded["batch"],
                edge_index=encoded["edge_index"],
                edge_attr=encoded["edge_attr"],
                tau_init=encoded["tau_init"],
                steric_atom=encoded["steric_payload"].get("atom_embedding") if self.config.use_3d_branch else None,
            )
            cyp_payload = self.cyp_branch(
                encoded["shared_atoms"],
                encoded["batch"],
                edge_index=encoded["edge_index"],
                edge_attr=encoded["edge_attr"],
                tau_init=encoded["tau_init"],
                group_membership=encoded["group_membership"] if self.config.use_hierarchical_pooling else None,
                group_assignments=encoded["group_assignments"],
                manual_mol=encoded["prior_payload"].get("mol_prior_embedding") if self.config.use_manual_engine_priors else None,
                steric_atom=encoded["steric_payload"].get("atom_embedding") if self.config.use_3d_branch else None,
                steric_mol=encoded["steric_payload"].get("mol_embedding") if self.config.use_3d_branch else None,
                som_summary=som_payload["mol_summary"],
            )
            cyp_logits, cyp_residual = self.cyp_head(
                cyp_payload["mol_features"],
                prior_logits=encoded["prior_payload"].get("cyp_prior_logits") if self.config.use_manual_engine_priors else None,
                prior_features=encoded["prior_payload"].get("mol_prior_embedding") if self.config.use_manual_engine_priors else None,
            )
            som_features = self._apply_cyp_conditioning(som_payload["atom_features"], cyp_logits, encoded["batch"])
            site_logits, site_residual = self.site_head(
                som_features,
                prior_logits=encoded["prior_payload"].get("atom_prior_logits") if self.config.use_manual_engine_priors else None,
                prior_features=encoded["prior_payload"].get("atom_prior_embedding") if self.config.use_manual_engine_priors else None,
            )
            site_logits = self._apply_bde_prior(site_logits, encoded.get("bde_values"))
            return self._build_common_outputs(
                encoded,
                som_payload,
                cyp_payload,
                site_logits,
                cyp_logits,
                site_residual,
                cyp_residual,
                final_atom_features=som_features,
                extra={"model_variant": "baseline"},
            )


    class AdvancedLiquidMetabolismPredictor(_BaseMetabolismPredictor):
        """Physics-aware upgraded predictor with optional advanced modules."""

        def __init__(self, config: ModelConfig):
            super().__init__(config)
            self.phase_module = PhaseAugmentedState(
                config.shared_hidden_dim,
                config.phase_hidden_dim,
                phase_scale=config.phase_scale,
                dropout=config.dropout,
            ) if config.use_phase_augmented_state else None
            self.atom_physics_residual = PhysicsResidualBranch(
                config.physics_dim,
                config.som_branch_dim,
                config.physics_residual_hidden_dim,
            ) if config.use_physics_residual else None
            self.mol_physics_residual = PhysicsResidualBranch(
                config.physics_dim,
                config.cyp_branch_dim,
                config.physics_residual_hidden_dim,
            ) if config.use_physics_residual else None
            self.energy_module = EnergyLandscape(
                config.som_branch_dim,
                config.energy_hidden_dim,
                dropout=config.dropout,
                energy_value_clip=config.energy_value_clip,
            ) if config.use_energy_module else None
            self.tunneling_module = BarrierCrossingModule(
                config.som_branch_dim,
                config.tunneling_hidden_dim,
                alpha_init=config.tunneling_alpha_init,
                barrier_min=config.tunneling_barrier_min,
                barrier_max=config.tunneling_barrier_max,
                probability_floor=config.tunneling_probability_floor,
                dropout=config.dropout,
            ) if config.use_tunneling_module else None
            self.graph_tunneling = GraphTunneling(
                config.som_branch_dim,
                projection_dim=config.graph_tunneling_dim,
                max_edges_per_node=config.max_tunneling_edges_per_node,
                dropout=config.dropout,
                residual_scale=config.tunnel_residual_scale,
                residual_scale_max=config.tunnel_residual_scale_max,
            ) if config.use_graph_tunneling else None
            self.higher_order = HigherOrderCoupling(
                config.som_branch_dim,
                config.higher_order_hidden_dim,
                topk=config.higher_order_topk,
                heads=config.higher_order_heads,
                dropout=config.dropout,
            ) if config.use_higher_order_coupling else None
            self.deliberation = DeliberationLoop(
                atom_dim=config.som_branch_dim,
                mol_dim=config.cyp_branch_dim,
                num_cyp_classes=config.num_cyp_classes,
                hidden_dim=config.deliberation_hidden_dim,
                num_steps=config.num_deliberation_steps,
                dropout=config.dropout,
                step_scale=config.deliberation_step_scale,
                max_state_norm=config.deliberation_max_state_norm,
            ) if config.use_deliberation_loop else None

        def forward(self, batch):
            encoded = self._encode_inputs(batch)
            shared_atoms = encoded["shared_atoms"]
            phase_stats = {}
            phase_values = None
            if self.phase_module is not None:
                shared_atoms, phase_values, phase_stats = self.phase_module(shared_atoms)
            som_payload = self.som_branch(
                shared_atoms,
                encoded["batch"],
                edge_index=encoded["edge_index"],
                edge_attr=encoded["edge_attr"],
                tau_init=encoded["tau_init"],
                steric_atom=encoded["steric_payload"].get("atom_embedding") if self.config.use_3d_branch else None,
            )
            cyp_payload = self.cyp_branch(
                shared_atoms,
                encoded["batch"],
                edge_index=encoded["edge_index"],
                edge_attr=encoded["edge_attr"],
                tau_init=encoded["tau_init"],
                group_membership=encoded["group_membership"] if self.config.use_hierarchical_pooling else None,
                group_assignments=encoded["group_assignments"],
                manual_mol=encoded["prior_payload"].get("mol_prior_embedding") if self.config.use_manual_engine_priors else None,
                steric_atom=encoded["steric_payload"].get("atom_embedding") if self.config.use_3d_branch else None,
                steric_mol=encoded["steric_payload"].get("mol_embedding") if self.config.use_3d_branch else None,
                som_summary=som_payload["mol_summary"],
            )

            som_features = som_payload["atom_features"]
            mol_features = cyp_payload["mol_features"]
            physics_atom_gate = None
            physics_mol_gate = None
            physics_stats = {}
            if self.atom_physics_residual is not None and self.mol_physics_residual is not None:
                som_features, physics_atom_gate, atom_stats = self.atom_physics_residual(som_features, encoded["physics_out"])
                mol_physics = scatter_mean(encoded["physics_out"], encoded["batch"], encoded["num_molecules"]) if encoded["num_molecules"] else mol_features.new_zeros((0, encoded["physics_out"].size(-1)))
                mol_features, physics_mol_gate, mol_stats = self.mol_physics_residual(mol_features, mol_physics)
                physics_stats = {"atom": atom_stats, "mol": mol_stats}

            energy_payload = {
                "node_energy": None,
                "group_energy": None,
                "mol_energy": None,
                "stats": {},
            }
            if self.energy_module is not None:
                energy_payload = self.energy_module(
                    som_features,
                    encoded["batch"],
                    mol_hidden=mol_features,
                    group_hidden=cyp_payload["group_embeddings"],
                    group_mask=cyp_payload["group_mask"],
                )
                if self.config.use_energy_dynamics:
                    som_features = som_features - 0.05 * torch.tanh(energy_payload["node_energy"])
                    mol_features = mol_features - 0.05 * torch.tanh(energy_payload["mol_energy"])

            tunneling_payload = {"barrier": None, "tunnel_prob": None, "stats": {}}
            if self.tunneling_module is not None:
                tunneling_payload = self.tunneling_module(som_features)

            graph_tunnel_payload = {
                "message": torch.zeros_like(som_features),
                "edge_index": torch.zeros((2, 0), dtype=torch.long, device=som_features.device),
                "edge_prob": torch.zeros((0, 1), dtype=som_features.dtype, device=som_features.device),
                "stats": {},
            }
            if self.graph_tunneling is not None:
                graph_tunnel_payload = self.graph_tunneling(
                    som_features,
                    encoded["batch"],
                    tunnel_prob=tunneling_payload["tunnel_prob"] if self.config.use_tunneling_for_messages else None,
                )
                som_features = som_features + graph_tunnel_payload["message"]

            higher_order_payload = {
                "update": torch.zeros_like(som_features),
                "selected_indices": torch.zeros((0,), dtype=torch.long, device=som_features.device),
                "stats": {},
            }
            if self.higher_order is not None:
                higher_order_payload = self.higher_order(
                    som_features,
                    encoded["batch"],
                    priority=tunneling_payload["tunnel_prob"],
                )
                som_features = som_features + higher_order_payload["update"]

            deliberation_payload = self.deliberation(
                som_features,
                mol_features,
                encoded["batch"],
                node_energy=energy_payload["node_energy"],
                tunnel_prob=tunneling_payload["tunnel_prob"],
            ) if self.deliberation is not None else {
                "atom_hidden": som_features,
                "mol_hidden": mol_features,
                "site_logits": [],
                "cyp_logits": [],
                "critic_scores": [],
                "stats": {},
            }

            final_atom_features = deliberation_payload["atom_hidden"]
            final_mol_features = deliberation_payload["mol_hidden"]
            cyp_logits, cyp_residual = self.cyp_head(
                final_mol_features,
                prior_logits=encoded["prior_payload"].get("cyp_prior_logits") if self.config.use_manual_engine_priors else None,
                prior_features=encoded["prior_payload"].get("mol_prior_embedding") if self.config.use_manual_engine_priors else None,
            )
            conditioned_atom_features = self._apply_cyp_conditioning(final_atom_features, cyp_logits, encoded["batch"])
            site_logits, site_residual = self.site_head(
                conditioned_atom_features,
                prior_logits=encoded["prior_payload"].get("atom_prior_logits") if self.config.use_manual_engine_priors else None,
                prior_features=encoded["prior_payload"].get("atom_prior_embedding") if self.config.use_manual_engine_priors else None,
            )
            site_logits = self._apply_bde_prior(site_logits, encoded.get("bde_values"))
            if self.config.use_tunneling_module and self.config.use_tunneling_for_site_scores:
                site_logits = site_logits + torch.log(tunneling_payload["tunnel_prob"].clamp(min=1.0e-6))

            cyp_payload = dict(cyp_payload)
            cyp_payload["mol_features"] = final_mol_features
            extra = {
                "model_variant": "advanced",
                "energy": energy_payload["stats"],
                "tunneling": tunneling_payload["stats"],
                "graph_tunneling": graph_tunnel_payload["stats"],
                "phase": phase_stats,
                "higher_order": higher_order_payload["stats"],
                "physics_residual": physics_stats,
                "deliberation": deliberation_payload["stats"],
                "hidden_norms": {
                    "som_features_mean": float(som_features.detach().norm(dim=-1).mean().item()) if som_features.numel() else 0.0,
                    "mol_features_mean": float(mol_features.detach().norm(dim=-1).mean().item()) if mol_features.numel() else 0.0,
                    "final_atom_mean": float(final_atom_features.detach().norm(dim=-1).mean().item()) if final_atom_features.numel() else 0.0,
                    "final_mol_mean": float(final_mol_features.detach().norm(dim=-1).mean().item()) if final_mol_features.numel() else 0.0,
                },
            }
            outputs = self._build_common_outputs(
                encoded,
                som_payload,
                cyp_payload,
                site_logits,
                cyp_logits,
                site_residual,
                cyp_residual,
                final_atom_features=conditioned_atom_features,
                extra=extra,
            )
            outputs.update(
                {
                    "energy_outputs": energy_payload,
                    "tunneling_outputs": tunneling_payload,
                    "graph_tunneling_outputs": graph_tunnel_payload,
                    "phase_outputs": {
                        "phase": phase_values,
                        "stats": phase_stats,
                    } if self.phase_module is not None else {},
                    "higher_order_outputs": higher_order_payload,
                    "physics_residual_outputs": {
                        "atom_gate": physics_atom_gate,
                        "mol_gate": physics_mol_gate,
                        "stats": physics_stats,
                    },
                    "deliberation_outputs": {
                        "site_logits": deliberation_payload["site_logits"],
                        "cyp_logits": deliberation_payload["cyp_logits"],
                        "critic_scores": deliberation_payload["critic_scores"],
                        "stats": deliberation_payload["stats"],
                    },
                }
            )
            return outputs


    class SelectiveHybridLiquidMetabolismPredictor(_BaseMetabolismPredictor):
        """Experimental selective hybrid model with local tunneling bias and one-step output refinement."""

        def __init__(self, config: ModelConfig):
            super().__init__(config)
            self.local_tunneling = BarrierCrossingModule(
                config.som_branch_dim,
                config.tunneling_hidden_dim,
                alpha_init=config.tunneling_alpha_init,
                barrier_min=config.tunneling_barrier_min,
                barrier_max=config.tunneling_barrier_max,
                probability_floor=config.tunneling_probability_floor,
                dropout=config.dropout,
            ) if config.use_local_tunneling_bias else None
            self.tunneling_bias = LocalTunnelingBias(
                scale=config.local_tunneling_scale,
                clamp_value=config.local_tunneling_clamp,
            ) if config.use_local_tunneling_bias else None
            self.output_refiner = OutputRefinementHead(
                atom_dim=config.som_branch_dim,
                hidden_dim=config.output_refinement_hidden_dim,
                scale=config.output_refinement_scale,
                dropout=config.dropout,
            ) if config.use_output_refinement else None

        def forward(self, batch):
            encoded = self._encode_inputs(batch)
            som_payload = self.som_branch(
                encoded["shared_atoms"],
                encoded["batch"],
                edge_index=encoded["edge_index"],
                edge_attr=encoded["edge_attr"],
                tau_init=encoded["tau_init"],
                steric_atom=encoded["steric_payload"].get("atom_embedding") if self.config.use_3d_branch else None,
            )
            cyp_payload = self.cyp_branch(
                encoded["shared_atoms"],
                encoded["batch"],
                edge_index=encoded["edge_index"],
                edge_attr=encoded["edge_attr"],
                tau_init=encoded["tau_init"],
                group_membership=encoded["group_membership"] if self.config.use_hierarchical_pooling else None,
                group_assignments=encoded["group_assignments"],
                manual_mol=encoded["prior_payload"].get("mol_prior_embedding") if self.config.use_manual_engine_priors else None,
                steric_atom=encoded["steric_payload"].get("atom_embedding") if self.config.use_3d_branch else None,
                steric_mol=encoded["steric_payload"].get("mol_embedding") if self.config.use_3d_branch else None,
                som_summary=som_payload["mol_summary"],
            )
            cyp_logits, cyp_residual = self.cyp_head(
                cyp_payload["mol_features"],
                prior_logits=encoded["prior_payload"].get("cyp_prior_logits") if self.config.use_manual_engine_priors else None,
                prior_features=encoded["prior_payload"].get("mol_prior_embedding") if self.config.use_manual_engine_priors else None,
            )
            som_features = self._apply_cyp_conditioning(som_payload["atom_features"], cyp_logits, encoded["batch"])
            site_logits, site_residual = self.site_head(
                som_features,
                prior_logits=encoded["prior_payload"].get("atom_prior_logits") if self.config.use_manual_engine_priors else None,
                prior_features=encoded["prior_payload"].get("atom_prior_embedding") if self.config.use_manual_engine_priors else None,
            )
            site_logits = self._apply_bde_prior(site_logits, encoded.get("bde_values"))

            tunneling_payload = {"barrier": None, "tunnel_prob": None, "stats": {}}
            tunnel_bias = torch.zeros_like(site_logits)
            tunnel_bias_stats = {
                "tunnel_prob_mean": 0.0,
                "tunnel_bias_mean": 0.0,
                "tunnel_bias_max": 0.0,
            }
            if self.local_tunneling is not None and self.tunneling_bias is not None:
                tunneling_payload = self.local_tunneling(som_payload["atom_features"])
                site_logits, tunnel_bias, tunnel_bias_stats = self.tunneling_bias(
                    site_logits,
                    tunneling_payload["tunnel_prob"],
                )

            refine_delta = torch.zeros_like(site_logits)
            refine_gate = torch.zeros_like(site_logits)
            refinement_stats = {
                "refine_gate_mean": 0.0,
                "refine_delta_mean": 0.0,
                "refine_delta_max": 0.0,
            }
            if self.output_refiner is not None:
                mol_context = cyp_payload["mol_features"][encoded["batch"]] if cyp_payload["mol_features"].numel() else None
                site_logits, refine_delta, refine_gate, refinement_stats = self.output_refiner(
                    som_payload["atom_features"],
                    site_logits,
                    mol_context=mol_context.mean(dim=-1, keepdim=True) if mol_context is not None else None,
                )

            outputs = self._build_common_outputs(
                encoded,
                som_payload,
                cyp_payload,
                site_logits,
                cyp_logits,
                site_residual,
                cyp_residual,
                final_atom_features=som_features,
                extra={
                    "model_variant": "hybrid_selective",
                    "hybrid_selective": {
                        **tunnel_bias_stats,
                        **refinement_stats,
                    },
                },
            )
            outputs.update(
                {
                    "tunneling_outputs": tunneling_payload,
                    "hybrid_outputs": {
                        "tunnel_bias": tunnel_bias,
                        "refine_delta": refine_delta,
                        "refine_gate": refine_gate,
                        "stats": {
                            **tunnel_bias_stats,
                            **refinement_stats,
                        },
                    },
                }
            )
            return outputs


    class LiquidMetabolismNetV2(nn.Module):
        """Config-driven wrapper that instantiates the baseline or advanced predictor."""

        def __init__(self, config: ModelConfig):
            super().__init__()
            self.config = config
            if config.model_variant == "advanced":
                self.impl = AdvancedLiquidMetabolismPredictor(config)
            elif config.model_variant == "hybrid_selective":
                self.impl = SelectiveHybridLiquidMetabolismPredictor(config)
            else:
                self.impl = BaselineLiquidMetabolismPredictor(config)

        def forward(self, batch):
            return self.impl(batch)

        @property
        def last_gate_values(self):
            return getattr(self.impl, "last_gate_values", None)

        @property
        def last_tau_history(self):
            return getattr(self.impl, "last_tau_history", None)
else:  # pragma: no cover
    class BaselineLiquidMetabolismPredictor:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class AdvancedLiquidMetabolismPredictor:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class SelectiveHybridLiquidMetabolismPredictor:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class LiquidMetabolismNetV2:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
