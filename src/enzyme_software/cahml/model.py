from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.cahml.components import (
    AtomChemistryEncoder,
    ChemistryAwareGating,
    DisagreementResolver,
    HierarchicalPredictor,
    MoleculeEncoder,
    PhysicsConstraints,
    UncertaintyEstimator,
)
from enzyme_software.cahml.config import CAHMLConfig, REACTION_TYPES


if TORCH_AVAILABLE:
    class CAHML(nn.Module):
        def __init__(self, config: Optional[CAHMLConfig] = None):
            super().__init__()
            self.config = config or CAHMLConfig()
            self.mol_encoder = MoleculeEncoder(
                fingerprint_dim=self.config.mol_fingerprint_dim,
                descriptor_dim=self.config.mol_descriptor_dim,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
            )
            self.atom_encoder = AtomChemistryEncoder(
                raw_feature_dim=self.config.atom_raw_feature_dim,
                smarts_dim=self.config.smarts_pattern_dim,
                hidden_dim=self.config.hidden_dim // 2,
                dropout=self.config.dropout,
            )
            self.chemistry_gating = ChemistryAwareGating(
                mol_feature_dim=self.mol_encoder.output_dim,
                atom_feature_dim=self.atom_encoder.output_dim,
                n_models=self.config.n_models,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
            )
            self.uncertainty_estimator = UncertaintyEstimator(
                n_models=self.config.n_models,
                atom_feature_dim=self.atom_encoder.output_dim,
                hidden_dim=max(16, self.config.hidden_dim // 2),
            )
            self.disagreement_resolver = DisagreementResolver(
                n_models=self.config.n_models,
                atom_feature_dim=self.atom_encoder.output_dim,
                hidden_dim=self.config.hidden_dim,
            )
            self.hierarchical_predictor = HierarchicalPredictor(
                mol_feature_dim=self.mol_encoder.output_dim,
                atom_feature_dim=self.atom_encoder.output_dim,
                n_models=self.config.n_models,
                n_cyp_classes=self.config.n_cyp_classes,
                n_reaction_types=self.config.n_reaction_types,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
                use_base_cyp_prior=self.config.use_base_cyp_prior,
            )
            self.physics_constraints = PhysicsConstraints(
                boost_factor=self.config.constraint_boost_factor,
                penalty_factor=self.config.constraint_penalty_factor,
            )

        @staticmethod
        def _normalize_site_predictions(base_site_predictions: torch.Tensor) -> torch.Tensor:
            scores = base_site_predictions.float().clone()
            # Sigmoid any out-of-range values (logits accidentally passed in)
            mask = (scores < 0.0) | (scores > 1.0)
            scores = torch.where(mask, torch.sigmoid(scores), scores)
            # Rank-normalize each model's column so the gate sees relative
            # orderings rather than raw probabilities. Without this, models
            # that share base weights (micropattern_xtb was fine-tuned from
            # hybrid_lnn) look nearly identical in prob space and the gate
            # ignores the redundant column. Rank percentiles make the
            # reranking signal visible regardless of absolute score scale.
            n_atoms = scores.shape[0]
            if n_atoms > 1:
                for col in range(scores.shape[1]):
                    col_scores = scores[:, col]
                    order = torch.argsort(col_scores)
                    ranks = torch.empty_like(order, dtype=torch.float32)
                    ranks[order] = torch.arange(order.numel(), dtype=torch.float32, device=scores.device)
                    scores[:, col] = ranks / float(n_atoms - 1)
            return scores.clamp(1.0e-4, 1.0 - 1.0e-4)

        def _build_explanation(
            self,
            gate_info: Dict[str, object],
            uncertainty_info: Dict[str, object],
            resolution_info: Dict[str, object],
            hierarchy_info: Dict[str, object],
            constraint_info: Dict[str, object],
            confidence: torch.Tensor,
        ) -> Dict[str, object]:
            model_names = self.config.model_names
            gate_weights = gate_info["gate_weights_mean"]
            if hasattr(gate_weights, "detach"):
                gate_weights = gate_weights.detach().cpu()
            trusted_model_idx = int(torch.argmax(gate_weights).item())
            trusted_model = model_names[trusted_model_idx]
            confidence_mean = float(confidence.mean().item())
            return {
                "trusted_model": trusted_model,
                "trust_reason": f"Chemistry gating assigned {float(gate_weights[trusted_model_idx]):.1%} average weight",
                "confidence_level": "high" if confidence_mean > 0.7 else "medium" if confidence_mean > 0.4 else "low",
                "disagreement_level": "high" if float(resolution_info["frac_high_disagreement"]) > 0.3 else "low",
                "cyp_reasoning": f"Predicted {hierarchy_info['cyp_prediction']} with {float(hierarchy_info['cyp_confidence']):.1%} confidence",
                "reaction_reasoning": f"Predicted {hierarchy_info['reaction_name']}",
                "constraints_applied": int(constraint_info.get("n_rules_applied", 0)),
                "boosted_sites": list(constraint_info.get("boosted_atoms", [])),
                "blocked_sites": list(constraint_info.get("blocked_atoms", [])),
            }

        def forward(
            self,
            mol_features_raw: torch.Tensor,
            atom_features_raw: torch.Tensor,
            smarts_matches: torch.Tensor,
            base_site_predictions: torch.Tensor,
            base_cyp_predictions: torch.Tensor,
        ) -> Dict[str, object]:
            base_site_probs = self._normalize_site_predictions(base_site_predictions)
            mol_features = self.mol_encoder(mol_features_raw)
            atom_features = self.atom_encoder(atom_features_raw, smarts_matches)
            gate_weights, gate_info = self.chemistry_gating(mol_features, atom_features, base_site_probs)
            gated_scores = (base_site_probs * gate_weights).sum(dim=-1)
            adjusted_weights, confidence, uncertainty_info = self.uncertainty_estimator(base_site_probs, atom_features, gate_weights)
            weighted_scores = (base_site_probs * adjusted_weights).sum(dim=-1)
            resolved_scores, resolution_info = self.disagreement_resolver(base_site_probs, atom_features, weighted_scores)
            site_logits, cyp_logits, rxn_logits, hierarchy_info = self.hierarchical_predictor(
                mol_features,
                atom_features,
                resolved_scores.clamp(1.0e-4, 1.0 - 1.0e-4),
                base_cyp_predictions.float(),
            )
            if self.config.use_physics_constraints:
                site_logits, constraint_info = self.physics_constraints.apply_constraints(atom_features_raw, smarts_matches, site_logits)
            else:
                constraint_info = {"n_rules_applied": 0, "blocked_atoms": [], "boosted_atoms": [], "penalized_atoms": [], "rules": []}
            site_probs = torch.sigmoid(site_logits)
            site_ranking = torch.argsort(site_probs, descending=True)
            cyp_probs = torch.softmax(cyp_logits, dim=-1)
            reaction_probs = torch.softmax(rxn_logits, dim=-1)
            explanation = self._build_explanation(gate_info, uncertainty_info, resolution_info, hierarchy_info, constraint_info, confidence)
            flags = {
                "high_uncertainty": bool(float(confidence.mean().item()) < 1.0 - self.config.uncertainty_threshold),
                "model_disagreement": bool(float(resolution_info["frac_high_disagreement"]) > 0.2),
                "needs_review": bool(float(confidence.mean().item()) < 0.4 or float(resolution_info["frac_high_disagreement"]) > 0.4),
            }
            return {
                "site_scores": site_logits,
                "site_probabilities": site_probs,
                "site_ranking": site_ranking,
                "site_confidence": confidence,
                "cyp_logits": cyp_logits,
                "cyp_prediction": int(torch.argmax(cyp_probs).item()),
                "cyp_confidence": float(cyp_probs.max().item()),
                "reaction_type": hierarchy_info["reaction_name"],
                "reaction_logits": rxn_logits,
                "reaction_confidence": float(reaction_probs.max().item()),
                "explanation": explanation,
                "flags": flags,
                "stage_outputs": {
                    "gating": gate_info,
                    "uncertainty": uncertainty_info,
                    "resolution": resolution_info,
                    "hierarchy": hierarchy_info,
                    "constraints": constraint_info,
                },
            }
else:  # pragma: no cover
    class CAHML:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
