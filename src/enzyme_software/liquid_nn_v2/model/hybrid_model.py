from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.features.xtb_features import FULL_XTB_FEATURE_DIM
from enzyme_software.liquid_nn_v2.features.route_prior import combine_lnn_with_prior
from enzyme_software.liquid_nn_v2.model.accessibility import AccessibilityHead
from enzyme_software.liquid_nn_v2.model.barrier import BarrierHead
from enzyme_software.liquid_nn_v2.model.event_context import SparseEventContext
from enzyme_software.liquid_nn_v2.model.hybrid_modules import TopKCrossAtomReranker
from enzyme_software.liquid_nn_v2.model.nexus_bridge import NexusHybridBridge
from enzyme_software.liquid_nn_v2.model.precedent_logbook import AuditedEpisodeLogbook
from enzyme_software.liquid_nn_v2.model.wave_field import WholeMoleculeWaveField


if TORCH_AVAILABLE:
    class HybridLNNModel(nn.Module):
        """Hybrid wrapper that adds manual priors plus a lightweight NEXUS bridge."""

        def __init__(self, base_lnn, prior_weight_init: float = 0.3):
            super().__init__()
            self.base_lnn = base_lnn
            self.config = getattr(base_lnn, "config", None)
            prior_weight_init = min(max(float(prior_weight_init), 1.0e-3), 1.0 - 1.0e-3)
            self.prior_weight_logit = nn.Parameter(torch.logit(torch.tensor(prior_weight_init)))
            self.nexus_bridge = None
            self.site_arbiter_head = None
            self.site_arbiter_uses_bridge = False
            self.nexus_sideinfo_uses_bridge = False
            self.nexus_sideinfo_proj = None
            self.nexus_sideinfo_gate = None
            self.nexus_sideinfo_scale_logit = None
            self.local_chem_proj = None
            self.local_chem_gate = None
            self.local_chem_scale_logit = None
            self.local_chem_logit_head = None
            self.local_chem_logit_scale_logit = None
            self.event_context_module = None
            self.accessibility_head = None
            self.barrier_head = None
            self.phase2_context_proj = None
            self.phase2_context_gate = None
            self.phase2_context_scale_logit = None
            self.phase2_context_logit_head = None
            self.phase2_context_logit_scale_logit = None
            self.phase5_proposer_proj = None
            self.phase5_proposer_gate = None
            self.phase5_proposer_scale_logit = None
            self.phase5_proposer_logit_head = None
            self.phase5_proposer_logit_scale_logit = None
            self.topk_reranker = None
            self.domain_adv_head = None
            domain_adv_weight = float(getattr(self.config, "domain_adv_weight", 0.0))
            if domain_adv_weight > 0.0:
                domain_hidden = int(getattr(self.config, "domain_adv_hidden_dim", 64))
                mol_dim = int(getattr(self.config, "cyp_branch_dim", getattr(self.config, "mol_dim", 128)))
                self.domain_adv_head = nn.Sequential(
                    nn.Linear(mol_dim, domain_hidden),
                    nn.SiLU(),
                    nn.Dropout(float(getattr(self.config, "dropout", 0.1))),
                    nn.Linear(domain_hidden, 6),
                )
            if bool(getattr(self.config, "use_nexus_bridge", True)):
                atom_dim = int(getattr(self.config, "som_branch_dim", getattr(self.config, "shared_hidden_dim", 128)))
                num_cyp = int(getattr(self.config, "num_cyp_classes", 9))
                steric_dim = int(getattr(self.config, "steric_feature_dim", 8))
                xtb_dim = FULL_XTB_FEATURE_DIM
                topology_dim = int(getattr(self.config, "nexus_topology_feature_dim", 5))
                graph_dim = int(max(16, int(getattr(self.config, "nexus_graph_dim", 48))))
                self.nexus_bridge = NexusHybridBridge(
                    atom_feature_dim=atom_dim,
                    num_cyp_classes=num_cyp,
                    steric_feature_dim=steric_dim,
                    xtb_feature_dim=xtb_dim,
                    topology_feature_dim=topology_dim,
                    wave_hidden_dim=int(getattr(self.config, "nexus_wave_hidden_dim", 64)),
                    graph_dim=graph_dim,
                    memory_capacity=int(getattr(self.config, "nexus_memory_capacity", 4096)),
                    memory_topk=int(getattr(self.config, "nexus_memory_topk", 32)),
                    wave_aux_weight=float(getattr(self.config, "nexus_wave_aux_weight", 0.10)),
                    analogical_aux_weight=float(getattr(self.config, "nexus_analogical_aux_weight", 0.08)),
                    analogical_cyp_aux_scale=float(getattr(self.config, "nexus_analogical_cyp_aux_scale", 0.10)),
                )
                self.nexus_bridge.set_memory_frozen(bool(getattr(self.config, "nexus_memory_frozen", False)))
                self.ana_cyp_weight_logit = nn.Parameter(
                    torch.logit(torch.tensor(float(getattr(self.config, "nexus_analogical_cyp_init", 0.12))))
                )
                if bool(getattr(self.config, "use_nexus_sideinfo_features", False)):
                    sideinfo_hidden = int(getattr(self.config, "nexus_sideinfo_hidden_dim", atom_dim))
                    sideinfo_dropout = float(getattr(self.config, "nexus_sideinfo_dropout", 0.10))
                    wave_sideinfo_in = 5 + int(WholeMoleculeWaveField.field_feature_dim)
                    analogical_sideinfo_in = 15 + num_cyp
                    sideinfo_in = wave_sideinfo_in + analogical_sideinfo_in
                    self.nexus_sideinfo_proj = nn.Sequential(
                        nn.Linear(sideinfo_in, sideinfo_hidden),
                        nn.SiLU(),
                        nn.Dropout(sideinfo_dropout),
                        nn.Linear(sideinfo_hidden, atom_dim),
                    )
                    self.nexus_sideinfo_gate = nn.Sequential(
                        nn.Linear(atom_dim + sideinfo_in, max(32, sideinfo_hidden // 2)),
                        nn.SiLU(),
                        nn.Dropout(sideinfo_dropout),
                        nn.Linear(max(32, sideinfo_hidden // 2), 1),
                        nn.Sigmoid(),
                    )
                    self.nexus_sideinfo_scale_logit = nn.Parameter(
                        torch.logit(torch.tensor(float(getattr(self.config, "nexus_sideinfo_init_scale", 0.20))))
                    )
                    self.nexus_sideinfo_uses_bridge = True
                if bool(getattr(self.config, "use_nexus_site_arbiter", True)):
                    arbiter_hidden = int(getattr(self.config, "nexus_site_arbiter_hidden_dim", 128))
                    arbiter_dropout = float(getattr(self.config, "nexus_site_arbiter_dropout", 0.10))
                    council_hidden = max(32, arbiter_hidden // 2)
                    board_context_dim = (
                        atom_dim
                        + 16
                        + graph_dim
                        + num_cyp
                        + 3
                        + 2
                        + num_cyp
                        + steric_dim
                        + xtb_dim
                        + topology_dim
                        + 10
                        + AuditedEpisodeLogbook.brief_dim
                        + 1
                        + 1
                        + 5
                    )
                    arbiter_in = board_context_dim + 9
                    vote_dropout = min(float(arbiter_dropout) + 0.10, 0.40)
                    self.lnn_vote_head = nn.Sequential(
                        nn.Linear(atom_dim + 1, council_hidden),
                        nn.SiLU(),
                        nn.Dropout(vote_dropout),
                        nn.Linear(council_hidden, 1),
                    )
                    self.wave_vote_head = nn.Sequential(
                        nn.Linear(14, council_hidden),
                        nn.SiLU(),
                        nn.Dropout(vote_dropout),
                        nn.Linear(council_hidden, 1),
                    )
                    self.analogical_vote_head = nn.Sequential(
                        nn.Linear(8 + AuditedEpisodeLogbook.brief_dim, council_hidden),
                        nn.SiLU(),
                        nn.Dropout(vote_dropout),
                        nn.Linear(council_hidden, 1),
                    )
                    self.lnn_conf_head = nn.Sequential(
                        nn.Linear(atom_dim, council_hidden),
                        nn.SiLU(),
                        nn.Dropout(vote_dropout),
                        nn.Linear(council_hidden, 1),
                    )
                    self.wave_conf_head = nn.Sequential(
                        nn.Linear(14, council_hidden),
                        nn.SiLU(),
                        nn.Dropout(vote_dropout),
                        nn.Linear(council_hidden, 1),
                    )
                    self.analogical_conf_head = nn.Sequential(
                        nn.Linear(8 + AuditedEpisodeLogbook.brief_dim, council_hidden),
                        nn.SiLU(),
                        nn.Dropout(vote_dropout),
                        nn.Linear(council_hidden, 1),
                    )
                    self.council_board_head = nn.Sequential(
                        nn.Linear(board_context_dim + 6, arbiter_hidden),
                        nn.SiLU(),
                        nn.Dropout(arbiter_dropout),
                        nn.Linear(arbiter_hidden, 3),
                    )
                    self.site_arbiter_head = nn.Sequential(
                        nn.Linear(arbiter_in, arbiter_hidden),
                        nn.SiLU(),
                        nn.Dropout(arbiter_dropout),
                        nn.Linear(arbiter_hidden, max(32, arbiter_hidden // 2)),
                        nn.SiLU(),
                        nn.Dropout(arbiter_dropout),
                        nn.Linear(max(32, arbiter_hidden // 2), 1),
                    )
                    # Start the arbiter as a no-op residual so training begins from
                    # the base site signal instead of a random council suppression.
                    nn.init.zeros_(self.site_arbiter_head[-1].weight)
                    nn.init.zeros_(self.site_arbiter_head[-1].bias)
                    self.site_arbiter_uses_bridge = True
            if bool(getattr(self.config, "use_local_chemistry_path", False)):
                atom_dim = int(getattr(self.config, "som_branch_dim", getattr(self.config, "shared_hidden_dim", 128)))
                chem_dim = 11 + 7 + 1
                chem_hidden = int(getattr(self.config, "local_chem_hidden_dim", atom_dim))
                chem_dropout = float(getattr(self.config, "local_chem_dropout", 0.05))
                self.local_chem_proj = nn.Sequential(
                    nn.Linear(chem_dim, chem_hidden),
                    nn.SiLU(),
                    nn.Dropout(chem_dropout),
                    nn.Linear(chem_hidden, atom_dim),
                )
                self.local_chem_gate = nn.Sequential(
                    nn.Linear(atom_dim + chem_dim, max(32, chem_hidden // 2)),
                    nn.SiLU(),
                    nn.Dropout(chem_dropout),
                    nn.Linear(max(32, chem_hidden // 2), 1),
                    nn.Sigmoid(),
                )
                self.local_chem_logit_head = nn.Sequential(
                    nn.Linear(chem_dim, max(16, chem_hidden // 2)),
                    nn.SiLU(),
                    nn.Dropout(chem_dropout),
                    nn.Linear(max(16, chem_hidden // 2), 1),
                )
                self.local_chem_scale_logit = nn.Parameter(
                    torch.logit(torch.tensor(float(getattr(self.config, "local_chem_init_scale", 0.08))))
                )
                self.local_chem_logit_scale_logit = nn.Parameter(
                    torch.logit(torch.tensor(float(getattr(self.config, "local_chem_logit_scale", 0.05))))
                )
                gate_linear = self.local_chem_gate[-2]
                if hasattr(gate_linear, "bias") and gate_linear.bias is not None:
                    nn.init.constant_(gate_linear.bias, -1.0)
            if any(
                bool(getattr(self.config, flag, False))
                for flag in ("use_event_context", "use_accessibility_head", "use_barrier_head")
            ):
                atom_dim = int(getattr(self.config, "som_branch_dim", getattr(self.config, "shared_hidden_dim", 128)))
                phase2_hidden = int(getattr(self.config, "phase2_context_hidden_dim", atom_dim))
                phase2_dropout = float(getattr(self.config, "phase2_context_dropout", 0.05))
                ctx_parts = 11 + 1  # local chem + normalized anomaly
                if bool(getattr(self.config, "use_event_context", False)):
                    self.event_context_module = SparseEventContext(
                        atom_dim=atom_dim,
                        hidden_dim=int(getattr(self.config, "event_context_hidden_dim", 24)),
                        rounds=int(getattr(self.config, "event_context_rounds", 3)),
                        dropout=phase2_dropout,
                    )
                    ctx_parts += int(getattr(self.event_context_module, "output_dim", 0))
                if bool(getattr(self.config, "use_accessibility_head", False)):
                    self.accessibility_head = AccessibilityHead(
                        hidden_dim=int(getattr(self.config, "accessibility_hidden_dim", 16)),
                        dropout=phase2_dropout,
                    )
                    ctx_parts += int(getattr(self.accessibility_head, "output_dim", 0))
                if bool(getattr(self.config, "use_barrier_head", False)):
                    self.barrier_head = BarrierHead(
                        hidden_dim=int(getattr(self.config, "barrier_hidden_dim", 32)),
                        dropout=phase2_dropout,
                    )
                    ctx_parts += int(getattr(self.barrier_head, "output_dim", 0))
                self.phase2_context_proj = nn.Sequential(
                    nn.Linear(ctx_parts, phase2_hidden),
                    nn.SiLU(),
                    nn.Dropout(phase2_dropout),
                    nn.Linear(phase2_hidden, atom_dim),
                )
                self.phase2_context_gate = nn.Sequential(
                    nn.Linear(atom_dim + ctx_parts, max(32, phase2_hidden // 2)),
                    nn.SiLU(),
                    nn.Dropout(phase2_dropout),
                    nn.Linear(max(32, phase2_hidden // 2), 1),
                    nn.Sigmoid(),
                )
                self.phase2_context_logit_head = nn.Sequential(
                    nn.Linear(ctx_parts, max(16, phase2_hidden // 2)),
                    nn.SiLU(),
                    nn.Dropout(phase2_dropout),
                    nn.Linear(max(16, phase2_hidden // 2), 1),
                )
                self.phase2_context_scale_logit = nn.Parameter(
                    torch.logit(torch.tensor(float(getattr(self.config, "phase2_context_init_scale", 0.10))))
                )
                self.phase2_context_logit_scale_logit = nn.Parameter(
                    torch.logit(torch.tensor(float(getattr(self.config, "phase2_context_logit_scale", 0.05))))
                )
                phase2_gate_linear = self.phase2_context_gate[-2]
                if hasattr(phase2_gate_linear, "bias") and phase2_gate_linear.bias is not None:
                    nn.init.constant_(phase2_gate_linear.bias, -1.0)
            if any(
                bool(getattr(self.config, flag, False))
                for flag in ("use_phase5_boundary_field", "use_phase5_accessibility", "use_phase5_cyp_profile")
            ):
                atom_dim = int(getattr(self.config, "som_branch_dim", getattr(self.config, "shared_hidden_dim", 128)))
                phase5_in = 18
                phase5_hidden = int(getattr(self.config, "phase5_proposer_hidden_dim", atom_dim))
                phase5_dropout = float(getattr(self.config, "phase5_proposer_dropout", 0.05))
                self.phase5_proposer_proj = nn.Sequential(
                    nn.Linear(phase5_in, phase5_hidden),
                    nn.SiLU(),
                    nn.Dropout(phase5_dropout),
                    nn.Linear(phase5_hidden, atom_dim),
                )
                self.phase5_proposer_gate = nn.Sequential(
                    nn.Linear(atom_dim + phase5_in, max(32, phase5_hidden // 2)),
                    nn.SiLU(),
                    nn.Dropout(phase5_dropout),
                    nn.Linear(max(32, phase5_hidden // 2), 1),
                    nn.Sigmoid(),
                )
                self.phase5_proposer_logit_head = nn.Sequential(
                    nn.Linear(phase5_in, max(16, phase5_hidden // 2)),
                    nn.SiLU(),
                    nn.Dropout(phase5_dropout),
                    nn.Linear(max(16, phase5_hidden // 2), 1),
                )
                self.phase5_proposer_scale_logit = nn.Parameter(
                    torch.logit(torch.tensor(float(getattr(self.config, "phase5_proposer_init_scale", 0.10))))
                )
                self.phase5_proposer_logit_scale_logit = nn.Parameter(
                    torch.logit(torch.tensor(float(getattr(self.config, "phase5_proposer_logit_scale", 0.06))))
                )
                phase5_gate_linear = self.phase5_proposer_gate[-2]
                if hasattr(phase5_gate_linear, "bias") and phase5_gate_linear.bias is not None:
                    nn.init.constant_(phase5_gate_linear.bias, -0.75)
            if bool(getattr(self.config, "use_topk_reranker", False)):
                atom_dim = int(getattr(self.config, "som_branch_dim", getattr(self.config, "shared_hidden_dim", 128)))
                mol_dim = int(getattr(self.config, "cyp_branch_dim", getattr(self.config, "mol_dim", 128)))
                extra_dim = 17
                self.topk_reranker = TopKCrossAtomReranker(
                    atom_dim=atom_dim,
                    mol_dim=mol_dim,
                    extra_dim=extra_dim,
                    hidden_dim=int(getattr(self.config, "topk_reranker_hidden_dim", atom_dim)),
                    top_k=int(getattr(self.config, "topk_reranker_k", 8)),
                    num_heads=int(getattr(self.config, "topk_reranker_heads", 4)),
                    num_layers=int(getattr(self.config, "topk_reranker_layers", 2)),
                    dropout=float(getattr(self.config, "topk_reranker_dropout", 0.10)),
                    residual_scale=float(getattr(self.config, "topk_reranker_residual_scale", 0.75)),
                    use_gate=bool(getattr(self.config, "topk_reranker_use_gate", True)),
                    gate_bias=float(getattr(self.config, "topk_reranker_gate_bias", -2.0)),
                )

        def _optional_feature(self, value, rows: int, width: int, *, device, dtype) -> torch.Tensor:
            if width <= 0:
                return torch.zeros(rows, 0, device=device, dtype=dtype)
            if value is None:
                return torch.zeros(rows, width, device=device, dtype=dtype)
            out = value.to(device=device, dtype=dtype)
            if out.ndim == 1:
                out = out.unsqueeze(-1)
            if out.size(-1) == width:
                return out
            if out.size(-1) > width:
                return out[..., :width]
            return torch.nn.functional.pad(out, (0, width - int(out.size(-1))))

        def _scaled_live(self, value: torch.Tensor, *, enabled: bool, grad_scale: float) -> torch.Tensor:
            base = value.detach()
            if not enabled or grad_scale <= 0.0:
                return base
            scale = float(grad_scale)
            return base + scale * (value - base)

        def _safe_vote_tensor(self, value: torch.Tensor) -> torch.Tensor:
            return torch.nan_to_num(value, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)

        def _match_last_dim(self, value: torch.Tensor, width: int) -> torch.Tensor:
            if int(value.size(-1)) == int(width):
                return value
            if int(value.size(-1)) > int(width):
                return value[..., :width]
            return torch.nn.functional.pad(value, (0, int(width) - int(value.size(-1))))

        def _base_impl(self):
            return getattr(self.base_lnn, "impl", self.base_lnn)

        def _gradient_reverse(self, value: torch.Tensor, scale: float) -> torch.Tensor:
            if scale <= 0.0:
                return value.detach()
            return value.detach() - (float(scale) * (value - value.detach()))

        def _apply_fixed_cyp_context(self, outputs: Dict[str, object]) -> Dict[str, object]:
            fixed_idx = int(getattr(self.config, "fixed_cyp_index", -1))
            if fixed_idx < 0:
                return outputs
            cyp_logits = outputs.get("cyp_logits")
            if cyp_logits is None or not getattr(cyp_logits, "numel", lambda: 0)():
                return outputs
            if fixed_idx >= int(cyp_logits.shape[-1]):
                return outputs
            fixed_logit = float(getattr(self.config, "fixed_cyp_logit", 8.0))
            forced = torch.full_like(cyp_logits, -fixed_logit)
            forced[:, fixed_idx] = fixed_logit
            outputs = dict(outputs)
            outputs.setdefault("cyp_logits_base", cyp_logits)
            outputs["cyp_logits"] = forced
            diagnostics = dict(outputs.get("diagnostics") or {})
            diagnostics["fixed_cyp_context"] = {
                "enabled": 1.0,
                "fixed_cyp_index": float(fixed_idx),
                "fixed_cyp_logit": float(fixed_logit),
            }
            outputs["diagnostics"] = diagnostics
            return outputs

        def _apply_candidate_mask(self, outputs: Dict[str, object], batch: Dict[str, object]) -> Dict[str, object]:
            candidate_mask = batch.get("candidate_mask")
            site_logits = outputs.get("site_logits")
            if candidate_mask is None or site_logits is None:
                return outputs
            mask = candidate_mask.to(device=site_logits.device, dtype=site_logits.dtype).view_as(site_logits)
            if mask.numel() != site_logits.numel():
                return outputs
            mask_mode = str(getattr(self.config, "candidate_mask_mode", "hard")).strip().lower()
            if mask_mode == "off":
                masked_logits = site_logits
            elif mask_mode == "soft":
                bias = float(getattr(self.config, "candidate_mask_logit_bias", 2.0))
                masked_logits = site_logits - ((1.0 - mask) * bias)
            else:
                masked_logits = torch.where(mask > 0.5, site_logits, torch.full_like(site_logits, -20.0))
            result = dict(outputs)
            result.setdefault("site_logits_base", site_logits)
            result["site_logits"] = masked_logits
            result["reranked_site_logits"] = masked_logits
            result["site_scores"] = torch.sigmoid(masked_logits)
            diagnostics = dict(result.get("diagnostics") or {})
            nexus_stats = dict(diagnostics.get("nexus_bridge") or {})
            nexus_stats["candidate_fraction"] = float(mask.detach().mean().item())
            diagnostics["nexus_bridge"] = nexus_stats
            result["diagnostics"] = diagnostics
            return result

        def _apply_local_chemistry(self, outputs: Dict[str, object], batch: Dict[str, object]) -> Dict[str, object]:
            if self.local_chem_proj is None or self.local_chem_gate is None:
                return outputs
            atom_features = outputs.get("atom_features")
            batch_index = batch.get("batch")
            local_chem = batch.get("local_chem_features")
            anomaly_features = batch.get("local_anomaly_features")
            anomaly_score = batch.get("local_anomaly_score_normalized")
            if atom_features is None or batch_index is None or local_chem is None:
                return outputs
            device = atom_features.device
            dtype = atom_features.dtype
            chem_atom = local_chem.to(device=device, dtype=dtype)
            num_molecules = int(outputs["cyp_logits"].size(0))
            mol_anom = self._optional_feature(anomaly_features, num_molecules, 7, device=device, dtype=dtype)
            mol_score = self._optional_feature(anomaly_score, num_molecules, 1, device=device, dtype=dtype)
            z = torch.cat([chem_atom, mol_anom[batch_index], mol_score[batch_index]], dim=-1)
            residual = torch.tanh(self.local_chem_proj(z))
            gate = self.local_chem_gate(torch.cat([atom_features, z], dim=-1))
            scale = torch.sigmoid(self.local_chem_scale_logit).to(device=device, dtype=dtype)
            chem_logit_scale = torch.sigmoid(self.local_chem_logit_scale_logit).to(device=device, dtype=dtype)
            fused_atom_features = atom_features + (scale * gate * residual)
            base_impl = self._base_impl()
            prior_payload = base_impl.manual_priors(
                batch,
                num_atoms=int(atom_features.size(0)),
                num_molecules=num_molecules,
                device=device,
            )
            site_logits, _site_residual = base_impl.site_head(
                fused_atom_features,
                prior_logits=prior_payload.get("atom_prior_logits") if getattr(self.base_lnn.config, "use_manual_engine_priors", False) else None,
                prior_features=prior_payload.get("atom_prior_embedding") if getattr(self.base_lnn.config, "use_manual_engine_priors", False) else None,
            )
            site_logits = base_impl._apply_bde_prior(site_logits, (batch.get("physics_features") or {}).get("bde_values"))
            chem_logit_residual = self.local_chem_logit_head(z) if self.local_chem_logit_head is not None else torch.zeros_like(site_logits)
            site_logits = site_logits + (chem_logit_scale * gate * chem_logit_residual)
            source_site_logits = {}
            if hasattr(base_impl, "_compute_source_site_logits"):
                source_site_logits = base_impl._compute_source_site_logits(fused_atom_features, prior_payload) or {}
            if source_site_logits and hasattr(base_impl, "_blend_source_site_logits"):
                site_logits = base_impl._blend_source_site_logits(
                    site_logits,
                    source_site_logits,
                    batch.get("graph_metadata"),
                    batch.get("batch"),
                )
            result = dict(outputs)
            result.setdefault("site_logits_base", outputs.get("site_logits"))
            result["site_logits"] = site_logits.clamp(-20.0, 20.0)
            result["site_scores"] = torch.sigmoid(result["site_logits"])
            result["atom_features"] = fused_atom_features
            diagnostics = dict(result.get("diagnostics") or {})
            diagnostics["local_chemistry"] = {
                "chem_scale": float(scale.detach().item()),
                "chem_logit_scale": float(chem_logit_scale.detach().item()),
                "chem_gate_mean": float(gate.detach().mean().item()),
                "chem_residual_norm_mean": float(residual.detach().norm(dim=-1).mean().item()) if residual.numel() else 0.0,
                "chem_logit_mean": float(chem_logit_residual.detach().mean().item()) if chem_logit_residual.numel() else 0.0,
                "chem_logit_std": float(chem_logit_residual.detach().std().item()) if chem_logit_residual.numel() > 1 else 0.0,
                "chem_feature_mean": float(chem_atom.detach().mean().item()) if chem_atom.numel() else 0.0,
                "anomaly_score_mean": float(mol_score.detach().mean().item()) if mol_score.numel() else 0.0,
            }
            result["diagnostics"] = diagnostics
            if source_site_logits:
                result["source_site_logits"] = source_site_logits
            return result

        def _apply_phase5_proposer(self, outputs: Dict[str, object], batch: Dict[str, object]) -> Dict[str, object]:
            if self.phase5_proposer_proj is None or self.phase5_proposer_gate is None:
                return outputs
            atom_features = outputs.get("atom_features")
            phase5_atom = batch.get("phase5_atom_features")
            if atom_features is None or phase5_atom is None:
                return outputs
            device = atom_features.device
            dtype = atom_features.dtype
            z_full = phase5_atom.to(device=device, dtype=dtype)
            pieces = []
            cursor = 0
            if bool(getattr(self.config, "use_phase5_boundary_field", False)):
                pieces.append(z_full[:, cursor : cursor + 8])
            cursor += 8
            if bool(getattr(self.config, "use_phase5_accessibility", False)):
                pieces.append(z_full[:, cursor : cursor + 3])
            cursor += 3
            if bool(getattr(self.config, "use_phase5_cyp_profile", False)):
                pieces.append(z_full[:, cursor : cursor + 8])
            if not pieces:
                return outputs
            z = torch.cat(pieces, dim=-1)
            z = self._match_last_dim(z, int(self.phase5_proposer_proj[0].in_features))
            residual = torch.tanh(self.phase5_proposer_proj(z))
            gate_in = torch.cat([atom_features, z], dim=-1)
            gate_in = self._match_last_dim(gate_in, int(self.phase5_proposer_gate[0].in_features))
            gate = self.phase5_proposer_gate(gate_in)
            scale = torch.sigmoid(self.phase5_proposer_scale_logit).to(device=device, dtype=dtype)
            logit_scale = torch.sigmoid(self.phase5_proposer_logit_scale_logit).to(device=device, dtype=dtype)
            fused_atom_features = atom_features + (scale * gate * residual)
            base_impl = self._base_impl()
            num_molecules = int(outputs["cyp_logits"].size(0))
            prior_payload = base_impl.manual_priors(
                batch,
                num_atoms=int(atom_features.size(0)),
                num_molecules=num_molecules,
                device=device,
            )
            site_logits, _site_residual = base_impl.site_head(
                fused_atom_features,
                prior_logits=prior_payload.get("atom_prior_logits") if getattr(self.base_lnn.config, "use_manual_engine_priors", False) else None,
                prior_features=prior_payload.get("atom_prior_embedding") if getattr(self.base_lnn.config, "use_manual_engine_priors", False) else None,
            )
            site_logits = base_impl._apply_bde_prior(site_logits, (batch.get("physics_features") or {}).get("bde_values"))
            logit_residual = self.phase5_proposer_logit_head(z) if self.phase5_proposer_logit_head is not None else torch.zeros_like(site_logits)
            site_logits = site_logits + (logit_scale * gate * logit_residual)
            source_site_logits = {}
            if hasattr(base_impl, "_compute_source_site_logits"):
                source_site_logits = base_impl._compute_source_site_logits(fused_atom_features, prior_payload) or {}
            if source_site_logits and hasattr(base_impl, "_blend_source_site_logits"):
                site_logits = base_impl._blend_source_site_logits(
                    site_logits,
                    source_site_logits,
                    batch.get("graph_metadata"),
                    batch.get("batch"),
                )
            result = dict(outputs)
            result.setdefault("site_logits_phase5_base", outputs.get("site_logits"))
            result["site_logits"] = site_logits.clamp(-20.0, 20.0)
            result["site_scores"] = torch.sigmoid(result["site_logits"])
            result["atom_features"] = fused_atom_features
            diagnostics = dict(result.get("diagnostics") or {})
            diagnostics["phase5_proposer"] = {
                "p5_scale": float(scale.detach().item()),
                "p5_logit_scale": float(logit_scale.detach().item()),
                "p5_gate_mean": float(gate.detach().mean().item()) if gate.numel() else 0.0,
                "p5_residual_norm_mean": float(residual.detach().norm(dim=-1).mean().item()) if residual.numel() else 0.0,
                "p5_feature_mean": float(z.detach().mean().item()) if z.numel() else 0.0,
                "p5_boundary_mean": float(z_full[:, 0:1].detach().mean().item()) if z_full.numel() else 0.0,
                "p5_access_mean": float(z_full[:, 9:10].detach().mean().item()) if z_full.size(-1) >= 10 else 0.0,
                "p5_heme_distance_mean": float(z_full[:, 4:5].detach().mean().item()) if z_full.size(-1) >= 5 else 0.0,
                "p5_profile_norm_mean": float(z_full[:, -8:].detach().norm(dim=-1).mean().item()) if z_full.size(-1) >= 8 else 0.0,
            }
            result["diagnostics"] = diagnostics
            if source_site_logits:
                result["source_site_logits"] = source_site_logits
            return result

        def _apply_phase2_context(self, outputs: Dict[str, object], batch: Dict[str, object]) -> Dict[str, object]:
            if self.phase2_context_proj is None or self.phase2_context_gate is None:
                return outputs
            atom_features = outputs.get("atom_features")
            batch_index = batch.get("batch")
            local_chem = batch.get("local_chem_features")
            anomaly_score = batch.get("local_anomaly_score_normalized")
            atom_coords = batch.get("atom_coordinates")
            candidate_mask = batch.get("candidate_mask")
            edge_index = batch.get("edge_index")
            if atom_features is None or batch_index is None or local_chem is None:
                return outputs
            device = atom_features.device
            dtype = atom_features.dtype
            rows = int(atom_features.size(0))
            num_molecules = int(outputs["cyp_logits"].size(0))
            chem_atom = local_chem.to(device=device, dtype=dtype)
            mol_score = self._optional_feature(anomaly_score, num_molecules, 1, device=device, dtype=dtype)

            event_outputs = None
            ctx_parts = [chem_atom, mol_score[batch_index]]
            if self.event_context_module is not None:
                event_outputs = self.event_context_module(
                    atom_features=atom_features,
                    edge_index=edge_index,
                    atom_coords=atom_coords,
                    local_chem_features=chem_atom,
                    candidate_mask=candidate_mask,
                )
                ctx_parts.append(event_outputs["features"])
            access_outputs = None
            if self.accessibility_head is not None:
                access_outputs = self.accessibility_head(
                    atom_coords=atom_coords,
                    batch_index=batch_index,
                    local_chem_features=chem_atom,
                    candidate_mask=candidate_mask,
                    event_outputs=event_outputs,
                )
                ctx_parts.append(access_outputs["features"])
            barrier_outputs = None
            if self.barrier_head is not None:
                barrier_outputs = self.barrier_head(
                    local_chem_features=chem_atom,
                    accessibility_outputs=access_outputs,
                    event_outputs=event_outputs,
                )
                ctx_parts.append(barrier_outputs["features"])

            z_ctx = torch.cat(ctx_parts, dim=-1)
            residual = torch.tanh(self.phase2_context_proj(z_ctx))
            gate = self.phase2_context_gate(torch.cat([atom_features, z_ctx], dim=-1))
            scale = torch.sigmoid(self.phase2_context_scale_logit).to(device=device, dtype=dtype)
            logit_scale = torch.sigmoid(self.phase2_context_logit_scale_logit).to(device=device, dtype=dtype)
            fused_atom_features = atom_features + (scale * gate * residual)

            base_impl = self._base_impl()
            prior_payload = base_impl.manual_priors(
                batch,
                num_atoms=rows,
                num_molecules=num_molecules,
                device=device,
            )
            site_logits, _site_residual = base_impl.site_head(
                fused_atom_features,
                prior_logits=prior_payload.get("atom_prior_logits") if getattr(self.base_lnn.config, "use_manual_engine_priors", False) else None,
                prior_features=prior_payload.get("atom_prior_embedding") if getattr(self.base_lnn.config, "use_manual_engine_priors", False) else None,
            )
            site_logits = base_impl._apply_bde_prior(site_logits, (batch.get("physics_features") or {}).get("bde_values"))
            logit_residual = self.phase2_context_logit_head(z_ctx) if self.phase2_context_logit_head is not None else torch.zeros_like(site_logits)
            site_logits = site_logits + (logit_scale * gate * logit_residual)
            source_site_logits = {}
            if hasattr(base_impl, "_compute_source_site_logits"):
                source_site_logits = base_impl._compute_source_site_logits(fused_atom_features, prior_payload) or {}
            if source_site_logits and hasattr(base_impl, "_blend_source_site_logits"):
                site_logits = base_impl._blend_source_site_logits(
                    site_logits,
                    source_site_logits,
                    batch.get("graph_metadata"),
                    batch.get("batch"),
                )

            result = dict(outputs)
            result.setdefault("site_logits_phase2_base", outputs.get("site_logits"))
            result["site_logits"] = site_logits.clamp(-20.0, 20.0)
            result["site_scores"] = torch.sigmoid(result["site_logits"])
            result["atom_features"] = fused_atom_features
            result["phase2_context_outputs"] = {
                "event_strain": None if event_outputs is None else event_outputs["strain"],
                "event_active_neighbor_count": None if event_outputs is None else event_outputs["active_neighbor_count"],
                "event_depth": None if event_outputs is None else event_outputs["event_depth"],
                "access_score": None if access_outputs is None else access_outputs["score"],
                "access_cost": None if access_outputs is None else access_outputs["cost"],
                "barrier_score": None if barrier_outputs is None else barrier_outputs["score"],
            }
            diagnostics = dict(result.get("diagnostics") or {})
            diagnostics["phase2_context"] = {
                "ctx_scale": float(scale.detach().item()),
                "ctx_logit_scale": float(logit_scale.detach().item()),
                "ctx_gate_mean": float(gate.detach().mean().item()) if gate.numel() else 0.0,
                "ctx_residual_norm_mean": float(residual.detach().norm(dim=-1).mean().item()) if residual.numel() else 0.0,
                "event_strain_mean": float(event_outputs["strain"].detach().mean().item()) if event_outputs is not None else 0.0,
                "access_score_mean": float(access_outputs["score"].detach().mean().item()) if access_outputs is not None else 0.0,
                "barrier_score_mean": float(barrier_outputs["score"].detach().mean().item()) if barrier_outputs is not None else 0.0,
            }
            result["diagnostics"] = diagnostics
            if source_site_logits:
                result["source_site_logits"] = source_site_logits
            return result

        def _topk_reranker_features(
            self,
            outputs: Dict[str, object],
            batch: Dict[str, object],
        ) -> torch.Tensor:
            site_logits = outputs["site_logits"]
            rows = int(site_logits.size(0))
            device = site_logits.device
            dtype = site_logits.dtype
            batch_index = batch.get("batch")
            num_molecules = int(outputs["cyp_logits"].size(0)) if outputs.get("cyp_logits") is not None else 0
            anomaly_score = batch.get("local_anomaly_score_normalized")
            anomaly_atom = (
                self._optional_feature(anomaly_score, num_molecules, 1, device=device, dtype=dtype)[batch_index]
                if batch_index is not None and num_molecules > 0
                else torch.zeros(rows, 1, device=device, dtype=dtype)
            )
            phase2 = outputs.get("phase2_context_outputs") or {}
            bde_values = ((batch.get("physics_features") or {}).get("bde_values") if isinstance(batch.get("physics_features"), dict) else None)
            pieces = [
                self._optional_feature(batch.get("local_chem_features"), rows, 11, device=device, dtype=dtype),
                anomaly_atom,
                self._optional_feature(phase2.get("event_strain"), rows, 1, device=device, dtype=dtype),
                self._optional_feature(phase2.get("access_score"), rows, 1, device=device, dtype=dtype),
                self._optional_feature(phase2.get("barrier_score"), rows, 1, device=device, dtype=dtype),
                self._optional_feature(bde_values, rows, 1, device=device, dtype=dtype),
                self._optional_feature(batch.get("candidate_mask"), rows, 1, device=device, dtype=dtype),
            ]
            return torch.cat(pieces, dim=-1)

        def _apply_topk_reranker(
            self,
            outputs: Dict[str, object],
            batch: Dict[str, object],
        ) -> Dict[str, object]:
            if self.topk_reranker is None:
                return outputs
            site_logits = outputs.get("site_logits")
            atom_features = outputs.get("atom_features")
            batch_index = batch.get("batch")
            if site_logits is None or atom_features is None or batch_index is None:
                return outputs
            if not getattr(site_logits, "numel", lambda: 0)() or not getattr(batch_index, "numel", lambda: 0)():
                return outputs
            proposal_logits = site_logits
            reranked_logits, reranker = self.topk_reranker(
                atom_features=atom_features,
                site_logits=proposal_logits,
                batch_index=batch_index,
                mol_features=outputs.get("mol_features"),
                extra_features=self._topk_reranker_features(outputs, batch),
            )
            reranked_logits = torch.nan_to_num(reranked_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            result = dict(outputs)
            result.setdefault("site_logits_base", proposal_logits)
            result["site_logits_proposal"] = proposal_logits
            result["site_scores_proposal"] = torch.sigmoid(proposal_logits)
            result["site_logits"] = reranked_logits
            result["reranked_site_logits"] = reranked_logits
            result["site_scores"] = torch.sigmoid(reranked_logits)
            result["topk_reranker_outputs"] = reranker
            diagnostics = dict(result.get("diagnostics") or {})
            diagnostics["topk_reranker"] = dict(reranker.get("stats") or {})
            result["diagnostics"] = diagnostics
            return result

        def _site_logits_from_sideinfo(
            self,
            outputs: Dict[str, object],
            bridge: Dict[str, object],
            batch: Dict[str, object],
        ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            atom_features = outputs["atom_features"]
            batch_index = batch["batch"]
            device = atom_features.device
            dtype = atom_features.dtype
            rows = int(atom_features.size(0))
            num_molecules = int(outputs["cyp_logits"].size(0))
            base_cyp = bridge["analogical_cyp_prior"][batch_index].detach()
            wave_preds = bridge["wave_predictions"]
            wave_field = bridge["wave_field"]
            wave_grad_scale = float(getattr(self.config, "nexus_live_wave_vote_grad_scale", 0.02))
            analogical_grad_scale = float(getattr(self.config, "nexus_live_analogical_vote_grad_scale", 0.02))
            wave_gap = wave_preds["predicted_gap"][batch_index].unsqueeze(-1).detach()
            wave_charges = wave_preds["predicted_charges"].unsqueeze(-1).detach()
            wave_fukui = wave_preds["predicted_fukui"].unsqueeze(-1).detach()
            wave_site_bias = self._scaled_live(bridge["wave_site_bias"], enabled=True, grad_scale=wave_grad_scale)
            wave_field_features = self._scaled_live(
                wave_field["atom_field_features"],
                enabled=True,
                grad_scale=wave_grad_scale,
            )
            wave_inputs = torch.cat(
                [
                    wave_charges,
                    wave_fukui,
                    wave_gap,
                    wave_site_bias,
                    bridge["wave_reliability"].detach(),
                    wave_field_features,
                ],
                dim=-1,
            )
            analogical_site_prior = self._scaled_live(
                bridge["analogical_site_prior"],
                enabled=True,
                grad_scale=analogical_grad_scale,
            )
            analogical_site_bias = self._scaled_live(
                bridge["analogical_site_bias"],
                enabled=True,
                grad_scale=analogical_grad_scale,
            )
            continuous_reasoning = self._scaled_live(
                bridge["continuous_reasoning_features"],
                enabled=True,
                grad_scale=analogical_grad_scale,
            )
            analogical_inputs = torch.cat(
                [
                    analogical_site_prior,
                    bridge["analogical_confidence"].detach(),
                    bridge["analogical_gate"].detach(),
                    analogical_site_bias,
                    continuous_reasoning,
                    bridge["precedent_brief"].detach(),
                    base_cyp,
                ],
                dim=-1,
            )
            sideinfo = torch.nan_to_num(torch.cat([wave_inputs, analogical_inputs], dim=-1), nan=0.0, posinf=4.0, neginf=-4.0)
            expected_sideinfo = int(self.nexus_sideinfo_proj[0].in_features)
            sideinfo = self._match_last_dim(sideinfo, expected_sideinfo)
            sideinfo_residual = torch.tanh(self.nexus_sideinfo_proj(sideinfo))
            gate_in = torch.cat([atom_features, sideinfo], dim=-1)
            expected_gate = int(self.nexus_sideinfo_gate[0].in_features)
            gate_in = self._match_last_dim(gate_in, expected_gate)
            sideinfo_gate = self.nexus_sideinfo_gate(gate_in)
            sideinfo_scale = torch.sigmoid(self.nexus_sideinfo_scale_logit).to(device=device, dtype=dtype)
            fused_atom_features = atom_features + sideinfo_scale * sideinfo_gate * sideinfo_residual
            base_impl = self._base_impl()
            prior_payload = base_impl.manual_priors(batch, num_atoms=rows, num_molecules=num_molecules, device=device)
            site_logits, _site_residual = base_impl.site_head(
                fused_atom_features,
                prior_logits=prior_payload.get("atom_prior_logits") if getattr(self.base_lnn.config, "use_manual_engine_priors", False) else None,
                prior_features=prior_payload.get("atom_prior_embedding") if getattr(self.base_lnn.config, "use_manual_engine_priors", False) else None,
            )
            site_logits = base_impl._apply_bde_prior(site_logits, (batch.get("physics_features") or {}).get("bde_values"))
            source_site_logits = {}
            if hasattr(base_impl, "_compute_source_site_logits"):
                source_site_logits = base_impl._compute_source_site_logits(fused_atom_features, prior_payload) or {}
            if source_site_logits and hasattr(base_impl, "_blend_source_site_logits"):
                site_logits = base_impl._blend_source_site_logits(
                    site_logits,
                    source_site_logits,
                    batch.get("graph_metadata"),
                    batch.get("batch"),
                )
            sideinfo_diag = {
                "fused_atom_features": fused_atom_features,
                "sideinfo_inputs": sideinfo,
                "sideinfo_residual": sideinfo_residual,
                "sideinfo_gate": sideinfo_gate,
                "sideinfo_scale": sideinfo_scale.view(1, 1),
            }
            if source_site_logits:
                sideinfo_diag["source_site_logits"] = source_site_logits
            return site_logits.clamp(-20.0, 20.0), sideinfo_diag

        def _site_logits_from_arbiter(
            self,
            outputs: Dict[str, object],
            bridge: Dict[str, object],
            batch: Dict[str, object],
        ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            atom_features = outputs["atom_features"]
            batch_index = batch["batch"]
            rows = int(atom_features.size(0))
            device = atom_features.device
            dtype = atom_features.dtype
            num_cyp = int(getattr(self.config, "num_cyp_classes", 9))
            steric_dim = int(getattr(self.config, "steric_feature_dim", 8))
            xtb_dim = FULL_XTB_FEATURE_DIM
            topology_dim = int(getattr(self.config, "nexus_topology_feature_dim", 5))

            graph_embeddings = bridge["graph_embeddings"][batch_index]
            base_cyp_context = torch.softmax(outputs["cyp_logits"], dim=-1)[batch_index]
            analogical_cyp_context = bridge["analogical_cyp_prior"][batch_index]
            confidence = bridge["analogical_confidence"]
            wave_preds = bridge["wave_predictions"]
            wave_field = bridge["wave_field"]
            wave_reliability = bridge.get("wave_reliability")
            if wave_reliability is None:
                wave_reliability = atom_features.new_ones((rows, 1))
            else:
                wave_reliability = wave_reliability.detach()
            wave_context_gate = torch.sigmoid((wave_reliability - 0.20) / 0.08)
            wave_vote_gate = torch.sigmoid((wave_reliability - 0.35) / 0.08)
            precedent_brief = bridge.get("precedent_brief")
            if precedent_brief is None:
                precedent_brief = atom_features.new_zeros((rows, AuditedEpisodeLogbook.brief_dim))
            analogical_gate = bridge.get("analogical_gate")
            if analogical_gate is None:
                analogical_gate = bridge["analogical_confidence"]
            analogical_gate = analogical_gate.detach().clamp(0.0, 1.0)
            analogical_context_gate = torch.sigmoid((analogical_gate - 0.12) / 0.10)
            analogical_vote_gate = torch.sigmoid((analogical_gate - 0.18) / 0.10)
            vote_scale = float(getattr(self.config, "nexus_vote_logit_scale", 2.0))
            live_wave_vote_inputs = bool(getattr(self.config, "nexus_live_wave_vote_inputs", True))
            live_analogical_vote_inputs = bool(getattr(self.config, "nexus_live_analogical_vote_inputs", True))
            wave_vote_grad_scale = float(getattr(self.config, "nexus_live_wave_vote_grad_scale", 0.05))
            analogical_vote_grad_scale = float(getattr(self.config, "nexus_live_analogical_vote_grad_scale", 0.05))
            atom_features_b = torch.tanh(atom_features)
            # Keep the LNN encoder live, but stop the main site loss from backpropagating
            # through the heavier wave/analogical sidecar. Those modules still learn via
            # their own auxiliary losses.
            multivectors_b = torch.tanh(bridge["atom_multivectors"].detach())
            graph_embeddings_b = torch.tanh(graph_embeddings.detach())
            wave_scalar = torch.cat(
                [
                    wave_preds["predicted_charges"].unsqueeze(-1),
                    wave_preds["predicted_fukui"].unsqueeze(-1),
                    wave_preds["predicted_gap"][batch_index].unsqueeze(-1),
                ],
                dim=-1,
            )
            wave_scalar_b = wave_context_gate * torch.tanh(wave_scalar.detach())
            steric = self._optional_feature(batch.get("atom_3d_features"), rows, steric_dim, device=device, dtype=dtype)
            xtb = self._optional_feature(batch.get("xtb_atom_features"), rows, xtb_dim, device=device, dtype=dtype)
            topology = self._optional_feature(batch.get("topology_atom_features"), rows, topology_dim, device=device, dtype=dtype)
            steric_b = torch.tanh(steric)
            xtb_b = torch.tanh(xtb)
            topology_b = topology  # features are already in [0, 1]; no tanh needed
            wave_field_b = wave_context_gate * torch.tanh(wave_field["atom_field_features"].detach())
            wave_scalar_vote = torch.tanh(
                self._safe_vote_tensor(
                    self._scaled_live(
                        wave_scalar,
                        enabled=live_wave_vote_inputs,
                        grad_scale=wave_vote_grad_scale,
                    )
                )
            )
            wave_field_vote = torch.tanh(
                self._safe_vote_tensor(
                    self._scaled_live(
                        wave_field["atom_field_features"],
                        enabled=live_wave_vote_inputs,
                        grad_scale=wave_vote_grad_scale,
                    )
                )
            )
            lnn_vote_raw = self.lnn_vote_head(
                torch.cat(
                    [
                        atom_features_b,
                        torch.tanh(outputs["site_logits"]),
                    ],
                    dim=-1,
                )
            )
            lnn_vote = vote_scale * torch.tanh(lnn_vote_raw)
            lnn_conf = torch.sigmoid(self.lnn_conf_head(atom_features_b))
            wave_site_bias_b = wave_context_gate * torch.tanh(bridge["wave_site_bias"].detach())
            wave_site_bias_vote = torch.tanh(
                self._safe_vote_tensor(
                    self._scaled_live(
                        bridge["wave_site_bias"],
                        enabled=live_wave_vote_inputs,
                        grad_scale=wave_vote_grad_scale,
                    )
                )
            )
            wave_vote_raw = self.wave_vote_head(
                torch.cat(
                    [
                        wave_field_vote,
                        wave_scalar_vote,
                        wave_site_bias_vote,
                    ],
                    dim=-1,
                )
            )
            wave_vote = wave_vote_gate * (vote_scale * torch.tanh(wave_vote_raw))
            wave_conf = wave_vote_gate * torch.sigmoid(
                self.wave_conf_head(
                    torch.cat(
                        [
                            wave_field_vote,
                            wave_scalar_vote,
                            wave_site_bias_vote,
                        ],
                        dim=-1,
                    )
                )
            )
            analogical_site_bias_b = analogical_context_gate * torch.tanh(bridge["analogical_site_bias"].detach())
            continuous_reasoning_b = analogical_context_gate * torch.tanh(bridge["continuous_reasoning_features"].detach())
            analogical_site_prior = analogical_context_gate * bridge["analogical_site_prior"].detach()
            confidence_b = analogical_context_gate * confidence.detach()
            analogical_cyp_context_b = analogical_context_gate * analogical_cyp_context.detach()
            analogical_site_prior_vote = self._safe_vote_tensor(
                self._scaled_live(
                    bridge["analogical_site_prior"],
                    enabled=live_analogical_vote_inputs,
                    grad_scale=analogical_vote_grad_scale,
                )
            )
            confidence_vote = self._safe_vote_tensor(
                self._scaled_live(
                    confidence,
                    enabled=live_analogical_vote_inputs,
                    grad_scale=analogical_vote_grad_scale,
                )
            )
            analogical_site_bias_vote = torch.tanh(
                self._safe_vote_tensor(
                    self._scaled_live(
                        bridge["analogical_site_bias"],
                        enabled=live_analogical_vote_inputs,
                        grad_scale=analogical_vote_grad_scale,
                    )
                )
            )
            continuous_reasoning_vote = torch.tanh(
                self._safe_vote_tensor(
                    self._scaled_live(
                        bridge["continuous_reasoning_features"],
                        enabled=live_analogical_vote_inputs,
                        grad_scale=analogical_vote_grad_scale,
                    )
                )
            )
            precedent_brief_vote = analogical_context_gate * self._safe_vote_tensor(
                self._scaled_live(
                    precedent_brief,
                    enabled=live_analogical_vote_inputs,
                    grad_scale=analogical_vote_grad_scale,
                )
            )
            analogical_vote_raw = self.analogical_vote_head(
                torch.cat(
                    [
                        analogical_site_prior_vote,
                        confidence_vote,
                        analogical_site_bias_vote,
                        continuous_reasoning_vote,
                        precedent_brief_vote,
                    ],
                    dim=-1,
                )
            )
            analogical_vote = analogical_vote_gate * (vote_scale * torch.tanh(analogical_vote_raw))
            analogical_conf = analogical_vote_gate * torch.sigmoid(
                self.analogical_conf_head(
                    torch.cat(
                        [
                            analogical_site_prior_vote,
                            confidence_vote,
                            analogical_site_bias_vote,
                            continuous_reasoning_vote,
                            precedent_brief_vote,
                        ],
                        dim=-1,
                    )
                )
            )
            board_context = torch.cat(
                [
                    atom_features_b,
                    multivectors_b,
                    graph_embeddings_b,
                    base_cyp_context,
                    wave_scalar_b,
                    analogical_site_prior,
                    confidence_b,
                    analogical_cyp_context_b,
                    steric_b,
                    xtb_b,
                    topology_b,
                    wave_field_b,
                    analogical_context_gate * precedent_brief.detach(),
                    wave_site_bias_b,
                    analogical_site_bias_b,
                    continuous_reasoning_b,
                ],
                dim=-1,
            )
            council_stream_meta = torch.cat(
                [
                    lnn_vote,
                    lnn_conf,
                    wave_vote,
                    wave_conf,
                    analogical_vote,
                    analogical_conf,
                ],
                dim=-1,
            )
            board_logits = self.council_board_head(
                torch.cat(
                    [
                        board_context,
                        council_stream_meta,
                    ],
                    dim=-1,
                )
            )
            board_weights = torch.softmax(board_logits, dim=-1)
            council_logit = (
                board_weights[:, 0:1] * lnn_vote
                + board_weights[:, 1:2] * wave_vote
                + board_weights[:, 2:3] * analogical_vote
            )
            # Per-molecule zero-centering of council_logit: same fix as arbiter.
            # Without this the council learns a global negative bias (mean ≈ -0.86)
            # that suppresses all atom logits uniformly instead of re-ranking.
            if batch_index.numel() > 0:
                num_mol_cncl = int(batch_index.max().item()) + 1
                cncl_sums = torch.zeros(num_mol_cncl, 1, device=device, dtype=dtype)
                cncl_cnts = torch.zeros(num_mol_cncl, 1, device=device, dtype=dtype)
                idx_exp_c = batch_index.unsqueeze(-1)
                cncl_sums.scatter_add_(0, idx_exp_c, council_logit)
                cncl_cnts.scatter_add_(0, idx_exp_c, torch.ones_like(council_logit))
                cncl_means = cncl_sums / cncl_cnts.clamp(min=1.0)
                council_logit = council_logit - cncl_means[batch_index]
            base_site_logits = outputs["site_logits"]
            # The council is the only active correction path now. The extra
            # arbiter residual was redundant and made the fusion stack harder to
            # reason about without showing strong standalone value.
            arbiter_residual = torch.zeros_like(base_site_logits)
            site_logits = base_site_logits + council_logit
            site_logits = torch.nan_to_num(site_logits, nan=0.0, posinf=20.0, neginf=-20.0)
            council = {
                "base_site_logits": base_site_logits,
                "lnn_vote": lnn_vote,
                "lnn_conf": lnn_conf,
                "wave_vote": wave_vote,
                "wave_conf": wave_conf,
                "analogical_vote": analogical_vote,
                "analogical_conf": analogical_conf,
                "council_logit": council_logit,
                "arbiter_residual": arbiter_residual,
                "board_weights": board_weights,
            }
            return site_logits.clamp(-20.0, 20.0), council

        def _apply_nexus_bridge(self, outputs: Dict[str, object], batch: Dict[str, object]) -> Dict[str, object]:
            if self.nexus_bridge is None:
                return outputs
            atom_features = outputs.get("atom_features")
            if atom_features is None:
                return outputs
            bridge = self.nexus_bridge(
                atom_features=atom_features,
                batch_index=batch["batch"],
                site_logits=outputs["site_logits"],
                cyp_logits=outputs["cyp_logits"],
                atom_3d_features=batch.get("atom_3d_features"),
                xtb_atom_features=batch.get("xtb_atom_features"),
                topology_atom_features=batch.get("topology_atom_features"),
                xtb_atom_valid_mask=batch.get("xtb_atom_valid_mask"),
                xtb_mol_valid=batch.get("xtb_mol_valid"),
                site_labels=batch.get("site_labels"),
                site_supervision_mask=batch.get("site_supervision_mask"),
                cyp_labels=batch.get("cyp_labels"),
                cyp_supervision_mask=batch.get("cyp_supervision_mask"),
                graph_molecule_keys=batch.get("graph_molecule_keys"),
            )
            result = dict(outputs)
            result["site_logits_base"] = outputs["site_logits"]
            result["cyp_logits_base"] = outputs["cyp_logits"]
            council = None
            sideinfo = None
            if self.nexus_sideinfo_uses_bridge:
                site_logits, sideinfo = self._site_logits_from_sideinfo(outputs, bridge, batch)
                cyp_logits = outputs["cyp_logits"]
                site_mode = "nexus_sideinfo"
            elif self.site_arbiter_uses_bridge:
                site_logits, council = self._site_logits_from_arbiter(outputs, bridge, batch)
                ana_cyp_weight = torch.sigmoid(self.ana_cyp_weight_logit)
                cyp_logits = outputs["cyp_logits"] + ana_cyp_weight * bridge["analogical_cyp_bias"].detach()
                site_mode = "nexus_council"
            else:
                site_logits = outputs["site_logits"]
                cyp_logits = outputs["cyp_logits"]
                site_mode = "base"
            cyp_logits = torch.nan_to_num(cyp_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            diagnostics = dict(result.get("diagnostics") or {})
            diagnostics["nexus_bridge"] = {
                **bridge["metrics"],
                "analogical_cyp_weight": float(torch.sigmoid(self.ana_cyp_weight_logit).detach().item()) if council is not None else 0.0,
                "site_mode": site_mode,
                "wave_vote_gate_mean": float(torch.sigmoid((bridge["wave_reliability"].detach() - 0.35) / 0.08).mean().item()) if bridge.get("wave_reliability") is not None else 0.0,
                "analogical_gate_mean": float(bridge["analogical_gate"].detach().mean().item()) if bridge.get("analogical_gate") is not None else 0.0,
                "site_logits_base_mean": float(outputs["site_logits"].detach().mean().item()),
                "site_logits_council_mean": float(council["council_logit"].detach().mean().item()) if council else 0.0,
                "site_logits_arbiter_residual_mean": float(council["arbiter_residual"].detach().mean().item()) if council else 0.0,
                "site_logits_final_mean": float(site_logits.detach().mean().item()),
                "sideinfo_gate_mean": float(sideinfo["sideinfo_gate"].detach().mean().item()) if sideinfo else 0.0,
                "sideinfo_residual_norm_mean": float(sideinfo["sideinfo_residual"].detach().norm(dim=-1).mean().item()) if sideinfo else 0.0,
                "sideinfo_scale": float(sideinfo["sideinfo_scale"].detach().view(-1)[0].item()) if sideinfo else 0.0,
            }
            result.update(
                {
                    "atom_features_base": outputs.get("atom_features"),
                    "atom_features": sideinfo["fused_atom_features"] if sideinfo else outputs.get("atom_features"),
                    "site_logits": site_logits,
                    "reranked_site_logits": site_logits,
                    "site_scores": torch.sigmoid(site_logits),
                    "cyp_logits": cyp_logits,
                    "nexus_bridge_outputs": bridge,
                    "nexus_bridge_losses": bridge["losses"],
                    "atom_multivectors": bridge["atom_multivectors"],
                    "site_vote_heads": council,
                    "nexus_sideinfo": sideinfo,
                    "source_site_logits": sideinfo.get("source_site_logits") if sideinfo else outputs.get("source_site_logits"),
                    "diagnostics": diagnostics,
                }
            )
            return result

        @torch.no_grad()
        def load_nexus_precedent_logbook(self, path: str, *, cyp_names: Optional[list[str]] = None) -> Dict[str, float]:
            if self.nexus_bridge is None:
                return {"cases": 0.0, "episodes": 0.0}
            return self.nexus_bridge.load_precedent_logbook(path, cyp_names=cyp_names)

        @torch.no_grad()
        def rebuild_nexus_memory(self, loader, *, device=None, max_batches: int | None = None) -> Dict[str, float]:
            if self.nexus_bridge is None:
                return {"used": 0.0, "memory_size": 0.0}
            from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs, move_to_device

            self.nexus_bridge.clear_memory()
            was_training = self.training
            self.eval()
            total_batches = 0
            total_used = 0.0
            try:
                for batch_idx, raw_batch in enumerate(loader):
                    if max_batches is not None and batch_idx >= int(max_batches):
                        break
                    if raw_batch is None:
                        continue
                    batch = raw_batch if isinstance(raw_batch, dict) else collate_molecule_graphs(raw_batch)
                    batch = move_to_device(batch, device or next(self.parameters()).device)
                    outputs = self.base_lnn(batch)
                    stats = self.nexus_bridge.ingest_batch(
                        atom_features=outputs["atom_features"],
                        batch_index=batch["batch"],
                        site_logits=outputs.get("site_logits"),
                        cyp_logits=outputs["cyp_logits"],
                        atom_3d_features=batch.get("atom_3d_features"),
                        xtb_atom_features=batch.get("xtb_atom_features"),
                        topology_atom_features=batch.get("topology_atom_features"),
                        site_labels=batch.get("site_labels"),
                        site_supervision_mask=batch.get("site_supervision_mask"),
                        cyp_labels=batch.get("cyp_labels"),
                        cyp_supervision_mask=batch.get("cyp_supervision_mask"),
                        graph_molecule_keys=batch.get("graph_molecule_keys"),
                    )
                    total_batches += 1
                    total_used += float(stats.get("used", 0.0))
            finally:
                self.train(was_training)
            if bool(getattr(self.config, "nexus_memory_frozen", False)):
                self.nexus_bridge.set_memory_frozen(True)
            return {
                "used": total_used,
                "batches": float(total_batches),
                "memory_size": float(self.nexus_bridge.memory.size()),
            }

        def forward(
            self,
            batch: Dict[str, object],
            route_prior: Optional[torch.Tensor] = None,
        ) -> Dict[str, object]:
            outputs = dict(self.base_lnn(batch))
            outputs = self._apply_fixed_cyp_context(outputs)
            outputs = self._apply_phase5_proposer(outputs, batch)
            outputs = self._apply_local_chemistry(outputs, batch)
            outputs = self._apply_phase2_context(outputs, batch)
            if self.domain_adv_head is not None:
                mol_features = outputs.get("mol_features")
                if mol_features is not None and getattr(mol_features, "numel", lambda: 0)():
                    rev = self._gradient_reverse(
                        mol_features,
                        scale=float(getattr(self.config, "domain_adv_grad_scale", 0.1)),
                    )
                    outputs["domain_logits"] = self.domain_adv_head(rev)
            outputs = self._apply_nexus_bridge(outputs, batch)
            outputs = self._apply_topk_reranker(outputs, batch)
            outputs = self._apply_candidate_mask(outputs, batch)
            prior = route_prior
            if prior is None:
                prior = batch.get("manual_engine_route_prior")
            if prior is None:
                outputs["hybrid_manual_prior"] = {
                    "prior_weight": float(torch.sigmoid(self.prior_weight_logit).detach().item()),
                    "used": 0.0,
                }
                return outputs

            prior = prior.to(device=outputs["cyp_logits"].device, dtype=outputs["cyp_logits"].dtype)
            weight = torch.sigmoid(self.prior_weight_logit)
            outputs["cyp_logits"] = combine_lnn_with_prior(
                outputs["cyp_logits"],
                prior,
                prior_weight=float(weight.detach().item()),
            )
            outputs["hybrid_manual_prior"] = {
                "prior_weight": float(weight.detach().item()),
                "used": 1.0,
            }
            return outputs

        @torch.no_grad()
        def refresh_nexus_memory(self, loader, *, device=None) -> int:
            """Clear and rebuild the analogical memory using the current encoder.

            Called once per epoch (after gradient updates) so every stored key
            is encoded by the same up-to-date network rather than a mix of
            mid-epoch snapshots.  Returns the number of atoms ingested.
            """
            if self.nexus_bridge is None:
                return 0
            from enzyme_software.liquid_nn_v2.training.utils import move_to_device
            self.nexus_bridge.clear_memory()
            was_training = self.training
            self.eval()
            ingested = 0
            try:
                for raw_batch in loader:
                    batch = move_to_device(raw_batch, device) if device is not None else raw_batch
                    outputs = self.base_lnn(batch)
                    cyp_logits = outputs.get("cyp_logits")
                    if cyp_logits is None or not cyp_logits.numel():
                        continue
                    atom_features = outputs.get("atom_features")
                    if atom_features is None:
                        continue
                    stats = self.nexus_bridge.ingest_batch(
                        atom_features=atom_features,
                        batch_index=batch["batch"],
                        site_logits=outputs.get("site_logits"),
                        cyp_logits=cyp_logits,
                        atom_3d_features=batch.get("atom_3d_features"),
                        xtb_atom_features=batch.get("xtb_atom_features"),
                        topology_atom_features=batch.get("topology_atom_features"),
                        site_labels=batch.get("site_labels"),
                        site_supervision_mask=batch.get("site_supervision_mask"),
                        graph_molecule_keys=batch.get("graph_molecule_keys"),
                    )
                    ingested += int(stats.get("added_atoms", 0))
            finally:
                if was_training:
                    self.train()
            return ingested

else:  # pragma: no cover
    class HybridLNNModel:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
