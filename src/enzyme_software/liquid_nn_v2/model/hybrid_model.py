from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.features.xtb_features import FULL_XTB_FEATURE_DIM
from enzyme_software.liquid_nn_v2.features.route_prior import combine_lnn_with_prior
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
            wave_gap = wave_preds["predicted_gap"][batch_index].unsqueeze(-1).detach()
            wave_inputs = torch.cat(
                [
                    wave_preds["predicted_charges"].unsqueeze(-1).detach(),
                    wave_preds["predicted_fukui"].unsqueeze(-1).detach(),
                    wave_gap,
                    bridge["wave_site_bias"].detach(),
                    bridge["wave_reliability"].detach(),
                    wave_field["atom_field_features"].detach(),
                ],
                dim=-1,
            )
            analogical_inputs = torch.cat(
                [
                    bridge["analogical_site_prior"].detach(),
                    bridge["analogical_confidence"].detach(),
                    bridge["analogical_gate"].detach(),
                    bridge["analogical_site_bias"].detach(),
                    bridge["continuous_reasoning_features"].detach(),
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
            sideinfo_diag = {
                "fused_atom_features": fused_atom_features,
                "sideinfo_inputs": sideinfo,
                "sideinfo_residual": sideinfo_residual,
                "sideinfo_gate": sideinfo_gate,
                "sideinfo_scale": sideinfo_scale.view(1, 1),
            }
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
            if self.domain_adv_head is not None:
                mol_features = outputs.get("mol_features")
                if mol_features is not None and getattr(mol_features, "numel", lambda: 0)():
                    rev = self._gradient_reverse(
                        mol_features,
                        scale=float(getattr(self.config, "domain_adv_grad_scale", 0.1)),
                    )
                    outputs["domain_logits"] = self.domain_adv_head(rev)
            outputs = self._apply_nexus_bridge(outputs, batch)
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
