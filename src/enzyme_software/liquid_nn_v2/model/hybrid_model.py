from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.features.xtb_features import FULL_XTB_FEATURE_DIM
from enzyme_software.liquid_nn_v2.features.route_prior import combine_lnn_with_prior
from enzyme_software.liquid_nn_v2.model.nexus_bridge import NexusHybridBridge
from enzyme_software.liquid_nn_v2.model.precedent_logbook import AuditedEpisodeLogbook


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
            precedent_brief = bridge.get("precedent_brief")
            if precedent_brief is None:
                precedent_brief = atom_features.new_zeros((rows, AuditedEpisodeLogbook.brief_dim))
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
            wave_scalar_b = wave_reliability * torch.tanh(wave_scalar.detach())
            steric = self._optional_feature(batch.get("atom_3d_features"), rows, steric_dim, device=device, dtype=dtype)
            xtb = self._optional_feature(batch.get("xtb_atom_features"), rows, xtb_dim, device=device, dtype=dtype)
            topology = self._optional_feature(batch.get("topology_atom_features"), rows, topology_dim, device=device, dtype=dtype)
            steric_b = torch.tanh(steric)
            xtb_b = torch.tanh(xtb)
            topology_b = topology  # features are already in [0, 1]; no tanh needed
            wave_field_b = wave_reliability * torch.tanh(wave_field["atom_field_features"].detach())
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
            wave_site_bias_b = wave_reliability * torch.tanh(bridge["wave_site_bias"].detach())
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
            wave_vote = wave_reliability * (vote_scale * torch.tanh(wave_vote_raw))
            wave_conf = wave_reliability * torch.sigmoid(
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
            analogical_site_bias_b = torch.tanh(bridge["analogical_site_bias"].detach())
            continuous_reasoning_b = torch.tanh(bridge["continuous_reasoning_features"].detach())
            analogical_site_prior = bridge["analogical_site_prior"].detach()
            confidence_b = confidence.detach()
            analogical_cyp_context_b = analogical_cyp_context.detach()
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
            precedent_brief_vote = self._safe_vote_tensor(
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
            analogical_vote = vote_scale * torch.tanh(analogical_vote_raw)
            analogical_conf = torch.sigmoid(
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
                    precedent_brief.detach(),
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
            arbiter_in = torch.cat(
                [
                    board_context,
                    board_weights,
                    lnn_vote,
                    lnn_conf,
                    wave_vote,
                    wave_conf,
                    analogical_vote,
                    analogical_conf,
                ],
                dim=-1,
            )
            arbiter_in = torch.nan_to_num(arbiter_in, nan=0.0, posinf=4.0, neginf=-4.0)
            base_site_logits = outputs["site_logits"]
            arbiter_residual = self.site_arbiter_head(arbiter_in)
            # Per-molecule zero-centering: force the arbiter to only change
            # *relative* rankings within a molecule, not the global logit
            # magnitude. Without this the arbiter learns a persistent negative
            # bias (mean ≈ -2.5) that suppresses all predictions uniformly
            # instead of pushing the true site above wrong atoms.
            if batch_index.numel() > 0:
                num_mol_arb = int(batch_index.max().item()) + 1
                arb_sums = torch.zeros(num_mol_arb, 1, device=device, dtype=dtype)
                arb_cnts = torch.zeros(num_mol_arb, 1, device=device, dtype=dtype)
                idx_exp = batch_index.unsqueeze(-1)
                arb_sums.scatter_add_(0, idx_exp, arbiter_residual)
                arb_cnts.scatter_add_(0, idx_exp, torch.ones_like(arbiter_residual))
                arb_means = arb_sums / arb_cnts.clamp(min=1.0)
                arbiter_residual = arbiter_residual - arb_means[batch_index]
            # Treat the council and arbiter as residual corrections on top of the
            # base site logits so they cannot bury the base model from step 1.
            site_logits = base_site_logits + council_logit + arbiter_residual
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
                cyp_logits=outputs["cyp_logits"],
                atom_3d_features=batch.get("atom_3d_features"),
                xtb_atom_features=batch.get("xtb_atom_features"),
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
            ana_cyp_weight = torch.sigmoid(self.ana_cyp_weight_logit)
            council = None
            if self.site_arbiter_head is not None and self.site_arbiter_uses_bridge:
                site_logits, council = self._site_logits_from_arbiter(outputs, bridge, batch)
                site_mode = "nexus_arbiter"
            else:
                site_logits = outputs["site_logits"]
                site_mode = "base"
            cyp_logits = outputs["cyp_logits"] + ana_cyp_weight * bridge["analogical_cyp_bias"].detach()
            cyp_logits = torch.nan_to_num(cyp_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            diagnostics = dict(result.get("diagnostics") or {})
            diagnostics["nexus_bridge"] = {
                **bridge["metrics"],
                "analogical_cyp_weight": float(ana_cyp_weight.detach().item()),
                "site_mode": site_mode,
                "site_logits_base_mean": float(outputs["site_logits"].detach().mean().item()),
                "site_logits_council_mean": float(council["council_logit"].detach().mean().item()) if council else 0.0,
                "site_logits_arbiter_residual_mean": float(council["arbiter_residual"].detach().mean().item()) if council else 0.0,
                "site_logits_final_mean": float(site_logits.detach().mean().item()),
            }
            result.update(
                {
                    "site_logits": site_logits,
                    "reranked_site_logits": site_logits,
                    "site_scores": torch.sigmoid(site_logits),
                    "cyp_logits": cyp_logits,
                    "nexus_bridge_outputs": bridge,
                    "nexus_bridge_losses": bridge["losses"],
                    "atom_multivectors": bridge["atom_multivectors"],
                    "site_vote_heads": council,
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
                        cyp_logits=outputs["cyp_logits"],
                        atom_3d_features=batch.get("atom_3d_features"),
                        xtb_atom_features=batch.get("xtb_atom_features"),
                        site_labels=batch.get("site_labels"),
                        site_supervision_mask=batch.get("site_supervision_mask"),
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
            outputs = self._apply_nexus_bridge(outputs, batch)
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
                        cyp_logits=cyp_logits,
                        atom_3d_features=batch.get("atom_3d_features"),
                        xtb_atom_features=batch.get("xtb_atom_features"),
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
