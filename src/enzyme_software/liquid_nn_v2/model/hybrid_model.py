from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
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
                xtb_dim = 6
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
                        + 10
                        + AuditedEpisodeLogbook.brief_dim
                        + 1
                        + 1
                        + 5
                    )
                    arbiter_in = board_context_dim + 9
                    self.lnn_vote_head = nn.Sequential(
                        nn.Linear(atom_dim + 1, council_hidden),
                        nn.SiLU(),
                        nn.Linear(council_hidden, 1),
                    )
                    self.wave_vote_head = nn.Sequential(
                        nn.Linear(14, council_hidden),
                        nn.SiLU(),
                        nn.Linear(council_hidden, 1),
                    )
                    self.analogical_vote_head = nn.Sequential(
                        nn.Linear(8 + AuditedEpisodeLogbook.brief_dim, council_hidden),
                        nn.SiLU(),
                        nn.Linear(council_hidden, 1),
                    )
                    self.lnn_conf_head = nn.Sequential(
                        nn.Linear(atom_dim, council_hidden),
                        nn.SiLU(),
                        nn.Linear(council_hidden, 1),
                    )
                    self.wave_conf_head = nn.Sequential(
                        nn.Linear(14, council_hidden),
                        nn.SiLU(),
                        nn.Linear(council_hidden, 1),
                    )
                    self.analogical_conf_head = nn.Sequential(
                        nn.Linear(8 + AuditedEpisodeLogbook.brief_dim, council_hidden),
                        nn.SiLU(),
                        nn.Linear(council_hidden, 1),
                    )
                    self.council_board_head = nn.Sequential(
                        nn.Linear(board_context_dim + 3, arbiter_hidden),
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
            xtb_dim = 6

            graph_embeddings = bridge["graph_embeddings"][batch_index]
            base_cyp_context = torch.softmax(outputs["cyp_logits"], dim=-1)[batch_index]
            analogical_cyp_context = bridge["analogical_cyp_prior"][batch_index]
            confidence = bridge["analogical_confidence"]
            wave_preds = bridge["wave_predictions"]
            wave_field = bridge["wave_field"]
            precedent_brief = bridge.get("precedent_brief")
            if precedent_brief is None:
                precedent_brief = atom_features.new_zeros((rows, AuditedEpisodeLogbook.brief_dim))
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
            wave_scalar_b = torch.tanh(wave_scalar.detach())
            steric = self._optional_feature(batch.get("atom_3d_features"), rows, steric_dim, device=device, dtype=dtype)
            xtb = self._optional_feature(batch.get("xtb_atom_features"), rows, xtb_dim, device=device, dtype=dtype)
            steric_b = torch.tanh(steric)
            xtb_b = torch.tanh(xtb)
            wave_field_b = torch.tanh(wave_field["atom_field_features"].detach())
            lnn_vote = self.lnn_vote_head(
                torch.cat(
                    [
                        atom_features_b,
                        torch.tanh(outputs["site_logits"]),
                    ],
                    dim=-1,
                )
            )
            lnn_conf = torch.sigmoid(self.lnn_conf_head(atom_features_b))
            wave_site_bias_b = torch.tanh(bridge["wave_site_bias"].detach())
            wave_vote = self.wave_vote_head(
                torch.cat(
                    [
                        wave_field_b,
                        wave_scalar_b,
                        wave_site_bias_b,
                    ],
                    dim=-1,
                )
            )
            wave_conf = torch.sigmoid(
                self.wave_conf_head(
                    torch.cat(
                        [
                            wave_field_b,
                            wave_scalar_b,
                            wave_site_bias_b,
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
            analogical_vote = self.analogical_vote_head(
                torch.cat(
                    [
                        analogical_site_prior,
                        confidence_b,
                        analogical_site_bias_b,
                        continuous_reasoning_b,
                        precedent_brief.detach(),
                    ],
                    dim=-1,
                )
            )
            analogical_conf = torch.sigmoid(
                self.analogical_conf_head(
                    torch.cat(
                        [
                            analogical_site_prior,
                            confidence_b,
                            analogical_site_bias_b,
                            continuous_reasoning_b,
                            precedent_brief.detach(),
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
                    lnn_conf,
                    wave_conf,
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
            arbiter_in = torch.cat(
                [
                    board_context,
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
            site_logits = council_logit + self.site_arbiter_head(arbiter_in)
            site_logits = torch.nan_to_num(site_logits, nan=0.0, posinf=20.0, neginf=-20.0)
            council = {
                "lnn_vote": lnn_vote,
                "lnn_conf": lnn_conf,
                "wave_vote": wave_vote,
                "wave_conf": wave_conf,
                "analogical_vote": analogical_vote,
                "analogical_conf": analogical_conf,
                "council_logit": council_logit,
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
                site_labels=batch.get("site_labels"),
                site_supervision_mask=batch.get("site_supervision_mask"),
                cyp_labels=batch.get("cyp_labels"),
                cyp_supervision_mask=batch.get("cyp_supervision_mask"),
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
else:  # pragma: no cover
    class HybridLNNModel:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
