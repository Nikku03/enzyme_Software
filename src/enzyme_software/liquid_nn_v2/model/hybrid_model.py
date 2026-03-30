from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.features.route_prior import combine_lnn_with_prior
from enzyme_software.liquid_nn_v2.model.nexus_bridge import NexusHybridBridge


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
            if bool(getattr(self.config, "use_nexus_bridge", True)):
                self.nexus_bridge = NexusHybridBridge(
                    atom_feature_dim=int(getattr(self.config, "som_branch_dim", getattr(self.config, "shared_hidden_dim", 128))),
                    num_cyp_classes=int(getattr(self.config, "num_cyp_classes", 9)),
                    steric_feature_dim=int(getattr(self.config, "steric_feature_dim", 8)),
                    xtb_feature_dim=6,
                    wave_hidden_dim=int(getattr(self.config, "nexus_wave_hidden_dim", 64)),
                    graph_dim=int(getattr(self.config, "nexus_graph_dim", 48)),
                    memory_capacity=int(getattr(self.config, "nexus_memory_capacity", 4096)),
                    memory_topk=int(getattr(self.config, "nexus_memory_topk", 32)),
                    wave_aux_weight=float(getattr(self.config, "nexus_wave_aux_weight", 0.10)),
                    analogical_aux_weight=float(getattr(self.config, "nexus_analogical_aux_weight", 0.08)),
                )
                self.nexus_bridge.set_memory_frozen(bool(getattr(self.config, "nexus_memory_frozen", False)))
                self.wave_site_weight_logit = nn.Parameter(
                    torch.logit(torch.tensor(float(getattr(self.config, "nexus_wave_site_init", 0.18))))
                )
                self.ana_site_weight_logit = nn.Parameter(
                    torch.logit(torch.tensor(float(getattr(self.config, "nexus_analogical_site_init", 0.20))))
                )
                self.ana_cyp_weight_logit = nn.Parameter(
                    torch.logit(torch.tensor(float(getattr(self.config, "nexus_analogical_cyp_init", 0.12))))
                )

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
            wave_weight = torch.sigmoid(self.wave_site_weight_logit)
            ana_site_weight = torch.sigmoid(self.ana_site_weight_logit)
            ana_cyp_weight = torch.sigmoid(self.ana_cyp_weight_logit)
            site_logits = (
                outputs["site_logits"]
                + wave_weight * bridge["wave_site_bias"]
                + ana_site_weight * bridge["analogical_site_bias"]
            )
            cyp_logits = outputs["cyp_logits"] + ana_cyp_weight * bridge["analogical_cyp_bias"]
            diagnostics = dict(result.get("diagnostics") or {})
            diagnostics["nexus_bridge"] = {
                **bridge["metrics"],
                "wave_site_weight": float(wave_weight.detach().item()),
                "analogical_site_weight": float(ana_site_weight.detach().item()),
                "analogical_cyp_weight": float(ana_cyp_weight.detach().item()),
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
                    "diagnostics": diagnostics,
                }
            )
            return result

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
