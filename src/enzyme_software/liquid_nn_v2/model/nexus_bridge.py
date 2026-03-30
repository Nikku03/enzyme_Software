from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.model.liquid_branch import scatter_mean
from nexus.reasoning.metric_learner import HGNNProjection
from nexus.reasoning_wave.metric_learner import WaveQuantumDistillationHead, quantum_distillation_loss


if TORCH_AVAILABLE:
    class _OnlineAnalogicalAtomMemory(nn.Module):
        def __init__(
            self,
            *,
            key_dim: int,
            graph_dim: int,
            num_cyp_classes: int,
            capacity: int = 4096,
            topk: int = 32,
            graph_weight: float = 0.25,
            temperature: float = 0.25,
        ) -> None:
            super().__init__()
            self.key_dim = int(key_dim)
            self.graph_dim = int(graph_dim)
            self.num_cyp_classes = int(num_cyp_classes)
            self.capacity = max(64, int(capacity))
            self.topk = max(1, int(topk))
            self.graph_weight = float(max(graph_weight, 0.0))
            self.temperature = float(max(temperature, 1.0e-3))
            self.register_buffer("keys", torch.zeros(self.capacity, self.key_dim))
            self.register_buffer("graph", torch.zeros(self.capacity, self.graph_dim))
            self.register_buffer("site", torch.zeros(self.capacity, 1))
            self.register_buffer("cyp", torch.zeros(self.capacity, self.num_cyp_classes))
            self.register_buffer("valid", torch.zeros(self.capacity, dtype=torch.bool))
            self.register_buffer("ptr", torch.zeros((), dtype=torch.long))

        def size(self) -> int:
            return int(self.valid.sum().item())

        def lookup(self, query_keys: torch.Tensor, query_graph: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
            size = self.size()
            if size <= 0:
                return None
            mem_keys = self.keys[:size]
            mem_graph = self.graph[:size]
            mem_site = self.site[:size]
            mem_cyp = self.cyp[:size]
            q_keys = F.normalize(query_keys.float(), p=2, dim=-1)
            q_graph = F.normalize(query_graph.float(), p=2, dim=-1)
            k_keys = F.normalize(mem_keys.float(), p=2, dim=-1)
            k_graph = F.normalize(mem_graph.float(), p=2, dim=-1)
            atom_scores = torch.matmul(q_keys, k_keys.transpose(0, 1))
            graph_scores = torch.matmul(q_graph, k_graph.transpose(0, 1))
            scores = atom_scores + self.graph_weight * graph_scores
            k = min(self.topk, size)
            top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
            weights = F.softmax(top_scores / self.temperature, dim=-1)
            site_prior = (weights.unsqueeze(-1) * mem_site[top_idx]).sum(dim=1)
            cyp_prior = (weights.unsqueeze(-1) * mem_cyp[top_idx]).sum(dim=1)
            confidence = weights.max(dim=-1, keepdim=True).values
            return {
                "site_prior": site_prior,
                "cyp_prior": cyp_prior,
                "confidence": confidence,
                "top_scores": top_scores,
            }

        @torch.no_grad()
        def update(
            self,
            *,
            atom_keys: torch.Tensor,
            atom_graph: torch.Tensor,
            site_labels: torch.Tensor,
            cyp_probs: torch.Tensor,
            supervision_mask: Optional[torch.Tensor] = None,
        ) -> None:
            if atom_keys.numel() == 0:
                return
            mask = torch.ones(atom_keys.size(0), dtype=torch.bool, device=atom_keys.device)
            if supervision_mask is not None:
                mask = supervision_mask.view(-1) > 0.5
            if not bool(mask.any()):
                return
            keys = atom_keys[mask].detach()
            graph = atom_graph[mask].detach()
            site = site_labels[mask].detach().view(-1, 1)
            cyp = cyp_probs[mask].detach()
            count = int(keys.size(0))
            ptr = int(self.ptr.item())
            if count >= self.capacity:
                keys = keys[-self.capacity :]
                graph = graph[-self.capacity :]
                site = site[-self.capacity :]
                cyp = cyp[-self.capacity :]
                count = self.capacity
                ptr = 0
            end = ptr + count
            if end <= self.capacity:
                sl = slice(ptr, end)
                self.keys[sl] = keys
                self.graph[sl] = graph
                self.site[sl] = site
                self.cyp[sl] = cyp
                self.valid[sl] = True
            else:
                first = self.capacity - ptr
                second = count - first
                self.keys[ptr:] = keys[:first]
                self.graph[ptr:] = graph[:first]
                self.site[ptr:] = site[:first]
                self.cyp[ptr:] = cyp[:first]
                self.valid[ptr:] = True
                self.keys[:second] = keys[first:]
                self.graph[:second] = graph[first:]
                self.site[:second] = site[first:]
                self.cyp[:second] = cyp[first:]
                self.valid[:second] = True
            self.ptr.fill_(end % self.capacity)


    class NexusHybridBridge(nn.Module):
        """
        Lightweight bridge that ports the useful NEXUS pieces into the hybrid LNN:
        - 16D learned atom multivectors
        - wave-style quantum distillation head
        - online analogical atom memory for site/CYP priors
        """

        def __init__(
            self,
            *,
            atom_feature_dim: int,
            num_cyp_classes: int,
            steric_feature_dim: int = 8,
            xtb_feature_dim: int = 6,
            wave_hidden_dim: int = 64,
            graph_dim: int = 48,
            memory_capacity: int = 4096,
            memory_topk: int = 32,
            wave_aux_weight: float = 0.10,
            analogical_aux_weight: float = 0.08,
        ) -> None:
            super().__init__()
            self.atom_feature_dim = int(atom_feature_dim)
            self.num_cyp_classes = int(num_cyp_classes)
            self.steric_feature_dim = int(max(0, steric_feature_dim))
            self.xtb_feature_dim = int(max(0, xtb_feature_dim))
            self.wave_aux_weight = float(max(0.0, wave_aux_weight))
            self.analogical_aux_weight = float(max(0.0, analogical_aux_weight))
            total_in = self.atom_feature_dim + self.steric_feature_dim + self.xtb_feature_dim
            hidden = max(64, int(wave_hidden_dim))
            self.atom_to_multivector = nn.Sequential(
                nn.Linear(total_in, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 16),
            )
            self.coord_proj = nn.Linear(self.steric_feature_dim, 3) if self.steric_feature_dim > 0 else None
            self.scalar_proj = nn.Linear(self.xtb_feature_dim or self.atom_feature_dim, 1)
            self.pseudo_proj = nn.Linear(self.xtb_feature_dim or self.atom_feature_dim, 1)
            self.wave_head = WaveQuantumDistillationHead(hidden_dim=hidden, dropout=0.05)
            self.wave_site_head = nn.Sequential(
                nn.Linear(16 + 2 + 3, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            self.graph_encoder = HGNNProjection(
                in_channels_16d=16,
                hidden_dim=hidden,
                poincare_dim=int(max(16, graph_dim)),
                dropout=0.05,
            )
            key_dim = hidden // 2
            self.atom_key = nn.Sequential(
                nn.Linear(16 + self.atom_feature_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, key_dim),
            )
            self.analogical_site_head = nn.Sequential(
                nn.Linear(16 + 1 + self.num_cyp_classes + 1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            self.analogical_cyp_head = nn.Sequential(
                nn.Linear(int(max(16, graph_dim)) + self.num_cyp_classes + 1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.num_cyp_classes),
            )
            self.memory = _OnlineAnalogicalAtomMemory(
                key_dim=key_dim,
                graph_dim=int(max(16, graph_dim)),
                num_cyp_classes=self.num_cyp_classes,
                capacity=memory_capacity,
                topk=memory_topk,
            )

        def _optional_feature(self, value: Optional[torch.Tensor], rows: int, width: int, *, device, dtype) -> torch.Tensor:
            if value is None:
                return torch.zeros(rows, width, device=device, dtype=dtype)
            out = value.to(device=device, dtype=dtype)
            if out.ndim == 1:
                out = out.unsqueeze(-1)
            if out.size(-1) == width:
                return out
            if out.size(-1) > width:
                return out[..., :width]
            return F.pad(out, (0, width - int(out.size(-1))))

        def _build_multivectors(
            self,
            atom_features: torch.Tensor,
            atom_3d_features: Optional[torch.Tensor],
            xtb_atom_features: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rows = int(atom_features.size(0))
            device = atom_features.device
            dtype = atom_features.dtype
            steric = self._optional_feature(atom_3d_features, rows, self.steric_feature_dim, device=device, dtype=dtype)
            xtb = self._optional_feature(xtb_atom_features, rows, self.xtb_feature_dim, device=device, dtype=dtype)
            fused = torch.cat([atom_features, steric, xtb], dim=-1)
            multivectors = self.atom_to_multivector(fused)
            if self.coord_proj is not None and steric.numel():
                coords = torch.tanh(self.coord_proj(steric))
                multivectors[:, 1:4] = multivectors[:, 1:4] + coords
            else:
                coords = torch.zeros(rows, 3, device=device, dtype=dtype)
            scalar_src = xtb if self.xtb_feature_dim > 0 else atom_features
            multivectors[:, 0:1] = multivectors[:, 0:1] + self.scalar_proj(scalar_src)
            multivectors[:, 15:16] = multivectors[:, 15:16] + self.pseudo_proj(scalar_src)
            return multivectors, coords

        def _group_graph_embedding(self, multivectors: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
            num_molecules = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
            if num_molecules == 0:
                return multivectors.new_zeros((0, self.memory.graph_dim))
            out = []
            for mol_idx in range(num_molecules):
                mask = batch_index == mol_idx
                out.append(self.graph_encoder(multivectors[mask]))
            return torch.stack(out, dim=0)

        def _wave_predictions(
            self,
            multivectors: torch.Tensor,
            coords: torch.Tensor,
            batch_index: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            num_molecules = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
            pred_charge = multivectors.new_zeros(multivectors.size(0))
            pred_fukui = multivectors.new_zeros(multivectors.size(0))
            pred_gap = multivectors.new_zeros(num_molecules)
            for mol_idx in range(num_molecules):
                mask = batch_index == mol_idx
                preds = self.wave_head(multivectors[mask], atom_coords=coords[mask])
                pred_charge[mask] = preds["predicted_charges"].view(-1)
                pred_fukui[mask] = preds["predicted_fukui"].view(-1)
                pred_gap[mol_idx] = preds["predicted_gap"].view(-1)[0]
            return {
                "predicted_charges": pred_charge,
                "predicted_fukui": pred_fukui,
                "predicted_gap": pred_gap,
            }

        def _wave_aux_loss(
            self,
            *,
            wave_preds: dict[str, torch.Tensor],
            xtb_atom_features: Optional[torch.Tensor],
            batch_index: torch.Tensor,
            atom_3d_features: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, Dict[str, float]]:
            if self.wave_aux_weight <= 0.0 or xtb_atom_features is None or xtb_atom_features.numel() == 0:
                zero = wave_preds["predicted_charges"].sum() * 0.0
                return zero, {"wave_aux_loss": 0.0, "wave_charge_loss": 0.0, "wave_fukui_loss": 0.0, "wave_gap_loss": 0.0}
            xtb = xtb_atom_features.float()
            target_charge = xtb[:, 0]
            target_fukui = (0.45 * xtb[:, 1].abs() + 0.35 * xtb[:, 3].clamp_min(0.0) + 0.20 * xtb[:, 5].clamp_min(0.0))
            gap_atom = 0.5 * xtb[:, 4].clamp_min(0.0) + 0.5 * xtb[:, 5].clamp_min(0.0)
            num_molecules = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
            target_gap = scatter_mean(gap_atom.unsqueeze(-1), batch_index, num_molecules).squeeze(-1)
            coords = None
            if atom_3d_features is not None and atom_3d_features.numel():
                steric = atom_3d_features.float()
                coords = steric[:, :3] if steric.size(-1) >= 3 else F.pad(steric, (0, 3 - int(steric.size(-1))))
            loss, metrics = quantum_distillation_loss(
                predicted_charges=wave_preds["predicted_charges"],
                predicted_fukui=wave_preds["predicted_fukui"],
                predicted_gap=wave_preds["predicted_gap"],
                target_charges=target_charge,
                target_fukui=target_fukui,
                target_gap=target_gap,
                atom_mask=torch.ones_like(target_charge),
            )
            total = float(self.wave_aux_weight) * loss
            return total, {
                "wave_aux_loss": float(total.detach().item()),
                "wave_charge_loss": float(metrics["charge_loss"].detach().item()),
                "wave_fukui_loss": float(metrics["fukui_loss"].detach().item()),
                "wave_gap_loss": float(metrics["gap_loss"].detach().item()),
            }

        def _analogical_aux_loss(
            self,
            *,
            site_prior: torch.Tensor,
            site_labels: Optional[torch.Tensor],
            site_supervision_mask: Optional[torch.Tensor],
            cyp_prior_by_mol: torch.Tensor,
            cyp_labels: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, Dict[str, float]]:
            if self.analogical_aux_weight <= 0.0 or site_labels is None:
                zero = site_prior.sum() * 0.0
                return zero, {
                    "analogical_aux_loss": 0.0,
                    "analogical_site_loss": 0.0,
                    "analogical_cyp_loss": 0.0,
                }
            labels = site_labels.float().view(-1, 1)
            prior_logits = torch.logit(site_prior.clamp(1.0e-4, 1.0 - 1.0e-4))
            site_loss_raw = F.binary_cross_entropy_with_logits(prior_logits, labels, reduction="none")
            if site_supervision_mask is not None:
                mask = site_supervision_mask.float().view(-1, 1)
                site_loss = (site_loss_raw * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                site_loss = site_loss_raw.mean()
            cyp_loss = prior_logits.sum() * 0.0
            if cyp_labels is not None and cyp_prior_by_mol.numel():
                cyp_loss = F.cross_entropy(torch.log(cyp_prior_by_mol.clamp_min(1.0e-6)), cyp_labels.long())
            total = float(self.analogical_aux_weight) * (site_loss + 0.25 * cyp_loss)
            return total, {
                "analogical_aux_loss": float(total.detach().item()),
                "analogical_site_loss": float(site_loss.detach().item()),
                "analogical_cyp_loss": float(cyp_loss.detach().item()),
            }

        def forward(
            self,
            *,
            atom_features: torch.Tensor,
            batch_index: torch.Tensor,
            cyp_logits: torch.Tensor,
            atom_3d_features: Optional[torch.Tensor] = None,
            xtb_atom_features: Optional[torch.Tensor] = None,
            site_labels: Optional[torch.Tensor] = None,
            site_supervision_mask: Optional[torch.Tensor] = None,
            cyp_labels: Optional[torch.Tensor] = None,
        ) -> Dict[str, object]:
            atom_features = atom_features.float()
            multivectors, coords = self._build_multivectors(atom_features, atom_3d_features, xtb_atom_features)
            wave_preds = self._wave_predictions(multivectors, coords, batch_index)
            wave_bias_input = torch.cat(
                [
                    multivectors,
                    wave_preds["predicted_charges"].unsqueeze(-1),
                    wave_preds["predicted_fukui"].unsqueeze(-1),
                    coords,
                ],
                dim=-1,
            )
            wave_site_bias = self.wave_site_head(wave_bias_input)
            graph_embeddings = self._group_graph_embedding(multivectors, batch_index)
            per_atom_graph = graph_embeddings[batch_index] if graph_embeddings.numel() else multivectors.new_zeros((multivectors.size(0), self.memory.graph_dim))
            atom_keys = self.atom_key(torch.cat([multivectors, atom_features], dim=-1))
            retrieval = self.memory.lookup(atom_keys, per_atom_graph)
            if retrieval is None:
                site_prior = torch.full((multivectors.size(0), 1), 0.5, device=multivectors.device, dtype=multivectors.dtype)
                cyp_prior = F.softmax(cyp_logits.detach(), dim=-1)[batch_index]
                confidence = multivectors.new_zeros((multivectors.size(0), 1))
                analogical_site_bias = multivectors.new_zeros((multivectors.size(0), 1))
                analogical_cyp_bias = cyp_logits.new_zeros(cyp_logits.shape)
                cyp_prior_by_mol = F.softmax(cyp_logits.detach(), dim=-1)
            else:
                site_prior = retrieval["site_prior"]
                cyp_prior = retrieval["cyp_prior"]
                confidence = retrieval["confidence"]
                analogical_site_input = torch.cat([multivectors, site_prior, cyp_prior, confidence], dim=-1)
                analogical_site_bias = self.analogical_site_head(analogical_site_input)
                num_molecules = int(cyp_logits.size(0))
                cyp_prior_by_mol = scatter_mean(cyp_prior, batch_index, num_molecules)
                graph_conf = scatter_mean(confidence, batch_index, num_molecules)
                analogical_cyp_bias = self.analogical_cyp_head(torch.cat([graph_embeddings, cyp_prior_by_mol, graph_conf], dim=-1))
            wave_aux_loss, wave_metrics = self._wave_aux_loss(
                wave_preds=wave_preds,
                xtb_atom_features=xtb_atom_features,
                batch_index=batch_index,
                atom_3d_features=atom_3d_features,
            )
            analogical_aux_loss, analogical_metrics = self._analogical_aux_loss(
                site_prior=site_prior,
                site_labels=site_labels,
                site_supervision_mask=site_supervision_mask,
                cyp_prior_by_mol=cyp_prior_by_mol,
                cyp_labels=cyp_labels,
            )
            total_aux_loss = wave_aux_loss + analogical_aux_loss
            if self.training and site_labels is not None and cyp_logits.numel():
                cyp_probs_atom = F.softmax(cyp_logits.detach(), dim=-1)[batch_index]
                self.memory.update(
                    atom_keys=atom_keys,
                    atom_graph=per_atom_graph,
                    site_labels=site_labels.float().view(-1, 1),
                    cyp_probs=cyp_probs_atom,
                    supervision_mask=site_supervision_mask,
                )
            metrics = {
                **wave_metrics,
                **analogical_metrics,
                "memory_size": float(self.memory.size()),
                "analogical_confidence_mean": float(confidence.detach().mean().item()) if confidence.numel() else 0.0,
                "wave_charge_mean": float(wave_preds["predicted_charges"].detach().mean().item()) if wave_preds["predicted_charges"].numel() else 0.0,
                "wave_fukui_mean": float(wave_preds["predicted_fukui"].detach().mean().item()) if wave_preds["predicted_fukui"].numel() else 0.0,
            }
            return {
                "atom_multivectors": multivectors,
                "graph_embeddings": graph_embeddings,
                "wave_predictions": wave_preds,
                "wave_site_bias": wave_site_bias,
                "analogical_site_prior": site_prior,
                "analogical_cyp_prior": cyp_prior_by_mol,
                "analogical_confidence": confidence,
                "analogical_site_bias": analogical_site_bias,
                "analogical_cyp_bias": analogical_cyp_bias,
                "losses": {
                    "total": total_aux_loss,
                    "wave": wave_aux_loss,
                    "analogical": analogical_aux_loss,
                },
                "metrics": metrics,
            }
else:  # pragma: no cover
    class NexusHybridBridge:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()


__all__ = ["NexusHybridBridge"]
