from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.features.xtb_features import FULL_XTB_FEATURE_DIM
from enzyme_software.liquid_nn_v2.model.liquid_branch import scatter_mean
from enzyme_software.liquid_nn_v2.model.precedent_logbook import AuditedEpisodeLogbook
from enzyme_software.liquid_nn_v2.model.wave_field import WholeMoleculeWaveField
from nexus.reasoning.metric_learner import HGNNProjection
from nexus.reasoning_wave.analogical_fusion import NexusDualDecoder, PGWCrossAttention
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
            hard_negative_ratio: float = 2.0,
            min_hard_negatives: int = 32,
            max_hard_negatives: int = 256,
        ) -> None:
            super().__init__()
            self.key_dim = int(key_dim)
            self.graph_dim = int(graph_dim)
            self.num_cyp_classes = int(num_cyp_classes)
            self.capacity = max(64, int(capacity))
            self.topk = max(1, int(topk))
            self.graph_weight = float(max(graph_weight, 0.0))
            self.temperature = float(max(temperature, 1.0e-3))
            self.hard_negative_ratio = float(max(hard_negative_ratio, 0.0))
            self.min_hard_negatives = max(0, int(min_hard_negatives))
            self.max_hard_negatives = max(self.min_hard_negatives, int(max_hard_negatives))
            self.register_buffer("keys", torch.zeros(self.capacity, self.key_dim))
            self.register_buffer("graph", torch.zeros(self.capacity, self.graph_dim))
            self.register_buffer("site", torch.zeros(self.capacity, 1))
            self.register_buffer("cyp", torch.zeros(self.capacity, self.num_cyp_classes))
            self.register_buffer("multivector", torch.zeros(self.capacity, 16))
            self.register_buffer("molecule_key", torch.zeros(self.capacity, dtype=torch.long))
            self.register_buffer("valid", torch.zeros(self.capacity, dtype=torch.bool))
            self.register_buffer("ptr", torch.zeros((), dtype=torch.long))

        def size(self) -> int:
            return int(self.valid.sum().item())

        @torch.no_grad()
        def clear(self) -> None:
            self.keys.zero_()
            self.graph.zero_()
            self.site.zero_()
            self.cyp.zero_()
            self.multivector.zero_()
            self.molecule_key.zero_()
            self.valid.zero_()
            self.ptr.zero_()

        def lookup(
            self,
            query_keys: torch.Tensor,
            query_graph: torch.Tensor,
            query_molecule_keys: Optional[torch.Tensor] = None,
        ) -> Optional[Dict[str, torch.Tensor]]:
            size = self.size()
            if size <= 0:
                return None
            mem_keys = self.keys[:size]
            mem_graph = self.graph[:size]
            mem_site = self.site[:size]
            mem_cyp = self.cyp[:size]
            mem_mv = self.multivector[:size]
            mem_molecule_key = self.molecule_key[:size]
            q_keys = F.normalize(query_keys.float(), p=2, dim=-1)
            q_graph = F.normalize(query_graph.float(), p=2, dim=-1)
            k_keys = F.normalize(mem_keys.float(), p=2, dim=-1)
            k_graph = F.normalize(mem_graph.float(), p=2, dim=-1)
            atom_scores = torch.matmul(q_keys, k_keys.transpose(0, 1))
            graph_scores = torch.matmul(q_graph, k_graph.transpose(0, 1))
            scores = atom_scores + self.graph_weight * graph_scores
            if query_molecule_keys is not None:
                qm = query_molecule_keys.to(device=scores.device, dtype=torch.long).view(-1, 1)
                same_molecule = qm == mem_molecule_key.view(1, -1)
                scores = scores.masked_fill(same_molecule, -1.0e9)
            k = min(self.topk, size)
            top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
            valid_rows = top_scores[:, 0] > -1.0e8
            valid_top = top_scores > -1.0e8
            scaled_scores = (top_scores / self.temperature).masked_fill(~valid_top, -1.0e9)
            weights = F.softmax(scaled_scores, dim=-1) * valid_top.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            site_prior = (weights.unsqueeze(-1) * mem_site[top_idx]).sum(dim=1)
            cyp_prior = (weights.unsqueeze(-1) * mem_cyp[top_idx]).sum(dim=1)
            best_score = top_scores[:, :1]
            if int(top_scores.size(-1)) > 1:
                support_margin = (top_scores[:, :1] - top_scores[:, 1:2]).clamp_min(0.0)
            else:
                support_margin = top_scores[:, :1].clamp_min(0.0)
            concentration = weights.pow(2).sum(dim=-1, keepdim=True)
            if k > 1:
                concentration = (concentration - (1.0 / float(k))) / (1.0 - (1.0 / float(k)))
            concentration = concentration.clamp(0.0, 1.0)
            score_conf = torch.sigmoid((best_score - 0.30) / 0.12)
            margin_conf = torch.sigmoid((support_margin - 0.04) / 0.04)
            confidence = (0.50 * score_conf) + (0.30 * margin_conf) + (0.20 * concentration)
            confidence = confidence * valid_rows.unsqueeze(-1).to(dtype=confidence.dtype)
            # Separate "there is some memory support" from "the retrieval is
            # selective enough to trust".  The benchmark failures showed that the
            # analogical branch is often active but diffuse (tiny margin /
            # concentration), which should cause abstention rather than a vote.
            selectivity = (
                torch.sigmoid((support_margin - 0.025) / 0.025)
                * torch.sigmoid((concentration - 0.10) / 0.08)
            )
            selectivity = selectivity * valid_rows.unsqueeze(-1).to(dtype=confidence.dtype)
            trust_gate = (confidence * selectivity).clamp(0.0, 1.0)
            return {
                "site_prior": site_prior,
                "cyp_prior": cyp_prior,
                "confidence": confidence,
                "selectivity": selectivity,
                "trust_gate": trust_gate,
                "best_score": best_score,
                "support_margin": support_margin,
                "concentration": concentration,
                "top_scores": top_scores,
                "top_weights": weights,
                "retrieved_multivectors": mem_mv[top_idx],
                "valid_rows": valid_rows.unsqueeze(-1),
            }

        @torch.no_grad()
        def update(
            self,
            *,
            atom_keys: torch.Tensor,
            atom_graph: torch.Tensor,
            atom_multivectors: torch.Tensor,
            site_labels: torch.Tensor,
            cyp_probs: torch.Tensor,
            molecule_keys: torch.Tensor,
            supervision_mask: Optional[torch.Tensor] = None,
            hard_negative_scores: Optional[torch.Tensor] = None,
        ) -> int:
            if atom_keys.numel() == 0:
                return 0
            mask = torch.ones(atom_keys.size(0), dtype=torch.bool, device=atom_keys.device)
            if supervision_mask is not None:
                mask = supervision_mask.view(-1) > 0.5
            if not bool(mask.any()):
                return 0
            labels_flat = site_labels.view(-1)
            positive_mask = mask & (labels_flat > 0.5)
            selected_mask = positive_mask.clone()
            negative_mask = mask & ~positive_mask
            if bool(negative_mask.any()):
                negative_idx = torch.nonzero(negative_mask, as_tuple=False).view(-1)
                if hard_negative_scores is not None:
                    hardness = hard_negative_scores.view(-1)[negative_idx].detach().float()
                else:
                    hardness = torch.zeros_like(negative_idx, dtype=atom_keys.dtype)
                positive_count = int(positive_mask.sum().item())
                if positive_count > 0:
                    desired_negatives = max(self.min_hard_negatives, int(round(self.hard_negative_ratio * positive_count)))
                else:
                    desired_negatives = self.min_hard_negatives
                desired_negatives = min(desired_negatives, self.max_hard_negatives, int(negative_idx.numel()))
                if desired_negatives > 0:
                    hard_order = torch.topk(hardness, k=desired_negatives, dim=0).indices
                    selected_mask[negative_idx[hard_order]] = True
            if not bool(selected_mask.any()):
                return 0
            keys = atom_keys[selected_mask].detach()
            graph = atom_graph[selected_mask].detach()
            multivector = atom_multivectors[selected_mask].detach()
            site = site_labels[selected_mask].detach().view(-1, 1)
            cyp = cyp_probs[selected_mask].detach()
            mol_keys = molecule_keys[selected_mask].detach().view(-1)
            count = int(keys.size(0))
            ptr = int(self.ptr.item())
            if count >= self.capacity:
                keys = keys[-self.capacity :]
                graph = graph[-self.capacity :]
                multivector = multivector[-self.capacity :]
                site = site[-self.capacity :]
                cyp = cyp[-self.capacity :]
                mol_keys = mol_keys[-self.capacity :]
                count = self.capacity
                ptr = 0
            end = ptr + count
            if end <= self.capacity:
                sl = slice(ptr, end)
                self.keys[sl] = keys
                self.graph[sl] = graph
                self.multivector[sl] = multivector
                self.site[sl] = site
                self.cyp[sl] = cyp
                self.molecule_key[sl] = mol_keys
                self.valid[sl] = True
            else:
                first = self.capacity - ptr
                second = count - first
                self.keys[ptr:] = keys[:first]
                self.graph[ptr:] = graph[:first]
                self.multivector[ptr:] = multivector[:first]
                self.site[ptr:] = site[:first]
                self.cyp[ptr:] = cyp[:first]
                self.molecule_key[ptr:] = mol_keys[:first]
                self.valid[ptr:] = True
                self.keys[:second] = keys[first:]
                self.graph[:second] = graph[first:]
                self.multivector[:second] = multivector[first:]
                self.site[:second] = site[first:]
                self.cyp[:second] = cyp[first:]
                self.molecule_key[:second] = mol_keys[first:]
                self.valid[:second] = True
            self.ptr.fill_(end % self.capacity)
            return count


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
            xtb_feature_dim: int = FULL_XTB_FEATURE_DIM,
            topology_feature_dim: int = 5,
            wave_hidden_dim: int = 64,
            graph_dim: int = 48,
            memory_capacity: int = 4096,
            memory_topk: int = 32,
            wave_aux_weight: float = 0.10,
            analogical_aux_weight: float = 0.08,
            analogical_cyp_aux_scale: float = 0.10,
        ) -> None:
            super().__init__()
            self.atom_feature_dim = int(atom_feature_dim)
            self.num_cyp_classes = int(num_cyp_classes)
            self.steric_feature_dim = int(max(0, steric_feature_dim))
            self.xtb_feature_dim = int(max(0, xtb_feature_dim))
            self.topology_feature_dim = int(max(0, topology_feature_dim))
            self.wave_aux_weight = float(max(0.0, wave_aux_weight))
            self.analogical_aux_weight = float(max(0.0, analogical_aux_weight))
            self.analogical_cyp_aux_scale = float(max(0.0, analogical_cyp_aux_scale))
            self.memory_frozen = False
            self.precedent_logbook = AuditedEpisodeLogbook(max_cases=max(4096, memory_capacity * 8), topk=max(8, min(32, memory_topk)))
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
            self.wave_field = WholeMoleculeWaveField(multivector_dim=16, hidden_dim=hidden)
            self.wave_site_head = nn.Sequential(
                nn.Linear(16 + 2 + 3 + WholeMoleculeWaveField.field_feature_dim + 2, hidden),
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
            backbone_key_dim = max(16, hidden // 4)
            local_key_dim = max(16, hidden // 4)
            local_descriptor_dim = self.steric_feature_dim + self.xtb_feature_dim + self.topology_feature_dim
            self.backbone_key_proj = nn.Sequential(
                nn.LayerNorm(self.atom_feature_dim),
                nn.Linear(self.atom_feature_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, backbone_key_dim),
            )
            self.local_descriptor_proj = (
                nn.Sequential(
                    nn.LayerNorm(local_descriptor_dim),
                    nn.Linear(local_descriptor_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, local_key_dim),
                )
                if local_descriptor_dim > 0
                else None
            )
            # Retrieval keys need both trainable bridge state and stable chemistry
            # signals. Bridge-only keys were too diffuse on the benchmark.
            _key_in_dim = 16 + WholeMoleculeWaveField.field_feature_dim + backbone_key_dim + local_key_dim
            self.atom_key = nn.Sequential(
                nn.LayerNorm(_key_in_dim),
                nn.Linear(_key_in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, key_dim),
            )
            self.continuous_cross_attention = PGWCrossAttention(hidden_dim=hidden)
            self.continuous_reason_proj = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 16),
            )
            self.continuous_dual_decoder = NexusDualDecoder(hidden_dim=max(16, hidden // 2))
            self.analogical_site_head = nn.Sequential(
                nn.Linear(16 + 1 + self.num_cyp_classes + 1 + AuditedEpisodeLogbook.brief_dim, hidden),
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

        def _continuous_reasoning(
            self,
            *,
            query_multivectors: torch.Tensor,
            retrieval: Optional[Dict[str, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            rows = int(query_multivectors.size(0))
            device = query_multivectors.device
            dtype = query_multivectors.dtype
            zero_bias = torch.zeros((rows, 1), device=device, dtype=dtype)
            zero_features = torch.zeros((rows, 5), device=device, dtype=dtype)
            if retrieval is None:
                return {
                    "site_bias": zero_bias,
                    "features": zero_features,
                }

            retrieved_mv = retrieval["retrieved_multivectors"].to(device=device, dtype=dtype)
            top_weights = retrieval["top_weights"].to(device=device, dtype=dtype)
            top_scores = retrieval["top_scores"].to(device=device, dtype=dtype)
            q_fp = query_multivectors.unsqueeze(1)
            pi_star = top_weights.unsqueeze(1)
            context = self.continuous_cross_attention(q_fp, retrieved_mv, pi_star).squeeze(1)
            reason_mv = self.continuous_reason_proj(context)
            y_fp_som, _y_fp_morph, y_ana_som, _y_ana_morph = self.continuous_dual_decoder(query_multivectors, reason_mv)
            support_mean = top_scores.mean(dim=-1, keepdim=True)
            if int(top_scores.size(-1)) > 1:
                support_margin = (top_scores[:, :1] - top_scores[:, 1:2]).clamp_min(0.0)
            else:
                support_margin = top_scores[:, :1]
            context_norm = (context.pow(2).sum(dim=-1, keepdim=True) + 1.0e-12).sqrt() / max(1.0, float(context.size(-1)) ** 0.5)
            valid_rows = retrieval.get("valid_rows")
            if valid_rows is not None:
                valid_rows_f = valid_rows.to(device=device, dtype=dtype)
                top_scores = top_scores * valid_rows_f
                support_mean = support_mean * valid_rows_f
                support_margin = support_margin * valid_rows_f
                context_norm = context_norm * valid_rows_f
            reason_features = torch.cat(
                [
                    y_ana_som.unsqueeze(-1),
                    y_fp_som.unsqueeze(-1),
                    support_mean,
                    support_margin,
                    context_norm,
                ],
                dim=-1,
            )
            if valid_rows is not None:
                valid_rows_f = valid_rows.to(device=device, dtype=dtype)
                reason_features = reason_features * valid_rows_f
                site_bias = y_ana_som.unsqueeze(-1) * valid_rows_f
            else:
                site_bias = y_ana_som.unsqueeze(-1)
            return {
                "site_bias": site_bias,
                "features": reason_features,
            }

        @torch.no_grad()
        def clear_memory(self) -> None:
            self.memory.clear()

        def set_memory_frozen(self, frozen: bool) -> None:
            self.memory_frozen = bool(frozen)

        @torch.no_grad()
        def load_precedent_logbook(self, path: str, *, cyp_names: Optional[list[str]] = None) -> Dict[str, float]:
            return self.precedent_logbook.load_jsonl(path, cyp_names=cyp_names, allowed_splits=("train",))

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

        def _sanitize_feature_tensor(
            self,
            value: torch.Tensor,
            *,
            clamp_value: float = 8.0,
        ) -> torch.Tensor:
            return torch.nan_to_num(
                value,
                nan=0.0,
                posinf=clamp_value,
                neginf=-clamp_value,
            ).clamp(min=-clamp_value, max=clamp_value)

        def _build_multivectors(
            self,
            atom_features: torch.Tensor,
            atom_3d_features: Optional[torch.Tensor],
            xtb_atom_features: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rows = int(atom_features.size(0))
            device = atom_features.device
            dtype = atom_features.dtype
            steric = self._sanitize_feature_tensor(
                self._optional_feature(atom_3d_features, rows, self.steric_feature_dim, device=device, dtype=dtype)
            )
            xtb = self._sanitize_feature_tensor(
                self._optional_feature(xtb_atom_features, rows, self.xtb_feature_dim, device=device, dtype=dtype)
            )
            fused = torch.cat([atom_features, steric, xtb], dim=-1)
            multivectors = self.atom_to_multivector(fused)
            if self.steric_feature_dim > 0 and steric.numel():
                if steric.size(-1) >= 3:
                    coords = steric[:, :3]
                else:
                    coords = F.pad(steric, (0, 3 - int(steric.size(-1))))
                coords = self._sanitize_feature_tensor(coords, clamp_value=4.0)
            else:
                coords = torch.zeros(rows, 3, device=device, dtype=dtype)
            if self.coord_proj is not None and steric.numel():
                coord_delta = 0.25 * torch.tanh(self.coord_proj(steric))
                coord_delta = self._sanitize_feature_tensor(coord_delta, clamp_value=1.0)
                multivectors[:, 1:4] = multivectors[:, 1:4] + coord_delta
            scalar_src = xtb if self.xtb_feature_dim > 0 else atom_features
            multivectors[:, 0:1] = multivectors[:, 0:1] + self.scalar_proj(scalar_src)
            multivectors[:, 15:16] = multivectors[:, 15:16] + self.pseudo_proj(scalar_src)
            return multivectors, coords

        def _build_atom_keys(
            self,
            *,
            atom_features: torch.Tensor,
            multivectors: torch.Tensor,
            atom_field_features: torch.Tensor,
            atom_3d_features: Optional[torch.Tensor],
            xtb_atom_features: Optional[torch.Tensor],
            topology_atom_features: Optional[torch.Tensor],
        ) -> torch.Tensor:
            rows = int(atom_features.size(0))
            device = atom_features.device
            dtype = atom_features.dtype
            backbone_features = self._sanitize_feature_tensor(atom_features)
            backbone_proj = self.backbone_key_proj(backbone_features)
            if self.local_descriptor_proj is not None:
                steric = self._sanitize_feature_tensor(
                    self._optional_feature(atom_3d_features, rows, self.steric_feature_dim, device=device, dtype=dtype)
                )
                xtb = self._sanitize_feature_tensor(
                    self._optional_feature(xtb_atom_features, rows, self.xtb_feature_dim, device=device, dtype=dtype)
                )
                topology = self._sanitize_feature_tensor(
                    self._optional_feature(topology_atom_features, rows, self.topology_feature_dim, device=device, dtype=dtype)
                )
                local_descriptor = torch.cat([steric, xtb, topology], dim=-1)
                local_proj = self.local_descriptor_proj(local_descriptor)
            else:
                local_proj = multivectors.new_zeros((rows, 0))
            key_input = torch.cat(
                [
                    multivectors,
                    self._sanitize_feature_tensor(atom_field_features),
                    backbone_proj,
                    local_proj,
                ],
                dim=-1,
            )
            return self.atom_key(key_input)

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
            atom_valid_mask: Optional[torch.Tensor] = None,
            mol_valid_mask: Optional[torch.Tensor] = None,
        ) -> dict[str, torch.Tensor]:
            num_molecules = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
            pred_charge = multivectors.new_zeros(multivectors.size(0))
            pred_fukui = multivectors.new_zeros(multivectors.size(0))
            pred_gap = multivectors.new_zeros(num_molecules)
            for mol_idx in range(num_molecules):
                if mol_valid_mask is not None and float(mol_valid_mask[mol_idx].item()) <= 0.5:
                    continue
                mask = batch_index == mol_idx
                if atom_valid_mask is not None:
                    mask = mask & (atom_valid_mask.view(-1) > 0.5)
                if not bool(mask.any()):
                    continue
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
            xtb_atom_valid_mask: Optional[torch.Tensor],
            xtb_mol_valid: Optional[torch.Tensor],
            batch_index: torch.Tensor,
            atom_3d_features: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, Dict[str, float]]:
            if self.wave_aux_weight <= 0.0 or xtb_atom_features is None or xtb_atom_features.numel() == 0:
                zero = wave_preds["predicted_charges"].sum() * 0.0
                return zero, {"wave_aux_loss": 0.0, "wave_charge_loss": 0.0, "wave_fukui_loss": 0.0, "wave_gap_loss": 0.0}
            xtb = xtb_atom_features.float()
            if float(xtb.detach().abs().sum().item()) < 1.0e-8:
                zero = wave_preds["predicted_charges"].sum() * 0.0
                return zero, {"wave_aux_loss": 0.0, "wave_charge_loss": 0.0, "wave_fukui_loss": 0.0, "wave_gap_loss": 0.0}
            atom_valid = torch.ones_like(wave_preds["predicted_charges"], dtype=torch.float32)
            if xtb_atom_valid_mask is not None and xtb_atom_valid_mask.numel():
                atom_valid = xtb_atom_valid_mask.float().view(-1).clamp(0.0, 1.0)
            if float(atom_valid.sum().detach().item()) < 1.0e-6:
                zero = wave_preds["predicted_charges"].sum() * 0.0
                return zero, {"wave_aux_loss": 0.0, "wave_charge_loss": 0.0, "wave_fukui_loss": 0.0, "wave_gap_loss": 0.0}
            target_charge = xtb[:, 0]
            target_fukui = (0.45 * xtb[:, 1].abs() + 0.35 * xtb[:, 3].clamp_min(0.0) + 0.20 * xtb[:, 5].clamp_min(0.0))
            gap_atom = 0.5 * xtb[:, 4].clamp_min(0.0) + 0.5 * xtb[:, 5].clamp_min(0.0)
            num_molecules = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
            idx = batch_index.long()
            gap_sum = torch.zeros(num_molecules, device=gap_atom.device, dtype=gap_atom.dtype)
            gap_count = torch.zeros(num_molecules, device=gap_atom.device, dtype=gap_atom.dtype)
            gap_sum.scatter_add_(0, idx, gap_atom * atom_valid)
            gap_count.scatter_add_(0, idx, atom_valid)
            target_gap = gap_sum / gap_count.clamp_min(1.0)
            if xtb_mol_valid is not None and xtb_mol_valid.numel():
                mol_valid = xtb_mol_valid.float().view(-1).clamp(0.0, 1.0)
            else:
                mol_valid = (gap_count > 0.0).float()
            charge_loss_raw = F.smooth_l1_loss(wave_preds["predicted_charges"], target_charge, reduction="none")
            fukui_loss_raw = F.smooth_l1_loss(wave_preds["predicted_fukui"], target_fukui, reduction="none")
            gap_loss_raw = F.smooth_l1_loss(wave_preds["predicted_gap"], target_gap, reduction="none")
            charge_loss = (charge_loss_raw * atom_valid).sum() / atom_valid.sum().clamp_min(1.0)
            fukui_loss = (fukui_loss_raw * atom_valid).sum() / atom_valid.sum().clamp_min(1.0)
            gap_loss = (gap_loss_raw * mol_valid).sum() / mol_valid.sum().clamp_min(1.0)
            loss = charge_loss + fukui_loss + gap_loss
            total = float(self.wave_aux_weight) * loss
            return total, {
                "wave_aux_loss": float(total.detach().item()),
                "wave_charge_loss": float(charge_loss.detach().item()),
                "wave_fukui_loss": float(fukui_loss.detach().item()),
                "wave_gap_loss": float(gap_loss.detach().item()),
            }

        def _analogical_aux_loss(
            self,
            *,
            site_prior: torch.Tensor,
            site_labels: Optional[torch.Tensor],
            site_supervision_mask: Optional[torch.Tensor],
            cyp_prior_by_mol: torch.Tensor,
            cyp_labels: Optional[torch.Tensor],
            cyp_supervision_mask: Optional[torch.Tensor],
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
                cyp_mask = None
                if cyp_supervision_mask is not None:
                    cyp_mask = cyp_supervision_mask.view(-1) > 0.5
                if cyp_mask is None:
                    cyp_loss = F.cross_entropy(torch.log(cyp_prior_by_mol.clamp_min(1.0e-6)), cyp_labels.long())
                elif bool(cyp_mask.any()):
                    cyp_loss = F.cross_entropy(
                        torch.log(cyp_prior_by_mol[cyp_mask].clamp_min(1.0e-6)),
                        cyp_labels[cyp_mask].long(),
                    )
            cyp_scale = float(getattr(self, "analogical_cyp_aux_scale", 0.10))
            total = float(self.analogical_aux_weight) * (site_loss + cyp_scale * cyp_loss)
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
            site_logits: Optional[torch.Tensor] = None,
            cyp_logits: torch.Tensor,
            atom_3d_features: Optional[torch.Tensor] = None,
            xtb_atom_features: Optional[torch.Tensor] = None,
            topology_atom_features: Optional[torch.Tensor] = None,
            xtb_atom_valid_mask: Optional[torch.Tensor] = None,
            xtb_mol_valid: Optional[torch.Tensor] = None,
            site_labels: Optional[torch.Tensor] = None,
            site_supervision_mask: Optional[torch.Tensor] = None,
            cyp_labels: Optional[torch.Tensor] = None,
            cyp_supervision_mask: Optional[torch.Tensor] = None,
            graph_molecule_keys: Optional[torch.Tensor] = None,
        ) -> Dict[str, object]:
            atom_features = atom_features.float()
            multivectors, coords = self._build_multivectors(atom_features, atom_3d_features, xtb_atom_features)
            if xtb_mol_valid is not None and xtb_mol_valid.numel():
                mol_valid_mask = xtb_mol_valid.float().to(device=batch_index.device, dtype=multivectors.dtype).view(-1, 1).clamp(0.0, 1.0)
            else:
                mol_valid_mask = None
            if xtb_atom_valid_mask is not None and xtb_atom_valid_mask.numel():
                atom_valid_mask = xtb_atom_valid_mask.float().to(device=multivectors.device, dtype=multivectors.dtype).view(-1, 1).clamp(0.0, 1.0)
            elif xtb_atom_features is not None and xtb_atom_features.numel():
                atom_valid_mask = torch.ones((multivectors.size(0), 1), device=multivectors.device, dtype=multivectors.dtype)
            else:
                atom_valid_mask = torch.zeros((multivectors.size(0), 1), device=multivectors.device, dtype=multivectors.dtype)
            wave_compute_mask = atom_valid_mask
            if mol_valid_mask is not None:
                wave_compute_mask = wave_compute_mask * mol_valid_mask[batch_index]
            masked_multivectors = multivectors * wave_compute_mask
            masked_coords = coords * wave_compute_mask
            stable_coords = coords.detach()
            wave_preds = self._wave_predictions(
                masked_multivectors,
                stable_coords * wave_compute_mask.detach(),
                batch_index,
                atom_valid_mask=atom_valid_mask,
                mol_valid_mask=mol_valid_mask,
            )
            wave_field = self.wave_field(masked_multivectors, stable_coords * wave_compute_mask.detach(), batch_index)
            wave_field["atom_field_features"] = wave_field["atom_field_features"] * wave_compute_mask
            wave_field["global_density"] = wave_field["global_density"] * wave_compute_mask.view(-1)
            wave_field["global_gap_proxy"] = wave_field["global_gap_proxy"] * (
                mol_valid_mask.view(-1) if mol_valid_mask is not None else torch.ones_like(wave_field["global_gap_proxy"])
            )
            wave_bias_input = torch.cat(
                [
                    masked_multivectors,
                    wave_preds["predicted_charges"].unsqueeze(-1),
                    wave_preds["predicted_fukui"].unsqueeze(-1),
                    masked_coords,
                    wave_field["atom_field_features"],
                    wave_field["global_density"].unsqueeze(-1),
                    wave_field["global_gap_proxy"][batch_index].unsqueeze(-1),
                ],
                dim=-1,
            )
            wave_site_bias = self.wave_site_head(wave_bias_input)
            wave_reliability = atom_valid_mask
            if xtb_atom_features is not None and xtb_atom_features.numel() and xtb_atom_features.size(-1) >= 7:
                xtb_confidence = xtb_atom_features[:, 6:7].float().to(device=multivectors.device, dtype=multivectors.dtype).clamp(0.0, 1.0)
                wave_reliability = wave_reliability * torch.where(wave_reliability > 0.5, xtb_confidence, torch.ones_like(xtb_confidence))
            if mol_valid_mask is not None:
                wave_reliability = wave_reliability * mol_valid_mask[batch_index]
            wave_site_bias = wave_reliability * wave_site_bias
            graph_embeddings = self._group_graph_embedding(multivectors, batch_index)
            per_atom_graph = graph_embeddings[batch_index] if graph_embeddings.numel() else multivectors.new_zeros((multivectors.size(0), self.memory.graph_dim))
            if graph_molecule_keys is not None and graph_molecule_keys.numel():
                atom_molecule_keys = graph_molecule_keys.to(device=batch_index.device, dtype=torch.long)[batch_index]
            else:
                atom_molecule_keys = torch.zeros_like(batch_index, dtype=torch.long)
            atom_keys = self._build_atom_keys(
                atom_features=atom_features,
                multivectors=multivectors,
                atom_field_features=wave_field["atom_field_features"],
                atom_3d_features=atom_3d_features,
                xtb_atom_features=xtb_atom_features,
                topology_atom_features=topology_atom_features,
            )
            precedent_query = torch.cat(
                [
                    multivectors,
                    wave_preds["predicted_charges"].unsqueeze(-1),
                    wave_preds["predicted_fukui"].unsqueeze(-1),
                    wave_field["atom_field_features"],
                ],
                dim=-1,
            )
            cyp_probs_by_mol = F.softmax(cyp_logits.detach(), dim=-1)
            cyp_value_lookup = torch.argmax(cyp_probs_by_mol, dim=-1, keepdim=True).float() + 1.0
            precedent = self.precedent_logbook.lookup(
                precedent_query,
                cyp_value_lookup[batch_index],
                query_molecule_keys=atom_molecule_keys,
            )
            precedent_brief = (
                precedent["brief"].to(device=multivectors.device, dtype=multivectors.dtype)
                if precedent is not None
                else multivectors.new_zeros((multivectors.size(0), AuditedEpisodeLogbook.brief_dim))
            )
            retrieval = self.memory.lookup(atom_keys, per_atom_graph, query_molecule_keys=atom_molecule_keys)
            best_score = None
            support_margin = None
            concentration = None
            selectivity = None
            if retrieval is None:
                site_prior = torch.full((multivectors.size(0), 1), 0.5, device=multivectors.device, dtype=multivectors.dtype)
                cyp_prior = F.softmax(cyp_logits.detach(), dim=-1)[batch_index]
                confidence = multivectors.new_zeros((multivectors.size(0), 1))
                analogical_gate = multivectors.new_zeros((multivectors.size(0), 1))
                analogical_site_bias = multivectors.new_zeros((multivectors.size(0), 1))
                analogical_cyp_bias = cyp_logits.new_zeros(cyp_logits.shape)
                cyp_prior_by_mol = F.softmax(cyp_logits.detach(), dim=-1)
                continuous_reasoning = self._continuous_reasoning(query_multivectors=multivectors, retrieval=None)
            else:
                site_prior = retrieval["site_prior"]
                cyp_prior = retrieval["cyp_prior"]
                confidence = retrieval["confidence"]
                best_score = retrieval.get("best_score")
                support_margin = retrieval.get("support_margin")
                concentration = retrieval.get("concentration")
                selectivity = retrieval.get("selectivity")
                valid_rows = retrieval.get("valid_rows")
                if valid_rows is not None:
                    valid_rows_f = valid_rows.to(device=multivectors.device, dtype=multivectors.dtype)
                    fallback_site_prior = torch.full_like(site_prior, 0.5)
                    fallback_cyp_prior = F.softmax(cyp_logits.detach(), dim=-1)[batch_index]
                    site_prior = torch.where(valid_rows_f > 0.5, site_prior, fallback_site_prior)
                    cyp_prior = torch.where(valid_rows_f > 0.5, cyp_prior, fallback_cyp_prior)
                    confidence = confidence * valid_rows_f
                continuous_reasoning = self._continuous_reasoning(query_multivectors=multivectors, retrieval=retrieval)
                analogical_gate = retrieval.get("trust_gate")
                if analogical_gate is None:
                    analogical_gate = confidence.clamp(0.0, 1.0)
                neutral_site_prior = torch.full_like(site_prior, 0.5)
                site_prior = analogical_gate * site_prior + (1.0 - analogical_gate) * neutral_site_prior
                analogical_site_input = torch.cat([multivectors, site_prior, cyp_prior, confidence, precedent_brief], dim=-1)
                analogical_site_bias = analogical_gate * (
                    self.analogical_site_head(analogical_site_input) + 0.25 * continuous_reasoning["site_bias"]
                )
                num_molecules = int(cyp_logits.size(0))
                cyp_prior_by_mol = scatter_mean(cyp_prior, batch_index, num_molecules)
                graph_conf = scatter_mean(analogical_gate, batch_index, num_molecules)
                analogical_cyp_bias = graph_conf * self.analogical_cyp_head(
                    torch.cat([graph_embeddings, cyp_prior_by_mol, graph_conf], dim=-1)
                )
                site_prior = analogical_gate * ((0.70 * site_prior) + (0.30 * torch.sigmoid(continuous_reasoning["site_bias"]))) + (1.0 - analogical_gate) * neutral_site_prior
            if retrieval is None:
                analogical_gate = multivectors.new_zeros((multivectors.size(0), 1))
            wave_aux_loss, wave_metrics = self._wave_aux_loss(
                wave_preds=wave_preds,
                xtb_atom_features=xtb_atom_features,
                xtb_atom_valid_mask=xtb_atom_valid_mask,
                xtb_mol_valid=xtb_mol_valid,
                batch_index=batch_index,
                atom_3d_features=atom_3d_features,
            )
            analogical_aux_loss, analogical_metrics = self._analogical_aux_loss(
                site_prior=site_prior,
                site_labels=site_labels,
                site_supervision_mask=site_supervision_mask,
                cyp_prior_by_mol=cyp_prior_by_mol,
                cyp_labels=cyp_labels,
                cyp_supervision_mask=cyp_supervision_mask,
            )
            total_aux_loss = wave_aux_loss + analogical_aux_loss
            if (not self.memory_frozen) and self.training and site_labels is not None and cyp_logits.numel():
                cyp_probs_by_mol = F.softmax(cyp_logits.detach(), dim=-1)
                if cyp_labels is not None:
                    true_cyp = F.one_hot(
                        cyp_labels.long().clamp(min=0),
                        num_classes=self.num_cyp_classes,
                    ).to(dtype=cyp_probs_by_mol.dtype, device=cyp_probs_by_mol.device)
                    if cyp_supervision_mask is not None:
                        cyp_mask = cyp_supervision_mask.float().view(-1, 1).to(device=cyp_probs_by_mol.device, dtype=cyp_probs_by_mol.dtype)
                        cyp_probs_by_mol = cyp_mask * true_cyp + (1.0 - cyp_mask) * cyp_probs_by_mol
                    else:
                        cyp_probs_by_mol = true_cyp
                cyp_probs_atom = cyp_probs_by_mol[batch_index]
                if site_logits is not None:
                    hard_negative_scores = torch.sigmoid(site_logits.detach().view(-1, 1))
                else:
                    hard_negative_scores = site_prior.detach()
                self.memory.update(
                    atom_keys=atom_keys,
                    atom_graph=per_atom_graph,
                    atom_multivectors=multivectors,
                    site_labels=site_labels.float().view(-1, 1),
                    cyp_probs=cyp_probs_atom,
                    molecule_keys=atom_molecule_keys,
                    supervision_mask=site_supervision_mask,
                    hard_negative_scores=hard_negative_scores,
                )
            metrics = {
                **wave_metrics,
                **analogical_metrics,
                "memory_size": float(self.memory.size()),
                "analogical_confidence_mean": float(confidence.detach().mean().item()) if confidence.numel() else 0.0,
                "analogical_best_score_mean": float(best_score.detach().mean().item()) if retrieval is not None and best_score is not None and best_score.numel() else 0.0,
                "analogical_margin_mean": float(support_margin.detach().mean().item()) if retrieval is not None and support_margin is not None and support_margin.numel() else 0.0,
                "analogical_concentration_mean": float(concentration.detach().mean().item()) if retrieval is not None and concentration is not None and concentration.numel() else 0.0,
                "analogical_selectivity_mean": float(selectivity.detach().mean().item()) if retrieval is not None and selectivity is not None and selectivity.numel() else 0.0,
                "analogical_gate_mean": float(analogical_gate.detach().mean().item()) if analogical_gate.numel() else 0.0,
                "wave_charge_mean": float(wave_preds["predicted_charges"].detach().mean().item()) if wave_preds["predicted_charges"].numel() else 0.0,
                "wave_fukui_mean": float(wave_preds["predicted_fukui"].detach().mean().item()) if wave_preds["predicted_fukui"].numel() else 0.0,
                "wave_reliability_mean": float(wave_reliability.detach().mean().item()) if wave_reliability.numel() else 0.0,
                "wave_valid_atom_fraction": float((xtb_atom_valid_mask.float().mean().item()) if xtb_atom_valid_mask is not None and xtb_atom_valid_mask.numel() else 0.0),
                "wave_valid_mol_fraction": float((xtb_mol_valid.float().mean().item()) if xtb_mol_valid is not None and xtb_mol_valid.numel() else 0.0),
                "wave_field_density_mean": float(wave_field["global_density"].detach().mean().item()) if wave_field["global_density"].numel() else 0.0,
                "wave_field_gap_proxy_mean": float(wave_field["global_gap_proxy"].detach().mean().item()) if wave_field["global_gap_proxy"].numel() else 0.0,
                "continuous_reasoning_mean": float(continuous_reasoning["site_bias"].detach().mean().item()) if continuous_reasoning["site_bias"].numel() else 0.0,
                "precedent_logbook_size": float(self.precedent_logbook.size()),
                "precedent_positive_support_mean": float(precedent_brief[:, :1].detach().mean().item()) if precedent_brief.numel() else 0.0,
                "precedent_negative_support_mean": float(precedent_brief[:, 1:2].detach().mean().item()) if precedent_brief.numel() else 0.0,
            }
            return {
                "atom_multivectors": multivectors,
                "graph_embeddings": graph_embeddings,
                "wave_predictions": wave_preds,
                "wave_field": wave_field,
                "wave_site_bias": wave_site_bias,
                "wave_reliability": wave_reliability,
                "analogical_site_prior": site_prior,
                "analogical_cyp_prior": cyp_prior_by_mol,
                "analogical_confidence": confidence,
                "analogical_gate": analogical_gate,
                "analogical_best_score": best_score,
                "analogical_margin": support_margin,
                "analogical_concentration": concentration,
                "analogical_selectivity": selectivity,
                "analogical_site_bias": analogical_site_bias,
                "analogical_cyp_bias": analogical_cyp_bias,
                "continuous_reasoning_features": continuous_reasoning["features"],
                "precedent_brief": precedent_brief,
                "losses": {
                    "total": total_aux_loss,
                    "wave": wave_aux_loss,
                    "analogical": analogical_aux_loss,
                },
                "metrics": metrics,
            }

        @torch.no_grad()
        def ingest_batch(
            self,
            *,
            atom_features: torch.Tensor,
            batch_index: torch.Tensor,
            site_logits: Optional[torch.Tensor] = None,
            cyp_logits: torch.Tensor,
            atom_3d_features: Optional[torch.Tensor] = None,
            xtb_atom_features: Optional[torch.Tensor] = None,
            topology_atom_features: Optional[torch.Tensor] = None,
            site_labels: Optional[torch.Tensor] = None,
            site_supervision_mask: Optional[torch.Tensor] = None,
            cyp_labels: Optional[torch.Tensor] = None,
            cyp_supervision_mask: Optional[torch.Tensor] = None,
            graph_molecule_keys: Optional[torch.Tensor] = None,
        ) -> Dict[str, float]:
            if site_labels is None or cyp_logits.numel() == 0:
                return {"memory_size": float(self.memory.size()), "used": 0.0, "added_atoms": 0.0}
            atom_features = atom_features.float()
            multivectors, _coords = self._build_multivectors(atom_features, atom_3d_features, xtb_atom_features)
            graph_embeddings = self._group_graph_embedding(multivectors, batch_index)
            per_atom_graph = graph_embeddings[batch_index] if graph_embeddings.numel() else multivectors.new_zeros((multivectors.size(0), self.memory.graph_dim))
            if graph_molecule_keys is not None and graph_molecule_keys.numel():
                atom_molecule_keys = graph_molecule_keys.to(device=batch_index.device, dtype=torch.long)[batch_index]
            else:
                atom_molecule_keys = torch.zeros_like(batch_index, dtype=torch.long)
            _wf = self.wave_field(multivectors, _coords, batch_index)
            atom_keys = self._build_atom_keys(
                atom_features=atom_features,
                multivectors=multivectors,
                atom_field_features=_wf["atom_field_features"],
                atom_3d_features=atom_3d_features,
                xtb_atom_features=xtb_atom_features,
                topology_atom_features=topology_atom_features,
            )
            cyp_probs_by_mol = F.softmax(cyp_logits.detach(), dim=-1)
            if cyp_labels is not None:
                true_cyp = F.one_hot(
                    cyp_labels.long().clamp(min=0),
                    num_classes=self.num_cyp_classes,
                ).to(dtype=cyp_probs_by_mol.dtype, device=cyp_probs_by_mol.device)
                if cyp_supervision_mask is not None:
                    cyp_mask = cyp_supervision_mask.float().view(-1, 1).to(device=cyp_probs_by_mol.device, dtype=cyp_probs_by_mol.dtype)
                    cyp_probs_by_mol = cyp_mask * true_cyp + (1.0 - cyp_mask) * cyp_probs_by_mol
                else:
                    cyp_probs_by_mol = true_cyp
            cyp_probs_atom = cyp_probs_by_mol[batch_index]
            supervision = site_supervision_mask.view(-1) > 0.5 if site_supervision_mask is not None else torch.ones_like(batch_index, dtype=torch.bool)
            if site_logits is not None:
                hard_negative_scores = torch.sigmoid(site_logits.detach().view(-1, 1))
            else:
                hard_negative_scores = torch.zeros((site_labels.numel(), 1), device=site_labels.device, dtype=site_labels.dtype)
            selected_atoms = self.memory.update(
                atom_keys=atom_keys,
                atom_graph=per_atom_graph,
                atom_multivectors=multivectors,
                site_labels=site_labels.float().view(-1, 1),
                cyp_probs=cyp_probs_atom,
                molecule_keys=atom_molecule_keys,
                supervision_mask=site_supervision_mask,
                hard_negative_scores=hard_negative_scores,
            )
            used = 1.0 if bool(supervision.any()) else 0.0
            return {"memory_size": float(self.memory.size()), "used": used, "added_atoms": float(selected_atoms)}
else:  # pragma: no cover
    class NexusHybridBridge:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()


__all__ = ["NexusHybridBridge"]
