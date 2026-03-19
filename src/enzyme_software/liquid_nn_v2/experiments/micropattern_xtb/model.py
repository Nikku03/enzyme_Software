from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig
from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.config import MicroPatternXTBConfig
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.reranker import MicroPatternReranker
from enzyme_software.liquid_nn_v2.features.micropattern_features import (
    CHEMISTRY_PRIOR_DIM,
    build_candidate_local_descriptor,
    chemistry_prior_matrix,
    ring_and_aromatic_arrays,
)


@dataclass
class CandidateMeta:
    global_index: int
    local_index: int
    molecule_index: int
    is_positive: bool
    base_logit: float


if TORCH_AVAILABLE:
    def load_base_hybrid_checkpoint(checkpoint_path: str | Path, device) -> HybridLNNModel:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        base_cfg = payload.get("config", {}).get("base_model") or ModelConfig.light_advanced(
            use_manual_engine_priors=True,
            use_3d_branch=True,
            return_intermediate_stats=True,
        ).__dict__
        base_model = LiquidMetabolismNetV2(ModelConfig(**base_cfg))
        model = HybridLNNModel(base_model)
        state_dict = payload.get("model_state_dict") or payload
        model.load_state_dict(state_dict, strict=False)
        return model


    class MicroPatternXTBHybridModel(nn.Module):
        def __init__(self, base_model: HybridLNNModel, config: MicroPatternXTBConfig):
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.reranker = MicroPatternReranker(
                hidden_dim=config.reranker_hidden_dim,
                dropout=config.reranker_dropout,
                scale_init=config.reranker_scale_init,
            )
            if config.freeze_base_model:
                self.set_base_trainable(False)

        def set_base_trainable(self, trainable: bool) -> None:
            for param in self.base_model.parameters():
                param.requires_grad = bool(trainable)

        def _run_base(self, batch: Dict[str, object]) -> Tuple[Dict[str, object], torch.Tensor, torch.Tensor]:
            impl = self.base_model.base_lnn.impl if hasattr(self.base_model, "base_lnn") else self.base_model.impl
            context = torch.enable_grad() if any(param.requires_grad for param in self.base_model.parameters()) else torch.no_grad()
            with context:
                encoded = impl._encode_inputs(batch)
                som_payload = impl.som_branch(
                    encoded["shared_atoms"],
                    encoded["batch"],
                    edge_index=encoded["edge_index"],
                    edge_attr=encoded["edge_attr"],
                    tau_init=encoded["tau_init"],
                    steric_atom=encoded["steric_payload"].get("atom_embedding") if impl.config.use_3d_branch else None,
                )
                cyp_payload = impl.cyp_branch(
                    encoded["shared_atoms"],
                    encoded["batch"],
                    edge_index=encoded["edge_index"],
                    edge_attr=encoded["edge_attr"],
                    tau_init=encoded["tau_init"],
                    group_membership=encoded["group_membership"] if impl.config.use_hierarchical_pooling else None,
                    group_assignments=encoded["group_assignments"],
                    manual_mol=encoded["prior_payload"].get("mol_prior_embedding") if impl.config.use_manual_engine_priors else None,
                    steric_atom=encoded["steric_payload"].get("atom_embedding") if impl.config.use_3d_branch else None,
                    steric_mol=encoded["steric_payload"].get("mol_embedding") if impl.config.use_3d_branch else None,
                    som_summary=som_payload["mol_summary"],
                )
                site_logits, site_residual = impl.site_head(
                    som_payload["atom_features"],
                    prior_logits=encoded["prior_payload"].get("atom_prior_logits") if impl.config.use_manual_engine_priors else None,
                    prior_features=encoded["prior_payload"].get("atom_prior_embedding") if impl.config.use_manual_engine_priors else None,
                )
                cyp_logits, cyp_residual = impl.cyp_head(
                    cyp_payload["mol_features"],
                    prior_logits=encoded["prior_payload"].get("cyp_prior_logits") if impl.config.use_manual_engine_priors else None,
                    prior_features=encoded["prior_payload"].get("mol_prior_embedding") if impl.config.use_manual_engine_priors else None,
                )
                base_outputs = impl._build_common_outputs(
                    encoded,
                    som_payload,
                    cyp_payload,
                    site_logits,
                    cyp_logits,
                    site_residual,
                    cyp_residual,
                    extra={"model_variant": "baseline"},
                )
                if hasattr(self.base_model, "prior_weight_logit"):
                    prior = batch.get("manual_engine_route_prior")
                    if prior is not None:
                        weight = torch.sigmoid(self.base_model.prior_weight_logit)
                        prior = prior.to(device=cyp_logits.device, dtype=cyp_logits.dtype)
                        base_outputs["cyp_logits_base"] = base_outputs["cyp_logits"]
                        from enzyme_software.liquid_nn_v2.features.route_prior import combine_lnn_with_prior

                        base_outputs["cyp_logits"] = combine_lnn_with_prior(
                            base_outputs["cyp_logits"],
                            prior,
                            prior_weight=float(weight.detach().item()),
                        )
            return base_outputs, som_payload["atom_features"], cyp_payload["mol_features"]

        def _candidate_indices(self, batch, base_site_logits) -> Tuple[List[List[int]], List[List[bool]]]:
            top_k = int(self.config.top_k_candidates)
            batch_index = batch["batch"]
            site_labels = batch["site_labels"].squeeze(-1)
            site_mask = batch.get("site_supervision_mask")
            if site_mask is None:
                site_mask = torch.zeros_like(site_labels).unsqueeze(-1)
            site_mask = site_mask.squeeze(-1)
            num_molecules = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
            per_molecule: List[List[int]] = []
            positives: List[List[bool]] = []
            for mol_idx in range(num_molecules):
                atom_idx = torch.nonzero(batch_index == mol_idx, as_tuple=False).view(-1)
                if atom_idx.numel() == 0:
                    per_molecule.append([])
                    positives.append([])
                    continue
                logits = base_site_logits[atom_idx].squeeze(-1)
                k = min(top_k, atom_idx.numel())
                top_local = torch.topk(logits, k=k, dim=0).indices.tolist()
                selected = [int(atom_idx[i].item()) for i in top_local]
                if bool((site_mask[atom_idx] > 0.5).any()) and bool((site_labels[atom_idx] > 0.5).any()):
                    pos_local = torch.nonzero((site_labels[atom_idx] > 0.5) & (site_mask[atom_idx] > 0.5), as_tuple=False).view(-1)
                    best_pos_local = int(pos_local[torch.argmax(logits[pos_local])].item()) if pos_local.numel() else None
                    if best_pos_local is not None:
                        pos_global = int(atom_idx[best_pos_local].item())
                        if pos_global not in selected:
                            if len(selected) >= k and selected:
                                selected[-1] = pos_global
                            else:
                                selected.append(pos_global)
                selected = list(dict.fromkeys(selected))[:top_k]
                selected_positive = [
                    bool(site_labels[idx].item() > 0.5 and site_mask[idx].item() > 0.5)
                    for idx in selected
                ]
                per_molecule.append(selected)
                positives.append(selected_positive)
            return per_molecule, positives

        def _build_reranker_batch(self, batch, atom_embeddings, mol_features, base_site_logits):
            per_molecule, positives = self._candidate_indices(batch, base_site_logits)
            top_k = int(self.config.top_k_candidates)
            radius = int(self.config.micropattern_radius)
            device = atom_embeddings.device
            feature_rows = []
            xtb_rows = []
            candidate_valid = []
            candidate_positive = []
            candidate_meta: List[Optional[CandidateMeta]] = []
            candidate_base_logits: List[float] = []
            per_graph_smiles = batch.get("canonical_smiles") or []
            batch_index = batch["batch"].detach().cpu().numpy()
            edge_index = batch["edge_index"].detach().cpu().numpy()
            edge_attr = batch["edge_attr"].detach().cpu().numpy() if batch.get("edge_attr") is not None else None
            atom_embeddings_np = atom_embeddings.detach().cpu().numpy()
            base_site_np = base_site_logits.detach().cpu().numpy().reshape(-1)
            manual_np = (
                batch["manual_engine_atom_features"].detach().cpu().numpy()
                if batch.get("manual_engine_atom_features") is not None
                else None
            )
            xtb_np = (
                batch["xtb_atom_features"].detach().cpu().numpy()
                if batch.get("xtb_atom_features") is not None
                else None
            )
            num_molecules = len(per_molecule)
            for mol_idx in range(num_molecules):
                mol_atom_idx = np.where(batch_index == mol_idx)[0]
                if mol_atom_idx.size == 0:
                    continue
                local_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(mol_atom_idx.tolist())}
                mol_mask = np.isin(edge_index[0], mol_atom_idx) & np.isin(edge_index[1], mol_atom_idx)
                mol_edge_index = edge_index[:, mol_mask].copy()
                for col in range(mol_edge_index.shape[1]):
                    mol_edge_index[0, col] = local_lookup[int(mol_edge_index[0, col])]
                    mol_edge_index[1, col] = local_lookup[int(mol_edge_index[1, col])]
                mol_edge_attr = edge_attr[mol_mask] if edge_attr is not None and mol_mask.any() else None
                mol_atom_emb = atom_embeddings_np[mol_atom_idx]
                mol_manual = manual_np[mol_atom_idx] if manual_np is not None else None
                mol_xtb = xtb_np[mol_atom_idx] if xtb_np is not None else None
                mol_smiles = per_graph_smiles[mol_idx] if mol_idx < len(per_graph_smiles) else ""
                mol_prior = chemistry_prior_matrix(mol_smiles, mol_atom_idx.size)
                ring_flags, aromatic_flags = ring_and_aromatic_arrays(
                    mol_smiles,
                    mol_atom_idx.size,
                )
                mol_context = mol_features[mol_idx].detach().cpu().numpy() if mol_features.numel() else np.zeros((0,), dtype=np.float32)
                for slot in range(top_k):
                    if slot < len(per_molecule[mol_idx]):
                        global_idx = per_molecule[mol_idx][slot]
                        local_idx = local_lookup[global_idx]
                        local_desc = build_candidate_local_descriptor(
                            edge_index=mol_edge_index,
                            num_atoms=mol_atom_idx.size,
                            center_idx=local_idx,
                            radius=radius,
                            atom_embeddings=mol_atom_emb,
                            manual_features=mol_manual,
                            xtb_features=mol_xtb,
                            prior_features=mol_prior,
                            ring_flags=ring_flags,
                            aromatic_flags=aromatic_flags,
                            edge_attr=mol_edge_attr,
                        )
                        scalar = np.asarray(
                            [
                                float(base_site_np[global_idx]),
                                float(torch.sigmoid(base_site_logits[global_idx]).detach().cpu().item()),
                                float(slot) / max(1.0, float(top_k - 1)),
                            ],
                            dtype=np.float32,
                        )
                        feature_rows.append(np.concatenate([local_desc, mol_context.astype(np.float32), scalar], axis=0))
                        xtb_rows.append(mol_xtb[local_idx] if mol_xtb is not None and local_idx < len(mol_xtb) else np.zeros((6,), dtype=np.float32))
                        candidate_valid.append(True)
                        is_positive = bool(positives[mol_idx][slot]) if slot < len(positives[mol_idx]) else False
                        candidate_positive.append(is_positive)
                        candidate_base_logits.append(float(base_site_np[global_idx]))
                        candidate_meta.append(
                            CandidateMeta(
                                global_index=global_idx,
                                local_index=local_idx,
                                molecule_index=mol_idx,
                                is_positive=is_positive,
                                base_logit=float(base_site_np[global_idx]),
                            )
                        )
                    else:
                        pad_dim = (
                            feature_rows[0].shape[0]
                            if feature_rows
                            else atom_embeddings_np.shape[1] * 3
                            + (manual_np.shape[1] * 3 if manual_np is not None else 0)
                            + (xtb_np.shape[1] * 3 if xtb_np is not None else 0)
                            + CHEMISTRY_PRIOR_DIM * 3
                            + mol_features.shape[1]
                            + 15
                        )
                        feature_rows.append(np.zeros((pad_dim,), dtype=np.float32))
                        xtb_rows.append(np.zeros((xtb_np.shape[1] if xtb_np is not None else 6,), dtype=np.float32))
                        candidate_valid.append(False)
                        candidate_positive.append(False)
                        candidate_base_logits.append(0.0)
                        candidate_meta.append(None)
            features = torch.as_tensor(np.asarray(feature_rows, dtype=np.float32), device=device)
            xtb_tensor = torch.as_tensor(np.asarray(xtb_rows, dtype=np.float32), device=device)
            candidate_valid_tensor = torch.as_tensor(candidate_valid, dtype=torch.bool, device=device).view(-1, top_k)
            candidate_positive_tensor = torch.as_tensor(candidate_positive, dtype=torch.bool, device=device).view(-1, top_k)
            return {
                "features": features,
                "xtb_features": xtb_tensor,
                "candidate_base_logits": torch.as_tensor(candidate_base_logits, device=device, dtype=base_site_logits.dtype).unsqueeze(-1),
                "candidate_valid": candidate_valid_tensor,
                "candidate_positive": candidate_positive_tensor,
                "candidate_meta": candidate_meta,
            }

        def forward(self, batch: Dict[str, object]) -> Dict[str, object]:
            base_outputs, atom_embeddings, mol_features = self._run_base(batch)
            base_site_logits = base_outputs["site_logits"]
            reranker_batch = self._build_reranker_batch(batch, atom_embeddings, mol_features, base_site_logits)
            top_k = int(self.config.top_k_candidates)
            features = reranker_batch["features"]
            xtb_features = reranker_batch["xtb_features"]
            candidate_meta = reranker_batch["candidate_meta"]
            if features.numel() == 0:
                return {
                    "base_outputs": base_outputs,
                    "base_site_logits": base_site_logits,
                    "reranked_site_logits": base_site_logits,
                    "base_candidate_scores": torch.empty((0, top_k), device=base_site_logits.device),
                    "reranked_candidate_scores": torch.empty((0, top_k), device=base_site_logits.device),
                    "candidate_valid": reranker_batch["candidate_valid"],
                    "candidate_positive": reranker_batch["candidate_positive"],
                    "candidate_meta": candidate_meta,
                    "stats": {"xtb_valid_molecules": 0.0, "xtb_valid_atoms": 0.0},
                }

            base_candidate_logits = reranker_batch["candidate_base_logits"]
            refined_candidate_logits, reranker_payload = self.reranker(features, xtb_features, base_candidate_logits)
            reranked_site_logits = base_site_logits.clone()
            for idx, meta in enumerate(candidate_meta):
                if meta is None:
                    continue
                reranked_site_logits[meta.global_index] = refined_candidate_logits[idx]

            candidate_rows = int(reranker_batch["candidate_valid"].shape[0]) if top_k else 0
            base_candidate_scores = base_candidate_logits.view(candidate_rows, top_k)
            reranked_candidate_scores = refined_candidate_logits.view(candidate_rows, top_k)
            xtb_valid_mask = batch.get("xtb_atom_valid_mask")
            xtb_valid_atoms = float(xtb_valid_mask.float().mean().item()) if xtb_valid_mask is not None else 0.0
            xtb_mol_valid = batch.get("xtb_mol_valid")
            xtb_valid_molecules = float(xtb_mol_valid.float().mean().item()) if xtb_mol_valid is not None else 0.0
            return {
                "base_outputs": base_outputs,
                "base_site_logits": base_site_logits,
                "reranked_site_logits": reranked_site_logits,
                "base_candidate_scores": base_candidate_scores,
                "reranked_candidate_scores": reranked_candidate_scores,
                "candidate_valid": reranker_batch["candidate_valid"],
                "candidate_positive": reranker_batch["candidate_positive"],
                "candidate_meta": candidate_meta,
                "stats": {
                    "xtb_valid_molecules": xtb_valid_molecules,
                    "xtb_valid_atoms": xtb_valid_atoms,
                    "reranker_gate_mean": reranker_payload["stats"]["gate_mean"],
                    "reranker_delta_mean": reranker_payload["stats"]["delta_mean"],
                    "reranker_delta_max": reranker_payload["stats"]["delta_max"],
                },
            }
else:  # pragma: no cover
    def load_base_hybrid_checkpoint(*args, **kwargs):
        require_torch()

    class MicroPatternXTBHybridModel:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
