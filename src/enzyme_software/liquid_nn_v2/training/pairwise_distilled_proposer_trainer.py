from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.training.hard_negative_mining import mine_hard_negative_pairs
from enzyme_software.liquid_nn_v2.training.loss import AdaptiveLossV2
from enzyme_software.liquid_nn_v2.training.metrics import compute_site_metrics_v2, compute_sourcewise_recall_at_k
from enzyme_software.liquid_nn_v2.training.pairwise_distillation import (
    build_pairwise_distilled_targets,
    zero_pairwise_distillation_metrics,
)
from enzyme_software.liquid_nn_v2.training.pairwise_probe import apply_candidate_mask_to_site_logits
from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs, move_to_device


if TORCH_AVAILABLE:
    def _graph_sources_from_metadata(metadata: List[dict]) -> List[str]:
        sources = []
        for entry in metadata:
            source = ""
            if isinstance(entry, dict):
                source = str(entry.get("source") or entry.get("data_source") or "").strip().lower()
            sources.append(source or "unknown")
        return sources


    @dataclass
    class PairwiseDistilledProposerTrainer:
        """Train a scalar proposer head from pairwise-teacher soft targets."""

        model: object
        pairwise_head: object
        distilled_head: object
        learning_rate: float = 1.0e-4
        weight_decay: float = 1.0e-4
        max_grad_norm: float = 5.0
        candidate_topk: int = 6
        target_temperature: float = 1.0
        loss_type: str = "kl"
        label_smoothing: float = 0.0
        restrict_to_candidates: bool = True
        use_frozen_backbone: bool = True
        use_frozen_pairwise_head: bool = True
        trainable_proposer_head_only: bool = True
        unfreeze_last_backbone_block: bool = False
        distilled_head_lr_scale: float = 1.0
        backbone_lr_scale: float = 0.1
        enable_supervised_site_loss: bool = False
        supervised_weight: float = 1.0
        distill_weight: float = 0.1
        use_main_site_loss_impl: bool = True
        device: Optional[torch.device] = None

        def __post_init__(self):
            self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if bool(self.trainable_proposer_head_only):
                self.use_frozen_backbone = True
                self.unfreeze_last_backbone_block = False
            self.model.to(self.device)
            self.pairwise_head.to(self.device)
            self.distilled_head.to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.pairwise_head.parameters():
                param.requires_grad = False
            base_impl = getattr(getattr(self.model, "base_lnn", None), "impl", None)
            self.last_backbone_block = getattr(base_impl, "som_branch", None)
            if (not self.use_frozen_backbone) and bool(self.unfreeze_last_backbone_block) and self.last_backbone_block is not None:
                for param in self.last_backbone_block.parameters():
                    param.requires_grad = True
            for param in self.distilled_head.parameters():
                param.requires_grad = True
            if not self.use_frozen_pairwise_head:
                raise NotImplementedError("pairwise_distilled_proposer currently expects a frozen pairwise teacher")
            if bool(self.enable_supervised_site_loss) and not bool(self.use_main_site_loss_impl):
                raise NotImplementedError(
                    "pairwise_distilled_proposer supervised mode currently reuses the mainline site loss implementation"
                )
            self.pairwise_head.eval()
            self.site_loss_wrapper = None
            if bool(self.enable_supervised_site_loss):
                model_config = getattr(self.model, "config", None)
                self.site_loss_wrapper = AdaptiveLossV2(
                    tau_reg_weight=0.0,
                    energy_loss_weight=0.0,
                    deliberation_loss_weight=0.0,
                    site_label_smoothing=float(getattr(model_config, "site_label_smoothing", 0.0)),
                    site_top1_margin_weight=float(getattr(model_config, "site_top1_margin_weight", 0.0)),
                    site_top1_margin_value=float(getattr(model_config, "site_top1_margin_value", 0.5)),
                    site_ranking_weight=float(getattr(model_config, "site_ranking_weight", 0.5)),
                    site_hard_negative_fraction=float(getattr(model_config, "site_hard_negative_fraction", 0.5)),
                    site_top1_margin_topk=int(getattr(model_config, "site_top1_margin_topk", 1)),
                    site_top1_margin_decay=float(getattr(model_config, "site_top1_margin_decay", 1.0)),
                    site_cover_weight=float(getattr(model_config, "site_cover_weight", 0.0)),
                    site_cover_margin=float(getattr(model_config, "site_cover_margin", 0.20)),
                    site_cover_topk=int(getattr(model_config, "site_cover_topk", 5)),
                    site_shortlist_weight=float(getattr(model_config, "site_shortlist_weight", 0.0)),
                    site_shortlist_temperature=float(getattr(model_config, "site_shortlist_temperature", 0.70)),
                    site_shortlist_topk=int(getattr(model_config, "site_shortlist_topk", 5)),
                    site_use_rank_weighted_shortlist=bool(
                        getattr(model_config, "site_use_rank_weighted_shortlist", False)
                    ),
                    site_hard_negative_weight=float(getattr(model_config, "site_hard_negative_weight", 0.0)),
                    site_hard_negative_margin=float(getattr(model_config, "site_hard_negative_margin", 0.20)),
                    site_hard_negative_max_per_true=int(getattr(model_config, "site_hard_negative_max_per_true", 3)),
                    site_use_top_score_hard_neg=bool(getattr(model_config, "site_use_top_score_hard_neg", True)),
                    site_use_graph_local_hard_neg=bool(getattr(model_config, "site_use_graph_local_hard_neg", True)),
                    site_use_3d_local_hard_neg=bool(getattr(model_config, "site_use_3d_local_hard_neg", True)),
                    site_use_rank_weighted_hard_neg=bool(
                        getattr(model_config, "site_use_rank_weighted_hard_neg", False)
                    ),
                ).to(self.device)
                for param in self.site_loss_wrapper.parameters():
                    param.requires_grad = False
                self.site_loss_wrapper.eval()
            param_groups = [
                {
                    "params": [param for param in self.distilled_head.parameters() if param.requires_grad],
                    "lr": float(self.learning_rate) * float(self.distilled_head_lr_scale),
                    "weight_decay": float(self.weight_decay),
                }
            ]
            if self.last_backbone_block is not None:
                backbone_params = [param for param in self.last_backbone_block.parameters() if param.requires_grad]
                if backbone_params:
                    param_groups.append(
                        {
                            "params": backbone_params,
                            "lr": float(self.learning_rate) * float(self.backbone_lr_scale),
                            "weight_decay": float(self.weight_decay),
                        }
                    )
            self.optimizer = torch.optim.AdamW(param_groups)
            self.trainable_module_summary = [
                {
                    "name": "distilled_proposer_head",
                    "lr": float(self.learning_rate) * float(self.distilled_head_lr_scale),
                    "param_count": int(sum(param.numel() for param in self.distilled_head.parameters() if param.requires_grad)),
                }
            ]
            if self.last_backbone_block is not None and any(param.requires_grad for param in self.last_backbone_block.parameters()):
                self.trainable_module_summary.append(
                    {
                        "name": "base_lnn.impl.som_branch",
                        "lr": float(self.learning_rate) * float(self.backbone_lr_scale),
                        "param_count": int(sum(param.numel() for param in self.last_backbone_block.parameters() if param.requires_grad)),
                    }
                )

        def _prepare_batch(self, batch_or_graphs):
            if isinstance(batch_or_graphs, dict):
                return move_to_device(batch_or_graphs, self.device)
            return move_to_device(collate_molecule_graphs(batch_or_graphs), self.device)

        def _candidate_mask(self, batch: Dict[str, object]):
            return batch.get("candidate_train_mask", batch.get("candidate_mask"))

        def _supervision_mask(self, batch: Dict[str, object]):
            return batch.get("effective_site_supervision_mask", batch.get("site_supervision_mask"))

        def _count_supervised_molecules(self, batch: Dict[str, object]) -> float:
            site_mask = self._supervision_mask(batch)
            batch_index = batch["batch"].view(-1)
            if site_mask is None:
                return float(int(batch_index.max().item()) + 1 if batch_index.numel() else 0)
            supervised = site_mask.view(-1) > 0.5
            if not bool(supervised.any()):
                return 0.0
            molecule_ids = torch.unique(batch_index[supervised])
            return float(int(molecule_ids.numel()))

        def _forward_model(self, batch: Dict[str, object]):
            if self.use_frozen_backbone:
                with torch.no_grad():
                    outputs = self.model(batch)
            else:
                outputs = self.model(batch)
            return outputs

        def _distillation_loss(self, student_logits, payload):
            molecules = payload.get("molecules") or []
            metrics = dict(payload.get("metrics") or {})
            if not molecules:
                metrics.update(
                    {
                        "distilled_kl_loss": 0.0,
                        "distilled_pred_argmax_match_target_fraction": 0.0,
                    }
                )
                return student_logits.sum() * 0.0, metrics
            losses = []
            pred_target_hits = []
            for molecule in molecules:
                candidate_indices = molecule["candidate_indices"].long()
                target_distribution = molecule["target_distribution"]
                candidate_logits = student_logits[candidate_indices].view(-1)
                if int(candidate_logits.numel()) <= 1:
                    continue
                if str(self.loss_type or "kl").strip().lower() == "mse":
                    predicted_probs = torch.softmax(candidate_logits, dim=0)
                    loss_value = torch.nn.functional.mse_loss(predicted_probs, target_distribution, reduction="mean")
                else:
                    log_probs = torch.log_softmax(candidate_logits, dim=0)
                    loss_value = torch.nn.functional.kl_div(log_probs, target_distribution, reduction="batchmean")
                losses.append(loss_value)
                pred_target_hits.append(float(int(torch.argmax(candidate_logits).item()) == int(molecule["target_argmax_index"])))
            if not losses:
                metrics.update(
                    {
                        "distilled_kl_loss": 0.0,
                        "distilled_pred_argmax_match_target_fraction": 0.0,
                    }
                )
                return student_logits.sum() * 0.0, metrics
            distilled_loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(distilled_loss)):
                raise FloatingPointError("Non-finite pairwise distilled proposer loss detected")
            metrics["distilled_kl_loss"] = float(distilled_loss.detach().item())
            metrics["distilled_pred_argmax_match_target_fraction"] = float(sum(pred_target_hits) / len(pred_target_hits))
            return distilled_loss, metrics

        def _supervised_site_loss(self, student_logits, batch: Dict[str, object]):
            metrics = {
                "distilled_supervised_loss": 0.0,
                "distilled_supervised_molecule_count": 0.0,
            }
            if not bool(self.enable_supervised_site_loss):
                return student_logits.sum() * 0.0, metrics
            if self.site_loss_wrapper is None:
                raise RuntimeError("Supervised distilled proposer requested without a site loss wrapper")
            site_mask = self._supervision_mask(batch)
            student_site_logits = student_logits.view_as(batch["site_labels"])
            site_loss, _ = self.site_loss_wrapper.site_loss(
                student_site_logits,
                batch["site_labels"],
                batch["batch"],
                supervision_mask=site_mask,
                node_weights=batch.get("node_confidence_weights"),
                graph_weights=batch.get("graph_confidence_weights"),
                candidate_mask=self._candidate_mask(batch),
                edge_index=batch.get("edge_index"),
                atom_coordinates=batch.get("atom_coordinates"),
            )
            if not bool(torch.isfinite(site_loss)):
                raise FloatingPointError("Non-finite supervised distilled proposer loss detected")
            metrics["distilled_supervised_loss"] = float(site_loss.detach().item())
            metrics["distilled_supervised_molecule_count"] = self._count_supervised_molecules(batch)
            return site_loss, metrics

        def _run_batch(self, batch: Dict[str, object]):
            outputs = self._forward_model(batch)
            atom_features = outputs.get("atom_features")
            teacher_logits = outputs.get("site_logits")
            if atom_features is None or teacher_logits is None:
                raise RuntimeError(
                    "pairwise_distilled_proposer requires model outputs['atom_features'] and outputs['site_logits']"
                )
            ranking_mask = self._candidate_mask(batch)
            config = getattr(self.model, "config", None)
            masked_teacher_logits = apply_candidate_mask_to_site_logits(
                teacher_logits.detach(),
                ranking_mask,
                mask_mode=str(getattr(config, "candidate_mask_mode", "hard") or "hard"),
                logit_bias=float(getattr(config, "candidate_mask_logit_bias", 2.0)),
            )
            payload = build_pairwise_distilled_targets(
                atom_embeddings=atom_features,
                teacher_site_logits=masked_teacher_logits,
                site_labels=batch["site_labels"],
                batch_index=batch["batch"],
                pairwise_head=self.pairwise_head,
                supervision_mask=self._supervision_mask(batch),
                candidate_mask=ranking_mask,
                candidate_topk=int(self.candidate_topk),
                temperature=float(self.target_temperature),
                label_smoothing=float(self.label_smoothing),
                restrict_to_candidates=bool(self.restrict_to_candidates),
            )
            student_logits = self.distilled_head(atom_features).view(-1)
            if not bool(torch.isfinite(student_logits).all()):
                raise FloatingPointError("Non-finite distilled proposer logits detected")
            distill_loss, distilled_metrics = self._distillation_loss(student_logits, payload)
            supervised_loss, supervised_metrics = self._supervised_site_loss(student_logits, batch)
            effective_supervised_weight = float(self.supervised_weight) if bool(self.enable_supervised_site_loss) else 0.0
            effective_distill_weight = float(self.distill_weight) if bool(self.enable_supervised_site_loss) else 1.0
            total_loss = (
                effective_supervised_weight * supervised_loss
                + effective_distill_weight * distill_loss
            )
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("Non-finite total distilled proposer loss detected")
            distilled_metrics.update(supervised_metrics)
            distilled_metrics["distilled_total_loss"] = float(total_loss.detach().item())
            distilled_metrics["distilled_supervised_weight"] = float(effective_supervised_weight)
            distilled_metrics["distilled_distill_weight"] = float(effective_distill_weight)
            distilled_metrics["distilled_total_molecule_count"] = float(
                max(
                    float(distilled_metrics.get("distilled_molecule_count", 0.0)),
                    float(supervised_metrics.get("distilled_supervised_molecule_count", 0.0)),
                )
            )
            masked_student_logits = apply_candidate_mask_to_site_logits(
                student_logits,
                ranking_mask,
                mask_mode=str(getattr(config, "candidate_mask_mode", "hard") or "hard"),
                logit_bias=float(getattr(config, "candidate_mask_logit_bias", 2.0)),
            )
            student_scores = torch.sigmoid(masked_student_logits)
            if not bool(torch.isfinite(student_scores).all()):
                raise FloatingPointError("Non-finite distilled proposer scores detected")
            return total_loss, student_scores, distilled_metrics

        def _merge_distilled_epoch_metrics(self, accum: Dict[str, float]) -> Dict[str, float]:
            metrics = zero_pairwise_distillation_metrics()
            metrics["skipped_singleton_candidate_molecules"] = float(accum.get("skipped_singleton_candidate_molecules", 0.0))
            distill_count = float(accum.get("distilled_molecule_count", 0.0))
            supervised_count = float(accum.get("distilled_supervised_molecule_count", 0.0))
            total_count = float(accum.get("distilled_total_molecule_count", 0.0))
            metrics["distilled_molecule_count"] = distill_count
            metrics["distilled_supervised_molecule_count"] = supervised_count
            metrics["distilled_total_molecule_count"] = total_count
            if distill_count > 0.0:
                for key in (
                    "distilled_kl_loss",
                    "distilled_target_entropy_mean",
                    "distilled_target_max_mean",
                    "distilled_target_argmax_match_true_fraction",
                    "distilled_pred_argmax_match_target_fraction",
                    "distilled_target_true_mass_mean",
                    "candidate_count_mean",
                ):
                    metrics[key] = float(accum.get(f"{key}_sum", 0.0)) / distill_count
            if supervised_count > 0.0:
                metrics["distilled_supervised_loss"] = (
                    float(accum.get("distilled_supervised_loss_sum", 0.0)) / supervised_count
                )
            if total_count > 0.0:
                for key in (
                    "distilled_total_loss",
                    "distilled_supervised_weight",
                    "distilled_distill_weight",
                ):
                    metrics[key] = float(accum.get(f"{key}_sum", 0.0)) / total_count
            return metrics

        def _accumulate_distilled_batch_metrics(self, accum: Dict[str, float], batch_metrics: Dict[str, float]) -> None:
            distill_count = float(batch_metrics.get("distilled_molecule_count", 0.0))
            supervised_count = float(batch_metrics.get("distilled_supervised_molecule_count", 0.0))
            total_count = float(batch_metrics.get("distilled_total_molecule_count", 0.0))
            accum["skipped_singleton_candidate_molecules"] = float(
                accum.get("skipped_singleton_candidate_molecules", 0.0)
            ) + float(batch_metrics.get("skipped_singleton_candidate_molecules", 0.0))
            accum["distilled_molecule_count"] = float(accum.get("distilled_molecule_count", 0.0)) + distill_count
            accum["distilled_supervised_molecule_count"] = float(
                accum.get("distilled_supervised_molecule_count", 0.0)
            ) + supervised_count
            accum["distilled_total_molecule_count"] = float(
                accum.get("distilled_total_molecule_count", 0.0)
            ) + total_count
            for key in (
                "distilled_kl_loss",
                "distilled_target_entropy_mean",
                "distilled_target_max_mean",
                "distilled_target_argmax_match_true_fraction",
                "distilled_pred_argmax_match_target_fraction",
                "distilled_target_true_mass_mean",
                "candidate_count_mean",
            ):
                accum[f"{key}_sum"] = float(accum.get(f"{key}_sum", 0.0)) + (float(batch_metrics.get(key, 0.0)) * distill_count)
            for key in ("distilled_supervised_loss",):
                accum[f"{key}_sum"] = float(accum.get(f"{key}_sum", 0.0)) + (float(batch_metrics.get(key, 0.0)) * supervised_count)
            for key in (
                "distilled_total_loss",
                "distilled_supervised_weight",
                "distilled_distill_weight",
            ):
                accum[f"{key}_sum"] = float(accum.get(f"{key}_sum", 0.0)) + (float(batch_metrics.get(key, 0.0)) * total_count)

        def _finalize_epoch_metrics(
            self,
            *,
            site_scores,
            site_labels,
            site_batches,
            site_supervision_masks,
            candidate_masks,
            merged_edge_parts,
            merged_coord_parts,
            graph_sources,
            distilled_accum,
        ) -> Dict[str, object]:
            if not site_scores:
                raise RuntimeError("pairwise_distilled_proposer received zero valid batches")
            merged_site_scores = torch.cat(site_scores, dim=0)
            merged_site_labels = torch.cat(site_labels, dim=0)
            merged_site_batch = torch.cat(site_batches, dim=0)
            merged_site_supervision_mask = torch.cat(site_supervision_masks, dim=0)
            merged_candidate_mask = torch.cat(candidate_masks, dim=0)
            merged_edge_index = (
                torch.cat(merged_edge_parts, dim=1) if merged_edge_parts else torch.zeros((2, 0), dtype=torch.long)
            )
            merged_atom_coordinates = torch.cat(merged_coord_parts, dim=0) if merged_coord_parts else None
            metrics = compute_site_metrics_v2(
                merged_site_scores,
                merged_site_labels,
                merged_site_batch,
                supervision_mask=merged_site_supervision_mask,
                ranking_mask=merged_candidate_mask,
            )
            metrics.update(
                mine_hard_negative_pairs(
                    merged_site_scores,
                    merged_site_labels,
                    merged_site_batch,
                    supervision_mask=merged_site_supervision_mask,
                    candidate_mask=merged_candidate_mask,
                    edge_index=merged_edge_index,
                    atom_coordinates=merged_atom_coordinates,
                    use_top_score=bool(getattr(self.model.config, "site_use_top_score_hard_neg", True)),
                    use_graph_local=bool(getattr(self.model.config, "site_use_graph_local_hard_neg", True)),
                    use_3d_local=bool(getattr(self.model.config, "site_use_3d_local_hard_neg", True)),
                    max_hard_negs_per_true=int(getattr(self.model.config, "site_hard_negative_max_per_true", 3)),
                )["stats"]
            )
            metrics["source_recall_at_6"] = compute_sourcewise_recall_at_k(
                merged_site_scores,
                merged_site_labels,
                merged_site_batch,
                graph_sources,
                k=6,
                supervision_mask=merged_site_supervision_mask,
                ranking_mask=merged_candidate_mask,
            )
            metrics.update(self._merge_distilled_epoch_metrics(distilled_accum))
            return metrics

        def train_loader_epoch(self, loader) -> Dict[str, object]:
            self.model.train(not self.use_frozen_backbone)
            self.pairwise_head.eval()
            self.distilled_head.train()
            distilled_accum: Dict[str, float] = {}
            site_scores = []
            site_labels = []
            site_supervision_masks = []
            candidate_masks = []
            site_batches = []
            merged_edge_parts = []
            merged_coord_parts = []
            graph_sources = []
            graph_offset = 0
            atom_offset = 0
            for raw_batch in loader:
                if raw_batch is None:
                    continue
                batch = self._prepare_batch(raw_batch)
                loss, student_scores, batch_metrics = self._run_batch(batch)
                if float(batch_metrics.get("distilled_total_molecule_count", 0.0)) <= 0.0:
                    self._accumulate_distilled_batch_metrics(distilled_accum, batch_metrics)
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [param for group in self.optimizer.param_groups for param in group["params"]],
                    float(self.max_grad_norm),
                )
                self.optimizer.step()
                self._accumulate_distilled_batch_metrics(distilled_accum, batch_metrics)
                site_scores.append(student_scores.detach().cpu())
                site_labels.append(batch["site_labels"].detach().cpu())
                site_supervision_masks.append(
                    self._supervision_mask(batch).detach().cpu()
                    if self._supervision_mask(batch) is not None
                    else torch.ones_like(batch["site_labels"]).detach().cpu()
                )
                candidate_masks.append(
                    self._candidate_mask(batch).detach().cpu()
                    if self._candidate_mask(batch) is not None
                    else torch.ones_like(batch["site_labels"]).detach().cpu()
                )
                site_batches.append(batch["batch"].detach().cpu() + graph_offset)
                edge_index = batch.get("edge_index")
                if edge_index is not None:
                    merged_edge_parts.append(edge_index.detach().cpu() + atom_offset)
                atom_coordinates = batch.get("atom_coordinates")
                if atom_coordinates is not None:
                    merged_coord_parts.append(atom_coordinates.detach().cpu())
                metadata = list(batch.get("graph_metadata") or [])
                graph_sources.extend(_graph_sources_from_metadata(metadata))
                graph_offset += len(metadata) if metadata else (int(batch["batch"].max().item()) + 1 if batch["batch"].numel() else 0)
                atom_offset += int(batch["site_labels"].shape[0])
            return self._finalize_epoch_metrics(
                site_scores=site_scores,
                site_labels=site_labels,
                site_batches=site_batches,
                site_supervision_masks=site_supervision_masks,
                candidate_masks=candidate_masks,
                merged_edge_parts=merged_edge_parts,
                merged_coord_parts=merged_coord_parts,
                graph_sources=graph_sources,
                distilled_accum=distilled_accum,
            )

        def evaluate_loader(self, loader) -> Dict[str, object]:
            self.model.eval()
            self.pairwise_head.eval()
            self.distilled_head.eval()
            distilled_accum: Dict[str, float] = {}
            site_scores = []
            site_labels = []
            site_supervision_masks = []
            candidate_masks = []
            site_batches = []
            merged_edge_parts = []
            merged_coord_parts = []
            graph_sources = []
            graph_offset = 0
            atom_offset = 0
            with torch.no_grad():
                for raw_batch in loader:
                    if raw_batch is None:
                        continue
                    batch = self._prepare_batch(raw_batch)
                    _loss, student_scores, batch_metrics = self._run_batch(batch)
                    self._accumulate_distilled_batch_metrics(distilled_accum, batch_metrics)
                    site_scores.append(student_scores.detach().cpu())
                    site_labels.append(batch["site_labels"].detach().cpu())
                    site_supervision_masks.append(
                        self._supervision_mask(batch).detach().cpu()
                        if self._supervision_mask(batch) is not None
                        else torch.ones_like(batch["site_labels"]).detach().cpu()
                    )
                    candidate_masks.append(
                        self._candidate_mask(batch).detach().cpu()
                        if self._candidate_mask(batch) is not None
                        else torch.ones_like(batch["site_labels"]).detach().cpu()
                    )
                    site_batches.append(batch["batch"].detach().cpu() + graph_offset)
                    edge_index = batch.get("edge_index")
                    if edge_index is not None:
                        merged_edge_parts.append(edge_index.detach().cpu() + atom_offset)
                    atom_coordinates = batch.get("atom_coordinates")
                    if atom_coordinates is not None:
                        merged_coord_parts.append(atom_coordinates.detach().cpu())
                    metadata = list(batch.get("graph_metadata") or [])
                    graph_sources.extend(_graph_sources_from_metadata(metadata))
                    graph_offset += len(metadata) if metadata else (int(batch["batch"].max().item()) + 1 if batch["batch"].numel() else 0)
                    atom_offset += int(batch["site_labels"].shape[0])
            return self._finalize_epoch_metrics(
                site_scores=site_scores,
                site_labels=site_labels,
                site_batches=site_batches,
                site_supervision_masks=site_supervision_masks,
                candidate_masks=candidate_masks,
                merged_edge_parts=merged_edge_parts,
                merged_coord_parts=merged_coord_parts,
                graph_sources=graph_sources,
                distilled_accum=distilled_accum,
            )
else:  # pragma: no cover
    @dataclass
    class PairwiseDistilledProposerTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
