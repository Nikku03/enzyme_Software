from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.config import TrainingConfig
from enzyme_software.liquid_nn_v2.training.hard_negative_mining import mine_hard_negative_pairs
from enzyme_software.liquid_nn_v2.training.episode_logger import EpisodeLogger
from enzyme_software.liquid_nn_v2.training.loss import AdaptiveLossV2, CandidateRerankLoss
from enzyme_software.liquid_nn_v2.training.metrics import (
    compute_cyp_metrics,
    compute_reranker_metrics,
    compute_site_metrics_v2,
    compute_sourcewise_recall_at_k,
)
from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs, move_to_device


if TORCH_AVAILABLE:
    _DOMAIN_SOURCE_TO_IDX = {
        "drugbank": 0,
        "az120": 1,
        "metxbiodb": 2,
        "attnsom": 3,
        "cyp_dbs_external": 4,
        "other": 5,
    }

    def _normalize_source_name(value: str) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "drugbank_existing": "drugbank",
            "metxbiodb_atom_only": "metxbiodb",
            "cyp_dbs_experimental": "cyp_dbs_external",
            "literature_curation": "literature",
        }
        return aliases.get(text, text)

    def _graph_sources_from_metadata(metadata: List[dict]) -> List[str]:
        sources: List[str] = []
        for meta in metadata:
            source = _normalize_source_name(str((meta or {}).get("site_source") or (meta or {}).get("source") or "unknown"))
            sources.append(source or "unknown")
        return sources

    class _NoOpScheduler:
        def step(self, val_metric: float) -> None:
            return None


    class _ManualAdamW:
        def __init__(
            self,
            params,
            *,
            lr: float,
            weight_decay: float,
            betas,
            eps: float = 1.0e-8,
        ):
            self.param_groups = [{
                "params": [param for param in params if param is not None],
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "betas": tuple(float(value) for value in betas),
                "eps": float(eps),
            }]
            self.state: Dict[int, Dict[str, object]] = {}

        def zero_grad(self, set_to_none: bool = False) -> None:
            for group in self.param_groups:
                for param in group["params"]:
                    grad = param.grad
                    if grad is None:
                        continue
                    if set_to_none:
                        param.grad = None
                    else:
                        grad.zero_()

        @torch.no_grad()
        def step(self) -> None:
            for group in self.param_groups:
                lr = float(group["lr"])
                weight_decay = float(group["weight_decay"])
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                for param in group["params"]:
                    grad = param.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("ManualAdamW does not support sparse gradients")
                    state = self.state.setdefault(
                        id(param),
                        {
                            "step": 0,
                            "exp_avg": torch.zeros_like(param),
                            "exp_avg_sq": torch.zeros_like(param),
                        },
                    )
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] = int(state["step"]) + 1
                    step = int(state["step"])

                    if weight_decay != 0.0:
                        param.mul_(1.0 - (lr * weight_decay))

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - (beta1 ** step)
                    bias_correction2 = 1.0 - (beta2 ** step)
                    denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(eps)
                    step_size = lr / bias_correction1
                    param.addcdiv_(exp_avg, denom, value=-step_size)


    @dataclass
    class Trainer:
        model: object
        config: TrainingConfig = field(default_factory=TrainingConfig)
        device: Optional[torch.device] = None
        cyp_class_weights: Optional[object] = None
        episode_logger: Optional[EpisodeLogger] = None

        def __post_init__(self):
            self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._disable_broken_torch_compile_wrappers()
            self.model.to(self.device)
            self.current_epoch = 0
            weights = None
            model_config = getattr(self.model, "config", None)
            self.site_logit_bias_warmup_epochs = max(0, int(getattr(model_config, "site_logit_bias_warmup_epochs", 8)))
            self.site_logit_bias_target = float(getattr(model_config, "site_logit_bias_target", -0.10))
            self.site_logit_bias_weight = max(0.0, float(getattr(model_config, "site_logit_bias_weight", 0.05)))
            self.site_source_weight_default = float(getattr(model_config, "site_source_weight_default", 1.0))
            self.site_source_weight_map = {
                "drugbank": float(getattr(model_config, "site_source_weight_drugbank", 1.0)),
                "az120": float(getattr(model_config, "site_source_weight_az120", 1.0)),
                "metxbiodb": float(getattr(model_config, "site_source_weight_metxbiodb", 1.0)),
                "attnsom": float(getattr(model_config, "site_source_weight_attnsom", 1.0)),
                "cyp_dbs_external": float(getattr(model_config, "site_source_weight_cyp_dbs_external", 1.0)),
            }
            self.source_site_aux_weight = float(getattr(model_config, "source_site_aux_weight", 0.0))
            if self.cyp_class_weights is not None:
                weights = self.cyp_class_weights.to(self.device) if hasattr(self.cyp_class_weights, "to") else torch.as_tensor(self.cyp_class_weights, dtype=torch.float32, device=self.device)
            self.loss_fn = AdaptiveLossV2(
                cyp_class_weights=weights,
                tau_reg_weight=float(getattr(model_config, "tau_prior_weight", 0.01)),
                energy_loss_weight=float(getattr(model_config, "energy_loss_weight", 0.0)),
                deliberation_loss_weight=float(getattr(model_config, "deliberation_loss_weight", 0.0)),
                energy_margin=float(getattr(model_config, "energy_margin", 0.15)),
                energy_loss_clip=float(getattr(model_config, "energy_loss_clip", 2.0)),
                site_label_smoothing=float(getattr(model_config, "nexus_site_label_smoothing", 0.05)),
                site_top1_margin_weight=float(getattr(model_config, "nexus_top1_margin_weight", 0.5)),
                site_top1_margin_value=float(getattr(model_config, "nexus_top1_margin_value", 0.5)),
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
                site_hard_negative_weight=float(getattr(model_config, "site_hard_negative_weight", 0.0)),
                site_hard_negative_margin=float(getattr(model_config, "site_hard_negative_margin", 0.20)),
                site_hard_negative_max_per_true=int(getattr(model_config, "site_hard_negative_max_per_true", 3)),
                site_use_top_score_hard_neg=bool(getattr(model_config, "site_use_top_score_hard_neg", True)),
                site_use_graph_local_hard_neg=bool(getattr(model_config, "site_use_graph_local_hard_neg", True)),
                site_use_3d_local_hard_neg=bool(getattr(model_config, "site_use_3d_local_hard_neg", True)),
                site_use_rank_weighted_hard_neg=bool(getattr(model_config, "site_use_rank_weighted_hard_neg", False)),
            )
            self.loss_fn.to(self.device)
            self.reranker_loss = CandidateRerankLoss(
                ce_weight=float(getattr(model_config, "topk_reranker_ce_weight", 0.25)),
                margin_weight=float(getattr(model_config, "topk_reranker_margin_weight", 0.25)),
                margin_value=float(getattr(model_config, "topk_reranker_margin_value", 0.30)),
            ).to(self.device)
            trainable_params = self._trainable_parameters()
            force_manual = os.environ.get("HYBRID_FORCE_MANUAL_OPTIMIZER", "").strip().lower() in {
                "1", "true", "yes", "on",
            }
            if force_manual:
                self._optimizer_backend = "manual"
                print("Using ManualAdamW due to HYBRID_FORCE_MANUAL_OPTIMIZER=1.", flush=True)
                self.optimizer = _ManualAdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    betas=self.config.betas,
                )
                self.scheduler = _NoOpScheduler()
            else:
                self._optimizer_backend = "torch"
                try:
                    self.optimizer = torch.optim.AdamW(
                        trainable_params,
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay,
                        betas=self.config.betas,
                    )
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        mode="max",
                        factor=self.config.scheduler_factor,
                        patience=self.config.scheduler_patience,
                        min_lr=1e-6,
                    )
                except AttributeError as exc:
                    if "torch._inductor" not in str(exc) and "custom_graph_pass" not in str(exc):
                        raise
                    print(
                        "Falling back to ManualAdamW due to broken torch compile runtime "
                        f"({exc}).",
                        flush=True,
                    )
                    self._optimizer_backend = "manual"
                    self.optimizer = _ManualAdamW(
                        trainable_params,
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay,
                        betas=self.config.betas,
                    )
                    self.scheduler = _NoOpScheduler()

        def _disable_broken_torch_compile_wrappers(self) -> None:
            # Some Colab torch builds hit a circular-import bug the first time
            # torch._compile's lazy _disable_dynamo wrapper touches torch._inductor.
            # Replace those optimizer wrappers with the original methods.
            try:
                import torch.optim.optimizer as optimizer_mod
            except Exception:
                return
            for name in ("add_param_group", "zero_grad"):
                wrapped = getattr(optimizer_mod.Optimizer, name, None)
                original = getattr(wrapped, "__wrapped__", None)
                if wrapped is None or original is None:
                    continue
                setattr(optimizer_mod.Optimizer, name, original)

        def step_scheduler(self, val_metric: float) -> None:
            self.scheduler.step(val_metric)

        def _trainable_parameters(self) -> List[torch.nn.Parameter]:
            params: List[torch.nn.Parameter] = []
            for module in (self.model, self.loss_fn):
                for param in module.parameters():
                    if param.requires_grad:
                        params.append(param)
            return params

        def _check_tensor_finite(self, name: str, value) -> None:
            if value is None or not hasattr(value, "numel") or value.numel() == 0:
                return
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Non-finite tensor detected: {name}")

        def _run_finite_checks(self, outputs: Dict[str, object], stats: Dict[str, float], *, epoch: Optional[int] = None, batch_idx: Optional[int] = None) -> None:
            model_config = getattr(self.model, "config", None)
            if not bool(getattr(model_config, "enable_finite_checks", False)):
                return
            self._check_tensor_finite("site_logits", outputs.get("site_logits"))
            self._check_tensor_finite("cyp_logits", outputs.get("cyp_logits"))
            energy_outputs = outputs.get("energy_outputs") or {}
            if isinstance(energy_outputs, dict):
                self._check_tensor_finite("energy_node", energy_outputs.get("node_energy"))
                self._check_tensor_finite("energy_mol", energy_outputs.get("mol_energy"))
            tunneling_outputs = outputs.get("tunneling_outputs") or {}
            if isinstance(tunneling_outputs, dict):
                self._check_tensor_finite("tunnel_prob", tunneling_outputs.get("tunnel_prob"))
            hidden_warn = float(getattr(model_config, "instability_hidden_norm_warn", 25.0))
            energy_warn = float(getattr(model_config, "instability_energy_warn", 8.0))
            warnings = []
            atom_hidden = float(stats.get("atom_hidden_norm_mean", 0.0))
            mol_hidden = float(stats.get("mol_hidden_norm_mean", 0.0))
            energy_max = float(stats.get("energy_max", 0.0))
            tunnel_msg_norm = float(stats.get("tunnel_msg_norm_mean", 0.0))
            if atom_hidden > hidden_warn:
                warnings.append(f"atom_hidden_norm_mean={atom_hidden:.2f}")
            if mol_hidden > hidden_warn:
                warnings.append(f"mol_hidden_norm_mean={mol_hidden:.2f}")
            if energy_max > energy_warn:
                warnings.append(f"energy_max={energy_max:.2f}")
            if tunnel_msg_norm > hidden_warn:
                warnings.append(f"tunnel_msg_norm_mean={tunnel_msg_norm:.2f}")
            if warnings:
                prefix = "Instability warning"
                if epoch is not None and batch_idx is not None:
                    prefix += f" epoch={epoch + 1} batch={batch_idx}"
                print(f"{prefix}: {', '.join(warnings)}", flush=True)

        def _clip_gradients(self) -> None:
            params = [param for param in self._trainable_parameters() if param.grad is not None]
            if params:
                torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)

        def _sanitize_trainable_parameters(self) -> int:
            _uninit_cls = getattr(torch.nn.parameter, "UninitializedParameter", None)
            fixed = 0
            for param in self._trainable_parameters():
                if _uninit_cls is not None and isinstance(param, _uninit_cls):
                    continue
                if not bool(torch.isfinite(param).all()):
                    param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
                    param.data.clamp_(-20.0, 20.0)
                    fixed += 1
            if hasattr(self.loss_fn, "log_var_site") and not bool(torch.isfinite(self.loss_fn.log_var_site).all()):
                self.loss_fn.log_var_site.data.zero_()
                fixed += 1
            if hasattr(self.loss_fn, "log_var_cyp") and not bool(torch.isfinite(self.loss_fn.log_var_cyp).all()):
                self.loss_fn.log_var_cyp.data.zero_()
                fixed += 1
            return fixed

        def _has_nonfinite_gradients(self) -> bool:
            _uninit_cls = getattr(torch.nn.parameter, "UninitializedParameter", None)
            for param in self._trainable_parameters():
                if _uninit_cls is not None and isinstance(param, _uninit_cls):
                    continue
                grad = param.grad
                if grad is not None and not bool(torch.isfinite(grad).all()):
                    return True
            return False

        def _nonfinite_gradient_names(self) -> List[str]:
            names: List[str] = []
            _uninit_cls = getattr(torch.nn.parameter, "UninitializedParameter", None)
            for module_name, module in (("model", self.model), ("loss_fn", self.loss_fn)):
                for param_name, param in module.named_parameters():
                    if not param.requires_grad:
                        continue
                    if _uninit_cls is not None and isinstance(param, _uninit_cls):
                        continue
                    grad = param.grad
                    if grad is not None and not bool(torch.isfinite(grad).all()):
                        names.append(f"{module_name}.{param_name}")
            return names

        def _sanitize_gradients(self) -> int:
            """Zero-out NaN/inf gradients in-place; return count of affected params."""
            _uninit_cls = getattr(torch.nn.parameter, "UninitializedParameter", None)
            fixed = 0
            for param in self._trainable_parameters():
                if _uninit_cls is not None and isinstance(param, _uninit_cls):
                    continue
                if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    fixed += 1
            return fixed

        def _prepare_batch(self, batch_or_graphs):
            if isinstance(batch_or_graphs, dict):
                return move_to_device(batch_or_graphs, self.device)
            return move_to_device(collate_molecule_graphs(batch_or_graphs), self.device)

        def _apply_confidence_weights(self, loss, batch):
            return loss

        def _sanitize_aux_logits(self, logits, clamp_value: float = 20.0):
            return torch.nan_to_num(
                logits,
                nan=0.0,
                posinf=clamp_value,
                neginf=-clamp_value,
            ).clamp(min=-clamp_value, max=clamp_value)

        def _graph_source_weights(self, batch: Dict[str, object]):
            metadata = list(batch.get("graph_metadata") or [])
            if not metadata:
                return None
            weights = []
            changed = False
            for meta in metadata:
                source = _normalize_source_name(str((meta or {}).get("site_source") or (meta or {}).get("source") or ""))
                weight = float(self.site_source_weight_map.get(source, self.site_source_weight_default))
                changed = changed or abs(weight - 1.0) > 1.0e-6
                weights.append(weight)
            if not weights or not changed:
                return None
            tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
            return tensor / tensor.mean().clamp_min(1.0e-6)

        def _augment_site_weights(self, batch: Dict[str, object]) -> Dict[str, float]:
            stats: Dict[str, float] = {}
            graph_source_weights = self._graph_source_weights(batch)
            batch_idx = batch.get("batch")
            if graph_source_weights is None or batch_idx is None or not hasattr(batch_idx, "numel") or batch_idx.numel() == 0:
                return stats
            graph_weights = batch.get("graph_confidence_weights")
            if graph_weights is None:
                graph_weights = torch.ones_like(graph_source_weights)
            else:
                graph_weights = graph_weights.to(device=graph_source_weights.device, dtype=graph_source_weights.dtype)
            new_graph_weights = graph_weights * graph_source_weights
            new_graph_weights = new_graph_weights / new_graph_weights.mean().clamp_min(1.0e-6)
            batch["graph_confidence_weights"] = new_graph_weights
            node_weights = batch.get("node_confidence_weights")
            source_to_node = graph_source_weights[batch_idx.long()].unsqueeze(-1)
            if node_weights is None:
                batch["node_confidence_weights"] = new_graph_weights[batch_idx.long()].unsqueeze(-1)
            else:
                node_weights = node_weights.to(device=source_to_node.device, dtype=source_to_node.dtype)
                batch["node_confidence_weights"] = node_weights * source_to_node
            stats["source_weight_mean"] = float(graph_source_weights.mean().detach().item())
            stats["source_weight_max"] = float(graph_source_weights.max().detach().item())
            stats["source_weight_min"] = float(graph_source_weights.min().detach().item())
            return stats

        def _graph_domain_targets(self, batch: Dict[str, object]):
            metadata = list(batch.get("graph_metadata") or [])
            if not metadata:
                return None
            labels = []
            for meta in metadata:
                source = _normalize_source_name(str((meta or {}).get("site_source") or (meta or {}).get("source") or ""))
                labels.append(int(_DOMAIN_SOURCE_TO_IDX.get(source, _DOMAIN_SOURCE_TO_IDX["other"])))
            return torch.tensor(labels, dtype=torch.long, device=self.device)

        def _source_align_loss(self, outputs: Dict[str, object], batch: Dict[str, object]):
            model_config = getattr(self.model, "config", None)
            align_weight = float(getattr(model_config, "source_align_weight", 0.0)) if model_config is not None else 0.0
            if align_weight <= 0.0:
                return None
            mol_features = outputs.get("mol_features")
            metadata = list(batch.get("graph_metadata") or [])
            if mol_features is None or not hasattr(mol_features, "shape") or mol_features.ndim != 2:
                return None
            if len(metadata) != int(mol_features.shape[0]) or len(metadata) < 4:
                return None
            main_sources = {"drugbank", "az120", "metxbiodb", "metxbiodb"}
            hard_sources = {"attnsom", "cyp_dbs_external"}
            source_names = [
                _normalize_source_name(str((meta or {}).get("site_source") or (meta or {}).get("source") or ""))
                for meta in metadata
            ]
            main_idx = [idx for idx, source in enumerate(source_names) if source in main_sources]
            hard_idx = [idx for idx, source in enumerate(source_names) if source in hard_sources]
            if len(main_idx) < 2 or len(hard_idx) < 2:
                return None
            main_x = mol_features[torch.tensor(main_idx, device=mol_features.device, dtype=torch.long)]
            hard_x = mol_features[torch.tensor(hard_idx, device=mol_features.device, dtype=torch.long)]
            main_mean = main_x.mean(dim=0)
            hard_mean = hard_x.mean(dim=0)
            mean_loss = F.mse_loss(hard_mean, main_mean)
            cov_weight = float(getattr(model_config, "source_align_cov_weight", 0.5))
            cov_loss = mean_loss * 0.0
            if cov_weight > 0.0 and int(main_x.shape[0]) > 1 and int(hard_x.shape[0]) > 1:
                main_centered = main_x - main_mean
                hard_centered = hard_x - hard_mean
                main_cov = (main_centered.transpose(0, 1) @ main_centered) / float(max(1, int(main_x.shape[0]) - 1))
                hard_cov = (hard_centered.transpose(0, 1) @ hard_centered) / float(max(1, int(hard_x.shape[0]) - 1))
                cov_loss = F.mse_loss(hard_cov, main_cov)
            total = align_weight * (mean_loss + (cov_weight * cov_loss))
            return total, {
                "source_align_loss": float(total.detach().item()),
                "source_align_mean_loss": float(mean_loss.detach().item()),
                "source_align_cov_loss": float(cov_loss.detach().item()),
                "source_align_weight": float(align_weight),
                "source_align_cov_weight": float(cov_weight),
                "source_align_main_count": float(len(main_idx)),
                "source_align_hard_count": float(len(hard_idx)),
            }

        def _enforce_candidate_mask(self, outputs: Dict[str, object], batch: Dict[str, object]) -> Dict[str, object]:
            candidate_mask = batch.get("candidate_mask")
            site_logits = outputs.get("site_logits")
            if candidate_mask is None or site_logits is None:
                return outputs
            mask = candidate_mask.to(device=site_logits.device, dtype=site_logits.dtype).view_as(site_logits)
            if mask.numel() != site_logits.numel():
                return outputs
            model_config = getattr(self.model, "config", None)
            mask_mode = str(getattr(model_config, "candidate_mask_mode", "hard")).strip().lower()
            if mask_mode == "off":
                masked_logits = site_logits
            elif mask_mode == "soft":
                bias = float(getattr(model_config, "candidate_mask_logit_bias", 2.0))
                masked_logits = site_logits - ((1.0 - mask) * bias)
            else:
                masked_logits = torch.where(mask > 0.5, site_logits, torch.full_like(site_logits, -20.0))
            outputs = dict(outputs)
            outputs["site_logits"] = masked_logits
            outputs["site_scores"] = torch.sigmoid(masked_logits)
            outputs.setdefault("site_logits_base", site_logits)
            outputs["candidate_fraction"] = float(mask.detach().mean().item())
            return outputs

        def _masked_bce_with_logits(self, logits, labels, supervision_mask):
            labels = labels.float().view_as(logits)
            raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            if supervision_mask is None:
                return raw.mean()
            mask = supervision_mask.float().view_as(logits)
            return (raw * mask).sum() / mask.sum().clamp_min(1.0)

        def _resolve_site_engine_supervision(
            self,
            *,
            batch: Dict[str, object],
            outputs: Dict[str, object],
            engine_name: str,
        ):
            site_mask = batch.get("effective_site_supervision_mask", batch.get("site_supervision_mask"))
            node_weights = batch.get("node_confidence_weights")
            graph_weights = batch.get("graph_confidence_weights")
            bridge = outputs.get("nexus_bridge_outputs") or {}

            if engine_name == "wave":
                reliability = bridge.get("wave_reliability")
                if reliability is not None:
                    reliability = reliability.detach().float().view(-1, 1).clamp(0.0, 1.0)
                    positive = (reliability > 0.05).float()
                    site_mask = positive if site_mask is None else site_mask.float().view_as(positive) * positive
                    node_weights = reliability if node_weights is None else node_weights.float().view_as(reliability) * reliability
                    xtb_mol_valid = batch.get("xtb_mol_valid")
                    if xtb_mol_valid is not None and xtb_mol_valid.numel():
                        graph_valid = xtb_mol_valid.detach().float().view(-1).clamp(0.0, 1.0)
                        graph_weights = graph_valid if graph_weights is None else graph_weights.float().view_as(graph_valid) * graph_valid
            elif engine_name == "analogical":
                analogical_gate = bridge.get("analogical_gate")
                if analogical_gate is not None:
                    analogical_gate = analogical_gate.detach().float().view(-1, 1).clamp(0.0, 1.0)
                    positive = (analogical_gate > 0.05).float()
                    site_mask = positive if site_mask is None else site_mask.float().view_as(positive) * positive
                    node_weights = analogical_gate if node_weights is None else node_weights.float().view_as(analogical_gate) * analogical_gate

            return site_mask, node_weights, graph_weights

        def _site_style_vote_loss(self, logits, batch: Dict[str, object], *, site_mask=None, node_weights=None, graph_weights=None):
            site_labels = batch["site_labels"]
            site_mask = batch.get("effective_site_supervision_mask", batch.get("site_supervision_mask")) if site_mask is None else site_mask
            node_weights = batch.get("node_confidence_weights") if node_weights is None else node_weights
            graph_weights = batch.get("graph_confidence_weights") if graph_weights is None else graph_weights
            site_batch = batch["batch"]
            logits = self._sanitize_aux_logits(logits)
            if site_mask is not None and float(site_mask.float().sum().detach().item()) < 1.0e-6:
                return logits.sum() * 0.0
            loss_value, _ = self.loss_fn.site_loss(
                logits,
                site_labels,
                site_batch,
                supervision_mask=site_mask,
                node_weights=node_weights,
                graph_weights=graph_weights,
            )
            return loss_value

        def _override_style_vote_loss(self, logits, batch: Dict[str, object], *, base_logits, site_mask=None, node_weights=None):
            labels = batch["site_labels"].float().view_as(logits)
            site_mask = batch.get("effective_site_supervision_mask", batch.get("site_supervision_mask")) if site_mask is None else site_mask
            node_weights = batch.get("node_confidence_weights") if node_weights is None else node_weights
            safe_logits = self._sanitize_aux_logits(logits)
            base_probs = torch.sigmoid(self._sanitize_aux_logits(base_logits.detach())).view_as(safe_logits)
            target = (labels - base_probs).clamp(-1.0, 1.0)
            pred = torch.tanh(safe_logits)
            raw = F.smooth_l1_loss(pred, target, reduction="none")
            if node_weights is not None:
                node_weights = node_weights.float().view_as(raw).to(device=raw.device, dtype=raw.dtype)
                raw = raw * node_weights
            if site_mask is None:
                return raw.mean()
            mask = site_mask.float().view_as(raw).to(device=raw.device, dtype=raw.dtype)
            if float(mask.sum().detach().item()) < 1.0e-6:
                return raw.sum() * 0.0
            denom = (mask * (node_weights if node_weights is not None else 1.0)).sum().clamp_min(1.0e-6) if node_weights is not None else mask.sum().clamp_min(1.0)
            return (raw * mask).sum() / denom

        def _source_site_aux_loss(self, outputs: Dict[str, object], batch: Dict[str, object]):
            if self.source_site_aux_weight <= 0.0:
                return None
            source_site_logits = outputs.get("source_site_logits")
            if not isinstance(source_site_logits, dict) or not source_site_logits:
                return None
            metadata = list(batch.get("graph_metadata") or [])
            batch_index = batch.get("batch")
            if not metadata or batch_index is None:
                return None
            site_mask = batch.get("effective_site_supervision_mask", batch.get("site_supervision_mask"))
            node_weights = batch.get("node_confidence_weights")
            graph_weights = batch.get("graph_confidence_weights")
            per_source_losses = []
            used_sources = []
            for source_name, logits in source_site_logits.items():
                mol_indices = [
                    idx for idx, meta in enumerate(metadata)
                    if _normalize_source_name(str((meta or {}).get("site_source") or (meta or {}).get("source") or "")) == str(source_name)
                ]
                if not mol_indices:
                    continue
                atom_mask = torch.zeros_like(batch_index, dtype=torch.float32, device=batch_index.device)
                for mol_idx in mol_indices:
                    atom_mask = torch.maximum(atom_mask, (batch_index == int(mol_idx)).float())
                atom_mask = atom_mask.view(-1, 1)
                effective_mask = atom_mask if site_mask is None else atom_mask * site_mask.float().view_as(atom_mask)
                if float(effective_mask.sum().detach().item()) < 1.0e-6:
                    continue
                loss_value, _ = self.loss_fn.site_loss(
                    self._sanitize_aux_logits(logits),
                    batch["site_labels"],
                    batch_index,
                    supervision_mask=effective_mask,
                    node_weights=node_weights,
                    graph_weights=graph_weights,
                )
                per_source_losses.append(loss_value)
                used_sources.append((str(source_name), len(mol_indices)))
            if not per_source_losses:
                return None
            total = torch.stack(per_source_losses).mean()
            stats = {
                "source_site_aux_loss": float(total.detach().item()),
                "source_site_aux_weight": float(self.source_site_aux_weight),
                "source_site_aux_sources": float(len(used_sources)),
            }
            for source_name, count in used_sources:
                stats[f"source_site_aux_count_{source_name}"] = float(count)
            return (self.source_site_aux_weight * total), stats

        def compute_loss(self, batch: Dict[str, object], outputs: Dict[str, object]):
            site_mask = batch.get("site_supervision_mask")
            candidate_train_mask = batch.get("candidate_train_mask")
            model_config = getattr(self.model, "config", None)
            mask_mode = str(getattr(model_config, "candidate_mask_mode", "hard")).strip().lower()
            if candidate_train_mask is not None and mask_mode == "hard":
                candidate_train_mask = candidate_train_mask.float()
                site_mask = candidate_train_mask if site_mask is None else site_mask.float().view_as(candidate_train_mask) * candidate_train_mask
            batch["effective_site_supervision_mask"] = site_mask
            source_weight_stats = self._augment_site_weights(batch)
            # Fix 5: inject size-aware per-graph/per-atom weights when none are set.
            # Large molecules get proportionally more gradient signal so the model
            # learns to rank sites in 26-40 and 41+ atom molecules.
            if batch.get("graph_confidence_weights") is None and "batch" in batch:
                batch_idx = batch["batch"]
                if batch_idx.numel() > 0:
                    num_mol = int(batch_idx.max().item()) + 1
                    mol_counts = torch.zeros(num_mol, dtype=torch.float32, device=batch_idx.device)
                    mol_counts.scatter_add_(0, batch_idx.long(), torch.ones(batch_idx.shape[0], dtype=torch.float32, device=batch_idx.device))
                    # weight = log(n_atoms) / log(16), clamped ≥ 1; normalised so mean = 1
                    raw_w = (torch.log(mol_counts.clamp(min=1.0)) / float(np.log(16.0))).clamp(min=1.0)
                    gcw = raw_w / raw_w.mean().clamp(min=1.0e-6)
                    batch["graph_confidence_weights"] = gcw
                    if batch.get("node_confidence_weights") is None:
                        batch["node_confidence_weights"] = gcw[batch_idx.long()].unsqueeze(-1)
            cyp_supervision_mask = batch.get("cyp_supervision_mask")
            if bool(getattr(model_config, "disable_cyp_task", False)):
                if cyp_supervision_mask is None:
                    cyp_supervision_mask = torch.zeros_like(batch["cyp_labels"], dtype=torch.float32, device=batch["cyp_labels"].device)
                else:
                    cyp_supervision_mask = torch.zeros_like(cyp_supervision_mask, dtype=torch.float32)
            loss, stats = self.loss_fn(
                outputs["site_logits"],
                outputs["cyp_logits"],
                batch["site_labels"],
                batch["cyp_labels"],
                batch["batch"],
                site_mask,
                cyp_supervision_mask,
                batch.get("graph_confidence_weights"),
                batch.get("node_confidence_weights"),
                outputs.get("tau_history"),
                batch.get("tau_init"),
                outputs.get("energy_outputs"),
                outputs.get("deliberation_outputs"),
                candidate_mask=batch.get("candidate_train_mask", batch.get("candidate_mask")),
                edge_index=batch.get("edge_index"),
                atom_coordinates=batch.get("atom_coordinates"),
            )
            stats = dict(stats)
            stats.update(source_weight_stats)
            if (
                self.site_logit_bias_weight > 0.0
                and self.current_epoch < self.site_logit_bias_warmup_epochs
            ):
                bias_logits = outputs.get("site_logits_base", outputs["site_logits"])
                bias_mask = site_mask
                if bias_mask is not None and float(bias_mask.float().sum().detach().item()) > 0.0:
                    flat_logits = bias_logits.view(-1)
                    flat_mask = bias_mask.view(-1) > 0.5
                    masked_mean = flat_logits[flat_mask].mean()
                    bias_excess = torch.relu(masked_mean - self.site_logit_bias_target)
                    logit_bias_loss = bias_excess.square()
                    loss = loss + (self.site_logit_bias_weight * logit_bias_loss)
                    stats["site_logit_bias_loss"] = float(logit_bias_loss.detach().item())
                    stats["site_logit_bias_mean"] = float(masked_mean.detach().item())
                    stats["site_logit_bias_target"] = float(self.site_logit_bias_target)
                    stats["site_logit_bias_weight"] = float(self.site_logit_bias_weight)
            bridge_losses = outputs.get("nexus_bridge_losses") or {}
            if isinstance(bridge_losses, dict):
                bridge_total = bridge_losses.get("total")
                if bridge_total is not None:
                    loss = loss + bridge_total
                    stats["nexus_bridge_loss"] = float(bridge_total.detach().item())
                for key, value in bridge_losses.items():
                    if key == "total" or value is None:
                        continue
                    if hasattr(value, "detach"):
                        stats[f"nexus_{key}_loss"] = float(value.detach().item())
            bridge = outputs.get("nexus_bridge_outputs") or {}
            if isinstance(bridge, dict) and bridge:
                wave_sideinfo_weight = float(getattr(model_config, "nexus_wave_sideinfo_aux_weight", 0.0))
                if wave_sideinfo_weight > 0.0 and bridge.get("wave_site_bias") is not None:
                    aux_site_mask, aux_node_weights, _aux_graph_weights = self._resolve_site_engine_supervision(
                        batch=batch,
                        outputs=outputs,
                        engine_name="wave",
                    )
                    wave_sideinfo_loss = self._override_style_vote_loss(
                        bridge["wave_site_bias"],
                        batch,
                        base_logits=outputs.get("site_logits_base", outputs["site_logits"]),
                        site_mask=aux_site_mask,
                        node_weights=aux_node_weights,
                    )
                    loss = loss + (wave_sideinfo_weight * wave_sideinfo_loss)
                    stats["nexus_wave_sideinfo_loss"] = float(wave_sideinfo_loss.detach().item())
                    stats["nexus_wave_sideinfo_weight"] = float(wave_sideinfo_weight)
                analogical_sideinfo_weight = float(getattr(model_config, "nexus_analogical_sideinfo_aux_weight", 0.0))
                if analogical_sideinfo_weight > 0.0 and bridge.get("analogical_site_bias") is not None:
                    aux_site_mask, aux_node_weights, _aux_graph_weights = self._resolve_site_engine_supervision(
                        batch=batch,
                        outputs=outputs,
                        engine_name="analogical",
                    )
                    analogical_sideinfo_loss = self._override_style_vote_loss(
                        bridge["analogical_site_bias"],
                        batch,
                        base_logits=outputs.get("site_logits_base", outputs["site_logits"]),
                        site_mask=aux_site_mask,
                        node_weights=aux_node_weights,
                    )
                    loss = loss + (analogical_sideinfo_weight * analogical_sideinfo_loss)
                    stats["nexus_analogical_sideinfo_loss"] = float(analogical_sideinfo_loss.detach().item())
                    stats["nexus_analogical_sideinfo_weight"] = float(analogical_sideinfo_weight)
            vote_heads = outputs.get("site_vote_heads") or {}
            model_config = getattr(self.model, "config", None)
            if isinstance(vote_heads, dict) and vote_heads:
                site_labels = batch["site_labels"]
                site_mask = batch.get("site_supervision_mask")
                aux_specs = [
                    ("lnn", "lnn_vote", float(getattr(model_config, "nexus_lnn_vote_aux_weight", 0.0)), "site"),
                    ("wave", "wave_vote", float(getattr(model_config, "nexus_wave_vote_aux_weight", 0.0)), "override"),
                    ("analogical", "analogical_vote", float(getattr(model_config, "nexus_analogical_vote_aux_weight", 0.0)), "override"),
                ]
                for name, key, weight, style in aux_specs:
                    tensor = vote_heads.get(key)
                    if tensor is None or weight <= 0.0:
                        continue
                    aux_site_mask, aux_node_weights, aux_graph_weights = self._resolve_site_engine_supervision(
                        batch=batch,
                        outputs=outputs,
                        engine_name=name,
                    )
                    if style == "override":
                        aux_loss = self._override_style_vote_loss(
                            tensor,
                            batch,
                            base_logits=outputs.get("site_logits_base", outputs["site_logits"]),
                            site_mask=aux_site_mask,
                            node_weights=aux_node_weights,
                        )
                    else:
                        aux_loss = self._site_style_vote_loss(
                            tensor,
                            batch,
                            site_mask=aux_site_mask,
                            node_weights=aux_node_weights,
                            graph_weights=aux_graph_weights,
                        )
                    loss = loss + (weight * aux_loss)
                    stats[f"nexus_{name}_vote_loss"] = float(aux_loss.detach().item())
                    stats[f"nexus_{name}_vote_weight"] = float(weight)
                    stats[f"nexus_{name}_vote_style"] = 1.0 if style == "override" else 0.0
                    if aux_site_mask is not None:
                        stats[f"nexus_{name}_vote_supervision_fraction"] = float(
                            aux_site_mask.float().mean().detach().item()
                        )
                final_probs = torch.sigmoid(outputs["site_logits"]).detach()
                consistency_specs = [
                    ("wave", "wave_vote", float(getattr(model_config, "nexus_wave_vote_consistency_weight", 0.0))),
                    ("analogical", "analogical_vote", float(getattr(model_config, "nexus_analogical_vote_consistency_weight", 0.0))),
                ]
                for name, key, weight in consistency_specs:
                    tensor = vote_heads.get(key)
                    if tensor is None or weight <= 0.0:
                        continue
                    aux_site_mask, _aux_node_weights, _aux_graph_weights = self._resolve_site_engine_supervision(
                        batch=batch,
                        outputs=outputs,
                        engine_name=name,
                    )
                    safe_tensor = self._sanitize_aux_logits(tensor)
                    raw = F.binary_cross_entropy_with_logits(safe_tensor, final_probs, reduction="none")
                    if aux_site_mask is not None:
                        mask = aux_site_mask.float().view_as(raw)
                        if float(mask.sum().detach().item()) < 1.0e-6:
                            continue
                        cons_loss = (raw * mask).sum() / mask.sum().clamp_min(1.0)
                    else:
                        cons_loss = raw.mean()
                    loss = loss + (weight * cons_loss)
                    stats[f"nexus_{name}_vote_consistency_loss"] = float(cons_loss.detach().item())
                    stats[f"nexus_{name}_vote_consistency_weight"] = float(weight)
                board_weights = vote_heads.get("board_weights")
                board_entropy_weight = float(getattr(model_config, "nexus_board_entropy_weight", 0.0))
                if board_weights is not None and board_entropy_weight > 0.0:
                    probs = board_weights.clamp_min(1.0e-6)
                    uniform_kl = (probs * (torch.log(probs) - float(np.log(1.0 / probs.size(-1))))).sum(dim=-1).mean()
                    loss = loss + (board_entropy_weight * uniform_kl)
                    stats["nexus_board_uniform_kl"] = float(uniform_kl.detach().item())
                    stats["nexus_board_entropy_weight"] = float(board_entropy_weight)
            domain_logits = outputs.get("domain_logits")
            domain_weight = float(getattr(model_config, "domain_adv_weight", 0.0)) if model_config is not None else 0.0
            if domain_logits is not None and domain_weight > 0.0:
                domain_targets = self._graph_domain_targets(batch)
                if domain_targets is not None and int(domain_targets.numel()) == int(domain_logits.shape[0]) and int(domain_targets.numel()) > 1:
                    domain_loss = F.cross_entropy(domain_logits, domain_targets)
                    loss = loss + (domain_weight * domain_loss)
                    domain_pred = torch.argmax(domain_logits.detach(), dim=-1)
                    domain_acc = (domain_pred == domain_targets).float().mean()
                    stats["domain_adv_loss"] = float(domain_loss.detach().item())
                    stats["domain_adv_weight"] = float(domain_weight)
                    stats["domain_adv_acc"] = float(domain_acc.detach().item())
            source_align = self._source_align_loss(outputs, batch)
            if source_align is not None:
                align_loss, align_stats = source_align
                loss = loss + align_loss
                stats.update(align_stats)
            source_site_aux = self._source_site_aux_loss(outputs, batch)
            if source_site_aux is not None:
                aux_loss, aux_stats = source_site_aux
                loss = loss + aux_loss
                stats.update(aux_stats)
            reranker_outputs = outputs.get("topk_reranker_outputs") or {}
            proposal_mask = reranker_outputs.get("selected_mask") if isinstance(reranker_outputs, dict) else None
            if proposal_mask is not None and (
                getattr(self.reranker_loss, "ce_weight", 0.0) > 0.0
                or getattr(self.reranker_loss, "margin_weight", 0.0) > 0.0
            ):
                rerank_loss, rerank_stats = self.reranker_loss(
                    outputs["site_logits"],
                    batch["site_labels"],
                    batch["batch"],
                    proposal_mask,
                    supervision_mask=site_mask,
                    graph_weights=batch.get("graph_confidence_weights"),
                )
                loss = loss + rerank_loss
                stats.update(rerank_stats)
                raw_delta = reranker_outputs.get("raw_delta")
                applied_delta = reranker_outputs.get("applied_delta")
                labels = batch["site_labels"].view(-1)
                prop_mask = proposal_mask.view(-1) > 0.5
                if raw_delta is not None:
                    raw_delta = raw_delta.view(-1)
                    true_mask = prop_mask & (labels > 0.5)
                    false_mask = prop_mask & ~(labels > 0.5)
                    if bool(true_mask.any()):
                        stats["candidate_rerank_raw_true_mean"] = float(raw_delta[true_mask].detach().mean().item())
                    if bool(false_mask.any()):
                        stats["candidate_rerank_raw_false_mean"] = float(raw_delta[false_mask].detach().mean().item())
                if applied_delta is not None:
                    applied_delta = applied_delta.view(-1)
                    true_mask = prop_mask & (labels > 0.5)
                    false_mask = prop_mask & ~(labels > 0.5)
                    if bool(true_mask.any()):
                        stats["candidate_rerank_applied_true_mean"] = float(applied_delta[true_mask].detach().mean().item())
                    if bool(false_mask.any()):
                        stats["candidate_rerank_applied_false_mean"] = float(applied_delta[false_mask].detach().mean().item())
            stats.update(self._collect_output_stats(outputs))
            weighted_loss = self._apply_confidence_weights(loss, batch)
            if weighted_loss is not loss:
                stats["confidence_scale"] = float((weighted_loss / loss).detach().item()) if loss.detach().abs().item() > 1.0e-12 else 1.0
                stats["total_loss"] = float(weighted_loss.item())
            return weighted_loss, stats

        def _collect_output_stats(self, outputs: Dict[str, object]) -> Dict[str, float]:
            metrics: Dict[str, float] = {}
            energy_outputs = outputs.get("energy_outputs") or {}
            tunneling_outputs = outputs.get("tunneling_outputs") or {}
            phase_outputs = outputs.get("phase_outputs") or {}
            deliberation_outputs = outputs.get("deliberation_outputs") or {}
            tau_stats = outputs.get("tau_stats") or {}
            diagnostics = outputs.get("diagnostics") or {}

            node_energy = energy_outputs.get("node_energy") if isinstance(energy_outputs, dict) else None
            if node_energy is not None:
                metrics["energy_mean"] = float(node_energy.detach().mean().item())
                metrics["energy_min"] = float(node_energy.detach().min().item())
                metrics["energy_max"] = float(node_energy.detach().max().item())
            mol_energy = energy_outputs.get("mol_energy") if isinstance(energy_outputs, dict) else None
            if mol_energy is not None and hasattr(mol_energy, "numel") and mol_energy.numel():
                metrics["mol_energy_mean"] = float(mol_energy.detach().mean().item())
                metrics["mol_energy_max"] = float(mol_energy.detach().max().item())
            tunnel_prob = tunneling_outputs.get("tunnel_prob") if isinstance(tunneling_outputs, dict) else None
            if tunnel_prob is not None:
                metrics["tunnel_prob_mean"] = float(tunnel_prob.detach().mean().item())
                metrics["tunnel_prob_max"] = float(tunnel_prob.detach().max().item())
            tunnel_stats = diagnostics.get("graph_tunneling") if isinstance(diagnostics, dict) else None
            if isinstance(tunnel_stats, dict):
                metrics["tunneling_edge_count"] = float(tunnel_stats.get("tunneling_edge_count", 0.0))
                metrics["tunnel_msg_norm_mean"] = float(tunnel_stats.get("tunnel_msg_norm_mean", 0.0))
                metrics["tunnel_gate_mean"] = float(tunnel_stats.get("tunnel_gate_mean", 0.0))
            hybrid_stats = diagnostics.get("hybrid_selective") if isinstance(diagnostics, dict) else None
            if isinstance(hybrid_stats, dict):
                metrics["tunnel_bias_mean"] = float(hybrid_stats.get("tunnel_bias_mean", 0.0))
                metrics["tunnel_bias_max"] = float(hybrid_stats.get("tunnel_bias_max", 0.0))
                metrics["refine_gate_mean"] = float(hybrid_stats.get("refine_gate_mean", 0.0))
                metrics["refine_delta_mean"] = float(hybrid_stats.get("refine_delta_mean", 0.0))
                metrics["refine_delta_max"] = float(hybrid_stats.get("refine_delta_max", 0.0))
            phase_stats = phase_outputs.get("stats") if isinstance(phase_outputs, dict) else None
            if phase_stats:
                metrics["phase_mean"] = float(phase_stats.get("phase_mean", 0.0))
                metrics["phase_var"] = float(phase_stats.get("phase_var", 0.0))
            deliberation_stats = deliberation_outputs.get("stats") if isinstance(deliberation_outputs, dict) else None
            if deliberation_stats:
                metrics["deliberation_steps"] = float(deliberation_stats.get("num_steps", 0.0))
                metrics["critic_mean"] = float(deliberation_stats.get("critic_mean", 0.0))
                metrics["atom_gate_mean"] = float(deliberation_stats.get("atom_gate_mean", 0.0))
                metrics["mol_gate_mean"] = float(deliberation_stats.get("mol_gate_mean", 0.0))
                metrics["atom_hidden_norm_mean"] = float(deliberation_stats.get("atom_hidden_norm_mean", 0.0))
                metrics["mol_hidden_norm_mean"] = float(deliberation_stats.get("mol_hidden_norm_mean", 0.0))
            if isinstance(tau_stats, dict) and tau_stats.get("shared"):
                final_shared = tau_stats["shared"][-1] if tau_stats["shared"] else {}
                if final_shared:
                    metrics["tau_mean"] = float(final_shared.get("mean", 0.0))
                    metrics["tau_std"] = float(final_shared.get("std", 0.0))
            if isinstance(diagnostics, dict):
                physics_stats = diagnostics.get("physics_residual")
                if isinstance(physics_stats, dict):
                    atom_stats = physics_stats.get("atom")
                    if isinstance(atom_stats, dict):
                        metrics["physics_gate_mean"] = float(atom_stats.get("gate_mean", 0.0))
                nexus_stats = diagnostics.get("nexus_bridge")
                if isinstance(nexus_stats, dict):
                    for key, value in nexus_stats.items():
                        try:
                            metrics[f"nexus_{key}"] = float(value)
                        except Exception:
                            continue
                reranker_stats = diagnostics.get("topk_reranker")
                if isinstance(reranker_stats, dict):
                    for key, value in reranker_stats.items():
                        try:
                            metrics[f"topk_reranker_{key}"] = float(value)
                        except Exception:
                            continue
                hidden_norms = diagnostics.get("hidden_norms")
                if isinstance(hidden_norms, dict):
                    metrics["som_hidden_norm_mean"] = float(hidden_norms.get("som_features_mean", 0.0))
                    metrics["mol_feature_norm_mean"] = float(hidden_norms.get("mol_features_mean", 0.0))
                    metrics["final_atom_norm_mean"] = float(hidden_norms.get("final_atom_mean", 0.0))
                    metrics["final_mol_norm_mean"] = float(hidden_norms.get("final_mol_mean", 0.0))
            return metrics

        def train_epoch(self, graphs: Iterable) -> Dict[str, float]:
            self.model.train()
            batch = self._prepare_batch(graphs)
            self._sanitize_trainable_parameters()
            outputs = self.model(batch)
            outputs = self._enforce_candidate_mask(outputs, batch)
            loss, stats = self.compute_loss(batch, outputs)
            self._run_finite_checks(outputs, stats)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss detected in train_epoch: stats={stats}")
            self.optimizer.zero_grad()
            loss.backward()
            if self._has_nonfinite_gradients():
                bad_names = self._nonfinite_gradient_names()
                fixed_grads = self._sanitize_gradients()
                fixed_params = self._sanitize_trainable_parameters()
                self.optimizer.zero_grad(set_to_none=True)
                sample = ", ".join(bad_names[:3]) if bad_names else "unknown"
                print(
                    "train_epoch: skipped optimizer step after zeroing NaN/inf gradients "
                    f"in {fixed_grads} grad(s) (sanitized_params={fixed_params}; examples={sample}).",
                    flush=True,
                )
                return stats
            self._clip_gradients()
            self.optimizer.step()
            return stats

        def train_loader_epoch(self, loader) -> Dict[str, float]:
            self.model.train()
            history = []
            self.current_epoch = int(getattr(loader, "_current_epoch", 0) or 0)
            for batch_idx, raw_batch in enumerate(loader):
                if raw_batch is None:
                    continue
                self._sanitize_trainable_parameters()
                batch = self._prepare_batch(raw_batch)
                outputs = self.model(batch)
                outputs = self._enforce_candidate_mask(outputs, batch)
                loss, stats = self.compute_loss(batch, outputs)
                self._run_finite_checks(outputs, stats, batch_idx=batch_idx)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite loss detected during training at batch {batch_idx}: stats={stats}"
                    )
                if self.episode_logger is not None:
                    self.episode_logger.log_step(
                        split="train",
                        epoch=getattr(loader, "_current_epoch", None),
                        batch_idx=batch_idx,
                        batch=batch,
                        stats=stats,
                        outputs=outputs,
                    )
                    self.episode_logger.log_examples(
                        split="train",
                        epoch=getattr(loader, "_current_epoch", None),
                        batch_idx=batch_idx,
                        batch=batch,
                        outputs=outputs,
                        stats=stats,
                    )
                self.optimizer.zero_grad()
                loss.backward()
                if self._has_nonfinite_gradients():
                    bad_names = self._nonfinite_gradient_names()
                    fixed_grads = self._sanitize_gradients()
                    fixed_params = self._sanitize_trainable_parameters()
                    self.optimizer.zero_grad(set_to_none=True)
                    sample = ", ".join(bad_names[:3]) if bad_names else "unknown"
                    print(
                        f"Batch {batch_idx}: skipped optimizer step after zeroing NaN/inf in {fixed_grads} grad(s) "
                        f"(sanitized_params={fixed_params}; examples={sample}).",
                        flush=True,
                    )
                    history.append(stats)
                    continue
                self._clip_gradients()
                self.optimizer.step()
                self._sanitize_trainable_parameters()
                history.append(stats)
            if not history:
                raise RuntimeError(
                    "train_loader_epoch received zero valid batches. "
                    "This usually means the dataset loader dropped every molecule."
                )
            keys = sorted({key for stats in history for key in stats.keys()})
            return {
                key: float(sum(float(stats.get(key, 0.0)) for stats in history) / len(history))
                for key in keys
            }

        def evaluate(self, graphs: Iterable) -> Dict[str, object]:
            self.model.eval()
            with torch.no_grad():
                batch = self._prepare_batch(graphs)
                outputs = self.model(batch)
                outputs = self._enforce_candidate_mask(outputs, batch)
            site_metrics = compute_site_metrics_v2(
                outputs["site_scores"],
                batch["site_labels"],
                batch["batch"],
                supervision_mask=batch.get("site_supervision_mask"),
                ranking_mask=batch.get("candidate_mask"),
            )
            graph_sources = _graph_sources_from_metadata(list(batch.get("graph_metadata") or []))
            site_metrics["source_recall_at_6"] = compute_sourcewise_recall_at_k(
                outputs["site_scores"],
                batch["site_labels"],
                batch["batch"],
                graph_sources,
                k=6,
                supervision_mask=batch.get("site_supervision_mask"),
                ranking_mask=batch.get("candidate_mask"),
            )
            site_metrics.update(
                mine_hard_negative_pairs(
                    outputs["site_scores"],
                    batch["site_labels"],
                    batch["batch"],
                    supervision_mask=batch.get("site_supervision_mask"),
                    candidate_mask=batch.get("candidate_mask"),
                    edge_index=batch.get("edge_index"),
                    atom_coordinates=batch.get("atom_coordinates"),
                    use_top_score=bool(getattr(self.model.config, "site_use_top_score_hard_neg", True)),
                    use_graph_local=bool(getattr(self.model.config, "site_use_graph_local_hard_neg", True)),
                    use_3d_local=bool(getattr(self.model.config, "site_use_3d_local_hard_neg", True)),
                    max_hard_negs_per_true=int(getattr(self.model.config, "site_hard_negative_max_per_true", 3)),
                )["stats"]
            )
            proposal_scores = outputs.get("site_scores_proposal")
            reranker_outputs = outputs.get("topk_reranker_outputs") or {}
            if proposal_scores is not None and reranker_outputs.get("selected_mask") is not None:
                site_metrics.update(
                    compute_reranker_metrics(
                        outputs["site_scores"],
                        proposal_scores,
                        batch["site_labels"],
                        batch["batch"],
                        reranker_outputs["selected_mask"],
                        supervision_mask=batch.get("site_supervision_mask"),
                    )
                )
            cyp_metrics = compute_cyp_metrics(
                outputs["cyp_logits"],
                batch["cyp_labels"],
                supervision_mask=batch.get("cyp_supervision_mask"),
            )
            return {**site_metrics, **cyp_metrics}

        def analyze_tau(self, loader=None) -> Dict[str, float]:
            target_loader = loader
            if target_loader is None:
                return {"tau_bde_correlation": 0.0}
            self.model.eval()
            tau_values = []
            tau_init_values = []
            with torch.no_grad():
                for raw_batch in target_loader:
                    if raw_batch is None:
                        continue
                    batch = self._prepare_batch(raw_batch)
                    outputs = self.model(batch)
                    tau_history = outputs.get("tau_history") or []
                    if tau_history:
                        tau_values.append(tau_history[-1].detach().cpu().reshape(-1))
                        tau_init_values.append(batch["tau_init"].detach().cpu().reshape(-1))
            if not tau_values:
                return {"tau_bde_correlation": 0.0}
            tau_final = torch.cat(tau_values)
            tau_init = torch.cat(tau_init_values)
            corr = float(torch.corrcoef(torch.stack([tau_final, tau_init]))[0, 1].item()) if tau_final.numel() > 1 else 0.0
            return {
                "tau_bde_correlation": corr,
                "tau_init_mean": float(tau_init.mean().item()),
                "tau_init_std": float(tau_init.std().item()),
                "tau_final_mean": float(tau_final.mean().item()),
                "tau_final_std": float(tau_final.std().item()),
            }

        def analyze_gates(self, loader=None) -> Dict[str, float]:
            target_loader = loader
            if target_loader is None:
                return {"gate_mean": 0.0}
            self.model.eval()
            gates = []
            with torch.no_grad():
                for raw_batch in target_loader:
                    if raw_batch is None:
                        continue
                    batch = self._prepare_batch(raw_batch)
                    outputs = self.model(batch)
                    gate_values = outputs.get("gate_values")
                    if gate_values is not None:
                        gates.append(gate_values.detach().cpu().reshape(-1))
            if not gates:
                return {"gate_mean": 0.0}
            g = torch.cat(gates)
            return {
                "gate_mean": float(g.mean().item()),
                "gate_std": float(g.std().item()),
                "gate_min": float(g.min().item()),
                "gate_max": float(g.max().item()),
            }

        def evaluate_loader(self, loader) -> Dict[str, object]:
            self.model.eval()
            site_scores = []
            proposal_site_scores = []
            site_labels = []
            site_supervision_masks = []
            candidate_masks = []
            proposal_masks = []
            site_batch = []
            merged_edge_parts = []
            merged_coord_parts = []
            graph_sources = []
            cyp_logits = []
            cyp_labels = []
            cyp_supervision_masks = []
            graph_offset = 0
            atom_offset = 0
            with torch.no_grad():
                for raw_batch in loader:
                    if raw_batch is None:
                        continue
                    batch = self._prepare_batch(raw_batch)
                    outputs = self.model(batch)
                    outputs = self._enforce_candidate_mask(outputs, batch)
                    if self.episode_logger is not None:
                        self.episode_logger.log_step(
                            split=str(getattr(loader, "_split_name", "eval")),
                            epoch=getattr(loader, "_current_epoch", None),
                            batch_idx=int(len(site_scores)),
                            batch=batch,
                            stats=None,
                            outputs=outputs,
                        )
                        self.episode_logger.log_examples(
                            split=str(getattr(loader, "_split_name", "eval")),
                            epoch=getattr(loader, "_current_epoch", None),
                            batch_idx=int(len(site_scores)),
                            batch=batch,
                            outputs=outputs,
                            stats=None,
                        )
                    site_scores.append(outputs["site_scores"].detach().cpu())
                    proposal_site_scores.append(
                        outputs.get("site_scores_proposal", outputs["site_scores"]).detach().cpu()
                    )
                    site_labels.append(batch["site_labels"].detach().cpu())
                    site_supervision_masks.append(
                        batch.get("site_supervision_mask", torch.ones_like(batch["site_labels"])).detach().cpu()
                    )
                    candidate_masks.append(
                        batch.get("candidate_mask", torch.ones_like(batch["site_labels"])).detach().cpu()
                    )
                    reranker_outputs = outputs.get("topk_reranker_outputs") or {}
                    proposal_masks.append(
                        reranker_outputs.get("selected_mask", torch.zeros_like(batch["site_labels"])).detach().cpu()
                    )
                    site_batch.append(batch["batch"].detach().cpu() + graph_offset)
                    edge_index = batch.get("edge_index")
                    if edge_index is not None:
                        merged_edge_parts.append(edge_index.detach().cpu() + atom_offset)
                    atom_coordinates = batch.get("atom_coordinates")
                    if atom_coordinates is not None:
                        merged_coord_parts.append(atom_coordinates.detach().cpu())
                    graph_sources.extend(_graph_sources_from_metadata(list(batch.get("graph_metadata") or [])))
                    cyp_logits.append(outputs["cyp_logits"].detach().cpu())
                    cyp_labels.append(batch["cyp_labels"].detach().cpu())
                    cyp_supervision_masks.append(
                        batch.get("cyp_supervision_mask", torch.ones_like(batch["cyp_labels"])).detach().cpu()
                    )
                    graph_offset += int(batch["cyp_labels"].shape[0])
                    atom_offset += int(batch["site_labels"].shape[0])
            if not site_scores:
                return {}
            merged_site_scores = torch.cat(site_scores, dim=0)
            merged_proposal_site_scores = torch.cat(proposal_site_scores, dim=0)
            merged_site_labels = torch.cat(site_labels, dim=0)
            merged_site_supervision_mask = torch.cat(site_supervision_masks, dim=0)
            merged_candidate_mask = torch.cat(candidate_masks, dim=0)
            merged_proposal_mask = torch.cat(proposal_masks, dim=0)
            merged_site_batch = torch.cat(site_batch, dim=0)
            merged_edge_index = torch.cat(merged_edge_parts, dim=1) if merged_edge_parts else torch.zeros((2, 0), dtype=torch.long)
            merged_atom_coordinates = torch.cat(merged_coord_parts, dim=0) if merged_coord_parts else None
            merged_cyp_logits = torch.cat(cyp_logits, dim=0)
            merged_cyp_labels = torch.cat(cyp_labels, dim=0)
            merged_cyp_supervision_mask = torch.cat(cyp_supervision_masks, dim=0)
            site_metrics = compute_site_metrics_v2(
                merged_site_scores,
                merged_site_labels,
                merged_site_batch,
                supervision_mask=merged_site_supervision_mask,
                ranking_mask=merged_candidate_mask,
            )
            site_metrics.update(
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
            site_metrics["source_recall_at_6"] = compute_sourcewise_recall_at_k(
                merged_site_scores,
                merged_site_labels,
                merged_site_batch,
                graph_sources,
                k=6,
                supervision_mask=merged_site_supervision_mask,
                ranking_mask=merged_candidate_mask,
            )
            site_metrics.update(
                compute_reranker_metrics(
                    merged_site_scores,
                    merged_proposal_site_scores,
                    merged_site_labels,
                    merged_site_batch,
                    merged_proposal_mask,
                    supervision_mask=merged_site_supervision_mask,
                )
            )
            cyp_metrics = compute_cyp_metrics(
                merged_cyp_logits,
                merged_cyp_labels,
                supervision_mask=merged_cyp_supervision_mask,
            )
            return {**site_metrics, **cyp_metrics}
else:  # pragma: no cover
    @dataclass
    class Trainer:  # type: ignore[override]
        model: object
        config: TrainingConfig = field(default_factory=TrainingConfig)
        device: object = None

        def __post_init__(self):
            require_torch()
