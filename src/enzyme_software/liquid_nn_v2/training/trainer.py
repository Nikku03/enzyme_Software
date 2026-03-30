from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.config import TrainingConfig
from enzyme_software.liquid_nn_v2.training.episode_logger import EpisodeLogger
from enzyme_software.liquid_nn_v2.training.loss import AdaptiveLossV2
from enzyme_software.liquid_nn_v2.training.metrics import compute_cyp_metrics, compute_site_metrics_v2
from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs, move_to_device


if TORCH_AVAILABLE:
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
            weights = None
            model_config = getattr(self.model, "config", None)
            if self.cyp_class_weights is not None:
                weights = self.cyp_class_weights.to(self.device) if hasattr(self.cyp_class_weights, "to") else torch.as_tensor(self.cyp_class_weights, dtype=torch.float32, device=self.device)
            self.loss_fn = AdaptiveLossV2(
                cyp_class_weights=weights,
                tau_reg_weight=float(getattr(model_config, "tau_prior_weight", 0.01)),
                energy_loss_weight=float(getattr(model_config, "energy_loss_weight", 0.0)),
                deliberation_loss_weight=float(getattr(model_config, "deliberation_loss_weight", 0.0)),
                energy_margin=float(getattr(model_config, "energy_margin", 0.15)),
                energy_loss_clip=float(getattr(model_config, "energy_loss_clip", 2.0)),
            )
            self.loss_fn.to(self.device)
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

        def _prepare_batch(self, batch_or_graphs):
            if isinstance(batch_or_graphs, dict):
                return move_to_device(batch_or_graphs, self.device)
            return move_to_device(collate_molecule_graphs(batch_or_graphs), self.device)

        def _apply_confidence_weights(self, loss, batch):
            return loss

        def _masked_bce_with_logits(self, logits, labels, supervision_mask):
            labels = labels.float().view_as(logits)
            raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            if supervision_mask is None:
                return raw.mean()
            mask = supervision_mask.float().view_as(logits)
            return (raw * mask).sum() / mask.sum().clamp_min(1.0)

        def _site_style_vote_loss(self, logits, batch: Dict[str, object]):
            site_labels = batch["site_labels"]
            site_mask = batch.get("site_supervision_mask")
            node_weights = batch.get("node_confidence_weights")
            graph_weights = batch.get("graph_confidence_weights")
            site_batch = batch["batch"]
            loss_value, _ = self.loss_fn.site_loss(
                logits,
                site_labels,
                site_batch,
                supervision_mask=site_mask,
                node_weights=node_weights,
                graph_weights=graph_weights,
            )
            return loss_value

        def compute_loss(self, batch: Dict[str, object], outputs: Dict[str, object]):
            loss, stats = self.loss_fn(
                outputs["site_logits"],
                outputs["cyp_logits"],
                batch["site_labels"],
                batch["cyp_labels"],
                batch["batch"],
                batch.get("site_supervision_mask"),
                batch.get("cyp_supervision_mask"),
                batch.get("graph_confidence_weights"),
                batch.get("node_confidence_weights"),
                outputs.get("tau_history"),
                batch.get("tau_init"),
                outputs.get("energy_outputs"),
                outputs.get("deliberation_outputs"),
            )
            stats = dict(stats)
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
            vote_heads = outputs.get("site_vote_heads") or {}
            model_config = getattr(self.model, "config", None)
            if isinstance(vote_heads, dict) and vote_heads:
                site_labels = batch["site_labels"]
                site_mask = batch.get("site_supervision_mask")
                aux_specs = [
                    ("lnn", "lnn_vote", float(getattr(model_config, "nexus_lnn_vote_aux_weight", 0.0))),
                    ("wave", "wave_vote", float(getattr(model_config, "nexus_wave_vote_aux_weight", 0.0))),
                    ("analogical", "analogical_vote", float(getattr(model_config, "nexus_analogical_vote_aux_weight", 0.0))),
                ]
                for name, key, weight in aux_specs:
                    tensor = vote_heads.get(key)
                    if tensor is None or weight <= 0.0:
                        continue
                    aux_loss = self._site_style_vote_loss(tensor, batch)
                    loss = loss + (weight * aux_loss)
                    stats[f"nexus_{name}_vote_loss"] = float(aux_loss.detach().item())
                    stats[f"nexus_{name}_vote_weight"] = float(weight)
                final_probs = torch.sigmoid(outputs["site_logits"]).detach()
                consistency_specs = [
                    ("wave", "wave_vote", float(getattr(model_config, "nexus_wave_vote_consistency_weight", 0.0))),
                    ("analogical", "analogical_vote", float(getattr(model_config, "nexus_analogical_vote_consistency_weight", 0.0))),
                ]
                for name, key, weight in consistency_specs:
                    tensor = vote_heads.get(key)
                    if tensor is None or weight <= 0.0:
                        continue
                    raw = F.binary_cross_entropy_with_logits(tensor, final_probs, reduction="none")
                    if site_mask is not None:
                        mask = site_mask.float().view_as(raw)
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
            outputs = self.model(batch)
            loss, stats = self.compute_loss(batch, outputs)
            self._run_finite_checks(outputs, stats)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss detected in train_epoch: stats={stats}")
            self.optimizer.zero_grad()
            loss.backward()
            self._clip_gradients()
            self.optimizer.step()
            return stats

        def train_loader_epoch(self, loader) -> Dict[str, float]:
            self.model.train()
            history = []
            for batch_idx, raw_batch in enumerate(loader):
                if raw_batch is None:
                    continue
                batch = self._prepare_batch(raw_batch)
                outputs = self.model(batch)
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
                self._clip_gradients()
                self.optimizer.step()
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
            site_metrics = compute_site_metrics_v2(
                outputs["site_scores"],
                batch["site_labels"],
                batch["batch"],
                supervision_mask=batch.get("site_supervision_mask"),
            )
            cyp_metrics = compute_cyp_metrics(outputs["cyp_logits"], batch["cyp_labels"])
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
            site_labels = []
            site_supervision_masks = []
            site_batch = []
            cyp_logits = []
            cyp_labels = []
            batch_offset = 0
            with torch.no_grad():
                for raw_batch in loader:
                    if raw_batch is None:
                        continue
                    batch = self._prepare_batch(raw_batch)
                    outputs = self.model(batch)
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
                    site_labels.append(batch["site_labels"].detach().cpu())
                    site_supervision_masks.append(
                        batch.get("site_supervision_mask", torch.ones_like(batch["site_labels"])).detach().cpu()
                    )
                    site_batch.append(batch["batch"].detach().cpu() + batch_offset)
                    cyp_logits.append(outputs["cyp_logits"].detach().cpu())
                    cyp_labels.append(batch["cyp_labels"].detach().cpu())
                    batch_offset += int(batch["cyp_labels"].shape[0])
            if not site_scores:
                return {}
            merged_site_scores = torch.cat(site_scores, dim=0)
            merged_site_labels = torch.cat(site_labels, dim=0)
            merged_site_supervision_mask = torch.cat(site_supervision_masks, dim=0)
            merged_site_batch = torch.cat(site_batch, dim=0)
            merged_cyp_logits = torch.cat(cyp_logits, dim=0)
            merged_cyp_labels = torch.cat(cyp_labels, dim=0)
            site_metrics = compute_site_metrics_v2(
                merged_site_scores,
                merged_site_labels,
                merged_site_batch,
                supervision_mask=merged_site_supervision_mask,
            )
            cyp_metrics = compute_cyp_metrics(merged_cyp_logits, merged_cyp_labels)
            return {**site_metrics, **cyp_metrics}
else:  # pragma: no cover
    @dataclass
    class Trainer:  # type: ignore[override]
        model: object
        config: TrainingConfig = field(default_factory=TrainingConfig)
        device: object = None

        def __post_init__(self):
            require_torch()
