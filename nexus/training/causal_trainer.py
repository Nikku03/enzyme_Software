from __future__ import annotations

from contextlib import nullcontext
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from nexus.core.dynamics_engine import NEXUS_Dynamics_Engine
from nexus.core.field_optimizer import Field_Gradient_Optimizer, FieldGradientOptimizationReport
from nexus.core.flux_analysis import NCFAFluxPropagator
from nexus.core.inference import NEXUS_Module1_Output
from nexus.layers.dag_learner import MetabolicDAGLearner
from nexus.physics.clifford_math import embed_coordinates
from nexus.pocket.ddi import DDIOccupancyState
from nexus.reasoning.baseline_memory import BaselineMemoryBank
from nexus.training.losses import GatedAnalogicalGodLoss, NEXUS_God_Loss


@dataclass
class OptimizerGroupSummary:
    electronic_count: int
    physics_count: int
    query_count: int
    loss_count: int


@dataclass
class TrainingStepResult:
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class Metabolic_Causal_Trainer(nn.Module):
    def __init__(
        self,
        model: Optional[NEXUS_Dynamics_Engine] = None,
        loss_fn: Optional[NEXUS_God_Loss] = None,
        field_optimizer: Optional[Field_Gradient_Optimizer] = None,
        dag_learner: Optional[MetabolicDAGLearner] = None,
        *,
        grad_clip_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        dynamics_steps: int = 8,
        dynamics_dt: float = 0.001,
        dynamics_summary_mode: str = "lite",
        checkpoint_dynamics: bool = True,
        curriculum_transition_step: int = 2000,
        enable_static_compile: bool = True,
        compile_mode: str = "reduce-overhead",
        compile_suppress_errors: bool = True,
        enable_bf16_hot_path: bool = True,
        enable_wsd_scheduler: bool = True,
        wsd_warmup_ratio: float = 0.1,
        wsd_stable_ratio: float = 0.8,
        wsd_decay_ratio: float = 0.1,
        wsd_decay_style: str = "linear",
        wsd_warmup_init_scale: float = 0.1,
        wsd_min_lr_scale: float = 0.05,
        low_memory_train_mode: bool = False,
        low_memory_scan_gradients: bool = False,
        use_galore: bool = True,
    ) -> None:
        super().__init__()
        self.model = model or NEXUS_Dynamics_Engine()
        self.loss_fn = loss_fn or NEXUS_God_Loss()
        self.field_optimizer = field_optimizer or Field_Gradient_Optimizer()
        self.dag_learner = dag_learner or MetabolicDAGLearner()
        self.flux_propagator = NCFAFluxPropagator()
        self.grad_clip_norm = float(grad_clip_norm)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if self.gradient_accumulation_steps < 1:
            self.gradient_accumulation_steps = 1
        self.dynamics_steps = int(dynamics_steps)
        self.dynamics_dt = float(dynamics_dt)
        self.dynamics_summary_mode = str(dynamics_summary_mode).lower()
        self.checkpoint_dynamics = bool(checkpoint_dynamics)
        self.curriculum_transition_step = int(curriculum_transition_step)
        self.enable_static_compile = bool(enable_static_compile)
        self.compile_mode = str(compile_mode)
        self.compile_suppress_errors = bool(compile_suppress_errors)
        self.enable_bf16_hot_path = bool(enable_bf16_hot_path)
        self.enable_wsd_scheduler = bool(enable_wsd_scheduler)
        self.wsd_warmup_ratio = float(wsd_warmup_ratio)
        self.wsd_stable_ratio = float(wsd_stable_ratio)
        self.wsd_decay_ratio = float(wsd_decay_ratio)
        self.wsd_decay_style = str(wsd_decay_style).lower()
        self.wsd_warmup_init_scale = float(wsd_warmup_init_scale)
        self.wsd_min_lr_scale = float(wsd_min_lr_scale)
        self.low_memory_train_mode = bool(low_memory_train_mode)
        self.low_memory_scan_gradients = bool(low_memory_scan_gradients)
        self.use_galore = bool(use_galore)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.total_training_steps: Optional[int] = None
        self.last_checkpoint_fallback = False
        self.last_dynamics_fallback = False
        self.static_compile_applied = False
        self.compiled_module_names: List[str] = []
        self.register_buffer("global_step_counter", torch.zeros((), dtype=torch.long))
        self.gated_loss = GatedAnalogicalGodLoss(confidence_threshold=0.6)
        # Memory bank is populated externally (trainer.memory_bank.populate_from_mols)
        # before training begins.  Left empty here so training still runs without it.
        self.memory_bank = BaselineMemoryBank(device="cpu")
        self._maybe_prepare_precision_runtime()
        self._validate_wsd_config()

    def _module_device(self) -> torch.device:
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def _autocast_enabled(self) -> bool:
        return self.enable_bf16_hot_path and self._module_device().type == "cuda"

    def _autocast_context(self):
        if not self._autocast_enabled():
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def _compile_target_specs(self) -> List[tuple[str, nn.Module, str]]:
        targets: List[tuple[str, nn.Module, str]] = []
        specs = [
            ("model.module1.field_engine.siren_field", self.model.module1.field_engine, "siren_field"),
            ("model.module1.field_engine.multiscale_engine.attention", self.model.module1.field_engine.multiscale_engine, "attention"),
            ("model.pocket_encoder.reversed_attention", self.model.pocket_encoder, "reversed_attention"),
            ("model.pocket_encoder.a_field_controller.read_attention", self.model.pocket_encoder.a_field_controller, "read_attention"),
        ]
        for name, root, attr in specs:
            if hasattr(root, attr):
                module = getattr(root, attr)
                if isinstance(module, nn.Module):
                    targets.append((name, root, attr))
        return targets

    def _maybe_prepare_precision_runtime(self) -> None:
        if self.static_compile_applied or not self.enable_static_compile:
            return
        if not hasattr(torch, "compile"):
            return
        if self._module_device().type != "cuda":
            return
        import torch._dynamo as dynamo

        dynamo.config.suppress_errors = self.compile_suppress_errors
        compiled: List[str] = []
        for name, root, attr in self._compile_target_specs():
            module = getattr(root, attr)
            try:
                setattr(root, attr, torch.compile(module))
                compiled.append(name)
            except Exception:
                continue
        self.compiled_module_names = compiled
        self.static_compile_applied = True

    def _validate_wsd_config(self) -> None:
        ratios = [self.wsd_warmup_ratio, self.wsd_stable_ratio, self.wsd_decay_ratio]
        if any(r < 0.0 for r in ratios):
            raise ValueError("WSD ratios must be non-negative")
        total = sum(ratios)
        if total <= 0.0:
            raise ValueError("At least one WSD phase ratio must be positive")
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError("WSD ratios must sum to 1.0")
        if self.wsd_decay_style not in {"linear", "cosine"}:
            raise ValueError("wsd_decay_style must be 'linear' or 'cosine'")
        if self.wsd_warmup_init_scale <= 0.0:
            raise ValueError("wsd_warmup_init_scale must be > 0")
        if self.wsd_min_lr_scale <= 0.0:
            raise ValueError("wsd_min_lr_scale must be > 0")

    def set_total_training_steps(self, total_steps: int) -> None:
        total = int(total_steps)
        if total <= 0:
            raise ValueError("total_steps must be > 0")
        self.total_training_steps = total
        if self.optimizer is not None:
            self.scheduler = self._build_wsd_scheduler(self.optimizer, total)

    def _wsd_phase_counts(self, total_steps: int) -> tuple[int, int, int]:
        total = max(int(total_steps), 1)
        warmup = int(round(total * self.wsd_warmup_ratio))
        stable = int(round(total * self.wsd_stable_ratio))
        decay = total - warmup - stable
        if warmup == 0 and self.wsd_warmup_ratio > 0.0:
            warmup = 1
        if decay == 0 and self.wsd_decay_ratio > 0.0:
            decay = 1
        if warmup + stable + decay > total:
            overflow = warmup + stable + decay - total
            stable = max(0, stable - overflow)
        if warmup + stable + decay < total:
            stable += total - (warmup + stable + decay)
        return warmup, stable, decay

    def _build_wsd_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        warmup_steps, stable_steps, decay_steps = self._wsd_phase_counts(total_steps)

        def lr_lambda(step: int) -> float:
            if total_steps <= 1:
                return 1.0
            current = min(max(int(step), 0), total_steps - 1)
            if warmup_steps > 0 and current < warmup_steps:
                progress = (current + 1) / warmup_steps
                return self.wsd_warmup_init_scale + (1.0 - self.wsd_warmup_init_scale) * progress
            stable_end = warmup_steps + stable_steps
            if current < stable_end or decay_steps <= 0:
                return 1.0
            decay_progress = min(max((current - stable_end + 1) / decay_steps, 0.0), 1.0)
            if self.wsd_decay_style == "cosine":
                cosine = 0.5 * (1.0 + math.cos(decay_progress * math.pi))
                return self.wsd_min_lr_scale + (1.0 - self.wsd_min_lr_scale) * cosine
            return 1.0 - (1.0 - self.wsd_min_lr_scale) * decay_progress

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if warmup_steps > 0:
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base_lr * self.wsd_warmup_init_scale
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        return scheduler

    @staticmethod
    def _to_fp32(tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            return tensor.to(dtype=torch.float32)
        return tensor

    def _move_to_device(self, obj: Any, device: Optional[torch.device] = None) -> Any:
        target = device or self._module_device()
        if torch.is_tensor(obj):
            return obj.to(target)
        if isinstance(obj, dict):
            return {key: self._move_to_device(value, target) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(value, target) for value in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(value, target) for value in obj)
        return obj

    @staticmethod
    def _sanitize_tensor(
        tensor: torch.Tensor,
        *,
        nan: float = 0.0,
        posinf: float = 1.0e4,
        neginf: float = -1.0e4,
        clamp: Optional[tuple[float, float]] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            return tensor
        out = torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)
        if clamp is not None and out.is_floating_point():
            out = out.clamp(min=clamp[0], max=clamp[1])
        return out

    def _module1_forward_hot_path(self, smiles: str) -> NEXUS_Module1_Output:
        seed = self.model.module1.agency(smiles)
        manifold = self.model.module1.refiner(seed)
        _ = self.model.module1.symmetry_engine(manifold)
        with self._autocast_context():
            field_state = self.model.module1.field_engine.build_state(manifold)
        scan = field_state.field.scan_reaction_volume(manifold)
        alignment_score = 0.5 * scan.alignment_tensor.mean(dim=-1) + 0.5 * scan.l2_alignment
        order = torch.argsort(scan.effective_reactivity, descending=True)
        return NEXUS_Module1_Output(
            seed=seed,
            manifold=manifold,
            field_state=field_state,
            scan=scan,
            ranked_atom_indices=scan.atom_indices[order],
            som_coordinates=scan.refined_peak_points[order],
            psi_peak=scan.refined_peak_values[order],
            approach_vector=scan.approach_vectors[order],
            alignment_score=alignment_score[order],
            exposure_score=scan.exposure_scores[order],
            effective_reactivity=scan.effective_reactivity[order],
        )

    def _build_pocket_encoding_hot_path(
        self,
        module1_out: NEXUS_Module1_Output,
        target_rank: int,
        protein_data: Mapping[str, Any],
    ):
        protein_coords = protein_data["coords"]
        isoform_embedding = protein_data["isoform_embedding"]
        target_point = module1_out.som_coordinates[target_rank].view(1, 3)
        components = module1_out.field_state.field.query_components(target_point, compute_observables=True)
        latent = components.get("latent_multivector")
        if latent is None:
            drug_mv = embed_coordinates(target_point.to(dtype=protein_coords.dtype))
        else:
            if latent.ndim == 3:
                drug_mv = latent.mean(dim=-2)
            else:
                drug_mv = latent
            drug_mv = drug_mv.to(device=protein_coords.device, dtype=protein_coords.dtype)
        with self._autocast_context():
            return self.model.pocket_encoder(
                drug_mv,
                protein_coords,
                isoform_embedding,
                sequence=protein_data.get("sequence"),
                sequence_embedding=protein_data.get("sequence_embedding"),
                variant_ids=protein_data.get("variant_ids"),
                variant_embedding=protein_data.get("variant_embedding"),
                residue_types=protein_data.get("residue_types"),
                conservation_scores=protein_data.get("conservation_scores"),
                allosteric=protein_data.get("allosteric"),
                t=protein_data.get("t"),
            )

    def _trainable(self, params: Iterable[nn.Parameter]) -> List[nn.Parameter]:
        return [p for p in params if p is not None and p.requires_grad]

    def _param_ids(self, params: Sequence[nn.Parameter]) -> set[int]:
        return {id(p) for p in params}

    def _unique(self, params: Sequence[nn.Parameter]) -> List[nn.Parameter]:
        seen: set[int] = set()
        unique: List[nn.Parameter] = []
        for param in params:
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            unique.append(param)
        return unique

    # Keywords that identify each parameter group regardless of DDP module prefixes.
    # tensor_splatter controls 3-D spatial coordinate mapping; alpha_raw gates the
    # equivariant splatter kernel.  Both require full-rank gradients for 3-D stability.
    _PHYSICS_KEYWORDS = frozenset({"coupling_lambda", "reactive_scale", "alpha_raw", "alpha_species_bias", "tensor_splatter"})
    _QUERY_KEYWORDS = frozenset({"query_engine"})
    # Heavy 512-dim layers: SIREN MLP, attention heads, DAG learner, hyper-network.
    # These get GaLore rank-128 projection.  Everything else in the electronic group
    # gets rank-32.  Only applies to ndim >= 2 parameters (weight matrices).
    _GALORE_DEEP_KEYWORDS = frozenset({"siren_field", "attention", "dag_learner", "hyper_net"})

    def _split_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        physics_params: List[nn.Parameter] = []
        query_params: List[nn.Parameter] = []
        electronic_params: List[nn.Parameter] = []
        seen: set[int] = set()

        # Include both the dynamics model and the DAG learner.  loss_fn.log_vars is
        # handled separately below because it belongs to its own optimizer group.
        param_sources = [
            ("model", self.model),
            ("dag_learner", self.dag_learner),
        ]
        for source_prefix, module in param_sources:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                pid = id(param)
                if pid in seen:
                    continue
                seen.add(pid)
                full_name = f"{source_prefix}.{name}"
                if any(kw in full_name for kw in self._PHYSICS_KEYWORDS):
                    physics_params.append(param)
                elif any(kw in full_name for kw in self._QUERY_KEYWORDS):
                    query_params.append(param)
                else:
                    electronic_params.append(param)

        loss_params = self._unique(
            self._trainable([self.loss_fn.log_vars])
            + list(self.gated_loss.parameters())
        )
        return {
            "electronic": electronic_params,
            "physics": physics_params,
            "query": query_params,
            "loss": loss_params,
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        self._maybe_prepare_precision_runtime()

        if not self.use_galore:
            # Plain AdamW — no SVD, safe on Colab / CPU.
            # Enumerate submodules explicitly so gated_loss.log_s and loss_fn.log_vars
            # are guaranteed to be in the optimizer regardless of PyTorch module-walk order.
            all_params = self._unique(
                [p for p in self.model.parameters() if p.requires_grad]
                + [p for p in self.dag_learner.parameters() if p.requires_grad]
                + [p for p in self.gated_loss.parameters() if p.requires_grad]
                + [p for p in self.loss_fn.parameters() if p.requires_grad]
            )
            self.optimizer = torch.optim.AdamW(all_params, lr=1.0e-4, weight_decay=1.0e-5)
            if self.enable_wsd_scheduler and self.total_training_steps is not None:
                self.scheduler = self._build_wsd_scheduler(self.optimizer, self.total_training_steps)
            return self.optimizer

        try:
            from galore_torch import GaLoreAdamW
        except ImportError as exc:
            raise ImportError(
                "galore_torch is required. Install with: pip install galore-torch"
            ) from exc

        # Robustify GaLore SVD against ill-conditioned untrained weight matrices.
        # CUDA's cusolver driver diverges on near-zero SIREN weights (error code 8).
        # Fall back to CPU LAPACK, which is always stable, then return to original device.
        try:
            import types as _types
            import galore_torch.galore_projector as _gp
            if not getattr(_gp.GradientProjector, "_svd_patched", False):
                _orig_get_orth = _gp.GradientProjector.get_orthogonal_matrix

                def _robust_get_orth(self, weights, rank, type):  # noqa: A002
                    try:
                        return _orig_get_orth(self, weights, rank, type)
                    except torch.linalg.LinAlgError:
                        # cusolver diverged — run SVD on CPU LAPACK and move result back
                        data = weights.data if hasattr(weights, "data") else weights
                        dev = data.device
                        cpu_proxy = _types.SimpleNamespace(data=data.cpu().float())
                        result = _orig_get_orth(self, cpu_proxy, rank, type)
                        if torch.is_tensor(result):
                            return result.to(dev)
                        return result

                _gp.GradientProjector.get_orthogonal_matrix = _robust_get_orth
                _gp.GradientProjector._svd_patched = True
        except Exception:
            pass  # galore_torch layout changed — skip patch

        groups = self._split_parameter_groups()

        # --- Full-rank groups: physics constants, query engine, God Loss scalars ---
        # These must never have gradients projected into a low-rank subspace.
        param_groups: list = []
        if groups["physics"]:
            param_groups.append({
                "params": groups["physics"],
                "lr": 1.0e-5,
                "weight_decay": 0.0,
                "name": "physics_constants",
            })
        if groups["query"]:
            param_groups.append({
                "params": groups["query"],
                "lr": 5.0e-4,
                "weight_decay": 1.0e-5,
                "name": "query_engine",
            })
        if groups["loss"]:
            param_groups.append({
                "params": groups["loss"],
                "lr": 1.0e-3,
                "weight_decay": 0.0,
                "name": "god_loss_balancer",
            })

        # --- GaLore routing for electronic parameters ---
        # Re-iterate with full parameter names so we can assign rank by layer depth.
        # 1-D parameters (biases, layer norms, scalars) cannot be projected into a
        # low-rank subspace and fall through to a standard full-rank group.
        claimed_ids: set[int] = (
            {id(p) for p in groups["physics"]}
            | {id(p) for p in groups["query"]}
            | {id(p) for p in groups["loss"]}
        )
        electronic_full_rank: list = []
        for source_prefix, module in [("model", self.model), ("dag_learner", self.dag_learner)]:
            for name, param in module.named_parameters():
                if not param.requires_grad or id(param) in claimed_ids:
                    continue
                claimed_ids.add(id(param))
                full_name = f"{source_prefix}.{name}"
                is_deep = any(kw in full_name for kw in self._GALORE_DEEP_KEYWORDS)
                if param.ndim >= 2:
                    # Weight matrix: project gradients into low-rank subspace.
                    param_groups.append({
                        "params": [param],
                        "rank": 128 if is_deep else 32,
                        "update_proj_gap": 200,
                        "scale": 0.25,
                        "proj_type": "std",
                        "lr": 1.0e-4,
                        "weight_decay": 1.0e-5,
                    })
                else:
                    # Bias / layer-norm / 1-D scalar: must stay full-rank.
                    electronic_full_rank.append(param)

        if electronic_full_rank:
            param_groups.append({
                "params": electronic_full_rank,
                "lr": 1.0e-4,
                "weight_decay": 1.0e-5,
                "name": "electronic_biases",
            })

        self.optimizer = GaLoreAdamW(param_groups, lr=1.0e-4, weight_decay=1.0e-5)
        if self.enable_wsd_scheduler and self.total_training_steps is not None:
            self.scheduler = self._build_wsd_scheduler(self.optimizer, self.total_training_steps)
        return self.optimizer

    def optimizer_summary(self) -> OptimizerGroupSummary:
        groups = self._split_parameter_groups()
        return OptimizerGroupSummary(
            electronic_count=sum(p.numel() for p in groups["electronic"]),
            physics_count=sum(p.numel() for p in groups["physics"]),
            query_count=sum(p.numel() for p in groups["query"]),
            loss_count=sum(p.numel() for p in groups["loss"]),
        )

    def get_all_parameters(self) -> List[nn.Parameter]:
        return self._unique(self._trainable(list(self.parameters())))

    def clip_gradients(self) -> float:
        params = self.get_all_parameters()
        if not params:
            return 0.0
        # Correct NaN gradients that may arise from AMP + unstable SVD in GaLore.
        for param in params:
            if param.grad is None:
                continue
            param.grad = torch.nan_to_num(
                param.grad,
                nan=0.0,
                posinf=self.grad_clip_norm,
                neginf=-self.grad_clip_norm,
            )
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_norm)
        if torch.is_tensor(total_norm):
            return float(total_norm.detach().cpu().item())
        return float(total_norm)

    def _unpack_single(self, batch: Any) -> Any:
        if isinstance(batch, (list, tuple)) and len(batch) == 1:
            return batch[0]
        return batch

    def _lookup(self, batch: Any, *keys: str, default: Any = None) -> Any:
        batch = self._unpack_single(batch)
        if isinstance(batch, Mapping):
            for key in keys:
                if key in batch:
                    return batch[key]
        else:
            for key in keys:
                if hasattr(batch, key):
                    return getattr(batch, key)
        return default

    def _resolve_smiles(self, batch: Any) -> str:
        smiles = self._lookup(batch, "smiles", "graph", default=None)
        if isinstance(smiles, (list, tuple)):
            smiles = smiles[0]
        if not isinstance(smiles, str):
            raise TypeError("Batch must provide a SMILES string under 'smiles' or 'graph'")
        return smiles

    def _resolve_protein_data(self, batch: Any) -> Optional[Dict[str, Any]]:
        protein = self._lookup(batch, "protein_data", "pocket", default=None)
        if protein is None:
            coords = self._lookup(batch, "protein_coords", default=None)
            isoform_embedding = self._lookup(batch, "isoform_embedding", default=None)
            if coords is None or isoform_embedding is None:
                return None
            protein = {
                "coords": coords,
                "isoform_embedding": isoform_embedding,
                "variant_ids": self._lookup(batch, "variant_ids", default=None),
                "variant_embedding": self._lookup(batch, "variant_embedding", default=None),
                "sequence": self._lookup(batch, "sequence", default=None),
                "sequence_embedding": self._lookup(batch, "sequence_embedding", default=None),
                "residue_types": self._lookup(batch, "residue_types", default=None),
                "conservation_scores": self._lookup(batch, "conservation_scores", default=None),
                "allosteric": self._lookup(batch, "allosteric", default=None),
                "ddi_occupancy": self._lookup(batch, "ddi_occupancy", default=None),
            }
        if not isinstance(protein, Mapping):
            raise TypeError("protein_data must be a mapping when provided")
        return dict(protein)

    def current_curriculum_stage(self) -> str:
        if int(self.global_step_counter.detach().cpu().item()) < self.curriculum_transition_step:
            return "field_reconstruction"
        return "supervised_ranking"

    def _resolve_true_atom_index(self, batch: Any, device: torch.device) -> torch.Tensor:
        direct = self._lookup(batch, "true_som_idx", "som_idx", "target_atom_index", default=None)
        if direct is not None:
            if isinstance(direct, (list, tuple)):
                direct = direct[0]
            return torch.as_tensor(direct, dtype=torch.long, device=device).view(())
        sites = self._lookup(batch, "site_atoms", "metabolism_sites", default=None)
        if sites is None:
            # No SoM label in batch (e.g. unlabelled ATTNSOM SDF).
            # Fall back to atom 0 so field-reconstruction and DAG losses still train;
            # the ranking loss will be noisy but won't crash.
            import warnings
            warnings.warn(
                "Batch has no SoM label — falling back to atom 0 for ranking loss. "
                "Add 'true_som_idx' to your dataset for supervised ranking.",
                UserWarning,
                stacklevel=4,
            )
            return torch.zeros((), dtype=torch.long, device=device)
        if isinstance(sites, torch.Tensor):
            if sites.numel() == 0:
                raise ValueError("Batch contains no site atoms")
            return sites.view(-1)[0].to(device=device, dtype=torch.long)
        if isinstance(sites, (list, tuple)):
            if len(sites) == 0:
                raise ValueError("Batch contains no site atoms")
            return torch.as_tensor(sites[0], dtype=torch.long, device=device).view(())
        raise TypeError("Unsupported site atom annotation format")

    def _resolve_exp_rate(
        self,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], bool]:
        value = self._lookup(
            batch,
            "exp_rate",
            "intrinsic_clearance",
            "kcat",
            "experimental_rate",
            default=None,
        )
        if value is None:
            return None, False
        if isinstance(value, (list, tuple)):
            value = value[0]
        return torch.as_tensor(value, dtype=dtype, device=device).clamp_min(1.0e-12), True

    def _scan_row_index(self, atom_indices: torch.Tensor, true_atom_index: torch.Tensor) -> torch.Tensor:
        matches = (atom_indices == true_atom_index).nonzero(as_tuple=False)
        if matches.numel() > 0:
            return matches[0, 0]
        nearest = torch.argmin((atom_indices.to(torch.float32) - true_atom_index.to(torch.float32)).abs())
        return nearest.to(dtype=torch.long)

    def _dynamics_summary(
        self,
        q_init_internal: torch.Tensor,
        target_point_internal: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        target_atom_index: torch.Tensor,
        accessibility_field=None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dynamics_summary_mode == "lite":
            return self._lite_dynamics_summary(
                q_init_internal,
                target_point_internal,
                smiles=smiles,
                species=species,
                target_atom_index=target_atom_index,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )
        reactive_reference = self.model.hamiltonian.reactive_reference.detach().clone()
        try:
            self.last_dynamics_fallback = False
            try:
                navigation = self.model.navigator(
                    self.model.hamiltonian,
                    q_init_internal,
                    smiles=smiles,
                    species=species,
                    target_atom_index=target_atom_index,
                    target_point=target_point_internal,
                    steps=self.dynamics_steps,
                    dt=self.dynamics_dt,
                    accessibility_field=accessibility_field,
                    ddi_occupancy=ddi_occupancy,
                )
                kinetics = self.model.kinetics(
                    self.model.hamiltonian,
                    navigation,
                    q_init_internal,
                    smiles=smiles,
                    species=species,
                    accessibility_field=accessibility_field,
                    ddi_occupancy=ddi_occupancy,
                )
                h_initial = self.model.hamiltonian(
                    q_init_internal,
                    torch.zeros_like(q_init_internal),
                    smiles=smiles,
                    species=species,
                    accessibility_field=accessibility_field,
                    ddi_occupancy=ddi_occupancy,
                )
                h_final = navigation.best.trajectory.h_path[-1]
                ts_eigenvalues = kinetics.ts_eigenvalues
                if ts_eigenvalues.numel() < 2:
                    ts_eigenvalues = F.pad(ts_eigenvalues, (0, 2 - ts_eigenvalues.numel()))
                pred_rate = kinetics.quantum_rate_rpmd.clamp_min(1.0e-12)
                if not bool(torch.isfinite(pred_rate).all().item()) or float(pred_rate.detach().cpu().item()) <= 0.0:
                    pred_rate = kinetics.metabolic_rate.clamp_min(1.0e-12)
                return pred_rate, h_initial, h_final, ts_eigenvalues
            except Exception as _dyn_err:
                import traceback as _tb
                print("\n[!!!] PHYSICS SURROGATE CRASH DETECTED [!!!]")
                _tb.print_exc()
                print(f"[!!!] smiles={smiles!r}  q_shape={q_init_internal.shape}  "
                      f"dtype={q_init_internal.dtype}  device={q_init_internal.device}")
                print("[!!!] -------------------------------- [!!!]\n", flush=True)
                self.last_dynamics_fallback = True
                h_initial = self.model.hamiltonian(
                    q_init_internal,
                    torch.zeros_like(q_init_internal),
                    smiles=smiles,
                    species=species,
                    accessibility_field=accessibility_field,
                    ddi_occupancy=ddi_occupancy,
                )
                pred_rate = torch.exp(-torch.relu(h_initial)).clamp_min(1.0e-12)
                ts_eigenvalues = torch.stack(
                    [
                        h_initial.new_tensor(-self.loss_fn.topology_margin),
                        h_initial.new_tensor(self.loss_fn.topology_margin),
                    ],
                    dim=0,
                )
                return pred_rate, h_initial, h_initial, ts_eigenvalues
        finally:
            self.model.hamiltonian.reactive_reference.copy_(reactive_reference)

    def _lite_dynamics_summary(
        self,
        q_init_internal: torch.Tensor,
        target_point_internal: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        target_atom_index: torch.Tensor,
        accessibility_field=None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.last_dynamics_fallback = False
        self.last_checkpoint_fallback = False
        p_init = torch.zeros_like(q_init_internal)
        trajectory = self.model.solver(
            self.model.hamiltonian,
            q_init_internal,
            p_init,
            steps=self.dynamics_steps,
            dt=self.dynamics_dt,
            smiles=smiles,
            species=species,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
        )
        h_initial = trajectory.h_path[0]
        h_final = trajectory.h_path[-1]
        atom_idx = target_atom_index.to(dtype=torch.long, device=q_init_internal.device)
        terminal_distance = (trajectory.q_path[-1, atom_idx] - target_point_internal).pow(2).sum().sqrt()
        action_scale = trajectory.action_integral.abs()
        effective_barrier = torch.relu(h_final - h_initial) + 0.1 * terminal_distance + 0.01 * action_scale
        pred_rate = torch.exp(-effective_barrier).clamp_min(1.0e-12)
        ts_eigenvalues = torch.stack(
            [
                h_initial.new_tensor(-self.loss_fn.topology_margin),
                h_initial.new_tensor(self.loss_fn.topology_margin),
            ],
            dim=0,
        )
        return pred_rate, h_initial, h_final, ts_eigenvalues

    def _dynamics_summary_checkpointed(
        self,
        q_init_internal: torch.Tensor,
        target_point_internal: torch.Tensor,
        *,
        smiles: str,
        species: torch.Tensor,
        target_atom_index: torch.Tensor,
        accessibility_field=None,
        ddi_occupancy: Optional[DDIOccupancyState] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def _wrapped(q_input: torch.Tensor, target_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return self._dynamics_summary(
                q_input,
                target_input,
                smiles=smiles,
                species=species,
                target_atom_index=target_atom_index,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )

        if self.checkpoint_dynamics:
            self.last_checkpoint_fallback = False
            return checkpoint.checkpoint(
                _wrapped,
                q_init_internal,
                target_point_internal,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        self.last_checkpoint_fallback = True
        return _wrapped(q_init_internal, target_point_internal)

    def _zero_sobolev_report(self, manifold) -> FieldGradientOptimizationReport:
        zero = torch.zeros((), dtype=manifold.pos.dtype, device=manifold.pos.device)
        return FieldGradientOptimizationReport(
            gradient_loss=zero,
            spectral_penalty=zero,
            alpha_calibration_loss=zero,
            total_loss=zero,
            atomic_gradients=torch.zeros_like(manifold.pos),
            vacuum_values=manifold.pos.new_zeros((1,)),
            vacuum_gradients=manifold.pos.new_zeros((1, 3)),
        )

    def forward_batch(self, batch: Any) -> TrainingStepResult:
        self._maybe_prepare_precision_runtime()
        smiles = self._resolve_smiles(batch)
        module1_out = self._module1_forward_hot_path(smiles)
        device = module1_out.manifold.pos.device
        true_atom_index = self._resolve_true_atom_index(batch, device=device)
        true_row_index = self._scan_row_index(module1_out.scan.atom_indices, true_atom_index)
        # som_coordinates / alignment_score are sorted by descending effective_reactivity;
        # _build_pocket_encoding expects a rank in that sorted order.
        true_ranked_index = self._scan_row_index(module1_out.ranked_atom_indices, true_atom_index)
        exp_rate, has_exp_rate = self._resolve_exp_rate(batch, device=device, dtype=torch.float32)
        protein_data = self._resolve_protein_data(batch)
        pocket_encoding = None
        accessibility_field = None
        ddi_occupancy = None
        if protein_data is not None:
            pocket_encoding = self._build_pocket_encoding_hot_path(
                module1_out,
                int(true_ranked_index.detach().cpu().item()),
                protein_data,
            )
            accessibility_field = pocket_encoding.accessibility_state
            ddi_occupancy = protein_data.get("ddi_occupancy")

        field = module1_out.field_state.field
        manifold_pos = self._sanitize_tensor(
            module1_out.manifold.pos,
            nan=0.0,
            posinf=25.0,
            neginf=-25.0,
            clamp=(-25.0, 25.0),
        )
        effective_reactivity = self._sanitize_tensor(
            self._to_fp32(module1_out.scan.effective_reactivity),
            nan=0.0,
            posinf=100.0,
            neginf=-100.0,
            clamp=(-100.0, 100.0),
        )
        alignment_score = self._sanitize_tensor(
            self._to_fp32(module1_out.alignment_score),
            nan=0.0,
            posinf=100.0,
            neginf=-100.0,
            clamp=(-100.0, 100.0),
        )
        target_point_world = self._sanitize_tensor(
            module1_out.scan.refined_peak_points[true_row_index],
            nan=0.0,
            posinf=25.0,
            neginf=-25.0,
            clamp=(-25.0, 25.0),
        )
        q_init_internal = self._sanitize_tensor(
            field.to_internal_coords(manifold_pos).to(dtype=self.model.solver_dtype),
            nan=0.0,
            posinf=25.0,
            neginf=-25.0,
            clamp=(-25.0, 25.0),
        )
        target_point_internal = self._sanitize_tensor(
            field.to_internal_coords(target_point_world.view(1, 3)).view(-1).to(dtype=self.model.solver_dtype),
            nan=0.0,
            posinf=25.0,
            neginf=-25.0,
            clamp=(-25.0, 25.0),
        )

        low_memory_train = self.low_memory_train_mode and self.training
        if low_memory_train:
            self.last_dynamics_fallback = True
            self.last_checkpoint_fallback = True
            keep_scan_grads = bool(self.low_memory_scan_gradients)
            _q_fp32 = q_init_internal.float()
            _dev_type = _q_fp32.device.type
            # Save reactive_reference before the call so a NaN psi_atoms cannot
            # permanently corrupt the exponential moving average across batches.
            _rr_saved = self.model.hamiltonian.reactive_reference.clone()
            try:
                with torch.autocast(device_type=_dev_type, enabled=False):
                    _h_raw = self._to_fp32(self.model.hamiltonian(
                        _q_fp32,
                        torch.zeros_like(_q_fp32),
                        smiles=smiles,
                        species=module1_out.manifold.species,
                        accessibility_field=accessibility_field,
                        ddi_occupancy=ddi_occupancy,
                    ))
            finally:
                # Only restore if the call produced a non-finite reference
                # (so valid updates from healthy batches are kept).
                if not torch.isfinite(self.model.hamiltonian.reactive_reference):
                    self.model.hamiltonian.reactive_reference.copy_(_rr_saved)
            # ── Raw hamiltonian diagnostic (fires before sanitize swallows NaN) ──
            if not torch.isfinite(_h_raw):
                print(
                    f"[HAM-RAW] h_raw={_h_raw.item():.6g}  smiles={smiles!r}  "
                    f"q_range=[{_q_fp32.min().item():.3g}, {_q_fp32.max().item():.3g}]  "
                    f"reactive_ref={self.model.hamiltonian.reactive_reference.item():.4g}",
                    flush=True,
                )
            # ────────────────────────────────────────────────────────────────────
            h_initial = self._sanitize_tensor(_h_raw,
                nan=25.0,
                posinf=100.0,
                neginf=-100.0,
                clamp=(-100.0, 100.0),
            )
            target_distance = self._to_fp32(
                (q_init_internal[true_atom_index] - target_point_internal).pow(2).sum().sqrt()
            )
            scan_logits = 4.0 * self._to_fp32(effective_reactivity) + 2.0 * self._to_fp32(alignment_score)
            target_scan_prob = torch.softmax(scan_logits, dim=0)[true_row_index]
            effective_barrier = (
                0.25 * F.softplus(self._to_fp32(h_initial) / 10.0)
                + 0.5 * torch.tanh(target_distance)
                - torch.log(target_scan_prob.clamp_min(1.0e-6))
            )
            pred_rate = self._sanitize_tensor(
                torch.exp(-effective_barrier),
                nan=1.0e-6,
                posinf=1.0,
                neginf=1.0e-6,
                clamp=(1.0e-6, 1.0),
            )
            if not keep_scan_grads:
                h_initial = h_initial.detach()
                pred_rate = pred_rate.detach()
            h_final = h_initial
            ts_eigenvalues = torch.stack(
                [
                    h_initial.new_tensor(-self.loss_fn.topology_margin),
                    h_initial.new_tensor(self.loss_fn.topology_margin),
                ],
                dim=0,
            )
            sobolev_report = self._zero_sobolev_report(module1_out.manifold)
            delta_E_tensor = self._sanitize_tensor(-effective_reactivity, clamp=(-100.0, 100.0))
            if not keep_scan_grads:
                delta_E_tensor = delta_E_tensor.detach()
        else:
            pred_rate, h_initial, h_final, ts_eigenvalues = self._dynamics_summary_checkpointed(
                q_init_internal,
                target_point_internal,
                smiles=smiles,
                species=module1_out.manifold.species,
                target_atom_index=true_atom_index,
                accessibility_field=accessibility_field,
                ddi_occupancy=ddi_occupancy,
            )

            sobolev_report = self.field_optimizer(field, module1_out.manifold)
            pred_rate = self._sanitize_tensor(self._to_fp32(pred_rate), nan=1.0e-12, posinf=1.0, neginf=1.0e-12, clamp=(1.0e-12, 1.0))
            h_initial = self._sanitize_tensor(self._to_fp32(h_initial), nan=25.0, posinf=100.0, neginf=-100.0, clamp=(-100.0, 100.0))
            h_final = self._sanitize_tensor(self._to_fp32(h_final), nan=25.0, posinf=100.0, neginf=-100.0, clamp=(-100.0, 100.0))
            ts_eigenvalues = self._sanitize_tensor(self._to_fp32(ts_eigenvalues), nan=0.0, posinf=100.0, neginf=-100.0, clamp=(-100.0, 100.0))
            delta_E_tensor = self._sanitize_tensor(-effective_reactivity, clamp=(-100.0, 100.0))
            sobolev_report = FieldGradientOptimizationReport(
                gradient_loss=self._sanitize_tensor(sobolev_report.gradient_loss, clamp=(0.0, 100.0)),
                spectral_penalty=self._sanitize_tensor(sobolev_report.spectral_penalty, clamp=(0.0, 100.0)),
                alpha_calibration_loss=self._sanitize_tensor(sobolev_report.alpha_calibration_loss, clamp=(0.0, 100.0)),
                total_loss=self._sanitize_tensor(sobolev_report.total_loss, clamp=(0.0, 100.0)),
                atomic_gradients=self._sanitize_tensor(sobolev_report.atomic_gradients, clamp=(-100.0, 100.0)),
                vacuum_values=self._sanitize_tensor(sobolev_report.vacuum_values, clamp=(-100.0, 100.0)),
                vacuum_gradients=self._sanitize_tensor(sobolev_report.vacuum_gradients, clamp=(-100.0, 100.0)),
            )

        if exp_rate is None:
            # ATTNSOM/Zaretzki-style SoM datasets typically supervise the
            # reactive site, not an experimental kinetic constant. Use the
            # current prediction as the target so the kinetic branch is
            # neutral rather than silently forcing a fake exp_rate=1.0.
            exp_rate = pred_rate.detach()

        # Initialise sub-losses as zeros so they are always defined for the metrics dict,
        # whether or not protein data was provided.
        _z = torch.zeros((), dtype=torch.float32, device=device)
        pocket_coord_recon_loss = _z.clone()
        pocket_mask_loss = _z.clone()
        pocket_steric_loss = _z.clone()
        reconstruction_loss = _z.clone()
        if pocket_encoding is not None:
            pocket_coord_recon_loss = F.mse_loss(
                pocket_encoding.refined_coords.to(dtype=torch.float32),
                protein_data["coords"].to(device=device, dtype=torch.float32),
            )
            pocket_mask_loss = F.mse_loss(
                pocket_encoding.accessibility_mask.to(dtype=torch.float32),
                pocket_encoding.accessibility_output.accessibility.to(dtype=torch.float32),
            )
            pocket_steric_loss = pocket_encoding.accessibility_output.steric_loss.to(dtype=torch.float32)
            reconstruction_loss = pocket_coord_recon_loss + pocket_mask_loss + pocket_steric_loss

        # T2: flux consistency — penalise models that route more than 100 % of substrate
        # mass to metabolic sites (mass creation).  sigmoid(-delta_E) gives an unnormalised
        # per-site activation; if multiple sites are simultaneously near-certain the row sum
        # exceeds 1.0 and the NCFA propagator fires a relu penalty.
        site_fluxes = torch.sigmoid(-delta_E_tensor)
        W_flux = site_fluxes.view(1, 1, -1)   # [1 batch, 1 source node, N_atoms]
        flux_consistency_loss = self.flux_propagator.compute_flux_consistency_loss(W_flux).to(dtype=torch.float32)

        # T3: ranking loss on geometric alignment scores — a structural signal independent
        # of the thermodynamic delta_E ranking already captured by som_loss.
        # alignment_score is sorted by descending effective_reactivity; true_ranked_index
        # (computed above) is the true SoM's position in that sorted order.
        ranking_loss = self.loss_fn.ranking_loss_fn(
            alignment_score.unsqueeze(0),
            true_ranked_index,
        )
        if low_memory_train and not self.low_memory_scan_gradients:
            ranking_loss = ranking_loss.detach()

        # T5: Metabolic DAG forward pass.
        # Build per-atom Clifford multivectors: position in the vector grade (indices 1-3),
        # effective reactivity in the scalar grade (index 0).  This gives the GraNDAG edge
        # predictor a geometry- and thermodynamics-aware representation of each candidate
        # metabolic site without requiring an additional neural forward pass.
        atom_mv = embed_coordinates(manifold_pos.to(dtype=exp_rate.dtype)).clone()
        atom_mv[..., 0] = effective_reactivity
        atom_mv = atom_mv.unsqueeze(0)   # [1, N_atoms, 8]  — single-compound batch dim
        dag_output = self.dag_learner(
            atom_mv,
            # Per-source activation barriers: [1, N, 1] broadcasts against [1, N, N] prior.
            delta_g_activations=delta_E_tensor.view(1, -1, 1),
            accessibility_mask=(
                pocket_encoding.accessibility_mask.to(dtype=torch.float32)
                if pocket_encoding is not None
                else None
            ),
            physics_loss=self._to_fp32(sobolev_report.total_loss),
            flux_consistency_loss=flux_consistency_loss,
        )
        dag_causal_loss = self._sanitize_tensor(self._to_fp32(dag_output.causal_loss), clamp=(0.0, 1.0e4))

        total_loss, loss_info = self.loss_fn(
            delta_E_tensor=delta_E_tensor,
            true_som_idx=true_row_index,
            pred_rate=self._to_fp32(pred_rate),
            exp_rate=exp_rate,
            sobolev_penalty=self._to_fp32(sobolev_report.total_loss),
            H_initial=self._to_fp32(h_initial),
            H_final=self._to_fp32(h_final),
            ts_eigenvalues=self._to_fp32(ts_eigenvalues),
            flux_consistency_loss=flux_consistency_loss,
            reconstruction_loss=reconstruction_loss,
            ranking_loss=ranking_loss,
        )
        # Normalise DAG causal loss by N² so the acyclicity penalty does not explode
        # on large molecules: for an N-atom molecule the penalty grows O(N²) otherwise.
        # Also cap the post-normalisation contribution at 4.0 so the untrained DAG's
        # acyclicity penalty (raw ≈ 8000+) does not overwhelm the SoM / ranking signal
        # (~2–4) and dominate SIREN learning during early training.
        n_atoms = float(module1_out.manifold.pos.size(0))
        dag_loss_scale = max(n_atoms * n_atoms, 1.0)
        dag_contribution = (dag_causal_loss / dag_loss_scale).clamp_max(4.0)
        total_loss = self._sanitize_tensor(self._to_fp32(total_loss), clamp=(0.0, 1.0e5)) + dag_contribution

        # SoM top-1 / top-2 accuracy — true_ranked_index is the rank of the
        # ground-truth SoM in the descending-effective_reactivity-sorted list, so
        # rank 0 = correctly predicted as the most reactive atom (top-1 hit).
        _rank = int(true_ranked_index.detach().cpu().item())
        _n    = int(delta_E_tensor.numel())
        som_top1 = torch.as_tensor(1.0 if _rank == 0 else 0.0, dtype=torch.float32, device=total_loss.device)
        som_top2 = torch.as_tensor(1.0 if _rank <= 1 else 0.0, dtype=torch.float32, device=total_loss.device)
        som_top3 = torch.as_tensor(1.0 if _rank <= 2 else 0.0, dtype=torch.float32, device=total_loss.device)

        metrics = {
            "loss_total": total_loss.detach(),
            "som_top1": som_top1,
            "som_top2": som_top2,
            "som_top3": som_top3,
            "som_rank": torch.as_tensor(float(_rank), dtype=torch.float32, device=total_loss.device),
            "som_n_atoms": torch.as_tensor(float(_n), dtype=torch.float32, device=total_loss.device),
            "loss_raw_som": loss_info["som_loss"],
            "loss_raw_rank": loss_info["ranking_loss"],
            "loss_raw_kinetics": loss_info["kinetic_loss"],
            "loss_raw_physics": loss_info["physics_loss"],
            "loss_raw_topology": loss_info["topology_loss"],
            "loss_raw_flux": loss_info["flux_loss"],
            "loss_raw_reconstruction": loss_info["reconstruction_loss"],
            "weight_som": loss_info["precision"][0],
            "weight_kinetics": loss_info["precision"][1],
            "weight_physics": loss_info["precision"][2],
            "weight_topology": loss_info["precision"][3],
            "weight_flux": loss_info["precision"][4],
            "weight_reconstruction": loss_info["precision"][5],
            "pred_rate": pred_rate.detach(),
            "exp_rate": exp_rate.detach(),
            "kinetics_supervised": torch.as_tensor(
                1.0 if has_exp_rate else 0.0,
                dtype=total_loss.dtype,
                device=total_loss.device,
            ),
            "true_atom_index": true_atom_index.detach().to(dtype=torch.float32),
            "true_row_index": true_row_index.detach().to(dtype=torch.float32),
            "target_effective_reactivity": effective_reactivity[true_row_index].detach(),
            "sobolev_gradient_loss": sobolev_report.gradient_loss.detach(),
            "sobolev_spectral_penalty": sobolev_report.spectral_penalty.detach(),
            "sobolev_alpha_loss": sobolev_report.alpha_calibration_loss.detach(),
            "hamiltonian_initial": h_initial.detach(),
            "hamiltonian_final": h_final.detach(),
            "checkpoint_fallback": torch.as_tensor(
                1.0 if self.last_checkpoint_fallback else 0.0,
                dtype=total_loss.dtype,
                device=total_loss.device,
            ),
            "static_compile_active": torch.as_tensor(
                1.0 if self.static_compile_applied and len(self.compiled_module_names) > 0 else 0.0,
                dtype=total_loss.dtype,
                device=total_loss.device,
            ),
            "bf16_hot_path": torch.as_tensor(
                1.0 if self._autocast_enabled() else 0.0,
                dtype=total_loss.dtype,
                device=total_loss.device,
            ),
            "curriculum_stage": torch.as_tensor(0.0 if self.current_curriculum_stage() == "field_reconstruction" else 1.0, dtype=total_loss.dtype, device=total_loss.device),
            "dynamics_fallback": torch.as_tensor(
                1.0 if self.last_dynamics_fallback else 0.0,
                dtype=total_loss.dtype,
                device=total_loss.device,
            ),
            "loss_recon_coords": pocket_coord_recon_loss.detach(),
            "loss_recon_mask": pocket_mask_loss.detach(),
            "loss_recon_steric": pocket_steric_loss.detach(),
            "dag_causal_loss": dag_causal_loss.detach(),
            "dag_acyclicity": self._sanitize_tensor(dag_output.acyclicity.detach(), clamp=(0.0, 1.0e4)),
            "dag_sparsity": self._sanitize_tensor(dag_output.sparsity.detach(), clamp=(0.0, 1.0e4)),
            "dag_adjacency_mean": self._sanitize_tensor(dag_output.raw_adjacency.abs().mean().detach(), clamp=(0.0, 1.0e4)),
            "dag_kinetic_penalty": self._sanitize_tensor(dag_output.kinetic_penalty.detach(), clamp=(0.0, 1.0e4)),
            "dag_thermo_penalty": self._sanitize_tensor(dag_output.thermodynamic_penalty.detach(), clamp=(0.0, 1.0e4)),
            "dag_affinity_penalty": self._sanitize_tensor(dag_output.affinity_penalty.detach(), clamp=(0.0, 1.0e4)),
            "dag_access_penalty": self._sanitize_tensor(dag_output.accessibility_penalty.detach(), clamp=(0.0, 1.0e4)),
            "dag_flux_penalty": self._sanitize_tensor(dag_output.flux_penalty.detach(), clamp=(0.0, 1.0e4)),
            "dag_recon_loss": self._sanitize_tensor(dag_output.reconstruction_loss.detach(), clamp=(0.0, 1.0e4)),
            "dag_manifold_recon_loss": self._sanitize_tensor(dag_output.manifold_recon_loss.detach(), clamp=(0.0, 1.0e4)),
            "dag_manifold_density_penalty": self._sanitize_tensor(dag_output.manifold_density_penalty.detach(), clamp=(0.0, 1.0e4)),
        }

        # ── Analogical Engine ──────────────────────────────────────────────
        # Only fires when the memory bank has been populated.  Wrapped in a
        # broad try/except so a bad retrieval or RDKit failure never crashes
        # the training loop.
        if self.memory_bank.memory_embeddings is not None:
            try:
                from rdkit import Chem as _Chem
                _query_mol = _Chem.MolFromSmiles(smiles)
                if _query_mol is not None:
                    _result = self.memory_bank.retrieve_and_transport(_query_mol)
                    # Re-index pred_ana (SMILES atom order) onto the scan's
                    # descending-reactivity-sorted atom order so target_idx
                    # (= true_row_index) aligns with pred_fp (= effective_reactivity).
                    _scan_atom_idx = module1_out.scan.atom_indices.to(
                        dtype=torch.long, device=device
                    )
                    N_scan = _scan_atom_idx.numel()
                    _pred_ana_scan = torch.zeros(
                        N_scan, dtype=torch.float32, device=device
                    )
                    _transport_mapped = False
                    if _result.transport_succeeded and _result.analogical_pred.numel() > 0:
                        _pred_atom = int(_result.analogical_pred.argmax().item())
                        _hits = (_scan_atom_idx == _pred_atom).nonzero(as_tuple=True)[0]
                        if len(_hits) > 0:
                            _pred_ana_scan[int(_hits[0].item())] = 1.0
                            _transport_mapped = True
                    _ana_loss, _ana_info = self.gated_loss(
                        pred_fp=effective_reactivity,
                        pred_ana=_pred_ana_scan,
                        target_idx=true_row_index,
                        retrieval_confidence=_result.confidence,
                        transport_succeeded=_transport_mapped,
                    )
                    _ana_loss = self._sanitize_tensor(
                        self._to_fp32(_ana_loss), clamp=(0.0, 100.0)
                    )
                    total_loss = total_loss + _ana_loss
                    metrics.update({
                        "ana_loss_total": _ana_loss.detach(),
                        "ana_gate_open": torch.as_tensor(
                            _ana_info["gate_open"], dtype=torch.float32, device=device
                        ),
                        "ana_confidence": torch.as_tensor(
                            _result.confidence, dtype=torch.float32, device=device
                        ),
                        "ana_weight_fp": torch.as_tensor(
                            _ana_info["weight_physics"], dtype=torch.float32, device=device
                        ),
                        "ana_weight_ana": torch.as_tensor(
                            _ana_info["weight_analogy"], dtype=torch.float32, device=device
                        ),
                        "ana_transport_ok": torch.as_tensor(
                            1.0 if _transport_mapped else 0.0,
                            dtype=torch.float32, device=device,
                        ),
                    })
            except Exception as _ana_err:
                # Print once so we can see if the analogical engine is broken,
                # but never let it crash the physics training loop.
                import traceback as _tb
                print(f"[ANA-ERR] {type(_ana_err).__name__}: {_ana_err}", flush=True)
                _tb.print_exc()
        # ──────────────────────────────────────────────────────────────────

        if self.optimizer is not None and self.optimizer.param_groups:
            metrics["lr"] = torch.as_tensor(
                self.optimizer.param_groups[0]["lr"],
                dtype=total_loss.dtype,
                device=total_loss.device,
            )
        if self.scheduler is not None:
            metrics["wsd_scheduler_active"] = torch.as_tensor(1.0, dtype=total_loss.dtype, device=total_loss.device)
        else:
            metrics["wsd_scheduler_active"] = torch.as_tensor(0.0, dtype=total_loss.dtype, device=total_loss.device)
        return TrainingStepResult(loss=total_loss, metrics=metrics)

    def training_step(self, batch: Any) -> Optional[Dict[str, torch.Tensor]]:
        """Performs a single forward and backward pass for one batch of data."""
        if self.optimizer is None:
            self.configure_optimizers()
        assert self.optimizer is not None
        batch = self._move_to_device(batch)
        result = self.forward_batch(batch)

        if not torch.isfinite(result.loss):
            import warnings
            warnings.warn(f"Non-finite loss ({result.loss.item()}) — skipping batch.", UserWarning, stacklevel=2)
            # Explicitly delete tensor to free memory, as the graph is not cleared by backward()
            del result
            if self._module_device().type == "cuda":
                torch.cuda.empty_cache()
            return None

        # Normalize loss for accumulation and perform backward pass
        loss = result.loss / self.gradient_accumulation_steps
        loss.backward()

        metrics = dict(result.metrics)
        # Grad norm is calculated in fit_epoch, so just add a placeholder
        metrics["grad_norm"] = torch.zeros((), dtype=result.loss.dtype, device=result.loss.device)
        return metrics

    def validation_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        batch = self._move_to_device(batch)
        result = self.forward_batch(batch)
        metrics = dict(result.metrics)
        metrics["grad_norm"] = torch.zeros((), dtype=result.loss.dtype, device=result.loss.device)
        return metrics

    def fit_epoch(self, dataloader, *, train: bool = True, log_every: int = 0) -> Dict[str, float]:
        reducer: Dict[str, List[float]] = {}
        self.train(mode=train)
        
        valid_batches_processed = 0
        if train:
            if self.optimizer is None: self.configure_optimizers()
            assert self.optimizer is not None
            self.optimizer.zero_grad(set_to_none=True)

        total_batches = len(dataloader)
        for i, batch in enumerate(dataloader, start=1):
            if train:
                metrics = self.training_step(batch)
                if metrics is None:
                    continue  # Skip batch if loss was NaN

                # Log metrics from the successful forward/backward pass
                for key, value in metrics.items():
                    if torch.is_tensor(value):
                        reducer.setdefault(key, []).append(float(value.detach().cpu().item()))
                
                valid_batches_processed += 1
                is_accumulation_step = (valid_batches_processed % self.gradient_accumulation_steps) == 0
                
                if is_accumulation_step:
                    grad_norm = self.clip_gradients()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    reducer.setdefault("grad_norm", []).append(grad_norm)

                self.global_step_counter.add_(1)
                if log_every > 0 and (i == 1 or i % log_every == 0 or i == total_batches):
                    running = {key: sum(values) / max(len(values), 1) for key, values in reducer.items()}
                    _ana_active = "ana_loss_total" in running
                    _ana_str = (
                        f" | ana_loss={running.get('ana_loss_total', float('nan')):.4g}"
                        f" gate={running.get('ana_gate_open', float('nan')):.2f}"
                        f" conf={running.get('ana_confidence', float('nan')):.3f}"
                        f" w_fp={running.get('ana_weight_fp', float('nan')):.3f}"
                        f" w_ana={running.get('ana_weight_ana', float('nan')):.3f}"
                        f" t_ok={running.get('ana_transport_ok', float('nan')):.2f}"
                    ) if _ana_active else ""
                    print(
                        f"batch={i}/{total_batches} "
                        f"loss_total={running.get('loss_total', float('nan')):.6g} "
                        f"top1={running.get('som_top1', float('nan')):.2%} "
                        f"top2={running.get('som_top2', float('nan')):.2%} "
                        f"pred_rate={running.get('pred_rate', float('nan')):.6g} "
                        f"dag_loss={running.get('dag_causal_loss', float('nan')):.6g}"
                        f"{_ana_str}",
                        flush=True,
                    )
            
            else:  # Validation loop
                metrics = self.validation_step(batch)
                for key, value in metrics.items():
                    if torch.is_tensor(value):
                        reducer.setdefault(key, []).append(float(value.detach().cpu().item()))
                if log_every > 0 and (i == 1 or i % log_every == 0 or i == total_batches):
                    running = {key: sum(values) / max(len(values), 1) for key, values in reducer.items()}
                    print(
                        f"val_batch={i}/{total_batches} "
                        f"loss_total={running.get('loss_total', float('nan')):.6g} "
                        f"top1={running.get('som_top1', float('nan')):.2%} "
                        f"top2={running.get('som_top2', float('nan')):.2%} "
                        f"pred_rate={running.get('pred_rate', float('nan')):.6g} "
                        f"dag_loss={running.get('dag_causal_loss', float('nan')):.6g}",
                        flush=True,
                    )
        
        # Handle case where the last batches didn't trigger an optimizer step
        if train and (valid_batches_processed % self.gradient_accumulation_steps) != 0:
            grad_norm = self.clip_gradients()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            reducer.setdefault("grad_norm", []).append(grad_norm)

        return {key: sum(values) / max(len(values), 1) for key, values in reducer.items()}


    def evaluate_epoch(self, dataloader, *, log_every: int = 0) -> Dict[str, float]:
        """Run one full eval pass and return accuracy + loss metrics.

        Returns the same dict as fit_epoch(train=False) but also prints a
        human-readable accuracy summary at the end.
        """
        metrics = self.fit_epoch(dataloader, train=False, log_every=log_every)
        top1  = metrics.get("som_top1",  float("nan"))
        top2  = metrics.get("som_top2",  float("nan"))
        top3  = metrics.get("som_top3",  float("nan"))
        rank  = metrics.get("som_rank",  float("nan"))
        n_atoms = metrics.get("som_n_atoms", float("nan"))
        print(
            f"\n── SoM accuracy ──────────────────────────────\n"
            f"  top-1 : {top1:.1%}\n"
            f"  top-2 : {top2:.1%}\n"
            f"  top-3 : {top3:.1%}\n"
            f"  mean rank  : {rank:.1f} / {n_atoms:.0f} atoms\n"
            f"──────────────────────────────────────────────\n",
            flush=True,
        )
        return metrics


def load_compound_records(dataset_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and "compounds" in payload:
        records = payload["compounds"]
    else:
        raise ValueError("Dataset JSON must be a list or a dict containing 'compounds'")
    if not isinstance(records, list):
        raise ValueError("Dataset compounds payload must be a list")
    return [dict(record) for record in records]


__all__ = [
    "Metabolic_Causal_Trainer",
    "OptimizerGroupSummary",
    "TrainingStepResult",
    "load_compound_records",
]
