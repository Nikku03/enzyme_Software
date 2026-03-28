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

# _StopRecomputationError is a PyTorch-internal control-flow signal used by
# non-reentrant checkpointing (use_reentrant=False) to stop forward
# recomputation at the right boundary.  It MUST propagate back to
# checkpoint.checkpoint() and must never be caught by user-level except
# handlers.  Catching it causes fallback code to run inside the recomputation
# frame, which triggers: AssertionError: target_frame.early_stop is set.
try:
    from torch.utils.checkpoint import _StopRecomputationError as _CheckpointStop
except ImportError:
    _CheckpointStop = None  # older PyTorch — sentinel that is never matched

from nexus.core.dynamics_engine import NEXUS_Dynamics_Engine
from nexus.core.field_optimizer import Field_Gradient_Optimizer, FieldGradientOptimizationReport
from nexus.core.flux_analysis import NCFAFluxPropagator
from nexus.core.inference import NEXUS_Module1_Output
from nexus.layers.dag_learner import MetabolicDAGLearner
from nexus.physics.clifford_math import embed_coordinates
from nexus.pocket.ddi import DDIOccupancyState
from nexus.reasoning.analogical_fusion import (
    HomoscedasticArbiterLoss,
    NexusDualDecoder,
    PGWCrossAttention,
)
from nexus.reasoning.hyperbolic_memory import HyperbolicMemoryBank
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


def compute_masked_morphism_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).to(dtype=torch.float32)
    targets_f = targets.to(dtype=torch.float32)
    mask_f = mask.to(dtype=torch.float32)
    preds_masked = preds * mask_f
    targets_masked = targets_f * mask_f

    tp = (preds_masked * targets_masked).sum(dim=(0, 1))
    fp = (preds_masked * (1.0 - targets_masked)).sum(dim=(0, 1))
    fn = ((1.0 - preds_masked) * targets_masked).sum(dim=(0, 1))
    eps = 1.0e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_per_class = 2.0 * (precision * recall) / (precision + recall + eps)
    macro_f1 = f1_per_class.mean()
    return macro_f1, f1_per_class


def compute_analogical_ood_metrics(
    *,
    retrieval_confidence: float,
    retrieval_mix_entropy: float,
    retrieval_mix_count: int,
    retrieval_diversity_score: float,
    transport_backend: str,
    transport_succeeded: bool,
    transported_mass: float,
    physics_analogy_agreement: float,
    fusion_available: bool,
    neuralgw_confidence: float,
) -> tuple[float, float, float]:
    max_entropy = math.log(max(int(retrieval_mix_count), 1)) if int(retrieval_mix_count) > 1 else 0.0
    entropy_norm = float(retrieval_mix_entropy / max(max_entropy, 1.0e-8)) if max_entropy > 0.0 else 0.0
    low_conf = 1.0 - max(0.0, min(float(retrieval_confidence), 1.0))
    low_diversity = 1.0 - max(0.0, min(float(retrieval_diversity_score), 1.0))
    disagreement = 1.0 - max(0.0, min(float(physics_analogy_agreement), 1.0))
    support_quality = max(0.0, min(float(transported_mass) / 0.25, 1.0))
    low_transport = 1.0 - support_quality
    backend_penalty = 1.0 if (
        (not transport_succeeded)
        or ("fallback" in str(transport_backend))
        or str(transport_backend) in {"prefilter_reject", "pgw_error", "pgw_low_mass", "mcs"}
    ) else 0.0
    fusion_penalty = 0.0 if fusion_available else 1.0
    ngw_penalty = 1.0 - max(0.0, min(float(neuralgw_confidence), 1.0))
    ood_score = (
        low_conf
        + entropy_norm
        + low_diversity
        + disagreement
        + low_transport
        + backend_penalty
        + fusion_penalty
        + ngw_penalty
    ) / 8.0
    hard_case = 1.0 if ood_score >= 0.55 else 0.0
    abstain = 1.0 if (ood_score >= 0.75 or retrieval_confidence < 0.35 or (not fusion_available and not transport_succeeded)) else 0.0
    return float(ood_score), float(hard_case), float(abstain)


class Metabolic_Causal_Trainer(nn.Module):
    _PRED_RATE_MAX = 1.0e12

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
        dag_loss_weight: float = 1.0,
        dag_loss_cap: float = 4.0,
        analogical_loss_weight: float = 1.0,
        physics_cache_mode: str = "off",
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
        self.dag_loss_weight = float(max(dag_loss_weight, 0.0))
        self.dag_loss_cap = float(max(dag_loss_cap, 0.0))
        self.analogical_loss_weight = float(max(analogical_loss_weight, 0.0))
        self.physics_cache_mode = str(physics_cache_mode).strip().lower() or "off"
        if self.physics_cache_mode not in {"off", "cached", "hybrid"}:
            self.physics_cache_mode = "off"
        self.physics_cache: Dict[str, Dict[str, Any]] = {}
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.total_training_steps: Optional[int] = None
        self.last_checkpoint_fallback = False
        self.last_dynamics_fallback = False
        self.last_kinetics_debug: Dict[str, torch.Tensor] = {}
        self.static_compile_applied = False
        self.compiled_module_names: List[str] = []
        self.register_buffer("global_step_counter", torch.zeros((), dtype=torch.long))
        self.gated_loss = GatedAnalogicalGodLoss(
            confidence_threshold=0.85,
            peak_threshold=0.09,
        )
        # Memory bank is populated externally (trainer.memory_bank.populate_from_mols)
        # before training begins.  Left empty here so training still runs without it.
        self.memory_bank = HyperbolicMemoryBank(device="cpu")
        self.current_epoch_index = 0
        if self.memory_bank.pgw is not None:
            # Register the NeuralGW student so it trains with the rest of the model.
            self.neuralgw_approximator = self.memory_bank.pgw.neural_approximator
        self.pgw_cross_attention = PGWCrossAttention(hidden_dim=32)
        self.analogical_dual_decoder = NexusDualDecoder(hidden_dim=32)
        self.analogical_arbiter = HomoscedasticArbiterLoss()
        self.analogical_trace_enabled = False
        self.analogical_trace_path: Optional[Path] = None
        self._active_phase = "train"
        self._active_batch_index = 0
        self._active_total_batches = 0
        self._maybe_prepare_precision_runtime()
        self._validate_wsd_config()

    def _module_device(self) -> torch.device:
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def sync_memory_bank_device(self, device: torch.device | str | None = None) -> torch.device:
        resolved = torch.device(device) if device is not None else self._module_device()
        self.memory_bank.set_device(resolved)
        if self.memory_bank.pgw is not None:
            self.neuralgw_approximator = self.memory_bank.pgw.neural_approximator
        return resolved

    def _autocast_enabled(self) -> bool:
        return self.enable_bf16_hot_path and self._module_device().type == "cuda"

    def _autocast_context(self):
        if not self._autocast_enabled():
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @staticmethod
    def physics_cache_key(smiles: str, true_atom_index: int) -> str:
        return f"{smiles}::atom={int(true_atom_index)}"

    def set_physics_cache(self, entries: Mapping[str, Any], *, mode: Optional[str] = None) -> int:
        if mode is not None:
            resolved_mode = str(mode).strip().lower() or "off"
            self.physics_cache_mode = resolved_mode if resolved_mode in {"off", "cached", "hybrid"} else "off"
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in dict(entries).items():
            if not isinstance(key, str):
                continue
            if isinstance(value, Mapping):
                normalized[key] = dict(value)
        self.physics_cache = normalized
        return len(self.physics_cache)

    def load_physics_cache(self, path: str | Path, *, mode: Optional[str] = None) -> int:
        payload = torch.load(Path(path), map_location="cpu")
        entries = payload.get("entries", payload) if isinstance(payload, Mapping) else {}
        if not isinstance(entries, Mapping):
            raise TypeError("Physics cache must be a mapping or contain an 'entries' mapping")
        return self.set_physics_cache(entries, mode=mode)

    def set_analogical_trace(
        self,
        path: str | Path,
        *,
        enabled: bool = True,
        truncate: bool = False,
    ) -> None:
        self.analogical_trace_enabled = bool(enabled)
        self.analogical_trace_path = Path(path)
        self.analogical_trace_path.parent.mkdir(parents=True, exist_ok=True)
        if truncate and self.analogical_trace_path.exists():
            self.analogical_trace_path.unlink()

    @staticmethod
    def _trace_scalar(value: Any) -> Any:
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() == 1:
                return float(value.item())
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return value

    def _append_analogical_trace(self, payload: Mapping[str, Any]) -> None:
        if not self.analogical_trace_enabled or self.analogical_trace_path is None:
            return
        normalized = {str(key): self._trace_scalar(value) for key, value in payload.items()}
        with self.analogical_trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(normalized) + "\n")

    def _lookup_physics_cache(
        self,
        smiles: str,
        true_atom_index: torch.Tensor | int,
        *,
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self.physics_cache_mode == "off" or not self.physics_cache:
            return None
        key = self.physics_cache_key(smiles, int(torch.as_tensor(true_atom_index).detach().cpu().item()))
        entry = self.physics_cache.get(key)
        if entry is None:
            if self.physics_cache_mode == "cached":
                raise KeyError(f"Missing physics cache entry for {key}")
            return None
        out: Dict[str, torch.Tensor] = {}
        for name, value in entry.items():
            if isinstance(value, (str, bytes)) or value is None:
                continue
            if torch.is_tensor(value):
                tensor = value.detach().to(device=device)
            else:
                try:
                    tensor = torch.as_tensor(value, device=device)
                except (TypeError, ValueError):
                    continue
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.float32)
            out[name] = tensor
        return out

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

    def _module1_node_multivectors(self, module1_out: NEXUS_Module1_Output) -> torch.Tensor:
        with self._autocast_context():
            _, latent = module1_out.field_state.field.raw_query(
                module1_out.manifold.pos,
                return_latent=True,
            )
        if latent.ndim == 3:
            return latent.mean(dim=-2)
        return latent

    def _project_module1_hyperbolic(self, module1_out: NEXUS_Module1_Output) -> torch.Tensor:
        node_multivectors = self._module1_node_multivectors(module1_out)
        return self.gated_loss.hyperbolic_projector(node_multivectors)

    def encode_smiles_for_memory_bank(self, smiles: str) -> Dict[str, torch.Tensor]:
        # The manifold refiner internally calls torch.autograd.grad, so it
        # cannot run under a surrounding no_grad context even for inference-
        # only memory-bank encoding.
        with torch.enable_grad():
            seed = self.model.module1.agency(smiles)
            manifold = self.model.module1.refiner(seed)
        with torch.no_grad():
            _ = self.model.module1.symmetry_engine(manifold)
            with self._autocast_context():
                field_state = self.model.module1.field_engine.build_state(manifold)
                _, latent = field_state.field.raw_query(manifold.pos, return_latent=True)
            if latent.ndim == 3:
                node_multivectors = latent.mean(dim=-2)
            else:
                node_multivectors = latent
            embedding = self.gated_loss.hyperbolic_projector(node_multivectors)
        return {
            "graph_embedding": embedding.detach().float().cpu(),
            "node_multivectors": node_multivectors.detach().float().cpu(),
        }

    def encode_mol_for_memory_bank(self, mol) -> Dict[str, torch.Tensor]:
        """Fast bank encoding that reads 3D positions from an existing SDF conformer.

        Skips both SMILES→3D generation (MolLlama + StructuralDiffusion) and
        MACE-OFF geometry optimisation — the two dominant costs in
        encode_smiles_for_memory_bank.  For a 457-molecule CYP3A4 bank this
        reduces bank population from ~30-90 minutes to ~30-90 seconds.

        Falls back to encode_smiles_for_memory_bank for molecules that have no
        embedded 3D conformer.
        """
        from rdkit import Chem as _Chem
        from nexus.core.generative_agency import NEXUS_Seed
        from nexus.core.manifold_refiner import Refined_NEXUS_Manifold
        from nexus.models.mol_llama_wrapper import LatentBlueprint

        if mol.GetNumConformers() == 0:
            return self.encode_smiles_for_memory_bank(_Chem.MolToSmiles(mol))

        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        device = self._module_device()

        # 3D positions straight from the SDF conformer.
        coords = [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(n_atoms)
        ]
        pos = torch.tensor(coords, dtype=torch.float32, device=device)

        species = torch.tensor(
            [a.GetAtomicNum() for a in mol.GetAtoms()],
            dtype=torch.long,
            device=device,
        )
        atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]

        # Chirality codes — same logic as NEXT_Mol_Generative_Agency.
        chirality_codes = torch.zeros(n_atoms, dtype=torch.long, device=device)
        for atom in mol.GetAtoms():
            tag = str(atom.GetChiralTag())
            if tag == "CHI_TETRAHEDRAL_CCW":
                chirality_codes[atom.GetIdx()] = 1
            elif tag == "CHI_TETRAHEDRAL_CW":
                chirality_codes[atom.GetIdx()] = -1

        formal_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        mol_smiles = _Chem.MolToSmiles(mol)

        # Dummy LatentBlueprint.  The field engine only reads
        # manifold.seed.metadata["formal_charge"] — the blueprint tensors are
        # never accessed after the positions have been generated.
        latent_dim = self.model.module1.agency.latent_dim
        dummy_blueprint = LatentBlueprint(
            sequence=torch.zeros(1, 1, latent_dim),
            pooled=torch.zeros(1, latent_dim),
            token_ids=torch.zeros(1, 1, dtype=torch.long),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            smiles=mol_smiles,
            source="sdf_fast_path",
            chirality_signature=torch.zeros(8),
        )

        seed = NEXUS_Seed(
            pos=pos,
            z=species,
            latent_blueprint=dummy_blueprint,
            smiles=mol_smiles,
            atom_symbols=atom_symbols,
            chirality_codes=chirality_codes,
            metadata={"formal_charge": formal_charge, "fast_path": True},
        )

        # Zero energy + forces — MACE-OFF is skipped; the topology kernel
        # uses forces only for RBF feature computation, not for gradient flow.
        manifold = Refined_NEXUS_Manifold(
            pos=pos,
            energy=torch.zeros((), dtype=pos.dtype, device=device),
            forces=torch.zeros_like(pos),
            species=species,
            seed=seed,
        )

        with torch.no_grad():
            with self._autocast_context():
                field_state = self.model.module1.field_engine.build_state(manifold)
                _, latent = field_state.field.raw_query(pos, return_latent=True)
            if latent.ndim == 3:
                node_multivectors = latent.mean(dim=-2)
            else:
                node_multivectors = latent
            embedding = self.gated_loss.hyperbolic_projector(node_multivectors)

        return {
            "graph_embedding": embedding.detach().float().cpu(),
            "node_multivectors": node_multivectors.detach().float().cpu(),
        }

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
    _NO_DECAY_MODULE_KEYWORDS = frozenset({
        "field_engine",
        "siren_field",
        "hamiltonian",
        "navigator",
        "potential",
        "quantum_enforcer",
        "symmetry_engine",
        "refiner",
        "solver",
    })
    _NO_DECAY_KEYWORDS = frozenset({
        "bias",
        "norm",
        "embedding",
        "log_vars",
        "log_s",
        "log_var",
        "batchnorm",
        "layernorm",
        "instancenorm",
        "groupnorm",
    })
    # Heavy 512-dim layers: SIREN MLP, attention heads, DAG learner, hyper-network.
    # These get GaLore rank-128 projection.  Everything else in the electronic group
    # gets rank-32.  Only applies to ndim >= 2 parameters (weight matrices).
    _GALORE_DEEP_KEYWORDS = frozenset({"siren_field", "attention", "dag_learner", "hyper_net"})

    def _should_decay_param(self, full_name: str, param: nn.Parameter) -> bool:
        name = full_name.lower()
        if any(kw in name for kw in self._PHYSICS_KEYWORDS):
            return False
        if any(kw in name for kw in self._QUERY_KEYWORDS):
            return False
        if any(kw in name for kw in self._NO_DECAY_MODULE_KEYWORDS):
            return False
        if any(kw in name for kw in self._NO_DECAY_KEYWORDS):
            return False
        return param.ndim >= 2

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
            + list(getattr(self, "neuralgw_approximator", nn.Module()).parameters())
            + list(self.pgw_cross_attention.parameters())
            + list(self.analogical_dual_decoder.parameters())
            + list(self.analogical_arbiter.parameters())
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
            # Plain AdamW with targeted decay:
            # decay only dense reasoning weights, never physics controls, query-engine
            # routing, embeddings, norms, biases, or homoscedastic balance scalars.
            decay_params: List[nn.Parameter] = []
            no_decay_params: List[nn.Parameter] = []
            seen: set[int] = set()
            for source_prefix, module in [
                ("model", self.model),
                ("dag_learner", self.dag_learner),
                ("gated_loss", self.gated_loss),
                ("loss_fn", self.loss_fn),
                ("neuralgw", getattr(self, "neuralgw_approximator", None)),
                ("pgw_cross_attention", self.pgw_cross_attention),
                ("analogical_dual_decoder", self.analogical_dual_decoder),
                ("analogical_arbiter", self.analogical_arbiter),
            ]:
                if module is None:
                    continue
                for name, param in module.named_parameters():
                    if not param.requires_grad:
                        continue
                    pid = id(param)
                    if pid in seen:
                        continue
                    seen.add(pid)
                    full_name = f"{source_prefix}.{name}"
                    if self._should_decay_param(full_name, param):
                        decay_params.append(param)
                    else:
                        no_decay_params.append(param)

            param_groups = []
            if decay_params:
                param_groups.append({"params": decay_params, "lr": 1.0e-4, "weight_decay": 1.0e-5})
            if no_decay_params:
                param_groups.append({"params": no_decay_params, "lr": 1.0e-4, "weight_decay": 0.0})
            self.optimizer = torch.optim.AdamW(param_groups, lr=1.0e-4, weight_decay=1.0e-5)
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
        prebuilt_field=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dynamics_summary_mode == "lite":
            self._set_last_kinetics_debug()
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
        # Inject prebuilt field so hamiltonian.compute_potential_energy skips the
        # expensive field_engine call on every ODE solver step inside navigator.
        if prebuilt_field is not None:
            self.model.hamiltonian._prebuilt_field_override = prebuilt_field
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
                self._set_last_kinetics_debug(
                    classical_barrier=kinetics.classical_barrier,
                    delta_g_dagger=kinetics.delta_g_dagger,
                    effective_delta_g_dagger=kinetics.effective_delta_g_dagger,
                    wigner_kappa=kinetics.wigner_kappa,
                    transmission_coefficient=kinetics.transmission_coefficient,
                    instanton_correction=kinetics.instanton_correction,
                    metabolic_rate=kinetics.metabolic_rate,
                    quantum_rate_rpmd=kinetics.quantum_rate_rpmd,
                    ts_valid=1.0 if navigation.best.ts_candidate is not None and navigation.best.ts_candidate.is_transition_state else 0.0,
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
                # Detach ts_eigenvalues to break the backward path through the
                # ts_search Hessian (create_graph=True) → SIREN grid chain.
                # When the outer loss.backward() recomputes through that path,
                # _prebuilt_field_override is already None (cleared by finally),
                # so the field rebuilds on a 1728-pt grid producing different
                # shapes/dtypes — the "saved [N,N,3] float64 vs recomputed
                # [N,3] float32" checkpoint mismatch.  The eigenvalue VALUES
                # still flow into the topology loss; only the Hessian-gradient
                # path is cut, which is noisy and extremely expensive anyway.
                ts_eigenvalues = kinetics.ts_eigenvalues.detach()
                if ts_eigenvalues.numel() < 2:
                    ts_eigenvalues = F.pad(ts_eigenvalues, (0, 2 - ts_eigenvalues.numel()))
                pred_rate = kinetics.quantum_rate_rpmd.clamp_min(1.0e-12)
                if not bool(torch.isfinite(pred_rate).all().item()) or float(pred_rate.detach().cpu().item()) <= 0.0:
                    pred_rate = kinetics.metabolic_rate.clamp_min(1.0e-12)
                return pred_rate, h_initial, h_final, ts_eigenvalues
            except Exception as _dyn_err:
                # ── checkpoint control-flow guard ──────────────────────────
                # Re-raise PyTorch-internal signals before ANY fallback work.
                # If _StopRecomputationError reaches this handler, the broad
                # except would run self.model.hamiltonian() inside the
                # recomputation frame → AssertionError: early_stop is set.
                if _CheckpointStop is not None and isinstance(_dyn_err, _CheckpointStop):
                    raise
                # Belt-and-suspenders: secondary failure if stop already leaked
                if isinstance(_dyn_err, AssertionError) and "early_stop" in str(_dyn_err):
                    raise
                # ──────────────────────────────────────────────────────────
                import traceback as _tb
                print("\n[!!!] PHYSICS SURROGATE CRASH DETECTED [!!!]")
                _tb.print_exc()
                print(f"[!!!] smiles={smiles!r}  q_shape={q_init_internal.shape}  "
                      f"dtype={q_init_internal.dtype}  device={q_init_internal.device}")
                print("[!!!] -------------------------------- [!!!]\n", flush=True)
                self.last_dynamics_fallback = True
                self._set_last_kinetics_debug()
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
            # Always clear the override so later calls don't accidentally reuse a
            # stale (potentially freed) field tensor.
            self.model.hamiltonian._prebuilt_field_override = None

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
        prebuilt_field=None,
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
                prebuilt_field=prebuilt_field,
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

    def _set_last_kinetics_debug(
        self,
        *,
        classical_barrier: torch.Tensor | float = 0.0,
        delta_g_dagger: torch.Tensor | float = 0.0,
        effective_delta_g_dagger: torch.Tensor | float = 0.0,
        wigner_kappa: torch.Tensor | float = 1.0,
        transmission_coefficient: torch.Tensor | float = 1.0,
        instanton_correction: torch.Tensor | float = 1.0,
        metabolic_rate: torch.Tensor | float = 1.0e-12,
        quantum_rate_rpmd: torch.Tensor | float = 1.0e-12,
        ts_valid: torch.Tensor | float = 0.0,
    ) -> None:
        device = self._module_device()
        self.last_kinetics_debug = {
            "kinetics_classical_barrier": self._to_fp32(torch.as_tensor(classical_barrier, device=device)).detach(),
            "kinetics_delta_g_dagger": self._to_fp32(torch.as_tensor(delta_g_dagger, device=device)).detach(),
            "kinetics_effective_delta_g_dagger": self._to_fp32(torch.as_tensor(effective_delta_g_dagger, device=device)).detach(),
            "kinetics_wigner_kappa": self._to_fp32(torch.as_tensor(wigner_kappa, device=device)).detach(),
            "kinetics_transmission": self._to_fp32(torch.as_tensor(transmission_coefficient, device=device)).detach(),
            "kinetics_instanton_correction": self._to_fp32(torch.as_tensor(instanton_correction, device=device)).detach(),
            "kinetics_metabolic_rate": self._to_fp32(torch.as_tensor(metabolic_rate, device=device)).detach(),
            "kinetics_quantum_rate_rpmd": self._to_fp32(torch.as_tensor(quantum_rate_rpmd, device=device)).detach(),
            "kinetics_ts_valid": self._to_fp32(torch.as_tensor(ts_valid, device=device)).detach(),
        }

    def forward_batch(self, batch: Any) -> TrainingStepResult:
        self._maybe_prepare_precision_runtime()
        smiles = self._resolve_smiles(batch)
        analogical_trace: Dict[str, Any] | None = None
        module1_out = self._module1_forward_hot_path(smiles)
        device = module1_out.manifold.pos.device
        true_atom_index = self._resolve_true_atom_index(batch, device=device)
        true_row_index = self._scan_row_index(module1_out.scan.atom_indices, true_atom_index)
        physics_cache_entry = self._lookup_physics_cache(
            smiles,
            true_atom_index,
            device=device,
        )
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
        if physics_cache_entry is not None:
            self.last_dynamics_fallback = bool(
                physics_cache_entry.get(
                    "dynamics_fallback",
                    torch.zeros((), dtype=torch.float32, device=device),
                ).detach().cpu().item()
                >= 0.5
            )
            self.last_checkpoint_fallback = bool(
                physics_cache_entry.get(
                    "checkpoint_fallback",
                    torch.ones((), dtype=torch.float32, device=device),
                ).detach().cpu().item()
                >= 0.5
            )
            self._set_last_kinetics_debug(
                classical_barrier=physics_cache_entry.get("kinetics_classical_barrier", 0.0),
                delta_g_dagger=physics_cache_entry.get("kinetics_delta_g_dagger", 0.0),
                effective_delta_g_dagger=physics_cache_entry.get("kinetics_effective_delta_g_dagger", 0.0),
                wigner_kappa=physics_cache_entry.get("kinetics_wigner_kappa", 1.0),
                transmission_coefficient=physics_cache_entry.get("kinetics_transmission", 1.0),
                instanton_correction=physics_cache_entry.get("kinetics_instanton_correction", 1.0),
                metabolic_rate=physics_cache_entry.get("kinetics_metabolic_rate", 1.0e-12),
                quantum_rate_rpmd=physics_cache_entry.get("kinetics_quantum_rate_rpmd", 1.0e-12),
                ts_valid=physics_cache_entry.get("kinetics_ts_valid", 0.0),
            )
            pred_rate = self._sanitize_tensor(
                self._to_fp32(physics_cache_entry.get("pred_rate", 1.0e-12)),
                nan=1.0e-12,
                posinf=self._PRED_RATE_MAX,
                neginf=1.0e-12,
                clamp=(1.0e-12, self._PRED_RATE_MAX),
            )
            pred_rate_raw = self._sanitize_tensor(
                self._to_fp32(physics_cache_entry.get("pred_rate_raw", pred_rate)),
                nan=1.0e-12,
                posinf=self._PRED_RATE_MAX,
                neginf=1.0e-12,
                clamp=(1.0e-12, self._PRED_RATE_MAX),
            )
            h_initial = self._sanitize_tensor(
                self._to_fp32(physics_cache_entry.get("hamiltonian_initial", 0.0)),
                nan=2.5,
                posinf=100.0,
                neginf=-100.0,
                clamp=(-100.0, 100.0),
            )
            h_final = self._sanitize_tensor(
                self._to_fp32(physics_cache_entry.get("hamiltonian_final", h_initial)),
                nan=2.5,
                posinf=100.0,
                neginf=-100.0,
                clamp=(-100.0, 100.0),
            )
            ts_eigenvalues = self._sanitize_tensor(
                self._to_fp32(physics_cache_entry.get("ts_eigenvalues", torch.zeros(2, device=device))),
                nan=0.0,
                posinf=100.0,
                neginf=-100.0,
                clamp=(-100.0, 100.0),
            )
            if ts_eigenvalues.numel() < 2:
                ts_eigenvalues = F.pad(ts_eigenvalues.view(-1), (0, 2 - ts_eigenvalues.numel()))
            else:
                ts_eigenvalues = ts_eigenvalues.view(-1)
            # Cached-physics mode already provides the expensive dynamics targets.
            # Skip the live Sobolev field optimizer here; it rebuilds a large
            # higher-order graph and largely defeats the cache's memory savings.
            sobolev_report = self._zero_sobolev_report(module1_out.manifold)
            delta_E_tensor = self._sanitize_tensor(-effective_reactivity, clamp=(-100.0, 100.0))
        elif low_memory_train:
            self.last_dynamics_fallback = True
            self.last_checkpoint_fallback = True
            self._set_last_kinetics_debug()
            keep_scan_grads = bool(self.low_memory_scan_gradients)
            _q_fp32 = q_init_internal.float()
            _dev_type = _q_fp32.device.type
            # Save reactive_reference before the call so a NaN psi_atoms cannot
            # permanently corrupt the exponential moving average across batches.
            _rr_saved = self.model.hamiltonian.reactive_reference.clone()
            # Inject the already-computed SIREN field so the Hamiltonian skips
            # its quantum_enforcer rebuild (1000-point grid → 4 GB OOM on L4).
            self.model.hamiltonian._prebuilt_field_override = field
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
                self.model.hamiltonian._prebuilt_field_override = None
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
            _n_atoms = max(int(_q_fp32.shape[0]), 1)
            h_initial = self._sanitize_tensor(_h_raw / _n_atoms,
                nan=2.5,
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
            pred_rate_raw = torch.exp(-effective_barrier)
            pred_rate = self._sanitize_tensor(
                pred_rate_raw,
                nan=1.0e-6,
                posinf=self._PRED_RATE_MAX,
                neginf=1.0e-6,
                clamp=(1.0e-6, self._PRED_RATE_MAX),
            )
            if not keep_scan_grads:
                h_initial = h_initial.detach()
                pred_rate = pred_rate.detach()
                pred_rate_raw = pred_rate_raw.detach()
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
                prebuilt_field=field,
            )
            pred_rate_raw = self._to_fp32(pred_rate)

            sobolev_report = self.field_optimizer(field, module1_out.manifold)
            pred_rate = self._sanitize_tensor(
                self._to_fp32(pred_rate),
                nan=1.0e-12,
                posinf=self._PRED_RATE_MAX,
                neginf=1.0e-12,
                clamp=(1.0e-12, self._PRED_RATE_MAX),
            )
            _n_atoms_dyn = max(int(q_init_internal.shape[0]), 1)
            h_initial = self._sanitize_tensor(self._to_fp32(h_initial) / _n_atoms_dyn, nan=2.5, posinf=100.0, neginf=-100.0, clamp=(-100.0, 100.0))
            h_final = self._sanitize_tensor(self._to_fp32(h_final) / _n_atoms_dyn, nan=2.5, posinf=100.0, neginf=-100.0, clamp=(-100.0, 100.0))
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
        pred_rate_raw = self._sanitize_tensor(
            self._to_fp32(pred_rate_raw),
            nan=1.0e-12,
            posinf=self._PRED_RATE_MAX,
            neginf=1.0e-12,
            clamp=(1.0e-12, self._PRED_RATE_MAX),
        )
        pred_rate_log10 = torch.log10(pred_rate_raw.clamp_min(1.0e-12))

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
        dag_contribution = (dag_causal_loss / dag_loss_scale).clamp_max(self.dag_loss_cap)
        dag_contribution = dag_contribution * self.dag_loss_weight
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
            "som_top1_fp": som_top1,
            "som_top2_fp": som_top2,
            "som_top3_fp": som_top3,
            "som_rank_fp": torch.as_tensor(float(_rank), dtype=torch.float32, device=total_loss.device),
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
            "pred_rate_raw": pred_rate_raw.detach(),
            "pred_rate_log10": pred_rate_log10.detach(),
            "physics_cache_hit": torch.as_tensor(
                1.0 if physics_cache_entry is not None else 0.0,
                dtype=total_loss.dtype,
                device=total_loss.device,
            ),
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
            "dag_loss_contribution": dag_contribution.detach(),
            "dag_loss_weight": torch.as_tensor(
                self.dag_loss_weight,
                dtype=total_loss.dtype,
                device=total_loss.device,
            ),
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
        metrics.update(self.last_kinetics_debug)

        # ── Analogical Engine ──────────────────────────────────────────────
        # Only fires when the memory bank has been populated.  Wrapped in a
        # broad try/except so a bad retrieval or RDKit failure never crashes
        # the training loop.
        if self.memory_bank.memory_embeddings is not None:
            try:
                from rdkit import Chem as _Chem
                _query_mol = _Chem.MolFromSmiles(smiles)
                if _query_mol is not None:
                    # Use the atom multivectors already assembled for the DAG learner
                    # (atom_mv: [1, N, 8]) instead of re-querying the SIREN field.
                    # Re-querying SIREN duplicates the most expensive op in the pipeline
                    # and the atom_mv already encodes position + reactivity in G(3,0,0).
                    _query_node_multivectors = atom_mv.squeeze(0).detach()  # [N, 8]

                    # Only project to hyperbolic space when the bank actually contains
                    # continuous (HGNN) embeddings — otherwise the projection is wasted.
                    _bank_has_continuous = (
                        self.memory_bank.memory_projected_mask is not None
                        and bool(self.memory_bank.memory_projected_mask.any().item())
                    )
                    if _bank_has_continuous:
                        _query_hyper_embed = self.gated_loss.hyperbolic_projector(
                            atom_mv.squeeze(0)
                        )
                    else:
                        _query_hyper_embed = None
                    _query_morphism_prior = None
                    _batch_morph_prior = batch.get("morphism_target")
                    if torch.is_tensor(_batch_morph_prior) and _batch_morph_prior.numel() > 0:
                        _morph_prior_src = _batch_morph_prior[0] if _batch_morph_prior.ndim >= 3 else _batch_morph_prior
                        if (
                            _morph_prior_src.ndim == 2
                            and 0 <= int(true_atom_index.item()) < _morph_prior_src.size(0)
                        ):
                            _query_morphism_prior = _morph_prior_src[int(true_atom_index.item())].to(
                                dtype=torch.float32,
                                device=device,
                            )

                    _result = self.memory_bank.retrieve_and_transport(
                        _query_mol,
                        query_smiles=smiles,
                        mechanism_encoder=self.gated_loss.mechanism_encoder,
                        query_embedding=_query_hyper_embed,
                        query_multivectors=_query_node_multivectors,
                        query_morphism_prior=_query_morphism_prior,
                    )
                    # Re-index pred_ana (SMILES atom order) onto the scan's
                    # descending-reactivity-sorted atom order so target_idx
                    # (= true_row_index) aligns with pred_fp (= effective_reactivity).
                    _scan_atom_idx = module1_out.scan.atom_indices.to(
                        dtype=torch.long, device=device
                    )
                    N_scan = _scan_atom_idx.numel()
                    _pred_ana_scan = torch.zeros(N_scan, dtype=torch.float32, device=device)
                    _transport_mapped = False
                    if _result.transport_succeeded and _result.analogical_pred.numel() > 0:
                        _query_pred = _result.analogical_pred.to(dtype=torch.float32, device=device).view(-1)
                        _valid = (_scan_atom_idx >= 0) & (_scan_atom_idx < _query_pred.numel())
                        if bool(_valid.any().item()):
                            _pred_ana_scan[_valid] = _query_pred[_scan_atom_idx[_valid]]
                            _transport_mapped = bool((_pred_ana_scan > 0).any().item())

                    # physics_analogy_agreement: cosine similarity between the physics
                    # field's reactivity ranking and the analogy-transported label.
                    # This is the key signal for the Watson gate — high agreement means
                    # both pathways agree on which atom is the SoM.
                    _eff_react_norm = effective_reactivity.detach().float()
                    _ana_scan_norm = _pred_ana_scan.detach().float()
                    _eff_r2 = (_eff_react_norm * _eff_react_norm).sum().clamp_min(1e-8)
                    _ana_r2 = (_ana_scan_norm * _ana_scan_norm).sum().clamp_min(1e-8)
                    _physics_analogy_agreement = float(
                        (_eff_react_norm * _ana_scan_norm).sum()
                        / (_eff_r2.sqrt() * _ana_r2.sqrt())
                    ) if _transport_mapped else 0.0

                    _ana_loss, _ana_info = self.gated_loss(
                        pred_fp=effective_reactivity,
                        pred_ana=_pred_ana_scan,
                        target_idx=true_row_index,
                        retrieval_confidence=_result.confidence,
                        transport_succeeded=_transport_mapped,
                        physics_analogy_agreement=_physics_analogy_agreement,
                    )
                    _ana_loss = self._sanitize_tensor(
                        self._to_fp32(_ana_loss), clamp=(0.0, 100.0)
                    )
                    _ana_loss_weighted = _ana_loss * self.analogical_loss_weight
                    total_loss = total_loss + _ana_loss_weighted
                    _neuralgw_distill_weighted = torch.zeros((), dtype=torch.float32, device=device)
                    if _result.transport_distill_loss is not None:
                        _distill_raw = self._sanitize_tensor(
                            self._to_fp32(_result.transport_distill_loss),
                            clamp=(0.0, 10.0),
                        )
                        _neuralgw_distill_weighted = 0.1 * _distill_raw
                        total_loss = total_loss + _neuralgw_distill_weighted

                    _fusion_available = False
                    _fusion_weight_ana = 0.0
                    _fusion_loss_weighted = torch.zeros((), dtype=torch.float32, device=device)
                    _fusion_sigma_fp = torch.zeros((), dtype=torch.float32, device=device)
                    _fusion_sigma_ana = torch.zeros((), dtype=torch.float32, device=device)
                    _morph_f1_macro_fp = torch.zeros((), dtype=torch.float32, device=device)
                    _morph_f1_macro_ana = torch.zeros((), dtype=torch.float32, device=device)
                    _morph_f1_epox_fp = torch.zeros((), dtype=torch.float32, device=device)
                    _morph_f1_epox_ana = torch.zeros((), dtype=torch.float32, device=device)
                    _morph_f1_available = torch.zeros((), dtype=torch.float32, device=device)
                    _fusion_error_message: str | None = None
                    _ood_score = 0.0
                    _hard_case = 0.0
                    _abstain = 0.0
                    if (
                        _result.transport_plan is not None
                        and _result.retrieved_node_multivectors is not None
                        and _result.transport_backend != "mcs"
                    ):
                        try:
                            _pi_star = self._to_fp32(
                                torch.as_tensor(_result.transport_plan, device=device)
                            ).detach()
                            _ret_mv = self._to_fp32(
                                torch.as_tensor(_result.retrieved_node_multivectors, device=device)
                            )
                            _query_mv = self._to_fp32(atom_mv.squeeze(0))
                            _scan_valid = (
                                (_scan_atom_idx >= 0)
                                & (_scan_atom_idx < _query_mv.size(0))
                                & (_scan_atom_idx < _pi_star.size(0))
                            )
                            if bool(_scan_valid.all().item()) and _pi_star.ndim == 2 and _ret_mv.ndim == 2:
                                _scan_pi = _pi_star.index_select(0, _scan_atom_idx)
                                if _scan_pi.size(1) == _ret_mv.size(0):
                                    _q_fp_scan = _query_mv.index_select(0, _scan_atom_idx).unsqueeze(0)
                                    _q_ana_scan = self.pgw_cross_attention(
                                        _q_fp_scan,
                                        _ret_mv.unsqueeze(0),
                                        _scan_pi.unsqueeze(0),
                                    )
                                    _som_target_scan = torch.zeros(
                                        (1, _q_fp_scan.size(1)),
                                        dtype=torch.float32,
                                        device=device,
                                    )
                                    _som_target_scan[0, int(true_row_index.item())] = 1.0
                                    _morph_target_scan = torch.zeros(
                                        (1, _q_fp_scan.size(1), 5),
                                        dtype=torch.float32,
                                        device=device,
                                    )
                                    _morph_mask_scan = torch.zeros_like(_morph_target_scan)
                                    _has_morph_label = torch.zeros((), dtype=torch.float32, device=device)
                                    _label_confidence = torch.zeros((), dtype=torch.float32, device=device)
                                    _batch_som = batch.get("som_target")
                                    if torch.is_tensor(_batch_som) and _batch_som.numel() > 0:
                                        _som_source = _batch_som[0] if _batch_som.ndim >= 2 else _batch_som.view(-1)
                                        if _som_source.numel() > 0:
                                            _som_valid = (_scan_atom_idx >= 0) & (_scan_atom_idx < _som_source.numel())
                                            if bool(_som_valid.all().item()):
                                                _som_target_scan = _som_source.index_select(0, _scan_atom_idx).to(
                                                    dtype=torch.float32,
                                                    device=device,
                                                ).unsqueeze(0)
                                    _batch_morph = batch.get("morphism_target")
                                    _batch_morph_mask = batch.get("morphism_loss_mask")
                                    if torch.is_tensor(_batch_morph) and _batch_morph.numel() > 0:
                                        _morph_source = _batch_morph[0] if _batch_morph.ndim >= 3 else _batch_morph
                                        if _morph_source.ndim == 2 and _morph_source.size(0) > 0:
                                            _morph_valid = (_scan_atom_idx >= 0) & (_scan_atom_idx < _morph_source.size(0))
                                            if bool(_morph_valid.all().item()):
                                                _morph_target_scan = _morph_source.index_select(0, _scan_atom_idx).to(
                                                    dtype=torch.float32,
                                                    device=device,
                                                ).unsqueeze(0)
                                    if torch.is_tensor(_batch_morph_mask) and _batch_morph_mask.numel() > 0:
                                        _mask_source = _batch_morph_mask[0] if _batch_morph_mask.ndim >= 3 else _batch_morph_mask
                                        if _mask_source.ndim == 2 and _mask_source.size(0) > 0:
                                            _mask_valid = (_scan_atom_idx >= 0) & (_scan_atom_idx < _mask_source.size(0))
                                            if bool(_mask_valid.all().item()):
                                                _morph_mask_scan = _mask_source.index_select(0, _scan_atom_idx).to(
                                                    dtype=torch.float32,
                                                    device=device,
                                                ).unsqueeze(0)
                                    _batch_has_morph = batch.get("has_morphism_label")
                                    if torch.is_tensor(_batch_has_morph) and _batch_has_morph.numel() > 0:
                                        _has_morph_label = _batch_has_morph.view(-1)[0].to(
                                            dtype=torch.float32,
                                            device=device,
                                        )
                                    _batch_label_conf = batch.get("label_confidence")
                                    if torch.is_tensor(_batch_label_conf) and _batch_label_conf.numel() > 0:
                                        _label_confidence = _batch_label_conf.view(-1)[0].to(
                                            dtype=torch.float32,
                                            device=device,
                                        )
                                    _ret_morph_target = None
                                    _ret_morph_mask = None
                                    _ret_has_morph = torch.zeros((), dtype=torch.float32, device=device)
                                    _ret_label_conf = torch.zeros((), dtype=torch.float32, device=device)
                                    if torch.is_tensor(_result.retrieved_morphism_target):
                                        _ret_morph_target = self._to_fp32(
                                            _result.retrieved_morphism_target.to(device=device)
                                        ).unsqueeze(0)
                                    if torch.is_tensor(_result.retrieved_morphism_loss_mask):
                                        _ret_morph_mask = self._to_fp32(
                                            _result.retrieved_morphism_loss_mask.to(device=device)
                                        ).unsqueeze(0)
                                    _ret_has_morph = torch.as_tensor(
                                        1.0 if _result.retrieved_has_morphism_label else 0.0,
                                        dtype=torch.float32,
                                        device=device,
                                    )
                                    _ret_label_conf = torch.as_tensor(
                                        float(_result.retrieved_label_confidence),
                                        dtype=torch.float32,
                                        device=device,
                                    )
                                    _y_hat_fp_som, _y_hat_fp_morph, _y_hat_ana_som, _y_hat_ana_morph = self.analogical_dual_decoder(
                                        _q_fp_scan,
                                        _q_ana_scan,
                                    )
                                    _bridge_loss = None
                                    if (
                                        _ret_morph_target is not None
                                        and _ret_morph_mask is not None
                                        and bool((_has_morph_label > 0).item())
                                        and bool((_ret_has_morph > 0).item())
                                        and _scan_pi.size(1) == _ret_morph_target.size(1)
                                    ):
                                        _target_alignment = torch.matmul(
                                            _morph_target_scan.squeeze(0),
                                            _ret_morph_target.squeeze(0).transpose(0, 1),
                                        ).clamp_max(1.0)
                                        _query_atom_mask = (_morph_mask_scan.squeeze(0).sum(dim=-1) > 0).to(dtype=torch.float32).unsqueeze(1)
                                        _ret_atom_mask = (_ret_morph_mask.squeeze(0).sum(dim=-1) > 0).to(dtype=torch.float32).unsqueeze(0)
                                        _joint_mask = _query_atom_mask * _ret_atom_mask
                                        if bool((_joint_mask.sum() > 0).item()):
                                            _bridge_raw = F.binary_cross_entropy(
                                                _scan_pi.clamp(1.0e-7, 1.0 - 1.0e-7),
                                                _target_alignment,
                                                reduction="none",
                                            )
                                            _bridge_loss = (_bridge_raw * _joint_mask).sum() / _joint_mask.sum().clamp_min(1.0)
                                            _bridge_loss = _bridge_loss * torch.minimum(
                                                _label_confidence.clamp_min(0.0),
                                                _ret_label_conf.clamp_min(0.0),
                                            )
                                    _fusion_loss, _fusion_info = self.analogical_arbiter(
                                        _y_hat_fp_som,
                                        _y_hat_fp_morph,
                                        _y_hat_ana_som,
                                        _y_hat_ana_morph,
                                        _som_target_scan,
                                        _morph_target_scan,
                                        _morph_mask_scan,
                                        label_confidence=_label_confidence,
                                        has_morphism_label=_has_morph_label,
                                        bridge_loss=_bridge_loss,
                                        current_epoch=int(self.current_epoch_index),
                                    )
                                    _fusion_loss = self._sanitize_tensor(
                                        self._to_fp32(_fusion_loss),
                                        clamp=(0.0, 20.0),
                                    )
                                    _fusion_loss_weighted = 0.25 * _fusion_loss
                                    total_loss = total_loss + _fusion_loss_weighted
                                    _fusion_sigma_fp = self._to_fp32(_fusion_info["sigma_fp"]).view(())
                                    _fusion_sigma_ana = self._to_fp32(_fusion_info["sigma_ana"]).view(())
                                    _weight_ana_t = self._to_fp32(_fusion_info["weight_ana"]).view(1, 1)
                                    _fusion_weight_ana = float(_weight_ana_t.detach().item())
                                    if bool((_has_morph_label > 0).item()) and bool((_morph_mask_scan.sum() > 0).item()):
                                        _morph_f1_macro_fp, _morph_f1_fp_cls = compute_masked_morphism_f1(
                                            _y_hat_fp_morph,
                                            _morph_target_scan,
                                            _morph_mask_scan,
                                        )
                                        _morph_f1_macro_ana, _morph_f1_ana_cls = compute_masked_morphism_f1(
                                            _y_hat_ana_morph,
                                            _morph_target_scan,
                                            _morph_mask_scan,
                                        )
                                        _morph_f1_epox_fp = _morph_f1_fp_cls[3].view(())
                                        _morph_f1_epox_ana = _morph_f1_ana_cls[3].view(())
                                        _morph_f1_available = torch.ones((), dtype=torch.float32, device=device)
                                    _y_final = (1.0 - _weight_ana_t) * _y_hat_fp_som + _weight_ana_t * _y_hat_ana_som
                                    _rank_order = torch.argsort(_y_final.squeeze(0), descending=True)
                                    _fused_rank = int(
                                        (_rank_order == true_row_index).nonzero(as_tuple=False)[0].item()
                                    )
                                    metrics.update({
                                        "som_top1": torch.as_tensor(
                                            1.0 if _fused_rank == 0 else 0.0,
                                            dtype=torch.float32,
                                            device=device,
                                        ),
                                        "som_top2": torch.as_tensor(
                                            1.0 if _fused_rank <= 1 else 0.0,
                                            dtype=torch.float32,
                                            device=device,
                                        ),
                                        "som_top3": torch.as_tensor(
                                            1.0 if _fused_rank <= 2 else 0.0,
                                            dtype=torch.float32,
                                            device=device,
                                        ),
                                        "som_rank": torch.as_tensor(
                                            float(_fused_rank),
                                            dtype=torch.float32,
                                            device=device,
                                        ),
                                        "ana_fusion_available": torch.as_tensor(
                                            1.0, dtype=torch.float32, device=device
                                        ),
                                        "ana_fusion_weight": torch.as_tensor(
                                            _fusion_weight_ana,
                                            dtype=torch.float32,
                                            device=device,
                                        ),
                                        "ana_fusion_loss_total": _fusion_loss_weighted.detach(),
                                        "ana_sigma_fp": _fusion_sigma_fp.detach(),
                                        "ana_sigma_ana": _fusion_sigma_ana.detach(),
                                        "ana_morph_loss_fp": self._to_fp32(_fusion_info["loss_fp_morph"]).detach(),
                                        "ana_morph_loss_ana": self._to_fp32(_fusion_info["loss_ana_morph"]).detach(),
                                        "ana_bridge_loss": self._to_fp32(_fusion_info["bridge_loss"]).detach(),
                                        "ana_has_morphism_label": self._to_fp32(_fusion_info["has_morphism_label"]).detach(),
                                        "ana_label_confidence": self._to_fp32(_fusion_info["label_confidence"]).detach(),
                                        "ana_burn_in_active": self._to_fp32(_fusion_info["burn_in_active"]).detach(),
                                        "morphism_f1_available": _morph_f1_available.detach(),
                                        "morphism_f1_macro_fp": _morph_f1_macro_fp.detach(),
                                        "morphism_f1_macro_ana": _morph_f1_macro_ana.detach(),
                                        "morphism_f1_epox_fp": _morph_f1_epox_fp.detach(),
                                        "morphism_f1_epox_ana": _morph_f1_epox_ana.detach(),
                                    })
                                    _fusion_available = True
                        except Exception as _fusion_err:
                            _fusion_error_message = f"{type(_fusion_err).__name__}: {_fusion_err}"
                            _fusion_available = False

                    _ood_score, _hard_case, _abstain = compute_analogical_ood_metrics(
                        retrieval_confidence=float(_result.confidence),
                        retrieval_mix_entropy=float(_result.retrieval_mix_entropy),
                        retrieval_mix_count=int(_result.retrieval_mix_count),
                        retrieval_diversity_score=float(_result.retrieval_diversity_score),
                        transport_backend=str(_result.transport_backend),
                        transport_succeeded=bool(_transport_mapped),
                        transported_mass=float(_result.transported_mass),
                        physics_analogy_agreement=float(_physics_analogy_agreement),
                        fusion_available=bool(_fusion_available),
                        neuralgw_confidence=float(_result.neuralgw_confidence),
                    )

                    analogical_trace = {
                        "smiles": smiles,
                        "phase": self._active_phase,
                        "epoch_index": int(self.current_epoch_index),
                        "batch_index": int(self._active_batch_index),
                        "total_batches": int(self._active_total_batches),
                        "global_step": int(self.global_step_counter.detach().cpu().item()),
                        "true_atom_index": int(true_atom_index.detach().cpu().item()),
                        "true_row_index": int(true_row_index.detach().cpu().item()),
                        "retrieval_embedding_space": _result.embedding_space,
                        "retrieval_confidence": float(_result.confidence),
                        "retrieval_mix_count": int(_result.retrieval_mix_count),
                        "retrieval_mix_entropy": float(_result.retrieval_mix_entropy),
                        "retrieval_candidate_count": int(_result.retrieval_candidate_count),
                        "retrieval_mechanism_overlap": float(_result.retrieval_mechanism_overlap),
                        "retrieval_diversity_score": float(_result.retrieval_diversity_score),
                        "retrieved_same_query": bool(_result.retrieved_same_query),
                        "retrieved_smiles": (
                            _Chem.MolToSmiles(_result.retrieved_mol)
                            if _result.retrieved_mol is not None
                            else None
                        ),
                        "retrieved_scaffold": _result.retrieved_scaffold,
                        "retrieved_som_idx": int(_result.retrieved_som_idx),
                        "transport_backend": _result.transport_backend,
                        "transport_error_message": _result.transport_error_message,
                        "neuralgw_route_reason": _result.neuralgw_route_reason,
                        "transport_plan_shape": (
                            list(_result.transport_plan.shape)
                            if _result.transport_plan is not None
                            else None
                        ),
                        "retrieved_node_multivectors_shape": (
                            list(torch.as_tensor(_result.retrieved_node_multivectors).shape)
                            if _result.retrieved_node_multivectors is not None
                            else None
                        ),
                        "transport_succeeded": bool(_transport_mapped),
                        "transport_support": int(_result.transport_support_size),
                        "transported_mass": float(_result.transported_mass),
                        "ana_gate_open": float(_ana_info["gate_open"]),
                        "ana_peak": float(_ana_info["analogy_peak"]),
                        "ana_gate_conf_ok": float(_ana_info["gate_conf_ok"]),
                        "ana_gate_peak_ok": float(_ana_info["gate_peak_ok"]),
                        "ana_watson_agreement": float(_physics_analogy_agreement),
                        "neuralgw_used_exact": bool(_result.neuralgw_used_exact),
                        "neuralgw_confidence": float(_result.neuralgw_confidence),
                        "neuralgw_distill_loss": float(_result.neuralgw_distill_loss),
                        "fusion_available": bool(_fusion_available),
                        "fusion_weight_ana": float(_fusion_weight_ana),
                        "fusion_error_message": _fusion_error_message,
                        "ood_score": float(_ood_score),
                        "hard_case": bool(_hard_case),
                        "abstain": bool(_abstain),
                    }

                    # Encoder supervision loss: teach MechanismEncoder to embed
                    # same-SoM-class molecules close together and different classes apart.
                    _enc_loss_val = 0.0
                    if (
                        _result.query_embed is not None
                        and _result.retrieved_embed_detached is not None
                    ):
                        from nexus.reasoning.metric_learner import (
                            encoder_supervision_loss as _enc_sup_loss,
                            hyperbolic_supervision_loss as _hyp_sup_loss,
                            _som_class as _som_cls,
                        )
                        _q_som_class = _som_cls(int(true_atom_index.item()), _query_mol)
                        _r_som_class = _som_cls(_result.retrieved_som_idx, _result.retrieved_mol)
                        _same_class = (_q_som_class == _r_som_class) and (_q_som_class >= 0)
                        if _result.embedding_space == "hyperbolic":
                            _enc_loss = _hyp_sup_loss(
                                _result.query_embed,
                                _result.retrieved_embed_detached,
                                _same_class,
                            )
                        else:
                            _enc_loss = _enc_sup_loss(
                                _result.query_embed,
                                _result.retrieved_embed_detached,
                                _same_class,
                            )
                        _enc_loss = self._sanitize_tensor(
                            self._to_fp32(_enc_loss), clamp=(0.0, 10.0)
                        )
                        # Weight encoder loss at 0.05 — gentle signal, not dominating physics.
                        total_loss = total_loss + 0.05 * _enc_loss
                        _enc_loss_val = float(_enc_loss.detach().item())

                    metrics.update({
                        "ana_loss_total": _ana_loss_weighted.detach(),
                        "ana_loss_raw": _ana_loss.detach(),
                        "ana_loss_weight": torch.as_tensor(
                            self.analogical_loss_weight,
                            dtype=torch.float32,
                            device=device,
                        ),
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
                        "ana_peak": torch.as_tensor(
                            _ana_info["analogy_peak"], dtype=torch.float32, device=device
                        ),
                        "ana_gate_conf_ok": torch.as_tensor(
                            _ana_info["gate_conf_ok"], dtype=torch.float32, device=device
                        ),
                        "ana_gate_peak_ok": torch.as_tensor(
                            _ana_info["gate_peak_ok"], dtype=torch.float32, device=device
                        ),
                        "ana_watson_agreement": torch.as_tensor(
                            _physics_analogy_agreement, dtype=torch.float32, device=device
                        ),
                        "ana_encoder_loss": torch.as_tensor(
                            _enc_loss_val, dtype=torch.float32, device=device
                        ),
                        "ana_transport_backend_is_fast": torch.as_tensor(
                            1.0 if _result.transport_backend == "neuralgw_fast" else 0.0,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "ana_transport_mass": torch.as_tensor(
                            _result.transported_mass,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "ana_transport_support": torch.as_tensor(
                            _result.transport_support_size,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "ana_retrieval_candidate_count": torch.as_tensor(
                            _result.retrieval_candidate_count,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "ana_retrieval_mechanism_overlap": torch.as_tensor(
                            _result.retrieval_mechanism_overlap,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "ana_retrieval_diversity": torch.as_tensor(
                            _result.retrieval_diversity_score,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "neuralgw_used_exact": torch.as_tensor(
                            1.0 if _result.neuralgw_used_exact else 0.0,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "neuralgw_confidence": torch.as_tensor(
                            _result.neuralgw_confidence,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "neuralgw_distill_loss": torch.as_tensor(
                            _result.neuralgw_distill_loss,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "neuralgw_distill_loss_total": _neuralgw_distill_weighted.detach(),
                        "ana_ood_score": torch.as_tensor(
                            _ood_score,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "ana_hard_case": torch.as_tensor(
                            _hard_case,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "ana_abstain": torch.as_tensor(
                            _abstain,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "ana_fusion_available": torch.as_tensor(
                            1.0 if _fusion_available else 0.0,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "ana_fusion_weight": torch.as_tensor(
                            _fusion_weight_ana,
                            dtype=torch.float32,
                            device=device,
                        ),
                        "ana_fusion_loss_total": _fusion_loss_weighted.detach(),
                        "ana_sigma_fp": _fusion_sigma_fp.detach(),
                        "ana_sigma_ana": _fusion_sigma_ana.detach(),
                        "ana_direct_lift_top1": (
                            metrics["som_top1"] - metrics["som_top1_fp"]
                        ).detach(),
                        "ana_direct_lift_rank": (
                            metrics["som_rank_fp"] - metrics["som_rank"]
                        ).detach(),
                        "ana_direct_lift_top1_hard": (
                            (metrics["som_top1"] - metrics["som_top1_fp"]) * torch.as_tensor(_hard_case, dtype=torch.float32, device=device)
                        ).detach(),
                        "ana_fused_top1_hard": (
                            metrics["som_top1"] * torch.as_tensor(_hard_case, dtype=torch.float32, device=device)
                        ).detach(),
                        "ana_fp_top1_hard": (
                            metrics["som_top1_fp"] * torch.as_tensor(_hard_case, dtype=torch.float32, device=device)
                        ).detach(),
                    })
            except Exception as _ana_err:
                # Print once so we can see if the analogical engine is broken,
                # but never let it crash the physics training loop.
                import traceback as _tb
                print(f"[ANA-ERR] {type(_ana_err).__name__}: {_ana_err}", flush=True)
                _tb.print_exc()
                analogical_trace = {
                    "smiles": smiles,
                    "phase": self._active_phase,
                    "epoch_index": int(self.current_epoch_index),
                    "batch_index": int(self._active_batch_index),
                    "total_batches": int(self._active_total_batches),
                    "global_step": int(self.global_step_counter.detach().cpu().item()),
                    "analogical_error_type": type(_ana_err).__name__,
                    "analogical_error": str(_ana_err),
                }
        # ──────────────────────────────────────────────────────────────────

        metrics["loss_total"] = total_loss.detach()
        if analogical_trace is not None:
            analogical_trace.update(
                {
                    "loss_total": metrics.get("loss_total"),
                    "som_top1": metrics.get("som_top1"),
                    "som_top2": metrics.get("som_top2"),
                    "som_rank": metrics.get("som_rank"),
                    "som_top1_fp": metrics.get("som_top1_fp"),
                    "som_top2_fp": metrics.get("som_top2_fp"),
                    "som_rank_fp": metrics.get("som_rank_fp"),
                    "pred_rate": metrics.get("pred_rate"),
                    "dag_causal_loss": metrics.get("dag_causal_loss"),
                    "morphism_f1_available": metrics.get("morphism_f1_available"),
                    "morphism_f1_macro_fp": metrics.get("morphism_f1_macro_fp"),
                    "morphism_f1_macro_ana": metrics.get("morphism_f1_macro_ana"),
                    "morphism_f1_epox_fp": metrics.get("morphism_f1_epox_fp"),
                    "morphism_f1_epox_ana": metrics.get("morphism_f1_epox_ana"),
                }
            )
            self._append_analogical_trace(analogical_trace)

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

    def fit_epoch(
        self,
        dataloader,
        *,
        train: bool = True,
        log_every: int = 0,
        on_batch_end=None,
    ) -> Dict[str, float]:
        reducer: Dict[str, List[float]] = {}
        self.train(mode=train)
        self._active_phase = "train" if train else "val"
        if train and self.memory_bank.pgw is not None:
            self.memory_bank.pgw.current_epoch = int(self.current_epoch_index)
        
        valid_batches_processed = 0
        if train:
            if self.optimizer is None: self.configure_optimizers()
            assert self.optimizer is not None
            self.optimizer.zero_grad(set_to_none=True)

        total_batches = len(dataloader)
        for i, batch in enumerate(dataloader, start=1):
            self._active_batch_index = i
            self._active_total_batches = total_batches
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
                running = {key: sum(values) / max(len(values), 1) for key, values in reducer.items()}
                if on_batch_end is not None:
                    on_batch_end(
                        batch_index=i,
                        total_batches=total_batches,
                        running_metrics=running,
                        step_metrics={k: float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()},
                        train=train,
                    )
                if log_every > 0 and (i == 1 or i % log_every == 0 or i == total_batches):
                    _ana_active = "ana_loss_total" in running
                    _ana_str = (
                        f" | ana_loss={running.get('ana_loss_total', float('nan')):.4g}"
                        f" gate={running.get('ana_gate_open', float('nan')):.2f}"
                        f" conf={running.get('ana_confidence', float('nan')):.3f}"
                        f" peak={running.get('ana_peak', float('nan')):.3f}"
                        f" conf_ok={running.get('ana_gate_conf_ok', float('nan')):.2f}"
                        f" peak_ok={running.get('ana_gate_peak_ok', float('nan')):.2f}"
                        f" w_fp={running.get('ana_weight_fp', float('nan')):.3f}"
                        f" w_ana={running.get('ana_weight_ana', float('nan')):.3f}"
                        f" t_ok={running.get('ana_transport_ok', float('nan')):.2f}"
                        f" fuse={running.get('ana_fusion_available', float('nan')):.2f}"
                        f" fuse_w={running.get('ana_fusion_weight', float('nan')):.2f}"
                        f" burn={running.get('ana_burn_in_active', float('nan')):.2f}"
                        f" ngw_exact={running.get('neuralgw_used_exact', float('nan')):.2f}"
                        f" ngw_conf={running.get('neuralgw_confidence', float('nan')):.3f}"
                        f" cand={running.get('ana_retrieval_candidate_count', float('nan')):.1f}"
                        f" mech_ov={running.get('ana_retrieval_mechanism_overlap', float('nan')):.2f}"
                        f" div={running.get('ana_retrieval_diversity', float('nan')):.2f}"
                        f" ood={running.get('ana_ood_score', float('nan')):.2f}"
                        f" hard={running.get('ana_hard_case', float('nan')):.2f}"
                        f" abstain={running.get('ana_abstain', float('nan')):.2f}"
                        f" lift={running.get('ana_direct_lift_top1', float('nan')):.2%}"
                        f" mf1_fp={running.get('morphism_f1_macro_fp', float('nan')):.3f}"
                        f" mf1_ana={running.get('morphism_f1_macro_ana', float('nan')):.3f}"
                        f" epox_fp={running.get('morphism_f1_epox_fp', float('nan')):.3f}"
                        f" epox_ana={running.get('morphism_f1_epox_ana', float('nan')):.3f}"
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
                running = {key: sum(values) / max(len(values), 1) for key, values in reducer.items()}
                if on_batch_end is not None:
                    on_batch_end(
                        batch_index=i,
                        total_batches=total_batches,
                        running_metrics=running,
                        step_metrics={k: float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()},
                        train=train,
                    )
                if log_every > 0 and (i == 1 or i % log_every == 0 or i == total_batches):
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

        if train:
            self.current_epoch_index += 1
        self._active_batch_index = 0
        self._active_total_batches = 0
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
