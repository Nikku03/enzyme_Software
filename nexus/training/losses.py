from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GodLossBreakdown:
    som_loss: torch.Tensor
    ranking_loss: torch.Tensor
    kinetic_loss: torch.Tensor
    physics_loss: torch.Tensor
    topology_loss: torch.Tensor
    flux_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    raw_losses: torch.Tensor


@dataclass
class AnalogicalLossBreakdown:
    fp_raw_loss: torch.Tensor
    analogical_raw_loss: torch.Tensor
    weighted_fp_loss: torch.Tensor
    weighted_analogical_loss: torch.Tensor
    total_loss: torch.Tensor


class ListMLERankingLoss(nn.Module):
    def __init__(self, eps: float = 1.0e-12) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        pred_scores: torch.Tensor,
        true_indices: torch.Tensor | int,
    ) -> torch.Tensor:
        if pred_scores.ndim == 1:
            pred_scores = pred_scores.unsqueeze(0)
        if pred_scores.ndim != 2:
            raise ValueError("pred_scores must have shape [N] or [B, N]")
        batch = pred_scores.size(0)
        target = torch.as_tensor(true_indices, device=pred_scores.device, dtype=torch.long).view(-1)
        if target.numel() == 1 and batch > 1:
            target = target.expand(batch)
        if target.numel() != batch:
            raise ValueError("true_indices must provide one target per batch item")
        # Sort descending so the cumulative suffix-sum denominator depends only on
        # score values, not on the order atoms happen to appear in the input tensor
        # (SDF file order, SMILES canonical order, etc.).  Without this sort,
        # log_probs[i] = score[i] - log(sum_{j>=i} exp(score[j])) gives a different
        # value for the same true SoM when the atom list is permuted.
        sorted_scores, sort_indices = torch.sort(pred_scores, descending=True, dim=-1)
        # Boolean mask: True at the position of the true SoM in the sorted tensor.
        true_mask = sort_indices == target.unsqueeze(-1)   # [B, N]
        # Numerically-stable suffix log-sum-exp.  After descending sort the maximum
        # is always sorted_scores[:, 0], so subtract it before exponentiation.
        shifted = sorted_scores - sorted_scores[:, :1]
        exp_scores = torch.exp(shifted)
        cum_sums = torch.cumsum(exp_scores.flip(-1), dim=-1).flip(-1).clamp_min(self.eps)
        log_probs = shifted - torch.log(cum_sums)
        return -log_probs[true_mask].mean()


class SoftmaxFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        eps: float = 1.0e-12,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        target_index: torch.Tensor | int,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [B, N]")
        batch = logits.size(0)
        target = torch.as_tensor(target_index, device=logits.device, dtype=torch.long).view(-1)
        if target.numel() == 1 and batch > 1:
            target = target.expand(batch)
        if target.numel() != batch:
            raise ValueError("target_index must provide one target per batch item")
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        target_probs = target_log_probs.exp().clamp_min(self.eps)
        focal_factor = (1.0 - target_probs).pow(self.gamma)
        if self.alpha is not None:
            focal_factor = focal_factor * self.alpha
        return -(focal_factor * target_log_probs).mean()


class NEXUS_God_Loss(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        topology_margin: float = 1.0e-3,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
        som_loss_mode: str = "listmle",
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        flux_loss_weight: float = 0.1,
        kinetic_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.topology_margin = float(topology_margin)
        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)
        self.som_loss_mode = str(som_loss_mode).lower()
        self.flux_loss_weight = float(flux_loss_weight)
        # Permanent scale applied to kinetic loss BEFORE homoscedastic weighting.
        # At random init, log(pred_rate) ≈ +17.7 vs log(exp_rate) ≈ -18.4, giving
        # huber_loss ≈ 36 which is 11× the SoM focal loss (≈3.2).  log_vars would
        # need ~75 epochs at lr=1e-3 to suppress this — we only train for 8–30.
        # A permanent 0.1 weight brings kinetic to ≈3.6, matching SoM magnitude,
        # so the field spends its gradient budget on SoM ranking from the start.
        self.kinetic_loss_weight = float(max(kinetic_loss_weight, 0.0))
        self.log_vars = nn.Parameter(torch.zeros(6))
        self.ranking_loss_fn = ListMLERankingLoss()
        self.focal_loss_fn = SoftmaxFocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    def compute_som_loss(
        self,
        delta_E_tensor: torch.Tensor,
        true_som_idx: torch.Tensor | int,
        beta: Optional[float] = None,
        som_soft_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if delta_E_tensor.ndim != 1:
            raise ValueError("delta_E_tensor must have shape [N_atoms]")
        logits = -(float(self.beta if beta is None else beta)) * delta_E_tensor
        if som_soft_target is not None:
            soft_target = torch.as_tensor(
                som_soft_target,
                dtype=torch.float32,
                device=delta_E_tensor.device,
            ).view(-1)
            if soft_target.numel() != delta_E_tensor.numel():
                raise ValueError("som_soft_target must match delta_E_tensor shape")
            soft_target = soft_target.clamp_min(0.0)
            soft_target = soft_target / soft_target.sum().clamp_min(1.0e-8)
            return -(soft_target * F.log_softmax(logits, dim=-1)).sum()
        if self.som_loss_mode == "listmle":
            return self.ranking_loss_fn(logits, true_som_idx)
        if self.som_loss_mode == "focal":
            return self.focal_loss_fn(logits.unsqueeze(0), true_som_idx)
        target = torch.as_tensor(true_som_idx, device=delta_E_tensor.device, dtype=torch.long).view(1)
        return F.cross_entropy(logits.unsqueeze(0), target)

    def compute_kinetic_loss(
        self,
        pred_rate: torch.Tensor,
        exp_rate: torch.Tensor,
    ) -> torch.Tensor:
        log_pred = torch.log(pred_rate.clamp_min(1.0e-12))
        log_exp = torch.log(exp_rate.clamp_min(1.0e-12))
        return F.huber_loss(log_pred, log_exp, delta=1.0)

    def compute_physics_loss(
        self,
        sobolev_penalty: torch.Tensor,
        H_initial: torch.Tensor,
        H_final: torch.Tensor,
    ) -> torch.Tensor:
        energy_drift = torch.abs(H_final - H_initial)
        return sobolev_penalty + energy_drift

    def compute_topology_loss(
        self,
        ts_eigenvalues: torch.Tensor,
        margin: Optional[float] = None,
    ) -> torch.Tensor:
        if ts_eigenvalues.ndim != 1 or ts_eigenvalues.numel() < 2:
            raise ValueError("ts_eigenvalues must have shape [N] with N >= 2")
        eigvals, _ = torch.sort(ts_eigenvalues)
        m = float(self.topology_margin if margin is None else margin)
        lambda_1 = eigvals[0]
        lambda_2 = eigvals[1]
        penalty_1 = F.relu(lambda_1 + m)
        penalty_2 = F.relu(m - lambda_2)
        return penalty_1 + penalty_2

    def compute_flux_consistency_loss(
        self,
        flux_consistency_loss: torch.Tensor,
    ) -> torch.Tensor:
        return flux_consistency_loss * self.flux_loss_weight

    def compute_reconstruction_loss(
        self,
        reconstruction_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if reconstruction_loss is None:
            return self.log_vars.new_zeros(())
        return reconstruction_loss

    def compute_raw_losses(
        self,
        *,
        delta_E_tensor: torch.Tensor,
        true_som_idx: torch.Tensor | int,
        pred_rate: torch.Tensor,
        exp_rate: torch.Tensor,
        sobolev_penalty: torch.Tensor,
        H_initial: torch.Tensor,
        H_final: torch.Tensor,
        ts_eigenvalues: torch.Tensor,
        flux_consistency_loss: Optional[torch.Tensor] = None,
        reconstruction_loss: Optional[torch.Tensor] = None,
        ranking_loss: Optional[torch.Tensor] = None,
        kinetic_loss_scale: float = 1.0,
        som_soft_target: Optional[torch.Tensor] = None,
    ) -> GodLossBreakdown:
        som_loss = self.compute_som_loss(
            delta_E_tensor,
            true_som_idx,
            som_soft_target=som_soft_target,
        )
        kinetic_loss = (
            self.compute_kinetic_loss(pred_rate, exp_rate)
            * float(kinetic_loss_scale)
            * self.kinetic_loss_weight
        )
        physics_loss = self.compute_physics_loss(sobolev_penalty, H_initial, H_final)
        topology_loss = self.compute_topology_loss(ts_eigenvalues)
        flux_loss = (
            self.compute_flux_consistency_loss(flux_consistency_loss)
            if flux_consistency_loss is not None
            else torch.zeros((), dtype=som_loss.dtype, device=som_loss.device)
        )
        recon_loss = self.compute_reconstruction_loss(reconstruction_loss)
        actual_ranking_loss = (
            ranking_loss
            if ranking_loss is not None
            else torch.zeros((), dtype=som_loss.dtype, device=som_loss.device)
        )
        raw_losses = torch.stack([som_loss, kinetic_loss, physics_loss, topology_loss, flux_loss, recon_loss], dim=0)
        return GodLossBreakdown(
            som_loss=som_loss,
            ranking_loss=actual_ranking_loss,
            kinetic_loss=kinetic_loss,
            physics_loss=physics_loss,
            topology_loss=topology_loss,
            flux_loss=flux_loss,
            reconstruction_loss=recon_loss,
            raw_losses=raw_losses,
        )

    def forward(
        self,
        *,
        delta_E_tensor: torch.Tensor,
        true_som_idx: torch.Tensor | int,
        pred_rate: torch.Tensor,
        exp_rate: torch.Tensor,
        sobolev_penalty: torch.Tensor,
        H_initial: torch.Tensor,
        H_final: torch.Tensor,
        ts_eigenvalues: torch.Tensor,
        flux_consistency_loss: Optional[torch.Tensor] = None,
        reconstruction_loss: Optional[torch.Tensor] = None,
        ranking_loss: Optional[torch.Tensor] = None,
        kinetic_loss_scale: float = 1.0,
        som_soft_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        breakdown = self.compute_raw_losses(
            delta_E_tensor=delta_E_tensor,
            true_som_idx=true_som_idx,
            pred_rate=pred_rate,
            exp_rate=exp_rate,
            sobolev_penalty=sobolev_penalty,
            H_initial=H_initial,
            H_final=H_final,
            ts_eigenvalues=ts_eigenvalues,
            flux_consistency_loss=flux_consistency_loss,
            reconstruction_loss=reconstruction_loss,
            kinetic_loss_scale=kinetic_loss_scale,
            ranking_loss=ranking_loss,
            som_soft_target=som_soft_target,
        )
        safe_log_vars = torch.clamp(self.log_vars, min=self.log_var_min, max=self.log_var_max)
        precision = torch.exp(-safe_log_vars)
        weighted_losses = precision * breakdown.raw_losses
        total_loss = 0.5 * torch.sum(weighted_losses + safe_log_vars)
        info = {
            "som_loss": breakdown.som_loss.detach(),
            "ranking_loss": breakdown.ranking_loss.detach(),
            "kinetic_loss": breakdown.kinetic_loss.detach(),
            "physics_loss": breakdown.physics_loss.detach(),
            "topology_loss": breakdown.topology_loss.detach(),
            "flux_loss": breakdown.flux_loss.detach(),
            "reconstruction_loss": breakdown.reconstruction_loss.detach(),
            "raw_losses": breakdown.raw_losses.detach(),
            "log_vars": self.log_vars.detach(),
            "safe_log_vars": safe_log_vars.detach(),
            "precision": precision.detach(),
            "weighted_losses": weighted_losses.detach(),
            "total_loss": total_loss.detach(),
        }
        return total_loss, info


class AnalogicalGodLoss(nn.Module):
    """
    Homoscedastic arbitration between:
    - a first-principles pathway (continuous physics / DAG physics loss)
    - an analogical pathway (retrieved motif / topological DAG logits)

    This module is intentionally standalone because the current trainer does not
    yet expose a live analogical retrieval branch.  It can be composed with the
    existing NEXUS_God_Loss once `pred_fp` and `pred_ana` are both available in
    the training step.
    """

    def __init__(
        self,
        base_physics_loss_fn: Callable[..., Any],
        *,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.base_physics_loss_fn = base_physics_loss_fn
        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)
        self.log_var_fp = nn.Parameter(torch.zeros(1))
        self.log_var_ana = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _unpack_loss(loss_out: Any) -> torch.Tensor:
        if torch.is_tensor(loss_out):
            return loss_out
        if isinstance(loss_out, tuple) and len(loss_out) > 0 and torch.is_tensor(loss_out[0]):
            return loss_out[0]
        raise TypeError("base_physics_loss_fn must return a Tensor or (Tensor, info) tuple")

    def forward(
        self,
        pred_fp: torch.Tensor,
        pred_ana: torch.Tensor,
        target_dag: torch.Tensor,
        physics_tensors: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        physics_kwargs = dict(physics_tensors or {})
        loss_fp_raw = self._unpack_loss(
            self.base_physics_loss_fn(pred_fp, target_dag, **physics_kwargs)
        )
        loss_ana_raw = F.binary_cross_entropy_with_logits(pred_ana, target_dag)

        safe_log_var_fp = torch.clamp(self.log_var_fp, min=self.log_var_min, max=self.log_var_max)
        safe_log_var_ana = torch.clamp(self.log_var_ana, min=self.log_var_min, max=self.log_var_max)

        precision_fp = torch.exp(-safe_log_var_fp)
        precision_ana = torch.exp(-safe_log_var_ana)

        weighted_loss_fp = precision_fp * loss_fp_raw + safe_log_var_fp
        weighted_loss_ana = precision_ana * loss_ana_raw + safe_log_var_ana
        total_loss = weighted_loss_fp + weighted_loss_ana

        breakdown = AnalogicalLossBreakdown(
            fp_raw_loss=loss_fp_raw,
            analogical_raw_loss=loss_ana_raw,
            weighted_fp_loss=weighted_loss_fp,
            weighted_analogical_loss=weighted_loss_ana,
            total_loss=total_loss,
        )
        info = {
            "loss_fp_raw": breakdown.fp_raw_loss.detach(),
            "loss_ana_raw": breakdown.analogical_raw_loss.detach(),
            "weighted_loss_fp": breakdown.weighted_fp_loss.detach(),
            "weighted_loss_ana": breakdown.weighted_analogical_loss.detach(),
            "sigma_fp": torch.exp(0.5 * safe_log_var_fp).detach(),
            "sigma_ana": torch.exp(0.5 * safe_log_var_ana).detach(),
            "precision_fp": precision_fp.detach(),
            "precision_ana": precision_ana.detach(),
            "log_var_fp": self.log_var_fp.detach(),
            "log_var_ana": self.log_var_ana.detach(),
            "safe_log_var_fp": safe_log_var_fp.detach(),
            "safe_log_var_ana": safe_log_var_ana.detach(),
            "total_loss": breakdown.total_loss.detach(),
        }
        return total_loss, info

class WatsonGate(nn.Module):
    """
    Soft learned gate that replaces the hard 3-condition boolean in
    GatedAnalogicalGodLoss.

    Input features (3-dim):
        [retrieval_confidence, ana_peak, physics_analogy_agreement]

    where physics_analogy_agreement = cosine_sim(effective_reactivity.detach(),
    pred_ana_scan) — how much the physics field agrees with the analogy.

    Architecture: Linear(3→16) → SiLU → Linear(16→1) → sigmoid
    Init: fc2 bias = -2.0  →  sigmoid(-2) ≈ 0.12 (conservative start; gate
    stays mostly closed initially and learns to open when evidence accumulates).

    Returns a scalar in (0, 1) used as gate_weight to scale the analogy branch:
        total_loss = loss_term_fp + gate_weight * loss_term_ana
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)
        nn.init.constant_(self.fc2.bias, -2.0)

    def forward(
        self,
        confidence: float,
        ana_peak: float,
        agreement: float,
    ) -> torch.Tensor:
        device = self.fc1.weight.device
        x = torch.tensor(
            [[confidence, ana_peak, agreement]],
            dtype=torch.float32,
            device=device,
        )
        h = F.silu(self.fc1(x))
        return torch.sigmoid(self.fc2(h)).squeeze()


class GatedAnalogicalGodLoss(nn.Module):
    """
    Homoscedastic arbitration between a first-principles pathway (pred_fp)
    and an analogical retrieval pathway (pred_ana).

    Uses precision parameters s = exp(log_s) instead of variance parameters
    to prevent the σ-collapse failure mode where both σ values drift to ∞.

    Gate: WatsonGate — a learned soft gate (replaces the old hard boolean).
    The gate takes three signals:
        1. retrieval_confidence  — quality of the retrieved analogue
        2. ana_peak              — sharpness of the transported SoM prediction
        3. physics_analogy_agreement — cosine agreement between physics and analogy

    The MechanismEncoder lives here so its parameters flow through the trainer's
    main optimizer without any extra registration.

    Loss form:
        L = 0.5 * s_fp  * L_fp  - 0.5 * log(s_fp)
          + gate_weight * (0.5 * s_ana * L_ana - 0.5 * log(s_ana))

    When gate_weight ≈ 0 the loss degenerates to physics-only.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,  # kept for backward compat, no longer used as hard cutoff
        peak_threshold: float = 0.1,         # kept for backward compat
        burn_in_epochs: int = 1,
        burn_in_gate_floor: float = 0.5,
    ) -> None:
        super().__init__()
        # Legacy thresholds kept for logging / external callers that read them.
        self.threshold = float(confidence_threshold)
        self.peak_threshold = float(max(peak_threshold, 0.0))
        self.burn_in_epochs = int(max(burn_in_epochs, 0))
        self.burn_in_gate_floor = float(min(max(burn_in_gate_floor, 0.0), 1.0))
        # log(s) where s = 1/σ².  Init at 0 → s=1 → equal weighting.
        self.log_s = nn.Parameter(torch.zeros(2))
        # Watson soft gate
        self.watson = WatsonGate()
        # Mechanism-aware embedding encoder (MechanismEncoder lives here so its
        # params are automatically included when the trainer calls
        # list(self.gated_loss.parameters()))
        from nexus.reasoning.metric_learner import HGNNProjection, MechanismEncoder
        self.mechanism_encoder = MechanismEncoder()
        self.hyperbolic_projector = HGNNProjection(hidden_dim=256, poincare_dim=128, c=1.0)

    def forward(
        self,
        pred_fp: torch.Tensor,
        pred_ana: torch.Tensor,
        target_idx: torch.Tensor,
        retrieval_confidence: float,
        transport_succeeded: bool,
        physics_analogy_agreement: float = 0.0,
        transported_mass: float = 0.0,
        transport_backend: str = "",
        current_epoch: int | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # ── shape guards ──────────────────────────────────────────────
        if pred_fp.dim() == 1:
            pred_fp = pred_fp.unsqueeze(0)
        if target_idx.dim() == 0:
            target_idx = target_idx.unsqueeze(0)

        loss_fp = F.cross_entropy(pred_fp, target_idx)
        s_fp = torch.exp(self.log_s[0])
        s_ana = torch.exp(self.log_s[1])

        if pred_ana.dim() == 1:
            pred_ana = pred_ana.unsqueeze(0)

        ana_peak = pred_ana.max(dim=-1).values.mean().item() if pred_ana.numel() > 0 else 0.0
        support_quality = max(0.0, min(float(transported_mass) / 0.25, 1.0))
        backend_name = str(transport_backend or "")
        confidence_bonus = 0.0
        transport_gate = 0.0
        if transport_succeeded:
            if support_quality >= 0.8:
                confidence_bonus += 0.30
            elif support_quality >= 0.3:
                confidence_bonus += 0.15
            elif support_quality >= 0.08:
                confidence_bonus += 0.05

            if "mcs_fallback" in backend_name and support_quality >= 0.8:
                confidence_bonus += 0.10
                transport_gate = 0.85
            elif "som_neighborhood" in backend_name:
                if support_quality >= 0.2:
                    confidence_bonus += 0.08
                    transport_gate = 0.40
                elif support_quality >= 0.08:
                    confidence_bonus += 0.03
                    transport_gate = 0.20
                else:
                    transport_gate = 0.05
            else:
                transport_gate = max(0.10, support_quality)

        effective_confidence = min(max(float(retrieval_confidence) + confidence_bonus, 0.0), 1.0)
        in_burn_in = int(current_epoch or 0) < self.burn_in_epochs

        # ── soft Watson gate ──────────────────────────────────────────
        # Gate weight is ~0.12 at init; rises as all three signals converge.
        # Skip analogy branch entirely if transport failed (no sensible label).
        if not transport_succeeded:
            gate_weight = pred_fp.new_zeros(())   # exactly 0, no grad through ana
        elif in_burn_in:
            gate_weight = pred_fp.new_tensor(max(transport_gate, self.burn_in_gate_floor))
        else:
            gate_weight = self.watson(effective_confidence, ana_peak, physics_analogy_agreement)
            # transport_gate is a quality signal [0..0.85].  Previously we
            # multiplied gate_weight × transport_gate, compounding the Watson
            # sigmoid (~0.12) with the transport quality floor (~0.10) to give
            # an effective gate of ~0.012 — too small to learn from.
            # Now we only scale down when transport quality is very low (< 0.15);
            # otherwise the Watson gate runs at its full learned value so it can
            # actually open as training progresses.
            if transport_gate < 0.15:
                gate_weight = gate_weight * pred_fp.new_tensor(transport_gate / 0.15)

        loss_term_fp = 0.5 * s_fp * loss_fp - 0.5 * self.log_s[0]

        # ── analogical loss ───────────────────────────────────────────
        # Convert one-hot pred_ana to logits in a numerically safe range.
        # Scaling to [-5, +5] keeps the analogy signal commensurate with the
        # physics logits.
        pred_ana_logits = 10.0 * (pred_ana - 0.5)   # maps {0,1} → {-5, +5}
        loss_ana = F.cross_entropy(pred_ana_logits, target_idx)
        loss_term_ana = 0.5 * s_ana * loss_ana - 0.5 * self.log_s[1]

        total_loss = loss_term_fp + gate_weight * loss_term_ana

        gate_w_scalar = float(gate_weight.detach().item()) if torch.is_tensor(gate_weight) else 0.0

        return total_loss, {
            "loss_raw_physics": loss_fp.item(),
            "loss_raw_analogy": loss_ana.item(),
            "gate_open": gate_w_scalar,
            "analogy_peak": ana_peak,
            "gate_conf_ok": 1.0 if effective_confidence >= self.threshold else 0.0,
            "gate_peak_ok": 1.0 if ana_peak >= self.peak_threshold else 0.0,
            "weight_physics": s_fp.item(),
            "weight_analogy": s_ana.item(),
            "watson_agreement": float(physics_analogy_agreement),
            "effective_confidence": float(effective_confidence),
            "transport_gate": float(transport_gate),
            "burn_in_active": float(1.0 if in_burn_in else 0.0),
        }


__all__ = [
    "AnalogicalGodLoss",
    "AnalogicalLossBreakdown",
    "GatedAnalogicalGodLoss",
    "GodLossBreakdown",
    "ListMLERankingLoss",
    "NEXUS_God_Loss",
]
