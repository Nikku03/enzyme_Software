from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.data.cyp_classes import ALL_CYP_CLASSES, MAJOR_CYP_CLASSES


CYP_COUNTS = {
    "CYP1A2": 50,
    "CYP2A6": 8,
    "CYP2B6": 8,
    "CYP2C8": 37,
    "CYP2C9": 49,
    "CYP2C19": 25,
    "CYP2D6": 69,
    "CYP2E1": 11,
    "CYP3A4": 316,
}

CYP_ORDER = list(ALL_CYP_CLASSES)
MAJOR_CYP_ORDER = list(MAJOR_CYP_CLASSES)


if TORCH_AVAILABLE:
    def compute_cyp_weights(counts: Optional[Dict[str, int]] = None, max_weight: float = 10.0, cyp_order=None):
        counts = counts or CYP_COUNTS
        order = list(cyp_order or CYP_ORDER)
        count_list = [max(1, int(counts.get(cyp, 1))) for cyp in order]
        total = float(sum(count_list))
        num_classes = len(count_list)
        raw = [total / (num_classes * c) for c in count_list]
        max_raw = max(raw) if raw else 1.0
        weights = [min(float(max_weight), (w / max_raw) * float(max_weight)) for w in raw]
        return torch.tensor(weights, dtype=torch.float32)


    def _align_class_weights(logits, weights, counts: Optional[Dict[str, int]] = None, max_weight: float = 10.0):
        if weights is None:
            num_classes = int(logits.size(-1))
            if num_classes == len(MAJOR_CYP_ORDER):
                return compute_cyp_weights(counts, max_weight=max_weight, cyp_order=MAJOR_CYP_ORDER).to(logits.device)
            if num_classes == len(CYP_ORDER):
                return compute_cyp_weights(counts, max_weight=max_weight, cyp_order=CYP_ORDER).to(logits.device)
            return None
        if int(weights.numel()) == int(logits.size(-1)):
            return weights.to(logits.device)
        num_classes = int(logits.size(-1))
        if num_classes == len(MAJOR_CYP_ORDER):
            return compute_cyp_weights(counts, max_weight=max_weight, cyp_order=MAJOR_CYP_ORDER).to(logits.device)
        if num_classes == len(CYP_ORDER):
            return compute_cyp_weights(counts, max_weight=max_weight, cyp_order=CYP_ORDER).to(logits.device)
        return None


    def focal_cross_entropy(logits, labels, gamma: float = 2.0, weights=None, label_smoothing: float = 0.1, counts: Optional[Dict[str, int]] = None, max_weight: float = 10.0):
        weights = _align_class_weights(logits, weights, counts=counts, max_weight=max_weight)
        ce = F.cross_entropy(
            logits,
            labels,
            weight=weights,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        pt = torch.exp(-ce)
        focal_weight = (1 - pt) ** float(gamma)
        return (focal_weight * ce).mean()


    class ImbalancedCYPLoss(nn.Module):
        def __init__(
            self,
            cyp_counts: Optional[Dict[str, int]] = None,
            gamma: float = 2.0,
            max_weight: float = 10.0,
            label_smoothing: float = 0.1,
            device=None,
        ):
            super().__init__()
            self.gamma = float(gamma)
            self.label_smoothing = float(label_smoothing)
            self.cyp_counts = cyp_counts or CYP_COUNTS
            self.max_weight = float(max_weight)
            weights = compute_cyp_weights(self.cyp_counts, max_weight)
            if device is not None:
                weights = weights.to(device)
            self.register_buffer("weights", weights)

        def forward(self, logits, labels):
            return focal_cross_entropy(
                logits,
                labels,
                gamma=self.gamma,
                weights=self.weights,
                label_smoothing=self.label_smoothing,
                counts=self.cyp_counts,
                max_weight=self.max_weight,
            )


    class FocalLoss(nn.Module):
        """Focal loss for imbalanced site prediction."""

        def __init__(self, gamma: float = 2.0, pos_weight: float = 10.0):
            super().__init__()
            self.gamma = float(gamma)
            self.pos_weight = float(pos_weight)

        def forward(self, logits, labels, supervision_mask=None):
            probs = torch.sigmoid(logits)
            p_t = torch.where(labels == 1, probs, 1 - probs)
            focal_weight = (1 - p_t) ** self.gamma
            class_weight = torch.where(
                labels == 1,
                torch.full_like(logits, self.pos_weight),
                torch.ones_like(logits),
            )
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = focal_weight * class_weight * bce
            if supervision_mask is None:
                return loss.mean()
            mask = supervision_mask.to(dtype=loss.dtype)
            denom = mask.sum().clamp_min(1.0)
            return (loss * mask).sum() / denom


    class RankingLoss(nn.Module):
        """Pairwise ranking loss for per-molecule site ordering."""

        def __init__(self, margin: float = 1.0):
            super().__init__()
            self.margin = float(margin)

        def forward(self, scores, labels, batch, supervision_mask=None):
            scores = scores.squeeze(-1)
            labels = labels.squeeze(-1)
            if supervision_mask is not None:
                supervision_mask = supervision_mask.squeeze(-1)
            num_molecules = int(batch.max().item()) + 1 if batch.numel() else 0
            total_loss = torch.tensor(0.0, device=scores.device)
            num_pairs = 0

            for mol_idx in range(num_molecules):
                mask = batch == mol_idx
                if supervision_mask is not None and not bool((supervision_mask[mask] > 0.5).any()):
                    continue
                mol_scores = scores[mask]
                mol_labels = labels[mask]
                pos_idx = torch.where(mol_labels == 1)[0]
                neg_idx = torch.where(mol_labels == 0)[0]
                if len(pos_idx) == 0 or len(neg_idx) == 0:
                    continue
                neg_scores = mol_scores[neg_idx]
                for p in pos_idx:
                    pos_score = mol_scores[p]
                    violations = F.relu(self.margin - (pos_score - neg_scores))
                    total_loss = total_loss + violations.mean()
                    num_pairs += 1

            if num_pairs == 0:
                return torch.tensor(0.0, device=scores.device)
            return total_loss / num_pairs


    class SiteOfMetabolismLoss(nn.Module):
        """Combined focal classification + ranking loss for SoM prediction."""

        def __init__(
            self,
            focal_gamma: float = 2.0,
            pos_weight: float = 10.0,
            ranking_margin: float = 1.0,
            ranking_weight: float = 0.5,
        ):
            super().__init__()
            self.focal = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
            self.ranking = RankingLoss(margin=ranking_margin)
            self.ranking_weight = float(ranking_weight)

        def forward(self, logits, labels, batch, supervision_mask=None):
            focal_loss = self.focal(logits, labels, supervision_mask=supervision_mask)
            ranking_loss = self.ranking(torch.sigmoid(logits), labels, batch, supervision_mask=supervision_mask)
            total = focal_loss + self.ranking_weight * ranking_loss
            return total, {
                "focal_loss": float(focal_loss.item()),
                "ranking_loss": float(ranking_loss.item()),
                "site_loss": float(total.item()),
            }


    class AdaptiveLossV2(nn.Module):
        """Improved multi-task loss with focal + ranking site loss."""

        def __init__(
            self,
            tau_reg_weight: float = 0.01,
            energy_loss_weight: float = 0.0,
            deliberation_loss_weight: float = 0.0,
            energy_margin: float = 0.15,
            energy_loss_clip: float = 2.0,
            cyp_class_weights=None,
            cyp_counts: Optional[Dict[str, int]] = None,
            cyp_focal_gamma: float = 2.0,
            cyp_max_weight: float = 10.0,
            cyp_label_smoothing: float = 0.1,
        ):
            super().__init__()
            self.site_loss = SiteOfMetabolismLoss(
                focal_gamma=2.0,
                pos_weight=15.0,
                ranking_margin=1.0,
                ranking_weight=0.5,
            )
            self.log_var_site = nn.Parameter(torch.tensor(0.0))
            self.log_var_cyp = nn.Parameter(torch.tensor(0.0))
            self.tau_reg_weight = float(tau_reg_weight)
            self.energy_loss_weight = float(energy_loss_weight)
            self.deliberation_loss_weight = float(deliberation_loss_weight)
            self.energy_margin = float(energy_margin)
            self.energy_loss_clip = max(0.1, float(energy_loss_clip))
            if cyp_class_weights is not None:
                weights = cyp_class_weights
                if not hasattr(weights, "to"):
                    weights = torch.as_tensor(weights, dtype=torch.float32)
                self.cyp_loss_fn = ImbalancedCYPLoss(
                    gamma=cyp_focal_gamma,
                    max_weight=cyp_max_weight,
                    label_smoothing=cyp_label_smoothing,
                    device=weights.device if hasattr(weights, "device") else None,
                )
                with torch.no_grad():
                    self.cyp_loss_fn.weights.copy_(weights.to(self.cyp_loss_fn.weights.device))
            else:
                self.cyp_loss_fn = ImbalancedCYPLoss(
                    cyp_counts=cyp_counts,
                    gamma=cyp_focal_gamma,
                    max_weight=cyp_max_weight,
                    label_smoothing=cyp_label_smoothing,
                )

        def forward(
            self,
            site_logits,
            cyp_logits,
            site_labels,
            cyp_labels,
            batch,
            site_supervision_mask=None,
            tau_history=None,
            tau_init=None,
            energy_outputs=None,
            deliberation_outputs=None,
        ):
            site_loss, site_dict = self.site_loss(
                site_logits,
                site_labels,
                batch,
                supervision_mask=site_supervision_mask,
            )
            cyp_loss = self.cyp_loss_fn(cyp_logits, cyp_labels)
            log_var_site = self.log_var_site.clamp(min=-4.0, max=4.0)
            log_var_cyp = self.log_var_cyp.clamp(min=-4.0, max=4.0)
            precision_site = torch.exp(-log_var_site)
            precision_cyp = torch.exp(-log_var_cyp)
            total_loss = (
                0.5 * precision_site * site_loss + 0.5 * log_var_site
                + 0.5 * precision_cyp * cyp_loss + 0.5 * log_var_cyp
            )
            tau_reg_loss = torch.tensor(0.0, device=site_logits.device)
            if tau_history is not None and tau_init is not None:
                valid = [F.mse_loss(tau, tau_init) for tau in tau_history if tau.shape == tau_init.shape]
                if valid:
                    tau_reg_loss = torch.stack(valid).mean()
                    total_loss = total_loss + self.tau_reg_weight * tau_reg_loss
            energy_loss = torch.tensor(0.0, device=site_logits.device)
            energy_gap = torch.tensor(0.0, device=site_logits.device)
            if self.energy_loss_weight > 0.0 and energy_outputs is not None:
                node_energy = energy_outputs.get("node_energy")
                if node_energy is not None and node_energy.shape[:1] == site_labels.shape[:1]:
                    node_energy = torch.nan_to_num(
                        node_energy,
                        nan=0.0,
                        posinf=self.energy_loss_clip,
                        neginf=-self.energy_loss_clip,
                    ).clamp(min=-self.energy_loss_clip, max=self.energy_loss_clip)
                    supervised_mask = (
                        site_supervision_mask > 0.5
                        if site_supervision_mask is not None
                        else torch.ones_like(site_labels, dtype=torch.bool)
                    )
                    pos_mask = (site_labels > 0.5) & supervised_mask
                    neg_mask = (site_labels <= 0.5) & supervised_mask
                    if bool(pos_mask.any()) and bool(neg_mask.any()):
                        pos_mean = node_energy[pos_mask].mean()
                        neg_mean = node_energy[neg_mask].mean()
                        energy_gap = (neg_mean - pos_mean).clamp(
                            min=-self.energy_loss_clip,
                            max=self.energy_loss_clip,
                        )
                        violation = torch.clamp(
                            self.energy_margin - energy_gap,
                            min=0.0,
                            max=self.energy_loss_clip,
                        )
                        energy_loss = F.smooth_l1_loss(
                            violation,
                            torch.zeros_like(violation),
                            beta=0.5,
                        )
                    else:
                        energy_loss = F.smooth_l1_loss(
                            node_energy,
                            torch.zeros_like(node_energy),
                            beta=0.5,
                        ).clamp(max=self.energy_loss_clip)
                    total_loss = total_loss + self.energy_loss_weight * energy_loss
            deliberation_loss = torch.tensor(0.0, device=site_logits.device)
            if self.deliberation_loss_weight > 0.0 and deliberation_outputs is not None:
                site_steps = deliberation_outputs.get("site_logits") or []
                cyp_steps = deliberation_outputs.get("cyp_logits") or []
                penalties = []
                if len(site_steps) > 1:
                    final_site = site_steps[-1].detach()
                    penalties.extend(F.mse_loss(step, final_site) for step in site_steps[:-1])
                if len(cyp_steps) > 1:
                    final_cyp = cyp_steps[-1].detach()
                    penalties.extend(F.mse_loss(step, final_cyp) for step in cyp_steps[:-1])
                if penalties:
                    deliberation_loss = torch.stack(penalties).mean()
                    total_loss = total_loss + self.deliberation_loss_weight * deliberation_loss
            return total_loss, {
                **site_dict,
                "cyp_loss": float(cyp_loss.item()),
                "tau_reg_loss": float(tau_reg_loss.item()),
                "energy_loss": float(energy_loss.item()),
                "energy_gap": float(energy_gap.item()),
                "deliberation_loss": float(deliberation_loss.item()),
                "total_loss": float(total_loss.item()),
                "site_weight": float(precision_site.item()),
                "cyp_weight": float(precision_cyp.item()),
                "log_var_site": float(log_var_site.item()),
                "log_var_cyp": float(log_var_cyp.item()),
            }


    AdaptiveLoss = AdaptiveLossV2
else:  # pragma: no cover
    def compute_cyp_weights(*args, **kwargs):
        require_torch()

    def focal_cross_entropy(*args, **kwargs):
        require_torch()

    class ImbalancedCYPLoss:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class FocalLoss:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class RankingLoss:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class SiteOfMetabolismLoss:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class AdaptiveLossV2:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class AdaptiveLoss:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
