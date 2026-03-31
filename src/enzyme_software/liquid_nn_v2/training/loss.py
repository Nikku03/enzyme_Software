from __future__ import annotations

from typing import Dict, Optional

from enzyme_software.losses import ListMLELoss, MIRankLoss
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


    def focal_cross_entropy(
        logits,
        labels,
        gamma: float = 2.0,
        weights=None,
        label_smoothing: float = 0.1,
        counts: Optional[Dict[str, int]] = None,
        max_weight: float = 10.0,
        sample_weights=None,
    ):
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
        loss = focal_weight * ce
        if sample_weights is not None:
            sample_weights = sample_weights.view(-1).to(dtype=loss.dtype, device=loss.device)
            denom = sample_weights.sum().clamp_min(1.0e-6)
            return (loss * sample_weights).sum() / denom
        return loss.mean()


    class ImbalancedCYPLoss(nn.Module):
        def __init__(
            self,
            cyp_counts: Optional[Dict[str, int]] = None,
            cyp_order=None,
            gamma: float = 2.0,
            max_weight: float = 10.0,
            label_smoothing: float = 0.1,
            device=None,
        ):
            super().__init__()
            self.gamma = float(gamma)
            self.label_smoothing = float(label_smoothing)
            self.cyp_counts = cyp_counts or CYP_COUNTS
            self.cyp_order = list(cyp_order or CYP_ORDER)
            self.max_weight = float(max_weight)
            weights = compute_cyp_weights(self.cyp_counts, max_weight, cyp_order=self.cyp_order)
            if device is not None:
                weights = weights.to(device)
            self.register_buffer("weights", weights)

        def forward(self, logits, labels, sample_weights=None):
            return focal_cross_entropy(
                logits,
                labels,
                gamma=self.gamma,
                weights=self.weights,
                label_smoothing=self.label_smoothing,
                counts=self.cyp_counts,
                max_weight=self.max_weight,
                sample_weights=sample_weights,
            )


    class FocalLoss(nn.Module):
        """Focal loss for imbalanced site prediction."""

        def __init__(self, gamma: float = 2.0, pos_weight: float = 10.0, label_smoothing: float = 0.0):
            super().__init__()
            self.gamma = float(gamma)
            self.pos_weight = float(pos_weight)
            self.label_smoothing = float(label_smoothing)

        def forward(self, logits, labels, supervision_mask=None, node_weights=None, batch=None):
            probs = torch.sigmoid(logits)
            p_t = torch.where(labels == 1, probs, 1 - probs)
            focal_weight = (1 - p_t) ** self.gamma
            # Smooth binary targets: positive → (1-ε), negative → ε/2
            if self.label_smoothing > 0.0:
                smooth_labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            else:
                smooth_labels = labels
            # Adaptive pos_weight: per-molecule (n_neg / n_pos), clamped to [3, 30]
            if batch is not None and batch.numel():
                pos_weight_per_atom = torch.full_like(logits, self.pos_weight)
                num_mol = int(batch.max().item()) + 1
                labels_flat = labels.view(-1)
                for mol_idx in range(num_mol):
                    mol_mask = (batch == mol_idx)
                    n_atoms = float(mol_mask.sum().item())
                    n_pos = float((labels_flat[mol_mask] > 0.5).sum().item())
                    w = ((n_atoms - n_pos) / n_pos) if n_pos > 0 else self.pos_weight
                    w = max(3.0, min(30.0, w))
                    pos_weight_per_atom[mol_mask] = w
                class_weight = torch.where(labels == 1, pos_weight_per_atom, torch.ones_like(logits))
            else:
                class_weight = torch.where(
                    labels == 1,
                    torch.full_like(logits, self.pos_weight),
                    torch.ones_like(logits),
                )
            bce = F.binary_cross_entropy_with_logits(logits, smooth_labels, reduction="none")
            loss = focal_weight * class_weight * bce
            if node_weights is not None:
                node_weights = node_weights.to(dtype=loss.dtype, device=loss.device)
                if node_weights.shape != loss.shape:
                    node_weights = node_weights.view_as(loss)
                loss = loss * node_weights
            if supervision_mask is None:
                return loss.mean()
            mask = supervision_mask.to(dtype=loss.dtype, device=loss.device)
            if node_weights is not None:
                denom = (mask * node_weights).sum().clamp_min(1.0e-6)
            else:
                denom = mask.sum().clamp_min(1.0)
            return (loss * mask).sum() / denom


    class RankingLoss(nn.Module):
        """MIRank loss reduced over molecules in a batch."""

        def __init__(self, margin: float = 1.0, hard_negative_fraction: Optional[float] = None):
            super().__init__()
            self.mirank = MIRankLoss(margin=margin, hard_negative_fraction=hard_negative_fraction)

        def forward(self, scores, labels, batch, supervision_mask=None, graph_weights=None):
            scores = scores.view(-1)
            labels = labels.view(-1)
            if supervision_mask is not None:
                supervision_mask = supervision_mask.view(-1)
            num_molecules = int(batch.max().item()) + 1 if batch.numel() else 0
            per_molecule = []
            per_weights = []
            for mol_idx in range(num_molecules):
                mask = batch == mol_idx
                if not bool(mask.any()):
                    continue
                mol_scores = scores[mask]
                mol_labels = labels[mask]
                if supervision_mask is not None:
                    supervised = supervision_mask[mask] > 0.5
                    if not bool(supervised.any()):
                        continue
                    mol_scores = mol_scores[supervised]
                    mol_labels = mol_labels[supervised]
                pos_idx = torch.nonzero(mol_labels > 0.5, as_tuple=False).view(-1).tolist()
                if not pos_idx or len(pos_idx) == int(mol_scores.numel()):
                    continue
                per_molecule.append(self.mirank(mol_scores, pos_idx))
                if graph_weights is not None and mol_idx < int(graph_weights.numel()):
                    per_weights.append(graph_weights[mol_idx])
                else:
                    per_weights.append(torch.tensor(1.0, device=scores.device))

            if not per_molecule:
                return scores.sum() * 0.0
            stacked_losses = torch.stack(per_molecule)
            stacked_weights = torch.stack([weight.to(dtype=stacked_losses.dtype, device=stacked_losses.device) for weight in per_weights])
            denom = stacked_weights.sum().clamp_min(1.0e-6)
            return (stacked_losses * stacked_weights).sum() / denom


    class SoftAPLoss(nn.Module):
        """Differentiable approximation to Average Precision.

        Treats the rank of each positive as a temperature-scaled sigmoid sum,
        then computes precision-at-rank per positive and averages.
        """

        def __init__(self, temperature: float = 0.1):
            super().__init__()
            self.temperature = float(temperature)

        def forward(self, scores, positive_indices):
            scores = scores.view(-1)
            if not positive_indices:
                return scores.sum() * 0.0
            pos_scores = scores[positive_indices]               # (P,)
            # soft_rank[i] ≈ number of items ranked at or above positive i
            diff = scores.unsqueeze(0) - pos_scores.unsqueeze(1)  # (P, N)
            soft_rank = torch.sigmoid(diff / self.temperature).sum(dim=1).clamp(min=1.0)  # (P,)
            # Sort positives by descending score; cumulative count = 1, 2, …, P
            sorted_order = pos_scores.argsort(descending=True)
            cum_pos = torch.arange(1, len(positive_indices) + 1, dtype=torch.float32, device=scores.device)
            precision = cum_pos / soft_rank[sorted_order]
            ap = precision.mean()
            return 1.0 - ap


    class ListwiseRankingLoss(nn.Module):
        """ListMLE reduced over molecules in a batch."""

        def __init__(self, temperature: float = 1.0):
            super().__init__()
            self.listmle = ListMLELoss(temperature=temperature)

        def forward(self, scores, labels, batch, supervision_mask=None, graph_weights=None):
            scores = scores.view(-1)
            labels = labels.view(-1)
            if supervision_mask is not None:
                supervision_mask = supervision_mask.view(-1)
            num_molecules = int(batch.max().item()) + 1 if batch.numel() else 0
            per_molecule = []
            per_weights = []
            for mol_idx in range(num_molecules):
                mask = batch == mol_idx
                if not bool(mask.any()):
                    continue
                mol_scores = scores[mask]
                mol_labels = labels[mask]
                if supervision_mask is not None:
                    supervised = supervision_mask[mask] > 0.5
                    if not bool(supervised.any()):
                        continue
                    mol_scores = mol_scores[supervised]
                    mol_labels = mol_labels[supervised]
                pos_idx = torch.nonzero(mol_labels > 0.5, as_tuple=False).view(-1).tolist()
                if not pos_idx:
                    continue
                per_molecule.append(self.listmle(mol_scores, pos_idx))
                if graph_weights is not None and mol_idx < int(graph_weights.numel()):
                    per_weights.append(graph_weights[mol_idx])
                else:
                    per_weights.append(torch.tensor(1.0, device=scores.device))
            if not per_molecule:
                return scores.sum() * 0.0
            stacked_losses = torch.stack(per_molecule)
            stacked_weights = torch.stack([weight.to(dtype=stacked_losses.dtype, device=stacked_losses.device) for weight in per_weights])
            denom = stacked_weights.sum().clamp_min(1.0e-6)
            return (stacked_losses * stacked_weights).sum() / denom


    class SiteOfMetabolismLoss(nn.Module):
        """Combined focal classification + ranking loss for SoM prediction."""

        def __init__(
            self,
            focal_gamma: float = 2.0,
            pos_weight: float = 10.0,
            ranking_margin: float = 1.0,
            ranking_weight: float = 1.0,
            listmle_weight: float = 0.25,
            hard_negative_fraction: Optional[float] = 0.5,
            softap_weight: float = 0.3,
            softap_temperature: float = 0.1,
            label_smoothing: float = 0.0,
            top1_margin_weight: float = 0.0,
            top1_margin_value: float = 0.5,
        ):
            super().__init__()
            self.focal = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight, label_smoothing=label_smoothing)
            self.top1_margin_weight = float(top1_margin_weight)
            self.top1_margin_value = float(top1_margin_value)
            self.ranking = RankingLoss(margin=ranking_margin, hard_negative_fraction=hard_negative_fraction)
            self.listmle = ListwiseRankingLoss()
            self.softap = SoftAPLoss(temperature=softap_temperature)
            self.ranking_weight = float(ranking_weight)
            self.listmle_weight = float(listmle_weight)
            self.softap_weight = float(softap_weight)

        def _top1_margin_loss(self, logits, labels, batch, supervision_mask=None):
            """Penalise each molecule where the top-scored atom is not a true site.

            For every molecule: loss = max(0, margin - (best_pos_logit - best_neg_logit)).
            This directly targets the rank-1 error case that top1 accuracy measures.
            """
            logits_flat = logits.view(-1)
            labels_flat = labels.view(-1)
            sup_flat = supervision_mask.view(-1) if supervision_mask is not None else None
            num_mol = int(batch.max().item()) + 1 if batch.numel() else 0
            losses = []
            for mol_idx in range(num_mol):
                mol_mask = batch == mol_idx
                if not bool(mol_mask.any()):
                    continue
                if sup_flat is not None:
                    supervised = sup_flat[mol_mask] > 0.5
                    if not bool(supervised.any()):
                        continue
                    mol_logits = logits_flat[mol_mask][supervised]
                    mol_labels = labels_flat[mol_mask][supervised]
                else:
                    mol_logits = logits_flat[mol_mask]
                    mol_labels = labels_flat[mol_mask]
                pos_mask = mol_labels > 0.5
                neg_mask = ~pos_mask
                if not bool(pos_mask.any()) or not bool(neg_mask.any()):
                    continue
                best_pos = mol_logits[pos_mask].max()
                best_neg = mol_logits[neg_mask].max()
                losses.append(torch.relu(self.top1_margin_value - (best_pos - best_neg)))
            if not losses:
                return logits.sum() * 0.0
            return torch.stack(losses).mean()

        def forward(self, logits, labels, batch, supervision_mask=None, node_weights=None, graph_weights=None):
            focal_loss = self.focal(logits, labels, supervision_mask=supervision_mask, node_weights=node_weights, batch=batch)
            ranking_loss = self.ranking(logits, labels, batch, supervision_mask=supervision_mask, graph_weights=graph_weights)
            listmle_loss = self.listmle(logits, labels, batch, supervision_mask=supervision_mask, graph_weights=graph_weights)
            # SoftAP: per-molecule, same iteration structure as RankingLoss
            softap_loss = logits.sum() * 0.0
            if self.softap_weight > 0.0 and batch is not None and batch.numel():
                scores_flat = logits.view(-1)
                labels_flat = labels.view(-1)
                sup_flat = supervision_mask.view(-1) if supervision_mask is not None else None
                num_molecules = int(batch.max().item()) + 1
                per_mol_softap = []
                for mol_idx in range(num_molecules):
                    mask = batch == mol_idx
                    if not bool(mask.any()):
                        continue
                    mol_scores = scores_flat[mask]
                    mol_labels = labels_flat[mask]
                    if sup_flat is not None:
                        supervised = sup_flat[mask] > 0.5
                        if not bool(supervised.any()):
                            continue
                        mol_scores = mol_scores[supervised]
                        mol_labels = mol_labels[supervised]
                    pos_idx = torch.nonzero(mol_labels > 0.5, as_tuple=False).view(-1).tolist()
                    if not pos_idx or len(pos_idx) == int(mol_scores.numel()):
                        continue
                    per_mol_softap.append(self.softap(mol_scores, pos_idx))
                if per_mol_softap:
                    softap_loss = torch.stack(per_mol_softap).mean()
            top1_margin_loss = logits.sum() * 0.0
            if self.top1_margin_weight > 0.0:
                top1_margin_loss = self._top1_margin_loss(logits, labels, batch, supervision_mask=supervision_mask)
            total = (
                focal_loss
                + self.ranking_weight * ranking_loss
                + self.listmle_weight * listmle_loss
                + self.softap_weight * softap_loss
                + self.top1_margin_weight * top1_margin_loss
            )
            return total, {
                "focal_loss": float(focal_loss.item()),
                "ranking_loss": float(ranking_loss.item()),
                "listmle_loss": float(listmle_loss.item()),
                "softap_loss": float(softap_loss.item()),
                "top1_margin_loss": float(top1_margin_loss.item()),
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
            site_label_smoothing: float = 0.0,
            site_top1_margin_weight: float = 0.0,
            site_top1_margin_value: float = 0.5,
        ):
            super().__init__()
            self.site_loss = SiteOfMetabolismLoss(
                focal_gamma=2.0,
                pos_weight=15.0,
                ranking_margin=1.0,
                ranking_weight=0.5,
                label_smoothing=float(site_label_smoothing),
                top1_margin_weight=float(site_top1_margin_weight),
                top1_margin_value=float(site_top1_margin_value),
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
                cyp_order = MAJOR_CYP_ORDER if int(weights.numel()) == len(MAJOR_CYP_ORDER) else CYP_ORDER
                self.cyp_loss_fn = ImbalancedCYPLoss(
                    cyp_order=cyp_order,
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
                    cyp_order=MAJOR_CYP_ORDER if cyp_counts and set(cyp_counts).issubset(set(MAJOR_CYP_ORDER)) else CYP_ORDER,
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
            cyp_supervision_mask=None,
            graph_confidence_weights=None,
            node_confidence_weights=None,
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
                node_weights=node_confidence_weights,
                graph_weights=graph_confidence_weights,
            )
            cyp_loss = site_loss * 0.0
            cyp_mask = None
            if cyp_supervision_mask is not None:
                cyp_mask = cyp_supervision_mask.view(-1) > 0.5
            if cyp_mask is None:
                cyp_loss = self.cyp_loss_fn(cyp_logits, cyp_labels, sample_weights=graph_confidence_weights)
            elif bool(cyp_mask.any()):
                masked_weights = graph_confidence_weights[cyp_mask] if graph_confidence_weights is not None else None
                cyp_loss = self.cyp_loss_fn(
                    cyp_logits[cyp_mask],
                    cyp_labels[cyp_mask],
                    sample_weights=masked_weights,
                )
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
