from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, nn, require_torch, torch

from .listmle_loss import ListMLELoss
from .mirank_loss import MIRankLoss


if TORCH_AVAILABLE:
    class FocalLoss(nn.Module):
        def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
            super().__init__()
            self.alpha = float(alpha)
            self.gamma = float(gamma)

        def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            scores = scores.view(-1)
            labels = labels.view(-1).float()
            probs = torch.sigmoid(scores)
            pt = torch.where(labels > 0.5, probs, 1.0 - probs)
            alpha = torch.where(labels > 0.5, torch.full_like(labels, self.alpha), torch.full_like(labels, 1.0 - self.alpha))
            focal_weight = (1.0 - pt).pow(self.gamma)
            bce = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
            return (alpha * focal_weight * bce).mean()


    class CombinedSiteRankingLoss(nn.Module):
        def __init__(
            self,
            mirank_weight: float = 1.0,
            listmle_weight: float = 0.5,
            bce_weight: float = 0.3,
            focal_weight: float = 0.2,
            margin: float = 1.0,
            temperature: float = 1.0,
            hard_negative_fraction: Optional[float] = 0.5,
        ):
            super().__init__()
            self.mirank = MIRankLoss(margin=margin, hard_negative_fraction=hard_negative_fraction)
            self.listmle = ListMLELoss(temperature=temperature)
            self.focal = FocalLoss()
            self.mirank_weight = float(mirank_weight)
            self.listmle_weight = float(listmle_weight)
            self.bce_weight = float(bce_weight)
            self.focal_weight = float(focal_weight)

        def forward(
            self,
            scores: torch.Tensor,
            labels: torch.Tensor,
            positive_indices: Optional[List[int]] = None,
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            scores = scores.view(-1)
            labels = labels.view(-1).float()
            if positive_indices is None:
                positive_indices = torch.nonzero(labels > 0.5, as_tuple=False).view(-1).tolist()
            total = scores.sum() * 0.0
            stats: Dict[str, float] = {}

            if self.mirank_weight > 0.0 and positive_indices:
                mirank_loss = self.mirank(scores, positive_indices)
                total = total + self.mirank_weight * mirank_loss
                stats["mirank"] = float(mirank_loss.item())

            if self.listmle_weight > 0.0 and positive_indices:
                listmle_loss = self.listmle(scores, positive_indices)
                total = total + self.listmle_weight * listmle_loss
                stats["listmle"] = float(listmle_loss.item())

            if self.bce_weight > 0.0:
                bce_loss = F.binary_cross_entropy_with_logits(scores, labels)
                total = total + self.bce_weight * bce_loss
                stats["bce"] = float(bce_loss.item())

            if self.focal_weight > 0.0:
                focal_loss = self.focal(scores, labels)
                total = total + self.focal_weight * focal_loss
                stats["focal"] = float(focal_loss.item())

            stats["total"] = float(total.item())
            return total, stats


    class SiteRankingLossV2(nn.Module):
        def __init__(
            self,
            mirank_weight: float = 1.0,
            bce_weight: float = 0.3,
            margin: float = 1.0,
            hard_negative_fraction: Optional[float] = None,
        ):
            super().__init__()
            self.mirank = MIRankLoss(margin=margin, hard_negative_fraction=hard_negative_fraction)
            self.mirank_weight = float(mirank_weight)
            self.bce_weight = float(bce_weight)

        def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
            scores = scores.view(-1)
            labels = labels.view(-1).float()
            positive_indices = torch.nonzero(labels > 0.5, as_tuple=False).view(-1).tolist()
            mirank_loss = self.mirank(scores, positive_indices) if positive_indices else scores.sum() * 0.0
            bce_loss = F.binary_cross_entropy_with_logits(scores, labels)
            total = self.mirank_weight * mirank_loss + self.bce_weight * bce_loss
            return total, {
                "mirank": float(mirank_loss.item()),
                "bce": float(bce_loss.item()),
                "total": float(total.item()),
            }

else:  # pragma: no cover
    class FocalLoss:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class CombinedSiteRankingLoss:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class SiteRankingLossV2:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
