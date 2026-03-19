from __future__ import annotations

from typing import List, Optional

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class MIRankLoss(nn.Module):
        """Pairwise margin ranking loss for site ordering."""

        def __init__(
            self,
            margin: float = 1.0,
            hard_negative_fraction: Optional[float] = None,
            soft_negative_threshold: Optional[float] = None,
            reduction: str = "mean",
        ):
            super().__init__()
            self.margin = float(margin)
            self.hard_negative_fraction = hard_negative_fraction
            self.soft_negative_threshold = soft_negative_threshold
            self.reduction = str(reduction)

        def forward(
            self,
            scores: torch.Tensor,
            positive_indices: List[int],
            negative_indices: Optional[List[int]] = None,
            weights: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            scores = scores.view(-1)
            if len(positive_indices) == 0:
                return scores.sum() * 0.0

            if negative_indices is None:
                positive_set = set(int(idx) for idx in positive_indices)
                negative_indices = [idx for idx in range(int(scores.numel())) if idx not in positive_set]
            if len(negative_indices) == 0:
                return scores.sum() * 0.0

            pos_scores = scores[positive_indices]
            neg_scores = scores[negative_indices]

            if self.hard_negative_fraction is not None and len(negative_indices) > 1:
                frac = min(max(float(self.hard_negative_fraction), 0.0), 1.0)
                n_hard = max(1, int(round(len(negative_indices) * frac)))
                hard_idx = neg_scores.argsort(descending=True)[:n_hard]
                neg_scores = neg_scores[hard_idx]

            if self.soft_negative_threshold is not None:
                keep = neg_scores < float(self.soft_negative_threshold)
                if bool(keep.any()):
                    neg_scores = neg_scores[keep]
                else:
                    return scores.sum() * 0.0

            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            losses = F.relu(self.margin - diff)

            if weights is not None:
                weight_vec = weights.view(-1)[positive_indices].unsqueeze(1)
                losses = losses * weight_vec

            if self.reduction == "sum":
                return losses.sum()
            if self.reduction == "none":
                return losses
            return losses.mean()


    class MarginRankingLoss(nn.Module):
        """Sampled margin ranking loss built on torch.nn.MarginRankingLoss."""

        def __init__(self, margin: float = 1.0, n_samples: int = 100):
            super().__init__()
            self.margin = float(margin)
            self.n_samples = int(n_samples)
            self.loss_fn = nn.MarginRankingLoss(margin=self.margin)

        def forward(
            self,
            scores: torch.Tensor,
            positive_indices: List[int],
            negative_indices: Optional[List[int]] = None,
        ) -> torch.Tensor:
            scores = scores.view(-1)
            if len(positive_indices) == 0:
                return scores.sum() * 0.0

            if negative_indices is None:
                positive_set = set(int(idx) for idx in positive_indices)
                negative_indices = [idx for idx in range(int(scores.numel())) if idx not in positive_set]
            if len(negative_indices) == 0:
                return scores.sum() * 0.0

            n_pairs = min(self.n_samples, len(positive_indices) * len(negative_indices))
            pos_tensor = torch.as_tensor(positive_indices, device=scores.device, dtype=torch.long)
            neg_tensor = torch.as_tensor(negative_indices, device=scores.device, dtype=torch.long)
            pos_samples = pos_tensor[torch.randint(len(pos_tensor), (n_pairs,), device=scores.device)]
            neg_samples = neg_tensor[torch.randint(len(neg_tensor), (n_pairs,), device=scores.device)]
            target = torch.ones((n_pairs,), device=scores.device, dtype=scores.dtype)
            return self.loss_fn(scores[pos_samples], scores[neg_samples], target)

else:  # pragma: no cover
    class MIRankLoss:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class MarginRankingLoss:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
