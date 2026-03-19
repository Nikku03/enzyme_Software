from __future__ import annotations

from typing import List, Optional

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class ListMLELoss(nn.Module):
        """Listwise ranking loss that promotes positives to the top of the list."""

        def __init__(self, temperature: float = 1.0, top_k: Optional[int] = None):
            super().__init__()
            self.temperature = float(temperature)
            self.top_k = top_k

        def forward(self, scores: torch.Tensor, positive_indices: List[int]) -> torch.Tensor:
            scores = scores.view(-1)
            if len(positive_indices) == 0:
                return scores.sum() * 0.0
            positive_set = set(int(idx) for idx in positive_indices)
            negative_indices = [idx for idx in range(int(scores.numel())) if idx not in positive_set]
            target_order = list(positive_indices) + negative_indices
            ordered_scores = scores[target_order] / max(self.temperature, 1.0e-6)
            n_positions = min(len(positive_indices), self.top_k or len(positive_indices))
            log_likelihood = ordered_scores.new_tensor(0.0)
            for idx in range(n_positions):
                remaining = ordered_scores[idx:]
                log_likelihood = log_likelihood + remaining[0] - torch.logsumexp(remaining, dim=0)
            return -log_likelihood / max(1, n_positions)


    class ApproxNDCGLoss(nn.Module):
        """Approximate NDCG objective for top-k ranking quality."""

        def __init__(self, temperature: float = 1.0, top_k: int = 10):
            super().__init__()
            self.temperature = float(temperature)
            self.top_k = int(top_k)

        def forward(self, scores: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
            scores = scores.view(-1)
            relevance = relevance.view(-1).float()
            if scores.numel() == 0 or not bool((relevance > 0).any()):
                return scores.sum() * 0.0
            scaled = scores / max(self.temperature, 1.0e-6)
            soft_scores = F.softmax(scaled, dim=0)
            k = min(self.top_k, int(scores.numel()))
            top_vals, top_idx = soft_scores.topk(k)
            discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=scores.device, dtype=scores.dtype))
            dcg = (relevance[top_idx] * discounts).sum()
            ideal_idx = relevance.argsort(descending=True)[:k]
            idcg = (relevance[ideal_idx] * discounts).sum().clamp_min(1.0e-6)
            return 1.0 - (dcg / idcg)

else:  # pragma: no cover
    class ListMLELoss:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class ApproxNDCGLoss:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
