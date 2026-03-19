from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES


class FeatureStacker:
    def __init__(self, model_names: Iterable[str], num_cyp: int = len(MAJOR_CYP_CLASSES)):
        require_torch()
        self.model_names = list(model_names)
        self.num_cyp = int(num_cyp)
        self.atom_feature_dim = len(self.model_names) * 3 + 2
        self.global_feature_dim = len(self.model_names) * self.num_cyp + len(self.model_names) + 1

    def _scores_for_model(self, pred: Optional[Dict[str, object]], num_atoms: int) -> torch.Tensor:
        if pred is None or pred.get("site_scores") is None:
            return torch.zeros((num_atoms,), dtype=torch.float32)
        scores = pred["site_scores"].detach().cpu().view(-1).float()
        if scores.shape[0] == num_atoms:
            return scores
        if scores.shape[0] < num_atoms:
            padded = torch.zeros((num_atoms,), dtype=torch.float32)
            padded[: scores.shape[0]] = scores
            return padded
        return scores[:num_atoms]

    def _cyp_for_model(self, pred: Optional[Dict[str, object]]) -> torch.Tensor:
        if pred is None or pred.get("cyp_probs") is None:
            return torch.full((self.num_cyp,), 1.0 / float(self.num_cyp), dtype=torch.float32)
        probs = pred["cyp_probs"].detach().cpu().view(-1).float()
        if probs.shape[0] < self.num_cyp:
            out = torch.full((self.num_cyp,), 1.0 / float(self.num_cyp), dtype=torch.float32)
            out[: probs.shape[0]] = probs
            return out
        return probs[: self.num_cyp]

    def stack(self, predictions: Dict[str, Optional[Dict[str, object]]]) -> Dict[str, torch.Tensor]:
        available = [pred for pred in predictions.values() if pred is not None]
        if not available:
            raise ValueError("No valid predictions to stack")
        num_atoms = int(available[0]["num_atoms"])
        site_scores = torch.stack([self._scores_for_model(predictions.get(name), num_atoms) for name in self.model_names], dim=1)
        site_ranks = torch.zeros_like(site_scores)
        for idx in range(site_scores.shape[1]):
            order = torch.argsort(site_scores[:, idx], descending=True)
            rank = torch.empty_like(order, dtype=torch.float32)
            rank[order] = torch.arange(order.numel(), dtype=torch.float32)
            site_ranks[:, idx] = rank / max(1.0, float(num_atoms - 1))
        site_presence = torch.tensor(
            [1.0 if predictions.get(name) is not None else 0.0 for name in self.model_names],
            dtype=torch.float32,
        )
        presence_atoms = site_presence.view(1, -1).expand(num_atoms, -1)
        score_var = site_scores.var(dim=1, keepdim=True, unbiased=False)
        score_agree = (site_scores.std(dim=1, keepdim=True, unbiased=False) < 0.1).float()
        atom_features = torch.cat([site_scores, site_ranks, presence_atoms, score_var, score_agree], dim=1)
        cyp_probs = torch.stack([self._cyp_for_model(predictions.get(name)) for name in self.model_names], dim=0)
        top_cyps = torch.argmax(cyp_probs, dim=1)
        cyp_agree = torch.tensor([1.0 if bool(torch.all(top_cyps == top_cyps[0])) else 0.0], dtype=torch.float32)
        global_features = torch.cat([cyp_probs.reshape(-1), site_presence, cyp_agree], dim=0)
        site_labels = available[0].get("site_labels")
        if site_labels is None:
            site_labels = torch.zeros((num_atoms,), dtype=torch.float32)
        else:
            site_labels = site_labels.detach().cpu().view(-1).float()
        cyp_label = available[0].get("cyp_label")
        if cyp_label is None:
            cyp_label = 0
        return {
            "atom_features": atom_features,
            "global_features": global_features,
            "site_scores_raw": site_scores,
            "cyp_probs_raw": cyp_probs,
            "site_labels": site_labels,
            "cyp_label": torch.tensor(int(cyp_label), dtype=torch.long),
            "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
            "available_models": site_presence,
        }
