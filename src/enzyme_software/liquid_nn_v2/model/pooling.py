from __future__ import annotations

from typing import Dict, Optional, Tuple

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.data.smarts_patterns import FUNCTIONAL_GROUP_SMARTS


if TORCH_AVAILABLE:
    def segment_sum(values, batch, num_segments: int):
        out = torch.zeros(num_segments, values.size(-1), device=values.device, dtype=values.dtype)
        if batch.numel() == 0:
            return out
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(values), values)
        return out


    def segment_softmax(scores, batch, num_segments: int):
        out = torch.zeros_like(scores)
        for idx in range(int(num_segments)):
            mask = batch == idx
            if torch.any(mask):
                out[mask] = torch.softmax(scores[mask], dim=0)
        return out


    def masked_softmax(scores, mask, dim: int = -1):
        if scores.numel() == 0:
            return scores
        masked_scores = scores.masked_fill(~mask, -1.0e9)
        weights = torch.softmax(masked_scores, dim=dim)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=dim, keepdim=True).clamp(min=1.0e-8)
        return weights / denom


    def pack_atom_features(atom_features, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if atom_features.ndim != 2:
            raise ValueError(f"Expected flat atom features [A, H], got {tuple(atom_features.shape)}")
        if batch.numel() == 0:
            hidden_dim = atom_features.size(-1)
            return atom_features.new_zeros((0, 0, hidden_dim)), atom_features.new_zeros((0, 0), dtype=torch.bool)
        num_molecules = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=num_molecules)
        max_atoms = int(counts.max().item()) if counts.numel() else 0
        padded = atom_features.new_zeros((num_molecules, max_atoms, atom_features.size(-1)))
        mask = torch.zeros((num_molecules, max_atoms), dtype=torch.bool, device=atom_features.device)
        cursor = torch.zeros(num_molecules, dtype=torch.long, device=atom_features.device)
        for atom_idx in range(atom_features.size(0)):
            mol_idx = int(batch[atom_idx].item())
            slot = int(cursor[mol_idx].item())
            padded[mol_idx, slot] = atom_features[atom_idx]
            mask[mol_idx, slot] = True
            cursor[mol_idx] += 1
        return padded, mask


    def build_group_membership_tensor(group_assignments, batch, max_atoms: Optional[int] = None):
        num_groups = len(FUNCTIONAL_GROUP_SMARTS)
        if batch.numel() == 0:
            return torch.zeros((0, 0, num_groups), dtype=torch.float32)
        num_molecules = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=num_molecules)
        max_atoms = max_atoms or int(counts.max().item())
        membership = torch.zeros((num_molecules, max_atoms, num_groups), dtype=torch.float32, device=batch.device)
        for mol_idx in range(num_molecules):
            start = int((batch < mol_idx).sum().item())
            for group_idx, group_name in enumerate(FUNCTIONAL_GROUP_SMARTS.keys()):
                atom_ids = group_assignments.get((mol_idx, group_name), [])
                for atom_id in atom_ids:
                    local_idx = int(atom_id) - start
                    if 0 <= local_idx < max_atoms:
                        membership[mol_idx, local_idx, group_idx] = 1.0
        return membership


    class ChemistryHierarchicalPooling(nn.Module):
        """Pool atoms to chemistry groups, then groups to molecule context."""

        def __init__(self, atom_dim: int, hidden_dim: int):
            super().__init__()
            self.group_names = list(FUNCTIONAL_GROUP_SMARTS.keys())
            self.atom_attention = nn.Sequential(
                nn.Linear(atom_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.group_attention = nn.Sequential(
                nn.Linear(atom_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.fallback_attention = nn.Sequential(
                nn.Linear(atom_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.out_norm = nn.LayerNorm(atom_dim)

        def forward(self, atom_embeddings, group_membership, atom_mask):
            if atom_embeddings.ndim != 3:
                raise ValueError(f"Expected atom embeddings [B, N, H], got {tuple(atom_embeddings.shape)}")
            if group_membership is None:
                group_membership = atom_embeddings.new_zeros(
                    (atom_embeddings.size(0), atom_embeddings.size(1), len(self.group_names))
                )
            if atom_mask is None:
                atom_mask = torch.ones(atom_embeddings.shape[:2], dtype=torch.bool, device=atom_embeddings.device)
            membership_mask = group_membership > 0
            raw_atom_scores = self.atom_attention(atom_embeddings).squeeze(-1)
            group_embeddings = []
            group_weights = []
            group_present = []
            for group_idx in range(group_membership.size(-1)):
                group_mask = atom_mask & membership_mask[:, :, group_idx]
                weights = masked_softmax(raw_atom_scores, group_mask, dim=1)
                pooled = torch.sum(atom_embeddings * weights.unsqueeze(-1), dim=1)
                pooled = torch.where(
                    group_mask.any(dim=1, keepdim=True),
                    pooled,
                    torch.zeros_like(pooled),
                )
                group_embeddings.append(pooled)
                group_weights.append(weights)
                group_present.append(group_mask.any(dim=1).float())
            group_embeddings = torch.stack(group_embeddings, dim=1)
            group_weights = torch.stack(group_weights, dim=-1)
            group_present = torch.stack(group_present, dim=1) > 0

            fallback_mask = atom_mask & (~membership_mask.any(dim=-1))
            fallback_weights = masked_softmax(self.fallback_attention(atom_embeddings).squeeze(-1), fallback_mask, dim=1)
            fallback_embedding = torch.sum(atom_embeddings * fallback_weights.unsqueeze(-1), dim=1)
            atom_only_weights = masked_softmax(self.fallback_attention(atom_embeddings).squeeze(-1), atom_mask, dim=1)
            atom_only_embedding = torch.sum(atom_embeddings * atom_only_weights.unsqueeze(-1), dim=1)

            group_scores = self.group_attention(group_embeddings).squeeze(-1)
            group_level_weights = masked_softmax(group_scores, group_present, dim=1)
            molecule_from_groups = torch.sum(group_embeddings * group_level_weights.unsqueeze(-1), dim=1)

            any_groups = group_present.any(dim=1, keepdim=True)
            uncovered = fallback_mask.any(dim=1, keepdim=True)
            molecule_embedding = torch.where(any_groups, molecule_from_groups, atom_only_embedding)
            molecule_embedding = torch.where(
                any_groups & uncovered,
                0.8 * molecule_embedding + 0.2 * fallback_embedding,
                molecule_embedding,
            )
            molecule_embedding = self.out_norm(molecule_embedding)
            return {
                "group_embeddings": group_embeddings,
                "group_mask": group_present.float(),
                "molecule_embedding": molecule_embedding,
                "attention_weights": {
                    "atom_to_group": group_weights,
                    "group_to_molecule": group_level_weights,
                    "fallback_atom_attention": fallback_weights,
                },
            }


    class ExplicitGroupPooling(nn.Module):
        """Legacy wrapper around the new hierarchical pooling module."""

        def __init__(self, atom_dim: int, group_dim: int):
            super().__init__()
            self.pool = ChemistryHierarchicalPooling(atom_dim=atom_dim, hidden_dim=max(8, group_dim))
            self.group_projection = nn.Linear(atom_dim, group_dim)

        def forward(self, atom_features, group_assignments, batch):
            padded_atoms, atom_mask = pack_atom_features(atom_features, batch)
            membership = build_group_membership_tensor(group_assignments, batch, max_atoms=padded_atoms.size(1))
            pooled = self.pool(padded_atoms, membership, atom_mask)
            group_features = self.group_projection(pooled["group_embeddings"])
            return group_features, pooled["group_mask"]


    class MoleculePooling(nn.Module):
        def __init__(self, atom_dim: int, mol_dim: int):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(atom_dim, max(8, atom_dim // 2)),
                nn.Tanh(),
                nn.Linear(max(8, atom_dim // 2), 1),
            )
            self.transform = nn.Linear(atom_dim, mol_dim)

        def forward(self, atom_features, batch):
            if batch.numel() == 0:
                return torch.zeros(0, self.transform.out_features, device=atom_features.device)
            num_molecules = int(batch.max().item()) + 1
            scores = self.attention(atom_features)
            weights = segment_softmax(scores, batch, num_molecules)
            pooled = segment_sum(atom_features * weights, batch, num_molecules)
            return self.transform(pooled)
else:  # pragma: no cover
    def segment_sum(*args, **kwargs):
        require_torch()

    def segment_softmax(*args, **kwargs):
        require_torch()

    def masked_softmax(*args, **kwargs):
        require_torch()

    def pack_atom_features(*args, **kwargs):
        require_torch()

    def build_group_membership_tensor(*args, **kwargs):
        require_torch()

    class ChemistryHierarchicalPooling:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class ExplicitGroupPooling:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class MoleculePooling:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
