from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, nn, require_torch, torch


def _safe_tensor(data, *, dtype=torch.float32):
    return torch.as_tensor(data, dtype=dtype)


def _stable_molecule_key(smiles: str, *, primary_cyp: str = "") -> int:
    canonical = " ".join(str(smiles or "").split())
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(canonical)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    cyp_text = str(primary_cyp or "").strip()
    payload = f"{canonical}\0{cyp_text}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)


def _cyp_class_match(query_cyp: torch.Tensor, memory_cyp: torch.Tensor) -> torch.Tensor:
    """
    CYP classes are categorical identities, not ordered numeric values.

    Return 1.0 only for exact known-class matches and 0.0 otherwise. Unknown
    classes (encoded as 0) abstain instead of being treated as a match.
    """
    qc = query_cyp.float().view(-1, 1)
    mc = memory_cyp.float().view(1, -1)
    known = (qc > 0.0) & (mc > 0.0)
    return ((qc == mc) & known).to(dtype=qc.dtype)


if TORCH_AVAILABLE:
    class AuditedEpisodeLogbook(nn.Module):
        """Audited precedent index built from immutable episode logs."""

        vector_dim: int = 28  # 16D multivector + charge + fukui + 10 field features
        brief_dim: int = 6

        def __init__(self, *, max_cases: int = 32768, topk: int = 16, cyp_weight: float = 0.15, temperature: float = 0.25):
            super().__init__()
            self.max_cases = max(128, int(max_cases))
            self.topk = max(1, int(topk))
            self.cyp_weight = float(max(cyp_weight, 0.0))
            self.temperature = float(max(temperature, 1.0e-3))
            self.register_buffer("case_vectors", torch.zeros(self.max_cases, self.vector_dim))
            self.register_buffer("case_labels", torch.zeros(self.max_cases, 1))
            self.register_buffer("case_cyp", torch.zeros(self.max_cases, 1))
            self.register_buffer("case_molecule_key", torch.zeros(self.max_cases, dtype=torch.long))
            self.register_buffer("valid", torch.zeros(self.max_cases, dtype=torch.bool))
            self.register_buffer("ptr", torch.zeros((), dtype=torch.long))

        def size(self) -> int:
            return int(self.valid.sum().item())

        @torch.no_grad()
        def clear(self) -> None:
            self.case_vectors.zero_()
            self.case_labels.zero_()
            self.case_cyp.zero_()
            self.case_molecule_key.zero_()
            self.valid.zero_()
            self.ptr.zero_()

        @staticmethod
        def _case_vector_from_episode(episode: dict, atom_idx: int) -> Optional[torch.Tensor]:
            wave = episode.get("wave") or {}
            multivectors = wave.get("atom_multivectors") or []
            charges = wave.get("predicted_charges") or []
            fukui = wave.get("predicted_fukui") or []
            field = wave.get("atom_field_features") or []
            if atom_idx >= len(multivectors) or atom_idx >= len(charges) or atom_idx >= len(fukui) or atom_idx >= len(field):
                return None
            mv = list(multivectors[atom_idx] or [])
            ff = list(field[atom_idx] or [])
            if len(mv) != 16 or len(ff) != 10:
                return None
            return _safe_tensor(mv + [float(charges[atom_idx]), float(fukui[atom_idx])] + ff)

        @torch.no_grad()
        def load_jsonl(
            self,
            path: str | Path,
            *,
            cyp_names: Optional[Iterable[str]] = None,
            allowed_splits: Optional[Iterable[str]] = ("train",),
        ) -> Dict[str, float]:
            self.clear()
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Precedent logbook not found: {path}")
            cyp_names = [str(v) for v in (cyp_names or [])]
            allowed = {str(v).strip() for v in (allowed_splits or ())}
            cyp_to_value = {name: float(idx + 1) for idx, name in enumerate(cyp_names)}
            case_vectors = []
            case_labels = []
            case_cyp = []
            case_molecule_key = []
            episodes = 0
            positive_cases = 0
            negative_cases = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("record_type") != "episode":
                        continue
                    if allowed and str(record.get("split", "")).strip() not in allowed:
                        continue
                    episodes += 1
                    input_meta = record.get("input") or {}
                    true_site_atoms = {int(v) for v in list((record.get("outcome") or {}).get("true_site_atoms", input_meta.get("site_atoms", [])))}
                    num_atoms = int(input_meta.get("num_atoms", 0))
                    if num_atoms <= 0:
                        continue
                    molecule_key = int(
                        _stable_molecule_key(
                            input_meta.get("canonical_smiles", input_meta.get("smiles", "")),
                            primary_cyp=input_meta.get("primary_cyp", ""),
                        )
                    )
                    cyp_value = float(cyp_to_value.get(str(input_meta.get("primary_cyp", "")).strip(), 0.0))
                    for atom_idx in range(num_atoms):
                        vec = self._case_vector_from_episode(record, atom_idx)
                        if vec is None:
                            continue
                        case_vectors.append(vec)
                        label = 1.0 if atom_idx in true_site_atoms else 0.0
                        case_labels.append(label)
                        case_cyp.append(cyp_value)
                        case_molecule_key.append(molecule_key)
                        positive_cases += int(label > 0.5)
                        negative_cases += int(label <= 0.5)
                        if len(case_vectors) >= self.max_cases:
                            break
                    if len(case_vectors) >= self.max_cases:
                        break
            if not case_vectors:
                return {"cases": 0.0, "episodes": float(episodes), "positive_cases": 0.0, "negative_cases": 0.0}
            vectors = torch.stack(case_vectors, dim=0)
            labels = _safe_tensor(case_labels).view(-1, 1)
            cyp = _safe_tensor(case_cyp).view(-1, 1)
            molecule_keys = torch.as_tensor(case_molecule_key, dtype=torch.long)
            count = int(vectors.size(0))
            self.case_vectors[:count] = vectors
            self.case_labels[:count] = labels
            self.case_cyp[:count] = cyp
            self.case_molecule_key[:count] = molecule_keys
            self.valid[:count] = True
            self.ptr.fill_(count % self.max_cases)
            return {
                "cases": float(count),
                "episodes": float(episodes),
                "positive_cases": float(positive_cases),
                "negative_cases": float(negative_cases),
            }

        @torch.no_grad()
        def update(
            self,
            *,
            case_vectors: torch.Tensor,
            case_labels: torch.Tensor,
            case_cyp_values: torch.Tensor,
            molecule_keys: torch.Tensor,
            supervision_mask: Optional[torch.Tensor] = None,
            hard_negative_scores: Optional[torch.Tensor] = None,
            hard_negative_ratio: float = 1.5,
            min_hard_negatives: int = 16,
            max_hard_negatives: int = 128,
        ) -> int:
            if case_vectors.numel() == 0:
                return 0
            mask = torch.ones(case_vectors.size(0), dtype=torch.bool, device=case_vectors.device)
            if supervision_mask is not None:
                mask = supervision_mask.view(-1) > 0.5
            if not bool(mask.any()):
                return 0
            labels_flat = case_labels.view(-1)
            positive_mask = mask & (labels_flat > 0.5)
            selected_mask = positive_mask.clone()
            negative_mask = mask & ~positive_mask
            if bool(negative_mask.any()):
                negative_idx = torch.nonzero(negative_mask, as_tuple=False).view(-1)
                if hard_negative_scores is not None:
                    hardness = hard_negative_scores.view(-1)[negative_idx].detach().float()
                else:
                    hardness = torch.zeros_like(negative_idx, dtype=case_vectors.dtype)
                positive_count = int(positive_mask.sum().item())
                desired_negatives = max(int(min_hard_negatives), int(round(float(hard_negative_ratio) * max(1, positive_count))))
                desired_negatives = min(desired_negatives, int(max_hard_negatives), int(negative_idx.numel()))
                if desired_negatives > 0:
                    hard_order = torch.topk(hardness, k=desired_negatives, dim=0).indices
                    selected_mask[negative_idx[hard_order]] = True
            if not bool(selected_mask.any()):
                return 0
            vectors = case_vectors[selected_mask].detach().float()
            labels = case_labels[selected_mask].detach().view(-1, 1).float()
            cyp = case_cyp_values[selected_mask].detach().view(-1, 1).float()
            mol_keys = molecule_keys[selected_mask].detach().view(-1).long()
            count = int(vectors.size(0))
            ptr = int(self.ptr.item())
            if count >= self.max_cases:
                vectors = vectors[-self.max_cases :]
                labels = labels[-self.max_cases :]
                cyp = cyp[-self.max_cases :]
                mol_keys = mol_keys[-self.max_cases :]
                count = self.max_cases
                ptr = 0
            end = ptr + count
            if end <= self.max_cases:
                sl = slice(ptr, end)
                self.case_vectors[sl] = vectors
                self.case_labels[sl] = labels
                self.case_cyp[sl] = cyp
                self.case_molecule_key[sl] = mol_keys
                self.valid[sl] = True
            else:
                first = self.max_cases - ptr
                second = count - first
                self.case_vectors[ptr:] = vectors[:first]
                self.case_labels[ptr:] = labels[:first]
                self.case_cyp[ptr:] = cyp[:first]
                self.case_molecule_key[ptr:] = mol_keys[:first]
                self.valid[ptr:] = True
                self.case_vectors[:second] = vectors[first:]
                self.case_labels[:second] = labels[first:]
                self.case_cyp[:second] = cyp[first:]
                self.case_molecule_key[:second] = mol_keys[first:]
                self.valid[:second] = True
            self.ptr.fill_(end % self.max_cases)
            return count

        def lookup(
            self,
            query_vectors: torch.Tensor,
            query_cyp_values: Optional[torch.Tensor] = None,
            query_molecule_keys: Optional[torch.Tensor] = None,
        ) -> Optional[Dict[str, torch.Tensor]]:
            size = self.size()
            if size <= 0 or query_vectors.numel() == 0:
                return None
            mem_vectors = self.case_vectors[:size]
            mem_labels = self.case_labels[:size]
            mem_cyp = self.case_cyp[:size]
            mem_molecule_key = self.case_molecule_key[:size]
            qv = F.normalize(query_vectors.float(), p=2, dim=-1)
            mv = F.normalize(mem_vectors.float(), p=2, dim=-1)
            scores = torch.matmul(qv, mv.transpose(0, 1))
            if query_cyp_values is not None:
                cyp_match = _cyp_class_match(query_cyp_values, mem_cyp)
                scores = scores + self.cyp_weight * cyp_match
            allowed = torch.ones_like(scores, dtype=torch.bool)
            if query_molecule_keys is not None:
                qm = query_molecule_keys.to(device=scores.device, dtype=torch.long).view(-1, 1)
                allowed = allowed & (qm != mem_molecule_key.view(1, -1))
                scores = scores.masked_fill(~allowed, -1.0e9)
            k = min(self.topk, size)
            top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
            valid_rows = top_scores[:, 0] > -1.0e8
            valid_top = top_scores > -1.0e8
            scaled_scores = (top_scores / self.temperature).masked_fill(~valid_top, -1.0e9)
            weights = F.softmax(scaled_scores, dim=-1) * valid_top.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            top_labels = mem_labels[top_idx]
            positive_support = (weights.unsqueeze(-1) * top_labels).sum(dim=1)
            negative_support = (weights.unsqueeze(-1) * (1.0 - top_labels)).sum(dim=1)
            distances = 1.0 - top_scores
            mean_distance = (weights * distances).sum(dim=-1, keepdim=True)
            best_distance = distances[:, :1]
            if int(top_scores.size(-1)) > 1:
                support_margin = (top_scores[:, :1] - top_scores[:, 1:2]).clamp_min(0.0)
            else:
                support_margin = top_scores[:, :1]
            cyp_alignment = (
                (
                    weights.unsqueeze(-1)
                    * (
                        (
                            (query_cyp_values.float().view(-1, 1, 1) == mem_cyp[top_idx])
                            & (query_cyp_values.float().view(-1, 1, 1) > 0.0)
                            & (mem_cyp[top_idx] > 0.0)
                        ).to(dtype=weights.dtype)
                    )
                ).sum(dim=1)
                if query_cyp_values is not None
                else positive_support.new_zeros((positive_support.size(0), 1))
            )
            brief = torch.cat(
                [
                    positive_support,
                    negative_support,
                    mean_distance,
                    best_distance,
                    support_margin,
                    cyp_alignment,
                ],
                dim=-1,
            )
            valid_rows_f = valid_rows.unsqueeze(-1).to(dtype=brief.dtype)
            brief = brief * valid_rows_f
            positive_support = positive_support * valid_rows_f
            negative_support = negative_support * valid_rows_f
            mean_distance = mean_distance * valid_rows_f
            best_distance = best_distance * valid_rows_f
            support_margin = support_margin * valid_rows_f
            cyp_alignment = cyp_alignment * valid_rows_f
            return {
                "brief": brief,
                "positive_support": positive_support,
                "negative_support": negative_support,
                "mean_distance": mean_distance,
                "best_distance": best_distance,
                "support_margin": support_margin,
                "cyp_alignment": cyp_alignment,
                "top_scores": top_scores,
                "top_weights": weights,
                "top_labels": top_labels,
                "valid_rows": valid_rows.unsqueeze(-1),
            }
else:  # pragma: no cover
    class AuditedEpisodeLogbook:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
