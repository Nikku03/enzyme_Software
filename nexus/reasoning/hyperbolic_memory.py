"""
nexus/reasoning/hyperbolic_memory.py

Hyperbolic analogue of the ECFP4 memory bank.

Fingerprints are first L2-normalised in Euclidean space, then projected
strictly inside the Poincare ball so nearest-neighbour lookup can use the
hyperbolic geodesic distance.  Transport remains discrete and scaffold-based
via RDKit MCS, matching the existing baseline memory-bank contract.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F

try:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    _RDKIT_OK = True
except Exception:
    _RDKIT_OK = False

from .baseline_memory import MemoryRetrievalResult, _extract_som_idx, _morgan_fp_tensor


class HyperbolicMemoryBank:
    """
    Poincare-ball memory bank with MCS-guided SoM transport.

    The external API intentionally matches BaselineMemoryBank so the trainer
    and Colab scripts can swap retrieval geometry without changing call sites.
    """

    def __init__(
        self,
        device: str = "cpu",
        *,
        curvature: float = 1.0,
        fp_radius: int = 2,
        fp_bits: int = 2048,
        identity_distance_threshold: float = 1.0e-4,
        poincare_radius: float = 0.95,
        mcs_timeout: int = 2,
    ) -> None:
        if not _RDKIT_OK:
            raise ImportError("RDKit is required for HyperbolicMemoryBank")
        self.device = device
        self.curvature = float(max(curvature, 1.0e-8))
        self.fp_radius = int(fp_radius)
        self.fp_bits = int(fp_bits)
        self.identity_distance_threshold = float(max(identity_distance_threshold, 0.0))
        self.poincare_radius = float(min(max(poincare_radius, 1.0e-3), 1.0 - 1.0e-5))
        self.mcs_timeout = int(max(mcs_timeout, 1))

        self.historical_mols: List = []
        self.historical_soms: List[int] = []
        self.memory_embeddings: torch.Tensor | None = None

    def _project_to_poincare(self, x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        # Normalise direction first so high-dimensional bit vectors do not all
        # collapse onto the same near-boundary norm after projection.
        unit = F.normalize(x, p=2, dim=-1)
        scale = x.norm(p=2, dim=-1, keepdim=True).tanh() * self.poincare_radius
        projected = unit * scale
        norm = projected.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        max_norm = 1.0 - eps
        return torch.where(norm > max_norm, projected / norm * max_norm, projected)

    def _poincare_distance(self, u: torch.Tensor, v: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        c = u.new_tensor(self.curvature)
        sqrt_c = torch.sqrt(c)
        sqdist = (u - v).pow(2).sum(dim=-1)
        squnorm = u.pow(2).sum(dim=-1)
        sqvnorm = v.pow(2).sum(dim=-1)

        denom_u = (1.0 - c * squnorm).clamp_min(eps)
        denom_v = (1.0 - c * sqvnorm).clamp_min(eps)
        arg = 1.0 + 2.0 * c * sqdist / (denom_u * denom_v)
        arg = arg.clamp_min(1.0 + eps)
        return torch.acosh(arg) / sqrt_c.clamp_min(eps)

    def populate_from_mols(self, mols: List) -> None:
        print(f"Populating Hyperbolic Memory Bank from {len(mols)} input molecules...")
        fps: List[torch.Tensor] = []
        skipped = 0

        for mol in mols:
            som_idx = _extract_som_idx(mol)
            if som_idx is None:
                skipped += 1
                continue
            fp = _morgan_fp_tensor(mol, radius=self.fp_radius, n_bits=self.fp_bits)
            fps.append(fp)
            self.historical_mols.append(mol)
            self.historical_soms.append(som_idx)

        if not fps:
            raise RuntimeError("No labelled molecules found — hyperbolic memory bank is empty.")

        raw = torch.stack(fps).to(self.device)
        self.memory_embeddings = self._project_to_poincare(raw)
        print(
            f"Hyperbolic Memory Bank Active: {len(self.historical_mols)} molecules "
            f"({skipped} skipped — no SoM label)."
        )

    def retrieve_and_transport(self, query_mol) -> MemoryRetrievalResult:
        if self.memory_embeddings is None:
            raise RuntimeError("Call populate_from_mols() before retrieve_and_transport().")

        n_query = query_mol.GetNumAtoms()
        analogical_pred = torch.zeros(n_query, dtype=torch.float32, device=self.device)

        q_fp = _morgan_fp_tensor(query_mol, radius=self.fp_radius, n_bits=self.fp_bits)
        q_embed = self._project_to_poincare(q_fp.unsqueeze(0).to(self.device))
        distances = self._poincare_distance(q_embed, self.memory_embeddings).squeeze(0)

        k = min(2, len(self.historical_mols))
        top_distances, top_indices = torch.topk(distances, k, largest=False)

        best_idx = int(top_indices[0].item())
        best_distance = float(top_distances[0].item())
        if best_distance <= self.identity_distance_threshold and k > 1:
            best_idx = int(top_indices[1].item())
            best_distance = float(top_distances[1].item())
        confidence = math.exp(-best_distance)

        retrieved_mol = self.historical_mols[best_idx]
        retrieved_som = self.historical_soms[best_idx]

        mcs_size = 0
        transport_ok = False
        res = rdFMCS.FindMCS(
            [query_mol, retrieved_mol],
            timeout=self.mcs_timeout,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )

        if not res.canceled and res.numAtoms > 0:
            mcs_size = res.numAtoms
            mcs_mol = Chem.MolFromSmarts(res.smartsString)
            match_query = query_mol.GetSubstructMatch(mcs_mol)
            match_retrieved = retrieved_mol.GetSubstructMatch(mcs_mol)

            if match_query and match_retrieved:
                try:
                    mcs_pos = match_retrieved.index(retrieved_som)
                    mapped_query_idx = match_query[mcs_pos]
                    analogical_pred[mapped_query_idx] = 1.0
                    transport_ok = True
                except ValueError:
                    pass

        return MemoryRetrievalResult(
            analogical_pred=analogical_pred,
            confidence=confidence,
            retrieved_mol=retrieved_mol,
            retrieved_som_idx=retrieved_som,
            transport_succeeded=transport_ok,
            mcs_size=mcs_size,
        )

    def batch_stats(self, mols: List, true_soms: List[int]) -> dict:
        n = len(mols)
        successes = top1_hits = 0
        total_conf = total_mcs = 0.0

        for mol, true_som in zip(mols, true_soms):
            result = self.retrieve_and_transport(mol)
            total_conf += result.confidence
            total_mcs += result.mcs_size
            if result.transport_succeeded:
                successes += 1
                pred_atom = int(result.analogical_pred.argmax().item())
                if pred_atom == true_som:
                    top1_hits += 1

        return {
            "n": n,
            "transport_success_rate": successes / max(n, 1),
            "top1_accuracy": top1_hits / max(n, 1),
            "mean_confidence": total_conf / max(n, 1),
            "mean_mcs_size": total_mcs / max(n, 1),
        }
