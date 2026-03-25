"""
nexus/reasoning/pgw_transport.py

Partial Fused Gromov-Wasserstein transport for analogical SoM routing.

This module replaces brittle exact scaffold matching with a soft transport plan
between query and retrieved molecules.  It uses:
  * intra-molecule distance matrices (topological by default, 3D when present)
  * cross-molecule atom feature costs
  * entropic partial fused Gromov-Wasserstein from POT

The output is a soft label over query atoms that can be consumed by the
existing trainer after reindexing into scan order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from rdkit import Chem
    _RDKIT_OK = True
except Exception:
    _RDKIT_OK = False

try:
    import ot
    _POT_OK = True
except Exception:
    ot = None
    _POT_OK = False


@dataclass
class PGWTransportResult:
    analogical_pred: torch.Tensor
    transport_succeeded: bool
    support_size: int
    transported_mass: float


class PGWTransporter:
    def __init__(
        self,
        *,
        device: str = "cpu",
        reg: float = 0.075,
        alpha: float = 0.7,
        mass_fraction: float = 0.8,
        num_itermax: int = 200,
        tol: float = 1.0e-7,
        min_transport_mass: float = 1.0e-3,
        support_threshold: float = 5.0e-2,
    ) -> None:
        if not _RDKIT_OK:
            raise ImportError("RDKit is required for PGWTransporter")
        if not _POT_OK:
            raise ImportError("POT is required for PGWTransporter")
        self.device = device
        self.reg = float(max(reg, 1.0e-6))
        self.alpha = float(min(max(alpha, 0.0), 1.0))
        self.mass_fraction = float(min(max(mass_fraction, 1.0e-3), 1.0))
        self.num_itermax = int(max(num_itermax, 10))
        self.tol = float(max(tol, 1.0e-10))
        self.min_transport_mass = float(max(min_transport_mass, 1.0e-8))
        self.support_threshold = float(max(support_threshold, 1.0e-8))

    @staticmethod
    def _distance_matrix(mol) -> torch.Tensor:
        if mol.GetNumConformers() > 0:
            mat = Chem.Get3DDistanceMatrix(mol)
        else:
            mat = Chem.GetDistanceMatrix(mol)
        tensor = torch.as_tensor(mat, dtype=torch.float64)
        scale = tensor.max().clamp_min(1.0)
        return tensor / scale

    @staticmethod
    def _atom_features(mol) -> torch.Tensor:
        rows = []
        for atom in mol.GetAtoms():
            rows.append(
                [
                    atom.GetAtomicNum() / 100.0,
                    atom.GetDegree() / 8.0,
                    atom.GetTotalNumHs(includeNeighbors=True) / 8.0,
                    atom.GetFormalCharge() / 4.0,
                    1.0 if atom.GetIsAromatic() else 0.0,
                    atom.GetMass() / 200.0,
                ]
            )
        feats = torch.tensor(rows, dtype=torch.float64)
        return F.normalize(feats, p=2, dim=-1)

    def _cross_feature_cost(self, query_mol, retrieved_mol) -> torch.Tensor:
        q_feat = self._atom_features(query_mol)
        r_feat = self._atom_features(retrieved_mol)
        # bounded [0, 2] Euclidean cost in feature space
        return torch.cdist(q_feat, r_feat, p=2).to(dtype=torch.float64)

    def transport_label(self, query_mol, retrieved_mol, retrieved_som_idx: int) -> PGWTransportResult:
        n_query = query_mol.GetNumAtoms()
        zero_pred = torch.zeros(n_query, dtype=torch.float32, device=self.device)

        if not (0 <= int(retrieved_som_idx) < retrieved_mol.GetNumAtoms()):
            return PGWTransportResult(zero_pred, False, 0, 0.0)

        c_query = self._distance_matrix(query_mol)
        c_retrieved = self._distance_matrix(retrieved_mol)
        m_cross = self._cross_feature_cost(query_mol, retrieved_mol)

        p = torch.full((c_query.size(0),), 1.0 / max(c_query.size(0), 1), dtype=torch.float64)
        q = torch.full((c_retrieved.size(0),), 1.0 / max(c_retrieved.size(0), 1), dtype=torch.float64)
        mass = self.mass_fraction * float(min(p.sum().item(), q.sum().item()))

        try:
            coupling = ot.gromov.entropic_partial_fused_gromov_wasserstein(
                m_cross,
                c_query,
                c_retrieved,
                p=p,
                q=q,
                reg=self.reg,
                m=mass,
                alpha=self.alpha,
                numItermax=self.num_itermax,
                tol=self.tol,
                symmetric=True,
                log=False,
                verbose=False,
            )
        except Exception:
            return PGWTransportResult(zero_pred, False, 0, 0.0)

        coupling_t = torch.as_tensor(coupling, dtype=torch.float32, device=self.device)
        source_column = coupling_t[:, int(retrieved_som_idx)]
        transported_mass = float(source_column.sum().item())
        if transported_mass < self.min_transport_mass or not torch.isfinite(source_column).all():
            return PGWTransportResult(zero_pred, False, 0, transported_mass)

        analogical_pred = source_column / source_column.sum().clamp_min(self.min_transport_mass)
        support_size = int((analogical_pred >= self.support_threshold).sum().item())
        return PGWTransportResult(
            analogical_pred=analogical_pred,
            transport_succeeded=True,
            support_size=support_size,
            transported_mass=transported_mass,
        )
