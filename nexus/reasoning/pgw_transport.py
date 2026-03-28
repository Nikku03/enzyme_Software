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
import torch.nn as nn
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
    transport_backend: str = "none"
    distill_loss: Optional[torch.Tensor] = None
    neuralgw_used_exact: bool = False
    neuralgw_confidence: float = 0.0
    neuralgw_distill_loss: float = 0.0


class NeuralGWApproximator(nn.Module):
    def __init__(self, hidden_dim: int = 64, temperature: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(max(hidden_dim, 8))
        self.temperature = float(max(temperature, 1.0e-3))
        self.compressor = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.bilinear_align = nn.Bilinear(self.hidden_dim, self.hidden_dim, 1)
        self.slack_q = nn.Parameter(torch.zeros(1))
        self.slack_ret = nn.Parameter(torch.zeros(1))

    def forward(self, q_multivectors: torch.Tensor, ret_multivectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_mv = q_multivectors.to(dtype=torch.float32)
        r_mv = ret_multivectors.to(dtype=torch.float32)
        n_atoms = q_mv.size(0)
        m_atoms = r_mv.size(0)

        h_q = self.compressor(q_mv)
        h_ret = self.compressor(r_mv)

        h_q_exp = h_q.unsqueeze(1).expand(n_atoms, m_atoms, -1)
        h_ret_exp = h_ret.unsqueeze(0).expand(n_atoms, m_atoms, -1)
        logits = self.bilinear_align(h_q_exp, h_ret_exp).squeeze(-1)

        logits_with_slack_col = torch.cat([logits, self.slack_q.expand(n_atoms, 1)], dim=1)
        pi_rows = F.softmax(logits_with_slack_col / self.temperature, dim=1)[:, :-1]

        logits_with_slack_row = torch.cat([logits, self.slack_ret.expand(1, m_atoms)], dim=0)
        pi_cols = F.softmax(logits_with_slack_row / self.temperature, dim=0)[:-1, :]

        pi_approx = 0.5 * (pi_rows + pi_cols)
        return pi_approx, logits


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
        dustbin_epsilon: float = 0.2,
        anchor_count: int = 5,
        linearize_above_atoms: int = 24,
        neuralgw_enabled: bool = True,
        neuralgw_hidden_dim: int = 64,
        neuralgw_temperature: float = 0.1,
        neuralgw_burn_in_epochs: int = 2,
        neuralgw_ambiguity_threshold: float = 0.7,
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
        self.dustbin_epsilon = float(min(max(dustbin_epsilon, 0.0), 0.95))
        self.anchor_count = int(max(anchor_count, 1))
        self.linearize_above_atoms = int(max(linearize_above_atoms, 4))
        self.neuralgw_enabled = bool(neuralgw_enabled)
        self.neuralgw_burn_in_epochs = int(max(neuralgw_burn_in_epochs, 0))
        self.neuralgw_ambiguity_threshold = float(min(max(neuralgw_ambiguity_threshold, 0.0), 1.0))
        self.current_epoch = 0
        self.neural_approximator = NeuralGWApproximator(
            hidden_dim=neuralgw_hidden_dim,
            temperature=neuralgw_temperature,
        )

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

    @staticmethod
    def _prepare_multivectors(multivectors: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if multivectors is None:
            return None
        mv = torch.as_tensor(multivectors, dtype=torch.float64)
        if mv.ndim == 1:
            mv = mv.unsqueeze(0)
        elif mv.ndim > 2:
            mv = mv.reshape(mv.size(0), -1)
        if mv.numel() == 0 or not bool(torch.isfinite(mv).all().item()):
            return None
        return mv

    @staticmethod
    def compute_z_kernel_matrix(multivectors: torch.Tensor) -> torch.Tensor:
        v_norm = F.normalize(multivectors, p=2, dim=-1)
        sim_matrix = torch.matmul(v_norm, v_norm.transpose(0, 1))
        z_dist = 1.0 - sim_matrix
        return z_dist / z_dist.max().clamp_min(1.0)

    def _anchor_indices(self, mol) -> torch.Tensor:
        ranked = sorted(
            (
                (-atom.GetAtomicNum(), -atom.GetDegree(), atom.GetIdx())
                for atom in mol.GetAtoms()
            )
        )
        chosen = [idx for _, _, idx in ranked[: min(self.anchor_count, mol.GetNumAtoms())]]
        return torch.tensor(chosen, dtype=torch.long)

    def _linearized_anchor_cost(
        self,
        query_mol,
        retrieved_mol,
        query_multivectors: torch.Tensor,
        retrieved_multivectors: torch.Tensor,
    ) -> torch.Tensor:
        q_z = self.compute_z_kernel_matrix(query_multivectors)
        r_z = self.compute_z_kernel_matrix(retrieved_multivectors)
        q_anchor = self._anchor_indices(query_mol)
        r_anchor = self._anchor_indices(retrieved_mol)
        k = min(int(q_anchor.numel()), int(r_anchor.numel()))
        q_sig = q_z.index_select(1, q_anchor[:k])  # [N, K]
        r_sig = r_z.index_select(1, r_anchor[:k])  # [M, K]
        return torch.cdist(q_sig, r_sig, p=2).to(dtype=torch.float64)

    def _partial_sinkhorn(self, cost: torch.Tensor, *, transported_mass: float) -> torch.Tensor:
        n, m = cost.shape
        mu = torch.full((n,), 1.0 / max(n, 1), dtype=torch.float64, device=cost.device)
        nu = torch.full((m,), 1.0 / max(m, 1), dtype=torch.float64, device=cost.device)
        K = torch.exp(-cost / self.reg).clamp_min(1.0e-12)
        pi = K
        for _ in range(self.num_itermax):
            row_sum = pi.sum(dim=1).clamp_min(1.0e-12)
            pi = pi * torch.minimum(mu / row_sum, torch.ones_like(row_sum)).unsqueeze(1)
            col_sum = pi.sum(dim=0).clamp_min(1.0e-12)
            pi = pi * torch.minimum(nu / col_sum, torch.ones_like(col_sum)).unsqueeze(0)
            current_mass = pi.sum().clamp_min(1.0e-12)
            pi = pi * (transported_mass / current_mass)
        return pi

    def _partial_z_gw(
        self,
        z_dist_query: torch.Tensor,
        z_dist_retrieved: torch.Tensor,
        *,
        cross_cost: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = z_dist_query.size(0)
        m = z_dist_retrieved.size(0)
        mu = torch.full((n,), 1.0 / max(n, 1), dtype=torch.float64, device=z_dist_query.device)
        nu = torch.full((m,), 1.0 / max(m, 1), dtype=torch.float64, device=z_dist_query.device)
        transported_mass = 1.0 - self.dustbin_epsilon
        pi = torch.full((n, m), transported_mass / max(n * m, 1), dtype=torch.float64, device=z_dist_query.device)
        additive_cost = cross_cost if cross_cost is not None else 0.0
        for _ in range(self.num_itermax):
            cost_gradient = -4.0 * torch.matmul(z_dist_query, torch.matmul(pi, z_dist_retrieved))
            score = -(cost_gradient + additive_cost) / self.reg
            score = score - score.max()
            K = torch.exp(score).clamp_min(1.0e-12)
            row_sum = K.sum(dim=1).clamp_min(1.0e-12)
            col_sum = K.sum(dim=0).clamp_min(1.0e-12)
            pi_next = K * (mu / row_sum).unsqueeze(1) * (nu / col_sum).unsqueeze(0)
            pi_next = pi_next * (transported_mass / pi_next.sum().clamp_min(1.0e-12))
            if torch.max(torch.abs(pi_next - pi)).item() < self.tol:
                pi = pi_next
                break
            pi = pi_next
        return pi

    @staticmethod
    def _normalize_coupling(pi: torch.Tensor, target_mass: float) -> torch.Tensor:
        pi = torch.clamp(pi, min=1.0e-9)
        return pi * (target_mass / pi.sum().clamp_min(1.0e-9))

    @staticmethod
    def _mean_entropy(probabilities: torch.Tensor, dim: int) -> torch.Tensor:
        safe = probabilities.clamp_min(1.0e-9)
        norm = safe / safe.sum(dim=dim, keepdim=True).clamp_min(1.0e-9)
        return -(norm * torch.log(norm)).sum(dim=dim).mean()

    def _calculate_neuralgw_confidence(self, pi_approx: torch.Tensor, *, target_mass: float) -> torch.Tensor:
        actual_mass = pi_approx.sum()
        mass_penalty = torch.abs(actual_mass - target_mass)
        row_entropy = self._mean_entropy(pi_approx, dim=1)
        col_entropy = self._mean_entropy(pi_approx, dim=0)
        return 1.0 / (1.0 + row_entropy + col_entropy + mass_penalty)

    def _exact_multivector_coupling(
        self,
        query_mol,
        retrieved_mol,
        q_mv: torch.Tensor,
        r_mv: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        transported_mass = 1.0 - self.dustbin_epsilon
        if max(q_mv.size(0), r_mv.size(0)) > self.linearize_above_atoms:
            anchor_cost = self._linearized_anchor_cost(query_mol, retrieved_mol, q_mv, r_mv)
            return self._partial_sinkhorn(anchor_cost, transported_mass=transported_mass), "zgw_linearized"
        c_query = self.compute_z_kernel_matrix(q_mv)
        c_retrieved = self.compute_z_kernel_matrix(r_mv)
        m_cross = self._cross_feature_cost(query_mol, retrieved_mol)
        return self._partial_z_gw(c_query, c_retrieved, cross_cost=m_cross), "zgw_exact"

    def _dynamic_multivector_router(
        self,
        query_mol,
        retrieved_mol,
        q_mv: torch.Tensor,
        r_mv: torch.Tensor,
    ) -> tuple[torch.Tensor, str, Optional[torch.Tensor], bool, float, float]:
        target_mass = 1.0 - self.dustbin_epsilon
        pi_approx, _ = self.neural_approximator(q_mv, r_mv)
        pi_approx = self._normalize_coupling(pi_approx, target_mass)
        confidence_t = self._calculate_neuralgw_confidence(pi_approx, target_mass=target_mass)
        confidence = float(confidence_t.detach().item())

        burn_in_complete = int(self.current_epoch) > self.neuralgw_burn_in_epochs
        use_exact = (not self.neuralgw_enabled) or (not burn_in_complete) or (confidence <= self.neuralgw_ambiguity_threshold)

        distill_loss: Optional[torch.Tensor] = None
        if use_exact:
            pi_final, backend = self._exact_multivector_coupling(query_mol, retrieved_mol, q_mv, r_mv)
            if burn_in_complete and self.neuralgw_enabled:
                teacher = self._normalize_coupling(pi_final.detach().to(dtype=torch.float32), target_mass)
                student = self._normalize_coupling(pi_approx, target_mass)
                distill_loss = F.kl_div(
                    torch.log(student.clamp_min(1.0e-9)),
                    teacher,
                    reduction="batchmean",
                )
                distill_value = float(distill_loss.detach().item())
            else:
                distill_value = 0.0
            return pi_final, backend, distill_loss, True, confidence, distill_value

        return pi_approx.to(dtype=torch.float64), "neuralgw_fast", None, False, confidence, 0.0

    def transport_label(
        self,
        query_mol,
        retrieved_mol,
        retrieved_som_idx: int,
        *,
        query_multivectors: Optional[torch.Tensor] = None,
        retrieved_multivectors: Optional[torch.Tensor] = None,
    ) -> PGWTransportResult:
        n_query = query_mol.GetNumAtoms()
        zero_pred = torch.zeros(n_query, dtype=torch.float32, device=self.device)

        if not (0 <= int(retrieved_som_idx) < retrieved_mol.GetNumAtoms()):
            return PGWTransportResult(zero_pred, False, 0, 0.0, "invalid_retrieved_som")

        q_mv = self._prepare_multivectors(query_multivectors)
        r_mv = self._prepare_multivectors(retrieved_multivectors)

        if (
            q_mv is not None
            and r_mv is not None
            and q_mv.size(0) == query_mol.GetNumAtoms()
            and r_mv.size(0) == retrieved_mol.GetNumAtoms()
        ):
            try:
                coupling_t, backend, distill_loss, used_exact, neuralgw_confidence, neuralgw_distill_loss = self._dynamic_multivector_router(
                    query_mol,
                    retrieved_mol,
                    q_mv,
                    r_mv,
                )
                coupling_t = coupling_t.to(dtype=torch.float32, device=self.device)
                source_column = coupling_t[:, int(retrieved_som_idx)]
                moved_mass = float(source_column.sum().item())
                if moved_mass >= self.min_transport_mass and bool(torch.isfinite(source_column).all().item()):
                    analogical_pred = source_column / source_column.sum().clamp_min(self.min_transport_mass)
                    support_size = int((analogical_pred >= self.support_threshold).sum().item())
                    return PGWTransportResult(
                        analogical_pred=analogical_pred,
                        transport_succeeded=True,
                        support_size=support_size,
                        transported_mass=moved_mass,
                        transport_backend=backend,
                        distill_loss=distill_loss,
                        neuralgw_used_exact=used_exact,
                        neuralgw_confidence=neuralgw_confidence,
                        neuralgw_distill_loss=neuralgw_distill_loss,
                    )
            except Exception:
                pass

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
            return PGWTransportResult(zero_pred, False, 0, 0.0, "pgw_error")

        coupling_t = torch.as_tensor(coupling, dtype=torch.float32, device=self.device)
        source_column = coupling_t[:, int(retrieved_som_idx)]
        transported_mass = float(source_column.sum().item())
        if transported_mass < self.min_transport_mass or not torch.isfinite(source_column).all():
            return PGWTransportResult(zero_pred, False, 0, transported_mass, "pgw_low_mass")

        analogical_pred = source_column / source_column.sum().clamp_min(self.min_transport_mass)
        support_size = int((analogical_pred >= self.support_threshold).sum().item())
        return PGWTransportResult(
            analogical_pred=analogical_pred,
            transport_succeeded=True,
            support_size=support_size,
            transported_mass=transported_mass,
            transport_backend="pgw_exact",
        )
