from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .manifold_projector import ManifoldProjectionOutput, MultivectorManifoldProjector


# ---------------------------------------------------------------------------
# GraN-DAG: Gradient-based Neural DAG edge predictor
# ---------------------------------------------------------------------------

# Invariant scalar descriptors per (Mi, Mj) pair:
#   scalar dot, vector dot, bivector dot, trivector dot, L2 distance, scalar delta
_PAIR_FEAT_DIM: int = 6


class GraNDAGEdgePredictor(nn.Module):
    """MLP edge predictor that maps a pair of Cl(3,0) multivectors to a
    weight distribution over reaction operators.

    Input : multivectors [..., n_nodes, 8]  (public layout)
    Output: operator_weights [..., n_nodes, n_nodes, n_reaction_ops]
            — non-negative (Softplus), diagonal zeroed by caller.

    Feature engineering keeps strict O(3)-invariance:
      * grade-wise dot products  (scalar, vector, bivector, trivector)
      * L2 distance in 8-D Clifford space
      * scalar-component delta  (mass / charge density proxy)

    Backbone is a 3-layer residual bottleneck:
      Layer 1 – expansion  : Linear(_PAIR_FEAT_DIM → 512) + ReLU
      Layer 2 – processing : Linear(512 → 512) + identity skip + ReLU
      Layer 3 – projection : Linear(512 → n_reaction_ops) + Softplus
    """

    def __init__(self, n_reaction_ops: int = 8) -> None:
        super().__init__()
        self.n_reaction_ops = int(n_reaction_ops)

        self.expand = nn.Sequential(
            nn.Linear(_PAIR_FEAT_DIM, 512),
            nn.ReLU(),
        )
        self.process = nn.Linear(512, 512)   # residual: ReLU(process(h) + h)
        self.project = nn.Sequential(
            nn.Linear(512, self.n_reaction_ops),
            nn.Softplus(),                   # non-negative — required by NOTEARS h(W)
        )
        # Sparse initialisation: bias the final layer strongly negative so
        # Softplus(-5) ≈ 0.007 per edge weight at init.  Default bias=0 gives
        # Softplus(0) ≈ 0.693, producing dag_adjacency_mean ≈ 0.75 which
        # instantly saturates the NOTEARS penalty at its 10000 cap and zeroes
        # all DAG gradients for the entire first epoch.
        nn.init.constant_(self.project[0].bias, -5.0)

    @staticmethod
    def _pair_invariants(mi: torch.Tensor, mj: torch.Tensor) -> torch.Tensor:
        """Build 6 invariant scalars from two broadcast-expanded multivectors.

        mi : [..., n, 1, 8]
        mj : [..., 1, n, 8]
        returns [..., n, n, 6]
        """
        scalar_dot   = mi[..., 0:1] * mj[..., 0:1]
        vector_dot   = (mi[..., 1:4] * mj[..., 1:4]).sum(dim=-1, keepdim=True)
        bivector_dot = (mi[..., 4:7] * mj[..., 4:7]).sum(dim=-1, keepdim=True)
        trivector_dot = mi[..., 7:8] * mj[..., 7:8]
        distance     = (mi - mj).norm(dim=-1, keepdim=True)
        scalar_delta = mi[..., 0:1] - mj[..., 0:1]
        return torch.cat(
            [scalar_dot, vector_dot, bivector_dot, trivector_dot, distance, scalar_delta],
            dim=-1,
        )

    def forward(self, multivectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multivectors: [..., n_nodes, 8]

        Returns:
            [..., n_nodes, n_nodes, n_reaction_ops]  — non-negative edge weights.
        """
        if multivectors.size(-1) != 8:
            raise ValueError("multivectors must have trailing dimension 8")
        mi = multivectors.unsqueeze(-2)   # [..., n, 1, 8]
        mj = multivectors.unsqueeze(-3)   # [..., 1, n, 8]
        pair_feat = self._pair_invariants(mi, mj)   # [..., n, n, 6]
        h = self.expand(pair_feat)                  # [..., n, n, 512]
        h = F.relu(self.process(h) + h)             # [..., n, n, 512]  residual
        return self.project(h)                       # [..., n, n, n_ops]


# ---------------------------------------------------------------------------

@dataclass
class MetabolicDAGOutput:
    adjacency: torch.Tensor        # [..., n, n]  physics-weighted DAG
    raw_adjacency: torch.Tensor    # [..., n, n]  pre-prior structural DAG
    logits: torch.Tensor           # [..., n, n]  exposed for API compatibility
    node_embeddings: torch.Tensor
    reconstruction: torch.Tensor
    physics_prior: torch.Tensor
    accessibility_prior: torch.Tensor
    acyclicity: torch.Tensor
    sparsity: torch.Tensor
    physics_loss: torch.Tensor
    kinetic_penalty: torch.Tensor
    thermodynamic_penalty: torch.Tensor
    affinity_penalty: torch.Tensor
    accessibility_penalty: torch.Tensor
    flux_penalty: torch.Tensor
    reconstruction_loss: torch.Tensor
    causal_loss: torch.Tensor
    spawned_mask: torch.Tensor
    operator_weights: torch.Tensor  # [..., n, n, n_reaction_ops]  per-op edge weights
    # Phase 1.5 – Admissible Manifold
    manifold_recon_loss: torch.Tensor   # scalar  L_recon (distance from valid manifold)
    manifold_density_penalty: torch.Tensor  # scalar  electronic density violation
    surface_energy: torch.Tensor        # [..., n_nodes]  per-node stability metric
    stability_mask: torch.Tensor        # [..., n_nodes]  bool: admissible spawn targets


class MetabolicDAGLearner(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 8,
        invariant_dim: int = 8,
        hidden_dim: int = 64,
        pair_hidden_dim: int = 128,
        n_reaction_ops: int = 8,
        beta: float = 100.0,
        alpha: float = 1.0e-2,
        gamma: float = 1.0,
        delta: float = 1.0,
        reconstruction_weight: float = 1.0,
        edge_threshold: float = 0.5,
        activation_threshold: float = 25.0,
        thermodynamic_margin: float = 0.0,
        prior_temperature: float = 5.0,
        affinity_scale: float = 2.0,
        manifold_weight: float = 1.0,
        manifold_latent_dim: int = 4,
        manifold_hidden_dim: int = 32,
        surface_energy_threshold: float = 2.0,
        radical_ratio_threshold: float = 3.0,
    ) -> None:
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.invariant_dim = int(invariant_dim)
        self.hidden_dim = int(hidden_dim)
        self.pair_hidden_dim = int(pair_hidden_dim)
        self.n_reaction_ops = int(n_reaction_ops)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.reconstruction_weight = float(reconstruction_weight)
        self.edge_threshold = float(edge_threshold)
        self.activation_threshold = float(activation_threshold)
        self.thermodynamic_margin = float(thermodynamic_margin)
        self.prior_temperature = float(prior_temperature)
        self.affinity_scale = float(affinity_scale)
        self.manifold_weight = float(manifold_weight)

        self.invariant_projector = nn.Sequential(
            nn.Linear(self.invariant_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        # GraN-DAG edge predictor replaces the old pair-embedding MLP.
        # It operates directly on raw multivectors and predicts per-operator
        # edge weights; the structural adjacency is their max over operators.
        self.gran_dag = GraNDAGEdgePredictor(n_reaction_ops=self.n_reaction_ops)
        # Phase 1.5 – Admissible Manifold Projector.
        # Validates that reconstructed child multivectors lie on the learned
        # chemical manifold; provides L_recon, density penalty, and the
        # stability_mask used to gate spawn_generation.
        self.manifold_projector = MultivectorManifoldProjector(
            latent_dim=manifold_latent_dim,
            hidden_dim=manifold_hidden_dim,
            surface_energy_threshold=surface_energy_threshold,
            radical_ratio_threshold=radical_ratio_threshold,
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.node_feature_dim),
        )

    def invariant_node_features(self, multivectors: torch.Tensor) -> torch.Tensor:
        scalar = multivectors[..., 0:1]
        vector_norm = multivectors[..., 1:4].norm(dim=-1, keepdim=True)
        bivector_norm = multivectors[..., 4:7].norm(dim=-1, keepdim=True)
        trivector = multivectors[..., 7:8]
        total_norm = multivectors.norm(dim=-1, keepdim=True)
        vector_bivector_overlap = (multivectors[..., 1:4] * multivectors[..., 4:7]).sum(dim=-1, keepdim=True)
        scalar_trivector_mix = scalar * trivector
        parity_balance = vector_norm - bivector_norm
        return torch.cat(
            [
                scalar,
                vector_norm,
                bivector_norm,
                trivector,
                total_norm,
                vector_bivector_overlap,
                scalar_trivector_mix,
                parity_balance,
            ],
            dim=-1,
        )

    def encode_nodes(self, multivectors: torch.Tensor) -> torch.Tensor:
        invariants = self.invariant_node_features(multivectors)
        return self.invariant_projector(invariants)

    def predict_adjacency(
        self,
        multivectors: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        physics_prior: Optional[torch.Tensor] = None,
        accessibility_prior: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Node embeddings are still produced for the reconstruction head.
        node_embeddings = self.encode_nodes(multivectors)

        # GraN-DAG: per-operator edge weights [..., n, n, n_ops], Softplus → ≥ 0
        operator_weights = self.gran_dag(multivectors)

        # Aggregate over operators: the structural DAG weight is the maximum
        # causal flux across all biotransformation types for each (i→j) pair.
        # Shape: [..., n, n]
        raw_adjacency = operator_weights.max(dim=-1).values
        adjacency = raw_adjacency

        # Zero the diagonal (no self-loops).
        d = adjacency.size(-1)
        diag_mask = torch.eye(d, dtype=torch.bool, device=adjacency.device)
        if physics_prior is not None:
            adjacency = adjacency * physics_prior
        if accessibility_prior is not None:
            adjacency = adjacency * accessibility_prior
        adjacency = adjacency.masked_fill(diag_mask, 0.0)
        raw_adjacency = raw_adjacency.masked_fill(diag_mask, 0.0)
        operator_weights = operator_weights.masked_fill(diag_mask.unsqueeze(-1), 0.0)

        if mask is not None:
            adjacency = adjacency * mask
            raw_adjacency = raw_adjacency * mask
            operator_weights = operator_weights * mask.unsqueeze(-1)

        # "logits" exposed for API compatibility; equals the structural adjacency.
        return adjacency, raw_adjacency, adjacency, node_embeddings, operator_weights

    def compute_physics_prior(
        self,
        *,
        delta_g_activations: Optional[torch.Tensor] = None,
        delta_g_rxn: Optional[torch.Tensor] = None,
        binding_affinity: Optional[torch.Tensor] = None,
        batch_size: int,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        prior = torch.ones((batch_size, n_nodes, n_nodes), dtype=dtype, device=device)
        temp = torch.as_tensor(self.prior_temperature, dtype=dtype, device=device).clamp_min(1.0e-6)
        if delta_g_activations is not None:
            prior = prior * torch.sigmoid(
                (
                    torch.as_tensor(self.activation_threshold, dtype=dtype, device=device)
                    - delta_g_activations.to(dtype=dtype, device=device)
                )
                / temp
            )
        if delta_g_rxn is not None:
            prior = prior * torch.sigmoid(
                -(delta_g_rxn.to(dtype=dtype, device=device) - self.thermodynamic_margin) / temp
            )
        if binding_affinity is not None:
            prior = prior * torch.sigmoid(
                -binding_affinity.to(dtype=dtype, device=device)
                / torch.as_tensor(self.affinity_scale, dtype=dtype, device=device).clamp_min(1.0e-6)
            )
        eye = torch.eye(n_nodes, dtype=dtype, device=device)
        return prior * (~eye.bool()).to(dtype)

    def compute_accessibility_prior(
        self,
        accessibility_mask: Optional[torch.Tensor],
        path_accessibility: Optional[torch.Tensor] = None,
        *,
        batch_size: int,
        n_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if accessibility_mask is None:
            prior = torch.ones((batch_size, n_nodes, n_nodes), dtype=dtype, device=device)
        else:
            acc = accessibility_mask.to(dtype=dtype, device=device)
            if acc.ndim == 1:
                acc = acc.unsqueeze(0).expand(batch_size, -1)
            if acc.ndim == 2:
                prior = torch.sqrt((acc.unsqueeze(-1) * acc.unsqueeze(-2)).clamp_min(0.0))
            elif acc.ndim == 3:
                prior = acc
            else:
                raise ValueError("accessibility_mask must have shape [N], [B,N], or [B,N,N]")
        if path_accessibility is not None:
            path_acc = path_accessibility.to(dtype=dtype, device=device)
            if path_acc.ndim == 1:
                path_acc = path_acc.unsqueeze(0).expand(batch_size, -1)
            if path_acc.ndim == 2:
                path_prior = torch.sqrt((path_acc.unsqueeze(-1) * path_acc.unsqueeze(-2)).clamp_min(0.0))
            elif path_acc.ndim == 3:
                path_prior = path_acc
            else:
                raise ValueError("path_accessibility must have shape [N], [B,N], or [B,N,N]")
            prior = prior * path_prior
        eye = torch.eye(n_nodes, dtype=dtype, device=device)
        return prior * (~eye.bool()).to(dtype)

    def compute_accessibility_penalty(
        self,
        adjacency: torch.Tensor,
        accessibility_prior: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if accessibility_prior is None:
            return torch.zeros((), dtype=adjacency.dtype, device=adjacency.device)
        return torch.mean(adjacency * (1.0 - accessibility_prior).clamp_min(0.0))

    def compute_notears_constraint(self, W_struct: torch.Tensor, n_nodes: Optional[int] = None) -> torch.Tensor:
        d = int(n_nodes if n_nodes is not None else W_struct.size(-1))
        W_sq = torch.nan_to_num(W_struct * W_struct, nan=0.0, posinf=25.0, neginf=0.0)
        # Stable NOTEARS surrogate:
        #   h(W) = tr(exp(W∘W / d)) - d
        #
        # In practice, exact matrix_exp backward is the main source of NaNs on
        # Colab/H100 runs once adjacency entries momentarily spike.  We replace
        # it with a truncated trace-exp series in float64:
        #   tr(I + A + A²/2 + A³/6 + A⁴/24 + A⁵/120) - d
        #
        # This preserves h(W)=0 at A=0, remains monotone for the positive A
        # used here, and gives a smooth acyclicity penalty without invoking the
        # unstable LinalgMatrixExpBackward0 path.
        # Adaptive clamp: max entry of A is 2/d, so the spectral radius of the
        # all-positive matrix A is ≤ d * (2/d) = 2.  This bounds the Taylor
        # series: tr(exp(A)) ≤ d·e² ≈ 7.4d and h(W) ≤ 6.4d.
        # With d=20: h(W) ≤ 128, contribution to total_loss ≤ 128/400 = 0.32.
        # The fixed clamp of 10.0 allowed entries near 10 → tr(A⁵)/120 ~ O(d⁴·10⁵)
        # causing the 511 spike at batch 20.
        A = (W_sq / float(max(d, 1))).clamp(0.0, 2.0 / float(max(d, 1))).to(dtype=torch.float64)
        eye = torch.eye(d, dtype=A.dtype, device=A.device).expand_as(A)
        A2 = A @ A
        A3 = A2 @ A
        A4 = A3 @ A
        A5 = A4 @ A
        expm_trace_approx = torch.diagonal(
            eye + A + 0.5 * A2 + (1.0 / 6.0) * A3 + (1.0 / 24.0) * A4 + (1.0 / 120.0) * A5,
            dim1=-2,
            dim2=-1,
        ).sum(dim=-1)
        trace = expm_trace_approx.to(dtype=W_struct.dtype)
        h_W = trace - torch.as_tensor(float(d), dtype=W_struct.dtype, device=W_struct.device)
        return h_W.mean()

    def acyclicity_constraint(self, adjacency: torch.Tensor) -> torch.Tensor:
        return self.compute_notears_constraint(adjacency, adjacency.size(-1))

    def reconstruction_loss(
        self,
        multivectors: torch.Tensor,
        node_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        message = torch.matmul(adjacency.transpose(-1, -2), node_embeddings)
        reconstruction = self.reconstruction_head(node_embeddings + message)
        loss = F.mse_loss(reconstruction, multivectors, reduction="mean")
        return reconstruction, loss

    def thermodynamic_physics_loss(
        self,
        adjacency: torch.Tensor,
        *,
        delta_g_activations: Optional[torch.Tensor] = None,
        delta_g_rxn: Optional[torch.Tensor] = None,
        binding_affinity: Optional[torch.Tensor] = None,
        physics_loss: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kinetic_penalty = adjacency.new_zeros(())
        thermo_penalty = adjacency.new_zeros(())
        affinity_penalty = adjacency.new_zeros(())
        loss = adjacency.new_zeros(())
        if delta_g_activations is not None:
            if delta_g_activations.ndim == 2:
                delta_g_activations = delta_g_activations.unsqueeze(0).expand(adjacency.size(0), -1, -1)
            kinetic_penalty = torch.mean(adjacency * torch.relu(delta_g_activations - self.activation_threshold))
            loss = loss + kinetic_penalty
        if delta_g_rxn is not None:
            if delta_g_rxn.ndim == 2:
                delta_g_rxn = delta_g_rxn.unsqueeze(0).expand(adjacency.size(0), -1, -1)
            thermo_penalty = torch.mean(adjacency * torch.relu(delta_g_rxn - self.thermodynamic_margin))
            loss = loss + thermo_penalty
        if binding_affinity is not None:
            if binding_affinity.ndim == 2:
                binding_affinity = binding_affinity.unsqueeze(0).expand(adjacency.size(0), -1, -1)
            affinity_penalty = torch.mean(adjacency * torch.relu(binding_affinity))
            loss = loss + affinity_penalty
        if physics_loss is not None:
            loss = loss + physics_loss.mean()
        return loss, kinetic_penalty, thermo_penalty, affinity_penalty

    def unified_causal_loss(
        self,
        adjacency: torch.Tensor,
        reconstruction_loss: torch.Tensor,
        *,
        delta_g_activations: Optional[torch.Tensor] = None,
        delta_g_rxn: Optional[torch.Tensor] = None,
        binding_affinity: Optional[torch.Tensor] = None,
        accessibility_prior: Optional[torch.Tensor] = None,
        physics_loss: Optional[torch.Tensor] = None,
        flux_consistency_loss: Optional[torch.Tensor] = None,
        manifold_recon_loss: Optional[torch.Tensor] = None,
        manifold_density_penalty: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        acyclicity = self.acyclicity_constraint(adjacency)
        sparsity = adjacency.abs().mean()
        physics_term, kinetic_penalty, thermo_penalty, affinity_penalty = self.thermodynamic_physics_loss(
            adjacency,
            delta_g_activations=delta_g_activations,
            delta_g_rxn=delta_g_rxn,
            binding_affinity=binding_affinity,
            physics_loss=physics_loss,
        )
        accessibility_penalty = self.compute_accessibility_penalty(adjacency, accessibility_prior)
        flux_penalty = (
            flux_consistency_loss.mean()
            if flux_consistency_loss is not None
            else torch.zeros((), dtype=adjacency.dtype, device=adjacency.device)
        )
        # Phase 1.5 manifold terms: L_recon + density penalty, weighted by manifold_weight
        manifold_term = torch.zeros((), dtype=adjacency.dtype, device=adjacency.device)
        if manifold_recon_loss is not None:
            manifold_term = manifold_term + manifold_recon_loss
        if manifold_density_penalty is not None:
            manifold_term = manifold_term + manifold_density_penalty
        total = (
            self.reconstruction_weight * reconstruction_loss
            + self.alpha * sparsity
            + self.beta * acyclicity
            + self.gamma * physics_term
            + self.gamma * accessibility_penalty
            + self.delta * flux_penalty
            + self.manifold_weight * manifold_term
        )
        return (
            total,
            acyclicity,
            sparsity,
            physics_term,
            kinetic_penalty,
            thermo_penalty,
            affinity_penalty,
            accessibility_penalty,
            flux_penalty,
        )

    def apply_digital_inhibition(
        self,
        W_original: torch.Tensor,
        inhibitor_mask_3d: torch.Tensor,
        reaction_trajectories: torch.Tensor,
        *,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from nexus.pocket.ddi import apply_digital_inhibition

        return apply_digital_inhibition(
            W_original,
            inhibitor_mask_3d,
            reaction_trajectories,
            threshold=threshold,
        )

    def spawn_generation(
        self,
        adjacency: torch.Tensor,
        *,
        threshold: Optional[float] = None,
        max_children: Optional[int] = None,
        operator_weights: Optional[torch.Tensor] = None,
        stability_mask: Optional[torch.Tensor] = None,
        child_generator: Optional[Callable[[int, int, int, int, torch.Tensor], object]] = None,
    ) -> torch.Tensor | list[object]:
        cutoff = self.edge_threshold if threshold is None else float(threshold)
        mask = adjacency > cutoff
        # Phase 1.5: gate edges whose destination node is not on the admissible
        # manifold (unstable surface energy or non-physical charge density).
        # stability_mask: [..., n_nodes] bool — True means the node may be a child.
        if stability_mask is not None:
            stable_dst = stability_mask.unsqueeze(-2)   # [..., 1, n] → broadcast to [..., n, n]
            mask = mask & stable_dst
        if child_generator is None:
            return mask

        if adjacency.ndim == 2:
            adj_batch = adjacency.unsqueeze(0)
            op_batch = operator_weights.unsqueeze(0) if operator_weights is not None and operator_weights.ndim == 3 else operator_weights
        else:
            adj_batch = adjacency
            op_batch = operator_weights

        flat_scores = adj_batch.reshape(-1)
        keep = flat_scores > cutoff
        if not bool(keep.any().item()):
            return []
        order = torch.argsort(flat_scores, descending=True)
        selected = order[keep[order]]
        if max_children is not None:
            selected = selected[: int(max_children)]

        n_nodes = adj_batch.size(-1)
        spawned: list[object] = []
        for flat_idx in selected:
            flat_int = int(flat_idx.detach().cpu().item())
            batch_idx = flat_int // (n_nodes * n_nodes)
            rem = flat_int % (n_nodes * n_nodes)
            src_idx = rem // n_nodes
            dst_idx = rem % n_nodes
            if src_idx == dst_idx:
                continue
            op_idx = 0
            if op_batch is not None:
                op_idx = int(torch.argmax(op_batch[batch_idx, src_idx, dst_idx]).detach().cpu().item())
            weight = adj_batch[batch_idx, src_idx, dst_idx]
            spawned.append(child_generator(batch_idx, src_idx, dst_idx, op_idx, weight))
        return spawned

    def forward(
        self,
        multivectors: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
        delta_g_activations: Optional[torch.Tensor] = None,
        delta_g_rxn: Optional[torch.Tensor] = None,
        binding_affinity: Optional[torch.Tensor] = None,
        accessibility_mask: Optional[torch.Tensor] = None,
        path_accessibility: Optional[torch.Tensor] = None,
        physics_loss: Optional[torch.Tensor] = None,
        flux_consistency_loss: Optional[torch.Tensor] = None,
    ) -> MetabolicDAGOutput:
        batch_size, n_nodes = multivectors.shape[:2]
        physics_prior = self.compute_physics_prior(
            delta_g_activations=delta_g_activations,
            delta_g_rxn=delta_g_rxn,
            binding_affinity=binding_affinity,
            batch_size=batch_size,
            n_nodes=n_nodes,
            dtype=multivectors.dtype,
            device=multivectors.device,
        )
        accessibility_prior = self.compute_accessibility_prior(
            accessibility_mask,
            path_accessibility=path_accessibility,
            batch_size=batch_size,
            n_nodes=n_nodes,
            dtype=multivectors.dtype,
            device=multivectors.device,
        )
        adjacency, raw_adjacency, logits, node_embeddings, operator_weights = self.predict_adjacency(
            multivectors,
            mask=mask,
            physics_prior=physics_prior,
            accessibility_prior=accessibility_prior,
        )
        reconstruction, recon_loss = self.reconstruction_loss(multivectors, node_embeddings, adjacency)

        # Phase 1.5 – Admissible Manifold Projection
        # Apply projector to the reconstruction (shape matches multivectors: [B, N, 8]).
        # This validates that spawned child multivectors lie on the chemical manifold.
        proj_out: ManifoldProjectionOutput = self.manifold_projector(reconstruction)

        total_loss, acyclicity, sparsity, physics_term, kinetic_penalty, thermo_penalty, affinity_penalty, accessibility_penalty, flux_penalty = self.unified_causal_loss(
            adjacency,
            recon_loss,
            delta_g_activations=delta_g_activations,
            delta_g_rxn=delta_g_rxn,
            binding_affinity=binding_affinity,
            accessibility_prior=accessibility_prior,
            physics_loss=physics_loss,
            flux_consistency_loss=flux_consistency_loss,
            manifold_recon_loss=proj_out.recon_loss,
            manifold_density_penalty=proj_out.density_penalty,
        )
        # spawn_generation uses stability_mask to gate admissible child nodes
        spawned_mask = self.spawn_generation(
            adjacency,
            threshold=threshold,
            stability_mask=proj_out.stability_mask,
        )
        return MetabolicDAGOutput(
            adjacency=adjacency,
            raw_adjacency=raw_adjacency,
            logits=logits,
            node_embeddings=node_embeddings,
            reconstruction=reconstruction,
            physics_prior=physics_prior,
            accessibility_prior=accessibility_prior,
            acyclicity=acyclicity,
            sparsity=sparsity,
            physics_loss=physics_term,
            kinetic_penalty=kinetic_penalty,
            thermodynamic_penalty=thermo_penalty,
            affinity_penalty=affinity_penalty,
            accessibility_penalty=accessibility_penalty,
            flux_penalty=flux_penalty,
            reconstruction_loss=recon_loss,
            causal_loss=total_loss,
            spawned_mask=spawned_mask,
            operator_weights=operator_weights,
            manifold_recon_loss=proj_out.recon_loss,
            manifold_density_penalty=proj_out.density_penalty,
            surface_energy=proj_out.surface_energy,
            stability_mask=proj_out.stability_mask,
        )


__all__ = [
    "GraNDAGEdgePredictor",
    "MetabolicDAGLearner",
    "MetabolicDAGOutput",
]
