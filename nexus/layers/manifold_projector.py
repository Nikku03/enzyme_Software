"""
Phase 1.5 – Admissible Manifold (Ontology Projection)
======================================================
Replaces heuristic ontology validators with a fully differentiable
"snap-to-grid" operation on the Cl(3,0) multivector manifold.

The projector is a bottleneck autoencoder that works entirely in the
O(3)-invariant feature space extracted from raw 8D multivectors.
Because the AE operates on invariants (norms, scalars, overlaps) rather
than raw grade components, every quantity it computes — reconstruction
loss, surface energy, density penalty, stability mask — is guaranteed
to be rotation-invariant.

Architecture
------------
1. _invariant_features : 8D multivector → 8 invariant scalars
2. Encoder             : Linear(8, hidden) → SiLU → Linear(hidden, latent_dim)
3. Decoder             : Linear(latent_dim, hidden) → SiLU → Linear(hidden, 8)
4. L_recon             : MSE(input_invariants, decoded_invariants)

Stability metrics (closed-form, no learned parameters)
-------------------------------------------------------
* Surface area energy  E_surf = ||bivector|| / (|scalar| + ε)
  High ratio → high curvature → thermodynamically unstable intermediate
* Electronic density   penalty for scalar < 0 (non-physical charge)
  and radical character = ||vector|| / |scalar| > threshold (unpaired e-)

Integration
-----------
MultivectorManifoldProjector is called in MetabolicDAGLearner.forward on
the reconstructed multivectors produced by the GraN-DAG reconstruction
head.  Its losses are folded into the unified causal loss; its
stability_mask gates which child nodes may be spawned.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ManifoldProjectionOutput:
    """All outputs of one manifold-projection forward pass."""
    projected_invariants: torch.Tensor  # [..., n_nodes, 8]  decoded valid invariants
    recon_loss: torch.Tensor            # scalar  MSE in invariant space (L_recon)
    surface_energy: torch.Tensor        # [..., n_nodes]  per-node stability metric
    density_penalty: torch.Tensor       # scalar  electronic density violation
    stability_mask: torch.Tensor        # [..., n_nodes]  bool: True = admissible node


class MultivectorManifoldProjector(nn.Module):
    """Differentiable ontology projector for Phase 1.5.

    Args:
        latent_dim:                 Bottleneck dimensionality (default 4).
                                    Smaller values enforce tighter manifold
                                    compression.
        hidden_dim:                 Width of encoder/decoder hidden layer
                                    (default 32).
        surface_energy_threshold:   Nodes with E_surf above this are flagged
                                    unstable.  Typical stable organics sit
                                    below 2.0 (default).
        radical_ratio_threshold:    ||vector|| / |scalar| limit above which
                                    a node is considered a radical (default 3.0).
    """

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dim: int = 32,
        surface_energy_threshold: float = 2.0,
        radical_ratio_threshold: float = 3.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.surface_energy_threshold = float(surface_energy_threshold)
        self.radical_ratio_threshold = float(radical_ratio_threshold)

        # Bottleneck AE operating on the 8-dim invariant feature space.
        self.encoder = nn.Sequential(
            nn.Linear(8, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 8),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _invariant_features(multivectors: torch.Tensor) -> torch.Tensor:
        """Extract 8 O(3)-invariant scalars from 8D public-layout multivectors.

        Layout: [1, e1, e2, e3, e12, e23, e31, e123]
        Indices:  0   1   2   3    4    5    6    7

        Returns [..., 8] invariant tensor:
          scalar, vector_norm, bivector_norm, trivector,
          total_norm, vector-bivector overlap,
          scalar-trivector mix, parity balance
        """
        scalar       = multivectors[..., 0:1]
        vector       = multivectors[..., 1:4]
        bivector     = multivectors[..., 4:7]
        trivector    = multivectors[..., 7:8]

        vector_norm        = vector.norm(dim=-1, keepdim=True)
        bivector_norm      = bivector.norm(dim=-1, keepdim=True)
        total_norm         = multivectors.norm(dim=-1, keepdim=True)
        vb_overlap         = (vector * bivector).sum(dim=-1, keepdim=True)
        scalar_trivec_mix  = scalar * trivector
        parity_balance     = vector_norm - bivector_norm

        return torch.cat(
            [scalar, vector_norm, bivector_norm, trivector,
             total_norm, vb_overlap, scalar_trivec_mix, parity_balance],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Stability metrics  (closed-form, rotation-invariant)
    # ------------------------------------------------------------------

    def surface_area_energy(self, multivectors: torch.Tensor) -> torch.Tensor:
        """Per-node surface area energy proxy.

        E_surf = ||bivector|| / (|scalar| + ε)

        Physically: ratio of rotational/angular field content to charge
        density.  High values indicate reactive intermediates with high
        curvature on the potential energy surface.

        Returns [..., n_nodes]
        """
        scalar_mag    = multivectors[..., 0].abs() + 1.0e-6
        bivector_norm = multivectors[..., 4:7].norm(dim=-1)
        return bivector_norm / scalar_mag

    def electronic_density_penalty(self, multivectors: torch.Tensor) -> torch.Tensor:
        """Scalar penalty for non-physical electronic states.

        Two terms:
          1. Negative scalar component  →  relu(-scalar).mean()
             Electron density must be non-negative everywhere.
          2. Radical character  →  relu(||vector||/|scalar| − threshold).mean()
             Extreme vector/scalar ratios indicate unpaired-electron (radical)
             character that would be pruned by a deterministic validator.

        Returns a scalar loss.
        """
        scalar       = multivectors[..., 0]
        vector_norm  = multivectors[..., 1:4].norm(dim=-1)
        scalar_mag   = scalar.abs() + 1.0e-6

        density_violation  = F.relu(-scalar)
        radical_ratio      = vector_norm / scalar_mag
        radical_violation  = F.relu(radical_ratio - self.radical_ratio_threshold)

        return (density_violation + radical_violation).mean()

    def stability_mask_for(self, multivectors: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True where a node passes both stability checks.

        A node is admissible if:
          * E_surf < surface_energy_threshold
          * scalar ≥ 0  (non-negative charge density)

        Returns [..., n_nodes] bool tensor.
        """
        e_surf   = self.surface_area_energy(multivectors)
        positive = multivectors[..., 0] >= 0.0
        return (e_surf < self.surface_energy_threshold) & positive

    # ------------------------------------------------------------------
    # Core projection
    # ------------------------------------------------------------------

    def project(self, multivectors: torch.Tensor) -> torch.Tensor:
        """Snap multivectors to the nearest point on the learned manifold.

        Operates in invariant space: encode → bottleneck → decode.

        Args:
            multivectors: [..., n_nodes, 8]

        Returns:
            [..., n_nodes, 8]  decoded (projected) invariant representation.
        """
        inv_feat  = self._invariant_features(multivectors)   # [..., n, 8]
        latent    = self.encoder(inv_feat)                    # [..., n, latent_dim]
        return self.decoder(latent)                           # [..., n, 8]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, multivectors: torch.Tensor) -> ManifoldProjectionOutput:
        """Run the full Phase 1.5 projection pass.

        Args:
            multivectors: [..., n_nodes, 8]  — typically the reconstructed
                          multivectors from the GraN-DAG reconstruction head.

        Returns:
            ManifoldProjectionOutput with all stability signals.
        """
        if multivectors.size(-1) != 8:
            raise ValueError("multivectors must have trailing dimension 8")

        # Manifold snap: encode → decode in invariant space
        inv_input  = self._invariant_features(multivectors)   # [..., n, 8]
        latent     = self.encoder(inv_input)                  # [..., n, latent_dim]
        inv_proj   = self.decoder(latent)                     # [..., n, 8]

        # L_recon: distance from the learned chemical manifold (invariant MSE)
        recon_loss = F.mse_loss(inv_proj, inv_input, reduction="mean")

        # Stability metrics applied to the input multivectors
        surf_energy      = self.surface_area_energy(multivectors)
        density_penalty  = self.electronic_density_penalty(multivectors)
        stab_mask        = self.stability_mask_for(multivectors)

        return ManifoldProjectionOutput(
            projected_invariants=inv_proj,
            recon_loss=recon_loss,
            surface_energy=surf_energy,
            density_penalty=density_penalty,
            stability_mask=stab_mask,
        )


__all__ = ["MultivectorManifoldProjector", "ManifoldProjectionOutput"]
