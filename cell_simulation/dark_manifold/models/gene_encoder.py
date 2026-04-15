"""
dark_manifold/models/gene_encoder.py

Gene feature encoder for essentiality prediction.

Adapted from nexus/reasoning/metric_learner.py MechanismEncoder.

Key differences:
- Input: Gene features (15-dim) instead of Morgan fingerprints (2048-dim)
- Architecture: Smaller MLP appropriate for lower-dimensional input
- Output: L2-normalized embedding for similarity computation
"""

from __future__ import annotations

import math
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneEncoder(nn.Module):
    """
    Encode gene features into embeddings for similarity-based retrieval.
    
    Adapted from MechanismEncoder:
    - Input: Gene features (centrality, expression, redundancy, etc.)
    - Output: L2-normalized embedding for Euclidean or Poincaré retrieval
    
    Architecture: feature_dim -> 128 -> 64 -> embed_dim
    """
    
    def __init__(
        self, 
        feature_dim: int = 15, 
        embed_dim: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, embed_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode gene features to L2-normalized embedding.
        
        Args:
            features: Gene feature tensor [..., feature_dim]
        
        Returns:
            L2-normalized embedding [..., embed_dim]
        """
        return F.normalize(self.net(features.float()), p=2, dim=-1)


class PoincareMath:
    """
    Numerically safe Poincaré ball operations.
    
    Copied from metric_learner.py for standalone use.
    """
    
    def __init__(self, c: float = 1.0, eps: float = 1e-15) -> None:
        self.c = float(max(c, 1e-8))
        self.eps = float(max(eps, 1e-15))
    
    def _sqrt_c(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(math.sqrt(self.c))
    
    @staticmethod
    def _safe_norm(x: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
        """||x|| with defined gradient at x=0."""
        return (x.pow(2).sum(dim=-1, keepdim=True) + eps * eps).sqrt()
    
    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Project back into ball if needed."""
        max_norm = (1.0 - 1e-5) / math.sqrt(self.c)
        norm = self._safe_norm(x, self.eps)
        safe = torch.where(norm > max_norm, x * (max_norm / norm), x)
        return torch.nan_to_num(safe, nan=0.0, posinf=max_norm, neginf=-max_norm)
    
    def exp_map_0(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from origin to Poincaré ball."""
        v = v.float()
        v_norm = self._safe_norm(v, self.eps)
        sqrt_c = self._sqrt_c(v)
        scale = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        return self._project(scale * v)
    
    def log_map_0(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from Poincaré ball to tangent space at origin."""
        y = self._project(y.float())
        y_norm = self._safe_norm(y, self.eps)
        sqrt_c = self._sqrt_c(y)
        clipped = (sqrt_c * y_norm).clamp_max(1.0 - 1e-6)
        scale = torch.atanh(clipped) / (sqrt_c * y_norm)
        return scale * y
    
    def distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Geodesic distance in Poincaré ball."""
        u = self._project(u.float())
        v = self._project(v.float())
        sqdist = (u - v).pow(2).sum(dim=-1, keepdim=True)
        squnorm = u.pow(2).sum(dim=-1, keepdim=True)
        sqvnorm = v.pow(2).sum(dim=-1, keepdim=True)
        c = u.new_tensor(self.c)
        arg = 1.0 + 2.0 * c * sqdist / (
            (1.0 - c * squnorm).clamp_min(self.eps) * 
            (1.0 - c * sqvnorm).clamp_min(self.eps)
        )
        return torch.acosh(arg.clamp_min(1.0 + self.eps)).squeeze(-1)


class HyperbolicGeneEncoder(nn.Module):
    """
    Project gene features into Poincaré ball for hyperbolic retrieval.
    
    Adapted from HGNNProjection:
    - Maps gene features to tangent space
    - Projects to Poincaré ball via exponential map
    - Enables hyperbolic similarity computation
    """
    
    def __init__(
        self,
        feature_dim: int = 15,
        hidden_dim: int = 128,
        poincare_dim: int = 64,
        c: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.poincare_dim = poincare_dim
        self.math = PoincareMath(c=c)
        
        self.tangent_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, poincare_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project gene features to Poincaré ball.
        
        Args:
            features: Gene feature tensor [..., feature_dim]
        
        Returns:
            Poincaré embedding [..., poincare_dim]
        """
        tangent = self.tangent_mlp(features.float())
        return self.math.exp_map_0(tangent)
    
    def distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute geodesic distance between embeddings."""
        return self.math.distance(u, v)


# ============================================================================
# GENE FEATURE DEFINITIONS
# ============================================================================

# Feature names and indices for reference
GENE_FEATURE_NAMES = [
    # FBA features (0-2)
    'biomass_ratio',           # 0: Knockout biomass / WT biomass
    'num_blocked_reactions',   # 1: Reactions blocked by knockout
    'flux_variability',        # 2: FVA range for associated reactions
    
    # Network topology (3-7)
    'degree_centrality',       # 3: Gene connectivity
    'betweenness_centrality',  # 4: Pathway bridging
    'closeness_centrality',    # 5: Average path length
    'is_hub',                  # 6: High connectivity flag
    'clustering_coefficient',  # 7: Local clustering
    
    # Redundancy features (8-10)
    'isozyme_count',           # 8: Number of isozymes
    'has_alternative_pathway', # 9: Bypass route exists
    'pathway_redundancy',      # 10: Overall pathway backup
    
    # Expression features (11-12)
    'expression_level',        # 11: mRNA/protein abundance
    'protein_halflife',        # 12: Stability
    
    # Thermodynamic features (13-14)
    'reaction_delta_g',        # 13: Thermodynamic favorability
    'is_irreversible',         # 14: One-way reaction flag
]

FEATURE_DIM = len(GENE_FEATURE_NAMES)  # 15


def get_feature_index(name: str) -> int:
    """Get index for a feature name."""
    return GENE_FEATURE_NAMES.index(name)


if __name__ == "__main__":
    # Quick test
    print("Testing GeneEncoder...")
    
    encoder = GeneEncoder(feature_dim=15, embed_dim=64)
    features = torch.randn(10, 15)  # 10 genes, 15 features each
    embeddings = encoder(features)
    print(f"Input: {features.shape}")
    print(f"Output: {embeddings.shape}")
    print(f"L2 norm: {embeddings.norm(dim=-1)}")  # Should all be ~1
    
    print("\nTesting HyperbolicGeneEncoder...")
    hyp_encoder = HyperbolicGeneEncoder(feature_dim=15, poincare_dim=64)
    poincare_embeddings = hyp_encoder(features)
    print(f"Poincaré embeddings: {poincare_embeddings.shape}")
    print(f"Norms (should be < 1): {poincare_embeddings.norm(dim=-1)}")
    
    # Test distance computation
    d = hyp_encoder.distance(poincare_embeddings[0:1], poincare_embeddings[1:2])
    print(f"Distance between gene 0 and 1: {d.item():.4f}")
