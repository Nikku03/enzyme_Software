"""
Stub for nexus.reasoning.metric_learner
"""

import torch
import torch.nn as nn


class HGNNProjection(nn.Module):
    """Stub for HGNNProjection - Hyperbolic Graph Neural Network projection."""
    
    def __init__(self, in_channels_16d=16, hidden_dim=128, poincare_dim=64, dropout=0.05, **kwargs):
        super().__init__()
        # Match the expected interface: input is 16d multivector, output is poincare_dim
        self.net = nn.Sequential(
            nn.Linear(in_channels_16d, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, poincare_dim),
        )
    
    def forward(self, x, edge_index=None, edge_attr=None, batch=None, **kwargs):
        # x shape: (num_atoms, 16) for a single molecule
        # Need to aggregate to single embedding, then project
        if x.dim() == 2 and x.size(0) > 1:
            # Mean pool over atoms
            x = x.mean(dim=0, keepdim=True)
        return self.net(x).squeeze(0)


__all__ = ['HGNNProjection']
