"""
Nexus Stub Module

This provides stub implementations of nexus classes so the model can load
without the full nexus package installed.

The stubs are minimal passthrough implementations that maintain tensor shapes
but don't provide the full nexus functionality.
"""

import torch
import torch.nn as nn

class HGNNProjection(nn.Module):
    """Stub for nexus.reasoning.metric_learner.HGNNProjection"""
    
    def __init__(self, in_dim=128, out_dim=64, **kwargs):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, edge_index=None, edge_attr=None, **kwargs):
        return self.proj(x)


class NexusDualDecoder(nn.Module):
    """Stub for nexus.reasoning_wave.analogical_fusion.NexusDualDecoder"""
    
    def __init__(self, dim=128, **kwargs):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, **kwargs):
        return self.proj(x)


class PGWCrossAttention(nn.Module):
    """Stub for nexus.reasoning_wave.analogical_fusion.PGWCrossAttention"""
    
    def __init__(self, dim=128, heads=4, **kwargs):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    
    def forward(self, query, key, value=None, **kwargs):
        if value is None:
            value = key
        out, _ = self.attn(query, key, value)
        return out


class WaveQuantumDistillationHead(nn.Module):
    """Stub for nexus.reasoning_wave.metric_learner.WaveQuantumDistillationHead"""
    
    def __init__(self, dim=128, out_dim=1, **kwargs):
        super().__init__()
        self.head = nn.Linear(dim, out_dim)
    
    def forward(self, x, **kwargs):
        return self.head(x)


def quantum_distillation_loss(pred, target, **kwargs):
    """Stub for nexus.reasoning_wave.metric_learner.quantum_distillation_loss"""
    return torch.tensor(0.0, device=pred.device)


# Export all
__all__ = [
    'HGNNProjection',
    'NexusDualDecoder', 
    'PGWCrossAttention',
    'WaveQuantumDistillationHead',
    'quantum_distillation_loss',
]
