"""
Phase 1: Relational Proposer for Site-of-Metabolism Prediction

This module replaces the scalar proposer head with a relational architecture
that can compare atoms to each other, addressing the core bottleneck identified
in the pairwise probe experiments (77% pairwise accuracy vs 15% scalar accuracy).

Key insight: The backbone embeddings contain discriminative signal that the
scalar head cannot extract because it scores atoms independently. A relational
head can compare atoms and learn "atom A is more likely than atom B."

Architecture:
    1. Self-attention over atom embeddings (cross-atom comparison)
    2. Optional pairwise MLP aggregation (proven to work at 77% acc)
    3. Final scoring head that uses relational context

Author: Claude (Anthropic) for Naresh Chhillar
Date: 2026-04-09
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List
import math

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch, F


if TORCH_AVAILABLE:
    
    class RelationalSelfAttention(nn.Module):
        """
        Multi-head self-attention for atom embeddings.
        
        Each atom attends to all other atoms in the molecule, learning
        which comparisons are informative for SoM prediction.
        """
        
        def __init__(
            self,
            embed_dim: int,
            num_heads: int = 4,
            dropout: float = 0.1,
            bias: bool = True,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
            
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.head_dim)
            
        def forward(
            self, 
            x: torch.Tensor, 
            batch: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: Atom embeddings [N_atoms, embed_dim]
                batch: Molecule assignment [N_atoms] 
                mask: Optional attention mask
                
            Returns:
                attended: Updated atom embeddings [N_atoms, embed_dim]
                attn_weights: Attention weights for analysis
            """
            N = x.size(0)
            
            # Project to Q, K, V
            q = self.q_proj(x)  # [N, embed_dim]
            k = self.k_proj(x)  # [N, embed_dim]
            v = self.v_proj(x)  # [N, embed_dim]
            
            # Create per-molecule attention mask
            # Atoms can only attend to atoms in the same molecule
            mol_mask = (batch.unsqueeze(0) == batch.unsqueeze(1))  # [N, N]
            
            # Compute attention scores
            # Reshape for multi-head attention
            q = q.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [heads, N, head_dim]
            k = k.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [heads, N, head_dim]
            v = v.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # [heads, N, head_dim]
            
            # Attention: [heads, N, N]
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            
            # Apply molecule mask (atoms only attend within same molecule)
            attn_scores = attn_scores.masked_fill(~mol_mask.unsqueeze(0), float('-inf'))
            
            # Apply optional additional mask
            if mask is not None:
                attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
            
            # Softmax and dropout
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Handle all-masked rows
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            attended = torch.matmul(attn_weights, v)  # [heads, N, head_dim]
            
            # Reshape back
            attended = attended.transpose(0, 1).contiguous().view(N, self.embed_dim)
            
            # Output projection
            attended = self.out_proj(attended)
            
            # Average attention weights across heads for analysis
            attn_weights_avg = attn_weights.mean(dim=0)  # [N, N]
            
            return attended, attn_weights_avg
    
    
    class PairwiseAggregator(nn.Module):
        """
        Aggregate pairwise comparisons to produce per-atom scores.
        
        For each atom i, compares it against all other atoms j in the molecule
        and aggregates "i beats j" signals into a final score.
        
        This mirrors the pairwise probe architecture that achieved 77% accuracy.
        """
        
        def __init__(
            self,
            embed_dim: int,
            hidden_dim: Optional[int] = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            hidden_dim = hidden_dim or embed_dim
            
            # Pairwise comparison MLP
            # Input: [z_i, z_j, z_i - z_j, z_i * z_j] = 4 * embed_dim
            self.pairwise_mlp = nn.Sequential(
                nn.Linear(4 * embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            
        def forward(
            self,
            x: torch.Tensor,
            batch: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                x: Atom embeddings [N_atoms, embed_dim]
                batch: Molecule assignment [N_atoms]
                
            Returns:
                scores: Per-atom aggregated pairwise scores [N_atoms, 1]
            """
            N = x.size(0)
            device = x.device
            
            # Create molecule mask
            mol_mask = (batch.unsqueeze(0) == batch.unsqueeze(1))  # [N, N]
            
            # Compute all pairwise features
            # This is O(N^2) but N is typically small (10-50 atoms per molecule)
            x_i = x.unsqueeze(1).expand(N, N, -1)  # [N, N, D]
            x_j = x.unsqueeze(0).expand(N, N, -1)  # [N, N, D]
            
            pairwise_features = torch.cat([
                x_i,
                x_j,
                x_i - x_j,
                x_i * x_j,
            ], dim=-1)  # [N, N, 4*D]
            
            # Compute pairwise scores
            pairwise_scores = self.pairwise_mlp(pairwise_features).squeeze(-1)  # [N, N]
            
            # Mask out cross-molecule pairs
            pairwise_scores = pairwise_scores.masked_fill(~mol_mask, 0.0)
            
            # Aggregate: for each atom i, sum P(i beats j) across all j in same molecule
            # Higher score = more atoms this atom beats
            aggregated = pairwise_scores.sum(dim=1, keepdim=True)  # [N, 1]
            
            # Normalize by number of atoms in each molecule
            atoms_per_mol = mol_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            aggregated = aggregated / atoms_per_mol
            
            return aggregated
    
    
    class RelationalProposer(nn.Module):
        """
        Relational Site-of-Metabolism Proposer.
        
        Replaces the scalar proposer head that scores atoms independently.
        Uses self-attention to let atoms "see" each other before scoring.
        
        Architecture:
            Input atom embeddings
                ↓
            Self-attention (cross-atom context)
                ↓
            [Optional] Pairwise aggregation
                ↓
            Scoring head with relational context
                ↓
            Site logits
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: Optional[int] = None,
            num_attention_heads: int = 4,
            num_attention_layers: int = 2,
            use_pairwise_aggregator: bool = True,
            dropout: float = 0.1,
            residual_scale: float = 0.1,
        ):
            super().__init__()
            self.input_dim = input_dim
            hidden_dim = hidden_dim or input_dim
            self.hidden_dim = hidden_dim
            self.num_attention_layers = num_attention_layers
            self.use_pairwise_aggregator = use_pairwise_aggregator
            self.residual_scale = residual_scale
            
            # Input projection if dimensions differ
            self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
            
            # Self-attention layers
            self.attention_layers = nn.ModuleList([
                RelationalSelfAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                )
                for _ in range(num_attention_layers)
            ])
            
            # Layer norms for residual connections
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(num_attention_layers)
            ])
            
            # Optional pairwise aggregator
            self.pairwise_aggregator = (
                PairwiseAggregator(hidden_dim, hidden_dim, dropout)
                if use_pairwise_aggregator
                else None
            )
            
            # Final scoring head
            # Input: attended features + optional pairwise score
            score_input_dim = hidden_dim + (1 if use_pairwise_aggregator else 0)
            self.score_head = nn.Sequential(
                nn.Linear(score_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            
            # Initialize final layer to near-zero for stable training start
            nn.init.zeros_(self.score_head[-1].weight)
            nn.init.zeros_(self.score_head[-1].bias)
            
        def forward(
            self,
            features: torch.Tensor,
            batch: torch.Tensor,
            prior_logits: Optional[torch.Tensor] = None,
            return_attention: bool = False,
        ) -> Tuple[torch.Tensor, Dict]:
            """
            Args:
                features: Atom embeddings [N_atoms, input_dim]
                batch: Molecule assignment [N_atoms]
                prior_logits: Optional prior scores to fuse [N_atoms, 1]
                return_attention: Whether to return attention weights
                
            Returns:
                logits: Site prediction logits [N_atoms, 1]
                diagnostics: Dict with attention weights, intermediate values
            """
            # Project to hidden dim
            x = self.input_proj(features)
            
            attention_weights = []
            
            # Apply self-attention layers with residual connections
            for attn, ln in zip(self.attention_layers, self.layer_norms):
                attended, attn_w = attn(x, batch)
                x = ln(x + self.residual_scale * attended)
                if return_attention:
                    attention_weights.append(attn_w)
            
            # Optional pairwise aggregation
            if self.pairwise_aggregator is not None:
                pairwise_score = self.pairwise_aggregator(x, batch)
                score_input = torch.cat([x, pairwise_score], dim=-1)
            else:
                pairwise_score = None
                score_input = x
            
            # Final scoring
            logits = self.score_head(score_input)
            
            # Optional fusion with prior logits
            if prior_logits is not None:
                if prior_logits.ndim == 1:
                    prior_logits = prior_logits.unsqueeze(-1)
                logits = logits + prior_logits
            
            diagnostics = {
                "attended_features": x,
                "pairwise_score": pairwise_score,
                "attention_weights": attention_weights if return_attention else None,
                "logits_mean": float(logits.detach().mean().item()),
                "logits_std": float(logits.detach().std().item()),
            }
            
            return logits, diagnostics
    

    class RelationalFusionHead(nn.Module):
        """
        Drop-in replacement for ResidualFusionHead that uses relational scoring.
        
        Maintains the same interface as ResidualFusionHead for compatibility:
            logits, diagnostics = head(features, prior_logits, prior_features)
            
        But internally uses RelationalProposer for cross-atom comparison.
        """
        
        def __init__(
            self,
            input_dim: int,
            output_dim: int = 1,
            *,
            prior_feature_dim: int = 0,
            hidden_dim: Optional[int] = None,
            fusion_mode: str = "gated_add",
            dropout: float = 0.1,
            prior_scale_init: float = 0.65,
            num_attention_heads: int = 4,
            num_attention_layers: int = 2,
            use_pairwise_aggregator: bool = True,
        ):
            super().__init__()
            assert output_dim == 1, "RelationalFusionHead only supports output_dim=1 (site prediction)"
            
            self.output_dim = output_dim
            self.prior_feature_dim = prior_feature_dim
            self.fusion_mode = fusion_mode
            hidden_dim = hidden_dim or max(32, input_dim)
            
            # Prior scale (same as ResidualFusionHead)
            init = min(max(float(prior_scale_init), 1e-3), 1.5)
            init_sigmoid = min(max(init / 1.5, 1e-3), 1.0 - 1e-3)
            self.prior_scale_logit = nn.Parameter(
                torch.logit(torch.tensor(init_sigmoid, dtype=torch.float32))
            )
            
            # The relational proposer
            self.relational_proposer = RelationalProposer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_attention_heads=num_attention_heads,
                num_attention_layers=num_attention_layers,
                use_pairwise_aggregator=use_pairwise_aggregator,
                dropout=dropout,
            )
            
            # Gate network for prior fusion (same as ResidualFusionHead)
            gate_in = input_dim + output_dim + prior_feature_dim
            self.gate_net = nn.Sequential(
                nn.Linear(gate_in, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid(),
            )
            
            # Store batch for relational attention (set externally before forward)
            self._batch = None
            
        def set_batch(self, batch: torch.Tensor):
            """Set molecule batch indices for attention masking."""
            self._batch = batch
            
        def forward(
            self,
            features: torch.Tensor,
            prior_logits: Optional[torch.Tensor] = None,
            prior_features: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Dict]:
            """
            Same interface as ResidualFusionHead.
            """
            # DEBUG: Print once to verify this path is being used
            if not hasattr(self, '_debug_printed'):
                print(f"[DEBUG] RelationalFusionHead.forward() called! features.shape={features.shape}", flush=True)
                self._debug_printed = True
                
            device = features.device
            N = features.size(0)
            
            # Handle batch - if not set, assume single molecule
            if self._batch is None:
                batch = torch.zeros(N, dtype=torch.long, device=device)
            else:
                batch = self._batch.to(device)
            
            # Get relational logits
            relational_logits, rel_diag = self.relational_proposer(features, batch)
            
            # Handle priors (same as ResidualFusionHead)
            prior_scale = 1.5 * torch.sigmoid(self.prior_scale_logit).to(device=device, dtype=features.dtype)
            
            if prior_logits is None:
                prior_logits = torch.zeros(N, self.output_dim, device=device, dtype=features.dtype)
            if prior_logits.ndim == 1:
                prior_logits = prior_logits.unsqueeze(-1)
                
            if self.prior_feature_dim > 0:
                if prior_features is None:
                    prior_features = torch.zeros(N, self.prior_feature_dim, device=device, dtype=features.dtype)
                elif prior_features.ndim == 1:
                    prior_features = prior_features.unsqueeze(-1)
            else:
                prior_features = torch.zeros(N, 0, device=device, dtype=features.dtype)
                
            scaled_prior_logits = prior_scale * prior_logits
            
            # Fusion
            if self.fusion_mode == "additive":
                gate = torch.ones_like(relational_logits)
                logits = scaled_prior_logits + relational_logits
            else:
                gate = self.gate_net(torch.cat([features, scaled_prior_logits, prior_features], dim=-1))
                logits = scaled_prior_logits + gate * relational_logits
            
            diagnostics = {
                "residual_logits": relational_logits,
                "prior_logits": scaled_prior_logits,
                "gate": gate,
                "prior_scale": prior_scale,
                "relational_diagnostics": rel_diag,
                "diagnostics": {
                    "residual_abs_mean": float(relational_logits.detach().abs().mean().item()),
                    "prior_abs_mean": float(prior_logits.detach().abs().mean().item()),
                    "gate_mean": float(gate.detach().mean().item()),
                    "prior_scale": float(prior_scale.detach().item()),
                },
            }
            
            return logits, diagnostics


else:  # pragma: no cover
    class RelationalSelfAttention:
        def __init__(self, *args, **kwargs):
            require_torch()
            
    class PairwiseAggregator:
        def __init__(self, *args, **kwargs):
            require_torch()
            
    class RelationalProposer:
        def __init__(self, *args, **kwargs):
            require_torch()
            
    class RelationalFusionHead:
        def __init__(self, *args, **kwargs):
            require_torch()
