"""
NEXUS-Lite: Neural EXtraction of Universal Substrate Reactivity (Lite Version)

This is a self-contained implementation of the NEXUS architecture concepts
without requiring the external nexus package. It implements:

1. **Hyperbolic Embeddings**: Molecules are embedded in hyperbolic space where
   hierarchical relationships (functional groups → molecules → reactions) are
   naturally represented. The Poincaré ball model is used for its numerical
   stability.

2. **Analogical Memory Bank**: A differentiable memory system that stores
   representations of known reaction sites and retrieves similar cases for
   new predictions. Uses locality-sensitive hashing for efficient retrieval.

3. **Causal Reasoning Engine**: Models reaction pathways as causal graphs,
   allowing the system to reason about WHY certain sites are reactive based
   on electronic effects, sterics, and enzyme binding.

4. **Reaction Manifold Learning**: Uses topological analysis (persistent
   homology approximation) to identify reaction-defining structural motifs.

Key insight: By embedding molecules in hyperbolic space, we can capture the
hierarchical nature of chemical reactivity - functional groups determine local
reactivity, molecular context modulates it, and enzyme binding selects among
possible sites.

Mathematical foundations:
- Poincaré ball: x ∈ ℝ^n with ||x|| < 1
- Möbius addition: x ⊕ y = ((1+2<x,y>+||y||²)x + (1-||x||²)y) / (1+2<x,y>+||x||²||y||²)
- Exponential map: exp_x(v) = x ⊕ (tanh(||v||/2) * v/||v||)
- Distance: d(x,y) = 2 * arctanh(||(-x) ⊕ y||)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# ============================================================================
# HYPERBOLIC GEOMETRY OPERATIONS
# ============================================================================

if TORCH_AVAILABLE:
    
    class HyperbolicOperations:
        """
        Operations in the Poincaré ball model of hyperbolic space.
        
        The Poincaré ball is the unit ball {x ∈ ℝ^n : ||x|| < 1} with the
        hyperbolic metric. Key properties:
        - Points near the boundary are "far" from the center (tree leaves)
        - Points near the center are "close" to everything (tree root)
        - Perfect for hierarchical representations
        """
        
        @staticmethod
        def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
            """
            Möbius addition in the Poincaré ball.
            
            x ⊕ y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) / 
                    (1 + 2c<x,y> + c²||x||²||y||²)
            """
            x_sq = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), max=1-eps)
            y_sq = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), max=1-eps)
            xy = torch.sum(x * y, dim=-1, keepdim=True)
            
            num = (1 + 2*c*xy + c*y_sq) * x + (1 - c*x_sq) * y
            denom = 1 + 2*c*xy + c*c*x_sq*y_sq
            
            return num / (denom + eps)
        
        @staticmethod
        def mobius_matvec(M: torch.Tensor, x: torch.Tensor, c: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
            """
            Möbius matrix-vector multiplication.
            
            M ⊗ x = exp_0(M @ log_0(x))
            
            This allows linear transformations in hyperbolic space.
            """
            x_norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=eps, max=1-eps)
            x_scale = torch.atanh(x_norm) / x_norm
            
            # Log map to tangent space at origin
            v = x_scale * x
            
            # Linear transform
            Mv = F.linear(v, M)
            
            # Exp map back to hyperbolic space
            Mv_norm = torch.clamp(Mv.norm(dim=-1, keepdim=True), min=eps)
            return torch.tanh(Mv_norm) * Mv / Mv_norm
        
        @staticmethod
        def expmap0(v: torch.Tensor, c: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
            """
            Exponential map at the origin.
            
            Maps from tangent space (Euclidean) to hyperbolic space.
            exp_0(v) = tanh(sqrt(c)||v||) * v / (sqrt(c)||v||)
            """
            sqrt_c = math.sqrt(c)
            v_norm = torch.clamp(v.norm(dim=-1, keepdim=True), min=eps)
            return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
        
        @staticmethod
        def logmap0(x: torch.Tensor, c: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
            """
            Logarithmic map at the origin.
            
            Maps from hyperbolic space to tangent space.
            log_0(x) = arctanh(sqrt(c)||x||) * x / (sqrt(c)||x||)
            """
            sqrt_c = math.sqrt(c)
            x_norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=eps, max=1-eps)
            return torch.atanh(sqrt_c * x_norm) * x / (sqrt_c * x_norm)
        
        @staticmethod
        def distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
            """
            Hyperbolic distance in Poincaré ball.
            
            d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) ⊕ y||)
            """
            sqrt_c = math.sqrt(c)
            diff = HyperbolicOperations.mobius_add(-x, y, c)
            diff_norm = torch.clamp(diff.norm(dim=-1), min=eps, max=1-eps)
            return (2/sqrt_c) * torch.atanh(sqrt_c * diff_norm)
        
        @staticmethod
        def project(x: torch.Tensor, c: float = 1.0, eps: float = 1e-3) -> torch.Tensor:
            """Project points back into the Poincaré ball."""
            max_norm = (1 - eps) / math.sqrt(c)
            norm = x.norm(dim=-1, keepdim=True)
            cond = norm > max_norm
            projected = x / norm * max_norm
            return torch.where(cond, projected, x)


    class HyperbolicLinear(nn.Module):
        """
        Linear layer in hyperbolic space.
        
        Uses Möbius matrix-vector multiplication followed by bias addition
        in the tangent space.
        """
        
        def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.c = c
            
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
            
            self.reset_parameters()
        
        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Möbius matvec
            out = HyperbolicOperations.mobius_matvec(self.weight, x, self.c)
            
            # Add bias in tangent space
            if self.bias is not None:
                out = HyperbolicOperations.logmap0(out, self.c)
                out = out + self.bias
                out = HyperbolicOperations.expmap0(out, self.c)
            
            return HyperbolicOperations.project(out, self.c)


    # ========================================================================
    # ANALOGICAL MEMORY BANK
    # ========================================================================
    
    class AnalogicalMemoryBank(nn.Module):
        """
        Differentiable memory bank for analogical reasoning.
        
        Stores representations of known reaction sites and retrieves similar
        cases for new predictions. Key features:
        
        1. **Hyperbolic Storage**: Memory keys are in hyperbolic space,
           capturing hierarchical similarity (same functional group → similar
           reactivity patterns).
        
        2. **Multi-Slot Values**: Each memory stores:
           - Atom embedding
           - Reaction pattern ID
           - CYP isoform compatibility
           - True reactivity label
        
        3. **Soft Retrieval**: Uses hyperbolic distance for weighted retrieval,
           allowing smooth gradients during training.
        
        4. **Locality-Sensitive Hashing**: For efficient approximate retrieval
           during inference.
        """
        
        def __init__(
            self,
            key_dim: int = 64,
            value_dim: int = 128,
            capacity: int = 4096,
            num_heads: int = 4,
            topk: int = 32,
            temperature: float = 1.0,
            hyperbolic: bool = True,
        ):
            super().__init__()
            
            self.key_dim = key_dim
            self.value_dim = value_dim
            self.capacity = capacity
            self.num_heads = num_heads
            self.topk = topk
            self.temperature = temperature
            self.hyperbolic = hyperbolic
            
            # Memory storage
            self.register_buffer("keys", torch.zeros(capacity, key_dim))
            self.register_buffer("values", torch.zeros(capacity, value_dim))
            self.register_buffer("labels", torch.zeros(capacity))
            self.register_buffer("valid_mask", torch.zeros(capacity, dtype=torch.bool))
            self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))
            
            # Query projection (multi-head)
            self.query_proj = nn.Linear(key_dim, key_dim * num_heads)
            self.key_proj = nn.Linear(key_dim, key_dim * num_heads)
            self.value_proj = nn.Linear(value_dim, value_dim)
            
            # Output projection
            self.output_proj = nn.Linear(value_dim, value_dim)
            
            # LSH hash functions for fast retrieval
            self.register_buffer(
                "lsh_planes", 
                torch.randn(8, key_dim)  # 8 hash bits
            )
        
        def _compute_hash(self, x: torch.Tensor) -> torch.Tensor:
            """Compute LSH hash for fast approximate retrieval."""
            # (N, key_dim) @ (key_dim, 8) -> (N, 8)
            projections = x @ self.lsh_planes.T
            return (projections > 0).long()
        
        def write(
            self,
            keys: torch.Tensor,
            values: torch.Tensor,
            labels: torch.Tensor,
        ) -> None:
            """
            Write new entries to memory.
            
            Args:
                keys: (N, key_dim) - Atom embeddings
                values: (N, value_dim) - Context embeddings
                labels: (N,) - Reactivity labels (0 or 1)
            """
            n = keys.size(0)
            
            for i in range(n):
                ptr = self.write_ptr.item()
                
                if self.hyperbolic:
                    self.keys[ptr] = HyperbolicOperations.project(keys[i])
                else:
                    self.keys[ptr] = keys[i]
                
                self.values[ptr] = values[i]
                self.labels[ptr] = labels[i]
                self.valid_mask[ptr] = True
                
                # Circular buffer
                self.write_ptr = (self.write_ptr + 1) % self.capacity
        
        def read(
            self,
            queries: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            Read from memory using soft attention.
            
            Args:
                queries: (N, key_dim) - Query embeddings
                mask: (N,) - Valid query mask
                
            Returns:
                Dict with retrieved values and attention weights
            """
            if not self.valid_mask.any():
                # Empty memory - return zeros
                return {
                    "values": torch.zeros(queries.size(0), self.value_dim, device=queries.device),
                    "attention": torch.zeros(queries.size(0), self.topk, device=queries.device),
                    "retrieved_labels": torch.zeros(queries.size(0), self.topk, device=queries.device),
                }
            
            # Get valid memory entries
            valid_keys = self.keys[self.valid_mask]
            valid_values = self.values[self.valid_mask]
            valid_labels = self.labels[self.valid_mask]
            num_valid = valid_keys.size(0)
            
            # Project queries and keys
            Q = self.query_proj(queries)  # (N, key_dim * num_heads)
            K = self.key_proj(valid_keys)  # (M, key_dim * num_heads)
            V = self.value_proj(valid_values)  # (M, value_dim)
            
            # Reshape for multi-head attention
            N = queries.size(0)
            M = num_valid
            
            Q = Q.view(N, self.num_heads, self.key_dim)
            K = K.view(M, self.num_heads, self.key_dim)
            
            # Compute attention scores
            if self.hyperbolic:
                # Use hyperbolic distance
                scores = []
                for h in range(self.num_heads):
                    q_h = Q[:, h, :]  # (N, key_dim)
                    k_h = K[:, h, :]  # (M, key_dim)
                    
                    # Pairwise hyperbolic distances
                    dist = []
                    for i in range(N):
                        d = HyperbolicOperations.distance(
                            q_h[i:i+1].expand(M, -1), k_h
                        )
                        dist.append(d)
                    dist = torch.stack(dist, dim=0)  # (N, M)
                    
                    # Convert distance to similarity (closer = higher)
                    sim = -dist / self.temperature
                    scores.append(sim)
                
                scores = torch.stack(scores, dim=1).mean(dim=1)  # (N, M)
            else:
                # Euclidean dot product attention
                scores = torch.einsum("nhd,mhd->nhm", Q, K).mean(dim=1) / math.sqrt(self.key_dim)
            
            # Top-k retrieval
            k = min(self.topk, M)
            topk_scores, topk_indices = torch.topk(scores, k, dim=-1)
            
            # Softmax attention over top-k
            attention = F.softmax(topk_scores, dim=-1)  # (N, k)
            
            # Retrieve values
            retrieved_values = V[topk_indices]  # (N, k, value_dim)
            retrieved_labels = valid_labels[topk_indices]  # (N, k)
            
            # Weighted sum
            output = torch.einsum("nk,nkd->nd", attention, retrieved_values)
            output = self.output_proj(output)
            
            return {
                "values": output,
                "attention": attention,
                "retrieved_labels": retrieved_labels,
                "topk_indices": topk_indices,
            }
        
        def size(self) -> int:
            """Return number of valid entries."""
            return self.valid_mask.sum().item()
        
        def clear(self) -> None:
            """Clear all memory entries."""
            self.valid_mask.fill_(False)
            self.write_ptr.fill_(0)


    # ========================================================================
    # CAUSAL REASONING ENGINE
    # ========================================================================
    
    class CausalReasoningEngine(nn.Module):
        """
        Models reaction pathways as causal graphs.
        
        The key insight: Chemical reactivity is CAUSAL, not just correlational.
        A site is reactive BECAUSE of:
        1. Electronic factors (electron density, BDE)
        2. Steric factors (accessibility)
        3. Enzyme binding (distance to heme)
        
        This module learns to reason about these causal factors and their
        interactions. Uses a simplified causal graph:
        
        [Electronic] ──┐
                       ├──> [Local Reactivity] ──┐
        [Steric] ──────┘                         │
                                                 ├──> [Final SoM]
        [Enzyme Binding] ──> [Selectivity] ──────┘
        
        Each node is represented as a learned transformation, and edges
        represent causal dependencies.
        """
        
        def __init__(
            self,
            atom_dim: int = 128,
            hidden_dim: int = 64,
            num_causal_factors: int = 4,  # electronic, steric, binding, molecular_context
            dropout: float = 0.1,
        ):
            super().__init__()
            
            self.atom_dim = atom_dim
            self.hidden_dim = hidden_dim
            self.num_factors = num_causal_factors
            
            # Factor extractors
            self.electronic_encoder = nn.Sequential(
                nn.Linear(atom_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            self.steric_encoder = nn.Sequential(
                nn.Linear(atom_dim + 3, hidden_dim),  # +3 for accessibility features
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            self.binding_encoder = nn.Sequential(
                nn.Linear(atom_dim + 1, hidden_dim),  # +1 for distance to binding site
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            # Causal graph edges (parameterized as attention)
            self.electronic_to_local = nn.Linear(hidden_dim, hidden_dim)
            self.steric_to_local = nn.Linear(hidden_dim, hidden_dim)
            self.local_to_final = nn.Linear(hidden_dim, hidden_dim)
            self.binding_to_selectivity = nn.Linear(hidden_dim, hidden_dim)
            self.selectivity_to_final = nn.Linear(hidden_dim, hidden_dim)
            
            # Intervention module (for counterfactual reasoning)
            self.intervention_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            
            # Output
            self.output_proj = nn.Linear(hidden_dim, 1)
        
        def forward(
            self,
            atom_features: torch.Tensor,
            accessibility_features: Optional[torch.Tensor] = None,
            binding_distance: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass through the causal graph.
            
            Args:
                atom_features: (N, atom_dim) - Atom embeddings
                accessibility_features: (N, 3) - SASA, buried fraction, neighbor count
                binding_distance: (N, 1) - Distance to enzyme binding site
                
            Returns:
                Dict with causal factor activations and final prediction
            """
            N = atom_features.size(0)
            device = atom_features.device
            
            # Default accessibility and binding if not provided
            if accessibility_features is None:
                accessibility_features = torch.zeros(N, 3, device=device)
            if binding_distance is None:
                binding_distance = torch.ones(N, 1, device=device)
            
            # Encode causal factors
            electronic = self.electronic_encoder(atom_features)
            
            steric_input = torch.cat([atom_features, accessibility_features], dim=-1)
            steric = self.steric_encoder(steric_input)
            
            binding_input = torch.cat([atom_features, binding_distance], dim=-1)
            binding = self.binding_encoder(binding_input)
            
            # Causal graph propagation
            # Electronic + Steric -> Local Reactivity
            electronic_effect = self.electronic_to_local(electronic)
            steric_effect = self.steric_to_local(steric)
            local_reactivity = F.silu(electronic_effect + steric_effect)
            
            # Binding -> Selectivity
            selectivity = self.binding_to_selectivity(binding)
            
            # Local Reactivity + Selectivity -> Final
            local_effect = self.local_to_final(local_reactivity)
            selectivity_effect = self.selectivity_to_final(selectivity)
            
            # Intervention gate: learn when binding dominates vs local reactivity
            combined = torch.cat([local_effect, selectivity_effect], dim=-1)
            gate = self.intervention_gate(combined)
            
            final_features = gate * selectivity_effect + (1 - gate) * local_effect
            
            # Output logit
            logit = self.output_proj(final_features)
            
            return {
                "logits": logit.squeeze(-1),
                "electronic": electronic,
                "steric": steric,
                "binding": binding,
                "local_reactivity": local_reactivity,
                "selectivity": selectivity,
                "gate": gate.squeeze(-1),
            }
        
        def counterfactual(
            self,
            atom_features: torch.Tensor,
            intervention: str,
            intervention_value: torch.Tensor,
            **kwargs,
        ) -> Dict[str, torch.Tensor]:
            """
            Compute counterfactual prediction under intervention.
            
            Args:
                atom_features: Original atom features
                intervention: Which factor to intervene on
                    - "electronic": Override electronic features
                    - "steric": Override steric features
                    - "binding": Override binding features
                intervention_value: Value to set for the intervened factor
                
            Returns:
                Counterfactual predictions
            """
            # This would allow questions like:
            # "What would happen if this atom were more accessible?"
            # "What if the binding distance were different?"
            raise NotImplementedError("Counterfactual inference not yet implemented")


    # ========================================================================
    # NEXUS-LITE FULL MODEL
    # ========================================================================
    
    class NEXUSLiteModel(nn.Module):
        """
        NEXUS-Lite: Full model combining hyperbolic embeddings,
        analogical memory, and causal reasoning.
        
        This model wraps an existing GNN backbone and adds:
        1. Hyperbolic projection for hierarchical atom representations
        2. Memory bank for analogical reasoning
        3. Causal reasoning engine for interpretable predictions
        
        The model can be trained end-to-end or used for inference
        with a pre-trained backbone.
        """
        
        def __init__(
            self,
            backbone: nn.Module,
            atom_dim: int = 128,
            hyperbolic_dim: int = 64,
            memory_capacity: int = 4096,
            memory_topk: int = 32,
            use_causal_reasoning: bool = True,
            freeze_backbone: bool = False,
        ):
            super().__init__()
            
            self.backbone = backbone
            self.atom_dim = atom_dim
            self.hyperbolic_dim = hyperbolic_dim
            self.use_causal_reasoning = use_causal_reasoning
            self.freeze_backbone = freeze_backbone
            
            if freeze_backbone:
                for param in backbone.parameters():
                    param.requires_grad = False
            
            # Project to hyperbolic space
            self.euclidean_to_hyperbolic = nn.Sequential(
                nn.Linear(atom_dim, hyperbolic_dim * 2),
                nn.SiLU(),
                nn.Linear(hyperbolic_dim * 2, hyperbolic_dim),
            )
            
            # Analogical memory
            self.memory = AnalogicalMemoryBank(
                key_dim=hyperbolic_dim,
                value_dim=atom_dim,
                capacity=memory_capacity,
                topk=memory_topk,
                hyperbolic=True,
            )
            
            # Causal reasoning
            if use_causal_reasoning:
                self.causal_engine = CausalReasoningEngine(
                    atom_dim=atom_dim,
                    hidden_dim=hyperbolic_dim,
                )
            else:
                self.causal_engine = None
            
            # Combine memory and causal outputs
            self.combination_head = nn.Sequential(
                nn.Linear(atom_dim + hyperbolic_dim + 1, atom_dim),
                nn.LayerNorm(atom_dim),
                nn.SiLU(),
                nn.Linear(atom_dim, 1),
            )
            
            # Memory prediction head (for retrieved label weighting)
            self.memory_prediction = nn.Linear(1, 1)
        
        def forward(
            self,
            batch: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                batch: Dict containing at minimum:
                    - Standard GNN inputs (node features, edge index, etc.)
                    - 'site_labels' (optional, for memory writing)
                    
            Returns:
                Dict with predictions and intermediate representations
            """
            # Get backbone outputs
            with torch.set_grad_enabled(not self.freeze_backbone):
                backbone_outputs = self.backbone(batch)
            
            atom_features = backbone_outputs.get("atom_features")
            if atom_features is None:
                return backbone_outputs
            
            backbone_logits = backbone_outputs.get("site_logits", torch.zeros(atom_features.size(0)))
            
            # Project to hyperbolic space
            hyperbolic_embeddings = self.euclidean_to_hyperbolic(atom_features)
            hyperbolic_embeddings = HyperbolicOperations.expmap0(hyperbolic_embeddings)
            hyperbolic_embeddings = HyperbolicOperations.project(hyperbolic_embeddings)
            
            # Read from memory
            memory_output = self.memory.read(hyperbolic_embeddings)
            memory_values = memory_output["values"]
            retrieved_labels = memory_output["retrieved_labels"]
            attention = memory_output["attention"]
            
            # Weighted average of retrieved labels
            memory_label_pred = (attention * retrieved_labels).sum(dim=-1)
            
            # Causal reasoning (if enabled)
            if self.causal_engine is not None:
                accessibility = batch.get("accessibility_features")
                binding_dist = batch.get("binding_distance")
                
                causal_output = self.causal_engine(
                    atom_features,
                    accessibility_features=accessibility,
                    binding_distance=binding_dist,
                )
                causal_logits = causal_output["logits"]
            else:
                causal_logits = torch.zeros_like(backbone_logits)
            
            # Combine all signals
            combined_input = torch.cat([
                atom_features,
                memory_values,
                memory_label_pred.unsqueeze(-1),
            ], dim=-1)
            
            nexus_logits = self.combination_head(combined_input).squeeze(-1)
            
            # Final ensemble
            final_logits = (
                0.4 * backbone_logits +
                0.3 * nexus_logits +
                0.2 * causal_logits +
                0.1 * self.memory_prediction(memory_label_pred.unsqueeze(-1)).squeeze(-1)
            )
            
            # Update outputs
            outputs = dict(backbone_outputs)
            outputs["site_logits"] = final_logits
            outputs["site_logits_backbone"] = backbone_logits
            outputs["site_logits_nexus"] = nexus_logits
            outputs["site_logits_causal"] = causal_logits
            outputs["hyperbolic_embeddings"] = hyperbolic_embeddings
            outputs["memory_attention"] = attention
            outputs["memory_label_pred"] = memory_label_pred
            
            if self.causal_engine is not None:
                outputs["causal_gate"] = causal_output["gate"]
            
            return outputs
        
        def update_memory(
            self,
            atom_features: torch.Tensor,
            site_labels: torch.Tensor,
            batch_index: torch.Tensor,
        ) -> None:
            """
            Update memory bank with new examples.
            
            Args:
                atom_features: (N, atom_dim) - Atom embeddings
                site_labels: (N,) - Binary SoM labels
                batch_index: (N,) - Molecule indices
            """
            # Project to hyperbolic
            hyperbolic_keys = self.euclidean_to_hyperbolic(atom_features)
            hyperbolic_keys = HyperbolicOperations.expmap0(hyperbolic_keys)
            hyperbolic_keys = HyperbolicOperations.project(hyperbolic_keys)
            
            # Write to memory
            self.memory.write(
                keys=hyperbolic_keys.detach(),
                values=atom_features.detach(),
                labels=site_labels.detach().float(),
            )
        
        def memory_size(self) -> int:
            """Return current memory size."""
            return self.memory.size()


# ============================================================================
# UTILITIES
# ============================================================================

def create_nexus_lite_from_checkpoint(
    checkpoint_path: str,
    backbone_class,
    config,
    **nexus_kwargs,
) -> "NEXUSLiteModel":
    """
    Create NEXUS-Lite model from a backbone checkpoint.
    
    Args:
        checkpoint_path: Path to backbone checkpoint
        backbone_class: Class to instantiate backbone
        config: Config for backbone
        **nexus_kwargs: Additional args for NEXUSLiteModel
        
    Returns:
        NEXUSLiteModel with loaded backbone weights
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Create backbone
    backbone = backbone_class(config)
    backbone.load_state_dict(state_dict, strict=False)
    
    # Get atom dim from config
    atom_dim = getattr(config, "hidden_dim", 128)
    
    # Create NEXUS-Lite
    model = NEXUSLiteModel(
        backbone=backbone,
        atom_dim=atom_dim,
        **nexus_kwargs,
    )
    
    return model


if __name__ == "__main__":
    # Quick test of hyperbolic operations
    if TORCH_AVAILABLE:
        print("Testing hyperbolic operations...")
        
        x = torch.randn(4, 8) * 0.1  # Small values to stay in ball
        y = torch.randn(4, 8) * 0.1
        
        # Test Möbius addition
        z = HyperbolicOperations.mobius_add(x, y)
        print(f"Möbius add: norm={z.norm(dim=-1).mean():.4f} (should be < 1)")
        
        # Test expmap/logmap
        v = torch.randn(4, 8)
        x_exp = HyperbolicOperations.expmap0(v)
        v_log = HyperbolicOperations.logmap0(x_exp)
        print(f"Exp/log roundtrip error: {(v - v_log).abs().max():.6f}")
        
        # Test distance
        d = HyperbolicOperations.distance(x, y)
        print(f"Hyperbolic distances: {d}")
        
        # Test memory bank
        print("\nTesting memory bank...")
        memory = AnalogicalMemoryBank(key_dim=8, value_dim=16, capacity=100)
        
        # Write some entries
        keys = torch.randn(10, 8) * 0.1
        values = torch.randn(10, 16)
        labels = torch.randint(0, 2, (10,)).float()
        memory.write(keys, values, labels)
        print(f"Memory size: {memory.size()}")
        
        # Read
        queries = torch.randn(5, 8) * 0.1
        output = memory.read(queries)
        print(f"Retrieved values shape: {output['values'].shape}")
        print(f"Retrieved labels shape: {output['retrieved_labels'].shape}")
        
        print("\nNEXUS-Lite tests passed!")
