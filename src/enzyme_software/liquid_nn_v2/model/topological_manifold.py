"""
Topological Reaction Manifold for Site-of-Metabolism Prediction.

This module implements topological data analysis (TDA) concepts for identifying
reaction sites. The key insight: reactive sites often correspond to topological
features in the molecular structure that persist across multiple scales.

Core Concepts:

1. **Persistent Homology**: Track topological features (connected components,
   loops, voids) across a filtration of distance thresholds. Features that
   persist across many scales are "important" - they correspond to structural
   motifs that define reactivity.

2. **Reaction Manifold**: Embed molecules in a space where similar reaction
   sites are nearby. This is learned from reaction data and captures the
   "shape" of chemical reactivity.

3. **Topological Signatures**: Each atom gets a signature based on:
   - Local persistent homology (what structures exist nearby)
   - Global topological context (where is this atom in the overall shape)
   - Reaction site correlation (how do similar topological features react)

4. **Metric Learning**: Learn a distance metric that captures chemical
   similarity for reactivity prediction. Atoms with similar topological
   signatures tend to have similar reactivity.

Mathematical Framework:
- Vietoris-Rips filtration on molecular graphs
- Persistence diagrams and barcodes for feature extraction
- Wasserstein distance for comparing topological signatures
- Topological attention mechanism for message passing
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# ============================================================================
# PERSISTENT HOMOLOGY COMPUTATION (Simplified Implementation)
# ============================================================================

@dataclass
class PersistenceBar:
    """A single bar in a persistence barcode."""
    birth: float
    death: float
    dimension: int  # 0 = connected component, 1 = loop, 2 = void
    generator: Optional[Set[int]] = None  # Atoms that generate this feature
    
    @property
    def persistence(self) -> float:
        """Lifetime of this feature."""
        return self.death - self.birth
    
    @property
    def midlife(self) -> float:
        """Midpoint of this feature's existence."""
        return (self.birth + self.death) / 2.0


@dataclass
class PersistenceDiagram:
    """Collection of persistence bars for a molecule."""
    bars: List[PersistenceBar] = field(default_factory=list)
    
    def filter_by_dimension(self, dim: int) -> List[PersistenceBar]:
        """Get all bars of a specific dimension."""
        return [b for b in self.bars if b.dimension == dim]
    
    def filter_by_persistence(self, min_persistence: float) -> List[PersistenceBar]:
        """Get all bars with persistence above threshold."""
        return [b for b in self.bars if b.persistence >= min_persistence]
    
    def total_persistence(self, dim: Optional[int] = None) -> float:
        """Sum of all bar persistences."""
        bars = self.bars if dim is None else self.filter_by_dimension(dim)
        return sum(b.persistence for b in bars)


class VietorisRipsComplex:
    """
    Compute Vietoris-Rips persistence on molecular graphs.
    
    This is a simplified implementation that:
    1. Uses graph distance (number of bonds) as the filtration parameter
    2. Tracks connected components (H0) and loops (H1)
    3. Assigns topological features to atoms based on generators
    
    For a full implementation, you would use libraries like GUDHI or Ripser.
    """
    
    def __init__(self, max_dimension: int = 1, max_filtration: float = 10.0):
        self.max_dimension = max_dimension
        self.max_filtration = max_filtration
    
    def _compute_distance_matrix(self, mol) -> np.ndarray:
        """Compute shortest path distances between all atoms."""
        n = mol.GetNumAtoms()
        dist = np.full((n, n), float('inf'))
        np.fill_diagonal(dist, 0)
        
        # Initialize with bond distances
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            dist[i, j] = dist[j, i] = 1
        
        # Floyd-Warshall for shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        return dist
    
    def _compute_h0_persistence(
        self,
        dist_matrix: np.ndarray,
    ) -> Tuple[List[PersistenceBar], Dict[int, List[PersistenceBar]]]:
        """
        Compute H0 (connected component) persistence.
        
        Returns:
            - List of persistence bars
            - Dict mapping atom indices to bars they generate
        """
        n = dist_matrix.shape[0]
        
        # Union-Find for tracking connected components
        parent = list(range(n))
        rank = [0] * n
        birth = [0.0] * n  # Birth time for each component
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y, merge_time):
            px, py = find(x), find(y)
            if px == py:
                return None
            
            # Merge smaller into larger
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            
            # Component py dies, px survives
            return PersistenceBar(
                birth=birth[py],
                death=merge_time,
                dimension=0,
                generator={py},
            )
        
        bars = []
        atom_bars = defaultdict(list)
        
        # Get unique distances sorted
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if dist_matrix[i, j] < float('inf'):
                    edges.append((dist_matrix[i, j], i, j))
        edges.sort()
        
        # Process edges in order of filtration
        for dist, i, j in edges:
            if dist > self.max_filtration:
                break
            bar = union(i, j, dist)
            if bar is not None:
                bars.append(bar)
                for atom_idx in bar.generator:
                    atom_bars[atom_idx].append(bar)
        
        # Add infinite bars for surviving components
        components = set()
        for i in range(n):
            root = find(i)
            if root not in components:
                components.add(root)
                bar = PersistenceBar(
                    birth=0.0,
                    death=float('inf'),
                    dimension=0,
                    generator={root},
                )
                bars.append(bar)
                atom_bars[root].append(bar)
        
        return bars, atom_bars
    
    def _compute_h1_persistence(
        self,
        mol,
        dist_matrix: np.ndarray,
    ) -> Tuple[List[PersistenceBar], Dict[int, List[PersistenceBar]]]:
        """
        Compute H1 (loop/cycle) persistence.
        
        Uses ring detection as a proxy for true H1 computation.
        """
        bars = []
        atom_bars = defaultdict(list)
        
        if not RDKIT_AVAILABLE:
            return bars, atom_bars
        
        # Get ring info from RDKit
        ring_info = mol.GetRingInfo()
        
        for ring_atoms in ring_info.AtomRings():
            ring_set = set(ring_atoms)
            
            # Birth time: when all atoms in ring are connected
            max_dist = 0
            for i in ring_atoms:
                for j in ring_atoms:
                    if i < j:
                        max_dist = max(max_dist, dist_matrix[i, j])
            
            # Death time: infinity (rings persist)
            # In true persistence, loops die when filled
            bar = PersistenceBar(
                birth=max_dist,
                death=float('inf'),
                dimension=1,
                generator=ring_set,
            )
            bars.append(bar)
            
            for atom_idx in ring_atoms:
                atom_bars[atom_idx].append(bar)
        
        return bars, atom_bars
    
    def compute(self, mol) -> Tuple[PersistenceDiagram, Dict[int, List[PersistenceBar]]]:
        """
        Compute full persistence diagram for a molecule.
        
        Returns:
            - PersistenceDiagram with all bars
            - Dict mapping atom indices to associated bars
        """
        dist_matrix = self._compute_distance_matrix(mol)
        
        h0_bars, h0_atom_bars = self._compute_h0_persistence(dist_matrix)
        
        all_bars = h0_bars
        all_atom_bars = dict(h0_atom_bars)
        
        if self.max_dimension >= 1:
            h1_bars, h1_atom_bars = self._compute_h1_persistence(mol, dist_matrix)
            all_bars.extend(h1_bars)
            for atom_idx, bars in h1_atom_bars.items():
                if atom_idx in all_atom_bars:
                    all_atom_bars[atom_idx].extend(bars)
                else:
                    all_atom_bars[atom_idx] = bars
        
        return PersistenceDiagram(bars=all_bars), all_atom_bars


def compute_topological_features(
    smiles: str,
    include_3d: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute topological features for each atom in a molecule.
    
    Features include:
    - Local H0 persistence (how "connected" is this atom locally)
    - Local H1 persistence (is this atom part of important rings)
    - Total persistence (global topological complexity near this atom)
    - Persistence entropy (diversity of topological features)
    
    Args:
        smiles: Molecule SMILES
        include_3d: Whether to include 3D-based features
        
    Returns:
        Dict with feature arrays for each atom
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit required")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    
    # Compute persistence
    vr = VietorisRipsComplex(max_dimension=1)
    diagram, atom_bars = vr.compute(mol)
    
    # Initialize feature arrays
    features = {
        "h0_total_persistence": np.zeros(n),
        "h0_max_persistence": np.zeros(n),
        "h0_num_features": np.zeros(n),
        "h1_total_persistence": np.zeros(n),
        "h1_max_persistence": np.zeros(n),
        "h1_num_features": np.zeros(n),
        "persistence_entropy": np.zeros(n),
        "local_complexity": np.zeros(n),
        "is_ring_atom": np.zeros(n),
        "ring_size_min": np.zeros(n),
        "ring_size_max": np.zeros(n),
    }
    
    # Get ring info
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    
    for atom_idx in range(n):
        # Get bars associated with this atom
        bars = atom_bars.get(atom_idx, [])
        
        # H0 features
        h0_bars = [b for b in bars if b.dimension == 0 and b.persistence < float('inf')]
        if h0_bars:
            features["h0_total_persistence"][atom_idx] = sum(b.persistence for b in h0_bars)
            features["h0_max_persistence"][atom_idx] = max(b.persistence for b in h0_bars)
            features["h0_num_features"][atom_idx] = len(h0_bars)
        
        # H1 features
        h1_bars = [b for b in bars if b.dimension == 1]
        if h1_bars:
            persistences = [b.persistence if b.persistence < float('inf') else 10.0 for b in h1_bars]
            features["h1_total_persistence"][atom_idx] = sum(persistences)
            features["h1_max_persistence"][atom_idx] = max(persistences)
            features["h1_num_features"][atom_idx] = len(h1_bars)
        
        # Persistence entropy
        all_persistences = [
            b.persistence if b.persistence < float('inf') else 10.0 
            for b in bars
        ]
        if all_persistences:
            total = sum(all_persistences)
            if total > 0:
                probs = [p/total for p in all_persistences]
                entropy = -sum(p * math.log(p + 1e-10) for p in probs)
                features["persistence_entropy"][atom_idx] = entropy
        
        # Local complexity: combination of H0 and H1
        features["local_complexity"][atom_idx] = (
            features["h0_total_persistence"][atom_idx] + 
            2.0 * features["h1_total_persistence"][atom_idx]
        )
        
        # Ring features
        atom_ring_sizes = [len(ring) for ring in atom_rings if atom_idx in ring]
        if atom_ring_sizes:
            features["is_ring_atom"][atom_idx] = 1.0
            features["ring_size_min"][atom_idx] = min(atom_ring_sizes)
            features["ring_size_max"][atom_idx] = max(atom_ring_sizes)
    
    # Normalize features
    for key in ["h0_total_persistence", "h1_total_persistence", "local_complexity"]:
        max_val = features[key].max()
        if max_val > 0:
            features[key] = features[key] / max_val
    
    return features


# ============================================================================
# TOPOLOGICAL NEURAL NETWORK COMPONENTS
# ============================================================================

if TORCH_AVAILABLE:
    
    class PersistenceImageEncoder(nn.Module):
        """
        Encode persistence diagrams as fixed-size vectors.
        
        Uses a differentiable "persistence image" representation where
        each point in the persistence diagram contributes to a Gaussian
        kernel centered at its (birth, death) coordinates.
        """
        
        def __init__(
            self,
            resolution: int = 16,
            sigma: float = 0.5,
            max_birth: float = 5.0,
            max_death: float = 10.0,
        ):
            super().__init__()
            
            self.resolution = resolution
            self.sigma = sigma
            
            # Create grid for persistence image
            birth_range = torch.linspace(0, max_birth, resolution)
            death_range = torch.linspace(0, max_death, resolution)
            
            self.register_buffer("birth_grid", birth_range.view(1, -1, 1))
            self.register_buffer("death_grid", death_range.view(1, 1, -1))
        
        def forward(
            self,
            births: torch.Tensor,
            deaths: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Convert persistence points to persistence image.
            
            Args:
                births: (B, max_bars) birth times
                deaths: (B, max_bars) death times
                mask: (B, max_bars) validity mask
                
            Returns:
                (B, resolution, resolution) persistence image
            """
            B = births.size(0)
            
            # Expand for broadcasting
            # births: (B, max_bars, 1, 1)
            births = births.unsqueeze(-1).unsqueeze(-1)
            deaths = deaths.unsqueeze(-1).unsqueeze(-1)
            
            # Compute Gaussian contribution
            # grid: (1, resolution, 1) and (1, 1, resolution)
            birth_contrib = torch.exp(-0.5 * ((births - self.birth_grid) / self.sigma) ** 2)
            death_contrib = torch.exp(-0.5 * ((deaths - self.death_grid) / self.sigma) ** 2)
            
            # Combined contribution: (B, max_bars, resolution, resolution)
            contrib = birth_contrib * death_contrib
            
            # Weight by persistence (death - birth)
            persistence = (deaths - births).squeeze(-1).squeeze(-1)
            weighted = contrib * persistence.unsqueeze(-1).unsqueeze(-1)
            
            # Apply mask if provided
            if mask is not None:
                weighted = weighted * mask.unsqueeze(-1).unsqueeze(-1)
            
            # Sum over bars
            image = weighted.sum(dim=1)  # (B, resolution, resolution)
            
            return image
    
    
    class TopologicalAttention(nn.Module):
        """
        Attention mechanism weighted by topological features.
        
        The key idea: atoms with similar topological contexts should attend
        more strongly to each other. This captures long-range dependencies
        based on structural similarity rather than just distance.
        """
        
        def __init__(
            self,
            atom_dim: int = 128,
            topo_dim: int = 11,  # Number of topological features
            num_heads: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()
            
            self.atom_dim = atom_dim
            self.topo_dim = topo_dim
            self.num_heads = num_heads
            
            # Project topology to attention bias
            self.topo_query = nn.Linear(topo_dim, atom_dim)
            self.topo_key = nn.Linear(topo_dim, atom_dim)
            
            # Standard attention components
            self.query = nn.Linear(atom_dim, atom_dim)
            self.key = nn.Linear(atom_dim, atom_dim)
            self.value = nn.Linear(atom_dim, atom_dim)
            
            # Output projection
            self.output = nn.Linear(atom_dim, atom_dim)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(atom_dim // num_heads)
        
        def forward(
            self,
            atom_features: torch.Tensor,      # (N, atom_dim)
            topo_features: torch.Tensor,      # (N, topo_dim)
            batch_index: Optional[torch.Tensor] = None,  # (N,)
        ) -> torch.Tensor:
            """
            Apply topological attention.
            
            Args:
                atom_features: Node features
                topo_features: Topological features per node
                batch_index: Molecule indices for batching
                
            Returns:
                Updated atom features
            """
            N = atom_features.size(0)
            
            # Compute queries, keys, values
            Q = self.query(atom_features)
            K = self.key(atom_features)
            V = self.value(atom_features)
            
            # Topological bias
            topo_Q = self.topo_query(topo_features)
            topo_K = self.topo_key(topo_features)
            
            # Combine semantic and topological attention
            # Attention = softmax((Q + topo_Q) @ (K + topo_K)^T / sqrt(d))
            Q_combined = Q + 0.3 * topo_Q
            K_combined = K + 0.3 * topo_K
            
            # Compute attention scores
            # For simplicity, using dense attention (should use sparse for large molecules)
            attn_scores = torch.matmul(Q_combined, K_combined.transpose(-2, -1)) / self.scale
            
            # Mask attention to within-molecule if batch_index provided
            if batch_index is not None:
                # Create mask: same molecule = 1, different molecule = -inf
                mask = batch_index.unsqueeze(0) == batch_index.unsqueeze(1)
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention
            attended = torch.matmul(attn_weights, V)
            
            # Output projection
            output = self.output(attended)
            
            return output
    
    
    class TopologicalGNN(nn.Module):
        """
        Graph neural network augmented with topological features.
        
        This model uses topological information at multiple levels:
        1. Node initialization includes topological features
        2. Message passing is weighted by topological similarity
        3. Readout considers global topological context
        """
        
        def __init__(
            self,
            atom_input_dim: int = 64,
            topo_dim: int = 11,
            hidden_dim: int = 128,
            output_dim: int = 1,
            num_layers: int = 3,
            num_heads: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()
            
            self.hidden_dim = hidden_dim
            
            # Initial embedding
            self.atom_embed = nn.Linear(atom_input_dim, hidden_dim)
            self.topo_embed = nn.Linear(topo_dim, hidden_dim // 4)
            
            # Topological attention layers
            self.layers = nn.ModuleList([
                TopologicalAttention(
                    atom_dim=hidden_dim,
                    topo_dim=topo_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
            
            # Layer norms
            self.norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ])
            
            # Feed-forward layers
            self.ffn = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_layers)
            ])
            
            # Output head
            self.output = nn.Sequential(
                nn.Linear(hidden_dim + topo_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        
        def forward(
            self,
            atom_features: torch.Tensor,     # (N, atom_input_dim)
            topo_features: torch.Tensor,     # (N, topo_dim)
            batch_index: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass.
            
            Returns:
                Dict with logits and intermediate representations
            """
            # Initial embedding
            h = self.atom_embed(atom_features)
            topo_h = self.topo_embed(topo_features)
            
            # Concatenate initial topology embedding
            h = h + F.pad(topo_h, (0, h.size(-1) - topo_h.size(-1)))
            
            # Topological attention layers
            for attn, norm, ffn in zip(self.layers, self.norms, self.ffn):
                # Attention with residual
                h_attn = attn(h, topo_features, batch_index)
                h = norm(h + h_attn)
                
                # FFN with residual
                h = norm(h + ffn(h))
            
            # Output with topology features
            output_input = torch.cat([h, topo_features], dim=-1)
            logits = self.output(output_input)
            
            return {
                "logits": logits.squeeze(-1),
                "atom_features": h,
                "topo_features": topo_features,
            }
    
    
    class TopologicalReactionManifold(nn.Module):
        """
        Learn a manifold embedding where similar reaction sites are close.
        
        This implements the "reaction manifold" concept:
        - Each atom is embedded in a learned manifold space
        - The manifold is trained so that atoms with similar reactivity
          are nearby in the embedding space
        - New predictions are made by finding nearest neighbors in the manifold
        """
        
        def __init__(
            self,
            input_dim: int = 128,
            manifold_dim: int = 32,
            num_prototypes: int = 64,
            temperature: float = 0.1,
        ):
            super().__init__()
            
            self.manifold_dim = manifold_dim
            self.num_prototypes = num_prototypes
            self.temperature = temperature
            
            # Encoder to manifold
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, manifold_dim * 2),
                nn.SiLU(),
                nn.Linear(manifold_dim * 2, manifold_dim),
            )
            
            # Learnable prototypes (representative reaction sites)
            self.prototypes = nn.Parameter(torch.randn(num_prototypes, manifold_dim))
            self.prototype_labels = nn.Parameter(torch.zeros(num_prototypes))
            
            # Prototype attention
            self.prototype_query = nn.Linear(manifold_dim, manifold_dim)
            
        def forward(
            self,
            atom_features: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """
            Embed atoms in reaction manifold and predict reactivity.
            
            Args:
                atom_features: (N, input_dim) atom representations
                
            Returns:
                Dict with manifold embeddings and predictions
            """
            # Encode to manifold
            embeddings = self.encoder(atom_features)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            # Normalize prototypes
            prototypes = F.normalize(self.prototypes, p=2, dim=-1)
            
            # Compute similarity to prototypes
            similarities = torch.matmul(embeddings, prototypes.T) / self.temperature
            
            # Soft assignment to prototypes
            assignments = F.softmax(similarities, dim=-1)
            
            # Weighted prediction based on prototype labels
            prototype_scores = torch.sigmoid(self.prototype_labels)
            predictions = torch.matmul(assignments, prototype_scores)
            
            return {
                "manifold_embedding": embeddings,
                "prototype_similarities": similarities,
                "prototype_assignments": assignments,
                "predictions": predictions,
            }
        
        def update_prototypes(
            self,
            atom_features: torch.Tensor,
            labels: torch.Tensor,
            learning_rate: float = 0.1,
        ) -> None:
            """
            Update prototypes with new examples (online learning).
            
            This allows the manifold to adapt to new reaction data.
            """
            with torch.no_grad():
                embeddings = self.encoder(atom_features)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                
                # Find closest prototype for each point
                prototypes = F.normalize(self.prototypes, p=2, dim=-1)
                similarities = torch.matmul(embeddings, prototypes.T)
                closest = similarities.argmax(dim=-1)
                
                # Update prototypes toward their assigned points
                for i in range(embeddings.size(0)):
                    proto_idx = closest[i].item()
                    direction = embeddings[i] - prototypes[proto_idx]
                    self.prototypes.data[proto_idx] += learning_rate * direction
                    
                    # Update label with exponential moving average
                    self.prototype_labels.data[proto_idx] = (
                        0.9 * self.prototype_labels.data[proto_idx] +
                        0.1 * labels[i].float()
                    )


def create_topological_som_predictor(
    backbone: nn.Module,
    atom_dim: int = 128,
    topo_dim: int = 11,
    manifold_dim: int = 32,
) -> nn.Module:
    """
    Create a complete SoM predictor with topological features.
    
    This wraps an existing backbone with:
    1. TopologicalGNN for topological message passing
    2. TopologicalReactionManifold for similarity-based prediction
    """
    
    class TopologicalSoMPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            
            self.topo_gnn = TopologicalGNN(
                atom_input_dim=atom_dim,
                topo_dim=topo_dim,
                hidden_dim=atom_dim,
            )
            
            self.manifold = TopologicalReactionManifold(
                input_dim=atom_dim,
                manifold_dim=manifold_dim,
            )
            
            # Fusion
            self.fusion = nn.Linear(atom_dim * 2, atom_dim)
            self.output = nn.Linear(atom_dim + 1, 1)
        
        def forward(
            self,
            batch: Dict[str, torch.Tensor],
            topo_features: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            # Backbone
            backbone_out = self.backbone(batch)
            atom_features = backbone_out.get("atom_features")
            backbone_logits = backbone_out.get("site_logits")
            
            if topo_features is None:
                # Use zeros if no topological features
                topo_features = torch.zeros(
                    atom_features.size(0), 11,
                    device=atom_features.device
                )
            
            # Topological GNN
            topo_out = self.topo_gnn(atom_features, topo_features)
            topo_features_updated = topo_out["atom_features"]
            
            # Manifold prediction
            manifold_out = self.manifold(topo_features_updated)
            manifold_pred = manifold_out["predictions"]
            
            # Fuse
            fused = self.fusion(torch.cat([atom_features, topo_features_updated], dim=-1))
            
            # Final prediction
            final_input = torch.cat([fused, manifold_pred.unsqueeze(-1)], dim=-1)
            final_logits = self.output(final_input).squeeze(-1)
            
            outputs = dict(backbone_out)
            outputs["site_logits"] = final_logits
            outputs["manifold_embedding"] = manifold_out["manifold_embedding"]
            outputs["manifold_pred"] = manifold_pred
            
            return outputs
    
    return TopologicalSoMPredictor()


if __name__ == "__main__":
    print("Testing Topological Reaction Manifold...")
    
    # Test topological feature computation
    test_smiles = "Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"  # Midazolam
    
    features = compute_topological_features(test_smiles)
    
    print(f"\nMidazolam topological features:")
    print(f"Num atoms (with H): {len(features['h0_total_persistence'])}")
    
    # Find atoms with highest topological complexity
    complexity = features["local_complexity"]
    top_complex = np.argsort(-complexity)[:5]
    
    print(f"\nTop 5 topologically complex atoms:")
    for idx in top_complex:
        print(f"  Atom {idx}: complexity={complexity[idx]:.3f}, "
              f"ring={features['is_ring_atom'][idx]:.0f}, "
              f"H1_persist={features['h1_total_persistence'][idx]:.3f}")
    
    if TORCH_AVAILABLE:
        print("\nTesting TopologicalGNN...")
        
        N = 10  # Number of atoms
        model = TopologicalGNN(
            atom_input_dim=64,
            topo_dim=11,
            hidden_dim=128,
        )
        
        atom_feat = torch.randn(N, 64)
        topo_feat = torch.randn(N, 11)
        
        out = model(atom_feat, topo_feat)
        print(f"Output logits shape: {out['logits'].shape}")
        print(f"Atom features shape: {out['atom_features'].shape}")
    
    print("\nTests passed!")
