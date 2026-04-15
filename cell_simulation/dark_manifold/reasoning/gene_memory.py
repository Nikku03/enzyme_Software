"""
dark_manifold/reasoning/gene_memory.py

Gene Memory Bank for similarity-based essentiality prediction.

Adapted from nexus/reasoning/hyperbolic_memory.py.

Key idea: Store known gene essentiality labels and retrieve similar genes
to predict unknown genes by analogy.

Two modes:
1. Euclidean: Simple cosine similarity (faster, good baseline)
2. Hyperbolic: Poincaré ball distance (better for hierarchical data)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RetrievalResult:
    """Result of memory retrieval."""
    gene: str
    distance: float
    essential: bool
    features: np.ndarray
    similarity: float = 0.0


class GeneMemoryBank:
    """
    Memory bank for gene essentiality prediction by analogy.
    
    Stores gene features and essentiality labels, then retrieves
    similar genes to predict new ones.
    
    Adapted from HyperbolicMemoryBank but simplified for genes.
    """
    
    def __init__(
        self,
        encoder=None,
        use_hyperbolic: bool = False,
        curvature: float = 1.0,
        device: str = "cpu",
    ):
        """
        Initialize memory bank.
        
        Args:
            encoder: GeneEncoder or HyperbolicGeneEncoder. If None, uses raw features.
            use_hyperbolic: Use Poincaré ball distance instead of cosine.
            curvature: Poincaré ball curvature (only if hyperbolic).
            device: Torch device.
        """
        self.encoder = encoder
        self.use_hyperbolic = use_hyperbolic
        self.curvature = curvature
        self.device = device
        
        # Storage
        self.gene_ids: List[str] = []
        self.features: List[np.ndarray] = []
        self.embeddings: List[torch.Tensor] = []
        self.essentiality: List[bool] = []
        
        # Poincaré math (if needed)
        if use_hyperbolic:
            from ..models.gene_encoder import PoincareMath
            self.poincare = PoincareMath(c=curvature)
    
    def store(self, gene: str, features: np.ndarray, essential: bool):
        """
        Add a gene to memory.
        
        Args:
            gene: Gene ID
            features: Feature vector (15-dim)
            essential: True if essential
        """
        self.gene_ids.append(gene)
        self.features.append(features)
        self.essentiality.append(essential)
        
        # Compute embedding
        with torch.no_grad():
            feat_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            
            if self.encoder is not None:
                embed = self.encoder(feat_tensor.unsqueeze(0)).squeeze(0)
            else:
                # Use normalized raw features
                embed = F.normalize(feat_tensor, p=2, dim=-1)
            
            self.embeddings.append(embed)
    
    def store_batch(
        self, 
        genes: List[str], 
        features: np.ndarray, 
        essentials: List[bool]
    ):
        """Store multiple genes at once."""
        for i, gene in enumerate(genes):
            self.store(gene, features[i], essentials[i])
    
    def _compute_distances(self, query_embed: torch.Tensor) -> torch.Tensor:
        """Compute distances from query to all stored embeddings."""
        if len(self.embeddings) == 0:
            return torch.tensor([])
        
        stored = torch.stack(self.embeddings)
        
        if self.use_hyperbolic:
            # Poincaré distance
            query_exp = query_embed.unsqueeze(0).expand(len(stored), -1)
            distances = self.poincare.distance(query_exp, stored)
        else:
            # Cosine distance = 1 - cosine similarity
            similarities = F.cosine_similarity(query_embed.unsqueeze(0), stored)
            distances = 1 - similarities
        
        return distances
    
    def retrieve(
        self, 
        features: np.ndarray, 
        k: int = 5,
        exclude_gene: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Find k most similar genes in memory.
        
        Args:
            features: Query gene features
            k: Number of neighbors to retrieve
            exclude_gene: Gene ID to exclude (for leave-one-out)
        
        Returns:
            List of RetrievalResult, sorted by distance
        """
        if len(self.embeddings) == 0:
            return []
        
        # Encode query
        with torch.no_grad():
            feat_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            
            if self.encoder is not None:
                query_embed = self.encoder(feat_tensor.unsqueeze(0)).squeeze(0)
            else:
                query_embed = F.normalize(feat_tensor, p=2, dim=-1)
        
        # Compute distances
        distances = self._compute_distances(query_embed)
        
        # Sort by distance
        sorted_indices = torch.argsort(distances).cpu().numpy()
        
        results = []
        for idx in sorted_indices:
            gene = self.gene_ids[idx]
            
            # Skip self-retrieval
            if exclude_gene is not None and gene == exclude_gene:
                continue
            
            results.append(RetrievalResult(
                gene=gene,
                distance=distances[idx].item(),
                essential=self.essentiality[idx],
                features=self.features[idx],
                similarity=1 - distances[idx].item(),
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def predict_by_analogy(
        self, 
        features: np.ndarray, 
        k: int = 5,
        exclude_gene: Optional[str] = None,
    ) -> Dict:
        """
        Predict essentiality by retrieving similar genes.
        
        Uses distance-weighted voting among k nearest neighbors.
        
        Args:
            features: Query gene features
            k: Number of neighbors to vote
            exclude_gene: Gene to exclude (for leave-one-out)
        
        Returns:
            Dict with:
                - essential_score: Weighted probability of essentiality
                - confidence: Agreement among neighbors
                - neighbors: Retrieved genes
        """
        neighbors = self.retrieve(features, k=k, exclude_gene=exclude_gene)
        
        if not neighbors:
            return {
                'essential_score': 0.5,
                'confidence': 0.0,
                'neighbors': [],
            }
        
        # Distance-weighted voting
        weights = []
        votes = []
        
        for n in neighbors:
            # Weight inversely proportional to distance
            weight = 1.0 / (n.distance + 0.01)
            weights.append(weight)
            votes.append(1.0 if n.essential else 0.0)
        
        total_weight = sum(weights)
        essential_score = sum(w * v for w, v in zip(weights, votes)) / total_weight
        
        # Confidence from agreement
        # High confidence = all neighbors agree
        vote_std = np.std(votes) if len(votes) > 1 else 0
        confidence = 1.0 - vote_std
        
        return {
            'essential_score': essential_score,
            'confidence': confidence,
            'neighbors': neighbors,
        }
    
    def leave_one_out_accuracy(self) -> Tuple[float, List[Dict]]:
        """
        Evaluate accuracy using leave-one-out cross-validation.
        
        Returns:
            (accuracy, list of predictions)
        """
        correct = 0
        predictions = []
        
        for i, gene in enumerate(self.gene_ids):
            true_essential = self.essentiality[i]
            features = self.features[i]
            
            # Predict without self
            pred = self.predict_by_analogy(features, k=5, exclude_gene=gene)
            pred_essential = pred['essential_score'] > 0.5
            
            is_correct = pred_essential == true_essential
            if is_correct:
                correct += 1
            
            predictions.append({
                'gene': gene,
                'true': true_essential,
                'pred_score': pred['essential_score'],
                'pred': pred_essential,
                'correct': is_correct,
                'confidence': pred['confidence'],
            })
        
        accuracy = correct / len(self.gene_ids) if self.gene_ids else 0
        return accuracy, predictions
    
    def __len__(self):
        return len(self.gene_ids)


def build_memory_bank(
    fba_model=None,
    encoder=None,
    use_hyperbolic: bool = False,
    verbose: bool = False,
) -> GeneMemoryBank:
    """
    Build a memory bank from all genes in the model.
    
    Args:
        fba_model: FBA model. If None, creates one.
        encoder: Gene encoder. If None, uses raw features.
        use_hyperbolic: Use hyperbolic distances.
        verbose: Print progress.
    
    Returns:
        Populated GeneMemoryBank
    """
    from ..data.gene_features import GeneFeatureExtractor
    from ..data.essentiality import is_essential, get_labeled_genes
    
    # Get features
    extractor = GeneFeatureExtractor(fba_model, verbose=verbose)
    features_dict = extractor.extract_all()
    
    # Build memory bank
    memory = GeneMemoryBank(encoder=encoder, use_hyperbolic=use_hyperbolic)
    
    # Get labeled genes
    labeled = set(get_labeled_genes())
    
    for gene, gf in features_dict.items():
        if gene in labeled:
            essential = is_essential(gene)
            memory.store(gene, gf.features, essential)
    
    if verbose:
        print(f"Built memory bank with {len(memory)} genes")
    
    return memory


if __name__ == "__main__":
    print("Testing GeneMemoryBank...")
    
    # Build memory bank
    memory = build_memory_bank(verbose=True)
    
    print(f"\nMemory bank size: {len(memory)}")
    
    # Test retrieval
    test_gene = 'JCVISYN3A_0207'  # pfkA
    if test_gene in memory.gene_ids:
        idx = memory.gene_ids.index(test_gene)
        features = memory.features[idx]
        
        print(f"\nRetrieving neighbors for {test_gene}:")
        neighbors = memory.retrieve(features, k=5, exclude_gene=test_gene)
        for n in neighbors:
            print(f"  {n.gene}: dist={n.distance:.3f}, essential={n.essential}")
        
        # Test prediction
        pred = memory.predict_by_analogy(features, k=5, exclude_gene=test_gene)
        print(f"\nPrediction for {test_gene}:")
        print(f"  essential_score: {pred['essential_score']:.3f}")
        print(f"  confidence: {pred['confidence']:.3f}")
    
    # Leave-one-out evaluation
    print("\nRunning leave-one-out evaluation...")
    accuracy, predictions = memory.leave_one_out_accuracy()
    print(f"Leave-one-out accuracy: {accuracy*100:.1f}%")
    
    # Show some errors
    errors = [p for p in predictions if not p['correct']]
    print(f"\nErrors ({len(errors)} total):")
    for e in errors[:5]:
        print(f"  {e['gene']}: pred={e['pred_score']:.2f}, true={e['true']}")
