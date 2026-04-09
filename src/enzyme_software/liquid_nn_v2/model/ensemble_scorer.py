"""Ensemble SoM Scorer combining ML model with Physics Rules.

The key insight for reaching 90%+ accuracy:
1. ML model (Phase 5) captures learned patterns from data: 47.4% Top-1
2. Physics scorer captures known chemistry rules: ~40-50% Top-1 (estimated)
3. ENSEMBLE should exceed either alone by covering different error modes

Ensemble strategies:
A) Weighted average: score = α * ML_score + (1-α) * physics_score
B) Rank fusion: average ranks from each method
C) Learned gating: use physics confidence to weight methods
D) Boosting: physics corrects ML errors and vice versa
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem
except Exception:
    Chem = None

from enzyme_software.liquid_nn_v2.model.physics_scorer import PhysicsSoMScorer


class EnsembleSoMScorer:
    """
    Combines ML model predictions with physics-based rules.
    
    The physics scorer handles clear-cut chemistry cases (benzylic, N-methyl, etc.)
    The ML model handles subtle cases that require learned context.
    """
    
    def __init__(
        self,
        ml_weight: float = 0.60,
        physics_weight: float = 0.40,
        use_rank_fusion: bool = True,
        boost_high_confidence_physics: bool = True,
        physics_confidence_threshold: float = 0.85,
    ):
        """
        Args:
            ml_weight: Weight for ML model predictions
            physics_weight: Weight for physics predictions
            use_rank_fusion: If True, use rank-based fusion instead of score-based
            boost_high_confidence_physics: If True, trust physics more when confident
            physics_confidence_threshold: Threshold for high-confidence physics prediction
        """
        self.ml_weight = ml_weight
        self.physics_weight = physics_weight
        self.use_rank_fusion = use_rank_fusion
        self.boost_high_confidence_physics = boost_high_confidence_physics
        self.physics_confidence_threshold = physics_confidence_threshold
        
        self.physics_scorer = PhysicsSoMScorer()
    
    def _scores_to_ranks(self, scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Convert scores to ranks (higher score = lower rank number)."""
        n = len(scores)
        ranks = np.zeros(n, dtype=np.float32)
        
        valid_indices = np.where(mask > 0.5)[0]
        if len(valid_indices) == 0:
            return ranks
        
        valid_scores = scores[valid_indices]
        order = np.argsort(-valid_scores)
        
        for rank, idx in enumerate(order):
            original_idx = valid_indices[idx]
            # Rank 1 = best, normalize to [0, 1] where 1 = best
            ranks[original_idx] = 1.0 - (rank / len(valid_indices))
        
        return ranks
    
    def combine_predictions(
        self,
        smiles: str,
        ml_scores: np.ndarray,
        ml_candidate_mask: Optional[np.ndarray] = None,
        atom_coordinates: Optional[np.ndarray] = None,
        heme_center: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Combine ML predictions with physics-based predictions.
        
        Args:
            smiles: Molecule SMILES string
            ml_scores: (N,) array of ML model scores
            ml_candidate_mask: (N,) binary mask of candidate atoms
            atom_coordinates: (N, 3) optional 3D coordinates
            heme_center: (1, 3) optional estimated heme center
            
        Returns:
            Dict with combined scores and diagnostics
        """
        num_atoms = len(ml_scores)
        
        # Get physics scores
        try:
            physics_result = self.physics_scorer.score_molecule(
                smiles,
                atom_coordinates=atom_coordinates,
                heme_center=heme_center,
            )
            physics_scores = physics_result["final_scores"]
            physics_patterns = physics_result["pattern_matches"]
            is_heavy = physics_result["is_heavy"]
        except Exception as e:
            # Fallback: use ML only
            return {
                "ensemble_scores": ml_scores,
                "ml_scores": ml_scores,
                "physics_scores": np.zeros_like(ml_scores),
                "physics_patterns": np.full(num_atoms, "", dtype=object),
                "method": "ml_only",
                "error": str(e),
            }
        
        # Ensure same length
        if len(physics_scores) != num_atoms:
            # Padding or truncation
            if len(physics_scores) < num_atoms:
                physics_scores = np.pad(physics_scores, (0, num_atoms - len(physics_scores)))
                is_heavy = np.pad(is_heavy, (0, num_atoms - len(is_heavy)))
            else:
                physics_scores = physics_scores[:num_atoms]
                is_heavy = is_heavy[:num_atoms]
        
        # Use candidate mask if provided, else use heavy atom mask
        if ml_candidate_mask is not None:
            mask = np.asarray(ml_candidate_mask, dtype=np.float32).reshape(-1)
        else:
            mask = is_heavy
        
        # Normalize scores to [0, 1]
        ml_norm = ml_scores.copy()
        if ml_norm.max() > ml_norm.min():
            ml_norm = (ml_norm - ml_norm.min()) / (ml_norm.max() - ml_norm.min())
        
        physics_norm = physics_scores.copy()
        if physics_norm.max() > physics_norm.min():
            physics_norm = (physics_norm - physics_norm.min()) / (physics_norm.max() - physics_norm.min())
        
        # Apply mask
        ml_norm = ml_norm * mask
        physics_norm = physics_norm * mask
        
        if self.use_rank_fusion:
            # Rank-based fusion (more robust to score scale differences)
            ml_ranks = self._scores_to_ranks(ml_norm, mask)
            physics_ranks = self._scores_to_ranks(physics_norm, mask)
            
            ensemble_ranks = (
                self.ml_weight * ml_ranks + 
                self.physics_weight * physics_ranks
            )
            ensemble_scores = ensemble_ranks
        else:
            # Score-based fusion
            ensemble_scores = (
                self.ml_weight * ml_norm + 
                self.physics_weight * physics_norm
            )
        
        # Boost high-confidence physics predictions
        if self.boost_high_confidence_physics:
            high_conf_physics = physics_norm >= self.physics_confidence_threshold
            # For atoms where physics is very confident, increase weight
            boost_factor = np.where(high_conf_physics, 1.3, 1.0)
            physics_contribution = self.physics_weight * physics_norm * boost_factor
            
            # Recompute with boosted physics
            if self.use_rank_fusion:
                physics_ranks_boosted = self._scores_to_ranks(physics_norm * boost_factor, mask)
                ensemble_scores = (
                    self.ml_weight * ml_ranks + 
                    self.physics_weight * physics_ranks_boosted
                )
            else:
                ensemble_scores = (
                    self.ml_weight * ml_norm + 
                    physics_contribution
                )
        
        # Find cases where physics and ML disagree
        ml_top = np.argmax(ml_norm)
        physics_top = np.argmax(physics_norm)
        ensemble_top = np.argmax(ensemble_scores)
        
        agreement = "full" if ml_top == physics_top else "partial"
        if ml_top == physics_top == ensemble_top:
            agreement = "full"
        
        return {
            "ensemble_scores": ensemble_scores,
            "ml_scores": ml_norm,
            "physics_scores": physics_norm,
            "physics_patterns": physics_patterns if len(physics_patterns) == num_atoms else np.full(num_atoms, "", dtype=object),
            "ml_top": int(ml_top),
            "physics_top": int(physics_top),
            "ensemble_top": int(ensemble_top),
            "agreement": agreement,
            "method": "ensemble",
        }
    
    def predict_top_k(
        self,
        smiles: str,
        ml_scores: np.ndarray,
        k: int = 3,
        **kwargs,
    ) -> List[Tuple[int, float, str]]:
        """
        Get top-k predictions from ensemble.
        
        Returns:
            List of (atom_idx, ensemble_score, info) tuples
        """
        result = self.combine_predictions(smiles, ml_scores, **kwargs)
        scores = result["ensemble_scores"]
        patterns = result.get("physics_patterns", np.full(len(scores), "", dtype=object))
        
        # Sort by score descending
        sorted_indices = np.argsort(-scores)
        
        top_k = []
        for idx in sorted_indices[:k]:
            pattern = str(patterns[idx]) if idx < len(patterns) else ""
            info = f"physics:{pattern}" if pattern else "ml"
            top_k.append((int(idx), float(scores[idx]), info))
        
        return top_k


def calibrate_ensemble_weights(
    dataset: List[Dict],
    ml_model_fn,
    *,
    weight_range: Tuple[float, float] = (0.3, 0.8),
    steps: int = 11,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Find optimal ensemble weights via grid search.
    
    Args:
        dataset: List of dicts with 'smiles', 'site_labels', 'ml_scores'
        ml_model_fn: Function that takes smiles and returns ML scores
        weight_range: Range of ML weights to try
        steps: Number of weight values to try
        
    Returns:
        Dict with best weights and performance
    """
    physics_scorer = PhysicsSoMScorer()
    
    best_weight = 0.5
    best_top1 = 0.0
    results = []
    
    weights = np.linspace(weight_range[0], weight_range[1], steps)
    
    for ml_weight in weights:
        ensemble = EnsembleSoMScorer(
            ml_weight=ml_weight,
            physics_weight=1.0 - ml_weight,
        )
        
        top1_correct = 0
        total = 0
        
        for item in dataset:
            smiles = item.get("smiles", "")
            true_sites = set(int(s) for s in item.get("site_labels", []) if isinstance(s, (int, float)))
            ml_scores = item.get("ml_scores")
            
            if not smiles or not true_sites or ml_scores is None:
                continue
            
            try:
                result = ensemble.combine_predictions(smiles, np.array(ml_scores))
                top1_pred = result["ensemble_top"]
                
                if top1_pred in true_sites:
                    top1_correct += 1
                total += 1
            except Exception:
                continue
        
        top1_acc = top1_correct / max(total, 1)
        results.append((ml_weight, top1_acc))
        
        if verbose:
            print(f"ML weight {ml_weight:.2f}: Top-1 = {top1_acc:.3f}")
        
        if top1_acc > best_top1:
            best_top1 = top1_acc
            best_weight = ml_weight
    
    return {
        "best_ml_weight": best_weight,
        "best_physics_weight": 1.0 - best_weight,
        "best_top1_accuracy": best_top1,
        "all_results": results,
    }
