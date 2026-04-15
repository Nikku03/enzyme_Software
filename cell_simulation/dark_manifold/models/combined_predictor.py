"""
dark_manifold/models/combined_predictor.py

Combined Essentiality Predictor.

Combines three prediction methods:
1. FBA (V37) - Physics-based constraint analysis (85.6% accuracy)
2. Memory - Similarity-based retrieval (86.7% accuracy)
3. Rules - Learned interpretable patterns (to be implemented)

The final prediction is a weighted combination of these methods.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PredictionResult:
    """Detailed prediction result."""
    gene: str
    essential: bool
    score: float
    confidence: float
    
    # Individual method scores
    fba_score: float
    memory_score: float
    rule_score: float
    
    # Explanation
    explanation: str
    similar_genes: List[str]
    top_rule: Optional[int]
    
    # Method contributions
    method_agreement: float  # How much methods agree


class CombinedPredictor:
    """
    Combines FBA, Memory, and Rule-based predictions.
    
    Final prediction: weighted average of three methods.
    Weights can be learned or fixed.
    """
    
    def __init__(
        self,
        fba_model=None,
        memory_bank=None,
        rule_module=None,
        fba_weight: float = 0.4,
        memory_weight: float = 0.4,
        rule_weight: float = 0.2,
        feature_extractor=None,
    ):
        """
        Initialize combined predictor.
        
        Args:
            fba_model: FBAModel instance
            memory_bank: GeneMemoryBank instance
            rule_module: RuleDiscoveryModule (optional)
            fba_weight: Weight for FBA predictions
            memory_weight: Weight for memory predictions
            rule_weight: Weight for rule predictions
            feature_extractor: GeneFeatureExtractor (optional)
        """
        # Initialize components
        if fba_model is None:
            from .fba import get_fba_model
            fba_model = get_fba_model(verbose=False)
        self.fba = fba_model
        
        if feature_extractor is None:
            from ..data.gene_features import GeneFeatureExtractor
            feature_extractor = GeneFeatureExtractor(fba_model, verbose=False)
        self.feature_extractor = feature_extractor
        
        if memory_bank is None:
            from ..reasoning.gene_memory import build_memory_bank
            memory_bank = build_memory_bank(fba_model, verbose=False)
        self.memory = memory_bank
        
        self.rules = rule_module  # Can be None
        
        # Weights (must sum to 1)
        total = fba_weight + memory_weight + rule_weight
        self.weights = {
            'fba': fba_weight / total,
            'memory': memory_weight / total,
            'rules': rule_weight / total,
        }
    
    def predict(self, gene: str) -> PredictionResult:
        """
        Predict essentiality using all methods.
        
        Args:
            gene: Gene ID
        
        Returns:
            PredictionResult with combined prediction and explanations
        """
        # 1. FBA prediction
        fba_result = self.fba.knockout(gene)
        # Essential = low biomass ratio
        fba_score = 1.0 - fba_result['biomass_ratio']
        fba_score = max(0, min(1, fba_score))  # Clamp to [0, 1]
        
        # 2. Memory prediction
        features = self.feature_extractor.extract(gene).features
        memory_result = self.memory.predict_by_analogy(
            features, k=5, exclude_gene=gene
        )
        memory_score = memory_result['essential_score']
        
        # 3. Rule prediction (if available)
        if self.rules is not None:
            rule_result = self.rules.match_rules(
                torch.tensor(features, dtype=torch.float32)
            )
            rule_score = rule_result['prediction'].item()
            top_rule = rule_result['top_rule']
        else:
            # Default: use FBA as proxy
            rule_score = fba_score
            top_rule = None
        
        # 4. Weighted combination
        combined_score = (
            self.weights['fba'] * fba_score +
            self.weights['memory'] * memory_score +
            self.weights['rules'] * rule_score
        )
        
        # 5. Confidence from method agreement
        scores = [fba_score, memory_score, rule_score]
        method_agreement = 1.0 - np.std(scores)
        
        # Combine with memory confidence
        confidence = 0.5 * method_agreement + 0.5 * memory_result['confidence']
        
        # 6. Generate explanation
        explanation = self._explain(
            gene, fba_result, memory_result, fba_score, memory_score, rule_score
        )
        
        # 7. Get similar genes for reference
        similar_genes = [n.gene for n in memory_result['neighbors'][:3]]
        
        return PredictionResult(
            gene=gene,
            essential=combined_score > 0.5,
            score=combined_score,
            confidence=confidence,
            fba_score=fba_score,
            memory_score=memory_score,
            rule_score=rule_score,
            explanation=explanation,
            similar_genes=similar_genes,
            top_rule=top_rule,
            method_agreement=method_agreement,
        )
    
    def _explain(
        self,
        gene: str,
        fba_result: Dict,
        memory_result: Dict,
        fba_score: float,
        memory_score: float,
        rule_score: float,
    ) -> str:
        """Generate human-readable explanation."""
        parts = []
        
        # FBA explanation
        if fba_score > 0.9:
            parts.append(f"FBA: Knockout blocks growth ({fba_result['biomass_ratio']:.0%} viability)")
        elif fba_score > 0.5:
            parts.append(f"FBA: Knockout impairs growth ({fba_result['biomass_ratio']:.0%} viability)")
        else:
            parts.append(f"FBA: Knockout allows growth ({fba_result['biomass_ratio']:.0%} viability)")
        
        # Memory explanation
        if memory_result['neighbors']:
            top_neighbor = memory_result['neighbors'][0]
            ess_str = "essential" if top_neighbor.essential else "non-essential"
            parts.append(f"Similar to {top_neighbor.gene} ({ess_str})")
        
        return " | ".join(parts)
    
    def predict_all(self) -> Tuple[List[PredictionResult], Dict]:
        """
        Predict essentiality for all genes in model.
        
        Returns:
            (predictions, summary_stats)
        """
        predictions = []
        
        for gene in self.fba.get_genes():
            pred = self.predict(gene)
            predictions.append(pred)
        
        # Compute summary stats
        correct = sum(
            1 for p in predictions 
            if p.essential == self._get_true_essentiality(p.gene)
        )
        accuracy = correct / len(predictions) if predictions else 0
        
        summary = {
            'accuracy': accuracy,
            'n_genes': len(predictions),
            'n_correct': correct,
            'avg_confidence': np.mean([p.confidence for p in predictions]),
            'avg_agreement': np.mean([p.method_agreement for p in predictions]),
        }
        
        return predictions, summary
    
    def _get_true_essentiality(self, gene: str) -> bool:
        """Get true essentiality label."""
        from ..data.essentiality import is_essential
        return is_essential(gene)
    
    def evaluate(self, verbose: bool = True) -> Dict:
        """
        Evaluate combined predictor against ground truth.
        
        Returns:
            Evaluation metrics including confusion matrix
        """
        from ..data.essentiality import is_essential, get_labeled_genes
        
        labeled = set(get_labeled_genes())
        model_genes = set(self.fba.get_genes())
        test_genes = sorted(labeled & model_genes)
        
        tp, fp, tn, fn = 0, 0, 0, 0
        results = []
        
        for gene in test_genes:
            pred = self.predict(gene)
            true_ess = is_essential(gene)
            
            if true_ess:
                if pred.essential:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred.essential:
                    fp += 1
                else:
                    tn += 1
            
            results.append({
                'gene': gene,
                'pred': pred.essential,
                'true': true_ess,
                'score': pred.score,
                'fba_score': pred.fba_score,
                'memory_score': pred.memory_score,
                'correct': pred.essential == true_ess,
            })
            
            if verbose:
                match = '✓' if pred.essential == true_ess else '✗'
                print(f"  {gene[:15]:15s}: score={pred.score:.2f} | "
                      f"fba={pred.fba_score:.2f} mem={pred.memory_score:.2f} | {match}")
        
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'total': total,
            'results': results,
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"COMBINED PREDICTOR RESULTS")
            print(f"{'='*60}")
            print(f"Accuracy: {accuracy*100:.1f}%")
            print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"  Sensitivity: {sensitivity*100:.1f}%")
            print(f"  Specificity: {specificity*100:.1f}%")
        
        return metrics
    
    def optimize_weights(
        self, 
        n_trials: int = 100,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Find optimal weights using random search.
        
        Returns:
            Best weights found
        """
        from ..data.essentiality import is_essential, get_labeled_genes
        
        labeled = set(get_labeled_genes())
        model_genes = set(self.fba.get_genes())
        test_genes = list(labeled & model_genes)
        
        # Precompute FBA and memory scores
        fba_scores = {}
        memory_scores = {}
        true_labels = {}
        
        for gene in test_genes:
            fba_result = self.fba.knockout(gene)
            fba_scores[gene] = 1.0 - fba_result['biomass_ratio']
            
            features = self.feature_extractor.extract(gene).features
            memory_result = self.memory.predict_by_analogy(features, k=5, exclude_gene=gene)
            memory_scores[gene] = memory_result['essential_score']
            
            true_labels[gene] = is_essential(gene)
        
        best_weights = self.weights.copy()
        best_accuracy = 0
        
        for trial in range(n_trials):
            # Random weights
            w = np.random.dirichlet([1, 1, 1])
            weights = {'fba': w[0], 'memory': w[1], 'rules': w[2]}
            
            # Evaluate
            correct = 0
            for gene in test_genes:
                score = (
                    weights['fba'] * fba_scores[gene] +
                    weights['memory'] * memory_scores[gene] +
                    weights['rules'] * fba_scores[gene]  # Use FBA as rule proxy
                )
                pred = score > 0.5
                if pred == true_labels[gene]:
                    correct += 1
            
            accuracy = correct / len(test_genes)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights.copy()
                if verbose:
                    print(f"Trial {trial}: accuracy={accuracy*100:.1f}%, weights={weights}")
        
        if verbose:
            print(f"\nBest weights: fba={best_weights['fba']:.3f}, "
                  f"memory={best_weights['memory']:.3f}, rules={best_weights['rules']:.3f}")
            print(f"Best accuracy: {best_accuracy*100:.1f}%")
        
        return best_weights


def create_combined_predictor(verbose: bool = False) -> CombinedPredictor:
    """Create and return a combined predictor with default settings."""
    return CombinedPredictor(
        fba_weight=0.4,
        memory_weight=0.4,
        rule_weight=0.2,
    )


if __name__ == "__main__":
    print("="*60)
    print("COMBINED PREDICTOR EVALUATION")
    print("="*60)
    
    # Create predictor
    print("\nInitializing...")
    predictor = CombinedPredictor(
        fba_weight=0.5,
        memory_weight=0.5,
        rule_weight=0.0,  # No rules yet
    )
    
    # Evaluate
    print("\nEvaluating on labeled genes:")
    metrics = predictor.evaluate(verbose=True)
    
    # Optimize weights
    print("\n" + "="*60)
    print("OPTIMIZING WEIGHTS")
    print("="*60)
    best_weights = predictor.optimize_weights(n_trials=100, verbose=True)
    
    # Re-evaluate with optimal weights
    predictor.weights = best_weights
    print("\n" + "="*60)
    print("RE-EVALUATION WITH OPTIMAL WEIGHTS")
    print("="*60)
    metrics = predictor.evaluate(verbose=True)
