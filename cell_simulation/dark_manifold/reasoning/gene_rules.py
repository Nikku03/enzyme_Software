"""
dark_manifold/reasoning/gene_rules.py

Gene Rule Discovery Module.

Learns interpretable rules for gene essentiality prediction.

Adapted from nexus/dark_manifold.py RuleDiscoveryModule.

Goal: Discover human-readable rules like:
- "Hub genes with no isozymes are essential"
- "Genes in irreversible reactions are essential"  
- "High centrality + low redundancy → essential"

Two approaches:
1. Neural rules: Learned prototypes matched via cosine similarity
2. Decision rules: Explicit threshold-based rules extracted from data
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Feature names for interpretability
FEATURE_NAMES = [
    'biomass_ratio',           # 0
    'blocked_reactions',       # 1
    'flux_variability',        # 2
    'degree_centrality',       # 3
    'betweenness',             # 4
    'closeness',               # 5
    'is_hub',                  # 6
    'clustering',              # 7
    'isozyme_count',           # 8
    'has_alternative',         # 9
    'pathway_redundancy',      # 10
    'expression',              # 11
    'protein_halflife',        # 12
    'delta_g',                 # 13
    'is_irreversible',         # 14
]


@dataclass
class Rule:
    """A discovered rule."""
    rule_id: int
    description: str
    conditions: List[Tuple[str, str, float]]  # [(feature, op, threshold), ...]
    prediction: str  # "essential" or "non-essential"
    confidence: float
    support: int  # Number of genes matching this rule
    accuracy: float  # Accuracy on matching genes


class NeuralRuleModule(nn.Module):
    """
    Neural rule discovery via learned prototypes.
    
    Adapted from RuleDiscoveryModule in dark_manifold.py.
    
    Each rule is a prototype vector in feature space.
    Genes are assigned to rules via cosine similarity.
    Rule prediction is the average essentiality of assigned genes.
    """
    
    def __init__(
        self,
        feature_dim: int = 15,
        num_rules: int = 16,
        hidden_dim: int = 32,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_rules = num_rules
        self.hidden_dim = hidden_dim
        
        # Rule prototypes (learned)
        self.rule_prototypes = nn.Parameter(torch.randn(num_rules, feature_dim))
        
        # Rule prediction heads
        self.rule_heads = nn.Parameter(torch.zeros(num_rules))
        
        # Rule confidence (learned)
        self.rule_confidence = nn.Parameter(torch.zeros(num_rules))
        
        # Statistics buffers
        self.register_buffer("rule_counts", torch.zeros(num_rules))
        self.register_buffer("rule_correct", torch.zeros(num_rules))
    
    def match_rules(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Find which rules match the input features.
        
        Args:
            features: Gene features (batch_size, feature_dim) or (feature_dim,)
        
        Returns:
            Dict with activations, prediction, top_rule, confidence
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # Normalize for cosine similarity
        features_norm = F.normalize(features, p=2, dim=-1)
        prototypes_norm = F.normalize(self.rule_prototypes, p=2, dim=-1)
        
        # Compute similarities
        similarities = torch.mm(features_norm, prototypes_norm.t())  # (batch, num_rules)
        
        # Soft assignment via softmax
        activations = F.softmax(similarities * 5.0, dim=-1)
        
        # Weighted prediction
        rule_preds = torch.sigmoid(self.rule_heads)
        prediction = (activations * rule_preds.unsqueeze(0)).sum(dim=-1)
        
        # Top rule and confidence
        top_rule = similarities.argmax(dim=-1)
        confidence = torch.sigmoid(self.rule_confidence)[top_rule]
        
        return {
            'activations': activations,
            'prediction': prediction,
            'top_rule': top_rule,
            'confidence': confidence,
            'similarities': similarities,
        }
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass returns essentiality prediction."""
        result = self.match_rules(features)
        return result['prediction']
    
    def update_statistics(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Update rule statistics based on predictions."""
        with torch.no_grad():
            result = self.match_rules(features)
            top_rules = result['top_rule']
            preds = (result['prediction'] > 0.5).float()
            correct = (preds == labels).float()
            
            for i in range(features.size(0)):
                rule_idx = top_rules[i].item()
                self.rule_counts[rule_idx] += 1
                self.rule_correct[rule_idx] += correct[i].item()
    
    def get_rule_accuracy(self, rule_idx: int) -> float:
        """Get accuracy for a specific rule."""
        if self.rule_counts[rule_idx] == 0:
            return 0.0
        return (self.rule_correct[rule_idx] / self.rule_counts[rule_idx]).item()
    
    def interpret_rules(self, top_k: int = 5) -> List[str]:
        """
        Generate human-readable descriptions of top rules.
        
        Looks at which features have highest weights in each prototype.
        """
        rules = []
        confidences = torch.sigmoid(self.rule_confidence).detach().numpy()
        
        # Sort rules by confidence
        rule_indices = np.argsort(-confidences)
        
        for rule_idx in rule_indices[:top_k]:
            prototype = self.rule_prototypes[rule_idx].detach().numpy()
            pred = torch.sigmoid(self.rule_heads[rule_idx]).item()
            conf = confidences[rule_idx]
            acc = self.get_rule_accuracy(rule_idx)
            count = int(self.rule_counts[rule_idx].item())
            
            # Find top 2 features
            top_features = np.argsort(np.abs(prototype))[-2:][::-1]
            
            conditions = []
            for f_idx in top_features:
                f_name = FEATURE_NAMES[f_idx]
                f_value = prototype[f_idx]
                if f_value > 0.3:
                    conditions.append(f"high {f_name}")
                elif f_value < -0.3:
                    conditions.append(f"low {f_name}")
            
            if not conditions:
                conditions = ["general pattern"]
            
            outcome = "ESSENTIAL" if pred > 0.5 else "NON-ESSENTIAL"
            
            rule_str = (
                f"Rule {rule_idx}: IF {' AND '.join(conditions)} "
                f"THEN {outcome} (conf={conf:.0%}, acc={acc:.0%}, n={count})"
            )
            rules.append(rule_str)
        
        return rules


class DecisionRuleExtractor:
    """
    Extract explicit threshold-based decision rules from data.
    
    Simpler and more interpretable than neural rules.
    Uses feature thresholds learned from data.
    """
    
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names or FEATURE_NAMES
        self.rules: List[Rule] = []
    
    def extract_rules(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        gene_ids: List[str],
        min_support: int = 3,
        min_confidence: float = 0.7,
    ) -> List[Rule]:
        """
        Extract decision rules from training data.
        
        Uses simple threshold-based rules on individual features
        and feature combinations.
        
        Args:
            features: (N, D) feature matrix
            labels: (N,) binary labels
            gene_ids: Gene identifiers
            min_support: Minimum genes for a rule
            min_confidence: Minimum accuracy for a rule
        
        Returns:
            List of discovered rules
        """
        self.rules = []
        rule_id = 0
        
        n_samples, n_features = features.shape
        
        # Single feature rules
        for f_idx in range(n_features):
            f_name = self.feature_names[f_idx]
            f_values = features[:, f_idx]
            
            # Try thresholds at quartiles
            for q in [0.25, 0.5, 0.75]:
                threshold = np.quantile(f_values, q)
                
                # High feature → Essential
                mask_high = f_values >= threshold
                if mask_high.sum() >= min_support:
                    acc = labels[mask_high].mean() if mask_high.sum() > 0 else 0
                    if acc >= min_confidence:
                        self.rules.append(Rule(
                            rule_id=rule_id,
                            description=f"IF {f_name} >= {threshold:.2f} THEN ESSENTIAL",
                            conditions=[(f_name, '>=', threshold)],
                            prediction='essential',
                            confidence=acc,
                            support=int(mask_high.sum()),
                            accuracy=acc,
                        ))
                        rule_id += 1
                
                # Low feature → Essential  
                mask_low = f_values < threshold
                if mask_low.sum() >= min_support:
                    acc = labels[mask_low].mean() if mask_low.sum() > 0 else 0
                    if acc >= min_confidence:
                        self.rules.append(Rule(
                            rule_id=rule_id,
                            description=f"IF {f_name} < {threshold:.2f} THEN ESSENTIAL",
                            conditions=[(f_name, '<', threshold)],
                            prediction='essential',
                            confidence=acc,
                            support=int(mask_low.sum()),
                            accuracy=acc,
                        ))
                        rule_id += 1
                
                # Non-essential rules
                if mask_high.sum() >= min_support:
                    acc = 1 - labels[mask_high].mean() if mask_high.sum() > 0 else 0
                    if acc >= min_confidence:
                        self.rules.append(Rule(
                            rule_id=rule_id,
                            description=f"IF {f_name} >= {threshold:.2f} THEN NON-ESSENTIAL",
                            conditions=[(f_name, '>=', threshold)],
                            prediction='non-essential',
                            confidence=acc,
                            support=int(mask_high.sum()),
                            accuracy=acc,
                        ))
                        rule_id += 1
        
        # Two-feature combination rules (top combinations only)
        important_features = [0, 3, 6, 8, 9]  # biomass, degree, hub, isozyme, alt_path
        
        for i, f1_idx in enumerate(important_features):
            for f2_idx in important_features[i+1:]:
                f1_name = self.feature_names[f1_idx]
                f2_name = self.feature_names[f2_idx]
                f1_vals = features[:, f1_idx]
                f2_vals = features[:, f2_idx]
                
                # Try median thresholds
                t1 = np.median(f1_vals)
                t2 = np.median(f2_vals)
                
                # Both high
                mask = (f1_vals >= t1) & (f2_vals >= t2)
                if mask.sum() >= min_support:
                    acc = labels[mask].mean()
                    if acc >= min_confidence:
                        self.rules.append(Rule(
                            rule_id=rule_id,
                            description=f"IF {f1_name} >= {t1:.2f} AND {f2_name} >= {t2:.2f} THEN ESSENTIAL",
                            conditions=[(f1_name, '>=', t1), (f2_name, '>=', t2)],
                            prediction='essential',
                            confidence=acc,
                            support=int(mask.sum()),
                            accuracy=acc,
                        ))
                        rule_id += 1
        
        # Sort by confidence
        self.rules.sort(key=lambda r: -r.confidence)
        
        return self.rules
    
    def predict(self, features: np.ndarray) -> Tuple[float, List[Rule]]:
        """
        Predict using extracted rules.
        
        Returns weighted vote from matching rules.
        """
        if len(self.rules) == 0:
            return 0.5, []
        
        matching_rules = []
        
        for rule in self.rules:
            matches = True
            for f_name, op, threshold in rule.conditions:
                f_idx = self.feature_names.index(f_name)
                f_val = features[f_idx]
                
                if op == '>=' and f_val < threshold:
                    matches = False
                    break
                elif op == '<' and f_val >= threshold:
                    matches = False
                    break
            
            if matches:
                matching_rules.append(rule)
        
        if not matching_rules:
            return 0.5, []
        
        # Weighted vote
        total_weight = 0
        essential_vote = 0
        
        for rule in matching_rules:
            weight = rule.confidence * rule.support
            total_weight += weight
            if rule.prediction == 'essential':
                essential_vote += weight
        
        score = essential_vote / total_weight if total_weight > 0 else 0.5
        return score, matching_rules
    
    def get_top_rules(self, n: int = 10) -> List[Rule]:
        """Get top N rules by confidence."""
        return self.rules[:n]
    
    def print_rules(self, n: int = 10):
        """Print top rules."""
        print(f"\nTop {n} Discovered Rules:")
        print("=" * 70)
        for rule in self.rules[:n]:
            print(f"  {rule.description}")
            print(f"    Confidence: {rule.confidence:.1%}, Support: {rule.support} genes")
        print("=" * 70)


class GeneRuleDiscovery:
    """
    Combined rule discovery using both neural and decision approaches.
    
    The neural module learns patterns that may not be captured by
    simple threshold rules, while decision rules provide interpretability.
    """
    
    def __init__(
        self,
        feature_dim: int = 15,
        num_neural_rules: int = 16,
        use_neural: bool = True,
        use_decision: bool = True,
    ):
        self.feature_dim = feature_dim
        self.use_neural = use_neural
        self.use_decision = use_decision
        
        if use_neural:
            self.neural_module = NeuralRuleModule(feature_dim, num_neural_rules)
        else:
            self.neural_module = None
        
        if use_decision:
            self.decision_extractor = DecisionRuleExtractor()
        else:
            self.decision_extractor = None
        
        self._fitted = False
    
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        gene_ids: List[str],
        epochs: int = 100,
        lr: float = 0.01,
        verbose: bool = False,
    ):
        """
        Fit rule discovery to training data.
        
        Args:
            features: (N, D) feature matrix
            labels: (N,) binary labels (1 = essential)
            gene_ids: Gene identifiers
            epochs: Training epochs for neural rules
            lr: Learning rate
            verbose: Print progress
        """
        # Extract decision rules
        if self.decision_extractor is not None:
            self.decision_extractor.extract_rules(
                features, labels, gene_ids,
                min_support=3, min_confidence=0.7,
            )
            if verbose:
                print(f"Extracted {len(self.decision_extractor.rules)} decision rules")
        
        # Train neural rules
        if self.neural_module is not None:
            X = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.float32)
            
            optimizer = torch.optim.Adam(self.neural_module.parameters(), lr=lr)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                pred = self.neural_module(X).squeeze()
                loss = F.binary_cross_entropy(pred, y)
                loss.backward()
                optimizer.step()
                
                # Update statistics
                self.neural_module.update_statistics(X, y)
                
                if verbose and (epoch + 1) % 20 == 0:
                    acc = ((pred > 0.5) == y).float().mean()
                    print(f"Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc.item():.1%}")
            
            if verbose:
                print("\nLearned Neural Rules:")
                for rule in self.neural_module.interpret_rules(5):
                    print(f"  {rule}")
        
        self._fitted = True
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Predict using discovered rules.
        
        Returns:
            Dict with prediction, confidence, matching rules
        """
        if not self._fitted:
            return {'prediction': 0.5, 'confidence': 0.0, 'rules': []}
        
        scores = []
        
        # Neural prediction
        if self.neural_module is not None:
            with torch.no_grad():
                X = torch.tensor(features, dtype=torch.float32)
                result = self.neural_module.match_rules(X)
                neural_score = result['prediction'].item()
                scores.append(neural_score)
        
        # Decision prediction
        if self.decision_extractor is not None:
            decision_score, matching_rules = self.decision_extractor.predict(features)
            scores.append(decision_score)
        else:
            matching_rules = []
        
        # Average predictions
        final_score = np.mean(scores) if scores else 0.5
        
        return {
            'prediction': final_score,
            'essential': final_score > 0.5,
            'confidence': 1.0 - np.std(scores) if len(scores) > 1 else 0.5,
            'matching_rules': matching_rules,
            'neural_score': scores[0] if self.neural_module else None,
            'decision_score': scores[1] if self.decision_extractor and len(scores) > 1 else None,
        }
    
    def get_interpretable_rules(self, n: int = 10) -> List[str]:
        """Get human-readable rules."""
        rules = []
        
        if self.decision_extractor is not None:
            for rule in self.decision_extractor.get_top_rules(n):
                rules.append(rule.description)
        
        if self.neural_module is not None:
            rules.extend(self.neural_module.interpret_rules(n))
        
        return rules


def train_rule_discovery(verbose: bool = True) -> GeneRuleDiscovery:
    """
    Train rule discovery on all labeled genes.
    
    Returns trained GeneRuleDiscovery instance.
    """
    from ..data.gene_features import GeneFeatureExtractor
    from ..data.essentiality import is_essential, get_labeled_genes
    from ..models.fba import get_fba_model
    
    # Get features
    fba = get_fba_model(verbose=False)
    extractor = GeneFeatureExtractor(fba, verbose=False)
    
    # Get labeled genes
    labeled = get_labeled_genes()
    model_genes = set(fba.get_genes())
    genes = [g for g in labeled if g in model_genes]
    
    # Extract features and labels
    features = []
    labels = []
    
    for gene in genes:
        gf = extractor.extract(gene)
        features.append(gf.features)
        labels.append(1.0 if is_essential(gene) else 0.0)
    
    features = np.stack(features)
    labels = np.array(labels)
    
    if verbose:
        print(f"Training on {len(genes)} genes...")
        print(f"  Essential: {int(labels.sum())}")
        print(f"  Non-essential: {int(len(labels) - labels.sum())}")
    
    # Train
    rules = GeneRuleDiscovery(feature_dim=15, num_neural_rules=16)
    rules.fit(features, labels, genes, epochs=100, verbose=verbose)
    
    return rules


if __name__ == "__main__":
    print("="*60)
    print("GENE RULE DISCOVERY")
    print("="*60)
    
    rules = train_rule_discovery(verbose=True)
    
    print("\n" + "="*60)
    print("INTERPRETABLE RULES")
    print("="*60)
    
    for rule in rules.get_interpretable_rules(10):
        print(f"  {rule}")
