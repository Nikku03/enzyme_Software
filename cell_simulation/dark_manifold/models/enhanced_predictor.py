"""
Enhanced essentiality predictor with external data.
Combines FBA + PPI + functional categories.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import sys
sys.path.insert(0, '.')

from dark_manifold.models.fba import get_fba_model
from dark_manifold.data.essentiality import GENE_ESSENTIALITY
from dark_manifold.data.gene_features import GeneFeatureExtractor
from dark_manifold.external_data.ppi_features import get_ppi_features, get_ppi_network


class EnhancedPredictor:
    """Combines FBA + PPI + ML for essentiality prediction."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.fba = get_fba_model(verbose=False)
        self.extractor = GeneFeatureExtractor(self.fba, verbose=False)
        self.ppi_network = get_ppi_network()
        self.model = None
        self.scaler = None
        
    def extract_features(self, gene: str, essentials: set = None) -> np.ndarray:
        """Extract all features for a gene."""
        if essentials is None:
            essentials = set()
            
        # Base features from FBA
        base = self.extractor.extract(gene)
        base_feat = base.features  # 15-dim
        
        # FBA biomass ratio (continuous)
        fba_result = self.fba.knockout(gene)
        biomass = fba_result['biomass_ratio']
        
        # PPI features
        ppi = get_ppi_features(gene, essentials)
        ppi_feat = [
            ppi['ppi_degree'],
            ppi['ppi_in_network'],
            ppi['ppi_neighbor_ess_frac'],
        ]
        
        # Combine
        return np.concatenate([base_feat, [biomass], ppi_feat])
    
    def fit(self, genes: list, labels: np.ndarray):
        """Train the model."""
        # Get essentials for neighbor features
        essentials = set(g for g, l in zip(genes, labels) if l == 1)
        
        # Extract features
        X = np.array([self.extract_features(g, essentials) for g in genes])
        
        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train with class weights
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42,
        )
        
        # Oversample minority class via sample weights
        sample_weights = np.where(labels == 0, 10, 1)  # Weight non-essential 10x
        self.model.fit(X_scaled, labels, sample_weight=sample_weights)
        
        return self
    
    def predict(self, gene: str, essentials: set = None) -> bool:
        """Predict essentiality."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X = self.extract_features(gene, essentials).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return bool(self.model.predict(X_scaled)[0])
    
    def predict_proba(self, gene: str, essentials: set = None) -> float:
        """Get essentiality probability."""
        if self.model is None:
            raise ValueError("Model not trained")
            
        X = self.extract_features(gene, essentials).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[0, 1]


def cross_validate(predictor_class=EnhancedPredictor, n_splits=5):
    """5-fold CV evaluation."""
    fba = get_fba_model(verbose=False)
    
    # Get labeled genes
    model_genes = set(fba.get_genes())
    labeled = set(g for g in GENE_ESSENTIALITY.keys())
    genes = np.array(sorted(model_genes & labeled))
    y = np.array([1 if GENE_ESSENTIALITY[g] in ['E', 'Q'] else 0 for g in genes])
    
    print(f"Dataset: {len(genes)} genes ({y.sum()} essential, {len(y)-y.sum()} non-essential)")
    
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_tp = all_fp = all_tn = all_fn = 0
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(genes, y)):
        train_genes = genes[train_idx]
        test_genes = genes[test_idx]
        train_y = y[train_idx]
        test_y = y[test_idx]
        
        # Train
        pred = predictor_class(verbose=False)
        pred.fit(train_genes.tolist(), train_y)
        
        # Get essentials from training set
        train_essentials = set(g for g, l in zip(train_genes, train_y) if l == 1)
        
        # Test
        for i, g in enumerate(test_genes):
            pred_ess = pred.predict(g, train_essentials)
            true_ess = test_y[i]
            
            if true_ess:
                if pred_ess:
                    all_tp += 1
                else:
                    all_fn += 1
            else:
                if pred_ess:
                    all_fp += 1
                else:
                    all_tn += 1
    
    acc = (all_tp + all_tn) / (all_tp + all_fp + all_tn + all_fn)
    sens = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 0
    spec = all_tn / (all_tn + all_fp) if (all_tn + all_fp) else 0
    bal = (sens + spec) / 2
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bal,
        'sensitivity': sens,
        'specificity': spec,
        'tp': all_tp, 'fp': all_fp, 'tn': all_tn, 'fn': all_fn,
    }


if __name__ == "__main__":
    print("="*60)
    print("ENHANCED PREDICTOR (FBA + PPI + ML)")
    print("="*60)
    
    results = cross_validate()
    
    print(f"\nResults (5-fold CV):")
    print(f"  Accuracy:     {results['accuracy']*100:.1f}%")
    print(f"  Balanced:     {results['balanced_accuracy']*100:.1f}%")
    print(f"  Sensitivity:  {results['sensitivity']*100:.0f}%")
    print(f"  Specificity:  {results['specificity']*100:.0f}%")
    print(f"  TP={results['tp']} FP={results['fp']} TN={results['tn']} FN={results['fn']}")
    
    print("\nCompare to FBA alone: Bal=69.5%")
