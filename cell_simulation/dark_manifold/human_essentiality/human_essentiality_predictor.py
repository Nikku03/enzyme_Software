"""
Human Gene Essentiality Predictor - Hedge Fund Model

Adapted from the JCVI-syn3A "hedge fund" approach for human cancer cell lines.

CONCEPT:
- Instead of cross-SPECIES consensus (bacteria), use cross-CELL-LINE consensus
- If gene X is essential in 80% of similar cell lines → likely essential here
- For context-dependent genes, use ML with expression/CNV features

USAGE:
    from human_essentiality_predictor import HumanEssentialityPredictor
    
    # Load DepMap data
    predictor = HumanEssentialityPredictor()
    predictor.fit(crispr_gene_effect_df)
    
    # Predict for held-out cell line
    predictions = predictor.predict(cell_line_id='ACH-000001')
    
    # Or predict for new cell line with expression data
    predictions = predictor.predict_new(expression_profile)

Author: Chhillar/Naresh
Date: April 2026
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class HumanEssentialityPredictor:
    """
    Hedge Fund Predictor for Human Gene Essentiality.
    
    Strategy:
    1. For pan-essential genes (consensus ≥ 0.9): Always predict essential
    2. For non-essential genes (consensus ≤ 0.1): Always predict non-essential
    3. For context-dependent genes: Use ML model with features
    
    Features:
    - Consensus essentiality across other cell lines
    - Variance in essentiality (context-dependence signal)
    - Mean/variance of continuous scores (e.g., Chronos, SBF)
    - Expression levels (if available)
    - Copy number (if available)
    - Mutation status (if available)
    """
    
    def __init__(
        self,
        consensus_high_threshold: float = 0.9,
        consensus_low_threshold: float = 0.1,
        ml_model: str = 'logistic',  # 'logistic', 'rf', 'gbm'
        use_expression: bool = False,
        use_cnv: bool = False,
    ):
        """
        Initialize the predictor.
        
        Args:
            consensus_high_threshold: Genes with consensus above this are "pan-essential"
            consensus_low_threshold: Genes with consensus below this are "non-essential"
            ml_model: Which ML model to use for context-dependent genes
            use_expression: Whether to use expression features
            use_cnv: Whether to use copy number features
        """
        self.consensus_high = consensus_high_threshold
        self.consensus_low = consensus_low_threshold
        self.use_expression = use_expression
        self.use_cnv = use_cnv
        
        # Initialize ML model
        if ml_model == 'logistic':
            self.ml_model = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced')
        elif ml_model == 'rf':
            self.ml_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')
        elif ml_model == 'gbm':
            self.ml_model = GradientBoostingClassifier(n_estimators=100, max_depth=5)
        else:
            raise ValueError(f"Unknown model: {ml_model}")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Store data
        self.essentiality_matrix = None
        self.gene_names = None
        self.cell_line_names = None
        self.expression_matrix = None
        self.cnv_matrix = None
        
    def fit(
        self,
        essentiality_df: pd.DataFrame,
        expression_df: Optional[pd.DataFrame] = None,
        cnv_df: Optional[pd.DataFrame] = None,
        binarize: bool = True,
        threshold: float = -0.5,
    ) -> 'HumanEssentialityPredictor':
        """
        Fit the predictor on DepMap data.
        
        Args:
            essentiality_df: Gene effect scores (genes x cell_lines)
                             Negative = essential, Positive = non-essential
            expression_df: Optional expression data (genes x cell_lines)
            cnv_df: Optional copy number data (genes x cell_lines)
            binarize: Whether to convert scores to binary (essential/not)
            threshold: Threshold for binarization (default: -0.5 for Chronos)
        
        Returns:
            self
        """
        print("Fitting Human Essentiality Predictor...")
        
        # Store gene and cell line names
        self.gene_names = list(essentiality_df.index)
        self.cell_line_names = list(essentiality_df.columns)
        
        # Binarize if needed
        if binarize:
            # In DepMap Chronos scores: negative = essential
            self.essentiality_matrix = (essentiality_df < threshold).astype(int)
            print(f"  Binarized scores with threshold {threshold}")
        else:
            self.essentiality_matrix = essentiality_df
        
        # Store optional features
        if expression_df is not None and self.use_expression:
            self.expression_matrix = expression_df
            print(f"  Expression data: {expression_df.shape}")
        
        if cnv_df is not None and self.use_cnv:
            self.cnv_matrix = cnv_df
            print(f"  CNV data: {cnv_df.shape}")
        
        # Calculate gene-level statistics
        self.gene_consensus = self.essentiality_matrix.mean(axis=1)
        self.gene_variance = self.essentiality_matrix.var(axis=1)
        
        # Classify genes
        n_pan = (self.gene_consensus >= self.consensus_high).sum()
        n_non = (self.gene_consensus <= self.consensus_low).sum()
        n_context = len(self.gene_names) - n_pan - n_non
        
        print(f"  Pan-essential genes (≥{self.consensus_high:.0%}): {n_pan}")
        print(f"  Non-essential genes (≤{self.consensus_low:.0%}): {n_non}")
        print(f"  Context-dependent genes: {n_context}")
        
        # Train ML model on context-dependent genes
        self._train_ml_model()
        
        self.is_fitted = True
        print("Fitting complete!")
        
        return self
    
    def _train_ml_model(self):
        """Train the ML model on context-dependent genes."""
        # Get context-dependent genes
        mask = (self.gene_consensus > self.consensus_low) & \
               (self.gene_consensus < self.consensus_high)
        
        context_genes = [g for g, m in zip(self.gene_names, mask) if m]
        
        if len(context_genes) < 100:
            print("  Warning: Too few context-dependent genes for ML training")
            return
        
        print(f"  Training ML model on {len(context_genes)} context-dependent genes...")
        
        # Build training data
        # For each (gene, cell_line) pair in context-dependent genes
        X_train = []
        y_train = []
        
        for gene in context_genes:
            for i, cell_line in enumerate(self.cell_line_names):
                # Features: consensus from OTHER cell lines
                other_cols = [c for c in self.cell_line_names if c != cell_line]
                
                gene_data = self.essentiality_matrix.loc[gene, other_cols]
                consensus = gene_data.mean()
                variance = gene_data.var()
                
                features = [consensus, variance, consensus**2, np.sqrt(max(0, consensus))]
                
                # Add expression if available
                if self.expression_matrix is not None:
                    if gene in self.expression_matrix.index and cell_line in self.expression_matrix.columns:
                        expr = self.expression_matrix.loc[gene, cell_line]
                        features.append(expr)
                
                X_train.append(features)
                y_train.append(self.essentiality_matrix.loc[gene, cell_line])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        
        # Train model
        self.ml_model.fit(X_train, y_train)
        
        # Report training accuracy
        y_pred = self.ml_model.predict(X_train)
        acc = accuracy_score(y_train, y_pred)
        bal_acc = balanced_accuracy_score(y_train, y_pred)
        print(f"  ML model training accuracy: {acc:.3f} (balanced: {bal_acc:.3f})")
    
    def predict(
        self,
        cell_line_id: str,
        return_confidence: bool = False,
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Predict gene essentiality for a cell line (leave-one-out style).
        
        Args:
            cell_line_id: ID of the cell line to predict for
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions (and optionally confidence scores)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if cell_line_id not in self.cell_line_names:
            raise ValueError(f"Unknown cell line: {cell_line_id}")
        
        # Get consensus from OTHER cell lines
        other_cols = [c for c in self.cell_line_names if c != cell_line_id]
        consensus = self.essentiality_matrix[other_cols].mean(axis=1)
        
        # Initialize predictions
        predictions = pd.Series(index=self.gene_names, dtype=int)
        confidence = pd.Series(index=self.gene_names, dtype=float)
        
        # Pan-essential: predict essential
        pan_mask = consensus >= self.consensus_high
        predictions[pan_mask] = 1
        confidence[pan_mask] = consensus[pan_mask]
        
        # Non-essential: predict non-essential
        non_mask = consensus <= self.consensus_low
        predictions[non_mask] = 0
        confidence[non_mask] = 1 - consensus[non_mask]
        
        # Context-dependent: use ML model
        context_mask = ~pan_mask & ~non_mask
        context_genes = predictions[context_mask].index.tolist()
        
        if len(context_genes) > 0:
            X_context = []
            for gene in context_genes:
                gene_cons = consensus[gene]
                gene_var = self.essentiality_matrix.loc[gene, other_cols].var()
                features = [gene_cons, gene_var, gene_cons**2, np.sqrt(max(0, gene_cons))]
                
                if self.expression_matrix is not None:
                    if gene in self.expression_matrix.index and cell_line_id in self.expression_matrix.columns:
                        features.append(self.expression_matrix.loc[gene, cell_line_id])
                
                X_context.append(features)
            
            X_context = np.array(X_context)
            X_context = self.scaler.transform(X_context)
            
            ml_pred = self.ml_model.predict(X_context)
            ml_prob = self.ml_model.predict_proba(X_context)[:, 1] if hasattr(self.ml_model, 'predict_proba') else ml_pred
            
            for i, gene in enumerate(context_genes):
                predictions[gene] = ml_pred[i]
                confidence[gene] = ml_prob[i] if ml_pred[i] == 1 else 1 - ml_prob[i]
        
        if return_confidence:
            return predictions, confidence
        return predictions
    
    def evaluate_loo(self, n_samples: Optional[int] = None) -> Dict:
        """
        Evaluate model using leave-one-out cross-validation.
        
        Args:
            n_samples: Number of cell lines to evaluate (None = all)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        cell_lines = self.cell_line_names[:n_samples] if n_samples else self.cell_line_names
        
        all_y_true = []
        all_y_pred = []
        
        print(f"Evaluating on {len(cell_lines)} cell lines...")
        
        for i, cell_line in enumerate(cell_lines):
            y_true = self.essentiality_matrix[cell_line].values
            y_pred = self.predict(cell_line).values
            
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(cell_lines)}...")
        
        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'n_samples': len(y_true),
            'n_cell_lines': len(cell_lines),
        }
        
        print(f"\nResults:")
        print(f"  Accuracy: {results['accuracy']:.3f}")
        print(f"  Balanced Accuracy: {results['balanced_accuracy']:.3f}")
        print(f"  F1 Score: {results['f1_score']:.3f}")
        
        return results
    
    def get_gene_classification(self) -> pd.DataFrame:
        """
        Get gene classification (pan-essential, context-dependent, non-essential).
        
        Returns:
            DataFrame with gene classifications
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        df = pd.DataFrame({
            'gene': self.gene_names,
            'consensus': self.gene_consensus.values,
            'variance': self.gene_variance.values,
        })
        
        def classify(row):
            if row['consensus'] >= self.consensus_high:
                return 'pan_essential'
            elif row['consensus'] <= self.consensus_low:
                return 'non_essential'
            else:
                return 'context_dependent'
        
        df['category'] = df.apply(classify, axis=1)
        
        return df.sort_values('consensus', ascending=False)


# =============================================================================
# QUICK DEMO WITH SAMPLE DATA
# =============================================================================

def demo_with_sample_data():
    """Run a quick demo using sample data."""
    print("="*70)
    print("DEMO: Human Essentiality Predictor")
    print("="*70)
    
    # Load sample data
    try:
        dep_mat = pd.read_csv('/home/claude/human_binary_essentiality.csv', index_col=0)
        print(f"\nLoaded sample data: {dep_mat.shape[0]} genes × {dep_mat.shape[1]} cell lines")
    except FileNotFoundError:
        print("Sample data not found. Please run the data loading script first.")
        return
    
    # Initialize predictor
    predictor = HumanEssentialityPredictor(
        consensus_high_threshold=0.9,
        consensus_low_threshold=0.1,
        ml_model='logistic',
    )
    
    # Fit (data is already binary)
    predictor.essentiality_matrix = dep_mat
    predictor.gene_names = list(dep_mat.index)
    predictor.cell_line_names = list(dep_mat.columns)
    predictor.gene_consensus = dep_mat.mean(axis=1)
    predictor.gene_variance = dep_mat.var(axis=1)
    predictor._train_ml_model()
    predictor.is_fitted = True
    
    # Evaluate
    results = predictor.evaluate_loo(n_samples=10)
    
    # Show gene classification
    gene_class = predictor.get_gene_classification()
    print("\nGene classification summary:")
    print(gene_class['category'].value_counts())
    
    return predictor, results


if __name__ == '__main__':
    demo_with_sample_data()
