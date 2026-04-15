"""
Dark Manifold: Neural-Enhanced Gene Essentiality Prediction

A hybrid FBA + Memory system for predicting gene essentiality in JCVI-syn3A.

Key components:
- FBA (V37): Physics-based flux balance analysis (85.6% baseline)
- Memory Bank: Similarity-based retrieval from known genes
- Rule Discovery: Interpretable patterns for essentiality

Combined approach achieves 88.9% accuracy on Hutchison 2016 benchmark.

Usage:
    from dark_manifold import predict_essentiality, evaluate
    
    # Quick prediction
    is_essential = predict_essentiality('JCVISYN3A_0207')
    
    # Full evaluation
    results = evaluate(verbose=True)

Author: Naresh Chhillar, 2026
"""

__version__ = '0.2.0'
__author__ = 'Naresh Chhillar'

# Lazy imports for faster startup
def get_fba_model(verbose=False):
    """Get the FBA model."""
    from .models.fba import get_fba_model as _get
    return _get(verbose=verbose)

def build_memory_bank(fba_model=None, verbose=False):
    """Build memory bank for similarity retrieval."""
    from .reasoning.gene_memory import build_memory_bank as _build
    return _build(fba_model, verbose=verbose)

def predict_essentiality(gene: str, method='combined') -> bool:
    """
    Predict if a gene is essential.
    
    Args:
        gene: Gene ID (e.g., 'JCVISYN3A_0207')
        method: 'fba', 'memory', or 'combined'
    
    Returns:
        True if predicted essential
    """
    if method == 'fba':
        from .models.fba import predict_essentiality as _pred
        return _pred(gene)
    elif method == 'memory':
        from .reasoning.gene_memory import build_memory_bank
        from .data.gene_features import GeneFeatureExtractor
        fba = get_fba_model()
        memory = build_memory_bank(fba)
        extractor = GeneFeatureExtractor(fba)
        features = extractor.extract(gene).features
        result = memory.predict_by_analogy(features, k=5, exclude_gene=gene)
        return result['essential_score'] > 0.5
    else:  # combined
        from .models.fba import get_fba_model
        from .reasoning.gene_memory import build_memory_bank
        from .data.gene_features import GeneFeatureExtractor
        
        fba = get_fba_model()
        memory = build_memory_bank(fba)
        extractor = GeneFeatureExtractor(fba)
        
        # FBA prediction
        fba_result = fba.knockout(gene)
        fba_pred = fba_result['essential']
        
        # Memory prediction
        features = extractor.extract(gene).features
        mem_result = memory.predict_by_analogy(features, k=5, exclude_gene=gene)
        mem_score = mem_result['essential_score']
        
        # Combined: FBA OR high-confidence memory
        return fba_pred or (mem_score > 0.6)

def evaluate(verbose=True):
    """Evaluate all methods against Hutchison 2016 benchmark."""
    from .validation.benchmark import run_benchmark
    return run_benchmark(verbose=verbose)

def run_benchmark(verbose=True):
    """Run the validation benchmark."""
    from .validation.benchmark import run_benchmark as _run
    return _run(verbose=verbose)

def info():
    """Print package information."""
    print(f"Dark Manifold v{__version__}")
    print(f"Author: {__author__}")
    print()
    print("Modules:")
    print("  - data.essentiality: Hutchison 2016 ground truth")
    print("  - data.gene_features: Gene feature extraction")
    print("  - models.fba: FBA core (V37, 85.6% accuracy)")
    print("  - models.gene_encoder: Neural gene encoder")
    print("  - models.combined_predictor: FBA + Memory ensemble")
    print("  - reasoning.gene_memory: Similarity-based retrieval")
    print("  - reasoning.gene_rules: Rule discovery")
    print("  - validation.benchmark: Hutchison benchmark")
