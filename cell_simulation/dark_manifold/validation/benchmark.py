"""
dark_manifold/validation/benchmark.py

Validation benchmark against Hutchison 2016 essentiality data.

CRITICAL: Run this after EVERY code change.
If accuracy drops below 85.6%, REJECT the change.

Usage:
    python -m dark_manifold.validation.benchmark
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..data.essentiality import (
    GENE_ESSENTIALITY, 
    get_gene_name,
    get_labeled_genes,
    is_essential,
)
from ..models.fba import FBAModel, get_fba_model


@dataclass
class BenchmarkResult:
    """Results from running the benchmark."""
    accuracy: float
    sensitivity: float  # True positive rate (essential genes)
    specificity: float  # True negative rate (non-essential genes)
    tp: int
    fp: int
    tn: int
    fn: int
    total_genes: int
    total_time_ms: float
    time_per_gene_ms: float
    predictions: List[Dict]
    
    def passed(self, threshold: float = 0.855) -> bool:
        """Check if benchmark passed the accuracy threshold."""
        return self.accuracy >= threshold
    
    def __str__(self) -> str:
        status = "✓ PASSED" if self.passed() else "✗ FAILED"
        return (
            f"\n{'='*60}\n"
            f"BENCHMARK RESULT: {status}\n"
            f"{'='*60}\n"
            f"Accuracy: {self.accuracy*100:.1f}%\n"
            f"  TP={self.tp}, FP={self.fp}, TN={self.tn}, FN={self.fn}\n"
            f"  Sensitivity (essential genes): {self.sensitivity*100:.1f}%\n"
            f"  Specificity (non-essential): {self.specificity*100:.1f}%\n"
            f"Time: {self.total_time_ms:.1f}ms ({self.time_per_gene_ms:.2f}ms/gene)\n"
            f"{'='*60}"
        )


def run_benchmark(
    model: Optional[FBAModel] = None,
    verbose: bool = True,
    threshold: float = 0.856,
) -> BenchmarkResult:
    """
    Run benchmark against Hutchison 2016 essentiality data.
    
    Args:
        model: FBA model to test. If None, uses default.
        verbose: Print detailed results.
        threshold: Minimum accuracy to pass (default: 85.6% = V37 baseline).
    
    Returns:
        BenchmarkResult with accuracy metrics.
    
    Raises:
        AssertionError if accuracy below threshold.
    """
    if model is None:
        model = get_fba_model(verbose=False)
    
    if verbose:
        print("\n" + "="*60)
        print("ESSENTIALITY BENCHMARK")
        print("Ground truth: Hutchison et al. 2016 Science")
        print("="*60)
    
    start_time = time.time()
    
    tp, fp, tn, fn = 0, 0, 0, 0
    predictions = []
    
    # Get genes that are both in model and have experimental labels
    model_genes = set(model.get_genes())
    labeled_genes = set(get_labeled_genes())
    test_genes = sorted(model_genes & labeled_genes)
    
    for gene in test_genes:
        result = model.knockout(gene)
        pred_essential = result['essential']
        true_essential = is_essential(gene)
        
        # Confusion matrix
        if true_essential:
            if pred_essential:
                tp += 1
                match = '✓'
            else:
                fn += 1
                match = '✗'
        else:
            if not pred_essential:
                tn += 1
                match = '✓'
            else:
                fp += 1
                match = '✗'
        
        predictions.append({
            'gene': gene,
            'name': get_gene_name(gene),
            'predicted_essential': pred_essential,
            'true_essential': true_essential,
            'correct': pred_essential == true_essential,
            'biomass_ratio': result['biomass_ratio'],
        })
        
        if verbose:
            gene_name = get_gene_name(gene)
            status = "ESSENTIAL" if pred_essential else f"viable ({result['biomass_ratio']:.0%})"
            exp_str = "essential" if true_essential else "non-ess"
            print(f"  Δ{gene_name:8s}: {status:20s} | exp: {exp_str:9s} [{match}]")
    
    total_time_ms = (time.time() - start_time) * 1000
    
    # Compute metrics
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    result = BenchmarkResult(
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        tp=tp, fp=fp, tn=tn, fn=fn,
        total_genes=total,
        total_time_ms=total_time_ms,
        time_per_gene_ms=total_time_ms / total if total > 0 else 0,
        predictions=predictions,
    )
    
    if verbose:
        print(result)
    
    return result


def validate_model(model: Optional[FBAModel] = None, threshold: float = 0.856) -> bool:
    """
    Quick validation that model meets accuracy threshold.
    
    Use this in CI/CD or after changes.
    
    Returns:
        True if passed, False otherwise.
    """
    result = run_benchmark(model, verbose=False, threshold=threshold)
    return result.passed(threshold)


def assert_baseline(model: Optional[FBAModel] = None):
    """
    Assert that model meets V37 baseline (85.6%).
    
    Raises AssertionError if accuracy is below baseline.
    Use this as a gate before merging changes.
    """
    result = run_benchmark(model, verbose=False)
    assert result.passed(), (
        f"BASELINE FAILED: Accuracy {result.accuracy*100:.1f}% < 85.6%\n"
        f"Reject this change and debug."
    )
    print(f"✓ Baseline passed: {result.accuracy*100:.1f}% accuracy")


def get_error_analysis(model: Optional[FBAModel] = None) -> Dict:
    """
    Detailed analysis of prediction errors.
    
    Returns dict with false positives and false negatives.
    """
    result = run_benchmark(model, verbose=False)
    
    false_positives = [
        p for p in result.predictions 
        if p['predicted_essential'] and not p['true_essential']
    ]
    false_negatives = [
        p for p in result.predictions 
        if not p['predicted_essential'] and p['true_essential']
    ]
    
    return {
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'fp_count': len(false_positives),
        'fn_count': len(false_negatives),
        'accuracy': result.accuracy,
    }


def print_error_analysis():
    """Print detailed error analysis."""
    analysis = get_error_analysis()
    
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    print(f"\nFalse Positives ({analysis['fp_count']}) - predicted essential but not:")
    for p in analysis['false_positives']:
        print(f"  {p['name']:10s} ({p['gene']}) - biomass ratio: {p['biomass_ratio']:.2%}")
    
    print(f"\nFalse Negatives ({analysis['fn_count']}) - missed essential genes:")
    for p in analysis['false_negatives']:
        print(f"  {p['name']:10s} ({p['gene']}) - biomass ratio: {p['biomass_ratio']:.2%}")
    
    print(f"\nThese {analysis['fp_count'] + analysis['fn_count']} errors are what the neural layer needs to fix.")


if __name__ == "__main__":
    print("="*60)
    print("DARK MANIFOLD VALIDATION BENCHMARK")
    print("="*60)
    
    result = run_benchmark(verbose=True)
    
    if result.passed():
        print("\n✓ BENCHMARK PASSED")
        print_error_analysis()
    else:
        print("\n✗ BENCHMARK FAILED")
        print(f"  Required: ≥85.6%")
        print(f"  Achieved: {result.accuracy*100:.1f}%")
        exit(1)
