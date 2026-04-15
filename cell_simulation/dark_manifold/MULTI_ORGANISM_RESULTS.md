# Multi-Organism Generalization Results

## Summary

The Universal Adaptive Predictor improves FBA-based gene essentiality prediction
across phylogenetically diverse bacteria without organism-specific rules.

### Actual Test Results

| Organism | Phylum | Genes | Essential % | FBA Baseline | Adaptive | Improvement |
|----------|--------|-------|-------------|--------------|----------|-------------|
| E. coli K-12 | Proteobacteria | 1,295 | 8.0% | 58.0% | 58.8% | **+0.8%** |
| JCVI-syn3A | Tenericutes | 90 | 91.1% | 69.5% | 78.2% | **+8.7%** |
| Salmonella | Proteobacteria | 1,707 | — | (no ground truth) | — | — |

**Both organisms with ground truth improved. No organism degraded.**

**Average improvement: +4.7%**

## How It Works

### The Key Innovation

The adaptive predictor uses FBA's own prediction rate as a proxy for class balance:

```python
def get_adaptive_thresholds(fba_essential_rate):
    if fba_essential_rate > 0.5:  # Minimal genome (Mycoplasma-like)
        # Aggressive kinetic corrections (catch more essentials)
        kinetic_thresh = 0.5
        condition_thresh = 0.2
    elif fba_essential_rate < 0.2:  # Complex genome (E. coli-like)
        # Aggressive condition-dependent corrections (remove false positives)
        kinetic_thresh = 0.95
        condition_thresh = 0.8
```

### Correction Rules

1. **Kinetic Correction** (catches false negatives)
   - FBA says non-essential BUT high biomass + translation/nucleotide gene
   - → Predict ESSENTIAL (kinetic bottleneck)
   - Works especially well on minimal genomes

2. **Condition-Dependent Correction** (removes false positives)
   - FBA says essential BUT cofactor/fermentation gene
   - → Predict NON-ESSENTIAL (condition-dependent)
   - Works especially well on complex genomes

## Detailed Results

### E. coli K-12 (Proteobacteria)

- Model: iJO1366
- Ground truth: Keio collection
- FBA essential rate: 15.2%
- Threshold regime: LOW_ESS (complex genome)

```
FBA:      TP= 31  FP=165  TN=1026  FN=73   Balanced Accuracy: 58.0%
Adaptive: TP= 29  FP=123  TN=1068  FN=75   Balanced Accuracy: 58.8%
```

**What happened:** Removed 42 false positives (cofactor/fermentation genes),
lost 2 true positives. Net: +42 correct predictions, +0.8% balanced accuracy.

### JCVI-syn3A (Tenericutes)

- Model: iMB155
- Ground truth: Hutchison 2016
- FBA essential rate: 85.6%
- Threshold regime: HIGH_ESS (minimal genome)

```
FBA:      TP=73  FP=4  TN=4  FN=9   Balanced Accuracy: 69.5%
Adaptive: TP=77  FP=3  TN=5  FN=5   Balanced Accuracy: 78.2%
```

**What happened:** Caught 4 kinetic-essential nucleotide genes:
- JCVISYN3A_0317 (nucleotide metabolism)
- JCVISYN3A_0005 (nucleotide metabolism)
- JCVISYN3A_0629 (nucleotide metabolism)
- JCVISYN3A_0381 (nucleotide metabolism)

All 4 corrections were correct. Net: +4 correct predictions, +8.7% balanced accuracy.

## Why This Works

The same algorithm adapts to both extremes because:

1. **Minimal genomes** have high essentiality (>80%) so FBA misses kinetic bottlenecks
   → We aggressively catch translation/nucleotide genes

2. **Complex genomes** have low essentiality (~8%) so FBA creates false positives
   → We aggressively remove condition-dependent genes

The FBA essential rate tells us which type of organism we have, and we adjust accordingly.

## Organisms Tested

### With Ground Truth (Actual Validation)
1. E. coli K-12 (Proteobacteria) ✓
2. JCVI-syn3A (Tenericutes) ✓

### FBA Only (No Ground Truth)
3. Salmonella typhimurium (Proteobacteria)

## Usage

```python
from dark_manifold.models.universal_adaptive import adaptive_predict, extract_gene_categories

# Prepare FBA results
fba_results = {
    'gene1': {'fba_essential': True, 'ratio': 0.0},
    'gene2': {'fba_essential': False, 'ratio': 0.95},
    # ...
}

# Prepare gene features
gene_features = {
    'gene1': {'categories': {'translation'}, 'single_gene_frac': 1.0},
    'gene2': {'categories': {'nucleotide'}, 'single_gene_frac': 0.8},
    # ...
}

# Run adaptive prediction
predictions, rules, thresholds = adaptive_predict(fba_results, gene_features)

print(f"Regime: {thresholds.regime}")
print(f"Kinetic corrections: {sum(1 for r in rules.values() if r == 'kinetic')}")
print(f"Condition corrections: {sum(1 for r in rules.values() if r == 'condition')}")
```

## Limitations

1. **Only 2 organisms validated** with ground truth
2. **FBA's fundamental blind spots remain:**
   - Transport genes (exchange reactions bypass transporters)
   - Regulatory essentiality (outside FBA scope)
   - Kinetic bottlenecks beyond translation/nucleotide

3. **For >70% balanced accuracy, need:**
   - Ortholog transfer from related organisms
   - Expression/proteomics data integration
   - Multi-organism machine learning

## Files

- `models/universal_adaptive.py` - The Universal Adaptive Predictor
- `models/fba.py` - FBA wrapper for JCVI-syn3A
- `data/salmonella_fba_results.json` - Salmonella FBA knockout results
- `data/essentiality.py` - JCVI-syn3A ground truth

## Citation

This work is part of the Dark Manifold Virtual Cell project.
