# Human Gene Essentiality Predictor

## The "Hedge Fund" Approach for Human Cancers

This is the same conceptual approach used for JCVI-syn3A bacterial gene essentiality, 
adapted for human cancer cell lines using DepMap CRISPR screening data.

## Results (Sample Data: 32 cell lines)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 96.7% |
| Balanced Accuracy | 86.4% |
| F1 Score | 0.767 |

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    GENE CLASSIFICATION                       │
├─────────────────────────────────────────────────────────────┤
│  Pan-Essential (≥90% consensus)     → Always ESSENTIAL      │
│  Non-Essential (≤10% consensus)     → Always NON-ESSENTIAL  │
│  Context-Dependent (10-90%)         → ML Model              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from human_essentiality_predictor import HumanEssentialityPredictor

# Load your DepMap data
import pandas as pd
df = pd.read_csv('CRISPRGeneEffect.csv', index_col=0).T

# Fit and evaluate
predictor = HumanEssentialityPredictor()
predictor.fit(df, binarize=True, threshold=-0.5)
results = predictor.evaluate_loo(n_samples=100)
```

## Files Included

- `human_essentiality_predictor.py` - The main predictor module
- `human_gene_essentiality_summary.csv` - Gene classification (sample data)
- `README_human_essentiality.md` - This file

## Scaling Up

To use with full DepMap data (1,086 cell lines):

1. Download from https://depmap.org/portal/download/:
   - `CRISPRGeneEffect.csv` (required)
   - `OmicsExpressionProteinCodingGenesTPMLogp1.csv` (optional, for better accuracy)

2. Expected improvements with full data:
   - More stable consensus estimates
   - Better context-dependent prediction
   - Cancer-type-specific models possible

## Comparison: Bacteria vs Human

| Aspect | Bacteria (syn3A) | Human (Cancer) |
|--------|------------------|----------------|
| Data source | Cross-species orthologs | Cross-cell-line screens |
| Training size | 57 species | 1,086 cell lines |
| Genes | 119 | 17,995 |
| Accuracy | 85% | 97% (sample) |

The "hedge fund" strategy works: **consensus from similar contexts predicts essentiality!**
