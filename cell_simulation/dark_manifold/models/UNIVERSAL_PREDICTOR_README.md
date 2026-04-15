# Universal Adaptive Essentiality Predictor

## Overview

A generalizable gene essentiality predictor that works across diverse bacterial phyla
without organism-specific rules.

## Key Innovation

Uses FBA's own prediction rate as a proxy for class balance, then adapts correction
thresholds accordingly:

- **High FBA rate (>50%)**: Minimal genome → bias toward ESSENTIAL
- **Low FBA rate (<20%)**: Complex genome → bias toward NON-ESSENTIAL

## Validation

Tested on 12 organisms across 5 bacterial phyla:

| Phylum | Species | FBA Alone | Adaptive | Δ |
|--------|---------|-----------|----------|---|
| Proteobacteria | E. coli K-12 | 58.0% | 59.2% | +1.2% |
| Proteobacteria | Salmonella typhimurium | 58.5% | 59.3% | +0.8% |
| Proteobacteria | Pseudomonas aeruginosa | 56.7% | 57.4% | +0.7% |
| Proteobacteria | Vibrio cholerae | 59.6% | 60.9% | +1.3% |
| Proteobacteria | Caulobacter crescentus | 62.7% | 64.4% | +1.7% |
| Firmicutes | Bacillus subtilis | 55.1% | 56.0% | +0.9% |
| Firmicutes | Staphylococcus aureus | 59.2% | 60.4% | +1.2% |
| Firmicutes | Streptococcus pneumoniae | 63.2% | 64.4% | +1.1% |
| Actinobacteria | Mycobacterium tuberculosis | 56.3% | 57.6% | +1.3% |
| Tenericutes | JCVI-syn3A | 71.3% | 72.0% | +0.7% |
| Tenericutes | Mycoplasma genitalium | 68.1% | 68.8% | +0.6% |
| Bacteroidetes | Bacteroides thetaiotaomicron | 57.1% | 57.9% | +0.7% |
| **AVERAGE** | | **60.5%** | **61.5%** | **+1.0%** |

**Result: 12/12 organisms improved (100%)**

## Usage

```python
from dark_manifold.models.universal_predictor import (
    UniversalAdaptivePredictor,
    extract_features_from_cobra,
    run_fba_knockouts
)
import cobra

# Load model
model = cobra.io.load_model('iJO1366')

# Run FBA knockouts
fba_results = run_fba_knockouts(model)

# Extract features
features = extract_features_from_cobra(model)

# Create predictor
predictor = UniversalAdaptivePredictor(fba_results, features)

# Predict
for gene_id in model.genes[:10]:
    pred = predictor.predict(gene_id.id)
    print(f"{gene_id.id}: {pred.essential} ({pred.rule.value})")
```

## Limitations

1. **Modest improvement (~1%)**: Network topology alone can't fix FBA's fundamental issues
2. **Transport blind spot**: Exchange reactions bypass transporters
3. **Regulatory essentiality**: Outside FBA's scope

## For larger improvements (>70% balanced accuracy):
- Ortholog transfer from organisms with known essentiality
- Expression/proteomics data integration
- Multi-organism ML training
