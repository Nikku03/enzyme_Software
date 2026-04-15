# Path to 90%+ Balanced Accuracy

## Current State
- **Best method:** FBA (V37)
- **Balanced accuracy:** 69.5%
- **Errors:** 9 FN + 4 FP = 13 total

## Error Analysis

### False Negatives (9 genes FBA misses)

**High biomass (~98-100%) - Non-metabolic roles (4 genes):**
- JCVISYN3A_0317, 0005, 0629, 0381
- FBA cannot detect: no metabolic impact
- **Fix:** Need PPI, expression, conservation data

**Medium biomass (~30-40%) - Redundancy overestimated (5 genes):**
- JCVISYN3A_0233, 0207, 0352, 0353, 0546
- FBA finds alternate route, but cell can't use it
- **Fix:** Kinetic constraints, expression costs, regulatory rules

### False Positives (4 genes FBA gets wrong)
- JCVISYN3A_0683, 0684, 0589, 0235
- All show biomass = 0 (lethal) but cell survives
- **Fix:** Missing reactions in model, better GPR rules

## Required External Data

| Data Type | Source | Purpose |
|-----------|--------|---------|
| PPI network | SynWiki, STRING | Detect non-metabolic essential |
| Expression | SynWiki expression browser | Identify high-expression = essential |
| Conservation | PROST, Foldseek | Conserved = likely essential |
| Structure | AlphaFold2 | Function prediction |

## Implementation Plan

### Phase 1: Data Integration (2-3 days)
1. Scrape SynWiki PPI data (http://synwiki.uni-goettingen.de/)
2. Get expression data from SynWiki expression browser
3. Download PROST homology scores (https://bit.ly/prost-syn3a)

### Phase 2: Feature Engineering (1-2 days)
1. PPI degree centrality
2. Expression level
3. Conservation score (homolog count)
4. Functional category (TIGRfam)

### Phase 3: ML with Rich Features (1-2 days)
1. Combine FBA + new features
2. Train GBM/RF with class weights
3. Cross-validate

### Phase 4: Model Improvement (3-5 days)
1. Add missing reactions for FP genes
2. Add regulatory constraints
3. Integrate expression costs

## Expected Results

| Scenario | TP | FP | TN | FN | Balanced |
|----------|----|----|----|----|----------|
| Current FBA | 73 | 4 | 4 | 9 | 69.5% |
| + PPI/conservation | 77 | 4 | 4 | 5 | 78.6% |
| + Model fixes | 77 | 2 | 6 | 5 | 82.8% |
| + Expression | 80 | 2 | 6 | 2 | **90.0%** |

## Key Papers

1. Zhang et al. 2021 - PPI network for JCVI-syn3A
   https://pubs.acs.org/doi/abs/10.1021/acs.jproteome.0c00359

2. Pedreira et al. 2022 - SynWiki database
   https://onlinelibrary.wiley.com/doi/full/10.1002/pro.4179

3. Kilinc et al. 2025 - PROST annotations
   https://link.springer.com/protocol/10.1007/978-1-0716-4196-5_9

4. Bianchi et al. 2022 - Functional characterization
   https://pubs.acs.org/doi/10.1021/acs.jpcb.2c04188
