# Dark Manifold: Project Plan
## 90% Essentiality Predictor + Synthetic Lethality Screen

**Goal:** Publishable tool in 8-10 weeks
**Target:** Bioinformatics / NAR / PLOS Computational Biology

---

## PHASE 1: CONSOLIDATION (Week 1)
### Goal: Single clean codebase with automated validation

### Task 1.1: Create Unified Module Structure
**Status: NOT STARTED**
**Time: 2 days**

```
dark_manifold/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── imb155.py          # Stoichiometry from V37
│   ├── essentiality.py    # Hutchison 2016 ground truth
│   └── kinetics.py        # BRENDA parameters from V38
├── models/
│   ├── __init__.py
│   ├── fba.py             # V37 FBA core (DON'T MODIFY)
│   ├── kinetic.py         # V38 kinetics (simplified)
│   ├── expression.py      # V47 gene expression
│   └── neural.py          # NEW: Neural refinement
├── analysis/
│   ├── __init__.py
│   ├── knockout.py        # Single gene knockout
│   ├── synthetic_lethal.py # Gene pair screening
│   └── visualization.py   # Plotting utilities
├── validation/
│   ├── __init__.py
│   ├── benchmark.py       # Run against Hutchison 2016
│   └── external.py        # M. genitalium validation
└── tests/
    ├── test_fba.py
    ├── test_neural.py
    └── test_integration.py
```

**What to copy:**
- [ ] V37 `core_cell_simulator.py` → `models/fba.py` + `data/imb155.py`
- [ ] V37 essentiality dict → `data/essentiality.py`
- [ ] V38 `KINETIC_PARAMS` → `data/kinetics.py`
- [ ] V47 `gene_expression_v2.py` → `models/expression.py`

**What to delete (archive to `archive/` branch):**
- [ ] All 43 notebooks (keep only 1 final one)
- [ ] V3-V36, V39-V46, V48-V53 directories
- [ ] Duplicate files

---

### Task 1.2: Automated Validation Script
**Status: NOT STARTED**
**Time: 1 day**

```python
# validation/benchmark.py
def run_benchmark(model, method='fba'):
    """
    Run against Hutchison 2016 ground truth.
    MUST be run after every code change.
    """
    results = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    
    for gene, true_ess in HUTCHISON_2016.items():
        pred = model.predict_essentiality(gene, method=method)
        # ... count results
    
    accuracy = (results['tp'] + results['tn']) / total
    
    # HARD REQUIREMENT
    assert accuracy >= 0.856, f"Accuracy {accuracy:.1%} below V37 baseline (85.6%)"
    
    return results
```

**Deliverable:** `python -m dark_manifold.validation.benchmark` runs in <5 seconds

---

### Task 1.3: Clean Repository
**Status: NOT STARTED**  
**Time: 0.5 days**

- [ ] Create `archive` branch with all old versions
- [ ] Delete old versions from `main`
- [ ] Update `.gitignore`
- [ ] Write `README.md` with installation + usage
- [ ] Add `requirements.txt`

---

## PHASE 2: NEURAL REFINEMENT LAYER (Weeks 2-3)
### Goal: Improve from 85.6% → 90%+ accuracy

### Task 2.1: Feature Engineering
**Status: NOT STARTED**
**Time: 3 days**

Features that FBA alone doesn't capture:

```python
def extract_gene_features(gene: str, fba_result: dict) -> np.ndarray:
    """
    Extract features for neural refinement.
    """
    return np.array([
        # From FBA
        fba_result['biomass_ratio'],           # 0-1, how much biomass drops
        fba_result['num_blocked_reactions'],   # int, cascade size
        
        # Network topology (NEW)
        get_centrality(gene),                  # How connected is this gene?
        get_betweenness(gene),                 # Is it a bottleneck?
        count_isozymes(gene),                  # Redundancy
        
        # Expression (from V47)
        get_expression_level(gene),            # High/low expression
        get_protein_halflife(gene),            # Stability
        
        # Thermodynamics (from V38)
        get_reaction_delta_g(gene),            # Reversibility
        is_near_equilibrium(gene),             # Can flux reroute?
        
        # Pathway context
        is_in_essential_pathway(gene),         # Glycolysis, etc.
        distance_to_biomass(gene),             # Hops to biomass reaction
    ])
```

**Deliverable:** `features.py` with 10-15 validated features

---

### Task 2.2: Build Neural Refinement Model
**Status: NOT STARTED**
**Time: 4 days**

```python
# models/neural.py
import torch
import torch.nn as nn

class EssentialityRefiner(nn.Module):
    """
    Learn corrections to FBA predictions.
    
    Input: gene features (15-dim)
    Output: essentiality score (0-1)
    """
    def __init__(self, input_dim=15, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


class NeuralRefinedPredictor:
    """
    FBA + Neural refinement.
    """
    def __init__(self, fba_model, neural_model):
        self.fba = fba_model
        self.neural = neural_model
    
    def predict(self, gene: str) -> dict:
        # Base FBA prediction
        fba_result = self.fba.knockout(gene)
        fba_score = fba_result['biomass_ratio']
        
        # Extract features
        features = extract_gene_features(gene, fba_result)
        
        # Neural refinement
        with torch.no_grad():
            neural_score = self.neural(torch.tensor(features)).item()
        
        # Combine (weighted)
        combined = 0.7 * (1 - fba_score) + 0.3 * neural_score
        
        return {
            'essential': combined > 0.5,
            'confidence': abs(combined - 0.5) * 2,
            'fba_score': fba_score,
            'neural_score': neural_score,
            'combined_score': combined
        }
```

**Deliverable:** Working `NeuralRefinedPredictor` class

---

### Task 2.3: Training Pipeline
**Status: NOT STARTED**
**Time: 3 days**

```python
# Training with cross-validation
def train_refiner(genes, labels, n_folds=5):
    """
    Train neural refiner with leave-one-out or k-fold CV.
    """
    from sklearn.model_selection import StratifiedKFold
    
    results = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(genes, labels)):
        # Train on fold
        model = EssentialityRefiner()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(100):
            # ... training loop
            pass
        
        # Validate
        val_acc = evaluate(model, genes[val_idx], labels[val_idx])
        results.append(val_acc)
    
    return np.mean(results), np.std(results)
```

**Target:** 90%+ accuracy in cross-validation

---

### Task 2.4: Error Analysis
**Status: NOT STARTED**
**Time: 2 days**

Understand what V37 gets wrong:

```
V37 Confusion Matrix (current):
                  Predicted
                  Ess    Non-Ess
Actual  Ess       73     9        (FN: missed essential)
        Non-Ess   4      4        (FP: false essential)

Key questions:
1. Which 9 essential genes does FBA miss? Why?
2. Which 4 non-essential genes does FBA wrongly call essential?
3. Can features distinguish these cases?
```

**Deliverable:** Document explaining each error case + feature that could fix it

---

## PHASE 3: SYNTHETIC LETHALITY (Weeks 4-5)
### Goal: Screen all gene pairs, find novel interactions

### Task 3.1: Pairwise Knockout Screen
**Status: NOT STARTED**
**Time: 3 days**

```python
# analysis/synthetic_lethal.py
def screen_synthetic_lethality(model, genes: list) -> pd.DataFrame:
    """
    Screen all gene pairs for synthetic lethality.
    
    SL = both non-essential alone, but essential together
    """
    # Filter to non-essential genes only
    non_essential = [g for g in genes if not model.predict(g)['essential']]
    
    results = []
    n = len(non_essential)
    total_pairs = n * (n - 1) // 2
    
    for i, gene_a in enumerate(non_essential):
        for gene_b in non_essential[i+1:]:
            # Double knockout
            double_result = model.double_knockout(gene_a, gene_b)
            
            if double_result['essential']:
                results.append({
                    'gene_a': gene_a,
                    'gene_b': gene_b,
                    'biomass_ratio': double_result['biomass_ratio'],
                    'confidence': double_result['confidence'],
                    'pathway_a': get_pathway(gene_a),
                    'pathway_b': get_pathway(gene_b),
                })
    
    return pd.DataFrame(results)
```

**Deliverable:** List of predicted synthetic lethal pairs

---

### Task 3.2: Literature Validation
**Status: NOT STARTED**
**Time: 2 days**

- [ ] Search for known synthetic lethal pairs in minimal cells
- [ ] Cross-reference predictions with literature
- [ ] Calculate precision/recall on known pairs

**Sources:**
- Hutchison 2016 supplementary data
- DEG database
- SynLethDB (if minimal cell data exists)

---

### Task 3.3: Prioritize Novel Predictions
**Status: NOT STARTED**
**Time: 2 days**

```python
def prioritize_novel_sl_pairs(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Rank SL pairs by:
    1. Confidence (neural model certainty)
    2. Biological interest (different pathways = more interesting)
    3. Druggability (if both genes are potential targets)
    """
    predictions['novelty_score'] = (
        predictions['confidence'] * 0.4 +
        predictions['cross_pathway'] * 0.3 +
        predictions['druggability'] * 0.3
    )
    return predictions.sort_values('novelty_score', ascending=False)
```

**Deliverable:** Top 20 novel SL pairs for potential experimental validation

---

## PHASE 4: EXTERNAL VALIDATION (Week 6)
### Goal: Test on organisms outside training data

### Task 4.1: M. genitalium Essentiality Data
**Status: NOT STARTED**
**Time: 2 days**

- [ ] Download M. genitalium essentiality data (Glass et al. 2006)
- [ ] Map genes between M. genitalium and JCVI-syn3A (high homology)
- [ ] Run predictions on mapped genes
- [ ] Calculate accuracy on held-out organism

**Target:** >80% accuracy on external organism

---

### Task 4.2: Cross-Species Analysis
**Status: NOT STARTED**
**Time: 2 days**

```python
def cross_species_validation(model, target_species: str):
    """
    Test model on different species.
    """
    # Load target species data
    genes, labels = load_species_data(target_species)
    
    # Map to syn3A features (by homology)
    mapped_features = map_by_homology(genes, 'JCVI-syn3A')
    
    # Predict
    predictions = [model.predict(f) for f in mapped_features]
    
    # Evaluate
    accuracy = calculate_accuracy(predictions, labels)
    
    return {
        'species': target_species,
        'n_genes': len(genes),
        'n_mapped': len(mapped_features),
        'accuracy': accuracy
    }
```

---

## PHASE 5: PAPER WRITING (Weeks 7-8)
### Goal: Submit to journal

### Task 5.1: Methods Section
**Status: NOT STARTED**
**Time: 2 days**

Sections:
1. Data sources (iMB155, Hutchison 2016, BRENDA)
2. FBA implementation
3. Feature engineering
4. Neural refinement architecture
5. Training procedure (cross-validation)
6. Synthetic lethality screening
7. External validation

---

### Task 5.2: Results Section
**Status: NOT STARTED**
**Time: 2 days**

Figures:
1. Architecture diagram (FBA + Neural layers)
2. ROC curve: FBA vs Neural-refined
3. Confusion matrices
4. Feature importance analysis
5. Synthetic lethality network graph
6. Cross-species validation bar chart

---

### Task 5.3: Figures and Tables
**Status: NOT STARTED**
**Time: 3 days**

| Figure | Content | Tool |
|--------|---------|------|
| Fig 1 | Architecture diagram | SVG/Illustrator |
| Fig 2 | Accuracy comparison | matplotlib |
| Fig 3 | Feature importance | SHAP + matplotlib |
| Fig 4 | SL network | networkx + Cytoscape |
| Fig 5 | External validation | matplotlib |
| Table 1 | Dataset statistics | LaTeX |
| Table 2 | Performance metrics | LaTeX |
| Table 3 | Top SL predictions | LaTeX |

---

### Task 5.4: Writing and Submission
**Status: NOT STARTED**
**Time: 3 days**

- [ ] Draft abstract
- [ ] Draft introduction
- [ ] Compile supplementary materials
- [ ] Format for target journal
- [ ] Submit

---

## SUMMARY: STATUS TRACKER

| Phase | Task | Status | % Done | Time Est |
|-------|------|--------|--------|----------|
| **1. Consolidation** | | | **0%** | **5 days** |
| | 1.1 Unified module structure | NOT STARTED | 0% | 2 days |
| | 1.2 Validation script | NOT STARTED | 0% | 1 day |
| | 1.3 Clean repository | NOT STARTED | 0% | 0.5 days |
| **2. Neural Refinement** | | | **0%** | **12 days** |
| | 2.1 Feature engineering | NOT STARTED | 0% | 3 days |
| | 2.2 Neural model | NOT STARTED | 0% | 4 days |
| | 2.3 Training pipeline | NOT STARTED | 0% | 3 days |
| | 2.4 Error analysis | NOT STARTED | 0% | 2 days |
| **3. Synthetic Lethality** | | | **0%** | **7 days** |
| | 3.1 Pairwise screen | NOT STARTED | 0% | 3 days |
| | 3.2 Literature validation | NOT STARTED | 0% | 2 days |
| | 3.3 Prioritize novel pairs | NOT STARTED | 0% | 2 days |
| **4. External Validation** | | | **0%** | **4 days** |
| | 4.1 M. genitalium data | NOT STARTED | 0% | 2 days |
| | 4.2 Cross-species analysis | NOT STARTED | 0% | 2 days |
| **5. Paper** | | | **0%** | **10 days** |
| | 5.1 Methods | NOT STARTED | 0% | 2 days |
| | 5.2 Results | NOT STARTED | 0% | 2 days |
| | 5.3 Figures | NOT STARTED | 0% | 3 days |
| | 5.4 Submission | NOT STARTED | 0% | 3 days |

**TOTAL: ~38 working days = 8 weeks**

---

## WHAT'S ALREADY DONE (Assets to Reuse)

| Asset | Location | Status | Reuse |
|-------|----------|--------|-------|
| FBA implementation | `v37_full_imb155/core_cell_simulator.py` | ✅ Working | Copy to `models/fba.py` |
| Stoichiometry matrix | `v37_full_imb155/core_cell_simulator.py` | ✅ 338 reactions | Extract to `data/imb155.py` |
| Essentiality ground truth | `v37_full_imb155/core_cell_simulator.py` | ✅ 90 genes | Extract to `data/essentiality.py` |
| Gene database | `v47_gene_expression/gene_expression_v2.py` | ✅ 177 genes | Copy to `models/expression.py` |
| Kinetic parameters | `v38_kinetic/kinetic_cell.py` | ✅ 16 enzymes | Extract to `data/kinetics.py` |
| Benchmark result | V37 run | ✅ 85.6% accuracy | Baseline to beat |

---

## DEPENDENCIES

```
# requirements.txt
numpy>=1.21
scipy>=1.7
pandas>=1.3
torch>=1.10
scikit-learn>=1.0
networkx>=2.6
matplotlib>=3.5
seaborn>=0.11
```

---

## MILESTONES

| Week | Milestone | Success Criteria |
|------|-----------|------------------|
| 1 | Clean codebase | `python -m dark_manifold.validation.benchmark` passes |
| 3 | Neural refinement | 90%+ accuracy in cross-validation |
| 5 | Synthetic lethality | Top 20 novel pairs identified |
| 6 | External validation | >80% on M. genitalium |
| 8 | Paper submitted | Manuscript to Bioinformatics |

---

## RISKS AND MITIGATIONS

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Can't reach 90% accuracy | Medium | High | Focus on specific error cases, add more features |
| No known SL pairs to validate | Low | Medium | Use pathway logic as proxy validation |
| M. genitalium mapping fails | Low | Medium | Use E. coli essential genes instead |
| Paper rejected | Medium | Medium | Have backup journals ready (PLOS ONE, BMC Bioinf) |

---

## NEXT IMMEDIATE STEPS

**Today:**
1. Create `dark_manifold/` directory structure
2. Copy V37 FBA code to `models/fba.py`
3. Extract essentiality data to `data/essentiality.py`
4. Write `validation/benchmark.py`
5. Verify 85.6% accuracy still holds

**This week:**
1. Complete Phase 1 consolidation
2. Archive old versions to branch
3. Start feature engineering for neural layer
