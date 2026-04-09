# CYP3A4 SoM Prediction: Path to 90% Top-1 Accuracy

## Current State
- **Best Top-1:** 48.6% (pre-phase champion)
- **Target:** 90% Top-1 on curated dataset
- **Gap:** +41.4 percentage points

## Root Causes Identified
1. **Label noise:** ~70% of gold hard-source cases have annotation problems
2. **Architecture bottleneck:** Scalar proposer head scores atoms independently, can't compare
3. **Source quality variance:** ATTNSOM/CYP_DBs_external severely underperform
4. **Training instability:** best_epoch=1 pattern persists

---

## Phase 0: Data Cleanup ✅ IN PROGRESS
**Goal:** Remove wrong labels, correct annotations, add confidence scores

### Tasks
- [ ] Remove wrong-enzyme molecules (nicotine, NNK, NNN)
- [ ] Correct zileuton annotation (→ sulfur atom)
- [ ] Correct diclofenac annotation (→ C5 position)
- [ ] Add `label_confidence` field (high/medium/low/wrong)
- [ ] Flag CYP3A4-minor molecules (mianserin, phenprocoumon, hydromorphone)
- [ ] Rebuild train/val/test splits with cleaned data
- [ ] Output: `cyp3a4_cleaned_dataset.json`

### Validation
- [ ] Verify atom indices match SMILES
- [ ] Cross-check with PubMed/DrugBank for corrected sites
- [ ] Document all changes in changelog

---

## Phase 1: Architecture Surgery ✅ COMPLETE
**Goal:** Replace scalar proposer with relational version

### Tasks
- [x] Implement `RelationalSelfAttention` (cross-atom comparison)
- [x] Implement `PairwiseAggregator` (proven 77% acc architecture)
- [x] Implement `RelationalProposer` (full proposer module)
- [x] Implement `RelationalFusionHead` (drop-in replacement)
- [x] Add config options (`use_relational_proposer`, etc.)
- [x] Integrate into model.py with backward compatibility
- [x] Unit tests pass
- [ ] A/B test on cleaned data (requires Colab GPU)
- [ ] Expected lift: +8-12% recall@K

### Files Created
- `src/enzyme_software/liquid_nn_v2/model/relational_proposer.py` (430 lines)
- `scripts/phase1_relational_proposer_train.py` (training harness)

### Parameter Comparison
| Head | Parameters | Notes |
|------|------------|-------|
| ResidualFusionHead (baseline) | 37,350 | Scalar scoring |
| RelationalFusionHead (phase1) | 168,292 | +cross-attention +pairwise |

### Usage
```python
# In config:
config.use_relational_proposer = True
config.relational_proposer_num_heads = 4
config.relational_proposer_num_layers = 2
config.relational_proposer_use_pairwise = True
```

---

## Phase 2: Pairwise Reranker Integration
**Goal:** Use proven pairwise head (77% acc) as inference reranker

### Tasks
- [ ] Load frozen pairwise head checkpoint
- [ ] Implement `pairwise_rerank()` function
- [ ] Integrate into eval pipeline
- [ ] Measure Top-1 lift from reranking alone
- [ ] Expected lift: +3-5%

---

## Phase 3: Loss Function Redesign
**Goal:** Unified loss that optimizes for recall@K directly

### Tasks
- [ ] Implement shortlist-aware ListMLE
- [ ] Add hard margin term (true vs hardest negative)
- [ ] Add recall@K reward signal
- [ ] Tune loss weights via grid search
- [ ] Expected lift: +2-4%

---

## Phase 4: Chemistry Features
**Goal:** Add CYP3A4-specific reactivity descriptors

### Tasks
- [ ] Fukui indices (nucleophilic/electrophilic)
- [ ] BDE estimates (xTB pipeline exists)
- [ ] α-carbon to nitrogen detection
- [ ] Benzylic/allylic position flags
- [ ] SASA (solvent accessible surface area)
- [ ] Distance to basic nitrogen
- [ ] Expected lift: +5-8%

---

## Phase 5: Source-Aware Training
**Goal:** Weight training samples by source reliability

### Source Weights (Initial)
| Source | Weight | Rationale |
|--------|--------|-----------|
| drugbank | 1.0 | High quality |
| metxbiodb | 0.9 | Good |
| peng_external | 0.8 | Published |
| rudik_external | 0.8 | Published |
| attnsom | 0.5 | Mixed quality |
| cyp_dbs_external | 0.3 | Problematic |

### Tasks
- [ ] Implement weighted sampling/loss
- [ ] Tune weights via validation performance
- [ ] Expected lift: +3-5%

---

## Phase 6: Calibrated Confidence & Abstention
**Goal:** Know when to abstain, achieve 92%+ precision on predictions made

### Tasks
- [ ] Train confidence estimator (gap between top-1 and top-2)
- [ ] Set abstention threshold via precision-recall tradeoff
- [ ] Report coverage vs accuracy curves
- [ ] Target: 92% accuracy @ 85% coverage

---

## Success Metrics

### Primary
- **Top-1 Accuracy:** 90% on curated test set
- **Top-3 Accuracy:** 97%+
- **Recall@6:** 95%+

### Secondary
- **Per-source breakdown:** All sources > 80% Top-1
- **Calibration:** ECE < 0.05
- **Abstention:** 92% precision @ 85% coverage

---

## Timeline
| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 0 | 1-2 days | Day 2 |
| Phase 1 | 3-4 days | Day 6 |
| Phase 2 | 2 days | Day 8 |
| Phase 3 | 2 days | Day 10 |
| Phase 4 | 3-4 days | Day 14 |
| Phase 5 | 2 days | Day 16 |
| Phase 6 | 2-3 days | Day 19 |

**Total estimated time:** 3 weeks

---

## Checkpoints & Artifacts

### Phase 0 Outputs
- `data/cleaned/cyp3a4_cleaned_dataset.json`
- `data/cleaned/cyp3a4_cleaned_splits/`
- `data/cleaned/CHANGELOG.md`

### Model Checkpoints
- `checkpoints/phase1_relational_proposer/`
- `checkpoints/phase2_pairwise_reranker/`
- `checkpoints/phase3_unified_loss/`
- `checkpoints/phase4_chemistry_features/`
- `checkpoints/phase5_source_weighted/`
- `checkpoints/phase6_calibrated/`

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| Label noise worse than estimated | Expand literature verification to all training data |
| Relational proposer overfits | Add dropout, reduce capacity, regularize attention |
| Chemistry features don't help | Ablation study, feature importance analysis |
| Source weighting hurts generalization | Cross-validation across source combinations |
| 90% not achievable | Reframe as "90% on high-confidence subset" |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-09 | Start with data cleanup | Highest impact, lowest risk |
| | | |

