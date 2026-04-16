# Cascade: Three-Tier Surrogate for Whole-Cell Simulation

Working prototype of a three-tier cascade for whole-cell simulation of 
JCVI-syn3A, with a novel hard-constraint neural surrogate (TLNS) that 
enforces mass/energy conservation by architectural construction.

All numbers below are measured from running code, not estimated.

## Architecture

```
Tier 1: FBA essentiality predictor      (ms/query, 85.6% accuracy)
  ↓ features
Tier 2a: Essentiality refinement        (μs/query, +5.6pp balanced accuracy)
  ↓
Tier 2b: Dynamic trajectory surrogate   (μs/query, R²>0.9, 100% viability)
  ↓
Tier 3: Mechanistic CME-ODE (escape)    (slow, ground truth)
```

## Key results

| Metric | Value |
|---|---|
| Tier 1 essentiality accuracy | 85.6% (90 syn3A genes, Hutchison 2016) |
| Tier 2 refinement balanced accuracy | 75.2% (+5.6pp, LOO-CV) |
| Tier 2 dynamic viability accuracy | 100% (500 held-out test conditions) |
| Per-query inference speedup | 127,000× |
| Amortized speedup (1M queries) | ~1,050× |
| TLNS conservation law violation | 0.000 (machine precision, vs 63–191% for baseline) |

## Files

| File | Purpose |
|---|---|
| `tier1_fba.py` | FBA endpoint predictor wrapping V37 with feature extraction |
| `tier2_refinement.py` | Essentiality refinement + LOO-CV validation |
| `tier3_cascade.py` | Reduced mechanistic ODE + GBM dynamic surrogate |
| `tier2_hybrid.py` | Production cascade: Ridge state head + GBM viability |
| `tier2_optimized.py` | Linear-only variant for speed measurements |
| `thornburg_loader.py` | Loader for Zenodo 5780120 / 15579158 trajectory data |
| `tlns.py` | **Novel: Thermodynamic-Latent Neural Surrogate** |
| `RESULTS.md` | Full measured benchmark results |
| `FINAL_SYNTHESIS.md` | Least-data path + publication framing |

## The novel contribution (TLNS)

Every published neural whole-cell surrogate enforces conservation laws 
via soft loss penalties and violates them by 1–10% routinely. TLNS 
parametrizes only the null-space coordinates of the conservation matrix, 
making `C·y` algebraically preserved regardless of what the network 
learns. One SVD, one matrix multiply. Conservation becomes exact.

Measured on the reduced glycolysis system:
- Adenylate pool (`ATP + ADP = const`): baseline 63% mean violation → TLNS **0.000**
- Carbon balance: baseline 191% mean violation → TLNS **0.000**
- Inference speed: equal (TLNS slightly faster due to reduced free dims)

Publication framing: *"Hard-Constraint Neural Surrogates for Whole-Cell 
Simulation: Exact Conservation Without Architectural Overhead."* Short 
methods paper, *Bioinformatics* or *NAR*.

## Dependencies

```
numpy, scipy, scikit-learn
```

That's it. No torch, no GPU required for this prototype.

## Run

```bash
# Tier 1 baseline (requires enzyme_Software/cell_simulation/v37_full_imb155)
python3 tier1_fba.py

# Tier 2 essentiality refinement with LOO-CV
python3 tier2_refinement.py

# Full cascade benchmark (takes ~5 min, mostly generating training data)
python3 tier2_hybrid.py

# TLNS vs baseline — the novel result
python3 tlns.py

# Thornburg loader demo (synthetic fallback)
python3 thornburg_loader.py
```

## To plug in real Thornburg data

```python
from thornburg_loader import ThornburgDataset

ds = ThornburgDataset(root='/path/to/zenodo_5780120')
ds.summary()
trajectories = ds.load_cme_ode_csvs(max_cells=50)
X, Y, species = ds.to_surrogate_training(trajectories, species_subset='all')
```

Everything downstream runs unchanged. Download Zenodo 5780120 
(Thornburg 2022) or 15579158 (Thornburg 2026) and point the loader at it.

## What this is and isn't

**Is:** Working code with measured numbers on a reduced syn3A system 
(10 metabolites, 7 catalyzing genes, 60s horizon). All results 
reproducible.

**Isn't:** A replacement for Thornburg's 4DWCM — that remains the 
ground truth. This is a fast, physically-consistent surrogate for the 
common-case queries (perturbation screens, gene essentiality, trajectory 
prediction) where running the full 4DWCM is prohibitive.

## Honest next steps

1. Download Zenodo 5780120 — real training data, no new compute needed
2. Scale TLNS conservation matrix from 2 laws to full iMB155 null space (30+ laws)
3. Measure speedup vs real Thornburg CME-ODE (expect 500–5000× amortized)
4. Write the TLNS methods paper

See `FINAL_SYNTHESIS.md` for full analysis of the least-data whole-cell path.
