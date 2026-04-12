# 🧬 Dark Manifold Virtual Cell v2.0

**A neural network that IS a living cell, not just simulates one.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nikku03/enzyme_Software/blob/main/dark_manifold_cell/Dark_Manifold_V2_Colab.ipynb)

## What's New in v2.0

- ✅ **Real biochemistry** from Luthey-Schulten Lab (eLife 2019, Cell 2022)
- ✅ **Quantum Field Theory** components (Green's function propagators)
- ✅ **Liquid Neural Networks** (continuous-time adaptive dynamics)
- ✅ **63 real metabolic reactions** with stoichiometry
- ✅ **452 protein-coding genes** with regulation

## Real Biochemistry (NEW!)

Unlike v1 which learned everything from scratch, v2 uses **real stoichiometry** from published JCVI-syn3A models:

| Source | Data |
|--------|------|
| Breuer et al. (2019) eLife | Metabolic network, 98% of reactions annotated |
| Thornburg et al. (2022) Cell | Kinetic parameters, protein concentrations |
| Luthey-Schulten Lab GitHub | SBML model, initial conditions |

The stoichiometry matrix `S[i,j]` tells us how metabolite `i` changes when reaction `j` fires - **real chemistry, not learned weights**.

## Results

| Model | Genes | Trajectory Corr | Notes |
|-------|-------|-----------------|-------|
| v1 (100-gene) | 100 | 0.922 | Learned stoichiometry |
| v2 (452-gene) | 452 | TBD | Real stoichiometry |

## Quick Start

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badge above, or:

1. Go to [Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub
3. Enter: `YOUR_USERNAME/dark-manifold-cell`
4. Select `Dark_Manifold_Virtual_Cell.ipynb`
5. Runtime → Run all

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/dark-manifold-cell.git
cd dark-manifold-cell

# Install dependencies
pip install torch numpy matplotlib

# Run 100-gene model
python dark_manifold_100gene.py
```

## Model Architecture

```
INPUT: Molecular concentrations [n_genes + n_metabolites]
         ↓
    ┌────────────────────────────────────┐
    │     DARK FIELD ENCODER             │
    │  LayerNorm + SiLU activation       │
    └────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────┐
    │   LEARNED STOICHIOMETRY (W_stoich) │
    │  Enzyme → Metabolite effects       │
    └────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────┐
    │   LEARNED REGULATION (W_reg)       │
    │  Gene → Gene effects               │
    └────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────┐
    │     DYNAMICS NETWORKS              │
    │  Metabolite + Gene dynamics        │
    └────────────────────────────────────┘
         ↓
OUTPUT: Next state
```

## JCVI-syn3A Pathways (100 genes)

| Pathway | Genes |
|---------|-------|
| Glycolysis | ptsI, ptsH, pgi, pfkA, fba, tpiA, gapA, pgk, gpmI, eno, pyk, ldh |
| Pentose Phosphate | zwf, pgl, gnd, rpe, rpiA, tktA, talA |
| Nucleotide Synthesis | purA-N, pyrB-E (15 genes) |
| Amino Acid tRNA Synthetases | alaS, argS, ... valS (20 genes) |
| Lipid Synthesis | accA-D, fabD-Z (10 genes) |
| ATP Synthase | atpA-H, ndk, adk (10 genes) |
| Transcription/Translation | rpoA-D, rpsA-B, rplA-B, fusA, tufA, etc. (15 genes) |
| Transporters | glcU, nupC, potA-D, oppA-F, secA, secY (13 genes) |

## Files

| File | Description |
|------|-------------|
| `dark_manifold_v2_real.py` | **v2** - Real biochemistry (452 genes, 115 metabolites, 63 reactions) |
| `dark_manifold_v2.py` | **v2** - Base architecture with QFT + Liquid Neural Networks |
| `dark_manifold_493gene_v2.py` | **v2** - Full 493-gene model |
| `Dark_Manifold_V2_Colab.ipynb` | Colab notebook for GPU training |
| `dark_manifold_100gene.py` | **v1** - Original 100-gene model |

## Architecture (v2)

```
Input: gene_state [B, 452], met_state [B, 115]
    ↓
Encoders (gene_embed, met_embed)
    ↓
Green's Function Layer → G(ω) = (ω + iη - H)^(-1)
    ↓ 
PathwayLiquidCell (τ varies by pathway)
    ↓
Reaction Rate Predictor → v = f(gene_expression)
    ↓
Metabolite Dynamics: dM/dt = S @ v (real stoichiometry!)
    ↓
gene_pred, met_pred, cellular_state
```

## Components from enzyme_Software

Integrates QFT and Liquid Neural Network concepts from CYP-Predict NEXUS:

| Source File | Concept | Integration |
|------------|---------|-------------|
| `hybrid_nexus_dynamic.py` | LiquidCell | PathwayLiquidCell |
| `quantum_field_theory.py` | Green's function | GeneNetworkGreensFunction |
| `hybrid_nexus_dynamic.py` | DynamicStateBank | CellularStateBank |

## Citation

If you use this code, please cite:

```
@software{dark_manifold_cell,
  title = {Dark Manifold Virtual Cell},
  author = {Chhillar, Naresh},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/dark-manifold-cell}
}
```

## References

- JCVI-syn3A: Hutchison et al. (2016) "Design and synthesis of a minimal bacterial genome" *Science*
- Whole-cell model: Thornburg et al. (2022) "Fundamental behaviors emerge from simulations of a living minimal cell" *Cell*

## License

MIT License
