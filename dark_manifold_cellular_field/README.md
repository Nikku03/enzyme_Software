# Dark Manifold Virtual Cell v2.1

A quantum field theory approach to whole-cell simulation of **JCVI-syn3A** (493 genes, 543kb genome).

## Key Features

### v2.1 Improvements over v2.0
- ✅ **Michaelis-Menten kinetics**: Reaction rates depend on substrate concentrations
- ✅ **Real kinetic parameters** from Thornburg et al. (2022) Cell
- ✅ **Allosteric regulation**: ATP inhibits PFK, ADP activates PYK
- ✅ **Energy charge tracking**: EC = (ATP + 0.5×ADP)/(ATP+ADP+AMP)
- ✅ **Modular architecture**: Proper component separation

### Architecture Components

```
src/
├── core/               # Main DarkManifoldCell model
├── kinetics/           # Michaelis-Menten, allosteric regulation
├── qft/                # Green's function, dark matter field
├── dynamics/           # Liquid cells, Hamiltonian, Neural ODE
├── spatial/            # Compartments, crowding, transport
├── memory/             # Hyperbolic memory, rule extraction
├── data/               # Biochemistry data, SBML loaders
└── utils/              # Training utilities
```

## Installation

```bash
pip install torch numpy
# Optional: for loading real SBML data
pip install openpyxl
```

## Quick Start

```python
import torch
from src import DarkManifoldCell, evaluate_model

# Create model
model = DarkManifoldCell(hidden_dim=128)

# Initialize states
gene_expr = torch.ones(1, model.n_genes) * 0.5
met_conc = model.init_conc.unsqueeze(0)

# Forward pass
out = model(gene_expr, met_conc)
print(f"Energy charge: {out['energy_charge'].item():.3f}")

# Rollout trajectory
trajectory = model.rollout(gene_expr, met_conc, n_steps=100)
print(f"Trajectory shape: {trajectory['met_trajectory'].shape}")
```

## Module Details

### Core Model (`src/core/`)
- `DarkManifoldCell`: Full model with QFT, liquid dynamics
- `DarkManifoldCellLite`: Faster version without QFT

### Kinetics (`src/kinetics/`)
- `MichaelisMentenKinetics`: v = Vmax × [S]/(Km + [S])
- `AllostericRegulation`: Inhibition/activation by effectors
- `CombinedKinetics`: MM + allosteric together

### QFT Components (`src/qft/`)
- `GeneNetworkGreensFunction`: G(ω) = (ω + iη - H)^(-1) for non-local interactions
- `DarkMatterField`: Continuous field mediating interactions
- `QuantumFluctuation`: Stochastic sampling for exploration

### Dynamics (`src/dynamics/`)
- `LiquidCell`: Basic liquid time-constant cell
- `PathwayLiquidCell`: Pathway-specific time constants
- `CellularStateBank`: Discrete states (growth, stress, division)
- `HamiltonianDynamics`: Energy-conserving neural network
- `MetabolicHamiltonian`: Thermodynamically consistent dynamics

### Spatial (`src/spatial/`)
- `CompartmentalizedMetabolism`: Cytoplasm/membrane/extracellular
- `MolecularCrowding`: Activity coefficients from crowding
- `MembraneTransport`: Active/passive transport

### Memory (`src/memory/`)
- `HyperbolicMemory`: Memory bank in Poincaré ball
- `NeuralFieldTuringMachine`: Continuous field memory
- `RuleExtractor`: Extract interpretable rules

### Data (`src/data/`)
- `REACTIONS`: 31 reactions with stoichiometry
- `KINETIC_PARAMS`: kcat, Km from Thornburg et al.
- `SBMLLoader`: Load from Luthey-Schulten SBML files

## Training

Use the Colab notebook for GPU training:
```
notebooks/Dark_Manifold_V21_Colab.ipynb
```

Expected results after 500 epochs:
- Gene correlation: 0.95+
- Metabolite correlation: 0.90+ (up from 0.38 in v2.0!)
- Energy charge: 0.85-0.95 (physiological range)

## Data Sources

- **Breuer et al. (2019) eLife**: Essential metabolism for minimal cell
- **Thornburg et al. (2022) Cell**: Fundamental behaviors emerge
- **Luthey-Schulten Lab GitHub**: SBML models, kinetic parameters
  - https://github.com/Luthey-Schulten-Lab/Minimal_Cell
  - https://github.com/Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation

## Citation

If using this code:
```
@software{dark_manifold_cell,
  title={Dark Manifold Virtual Cell},
  author={Chhillar, Naresh},
  year={2026},
  version={2.1.0}
}
```

## License

MIT License
