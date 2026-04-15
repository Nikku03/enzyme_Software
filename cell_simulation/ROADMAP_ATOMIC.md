# Dark Manifold Virtual Cell: Roadmap to Atomic Resolution

## Vision
Simulate JCVI-syn3A from gene knockouts to individual atom movements, bridging 15 orders of magnitude in time (femtoseconds to hours) and space (angstroms to microns).

## Current State (April 2026)

### V37: Flux Balance Analysis ✅
- **Resolution**: Steady-state fluxes
- **Accuracy**: 85.6% gene essentiality
- **Speed**: 2.75ms/knockout
- **Limits**: No time dynamics, no concentrations

### V38: Kinetic ODE ✅ 
- **Resolution**: Metabolite concentrations over time
- **Speed**: ~100ms for 2h simulation
- **Features**: Michaelis-Menten, allosteric regulation, energy charge
- **Limits**: Deterministic, no molecular noise, no spatial organization

---

## Development Roadmap

### V39: Stochastic Kinetics (Next)
**Goal**: Individual molecule counts, gene expression noise

**Key Features**:
- Gillespie SSA (Stochastic Simulation Algorithm)
- Tau-leaping for speedup
- Protein copy number distributions
- Transcription bursts
- Translation noise

**Biological Insights**:
- ~500 copies of most proteins in JCVI-syn3A
- Significant molecule-to-molecule variation
- Stochastic gene expression

**Implementation**:
```python
# Gillespie algorithm core
def gillespie_step(state, propensities):
    a_sum = sum(propensities)
    tau = -log(random()) / a_sum  # Time to next reaction
    r = random() * a_sum
    # Select reaction proportional to propensity
    reaction = select_reaction(r, propensities)
    return tau, reaction
```

**Timeline**: 2 weeks

---

### V40: Spatial Model
**Goal**: 3D positions, diffusion, compartmentalization

**Key Features**:
- Reaction-diffusion PDEs
- Membrane as a boundary
- Nucleoid region (DNA localization)
- Ribosome clustering
- Protein localization patterns

**Technical Approach**:
- Finite element method (FEniCS)
- OR lattice-based (lattice Boltzmann)
- OR particle-based (GFRD - Green's Function Reaction Dynamics)

**Biological Data**:
- JCVI-syn3A is ~400nm diameter
- ~500,000 proteins total
- ~200 ribosomes
- 1 circular chromosome (531kb)

**Timeline**: 1 month

---

### V41: Coarse-Grained Molecular Dynamics
**Goal**: Protein shapes, membrane dynamics, large-scale motions

**Key Features**:
- Martini 3 force field
- 4:1 atom mapping (4 atoms → 1 bead)
- Membrane lipid dynamics
- Protein folding/unfolding
- Protein-protein interactions

**Scale**:
- ~100 million atoms → ~25 million beads
- Accessible timescale: microseconds
- Timestep: 20-40 fs

**Implementation**:
- GROMACS for MD engine
- Martinize2 for protein mapping
- Insane for membrane building

**Biological Questions**:
- How do ribosomes diffuse?
- Membrane protein organization
- Chromosome compaction

**Timeline**: 2 months

---

### V42: Hybrid Multi-Scale Coupling
**Goal**: Seamlessly connect all scales

**Architecture**:
```
┌─────────────────────────────────────────┐
│         DARK FIELD MEDIATOR             │
│   (Learned latent space coupling)       │
└─────────────────────────────────────────┘
       ↑↓              ↑↓              ↑↓
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Kinetic │    │ Spatial  │    │   CG-MD  │
│   ODE    │←→│  PDE/SSA  │←→│  Martini │
└──────────┘    └──────────┘    └──────────┘
```

**Coupling Methods**:
1. **Equation-free multiscale**: Lift/restrict operators
2. **Heterogeneous Multiscale Method (HMM)**: Micro informs macro
3. **Neural Operator Coupling**: FNO learns scale transitions

**Key Innovation - Dark Field Theory**:
- The "dark field" is a learned continuous latent representation
- Information flows through field, not direct coupling
- Enables superposition (quantum-inspired uncertainty)
- Pair production: predictions + uncertainties

**Timeline**: 3 months

---

### V43: Atomic Resolution Windows
**Goal**: All-atom MD for selected subsystems

**Target Subsystems**:
1. **Ribosome active site** during translation
2. **ATP synthase** proton translocation
3. **DNA polymerase** nucleotide incorporation
4. **Membrane channel** ion transport
5. **Enzyme active sites** during catalysis

**Technical Approach**:
- QM/MM for bond breaking/forming
- Machine Learning Potentials (MACE, ANI-2x, Allegro)
- Adaptive resolution (all-atom ↔ coarse-grained)

**Scale**:
- 10,000-100,000 atoms per window
- Femtosecond to nanosecond
- GPU-accelerated (JAX-MD, OpenMM)

**Implementation**:
```python
# Adaptive resolution switching
class AdaptiveSimulator:
    def __init__(self):
        self.cg_simulator = MartiniSimulator()
        self.aa_simulator = OpenMMSimulator()
        self.ml_potential = MACEPotential()
    
    def step(self, region_of_interest):
        if requires_atomic_detail(region_of_interest):
            # Backmapping CG → AA
            aa_coords = backmap(self.cg_state, region_of_interest)
            # All-atom simulation
            self.aa_simulator.step(aa_coords)
            # Forward map AA → CG
            self.cg_state = forward_map(self.aa_state)
        else:
            self.cg_simulator.step()
```

**Timeline**: 4 months

---

### V44: Full Integration - Dark Manifold Virtual Cell
**Goal**: Complete multi-scale cell simulator

**Features**:
- Gene knockout → atomic response
- Emergent phenomena from bottom-up
- Uncertainty quantification at all scales
- Real-time visualization

**Validation**:
1. Reproduce Hutchison 2016 essentiality data
2. Match proteomics (Breuer 2019)
3. Match metabolomics
4. Match growth rate (2h doubling)
5. Match cell size/morphology

**Compute Requirements**:
- GPU cluster (8+ A100s)
- Or TPU pod
- Or distributed simulation

---

## Technical Innovations

### 1. Dark Field Theory
The "dark matter" of the cell - a continuous latent field that:
- Mediates information between scales
- Encodes uncertainty as superposition
- Collapses to specific states when observed/measured
- Enables efficient sparse computation via importance sampling

### 2. Neural Operators for Scale Bridging
- Fourier Neural Operators (FNO) for PDE solving
- DeepONet for operator learning
- Graph Neural Networks for molecular systems
- Equivariant networks for 3D structure

### 3. Pair Production Learning
Every prediction generates:
- The prediction itself (matter)
- Its uncertainty/negation (antimatter)
- Training captures signal before "annihilation"
- Natural connection to contrastive learning

### 4. Hyperbolic Memory Bank
- Memories stored in hyperbolic space (Poincaré ball)
- Hierarchical structure preserved
- PGW (Parallel Geodesic Walk) transport
- Enables analogical reasoning across scales

---

## Data Requirements

### Structural Data
- [ ] All 473 protein structures (AlphaFold)
- [ ] Membrane lipid composition
- [ ] Chromosome structure (Hi-C if available)
- [ ] Ribosome structure (from E. coli homology)

### Kinetic Data
- [ ] Enzyme kinetics from BRENDA
- [ ] Metabolite concentrations from metabolomics
- [ ] Protein copy numbers from proteomics
- [ ] Growth rate measurements

### Validation Data
- [ ] Hutchison 2016 essentiality
- [ ] Breuer 2019 metabolomics
- [ ] Any available time-series data

---

## Milestones

| Version | Feature | Timeline | Accuracy Target |
|---------|---------|----------|-----------------|
| V37 | FBA essentiality | ✅ Done | 85.6% |
| V38 | Kinetic ODE | ✅ Done | - |
| V39 | Stochastic | 2 weeks | Match noise data |
| V40 | Spatial | 1 month | Correct localization |
| V41 | CG-MD | 2 months | Membrane dynamics |
| V42 | Multi-scale coupling | 3 months | Seamless transitions |
| V43 | Atomic windows | 4 months | QM accuracy |
| V44 | Full integration | 6 months | 95%+ essentiality |

---

## Open Questions

1. **How to handle the timescale gap?**
   - Femtosecond atomic → hour cell division = 10^18 ratio
   - Need intelligent sampling / rare event methods

2. **What determines essentiality beyond metabolism?**
   - Protein quality control?
   - Membrane integrity?
   - Chromosome segregation?

3. **Can we discover new biology?**
   - Predict synthetic lethal pairs
   - Identify essential non-metabolic functions
   - Design minimal minimal cell

4. **How minimal can we go?**
   - JCVI-syn3A has 473 genes
   - What would V45 predict as truly minimal?

---

## References

1. Hutchison et al. 2016. Science. "Design and synthesis of a minimal bacterial genome"
2. Breuer et al. 2019. eLife. "Essential metabolism for a minimal cell"
3. Thornburg et al. 2022. Cell. "Fundamental behaviors emerge from simulations of a living minimal cell"
4. Martini 3 force field - Souza et al. 2021
5. MACE ML potential - Batatia et al. 2022
6. Fourier Neural Operator - Li et al. 2021
