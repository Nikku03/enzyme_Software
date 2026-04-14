# Data Required for "The Cell Knows Itself" Model

## The Core Insight

A cell doesn't have a controller. It doesn't make decisions. 

**Every molecular interaction IS the computation:**
- Concentration × Affinity → Binding probability
- Binding → Conformational change → Activity change
- Activity → Flux → Concentration change
- Loop forever

The "program" of the cell is encoded in:
1. **Kd values** - What binds to what, how tightly?
2. **kcat values** - How fast do reactions go?
3. **Allosteric coefficients** - How does binding at one site affect another?
4. **ΔG° values** - Which direction does the reaction go?
5. **Concentrations** - The current state

---

## Data Sources Available

### 1. KINETIC PARAMETERS (kcat, Km)

**BRENDA Database** (brenda-enzymes.org)
- 2.7 million manually annotated entries
- 45,000+ enzymes from 6,900+ organisms
- Coverage: kcat, Km, Ki, Vmax, pH/temp optimum
- Mycoplasma-specific: LIMITED (~100-200 entries)
- **Can query via SOAP API or download bulk data**

**Status for syn3A:**
| Parameter | Estimated Coverage | Source |
|-----------|-------------------|--------|
| kcat | ~30% of reactions | BRENDA + literature |
| Km | ~40% of reactions | BRENDA + literature |
| Ki (inhibitors) | ~10% | BRENDA |
| Temperature optimum | ~20% | BRENDA |

**Gap:** ~60-70% of syn3A enzymes have NO measured kinetic parameters.

---

### 2. THERMODYNAMICS (ΔG°)

**eQuilibrator** (equilibrator.weizmann.ac.il)
- Gibbs energy for ~5,000 metabolites
- Can estimate ΔG° for any reaction via group contribution
- Python API: `equilibrator-api` on PyPI
- Accuracy: ±5-10 kJ/mol for most reactions

**Status for syn3A:**
| Data | Coverage | Notes |
|------|----------|-------|
| Metabolite ΔfG° | ~90% | Most common metabolites covered |
| Reaction ΔrG° | ~95% | Can be calculated from metabolites |
| pH/ionic strength effects | ✓ | Built into eQuilibrator |

**This is actually pretty complete!**

---

### 3. ALLOSTERIC REGULATION

**ASD - AlloSteric Database** (mdl.shsmu.edu.cn/ASD)
- ~100,000 allosteric proteins
- Activators and inhibitors with binding data
- Problem: Mostly eukaryotic, very few bacterial

**BRENDA (inhibitor/activator fields)**
- Ki values for inhibitors
- Activator information in commentary
- Coverage for Mycoplasma: POOR

**Status for syn3A:**
| Interaction Type | Known | Unknown |
|-----------------|-------|---------|
| ATP inhibition of glycolytic enzymes | ✓ | - |
| Feedback inhibition in biosynthesis | Partial | Most |
| Allosteric activators | Very few | Most |

**Gap:** This is the BIGGEST missing piece. We don't know most allosteric interactions in syn3A.

---

### 4. PROTEIN-PROTEIN BINDING (Kd)

**BindingDB** (bindingdb.org)
- 2.8 million binding measurements
- Mostly drug-target (not protein-protein)

**PDBbind** (pdbbind.org.cn)
- ~23,000 protein-ligand complexes with Kd
- Focused on drug discovery

**EVcomplex** (cited in syn3A paper)
- Predicted 33 protein-protein interactions in syn3A
- No binding affinities, just interaction probability

**Status for syn3A:**
| Interaction | Data Available |
|-------------|---------------|
| Enzyme-substrate Kd | Same as Km (~40%) |
| Protein-protein Kd | ALMOST NONE |
| Enzyme-cofactor Kd | ~30% |
| Ribosome assembly | Partial (from literature) |

**Gap:** We don't know most protein-protein binding affinities.

---

### 5. EXISTING SYN3A MODELS

**Luthey-Schulten Lab (Cell 2022)**
- Complete kinetic model of syn3A
- GitHub: Luthey-Schulten-Lab/Minimal_Cell_4DWCM
- Files include:
  - `kinetic_params.xlsx` - All kinetic parameters used
  - `initial_concentration.xlsx` - Starting metabolite/protein levels
  - `Syn3A_updated.xml` - SBML model with reactions
  
**This is GOLD. They already solved the parameter gathering problem.**

Their model has:
- 304 metabolites
- 338 reactions  
- 155 genes
- Kinetic parameters for ALL reactions (estimated where unknown)
- mRNA half-lives
- Protein copy numbers

---

## Data Inventory Summary

| Data Type | Total Needed | Available | Coverage | Source |
|-----------|-------------|-----------|----------|--------|
| **Stoichiometry** | 338 rxns | 338 | 100% | iMB155 SBML |
| **kcat** | 338 rxns | ~100 | 30% | BRENDA + Luthey-Schulten |
| **Km** | ~500 pairs | ~200 | 40% | BRENDA + Luthey-Schulten |
| **ΔG°** | 338 rxns | ~320 | 95% | eQuilibrator |
| **Allosteric regulation** | ~50+ | ~10 | 20% | Literature + BRENDA |
| **Protein-protein Kd** | ~100 | ~5 | 5% | EVcomplex predictions |
| **Gene expression** | 452 genes | 452 | 100% | Proteomics data |
| **Initial concentrations** | 304 mets | 304 | 100% | Luthey-Schulten |

---

## Critical Gaps & How to Fill Them

### Gap 1: Missing Kinetic Parameters
**Solution:** Use Luthey-Schulten parameters (they already estimated them)
- Download from GitHub
- They used: literature + BRENDA + inference from similar enzymes

### Gap 2: Allosteric Regulation  
**Solution:** Start with known metabolic regulation patterns
- ATP inhibits: PFK, pyruvate kinase, citrate synthase
- ADP activates: PFK
- GTP inhibits: glutamate dehydrogenase
- Product inhibition: most biosynthetic enzymes

For unknown: Use Hill coefficient n = 1 (no cooperativity) until data available

### Gap 3: Protein-Protein Binding
**Solution:** Two approaches:
1. Assume rapid equilibrium (binding is fast, use steady-state)
2. Estimate Kd from complex stability (ΔG = -RT ln(Kd))

### Gap 4: Transcription Factor Binding
**Irrelevant for syn3A!** 
- syn3A has almost no transcriptional regulation
- Only 2-3 sigma factors, minimal TF networks
- Most regulation is post-translational (allosteric)

---

## Recommended Data Loading Strategy

```python
# 1. Load Luthey-Schulten model (they did the hard work)
wget https://github.com/Luthey-Schulten-Lab/Minimal_Cell_4DWCM/raw/main/model_data/kinetic_params.xlsx
wget https://github.com/Luthey-Schulten-Lab/Minimal_Cell_4DWCM/raw/main/model_data/initial_concentration.xlsx
wget https://github.com/Luthey-Schulten-Lab/Minimal_Cell_4DWCM/raw/main/model_data/Syn3A_updated.xml

# 2. Get thermodynamics from eQuilibrator
pip install equilibrator-api
# Query ΔG° for each reaction

# 3. Build allosteric regulation matrix from literature
# Start with well-known feedback loops (ATP/ADP, NAD+/NADH)

# 4. Run model with:
# - Full kinetics from Luthey-Schulten
# - Thermodynamic constraints from eQuilibrator  
# - Basic allosteric patterns from textbooks
# - Protein-protein binding assumed fast (steady-state)
```

---

## The "Cell Knows Itself" Implementation

What we need for the new architecture:

```python
class PhysicsCell:
    """A cell that only knows physics, not biology."""
    
    def __init__(self):
        # The cell's "knowledge" - all from databases
        self.Kd_matrix = load_binding_affinities()    # Who binds whom
        self.kcat_vector = load_catalytic_rates()     # How fast
        self.dG_vector = load_thermodynamics()        # Which direction
        self.allosteric_map = load_regulation()       # How binding affects activity
        
        # The cell's "state" - just concentrations
        self.concentrations = load_initial_state()
    
    def step(self, dt):
        """Physics does everything."""
        
        # 1. Compute all binding probabilities
        # P(A binds B) = [A][B] / (Kd + [A])
        
        # 2. Compute all reaction rates  
        # v = kcat * [E] * saturation * thermodynamic_drive
        
        # 3. Compute thermodynamic driving force
        # drive = 1 - exp(ΔG / RT)  # 0 at equilibrium, 1 far from eq
        
        # 4. Apply allosteric modifications
        # kcat_effective = kcat * allosteric_factor(inhibitors, activators)
        
        # 5. Update concentrations
        # d[S]/dt = Σ(stoich * flux)
        
        # That's it. No decisions. Just physics.
```

---

## Bottom Line

**What we have:**
- 100% of stoichiometry
- 95% of thermodynamics  
- 100% of gene expression data
- 100% of initial concentrations
- ~40% of kinetics (rest can be estimated)

**What we're missing:**
- Comprehensive allosteric regulation (~20% known)
- Protein-protein binding affinities (~5% known)

**The good news:**
The Luthey-Schulten lab already built a working model with estimated parameters. We can use their work as a starting point and improve it with the physics-first approach.

**The insight:**
A cell doesn't need to "know" anything except:
1. What's touching me right now? (concentrations)
2. How tightly does it stick? (Kd)
3. How fast can I process it? (kcat)
4. Which way should it go? (ΔG)

Everything else emerges.
