"""
Dark Manifold V48d: Complete Minimal Cell Metabolism
====================================================

A publication-quality metabolic model for JCVI-syn3A.

Features:
- Full glycolysis with proper stoichiometry
- Nucleotide biosynthesis (NTPs and dNTPs)
- Amino acid pools with uptake
- ATP/GTP energy coupling
- NAD+/NADH redox balance
- Proper feedback regulation for homeostasis
- Coupling-ready for gene expression module

Based on iMB155 reconstruction (Breuer et al. 2019)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

CELL_VOLUME = 4.0e-17       # L (40 fL for JCVI-syn3A)
AVOGADRO = 6.022e23
CELL_DRY_MASS = 1.0e-13     # g (~0.1 pg)

def mM_to_molecules(c_mM: float) -> float:
    """Convert mM to molecules per cell."""
    return c_mM * 1e-3 * CELL_VOLUME * AVOGADRO

def molecules_to_mM(n: float) -> float:
    """Convert molecules per cell to mM."""
    return (n / AVOGADRO) / CELL_VOLUME * 1000

# Typical molecule counts
# ATP: ~3 mM ≈ 72,000 molecules
# Ribosome: ~1000 per cell
# mRNA: ~500 per cell
# Protein: ~50,000 per cell


# ============================================================================
# METABOLITE DEFINITIONS
# ============================================================================

@dataclass
class Metabolite:
    """Metabolite with thermodynamic and kinetic properties."""
    name: str
    formula: str = ""
    charge: int = 0
    compartment: str = "cytoplasm"
    
    # Concentration bounds (mM)
    c_min: float = 1e-6
    c_max: float = 100.0
    c_initial: float = 1.0
    
    # Is this a boundary metabolite (fixed)?
    boundary: bool = False


# ============================================================================
# REACTION DEFINITIONS
# ============================================================================

@dataclass 
class Reaction:
    """Metabolic reaction with kinetics."""
    name: str
    stoichiometry: Dict[str, float]  # metabolite: coefficient (- for substrates)
    
    # Kinetics
    vmax_forward: float = 1.0        # mM/min
    vmax_reverse: float = 0.0        # mM/min (0 = irreversible)
    
    # Michaelis constants
    Km: Dict[str, float] = field(default_factory=dict)
    
    # Inhibition
    Ki: Dict[str, float] = field(default_factory=dict)  # Competitive inhibitors
    
    # Activation
    Ka: Dict[str, float] = field(default_factory=dict)  # Activators
    
    # Associated gene(s)
    genes: List[str] = field(default_factory=list)
    
    # Thermodynamics
    delta_G0: float = 0.0  # kJ/mol (standard Gibbs free energy)
    
    @property
    def substrates(self) -> Dict[str, float]:
        return {m: -s for m, s in self.stoichiometry.items() if s < 0}
    
    @property
    def products(self) -> Dict[str, float]:
        return {m: s for m, s in self.stoichiometry.items() if s > 0}


# ============================================================================
# RATE LAWS
# ============================================================================

def michaelis_menten(S: float, Vmax: float, Km: float) -> float:
    """Simple Michaelis-Menten kinetics."""
    return Vmax * S / (Km + S)

def ordered_bi_bi(A: float, B: float, Vmax: float, 
                  Km_A: float, Km_B: float, Ki_A: float = None) -> float:
    """Ordered bi-bi mechanism (like many dehydrogenases)."""
    if Ki_A is None:
        Ki_A = Km_A
    denom = Ki_A * Km_B + Km_B * A + Km_A * B + A * B
    return Vmax * A * B / denom if denom > 0 else 0

def reversible_mm(S: float, P: float, Vmax_f: float, Vmax_r: float,
                  Km_S: float, Km_P: float) -> float:
    """Reversible Michaelis-Menten."""
    num = Vmax_f * S / Km_S - Vmax_r * P / Km_P
    denom = 1 + S / Km_S + P / Km_P
    return num / denom if denom > 0 else 0

def allosteric_inhibition(v: float, I: float, Ki: float, n: float = 1) -> float:
    """Apply allosteric inhibition to a rate."""
    return v * (Ki ** n) / (Ki ** n + I ** n)

def allosteric_activation(v: float, A: float, Ka: float, n: float = 1) -> float:
    """Apply allosteric activation to a rate."""
    return v * (A ** n) / (Ka ** n + A ** n)


# ============================================================================
# MINIMAL CELL METABOLISM MODEL
# ============================================================================

class MinimalCellMetabolism:
    """
    Complete metabolic model for JCVI-syn3A minimal cell.
    
    Pathways:
    1. Glycolysis (ATP production)
    2. Lactate fermentation (NAD+ regeneration)
    3. Pentose phosphate pathway (NADPH, Ribose-5-P)
    4. Nucleotide biosynthesis (NTPs, dNTPs)
    5. Amino acid uptake and pools
    6. Lipid precursors
    
    Coupling points for gene expression:
    - Transcription rate depends on NTP availability
    - Translation rate depends on ATP, GTP, amino acid availability
    - Protein/RNA degradation returns precursors
    """
    
    def __init__(self, growth_rate: float = 0.01):
        """
        Initialize metabolism.
        
        Args:
            growth_rate: Specific growth rate (1/min), default ~70 min doubling
        """
        self.mu = growth_rate
        
        # Build metabolite list
        self._build_metabolites()
        
        # Build reaction network
        self._build_reactions()
        
        # State indexing
        self.state_names = list(self.metabolites.keys())
        self.n_states = len(self.state_names)
        self.idx = {name: i for i, name in enumerate(self.state_names)}
        
        # External/boundary conditions
        self.external = {
            'Glc_ext': 20.0,      # External glucose (mM)
            'Lac_ext': 0.0,       # External lactate
            'AA_ext': 2.0,        # External amino acids (each)
            'Pi_ext': 10.0,       # External phosphate
            'O2': 0.0,            # No oxygen (fermentative)
        }
        
        # Biosynthetic demands (set by gene expression module)
        self.demands = {
            'protein_synthesis': 0.0,    # mM AA/min
            'rna_synthesis': 0.0,        # mM NTP/min
            'dna_synthesis': 0.0,        # mM dNTP/min
            'lipid_synthesis': 0.0,      # mM precursor/min
        }
        
        self._print_summary()
    
    def _build_metabolites(self):
        """Define all metabolites."""
        self.metabolites = {}
        
        # ===== ENERGY CARRIERS =====
        self.metabolites['ATP'] = Metabolite('ATP', 'C10H12N5O13P3', -4, c_initial=3.0)
        self.metabolites['ADP'] = Metabolite('ADP', 'C10H12N5O10P2', -3, c_initial=0.5)
        self.metabolites['AMP'] = Metabolite('AMP', 'C10H12N5O7P', -2, c_initial=0.1)
        self.metabolites['GTP'] = Metabolite('GTP', 'C10H12N5O14P3', -4, c_initial=0.8)
        self.metabolites['GDP'] = Metabolite('GDP', 'C10H12N5O11P2', -3, c_initial=0.2)
        self.metabolites['Pi'] = Metabolite('Pi', 'HO4P', -2, c_initial=10.0)
        self.metabolites['PPi'] = Metabolite('PPi', 'HO7P2', -3, c_initial=0.01)
        
        # ===== REDOX CARRIERS =====
        self.metabolites['NAD'] = Metabolite('NAD+', 'C21H26N7O14P2', -1, c_initial=1.0)
        self.metabolites['NADH'] = Metabolite('NADH', 'C21H27N7O14P2', -2, c_initial=0.05)
        self.metabolites['NADP'] = Metabolite('NADP+', 'C21H25N7O17P3', -3, c_initial=0.1)
        self.metabolites['NADPH'] = Metabolite('NADPH', 'C21H26N7O17P3', -4, c_initial=0.2)
        
        # ===== GLYCOLYSIS INTERMEDIATES =====
        self.metabolites['Glc'] = Metabolite('Glucose', 'C6H12O6', 0, c_initial=0.5)
        self.metabolites['G6P'] = Metabolite('Glucose-6-P', 'C6H11O9P', -2, c_initial=0.2)
        self.metabolites['F6P'] = Metabolite('Fructose-6-P', 'C6H11O9P', -2, c_initial=0.1)
        self.metabolites['FBP'] = Metabolite('Fructose-1,6-bisP', 'C6H10O12P2', -4, c_initial=0.05)
        self.metabolites['DHAP'] = Metabolite('DHAP', 'C3H5O6P', -2, c_initial=0.1)
        self.metabolites['G3P'] = Metabolite('G3P', 'C3H5O6P', -2, c_initial=0.05)
        self.metabolites['BPG'] = Metabolite('1,3-BPG', 'C3H4O10P2', -4, c_initial=0.01)
        self.metabolites['3PG'] = Metabolite('3-PG', 'C3H4O7P', -3, c_initial=0.2)
        self.metabolites['2PG'] = Metabolite('2-PG', 'C3H4O7P', -3, c_initial=0.05)
        self.metabolites['PEP'] = Metabolite('PEP', 'C3H2O6P', -3, c_initial=0.1)
        self.metabolites['Pyr'] = Metabolite('Pyruvate', 'C3H3O3', -1, c_initial=0.3)
        self.metabolites['Lac'] = Metabolite('Lactate', 'C3H5O3', -1, c_initial=0.5)
        
        # ===== PENTOSE PHOSPHATE PATHWAY =====
        self.metabolites['6PGL'] = Metabolite('6-P-Gluconolactone', 'C6H9O9P', -2, c_initial=0.01)
        self.metabolites['6PG'] = Metabolite('6-P-Gluconate', 'C6H10O10P', -3, c_initial=0.05)
        self.metabolites['Ru5P'] = Metabolite('Ribulose-5-P', 'C5H9O8P', -2, c_initial=0.05)
        self.metabolites['R5P'] = Metabolite('Ribose-5-P', 'C5H9O8P', -2, c_initial=0.1)
        self.metabolites['X5P'] = Metabolite('Xylulose-5-P', 'C5H9O8P', -2, c_initial=0.02)
        self.metabolites['S7P'] = Metabolite('Sedoheptulose-7-P', 'C7H13O10P', -2, c_initial=0.02)
        self.metabolites['E4P'] = Metabolite('Erythrose-4-P', 'C4H7O7P', -2, c_initial=0.01)
        
        # ===== NUCLEOTIDES =====
        self.metabolites['PRPP'] = Metabolite('PRPP', 'C5H8O14P3', -5, c_initial=0.05)
        self.metabolites['IMP'] = Metabolite('IMP', 'C10H11N4O8P', -2, c_initial=0.02)
        self.metabolites['GMP'] = Metabolite('GMP', 'C10H12N5O8P', -2, c_initial=0.05)
        self.metabolites['UMP'] = Metabolite('UMP', 'C9H11N2O9P', -2, c_initial=0.05)
        self.metabolites['CMP'] = Metabolite('CMP', 'C9H12N3O8P', -2, c_initial=0.05)
        self.metabolites['UTP'] = Metabolite('UTP', 'C9H11N2O15P3', -4, c_initial=0.5)
        self.metabolites['CTP'] = Metabolite('CTP', 'C9H12N3O14P3', -4, c_initial=0.3)
        self.metabolites['UDP'] = Metabolite('UDP', 'C9H11N2O12P2', -3, c_initial=0.1)
        self.metabolites['CDP'] = Metabolite('CDP', 'C9H12N3O11P2', -3, c_initial=0.1)
        
        # dNTPs
        self.metabolites['dATP'] = Metabolite('dATP', 'C10H12N5O12P3', -4, c_initial=0.02)
        self.metabolites['dGTP'] = Metabolite('dGTP', 'C10H12N5O13P3', -4, c_initial=0.02)
        self.metabolites['dCTP'] = Metabolite('dCTP', 'C9H12N3O13P3', -4, c_initial=0.02)
        self.metabolites['dTTP'] = Metabolite('dTTP', 'C10H13N2O14P3', -4, c_initial=0.02)
        
        # ===== AMINO ACIDS (simplified as pool) =====
        self.metabolites['AA'] = Metabolite('AA_pool', '', 0, c_initial=5.0)
        
        # ===== OTHER =====
        self.metabolites['CoA'] = Metabolite('CoA', 'C21H32N7O16P3S', -4, c_initial=0.5)
        self.metabolites['AcCoA'] = Metabolite('Acetyl-CoA', 'C23H34N7O17P3S', -4, c_initial=0.2)
        
    def _build_reactions(self):
        """Define all reactions with kinetics."""
        self.reactions = []
        
        # ==================== GLYCOLYSIS ====================
        
        # 1. Glucose uptake (PTS or permease)
        self.reactions.append(Reaction(
            name='GLC_uptake',
            stoichiometry={'Glc': 1, 'PEP': -1, 'Pyr': 1},  # PTS
            vmax_forward=1.0,
            Km={'PEP': 0.1},
            genes=['ptsG', 'ptsH', 'ptsI']
        ))
        
        # 2. Hexokinase / Glucokinase
        self.reactions.append(Reaction(
            name='HK',
            stoichiometry={'Glc': -1, 'ATP': -1, 'G6P': 1, 'ADP': 1},
            vmax_forward=0.5,
            Km={'Glc': 0.1, 'ATP': 0.5},
            genes=['glk']
        ))
        
        # 3. Phosphoglucose isomerase
        self.reactions.append(Reaction(
            name='PGI',
            stoichiometry={'G6P': -1, 'F6P': 1},
            vmax_forward=5.0,
            vmax_reverse=4.0,
            Km={'G6P': 0.3, 'F6P': 0.1},
            genes=['pgi']
        ))
        
        # 4. Phosphofructokinase (KEY REGULATORY STEP)
        self.reactions.append(Reaction(
            name='PFK',
            stoichiometry={'F6P': -1, 'ATP': -1, 'FBP': 1, 'ADP': 1},
            vmax_forward=2.0,
            Km={'F6P': 0.1, 'ATP': 0.1},
            Ki={'ATP': 1.0},  # ATP inhibition at high conc
            Ka={'ADP': 0.5, 'AMP': 0.1},  # ADP/AMP activation
            genes=['pfkA']
        ))
        
        # 5. Aldolase
        self.reactions.append(Reaction(
            name='FBA',
            stoichiometry={'FBP': -1, 'DHAP': 1, 'G3P': 1},
            vmax_forward=3.0,
            vmax_reverse=1.0,
            Km={'FBP': 0.05},
            genes=['fbaA']
        ))
        
        # 6. Triose phosphate isomerase (very fast)
        self.reactions.append(Reaction(
            name='TPI',
            stoichiometry={'DHAP': -1, 'G3P': 1},
            vmax_forward=50.0,
            vmax_reverse=25.0,
            Km={'DHAP': 1.0, 'G3P': 0.5},
            genes=['tpiA']
        ))
        
        # 7. GAPDH
        self.reactions.append(Reaction(
            name='GAPDH',
            stoichiometry={'G3P': -1, 'NAD': -1, 'Pi': -1, 'BPG': 1, 'NADH': 1},
            vmax_forward=5.0,
            Km={'G3P': 0.1, 'NAD': 0.1, 'Pi': 1.0},
            genes=['gapA']
        ))
        
        # 8. Phosphoglycerate kinase (ATP synthesis!)
        self.reactions.append(Reaction(
            name='PGK',
            stoichiometry={'BPG': -1, 'ADP': -1, '3PG': 1, 'ATP': 1},
            vmax_forward=10.0,
            vmax_reverse=5.0,
            Km={'BPG': 0.01, 'ADP': 0.2},
            genes=['pgk']
        ))
        
        # 9. Phosphoglycerate mutase
        self.reactions.append(Reaction(
            name='PGM',
            stoichiometry={'3PG': -1, '2PG': 1},
            vmax_forward=10.0,
            vmax_reverse=8.0,
            Km={'3PG': 0.2, '2PG': 0.1},
            genes=['gpmA']
        ))
        
        # 10. Enolase
        self.reactions.append(Reaction(
            name='ENO',
            stoichiometry={'2PG': -1, 'PEP': 1},
            vmax_forward=8.0,
            vmax_reverse=4.0,
            Km={'2PG': 0.1, 'PEP': 0.2},
            genes=['eno']
        ))
        
        # 11. Pyruvate kinase (ATP synthesis!)
        self.reactions.append(Reaction(
            name='PYK',
            stoichiometry={'PEP': -1, 'ADP': -1, 'Pyr': 1, 'ATP': 1},
            vmax_forward=5.0,
            Km={'PEP': 0.3, 'ADP': 0.3},
            Ka={'FBP': 0.1},  # FBP activation (feed-forward)
            genes=['pykF']
        ))
        
        # 12. Lactate dehydrogenase (NAD+ regeneration!)
        self.reactions.append(Reaction(
            name='LDH',
            stoichiometry={'Pyr': -1, 'NADH': -1, 'Lac': 1, 'NAD': 1},
            vmax_forward=10.0,
            vmax_reverse=0.5,
            Km={'Pyr': 0.5, 'NADH': 0.02},
            genes=['ldh']
        ))
        
        # ==================== PENTOSE PHOSPHATE PATHWAY ====================
        
        # 13. Glucose-6-P dehydrogenase (NADPH production)
        self.reactions.append(Reaction(
            name='G6PDH',
            stoichiometry={'G6P': -1, 'NADP': -1, '6PGL': 1, 'NADPH': 1},
            vmax_forward=0.5,
            Km={'G6P': 0.05, 'NADP': 0.01},
            genes=['zwf']
        ))
        
        # 14. 6-Phosphogluconolactonase
        self.reactions.append(Reaction(
            name='PGL',
            stoichiometry={'6PGL': -1, '6PG': 1},
            vmax_forward=5.0,
            Km={'6PGL': 0.05},
            genes=['pgl']
        ))
        
        # 15. 6-Phosphogluconate dehydrogenase
        self.reactions.append(Reaction(
            name='6PGDH',
            stoichiometry={'6PG': -1, 'NADP': -1, 'Ru5P': 1, 'NADPH': 1},
            vmax_forward=0.5,
            Km={'6PG': 0.03, 'NADP': 0.01},
            genes=['gnd']
        ))
        
        # 16. Ribose-5-P isomerase
        self.reactions.append(Reaction(
            name='RPI',
            stoichiometry={'Ru5P': -1, 'R5P': 1},
            vmax_forward=5.0,
            vmax_reverse=3.0,
            Km={'Ru5P': 0.5, 'R5P': 0.3},
            genes=['rpiA']
        ))
        
        # ==================== NUCLEOTIDE SYNTHESIS ====================
        
        # 17. PRPP synthetase
        self.reactions.append(Reaction(
            name='PRPP_syn',
            stoichiometry={'R5P': -1, 'ATP': -1, 'PRPP': 1, 'AMP': 1},
            vmax_forward=0.2,
            Km={'R5P': 0.05, 'ATP': 0.5},
            Ki={'ADP': 0.3, 'GDP': 0.3},  # Feedback inhibition
            genes=['prsA']
        ))
        
        # 18-19. Purine synthesis (simplified IMP → AMP/GMP)
        self.reactions.append(Reaction(
            name='AMP_syn',
            stoichiometry={'IMP': -1, 'GTP': -1, 'AMP': 1, 'GDP': 1},
            vmax_forward=0.1,
            Km={'IMP': 0.02, 'GTP': 0.2},
            genes=['purA', 'purB']
        ))
        
        self.reactions.append(Reaction(
            name='GMP_syn',
            stoichiometry={'IMP': -1, 'ATP': -1, 'NAD': -1, 'GMP': 1, 'ADP': 1, 'NADH': 1},
            vmax_forward=0.1,
            Km={'IMP': 0.02, 'ATP': 0.5},
            genes=['guaA', 'guaB']
        ))
        
        # 20-23. Nucleoside monophosphate kinases
        self.reactions.append(Reaction(
            name='ADK',  # Adenylate kinase
            stoichiometry={'AMP': -1, 'ATP': -1, 'ADP': 2},
            vmax_forward=20.0,
            vmax_reverse=20.0,
            Km={'AMP': 0.1, 'ATP': 0.2, 'ADP': 0.3},
            genes=['adk']
        ))
        
        self.reactions.append(Reaction(
            name='GMK',  # Guanylate kinase
            stoichiometry={'GMP': -1, 'ATP': -1, 'GDP': 1, 'ADP': 1},
            vmax_forward=5.0,
            Km={'GMP': 0.05, 'ATP': 0.3},
            genes=['gmk']
        ))
        
        self.reactions.append(Reaction(
            name='CMK',
            stoichiometry={'CMP': -1, 'ATP': -1, 'CDP': 1, 'ADP': 1},
            vmax_forward=3.0,
            Km={'CMP': 0.05, 'ATP': 0.3},
            genes=['cmk']
        ))
        
        self.reactions.append(Reaction(
            name='UMK',
            stoichiometry={'UMP': -1, 'ATP': -1, 'UDP': 1, 'ADP': 1},
            vmax_forward=3.0,
            Km={'UMP': 0.05, 'ATP': 0.3},
            genes=['pyrH']
        ))
        
        # 24. NDP kinase (equilibrates all NDP/NTP pools)
        for nuc in ['G', 'C', 'U']:
            self.reactions.append(Reaction(
                name=f'NDK_{nuc}',
                stoichiometry={f'{nuc}DP': -1, 'ATP': -1, f'{nuc}TP': 1, 'ADP': 1},
                vmax_forward=10.0,
                vmax_reverse=8.0,
                Km={f'{nuc}DP': 0.1, 'ATP': 0.2},
                genes=['ndk']
            ))
        
        # ==================== dNTP SYNTHESIS ====================
        
        # 25. Ribonucleotide reductase (NDP → dNDP)
        for nuc in ['A', 'G', 'C', 'U']:
            dnuc = 'd' + nuc
            if nuc == 'U':
                dnuc = 'dT'  # dUDP → dTTP pathway
            self.reactions.append(Reaction(
                name=f'RNR_{nuc}',
                stoichiometry={f'{nuc}DP' if nuc != 'A' else 'ADP': -1, 
                              'NADPH': -1, 
                              f'{dnuc}TP': 1 if nuc != 'U' else 0,  # Simplified
                              'NADP': 1},
                vmax_forward=0.02,
                Km={f'{nuc}DP' if nuc != 'A' else 'ADP': 0.1, 'NADPH': 0.05},
                genes=['nrdA', 'nrdB']
            ))
        
        # ==================== AMINO ACID UPTAKE ====================
        
        # 26. Amino acid transporter (simplified)
        self.reactions.append(Reaction(
            name='AA_uptake',
            stoichiometry={'AA': 1, 'ATP': -0.1, 'ADP': 0.1},  # Some ATP cost
            vmax_forward=1.0,
            Km={'ATP': 0.5},
            genes=['aapJ', 'aapM', 'aapP']
        ))
        
        # ==================== BIOSYNTHESIS CONSUMERS ====================
        
        # 27. Protein synthesis (4 ATP equiv per AA)
        self.reactions.append(Reaction(
            name='protein_syn',
            stoichiometry={'AA': -1, 'ATP': -2, 'GTP': -2, 'ADP': 2, 'GDP': 2, 'Pi': 4},
            vmax_forward=0.5,
            Km={'AA': 0.5, 'ATP': 0.5, 'GTP': 0.3},
            genes=['rpsA', 'rplA', 'tuf', 'fusA']  # Ribosome components
        ))
        
        # 28. RNA synthesis (NTP consumption)
        self.reactions.append(Reaction(
            name='rna_syn',
            stoichiometry={'ATP': -0.25, 'GTP': -0.25, 'CTP': -0.25, 'UTP': -0.25,
                          'ADP': 0.25, 'GDP': 0.25, 'CDP': 0.25, 'UDP': 0.25,
                          'PPi': 1},
            vmax_forward=0.1,
            Km={'ATP': 0.3, 'GTP': 0.3, 'CTP': 0.2, 'UTP': 0.2},
            genes=['rpoA', 'rpoB', 'rpoC']  # RNAP
        ))
        
        # 29. DNA synthesis (dNTP consumption)
        self.reactions.append(Reaction(
            name='dna_syn',
            stoichiometry={'dATP': -0.25, 'dGTP': -0.25, 'dCTP': -0.25, 'dTTP': -0.25,
                          'PPi': 1},
            vmax_forward=0.01,
            Km={'dATP': 0.01, 'dGTP': 0.01},
            genes=['dnaE', 'dnaN', 'dnaX']  # DNA Pol
        ))
        
        # ==================== MAINTENANCE ====================
        
        # 30. Maintenance ATP consumption
        self.reactions.append(Reaction(
            name='maintenance',
            stoichiometry={'ATP': -1, 'ADP': 1, 'Pi': 1},
            vmax_forward=0.2,
            Km={'ATP': 0.5}
        ))
        
        # 31. PPi hydrolysis (drives biosynthesis forward)
        self.reactions.append(Reaction(
            name='PPase',
            stoichiometry={'PPi': -1, 'Pi': 2},
            vmax_forward=10.0,  # Very fast
            Km={'PPi': 0.01},
            genes=['ppa']
        ))
        
        # Index reactions
        self.reaction_names = [r.name for r in self.reactions]
        self.n_reactions = len(self.reactions)
    
    def _print_summary(self):
        """Print model summary."""
        print(f"\n{'='*70}")
        print("MINIMAL CELL METABOLISM MODEL")
        print("="*70)
        print(f"Metabolites: {self.n_states}")
        print(f"Reactions: {self.n_reactions}")
        print(f"Growth rate: {self.mu:.4f} /min (doubling: {np.log(2)/self.mu:.1f} min)")
        
        # Count reaction types
        glyc = sum(1 for r in self.reactions if r.name in 
                   ['GLC_uptake', 'HK', 'PGI', 'PFK', 'FBA', 'TPI', 
                    'GAPDH', 'PGK', 'PGM', 'ENO', 'PYK', 'LDH'])
        ppp = sum(1 for r in self.reactions if r.name in 
                  ['G6PDH', 'PGL', '6PGDH', 'RPI'])
        nuc = sum(1 for r in self.reactions if 'syn' in r.name.lower() or 
                  'NDK' in r.name or 'ADK' in r.name or 'GMK' in r.name or
                  'CMK' in r.name or 'UMK' in r.name or 'RNR' in r.name)
        
        print(f"\nReaction categories:")
        print(f"  Glycolysis: {glyc}")
        print(f"  Pentose phosphate: {ppp}")
        print(f"  Nucleotide metabolism: {nuc}")
        print(f"  Other: {self.n_reactions - glyc - ppp - nuc}")
    
    def get_initial_state(self) -> np.ndarray:
        """Get initial concentration vector."""
        y0 = np.zeros(self.n_states)
        for name, met in self.metabolites.items():
            y0[self.idx[name]] = met.c_initial
        return y0
    
    def calculate_rate(self, rxn: Reaction, conc: Dict[str, float]) -> float:
        """Calculate reaction rate given concentrations."""
        
        # Forward rate
        v_f = rxn.vmax_forward
        
        # Substrate saturation
        for sub, stoich in rxn.substrates.items():
            if sub in conc:
                S = conc[sub]
                Km = rxn.Km.get(sub, 0.1)
                v_f *= S / (Km + S)
        
        # Competitive inhibition
        for inh, Ki in rxn.Ki.items():
            if inh in conc:
                I = conc[inh]
                v_f *= Ki / (Ki + I)
        
        # Allosteric activation
        for act, Ka in rxn.Ka.items():
            if act in conc:
                A = conc[act]
                v_f *= (1 + A / Ka) / (1 + Ka / (A + 1e-9))
        
        # Reverse rate (if reversible)
        v_r = 0
        if rxn.vmax_reverse > 0:
            v_r = rxn.vmax_reverse
            for prod, stoich in rxn.products.items():
                if prod in conc:
                    P = conc[prod]
                    Km = rxn.Km.get(prod, 0.1)
                    v_r *= P / (Km + P)
        
        return v_f - v_r
    
    def ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Calculate time derivatives."""
        
        dydt = np.zeros(self.n_states)
        
        # Build concentration dict (ensure non-negative)
        conc = {name: max(y[i], 1e-12) for i, name in enumerate(self.state_names)}
        
        # Add external metabolites
        conc['Glc_ext'] = self.external['Glc_ext']
        conc['AA_ext'] = self.external['AA_ext']
        
        # Calculate fluxes
        fluxes = {}
        for rxn in self.reactions:
            v = self.calculate_rate(rxn, conc)
            fluxes[rxn.name] = v
            
            # Update concentrations
            for met, stoich in rxn.stoichiometry.items():
                if met in self.idx:
                    dydt[self.idx[met]] += stoich * v
        
        # Add external demands (from gene expression module)
        if self.demands['protein_synthesis'] > 0:
            # This is handled by protein_syn reaction rate
            pass
        
        # Dilution by growth (all internal metabolites)
        for i in range(self.n_states):
            dydt[i] -= self.mu * y[i]
        
        return dydt
    
    def simulate(self, t_span: Tuple[float, float],
                 t_eval: Optional[np.ndarray] = None,
                 events: List = None) -> dict:
        """Run simulation."""
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        y0 = self.get_initial_state()
        
        print(f"\nSimulating {t_span[0]:.0f} to {t_span[1]:.0f} min...")
        
        solution = solve_ivp(
            self.ode_rhs,
            t_span,
            y0,
            t_eval=t_eval,
            method='LSODA',
            rtol=1e-8,
            atol=1e-10,
            events=events
        )
        
        # Build result dict
        result = {'t': solution.t, 'success': solution.success}
        for i, name in enumerate(self.state_names):
            result[name] = solution.y[i, :]
        
        # Calculate derived quantities
        result['energy_charge'] = (result['ATP'] + 0.5 * result['ADP']) / \
                                  (result['ATP'] + result['ADP'] + result['AMP'] + 1e-12)
        
        result['total_NTP'] = result['ATP'] + result['GTP'] + result['CTP'] + result['UTP']
        result['total_dNTP'] = result['dATP'] + result['dGTP'] + result['dCTP'] + result['dTTP']
        
        return result
    
    def analyze(self, result: dict, verbose: bool = True) -> dict:
        """Analyze simulation results."""
        
        t_final = result['t'][-1]
        
        # Final concentrations
        final = {name: result[name][-1] for name in self.state_names}
        
        # Energy charge
        ec = result['energy_charge'][-1]
        atp_adp = final['ATP'] / (final['ADP'] + 1e-12)
        
        # Redox state
        nad_ratio = final['NAD'] / (final['NADH'] + 1e-12)
        nadp_ratio = final['NADP'] / (final['NADPH'] + 1e-12)
        
        analysis = {
            'time': t_final,
            'energy_charge': ec,
            'ATP_ADP_ratio': atp_adp,
            'NAD_NADH_ratio': nad_ratio,
            'NADP_NADPH_ratio': nadp_ratio,
            'total_adenine': final['ATP'] + final['ADP'] + final['AMP'],
            'total_NTP': result['total_NTP'][-1],
            'total_dNTP': result['total_dNTP'][-1],
            'AA_pool': final['AA'],
            'lactate': final['Lac'],
            'pyruvate': final['Pyr'],
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"METABOLIC ANALYSIS (t = {t_final:.1f} min)")
            print("="*70)
            
            print("\n┌─────────────────────────────────────────────────────────────────┐")
            print("│                      ENERGY STATUS                              │")
            print("├─────────────────────────────────────────────────────────────────┤")
            print(f"│  ATP:           {final['ATP']:>8.3f} mM                                │")
            print(f"│  ADP:           {final['ADP']:>8.3f} mM                                │")
            print(f"│  AMP:           {final['AMP']:>8.3f} mM                                │")
            print(f"│  ATP/ADP:       {atp_adp:>8.1f}                                    │")
            print(f"│  Energy charge: {ec:>8.3f}   ", end="")
            if ec > 0.85:
                print("✓ HEALTHY                      │")
            elif ec > 0.7:
                print("△ MODERATE                     │")
            else:
                print("✗ STRESSED                     │")
            print("└─────────────────────────────────────────────────────────────────┘")
            
            print("\n┌─────────────────────────────────────────────────────────────────┐")
            print("│                      REDOX STATUS                               │")
            print("├─────────────────────────────────────────────────────────────────┤")
            print(f"│  NAD+:          {final['NAD']:>8.3f} mM                                │")
            print(f"│  NADH:          {final['NADH']:>8.3f} mM                                │")
            print(f"│  NAD+/NADH:     {nad_ratio:>8.1f}   ", end="")
            if nad_ratio > 5:
                print("✓ OXIDIZED (glycolysis OK)     │")
            else:
                print("△ REDUCED                      │")
            print(f"│  NADP+:         {final['NADP']:>8.3f} mM                                │")
            print(f"│  NADPH:         {final['NADPH']:>8.3f} mM                                │")
            print("└─────────────────────────────────────────────────────────────────┘")
            
            print("\n┌─────────────────────────────────────────────────────────────────┐")
            print("│                    NUCLEOTIDE POOLS                             │")
            print("├─────────────────────────────────────────────────────────────────┤")
            print(f"│  GTP:           {final['GTP']:>8.3f} mM                                │")
            print(f"│  CTP:           {final['CTP']:>8.3f} mM                                │")
            print(f"│  UTP:           {final['UTP']:>8.3f} mM                                │")
            print(f"│  Total NTPs:    {analysis['total_NTP']:>8.3f} mM                                │")
            print(f"│  Total dNTPs:   {analysis['total_dNTP']:>8.3f} mM                                │")
            print("└─────────────────────────────────────────────────────────────────┘")
            
            print("\n┌─────────────────────────────────────────────────────────────────┐")
            print("│                   PRECURSOR POOLS                               │")
            print("├─────────────────────────────────────────────────────────────────┤")
            print(f"│  Amino acids:   {final['AA']:>8.2f} mM                                │")
            print(f"│  R5P:           {final['R5P']:>8.3f} mM                                │")
            print(f"│  PRPP:          {final['PRPP']:>8.3f} mM                                │")
            print("└─────────────────────────────────────────────────────────────────┘")
            
            print("\n┌─────────────────────────────────────────────────────────────────┐")
            print("│                     GLYCOLYSIS                                  │")
            print("├─────────────────────────────────────────────────────────────────┤")
            print(f"│  Glucose (int): {final['Glc']:>8.3f} mM                                │")
            print(f"│  Pyruvate:      {final['Pyr']:>8.3f} mM                                │")
            print(f"│  Lactate:       {final['Lac']:>8.2f} mM                                │")
            print("└─────────────────────────────────────────────────────────────────┘")
            
            # Homeostasis check
            print(f"\n{'='*70}")
            print("HOMEOSTASIS CHECK")
            print("="*70)
            
            key_mets = ['ATP', 'NAD', 'AA', 'GTP']
            all_stable = True
            
            for met in key_mets:
                initial = result[met][0]
                final_val = result[met][-1]
                change = (final_val - initial) / (initial + 1e-12) * 100
                
                if abs(change) < 20:
                    status = "✓ STABLE"
                elif abs(change) < 50:
                    status = "△ ADJUSTING"
                    all_stable = False
                else:
                    status = "✗ UNSTABLE"
                    all_stable = False
                
                print(f"  {met:<6}: {initial:>6.3f} → {final_val:>6.3f} mM ({change:>+6.1f}%) {status}")
            
            if all_stable:
                print("\n🎉 METABOLISM IS IN HOMEOSTASIS!")
            else:
                print("\n⚠ System still adjusting (run longer or tune parameters)")
        
        return analysis


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("DARK MANIFOLD V48d: COMPLETE MINIMAL CELL METABOLISM")
    print("="*70)
    
    # Create model
    model = MinimalCellMetabolism(growth_rate=0.01)  # ~70 min doubling
    
    # Run simulation (3 hours to reach steady state)
    result = model.simulate(t_span=(0, 180))
    
    # Analyze
    analysis = model.analyze(result)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    t = result['t']
    
    # 1. Energy nucleotides
    ax = axes[0, 0]
    ax.plot(t, result['ATP'], 'b-', lw=2, label='ATP')
    ax.plot(t, result['ADP'], 'r-', lw=2, label='ADP')
    ax.plot(t, result['AMP'], 'g--', lw=1.5, label='AMP')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Adenine Nucleotides')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Energy charge
    ax = axes[0, 1]
    ax.plot(t, result['energy_charge'], 'k-', lw=2)
    ax.axhline(0.85, color='g', ls='--', alpha=0.5)
    ax.axhline(0.70, color='r', ls='--', alpha=0.5)
    ax.fill_between(t, 0.85, 0.95, alpha=0.1, color='green')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Energy Charge')
    ax.set_title('Cellular Energy Charge')
    ax.set_ylim([0.5, 1.0])
    ax.grid(alpha=0.3)
    
    # 3. Redox
    ax = axes[0, 2]
    ax.plot(t, result['NAD'], 'b-', lw=2, label='NAD+')
    ax.plot(t, result['NADH'], 'b--', lw=2, label='NADH')
    ax.plot(t, result['NADP'], 'r-', lw=2, label='NADP+')
    ax.plot(t, result['NADPH'], 'r--', lw=2, label='NADPH')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Redox Cofactors')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. NTPs
    ax = axes[1, 0]
    ax.plot(t, result['ATP'], 'b-', lw=2, label='ATP')
    ax.plot(t, result['GTP'], 'g-', lw=2, label='GTP')
    ax.plot(t, result['CTP'], 'r-', lw=2, label='CTP')
    ax.plot(t, result['UTP'], 'm-', lw=2, label='UTP')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Nucleoside Triphosphates')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 5. Amino acids & precursors
    ax = axes[1, 1]
    ax.plot(t, result['AA'], 'b-', lw=2, label='Amino acids')
    ax.plot(t, result['R5P']*10, 'g-', lw=2, label='Ribose-5-P (×10)')
    ax.plot(t, result['PRPP']*10, 'r-', lw=2, label='PRPP (×10)')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Biosynthetic Precursors')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 6. Glycolysis
    ax = axes[1, 2]
    ax.plot(t, result['Glc'], 'b-', lw=2, label='Glucose')
    ax.plot(t, result['Pyr'], 'r-', lw=2, label='Pyruvate')
    ax.plot(t, result['Lac'], 'g-', lw=2, label='Lactate')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Glycolysis')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metabolism_complete.png', dpi=150)
    print("\n✓ Saved: metabolism_complete.png")
    
    return model, result, analysis


if __name__ == '__main__':
    model, result, analysis = main()
