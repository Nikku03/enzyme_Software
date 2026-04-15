"""
Dark Manifold V48: Metabolism Module
=====================================

Full metabolic simulation for JCVI-syn3A:
- Glycolysis (ATP production)
- Pentose Phosphate Pathway (NADPH, ribose-5-P)
- Nucleotide biosynthesis (NTPs, dNTPs)
- Amino acid pools
- Lipid precursors

Coupled to gene expression:
- Transcription consumes NTPs
- Translation consumes ATP, GTP, amino acids
- All rates depend on metabolite availability
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys
sys.path.append('..')

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Cell parameters (JCVI-syn3A / Mycoplasma)
CELL_VOLUME = 5e-17          # L (50 fL - small cell)
AVOGADRO = 6.022e23
CELL_DRY_WEIGHT = 1e-13      # g (0.1 pg)

# Conversion: molecules to mM
def molecules_to_mM(n_molecules):
    """Convert molecule count to mM concentration."""
    return (n_molecules / AVOGADRO) / CELL_VOLUME * 1000

def mM_to_molecules(conc_mM):
    """Convert mM concentration to molecule count."""
    return conc_mM * 1e-3 * CELL_VOLUME * AVOGADRO

# Typical concentrations (mM)
ATP_CONC = 3.0          # mM
ADP_CONC = 0.5          # mM
GTP_CONC = 1.0          # mM
GDP_CONC = 0.2          # mM
NTP_TOTAL = 5.0         # mM (all 4 NTPs)
AA_TOTAL = 10.0         # mM (all 20 amino acids)
GLUCOSE_EXT = 10.0      # mM (external glucose)

# Convert to molecules
ATP_MOLECULES = mM_to_molecules(ATP_CONC)      # ~90,000
ADP_MOLECULES = mM_to_molecules(ADP_CONC)      # ~15,000
GTP_MOLECULES = mM_to_molecules(GTP_CONC)      # ~30,000


# ============================================================================
# METABOLITE CLASS
# ============================================================================

@dataclass
class Metabolite:
    """A metabolite pool with concentration dynamics."""
    name: str
    initial_conc: float      # mM
    
    # For charged molecules (ATP, NADH)
    is_energy_carrier: bool = False
    partner: str = None      # e.g., ATP's partner is ADP
    
    # Michaelis-Menten parameters
    Km: float = 0.1          # mM (typical Km)
    
    @property
    def initial_molecules(self) -> float:
        return mM_to_molecules(self.initial_conc)


# ============================================================================
# REACTION CLASS  
# ============================================================================

@dataclass
class Reaction:
    """A metabolic reaction with stoichiometry and kinetics."""
    name: str
    substrates: Dict[str, float]     # metabolite: stoichiometry
    products: Dict[str, float]       # metabolite: stoichiometry
    
    # Kinetics
    vmax: float = 1.0               # mM/min
    Km: Dict[str, float] = None     # substrate: Km
    
    # Enzyme
    enzyme_gene: str = None         # Gene encoding enzyme
    
    # Reversibility
    reversible: bool = False
    Keq: float = 1.0               # Equilibrium constant
    
    def __post_init__(self):
        if self.Km is None:
            self.Km = {s: 0.1 for s in self.substrates}


# ============================================================================
# GLYCOLYSIS
# ============================================================================

def build_glycolysis() -> List[Reaction]:
    """
    Glycolysis: Glucose → 2 Pyruvate + 2 ATP + 2 NADH
    
    Net reaction: Glucose + 2 NAD+ + 2 ADP + 2 Pi → 
                  2 Pyruvate + 2 NADH + 2 ATP + 2 H2O
    """
    
    reactions = []
    
    # 1. Glucose uptake (PTS system in some, permease in others)
    reactions.append(Reaction(
        name='glucose_uptake',
        substrates={'glucose_ext': 1, 'PEP': 1},
        products={'glucose_6P': 1, 'pyruvate': 1},
        vmax=0.5,  # mM/min
        enzyme_gene='ptsG',
    ))
    
    # 2. Glucose-6-P isomerase (G6P → F6P)
    reactions.append(Reaction(
        name='pgi',
        substrates={'glucose_6P': 1},
        products={'fructose_6P': 1},
        vmax=2.0,
        reversible=True,
        Keq=0.4,
        enzyme_gene='pgi',
    ))
    
    # 3. Phosphofructokinase (F6P + ATP → F1,6BP + ADP) - committed step
    reactions.append(Reaction(
        name='pfkA',
        substrates={'fructose_6P': 1, 'ATP': 1},
        products={'fructose_1_6_bisP': 1, 'ADP': 1},
        vmax=1.5,
        Km={'fructose_6P': 0.1, 'ATP': 0.05},
        enzyme_gene='pfkA',
    ))
    
    # 4. Aldolase (F1,6BP → DHAP + G3P)
    reactions.append(Reaction(
        name='fbaA',
        substrates={'fructose_1_6_bisP': 1},
        products={'DHAP': 1, 'G3P': 1},
        vmax=2.0,
        reversible=True,
        enzyme_gene='fbaA',
    ))
    
    # 5. Triose-P isomerase (DHAP ⟷ G3P)
    reactions.append(Reaction(
        name='tpiA',
        substrates={'DHAP': 1},
        products={'G3P': 1},
        vmax=10.0,  # Very fast
        reversible=True,
        Keq=0.05,
        enzyme_gene='tpiA',
    ))
    
    # 6. GAPDH (G3P + NAD+ + Pi → 1,3BPG + NADH)
    reactions.append(Reaction(
        name='gapA',
        substrates={'G3P': 1, 'NAD': 1, 'Pi': 1},
        products={'BPG_1_3': 1, 'NADH': 1},
        vmax=3.0,
        enzyme_gene='gapA',
    ))
    
    # 7. PGK (1,3BPG + ADP → 3PG + ATP) - first ATP!
    reactions.append(Reaction(
        name='pgk',
        substrates={'BPG_1_3': 1, 'ADP': 1},
        products={'PG_3': 1, 'ATP': 1},
        vmax=3.0,
        enzyme_gene='pgk',
    ))
    
    # 8. PGM (3PG → 2PG)
    reactions.append(Reaction(
        name='gpmA',
        substrates={'PG_3': 1},
        products={'PG_2': 1},
        vmax=5.0,
        reversible=True,
        enzyme_gene='gpmA',
    ))
    
    # 9. Enolase (2PG → PEP + H2O)
    reactions.append(Reaction(
        name='eno',
        substrates={'PG_2': 1},
        products={'PEP': 1},
        vmax=4.0,
        reversible=True,
        enzyme_gene='eno',
    ))
    
    # 10. Pyruvate kinase (PEP + ADP → Pyruvate + ATP) - second ATP!
    reactions.append(Reaction(
        name='pykF',
        substrates={'PEP': 1, 'ADP': 1},
        products={'pyruvate': 1, 'ATP': 1},
        vmax=2.0,
        Km={'PEP': 0.3, 'ADP': 0.2},
        enzyme_gene='pykF',
    ))
    
    # 11. Lactate dehydrogenase (Pyruvate + NADH → Lactate + NAD+)
    # Regenerates NAD+ for glycolysis
    reactions.append(Reaction(
        name='ldh',
        substrates={'pyruvate': 1, 'NADH': 1},
        products={'lactate': 1, 'NAD': 1},
        vmax=3.0,
        enzyme_gene='ldh',
    ))
    
    return reactions


# ============================================================================
# NUCLEOTIDE METABOLISM
# ============================================================================

def build_nucleotide_metabolism() -> List[Reaction]:
    """
    Nucleotide biosynthesis and interconversion.
    
    Simplified: 
    - Ribose-5-P + ATP → NMPs → NDPs → NTPs
    - NDP kinase equilibrates all NDPs/NTPs
    """
    
    reactions = []
    
    # Pentose phosphate pathway (simplified)
    reactions.append(Reaction(
        name='ppp_oxidative',
        substrates={'glucose_6P': 1, 'NADP': 2},
        products={'ribose_5P': 1, 'NADPH': 2, 'CO2': 1},
        vmax=0.3,
        enzyme_gene='zwf',
    ))
    
    # PRPP synthesis (ribose-5-P + ATP → PRPP + AMP)
    reactions.append(Reaction(
        name='prsA',
        substrates={'ribose_5P': 1, 'ATP': 1},
        products={'PRPP': 1, 'AMP': 1},
        vmax=0.2,
        enzyme_gene='prsA',
    ))
    
    # Purine synthesis (simplified: PRPP → IMP → AMP/GMP)
    reactions.append(Reaction(
        name='purine_denovo',
        substrates={'PRPP': 1, 'glutamine': 2, 'glycine': 1, 'ATP': 5, 'formate': 2},
        products={'IMP': 1, 'ADP': 5, 'glutamate': 2},
        vmax=0.1,
    ))
    
    # IMP → AMP
    reactions.append(Reaction(
        name='purA',
        substrates={'IMP': 1, 'aspartate': 1, 'GTP': 1},
        products={'AMP': 1, 'fumarate': 1, 'GDP': 1},
        vmax=0.2,
        enzyme_gene='purA',
    ))
    
    # IMP → GMP
    reactions.append(Reaction(
        name='guaB',
        substrates={'IMP': 1, 'NAD': 1, 'glutamine': 1, 'ATP': 1},
        products={'GMP': 1, 'NADH': 1, 'glutamate': 1, 'ADP': 1},
        vmax=0.2,
        enzyme_gene='guaB',
    ))
    
    # Pyrimidine synthesis (simplified)
    reactions.append(Reaction(
        name='pyrimidine_denovo',
        substrates={'carbamoyl_P': 1, 'aspartate': 1, 'PRPP': 1},
        products={'UMP': 1, 'Pi': 1},
        vmax=0.1,
    ))
    
    # UMP → CTP
    reactions.append(Reaction(
        name='pyrG',
        substrates={'UTP': 1, 'glutamine': 1, 'ATP': 1},
        products={'CTP': 1, 'glutamate': 1, 'ADP': 1},
        vmax=0.3,
        enzyme_gene='pyrG',
    ))
    
    # Nucleoside monophosphate kinases
    # AMP + ATP ⟷ 2 ADP
    reactions.append(Reaction(
        name='adk',
        substrates={'AMP': 1, 'ATP': 1},
        products={'ADP': 2},
        vmax=5.0,
        reversible=True,
        Keq=1.0,
        enzyme_gene='adk',
    ))
    
    # GMP + ATP → GDP + ADP
    reactions.append(Reaction(
        name='gmk',
        substrates={'GMP': 1, 'ATP': 1},
        products={'GDP': 1, 'ADP': 1},
        vmax=3.0,
        enzyme_gene='gmk',
    ))
    
    # UMP + ATP → UDP + ADP
    reactions.append(Reaction(
        name='pyrH',
        substrates={'UMP': 1, 'ATP': 1},
        products={'UDP': 1, 'ADP': 1},
        vmax=3.0,
        enzyme_gene='pyrH',
    ))
    
    # CMP + ATP → CDP + ADP
    reactions.append(Reaction(
        name='cmk',
        substrates={'CMP': 1, 'ATP': 1},
        products={'CDP': 1, 'ADP': 1},
        vmax=3.0,
        enzyme_gene='cmk',
    ))
    
    # NDP kinase (equilibrates all NDP/NTP pools)
    # NDP + ATP ⟷ NTP + ADP
    for nuc in ['GDP', 'UDP', 'CDP']:
        ntp = nuc[0] + 'TP'
        reactions.append(Reaction(
            name=f'ndk_{nuc}',
            substrates={nuc: 1, 'ATP': 1},
            products={ntp: 1, 'ADP': 1},
            vmax=10.0,  # Very fast
            reversible=True,
            Keq=1.0,
            enzyme_gene='ndk',
        ))
    
    # dNTP synthesis (ribonucleotide reductase)
    for nuc in ['ADP', 'GDP', 'CDP', 'UDP']:
        dnuc = 'd' + nuc
        reactions.append(Reaction(
            name=f'rnr_{nuc}',
            substrates={nuc: 1, 'NADPH': 1},
            products={dnuc: 1, 'NADP': 1},
            vmax=0.1,
            enzyme_gene='nrdA',
        ))
    
    # dNDP → dNTP (using ndk)
    for dnuc in ['dADP', 'dGDP', 'dCDP']:
        dntp = dnuc[:-1] + 'TP'
        reactions.append(Reaction(
            name=f'ndk_{dnuc}',
            substrates={dnuc: 1, 'ATP': 1},
            products={dntp: 1, 'ADP': 1},
            vmax=5.0,
            enzyme_gene='ndk',
        ))
    
    # dUDP → dUMP → dTMP → dTDP → dTTP
    reactions.append(Reaction(
        name='thymidylate',
        substrates={'dUMP': 1, 'methylene_THF': 1},
        products={'dTMP': 1, 'DHF': 1},
        vmax=0.2,
        enzyme_gene='thyA',
    ))
    
    reactions.append(Reaction(
        name='tmk',
        substrates={'dTMP': 1, 'ATP': 1},
        products={'dTDP': 1, 'ADP': 1},
        vmax=2.0,
        enzyme_gene='tmk',
    ))
    
    reactions.append(Reaction(
        name='ndk_dTDP',
        substrates={'dTDP': 1, 'ATP': 1},
        products={'dTTP': 1, 'ADP': 1},
        vmax=5.0,
        enzyme_gene='ndk',
    ))
    
    return reactions


# ============================================================================
# AMINO ACID METABOLISM
# ============================================================================

def build_amino_acid_metabolism() -> List[Reaction]:
    """
    Amino acid uptake and interconversion.
    
    JCVI-syn3A imports most amino acids from medium.
    Some interconversions are possible.
    """
    
    reactions = []
    
    # Amino acid uptake (simplified - one transporter)
    amino_acids = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly',
                   'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser',
                   'Thr', 'Trp', 'Tyr', 'Val']
    
    for aa in amino_acids:
        reactions.append(Reaction(
            name=f'uptake_{aa}',
            substrates={f'{aa}_ext': 1, 'ATP': 1},  # Active transport
            products={aa: 1, 'ADP': 1, 'Pi': 1},
            vmax=0.1,  # Slow but steady
            Km={f'{aa}_ext': 0.05},  # High affinity
        ))
    
    # Some key interconversions
    # Glutamate ⟷ Glutamine
    reactions.append(Reaction(
        name='glnA',
        substrates={'Glu': 1, 'ATP': 1, 'NH4': 1},
        products={'Gln': 1, 'ADP': 1, 'Pi': 1},
        vmax=0.5,
        enzyme_gene='glnA',
    ))
    
    # Aspartate ⟷ Asparagine
    reactions.append(Reaction(
        name='asnA',
        substrates={'Asp': 1, 'ATP': 1, 'NH4': 1},
        products={'Asn': 1, 'ADP': 1, 'Pi': 1},
        vmax=0.3,
        enzyme_gene='asnA',
    ))
    
    # Serine → Glycine + C1
    reactions.append(Reaction(
        name='glyA',
        substrates={'Ser': 1, 'THF': 1},
        products={'Gly': 1, 'methylene_THF': 1},
        vmax=0.3,
        reversible=True,
        enzyme_gene='glyA',
    ))
    
    return reactions


# ============================================================================
# ATP CONSUMPTION PROCESSES
# ============================================================================

def build_atp_consumers() -> List[Reaction]:
    """
    Major ATP-consuming processes.
    
    This is where the energy goes!
    """
    
    reactions = []
    
    # Protein synthesis (dominant ATP consumer)
    # ~4 ATP equivalents per amino acid (2 ATP charging + 2 GTP elongation)
    reactions.append(Reaction(
        name='protein_synthesis',
        substrates={'ATP': 2, 'GTP': 2, 'AA_pool': 1},
        products={'ADP': 2, 'GDP': 2, 'Pi': 4, 'protein_mass': 1},
        vmax=1.0,  # Adjusted by model based on ribosome activity
    ))
    
    # RNA synthesis
    # 1 NTP per nucleotide
    reactions.append(Reaction(
        name='rna_synthesis',
        substrates={'NTP_pool': 1},
        products={'NMP_incorporated': 1, 'PPi': 1},
        vmax=0.5,
    ))
    
    # DNA synthesis (during replication)
    reactions.append(Reaction(
        name='dna_synthesis',
        substrates={'dNTP_pool': 1},
        products={'dNMP_incorporated': 1, 'PPi': 1},
        vmax=0.1,  # Only during replication
    ))
    
    # Membrane maintenance
    reactions.append(Reaction(
        name='membrane_maintenance',
        substrates={'ATP': 1},
        products={'ADP': 1, 'Pi': 1},
        vmax=0.1,  # Basal
    ))
    
    # Chaperone activity (protein folding)
    reactions.append(Reaction(
        name='chaperone_atp',
        substrates={'ATP': 1},
        products={'ADP': 1, 'Pi': 1},
        vmax=0.2,  # ~10% of total ATP
    ))
    
    return reactions


# ============================================================================
# METABOLISM MODEL CLASS
# ============================================================================

class MetabolismModel:
    """
    Full metabolic model with coupled pools.
    
    State vector:
    - All metabolite concentrations (mM)
    
    Features:
    - Glycolysis for ATP production
    - Nucleotide biosynthesis
    - Amino acid pools
    - Coupling to gene expression (via consumption rates)
    """
    
    def __init__(self):
        # Build reactions
        self.reactions = []
        self.reactions.extend(build_glycolysis())
        self.reactions.extend(build_nucleotide_metabolism())
        self.reactions.extend(build_amino_acid_metabolism())
        self.reactions.extend(build_atp_consumers())
        
        # Index reactions
        self.reaction_names = [r.name for r in self.reactions]
        
        # Collect all metabolites
        self._collect_metabolites()
        
        # Initialize concentrations
        self._initialize_concentrations()
        
        # External conditions
        self.glucose_ext = 10.0  # mM
        self.aa_ext = {aa: 1.0 for aa in 
                       ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly',
                        'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser',
                        'Thr', 'Trp', 'Tyr', 'Val']}
        
        self._print_summary()
    
    def _collect_metabolites(self):
        """Find all metabolites in reactions."""
        self.metabolites = set()
        for rxn in self.reactions:
            self.metabolites.update(rxn.substrates.keys())
            self.metabolites.update(rxn.products.keys())
        self.metabolites = sorted(list(self.metabolites))
        self.n_metabolites = len(self.metabolites)
        self.met_idx = {m: i for i, m in enumerate(self.metabolites)}
    
    def _initialize_concentrations(self):
        """Set initial metabolite concentrations."""
        self.initial_conc = {}
        
        # Energy carriers
        self.initial_conc['ATP'] = 3.0
        self.initial_conc['ADP'] = 0.5
        self.initial_conc['AMP'] = 0.1
        self.initial_conc['GTP'] = 1.0
        self.initial_conc['GDP'] = 0.2
        self.initial_conc['GMP'] = 0.05
        
        # Other NTPs
        self.initial_conc['UTP'] = 1.0
        self.initial_conc['UDP'] = 0.2
        self.initial_conc['UMP'] = 0.05
        self.initial_conc['CTP'] = 0.8
        self.initial_conc['CDP'] = 0.15
        self.initial_conc['CMP'] = 0.05
        
        # dNTPs (much lower)
        for dntp in ['dATP', 'dGTP', 'dCTP', 'dTTP']:
            self.initial_conc[dntp] = 0.05
        for dndp in ['dADP', 'dGDP', 'dCDP', 'dTDP', 'dUDP']:
            self.initial_conc[dndp] = 0.02
        self.initial_conc['dUMP'] = 0.01
        self.initial_conc['dTMP'] = 0.01
        
        # Cofactors
        self.initial_conc['NAD'] = 1.0
        self.initial_conc['NADH'] = 0.1
        self.initial_conc['NADP'] = 0.3
        self.initial_conc['NADPH'] = 0.5
        self.initial_conc['Pi'] = 5.0
        self.initial_conc['PPi'] = 0.1
        
        # Glycolysis intermediates
        self.initial_conc['glucose_6P'] = 0.5
        self.initial_conc['fructose_6P'] = 0.2
        self.initial_conc['fructose_1_6_bisP'] = 0.1
        self.initial_conc['DHAP'] = 0.3
        self.initial_conc['G3P'] = 0.1
        self.initial_conc['BPG_1_3'] = 0.05
        self.initial_conc['PG_3'] = 0.3
        self.initial_conc['PG_2'] = 0.1
        self.initial_conc['PEP'] = 0.3
        self.initial_conc['pyruvate'] = 0.5
        self.initial_conc['lactate'] = 0.1
        
        # Nucleotide precursors
        self.initial_conc['ribose_5P'] = 0.2
        self.initial_conc['PRPP'] = 0.05
        self.initial_conc['IMP'] = 0.1
        
        # Amino acids
        for aa in ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly',
                   'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser',
                   'Thr', 'Trp', 'Tyr', 'Val']:
            self.initial_conc[aa] = 0.5  # mM
            self.initial_conc[f'{aa}_ext'] = 1.0  # External
        
        # Folate
        self.initial_conc['THF'] = 0.1
        self.initial_conc['methylene_THF'] = 0.05
        self.initial_conc['DHF'] = 0.02
        
        # Other
        self.initial_conc['glucose_ext'] = 10.0
        self.initial_conc['NH4'] = 1.0
        self.initial_conc['CO2'] = 1.0
        self.initial_conc['formate'] = 0.5
        self.initial_conc['glycine'] = 0.5  # Also an AA
        self.initial_conc['glutamine'] = 0.5  # Gln
        self.initial_conc['glutamate'] = 0.5  # Glu
        self.initial_conc['aspartate'] = 0.5  # Asp
        self.initial_conc['fumarate'] = 0.1
        self.initial_conc['carbamoyl_P'] = 0.05
        
        # Pooled metabolites for consumers
        self.initial_conc['AA_pool'] = 10.0
        self.initial_conc['NTP_pool'] = 5.0
        self.initial_conc['dNTP_pool'] = 0.2
        self.initial_conc['protein_mass'] = 0.0
        self.initial_conc['NMP_incorporated'] = 0.0
        self.initial_conc['dNMP_incorporated'] = 0.0
        
        # Set defaults for any missing metabolites
        for met in self.metabolites:
            if met not in self.initial_conc:
                self.initial_conc[met] = 0.01  # Default very low
    
    def _print_summary(self):
        """Print model summary."""
        print(f"\n{'='*70}")
        print("METABOLISM MODEL")
        print("="*70)
        print(f"Metabolites: {self.n_metabolites}")
        print(f"Reactions: {len(self.reactions)}")
        
        # Categorize reactions
        categories = defaultdict(int)
        for rxn in self.reactions:
            if 'glucose' in rxn.name or rxn.name in ['pgi', 'pfkA', 'fbaA', 'tpiA', 
                                                       'gapA', 'pgk', 'gpmA', 'eno', 
                                                       'pykF', 'ldh']:
                categories['glycolysis'] += 1
            elif 'uptake' in rxn.name:
                categories['transport'] += 1
            elif any(x in rxn.name for x in ['ndk', 'adk', 'gmk', 'cmk', 'tmk', 
                                              'pyrH', 'pyrG', 'rnr', 'prsA']):
                categories['nucleotide'] += 1
            elif 'synthesis' in rxn.name:
                categories['biosynthesis'] += 1
            else:
                categories['other'] += 1
        
        print("\nReactions by category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
    
    def get_initial_state(self) -> np.ndarray:
        """Get initial state vector."""
        state = np.zeros(self.n_metabolites)
        for met, conc in self.initial_conc.items():
            if met in self.met_idx:
                state[self.met_idx[met]] = conc
        return state
    
    def reaction_rate(self, rxn: Reaction, concentrations: Dict[str, float]) -> float:
        """Calculate reaction rate using Michaelis-Menten kinetics."""
        
        # Get substrate concentrations
        rate = rxn.vmax
        
        for substrate, stoich in rxn.substrates.items():
            conc = concentrations.get(substrate, 0)
            Km = rxn.Km.get(substrate, 0.1)
            
            # Michaelis-Menten
            if conc > 0 and Km > 0:
                rate *= (conc / (Km + conc))
            else:
                rate = 0
                break
        
        # Reversibility (simplified mass action)
        if rxn.reversible and rate > 0:
            product_term = 1.0
            for product, stoich in rxn.products.items():
                conc = concentrations.get(product, 0)
                product_term *= (conc ** stoich)
            
            substrate_term = 1.0
            for substrate, stoich in rxn.substrates.items():
                conc = concentrations.get(substrate, 0)
                substrate_term *= (conc ** stoich)
            
            if substrate_term > 0:
                ratio = product_term / (substrate_term * rxn.Keq)
                rate *= max(0, 1 - ratio)  # Slow down near equilibrium
        
        return rate
    
    def ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """ODE right-hand side."""
        
        dydt = np.zeros_like(y)
        
        # Build concentration dict
        conc = {met: max(y[i], 0) for i, met in enumerate(self.metabolites)}
        
        # Update external pools (constant in this version)
        conc['glucose_ext'] = self.glucose_ext
        for aa, ext_conc in self.aa_ext.items():
            conc[f'{aa}_ext'] = ext_conc
        
        # Update pooled metabolites
        conc['AA_pool'] = sum(conc.get(aa, 0) for aa in 
                             ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly',
                              'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser',
                              'Thr', 'Trp', 'Tyr', 'Val'])
        conc['NTP_pool'] = conc.get('ATP', 0) + conc.get('GTP', 0) + \
                          conc.get('CTP', 0) + conc.get('UTP', 0)
        conc['dNTP_pool'] = conc.get('dATP', 0) + conc.get('dGTP', 0) + \
                           conc.get('dCTP', 0) + conc.get('dTTP', 0)
        
        # Calculate reaction rates and update concentrations
        for rxn in self.reactions:
            rate = self.reaction_rate(rxn, conc)
            
            # Consume substrates
            for substrate, stoich in rxn.substrates.items():
                if substrate in self.met_idx:
                    dydt[self.met_idx[substrate]] -= rate * stoich
            
            # Produce products
            for product, stoich in rxn.products.items():
                if product in self.met_idx:
                    dydt[self.met_idx[product]] += rate * stoich
        
        return dydt
    
    def simulate(self, t_span: Tuple[float, float], 
                 t_eval: Optional[np.ndarray] = None) -> dict:
        """Run simulation."""
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)
        
        initial_state = self.get_initial_state()
        
        print(f"Simulating metabolism from t={t_span[0]} to t={t_span[1]} min...")
        
        solution = solve_ivp(
            self.ode_rhs,
            t_span,
            initial_state,
            t_eval=t_eval,
            method='LSODA',
            rtol=1e-6,
            atol=1e-9,
        )
        
        # Build result dict
        result = {'t': solution.t}
        for i, met in enumerate(self.metabolites):
            result[met] = solution.y[i, :]
        
        return result
    
    def get_atp_balance(self, result: dict) -> dict:
        """Analyze ATP production and consumption."""
        
        t = result['t']
        atp = result['ATP']
        adp = result['ADP']
        
        # Energy charge = (ATP + 0.5*ADP) / (ATP + ADP + AMP)
        amp = result.get('AMP', np.zeros_like(atp))
        total_adenine = atp + adp + amp
        energy_charge = (atp + 0.5 * adp) / (total_adenine + 1e-10)
        
        return {
            'ATP_final': atp[-1],
            'ADP_final': adp[-1],
            'AMP_final': amp[-1],
            'ATP_ADP_ratio': atp[-1] / (adp[-1] + 1e-10),
            'energy_charge': energy_charge[-1],
            'total_adenine': total_adenine[-1],
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("DARK MANIFOLD V48: METABOLISM")
    print("="*70)
    
    # Create model
    model = MetabolismModel()
    
    # Simulate 30 minutes
    result = model.simulate(t_span=(0, 30))
    
    # Analyze
    atp_balance = model.get_atp_balance(result)
    
    print(f"\n{'='*70}")
    print("RESULTS (t = 30 min)")
    print("="*70)
    
    print("\nEnergy status:")
    print(f"  ATP: {atp_balance['ATP_final']:.2f} mM")
    print(f"  ADP: {atp_balance['ADP_final']:.2f} mM")
    print(f"  AMP: {atp_balance['AMP_final']:.2f} mM")
    print(f"  ATP/ADP ratio: {atp_balance['ATP_ADP_ratio']:.1f}")
    print(f"  Energy charge: {atp_balance['energy_charge']:.2f} (healthy: 0.8-0.95)")
    
    print("\nNucleotides:")
    for nuc in ['GTP', 'UTP', 'CTP']:
        print(f"  {nuc}: {result[nuc][-1]:.2f} mM")
    
    print("\nGlycolysis intermediates:")
    for met in ['glucose_6P', 'fructose_6P', 'pyruvate', 'lactate']:
        print(f"  {met}: {result[met][-1]:.3f} mM")
    
    print("\nAmino acids (sample):")
    for aa in ['Glu', 'Gln', 'Asp', 'Ala']:
        print(f"  {aa}: {result[aa][-1]:.2f} mM")
    
    # Check homeostasis
    print(f"\n{'='*70}")
    print("HOMEOSTASIS CHECK")
    print("="*70)
    
    atp_initial = model.initial_conc['ATP']
    atp_final = result['ATP'][-1]
    atp_change = (atp_final - atp_initial) / atp_initial * 100
    
    print(f"ATP change: {atp_change:+.1f}%")
    
    if abs(atp_change) < 20:
        print("✓ ATP pool is stable (good homeostasis)")
    elif atp_change > 0:
        print("⚠ ATP accumulating (production > consumption)")
    else:
        print("⚠ ATP depleting (consumption > production)")
    
    return model, result


if __name__ == '__main__':
    model, result = main()
