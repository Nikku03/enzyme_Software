"""
Dark Manifold V47b: Enhanced Gene Expression Model
==================================================

Improvements over v1:
1. Proper resource accounting (RNAP/ribosome engagement times)
2. Gene length affects elongation time
3. More genes (full JCVI-syn3A set)
4. Regulation: autorepression, operons
5. Cell growth and dilution
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ============================================================================
# CONSTANTS
# ============================================================================

# Physical constants for Mycoplasma/minimal cell
CELL_VOLUME = 5e-17  # liters (50 femtoliters, small cell)
AVOGADRO = 6.022e23

# Macromolecular composition (from literature)
PROTEIN_PER_CELL = 50000          # Total protein molecules
MRNA_PER_CELL = 500               # Total mRNA molecules  
RRNA_PER_CELL = 3000              # Ribosomal RNA (for 1000 ribosomes)

# Kinetic parameters
RNAP_SPEED = 40             # nt/s transcription
RIBOSOME_SPEED = 15         # aa/s translation
MRNA_HALFLIFE_MEAN = 3.0    # min
PROTEIN_HALFLIFE_MEAN = 60  # min (dilution dominates)

# Cell cycle
DOUBLING_TIME = 90          # min (Mycoplasma ~1-2 hr)
GROWTH_RATE = np.log(2) / DOUBLING_TIME  # per min


# ============================================================================
# GENE DATABASE
# ============================================================================

@dataclass
class Gene:
    """Gene with full expression parameters."""
    name: str
    length_nt: int
    length_aa: int
    category: str
    
    # Promoter properties
    promoter_strength: float = 1.0   # Relative (0-10)
    sigma_factor: str = 'sig70'      # Which sigma factor
    
    # RBS properties  
    rbs_strength: float = 1.0        # Relative (0-10)
    
    # Codon optimization
    codon_adaptation: float = 0.7    # CAI (0-1)
    
    # mRNA properties
    mrna_halflife: float = 3.0       # min
    
    # Protein properties
    protein_halflife: float = 60.0   # min
    
    # Operon membership
    operon: Optional[str] = None     # Operon name if part of one
    operon_position: int = 0         # Position in operon (1-indexed)
    
    # Regulation
    autorepression: bool = False     # Does protein repress own gene?
    K_autorepression: float = 100    # Repression constant
    
    # Calculated rates (set by model)
    k_tx_init: float = 0.0           # Transcription initiation (1/min)
    k_tl_init: float = 0.0           # Translation initiation (1/min/mRNA)
    
    @property
    def delta_m(self) -> float:
        return np.log(2) / self.mrna_halflife
    
    @property
    def delta_p(self) -> float:
        return np.log(2) / self.protein_halflife + GROWTH_RATE  # Decay + dilution
    
    @property 
    def tx_elongation_time(self) -> float:
        """Time for RNAP to transcribe gene (min)."""
        return self.length_nt / (RNAP_SPEED * 60)
    
    @property
    def tl_elongation_time(self) -> float:
        """Time for ribosome to translate mRNA (min)."""
        return self.length_aa / (RIBOSOME_SPEED * 60)


def build_full_syn3a_genes() -> Dict[str, Gene]:
    """Build comprehensive JCVI-syn3A gene database."""
    
    genes = {}
    
    # ========== RIBOSOMAL PROTEIN OPERONS ==========
    # S10 operon (rpsJ-rplC-rplD-rplW-rplB-rpsS-rplV-rpsC-rplP-rpmC-rpsQ)
    s10_operon = [
        ('rpsJ', 103, 'ribosome_30S'), ('rplC', 209, 'ribosome_50S'),
        ('rplD', 203, 'ribosome_50S'), ('rplW', 104, 'ribosome_50S'),
        ('rplB', 277, 'ribosome_50S'), ('rpsS', 93, 'ribosome_30S'),
        ('rplV', 110, 'ribosome_50S'), ('rpsC', 233, 'ribosome_30S'),
        ('rplP', 136, 'ribosome_50S'), ('rpmC', 64, 'ribosome_50S'),
        ('rpsQ', 93, 'ribosome_30S'),
    ]
    for i, (name, length_aa, cat) in enumerate(s10_operon):
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category=cat, promoter_strength=9.0, rbs_strength=8.0,
            codon_adaptation=0.9, mrna_halflife=5.0, protein_halflife=180.0,
            operon='S10', operon_position=i+1
        )
    
    # spc operon (rplN-rplX-rplE-rpsN-rpsH-rplF-rplR-rpsE-rpmD-rplO)
    spc_operon = [
        ('rplN', 123, 'ribosome_50S'), ('rplX', 104, 'ribosome_50S'),
        ('rplE', 182, 'ribosome_50S'), ('rpsN', 101, 'ribosome_30S'),
        ('rpsH', 138, 'ribosome_30S'), ('rplF', 177, 'ribosome_50S'),
        ('rplR', 117, 'ribosome_50S'), ('rpsE', 167, 'ribosome_30S'),
        ('rpmD', 60, 'ribosome_50S'), ('rplO', 144, 'ribosome_50S'),
    ]
    for i, (name, length_aa, cat) in enumerate(spc_operon):
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category=cat, promoter_strength=9.0, rbs_strength=8.0,
            codon_adaptation=0.9, mrna_halflife=5.0, protein_halflife=180.0,
            operon='spc', operon_position=i+1
        )
    
    # str operon (rpsL-rpsG-fusA-tuf)
    str_operon = [
        ('rpsL', 124, 'ribosome_30S'), ('rpsG', 156, 'ribosome_30S'),
        ('fusA', 697, 'translation_factor'), ('tuf', 394, 'translation_factor'),
    ]
    for i, (name, length_aa, cat) in enumerate(str_operon):
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category=cat, promoter_strength=9.0, rbs_strength=9.0,
            codon_adaptation=0.95, mrna_halflife=5.0, protein_halflife=180.0,
            operon='str', operon_position=i+1
        )
    
    # Additional ribosomal proteins (not in main operons shown)
    other_ribosomal = [
        ('rpsA', 557, 'ribosome_30S'), ('rpsB', 261, 'ribosome_30S'),
        ('rpsD', 206, 'ribosome_30S'), ('rpsF', 131, 'ribosome_30S'),
        ('rpsI', 130, 'ribosome_30S'), ('rpsK', 129, 'ribosome_30S'),
        ('rpsM', 119, 'ribosome_30S'), ('rpsO', 89, 'ribosome_30S'),
        ('rpsP', 82, 'ribosome_30S'), ('rpsR', 75, 'ribosome_30S'),
        ('rpsT', 75, 'ribosome_30S'),
        ('rplA', 231, 'ribosome_50S'), ('rplJ', 166, 'ribosome_50S'),
        ('rplK', 144, 'ribosome_50S'), ('rplL', 124, 'ribosome_50S'),
        ('rplM', 144, 'ribosome_50S'), ('rplQ', 130, 'ribosome_50S'),
        ('rplS', 115, 'ribosome_50S'), ('rplT', 118, 'ribosome_50S'),
        ('rplU', 103, 'ribosome_50S'), ('rpmA', 89, 'ribosome_50S'),
        ('rpmB', 81, 'ribosome_50S'), ('rpmE', 50, 'ribosome_50S'),
        ('rpmF', 58, 'ribosome_50S'), ('rpmG', 53, 'ribosome_50S'),
        ('rpmH', 46, 'ribosome_50S'), ('rpmI', 66, 'ribosome_50S'),
        ('rpmJ', 38, 'ribosome_50S'),
    ]
    for name, length_aa, cat in other_ribosomal:
        if name not in genes:
            genes[name] = Gene(
                name=name, length_nt=length_aa*3, length_aa=length_aa,
                category=cat, promoter_strength=8.0, rbs_strength=8.0,
                codon_adaptation=0.9, mrna_halflife=5.0, protein_halflife=180.0
            )
    
    # ========== tRNA SYNTHETASES ==========
    trna_synthetases = [
        ('alaS', 876), ('argS', 577), ('asnS', 467), ('aspS', 590),
        ('cysS', 461), ('glnS', 554), ('gltX', 471), ('glyS', 465),
        ('hisS', 424), ('ileS', 939), ('leuS', 860), ('lysS', 505),
        ('metG', 662), ('pheS', 350), ('pheT', 795), ('proS', 478),
        ('serS', 421), ('thrS', 642), ('trpS', 334), ('tyrS', 424),
        ('valS', 880),
    ]
    for name, length_aa in trna_synthetases:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='tRNA_synthetase', promoter_strength=5.0, rbs_strength=5.0,
            codon_adaptation=0.8, mrna_halflife=3.0, protein_halflife=120.0
        )
    
    # ========== TRANSLATION FACTORS ==========
    translation_factors = [
        ('infA', 73), ('infB', 741), ('infC', 181),
        ('tsf', 283), ('prfA', 360), ('prfB', 365),
        ('frr', 185), ('efp', 186),
    ]
    for name, length_aa in translation_factors:
        if name not in genes:
            genes[name] = Gene(
                name=name, length_nt=length_aa*3, length_aa=length_aa,
                category='translation_factor', promoter_strength=6.0, rbs_strength=7.0,
                codon_adaptation=0.85, mrna_halflife=4.0, protein_halflife=120.0
            )
    
    # ========== RNAP AND SIGMA FACTORS ==========
    # rpoBC operon
    genes['rpoB'] = Gene(
        name='rpoB', length_nt=1342*3, length_aa=1342,
        category='RNAP', promoter_strength=4.0, rbs_strength=4.0,
        codon_adaptation=0.75, mrna_halflife=3.0, protein_halflife=180.0,
        operon='rpoBC', operon_position=1
    )
    genes['rpoC'] = Gene(
        name='rpoC', length_nt=1524*3, length_aa=1524,
        category='RNAP', promoter_strength=4.0, rbs_strength=4.0,
        codon_adaptation=0.75, mrna_halflife=3.0, protein_halflife=180.0,
        operon='rpoBC', operon_position=2
    )
    genes['rpoA'] = Gene(
        name='rpoA', length_nt=329*3, length_aa=329,
        category='RNAP', promoter_strength=4.0, rbs_strength=4.0,
        codon_adaptation=0.75, mrna_halflife=3.0, protein_halflife=180.0
    )
    genes['rpoD'] = Gene(
        name='rpoD', length_nt=437*3, length_aa=437,
        category='RNAP', promoter_strength=3.0, rbs_strength=3.0,
        codon_adaptation=0.7, mrna_halflife=2.5, protein_halflife=120.0,
        autorepression=True, K_autorepression=200  # Sigma-70 autoregulates
    )
    
    # ========== DNA REPLICATION ==========
    replication = [
        ('dnaA', 454), ('dnaB', 471), ('dnaC', 259), ('dnaE', 1160),
        ('dnaG', 342), ('dnaN', 378), ('dnaX', 455),
        ('gyrA', 820), ('gyrB', 647), ('parC', 750), ('parE', 630),
        ('ssb', 166), ('ligA', 671), ('polA', 605),
    ]
    for name, length_aa in replication:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='replication', promoter_strength=2.0, rbs_strength=2.5,
            codon_adaptation=0.7, mrna_halflife=2.0, protein_halflife=90.0
        )
    # DnaA autorepresses its own promoter
    genes['dnaA'].autorepression = True
    genes['dnaA'].K_autorepression = 50
    
    # ========== CELL DIVISION ==========
    division = [
        ('ftsZ', 395), ('ftsA', 420), ('ftsB', 105), ('ftsL', 121),
        ('ftsQ', 276), ('ftsW', 414), ('ftsI', 588), ('ftsH', 640),
    ]
    for name, length_aa in division:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='division', promoter_strength=2.5, rbs_strength=3.0,
            codon_adaptation=0.7, mrna_halflife=2.0, protein_halflife=60.0
        )
    
    # ========== MEMBRANE/SECRETION ==========
    membrane = [
        ('secA', 820), ('secY', 435), ('secE', 68), ('secG', 110),
        ('secD', 523), ('secF', 315),
        ('ffh', 453), ('ftsY', 497),
        ('yidC', 270), ('lepB', 324), ('lspA', 164),
    ]
    for name, length_aa in membrane:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='membrane', promoter_strength=3.0, rbs_strength=3.5,
            codon_adaptation=0.7, mrna_halflife=2.5, protein_halflife=90.0
        )
    
    # ========== CHAPERONES ==========
    # groE operon
    genes['groES'] = Gene(
        name='groES', length_nt=97*3, length_aa=97,
        category='chaperone', promoter_strength=6.0, rbs_strength=6.0,
        codon_adaptation=0.85, mrna_halflife=4.0, protein_halflife=120.0,
        operon='groE', operon_position=1
    )
    genes['groEL'] = Gene(
        name='groEL', length_nt=548*3, length_aa=548,
        category='chaperone', promoter_strength=6.0, rbs_strength=6.0,
        codon_adaptation=0.85, mrna_halflife=4.0, protein_halflife=120.0,
        operon='groE', operon_position=2
    )
    # dnaK-dnaJ-grpE operon
    genes['dnaK'] = Gene(
        name='dnaK', length_nt=638*3, length_aa=638,
        category='chaperone', promoter_strength=5.0, rbs_strength=5.0,
        codon_adaptation=0.8, mrna_halflife=3.0, protein_halflife=120.0,
        operon='dnaKJ', operon_position=1
    )
    genes['dnaJ'] = Gene(
        name='dnaJ', length_nt=376*3, length_aa=376,
        category='chaperone', promoter_strength=5.0, rbs_strength=5.0,
        codon_adaptation=0.8, mrna_halflife=3.0, protein_halflife=120.0,
        operon='dnaKJ', operon_position=2
    )
    genes['grpE'] = Gene(
        name='grpE', length_nt=197*3, length_aa=197,
        category='chaperone', promoter_strength=5.0, rbs_strength=5.0,
        codon_adaptation=0.8, mrna_halflife=3.0, protein_halflife=120.0
    )
    
    # ========== METABOLISM ==========
    metabolism = [
        # Glycolysis
        ('pgi', 445), ('pfkA', 320), ('fbaA', 359), ('tpiA', 255),
        ('gapA', 337), ('pgk', 400), ('gpmA', 250), ('eno', 432), ('pykF', 470),
        # Nucleotide metabolism
        ('ndk', 143), ('adk', 214), ('cmk', 227), ('gmk', 220), ('tmk', 212),
        ('pyrG', 545), ('pyrH', 241), ('prsA', 316),
        # ATP synthesis
        ('atpA', 505), ('atpB', 156), ('atpC', 286), ('atpD', 470),
        ('atpE', 79), ('atpF', 156), ('atpG', 287), ('atpH', 177),
    ]
    for name, length_aa in metabolism:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='metabolism', promoter_strength=4.0, rbs_strength=4.0,
            codon_adaptation=0.75, mrna_halflife=2.5, protein_halflife=90.0
        )
    
    # ========== LIPID SYNTHESIS ==========
    lipid = [
        ('accA', 319), ('accB', 156), ('accC', 449), ('accD', 304),
        ('fabD', 309), ('fabG', 244), ('fabH', 317), ('fabI', 262),
        ('fabZ', 151), ('acpP', 78), ('acpS', 126),
        ('plsB', 807), ('plsC', 245), ('cdsA', 270),
    ]
    for name, length_aa in lipid:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='lipid', promoter_strength=3.5, rbs_strength=3.5,
            codon_adaptation=0.7, mrna_halflife=2.5, protein_halflife=90.0
        )
    
    # ========== COFACTOR SYNTHESIS ==========
    cofactors = [
        ('coaA', 316), ('coaD', 159), ('coaE', 206),
        ('folA', 159), ('folC', 422), ('folD', 287), ('folE', 221),
        ('nadD', 213), ('nadE', 275),
        ('ribA', 196), ('ribB', 217), ('ribC', 213), ('ribD', 366), ('ribF', 313),
    ]
    for name, length_aa in cofactors:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='cofactor', promoter_strength=3.0, rbs_strength=3.0,
            codon_adaptation=0.7, mrna_halflife=2.5, protein_halflife=90.0
        )
    
    return genes


# ============================================================================
# ENHANCED MODEL
# ============================================================================

class EnhancedGeneExpressionModel:
    """
    Enhanced ODE model with:
    - Resource engagement (RNAP and ribosome sequestration during elongation)
    - Gene length effects
    - Operon structure
    - Autorepression
    - Cell growth (dilution)
    """
    
    def __init__(self, genes: Dict[str, Gene]):
        self.genes = genes
        self.gene_list = list(genes.keys())
        self.n_genes = len(self.gene_list)
        
        # Resource pools
        self.total_RNAP = 200          # Total RNAP holoenzyme
        self.total_ribosomes = 1000    # Total ribosomes
        
        # Scaling to match ~500 mRNA, ~50000 protein per cell
        self.k_tx_scale = 0.5          # Global transcription scaling
        self.k_tl_scale = 2.0          # Global translation scaling
        
        # Calculate rates
        self._calculate_rates()
        
        # Index operons
        self._index_operons()
        
        # State indices
        self.mrna_idx = {name: i for i, name in enumerate(self.gene_list)}
        self.protein_idx = {name: self.n_genes + i for i, name in enumerate(self.gene_list)}
        
        self._print_summary()
    
    def _calculate_rates(self):
        """Set rates for each gene."""
        for gene in self.genes.values():
            # Transcription initiation rate
            gene.k_tx_init = self.k_tx_scale * (gene.promoter_strength / 10.0)
            
            # Translation initiation rate (per mRNA per min)
            gene.k_tl_init = self.k_tl_scale * (gene.rbs_strength / 10.0) * gene.codon_adaptation
    
    def _index_operons(self):
        """Group genes by operon for coordinated transcription."""
        self.operons = defaultdict(list)
        for name, gene in self.genes.items():
            if gene.operon:
                self.operons[gene.operon].append(name)
        
        # Sort by position
        for operon, gene_names in self.operons.items():
            self.operons[operon] = sorted(
                gene_names, 
                key=lambda x: self.genes[x].operon_position
            )
    
    def _print_summary(self):
        """Print model summary."""
        print(f"\n{'='*70}")
        print("ENHANCED GENE EXPRESSION MODEL")
        print("="*70)
        print(f"Total genes: {self.n_genes}")
        print(f"Total RNAP: {self.total_RNAP}")
        print(f"Total ribosomes: {self.total_ribosomes}")
        print(f"Operons: {len(self.operons)}")
        
        # Count by category
        by_cat = defaultdict(int)
        for gene in self.genes.values():
            by_cat[gene.category] += 1
        
        print("\nGenes by category:")
        for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
            print(f"  {cat:<20} {count:>3}")
    
    def get_initial_state(self) -> np.ndarray:
        """Initialize at approximate steady state."""
        state = np.zeros(2 * self.n_genes)
        
        for i, name in enumerate(self.gene_list):
            gene = self.genes[name]
            
            # Rough steady state
            mrna_ss = gene.k_tx_init / gene.delta_m * 0.5  # Conservative
            protein_ss = gene.k_tl_init * mrna_ss / gene.delta_p
            
            state[i] = mrna_ss
            state[self.n_genes + i] = protein_ss
        
        return state
    
    def ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """ODE right-hand side with resource competition."""
        
        dydt = np.zeros_like(y)
        
        # Calculate resource usage
        rnap_engaged = 0
        ribosomes_engaged = 0
        
        for i, name in enumerate(self.gene_list):
            gene = self.genes[name]
            mrna = max(y[i], 0)
            
            # RNAP engaged in transcription
            # (transcription rate * elongation time)
            rnap_engaged += gene.k_tx_init * gene.tx_elongation_time
            
            # Ribosomes engaged in translation
            # (mRNA * initiation rate * elongation time)
            ribosomes_engaged += mrna * gene.k_tl_init * gene.tl_elongation_time
        
        # Free resources
        free_RNAP = max(self.total_RNAP - rnap_engaged, 1)
        free_ribosomes = max(self.total_ribosomes - ribosomes_engaged, 1)
        
        # Resource saturation
        f_RNAP = free_RNAP / self.total_RNAP
        f_ribosome = free_ribosomes / self.total_ribosomes
        
        # Gene expression dynamics
        for i, name in enumerate(self.gene_list):
            gene = self.genes[name]
            mrna = max(y[i], 0)
            protein = max(y[self.n_genes + i], 0)
            
            # Autorepression
            repression = 1.0
            if gene.autorepression and protein > 0:
                repression = gene.K_autorepression / (gene.K_autorepression + protein)
            
            # Transcription
            tx_rate = gene.k_tx_init * f_RNAP * repression
            dydt[i] = tx_rate - gene.delta_m * mrna
            
            # Translation
            tl_rate = gene.k_tl_init * mrna * f_ribosome
            dydt[self.n_genes + i] = tl_rate - gene.delta_p * protein
        
        return dydt
    
    def simulate(self, t_span: Tuple[float, float], 
                 t_eval: Optional[np.ndarray] = None,
                 initial_state: Optional[np.ndarray] = None) -> dict:
        """Run simulation."""
        
        if initial_state is None:
            initial_state = self.get_initial_state()
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)
        
        print(f"Simulating {t_span[0]:.0f} to {t_span[1]:.0f} min ({self.n_genes} genes)...")
        
        solution = solve_ivp(
            self.ode_rhs,
            t_span,
            initial_state,
            t_eval=t_eval,
            method='LSODA',
            rtol=1e-6,
            atol=1e-9
        )
        
        return {
            't': solution.t,
            'mrna': solution.y[:self.n_genes, :],
            'protein': solution.y[self.n_genes:, :],
        }
    
    def get_expression_summary(self, result: dict) -> dict:
        """Summarize expression at final time point."""
        
        mrna = result['mrna'][:, -1]
        protein = result['protein'][:, -1]
        
        summary = {
            'total_mrna': mrna.sum(),
            'total_protein': protein.sum(),
            'by_category': {},
            'top_genes': [],
        }
        
        # By category
        for i, name in enumerate(self.gene_list):
            cat = self.genes[name].category
            if cat not in summary['by_category']:
                summary['by_category'][cat] = {'mrna': 0, 'protein': 0, 'genes': 0}
            summary['by_category'][cat]['mrna'] += mrna[i]
            summary['by_category'][cat]['protein'] += protein[i]
            summary['by_category'][cat]['genes'] += 1
        
        # Top genes
        top_idx = np.argsort(protein)[::-1][:20]
        for idx in top_idx:
            summary['top_genes'].append({
                'name': self.gene_list[idx],
                'category': self.genes[self.gene_list[idx]].category,
                'protein': protein[idx],
                'mrna': mrna[idx],
            })
        
        return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("DARK MANIFOLD V47b: ENHANCED GENE EXPRESSION")
    print("="*70)
    
    # Build genes
    genes = build_full_syn3a_genes()
    
    # Create model
    model = EnhancedGeneExpressionModel(genes)
    
    # Simulate 3 hours (180 min) - should reach steady state
    result = model.simulate(t_span=(0, 180))
    
    # Summary
    summary = model.get_expression_summary(result)
    
    print(f"\n{'='*70}")
    print("STEADY STATE SUMMARY")
    print("="*70)
    print(f"Total mRNA: {summary['total_mrna']:.0f} copies")
    print(f"Total protein: {summary['total_protein']:.0f} copies")
    
    print(f"\n{'='*70}")
    print("EXPRESSION BY CATEGORY")
    print("="*70)
    for cat, data in sorted(summary['by_category'].items(), 
                            key=lambda x: -x[1]['protein']):
        pct = data['protein'] / summary['total_protein'] * 100
        print(f"  {cat:<20} {data['protein']:>8.0f} protein ({pct:>5.1f}%), "
              f"{data['mrna']:>6.1f} mRNA, {data['genes']:>3} genes")
    
    print(f"\n{'='*70}")
    print("TOP 20 EXPRESSED PROTEINS")
    print("="*70)
    for i, g in enumerate(summary['top_genes']):
        print(f"  {i+1:>2}. {g['name']:<10} {g['protein']:>7.0f} copies  ({g['category']})")
    
    # Check biology
    print(f"\n{'='*70}")
    print("BIOLOGICAL VALIDATION")
    print("="*70)
    
    # Ribosomal protein fraction
    ribo_protein = sum(d['protein'] for cat, d in summary['by_category'].items() 
                       if 'ribosome' in cat)
    ribo_pct = ribo_protein / summary['total_protein'] * 100
    print(f"Ribosomal proteins: {ribo_pct:.1f}% (expected: 25-50% for fast growth)")
    
    # Translation machinery
    tl_protein = ribo_protein + summary['by_category'].get('translation_factor', {}).get('protein', 0)
    tl_protein += summary['by_category'].get('tRNA_synthetase', {}).get('protein', 0)
    tl_pct = tl_protein / summary['total_protein'] * 100
    print(f"Translation machinery: {tl_pct:.1f}% (expected: 40-60%)")
    
    # Chaperones
    chap_pct = summary['by_category'].get('chaperone', {}).get('protein', 0) / summary['total_protein'] * 100
    print(f"Chaperones: {chap_pct:.1f}% (expected: 2-5%)")
    
    return model, result, summary


if __name__ == '__main__':
    model, result, summary = main()
