"""
Dark Manifold V47: Gene Expression Dynamics
============================================

Transcription + Translation simulation for JCVI-syn3A minimal cell.

Model: ODE-based gene expression with resource competition

Variables per gene:
- mRNA copy number
- Protein copy number

Global resources:
- Free RNAP
- Free ribosomes
- NTP pool
- Amino acid pools
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# JCVI-SYN3A GENE DATABASE WITH EXPRESSION PARAMETERS
# ============================================================================

@dataclass
class Gene:
    """Gene with expression parameters."""
    name: str
    length_nt: int          # Gene length in nucleotides
    length_aa: int          # Protein length in amino acids
    category: str           # Functional category
    
    # Expression parameters (will be estimated)
    promoter_strength: float = 1.0   # Relative (0-10)
    rbs_strength: float = 1.0        # Relative (0-10)
    codon_adaptation: float = 1.0    # CAI (0-1)
    
    # Derived rates (calculated)
    k_tx: float = 0.0       # Transcription rate
    k_tl: float = 0.0       # Translation rate
    
    # Decay rates
    mrna_half_life: float = 3.0     # minutes
    protein_half_life: float = 60.0  # minutes
    
    @property
    def delta_m(self) -> float:
        """mRNA decay rate (1/min)."""
        return np.log(2) / self.mrna_half_life
    
    @property
    def delta_p(self) -> float:
        """Protein decay rate (1/min)."""
        return np.log(2) / self.protein_half_life


def build_syn3a_genes() -> Dict[str, Gene]:
    """
    Build JCVI-syn3A gene database with expression parameters.
    
    Based on:
    - Hutchison et al. 2016 (gene list)
    - Karr et al. 2012 (M. genitalium parameters)
    - Typical Mycoplasma expression data
    """
    
    genes = {}
    
    # ========== RIBOSOMAL PROTEINS (high expression) ==========
    # 30S subunit proteins
    ribosomal_30S = [
        ('rpsA', 557), ('rpsB', 261), ('rpsC', 233), ('rpsD', 206),
        ('rpsE', 167), ('rpsF', 131), ('rpsG', 156), ('rpsH', 138),
        ('rpsI', 130), ('rpsJ', 103), ('rpsK', 129), ('rpsL', 124),
        ('rpsM', 119), ('rpsN', 101), ('rpsO', 89), ('rpsP', 82),
        ('rpsQ', 93), ('rpsR', 75), ('rpsS', 93),
    ]
    for name, length_aa in ribosomal_30S:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='ribosome_30S',
            promoter_strength=8.0,  # High expression
            rbs_strength=8.0,
            codon_adaptation=0.9,
            mrna_half_life=5.0,     # Stable mRNA
            protein_half_life=120.0  # Very stable protein
        )
    
    # 50S subunit proteins
    ribosomal_50S = [
        ('rplA', 231), ('rplB', 277), ('rplC', 209), ('rplD', 203),
        ('rplE', 182), ('rplF', 177), ('rplJ', 166), ('rplK', 144),
        ('rplL', 124), ('rplM', 144), ('rplN', 123), ('rplO', 144),
        ('rplP', 136), ('rplQ', 130), ('rplR', 117), ('rplS', 115),
        ('rplT', 118), ('rplU', 103), ('rplV', 110), ('rplW', 104),
        ('rplX', 104), ('rpmA', 89), ('rpmB', 81), ('rpmC', 64),
        ('rpmD', 60), ('rpmE', 50), ('rpmF', 58), ('rpmG', 53),
        ('rpmH', 46), ('rpmI', 66), ('rpmJ', 38),
    ]
    for name, length_aa in ribosomal_50S:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='ribosome_50S',
            promoter_strength=8.0,
            rbs_strength=8.0,
            codon_adaptation=0.9,
            mrna_half_life=5.0,
            protein_half_life=120.0
        )
    
    # ========== tRNA SYNTHETASES (medium-high expression) ==========
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
            category='tRNA_synthetase',
            promoter_strength=5.0,
            rbs_strength=5.0,
            codon_adaptation=0.8,
            mrna_half_life=3.0,
            protein_half_life=90.0
        )
    
    # ========== TRANSLATION FACTORS (high expression) ==========
    translation_factors = [
        ('infA', 73), ('infB', 741), ('infC', 181),
        ('fusA', 697), ('tsf', 283), ('tuf', 394),
        ('prfA', 360), ('prfB', 365), ('frr', 185),
        ('efp', 186),
    ]
    for name, length_aa in translation_factors:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='translation_factor',
            promoter_strength=7.0,
            rbs_strength=7.0,
            codon_adaptation=0.85,
            mrna_half_life=4.0,
            protein_half_life=90.0
        )
    
    # ========== RNA POLYMERASE (medium expression) ==========
    rnap_genes = [
        ('rpoA', 329), ('rpoB', 1342), ('rpoC', 1524), ('rpoD', 437),
        ('rpoE', 192),
    ]
    for name, length_aa in rnap_genes:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='RNAP',
            promoter_strength=4.0,
            rbs_strength=4.0,
            codon_adaptation=0.75,
            mrna_half_life=3.0,
            protein_half_life=120.0
        )
    
    # ========== DNA REPLICATION (low-medium expression) ==========
    replication_genes = [
        ('dnaA', 454), ('dnaB', 471), ('dnaC', 259), ('dnaE', 1160),
        ('dnaG', 342), ('dnaN', 378), ('dnaX', 455),
        ('gyrA', 820), ('gyrB', 647),
        ('ssb', 166), ('ligA', 671),
        ('polA', 605),
    ]
    for name, length_aa in replication_genes:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='replication',
            promoter_strength=2.0,  # Low expression
            rbs_strength=3.0,
            codon_adaptation=0.7,
            mrna_half_life=2.0,
            protein_half_life=60.0
        )
    
    # ========== CELL DIVISION (low expression, cell-cycle regulated) ==========
    division_genes = [
        ('ftsZ', 395), ('ftsA', 420), ('ftsH', 640),
    ]
    for name, length_aa in division_genes:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='division',
            promoter_strength=2.5,
            rbs_strength=3.0,
            codon_adaptation=0.7,
            mrna_half_life=2.0,
            protein_half_life=45.0
        )
    
    # ========== MEMBRANE/TRANSPORT (variable expression) ==========
    membrane_genes = [
        ('secA', 820), ('secY', 435), ('secE', 68),
        ('ffh', 453), ('ftsY', 497),
        ('yidC', 270), ('lepB', 324),
    ]
    for name, length_aa in membrane_genes:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='membrane',
            promoter_strength=3.0,
            rbs_strength=3.5,
            codon_adaptation=0.7,
            mrna_half_life=2.5,
            protein_half_life=60.0
        )
    
    # ========== METABOLISM (medium expression) ==========
    metabolism_genes = [
        # Glycolysis
        ('pgi', 445), ('pfkA', 320), ('fbaA', 359), ('tpiA', 255),
        ('gapA', 337), ('pgk', 400), ('gpmA', 250), ('eno', 432),
        ('pykF', 470),
        # Nucleotide synthesis
        ('ndk', 143), ('adk', 214), ('cmk', 227), ('gmk', 220), ('tmk', 212),
        ('pyrG', 545), ('pyrH', 241),
    ]
    for name, length_aa in metabolism_genes:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='metabolism',
            promoter_strength=4.0,
            rbs_strength=4.0,
            codon_adaptation=0.75,
            mrna_half_life=2.5,
            protein_half_life=60.0
        )
    
    # ========== CHAPERONES (high expression) ==========
    chaperone_genes = [
        ('groEL', 548), ('groES', 97),
        ('dnaK', 638), ('dnaJ', 376), ('grpE', 197),
    ]
    for name, length_aa in chaperone_genes:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='chaperone',
            promoter_strength=6.0,
            rbs_strength=6.0,
            codon_adaptation=0.85,
            mrna_half_life=4.0,
            protein_half_life=90.0
        )
    
    # ========== LIPID SYNTHESIS (medium expression) ==========
    lipid_genes = [
        ('accA', 319), ('accB', 156), ('accC', 449), ('accD', 304),
        ('fabD', 309), ('fabG', 244), ('fabH', 317), ('fabI', 262),
        ('fabZ', 151), ('acpP', 78), ('acpS', 126),
        ('plsB', 807), ('plsC', 245),
    ]
    for name, length_aa in lipid_genes:
        genes[name] = Gene(
            name=name, length_nt=length_aa*3, length_aa=length_aa,
            category='lipid',
            promoter_strength=3.5,
            rbs_strength=3.5,
            codon_adaptation=0.7,
            mrna_half_life=2.5,
            protein_half_life=60.0
        )
    
    return genes


# ============================================================================
# GENE EXPRESSION MODEL
# ============================================================================

class GeneExpressionModel:
    """
    ODE model for gene expression dynamics.
    
    State vector: [mRNA_1, mRNA_2, ..., mRNA_n, Protein_1, ..., Protein_n, 
                   free_RNAP, free_ribosomes]
    """
    
    def __init__(self, genes: Dict[str, Gene]):
        self.genes = genes
        self.gene_list = list(genes.keys())
        self.n_genes = len(self.gene_list)
        
        # Resource parameters
        self.total_RNAP = 300          # Total RNAP molecules
        self.total_ribosomes = 1500    # Total ribosomes
        
        # Global rate constants
        self.k_tx_max = 0.5            # Max transcription initiation (1/min)
        self.k_tl_max = 2.0            # Max translation initiation (1/min)
        
        # Resource binding constants (Michaelis-Menten)
        self.K_RNAP = 50               # RNAP binding constant
        self.K_ribosome = 100          # Ribosome binding constant
        
        # Elongation speeds
        self.v_tx = 40 * 60            # Transcription: 40 nt/s = 2400 nt/min
        self.v_tl = 15 * 60            # Translation: 15 aa/s = 900 aa/min
        
        # Calculate rates for each gene
        self._calculate_rates()
        
        # State indices
        self.mrna_start = 0
        self.protein_start = self.n_genes
        self.rnap_idx = 2 * self.n_genes
        self.ribosome_idx = 2 * self.n_genes + 1
        
        print(f"Gene Expression Model initialized:")
        print(f"  Genes: {self.n_genes}")
        print(f"  RNAP: {self.total_RNAP}")
        print(f"  Ribosomes: {self.total_ribosomes}")
    
    def _calculate_rates(self):
        """Calculate transcription and translation rates for each gene."""
        for gene in self.genes.values():
            # Transcription rate depends on promoter strength
            gene.k_tx = self.k_tx_max * (gene.promoter_strength / 10.0)
            
            # Translation rate depends on RBS strength and codon adaptation
            gene.k_tl = self.k_tl_max * (gene.rbs_strength / 10.0) * gene.codon_adaptation
    
    def get_initial_state(self) -> np.ndarray:
        """
        Initialize state vector with steady-state estimates.
        
        At steady state:
        mRNA = k_tx / delta_m
        Protein = k_tl * mRNA / delta_p
        """
        state = np.zeros(2 * self.n_genes + 2)
        
        for i, name in enumerate(self.gene_list):
            gene = self.genes[name]
            
            # Steady-state mRNA (copy number per cell)
            mrna_ss = gene.k_tx / gene.delta_m
            state[self.mrna_start + i] = mrna_ss
            
            # Steady-state protein
            protein_ss = gene.k_tl * mrna_ss / gene.delta_p
            state[self.protein_start + i] = protein_ss
        
        # Initial free resources (estimate)
        state[self.rnap_idx] = self.total_RNAP * 0.3  # 30% free
        state[self.ribosome_idx] = self.total_ribosomes * 0.2  # 20% free
        
        return state
    
    def ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of ODE system.
        
        d[mRNA_i]/dt = k_tx_i * f(RNAP) - δm_i * [mRNA_i]
        d[Protein_i]/dt = k_tl_i * [mRNA_i] * f(ribosome) - δp_i * [Protein_i]
        
        Where f(resource) is a saturation function.
        """
        dydt = np.zeros_like(y)
        
        # Get free resources
        free_RNAP = max(y[self.rnap_idx], 0)
        free_ribosomes = max(y[self.ribosome_idx], 0)
        
        # Resource availability (Michaelis-Menten-like)
        f_RNAP = free_RNAP / (self.K_RNAP + free_RNAP) if free_RNAP > 0 else 0
        f_ribosome = free_ribosomes / (self.K_ribosome + free_ribosomes) if free_ribosomes > 0 else 0
        
        # Track resource usage
        RNAP_used = 0
        ribosomes_used = 0
        
        for i, name in enumerate(self.gene_list):
            gene = self.genes[name]
            
            mrna = max(y[self.mrna_start + i], 0)
            protein = max(y[self.protein_start + i], 0)
            
            # Transcription: production - decay
            tx_rate = gene.k_tx * f_RNAP
            dydt[self.mrna_start + i] = tx_rate - gene.delta_m * mrna
            
            # Translation: production - decay
            tl_rate = gene.k_tl * mrna * f_ribosome
            dydt[self.protein_start + i] = tl_rate - gene.delta_p * protein
            
            # Resource usage (proportional to active genes)
            RNAP_used += tx_rate / self.k_tx_max
            ribosomes_used += tl_rate / self.k_tl_max
        
        # Resource dynamics (simplified: fast equilibration)
        # Free = Total - Used
        target_free_RNAP = max(self.total_RNAP - RNAP_used, 0)
        target_free_ribo = max(self.total_ribosomes - ribosomes_used, 0)
        
        # Fast relaxation to equilibrium
        k_relax = 10.0
        dydt[self.rnap_idx] = k_relax * (target_free_RNAP - free_RNAP)
        dydt[self.ribosome_idx] = k_relax * (target_free_ribo - free_ribosomes)
        
        return dydt
    
    def simulate(self, t_span: Tuple[float, float], 
                 t_eval: Optional[np.ndarray] = None,
                 initial_state: Optional[np.ndarray] = None) -> dict:
        """Run simulation."""
        
        if initial_state is None:
            initial_state = self.get_initial_state()
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)
        
        print(f"Simulating from t={t_span[0]} to t={t_span[1]} min...")
        
        solution = solve_ivp(
            self.ode_rhs, 
            t_span, 
            initial_state,
            t_eval=t_eval,
            method='LSODA',
            rtol=1e-6,
            atol=1e-9
        )
        
        if not solution.success:
            print(f"Warning: Integration failed - {solution.message}")
        
        return {
            't': solution.t,
            'y': solution.y,
            'mrna': solution.y[self.mrna_start:self.protein_start, :],
            'protein': solution.y[self.protein_start:self.rnap_idx, :],
            'free_RNAP': solution.y[self.rnap_idx, :],
            'free_ribosomes': solution.y[self.ribosome_idx, :],
        }
    
    def perturb_gene(self, state: np.ndarray, gene_name: str, 
                     mrna_factor: float = 1.0, protein_factor: float = 1.0) -> np.ndarray:
        """Apply perturbation to a gene's expression."""
        new_state = state.copy()
        
        if gene_name in self.gene_list:
            i = self.gene_list.index(gene_name)
            new_state[self.mrna_start + i] *= mrna_factor
            new_state[self.protein_start + i] *= protein_factor
        
        return new_state
    
    def knockout_gene(self, gene_name: str):
        """Knockout a gene by setting its rates to zero."""
        if gene_name in self.genes:
            self.genes[gene_name].k_tx = 0
            self.genes[gene_name].k_tl = 0
            print(f"Knocked out gene: {gene_name}")


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_steady_state(model: GeneExpressionModel, result: dict):
    """Analyze steady-state expression levels."""
    
    # Get final values
    mrna_final = result['mrna'][:, -1]
    protein_final = result['protein'][:, -1]
    
    print("\n" + "="*70)
    print("STEADY STATE ANALYSIS")
    print("="*70)
    
    print(f"\nTotal mRNA copies: {mrna_final.sum():.0f}")
    print(f"Total protein copies: {protein_final.sum():.0f}")
    print(f"Free RNAP: {result['free_RNAP'][-1]:.0f}")
    print(f"Free ribosomes: {result['free_ribosomes'][-1]:.0f}")
    
    # Top expressed genes
    print("\n" + "-"*50)
    print("TOP 10 EXPRESSED GENES (by protein copy number)")
    print("-"*50)
    
    protein_rank = np.argsort(protein_final)[::-1]
    for i, idx in enumerate(protein_rank[:10]):
        name = model.gene_list[idx]
        gene = model.genes[name]
        print(f"{i+1:2}. {name:<10} {protein_final[idx]:>8.0f} copies  ({gene.category})")
    
    # By category
    print("\n" + "-"*50)
    print("EXPRESSION BY CATEGORY")
    print("-"*50)
    
    category_protein = {}
    for i, name in enumerate(model.gene_list):
        cat = model.genes[name].category
        if cat not in category_protein:
            category_protein[cat] = 0
        category_protein[cat] += protein_final[i]
    
    for cat, total in sorted(category_protein.items(), key=lambda x: -x[1]):
        pct = total / protein_final.sum() * 100
        print(f"  {cat:<20} {total:>10.0f} ({pct:>5.1f}%)")


def plot_expression(model: GeneExpressionModel, result: dict, 
                    genes_to_plot: List[str], filename: str):
    """Plot expression dynamics for selected genes."""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    t = result['t']
    
    # mRNA
    ax = axes[0]
    for name in genes_to_plot:
        if name in model.gene_list:
            i = model.gene_list.index(name)
            ax.plot(t, result['mrna'][i, :], label=name)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('mRNA copies')
    ax.set_title('mRNA Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Protein
    ax = axes[1]
    for name in genes_to_plot:
        if name in model.gene_list:
            i = model.gene_list.index(name)
            ax.plot(t, result['protein'][i, :], label=name)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Protein copies')
    ax.set_title('Protein Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved plot: {filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("DARK MANIFOLD V47: GENE EXPRESSION DYNAMICS")
    print("JCVI-syn3A Minimal Cell")
    print("="*70)
    
    # Build gene database
    genes = build_syn3a_genes()
    print(f"\nLoaded {len(genes)} genes")
    
    # Create model
    model = GeneExpressionModel(genes)
    
    # Run simulation (2 hours = 120 minutes)
    result = model.simulate(t_span=(0, 120))
    
    # Analyze
    analyze_steady_state(model, result)
    
    # Plot some key genes
    key_genes = ['rpsA', 'rplB', 'tuf', 'groEL', 'dnaK', 'rpoB', 'ftsZ', 'dnaA']
    plot_expression(model, result, key_genes, 'expression_dynamics.png')
    
    # Plot resources
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result['t'], result['free_RNAP'], label='Free RNAP')
    ax.plot(result['t'], result['free_ribosomes'], label='Free Ribosomes')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Copy number')
    ax.set_title('Free Resources Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('resources.png', dpi=150)
    print("Saved plot: resources.png")
    
    return model, result


if __name__ == '__main__':
    model, result = main()
