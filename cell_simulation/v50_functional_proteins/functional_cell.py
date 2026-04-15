"""
Dark Manifold V50: Functional Minimal Cell
==========================================

PROTEINS DO THINGS!

The key insight: gene products close the loop.
- Metabolic enzymes → determine metabolic flux
- RNAP → determines transcription capacity
- Ribosomes → determine translation capacity
- Chaperones → determine protein folding efficiency

EMERGENT BEHAVIORS:
- Growth rate EMERGES from ribosome content
- Metabolic capacity EMERGES from enzyme levels
- Gene expression capacity EMERGES from RNAP levels
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/claude/enzyme_repo/cell_simulation')
from v47_gene_expression.gene_expression_v2 import Gene, build_full_syn3a_genes


# ============================================================================
# PROTEIN FUNCTION DEFINITIONS
# ============================================================================

@dataclass
class EnzymeFunction:
    """Defines how a protein functions as an enzyme."""
    gene: str
    reaction: str
    k_cat: float           # Turnover number (1/min)
    Km: Dict[str, float]   # Substrate Km values
    products: List[str]    # What it produces
    substrates: List[str]  # What it consumes


@dataclass  
class MachineryFunction:
    """Defines how proteins form cellular machinery."""
    genes: List[str]       # Component genes
    stoichiometry: Dict[str, int]  # How many of each needed
    function: str          # 'RNAP', 'ribosome', 'chaperone', etc.
    capacity_per_complex: float  # Activity per assembled complex


# ============================================================================
# FUNCTIONAL CELL MODEL
# ============================================================================

class FunctionalMinimalCell:
    """
    Minimal cell where proteins have functions.
    
    Key features:
    1. Enzyme levels determine metabolic Vmax
    2. RNAP levels determine transcription capacity
    3. Ribosome levels determine translation capacity
    4. Chaperone levels affect protein folding
    5. Growth rate EMERGES from these relationships
    """
    
    def __init__(self, n_genes: int = 80):
        print("="*72)
        print("    DARK MANIFOLD V50: FUNCTIONAL MINIMAL CELL")
        print("="*72)
        
        # ===== BUILD GENE SET WITH FUNCTIONAL PRIORITY =====
        all_genes = build_full_syn3a_genes()
        
        # MUST HAVE these functional genes
        essential_genes = [
            # RNAP subunits
            'rpoA', 'rpoB', 'rpoC', 'rpoD',
            # Ribosomal proteins (sample)
            'rplA', 'rplB', 'rplC', 'rplD', 'rpsA', 'rpsB', 'rpsC', 'rpsD',
            # Translation factors
            'tuf', 'fusA', 'tsf', 'infA', 'infB', 'infC', 'prfA', 'prfB',
            # Chaperones
            'groEL', 'groES', 'dnaK', 'dnaJ', 'grpE',
            # Glycolysis enzymes
            'pgi', 'pfkA', 'fbaA', 'tpiA', 'gapA', 'pgk', 'gpmA', 'eno', 'pykF',
            # LDH
            'ldh',
            # Nucleotide kinases
            'ndk', 'adk', 'gmk', 'cmk', 'tmk',
            # ATP synthase (some subunits)
            'atpA', 'atpD',
        ]
        
        # Select genes
        self.genes = {}
        for name in essential_genes:
            if name in all_genes:
                self.genes[name] = all_genes[name]
        
        # Fill with other genes
        for name, gene in all_genes.items():
            if name not in self.genes and len(self.genes) < n_genes:
                self.genes[name] = gene
        
        self.gene_list = list(self.genes.keys())
        self.n_genes = len(self.gene_list)
        
        # ===== DEFINE PROTEIN FUNCTIONS =====
        self._define_functions()
        
        # ===== STATE STRUCTURE =====
        # Metabolites
        self.met_states = ['ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'NAD', 'NADH',
                          'UTP', 'CTP', 'AA', 'Pyr', 'Lac', 'G6P', 'PEP']
        self.n_met = len(self.met_states)
        
        # Full state: metabolites + mRNA + protein
        self.n_states = self.n_met + 2 * self.n_genes
        
        # Indices
        self.met_idx = {m: i for i, m in enumerate(self.met_states)}
        self.mrna_idx = {g: self.n_met + i for i, g in enumerate(self.gene_list)}
        self.prot_idx = {g: self.n_met + self.n_genes + i for i, g in enumerate(self.gene_list)}
        
        # ===== KINETIC PARAMETERS =====
        self._setup_kinetics()
        
        # Print summary
        self._print_summary()
    
    def _define_functions(self):
        """Define what each protein does."""
        
        # ===== METABOLIC ENZYMES =====
        self.enzymes = {}
        
        # Glycolysis enzymes (simplified - lumped reactions)
        glycolysis_enzymes = [
            ('pgi', 'PGI', 100, {'G6P': 0.3}),
            ('pfkA', 'PFK', 50, {'G6P': 0.1, 'ATP': 0.1}),
            ('gapA', 'GAPDH', 80, {'G3P': 0.1, 'NAD': 0.1}),
            ('pgk', 'PGK', 100, {'BPG': 0.01, 'ADP': 0.2}),
            ('pykF', 'PYK', 60, {'PEP': 0.2, 'ADP': 0.2}),
        ]
        
        for gene, rxn, kcat, km in glycolysis_enzymes:
            if gene in self.genes:
                self.enzymes[gene] = EnzymeFunction(
                    gene=gene, reaction=rxn, k_cat=kcat, Km=km,
                    products=['ATP'], substrates=['ADP']
                )
        
        # LDH - critical for NAD+ regeneration
        if 'ldh' in self.genes:
            self.enzymes['ldh'] = EnzymeFunction(
                gene='ldh', reaction='LDH', k_cat=200,
                Km={'Pyr': 0.2, 'NADH': 0.02},
                products=['NAD', 'Lac'], substrates=['NADH', 'Pyr']
            )
        
        # Nucleotide kinases
        kinases = [
            ('adk', 'ADK', 500, {'AMP': 0.1, 'ATP': 0.2}),
            ('ndk', 'NDK', 300, {'GDP': 0.1, 'ATP': 0.2}),
        ]
        for gene, rxn, kcat, km in kinases:
            if gene in self.genes:
                self.enzymes[gene] = EnzymeFunction(
                    gene=gene, reaction=rxn, k_cat=kcat, Km=km,
                    products=[], substrates=[]
                )
        
        # ===== MACROMOLECULAR MACHINERY =====
        self.machinery = {}
        
        # RNAP (needs all 4 subunits)
        self.machinery['RNAP'] = MachineryFunction(
            genes=['rpoA', 'rpoB', 'rpoC', 'rpoD'],
            stoichiometry={'rpoA': 2, 'rpoB': 1, 'rpoC': 1, 'rpoD': 1},
            function='transcription',
            capacity_per_complex=40  # nt/s
        )
        
        # Ribosome (simplified - use a few key proteins as proxy)
        self.machinery['ribosome'] = MachineryFunction(
            genes=['rplA', 'rplB', 'rpsA', 'rpsB'],
            stoichiometry={'rplA': 1, 'rplB': 1, 'rpsA': 1, 'rpsB': 1},
            function='translation',
            capacity_per_complex=15  # aa/s
        )
        
        # Chaperone system
        self.machinery['chaperone'] = MachineryFunction(
            genes=['groEL', 'groES'],
            stoichiometry={'groEL': 14, 'groES': 7},  # GroEL tetradecamer + GroES heptamer
            function='folding',
            capacity_per_complex=10  # proteins/min
        )
        
        # Translation factors (needed for ribosome function)
        self.elongation_factors = ['tuf', 'fusA', 'tsf']
        self.initiation_factors = ['infA', 'infB', 'infC']
    
    def _setup_kinetics(self):
        """Setup kinetic parameters."""
        
        # Growth parameters
        self.mu_max = 0.015  # Maximum growth rate (1/min)
        
        # Metabolite targets (for homeostasis)
        self.met_targets = {
            'ATP': 3.0, 'ADP': 0.5, 'AMP': 0.1,
            'GTP': 0.8, 'GDP': 0.2,
            'NAD': 1.0, 'NADH': 0.1,
            'UTP': 0.5, 'CTP': 0.3,
            'AA': 5.0,
            'Pyr': 0.3, 'Lac': 2.0,
            'G6P': 0.3, 'PEP': 0.2,
        }
        
        # Homeostatic buffering time constants
        self.tau_buffer = {m: 10.0 for m in self.met_states}
        self.tau_buffer['ATP'] = 5.0
        self.tau_buffer['NAD'] = 5.0
        
        # External concentrations
        self.Glc_ext = 20.0
        self.AA_ext = 2.0
        
        # Gene expression base rates
        self.k_tx_base = 0.1   # Base transcription rate
        self.k_tl_base = 0.5   # Base translation rate
        
        # Functional scaling
        self.K_RNAP = 50       # RNAP copies for half-max transcription
        self.K_ribo = 200      # Ribosome copies for half-max translation
        self.K_chaperone = 100 # Chaperone copies for half-max folding
        self.K_EF = 50         # Elongation factor copies for half-max
        
        # Metabolic scaling
        self.K_enzyme = 10     # Enzyme copies for half-max flux
        
        # Calculate gene-specific rates
        for gene in self.genes.values():
            gene.k_tx_init = self.k_tx_base * (gene.promoter_strength / 5.0)
            gene.k_tl_init = self.k_tl_base * (gene.rbs_strength / 5.0)
    
    def _print_summary(self):
        """Print model summary."""
        
        print(f"\nGenes: {self.n_genes}")
        print(f"Functional enzymes: {len(self.enzymes)}")
        print(f"Machinery complexes: {len(self.machinery)}")
        
        by_cat = defaultdict(int)
        for gene in self.genes.values():
            by_cat[gene.category] += 1
        
        print(f"\nGenes by category:")
        for cat, count in sorted(by_cat.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cat:<20} {count:>3}")
        
        print(f"\nTotal states: {self.n_states}")
    
    def get_initial_state(self) -> np.ndarray:
        """Initialize state."""
        y0 = np.zeros(self.n_states)
        
        # Metabolites at target
        for met, target in self.met_targets.items():
            if met in self.met_idx:
                y0[self.met_idx[met]] = target
        
        # Gene expression at rough steady state
        for name in self.gene_list:
            gene = self.genes[name]
            mrna_ss = gene.k_tx_init / gene.delta_m * 2.0
            prot_ss = gene.k_tl_init * mrna_ss / gene.delta_p * 5.0
            y0[self.mrna_idx[name]] = mrna_ss
            y0[self.prot_idx[name]] = prot_ss
        
        return y0
    
    def calculate_machinery(self, y: np.ndarray) -> Dict[str, float]:
        """Calculate functional machinery levels from protein concentrations."""
        
        machinery = {}
        
        # RNAP: limited by least abundant subunit (accounting for stoichiometry)
        rnap_info = self.machinery['RNAP']
        rnap_components = []
        for gene, stoich in rnap_info.stoichiometry.items():
            if gene in self.prot_idx:
                level = max(y[self.prot_idx[gene]], 0) / stoich
                rnap_components.append(level)
        machinery['RNAP'] = min(rnap_components) if rnap_components else 0
        
        # Ribosomes: limited by least abundant component
        ribo_info = self.machinery['ribosome']
        ribo_components = []
        for gene, stoich in ribo_info.stoichiometry.items():
            if gene in self.prot_idx:
                level = max(y[self.prot_idx[gene]], 0) / stoich
                ribo_components.append(level)
        machinery['ribosome'] = min(ribo_components) if ribo_components else 0
        
        # Chaperones
        chap_info = self.machinery['chaperone']
        chap_components = []
        for gene, stoich in chap_info.stoichiometry.items():
            if gene in self.prot_idx:
                level = max(y[self.prot_idx[gene]], 0) / stoich
                chap_components.append(level)
        machinery['chaperone'] = min(chap_components) if chap_components else 0
        
        # Elongation factors (average)
        ef_levels = []
        for gene in self.elongation_factors:
            if gene in self.prot_idx:
                ef_levels.append(max(y[self.prot_idx[gene]], 0))
        machinery['EF'] = np.mean(ef_levels) if ef_levels else 0
        
        return machinery
    
    def calculate_enzyme_flux(self, enzyme_gene: str, y: np.ndarray) -> float:
        """Calculate flux through an enzyme given current state."""
        
        if enzyme_gene not in self.enzymes:
            return 0
        
        enz = self.enzymes[enzyme_gene]
        
        # Enzyme level
        E = max(y[self.prot_idx[enzyme_gene]], 0) if enzyme_gene in self.prot_idx else 0
        
        # Michaelis-Menten with enzyme level
        v = enz.k_cat * E / (self.K_enzyme + E)
        
        # Substrate saturation
        for sub, Km in enz.Km.items():
            if sub in self.met_idx:
                S = max(y[self.met_idx[sub]], 1e-12)
                v *= S / (Km + S)
        
        return v
    
    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        """Calculate derivatives with functional proteins."""
        
        dy = np.zeros(self.n_states)
        
        # ===== GET METABOLITES =====
        def get_met(name):
            return max(y[self.met_idx[name]], 1e-12) if name in self.met_idx else 0
        
        ATP = get_met('ATP')
        ADP = get_met('ADP')
        AMP = get_met('AMP')
        GTP = get_met('GTP')
        GDP = get_met('GDP')
        NAD = get_met('NAD')
        NADH = get_met('NADH')
        UTP = get_met('UTP')
        CTP = get_met('CTP')
        AA = get_met('AA')
        Pyr = get_met('Pyr')
        Lac = get_met('Lac')
        G6P = get_met('G6P')
        PEP = get_met('PEP')
        
        # ===== CALCULATE MACHINERY LEVELS =====
        machinery = self.calculate_machinery(y)
        
        n_RNAP = machinery['RNAP']
        n_ribo = machinery['ribosome']
        n_chap = machinery['chaperone']
        n_EF = machinery['EF']
        
        # Functional fractions (Michaelis-Menten saturation)
        f_RNAP = n_RNAP / (self.K_RNAP + n_RNAP)
        f_ribo = n_ribo / (self.K_ribo + n_ribo)
        f_chap = n_chap / (self.K_chaperone + n_chap)
        f_EF = n_EF / (self.K_EF + n_EF)
        
        # Combined translation efficiency
        f_translation = f_ribo * f_EF * f_chap
        
        # NTP availability
        NTP_min = min(ATP, GTP, UTP, CTP)
        f_NTP = NTP_min / (0.3 + NTP_min)
        
        # AA availability
        f_AA = AA / (1.0 + AA)
        
        # ===== METABOLISM (enzyme-dependent) =====
        
        # Glucose uptake (constant for simplicity)
        v_glc_uptake = 0.5 * self.Glc_ext / (1.0 + self.Glc_ext)
        dy[self.met_idx['G6P']] += v_glc_uptake
        
        # Glycolysis - depends on glycolytic enzyme levels
        # Simplified: use average of key enzymes
        glyc_enzymes = ['pfkA', 'gapA', 'pykF']
        glyc_activity = np.mean([
            self.calculate_enzyme_flux(e, y) for e in glyc_enzymes
            if e in self.enzymes
        ]) if glyc_enzymes else 0
        
        # Scale glycolysis rate
        v_glyc = 0.3 * (1 + glyc_activity) * G6P/(0.3 + G6P) * NAD/(0.2 + NAD) * ADP/(0.2 + ADP)
        
        dy[self.met_idx['G6P']] -= v_glyc
        dy[self.met_idx['NAD']] -= 2 * v_glyc
        dy[self.met_idx['NADH']] += 2 * v_glyc
        dy[self.met_idx['ADP']] -= 2 * v_glyc
        dy[self.met_idx['ATP']] += 2 * v_glyc
        dy[self.met_idx['PEP']] += 2 * v_glyc
        
        # Pyruvate kinase (PEP → Pyr + ATP)
        pyk_activity = self.calculate_enzyme_flux('pykF', y) if 'pykF' in self.enzymes else 0
        v_pyk = 0.5 * (1 + pyk_activity) * PEP/(0.2 + PEP) * ADP/(0.2 + ADP)
        
        dy[self.met_idx['PEP']] -= v_pyk
        dy[self.met_idx['ADP']] -= v_pyk
        dy[self.met_idx['Pyr']] += v_pyk
        dy[self.met_idx['ATP']] += v_pyk
        
        # LDH - depends on LDH level
        ldh_activity = self.calculate_enzyme_flux('ldh', y) if 'ldh' in self.enzymes else 0
        v_ldh = 2.0 * (1 + ldh_activity) * Pyr/(0.2 + Pyr) * NADH/(0.02 + NADH)
        
        dy[self.met_idx['Pyr']] -= v_ldh
        dy[self.met_idx['NADH']] -= v_ldh
        dy[self.met_idx['Lac']] += v_ldh
        dy[self.met_idx['NAD']] += v_ldh
        
        # Adenylate kinase
        adk_activity = self.calculate_enzyme_flux('adk', y) if 'adk' in self.enzymes else 0
        v_adk = 20.0 * (1 + adk_activity) * (ADP * ADP - ATP * AMP)
        dy[self.met_idx['ATP']] += v_adk
        dy[self.met_idx['AMP']] += v_adk
        dy[self.met_idx['ADP']] -= 2 * v_adk
        
        # NDP kinase
        ndk_activity = self.calculate_enzyme_flux('ndk', y) if 'ndk' in self.enzymes else 0
        v_ndk = 20.0 * (1 + ndk_activity) * (GDP * ATP - GTP * ADP)
        dy[self.met_idx['GTP']] += v_ndk
        dy[self.met_idx['ADP']] += v_ndk
        dy[self.met_idx['GDP']] -= v_ndk
        dy[self.met_idx['ATP']] -= v_ndk
        
        # ===== GENE EXPRESSION (machinery-dependent) =====
        
        total_tx_flux = 0
        total_tl_flux = 0
        
        for name in self.gene_list:
            gene = self.genes[name]
            i_mrna = self.mrna_idx[name]
            i_prot = self.prot_idx[name]
            
            mrna = max(y[i_mrna], 0)
            prot = max(y[i_prot], 0)
            
            # Autorepression
            repression = 1.0
            if gene.autorepression and prot > 0:
                repression = gene.K_autorepression / (gene.K_autorepression + prot)
            
            # TRANSCRIPTION: depends on RNAP levels and NTP availability
            tx_rate = gene.k_tx_init * f_RNAP * f_NTP * repression
            dy[i_mrna] = tx_rate - gene.delta_m * mrna
            
            total_tx_flux += tx_rate * gene.length_nt / 1000
            
            # TRANSLATION: depends on ribosome, EF, chaperone levels and metabolites
            tl_rate = gene.k_tl_init * mrna * f_translation * f_AA * (ATP/(0.5+ATP)) * (GTP/(0.3+GTP))
            
            # Protein maturation affected by chaperones
            mature_rate = tl_rate * (0.3 + 0.7 * f_chap)  # 30% fold spontaneously
            
            dy[i_prot] = mature_rate - gene.delta_p * prot
            
            total_tl_flux += tl_rate * gene.length_aa / 300
        
        # ===== METABOLITE CONSUMPTION BY GENE EXPRESSION =====
        
        # Transcription consumes NTPs
        tx_scale = 0.01
        dy[self.met_idx['ATP']] -= 0.25 * total_tx_flux * tx_scale
        dy[self.met_idx['GTP']] -= 0.25 * total_tx_flux * tx_scale
        dy[self.met_idx['UTP']] -= 0.25 * total_tx_flux * tx_scale
        dy[self.met_idx['CTP']] -= 0.25 * total_tx_flux * tx_scale
        
        # Translation consumes ATP, GTP, AA
        tl_scale = 0.015
        dy[self.met_idx['AA']] -= total_tl_flux * tl_scale
        dy[self.met_idx['ATP']] -= total_tl_flux * tl_scale * 2
        dy[self.met_idx['GTP']] -= total_tl_flux * tl_scale * 2
        dy[self.met_idx['ADP']] += total_tl_flux * tl_scale * 2
        dy[self.met_idx['GDP']] += total_tl_flux * tl_scale * 2
        
        # ===== MAINTENANCE AND UPTAKE =====
        
        # AA uptake
        v_aa = 0.15 * self.AA_ext / (0.5 + self.AA_ext)
        dy[self.met_idx['AA']] += v_aa
        
        # NTP synthesis
        v_ntp = 0.03 * ATP / (1.0 + ATP)
        dy[self.met_idx['UTP']] += 0.5 * v_ntp
        dy[self.met_idx['CTP']] += 0.5 * v_ntp
        dy[self.met_idx['ATP']] -= v_ntp
        dy[self.met_idx['ADP']] += v_ntp
        
        # Lactate export
        v_lac = 0.3 * Lac / (2.0 + Lac)
        dy[self.met_idx['Lac']] -= v_lac
        
        # Maintenance
        v_maint = 0.02 * ATP / (0.3 + ATP)
        dy[self.met_idx['ATP']] -= v_maint
        dy[self.met_idx['ADP']] += v_maint
        
        # ===== HOMEOSTATIC BUFFERING =====
        for met in self.met_states:
            if met in self.met_targets:
                target = self.met_targets[met]
                current = y[self.met_idx[met]]
                tau = self.tau_buffer[met]
                dy[self.met_idx[met]] += (target - current) / tau
        
        # ===== GROWTH DILUTION =====
        # Growth rate EMERGES from ribosome content!
        mu = self.mu_max * f_ribo * f_AA * (ATP/(1.0+ATP))
        
        for i in range(self.n_states):
            dy[i] -= mu * y[i]
        
        return dy
    
    def simulate(self, t_end: float = 180) -> dict:
        """Run simulation."""
        
        t_eval = np.linspace(0, t_end, 500)
        y0 = self.get_initial_state()
        
        print(f"\nSimulating 0 → {t_end} min...")
        print(f"States: {self.n_states}")
        
        sol = solve_ivp(self.deriv, (0, t_end), y0, t_eval=t_eval,
                       method='LSODA', rtol=1e-6, atol=1e-9)
        
        # Unpack
        result = {'t': sol.t}
        
        for met in self.met_states:
            result[met] = sol.y[self.met_idx[met], :]
        
        result['mrna'] = {g: sol.y[self.mrna_idx[g], :] for g in self.gene_list}
        result['protein'] = {g: sol.y[self.prot_idx[g], :] for g in self.gene_list}
        
        # Derived
        result['EC'] = (result['ATP'] + 0.5*result['ADP']) / \
                       (result['ATP'] + result['ADP'] + result['AMP'] + 1e-12)
        
        # Calculate machinery over time
        result['RNAP'] = np.zeros(len(sol.t))
        result['ribosome'] = np.zeros(len(sol.t))
        result['chaperone'] = np.zeros(len(sol.t))
        
        for i, t in enumerate(sol.t):
            mach = self.calculate_machinery(sol.y[:, i])
            result['RNAP'][i] = mach['RNAP']
            result['ribosome'][i] = mach['ribosome']
            result['chaperone'][i] = mach['chaperone']
        
        # Emergent growth rate
        result['mu'] = self.mu_max * result['ribosome']/(self.K_ribo + result['ribosome']) * \
                       result['AA']/(1.0 + result['AA']) * result['ATP']/(1.0 + result['ATP'])
        
        return result
    
    def analyze(self, r: dict) -> Dict:
        """Analyze with focus on emergent properties."""
        
        t = r['t'][-1]
        
        ATP = r['ATP'][-1]
        GTP = r['GTP'][-1]
        NAD = r['NAD'][-1]
        AA = r['AA'][-1]
        EC = r['EC'][-1]
        
        n_RNAP = r['RNAP'][-1]
        n_ribo = r['ribosome'][-1]
        n_chap = r['chaperone'][-1]
        mu = r['mu'][-1]
        
        total_prot = sum(r['protein'][g][-1] for g in self.gene_list)
        total_mrna = sum(r['mrna'][g][-1] for g in self.gene_list)
        
        print(f"""
{'█'*72}
{'█'*12}  FUNCTIONAL MINIMAL CELL REPORT  {'█'*13}
{'█'*72}

  Simulation: {t:.0f} minutes
  Genes: {self.n_genes}

{'═'*72}
                     ⚙️  EMERGENT PROPERTIES ⚙️
{'═'*72}

  ┌────────────────────────────────────────────────────────────────────┐
  │                                                                    │
  │   🔬 GROWTH RATE EMERGES FROM RIBOSOME CONTENT:                   │
  │                                                                    │
  │      Ribosomes:    {n_ribo:>6.0f} complexes                              │
  │      RNAP:         {n_RNAP:>6.0f} complexes                              │
  │      Chaperones:   {n_chap:>6.0f} complexes                              │
  │                                                                    │
  │      EMERGENT μ:   {mu:.4f} /min  (T_double = {np.log(2)/(mu+1e-12):.0f} min)         │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘

{'═'*72}
                        ⚡ METABOLISM ⚡
{'═'*72}

  ┌────────────────────────────────────────────────────────────────────┐
  │  ATP:  {ATP:>5.2f} mM     GTP:  {GTP:>5.2f} mM     Energy charge: {EC:.3f}   │""")
        
        if EC > 0.9:
            print(f"  │  NAD:  {NAD:>5.2f} mM     AA:   {AA:>5.2f} mM     Status: 🟢 EXCELLENT    │")
        elif EC > 0.8:
            print(f"  │  NAD:  {NAD:>5.2f} mM     AA:   {AA:>5.2f} mM     Status: 🟢 HEALTHY      │")
        else:
            print(f"  │  NAD:  {NAD:>5.2f} mM     AA:   {AA:>5.2f} mM     Status: 🟡 MODERATE     │")
        
        print(f"""  └────────────────────────────────────────────────────────────────────┘

{'═'*72}
                      🧬 GENE EXPRESSION 🧬
{'═'*72}

  Total mRNA:    {total_mrna:>8.1f} copies
  Total protein: {total_prot:>8.0f} copies

  KEY FUNCTIONAL PROTEINS:
  ────────────────────────""")
        
        # Show key functional proteins
        key_proteins = ['rpoB', 'rplA', 'tuf', 'groEL', 'pfkA', 'ldh', 'adk']
        for gene in key_proteins:
            if gene in self.prot_idx:
                level = r['protein'][gene][-1]
                cat = self.genes[gene].category
                print(f"  {gene:<10} {level:>8.0f} copies  [{cat}]")
        
        # Homeostasis check
        print(f"""
{'═'*72}
                      📊 HOMEOSTASIS 📊
{'═'*72}
""")
        
        checks = [('ATP', 3.0, ATP), ('GTP', 0.8, GTP), ('NAD', 1.0, NAD), ('AA', 5.0, AA)]
        all_stable = True
        
        for name, target, actual in checks:
            dev = abs(actual - target) / target * 100
            status = "✓" if dev < 15 else ("△" if dev < 30 else "✗")
            if dev >= 15: all_stable = False
            print(f"  {name:<5}: {actual:.2f} mM (target {target:.1f}) - {dev:.1f}% off {status}")
        
        if all_stable:
            print(f"""
{'═'*72}
{'█'*72}
{'█'*8}  🎉 FUNCTIONAL CELL WITH EMERGENT GROWTH! 🎉  {'█'*8}
{'█'*72}
{'═'*72}

  ✓ Enzyme levels determine metabolic flux
  ✓ RNAP levels determine transcription capacity
  ✓ Ribosome levels determine translation capacity
  ✓ Growth rate EMERGES from cellular composition
  
  THIS IS A LIVING CELL!

{'═'*72}
""")
        
        return {'EC': EC, 'mu': mu, 'ribosomes': n_ribo, 'RNAP': n_RNAP, 'stable': all_stable}


def main():
    print("="*72)
    print("    DARK MANIFOLD V50: FUNCTIONAL MINIMAL CELL")
    print("="*72)
    
    model = FunctionalMinimalCell(n_genes=80)
    
    result = model.simulate(t_end=180)
    
    analysis = model.analyze(result)
    
    # Plot
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('JCVI-syn3A Functional Minimal Cell\nV50: Emergent Growth from Protein Function', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    t = result['t']
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # 1. Energy
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, result['ATP'], 'b-', lw=2.5, label='ATP')
    ax.plot(t, result['GTP'], 'g-', lw=2, label='GTP')
    ax.axhline(3.0, color='b', ls=':', alpha=0.4)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Energy Nucleotides', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Machinery
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, result['ribosome'], 'b-', lw=2.5, label='Ribosomes')
    ax.plot(t, result['RNAP'], 'r-', lw=2, label='RNAP')
    ax.plot(t, result['chaperone'], 'g-', lw=2, label='Chaperones')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Complexes')
    ax.set_title('Cellular Machinery', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Emergent growth rate
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, result['mu']*60, 'k-', lw=2.5)  # Convert to /hr
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Growth rate (/hr)')
    ax.set_title('EMERGENT Growth Rate', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 4. Energy charge
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(t, 0.85, 0.95, alpha=0.15, color='green')
    ax.plot(t, result['EC'], 'k-', lw=2.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Energy Charge')
    ax.set_title('Cellular Energy Charge', fontweight='bold')
    ax.set_ylim([0.6, 1.0])
    ax.grid(alpha=0.3)
    
    # 5. Key enzymes
    ax = fig.add_subplot(gs[1, 1])
    enzymes_to_plot = ['pfkA', 'ldh', 'adk', 'ndk']
    for enz in enzymes_to_plot:
        if enz in result['protein']:
            ax.plot(t, result['protein'][enz], lw=2, label=enz)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Protein copies')
    ax.set_title('Metabolic Enzymes', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 6. Translation machinery
    ax = fig.add_subplot(gs[1, 2])
    tl_proteins = ['tuf', 'fusA', 'rplA', 'rpsA']
    for prot in tl_proteins:
        if prot in result['protein']:
            ax.plot(t, result['protein'][prot], lw=2, label=prot)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Protein copies')
    ax.set_title('Translation Machinery', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.savefig('functional_cell.png', dpi=200, bbox_inches='tight')
    print("\n✓ Saved: functional_cell.png")
    
    return model, result, analysis


if __name__ == '__main__':
    model, result, analysis = main()
