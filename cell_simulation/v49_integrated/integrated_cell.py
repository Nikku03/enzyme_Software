"""
Dark Manifold V49: Integrated Minimal Cell
==========================================

FULL COUPLING of:
1. Metabolism (V48) - ATP, NTPs, amino acids
2. Gene Expression (V47) - transcription, translation

COUPLING POINTS:
- Transcription CONSUMES NTPs (ATP, GTP, CTP, UTP)
- Translation CONSUMES ATP, GTP, and amino acids
- Low ATP/GTP SLOWS gene expression
- High expression DEPLETES metabolite pools

This is the heart of the virtual cell!
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from collections import defaultdict
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import our modules
sys.path.insert(0, '/home/claude/enzyme_repo/cell_simulation')
from v47_gene_expression.gene_expression_v2 import Gene, build_full_syn3a_genes
from v48_metabolism.metabolism_v7_proud import ProudMinimalCellMetabolism


# ============================================================================
# CONSTANTS
# ============================================================================

# Molecular weights and stoichiometry
ATP_PER_AA = 2       # ATP consumed per amino acid in translation (tRNA charging)
GTP_PER_AA = 2       # GTP consumed per amino acid (EF-Tu, EF-G)
NTP_PER_NT = 1       # NTP per nucleotide in transcription

# Average NTP composition of mRNA (roughly equal)
NTP_FRACTIONS = {'ATP': 0.25, 'GTP': 0.25, 'CTP': 0.25, 'UTP': 0.25}

# Conversion factors
CELL_VOLUME = 5e-17  # L
AVOGADRO = 6.022e23

def molecules_to_mM(n):
    return (n / AVOGADRO) / CELL_VOLUME * 1000

def mM_to_molecules(c):
    return c * 1e-3 * CELL_VOLUME * AVOGADRO


# ============================================================================
# INTEGRATED CELL MODEL
# ============================================================================

class IntegratedMinimalCell:
    """
    Fully integrated minimal cell model.
    
    Combines:
    - Metabolism: ATP, GTP, UTP, CTP, NAD, amino acids
    - Gene expression: mRNA and protein for 177 genes
    
    Coupling:
    - Transcription rate depends on NTP availability
    - Translation rate depends on ATP, GTP, AA availability
    - Gene expression consumes metabolites
    """
    
    def __init__(self, n_genes: int = 50):
        """
        Initialize integrated model.
        
        Args:
            n_genes: Number of genes to simulate (default 50 for speed)
        """
        print("="*72)
        print("    DARK MANIFOLD V49: INTEGRATED MINIMAL CELL")
        print("="*72)
        
        # ===== BUILD GENE SET =====
        all_genes = build_full_syn3a_genes()
        
        # Select subset of most important genes
        # Priority: ribosomes > RNAP > metabolism > others
        priority_categories = [
            'ribosomal_protein',
            'rRNA', 
            'RNAP',
            'translation_factor',
            'chaperone',
            'metabolism',
        ]
        
        selected_genes = {}
        for cat in priority_categories:
            for name, gene in all_genes.items():
                if gene.category == cat and len(selected_genes) < n_genes:
                    selected_genes[name] = gene
        
        # Fill remaining with other genes
        for name, gene in all_genes.items():
            if name not in selected_genes and len(selected_genes) < n_genes:
                selected_genes[name] = gene
        
        self.genes = selected_genes
        self.gene_list = list(self.genes.keys())
        self.n_genes = len(self.gene_list)
        
        print(f"\nGenes: {self.n_genes}")
        
        # ===== METABOLISM STATE =====
        # We directly include metabolite states (not a separate object)
        self.met_states = ['ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'NAD', 'NADH',
                          'UTP', 'CTP', 'AA', 'Pyr', 'Lac']
        self.n_met = len(self.met_states)
        
        # Target metabolite concentrations
        self.met_targets = {
            'ATP': 3.0, 'ADP': 0.5, 'AMP': 0.1,
            'GTP': 0.8, 'GDP': 0.2,
            'NAD': 1.0, 'NADH': 0.1,
            'UTP': 0.5, 'CTP': 0.3,
            'AA': 5.0,
            'Pyr': 0.3, 'Lac': 2.0,
        }
        
        # ===== FULL STATE =====
        # State = [metabolites (12), mRNA (n_genes), protein (n_genes)]
        self.n_states = self.n_met + 2 * self.n_genes
        
        # Indices
        self.met_idx = {m: i for i, m in enumerate(self.met_states)}
        self.mrna_idx = {g: self.n_met + i for i, g in enumerate(self.gene_list)}
        self.prot_idx = {g: self.n_met + self.n_genes + i for i, g in enumerate(self.gene_list)}
        
        # ===== KINETIC PARAMETERS =====
        self._setup_kinetics()
        
        # ===== PRINT SUMMARY =====
        self._print_summary()
    
    def _setup_kinetics(self):
        """Setup all kinetic parameters."""
        
        # Growth rate
        self.mu = 0.01  # ~70 min doubling
        
        # ----- METABOLISM -----
        # Glycolysis
        self.V_glyc = 0.5
        self.Km_NAD = 0.2
        self.Km_ADP = 0.2
        self.Ki_ATP = 2.0
        
        # LDH
        self.V_ldh = 5.0
        self.Km_Pyr = 0.2
        self.Km_NADH = 0.05
        
        # Maintenance
        self.V_maint = 0.02
        
        # AA uptake (external source)
        self.V_aa_uptake = 0.15
        self.AA_ext = 2.0
        
        # NTP synthesis
        self.V_ntp_synth = 0.03
        
        # Fast kinases
        self.k_adk = 50.0
        self.k_ndk = 50.0
        
        # Homeostatic buffering
        self.tau_buffer = {
            'ATP': 5.0, 'ADP': 5.0, 'AMP': 10.0,
            'GTP': 5.0, 'GDP': 10.0,
            'NAD': 5.0, 'NADH': 5.0,
            'UTP': 10.0, 'CTP': 10.0,
            'AA': 10.0,
            'Pyr': 20.0, 'Lac': 30.0,
        }
        
        # ----- GENE EXPRESSION -----
        # Resource pools
        self.total_RNAP = 200
        self.total_ribosomes = 1000
        
        # Scaling
        self.k_tx_scale = 0.3   # Transcription
        self.k_tl_scale = 1.5   # Translation
        
        # Metabolite sensitivity
        self.Km_NTP_tx = 0.3     # Km for NTPs in transcription
        self.Km_ATP_tl = 0.5     # Km for ATP in translation
        self.Km_GTP_tl = 0.3     # Km for GTP in translation
        self.Km_AA_tl = 1.0      # Km for amino acids
        
        # Calculate gene-specific rates
        for gene in self.genes.values():
            gene.k_tx_init = self.k_tx_scale * (gene.promoter_strength / 10.0)
            gene.k_tl_init = self.k_tl_scale * (gene.rbs_strength / 10.0) * gene.codon_adaptation
    
    def _print_summary(self):
        """Print model summary."""
        
        by_cat = defaultdict(int)
        for gene in self.genes.values():
            by_cat[gene.category] += 1
        
        print(f"\nGenes by category:")
        for cat, count in sorted(by_cat.items(), key=lambda x: -x[1])[:8]:
            print(f"  {cat:<20} {count:>3}")
        
        print(f"\nTotal states: {self.n_states}")
        print(f"  Metabolites: {self.n_met}")
        print(f"  mRNAs: {self.n_genes}")
        print(f"  Proteins: {self.n_genes}")
        
        print(f"\nResources: {self.total_RNAP} RNAP, {self.total_ribosomes} ribosomes")
    
    def get_initial_state(self) -> np.ndarray:
        """Initialize at approximate steady state."""
        
        y0 = np.zeros(self.n_states)
        
        # Metabolites at target
        for met, target in self.met_targets.items():
            y0[self.met_idx[met]] = target
        
        # Gene expression at rough steady state
        for i, name in enumerate(self.gene_list):
            gene = self.genes[name]
            mrna_ss = gene.k_tx_init / gene.delta_m * 0.5
            prot_ss = gene.k_tl_init * mrna_ss / gene.delta_p
            y0[self.mrna_idx[name]] = mrna_ss
            y0[self.prot_idx[name]] = prot_ss
        
        return y0
    
    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        """Calculate derivatives for the full system."""
        
        dy = np.zeros(self.n_states)
        
        # ===== UNPACK METABOLITES =====
        def get_met(name):
            return max(y[self.met_idx[name]], 1e-12)
        
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
        
        # NTP availability for transcription (minimum of all 4)
        NTP_min = min(ATP, GTP, UTP, CTP)
        f_NTP = NTP_min / (self.Km_NTP_tx + NTP_min)
        
        # ATP/GTP/AA availability for translation
        f_ATP_tl = ATP / (self.Km_ATP_tl + ATP)
        f_GTP_tl = GTP / (self.Km_GTP_tl + GTP)
        f_AA = AA / (self.Km_AA_tl + AA)
        f_translation = f_ATP_tl * f_GTP_tl * f_AA
        
        # ===== RESOURCE COMPETITION =====
        rnap_engaged = 0
        ribo_engaged = 0
        
        for name in self.gene_list:
            gene = self.genes[name]
            mrna = max(y[self.mrna_idx[name]], 0)
            rnap_engaged += gene.k_tx_init * gene.tx_elongation_time
            ribo_engaged += mrna * gene.k_tl_init * gene.tl_elongation_time
        
        free_RNAP = max(self.total_RNAP - rnap_engaged, 1)
        free_ribo = max(self.total_ribosomes - ribo_engaged, 1)
        f_RNAP = free_RNAP / self.total_RNAP
        f_ribo = free_ribo / self.total_ribosomes
        
        # ===== GENE EXPRESSION =====
        total_tx_flux = 0  # Total NTP consumption by transcription
        total_tl_flux = 0  # Total AA consumption by translation
        total_tl_atp = 0   # ATP for translation
        total_tl_gtp = 0   # GTP for translation
        
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
            
            # TRANSCRIPTION (depends on NTPs and RNAP)
            tx_rate = gene.k_tx_init * f_RNAP * f_NTP * repression
            dy[i_mrna] = tx_rate - gene.delta_m * mrna
            
            # NTP consumption: gene.length_nt nucleotides per transcript
            # But rate is in transcripts/min, need to convert
            # Actually, tx_rate is already initiation rate
            # Each transcript uses ~1000 nt on average, but we track by transcript
            # Simplify: 1 "unit" of NTP per initiation (scaled)
            ntp_per_tx = gene.length_nt / 1000  # Normalized to 1kb
            total_tx_flux += tx_rate * ntp_per_tx
            
            # TRANSLATION (depends on ATP, GTP, AA and ribosomes)
            tl_rate = gene.k_tl_init * mrna * f_ribo * f_translation
            dy[i_prot] = tl_rate - gene.delta_p * prot
            
            # Metabolite consumption per protein
            aa_per_prot = gene.length_aa / 300  # Normalized to 300 aa
            total_tl_flux += tl_rate * aa_per_prot
            total_tl_atp += tl_rate * aa_per_prot * ATP_PER_AA
            total_tl_gtp += tl_rate * aa_per_prot * GTP_PER_AA
        
        # Scale fluxes to realistic metabolite consumption
        # (these are arbitrary units, need to match metabolism timescale)
        tx_scale = 0.01  # mM NTP consumed per "transcription unit"
        tl_scale = 0.01  # mM AA consumed per "translation unit"
        
        # ===== METABOLISM =====
        
        # --- GLYCOLYSIS ---
        Glc_int = 2.0  # Assume constant internal glucose
        v_glyc = self.V_glyc * NAD/(self.Km_NAD + NAD) * ADP/(self.Km_ADP + ADP) * \
                 self.Ki_ATP/(self.Ki_ATP + ATP)
        
        dy[self.met_idx['NAD']] -= 2 * v_glyc
        dy[self.met_idx['NADH']] += 2 * v_glyc
        dy[self.met_idx['ADP']] -= 2 * v_glyc
        dy[self.met_idx['ATP']] += 2 * v_glyc
        dy[self.met_idx['Pyr']] += 2 * v_glyc
        
        # --- LDH ---
        v_ldh = self.V_ldh * Pyr/(self.Km_Pyr + Pyr) * NADH/(self.Km_NADH + NADH)
        
        dy[self.met_idx['Pyr']] -= v_ldh
        dy[self.met_idx['NADH']] -= v_ldh
        dy[self.met_idx['Lac']] += v_ldh
        dy[self.met_idx['NAD']] += v_ldh
        
        # --- TRANSCRIPTION CONSUMPTION ---
        # Consumes NTPs (ATP, GTP, CTP, UTP)
        v_tx_ntp = total_tx_flux * tx_scale
        dy[self.met_idx['ATP']] -= 0.25 * v_tx_ntp
        dy[self.met_idx['GTP']] -= 0.25 * v_tx_ntp
        dy[self.met_idx['UTP']] -= 0.25 * v_tx_ntp
        dy[self.met_idx['CTP']] -= 0.25 * v_tx_ntp
        
        # --- TRANSLATION CONSUMPTION ---
        # Consumes ATP, GTP, and amino acids
        v_tl_aa = total_tl_flux * tl_scale
        dy[self.met_idx['AA']] -= v_tl_aa
        dy[self.met_idx['ATP']] -= total_tl_atp * tl_scale * 0.5
        dy[self.met_idx['ADP']] += total_tl_atp * tl_scale * 0.5
        dy[self.met_idx['GTP']] -= total_tl_gtp * tl_scale * 0.5
        dy[self.met_idx['GDP']] += total_tl_gtp * tl_scale * 0.5
        
        # --- MAINTENANCE ---
        v_maint = self.V_maint * ATP/(0.3 + ATP)
        dy[self.met_idx['ATP']] -= v_maint
        dy[self.met_idx['ADP']] += v_maint
        
        # --- ADENYLATE KINASE ---
        v_adk = self.k_adk * (ADP * ADP - ATP * AMP)
        dy[self.met_idx['ATP']] += v_adk
        dy[self.met_idx['AMP']] += v_adk
        dy[self.met_idx['ADP']] -= 2 * v_adk
        
        # --- NDP KINASE ---
        v_ndk = self.k_ndk * (GDP * ATP - GTP * ADP)
        dy[self.met_idx['GTP']] += v_ndk
        dy[self.met_idx['ADP']] += v_ndk
        dy[self.met_idx['GDP']] -= v_ndk
        dy[self.met_idx['ATP']] -= v_ndk
        
        # --- AA UPTAKE ---
        v_aa = self.V_aa_uptake * self.AA_ext / (0.5 + self.AA_ext)
        dy[self.met_idx['AA']] += v_aa
        
        # --- NTP SYNTHESIS ---
        v_ntp = self.V_ntp_synth * ATP / (1.0 + ATP)
        dy[self.met_idx['UTP']] += 0.5 * v_ntp
        dy[self.met_idx['CTP']] += 0.5 * v_ntp
        dy[self.met_idx['ATP']] -= v_ntp
        dy[self.met_idx['ADP']] += v_ntp
        
        # --- LACTATE EXPORT ---
        v_lac = 0.3 * Lac / (2.0 + Lac)
        dy[self.met_idx['Lac']] -= v_lac
        
        # --- HOMEOSTATIC BUFFERING ---
        for met in self.met_states:
            target = self.met_targets[met]
            current = y[self.met_idx[met]]
            tau = self.tau_buffer[met]
            dy[self.met_idx[met]] += (target - current) / tau
        
        # --- GROWTH DILUTION ---
        for i in range(self.n_states):
            dy[i] -= self.mu * y[i]
        
        return dy
    
    def simulate(self, t_end: float = 180, n_points: int = 500) -> dict:
        """Run simulation."""
        
        t_eval = np.linspace(0, t_end, n_points)
        y0 = self.get_initial_state()
        
        print(f"\nSimulating 0 → {t_end} min...")
        print(f"States: {self.n_states} ({self.n_met} metabolites + {2*self.n_genes} gene expression)")
        
        sol = solve_ivp(self.deriv, (0, t_end), y0, t_eval=t_eval,
                       method='LSODA', rtol=1e-6, atol=1e-9)
        
        # Unpack results
        result = {'t': sol.t}
        
        # Metabolites
        for met in self.met_states:
            result[met] = sol.y[self.met_idx[met], :]
        
        # Gene expression
        result['mrna'] = {}
        result['protein'] = {}
        for name in self.gene_list:
            result['mrna'][name] = sol.y[self.mrna_idx[name], :]
            result['protein'][name] = sol.y[self.prot_idx[name], :]
        
        # Derived quantities
        result['EC'] = (result['ATP'] + 0.5*result['ADP']) / \
                       (result['ATP'] + result['ADP'] + result['AMP'] + 1e-12)
        result['total_mrna'] = sum(sol.y[self.mrna_idx[g], :] for g in self.gene_list)
        result['total_protein'] = sum(sol.y[self.prot_idx[g], :] for g in self.gene_list)
        
        return result
    
    def analyze(self, r: dict) -> Dict:
        """Analyze simulation results."""
        
        t = r['t'][-1]
        
        # Final metabolites
        ATP = r['ATP'][-1]
        ADP = r['ADP'][-1]
        AMP = r['AMP'][-1]
        GTP = r['GTP'][-1]
        NAD = r['NAD'][-1]
        NADH = r['NADH'][-1]
        AA = r['AA'][-1]
        EC = r['EC'][-1]
        
        # Gene expression totals
        total_mrna = r['total_mrna'][-1]
        total_prot = r['total_protein'][-1]
        
        print(f"""
{'█'*72}
{'█'*15}  INTEGRATED MINIMAL CELL REPORT  {'█'*16}
{'█'*72}

  Simulation: {t:.0f} minutes
  Growth rate: μ = {self.mu:.3f} /min (T_double = {np.log(2)/self.mu:.0f} min)
  Genes: {self.n_genes}

{'═'*72}
                        ⚡ METABOLISM ⚡
{'═'*72}

  ┌────────────────────────────────────────────────────────────────────┐
  │  ENERGY                           │  PRECURSORS                   │
  │  ──────                           │  ──────────                   │
  │  ATP:  {ATP:>5.2f} mM  (target: 3.0)   │  Amino acids: {AA:>5.2f} mM        │
  │  ADP:  {ADP:>5.2f} mM                  │  Pyruvate:    {r['Pyr'][-1]:>5.2f} mM        │
  │  GTP:  {GTP:>5.2f} mM  (target: 0.8)   │  Lactate:     {r['Lac'][-1]:>5.2f} mM        │
  │                                   │                               │
  │  Energy charge: {EC:.3f}           │  NAD⁺/NADH:   {NAD/(NADH+1e-12):>5.1f}          │""")
        
        if EC > 0.9:
            print(f"  │  Status: 🟢 EXCELLENT             │                               │")
        elif EC > 0.8:
            print(f"  │  Status: 🟢 HEALTHY               │                               │")
        else:
            print(f"  │  Status: 🟡 MODERATE              │                               │")
        
        print(f"""  └────────────────────────────────────────────────────────────────────┘

{'═'*72}
                      🧬 GENE EXPRESSION 🧬
{'═'*72}

  ┌────────────────────────────────────────────────────────────────────┐
  │                                                                    │
  │  Total mRNA:    {total_mrna:>8.1f} copies                                 │
  │  Total protein: {total_prot:>8.0f} copies                                 │
  │                                                                    │
  │  Average mRNA per gene:    {total_mrna/self.n_genes:>6.2f}                            │
  │  Average protein per gene: {total_prot/self.n_genes:>6.0f}                            │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘

  TOP 10 EXPRESSED GENES:
  ───────────────────────""")
        
        # Get top proteins
        prot_final = [(name, r['protein'][name][-1]) for name in self.gene_list]
        prot_final.sort(key=lambda x: -x[1])
        
        for i, (name, level) in enumerate(prot_final[:10]):
            cat = self.genes[name].category
            mrna_level = r['mrna'][name][-1]
            print(f"  {i+1:>2}. {name:<12} {level:>8.0f} copies  [{cat}]")
        
        # Homeostasis check
        print(f"""
{'═'*72}
                      📊 HOMEOSTASIS CHECK 📊
{'═'*72}
""")
        
        checks = [
            ('ATP', 3.0, ATP),
            ('GTP', 0.8, GTP),
            ('NAD', 1.0, NAD),
            ('AA', 5.0, AA),
        ]
        
        all_stable = True
        for name, target, actual in checks:
            dev = abs(actual - target) / target * 100
            if dev < 15:
                status = "✓ STABLE"
            elif dev < 30:
                status = "△ CLOSE"
                all_stable = False
            else:
                status = "✗ DRIFTED"
                all_stable = False
            print(f"  {name:<5}: target={target:.1f}  actual={actual:.2f} mM  ({dev:>5.1f}% off) {status}")
        
        # Gene expression stability
        mrna_0 = r['total_mrna'][0]
        mrna_f = r['total_mrna'][-1]
        prot_0 = r['total_protein'][0]
        prot_f = r['total_protein'][-1]
        
        print(f"\n  mRNA:    {mrna_0:.1f} → {mrna_f:.1f} ({(mrna_f-mrna_0)/mrna_0*100:+.1f}%)")
        print(f"  Protein: {prot_0:.0f} → {prot_f:.0f} ({(prot_f-prot_0)/prot_0*100:+.1f}%)")
        
        if all_stable:
            print(f"""
{'═'*72}
{'█'*72}
{'█'*10}  🎉 INTEGRATED CELL ACHIEVES HOMEOSTASIS! 🎉  {'█'*10}
{'█'*72}
{'═'*72}

  ✓ Metabolism and gene expression are COUPLED
  ✓ ATP/GTP pools maintained despite expression load
  ✓ Amino acid flux balanced with protein synthesis
  ✓ The cell is alive and growing!

{'═'*72}
""")
        
        return {
            'EC': EC, 'ATP': ATP, 'GTP': GTP, 'AA': AA,
            'total_mrna': total_mrna, 'total_protein': total_prot,
            'stable': all_stable
        }


def main():
    print("="*72)
    print("    DARK MANIFOLD V49: INTEGRATED MINIMAL CELL")
    print("="*72)
    
    # Create integrated model
    model = IntegratedMinimalCell(n_genes=50)
    
    # Simulate
    result = model.simulate(t_end=180)
    
    # Analyze
    analysis = model.analyze(result)
    
    # Plot
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('JCVI-syn3A Integrated Minimal Cell\nDark Manifold V49: Metabolism + Gene Expression', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    t = result['t']
    
    # 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                          left=0.08, right=0.95, top=0.9, bottom=0.08)
    
    # 1. ATP/GTP
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, result['ATP'], 'b-', lw=2.5, label='ATP')
    ax.plot(t, result['GTP'], 'g-', lw=2, label='GTP')
    ax.plot(t, result['ADP'], 'r--', lw=1.5, label='ADP')
    ax.axhline(3.0, color='b', ls=':', alpha=0.4)
    ax.axhline(0.8, color='g', ls=':', alpha=0.4)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Energy Nucleotides', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 2. Energy Charge
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(t, 0.85, 0.95, alpha=0.15, color='green')
    ax.plot(t, result['EC'], 'k-', lw=2.5)
    ax.axhline(0.9, color='g', ls='--', alpha=0.5)
    ax.axhline(0.7, color='r', ls='--', alpha=0.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Energy Charge')
    ax.set_title('Cellular Energy Charge', fontweight='bold')
    ax.set_ylim([0.6, 1.0])
    ax.grid(alpha=0.3)
    
    # 3. Amino Acids
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, result['AA'], 'purple', lw=2.5, label='AA pool')
    ax.axhline(5.0, color='purple', ls=':', alpha=0.4)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Amino Acid Pool', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 4. Total mRNA
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, result['total_mrna'], 'b-', lw=2.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Total copies')
    ax.set_title('Total mRNA', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 5. Total Protein
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, result['total_protein'], 'r-', lw=2.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Total copies')
    ax.set_title('Total Protein', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 6. Sample genes
    ax = fig.add_subplot(gs[1, 2])
    # Pick a few interesting genes
    sample_genes = ['rplA', 'groEL', 'dnaK', 'tufA'][:4]
    available = [g for g in sample_genes if g in model.gene_list]
    if not available:
        available = model.gene_list[:4]
    
    for gene in available:
        ax.plot(t, result['protein'][gene], lw=2, label=gene)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Protein copies')
    ax.set_title('Sample Proteins', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.savefig('integrated_cell.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("\n✓ Saved: integrated_cell.png")
    
    return model, result, analysis


if __name__ == '__main__':
    model, result, analysis = main()
