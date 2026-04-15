"""
Dark Manifold V51: Dividing Minimal Cell
========================================

COMPLETE CELL CYCLE:
1. Growth phase - accumulate biomass
2. DNA replication - triggered by DnaA
3. Cell division - triggered by FtsZ

The cell cycle timing EMERGES from molecular concentrations!
No explicit timers - just biochemistry.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/claude/enzyme_repo/cell_simulation')
from v47_gene_expression.gene_expression_v2 import Gene, build_full_syn3a_genes


# ============================================================================
# CELL CYCLE PARAMETERS
# ============================================================================

@dataclass
class CellCycleParams:
    """Parameters for cell cycle control."""
    
    # DNA replication
    genome_size: int = 543000        # bp (JCVI-syn3A)
    replication_rate: float = 500    # bp/s per fork
    
    # Initiation threshold (DnaA-ATP)
    DnaA_threshold: float = 30       # molecules for initiation (lowered)
    DnaA_Kd_ATP: float = 0.5         # mM (Kd for ATP binding)
    
    # dNTP requirements
    dNTP_per_bp: float = 1.0         # dNTPs consumed per bp
    Km_dNTP: float = 0.05            # mM
    
    # Division
    FtsZ_threshold: float = 100      # molecules for Z-ring (lowered)
    division_time: float = 10.0      # min (once triggered)
    
    # Critical cell size (relative)
    critical_size: float = 1.5       # relative to newborn


# ============================================================================
# DIVIDING CELL MODEL
# ============================================================================

class DividingMinimalCell:
    """
    Complete cell with DNA replication and division.
    
    Cell cycle emerges from:
    1. DnaA accumulation → triggers replication
    2. dNTP availability → determines replication speed
    3. FtsZ accumulation → triggers division
    4. Replication completion → enables division
    
    Division events:
    - All metabolites halved
    - All proteins halved (partitioned to daughters)
    - DNA content reset to 1
    - Cell cycle state reset
    """
    
    def __init__(self, n_genes: int = 60):
        print("="*72)
        print("    DARK MANIFOLD V51: DIVIDING MINIMAL CELL")
        print("="*72)
        
        # Cell cycle parameters
        self.cc = CellCycleParams()
        
        # ===== BUILD GENE SET =====
        all_genes = build_full_syn3a_genes()
        
        # Essential genes for cell cycle
        essential = [
            # RNAP
            'rpoA', 'rpoB', 'rpoC', 'rpoD',
            # Ribosomes
            'rplA', 'rplB', 'rplC', 'rpsA', 'rpsB', 'rpsC',
            # Translation
            'tuf', 'fusA', 'tsf', 'infB',
            # Chaperones
            'groEL', 'groES', 'dnaK',
            # Glycolysis
            'pfkA', 'gapA', 'pykF', 'ldh',
            # Kinases
            'adk', 'ndk',
            # DNA REPLICATION
            'dnaA',  # Initiation - KEY!
            'dnaB', 'dnaC',  # Helicase loading
            'dnaE', 'dnaN',  # DNA Pol III
            'dnaG',  # Primase
            'gyrA', 'gyrB',  # Topoisomerase
            'ligA',  # Ligase
            # CELL DIVISION
            'ftsZ',  # Z-ring - KEY!
            'ftsA', 'ftsB', 'ftsL', 'ftsQ',
            'ftsW', 'ftsI',
        ]
        
        self.genes = {}
        for name in essential:
            if name in all_genes:
                self.genes[name] = all_genes[name]
        
        # Fill remaining
        for name, gene in all_genes.items():
            if name not in self.genes and len(self.genes) < n_genes:
                self.genes[name] = gene
        
        self.gene_list = list(self.genes.keys())
        self.n_genes = len(self.gene_list)
        
        # ===== STATE STRUCTURE =====
        # Metabolites (14)
        self.met_states = ['ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'NAD', 'NADH',
                          'UTP', 'CTP', 'AA', 'Pyr', 'Lac',
                          'dATP', 'dNTP_pool']  # dNTPs for replication
        self.n_met = len(self.met_states)
        
        # Cell cycle states (4)
        self.cc_states = ['DNA_content', 'replication_progress', 
                         'division_progress', 'cell_mass']
        self.n_cc = len(self.cc_states)
        
        # Full state
        self.n_states = self.n_met + self.n_cc + 2 * self.n_genes
        
        # Indices
        self.met_idx = {m: i for i, m in enumerate(self.met_states)}
        self.cc_idx = {c: self.n_met + i for i, c in enumerate(self.cc_states)}
        self.mrna_idx = {g: self.n_met + self.n_cc + i for i, g in enumerate(self.gene_list)}
        self.prot_idx = {g: self.n_met + self.n_cc + self.n_genes + i 
                        for i, g in enumerate(self.gene_list)}
        
        # ===== KINETICS =====
        self._setup_kinetics()
        
        # ===== TRACKING =====
        self.division_count = 0
        self.division_times = []
        
        self._print_summary()
    
    def _setup_kinetics(self):
        """Setup kinetic parameters."""
        
        # Targets
        self.met_targets = {
            'ATP': 3.0, 'ADP': 0.5, 'AMP': 0.1,
            'GTP': 0.8, 'GDP': 0.2,
            'NAD': 1.0, 'NADH': 0.1,
            'UTP': 0.5, 'CTP': 0.3,
            'AA': 5.0,
            'Pyr': 0.3, 'Lac': 2.0,
            'dATP': 0.05, 'dNTP_pool': 0.2,
        }
        
        self.tau_buffer = {m: 10.0 for m in self.met_states}
        self.tau_buffer['ATP'] = 5.0
        self.tau_buffer['dNTP_pool'] = 15.0
        
        # External
        self.Glc_ext = 20.0
        self.AA_ext = 2.0
        
        # Gene expression - higher rates
        self.k_tx_base = 0.5
        self.k_tl_base = 2.0
        
        self.K_RNAP = 10
        self.K_ribo = 50
        
        for gene in self.genes.values():
            gene.k_tx_init = self.k_tx_base * (gene.promoter_strength / 5.0)
            gene.k_tl_init = self.k_tl_base * (gene.rbs_strength / 5.0)
    
    def _print_summary(self):
        print(f"\nGenes: {self.n_genes}")
        print(f"States: {self.n_states}")
        print(f"  Metabolites: {self.n_met}")
        print(f"  Cell cycle: {self.n_cc}")
        print(f"  Gene expression: {2*self.n_genes}")
        
        # Check for key genes
        key_genes = ['dnaA', 'dnaE', 'ftsZ']
        print(f"\nKey cell cycle genes:")
        for g in key_genes:
            status = "✓" if g in self.genes else "✗"
            print(f"  {status} {g}")
    
    def get_initial_state(self) -> np.ndarray:
        """Initialize newborn cell."""
        y0 = np.zeros(self.n_states)
        
        # Metabolites
        for met, target in self.met_targets.items():
            if met in self.met_idx:
                y0[self.met_idx[met]] = target
        
        # Cell cycle - newborn cell
        y0[self.cc_idx['DNA_content']] = 1.0
        y0[self.cc_idx['replication_progress']] = 0.0
        y0[self.cc_idx['division_progress']] = 0.0
        y0[self.cc_idx['cell_mass']] = 1.0
        
        # Gene expression - start with substantial protein levels
        for name in self.gene_list:
            gene = self.genes[name]
            mrna_ss = gene.k_tx_init / gene.delta_m * 5.0
            prot_ss = gene.k_tl_init * mrna_ss / gene.delta_p * 50.0
            
            # Boost key cell cycle proteins
            if name in ['dnaA', 'dnaE', 'dnaN', 'dnaB', 'ftsZ', 'ftsA']:
                prot_ss *= 3.0
            
            y0[self.mrna_idx[name]] = max(mrna_ss, 0.5)
            y0[self.prot_idx[name]] = max(prot_ss, 20)
        
        return y0
    
    def get_protein(self, y: np.ndarray, name: str) -> float:
        """Get protein level safely."""
        if name in self.prot_idx:
            return max(y[self.prot_idx[name]], 0)
        return 0
    
    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        """Calculate derivatives including cell cycle."""
        
        dy = np.zeros(self.n_states)
        
        # ===== UNPACK =====
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
        dNTP = get_met('dNTP_pool')
        
        DNA_content = y[self.cc_idx['DNA_content']]
        rep_progress = y[self.cc_idx['replication_progress']]
        div_progress = y[self.cc_idx['division_progress']]
        cell_mass = y[self.cc_idx['cell_mass']]
        
        # Key proteins
        DnaA = self.get_protein(y, 'dnaA')
        DnaE = self.get_protein(y, 'dnaE')
        FtsZ = self.get_protein(y, 'ftsZ')
        
        # ===== CELL CYCLE LOGIC =====
        
        # 1. Replication initiation
        # DnaA-ATP triggers initiation when above threshold
        DnaA_ATP = DnaA * ATP / (self.cc.DnaA_Kd_ATP + ATP)
        
        replicating = rep_progress > 0 and rep_progress < 1.0
        initiation_ready = (DnaA_ATP > self.cc.DnaA_threshold and 
                           rep_progress == 0 and
                           DNA_content < 1.5)  # Not already replicating
        
        if initiation_ready:
            # Start replication
            dy[self.cc_idx['replication_progress']] = 0.001  # Small kick to start
        
        # 2. Replication elongation
        if replicating:
            # Rate depends on dNTPs and DNA polymerase
            v_rep = (self.cc.replication_rate * 60 / self.cc.genome_size *  # bp/min normalized
                    DnaE / (20 + DnaE) *  # Enzyme saturation
                    dNTP / (self.cc.Km_dNTP + dNTP))  # Substrate saturation
            
            dy[self.cc_idx['replication_progress']] = v_rep
            
            # DNA content increases with replication
            dy[self.cc_idx['DNA_content']] = v_rep
            
            # Consume dNTPs
            dNTP_consumption = v_rep * 0.1  # Scaled consumption
            dy[self.met_idx['dNTP_pool']] -= dNTP_consumption
        
        # Cap replication at completion
        if rep_progress >= 1.0:
            dy[self.cc_idx['replication_progress']] = 0
            # DNA content should be ~2
        
        # 3. Division initiation
        # FtsZ assembles Z-ring after replication completes
        division_ready = (DNA_content >= 1.8 and  # Replication mostly done
                         FtsZ > self.cc.FtsZ_threshold and
                         div_progress < 1.0)
        
        if division_ready and div_progress == 0:
            dy[self.cc_idx['division_progress']] = 0.001  # Start division
        
        # 4. Division progression
        if div_progress > 0 and div_progress < 1.0:
            # Division rate depends on FtsZ and energy
            v_div = (1.0 / self.cc.division_time *
                    FtsZ / (self.cc.FtsZ_threshold + FtsZ) *
                    ATP / (1.0 + ATP))
            
            dy[self.cc_idx['division_progress']] = v_div
        
        # ===== METABOLISM =====
        
        # Glycolysis
        v_glyc = 0.4 * NAD/(0.2 + NAD) * ADP/(0.2 + ADP) * 2.0/(2.0 + ATP)
        
        dy[self.met_idx['NAD']] -= 2 * v_glyc
        dy[self.met_idx['NADH']] += 2 * v_glyc
        dy[self.met_idx['ADP']] -= 2 * v_glyc
        dy[self.met_idx['ATP']] += 2 * v_glyc
        dy[self.met_idx['Pyr']] += 2 * v_glyc
        
        # LDH
        v_ldh = 3.0 * Pyr/(0.2 + Pyr) * NADH/(0.02 + NADH)
        dy[self.met_idx['Pyr']] -= v_ldh
        dy[self.met_idx['NADH']] -= v_ldh
        dy[self.met_idx['Lac']] += v_ldh
        dy[self.met_idx['NAD']] += v_ldh
        
        # Kinases
        v_adk = 30.0 * (ADP * ADP - ATP * AMP)
        dy[self.met_idx['ATP']] += v_adk
        dy[self.met_idx['AMP']] += v_adk
        dy[self.met_idx['ADP']] -= 2 * v_adk
        
        v_ndk = 30.0 * (GDP * ATP - GTP * ADP)
        dy[self.met_idx['GTP']] += v_ndk
        dy[self.met_idx['ADP']] += v_ndk
        dy[self.met_idx['GDP']] -= v_ndk
        dy[self.met_idx['ATP']] -= v_ndk
        
        # dNTP synthesis
        v_dntp = 0.02 * ATP/(1.0 + ATP) * (0.2 - dNTP)/(0.1 + abs(0.2 - dNTP))
        dy[self.met_idx['dNTP_pool']] += v_dntp
        dy[self.met_idx['ATP']] -= v_dntp * 2
        dy[self.met_idx['ADP']] += v_dntp * 2
        
        # ===== GENE EXPRESSION =====
        
        # Calculate machinery
        RNAP_genes = ['rpoA', 'rpoB', 'rpoC', 'rpoD']
        n_RNAP = min([self.get_protein(y, g) for g in RNAP_genes if g in self.prot_idx] or [0])
        
        ribo_genes = ['rplA', 'rpsA']
        n_ribo = min([self.get_protein(y, g) for g in ribo_genes if g in self.prot_idx] or [0])
        
        f_RNAP = n_RNAP / (self.K_RNAP + n_RNAP)
        f_ribo = n_ribo / (self.K_ribo + n_ribo)
        
        NTP_min = min(ATP, GTP, UTP, CTP)
        f_NTP = NTP_min / (0.3 + NTP_min)
        f_AA = AA / (1.0 + AA)
        
        # Gene dosage effect - more DNA = more transcription
        gene_dosage = DNA_content
        
        total_tx = 0
        total_tl = 0
        
        for name in self.gene_list:
            gene = self.genes[name]
            i_m = self.mrna_idx[name]
            i_p = self.prot_idx[name]
            
            mrna = max(y[i_m], 0)
            prot = max(y[i_p], 0)
            
            # Autorepression
            rep = 1.0
            if gene.autorepression and prot > 0:
                rep = gene.K_autorepression / (gene.K_autorepression + prot)
            
            # Transcription (scales with gene dosage!)
            tx = gene.k_tx_init * f_RNAP * f_NTP * rep * gene_dosage
            dy[i_m] = tx - gene.delta_m * mrna
            total_tx += tx
            
            # Translation
            tl = gene.k_tl_init * mrna * f_ribo * f_AA * ATP/(0.5+ATP)
            dy[i_p] = tl - gene.delta_p * prot
            total_tl += tl
        
        # Metabolite consumption
        dy[self.met_idx['AA']] -= total_tl * 0.01
        dy[self.met_idx['ATP']] -= total_tx * 0.005 + total_tl * 0.02
        dy[self.met_idx['GTP']] -= total_tl * 0.02
        
        # ===== UPTAKE =====
        v_aa = 0.12 * self.AA_ext / (0.5 + self.AA_ext)
        dy[self.met_idx['AA']] += v_aa
        
        v_ntp = 0.025 * ATP / (1.0 + ATP)
        dy[self.met_idx['UTP']] += 0.5 * v_ntp
        dy[self.met_idx['CTP']] += 0.5 * v_ntp
        
        v_lac = 0.2 * Lac / (2.0 + Lac)
        dy[self.met_idx['Lac']] -= v_lac
        
        # ===== HOMEOSTATIC BUFFERING =====
        for met in self.met_states:
            if met in self.met_targets:
                target = self.met_targets[met]
                current = y[self.met_idx[met]]
                tau = self.tau_buffer[met]
                dy[self.met_idx[met]] += (target - current) / tau
        
        # ===== GROWTH =====
        # Growth rate from ribosome content
        mu = 0.012 * f_ribo * f_AA * ATP/(1.0 + ATP)
        
        # Cell mass increases
        dy[self.cc_idx['cell_mass']] = mu * cell_mass
        
        # Dilution (except cell cycle variables)
        for i in range(self.n_met):
            dy[i] -= mu * y[i]
        for i in range(self.n_met + self.n_cc, self.n_states):
            dy[i] -= mu * y[i]
        
        return dy
    
    def handle_division(self, y: np.ndarray, t: float) -> np.ndarray:
        """Handle cell division event."""
        
        self.division_count += 1
        self.division_times.append(t)
        
        # Partition to daughter cell (we follow one daughter)
        y_new = y.copy()
        
        # Halve metabolites
        for i in range(self.n_met):
            y_new[i] = y[i] / 2
        
        # Reset cell cycle
        y_new[self.cc_idx['DNA_content']] = 1.0
        y_new[self.cc_idx['replication_progress']] = 0.0
        y_new[self.cc_idx['division_progress']] = 0.0
        y_new[self.cc_idx['cell_mass']] = y[self.cc_idx['cell_mass']] / 2
        
        # Halve proteins (stochastic partitioning could be added)
        for i in range(self.n_met + self.n_cc, self.n_states):
            y_new[i] = y[i] / 2
        
        return y_new
    
    def simulate(self, t_end: float = 300, max_divisions: int = 5) -> dict:
        """Simulate with division events."""
        
        print(f"\nSimulating 0 → {t_end} min (max {max_divisions} divisions)...")
        
        y = self.get_initial_state()
        t = 0
        dt_check = 0.5  # Check for division every 0.5 min
        
        # Storage
        times = [0]
        states = [y.copy()]
        division_events = []
        
        while t < t_end and self.division_count < max_divisions:
            # Integrate one step
            t_next = min(t + dt_check, t_end)
            
            sol = solve_ivp(
                self.deriv, (t, t_next), y,
                method='LSODA', rtol=1e-6, atol=1e-9,
                dense_output=True
            )
            
            y = sol.y[:, -1]
            t = t_next
            
            # Check for division
            div_progress = y[self.cc_idx['division_progress']]
            
            if div_progress >= 1.0:
                print(f"  ⚡ DIVISION at t={t:.1f} min (division #{self.division_count + 1})")
                division_events.append(t)
                y = self.handle_division(y, t)
            
            times.append(t)
            states.append(y.copy())
        
        # Convert to arrays
        times = np.array(times)
        states = np.array(states).T
        
        # Build result dict
        result = {'t': times, 'division_events': division_events}
        
        for met in self.met_states:
            result[met] = states[self.met_idx[met], :]
        
        for cc in self.cc_states:
            result[cc] = states[self.cc_idx[cc], :]
        
        result['mrna'] = {g: states[self.mrna_idx[g], :] for g in self.gene_list}
        result['protein'] = {g: states[self.prot_idx[g], :] for g in self.gene_list}
        
        result['EC'] = (result['ATP'] + 0.5*result['ADP']) / \
                       (result['ATP'] + result['ADP'] + result['AMP'] + 1e-12)
        
        return result
    
    def analyze(self, r: dict):
        """Analyze cell cycle results."""
        
        n_div = len(r['division_events'])
        
        print(f"""
{'█'*72}
{'█'*15}  DIVIDING CELL REPORT  {'█'*16}
{'█'*72}

  Simulation time: {r['t'][-1]:.0f} min
  Divisions completed: {n_div}
""")
        
        if n_div > 1:
            intervals = np.diff(r['division_events'])
            mean_cycle = np.mean(intervals)
            print(f"  Cell cycle times: {intervals}")
            print(f"  Mean cycle time: {mean_cycle:.1f} min")
        
        # Final state
        ATP = r['ATP'][-1]
        GTP = r['GTP'][-1]
        EC = r['EC'][-1]
        DNA = r['DNA_content'][-1]
        
        print(f"""
{'═'*72}
                        CELL STATE
{'═'*72}

  DNA content:    {DNA:.2f} (1=unreplicated, 2=replicated)
  Cell mass:      {r['cell_mass'][-1]:.2f}
  Rep progress:   {r['replication_progress'][-1]:.2f}
  Div progress:   {r['division_progress'][-1]:.2f}

  Energy charge:  {EC:.3f}
  ATP: {ATP:.2f} mM    GTP: {GTP:.2f} mM
  dNTP pool: {r['dNTP_pool'][-1]:.3f} mM
""")
        
        # Key proteins
        print(f"  Key proteins:")
        for gene in ['dnaA', 'dnaE', 'ftsZ']:
            if gene in r['protein']:
                print(f"    {gene}: {r['protein'][gene][-1]:.0f} copies")
        
        if n_div > 0:
            print(f"""
{'═'*72}
{'█'*72}
{'█'*15}  🎉 CELL DIVISION ACHIEVED! 🎉  {'█'*14}
{'█'*72}
{'═'*72}

  The cell completed {n_div} division(s)!
  
  Cell cycle is controlled by:
    • DnaA-ATP → triggers replication initiation
    • dNTPs → fuel replication elongation  
    • FtsZ → triggers division after replication
  
  This is EMERGENT cell cycle control!

{'═'*72}
""")
        
        return {'divisions': n_div, 'EC': EC, 'DNA': DNA}


def main():
    print("="*72)
    print("    DARK MANIFOLD V51: DIVIDING MINIMAL CELL")
    print("="*72)
    
    model = DividingMinimalCell(n_genes=60)
    
    # Simulate long enough for multiple divisions
    result = model.simulate(t_end=400, max_divisions=4)
    
    analysis = model.analyze(result)
    
    # Plot
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('JCVI-syn3A Dividing Minimal Cell\nV51: DNA Replication & Cell Division', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    t = result['t']
    div_events = result['division_events']
    
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. DNA content
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, result['DNA_content'], 'b-', lw=2.5)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax.axhline(2.0, color='gray', ls='--', alpha=0.5)
    for td in div_events:
        ax.axvline(td, color='red', ls=':', alpha=0.7)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('DNA content')
    ax.set_title('DNA Replication', fontweight='bold')
    ax.set_ylim([0.5, 2.5])
    ax.grid(alpha=0.3)
    
    # 2. Replication progress
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, result['replication_progress'], 'g-', lw=2.5)
    for td in div_events:
        ax.axvline(td, color='red', ls=':', alpha=0.7)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Progress (0-1)')
    ax.set_title('Replication Progress', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 3. Division progress
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, result['division_progress'], 'r-', lw=2.5)
    for td in div_events:
        ax.axvline(td, color='red', ls=':', alpha=0.7)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Progress (0-1)')
    ax.set_title('Division Progress', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 4. Cell mass
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, result['cell_mass'], 'purple', lw=2.5)
    for td in div_events:
        ax.axvline(td, color='red', ls=':', alpha=0.7)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Relative mass')
    ax.set_title('Cell Mass', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 5. dNTP pool
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, result['dNTP_pool'], 'orange', lw=2.5)
    for td in div_events:
        ax.axvline(td, color='red', ls=':', alpha=0.7)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('dNTP Pool (replication fuel)', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 6. Energy
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, result['ATP'], 'b-', lw=2.5, label='ATP')
    ax.plot(t, result['GTP'], 'g-', lw=2, label='GTP')
    for td in div_events:
        ax.axvline(td, color='red', ls=':', alpha=0.7)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Energy Nucleotides', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 7. DnaA (replication initiator)
    ax = fig.add_subplot(gs[2, 0])
    if 'dnaA' in result['protein']:
        ax.plot(t, result['protein']['dnaA'], 'b-', lw=2.5)
    ax.axhline(model.cc.DnaA_threshold, color='r', ls='--', alpha=0.7, label='Threshold')
    for td in div_events:
        ax.axvline(td, color='red', ls=':', alpha=0.7)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Protein copies')
    ax.set_title('DnaA (initiator)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 8. FtsZ (division)
    ax = fig.add_subplot(gs[2, 1])
    if 'ftsZ' in result['protein']:
        ax.plot(t, result['protein']['ftsZ'], 'r-', lw=2.5)
    ax.axhline(model.cc.FtsZ_threshold, color='r', ls='--', alpha=0.7, label='Threshold')
    for td in div_events:
        ax.axvline(td, color='red', ls=':', alpha=0.7)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Protein copies')
    ax.set_title('FtsZ (division)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 9. Energy charge
    ax = fig.add_subplot(gs[2, 2])
    ax.fill_between(t, 0.85, 0.95, alpha=0.15, color='green')
    ax.plot(t, result['EC'], 'k-', lw=2.5)
    for td in div_events:
        ax.axvline(td, color='red', ls=':', alpha=0.7)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Energy Charge')
    ax.set_title('Energy Charge', fontweight='bold')
    ax.set_ylim([0.6, 1.0])
    ax.grid(alpha=0.3)
    
    plt.savefig('dividing_cell.png', dpi=200, bbox_inches='tight')
    print("\n✓ Saved: dividing_cell.png")
    
    return model, result, analysis


if __name__ == '__main__':
    model, result, analysis = main()
