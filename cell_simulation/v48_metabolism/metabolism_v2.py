"""
Dark Manifold V48b: Balanced Metabolism
=======================================

Fixed issues:
1. ATP production/consumption balance
2. Proper glycolysis stoichiometry
3. Homeostatic feedback
4. Realistic consumer rates tied to cell state
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ============================================================================
# CONSTANTS
# ============================================================================

CELL_VOLUME = 5e-17  # L
AVOGADRO = 6.022e23

def mM_to_molecules(conc_mM):
    return conc_mM * 1e-3 * CELL_VOLUME * AVOGADRO

def molecules_to_mM(n):
    return (n / AVOGADRO) / CELL_VOLUME * 1000


# ============================================================================
# SIMPLIFIED BALANCED MODEL
# ============================================================================

class BalancedMetabolism:
    """
    Metabolic model with proper ATP balance.
    
    Key insight: ATP production must equal consumption at steady state.
    
    Production:
    - Glycolysis: Glucose → 2 ATP + 2 Pyruvate
    
    Consumption:
    - Protein synthesis: ~4 ATP per amino acid
    - RNA synthesis: ~2 ATP per nucleotide (NTP + polymerization)
    - DNA synthesis: ~2 ATP per nucleotide
    - Membrane/transport: ~10% of total
    - Maintenance: ~5% of total
    
    The trick: Scale consumption to match production capacity
    """
    
    def __init__(self):
        # ====== ENERGY POOLS ======
        self.ATP = 3.0          # mM
        self.ADP = 0.5          # mM
        self.AMP = 0.1          # mM
        self.Pi = 5.0           # mM
        
        self.GTP = 1.0          # mM
        self.GDP = 0.2          # mM
        
        # ====== REDOX POOLS ======
        self.NAD = 1.0          # mM
        self.NADH = 0.1         # mM
        
        # ====== NUCLEOTIDES ======
        self.NTP_pool = 3.0     # mM (ATP+GTP+CTP+UTP for RNA)
        self.dNTP_pool = 0.2    # mM (for DNA)
        
        # ====== AMINO ACIDS ======
        self.AA_pool = 10.0     # mM (total of all 20)
        
        # ====== GLYCOLYSIS ======
        self.glucose_ext = 10.0 # mM (external, constant)
        self.pyruvate = 0.5     # mM
        self.lactate = 0.1      # mM
        
        # ====== KINETIC PARAMETERS ======
        
        # Glycolysis: Glucose → 2 Pyruvate + 2 ATP + 2 NADH
        self.v_glycolysis_max = 0.5     # mM glucose/min
        self.Km_glucose = 0.5           # mM
        self.Km_ADP_glyc = 0.2          # mM
        self.Km_NAD = 0.2               # mM
        
        # Lactate dehydrogenase: Pyruvate + NADH → Lactate + NAD+
        self.v_ldh_max = 1.0            # mM/min
        self.Km_pyruvate = 0.5          # mM
        self.Km_NADH = 0.05             # mM
        
        # ATP synthase (if present) - runs in reverse in Mycoplasma
        # We'll ignore this for now (no ox phos)
        
        # ====== CONSUMER RATES ======
        # These are set to match glycolytic capacity at steady state
        
        # Protein synthesis: ~70-80% of ATP
        self.v_protein_max = 0.2        # mM AA/min → uses 0.8 mM ATP/min
        self.Km_AA = 1.0                # mM
        self.Km_ATP_protein = 0.5       # mM
        self.Km_GTP_protein = 0.3       # mM
        
        # RNA synthesis: ~10% of ATP
        self.v_rna_max = 0.05           # mM NTP/min
        self.Km_NTP = 0.5               # mM
        self.Km_ATP_rna = 0.5           # mM
        
        # DNA synthesis: ~2% (only during replication)
        self.v_dna_max = 0.01           # mM dNTP/min
        self.replicating = False        # Toggle
        
        # Maintenance: ~5-10%
        self.v_maintenance = 0.05       # mM ATP/min (basal)
        
        # ====== BIOSYNTHESIS (replenishment) ======
        
        # Amino acid uptake
        self.v_aa_uptake_max = 0.3      # mM/min
        self.AA_ext = 1.0               # mM (external)
        self.Km_AA_ext = 0.1            # mM
        
        # Nucleotide synthesis
        self.v_ntp_synth_max = 0.1      # mM/min
        self.Km_ribose5P = 0.1          # mM
        self.ribose5P = 0.2             # mM (from PPP)
        
        # State names for ODE
        self.state_names = [
            'ATP', 'ADP', 'AMP', 'Pi',
            'GTP', 'GDP',
            'NAD', 'NADH',
            'NTP_pool', 'dNTP_pool',
            'AA_pool',
            'pyruvate', 'lactate',
            'ribose5P',
            'protein_made', 'rna_made', 'dna_made'
        ]
        self.n_states = len(self.state_names)
        self.idx = {name: i for i, name in enumerate(self.state_names)}
        
        self._print_summary()
    
    def _print_summary(self):
        print(f"\n{'='*70}")
        print("BALANCED METABOLISM MODEL")
        print("="*70)
        print(f"States: {self.n_states}")
        print(f"\nInitial ATP/ADP ratio: {self.ATP/self.ADP:.1f}")
        print(f"Initial energy charge: {(self.ATP + 0.5*self.ADP)/(self.ATP+self.ADP+self.AMP):.2f}")
        
        # Estimate fluxes
        print("\nEstimated fluxes at initial conditions:")
        glyc = self.v_glycolysis_max * (self.glucose_ext/(self.Km_glucose + self.glucose_ext)) * \
               (self.ADP/(self.Km_ADP_glyc + self.ADP)) * (self.NAD/(self.Km_NAD + self.NAD))
        atp_prod = glyc * 2  # 2 ATP per glucose
        
        print(f"  Glycolysis: {glyc:.3f} mM glucose/min → {atp_prod:.3f} mM ATP/min")
        
        prot = self.v_protein_max * (self.AA_pool/(self.Km_AA + self.AA_pool)) * \
               (self.ATP/(self.Km_ATP_protein + self.ATP))
        atp_prot = prot * 4  # 4 ATP per AA
        print(f"  Protein synthesis: {prot:.3f} mM AA/min → {atp_prot:.3f} mM ATP/min")
        
        rna = self.v_rna_max * (self.NTP_pool/(self.Km_NTP + self.NTP_pool))
        atp_rna = rna * 2
        print(f"  RNA synthesis: {rna:.3f} mM NTP/min → {atp_rna:.3f} mM ATP/min")
        
        atp_maint = self.v_maintenance
        print(f"  Maintenance: {atp_maint:.3f} mM ATP/min")
        
        total_consumption = atp_prot + atp_rna + atp_maint
        print(f"\nTotal ATP production: {atp_prod:.3f} mM/min")
        print(f"Total ATP consumption: {total_consumption:.3f} mM/min")
        print(f"Balance: {atp_prod - total_consumption:+.3f} mM/min")
    
    def get_initial_state(self) -> np.ndarray:
        y0 = np.zeros(self.n_states)
        y0[self.idx['ATP']] = self.ATP
        y0[self.idx['ADP']] = self.ADP
        y0[self.idx['AMP']] = self.AMP
        y0[self.idx['Pi']] = self.Pi
        y0[self.idx['GTP']] = self.GTP
        y0[self.idx['GDP']] = self.GDP
        y0[self.idx['NAD']] = self.NAD
        y0[self.idx['NADH']] = self.NADH
        y0[self.idx['NTP_pool']] = self.NTP_pool
        y0[self.idx['dNTP_pool']] = self.dNTP_pool
        y0[self.idx['AA_pool']] = self.AA_pool
        y0[self.idx['pyruvate']] = self.pyruvate
        y0[self.idx['lactate']] = self.lactate
        y0[self.idx['ribose5P']] = self.ribose5P
        return y0
    
    def ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Calculate derivatives."""
        
        dydt = np.zeros_like(y)
        
        # Unpack state (with safety)
        ATP = max(y[self.idx['ATP']], 1e-6)
        ADP = max(y[self.idx['ADP']], 1e-6)
        AMP = max(y[self.idx['AMP']], 1e-6)
        Pi = max(y[self.idx['Pi']], 1e-6)
        GTP = max(y[self.idx['GTP']], 1e-6)
        GDP = max(y[self.idx['GDP']], 1e-6)
        NAD = max(y[self.idx['NAD']], 1e-6)
        NADH = max(y[self.idx['NADH']], 1e-6)
        NTP_pool = max(y[self.idx['NTP_pool']], 1e-6)
        dNTP_pool = max(y[self.idx['dNTP_pool']], 1e-6)
        AA_pool = max(y[self.idx['AA_pool']], 1e-6)
        pyruvate = max(y[self.idx['pyruvate']], 1e-6)
        lactate = max(y[self.idx['lactate']], 1e-6)
        ribose5P = max(y[self.idx['ribose5P']], 1e-6)
        
        # ====== GLYCOLYSIS ======
        # Glucose + 2 NAD+ + 2 ADP + 2 Pi → 2 Pyruvate + 2 NADH + 2 ATP
        v_glyc = self.v_glycolysis_max * \
                 (self.glucose_ext / (self.Km_glucose + self.glucose_ext)) * \
                 (ADP / (self.Km_ADP_glyc + ADP)) * \
                 (NAD / (self.Km_NAD + NAD))
        
        # Products: 2 pyruvate, 2 NADH, 2 ATP per glucose
        dydt[self.idx['pyruvate']] += 2 * v_glyc
        dydt[self.idx['NADH']] += 2 * v_glyc
        dydt[self.idx['NAD']] -= 2 * v_glyc
        dydt[self.idx['ATP']] += 2 * v_glyc
        dydt[self.idx['ADP']] -= 2 * v_glyc
        dydt[self.idx['Pi']] -= 2 * v_glyc
        
        # ====== LACTATE DEHYDROGENASE ======
        # Pyruvate + NADH → Lactate + NAD+ (regenerates NAD+)
        v_ldh = self.v_ldh_max * \
                (pyruvate / (self.Km_pyruvate + pyruvate)) * \
                (NADH / (self.Km_NADH + NADH))
        
        dydt[self.idx['pyruvate']] -= v_ldh
        dydt[self.idx['NADH']] -= v_ldh
        dydt[self.idx['lactate']] += v_ldh
        dydt[self.idx['NAD']] += v_ldh
        
        # ====== PROTEIN SYNTHESIS ======
        # AA + 2 ATP + 2 GTP → Protein + 2 ADP + 2 GDP + 4 Pi
        v_protein = self.v_protein_max * \
                    (AA_pool / (self.Km_AA + AA_pool)) * \
                    (ATP / (self.Km_ATP_protein + ATP)) * \
                    (GTP / (self.Km_GTP_protein + GTP))
        
        dydt[self.idx['AA_pool']] -= v_protein
        dydt[self.idx['ATP']] -= 2 * v_protein
        dydt[self.idx['ADP']] += 2 * v_protein
        dydt[self.idx['GTP']] -= 2 * v_protein
        dydt[self.idx['GDP']] += 2 * v_protein
        dydt[self.idx['Pi']] += 4 * v_protein
        dydt[self.idx['protein_made']] += v_protein
        
        # ====== RNA SYNTHESIS ======
        # NTP → NMP_incorporated + PPi (simplified)
        v_rna = self.v_rna_max * \
                (NTP_pool / (self.Km_NTP + NTP_pool)) * \
                (ATP / (0.5 + ATP))  # Energy-dependent
        
        dydt[self.idx['NTP_pool']] -= v_rna
        dydt[self.idx['rna_made']] += v_rna
        # PPi → 2 Pi (fast pyrophosphatase)
        dydt[self.idx['Pi']] += 2 * v_rna
        
        # ====== DNA SYNTHESIS (if replicating) ======
        if self.replicating:
            v_dna = self.v_dna_max * \
                    (dNTP_pool / (0.05 + dNTP_pool)) * \
                    (ATP / (0.5 + ATP))
            
            dydt[self.idx['dNTP_pool']] -= v_dna
            dydt[self.idx['dna_made']] += v_dna
            dydt[self.idx['Pi']] += 2 * v_dna
        
        # ====== MAINTENANCE ATP ======
        # Basal ATP consumption (membrane potential, etc.)
        v_maint = self.v_maintenance * (ATP / (0.5 + ATP))
        dydt[self.idx['ATP']] -= v_maint
        dydt[self.idx['ADP']] += v_maint
        dydt[self.idx['Pi']] += v_maint
        
        # ====== ADENYLATE KINASE ======
        # 2 ADP ⟷ ATP + AMP (fast equilibration, Keq ≈ 1)
        # This maintains adenylate pool balance
        v_adk = 5.0 * (ADP * ADP - ATP * AMP)  # Fast
        dydt[self.idx['ATP']] += v_adk
        dydt[self.idx['AMP']] += v_adk
        dydt[self.idx['ADP']] -= 2 * v_adk
        
        # ====== NDP KINASE ======
        # GDP + ATP ⟷ GTP + ADP (fast)
        v_ndk = 5.0 * (GDP * ATP - GTP * ADP)
        dydt[self.idx['GTP']] += v_ndk
        dydt[self.idx['ADP']] += v_ndk
        dydt[self.idx['GDP']] -= v_ndk
        dydt[self.idx['ATP']] -= v_ndk
        
        # ====== AMINO ACID UPTAKE ======
        # External AA → Internal AA (ATP-dependent)
        v_aa_uptake = self.v_aa_uptake_max * \
                      (self.AA_ext / (self.Km_AA_ext + self.AA_ext)) * \
                      (ATP / (0.5 + ATP))
        
        dydt[self.idx['AA_pool']] += v_aa_uptake
        dydt[self.idx['ATP']] -= v_aa_uptake * 0.1  # Some cost
        dydt[self.idx['ADP']] += v_aa_uptake * 0.1
        
        # ====== NTP SYNTHESIS ======
        # Ribose-5-P + ATP → NTPs (simplified)
        v_ntp_synth = self.v_ntp_synth_max * \
                      (ribose5P / (self.Km_ribose5P + ribose5P)) * \
                      (ATP / (0.5 + ATP))
        
        dydt[self.idx['NTP_pool']] += v_ntp_synth
        dydt[self.idx['ribose5P']] -= v_ntp_synth
        dydt[self.idx['ATP']] -= 2 * v_ntp_synth  # Costs ATP
        dydt[self.idx['ADP']] += 2 * v_ntp_synth
        
        # ====== PENTOSE PHOSPHATE PATHWAY ======
        # Glucose-6-P → Ribose-5-P + CO2 (simplified, constitutive)
        v_ppp = 0.05 * (self.glucose_ext / (1.0 + self.glucose_ext))
        dydt[self.idx['ribose5P']] += v_ppp
        
        return dydt
    
    def simulate(self, t_span: Tuple[float, float], 
                 t_eval: Optional[np.ndarray] = None) -> dict:
        """Run simulation."""
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)
        
        y0 = self.get_initial_state()
        
        print(f"\nSimulating t={t_span[0]} to t={t_span[1]} min...")
        
        solution = solve_ivp(
            self.ode_rhs,
            t_span,
            y0,
            t_eval=t_eval,
            method='LSODA',
            rtol=1e-8,
            atol=1e-10,
        )
        
        result = {'t': solution.t}
        for i, name in enumerate(self.state_names):
            result[name] = solution.y[i, :]
        
        return result
    
    def analyze(self, result: dict):
        """Analyze simulation results."""
        
        t = result['t']
        
        print(f"\n{'='*70}")
        print("METABOLIC STATE")
        print("="*70)
        
        # Energy state
        ATP = result['ATP'][-1]
        ADP = result['ADP'][-1]
        AMP = result['AMP'][-1]
        total_A = ATP + ADP + AMP
        energy_charge = (ATP + 0.5*ADP) / total_A
        
        print(f"\nEnergy status (t = {t[-1]:.1f} min):")
        print(f"  ATP: {ATP:.3f} mM")
        print(f"  ADP: {ADP:.3f} mM")
        print(f"  AMP: {AMP:.3f} mM")
        print(f"  ATP/ADP ratio: {ATP/ADP:.1f}")
        print(f"  Energy charge: {energy_charge:.3f} (healthy: 0.8-0.95)")
        
        # GTP
        GTP = result['GTP'][-1]
        GDP = result['GDP'][-1]
        print(f"\n  GTP: {GTP:.3f} mM")
        print(f"  GDP: {GDP:.3f} mM")
        print(f"  GTP/GDP ratio: {GTP/GDP:.1f}")
        
        # Redox
        NAD = result['NAD'][-1]
        NADH = result['NADH'][-1]
        print(f"\n  NAD+: {NAD:.3f} mM")
        print(f"  NADH: {NADH:.3f} mM")
        print(f"  NAD+/NADH ratio: {NAD/NADH:.1f} (glycolysis needs >1)")
        
        # Pools
        print(f"\nMetabolite pools:")
        print(f"  Amino acids: {result['AA_pool'][-1]:.2f} mM")
        print(f"  NTP pool: {result['NTP_pool'][-1]:.2f} mM")
        print(f"  dNTP pool: {result['dNTP_pool'][-1]:.3f} mM")
        print(f"  Ribose-5-P: {result['ribose5P'][-1]:.3f} mM")
        
        # Glycolysis
        print(f"\nGlycolysis:")
        print(f"  Pyruvate: {result['pyruvate'][-1]:.3f} mM")
        print(f"  Lactate: {result['lactate'][-1]:.2f} mM")
        
        # Production
        print(f"\nBiosynthesis (cumulative):")
        print(f"  Protein made: {result['protein_made'][-1]:.2f} mM AA incorporated")
        print(f"  RNA made: {result['rna_made'][-1]:.2f} mM NTP incorporated")
        print(f"  DNA made: {result['dna_made'][-1]:.4f} mM dNTP incorporated")
        
        # Stability check
        print(f"\n{'='*70}")
        print("STABILITY CHECK")
        print("="*70)
        
        atp_start = result['ATP'][0]
        atp_end = result['ATP'][-1]
        atp_change = (atp_end - atp_start) / atp_start * 100
        
        if abs(atp_change) < 10:
            print(f"✓ ATP stable ({atp_change:+.1f}%) - Metabolism is balanced!")
        elif atp_change > 0:
            print(f"△ ATP increased ({atp_change:+.1f}%) - Production > consumption")
        else:
            print(f"▽ ATP decreased ({atp_change:+.1f}%) - Consumption > production")
        
        # Health assessment
        if energy_charge > 0.85:
            print("✓ Energy charge healthy (>0.85)")
        elif energy_charge > 0.7:
            print("△ Energy charge moderate (0.7-0.85)")
        else:
            print("✗ Energy charge low (<0.7) - Cell stressed!")
        
        return {
            'ATP': ATP, 'ADP': ADP, 'AMP': AMP,
            'energy_charge': energy_charge,
            'GTP': GTP, 'GDP': GDP,
            'NAD': NAD, 'NADH': NADH,
            'AA_pool': result['AA_pool'][-1],
            'NTP_pool': result['NTP_pool'][-1],
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("DARK MANIFOLD V48b: BALANCED METABOLISM")
    print("="*70)
    
    model = BalancedMetabolism()
    
    # Run for 60 minutes (should reach steady state)
    result = model.simulate(t_span=(0, 60))
    
    # Analyze
    state = model.analyze(result)
    
    # Plot key metabolites
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    t = result['t']
    
    # ATP/ADP
    ax = axes[0, 0]
    ax.plot(t, result['ATP'], 'b-', label='ATP', linewidth=2)
    ax.plot(t, result['ADP'], 'r-', label='ADP', linewidth=2)
    ax.plot(t, result['AMP'], 'g--', label='AMP', linewidth=1)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Adenine Nucleotides')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy charge
    ax = axes[0, 1]
    total_A = result['ATP'] + result['ADP'] + result['AMP']
    energy_charge = (result['ATP'] + 0.5*result['ADP']) / total_A
    ax.plot(t, energy_charge, 'k-', linewidth=2)
    ax.axhline(y=0.85, color='g', linestyle='--', label='Healthy threshold')
    ax.axhline(y=0.7, color='r', linestyle='--', label='Stress threshold')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Energy Charge')
    ax.set_title('Cellular Energy Charge')
    ax.set_ylim([0.5, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Metabolite pools
    ax = axes[1, 0]
    ax.plot(t, result['AA_pool'], 'b-', label='Amino acids', linewidth=2)
    ax.plot(t, result['NTP_pool'], 'r-', label='NTPs', linewidth=2)
    ax.plot(t, result['dNTP_pool']*10, 'g-', label='dNTPs (×10)', linewidth=2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Precursor Pools')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Biosynthesis
    ax = axes[1, 1]
    ax.plot(t, result['protein_made'], 'b-', label='Protein', linewidth=2)
    ax.plot(t, result['rna_made'], 'r-', label='RNA', linewidth=2)
    ax.plot(t, result['dna_made']*100, 'g-', label='DNA (×100)', linewidth=2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Cumulative (mM)')
    ax.set_title('Biosynthesis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metabolism_dynamics.png', dpi=150)
    print("\nSaved: metabolism_dynamics.png")
    
    return model, result, state


if __name__ == '__main__':
    model, result, state = main()
