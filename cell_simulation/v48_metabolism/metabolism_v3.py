"""
Dark Manifold V48c: Homeostatic Metabolism
==========================================

Key fix: Production and consumption must self-balance through feedback.

The cell achieves homeostasis through:
1. ATP inhibits glycolysis (product inhibition)
2. ADP stimulates glycolysis (substrate for ATP synthesis)
3. Low ATP slows biosynthesis (ATP is substrate)
4. These feedbacks automatically balance the system
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class HomeostaticMetabolism:
    """
    Self-balancing metabolic model.
    
    Key: Let the cell find its own steady state through feedback.
    """
    
    def __init__(self, growth_rate: float = 0.01):
        """
        Initialize metabolism.
        
        Args:
            growth_rate: Cell growth rate (1/min), ~0.01 = 70 min doubling
        """
        self.growth_rate = growth_rate  # Dilution rate
        
        # ===== STATE VARIABLES (concentrations in mM) =====
        self.state_names = [
            'ATP', 'ADP', 'AMP',          # Adenine nucleotides
            'GTP', 'GDP',                  # Guanine nucleotides
            'NAD', 'NADH',                 # Redox cofactors
            'NTP_pool',                    # Ribonucleotides for RNA
            'dNTP_pool',                   # Deoxyribonucleotides for DNA
            'AA_pool',                     # Amino acid pool
            'glucose_int',                 # Internal glucose
            'pyruvate',                    # Pyruvate
            'lactate',                     # Lactate
            'Pi',                          # Inorganic phosphate
        ]
        self.n_states = len(self.state_names)
        self.idx = {name: i for i, name in enumerate(self.state_names)}
        
        # ===== INITIAL CONDITIONS =====
        self.initial = {
            'ATP': 3.0,
            'ADP': 0.5,
            'AMP': 0.1,
            'GTP': 1.0,
            'GDP': 0.2,
            'NAD': 1.0,
            'NADH': 0.1,
            'NTP_pool': 3.0,
            'dNTP_pool': 0.2,
            'AA_pool': 5.0,
            'glucose_int': 1.0,
            'pyruvate': 0.5,
            'lactate': 1.0,
            'Pi': 10.0,
        }
        
        # ===== EXTERNAL CONCENTRATIONS (constant) =====
        self.glucose_ext = 20.0  # mM
        self.AA_ext = 2.0        # mM (each amino acid)
        
        # ===== KINETIC PARAMETERS =====
        
        # Glucose uptake
        self.V_glc_uptake = 1.0   # mM/min
        self.Km_glc_ext = 1.0     # mM
        
        # Glycolysis (overall: Glucose → 2 Pyruvate + 2 ATP + 2 NADH)
        self.V_glycolysis = 2.0   # mM glucose/min (max)
        self.Km_glc_int = 0.5     # mM
        self.Km_NAD_glyc = 0.2    # mM
        self.Km_ADP_glyc = 0.1    # mM
        self.Ki_ATP_glyc = 2.0    # mM (ATP inhibits glycolysis!)
        
        # Lactate dehydrogenase (Pyruvate + NADH → Lactate + NAD+)
        self.V_ldh = 5.0          # mM/min (fast)
        self.Km_pyr = 0.5         # mM
        self.Km_NADH = 0.05       # mM
        
        # Protein synthesis (THE MAJOR ATP CONSUMER)
        # 4 ATP equivalents per amino acid (2 ATP + 2 GTP)
        self.V_protein = 0.3      # mM AA/min
        self.Km_AA = 1.0          # mM
        self.Km_ATP_prot = 1.0    # mM
        self.Km_GTP_prot = 0.5    # mM
        
        # RNA synthesis
        self.V_rna = 0.1          # mM NTP/min
        self.Km_NTP = 1.0         # mM
        self.Km_ATP_rna = 0.5     # mM
        
        # DNA synthesis (only during replication, ~0 normally)
        self.V_dna = 0.005        # mM dNTP/min (continuous low level)
        self.Km_dNTP = 0.1        # mM
        
        # Maintenance ATP consumption
        self.V_maint = 0.1        # mM ATP/min
        
        # Amino acid uptake
        self.V_aa_uptake = 0.5    # mM/min
        self.Km_AA_ext = 0.5      # mM
        
        # NTP synthesis from ATP + precursors
        self.V_ntp_synth = 0.2    # mM/min
        
        # dNTP synthesis
        self.V_dntp_synth = 0.01  # mM/min
        
        # Adenylate kinase: 2 ADP ⟷ ATP + AMP (Keq ≈ 1)
        self.k_adk = 10.0         # Fast equilibration
        
        # NDP kinase: NDP + ATP ⟷ NTP + ADP
        self.k_ndk = 10.0         # Fast equilibration
        
    def get_initial_state(self) -> np.ndarray:
        y0 = np.zeros(self.n_states)
        for name, val in self.initial.items():
            y0[self.idx[name]] = val
        return y0
    
    def energy_charge(self, ATP, ADP, AMP) -> float:
        """Calculate adenylate energy charge."""
        total = ATP + ADP + AMP
        if total > 0:
            return (ATP + 0.5 * ADP) / total
        return 0
    
    def ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Calculate time derivatives."""
        
        dydt = np.zeros(self.n_states)
        
        # Unpack state (ensure non-negative)
        ATP = max(y[self.idx['ATP']], 1e-9)
        ADP = max(y[self.idx['ADP']], 1e-9)
        AMP = max(y[self.idx['AMP']], 1e-9)
        GTP = max(y[self.idx['GTP']], 1e-9)
        GDP = max(y[self.idx['GDP']], 1e-9)
        NAD = max(y[self.idx['NAD']], 1e-9)
        NADH = max(y[self.idx['NADH']], 1e-9)
        NTP = max(y[self.idx['NTP_pool']], 1e-9)
        dNTP = max(y[self.idx['dNTP_pool']], 1e-9)
        AA = max(y[self.idx['AA_pool']], 1e-9)
        glc = max(y[self.idx['glucose_int']], 1e-9)
        pyr = max(y[self.idx['pyruvate']], 1e-9)
        lac = max(y[self.idx['lactate']], 1e-9)
        Pi = max(y[self.idx['Pi']], 1e-9)
        
        # ========== GLUCOSE UPTAKE ==========
        # External glucose → Internal glucose
        v_uptake = self.V_glc_uptake * (self.glucose_ext / (self.Km_glc_ext + self.glucose_ext))
        dydt[self.idx['glucose_int']] += v_uptake
        
        # ========== GLYCOLYSIS ==========
        # Glucose + 2 NAD+ + 2 ADP → 2 Pyruvate + 2 NADH + 2 ATP
        # KEY: ATP inhibits (product inhibition), ADP activates
        v_glyc = self.V_glycolysis * \
                 (glc / (self.Km_glc_int + glc)) * \
                 (NAD / (self.Km_NAD_glyc + NAD)) * \
                 (ADP / (self.Km_ADP_glyc + ADP)) * \
                 (self.Ki_ATP_glyc / (self.Ki_ATP_glyc + ATP))  # ATP inhibition!
        
        dydt[self.idx['glucose_int']] -= v_glyc
        dydt[self.idx['NAD']] -= 2 * v_glyc
        dydt[self.idx['NADH']] += 2 * v_glyc
        dydt[self.idx['ADP']] -= 2 * v_glyc
        dydt[self.idx['ATP']] += 2 * v_glyc
        dydt[self.idx['pyruvate']] += 2 * v_glyc
        dydt[self.idx['Pi']] -= 2 * v_glyc
        
        # ========== LACTATE DEHYDROGENASE ==========
        # Pyruvate + NADH → Lactate + NAD+ (regenerates NAD+!)
        v_ldh = self.V_ldh * \
                (pyr / (self.Km_pyr + pyr)) * \
                (NADH / (self.Km_NADH + NADH))
        
        dydt[self.idx['pyruvate']] -= v_ldh
        dydt[self.idx['NADH']] -= v_ldh
        dydt[self.idx['lactate']] += v_ldh
        dydt[self.idx['NAD']] += v_ldh
        
        # ========== PROTEIN SYNTHESIS ==========
        # AA + 2 ATP + 2 GTP → Protein + 2 ADP + 2 GDP + 4 Pi
        v_prot = self.V_protein * \
                 (AA / (self.Km_AA + AA)) * \
                 (ATP / (self.Km_ATP_prot + ATP)) * \
                 (GTP / (self.Km_GTP_prot + GTP))
        
        dydt[self.idx['AA_pool']] -= v_prot
        dydt[self.idx['ATP']] -= 2 * v_prot
        dydt[self.idx['ADP']] += 2 * v_prot
        dydt[self.idx['GTP']] -= 2 * v_prot
        dydt[self.idx['GDP']] += 2 * v_prot
        dydt[self.idx['Pi']] += 4 * v_prot
        
        # ========== RNA SYNTHESIS ==========
        # NTP → NMP + PPi (uses NTPs, costs polymerization energy)
        v_rna = self.V_rna * \
                (NTP / (self.Km_NTP + NTP)) * \
                (ATP / (self.Km_ATP_rna + ATP))
        
        dydt[self.idx['NTP_pool']] -= v_rna
        dydt[self.idx['Pi']] += 2 * v_rna  # PPi → 2 Pi
        
        # ========== DNA SYNTHESIS ==========
        v_dna = self.V_dna * (dNTP / (self.Km_dNTP + dNTP))
        dydt[self.idx['dNTP_pool']] -= v_dna
        dydt[self.idx['Pi']] += 2 * v_dna
        
        # ========== MAINTENANCE ==========
        # Basal ATP hydrolysis
        v_maint = self.V_maint * (ATP / (0.5 + ATP))
        dydt[self.idx['ATP']] -= v_maint
        dydt[self.idx['ADP']] += v_maint
        dydt[self.idx['Pi']] += v_maint
        
        # ========== ADENYLATE KINASE ==========
        # 2 ADP ⟷ ATP + AMP (fast equilibration)
        # Keq = [ATP][AMP]/[ADP]² ≈ 1
        v_adk = self.k_adk * (ADP * ADP - ATP * AMP)
        dydt[self.idx['ATP']] += v_adk
        dydt[self.idx['AMP']] += v_adk
        dydt[self.idx['ADP']] -= 2 * v_adk
        
        # ========== NDP KINASE ==========
        # GDP + ATP ⟷ GTP + ADP
        v_ndk_gtp = self.k_ndk * (GDP * ATP - GTP * ADP)
        dydt[self.idx['GTP']] += v_ndk_gtp
        dydt[self.idx['ADP']] += v_ndk_gtp
        dydt[self.idx['GDP']] -= v_ndk_gtp
        dydt[self.idx['ATP']] -= v_ndk_gtp
        
        # ========== AMINO ACID UPTAKE ==========
        v_aa = self.V_aa_uptake * (self.AA_ext / (self.Km_AA_ext + self.AA_ext)) * \
               (ATP / (0.5 + ATP))  # Energy-dependent
        dydt[self.idx['AA_pool']] += v_aa
        # Small ATP cost for transport
        dydt[self.idx['ATP']] -= 0.1 * v_aa
        dydt[self.idx['ADP']] += 0.1 * v_aa
        
        # ========== NTP SYNTHESIS ==========
        # Simplified: precursors + ATP → NTPs
        v_ntp = self.V_ntp_synth * (ATP / (1.0 + ATP))
        dydt[self.idx['NTP_pool']] += v_ntp
        dydt[self.idx['ATP']] -= v_ntp  # Costs ATP
        dydt[self.idx['ADP']] += v_ntp
        
        # ========== dNTP SYNTHESIS ==========
        v_dntp = self.V_dntp_synth * (ATP / (0.5 + ATP)) * (NTP / (1.0 + NTP))
        dydt[self.idx['dNTP_pool']] += v_dntp
        
        # ========== DILUTION BY GROWTH ==========
        # All metabolites diluted by cell growth
        for i in range(self.n_states):
            dydt[i] -= self.growth_rate * y[i]
        
        return dydt
    
    def simulate(self, t_span: Tuple[float, float], 
                 t_eval: Optional[np.ndarray] = None) -> dict:
        """Run simulation."""
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        y0 = self.get_initial_state()
        
        print(f"\nSimulating {t_span[0]:.0f} to {t_span[1]:.0f} min...")
        print(f"Growth rate: {self.growth_rate:.4f} /min (doubling time: {np.log(2)/self.growth_rate:.1f} min)")
        
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
        
        # Calculate derived quantities
        result['energy_charge'] = (result['ATP'] + 0.5 * result['ADP']) / \
                                  (result['ATP'] + result['ADP'] + result['AMP'])
        result['ATP_ADP_ratio'] = result['ATP'] / result['ADP']
        
        return result
    
    def analyze(self, result: dict):
        """Analyze results."""
        
        t = result['t']
        
        print(f"\n{'='*70}")
        print(f"RESULTS AT t = {t[-1]:.1f} min")
        print("="*70)
        
        # Energy
        print("\n>>> ENERGY STATUS <<<")
        print(f"  ATP:      {result['ATP'][-1]:.3f} mM")
        print(f"  ADP:      {result['ADP'][-1]:.3f} mM")
        print(f"  AMP:      {result['AMP'][-1]:.3f} mM")
        print(f"  ATP/ADP:  {result['ATP_ADP_ratio'][-1]:.1f}")
        print(f"  Energy charge: {result['energy_charge'][-1]:.3f}")
        
        # Redox
        print("\n>>> REDOX STATUS <<<")
        print(f"  NAD+:     {result['NAD'][-1]:.3f} mM")
        print(f"  NADH:     {result['NADH'][-1]:.3f} mM")
        print(f"  NAD+/NADH: {result['NAD'][-1]/result['NADH'][-1]:.1f}")
        
        # Pools
        print("\n>>> PRECURSOR POOLS <<<")
        print(f"  Amino acids: {result['AA_pool'][-1]:.2f} mM")
        print(f"  NTPs:        {result['NTP_pool'][-1]:.2f} mM")
        print(f"  dNTPs:       {result['dNTP_pool'][-1]:.3f} mM")
        
        # Glycolysis
        print("\n>>> GLYCOLYSIS <<<")
        print(f"  Glucose (int): {result['glucose_int'][-1]:.2f} mM")
        print(f"  Pyruvate:      {result['pyruvate'][-1]:.2f} mM")
        print(f"  Lactate:       {result['lactate'][-1]:.1f} mM")
        
        # Stability
        print(f"\n{'='*70}")
        print("HOMEOSTASIS CHECK")
        print("="*70)
        
        # Check each key metabolite
        checks = ['ATP', 'NAD', 'AA_pool', 'NTP_pool']
        all_stable = True
        
        for met in checks:
            initial = result[met][0]
            final = result[met][-1]
            change = (final - initial) / initial * 100
            
            if abs(change) < 15:
                status = "✓ STABLE"
            else:
                status = "△ CHANGING"
                all_stable = False
            
            print(f"  {met:<12}: {initial:.2f} → {final:.2f} mM ({change:+.1f}%) {status}")
        
        if all_stable:
            print("\n🎉 HOMEOSTASIS ACHIEVED! Cell is in steady state.")
        else:
            print("\n⚠ System still adjusting to steady state.")
        
        # Health assessment
        ec = result['energy_charge'][-1]
        if ec > 0.85:
            print(f"\n✓ Energy charge {ec:.2f} is HEALTHY (>0.85)")
        elif ec > 0.7:
            print(f"\n△ Energy charge {ec:.2f} is MODERATE (0.7-0.85)")
        else:
            print(f"\n✗ Energy charge {ec:.2f} is LOW (<0.7) - STRESSED!")
        
        return result


def main():
    print("="*70)
    print("DARK MANIFOLD V48c: HOMEOSTATIC METABOLISM")
    print("="*70)
    
    # Create model with realistic growth rate
    # Doubling time ~70 min → growth rate = ln(2)/70 ≈ 0.01 /min
    model = HomeostaticMetabolism(growth_rate=0.01)
    
    # Simulate 2 hours (should definitely reach steady state)
    result = model.simulate(t_span=(0, 120))
    
    # Analyze
    model.analyze(result)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t = result['t']
    
    # Energy nucleotides
    ax = axes[0, 0]
    ax.plot(t, result['ATP'], 'b-', lw=2, label='ATP')
    ax.plot(t, result['ADP'], 'r-', lw=2, label='ADP')
    ax.plot(t, result['AMP'], 'g--', lw=1.5, label='AMP')
    ax.plot(t, result['GTP'], 'c-', lw=1.5, label='GTP')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Energy Nucleotides')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Energy charge
    ax = axes[0, 1]
    ax.plot(t, result['energy_charge'], 'k-', lw=2)
    ax.axhline(0.9, color='g', ls='--', label='Optimal')
    ax.axhline(0.7, color='r', ls='--', label='Stress')
    ax.fill_between(t, 0.85, 0.95, alpha=0.2, color='green', label='Healthy range')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Energy Charge')
    ax.set_title('Cellular Energy Charge')
    ax.set_ylim([0.6, 1.0])
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Precursor pools
    ax = axes[1, 0]
    ax.plot(t, result['AA_pool'], 'b-', lw=2, label='Amino acids')
    ax.plot(t, result['NTP_pool'], 'r-', lw=2, label='NTPs')
    ax.plot(t, result['dNTP_pool']*10, 'g-', lw=2, label='dNTPs (×10)')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Precursor Pools')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Glycolysis products
    ax = axes[1, 1]
    ax.plot(t, result['glucose_int'], 'b-', lw=2, label='Glucose (int)')
    ax.plot(t, result['pyruvate'], 'r-', lw=2, label='Pyruvate')
    ax.plot(t, result['lactate'], 'g-', lw=2, label='Lactate')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    ax.set_title('Glycolysis')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metabolism_homeostasis.png', dpi=150)
    print("\n✓ Saved: metabolism_homeostasis.png")
    
    return model, result


if __name__ == '__main__':
    model, result = main()
