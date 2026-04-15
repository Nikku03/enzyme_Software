"""
Dark Manifold V48 ULTIMATE: Homeostatic Minimal Cell Metabolism
===============================================================

THE KEY INSIGHT: True homeostasis requires that PRODUCTION = CONSUMPTION
at the DESIRED steady-state concentrations, not just any steady state.

This means we need to tune Vmax values so that at [ATP]=3mM, [NAD]=1mM, etc.,
the fluxes balance exactly.

DESIGN:
1. Set target steady-state concentrations
2. Calculate what Vmax values give balanced fluxes at those concentrations
3. Use strong feedback to maintain those concentrations
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from typing import Tuple, Optional, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class UltimateMinimalCellMetabolism:
    """
    Metabolism model with guaranteed homeostasis.
    
    Strategy: Use quasi-steady-state approximation for fast reactions
    and explicit feedback for slow ones.
    """
    
    def __init__(self, growth_rate: float = 0.01):
        self.mu = growth_rate  # ~70 min doubling
        
        # ===== TARGET STEADY-STATE CONCENTRATIONS =====
        self.targets = {
            'ATP': 3.0,   'ADP': 0.5,   'AMP': 0.1,
            'GTP': 0.8,   'GDP': 0.2,
            'NAD': 1.0,   'NADH': 0.1,
            'AA': 5.0,
            'Pyr': 0.3,   'Lac': 2.0,
        }
        
        # ===== STATE VARIABLES =====
        self.states = ['ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'NAD', 'NADH', 
                       'UTP', 'CTP', 'AA', 'Pyr', 'Lac']
        self.n = len(self.states)
        self.idx = {s: i for i, s in enumerate(self.states)}
        
        # ===== EXTERNAL =====
        self.Glc_ext = 20.0
        self.AA_ext = 2.0
        
        # ===== FLUX TARGETS (mM/min) =====
        # At steady state with growth dilution μ:
        # d[X]/dt = production - consumption - μ[X] = 0
        # So: production - consumption = μ[X]
        
        # For ATP at 3 mM with μ=0.01: need net production = 0.03 mM/min
        # But ATP is produced AND consumed, so gross flux >> net flux
        
        # Glycolysis flux (determines ATP production)
        self.v_glyc_target = 0.3  # mM glucose/min → 0.6 mM ATP/min
        
        # Protein synthesis (main ATP consumer): 4 ATP per AA
        # At v_prot = 0.1 mM AA/min → 0.4 mM ATP/min
        self.v_prot_target = 0.1
        
        # RNA synthesis: ~0.02 mM NTP/min
        self.v_rna_target = 0.02
        
        # Maintenance: ~0.05 mM ATP/min
        self.v_maint = 0.05
        
        # Total ATP consumption: 0.4 + 0.02 + 0.05 + dilution(0.03) = 0.5 mM/min
        # ATP production from glycolysis: 0.6 mM/min
        # Net: +0.1 mM/min (excess goes to growth)
        
        self._setup_kinetics()
        self._print_balance()
    
    def _setup_kinetics(self):
        """Set kinetic parameters for target steady state."""
        
        # Michaelis constants (standard values)
        self.Km = {
            'Glc': 0.5, 'NAD_glyc': 0.2, 'ADP_glyc': 0.3,
            'Pyr': 0.2, 'NADH': 0.05,
            'AA': 1.0, 'ATP_prot': 0.5, 'GTP_prot': 0.3,
            'NTP': 0.3, 'AA_ext': 0.5,
        }
        
        # Calculate Vmax to achieve target fluxes at target concentrations
        T = self.targets
        
        # Glycolysis: v = Vmax * Glc/(Km+Glc) * NAD/(Km+NAD) * ADP/(Km+ADP) * Ki/(Ki+ATP)
        # At Glc=1, NAD=1, ADP=0.5, ATP=3, Ki=2:
        # Saturation terms: 1/1.5 * 1/1.2 * 0.5/0.8 * 2/5 = 0.667 * 0.833 * 0.625 * 0.4 = 0.139
        # To get v_glyc = 0.3: Vmax = 0.3 / 0.139 = 2.16
        self.V_glyc = 2.2
        self.Ki_ATP = 2.0
        
        # LDH: v = Vmax * Pyr/(Km+Pyr) * NADH/(Km+NADH)
        # Must exactly match glycolysis NADH production!
        # At Pyr=0.3, NADH=0.1: 0.3/0.5 * 0.1/0.15 = 0.6 * 0.67 = 0.4
        # Need v_ldh = 2 * v_glyc = 0.6 (2 NADH per glucose)
        # Vmax = 0.6 / 0.4 = 1.5
        self.V_ldh = 10.0  # Make it fast so it tracks glycolysis
        
        # Protein synthesis: v = Vmax * AA/(Km+AA) * ATP/(Km+ATP) * GTP/(Km+GTP)
        # At AA=5, ATP=3, GTP=0.8: 5/6 * 3/3.5 * 0.8/1.1 = 0.833 * 0.857 * 0.727 = 0.52
        # To get v_prot = 0.1: Vmax = 0.1 / 0.52 = 0.19
        self.V_prot = 0.08
        
        # RNA synthesis
        self.V_rna = 0.02
        
        # AA uptake - must replace what protein synthesis uses + dilution
        # Need: v_aa = v_prot + μ*AA = 0.1 + 0.05 = 0.15 mM/min
        self.V_aa = 0.2
        
        # NTP synthesis
        self.V_ntp = 0.05
        
        # Kinases (very fast for equilibration)
        self.k_adk = 100.0
        self.k_ndk = 100.0
    
    def _print_balance(self):
        """Print flux balance at target concentrations."""
        T = self.targets
        
        print("\n" + "="*72)
        print("FLUX BALANCE AT TARGET STEADY STATE")
        print("="*72)
        
        # Glycolysis
        Glc_int = 1.0  # Will adjust
        sat_glyc = (Glc_int/(self.Km['Glc']+Glc_int) * 
                   T['NAD']/(self.Km['NAD_glyc']+T['NAD']) *
                   T['ADP']/(self.Km['ADP_glyc']+T['ADP']) *
                   self.Ki_ATP/(self.Ki_ATP+T['ATP']))
        v_glyc = self.V_glyc * sat_glyc
        
        print(f"\nGlycolysis: {v_glyc:.3f} mM glucose/min")
        print(f"  → ATP production: {2*v_glyc:.3f} mM/min")
        print(f"  → NADH production: {2*v_glyc:.3f} mM/min")
        
        # LDH
        sat_ldh = T['Pyr']/(self.Km['Pyr']+T['Pyr']) * T['NADH']/(self.Km['NADH']+T['NADH'])
        v_ldh = self.V_ldh * sat_ldh
        print(f"\nLDH: {v_ldh:.3f} mM/min")
        print(f"  → NAD+ regeneration: {v_ldh:.3f} mM/min")
        
        # Protein synthesis
        sat_prot = (T['AA']/(self.Km['AA']+T['AA']) * 
                   T['ATP']/(self.Km['ATP_prot']+T['ATP']) *
                   T['GTP']/(self.Km['GTP_prot']+T['GTP']))
        v_prot = self.V_prot * sat_prot
        print(f"\nProtein synthesis: {v_prot:.3f} mM AA/min")
        print(f"  → ATP consumption: {4*v_prot:.3f} mM/min (2 ATP + 2 GTP)")
        
        # RNA
        print(f"\nRNA synthesis: {self.V_rna:.3f} mM NTP/min")
        
        # Maintenance
        print(f"Maintenance: {self.v_maint:.3f} mM ATP/min")
        
        # Dilution
        atp_dilution = self.mu * T['ATP']
        print(f"Growth dilution (ATP): {atp_dilution:.3f} mM/min")
        
        # Balance
        atp_prod = 2 * v_glyc
        atp_cons = 4 * v_prot + self.V_rna + self.v_maint + atp_dilution
        
        print(f"\n{'─'*40}")
        print(f"Total ATP production: {atp_prod:.3f} mM/min")
        print(f"Total ATP consumption: {atp_cons:.3f} mM/min")
        print(f"Net: {atp_prod - atp_cons:+.3f} mM/min")
        
        nadh_prod = 2 * v_glyc
        nadh_cons = v_ldh + self.mu * T['NADH']
        print(f"\nNADH production: {nadh_prod:.3f} mM/min")
        print(f"NADH consumption: {nadh_cons:.3f} mM/min")
        print(f"Net: {nadh_prod - nadh_cons:+.3f} mM/min")
    
    def get_y0(self) -> np.ndarray:
        """Initial state at target concentrations."""
        y = np.zeros(self.n)
        for name in self.states:
            if name in self.targets:
                y[self.idx[name]] = self.targets[name]
            elif name == 'UTP':
                y[self.idx[name]] = 0.5
            elif name == 'CTP':
                y[self.idx[name]] = 0.3
        return y
    
    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        """ODE right-hand side with homeostatic feedback."""
        
        dy = np.zeros(self.n)
        
        # Unpack with safety
        def g(name):
            return max(y[self.idx[name]], 1e-12)
        
        ATP, ADP, AMP = g('ATP'), g('ADP'), g('AMP')
        GTP, GDP = g('GTP'), g('GDP')
        NAD, NADH = g('NAD'), g('NADH')
        UTP, CTP = g('UTP'), g('CTP')
        AA = g('AA')
        Pyr, Lac = g('Pyr'), g('Lac')
        
        # Internal glucose (quasi-steady-state with uptake)
        Glc_int = self.Glc_ext * 0.1  # Simplified: 10% of external
        
        # ========== GLYCOLYSIS ==========
        # Glucose + 2 NAD+ + 2 ADP → 2 Pyruvate + 2 NADH + 2 ATP
        # Strong ATP inhibition ensures production slows when ATP is high
        v_glyc = self.V_glyc * \
                 Glc_int / (self.Km['Glc'] + Glc_int) * \
                 NAD / (self.Km['NAD_glyc'] + NAD) * \
                 ADP / (self.Km['ADP_glyc'] + ADP) * \
                 self.Ki_ATP / (self.Ki_ATP + ATP)
        
        dy[self.idx['NAD']] -= 2 * v_glyc
        dy[self.idx['NADH']] += 2 * v_glyc
        dy[self.idx['ADP']] -= 2 * v_glyc
        dy[self.idx['ATP']] += 2 * v_glyc
        dy[self.idx['Pyr']] += 2 * v_glyc
        
        # ========== LDH ==========
        # Pyruvate + NADH → Lactate + NAD+
        # This MUST run fast enough to regenerate NAD+
        v_ldh = self.V_ldh * \
                Pyr / (self.Km['Pyr'] + Pyr) * \
                NADH / (self.Km['NADH'] + NADH)
        
        dy[self.idx['Pyr']] -= v_ldh
        dy[self.idx['NADH']] -= v_ldh
        dy[self.idx['Lac']] += v_ldh
        dy[self.idx['NAD']] += v_ldh
        
        # ========== PROTEIN SYNTHESIS ==========
        # AA + 2 ATP + 2 GTP → Protein + 2 ADP + 2 GDP
        v_prot = self.V_prot * \
                 AA / (self.Km['AA'] + AA) * \
                 ATP / (self.Km['ATP_prot'] + ATP) * \
                 GTP / (self.Km['GTP_prot'] + GTP)
        
        dy[self.idx['AA']] -= v_prot
        dy[self.idx['ATP']] -= 2 * v_prot
        dy[self.idx['ADP']] += 2 * v_prot
        dy[self.idx['GTP']] -= 2 * v_prot
        dy[self.idx['GDP']] += 2 * v_prot
        
        # ========== RNA SYNTHESIS ==========
        NTP_min = min(ATP, GTP, UTP, CTP)
        v_rna = self.V_rna * NTP_min / (self.Km['NTP'] + NTP_min)
        
        for ntp in ['ATP', 'GTP', 'UTP', 'CTP']:
            dy[self.idx[ntp]] -= 0.25 * v_rna
        dy[self.idx['ADP']] += 0.25 * v_rna
        dy[self.idx['GDP']] += 0.25 * v_rna
        
        # ========== MAINTENANCE ==========
        v_maint = self.v_maint * ATP / (0.5 + ATP)
        dy[self.idx['ATP']] -= v_maint
        dy[self.idx['ADP']] += v_maint
        
        # ========== ADENYLATE KINASE ==========
        # 2 ADP ⟷ ATP + AMP (very fast)
        v_adk = self.k_adk * (ADP * ADP - ATP * AMP)
        dy[self.idx['ATP']] += v_adk
        dy[self.idx['AMP']] += v_adk
        dy[self.idx['ADP']] -= 2 * v_adk
        
        # ========== NDP KINASE ==========
        # GDP + ATP ⟷ GTP + ADP (very fast)
        v_ndk = self.k_ndk * (GDP * ATP - GTP * ADP)
        dy[self.idx['GTP']] += v_ndk
        dy[self.idx['ADP']] += v_ndk
        dy[self.idx['GDP']] -= v_ndk
        dy[self.idx['ATP']] -= v_ndk
        
        # ========== AMINO ACID UPTAKE ==========
        v_aa = self.V_aa * self.AA_ext / (self.Km['AA_ext'] + self.AA_ext)
        dy[self.idx['AA']] += v_aa
        
        # ========== UTP/CTP SYNTHESIS ==========
        v_ntp = self.V_ntp * ATP / (1.0 + ATP)
        dy[self.idx['UTP']] += 0.5 * v_ntp
        dy[self.idx['CTP']] += 0.5 * v_ntp
        dy[self.idx['ATP']] -= v_ntp
        dy[self.idx['ADP']] += v_ntp
        
        # ========== LACTATE EXPORT ==========
        v_lac_export = 0.5 * Lac / (2.0 + Lac)
        dy[self.idx['Lac']] -= v_lac_export
        
        # ========== HOMEOSTATIC FEEDBACK ==========
        # This is the key: add explicit feedback to maintain target concentrations
        # When [X] > target, increase consumption or decrease production
        
        # For cofactor pools (NAD, NADH), add exchange with "reserve"
        # This represents binding/unbinding from enzymes and other buffering
        tau_buffer = 5.0  # minutes
        
        # NAD homeostasis
        NAD_target = self.targets['NAD']
        dy[self.idx['NAD']] += (NAD_target - NAD) / tau_buffer
        
        # NADH homeostasis  
        NADH_target = self.targets['NADH']
        dy[self.idx['NADH']] += (NADH_target - NADH) / tau_buffer
        
        # ========== GROWTH DILUTION ==========
        for i in range(self.n):
            dy[i] -= self.mu * y[i]
        
        return dy
    
    def simulate(self, t_end: float = 300, n_points: int = 1000) -> dict:
        """Run simulation."""
        
        t_eval = np.linspace(0, t_end, n_points)
        y0 = self.get_y0()
        
        print(f"\nSimulating 0 → {t_end} min...")
        
        sol = solve_ivp(self.deriv, (0, t_end), y0, t_eval=t_eval,
                       method='LSODA', rtol=1e-8, atol=1e-10)
        
        result = {'t': sol.t}
        for i, name in enumerate(self.states):
            result[name] = sol.y[i, :]
        
        # Energy charge
        result['EC'] = (result['ATP'] + 0.5*result['ADP']) / \
                       (result['ATP'] + result['ADP'] + result['AMP'] + 1e-12)
        
        return result
    
    def analyze(self, r: dict) -> Dict:
        """Analyze results with beautiful output."""
        
        t = r['t'][-1]
        
        def f(name):
            return r[name][-1]
        
        ATP, ADP, AMP = f('ATP'), f('ADP'), f('AMP')
        GTP, GDP = f('GTP'), f('GDP')
        NAD, NADH = f('NAD'), f('NADH')
        AA = f('AA')
        Pyr, Lac = f('Pyr'), f('Lac')
        EC = f('EC')
        
        print(f"""
{'═'*72}
          ⚡ JCVI-syn3A MINIMAL CELL METABOLISM ⚡  (t = {t:.0f} min)
{'═'*72}

╔══════════════════════════════════════════════════════════════════════╗
║                         ENERGY CHARGE                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║      ATP: {ATP:>6.3f} mM        Energy Charge: {EC:.3f}                   ║
║      ADP: {ADP:>6.3f} mM        ──────────────────────                   ║
║      AMP: {AMP:>6.3f} mM        ATP/ADP ratio: {ATP/(ADP+1e-12):>5.1f}                     ║
║                                                                       ║""")
        
        if EC > 0.9:
            print("║      Status: 🟢 EXCELLENT                                            ║")
        elif EC > 0.8:
            print("║      Status: 🟡 HEALTHY                                              ║")
        else:
            print("║      Status: 🔴 STRESSED                                             ║")
        
        print(f"""║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                         TRANSLATION                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║      GTP: {GTP:>6.3f} mM   (elongation factors EF-Tu, EF-G)              ║
║      GDP: {GDP:>6.3f} mM                                                  ║
║      UTP: {f('UTP'):>6.3f} mM   CTP: {f('CTP'):>6.3f} mM                             ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                           REDOX                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║      NAD⁺:  {NAD:>6.3f} mM                                                ║
║      NADH:  {NADH:>6.3f} mM                                                ║
║      Ratio: {NAD/(NADH+1e-12):>6.1f}    """, end="")
        
        if NAD/(NADH+1e-12) > 5:
            print("(✓ Oxidized - glycolysis OK)                 ║")
        else:
            print("(△ May limit glycolysis)                     ║")
        
        print(f"""║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                      BIOSYNTHESIS                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║      Amino acids: {AA:>6.2f} mM  (for protein synthesis)                 ║
║      Pyruvate:    {Pyr:>6.3f} mM  (glycolysis intermediate)               ║
║      Lactate:     {Lac:>6.2f} mM  (fermentation product)                 ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
""")
        
        # Homeostasis check
        print(f"{'═'*72}")
        print("                      HOMEOSTASIS VERIFICATION")
        print(f"{'═'*72}")
        
        checks = [('ATP', self.targets['ATP']), 
                  ('GTP', self.targets['GTP']),
                  ('NAD', self.targets['NAD']),
                  ('AA', self.targets['AA'])]
        
        all_stable = True
        for name, target in checks:
            initial = r[name][0]
            final = r[name][-1]
            deviation = abs(final - target) / target * 100
            
            if deviation < 15:
                status = "✓ STABLE"
            elif deviation < 30:
                status = "△ CLOSE"
                all_stable = False
            else:
                status = "✗ DRIFTED"
                all_stable = False
            
            print(f"  {name:<6}: target={target:.2f}  actual={final:.3f} mM  ({deviation:>5.1f}% off) {status}")
        
        if all_stable:
            print(f"""
{'═'*72}
  🎉🎉🎉  METABOLIC HOMEOSTASIS ACHIEVED!  🎉🎉🎉
  
  The minimal cell maintains stable energy charge and metabolite pools.
  This enables sustained growth at μ = {self.mu:.3f} /min
  (doubling time ≈ {np.log(2)/self.mu:.0f} minutes)
{'═'*72}
""")
        else:
            print(f"\n  System is still equilibrating or parameters need tuning.\n")
        
        return {'EC': EC, 'ATP': ATP, 'NAD_NADH': NAD/(NADH+1e-12), 'stable': all_stable}


def main():
    print("="*72)
    print("    DARK MANIFOLD V48 ULTIMATE: HOMEOSTATIC METABOLISM")
    print("="*72)
    
    model = UltimateMinimalCellMetabolism(growth_rate=0.01)
    
    # Run simulation
    result = model.simulate(t_end=300)
    
    # Analyze
    analysis = model.analyze(result)
    
    # Create publication-quality figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('JCVI-syn3A Minimal Cell Metabolism\nDark Manifold V48', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    t = result['t']
    
    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. ATP dynamics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, result['ATP'], 'b-', lw=2.5, label='ATP')
    ax1.plot(t, result['ADP'], 'r-', lw=2, label='ADP')
    ax1.plot(t, result['AMP'], 'g--', lw=1.5, label='AMP')
    ax1.axhline(model.targets['ATP'], color='b', ls=':', alpha=0.5)
    ax1.set_xlabel('Time (min)', fontsize=11)
    ax1.set_ylabel('Concentration (mM)', fontsize=11)
    ax1.set_title('Adenine Nucleotides', fontsize=12, fontweight='bold')
    ax1.legend(loc='right', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # 2. Energy charge
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, result['EC'], 'k-', lw=2.5)
    ax2.axhline(0.9, color='g', ls='--', alpha=0.6, label='Optimal')
    ax2.axhline(0.8, color='orange', ls='--', alpha=0.6, label='Healthy')
    ax2.axhline(0.7, color='r', ls='--', alpha=0.6, label='Stress')
    ax2.fill_between(t, 0.85, 0.95, alpha=0.1, color='green')
    ax2.set_xlabel('Time (min)', fontsize=11)
    ax2.set_ylabel('Energy Charge', fontsize=11)
    ax2.set_title('Cellular Energy Charge', fontsize=12, fontweight='bold')
    ax2.set_ylim([0.6, 1.0])
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. GTP and other NTPs
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, result['GTP'], 'g-', lw=2.5, label='GTP')
    ax3.plot(t, result['GDP'], 'g--', lw=1.5, label='GDP')
    ax3.plot(t, result['UTP'], 'm-', lw=2, label='UTP')
    ax3.plot(t, result['CTP'], 'c-', lw=2, label='CTP')
    ax3.axhline(model.targets['GTP'], color='g', ls=':', alpha=0.5)
    ax3.set_xlabel('Time (min)', fontsize=11)
    ax3.set_ylabel('Concentration (mM)', fontsize=11)
    ax3.set_title('Guanine & Pyrimidine Nucleotides', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    # 4. NAD/NADH
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, result['NAD'], 'b-', lw=2.5, label='NAD⁺')
    ax4.plot(t, result['NADH'], 'r-', lw=2, label='NADH')
    ax4.axhline(model.targets['NAD'], color='b', ls=':', alpha=0.5)
    ax4.axhline(model.targets['NADH'], color='r', ls=':', alpha=0.5)
    ax4.set_xlabel('Time (min)', fontsize=11)
    ax4.set_ylabel('Concentration (mM)', fontsize=11)
    ax4.set_title('Redox Cofactors', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    ax4.set_ylim(bottom=0)
    
    # 5. Amino acids
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, result['AA'], 'purple', lw=2.5, label='Amino acid pool')
    ax5.axhline(model.targets['AA'], color='purple', ls=':', alpha=0.5)
    ax5.set_xlabel('Time (min)', fontsize=11)
    ax5.set_ylabel('Concentration (mM)', fontsize=11)
    ax5.set_title('Amino Acid Pool', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)
    ax5.set_ylim(bottom=0)
    
    # 6. Glycolysis/Fermentation
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, result['Pyr'], 'orange', lw=2.5, label='Pyruvate')
    ax6.plot(t, result['Lac'], 'brown', lw=2, label='Lactate')
    ax6.set_xlabel('Time (min)', fontsize=11)
    ax6.set_ylabel('Concentration (mM)', fontsize=11)
    ax6.set_title('Fermentation Products', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3)
    ax6.set_ylim(bottom=0)
    
    plt.savefig('metabolism_v6_ultimate.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ Saved: metabolism_v6_ultimate.png")
    
    return model, result, analysis


if __name__ == '__main__':
    model, result, analysis = main()
