"""
Dark Manifold V48e: Balanced Minimal Cell Metabolism
====================================================

KEY INSIGHT: ATP balance is everything.

Production: Glycolysis produces 2 ATP per glucose
Consumption: Protein synthesis is 70-80% of ATP budget

For homeostasis:
  ATP_production ≈ ATP_consumption

Let's calculate:
- Glucose uptake: ~0.5 mM/min
- ATP from glycolysis: ~1.0 mM/min (2 per glucose)
- Available for biosynthesis: ~0.8 mM/min (after maintenance)
- Protein synthesis at 4 ATP/AA: ~0.2 mM AA/min

This sets the growth rate!
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class BalancedMinimalCellMetabolism:
    """
    Properly balanced metabolic model.
    
    The key is that production and consumption are coupled through
    ATP/ADP ratio - when ATP drops, biosynthesis slows automatically.
    """
    
    def __init__(self):
        # ===== STATE VARIABLES =====
        self.state_names = [
            # Energy
            'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'Pi',
            # Redox
            'NAD', 'NADH',
            # Nucleotides
            'UTP', 'CTP',
            # Precursors
            'AA', 'R5P',
            # Glycolysis
            'G6P', 'FBP', 'PEP', 'Pyr', 'Lac',
        ]
        self.n_states = len(self.state_names)
        self.idx = {name: i for i, name in enumerate(self.state_names)}
        
        # ===== INITIAL CONDITIONS (mM) =====
        self.initial = {
            'ATP': 3.0, 'ADP': 0.5, 'AMP': 0.1,
            'GTP': 0.8, 'GDP': 0.2, 'Pi': 10.0,
            'NAD': 1.0, 'NADH': 0.1,
            'UTP': 0.5, 'CTP': 0.3,
            'AA': 5.0, 'R5P': 0.2,
            'G6P': 0.3, 'FBP': 0.1, 'PEP': 0.2, 'Pyr': 0.3, 'Lac': 1.0,
        }
        
        # ===== EXTERNAL CONDITIONS =====
        self.Glc_ext = 20.0  # mM glucose
        self.AA_ext = 2.0    # mM amino acids
        
        # ===== KINETIC PARAMETERS =====
        # Designed for ATP balance at steady state
        
        # Glucose uptake
        self.V_glc = 0.8      # mM/min
        self.Km_glc = 2.0     # mM
        
        # Upper glycolysis (G6P → FBP)
        self.V_pfk = 2.0      # mM/min
        self.Km_G6P = 0.1
        self.Km_ATP_pfk = 0.2
        self.Ki_ATP_pfk = 3.0  # High ATP inhibits (key regulation!)
        
        # Lower glycolysis (FBP → 2 PEP)
        self.V_lower = 5.0    # mM/min (fast)
        self.Km_FBP = 0.05
        self.Km_NAD = 0.2
        
        # Pyruvate kinase (PEP + ADP → Pyr + ATP)
        self.V_pyk = 3.0      # mM/min
        self.Km_PEP = 0.2
        self.Km_ADP_pyk = 0.3
        
        # LDH (Pyr + NADH → Lac + NAD+)
        self.V_ldh = 5.0      # mM/min
        self.Km_Pyr = 0.3
        self.Km_NADH = 0.02
        
        # Protein synthesis (THE MAIN ATP CONSUMER)
        self.V_protein = 0.15  # mM AA/min (tuned for balance!)
        self.Km_AA = 1.0
        self.Km_ATP_prot = 0.8
        self.Km_GTP_prot = 0.3
        
        # RNA synthesis
        self.V_rna = 0.03     # mM NTP/min
        self.Km_NTP = 0.3
        
        # Maintenance
        self.V_maint = 0.1    # mM ATP/min
        
        # AA uptake
        self.V_aa = 0.3       # mM/min
        self.Km_AA_ext = 0.5
        
        # NTP synthesis
        self.V_ntp = 0.1      # mM/min
        
        # Adenylate kinase (fast equilibration)
        self.k_adk = 20.0
        
        # NDP kinase (fast)
        self.k_ndk = 20.0
        
        # PPP (for NADPH and R5P)
        self.V_ppp = 0.1      # mM/min
        
        # Growth dilution
        self.mu = 0.01        # ~70 min doubling
        
        self._estimate_fluxes()
    
    def _estimate_fluxes(self):
        """Estimate steady-state fluxes for balance check."""
        print("\n" + "="*70)
        print("FLUX BALANCE ESTIMATE")
        print("="*70)
        
        # At steady state with typical concentrations
        ATP, ADP = 3.0, 0.5
        
        # Glucose flux
        v_glc = self.V_glc * self.Glc_ext / (self.Km_glc + self.Glc_ext)
        print(f"Glucose uptake: {v_glc:.3f} mM/min")
        
        # ATP production from glycolysis (2 ATP per glucose)
        atp_prod = 2 * v_glc
        print(f"ATP production (glycolysis): {atp_prod:.3f} mM/min")
        
        # ATP consumption
        v_prot = self.V_protein * 5.0/(self.Km_AA + 5.0) * ATP/(self.Km_ATP_prot + ATP)
        atp_prot = 4 * v_prot  # 2 ATP + 2 GTP per AA
        print(f"Protein synthesis: {v_prot:.3f} mM AA/min → {atp_prot:.3f} mM ATP/min")
        
        v_rna = self.V_rna * 1.0/(self.Km_NTP + 1.0)
        atp_rna = v_rna  # ~1 ATP equiv per NTP
        print(f"RNA synthesis: {v_rna:.3f} mM/min → {atp_rna:.3f} mM ATP/min")
        
        atp_maint = self.V_maint
        print(f"Maintenance: {atp_maint:.3f} mM ATP/min")
        
        atp_cons = atp_prot + atp_rna + atp_maint
        print(f"\nTotal ATP production: {atp_prod:.3f} mM/min")
        print(f"Total ATP consumption: {atp_cons:.3f} mM/min")
        print(f"Balance: {atp_prod - atp_cons:+.3f} mM/min")
        
        if abs(atp_prod - atp_cons) < 0.1:
            print("✓ Fluxes are balanced!")
        elif atp_prod > atp_cons:
            print("△ Production > consumption (ATP will accumulate)")
        else:
            print("△ Consumption > production (ATP will deplete)")
    
    def get_initial_state(self) -> np.ndarray:
        y0 = np.zeros(self.n_states)
        for name, val in self.initial.items():
            y0[self.idx[name]] = val
        return y0
    
    def ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        dydt = np.zeros(self.n_states)
        
        # Unpack (with safety)
        def get(name):
            return max(y[self.idx[name]], 1e-12)
        
        ATP = get('ATP')
        ADP = get('ADP')
        AMP = get('AMP')
        GTP = get('GTP')
        GDP = get('GDP')
        Pi = get('Pi')
        NAD = get('NAD')
        NADH = get('NADH')
        UTP = get('UTP')
        CTP = get('CTP')
        AA = get('AA')
        R5P = get('R5P')
        G6P = get('G6P')
        FBP = get('FBP')
        PEP = get('PEP')
        Pyr = get('Pyr')
        Lac = get('Lac')
        
        # ========== GLYCOLYSIS ==========
        
        # 1. Glucose uptake → G6P
        v_glc = self.V_glc * self.Glc_ext / (self.Km_glc + self.Glc_ext)
        dydt[self.idx['G6P']] += v_glc
        
        # 2. PFK: G6P + ATP → FBP + ADP (with ATP inhibition!)
        # Key: High ATP inhibits to prevent overproduction
        atp_inhibition = self.Ki_ATP_pfk / (self.Ki_ATP_pfk + ATP)
        adp_activation = 1 + ADP / 0.5  # ADP activates
        v_pfk = self.V_pfk * G6P/(self.Km_G6P + G6P) * ATP/(self.Km_ATP_pfk + ATP) * \
                atp_inhibition * adp_activation
        
        dydt[self.idx['G6P']] -= v_pfk
        dydt[self.idx['ATP']] -= v_pfk
        dydt[self.idx['FBP']] += v_pfk
        dydt[self.idx['ADP']] += v_pfk
        
        # 3. Lower glycolysis: FBP + 2NAD+ → 2PEP + 2NADH + 2ATP
        # (Lumped: aldolase + TPI + GAPDH + PGK)
        v_lower = self.V_lower * FBP/(self.Km_FBP + FBP) * NAD/(self.Km_NAD + NAD) * \
                  ADP/(0.3 + ADP)  # Needs ADP as substrate
        
        dydt[self.idx['FBP']] -= v_lower
        dydt[self.idx['NAD']] -= 2 * v_lower
        dydt[self.idx['NADH']] += 2 * v_lower
        dydt[self.idx['PEP']] += 2 * v_lower
        dydt[self.idx['ATP']] += 2 * v_lower  # From PGK
        dydt[self.idx['ADP']] -= 2 * v_lower
        
        # 4. Pyruvate kinase: PEP + ADP → Pyr + ATP
        v_pyk = self.V_pyk * PEP/(self.Km_PEP + PEP) * ADP/(self.Km_ADP_pyk + ADP)
        
        dydt[self.idx['PEP']] -= v_pyk
        dydt[self.idx['ADP']] -= v_pyk
        dydt[self.idx['Pyr']] += v_pyk
        dydt[self.idx['ATP']] += v_pyk
        
        # 5. LDH: Pyr + NADH → Lac + NAD+ (regenerates NAD+!)
        v_ldh = self.V_ldh * Pyr/(self.Km_Pyr + Pyr) * NADH/(self.Km_NADH + NADH)
        
        dydt[self.idx['Pyr']] -= v_ldh
        dydt[self.idx['NADH']] -= v_ldh
        dydt[self.idx['Lac']] += v_ldh
        dydt[self.idx['NAD']] += v_ldh
        
        # ========== PROTEIN SYNTHESIS ==========
        # AA + 2ATP + 2GTP → protein + 2ADP + 2GDP + 4Pi
        v_prot = self.V_protein * AA/(self.Km_AA + AA) * \
                 ATP/(self.Km_ATP_prot + ATP) * GTP/(self.Km_GTP_prot + GTP)
        
        dydt[self.idx['AA']] -= v_prot
        dydt[self.idx['ATP']] -= 2 * v_prot
        dydt[self.idx['ADP']] += 2 * v_prot
        dydt[self.idx['GTP']] -= 2 * v_prot
        dydt[self.idx['GDP']] += 2 * v_prot
        dydt[self.idx['Pi']] += 4 * v_prot
        
        # ========== RNA SYNTHESIS ==========
        # Consumes NTPs
        NTP_avg = (ATP + GTP + UTP + CTP) / 4
        v_rna = self.V_rna * NTP_avg/(self.Km_NTP + NTP_avg)
        
        dydt[self.idx['ATP']] -= 0.25 * v_rna
        dydt[self.idx['GTP']] -= 0.25 * v_rna
        dydt[self.idx['UTP']] -= 0.25 * v_rna
        dydt[self.idx['CTP']] -= 0.25 * v_rna
        dydt[self.idx['ADP']] += 0.25 * v_rna
        dydt[self.idx['GDP']] += 0.25 * v_rna
        dydt[self.idx['Pi']] += 2 * v_rna  # PPi → 2Pi
        
        # ========== MAINTENANCE ==========
        v_maint = self.V_maint * ATP/(0.5 + ATP)
        dydt[self.idx['ATP']] -= v_maint
        dydt[self.idx['ADP']] += v_maint
        dydt[self.idx['Pi']] += v_maint
        
        # ========== ADENYLATE KINASE ==========
        # 2 ADP ⟷ ATP + AMP (maintains adenylate balance)
        v_adk = self.k_adk * (ADP * ADP - ATP * AMP)
        dydt[self.idx['ATP']] += v_adk
        dydt[self.idx['AMP']] += v_adk
        dydt[self.idx['ADP']] -= 2 * v_adk
        
        # ========== NDP KINASE ==========
        # GDP + ATP ⟷ GTP + ADP
        v_ndk_g = self.k_ndk * (GDP * ATP - GTP * ADP)
        dydt[self.idx['GTP']] += v_ndk_g
        dydt[self.idx['ADP']] += v_ndk_g
        dydt[self.idx['GDP']] -= v_ndk_g
        dydt[self.idx['ATP']] -= v_ndk_g
        
        # ========== AMINO ACID UPTAKE ==========
        v_aa = self.V_aa * self.AA_ext/(self.Km_AA_ext + self.AA_ext) * \
               ATP/(0.5 + ATP)  # Energy-dependent
        dydt[self.idx['AA']] += v_aa
        dydt[self.idx['ATP']] -= 0.1 * v_aa  # Small cost
        dydt[self.idx['ADP']] += 0.1 * v_aa
        
        # ========== NTP SYNTHESIS ==========
        # Simplified: ATP → UTP, CTP
        v_ntp = self.V_ntp * ATP/(1.0 + ATP) * R5P/(0.1 + R5P)
        dydt[self.idx['UTP']] += 0.5 * v_ntp
        dydt[self.idx['CTP']] += 0.5 * v_ntp
        dydt[self.idx['ATP']] -= v_ntp
        dydt[self.idx['ADP']] += v_ntp
        dydt[self.idx['R5P']] -= 0.5 * v_ntp
        
        # ========== PENTOSE PHOSPHATE PATHWAY ==========
        # G6P → R5P (provides ribose for nucleotides)
        v_ppp = self.V_ppp * G6P/(0.2 + G6P)
        dydt[self.idx['G6P']] -= v_ppp
        dydt[self.idx['R5P']] += v_ppp
        
        # ========== GROWTH DILUTION ==========
        for i in range(self.n_states):
            dydt[i] -= self.mu * y[i]
        
        return dydt
    
    def simulate(self, t_span: Tuple[float, float],
                 t_eval: Optional[np.ndarray] = None) -> dict:
        
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
            atol=1e-10
        )
        
        result = {'t': solution.t}
        for i, name in enumerate(self.state_names):
            result[name] = solution.y[i, :]
        
        # Derived quantities
        result['energy_charge'] = (result['ATP'] + 0.5*result['ADP']) / \
                                  (result['ATP'] + result['ADP'] + result['AMP'] + 1e-12)
        result['total_NTP'] = result['ATP'] + result['GTP'] + result['UTP'] + result['CTP']
        
        return result
    
    def analyze(self, result: dict) -> dict:
        """Detailed analysis."""
        
        t = result['t'][-1]
        
        # Final values
        ATP = result['ATP'][-1]
        ADP = result['ADP'][-1]
        AMP = result['AMP'][-1]
        GTP = result['GTP'][-1]
        NAD = result['NAD'][-1]
        NADH = result['NADH'][-1]
        AA = result['AA'][-1]
        Lac = result['Lac'][-1]
        ec = result['energy_charge'][-1]
        
        print(f"\n{'='*70}")
        print(f"METABOLIC STATE AT t = {t:.0f} min")
        print("="*70)
        
        # Energy box
        print("\n╔════════════════════════════════════════════════════════════════════╗")
        print("║                         ENERGY STATUS                              ║")
        print("╠════════════════════════════════════════════════════════════════════╣")
        print(f"║  ATP:              {ATP:>7.3f} mM                                     ║")
        print(f"║  ADP:              {ADP:>7.3f} mM                                     ║")
        print(f"║  AMP:              {AMP:>7.3f} mM                                     ║")
        print(f"║  ATP/ADP ratio:    {ATP/(ADP+1e-12):>7.1f}                                       ║")
        print(f"║  Energy charge:    {ec:>7.3f}   ", end="")
        if ec > 0.85:
            print("✓ HEALTHY                         ║")
        elif ec > 0.70:
            print("△ MODERATE                        ║")
        else:
            print("✗ STRESSED                        ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        
        # GTP box
        print("\n╔════════════════════════════════════════════════════════════════════╗")
        print("║                       TRANSLATION FUEL                             ║")
        print("╠════════════════════════════════════════════════════════════════════╣")
        print(f"║  GTP:              {GTP:>7.3f} mM  (needed for ribosomes)             ║")
        print(f"║  GDP:              {result['GDP'][-1]:>7.3f} mM                                     ║")
        print(f"║  UTP:              {result['UTP'][-1]:>7.3f} mM  (for RNA synthesis)              ║")
        print(f"║  CTP:              {result['CTP'][-1]:>7.3f} mM                                     ║")
        print(f"║  Total NTPs:       {result['total_NTP'][-1]:>7.2f} mM                                     ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        
        # Redox box
        print("\n╔════════════════════════════════════════════════════════════════════╗")
        print("║                         REDOX STATUS                               ║")
        print("╠════════════════════════════════════════════════════════════════════╣")
        print(f"║  NAD+:             {NAD:>7.3f} mM                                     ║")
        print(f"║  NADH:             {NADH:>7.3f} mM                                     ║")
        print(f"║  NAD+/NADH ratio:  {NAD/(NADH+1e-12):>7.1f}   ", end="")
        if NAD/(NADH+1e-12) > 3:
            print("✓ GLYCOLYSIS READY             ║")
        else:
            print("△ REDUCTIVE                    ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        
        # Precursors box
        print("\n╔════════════════════════════════════════════════════════════════════╗")
        print("║                      BIOSYNTHESIS READY                            ║")
        print("╠════════════════════════════════════════════════════════════════════╣")
        print(f"║  Amino acids:      {AA:>7.2f} mM  (for protein synthesis)           ║")
        print(f"║  Ribose-5-P:       {result['R5P'][-1]:>7.3f} mM  (for nucleotides)               ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        
        # Glycolysis
        print("\n╔════════════════════════════════════════════════════════════════════╗")
        print("║                         GLYCOLYSIS                                 ║")
        print("╠════════════════════════════════════════════════════════════════════╣")
        print(f"║  G6P:              {result['G6P'][-1]:>7.3f} mM                                     ║")
        print(f"║  FBP:              {result['FBP'][-1]:>7.3f} mM                                     ║")
        print(f"║  PEP:              {result['PEP'][-1]:>7.3f} mM                                     ║")
        print(f"║  Pyruvate:         {result['Pyr'][-1]:>7.3f} mM                                     ║")
        print(f"║  Lactate:          {Lac:>7.2f} mM  (fermentation product)           ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        
        # Stability check
        print(f"\n{'='*70}")
        print("HOMEOSTASIS CHECK")
        print("="*70)
        
        checks = {
            'ATP': (result['ATP'][0], ATP),
            'GTP': (result['GTP'][0], GTP),
            'NAD': (result['NAD'][0], NAD),
            'AA': (result['AA'][0], AA),
        }
        
        all_stable = True
        for name, (initial, final) in checks.items():
            change = (final - initial) / (initial + 1e-12) * 100
            if abs(change) < 20:
                status = "✓ STABLE"
            elif abs(change) < 50:
                status = "△ ADJUSTING"
                all_stable = False
            else:
                status = "✗ UNSTABLE"
                all_stable = False
            print(f"  {name:<6}: {initial:>6.3f} → {final:>6.3f} mM ({change:>+6.1f}%) {status}")
        
        if all_stable:
            print("\n" + "="*70)
            print("🎉 HOMEOSTASIS ACHIEVED! THE CELL IS IN METABOLIC STEADY STATE! 🎉")
            print("="*70)
        
        return {
            'energy_charge': ec,
            'ATP': ATP, 'ADP': ADP, 'GTP': GTP,
            'NAD_NADH': NAD/(NADH+1e-12),
            'AA': AA, 'lactate': Lac
        }


def main():
    print("="*70)
    print("DARK MANIFOLD V48e: BALANCED MINIMAL CELL METABOLISM")
    print("="*70)
    
    model = BalancedMinimalCellMetabolism()
    
    # Simulate 3 hours
    result = model.simulate(t_span=(0, 180))
    
    # Analyze
    analysis = model.analyze(result)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    t = result['t']
    
    # 1. ATP/ADP/AMP
    ax = axes[0, 0]
    ax.plot(t, result['ATP'], 'b-', lw=2.5, label='ATP')
    ax.plot(t, result['ADP'], 'r-', lw=2, label='ADP')
    ax.plot(t, result['AMP'], 'g--', lw=1.5, label='AMP')
    ax.set_xlabel('Time (min)', fontsize=11)
    ax.set_ylabel('Concentration (mM)', fontsize=11)
    ax.set_title('Adenine Nucleotides', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 2. Energy charge
    ax = axes[0, 1]
    ax.plot(t, result['energy_charge'], 'k-', lw=2.5)
    ax.axhline(0.85, color='green', ls='--', alpha=0.7, label='Healthy threshold')
    ax.axhline(0.70, color='red', ls='--', alpha=0.7, label='Stress threshold')
    ax.fill_between(t, 0.85, 0.95, alpha=0.15, color='green')
    ax.set_xlabel('Time (min)', fontsize=11)
    ax.set_ylabel('Energy Charge', fontsize=11)
    ax.set_title('Cellular Energy Charge', fontsize=12, fontweight='bold')
    ax.set_ylim([0.5, 1.0])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 3. GTP and other NTPs
    ax = axes[0, 2]
    ax.plot(t, result['GTP'], 'g-', lw=2.5, label='GTP')
    ax.plot(t, result['UTP'], 'm-', lw=2, label='UTP')
    ax.plot(t, result['CTP'], 'c-', lw=2, label='CTP')
    ax.set_xlabel('Time (min)', fontsize=11)
    ax.set_ylabel('Concentration (mM)', fontsize=11)
    ax.set_title('Nucleoside Triphosphates', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 4. NAD/NADH
    ax = axes[1, 0]
    ax.plot(t, result['NAD'], 'b-', lw=2.5, label='NAD+')
    ax.plot(t, result['NADH'], 'r-', lw=2, label='NADH')
    ax.set_xlabel('Time (min)', fontsize=11)
    ax.set_ylabel('Concentration (mM)', fontsize=11)
    ax.set_title('Redox Cofactors', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 5. Precursors
    ax = axes[1, 1]
    ax.plot(t, result['AA'], 'b-', lw=2.5, label='Amino acids')
    ax.plot(t, result['R5P']*5, 'g-', lw=2, label='Ribose-5-P (×5)')
    ax.set_xlabel('Time (min)', fontsize=11)
    ax.set_ylabel('Concentration (mM)', fontsize=11)
    ax.set_title('Biosynthetic Precursors', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 6. Glycolysis products
    ax = axes[1, 2]
    ax.plot(t, result['Pyr'], 'r-', lw=2.5, label='Pyruvate')
    ax.plot(t, result['Lac'], 'g-', lw=2, label='Lactate')
    ax.plot(t, result['PEP'], 'b--', lw=1.5, label='PEP')
    ax.set_xlabel('Time (min)', fontsize=11)
    ax.set_ylabel('Concentration (mM)', fontsize=11)
    ax.set_title('Glycolysis', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('metabolism_balanced.png', dpi=150)
    print("\n✓ Saved: metabolism_balanced.png")
    
    return model, result, analysis


if __name__ == '__main__':
    model, result, analysis = main()
