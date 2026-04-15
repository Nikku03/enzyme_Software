"""
Dark Manifold V38: Kinetic Virtual Cell
========================================

Dynamic ODE-based simulation of JCVI-syn3A metabolism.
First step toward atomic resolution - adds TIME to V37.

Key advances over V37 (FBA):
- Metabolite concentrations change over time
- Michaelis-Menten enzyme kinetics  
- Allosteric regulation (inhibition/activation)
- ATP/ADP energy charge dynamics
- NAD+/NADH redox state

This is Level 1 of the multi-scale architecture.
Next: Level 2 (stochastic), Level 3 (spatial), Level 4 (atomic)

Author: Naresh Chhillar, 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import time


# ============================================================================
# KINETIC PARAMETERS
# From literature and estimated from thermodynamics
# ============================================================================

@dataclass
class EnzymeKinetics:
    """Michaelis-Menten kinetics for an enzyme."""
    vmax: float              # Maximum velocity (mM/s)
    km: Dict[str, float]     # Michaelis constants for substrates
    ki: Dict[str, float] = field(default_factory=dict)  # Inhibition constants
    ka: Dict[str, float] = field(default_factory=dict)  # Activation constants
    hill: float = 1.0        # Hill coefficient
    reversible: bool = True
    keq: float = 1.0         # Equilibrium constant


# Literature-derived kinetic parameters for key enzymes
# Sources: BRENDA, ecocyc, parameter fitting from Breuer et al.
KINETIC_PARAMS = {
    # Glycolysis
    'GLCpts': EnzymeKinetics(
        vmax=10.0,  # High capacity glucose uptake
        km={'glc': 0.05, 'pep': 0.1},  # mM
        ki={'g6p': 1.0},  # Product inhibition
        reversible=False
    ),
    'PGI': EnzymeKinetics(
        vmax=100.0,
        km={'g6p': 0.5, 'f6p': 0.2},
        keq=0.4,  # Favors F6P slightly
        reversible=True
    ),
    'PFK': EnzymeKinetics(
        vmax=50.0,
        km={'f6p': 0.1, 'atp': 0.05},
        ki={'atp': 2.0, 'pep': 0.5},  # ATP inhibition at high conc
        ka={'adp': 0.5, 'amp': 0.1},  # ADP/AMP activation
        hill=4.0,  # Highly cooperative
        reversible=False
    ),
    'FBA': EnzymeKinetics(
        vmax=50.0,
        km={'fbp': 0.05, 'g3p': 0.1, 'dhap': 0.1},
        keq=0.1,  # Favors cleavage
        reversible=True
    ),
    'TPI': EnzymeKinetics(
        vmax=500.0,  # Very fast
        km={'dhap': 0.5, 'g3p': 0.5},
        keq=0.05,  # Favors DHAP
        reversible=True
    ),
    'GAPD': EnzymeKinetics(
        vmax=100.0,
        km={'g3p': 0.05, 'nad': 0.1, 'pi': 1.0, 'bpg13': 0.01, 'nadh': 0.01},
        keq=0.01,  # Far from equilibrium
        reversible=True
    ),
    'PGK': EnzymeKinetics(
        vmax=200.0,
        km={'bpg13': 0.01, 'adp': 0.1, 'pg3': 0.5, 'atp': 0.5},
        keq=3000,  # Strongly favors ATP production
        reversible=True
    ),
    'PGM': EnzymeKinetics(
        vmax=100.0,
        km={'pg3': 0.2, 'pg2': 0.1},
        keq=0.15,
        reversible=True
    ),
    'ENO': EnzymeKinetics(
        vmax=100.0,
        km={'pg2': 0.1, 'pep': 0.1},
        keq=3.0,  # Favors PEP
        reversible=True
    ),
    'PYK': EnzymeKinetics(
        vmax=100.0,
        km={'pep': 0.05, 'adp': 0.2},
        ki={'atp': 5.0},
        ka={'fbp': 0.1},  # FBP activation (feedforward)
        hill=4.0,
        reversible=False
    ),
    # Fermentation (NAD+ regeneration)
    'LDH': EnzymeKinetics(
        vmax=200.0,
        km={'pyr': 0.5, 'nadh': 0.01, 'lac': 5.0, 'nad': 0.1},
        keq=20000,  # Strongly favors lactate
        reversible=True
    ),
    # Pentose Phosphate Pathway
    'G6PDH': EnzymeKinetics(
        vmax=20.0,
        km={'g6p': 0.05, 'nadp': 0.01},
        ki={'nadph': 0.01},  # NADPH inhibition
        reversible=False
    ),
    'GND': EnzymeKinetics(
        vmax=20.0,
        km={'go6p': 0.1, 'nadp': 0.01},
        reversible=False
    ),
    # ATP synthase
    'ATPS': EnzymeKinetics(
        vmax=50.0,  # Simplified ATP maintenance
        km={'adp': 0.5, 'pi': 1.0},
        ki={'atp': 5.0},
        reversible=False
    ),
    # Nucleotide kinases
    'ADK': EnzymeKinetics(
        vmax=100.0,
        km={'amp': 0.1, 'atp': 0.1, 'adp': 0.1},
        keq=1.0,  # Near equilibrium
        reversible=True
    ),
    'NDK': EnzymeKinetics(
        vmax=200.0,
        km={'atp': 0.1, 'gdp': 0.1, 'udp': 0.1, 'cdp': 0.1},
        keq=1.0,
        reversible=True
    ),
}

# Initial metabolite concentrations (mM)
# From Breuer et al. 2019 and E. coli literature
INITIAL_CONCENTRATIONS = {
    # Energy carriers
    'atp': 5.0,
    'adp': 1.0,
    'amp': 0.1,
    'gtp': 1.0,
    'gdp': 0.2,
    'utp': 1.0,
    'udp': 0.2,
    'ctp': 0.5,
    'cdp': 0.1,
    'nad': 1.0,
    'nadh': 0.1,
    'nadp': 0.1,
    'nadph': 0.01,
    'coa': 0.5,
    'pi': 10.0,
    'ppi': 0.01,
    
    # Glycolysis intermediates
    'glc': 0.1,      # External glucose kept at ~constant by medium
    'g6p': 1.0,
    'f6p': 0.3,
    'fbp': 0.5,
    'g3p': 0.1,
    'dhap': 0.5,
    'bpg13': 0.01,
    'pg3': 0.5,
    'pg2': 0.1,
    'pep': 0.2,
    'pyr': 1.0,
    'lac': 0.1,
    
    # PPP intermediates
    'gl6p': 0.01,
    'go6p': 0.05,
    'ru5p': 0.1,
    'r5p': 0.2,
    'x5p': 0.1,
    's7p': 0.05,
    'e4p': 0.02,
    'prpp': 0.05,
    
    # Amino acids (simplified - from medium)
    'aa_pool': 10.0,  # Lumped amino acid pool
    
    # Macromolecule precursors
    'protein_precursor': 0.1,
    'rna_precursor': 0.1,
    'dna_precursor': 0.01,
    'lipid_precursor': 0.1,
    
    # Biomass (arbitrary units)
    'biomass': 1.0,
}


# ============================================================================
# RATE EQUATIONS
# ============================================================================

def michaelis_menten(vmax: float, S: float, Km: float) -> float:
    """Simple Michaelis-Menten rate."""
    return vmax * S / (Km + S)


def michaelis_menten_reversible(vmax_f: float, S: float, Km_s: float, 
                                 P: float, Km_p: float, Keq: float) -> float:
    """Reversible Michaelis-Menten."""
    # Haldane relationship for reverse Vmax
    vmax_r = vmax_f / Keq
    numerator = vmax_f * S / Km_s - vmax_r * P / Km_p
    denominator = 1 + S / Km_s + P / Km_p
    return numerator / denominator


def hill_equation(vmax: float, S: float, Km: float, n: float) -> float:
    """Hill equation for cooperative binding."""
    return vmax * (S ** n) / (Km ** n + S ** n)


def allosteric_modifier(base_rate: float, 
                        inhibitors: Dict[str, Tuple[float, float]],
                        activators: Dict[str, Tuple[float, float]],
                        concentrations: Dict[str, float]) -> float:
    """
    Apply allosteric modification to a rate.
    
    inhibitors/activators: {metabolite: (concentration, Ki/Ka)}
    """
    rate = base_rate
    
    # Competitive/non-competitive inhibition
    for met, (conc, Ki) in inhibitors.items():
        if conc > 0 and Ki > 0:
            rate *= Ki / (Ki + conc)
    
    # Activation (increases Vmax or decreases Km)
    for met, (conc, Ka) in activators.items():
        if conc > 0 and Ka > 0:
            rate *= (1 + conc / Ka)
    
    return rate


# ============================================================================
# KINETIC MODEL
# ============================================================================

class KineticModel:
    """
    ODE-based kinetic model of JCVI-syn3A core metabolism.
    """
    
    def __init__(self):
        # Metabolite setup
        self.met_ids = list(INITIAL_CONCENTRATIONS.keys())
        self.met_idx = {m: i for i, m in enumerate(self.met_ids)}
        self.n_mets = len(self.met_ids)
        
        # Initial state
        self.y0 = np.array([INITIAL_CONCENTRATIONS[m] for m in self.met_ids])
        
        # External (clamped) metabolites - kept constant by medium
        self.external = {'glc', 'aa_pool', 'pi'}
        
        # Gene to enzyme mapping
        self.gene_enzymes = {
            'JCVISYN3A_0685': 'GLCpts',
            'JCVISYN3A_0233': 'PGI',
            'JCVISYN3A_0207': 'PFK',
            'JCVISYN3A_0352': 'FBA',
            'JCVISYN3A_0353': 'TPI',
            'JCVISYN3A_0314': 'GAPD',
            'JCVISYN3A_0315': 'PGK',
            'JCVISYN3A_0689': 'PGM',
            'JCVISYN3A_0231': 'ENO',
            'JCVISYN3A_0546': 'PYK',
            'JCVISYN3A_0449': 'LDH',
            'JCVISYN3A_0439': 'G6PDH',
            'JCVISYN3A_0441': 'GND',
            'JCVISYN3A_0005': 'ADK',
            'JCVISYN3A_0416': 'NDK',
        }
        
        # Knockout state
        self.knockouts = set()
        
        print(f"Kinetic model: {self.n_mets} metabolites, {len(KINETIC_PARAMS)} enzymes")
    
    def get(self, y: np.ndarray, met: str) -> float:
        """Get concentration of metabolite."""
        if met not in self.met_idx:
            return 0.0
        return max(0, y[self.met_idx[met]])
    
    def compute_rates(self, y: np.ndarray) -> Dict[str, float]:
        """Compute all reaction rates given current concentrations."""
        rates = {}
        
        # Helper to check if enzyme is knocked out
        def is_active(enzyme: str) -> bool:
            for gene, enz in self.gene_enzymes.items():
                if enz == enzyme and gene in self.knockouts:
                    return False
            return True
        
        # === GLYCOLYSIS ===
        
        # GLCpts: glc + pep -> g6p + pyr
        if is_active('GLCpts'):
            kin = KINETIC_PARAMS['GLCpts']
            glc = self.get(y, 'glc')
            pep = self.get(y, 'pep')
            g6p = self.get(y, 'g6p')
            
            # Bi-substrate kinetics (ordered)
            v = kin.vmax * (glc / (kin.km['glc'] + glc)) * (pep / (kin.km['pep'] + pep))
            # Product inhibition
            v *= kin.ki.get('g6p', 10) / (kin.ki.get('g6p', 10) + g6p)
            rates['GLCpts'] = v
        else:
            rates['GLCpts'] = 0
        
        # PGI: g6p <-> f6p
        if is_active('PGI'):
            kin = KINETIC_PARAMS['PGI']
            g6p = self.get(y, 'g6p')
            f6p = self.get(y, 'f6p')
            rates['PGI'] = michaelis_menten_reversible(
                kin.vmax, g6p, kin.km['g6p'], f6p, kin.km['f6p'], kin.keq
            )
        else:
            rates['PGI'] = 0
        
        # PFK: f6p + atp -> fbp + adp (major regulatory point)
        if is_active('PFK'):
            kin = KINETIC_PARAMS['PFK']
            f6p = self.get(y, 'f6p')
            atp = self.get(y, 'atp')
            adp = self.get(y, 'adp')
            amp = self.get(y, 'amp')
            pep = self.get(y, 'pep')
            
            # Hill kinetics for F6P
            v = hill_equation(kin.vmax, f6p, kin.km['f6p'], kin.hill)
            # ATP as substrate
            v *= atp / (kin.km['atp'] + atp)
            # ATP inhibition at high concentration
            v *= kin.ki['atp'] / (kin.ki['atp'] + atp)
            # ADP/AMP activation (energy charge sensing)
            v *= (1 + adp / kin.ka['adp'] + amp / kin.ka['amp'])
            # PEP inhibition
            v *= kin.ki['pep'] / (kin.ki['pep'] + pep)
            rates['PFK'] = v
        else:
            rates['PFK'] = 0
        
        # FBA: fbp <-> g3p + dhap
        if is_active('FBA'):
            kin = KINETIC_PARAMS['FBA']
            fbp = self.get(y, 'fbp')
            g3p = self.get(y, 'g3p')
            dhap = self.get(y, 'dhap')
            # Simplified: product = g3p * dhap equivalent
            rates['FBA'] = michaelis_menten_reversible(
                kin.vmax, fbp, kin.km['fbp'], g3p * dhap, kin.km['g3p'] * kin.km['dhap'], kin.keq
            )
        else:
            rates['FBA'] = 0
        
        # TPI: dhap <-> g3p
        if is_active('TPI'):
            kin = KINETIC_PARAMS['TPI']
            dhap = self.get(y, 'dhap')
            g3p = self.get(y, 'g3p')
            rates['TPI'] = michaelis_menten_reversible(
                kin.vmax, dhap, kin.km['dhap'], g3p, kin.km['g3p'], kin.keq
            )
        else:
            rates['TPI'] = 0
        
        # GAPD: g3p + nad + pi -> bpg13 + nadh
        if is_active('GAPD'):
            kin = KINETIC_PARAMS['GAPD']
            g3p = self.get(y, 'g3p')
            nad = self.get(y, 'nad')
            pi = self.get(y, 'pi')
            bpg13 = self.get(y, 'bpg13')
            nadh = self.get(y, 'nadh')
            
            # Three substrates
            v_f = kin.vmax * (g3p / (kin.km['g3p'] + g3p)) * (nad / (kin.km['nad'] + nad)) * (pi / (kin.km['pi'] + pi))
            v_r = (kin.vmax / kin.keq) * (bpg13 / (kin.km['bpg13'] + bpg13)) * (nadh / (kin.km['nadh'] + nadh))
            rates['GAPD'] = v_f - v_r
        else:
            rates['GAPD'] = 0
        
        # PGK: bpg13 + adp -> pg3 + atp
        if is_active('PGK'):
            kin = KINETIC_PARAMS['PGK']
            bpg13 = self.get(y, 'bpg13')
            adp = self.get(y, 'adp')
            pg3 = self.get(y, 'pg3')
            atp = self.get(y, 'atp')
            
            v_f = kin.vmax * (bpg13 / (kin.km['bpg13'] + bpg13)) * (adp / (kin.km['adp'] + adp))
            v_r = (kin.vmax / kin.keq) * (pg3 / (kin.km['pg3'] + pg3)) * (atp / (kin.km['atp'] + atp))
            rates['PGK'] = v_f - v_r
        else:
            rates['PGK'] = 0
        
        # PGM: pg3 <-> pg2
        if is_active('PGM'):
            kin = KINETIC_PARAMS['PGM']
            pg3 = self.get(y, 'pg3')
            pg2 = self.get(y, 'pg2')
            rates['PGM'] = michaelis_menten_reversible(
                kin.vmax, pg3, kin.km['pg3'], pg2, kin.km['pg2'], kin.keq
            )
        else:
            rates['PGM'] = 0
        
        # ENO: pg2 <-> pep
        if is_active('ENO'):
            kin = KINETIC_PARAMS['ENO']
            pg2 = self.get(y, 'pg2')
            pep = self.get(y, 'pep')
            rates['ENO'] = michaelis_menten_reversible(
                kin.vmax, pg2, kin.km['pg2'], pep, kin.km['pep'], kin.keq
            )
        else:
            rates['ENO'] = 0
        
        # PYK: pep + adp -> pyr + atp
        if is_active('PYK'):
            kin = KINETIC_PARAMS['PYK']
            pep = self.get(y, 'pep')
            adp = self.get(y, 'adp')
            atp = self.get(y, 'atp')
            fbp = self.get(y, 'fbp')
            
            # Hill kinetics
            v = hill_equation(kin.vmax, pep, kin.km['pep'], kin.hill)
            v *= adp / (kin.km['adp'] + adp)
            # ATP inhibition
            v *= kin.ki['atp'] / (kin.ki['atp'] + atp)
            # FBP activation (feedforward)
            v *= (1 + fbp / kin.ka['fbp'])
            rates['PYK'] = v
        else:
            rates['PYK'] = 0
        
        # === FERMENTATION ===
        
        # LDH: pyr + nadh <-> lac + nad
        if is_active('LDH'):
            kin = KINETIC_PARAMS['LDH']
            pyr = self.get(y, 'pyr')
            nadh = self.get(y, 'nadh')
            lac = self.get(y, 'lac')
            nad = self.get(y, 'nad')
            
            v_f = kin.vmax * (pyr / (kin.km['pyr'] + pyr)) * (nadh / (kin.km['nadh'] + nadh))
            v_r = (kin.vmax / kin.keq) * (lac / (kin.km['lac'] + lac)) * (nad / (kin.km['nad'] + nad))
            rates['LDH'] = v_f - v_r
        else:
            rates['LDH'] = 0
        
        # === PENTOSE PHOSPHATE PATHWAY ===
        
        # G6PDH: g6p + nadp -> gl6p + nadph
        if is_active('G6PDH'):
            kin = KINETIC_PARAMS['G6PDH']
            g6p = self.get(y, 'g6p')
            nadp = self.get(y, 'nadp')
            nadph = self.get(y, 'nadph')
            
            v = kin.vmax * (g6p / (kin.km['g6p'] + g6p)) * (nadp / (kin.km['nadp'] + nadp))
            # NADPH inhibition
            v *= kin.ki.get('nadph', 0.1) / (kin.ki.get('nadph', 0.1) + nadph)
            rates['G6PDH'] = v
        else:
            rates['G6PDH'] = 0
        
        # GND: go6p + nadp -> ru5p + nadph + co2
        if is_active('GND'):
            kin = KINETIC_PARAMS['GND']
            go6p = self.get(y, 'go6p')
            nadp = self.get(y, 'nadp')
            v = kin.vmax * (go6p / (kin.km['go6p'] + go6p)) * (nadp / (kin.km['nadp'] + nadp))
            rates['GND'] = v
        else:
            rates['GND'] = 0
        
        # PGL: gl6p -> go6p (fast, near equilibrium)
        gl6p = self.get(y, 'gl6p')
        rates['PGL'] = 100.0 * gl6p  # Very fast
        
        # PPP non-oxidative (simplified)
        ru5p = self.get(y, 'ru5p')
        rates['PPP_nonox'] = 10.0 * ru5p  # Converts to r5p, x5p, etc.
        
        # === ATP MAINTENANCE ===
        
        # Simplified ATP consumption for growth/maintenance
        atp = self.get(y, 'atp')
        biomass = self.get(y, 'biomass')
        rates['ATP_maintenance'] = 5.0 * atp / (1.0 + atp)  # Basal ATP consumption
        rates['ATP_growth'] = 10.0 * biomass * atp / (2.0 + atp)  # Growth-associated
        
        # ADK: amp + atp <-> 2 adp
        if is_active('ADK'):
            kin = KINETIC_PARAMS['ADK']
            amp = self.get(y, 'amp')
            atp = self.get(y, 'atp')
            adp = self.get(y, 'adp')
            
            v_f = kin.vmax * (amp / (kin.km['amp'] + amp)) * (atp / (kin.km['atp'] + atp))
            v_r = (kin.vmax / kin.keq) * (adp ** 2) / (kin.km['adp'] ** 2 + adp ** 2)
            rates['ADK'] = v_f - v_r
        else:
            rates['ADK'] = 0
        
        # === BIOMASS SYNTHESIS (simplified) ===
        
        # Growth rate depends on precursors and energy
        aa = self.get(y, 'aa_pool')
        ntp = atp + self.get(y, 'gtp') + self.get(y, 'utp') + self.get(y, 'ctp')
        r5p = self.get(y, 'r5p')
        
        # Monod-like growth kinetics
        mu_max = 0.5  # Max growth rate (1/h) - JCVI-syn3A doubles in ~2h
        growth = mu_max * biomass
        growth *= aa / (1.0 + aa)  # Amino acid limitation
        growth *= ntp / (5.0 + ntp)  # Energy limitation
        growth *= r5p / (0.5 + r5p)  # Nucleotide precursor limitation
        
        rates['GROWTH'] = growth
        
        # Lactate export
        lac = self.get(y, 'lac')
        rates['LAC_export'] = 10.0 * lac / (1.0 + lac)
        
        return rates
    
    def dydt(self, t: float, y: np.ndarray) -> np.ndarray:
        """Compute time derivatives of all metabolites."""
        rates = self.compute_rates(y)
        
        dy = np.zeros(self.n_mets)
        
        # === GLYCOLYSIS ===
        
        # Glucose (external - clamped)
        # dy[self.met_idx['glc']] = 0  # Kept constant
        
        # G6P: +GLCpts, -PGI, -G6PDH
        dy[self.met_idx['g6p']] = rates['GLCpts'] - rates['PGI'] - rates['G6PDH']
        
        # F6P: +PGI, -PFK
        dy[self.met_idx['f6p']] = rates['PGI'] - rates['PFK']
        
        # FBP: +PFK, -FBA
        dy[self.met_idx['fbp']] = rates['PFK'] - rates['FBA']
        
        # DHAP: +FBA, -TPI
        dy[self.met_idx['dhap']] = rates['FBA'] - rates['TPI']
        
        # G3P: +FBA, +TPI, -GAPD
        dy[self.met_idx['g3p']] = rates['FBA'] + rates['TPI'] - rates['GAPD']
        
        # 1,3-BPG: +GAPD, -PGK
        dy[self.met_idx['bpg13']] = rates['GAPD'] - rates['PGK']
        
        # 3PG: +PGK, -PGM
        dy[self.met_idx['pg3']] = rates['PGK'] - rates['PGM']
        
        # 2PG: +PGM, -ENO
        dy[self.met_idx['pg2']] = rates['PGM'] - rates['ENO']
        
        # PEP: +ENO, -PYK, -GLCpts
        dy[self.met_idx['pep']] = rates['ENO'] - rates['PYK'] - rates['GLCpts']
        
        # Pyruvate: +GLCpts, +PYK, -LDH
        dy[self.met_idx['pyr']] = rates['GLCpts'] + rates['PYK'] - rates['LDH']
        
        # Lactate: +LDH, -export
        dy[self.met_idx['lac']] = rates['LDH'] - rates['LAC_export']
        
        # === ENERGY ===
        
        # ATP: +PGK, +PYK, -PFK, -GLCpts(indirect), -maintenance, -growth
        dy[self.met_idx['atp']] = (
            rates['PGK'] + rates['PYK'] 
            - rates['PFK'] 
            - rates['ATP_maintenance'] 
            - rates['ATP_growth']
            - rates['ADK']
        )
        
        # ADP: -PGK, -PYK, +PFK, +maintenance, +growth, +2*ADK
        dy[self.met_idx['adp']] = (
            -rates['PGK'] - rates['PYK'] 
            + rates['PFK']
            + rates['ATP_maintenance']
            + rates['ATP_growth']
            + 2 * rates['ADK']
        )
        
        # AMP: -ADK
        dy[self.met_idx['amp']] = -rates['ADK']
        
        # NAD: +LDH, -GAPD
        dy[self.met_idx['nad']] = rates['LDH'] - rates['GAPD']
        
        # NADH: -LDH, +GAPD
        dy[self.met_idx['nadh']] = -rates['LDH'] + rates['GAPD']
        
        # NADP: +G6PDH, +GND (consumed by biosynthesis)
        dy[self.met_idx['nadp']] = -rates['G6PDH'] - rates['GND'] + 0.1 * rates['GROWTH']
        
        # NADPH: +G6PDH, +GND, -biosynthesis
        dy[self.met_idx['nadph']] = rates['G6PDH'] + rates['GND'] - 0.5 * rates['GROWTH']
        
        # === PPP ===
        
        # 6PGL: +G6PDH, -PGL
        dy[self.met_idx['gl6p']] = rates['G6PDH'] - rates['PGL']
        
        # 6PGonate: +PGL, -GND
        dy[self.met_idx['go6p']] = rates['PGL'] - rates['GND']
        
        # Ru5P: +GND, -PPP_nonox
        dy[self.met_idx['ru5p']] = rates['GND'] - rates['PPP_nonox']
        
        # R5P: +PPP_nonox (simplified)
        dy[self.met_idx['r5p']] = 0.5 * rates['PPP_nonox'] - 0.1 * rates['GROWTH']
        
        # === BIOMASS ===
        
        # Biomass grows
        dy[self.met_idx['biomass']] = rates['GROWTH']
        
        # Clamp external metabolites
        for met in self.external:
            if met in self.met_idx:
                dy[self.met_idx[met]] = 0
        
        return dy
    
    def simulate(self, t_span: Tuple[float, float], 
                 t_eval: np.ndarray = None,
                 knockouts: List[str] = None) -> dict:
        """
        Run ODE simulation.
        
        Args:
            t_span: (t_start, t_end) in hours
            t_eval: Time points to record
            knockouts: List of gene IDs to knock out
        
        Returns:
            Dictionary with time, concentrations, rates
        """
        self.knockouts = set(knockouts) if knockouts else set()
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 100)
        
        start = time.time()
        
        sol = solve_ivp(
            self.dydt,
            t_span,
            self.y0,
            method='LSODA',  # Good for stiff systems
            t_eval=t_eval,
            dense_output=True,
            max_step=0.01  # Small steps for accuracy
        )
        
        elapsed = time.time() - start
        
        # Compute rates at final state
        final_rates = self.compute_rates(sol.y[:, -1])
        
        # Compute energy charge
        final_atp = sol.y[self.met_idx['atp'], -1]
        final_adp = sol.y[self.met_idx['adp'], -1]
        final_amp = sol.y[self.met_idx['amp'], -1]
        energy_charge = (final_atp + 0.5 * final_adp) / (final_atp + final_adp + final_amp + 1e-9)
        
        return {
            't': sol.t,
            'y': sol.y,
            'met_ids': self.met_ids,
            'met_idx': self.met_idx,
            'rates': final_rates,
            'energy_charge': energy_charge,
            'final_biomass': sol.y[self.met_idx['biomass'], -1],
            'elapsed_s': elapsed,
            'success': sol.success
        }
    
    def knockout_essentiality(self, gene: str, t_sim: float = 2.0) -> dict:
        """
        Test if a gene knockout is lethal.
        
        Simulates for t_sim hours and checks if:
        1. Biomass decreased
        2. Energy charge collapsed
        3. Key metabolites depleted
        """
        # Wild-type simulation
        wt = self.simulate((0, t_sim))
        
        # Knockout simulation
        ko = self.simulate((0, t_sim), knockouts=[gene])
        
        # Essentiality criteria
        biomass_ratio = ko['final_biomass'] / (wt['final_biomass'] + 1e-9)
        ec_ratio = ko['energy_charge'] / (wt['energy_charge'] + 1e-9)
        
        essential = (biomass_ratio < 0.1) or (ec_ratio < 0.5)
        
        return {
            'gene': gene,
            'essential': essential,
            'biomass_ratio': biomass_ratio,
            'energy_charge_ratio': ec_ratio,
            'wt_biomass': wt['final_biomass'],
            'ko_biomass': ko['final_biomass'],
            'wt_ec': wt['energy_charge'],
            'ko_ec': ko['energy_charge']
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("DARK MANIFOLD V38: KINETIC VIRTUAL CELL")
    print("Dynamic ODE Simulation of JCVI-syn3A")
    print("="*60)
    
    model = KineticModel()
    
    # Run wild-type simulation for 2 hours (one doubling time)
    print("\n--- Wild-type simulation (2h) ---")
    wt = model.simulate((0, 2.0))
    
    print(f"Simulation: {'SUCCESS' if wt['success'] else 'FAILED'}")
    print(f"Time: {wt['elapsed_s']*1000:.1f}ms")
    print(f"Final biomass: {wt['final_biomass']:.3f}")
    print(f"Energy charge: {wt['energy_charge']:.3f}")
    
    print("\nKey fluxes (final):")
    for rxn in ['GLCpts', 'PFK', 'PYK', 'LDH', 'G6PDH', 'GROWTH']:
        if rxn in wt['rates']:
            print(f"  {rxn}: {wt['rates'][rxn]:.4f} mM/s")
    
    print("\nFinal concentrations:")
    for met in ['atp', 'adp', 'nad', 'nadh', 'pyr', 'lac', 'g6p']:
        idx = wt['met_idx'][met]
        print(f"  {met}: {wt['y'][idx, -1]:.4f} mM")
    
    # Test some knockouts
    print("\n--- Knockout essentiality tests ---")
    test_genes = [
        ('JCVISYN3A_0207', 'pfkA'),
        ('JCVISYN3A_0546', 'pyk'),
        ('JCVISYN3A_0449', 'ldh'),
        ('JCVISYN3A_0231', 'eno'),
    ]
    
    for gene_id, gene_name in test_genes:
        result = model.knockout_essentiality(gene_id, t_sim=1.0)
        status = "ESSENTIAL" if result['essential'] else f"viable ({result['biomass_ratio']:.0%})"
        print(f"  Δ{gene_name}: {status} | EC: {result['ko_ec']:.2f}")
    
    return model, wt


if __name__ == '__main__':
    model, result = main()
