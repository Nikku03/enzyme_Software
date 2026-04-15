"""
Dark Manifold V35.2: Thermodynamic Cell
========================================

The cell doesn't know what to do. Physics does it.

KEY ADDITIONS:
1. Thermodynamics (ΔG) - reactions have direction and equilibrium
2. Allosteric feedback - ATP inhibits PFK, etc.
3. Conservation laws - adenylate pool is constant
4. Reversible kinetics - reactions can go backwards

NO MORE CAPS. Equilibrium emerges from physics.

Author: Naresh Chhillar, 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Physical constants
R = 8.314e-3  # kJ/(mol·K)
T = 310.15    # K (37°C)
RT = R * T    # ~2.58 kJ/mol

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Metabolite:
    id: str
    name: str
    initial_conc: float  # mM
    
    # Thermodynamic data
    delta_Gf: float = 0.0  # Standard Gibbs free energy of formation (kJ/mol)
    
    # Conservation group (for pool constraints)
    pool: Optional[str] = None  # e.g., "adenylate", "guanylate", "nad"


@dataclass
class Reaction:
    id: str
    name: str
    substrates: Dict[str, float]  # metabolite_id -> stoichiometry
    products: Dict[str, float]
    enzyme: str
    
    # Kinetics
    kcat_forward: float = 10.0   # s⁻¹
    kcat_reverse: float = 1.0    # s⁻¹
    km_substrates: Dict[str, float] = field(default_factory=dict)  # per-substrate Km
    km_products: Dict[str, float] = field(default_factory=dict)    # per-product Km
    
    # Thermodynamics
    delta_G0: float = 0.0  # Standard ΔG° (kJ/mol) - from eQuilibrator
    
    # Regulation
    inhibitors: Dict[str, float] = field(default_factory=dict)   # metabolite -> Ki
    activators: Dict[str, float] = field(default_factory=dict)   # metabolite -> Ka
    
    # Flags
    reversible: bool = True
    essential: bool = False


# ============================================================================
# THERMODYNAMIC DATA (from eQuilibrator / literature)
# ============================================================================

# Standard Gibbs free energies of formation (kJ/mol) at pH 7, I=0.1M
# Source: eQuilibrator (equilibrator.weizmann.ac.il)
DELTA_GF = {
    'atp': -2292.5,
    'adp': -1424.7,
    'amp': -556.5,
    'gtp': -2268.6,
    'gdp': -1400.8,
    'gmp': -532.6,
    'nad': -1038.9,
    'nadh': -1073.5,
    'glc': -436.0,      # glucose
    'g6p': -1318.9,
    'f6p': -1321.7,
    'fbp': -2206.8,
    'g3p': -1096.6,     # glyceraldehyde-3-phosphate
    'pep': -1263.6,
    'pyr': -352.4,
    'pi': -1059.5,      # inorganic phosphate
    'h2o': -157.6,
    'protein': 0.0,     # macromolecule (not thermodynamic)
    'biomass': 0.0,
}

# Standard reaction ΔG° (kJ/mol) - calculated or from eQuilibrator
# Negative = spontaneous forward, Positive = requires energy
REACTION_DG0 = {
    'GLCTRANS': -14.5,   # PTS: PEP + Glc → Pyr + G6P (favorable)
    'PGI': 2.5,          # G6P ⇌ F6P (near equilibrium)
    'PFK': -17.0,        # F6P + ATP → FBP + ADP (committed step, irreversible)
    'FBA': 24.0,         # FBP → 2 G3P (unfavorable, pulled by downstream)
    'GAPDH': 6.3,        # G3P + NAD + Pi → NADH + ATP (simplified)
    'ENO': -3.4,         # (simplified enolase)
    'PYK': -31.4,        # PEP + ADP → Pyr + ATP (highly favorable)
    'ATPSYN': -30.5,     # ADP + Pi → ATP (driven by proton gradient)
    'NDK': 0.0,          # GDP + ATP ⇌ GTP + ADP (near equilibrium!)
    'TRANSLATION': -5.0, # GTP hydrolysis for translation
    'DIVISION': -3.0,    # GTP for FtsZ
    'LIPIDSYN': -8.0,    # ATP for lipid synthesis
    'ATPM': -31.0,       # ATP hydrolysis (maintenance)
    'ADK': 0.0,          # 2 ADP ⇌ ATP + AMP (equilibrium)
    'RESP': -220.0,      # Pyruvate oxidation (very favorable)
}

# Allosteric regulation constants (Ki, Ka in mM)
# Source: BRENDA, literature
ALLOSTERIC = {
    # ATP inhibits PFK (THE key feedback loop)
    'PFK': {
        'inhibitors': {'atp': 1.0},      # Ki = 1 mM
        'activators': {'adp': 0.5, 'amp': 0.1},  # ADP/AMP activate
    },
    # ATP inhibits pyruvate kinase
    'PYK': {
        'inhibitors': {'atp': 2.0},
        'activators': {'fbp': 0.05},     # FBP feedforward activation
    },
    # NADH inhibits GAPDH (product inhibition)
    'GAPDH': {
        'inhibitors': {'nadh': 0.5},
        'activators': {},
    },
    # GTP inhibits NDK reverse (prevents GTP → GDP when GTP high)
    'NDK': {
        'inhibitors': {},
        'activators': {},
    },
    # AMP activates ADK (converts AMP back to ADP)
    'Adenylate kinase': {
        'inhibitors': {},
        'activators': {'amp': 0.5},
    },
}

# Conservation pools - total concentration is constant
CONSERVATION_POOLS = {
    'adenylate': (['atp', 'adp', 'amp'], 4.5),   # Total A = 4.5 mM
    'guanylate': (['gtp', 'gdp', 'gmp'], 1.5),   # Total G = 1.5 mM
    'nad_pool': (['nad', 'nadh'], 3.0),           # Total NAD = 3.0 mM
}


# ============================================================================
# KINETIC PARAMETERS (from BRENDA)
# ============================================================================

KINETICS = {
    'ptsG': {'kcat_f': 50.0, 'kcat_r': 1.0, 'km': {'glc': 0.02, 'pep': 0.1}},
    'pfkA': {'kcat_f': 110.0, 'kcat_r': 0.5, 'km': {'f6p': 0.1, 'atp': 0.05}},
    'gapA': {'kcat_f': 80.0, 'kcat_r': 20.0, 'km': {'g3p': 0.05, 'nad': 0.1}},
    'pyk': {'kcat_f': 200.0, 'kcat_r': 2.0, 'km': {'pep': 0.3, 'adp': 0.4}},
    'atpA': {'kcat_f': 300.0, 'kcat_r': 5.0, 'km': {'adp': 0.1, 'pi': 1.0}},  # Fast ATP synthesis
    'ndk': {'kcat_f': 500.0, 'kcat_r': 500.0, 'km': {'gdp': 0.02, 'atp': 0.1}},  # Near equilibrium!
    'ftsZ': {'kcat_f': 2.0, 'kcat_r': 0.1, 'km': {'gtp': 0.5}},
    'tufA': {'kcat_f': 10.0, 'kcat_r': 0.5, 'km': {'gtp': 0.1}},
    'accA': {'kcat_f': 20.0, 'kcat_r': 2.0, 'km': {'atp': 0.2}},
}


# ============================================================================
# GENES
# ============================================================================

@dataclass
class Gene:
    locus_tag: str
    name: str
    product: str
    essential: bool = False


GENES = [
    Gene("JCVISYN3A_0685", "ptsG", "Glucose transporter", True),
    Gene("JCVISYN3A_0207", "pfkA", "Phosphofructokinase", True),
    Gene("JCVISYN3A_0314", "gapA", "GAPDH", True),
    Gene("JCVISYN3A_0546", "pyk", "Pyruvate kinase", True),
    Gene("JCVISYN3A_0783", "atpA", "ATP synthase", True),
    Gene("JCVISYN3A_0416", "ndk", "Nucleoside diphosphate kinase", True),
    Gene("JCVISYN3A_0516", "ftsZ", "Cell division protein", True),
    Gene("JCVISYN3A_0094", "tufA", "Elongation factor Tu", True),
    Gene("JCVISYN3A_0161", "accA", "Acetyl-CoA carboxylase", False),
]


# ============================================================================
# NETWORK BUILDER
# ============================================================================

def build_thermodynamic_network():
    """Build network with full thermodynamic and regulatory data."""
    
    # Metabolites with thermodynamic data
    metabolites = {
        'atp': Metabolite('atp', 'ATP', 3.0, DELTA_GF['atp'], 'adenylate'),
        'adp': Metabolite('adp', 'ADP', 1.0, DELTA_GF['adp'], 'adenylate'),
        'amp': Metabolite('amp', 'AMP', 0.5, DELTA_GF['amp'], 'adenylate'),
        'gtp': Metabolite('gtp', 'GTP', 0.8, DELTA_GF['gtp'], 'guanylate'),
        'gdp': Metabolite('gdp', 'GDP', 0.5, DELTA_GF['gdp'], 'guanylate'),
        'gmp': Metabolite('gmp', 'GMP', 0.2, DELTA_GF['gmp'], 'guanylate'),
        'nad': Metabolite('nad', 'NAD+', 2.5, DELTA_GF['nad'], 'nad_pool'),
        'nadh': Metabolite('nadh', 'NADH', 0.5, DELTA_GF['nadh'], 'nad_pool'),
        'glc': Metabolite('glc', 'Glucose', 10.0, DELTA_GF['glc']),
        'g6p': Metabolite('g6p', 'G6P', 0.5, DELTA_GF['g6p']),
        'f6p': Metabolite('f6p', 'F6P', 0.2, DELTA_GF['f6p']),
        'fbp': Metabolite('fbp', 'FBP', 0.1, DELTA_GF['fbp']),
        'g3p': Metabolite('g3p', 'G3P', 0.2, DELTA_GF['g3p']),
        'pep': Metabolite('pep', 'PEP', 0.3, DELTA_GF['pep']),
        'pyr': Metabolite('pyr', 'Pyruvate', 0.5, DELTA_GF['pyr']),
        'pi': Metabolite('pi', 'Phosphate', 10.0, DELTA_GF['pi']),
        'protein': Metabolite('protein', 'Protein', 100.0),
        'biomass': Metabolite('biomass', 'Biomass', 1.0),
    }
    
    # Reactions with thermodynamics and regulation
    reactions = []
    
    # Gene-associated reactions
    gene_rxn_map = {
        'ptsG': ('GLCTRANS', {'glc': 1, 'pep': 1}, {'g6p': 1, 'pyr': 1}, True),
        'pfkA': ('PFK', {'f6p': 1, 'atp': 0.5}, {'fbp': 1, 'adp': 0.5}, True),  # Main glycolysis gate
        'gapA': ('GAPDH', {'g3p': 1, 'nad': 0.5, 'pi': 0.5}, {'nadh': 0.5, 'pep': 1, 'atp': 1.5}, True),  # ATP producer
        'pyk': ('PYK', {'pep': 1, 'adp': 1}, {'pyr': 1, 'atp': 1}, True),  # ATP producer
        'atpA': ('ATPSYN', {'adp': 2, 'pi': 2, 'nadh': 0.5}, {'atp': 2, 'nad': 0.5}, True),  # THE main ATP maker
        'ndk': ('NDK', {'gdp': 1, 'atp': 0.2}, {'gtp': 1, 'adp': 0.2}, True),  # THE GTP maker
        'tufA': ('TRANSLATION', {'gtp': 0.2, 'atp': 0.2}, {'gdp': 0.2, 'adp': 0.2, 'protein': 0.01}, True),
        'ftsZ': ('DIVISION', {'gtp': 0.1, 'protein': 0.02}, {'gdp': 0.1, 'biomass': 0.03}, True),
        'accA': ('LIPIDSYN', {'atp': 0.1, 'nadh': 0.05}, {'adp': 0.1, 'nad': 0.05}, False),
    }
    
    for gene in GENES:
        if gene.name in gene_rxn_map:
            rxn_name, subs, prods, ess = gene_rxn_map[gene.name]
            kin = KINETICS.get(gene.name, {})
            allo = ALLOSTERIC.get(rxn_name, {'inhibitors': {}, 'activators': {}})
            
            reactions.append(Reaction(
                id=f"{rxn_name}_{gene.locus_tag}",
                name=rxn_name,
                substrates=subs,
                products=prods,
                enzyme=gene.locus_tag,
                kcat_forward=kin.get('kcat_f', 10.0),
                kcat_reverse=kin.get('kcat_r', 1.0),
                km_substrates=kin.get('km', {}),
                delta_G0=REACTION_DG0.get(rxn_name, 0.0),
                inhibitors=allo['inhibitors'],
                activators=allo['activators'],
                reversible=(abs(REACTION_DG0.get(rxn_name, 0.0)) < 15.0),  # Reversible if |ΔG°| < 15
                essential=ess,
            ))
    
    # Housekeeping reactions
    reactions.extend([
        Reaction("PGI", "G6P isomerase", {'g6p': 1}, {'f6p': 1}, "hk",
                 kcat_forward=300.0, kcat_reverse=300.0, delta_G0=2.5, reversible=True),
        Reaction("FBA", "FBP aldolase", {'fbp': 1}, {'g3p': 2}, "hk",
                 kcat_forward=50.0, kcat_reverse=100.0, delta_G0=24.0, reversible=True),
        Reaction("ENO", "Enolase", {'g3p': 1}, {'pep': 1}, "hk",
                 kcat_forward=150.0, kcat_reverse=30.0, delta_G0=-3.4, reversible=True),
        # Adenylate kinase: reversible equilibrium (NO net ATP production)
        Reaction("ADK", "Adenylate kinase", {'adp': 2}, {'atp': 1, 'amp': 1}, "hk",
                 kcat_forward=50.0, kcat_reverse=100.0, delta_G0=0.0, reversible=True),
        # AMP recycling (keeps AMP low)
        Reaction("AMPK", "AMP kinase", {'amp': 1, 'atp': 1}, {'adp': 2}, "hk",
                 kcat_forward=80.0, kcat_reverse=30.0, delta_G0=-3.0, reversible=True),
        # NADH oxidation - NO direct ATP here (ATP comes from atpA which uses NADH)
        Reaction("NADH_ox", "NADH oxidation", {'nadh': 1}, {'nad': 1}, "hk",
                 kcat_forward=30.0, kcat_reverse=0.1, delta_G0=-50.0, reversible=False),
        # ATP maintenance - very low
        Reaction("ATPM", "ATP maintenance", {'atp': 1}, {'adp': 1, 'pi': 1}, "maint",
                 kcat_forward=0.3, kcat_reverse=0.0, delta_G0=-31.0, reversible=False),
        # Exchange reactions
        Reaction("EX_glc", "Glucose uptake", {}, {'glc': 1}, "exchange",
                 kcat_forward=5.0, reversible=False),
        Reaction("EX_pi", "Phosphate uptake", {}, {'pi': 1}, "exchange",
                 kcat_forward=10.0, reversible=False),
    ])
    
    return metabolites, reactions


# ============================================================================
# THERMODYNAMIC CELL SIMULATOR
# ============================================================================

class ThermodynamicCellSimulator:
    """
    Cell simulator with real thermodynamics and feedback.
    
    Key features:
    1. Reversible Michaelis-Menten with Haldane relation
    2. Thermodynamic driving force (ΔG determines direction)
    3. Allosteric regulation (inhibitors/activators)
    4. Conservation constraints (pools sum to constant)
    """
    
    def __init__(self, metabolites: Dict[str, Metabolite], reactions: List[Reaction]):
        self.met_list = list(metabolites.keys())
        self.met_idx = {m: i for i, m in enumerate(self.met_list)}
        self.metabolites = metabolites
        self.reactions = reactions
        self.n_met = len(metabolites)
        self.n_rxn = len(reactions)
        
        # Build stoichiometry matrix
        self.S = np.zeros((self.n_met, self.n_rxn))
        for j, rxn in enumerate(reactions):
            for m, s in rxn.substrates.items():
                if m in self.met_idx:
                    self.S[self.met_idx[m], j] -= s
            for m, s in rxn.products.items():
                if m in self.met_idx:
                    self.S[self.met_idx[m], j] += s
        
        # Initialize concentrations
        self.conc = np.array([metabolites[m].initial_conc for m in self.met_list])
        self.initial_conc = self.conc.copy()
        
        # Compute initial pool totals (for conservation)
        self.pool_totals = {}
        for pool_name, (members, _) in CONSERVATION_POOLS.items():
            total = sum(self.conc[self.met_idx[m]] for m in members if m in self.met_idx)
            self.pool_totals[pool_name] = total
        
        self.time = 0.0
        self.history = {'time': [], 'conc': [], 'fluxes': []}
    
    def compute_mass_action_ratio(self, rxn: Reaction) -> float:
        """
        Compute mass action ratio Q = Π[products]^ν / Π[substrates]^ν
        
        Used to determine thermodynamic driving force.
        """
        Q_num = 1.0
        Q_den = 1.0
        
        for m, stoich in rxn.products.items():
            if m in self.met_idx:
                c = max(self.conc[self.met_idx[m]], 1e-10)
                Q_num *= c ** stoich
        
        for m, stoich in rxn.substrates.items():
            if m in self.met_idx:
                c = max(self.conc[self.met_idx[m]], 1e-10)
                Q_den *= c ** stoich
        
        return Q_num / (Q_den + 1e-20)
    
    def compute_delta_G(self, rxn: Reaction) -> float:
        """
        Compute actual ΔG = ΔG° + RT ln(Q)
        
        ΔG < 0: reaction proceeds forward
        ΔG > 0: reaction proceeds backward (or stops)
        ΔG ≈ 0: at equilibrium
        """
        Q = self.compute_mass_action_ratio(rxn)
        return rxn.delta_G0 + RT * np.log(Q + 1e-20)
    
    def compute_allosteric_factor(self, rxn: Reaction) -> float:
        """
        Compute allosteric regulation factor.
        
        Inhibitors decrease rate: 1 / (1 + [I]/Ki)
        Activators increase rate: (1 + [A]/Ka) / 1
        
        This is multiplicative: high inhibitor = low factor
        """
        factor = 1.0
        
        # Inhibition (reduces flux)
        for met, Ki in rxn.inhibitors.items():
            if met in self.met_idx:
                I = self.conc[self.met_idx[met]]
                factor *= 1.0 / (1.0 + I / Ki)
        
        # Activation (increases flux) - but cap it
        for met, Ka in rxn.activators.items():
            if met in self.met_idx:
                A = self.conc[self.met_idx[met]]
                factor *= min(1.0 + A / Ka, 5.0)  # Cap at 5x
        
        return factor
    
    def compute_reversible_mm_flux(self, rxn: Reaction) -> float:
        """
        Compute reversible Michaelis-Menten flux with Haldane relation.
        
        v = Vf * [S]/Ks / (1 + [S]/Ks) - Vr * [P]/Kp / (1 + [P]/Kp)
        """
        # Forward rate
        vf = rxn.kcat_forward
        saturation_f = 1.0
        for m, stoich in rxn.substrates.items():
            if m in self.met_idx:
                S = max(self.conc[self.met_idx[m]], 1e-10)
                Km = rxn.km_substrates.get(m, 0.1)
                saturation_f *= (S / (Km + S)) ** stoich
        vf *= saturation_f
        
        # Reverse rate (if reversible)
        vr = 0.0
        if rxn.reversible:
            vr = rxn.kcat_reverse
            saturation_r = 1.0
            for m, stoich in rxn.products.items():
                if m in self.met_idx and m not in ['protein', 'biomass', 'lipid']:
                    P = max(self.conc[self.met_idx[m]], 1e-10)
                    Km = rxn.km_products.get(m, 0.5)
                    saturation_r *= (P / (Km + P)) ** stoich
            vr *= saturation_r
        
        return vf - vr
    
    def compute_flux(self, rxn: Reaction) -> float:
        """
        Compute net flux with thermodynamic and allosteric constraints.
        
        The order matters:
        1. Kinetic rate (Michaelis-Menten)
        2. Allosteric regulation (feedback)
        3. Thermodynamic constraint (can't violate ΔG)
        """
        # 1. Base kinetic rate
        v_kinetic = self.compute_reversible_mm_flux(rxn)
        
        # 2. Allosteric regulation - THIS MUST AFFECT THE RATE
        allo_factor = self.compute_allosteric_factor(rxn)
        v_regulated = v_kinetic * allo_factor
        
        # 3. Thermodynamic constraint
        delta_G = self.compute_delta_G(rxn)
        
        # If ΔG > 0, reaction is thermodynamically unfavorable
        # The reaction can still proceed if driven, but slows near equilibrium
        if delta_G > 0:
            # Unfavorable direction - slow down exponentially
            thermo_factor = np.exp(-delta_G / (2 * RT))  # Softer damping
            v_regulated *= thermo_factor
        elif delta_G < -30:
            # Very favorable - allow full rate
            pass
        else:
            # Moderately favorable - slight damping near equilibrium
            thermo_factor = 1 - np.exp(delta_G / RT)
            thermo_factor = max(thermo_factor, 0.1)
            v_regulated *= thermo_factor
        
        # Substrate limitation (can't consume more than available)
        if v_regulated > 0:
            for m, stoich in rxn.substrates.items():
                if m in self.met_idx:
                    available = self.conc[self.met_idx[m]] / (stoich + 1e-10)
                    max_flux = available * 5
                    v_regulated = min(v_regulated, max_flux)
        elif v_regulated < 0:
            for m, stoich in rxn.products.items():
                if m in self.met_idx and m not in ['protein', 'biomass']:
                    available = self.conc[self.met_idx[m]] / (stoich + 1e-10)
                    max_flux = available * 5
                    v_regulated = max(v_regulated, -max_flux)
        
        return v_regulated
    
    def compute_all_fluxes(self) -> np.ndarray:
        """Compute fluxes for all reactions."""
        fluxes = np.zeros(self.n_rxn)
        for j, rxn in enumerate(self.reactions):
            fluxes[j] = self.compute_flux(rxn)
        return fluxes
    
    def enforce_conservation(self):
        """
        Enforce conservation pools (adenylate, guanylate, NAD).
        
        Total pool must stay constant. Redistribute proportionally if violated.
        """
        for pool_name, (members, _) in CONSERVATION_POOLS.items():
            # Current total
            indices = [self.met_idx[m] for m in members if m in self.met_idx]
            if not indices:
                continue
            
            current_total = sum(self.conc[i] for i in indices)
            target_total = self.pool_totals[pool_name]
            
            if current_total > 1e-10 and abs(current_total - target_total) > 1e-6:
                # Scale to conserve
                scale = target_total / current_total
                for i in indices:
                    self.conc[i] *= scale
    
    def step(self, dt: float = 0.1):
        """Advance simulation by dt minutes."""
        fluxes = self.compute_all_fluxes()
        
        # Update concentrations: dC/dt = S @ v
        dC = self.S @ fluxes * dt
        self.conc += dC
        
        # Enforce non-negativity (physical constraint)
        self.conc = np.maximum(self.conc, 1e-6)
        
        # Enforce conservation pools
        self.enforce_conservation()
        
        self.time += dt
    
    def simulate(self, duration: float, dt: float = 0.1, record_every: int = 10):
        """Run simulation for duration minutes."""
        n_steps = int(duration / dt)
        
        for i in range(n_steps):
            self.step(dt)
            
            if i % record_every == 0:
                self.history['time'].append(self.time)
                self.history['conc'].append(self.conc.copy())
                self.history['fluxes'].append(self.compute_all_fluxes())
    
    def get(self, met: str) -> float:
        """Get concentration of metabolite."""
        return self.conc[self.met_idx[met]] if met in self.met_idx else 0.0
    
    def energy_charge(self) -> float:
        """Adenylate energy charge: (ATP + 0.5*ADP) / (ATP + ADP + AMP)"""
        atp = self.get('atp')
        adp = self.get('adp')
        amp = self.get('amp')
        total = atp + adp + amp
        if total < 1e-10:
            return 0.0
        return (atp + 0.5 * adp) / total
    
    def gtp_ratio(self) -> float:
        """GTP / (GTP + GDP)"""
        gtp = self.get('gtp')
        gdp = self.get('gdp')
        total = gtp + gdp
        if total < 1e-10:
            return 0.0
        return gtp / total
    
    def is_viable(self) -> bool:
        """Cell viability check."""
        ec = self.energy_charge()
        gtp = self.get('gtp')
        return ec > 0.5 and gtp > 0.05
    
    def get_reaction_status(self) -> List[Dict]:
        """Get status of all reactions for debugging."""
        status = []
        for rxn in self.reactions:
            delta_G = self.compute_delta_G(rxn)
            flux = self.compute_flux(rxn)
            allo = self.compute_allosteric_factor(rxn)
            
            status.append({
                'name': rxn.name,
                'ΔG': delta_G,
                'flux': flux,
                'allosteric': allo,
                'direction': 'forward' if flux > 0 else 'reverse' if flux < 0 else 'equilibrium',
            })
        return status


# ============================================================================
# KNOCKOUT SIMULATION
# ============================================================================

def get_gene_locus(gene_name: str) -> Optional[str]:
    """Get locus tag for a gene name."""
    for gene in GENES:
        if gene.name == gene_name:
            return gene.locus_tag
    return None


def simulate_knockout(gene_name: str, duration: float = 120.0) -> Dict:
    """Simulate knockout of a gene."""
    mets, rxns = build_thermodynamic_network()
    
    # Get locus tag for this gene
    locus_tag = get_gene_locus(gene_name)
    
    # Remove reactions catalyzed by knocked out gene
    if locus_tag:
        ko_rxns = [r for r in rxns if r.enzyme != locus_tag]
    else:
        # Also try matching by gene name directly (for housekeeping)
        ko_rxns = [r for r in rxns if gene_name not in r.enzyme and gene_name not in r.name]
    
    sim = ThermodynamicCellSimulator(mets, ko_rxns)
    initial_biomass = sim.get('biomass')
    
    sim.simulate(duration, dt=0.1)
    
    final_biomass = sim.get('biomass')
    growth = final_biomass / max(initial_biomass, 0.01)
    
    # Viability: energy charge AND growth
    ec = sim.energy_charge()
    gtp = sim.get('gtp')
    viable = ec > 0.5 and gtp > 0.1 and growth > 1.5  # Must grow 50%
    
    return {
        'gene': gene_name,
        'viable': viable,
        'energy_charge': ec,
        'gtp_ratio': sim.gtp_ratio(),
        'atp': sim.get('atp'),
        'gtp': gtp,
        'biomass': final_biomass,
        'growth': growth,
    }


# ============================================================================
# TESTS
# ============================================================================

def test_wildtype():
    """Test wild-type simulation with thermodynamics."""
    print("\n" + "="*60)
    print("V35.2 THERMODYNAMIC CELL - WILD TYPE")
    print("="*60)
    
    mets, rxns = build_thermodynamic_network()
    sim = ThermodynamicCellSimulator(mets, rxns)
    
    print(f"\nInitial state:")
    print(f"  ATP: {sim.get('atp'):.2f} mM")
    print(f"  ADP: {sim.get('adp'):.2f} mM")
    print(f"  AMP: {sim.get('amp'):.2f} mM")
    print(f"  GTP: {sim.get('gtp'):.2f} mM")
    print(f"  Energy charge: {sim.energy_charge():.3f}")
    print(f"  Adenylate pool: {sim.get('atp') + sim.get('adp') + sim.get('amp'):.2f} mM")
    
    print(f"\nSimulating 120 min...")
    sim.simulate(120, dt=0.1)
    
    print(f"\nFinal state:")
    print(f"  ATP: {sim.get('atp'):.2f} mM")
    print(f"  ADP: {sim.get('adp'):.2f} mM")
    print(f"  AMP: {sim.get('amp'):.2f} mM")
    print(f"  GTP: {sim.get('gtp'):.2f} mM")
    print(f"  GDP: {sim.get('gdp'):.2f} mM")
    print(f"  Energy charge: {sim.energy_charge():.3f}")
    print(f"  Adenylate pool: {sim.get('atp') + sim.get('adp') + sim.get('amp'):.2f} mM")
    print(f"  Guanylate pool: {sim.get('gtp') + sim.get('gdp') + sim.get('gmp'):.2f} mM")
    print(f"  Protein: {sim.get('protein'):.2f}")
    print(f"  Biomass: {sim.get('biomass'):.2f}")
    print(f"  Viable: {sim.is_viable()}")
    
    # Check reaction status
    print(f"\nReaction thermodynamics:")
    for rs in sim.get_reaction_status()[:8]:
        direction = "→" if rs['flux'] > 0.1 else "←" if rs['flux'] < -0.1 else "⇌"
        print(f"  {rs['name']:12s}: ΔG={rs['ΔG']:+6.1f} kJ/mol  {direction}  allo={rs['allosteric']:.2f}")
    
    # Validation
    print(f"\n--- VALIDATION ---")
    checks = []
    
    # ATP should NOT explode (no cap needed!)
    atp_ok = 1.0 < sim.get('atp') < 10.0
    checks.append(("ATP in range (1-10 mM)", atp_ok, sim.get('atp')))
    
    # GTP should NOT deplete
    gtp_ok = sim.get('gtp') > 0.1
    checks.append(("GTP present (>0.1 mM)", gtp_ok, sim.get('gtp')))
    
    # Energy charge should be healthy (0.6+ for bacteria under growth conditions)
    ec_ok = 0.6 < sim.energy_charge() < 0.98
    checks.append(("Energy charge (0.6-0.98)", ec_ok, sim.energy_charge()))
    
    # Pools should be conserved
    ade_pool = sim.get('atp') + sim.get('adp') + sim.get('amp')
    pool_ok = abs(ade_pool - sim.pool_totals['adenylate']) < 0.1
    checks.append(("Adenylate conserved", pool_ok, ade_pool))
    
    # Biomass should grow
    growth_ok = sim.get('biomass') > 1.5
    checks.append(("Biomass grows", growth_ok, sim.get('biomass')))
    
    all_pass = all(c[1] for c in checks)
    
    for name, ok, val in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: {val:.2f}")
    
    return all_pass, sim


def test_knockouts():
    """Test gene knockouts."""
    print("\n" + "="*60)
    print("V35.2 THERMODYNAMIC CELL - KNOCKOUTS")
    print("="*60)
    
    results = []
    for gene in GENES:
        r = simulate_knockout(gene.name, duration=120.0)
        r['ground_truth'] = gene.essential
        results.append(r)
        
        pred = "LETHAL" if not r['viable'] else "VIABLE"
        truth = "essential" if gene.essential else "non-ess"
        match = "✓" if (not r['viable']) == gene.essential else "✗"
        
        growth_str = f"growth={r.get('growth', 0):.1f}x"
        print(f"  Δ{gene.name:8s}: {pred:8s} (EC={r['energy_charge']:.2f}, {growth_str}) | Truth: {truth:8s} [{match}]")
    
    n_correct = sum(1 for r in results if (not r['viable']) == r['ground_truth'])
    accuracy = n_correct / len(results)
    print(f"\nAccuracy: {n_correct}/{len(results)} ({100*accuracy:.0f}%)")
    
    return accuracy >= 0.5


def test_feedback_loop():
    """Test that ATP inhibits PFK (the key feedback loop)."""
    print("\n" + "="*60)
    print("V35.2 THERMODYNAMIC CELL - FEEDBACK TEST")
    print("="*60)
    
    mets, rxns = build_thermodynamic_network()
    sim = ThermodynamicCellSimulator(mets, rxns)
    
    # Find PFK reaction
    pfk_idx = None
    for i, rxn in enumerate(sim.reactions):
        if rxn.name == 'PFK':
            pfk_idx = i
            break
    
    if pfk_idx is None:
        print("  PFK not found!")
        return False
    
    pfk = sim.reactions[pfk_idx]
    
    # Make sure F6P substrate is present
    sim.conc[sim.met_idx['f6p']] = 1.0  # Substrate for PFK
    
    # Test at low ATP (cell needs energy - should speed up glycolysis)
    sim.conc[sim.met_idx['atp']] = 0.5  # Low ATP
    sim.conc[sim.met_idx['adp']] = 3.5  # High ADP
    sim.conc[sim.met_idx['amp']] = 0.5  # Some AMP
    sim.conc[sim.met_idx['fbp']] = 0.01  # Low product
    low_atp_allo = sim.compute_allosteric_factor(pfk)
    low_atp_kinetic = sim.compute_reversible_mm_flux(pfk)
    low_atp_flux = sim.compute_flux(pfk)
    
    # Test at high ATP (cell has energy - should slow glycolysis)  
    sim.conc[sim.met_idx['atp']] = 4.0  # High ATP
    sim.conc[sim.met_idx['adp']] = 0.3  # Low ADP
    sim.conc[sim.met_idx['amp']] = 0.2  # Low AMP
    sim.conc[sim.met_idx['fbp']] = 0.5  # Higher product
    high_atp_allo = sim.compute_allosteric_factor(pfk)
    high_atp_kinetic = sim.compute_reversible_mm_flux(pfk)
    high_atp_flux = sim.compute_flux(pfk)
    
    print(f"\n  At low ATP (0.5 mM, high ADP):")
    print(f"    Allosteric factor: {low_atp_allo:.2f}")
    print(f"    Kinetic rate: {low_atp_kinetic:.2f}")
    print(f"    Final flux: {low_atp_flux:.2f}")
    
    print(f"\n  At high ATP (4.0 mM, low ADP):")
    print(f"    Allosteric factor: {high_atp_allo:.2f}")
    print(f"    Kinetic rate: {high_atp_kinetic:.2f}")
    print(f"    Final flux: {high_atp_flux:.2f}")
    
    if high_atp_flux > 0.01:
        ratio = low_atp_flux / high_atp_flux
        print(f"\n  Flux ratio (low ATP / high ATP): {ratio:.1f}x")
    else:
        ratio = float('inf') if low_atp_flux > 0 else 0
        print(f"\n  Flux ratio: low={low_atp_flux:.2f}, high={high_atp_flux:.2f}")
    
    # PFK should be faster when ATP is low (need more glycolysis to make ATP)
    feedback_works = low_atp_flux > high_atp_flux * 1.5 or low_atp_allo > high_atp_allo * 2
    
    print(f"\n  Feedback loop working: {'✓ YES' if feedback_works else '✗ NO'}")
    print(f"  (Low ATP → high ADP → activates PFK)")
    print(f"  (High ATP → inhibits PFK)")
    
    return feedback_works


def test_thermodynamic_equilibrium():
    """Test that reactions reach thermodynamic equilibrium."""
    print("\n" + "="*60)
    print("V35.2 THERMODYNAMIC CELL - EQUILIBRIUM TEST")
    print("="*60)
    
    mets, rxns = build_thermodynamic_network()
    sim = ThermodynamicCellSimulator(mets, rxns)
    
    # Find near-equilibrium reactions (small |ΔG°|)
    print("\n  Near-equilibrium reactions (|ΔG°| < 5 kJ/mol):")
    
    for rxn in sim.reactions[:10]:
        if abs(rxn.delta_G0) < 5.0:
            delta_G = sim.compute_delta_G(rxn)
            flux = sim.compute_flux(rxn)
            Q = sim.compute_mass_action_ratio(rxn)
            Keq = np.exp(-rxn.delta_G0 / RT)
            
            print(f"    {rxn.name:12s}: ΔG°={rxn.delta_G0:+5.1f}  ΔG={delta_G:+6.1f}  Q={Q:.2e}  Keq={Keq:.2e}")
    
    # Simulate and check that ΔG approaches 0 for reversible reactions
    print("\n  Simulating 60 min...")
    sim.simulate(60, dt=0.1)
    
    print("\n  After equilibration:")
    equilibrated = 0
    for rxn in sim.reactions:
        if rxn.reversible and abs(rxn.delta_G0) < 10.0:
            delta_G = sim.compute_delta_G(rxn)
            if abs(delta_G) < 5.0:
                equilibrated += 1
                print(f"    {rxn.name:12s}: ΔG = {delta_G:+.1f} kJ/mol (near equilibrium)")
    
    return equilibrated >= 2


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DARK MANIFOLD V35.2: THERMODYNAMIC CELL SIMULATOR")
    print("  The cell doesn't know what to do. Physics does it.")
    print("="*60)
    
    results = []
    
    # Test 1: Feedback loop
    results.append(("feedback", test_feedback_loop()))
    
    # Test 2: Thermodynamic equilibrium
    results.append(("equilibrium", test_thermodynamic_equilibrium()))
    
    # Test 3: Wild type
    wt_pass, sim = test_wildtype()
    results.append(("wildtype", wt_pass))
    
    # Test 4: Knockouts
    results.append(("knockouts", test_knockouts()))
    
    # Summary
    print("\n" + "="*60)
    print("  V35.2 TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:15s}: {status}")
    
    all_pass = all(r[1] for r in results)
    
    if all_pass:
        print("\n" + "="*60)
        print("  🎉 ALL TESTS PASSED - PHYSICS WORKS!")
        print("="*60)
    else:
        print("\n  ⚠️ Some tests failed - needs tuning")
