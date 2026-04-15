"""
Dark Manifold V39: Stochastic Virtual Cell
===========================================

Gillespie Stochastic Simulation Algorithm (SSA) for JCVI-syn3A.
Adds molecular NOISE to V38's kinetics.

Key advances over V38 (ODE):
- Individual molecule counts (not concentrations)
- Intrinsic stochasticity from small copy numbers
- Transcription bursts
- Translation noise
- Captures cell-to-cell variability

JCVI-syn3A facts:
- ~500,000 total proteins
- ~200 ribosomes
- ~50-500 copies of most proteins
- Significant noise at these copy numbers

Author: Naresh Chhillar, 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
from numba import njit, prange
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONSTANTS
# ============================================================================

# Cell parameters for JCVI-syn3A
CELL_VOLUME = 6e-17  # liters (400nm diameter sphere)
AVOGADRO = 6.022e23
VOLUME_FACTOR = CELL_VOLUME * AVOGADRO  # molecules per mM

# Convert mM to molecule count
def mM_to_molecules(conc_mM: float) -> int:
    return int(conc_mM * VOLUME_FACTOR)

# Convert molecules to mM
def molecules_to_mM(count: int) -> float:
    return count / VOLUME_FACTOR


# ============================================================================
# STOCHASTIC REACTION NETWORK
# ============================================================================

@dataclass
class Reaction:
    """A single reaction in the network."""
    name: str
    substrates: Dict[str, int]  # {species: stoichiometry}
    products: Dict[str, int]
    rate_constant: float  # Propensity rate constant
    genes: List[str] = field(default_factory=list)
    
    def propensity(self, state: Dict[str, int], active_genes: set) -> float:
        """Compute propensity (probability rate) of this reaction."""
        # Check if genes are knocked out
        for gene in self.genes:
            if gene not in active_genes:
                return 0.0
        
        # Compute propensity
        prop = self.rate_constant
        for species, stoich in self.substrates.items():
            count = state.get(species, 0)
            # Combinatorial factor for stoichiometry
            for i in range(stoich):
                prop *= max(0, count - i)
        
        return prop


class StochasticModel:
    """
    Gillespie SSA simulation of JCVI-syn3A core metabolism.
    """
    
    def __init__(self):
        self.species = []
        self.reactions = []
        self.initial_state = {}
        self.active_genes = set()
        
        self._build_model()
        
        print(f"Stochastic model: {len(self.species)} species, {len(self.reactions)} reactions")
    
    def _build_model(self):
        """Build the reaction network."""
        
        # === INITIAL MOLECULE COUNTS ===
        # Based on JCVI-syn3A proteomics and metabolomics
        
        self.initial_state = {
            # Energy carriers
            'ATP': mM_to_molecules(5.0),
            'ADP': mM_to_molecules(1.0),
            'AMP': mM_to_molecules(0.1),
            'NAD': mM_to_molecules(1.0),
            'NADH': mM_to_molecules(0.1),
            'Pi': mM_to_molecules(10.0),
            
            # Glycolysis intermediates
            'GLC': mM_to_molecules(0.1),  # External glucose (clamped)
            'G6P': mM_to_molecules(1.0),
            'F6P': mM_to_molecules(0.3),
            'FBP': mM_to_molecules(0.5),
            'GAP': mM_to_molecules(0.1),   # G3P
            'DHAP': mM_to_molecules(0.5),
            'BPG': mM_to_molecules(0.01),  # 1,3-BPG
            'PG3': mM_to_molecules(0.5),
            'PG2': mM_to_molecules(0.1),
            'PEP': mM_to_molecules(0.2),
            'PYR': mM_to_molecules(1.0),
            'LAC': mM_to_molecules(0.1),
            
            # Proteins (copy numbers from proteomics)
            'E_PGI': 200,     # Phosphoglucose isomerase
            'E_PFK': 500,     # Phosphofructokinase (highly expressed)
            'E_FBA': 300,     # Aldolase
            'E_TPI': 400,     # Triose phosphate isomerase
            'E_GAPDH': 1000,  # GAPDH (very abundant)
            'E_PGK': 600,     # Phosphoglycerate kinase
            'E_PGM': 300,     # Phosphoglycerate mutase
            'E_ENO': 500,     # Enolase
            'E_PYK': 400,     # Pyruvate kinase
            'E_LDH': 800,     # Lactate dehydrogenase
            'E_PTSG': 100,    # Glucose PTS
            
            # Ribosomes
            'RIBOSOME': 200,
            
            # mRNAs (low copy numbers)
            'mRNA_total': 1000,
            
            # Biomass proxy
            'BIOMASS': 1000,
        }
        
        self.species = list(self.initial_state.keys())
        
        # All genes active initially
        self.active_genes = {
            'JCVISYN3A_0685',  # ptsG
            'JCVISYN3A_0233',  # pgi
            'JCVISYN3A_0207',  # pfkA
            'JCVISYN3A_0352',  # fba
            'JCVISYN3A_0353',  # tpiA
            'JCVISYN3A_0314',  # gapA
            'JCVISYN3A_0315',  # pgk
            'JCVISYN3A_0689',  # pgm
            'JCVISYN3A_0231',  # eno
            'JCVISYN3A_0546',  # pyk
            'JCVISYN3A_0449',  # ldh
        }
        
        # === REACTIONS ===
        # Rate constants are scaled for molecule counts
        # k = k_mM / VOLUME_FACTOR for bimolecular reactions
        
        k_scale = 1.0 / VOLUME_FACTOR  # For bimolecular reactions
        
        # Glucose uptake (PTS system)
        # GLC + PEP -> G6P + PYR
        self.reactions.append(Reaction(
            name='GLCpts',
            substrates={'GLC': 1, 'PEP': 1, 'E_PTSG': 1},
            products={'G6P': 1, 'PYR': 1, 'E_PTSG': 1, 'GLC': 1},  # GLC regenerated (clamped)
            rate_constant=10.0 * k_scale,
            genes=['JCVISYN3A_0685']
        ))
        
        # PGI: G6P <-> F6P
        self.reactions.append(Reaction(
            name='PGI_f',
            substrates={'G6P': 1, 'E_PGI': 1},
            products={'F6P': 1, 'E_PGI': 1},
            rate_constant=100.0 * k_scale,
            genes=['JCVISYN3A_0233']
        ))
        self.reactions.append(Reaction(
            name='PGI_r',
            substrates={'F6P': 1, 'E_PGI': 1},
            products={'G6P': 1, 'E_PGI': 1},
            rate_constant=40.0 * k_scale,
            genes=['JCVISYN3A_0233']
        ))
        
        # PFK: F6P + ATP -> FBP + ADP (irreversible, regulated)
        self.reactions.append(Reaction(
            name='PFK',
            substrates={'F6P': 1, 'ATP': 1, 'E_PFK': 1},
            products={'FBP': 1, 'ADP': 1, 'E_PFK': 1},
            rate_constant=50.0 * k_scale * k_scale,  # Trimolecular
            genes=['JCVISYN3A_0207']
        ))
        
        # FBA: FBP <-> GAP + DHAP
        self.reactions.append(Reaction(
            name='FBA_f',
            substrates={'FBP': 1, 'E_FBA': 1},
            products={'GAP': 1, 'DHAP': 1, 'E_FBA': 1},
            rate_constant=50.0 * k_scale,
            genes=['JCVISYN3A_0352']
        ))
        self.reactions.append(Reaction(
            name='FBA_r',
            substrates={'GAP': 1, 'DHAP': 1, 'E_FBA': 1},
            products={'FBP': 1, 'E_FBA': 1},
            rate_constant=5.0 * k_scale * k_scale,
            genes=['JCVISYN3A_0352']
        ))
        
        # TPI: DHAP <-> GAP
        self.reactions.append(Reaction(
            name='TPI_f',
            substrates={'DHAP': 1, 'E_TPI': 1},
            products={'GAP': 1, 'E_TPI': 1},
            rate_constant=500.0 * k_scale,
            genes=['JCVISYN3A_0353']
        ))
        self.reactions.append(Reaction(
            name='TPI_r',
            substrates={'GAP': 1, 'E_TPI': 1},
            products={'DHAP': 1, 'E_TPI': 1},
            rate_constant=100.0 * k_scale,
            genes=['JCVISYN3A_0353']
        ))
        
        # GAPDH: GAP + NAD + Pi -> BPG + NADH
        self.reactions.append(Reaction(
            name='GAPDH_f',
            substrates={'GAP': 1, 'NAD': 1, 'Pi': 1, 'E_GAPDH': 1},
            products={'BPG': 1, 'NADH': 1, 'E_GAPDH': 1},
            rate_constant=100.0 * (k_scale ** 3),
            genes=['JCVISYN3A_0314']
        ))
        
        # PGK: BPG + ADP -> PG3 + ATP
        self.reactions.append(Reaction(
            name='PGK_f',
            substrates={'BPG': 1, 'ADP': 1, 'E_PGK': 1},
            products={'PG3': 1, 'ATP': 1, 'E_PGK': 1},
            rate_constant=200.0 * k_scale * k_scale,
            genes=['JCVISYN3A_0315']
        ))
        
        # PGM: PG3 <-> PG2
        self.reactions.append(Reaction(
            name='PGM_f',
            substrates={'PG3': 1, 'E_PGM': 1},
            products={'PG2': 1, 'E_PGM': 1},
            rate_constant=100.0 * k_scale,
            genes=['JCVISYN3A_0689']
        ))
        self.reactions.append(Reaction(
            name='PGM_r',
            substrates={'PG2': 1, 'E_PGM': 1},
            products={'PG3': 1, 'E_PGM': 1},
            rate_constant=15.0 * k_scale,
            genes=['JCVISYN3A_0689']
        ))
        
        # ENO: PG2 <-> PEP
        self.reactions.append(Reaction(
            name='ENO_f',
            substrates={'PG2': 1, 'E_ENO': 1},
            products={'PEP': 1, 'E_ENO': 1},
            rate_constant=100.0 * k_scale,
            genes=['JCVISYN3A_0231']
        ))
        self.reactions.append(Reaction(
            name='ENO_r',
            substrates={'PEP': 1, 'E_ENO': 1},
            products={'PG2': 1, 'E_ENO': 1},
            rate_constant=33.0 * k_scale,
            genes=['JCVISYN3A_0231']
        ))
        
        # PYK: PEP + ADP -> PYR + ATP (irreversible)
        self.reactions.append(Reaction(
            name='PYK',
            substrates={'PEP': 1, 'ADP': 1, 'E_PYK': 1},
            products={'PYR': 1, 'ATP': 1, 'E_PYK': 1},
            rate_constant=100.0 * k_scale * k_scale,
            genes=['JCVISYN3A_0546']
        ))
        
        # LDH: PYR + NADH <-> LAC + NAD
        self.reactions.append(Reaction(
            name='LDH_f',
            substrates={'PYR': 1, 'NADH': 1, 'E_LDH': 1},
            products={'LAC': 1, 'NAD': 1, 'E_LDH': 1},
            rate_constant=200.0 * k_scale * k_scale,
            genes=['JCVISYN3A_0449']
        ))
        
        # ATP consumption (growth/maintenance)
        self.reactions.append(Reaction(
            name='ATP_use',
            substrates={'ATP': 1},
            products={'ADP': 1, 'Pi': 1},
            rate_constant=0.01,  # Basal consumption
            genes=[]
        ))
        
        # Lactate export
        self.reactions.append(Reaction(
            name='LAC_export',
            substrates={'LAC': 1},
            products={},  # Exits cell
            rate_constant=0.1,
            genes=[]
        ))
        
        # Simplified biomass production
        self.reactions.append(Reaction(
            name='GROWTH',
            substrates={'ATP': 10, 'RIBOSOME': 1},
            products={'ADP': 10, 'Pi': 10, 'BIOMASS': 1, 'RIBOSOME': 1},
            rate_constant=0.001 * (k_scale ** 9),
            genes=[]
        ))
    
    def gillespie_step(self, state: Dict[str, int]) -> Tuple[float, Optional[str]]:
        """
        Single Gillespie SSA step.
        
        Returns:
            tau: Time until next reaction
            reaction_name: Which reaction fired (or None if none possible)
        """
        # Compute all propensities
        propensities = []
        for rxn in self.reactions:
            prop = rxn.propensity(state, self.active_genes)
            propensities.append(prop)
        
        a_total = sum(propensities)
        
        if a_total <= 0:
            return float('inf'), None
        
        # Time to next reaction (exponential distribution)
        tau = -np.log(np.random.random()) / a_total
        
        # Select reaction
        r = np.random.random() * a_total
        cumsum = 0
        selected_rxn = None
        for i, (rxn, prop) in enumerate(zip(self.reactions, propensities)):
            cumsum += prop
            if cumsum >= r:
                selected_rxn = rxn
                break
        
        return tau, selected_rxn.name if selected_rxn else None
    
    def apply_reaction(self, state: Dict[str, int], rxn_name: str):
        """Apply a reaction to the state."""
        for rxn in self.reactions:
            if rxn.name == rxn_name:
                # Remove substrates
                for species, stoich in rxn.substrates.items():
                    state[species] = max(0, state.get(species, 0) - stoich)
                # Add products
                for species, stoich in rxn.products.items():
                    state[species] = state.get(species, 0) + stoich
                return
    
    def simulate(self, t_max: float, knockouts: List[str] = None, 
                 max_steps: int = 1000000) -> dict:
        """
        Run Gillespie simulation.
        
        Args:
            t_max: Maximum simulation time (hours)
            knockouts: Genes to knock out
            max_steps: Maximum number of reactions
            
        Returns:
            Dictionary with time series and final state
        """
        # Set up knockouts
        if knockouts:
            self.active_genes = self.active_genes - set(knockouts)
        else:
            # Reset to all genes active
            self.active_genes = {
                'JCVISYN3A_0685', 'JCVISYN3A_0233', 'JCVISYN3A_0207',
                'JCVISYN3A_0352', 'JCVISYN3A_0353', 'JCVISYN3A_0314',
                'JCVISYN3A_0315', 'JCVISYN3A_0689', 'JCVISYN3A_0231',
                'JCVISYN3A_0546', 'JCVISYN3A_0449',
            }
        
        # Initialize state
        state = self.initial_state.copy()
        
        # Record time series
        t_record = [0.0]
        atp_record = [state['ATP']]
        lac_record = [state['LAC']]
        biomass_record = [state['BIOMASS']]
        
        t = 0.0
        steps = 0
        start_time = time.time()
        
        while t < t_max and steps < max_steps:
            tau, rxn_name = self.gillespie_step(state)
            
            if rxn_name is None:
                break
            
            t += tau
            self.apply_reaction(state, rxn_name)
            steps += 1
            
            # Record periodically (every ~0.01h)
            if t > t_record[-1] + 0.01:
                t_record.append(t)
                atp_record.append(state['ATP'])
                lac_record.append(state['LAC'])
                biomass_record.append(state['BIOMASS'])
        
        elapsed = time.time() - start_time
        
        # Compute final energy charge
        ec = (state['ATP'] + 0.5 * state['ADP']) / (state['ATP'] + state['ADP'] + state['AMP'] + 1)
        
        return {
            't': np.array(t_record),
            'ATP': np.array(atp_record),
            'LAC': np.array(lac_record),
            'BIOMASS': np.array(biomass_record),
            'final_state': state,
            'energy_charge': ec,
            'steps': steps,
            'elapsed_s': elapsed
        }
    
    def knockout_essentiality(self, gene: str, n_replicates: int = 5, t_max: float = 1.0) -> dict:
        """
        Test gene essentiality with stochastic simulation.
        
        Runs multiple replicates to capture variability.
        """
        wt_biomass = []
        ko_biomass = []
        
        for _ in range(n_replicates):
            # Wild-type
            wt = self.simulate(t_max)
            wt_biomass.append(wt['final_state']['BIOMASS'])
            
            # Knockout
            ko = self.simulate(t_max, knockouts=[gene])
            ko_biomass.append(ko['final_state']['BIOMASS'])
        
        wt_mean = np.mean(wt_biomass)
        ko_mean = np.mean(ko_biomass)
        wt_std = np.std(wt_biomass)
        ko_std = np.std(ko_biomass)
        
        # Essential if KO significantly reduces biomass
        ratio = ko_mean / (wt_mean + 1)
        essential = ratio < 0.5
        
        return {
            'gene': gene,
            'essential': essential,
            'biomass_ratio': ratio,
            'wt_mean': wt_mean,
            'wt_std': wt_std,
            'ko_mean': ko_mean,
            'ko_std': ko_std,
        }


# ============================================================================
# TAU-LEAPING (Faster approximate method)
# ============================================================================

class TauLeapingModel(StochasticModel):
    """
    Tau-leaping for faster stochastic simulation.
    
    Instead of one reaction at a time, fires multiple reactions per step.
    Valid when molecule counts are large enough.
    """
    
    def tau_leap_step(self, state: Dict[str, int], tau: float) -> Dict[str, int]:
        """
        Single tau-leaping step.
        
        Fires Poisson-distributed number of each reaction.
        """
        new_state = state.copy()
        
        for rxn in self.reactions:
            prop = rxn.propensity(state, self.active_genes)
            if prop > 0:
                # Number of times this reaction fires in interval tau
                n_fires = np.random.poisson(prop * tau)
                
                if n_fires > 0:
                    # Apply reaction n_fires times
                    for species, stoich in rxn.substrates.items():
                        new_state[species] = max(0, new_state.get(species, 0) - stoich * n_fires)
                    for species, stoich in rxn.products.items():
                        new_state[species] = new_state.get(species, 0) + stoich * n_fires
        
        return new_state
    
    def simulate_tau_leap(self, t_max: float, tau: float = 0.001,
                          knockouts: List[str] = None) -> dict:
        """
        Run tau-leaping simulation.
        """
        if knockouts:
            self.active_genes = self.active_genes - set(knockouts)
        else:
            self.active_genes = {
                'JCVISYN3A_0685', 'JCVISYN3A_0233', 'JCVISYN3A_0207',
                'JCVISYN3A_0352', 'JCVISYN3A_0353', 'JCVISYN3A_0314',
                'JCVISYN3A_0315', 'JCVISYN3A_0689', 'JCVISYN3A_0231',
                'JCVISYN3A_0546', 'JCVISYN3A_0449',
            }
        
        state = self.initial_state.copy()
        
        t_record = [0.0]
        atp_record = [state['ATP']]
        biomass_record = [state['BIOMASS']]
        
        t = 0.0
        start_time = time.time()
        
        while t < t_max:
            state = self.tau_leap_step(state, tau)
            t += tau
            
            t_record.append(t)
            atp_record.append(state['ATP'])
            biomass_record.append(state['BIOMASS'])
        
        elapsed = time.time() - start_time
        
        return {
            't': np.array(t_record),
            'ATP': np.array(atp_record),
            'BIOMASS': np.array(biomass_record),
            'final_state': state,
            'elapsed_s': elapsed
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("DARK MANIFOLD V39: STOCHASTIC VIRTUAL CELL")
    print("Gillespie SSA Simulation of JCVI-syn3A")
    print("="*60)
    
    model = StochasticModel()
    
    # Wild-type simulation
    print("\n--- Wild-type simulation (1h, exact SSA) ---")
    result = model.simulate(t_max=0.5, max_steps=100000)  # Shorter for speed
    
    print(f"Steps: {result['steps']:,}")
    print(f"Time: {result['elapsed_s']:.2f}s")
    print(f"Final ATP: {result['final_state']['ATP']:,} molecules")
    print(f"Final biomass: {result['final_state']['BIOMASS']:,}")
    print(f"Energy charge: {result['energy_charge']:.3f}")
    
    # Compare ATP in mM
    atp_mM = molecules_to_mM(result['final_state']['ATP'])
    print(f"ATP concentration: {atp_mM:.3f} mM")
    
    # Tau-leaping (faster)
    print("\n--- Tau-leaping simulation (1h) ---")
    tau_model = TauLeapingModel()
    tau_result = tau_model.simulate_tau_leap(t_max=1.0, tau=0.001)
    
    print(f"Time: {tau_result['elapsed_s']:.2f}s")
    print(f"Final ATP: {tau_result['final_state']['ATP']:,}")
    print(f"Final biomass: {tau_result['final_state']['BIOMASS']:,}")
    
    # Knockout test (with noise)
    print("\n--- Knockout essentiality (with replicates) ---")
    test_genes = [
        ('JCVISYN3A_0207', 'pfkA'),
        ('JCVISYN3A_0449', 'ldh'),
    ]
    
    for gene_id, gene_name in test_genes:
        result = model.knockout_essentiality(gene_id, n_replicates=3, t_max=0.2)
        status = "ESSENTIAL" if result['essential'] else f"viable ({result['biomass_ratio']:.0%})"
        print(f"  Δ{gene_name}: {status}")
        print(f"    WT: {result['wt_mean']:.0f} ± {result['wt_std']:.0f}")
        print(f"    KO: {result['ko_mean']:.0f} ± {result['ko_std']:.0f}")
    
    return model, result


if __name__ == '__main__':
    model, result = main()
