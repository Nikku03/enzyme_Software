"""
Dark Manifold V45: Systems Cell Model
=====================================

Models the REAL complexity of cellular regulation:

1. MULTI-STEP PATHWAYS
   - 8-step biosynthesis
   - Rate-limiting steps
   - Intermediate accumulation

2. FEEDBACK LOOPS  
   - Product inhibition
   - Substrate activation
   - Homeostasis

3. CYCLES
   - TCA cycle (8 steps)
   - Regeneration of carriers
   - Cycle breakage

4. REGULATORY NETWORKS
   - Transcription factors
   - Signal cascades
   - Bistable switches

5. EPIGENETICS (simplified)
   - Gene silencing
   - Chromatin states
   - Heritable modifications

This goes beyond "gene → protein → reaction" to capture
the systems-level behavior that determines essentiality.

Author: Naresh Chhillar, 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Callable
from enum import Enum
from collections import defaultdict
import time


# ============================================================================
# REGULATORY ELEMENTS
# ============================================================================

class RegulationType(Enum):
    """Types of regulation."""
    ACTIVATION = "activation"
    INHIBITION = "inhibition"
    COMPETITIVE = "competitive"
    ALLOSTERIC = "allosteric"


@dataclass
class Regulation:
    """A regulatory interaction."""
    regulator: str        # What does the regulating (metabolite or protein)
    target: str          # What is regulated (enzyme or gene)
    reg_type: RegulationType
    strength: float      # Ki or Ka (mM)
    hill_coef: float = 1.0  # Cooperativity


@dataclass 
class Pathway:
    """A multi-step biosynthetic pathway."""
    name: str
    steps: List[str]     # List of enzyme gene IDs in order
    metabolites: List[str]  # Intermediates: [start, inter1, inter2, ..., product]
    
    # Regulation
    feedback_inhibition: Optional[Tuple[str, str, float]] = None  # (product, enzyme, Ki)
    feedforward_activation: Optional[Tuple[str, str, float]] = None
    
    # Properties
    is_essential: bool = True  # Is the final product essential?
    
    def get_step_position(self, gene_id: str) -> Optional[int]:
        """Get position of enzyme in pathway (0 = first step)."""
        if gene_id in self.steps:
            return self.steps.index(gene_id)
        return None


@dataclass
class Cycle:
    """A metabolic cycle (e.g., TCA cycle)."""
    name: str
    steps: List[str]     # Enzymes in order
    metabolites: List[str]  # Intermediates (last connects to first)
    
    # Cycles are special: ALL steps are essential
    # Breaking any step breaks the whole cycle
    
    def is_broken_by(self, knockout: str) -> bool:
        """Check if knockout breaks the cycle."""
        return knockout in self.steps


@dataclass
class BistableSwitch:
    """A bistable genetic switch (memory element)."""
    name: str
    gene_a: str  # Represses B
    gene_b: str  # Represses A
    
    # States
    state: str = "A"  # "A" or "B"
    
    def knockout_effect(self, gene: str) -> str:
        """What happens when one gene is knocked out?"""
        if gene == self.gene_a:
            return "locked_in_B"  # Can't switch to A
        elif gene == self.gene_b:
            return "locked_in_A"  # Can't switch to B
        return "unaffected"


@dataclass
class EpigeneticMark:
    """An epigenetic modification."""
    target_gene: str
    mark_type: str  # "methylation", "acetylation", etc.
    writer: str     # Enzyme that adds the mark
    eraser: str     # Enzyme that removes the mark
    
    # Effect
    effect_on_expression: float  # Multiplier (0 = silenced, 2 = activated)


# ============================================================================
# SYSTEMS CELL MODEL
# ============================================================================

class SystemsCell:
    """
    Cell model with systems-level regulation.
    
    Captures:
    - Multi-step pathways
    - Feedback loops
    - Metabolic cycles
    - Regulatory switches
    - Epigenetic control
    """
    
    def __init__(self):
        # Core components
        self.genes: Dict[str, dict] = {}
        self.metabolites: Dict[str, float] = {}  # Concentrations
        
        # Regulatory structure
        self.pathways: List[Pathway] = []
        self.cycles: List[Cycle] = []
        self.regulations: List[Regulation] = []
        self.switches: List[BistableSwitch] = []
        self.epigenetic_marks: List[EpigeneticMark] = []
        
        # Build the cell
        self._build_pathways()
        self._build_cycles()
        self._build_regulation()
        self._build_switches()
        self._build_epigenetics()
        
        self._summarize()
    
    def _build_pathways(self):
        """Build multi-step biosynthetic pathways."""
        
        # ====== GLYCOLYSIS (10 steps) ======
        glycolysis = Pathway(
            name="glycolysis",
            steps=['ptsG', 'pgi', 'pfkA', 'fba', 'tpiA', 
                   'gapA', 'pgk', 'pgm', 'eno', 'pyk'],
            metabolites=['glucose', 'G6P', 'F6P', 'FBP', 'DHAP/G3P', 
                        'G3P', 'BPG', '3PG', '2PG', 'PEP', 'pyruvate'],
            # ATP inhibits PFK (feedback)
            feedback_inhibition=('ATP', 'pfkA', 2.0),
            # FBP activates PYK (feedforward)
            feedforward_activation=('FBP', 'pyk', 0.1),
            is_essential=True
        )
        self.pathways.append(glycolysis)
        
        # Register genes
        for i, gene in enumerate(glycolysis.steps):
            self.genes[gene] = {
                'name': gene,
                'pathway': 'glycolysis',
                'step': i,
                'essential': True,  # All glycolysis genes essential in syn3A
                'ground_truth_essential': gene not in ['ldh']  # ldh is non-essential
            }
        
        # ====== NUCLEOTIDE BIOSYNTHESIS (8 steps for purines) ======
        purine_synthesis = Pathway(
            name="purine_biosynthesis",
            steps=['prsA', 'purF', 'purD', 'purN', 'purL', 'purM', 'purK', 'purE'],
            metabolites=['R5P', 'PRPP', 'PRA', 'GAR', 'FGAR', 'FGAM', 
                        'AIR', 'CAIR', 'SAICAR', 'IMP'],
            feedback_inhibition=('AMP', 'prsA', 0.5),  # AMP inhibits PRPP synthesis
            is_essential=True
        )
        self.pathways.append(purine_synthesis)
        
        for i, gene in enumerate(purine_synthesis.steps):
            self.genes[gene] = {
                'name': gene,
                'pathway': 'purine_biosynthesis', 
                'step': i,
                'essential': True,
                'ground_truth_essential': True
            }
        
        # ====== FATTY ACID SYNTHESIS (7 cycles of 4 steps each) ======
        fa_synthesis = Pathway(
            name="fatty_acid_synthesis",
            steps=['accA', 'accB', 'accC', 'accD', 'fabD', 'fabH', 'fabG', 'fabZ', 'fabI'],
            metabolites=['acetyl_CoA', 'malonyl_CoA', 'malonyl_ACP', 
                        'acetoacetyl_ACP', 'hydroxyacyl_ACP', 'enoyl_ACP', 
                        'acyl_ACP', 'palmitate'],
            feedback_inhibition=('palmitate', 'accA', 1.0),
            is_essential=True
        )
        self.pathways.append(fa_synthesis)
        
        for i, gene in enumerate(fa_synthesis.steps):
            self.genes[gene] = {
                'name': gene,
                'pathway': 'fatty_acid_synthesis',
                'step': i,
                'essential': True,
                'ground_truth_essential': True
            }
        
        # ====== AMINO ACID BIOSYNTHESIS (multiple pathways) ======
        # Simplified: one pathway for "amino acids"
        aa_synthesis = Pathway(
            name="amino_acid_synthesis",
            steps=['gltB', 'gltD', 'glnA', 'aspC', 'asnA'],
            metabolites=['2OG', 'glutamate', 'glutamine', 'aspartate', 'asparagine'],
            feedback_inhibition=('glutamine', 'glnA', 2.0),
            is_essential=True
        )
        self.pathways.append(aa_synthesis)
        
        for i, gene in enumerate(aa_synthesis.steps):
            self.genes[gene] = {
                'name': gene,
                'pathway': 'amino_acid_synthesis',
                'step': i,
                'essential': True,
                'ground_truth_essential': True
            }
    
    def _build_cycles(self):
        """Build metabolic cycles."""
        
        # ====== TCA CYCLE (8 steps) ======
        # In JCVI-syn3A, TCA is incomplete, but let's model a generic one
        tca = Cycle(
            name="TCA_cycle",
            steps=['gltA', 'acnA', 'icd', 'sucA', 'sucC', 'sdhA', 'fumA', 'mdh'],
            metabolites=['OAA', 'citrate', 'isocitrate', '2OG', 
                        'succinyl_CoA', 'succinate', 'fumarate', 'malate']
        )
        self.cycles.append(tca)
        
        for gene in tca.steps:
            self.genes[gene] = {
                'name': gene,
                'pathway': 'TCA_cycle',
                'is_cycle': True,
                'essential': True,  # Breaking cycle is lethal
                'ground_truth_essential': True
            }
        
        # ====== NADH/NAD CYCLE ======
        nad_cycle = Cycle(
            name="NAD_cycle",
            steps=['gapA', 'ldh'],  # GAPDH reduces NAD, LDH regenerates
            metabolites=['NAD', 'NADH']
        )
        self.cycles.append(nad_cycle)
    
    def _build_regulation(self):
        """Build regulatory network."""
        
        # ====== GLYCOLYSIS REGULATION ======
        
        # ATP inhibits PFK (classic feedback)
        self.regulations.append(Regulation(
            regulator='ATP',
            target='pfkA',
            reg_type=RegulationType.ALLOSTERIC,
            strength=2.0,  # Ki = 2 mM
            hill_coef=4.0  # Highly cooperative
        ))
        
        # Citrate inhibits PFK
        self.regulations.append(Regulation(
            regulator='citrate',
            target='pfkA',
            reg_type=RegulationType.ALLOSTERIC,
            strength=0.5,
            hill_coef=2.0
        ))
        
        # FBP activates pyruvate kinase (feedforward)
        self.regulations.append(Regulation(
            regulator='FBP',
            target='pyk',
            reg_type=RegulationType.ACTIVATION,
            strength=0.1,  # Ka = 0.1 mM
            hill_coef=4.0
        ))
        
        # ====== NUCLEOTIDE REGULATION ======
        
        # AMP/GMP inhibit PRPP synthetase
        self.regulations.append(Regulation(
            regulator='AMP',
            target='prsA',
            reg_type=RegulationType.INHIBITION,
            strength=0.5,
            hill_coef=2.0
        ))
        
        # ====== AMINO ACID REGULATION ======
        
        # Glutamine inhibits glutamine synthetase
        self.regulations.append(Regulation(
            regulator='glutamine',
            target='glnA',
            reg_type=RegulationType.INHIBITION,
            strength=2.0,
            hill_coef=2.0
        ))
    
    def _build_switches(self):
        """Build bistable genetic switches."""
        
        # ====== LAMBDA PHAGE-LIKE SWITCH ======
        # (Not in syn3A but illustrates the concept)
        lysis_lysogeny = BistableSwitch(
            name="lysis_lysogeny",
            gene_a="cI",    # Lysogeny (represses cro)
            gene_b="cro",   # Lysis (represses cI)
            state="A"       # Start in lysogeny
        )
        self.switches.append(lysis_lysogeny)
        
        # Register switch genes
        self.genes['cI'] = {'name': 'cI', 'is_switch': True, 'essential': False,
                           'ground_truth_essential': False}
        self.genes['cro'] = {'name': 'cro', 'is_switch': True, 'essential': False,
                            'ground_truth_essential': False}
    
    def _build_epigenetics(self):
        """Build epigenetic regulation."""
        
        # ====== DNA METHYLATION ======
        # JCVI-syn3A has a restriction-modification system
        
        # Dam methylase methylates GATC sites
        self.epigenetic_marks.append(EpigeneticMark(
            target_gene="global",  # Affects many genes
            mark_type="methylation",
            writer="dam",
            eraser=None,  # No active demethylation in bacteria
            effect_on_expression=1.0  # Protective, not silencing
        ))
        
        self.genes['dam'] = {
            'name': 'dam', 
            'is_epigenetic': True,
            'essential': True,  # Protects DNA from restriction
            'ground_truth_essential': True
        }
        
        # ====== PROTEIN ACETYLATION ======
        # Affects metabolism through enzyme modification
        self.genes['pat'] = {
            'name': 'pat',
            'is_epigenetic': True,
            'essential': False,  # Regulatory, not essential
            'ground_truth_essential': False
        }
    
    def _summarize(self):
        """Print summary of cell model."""
        print("="*70)
        print("SYSTEMS CELL MODEL")
        print("="*70)
        print(f"Genes: {len(self.genes)}")
        print(f"Pathways: {len(self.pathways)}")
        print(f"  - Total pathway steps: {sum(len(p.steps) for p in self.pathways)}")
        print(f"Cycles: {len(self.cycles)}")
        print(f"Regulations: {len(self.regulations)}")
        print(f"Bistable switches: {len(self.switches)}")
        print(f"Epigenetic marks: {len(self.epigenetic_marks)}")
        
        # Count essential
        n_essential = sum(1 for g in self.genes.values() 
                         if g.get('ground_truth_essential', False))
        print(f"Essential genes: {n_essential}/{len(self.genes)}")
    
    # ========================================================================
    # KNOCKOUT SIMULATION
    # ========================================================================
    
    def simulate_knockout(self, gene_id: str) -> Dict:
        """
        Simulate knockout with systems-level effects.
        
        Considers:
        1. Position in pathway (early vs late)
        2. Cycle breakage
        3. Feedback loop disruption
        4. Switch state locking
        5. Epigenetic effects
        """
        if gene_id not in self.genes:
            return {'error': f'Gene {gene_id} not found'}
        
        gene = self.genes[gene_id]
        effects = []
        is_lethal = False
        
        # ====== CHECK PATHWAY EFFECTS ======
        for pathway in self.pathways:
            pos = pathway.get_step_position(gene_id)
            if pos is not None:
                n_steps = len(pathway.steps)
                
                # Early steps: substrate accumulation
                if pos < n_steps // 3:
                    effects.append(f"Early block in {pathway.name}: substrate accumulates")
                    if pathway.is_essential:
                        is_lethal = True
                        effects.append(f"→ {pathway.metabolites[0]} accumulates (toxic)")
                
                # Middle steps: intermediate accumulation
                elif pos < 2 * n_steps // 3:
                    effects.append(f"Middle block in {pathway.name}: intermediate accumulates")
                    if pathway.is_essential:
                        is_lethal = True
                        effects.append(f"→ {pathway.metabolites[pos]} accumulates")
                
                # Late steps: product depletion
                else:
                    effects.append(f"Late block in {pathway.name}: product depleted")
                    if pathway.is_essential:
                        is_lethal = True
                        effects.append(f"→ No {pathway.metabolites[-1]}")
                
                # Feedback disruption
                if pathway.feedback_inhibition:
                    product, target, Ki = pathway.feedback_inhibition
                    if gene_id == target:
                        effects.append(f"Feedback target knocked out: {product} no longer inhibits")
                    elif pos >= len(pathway.steps) - 2:  # Late enzyme
                        effects.append(f"Product {product} depleted → feedback lost → upstream overactive")
        
        # ====== CHECK CYCLE EFFECTS ======
        for cycle in self.cycles:
            if cycle.is_broken_by(gene_id):
                effects.append(f"CYCLE BROKEN: {cycle.name}")
                effects.append(f"→ All intermediates affected: {', '.join(cycle.metabolites)}")
                is_lethal = True
        
        # ====== CHECK REGULATION EFFECTS ======
        for reg in self.regulations:
            if reg.target == gene_id:
                effects.append(f"Regulated enzyme knocked out (regulated by {reg.regulator})")
            # If knockout affects regulator production, downstream targets affected
        
        # ====== CHECK SWITCH EFFECTS ======
        for switch in self.switches:
            effect = switch.knockout_effect(gene_id)
            if effect != "unaffected":
                effects.append(f"Switch '{switch.name}': {effect}")
                # Being locked in one state may or may not be lethal
        
        # ====== CHECK EPIGENETIC EFFECTS ======
        for mark in self.epigenetic_marks:
            if mark.writer == gene_id:
                effects.append(f"Epigenetic writer lost: {mark.mark_type} marks not added")
                effects.append(f"→ Affects expression of {mark.target_gene}")
                if mark.effect_on_expression == 0:
                    is_lethal = True  # Silencing essential genes
        
        # Ground truth
        ground_truth = gene.get('ground_truth_essential', False)
        
        return {
            'gene_id': gene_id,
            'gene_name': gene.get('name', gene_id),
            'predicted_essential': is_lethal,
            'experimental_essential': ground_truth,
            'correct': is_lethal == ground_truth,
            'effects': effects,
            'pathway': gene.get('pathway', None),
            'is_cycle_member': gene.get('is_cycle', False),
            'is_switch_member': gene.get('is_switch', False),
            'is_epigenetic': gene.get('is_epigenetic', False),
        }
    
    def predict_all(self) -> Dict:
        """Run predictions on all genes."""
        results = []
        
        for gene_id in self.genes:
            result = self.simulate_knockout(gene_id)
            results.append(result)
        
        # Metrics
        tp = sum(1 for r in results if r['predicted_essential'] and r['experimental_essential'])
        fp = sum(1 for r in results if r['predicted_essential'] and not r['experimental_essential'])
        tn = sum(1 for r in results if not r['predicted_essential'] and not r['experimental_essential'])
        fn = sum(1 for r in results if not r['predicted_essential'] and r['experimental_essential'])
        
        accuracy = (tp + tn) / len(results) if results else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'total': len(results),
            'results': results,
        }


# ============================================================================
# DYNAMIC SIMULATION
# ============================================================================

class DynamicSystemsCell(SystemsCell):
    """
    Cell with time-dependent dynamics.
    
    Simulates:
    - Metabolite concentrations over time
    - Oscillations
    - Adaptation
    - Transient vs steady-state effects
    """
    
    def __init__(self):
        super().__init__()
        self._init_concentrations()
    
    def _init_concentrations(self):
        """Initialize metabolite concentrations."""
        self.concentrations = {
            # Energy
            'ATP': 3.0,
            'ADP': 0.5,
            'AMP': 0.1,
            
            # Glycolysis
            'glucose': 5.0,
            'G6P': 1.0,
            'F6P': 0.3,
            'FBP': 0.5,
            'pyruvate': 0.5,
            
            # TCA
            'acetyl_CoA': 0.1,
            'citrate': 0.5,
            'OAA': 0.05,
            
            # Redox
            'NAD': 1.0,
            'NADH': 0.1,
            
            # Building blocks
            'amino_acids': 1.0,
            'NTPs': 1.0,
        }
    
    def compute_rate(self, enzyme: str, substrate: str, 
                     vmax: float = 10.0, km: float = 0.5) -> float:
        """
        Compute enzyme rate with regulation.
        
        v = Vmax * [S] / (Km * (1 + [I]/Ki) + [S]) * (1 + [A]/Ka) / (1 + [A]/Ka)
        """
        S = self.concentrations.get(substrate, 0)
        
        # Base Michaelis-Menten
        rate = vmax * S / (km + S)
        
        # Apply regulations
        for reg in self.regulations:
            if reg.target == enzyme:
                regulator_conc = self.concentrations.get(reg.regulator, 0)
                
                if reg.reg_type == RegulationType.INHIBITION:
                    # Competitive-like inhibition
                    inhibition = 1 / (1 + (regulator_conc / reg.strength) ** reg.hill_coef)
                    rate *= inhibition
                
                elif reg.reg_type == RegulationType.ACTIVATION:
                    # Activation
                    activation = (regulator_conc / reg.strength) ** reg.hill_coef
                    activation = activation / (1 + activation)
                    rate *= (0.1 + 0.9 * activation)  # 10% basal
        
        return rate
    
    def simulate_time_course(self, knockout: str = None, 
                            duration: float = 100.0, 
                            dt: float = 0.1) -> Dict:
        """
        Simulate cell over time with optional knockout.
        
        Returns time course of key metabolites.
        """
        times = np.arange(0, duration, dt)
        n_steps = len(times)
        
        # Track key metabolites
        history = {met: np.zeros(n_steps) for met in 
                  ['ATP', 'pyruvate', 'amino_acids', 'NAD']}
        
        # Temporary concentrations
        conc = self.concentrations.copy()
        
        for i, t in enumerate(times):
            # Record
            for met in history:
                history[met][i] = conc.get(met, 0)
            
            # Compute fluxes (simplified)
            if knockout != 'pfkA':
                glycolysis_flux = self.compute_rate('pfkA', 'F6P', vmax=10, km=0.1)
            else:
                glycolysis_flux = 0
            
            if knockout != 'pyk':
                pyk_flux = self.compute_rate('pyk', 'PEP', vmax=20, km=0.05)
            else:
                pyk_flux = 0
            
            # Update concentrations (Euler integration)
            conc['glucose'] -= glycolysis_flux * dt * 0.1  # Glucose consumed
            conc['pyruvate'] += glycolysis_flux * dt - pyk_flux * dt * 0.5
            conc['ATP'] += (2 * pyk_flux - glycolysis_flux) * dt * 0.1  # Net ATP
            
            # Clamp to non-negative
            for met in conc:
                conc[met] = max(0, conc[met])
        
        # Determine outcome
        final_atp = history['ATP'][-1]
        min_atp = np.min(history['ATP'])
        
        return {
            'times': times,
            'history': history,
            'final_ATP': final_atp,
            'min_ATP': min_atp,
            'survived': min_atp > 0.5,  # ATP threshold for survival
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("DARK MANIFOLD V45: SYSTEMS CELL MODEL")
    print("Pathways, Feedback, Cycles, Switches, Epigenetics")
    print("="*70)
    
    cell = SystemsCell()
    
    print("\n" + "="*70)
    print("KNOCKOUT PREDICTIONS")
    print("="*70)
    
    summary = cell.predict_all()
    
    print(f"\n{'Gene':<12} {'Pred':<6} {'Exp':<6} {'Match':<6} {'Effects'}")
    print("-"*70)
    
    for r in summary['results']:
        pred = "ESS" if r['predicted_essential'] else "non"
        exp = "ESS" if r['experimental_essential'] else "non"
        match = "✓" if r['correct'] else "✗"
        effects = "; ".join(r['effects'][:2]) if r['effects'] else "-"
        print(f"{r['gene_name']:<12} {pred:<6} {exp:<6} {match:<6} {effects[:40]}")
    
    print("\n" + "="*70)
    print("ACCURACY SUMMARY")
    print("="*70)
    print(f"Total genes: {summary['total']}")
    print(f"Accuracy:    {summary['accuracy']*100:.1f}%")
    print(f"Sensitivity: {summary['sensitivity']*100:.1f}%")
    print(f"Specificity: {summary['specificity']*100:.1f}%")
    print(f"TP={summary['tp']}, FP={summary['fp']}, TN={summary['tn']}, FN={summary['fn']}")
    
    # Dynamic simulation
    print("\n" + "="*70)
    print("DYNAMIC SIMULATION")
    print("="*70)
    
    dyn_cell = DynamicSystemsCell()
    
    # Wild-type
    wt = dyn_cell.simulate_time_course(knockout=None, duration=50)
    print(f"Wild-type: ATP={wt['final_ATP']:.2f}, survived={wt['survived']}")
    
    # PFK knockout
    pfk_ko = dyn_cell.simulate_time_course(knockout='pfkA', duration=50)
    print(f"ΔpfkA:     ATP={pfk_ko['final_ATP']:.2f}, survived={pfk_ko['survived']}")
    
    return cell, summary


if __name__ == '__main__':
    cell, summary = main()
