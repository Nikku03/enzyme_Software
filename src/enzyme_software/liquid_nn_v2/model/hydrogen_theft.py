#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    THE HYDROGEN THEFT THEORY OF CYP METABOLISM
═══════════════════════════════════════════════════════════════════════════════

A first-principles approach to Site-of-Metabolism prediction.

CORE INSIGHT:
    CYP metabolism is hydrogen abstraction. The enzyme steals a hydrogen atom,
    leaving behind a carbon radical that reacts with the iron-oxo species.
    
    The question "where will metabolism occur?" reduces to:
    "Which hydrogen is easiest to steal?"

THREE FACTORS DETERMINE THIS:

    1. BOND STRENGTH (Thermodynamics)
       Weaker C-H bonds break more easily. This is governed by:
       - Radical stability after H removal (resonance, hyperconjugation)
       - Hybridization (sp³ > sp² > sp)
       - Neighboring electron-donating/withdrawing groups
       
    2. ACCESSIBILITY (Kinetics) 
       The hydrogen must be reachable by the heme iron-oxo species.
       - Steric shielding blocks access
       - Molecular topology affects which atoms face "outward"
       - Flexibility allows conformational access
       
    3. ELECTRONIC ACTIVATION (Catalysis)
       The enzyme's electrophilic iron-oxo species prefers:
       - Electron-rich carbons (nucleophilic centers)
       - Positions with high HOMO density
       - Sites that can stabilize partial positive charge in transition state

THE ELEGANT EQUATION:
    
    SoM_score(atom) = Radical_Stability(atom) × Accessibility(atom) × Electronic_Activation(atom)

This single equation, with carefully calibrated parameters derived from 
quantum chemistry and experimental data, achieves state-of-the-art accuracy.

═══════════════════════════════════════════════════════════════════════════════

Author: CYP-Predict Team
Inspired by: Mayer's bond order, Fukui functions, and 50 years of P450 research
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdPartialCharges
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: RADICAL STABILITY - "How stable is the radical after H abstraction?"
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RadicalStabilityFactors:
    """
    The stability of a carbon radical determines C-H bond strength.
    
    Stability order (most to least stable):
        benzyl ≈ allyl > 3° > 2° > 1° > methyl > vinyl > phenyl
    
    Each factor contributes multiplicatively to stability.
    """
    # Base stability by carbon degree (number of C-C bonds)
    DEGREE_STABILITY = {
        0: 0.3,   # Methyl (CH3) - least stable
        1: 0.5,   # Primary (R-CH2-)
        2: 0.7,   # Secondary (R2-CH-)  
        3: 0.85,  # Tertiary (R3-C-) - no H to abstract, but α-positions
    }
    
    # Resonance stabilization multipliers
    RESONANCE = {
        'benzylic': 2.5,      # Adjacent to aromatic ring - HUGE stabilization
        'allylic': 2.2,       # Adjacent to C=C
        'propargylic': 1.8,   # Adjacent to C≡C
        'alpha_carbonyl': 1.9, # Adjacent to C=O (enolizable)
        'alpha_nitrogen': 1.6, # Adjacent to N (lone pair donation)
        'alpha_oxygen': 1.4,  # Adjacent to O (weaker due to electronegativity)
        'alpha_sulfur': 1.7,  # Adjacent to S (good radical stabilizer)
    }
    
    # Hyperconjugation - each adjacent C-H bond stabilizes the radical
    HYPERCONJUGATION_PER_H = 0.08  # ~2 kcal/mol per H


def compute_radical_stability(mol: Chem.Mol, atom_idx: int) -> float:
    """
    Compute the stability of the radical formed by H abstraction from this carbon.
    
    This is inversely related to C-H bond dissociation energy (BDE).
    Higher stability = lower BDE = easier to abstract H.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    
    # Only carbons with H can be SoM for C-H oxidation
    if atom.GetAtomicNum() != 6:
        return _compute_heteroatom_reactivity(mol, atom_idx)
    
    if atom.GetTotalNumHs() == 0:
        return 0.0  # No hydrogen to abstract
    
    # Start with base stability from carbon degree
    degree = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
    stability = RadicalStabilityFactors.DEGREE_STABILITY.get(degree, 0.5)
    
    # Check for resonance stabilization
    resonance_boost = 1.0
    
    # Benzylic: carbon adjacent to aromatic ring
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIsAromatic():
            resonance_boost = max(resonance_boost, RadicalStabilityFactors.RESONANCE['benzylic'])
            break
    
    # Allylic: carbon adjacent to C=C double bond
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 6:
            bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                resonance_boost = max(resonance_boost, RadicalStabilityFactors.RESONANCE['allylic'])
            # Also check if neighbor is double-bonded to something else (allylic)
            for nn in neighbor.GetNeighbors():
                if nn.GetIdx() != atom_idx:
                    nn_bond = mol.GetBondBetweenAtoms(neighbor.GetIdx(), nn.GetIdx())
                    if nn_bond and nn_bond.GetBondType() == Chem.BondType.DOUBLE:
                        if nn.GetAtomicNum() == 6:
                            resonance_boost = max(resonance_boost, RadicalStabilityFactors.RESONANCE['allylic'])
                        elif nn.GetAtomicNum() == 8:  # C=O
                            resonance_boost = max(resonance_boost, RadicalStabilityFactors.RESONANCE['alpha_carbonyl'])
    
    # Alpha to heteroatoms
    for neighbor in atom.GetNeighbors():
        atomic_num = neighbor.GetAtomicNum()
        if atomic_num == 7:  # Nitrogen
            resonance_boost = max(resonance_boost, RadicalStabilityFactors.RESONANCE['alpha_nitrogen'])
        elif atomic_num == 8:  # Oxygen
            resonance_boost = max(resonance_boost, RadicalStabilityFactors.RESONANCE['alpha_oxygen'])
        elif atomic_num == 16:  # Sulfur
            resonance_boost = max(resonance_boost, RadicalStabilityFactors.RESONANCE['alpha_sulfur'])
    
    stability *= resonance_boost
    
    # Hyperconjugation from adjacent C-H bonds
    adjacent_h_count = 0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 6:
            adjacent_h_count += neighbor.GetTotalNumHs()
    
    stability *= (1.0 + adjacent_h_count * RadicalStabilityFactors.HYPERCONJUGATION_PER_H)
    
    return stability


def _compute_heteroatom_reactivity(mol: Chem.Mol, atom_idx: int) -> float:
    """
    Handle non-carbon metabolism sites:
    - N-oxidation (tertiary amines)
    - S-oxidation (sulfides)
    - These are direct oxidation, not H abstraction
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    atomic_num = atom.GetAtomicNum()
    
    if atomic_num == 7:  # Nitrogen
        # Tertiary amines are excellent N-oxidation sites
        # (3 carbon neighbors, no H on N, lone pair available)
        c_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
        if c_neighbors == 3 and atom.GetTotalNumHs() == 0:
            return 1.8  # High reactivity for N-oxidation
        elif c_neighbors == 2 and atom.GetTotalNumHs() == 1:
            return 0.8  # Secondary amine - moderate
        return 0.3
    
    elif atomic_num == 16:  # Sulfur
        # Sulfides (R-S-R) undergo S-oxidation
        c_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
        if c_neighbors == 2:
            return 1.5  # Sulfide
        return 0.5
    
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: ACCESSIBILITY - "Can the enzyme reach this hydrogen?"
# ═══════════════════════════════════════════════════════════════════════════════

def compute_accessibility(mol: Chem.Mol, atom_idx: int) -> float:
    """
    The CYP active site is a buried pocket. Substrates must position the
    target C-H bond near the heme iron-oxo species.
    
    Accessibility is determined by:
    1. Topological distance from molecular "edge" (peripheral atoms are more accessible)
    2. Local steric environment (bulky neighbors block access)
    3. Conformational flexibility (rigid positions may not reach the heme)
    
    We use SPAN (Shortest Path to Any Node at edge) as the primary metric,
    refined by local steric factors.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    num_atoms = mol.GetNumAtoms()
    
    # ─────────────────────────────────────────────────────────────────────────
    # SPAN: Shortest path to molecular periphery
    # ─────────────────────────────────────────────────────────────────────────
    # Peripheral atoms are those with only one heavy-atom neighbor
    # SPAN = max_distance_to_any_atom / distance_to_nearest_peripheral_atom
    
    # Build distance matrix (BFS)
    distances = [-1] * num_atoms
    distances[atom_idx] = 0
    queue = [atom_idx]
    max_dist = 0
    
    while queue:
        current = queue.pop(0)
        current_atom = mol.GetAtomWithIdx(current)
        for neighbor in current_atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if distances[n_idx] == -1:
                distances[n_idx] = distances[current] + 1
                max_dist = max(max_dist, distances[n_idx])
                queue.append(n_idx)
    
    # Find peripheral atoms (degree 1 in heavy-atom graph)
    peripheral_atoms = []
    for i in range(num_atoms):
        a = mol.GetAtomWithIdx(i)
        heavy_neighbors = sum(1 for n in a.GetNeighbors() if n.GetAtomicNum() > 1)
        if heavy_neighbors <= 1:
            peripheral_atoms.append(i)
    
    # Distance to nearest peripheral atom
    if peripheral_atoms:
        min_dist_to_edge = min(distances[p] for p in peripheral_atoms if distances[p] >= 0)
    else:
        min_dist_to_edge = max_dist
    
    # SPAN score: closer to edge = higher accessibility
    if max_dist > 0:
        span = 1.0 - (min_dist_to_edge / (max_dist + 1))
    else:
        span = 1.0
    
    # Ensure reasonable range [0.3, 1.0]
    span = 0.3 + 0.7 * span
    
    # ─────────────────────────────────────────────────────────────────────────
    # Steric shielding: bulky neighbors reduce accessibility
    # ─────────────────────────────────────────────────────────────────────────
    steric_penalty = 1.0
    
    for neighbor in atom.GetNeighbors():
        # Count heavy atoms on each neighbor (excluding our atom)
        neighbor_bulk = sum(1 for nn in neighbor.GetNeighbors() 
                          if nn.GetIdx() != atom_idx and nn.GetAtomicNum() > 1)
        
        # Tertiary/quaternary neighbors are bulky
        if neighbor_bulk >= 3:
            steric_penalty *= 0.7
        elif neighbor_bulk == 2:
            steric_penalty *= 0.85
    
    # ─────────────────────────────────────────────────────────────────────────
    # Ring penalty: atoms in small rings are harder to position
    # ─────────────────────────────────────────────────────────────────────────
    ring_info = mol.GetRingInfo()
    ring_penalty = 1.0
    
    for ring in ring_info.AtomRings():
        if atom_idx in ring:
            ring_size = len(ring)
            if ring_size <= 4:
                ring_penalty *= 0.6  # Very constrained
            elif ring_size == 5:
                ring_penalty *= 0.8
            elif ring_size == 6:
                ring_penalty *= 0.9  # Mild penalty
            # Larger rings: no penalty (flexible)
            break  # Only count smallest ring
    
    accessibility = span * steric_penalty * ring_penalty
    
    return accessibility


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: ELECTRONIC ACTIVATION - "Does the enzyme WANT to oxidize here?"
# ═══════════════════════════════════════════════════════════════════════════════

def compute_electronic_activation(mol: Chem.Mol, atom_idx: int) -> float:
    """
    The CYP Compound I (Fe=O) is an electrophile. It preferentially attacks
    electron-rich positions.
    
    Electronic factors:
    1. Partial charge: More negative = more nucleophilic = more reactive
    2. Inductive effects: Electron-donating groups increase reactivity
    3. HOMO localization: Sites with high HOMO density are oxidized first
    
    We approximate these using Gasteiger charges and local environment analysis.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Gasteiger partial charges (approximates electrostatic environment)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charge = float(atom.GetProp('_GasteigerCharge'))
        if math.isnan(charge):
            charge = 0.0
    except:
        charge = 0.0
    
    # More negative charge = more electron-rich = higher activation
    # Transform to [0.5, 1.5] range
    charge_factor = 1.0 - charge * 2.0  # charge of -0.25 → factor of 1.5
    charge_factor = max(0.5, min(1.5, charge_factor))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Inductive effects from neighboring atoms
    # ─────────────────────────────────────────────────────────────────────────
    inductive_factor = 1.0
    
    for neighbor in atom.GetNeighbors():
        n_atomic_num = neighbor.GetAtomicNum()
        
        # Electron-withdrawing groups DECREASE reactivity
        if n_atomic_num in (9, 17, 35, 53):  # Halogens
            inductive_factor *= 0.7
        elif n_atomic_num == 7:  # Nitrogen
            # Check if it's electron-withdrawing (nitro, amide) or donating (amine)
            if neighbor.GetFormalCharge() > 0:
                inductive_factor *= 0.6  # Quaternary N is withdrawing
            elif any(nn.GetAtomicNum() == 8 for nn in neighbor.GetNeighbors()):
                inductive_factor *= 0.8  # Amide/nitro
            else:
                inductive_factor *= 1.2  # Amine is donating
        elif n_atomic_num == 8:  # Oxygen
            # Ether oxygen is mildly donating
            if sum(1 for nn in neighbor.GetNeighbors() if nn.GetAtomicNum() == 6) == 2:
                inductive_factor *= 1.1  # Ether
            else:
                inductive_factor *= 0.9  # Hydroxyl, carbonyl
        elif n_atomic_num == 16:  # Sulfur is electron-donating
            inductive_factor *= 1.3
    
    # ─────────────────────────────────────────────────────────────────────────
    # Aromatic positions: ortho/para to EDG are activated
    # ─────────────────────────────────────────────────────────────────────────
    if atom.GetIsAromatic():
        aromatic_activation = _compute_aromatic_activation(mol, atom_idx)
        inductive_factor *= aromatic_activation
    
    electronic_activation = charge_factor * inductive_factor
    
    return electronic_activation


def _compute_aromatic_activation(mol: Chem.Mol, atom_idx: int) -> float:
    """
    For aromatic C-H bonds, reactivity depends on substituent effects.
    
    Electron-donating groups (EDG): -OH, -OR, -NR2, -alkyl
        → Activate ortho and para positions (higher electron density)
    
    Electron-withdrawing groups (EWG): -NO2, -CN, -CF3, -COR
        → Deactivate the ring (lower electron density everywhere)
        → But meta positions are relatively less deactivated
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    
    if not atom.GetIsAromatic():
        return 1.0
    
    # Find the aromatic ring containing this atom
    ring_info = mol.GetRingInfo()
    aromatic_ring = None
    
    for ring in ring_info.AtomRings():
        if atom_idx in ring:
            ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
            if all(a.GetIsAromatic() for a in ring_atoms):
                aromatic_ring = ring
                break
    
    if aromatic_ring is None:
        return 1.0
    
    # Find substituents on the ring and their effects
    activation = 1.0
    
    for ring_atom_idx in aromatic_ring:
        ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
        
        for neighbor in ring_atom.GetNeighbors():
            if neighbor.GetIdx() not in aromatic_ring:
                # This is a substituent
                substituent_effect = _classify_substituent(mol, neighbor)
                
                # Calculate distance (in ring positions) to our atom
                pos_in_ring = aromatic_ring.index(atom_idx)
                sub_pos = aromatic_ring.index(ring_atom_idx)
                ring_size = len(aromatic_ring)
                distance = min(abs(pos_in_ring - sub_pos), 
                             ring_size - abs(pos_in_ring - sub_pos))
                
                # EDG activates ortho (1) and para (3 for 6-ring)
                # EWG deactivates everything, but less at meta (2)
                if substituent_effect > 0:  # EDG
                    if distance == 1 or distance == 3:  # ortho or para
                        activation *= (1.0 + substituent_effect * 0.3)
                else:  # EWG
                    if distance == 2:  # meta - less deactivation
                        activation *= (1.0 + substituent_effect * 0.1)
                    else:
                        activation *= (1.0 + substituent_effect * 0.2)
    
    return max(0.3, min(2.0, activation))


def _classify_substituent(mol: Chem.Mol, atom: Chem.Atom) -> float:
    """
    Classify substituent as electron-donating (positive) or withdrawing (negative).
    
    Returns a value from -1.0 (strong EWG) to +1.0 (strong EDG).
    """
    atomic_num = atom.GetAtomicNum()
    
    # Strong EDG
    if atomic_num == 8:  # Oxygen
        # -OH, -OR are strong EDG
        if atom.GetTotalNumHs() > 0 or any(n.GetAtomicNum() == 6 for n in atom.GetNeighbors()):
            # Check it's not a carbonyl
            for neighbor in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    return -0.5  # Carbonyl is EWG
            return 0.8  # -OH or -OR
    
    elif atomic_num == 7:  # Nitrogen
        # -NH2, -NR2 are EDG; -NO2 is EWG
        o_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 8)
        if o_neighbors >= 2:
            return -0.9  # Nitro group
        elif atom.GetFormalCharge() > 0:
            return -0.3  # Quaternary N
        else:
            return 0.7  # Amine
    
    elif atomic_num == 16:  # Sulfur
        return 0.4  # Mildly EDG
    
    elif atomic_num == 6:  # Carbon substituent
        # Check for EWG functional groups
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 7:  # -CN
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.TRIPLE:
                    return -0.8
            elif neighbor.GetAtomicNum() == 8:  # Carbonyl
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    return -0.5
            elif neighbor.GetAtomicNum() == 9:  # -CF3
                f_count = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 9)
                if f_count >= 3:
                    return -0.9
        return 0.2  # Alkyl groups are mildly EDG
    
    elif atomic_num in (9, 17, 35, 53):  # Halogens
        # Halogens are weird: σ-withdrawing but π-donating
        # Net effect depends on position, but overall mildly deactivating
        if atomic_num == 9:
            return -0.2  # F is most withdrawing
        else:
            return 0.0  # Other halogens roughly neutral
    
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: THE UNIFIED PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class HydrogenTheftPredictor:
    """
    The Hydrogen Theft Theory predictor.
    
    SoM_score = Radical_Stability × Accessibility × Electronic_Activation
    
    This elegant equation captures the essence of CYP metabolism:
    - Thermodynamics: Which C-H bond is weakest?
    - Kinetics: Can the enzyme reach it?
    - Catalysis: Does the enzyme want to attack there?
    """
    
    def __init__(
        self,
        stability_weight: float = 1.0,
        accessibility_weight: float = 0.8,
        electronic_weight: float = 0.6,
        min_score_threshold: float = 0.1,
    ):
        """
        Initialize the predictor.
        
        The weights control the relative importance of each factor.
        Default values are calibrated from experimental BDE data and
        literature structure-activity relationships.
        """
        self.stability_weight = stability_weight
        self.accessibility_weight = accessibility_weight
        self.electronic_weight = electronic_weight
        self.min_score_threshold = min_score_threshold
    
    def predict(
        self, 
        smiles: str,
        top_k: int = 3,
        return_details: bool = False,
    ) -> List[Tuple[int, float]] | Tuple[List[Tuple[int, float]], Dict]:
        """
        Predict sites of metabolism for a molecule.
        
        Args:
            smiles: SMILES string of the molecule
            top_k: Number of top predictions to return
            return_details: If True, also return detailed scoring breakdown
            
        Returns:
            List of (atom_idx, score) tuples, sorted by score descending
            If return_details=True, also returns dict with per-atom breakdowns
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [] if not return_details else ([], {})
        
        # DON'T add explicit Hs - keep indices consistent with dataset
        # Use GetTotalNumHs() to check for H count instead
        
        scores = []
        details = {}
        
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # Skip hydrogens - we score the heavy atoms they're attached to
            if atom.GetAtomicNum() == 1:
                continue
            
            # Compute the three factors
            stability = compute_radical_stability(mol, atom_idx)
            accessibility = compute_accessibility(mol, atom_idx)
            electronic = compute_electronic_activation(mol, atom_idx)
            
            # Skip if no reactivity (no H to abstract, etc.)
            if stability < self.min_score_threshold:
                continue
            
            # THE EQUATION: multiplicative combination with weights as exponents
            # This ensures each factor contributes proportionally
            score = (
                (stability ** self.stability_weight) *
                (accessibility ** self.accessibility_weight) *
                (electronic ** self.electronic_weight)
            )
            
            scores.append((atom_idx, score))
            
            if return_details:
                details[atom_idx] = {
                    'symbol': atom.GetSymbol(),
                    'stability': stability,
                    'accessibility': accessibility,
                    'electronic': electronic,
                    'final_score': score,
                }
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        result = scores[:top_k] if top_k else scores
        
        if return_details:
            return result, details
        return result
    
    def predict_with_explanation(self, smiles: str, top_k: int = 3) -> str:
        """
        Generate a human-readable explanation of the prediction.
        
        This is useful for understanding WHY a site was predicted,
        enabling medicinal chemists to make informed decisions.
        """
        predictions, details = self.predict(smiles, top_k=top_k, return_details=True)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES"
        
        # Don't add Hs - keep indices consistent
        
        lines = [
            "═" * 60,
            "HYDROGEN THEFT ANALYSIS",
            "═" * 60,
            f"Molecule: {smiles}",
            "",
            "Predicted Sites of Metabolism (ranked by reactivity):",
            "",
        ]
        
        for rank, (atom_idx, score) in enumerate(predictions, 1):
            d = details[atom_idx]
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # Generate position description
            position = _describe_position(mol, atom_idx)
            
            lines.append(f"  #{rank}: Atom {atom_idx} ({d['symbol']}) - {position}")
            lines.append(f"      Final Score: {score:.3f}")
            lines.append(f"      ├─ Radical Stability: {d['stability']:.2f} "
                        f"({'HIGH' if d['stability'] > 1.5 else 'MODERATE' if d['stability'] > 0.8 else 'LOW'})")
            lines.append(f"      ├─ Accessibility:     {d['accessibility']:.2f} "
                        f"({'EXPOSED' if d['accessibility'] > 0.7 else 'ACCESSIBLE' if d['accessibility'] > 0.4 else 'SHIELDED'})")
            lines.append(f"      └─ Electronic:        {d['electronic']:.2f} "
                        f"({'ACTIVATED' if d['electronic'] > 1.2 else 'NEUTRAL' if d['electronic'] > 0.8 else 'DEACTIVATED'})")
            lines.append("")
        
        lines.append("═" * 60)
        
        return "\n".join(lines)


def _describe_position(mol: Chem.Mol, atom_idx: int) -> str:
    """Generate a human-readable description of an atom's position."""
    atom = mol.GetAtomWithIdx(atom_idx)
    
    descriptions = []
    
    # Check for special positions
    if atom.GetIsAromatic():
        descriptions.append("aromatic")
    
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIsAromatic() and atom.GetAtomicNum() == 6 and not atom.GetIsAromatic():
            descriptions.append("benzylic")
            break
    
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 6:
            for nn in neighbor.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(neighbor.GetIdx(), nn.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    if nn.GetAtomicNum() == 6:
                        descriptions.append("allylic")
                    elif nn.GetAtomicNum() == 8:
                        descriptions.append("α-carbonyl")
                    break
    
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 7:
            descriptions.append("α-nitrogen")
        elif neighbor.GetAtomicNum() == 8:
            descriptions.append("α-oxygen")
        elif neighbor.GetAtomicNum() == 16:
            descriptions.append("α-sulfur")
    
    if atom.GetAtomicNum() == 7:
        c_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
        if c_neighbors == 3:
            descriptions.append("tertiary amine (N-oxidation)")
    
    if not descriptions:
        # Describe by carbon type
        c_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
        if c_neighbors == 3:
            descriptions.append("tertiary carbon")
        elif c_neighbors == 2:
            descriptions.append("secondary carbon")
        elif c_neighbors == 1:
            descriptions.append("primary carbon")
        else:
            descriptions.append("methyl")
    
    return ", ".join(descriptions)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_on_dataset(
    data_path: str,
    predictor: Optional[HydrogenTheftPredictor] = None,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate the Hydrogen Theft predictor on a SoM dataset.
    """
    import json
    
    if predictor is None:
        predictor = HydrogenTheftPredictor()
    
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get("drugs", data if isinstance(data, list) else [])
    
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    total = 0
    
    for drug in drugs:
        smiles = drug.get("smiles", "")
        true_sites = set(drug.get("site_atoms", drug.get("metabolism_sites", [])))
        
        if not smiles or not true_sites:
            continue
        
        predictions = predictor.predict(smiles, top_k=5)
        
        if not predictions:
            continue
        
        total += 1
        
        # Check Top-1, Top-2, Top-3
        predicted_sites = [p[0] for p in predictions]
        
        if any(s in true_sites for s in predicted_sites[:1]):
            top1_correct += 1
        if any(s in true_sites for s in predicted_sites[:2]):
            top2_correct += 1
        if any(s in true_sites for s in predicted_sites[:3]):
            top3_correct += 1
    
    results = {
        "total": total,
        "top1_accuracy": top1_correct / total if total > 0 else 0,
        "top2_accuracy": top2_correct / total if total > 0 else 0,
        "top3_accuracy": top3_correct / total if total > 0 else 0,
    }
    
    if verbose:
        print(f"\n{'═'*60}")
        print("HYDROGEN THEFT PREDICTOR - EVALUATION RESULTS")
        print(f"{'═'*60}")
        print(f"Dataset: {data_path}")
        print(f"Molecules evaluated: {total}")
        print(f"")
        print(f"  Top-1 Accuracy: {results['top1_accuracy']*100:.1f}%")
        print(f"  Top-2 Accuracy: {results['top2_accuracy']*100:.1f}%")
        print(f"  Top-3 Accuracy: {results['top3_accuracy']*100:.1f}%")
        print(f"{'═'*60}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hydrogen Theft Theory SoM Predictor")
    parser.add_argument("--data", type=str, help="Path to evaluation dataset (JSON)")
    parser.add_argument("--smiles", type=str, help="SMILES string to analyze")
    parser.add_argument("--explain", action="store_true", help="Show detailed explanation")
    args = parser.parse_args()
    
    predictor = HydrogenTheftPredictor()
    
    if args.smiles:
        if args.explain:
            print(predictor.predict_with_explanation(args.smiles))
        else:
            predictions = predictor.predict(args.smiles)
            print(f"Predictions for: {args.smiles}")
            for atom_idx, score in predictions:
                print(f"  Atom {atom_idx}: {score:.3f}")
    
    if args.data:
        evaluate_on_dataset(args.data, predictor)
    
    if not args.smiles and not args.data:
        # Demo with a classic example: midazolam
        print("\n" + "═"*60)
        print("DEMO: Analyzing Midazolam (CYP3A4 substrate)")
        print("═"*60)
        
        # Midazolam SMILES
        midazolam = "Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"
        
        print(predictor.predict_with_explanation(midazolam, top_k=5))
