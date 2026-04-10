#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    THE HYDROGEN THEFT THEORY v2
                    First-Principles CYP Metabolism Prediction
═══════════════════════════════════════════════════════════════════════════════

INSIGHT FROM ERROR ANALYSIS:
    The most common CYP3A4 reactions are NOT what textbooks emphasize:
    
    1. N-DEALKYLATION (α-carbon next to N) - 40%+ of reactions
       The carbon adjacent to nitrogen loses an H, leading to C=N+
       which hydrolyzes to release an aldehyde + amine
       
    2. O-DEALKYLATION (α-carbon next to O) - 20%+ of reactions  
       Same mechanism for ethers: C-O bond cleaves after α-oxidation
       
    3. ALIPHATIC HYDROXYLATION (secondary/primary carbons) - 20%+
       Standard C-H oxidation on alkyl chains
       
    4. AROMATIC HYDROXYLATION - 10%
       Much less common than expected!
       
    5. N-OXIDATION (tertiary amines) - 5%
       Direct oxidation of nitrogen lone pair

THE KEY PRINCIPLE:
    "The heteroatom ACTIVATES its α-carbon for oxidation"
    
    Why? The nitrogen/oxygen lone pair donates electrons into the 
    developing radical, stabilizing it through hyperconjugation.
    This is why drugs with -N(CH3)2 groups almost always undergo
    N-demethylation rather than other reactions.

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdPartialCharges
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# THE REACTION HIERARCHY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ReactionType:
    """A metabolic reaction type with its relative frequency and activation."""
    name: str
    base_score: float  # Relative likelihood (higher = more common)
    description: str


# Ordered by frequency in CYP3A4 metabolism
REACTION_TYPES = {
    'N_DEALKYLATION': ReactionType(
        'N-Dealkylation', 
        2.5,  # VERY common - boost α-N carbons
        'Oxidation of carbon adjacent to nitrogen, releasing aldehyde'
    ),
    'O_DEALKYLATION': ReactionType(
        'O-Dealkylation',
        2.2,  # Very common for ethers
        'Oxidation of carbon adjacent to ether oxygen'
    ),
    'S_DEALKYLATION': ReactionType(
        'S-Dealkylation',
        1.8,  # Less common but occurs
        'Oxidation of carbon adjacent to thioether'
    ),
    'BENZYLIC': ReactionType(
        'Benzylic Hydroxylation',
        1.8,  # Common, resonance-stabilized
        'Oxidation at carbon adjacent to aromatic ring'
    ),
    'ALLYLIC': ReactionType(
        'Allylic Hydroxylation',
        1.6,  # Resonance-stabilized
        'Oxidation at carbon adjacent to double bond'
    ),
    'ALPHA_CARBONYL': ReactionType(
        'α-Carbonyl Oxidation',
        1.4,  # Enolizable position
        'Oxidation adjacent to ketone/aldehyde'
    ),
    'TERTIARY_C': ReactionType(
        'Tertiary C-H Hydroxylation',
        1.2,  # Stable radical but sterically hindered
        'Oxidation at R3C-H position'
    ),
    'SECONDARY_C': ReactionType(
        'Secondary C-H Hydroxylation',
        1.0,  # Baseline aliphatic
        'Oxidation at R2CH2 position'
    ),
    'PRIMARY_C': ReactionType(
        'Primary C-H Hydroxylation',
        0.7,  # Less stable radical
        'Oxidation at RCH3 (terminal methyl)'
    ),
    'N_OXIDATION': ReactionType(
        'N-Oxidation',
        0.8,  # Less common than expected
        'Direct oxidation of tertiary amine nitrogen'
    ),
    'S_OXIDATION': ReactionType(
        'S-Oxidation',
        1.0,  # Sulfides are good substrates
        'Oxidation of thioether sulfur'
    ),
    'AROMATIC': ReactionType(
        'Aromatic Hydroxylation',
        0.5,  # Much less common than people think!
        'Epoxidation/NIH shift on aromatic ring'
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# THE CORE SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def score_atom(mol: Chem.Mol, atom_idx: int) -> Tuple[float, str, Dict]:
    """
    Score an atom for metabolism potential.
    
    Returns:
        (score, reaction_type, details_dict)
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    symbol = atom.GetSymbol()
    atomic_num = atom.GetAtomicNum()
    num_h = atom.GetTotalNumHs()
    
    # ─────────────────────────────────────────────────────────────────────────
    # RULE 1: Must have abstractable hydrogen (for C-H oxidation)
    #         or be an oxidizable heteroatom (N, S)
    # ─────────────────────────────────────────────────────────────────────────
    
    if atomic_num == 6:  # Carbon
        if num_h == 0:
            return 0.0, 'NONE', {}
        return _score_carbon(mol, atom_idx)
    
    elif atomic_num == 7:  # Nitrogen
        return _score_nitrogen(mol, atom_idx)
    
    elif atomic_num == 16:  # Sulfur
        return _score_sulfur(mol, atom_idx)
    
    else:
        return 0.0, 'NONE', {}


def _score_carbon(mol: Chem.Mol, atom_idx: int) -> Tuple[float, str, Dict]:
    """Score a carbon atom for C-H oxidation."""
    atom = mol.GetAtomWithIdx(atom_idx)
    num_h = atom.GetTotalNumHs()
    
    details = {
        'num_h': num_h,
        'factors': [],
    }
    
    # ─────────────────────────────────────────────────────────────────────────
    # PRIORITY 1: α-HETEROATOM POSITIONS (N-dealkylation, O-dealkylation)
    # These are the MOST common CYP3A4 reactions!
    # ─────────────────────────────────────────────────────────────────────────
    
    for neighbor in atom.GetNeighbors():
        n_atomic_num = neighbor.GetAtomicNum()
        
        # α to Nitrogen - HIGHLY activated for N-dealkylation
        if n_atomic_num == 7:
            # Check if nitrogen is sp3 (amine) - these are the reactive ones
            n_neighbors = [nn for nn in neighbor.GetNeighbors()]
            n_degree = len(n_neighbors)
            
            # Tertiary/secondary amine nitrogens activate their α-carbons
            if n_degree >= 2:
                score = REACTION_TYPES['N_DEALKYLATION'].base_score
                
                # Methyl groups on N are MOST reactive (N-demethylation)
                if num_h == 3 and atom.GetDegree() == 1:
                    score *= 1.3  # N-CH3 is the classic substrate
                    details['factors'].append('N-methyl (highly reactive)')
                elif num_h == 2:
                    score *= 1.1  # N-CH2-R also common
                    details['factors'].append('N-CH2-R')
                
                # Multiple equivalent methyls? All are reactive
                details['factors'].append(f'α-nitrogen (degree {n_degree})')
                return score * _accessibility_factor(mol, atom_idx), 'N_DEALKYLATION', details
        
        # α to Oxygen - O-dealkylation (ethers, aryl ethers)
        if n_atomic_num == 8:
            # Check if it's an ether oxygen (not carbonyl, not hydroxyl)
            o_bonds = neighbor.GetBonds()
            is_ether = all(b.GetBondType() == Chem.BondType.SINGLE for b in o_bonds)
            o_h = neighbor.GetTotalNumHs()
            
            if is_ether and o_h == 0:  # Ether oxygen, no -OH
                score = REACTION_TYPES['O_DEALKYLATION'].base_score
                
                # Methoxy groups are common
                if num_h == 3 and atom.GetDegree() == 1:
                    score *= 1.2
                    details['factors'].append('O-methyl')
                
                details['factors'].append('α-ether')
                return score * _accessibility_factor(mol, atom_idx), 'O_DEALKYLATION', details
        
        # α to Sulfur - S-dealkylation
        if n_atomic_num == 16:
            score = REACTION_TYPES['S_DEALKYLATION'].base_score
            details['factors'].append('α-sulfur')
            return score * _accessibility_factor(mol, atom_idx), 'S_DEALKYLATION', details
    
    # ─────────────────────────────────────────────────────────────────────────
    # PRIORITY 2: RESONANCE-STABILIZED POSITIONS (benzylic, allylic)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Benzylic - adjacent to aromatic ring
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIsAromatic():
            score = REACTION_TYPES['BENZYLIC'].base_score
            
            # Primary benzylic (ArCH3) most reactive
            if num_h == 3:
                score *= 1.2
            # Secondary benzylic also good
            elif num_h == 2:
                score *= 1.1
            
            details['factors'].append('benzylic')
            return score * _accessibility_factor(mol, atom_idx), 'BENZYLIC', details
    
    # Allylic - adjacent to C=C
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 6:
            for nn in neighbor.GetNeighbors():
                if nn.GetIdx() != atom_idx:
                    bond = mol.GetBondBetweenAtoms(neighbor.GetIdx(), nn.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        if nn.GetAtomicNum() == 6:  # C=C, not C=O
                            score = REACTION_TYPES['ALLYLIC'].base_score
                            details['factors'].append('allylic')
                            return score * _accessibility_factor(mol, atom_idx), 'ALLYLIC', details
                        elif nn.GetAtomicNum() == 8:  # α to C=O
                            score = REACTION_TYPES['ALPHA_CARBONYL'].base_score
                            details['factors'].append('α-carbonyl')
                            return score * _accessibility_factor(mol, atom_idx), 'ALPHA_CARBONYL', details
    
    # ─────────────────────────────────────────────────────────────────────────
    # PRIORITY 3: ALIPHATIC C-H (tertiary > secondary > primary)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Count carbon neighbors
    c_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
    
    if c_neighbors >= 3:  # Tertiary
        score = REACTION_TYPES['TERTIARY_C'].base_score
        details['factors'].append('tertiary carbon')
        return score * _accessibility_factor(mol, atom_idx), 'TERTIARY_C', details
    
    elif c_neighbors == 2:  # Secondary
        score = REACTION_TYPES['SECONDARY_C'].base_score
        details['factors'].append('secondary carbon')
        return score * _accessibility_factor(mol, atom_idx), 'SECONDARY_C', details
    
    elif c_neighbors == 1:  # Primary (terminal methyl)
        score = REACTION_TYPES['PRIMARY_C'].base_score
        details['factors'].append('primary/methyl')
        return score * _accessibility_factor(mol, atom_idx), 'PRIMARY_C', details
    
    # ─────────────────────────────────────────────────────────────────────────
    # PRIORITY 4: AROMATIC C-H (less common than expected!)
    # ─────────────────────────────────────────────────────────────────────────
    
    if atom.GetIsAromatic():
        score = REACTION_TYPES['AROMATIC'].base_score
        details['factors'].append('aromatic C-H')
        
        # Check for activation by EDG (OH, OR, NR2)
        # Ortho/para to EDG are more reactive
        activation = _aromatic_activation(mol, atom_idx)
        score *= activation
        
        return score * _accessibility_factor(mol, atom_idx), 'AROMATIC', details
    
    return 0.1, 'UNKNOWN', details


def _score_nitrogen(mol: Chem.Mol, atom_idx: int) -> Tuple[float, str, Dict]:
    """Score nitrogen for N-oxidation."""
    atom = mol.GetAtomWithIdx(atom_idx)
    
    details = {'factors': []}
    
    # Only tertiary amines undergo N-oxidation
    # (Secondary/primary are too electron-poor after H-bonding)
    c_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
    h_count = atom.GetTotalNumHs()
    
    if c_neighbors == 3 and h_count == 0:
        # Tertiary amine - N-oxidation substrate
        score = REACTION_TYPES['N_OXIDATION'].base_score
        details['factors'].append('tertiary amine')
        
        # Aromatic amines are less basic, less reactive
        if atom.GetIsAromatic():
            score *= 0.3
            details['factors'].append('aromatic (deactivated)')
        
        # Check for steric hindrance
        bulky = sum(1 for n in atom.GetNeighbors() 
                   if n.GetAtomicNum() == 6 and n.GetDegree() > 2)
        if bulky >= 2:
            score *= 0.7
            details['factors'].append('sterically hindered')
        
        return score * _accessibility_factor(mol, atom_idx), 'N_OXIDATION', details
    
    return 0.0, 'NONE', details


def _score_sulfur(mol: Chem.Mol, atom_idx: int) -> Tuple[float, str, Dict]:
    """Score sulfur for S-oxidation."""
    atom = mol.GetAtomWithIdx(atom_idx)
    
    details = {'factors': []}
    
    # Sulfides (R-S-R) undergo S-oxidation
    c_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
    
    if c_neighbors == 2:
        score = REACTION_TYPES['S_OXIDATION'].base_score
        details['factors'].append('sulfide')
        return score * _accessibility_factor(mol, atom_idx), 'S_OXIDATION', details
    
    return 0.0, 'NONE', details


# ═══════════════════════════════════════════════════════════════════════════════
# ACCESSIBILITY FACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def _accessibility_factor(mol: Chem.Mol, atom_idx: int) -> float:
    """
    How accessible is this atom to the CYP active site?
    
    Uses SPAN (distance to molecular periphery) as primary metric.
    Returns value in [0.5, 1.2] - most atoms are reasonably accessible.
    """
    num_atoms = mol.GetNumAtoms()
    
    if num_atoms <= 3:
        return 1.0
    
    # BFS to find distances
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
    
    # Peripheral atoms (only 1 neighbor)
    peripheral = [i for i in range(num_atoms) 
                  if mol.GetAtomWithIdx(i).GetDegree() == 1]
    
    if not peripheral:
        peripheral = list(range(num_atoms))
    
    min_dist_to_edge = min(distances[p] for p in peripheral if distances[p] >= 0)
    
    # SPAN: closer to edge = more accessible
    if max_dist > 0:
        span = 1.0 - 0.5 * (min_dist_to_edge / (max_dist + 1))
    else:
        span = 1.0
    
    # Clamp to reasonable range [0.5, 1.2]
    return max(0.5, min(1.2, span))


def _aromatic_activation(mol: Chem.Mol, atom_idx: int) -> float:
    """
    Check if aromatic position is activated by electron-donating groups.
    Returns multiplier in [0.5, 1.5].
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    
    if not atom.GetIsAromatic():
        return 1.0
    
    # Find EDG substituents (OH, OR, NR2) on the ring
    ring_info = mol.GetRingInfo()
    
    for ring in ring_info.AtomRings():
        if atom_idx in ring and len(ring) == 6:  # 6-membered aromatic
            ring_atoms = set(ring)
            
            for ring_atom_idx in ring:
                ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
                
                for neighbor in ring_atom.GetNeighbors():
                    if neighbor.GetIdx() not in ring_atoms:
                        # Substituent found
                        n_atomic = neighbor.GetAtomicNum()
                        
                        if n_atomic == 8:  # -OH or -OR (strong EDG)
                            # Distance in ring
                            pos1 = ring.index(atom_idx)
                            pos2 = ring.index(ring_atom_idx)
                            dist = min(abs(pos1 - pos2), 6 - abs(pos1 - pos2))
                            
                            if dist in (1, 3):  # ortho or para
                                return 1.4
                        
                        elif n_atomic == 7:  # -NR2 (strong EDG)
                            pos1 = ring.index(atom_idx)
                            pos2 = ring.index(ring_atom_idx)
                            dist = min(abs(pos1 - pos2), 6 - abs(pos1 - pos2))
                            
                            if dist in (1, 3):
                                return 1.3
            break
    
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# THE PREDICTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class HydrogenTheftPredictor:
    """
    The Hydrogen Theft Theory predictor v2.
    
    Based on mechanistic understanding of CYP metabolism:
    1. N/O-dealkylation dominates (α-heteroatom carbons)
    2. Resonance-stabilized positions (benzylic, allylic) are activated
    3. Aromatic hydroxylation is less common than expected
    """
    
    def __init__(self):
        pass
    
    def predict(
        self, 
        smiles: str,
        top_k: int = 3,
        return_details: bool = False,
    ) -> List[Tuple[int, float]]:
        """Predict sites of metabolism."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [] if not return_details else ([], {})
        
        scores = []
        details = {}
        
        for atom_idx in range(mol.GetNumAtoms()):
            score, rxn_type, atom_details = score_atom(mol, atom_idx)
            
            if score > 0.01:
                scores.append((atom_idx, score, rxn_type))
                details[atom_idx] = {
                    'score': score,
                    'reaction_type': rxn_type,
                    **atom_details
                }
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        result = [(idx, sc) for idx, sc, _ in scores[:top_k]]
        
        if return_details:
            return result, details
        return result
    
    def predict_with_explanation(self, smiles: str, top_k: int = 5) -> str:
        """Generate human-readable prediction explanation."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES"
        
        predictions, details = self.predict(smiles, top_k=top_k, return_details=True)
        
        lines = [
            "═" * 60,
            "HYDROGEN THEFT ANALYSIS v2",
            "═" * 60,
            f"Molecule: {smiles}",
            "",
            "Predicted Sites (ranked by reactivity):",
        ]
        
        for rank, (atom_idx, score) in enumerate(predictions, 1):
            d = details[atom_idx]
            atom = mol.GetAtomWithIdx(atom_idx)
            rxn = d['reaction_type']
            rxn_desc = REACTION_TYPES.get(rxn, ReactionType('Unknown', 0, '')).description
            
            lines.append(f"")
            lines.append(f"  #{rank}: Atom {atom_idx} ({atom.GetSymbol()}) - Score: {score:.2f}")
            lines.append(f"      Reaction: {rxn}")
            lines.append(f"      {rxn_desc}")
            if d.get('factors'):
                lines.append(f"      Factors: {', '.join(d['factors'])}")
        
        lines.append("")
        lines.append("═" * 60)
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_on_dataset(data_path: str, verbose: bool = True) -> Dict:
    """Evaluate on a SoM dataset."""
    import json
    
    predictor = HydrogenTheftPredictor()
    
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get("drugs", [])
    
    top1 = top2 = top3 = total = 0
    
    for drug in drugs:
        smiles = drug.get("smiles", "")
        true_sites = set(drug.get("site_atoms", []))
        
        if not smiles or not true_sites:
            continue
        
        preds = predictor.predict(smiles, top_k=5)
        if not preds:
            continue
        
        total += 1
        pred_sites = [p[0] for p in preds]
        
        if any(s in true_sites for s in pred_sites[:1]):
            top1 += 1
        if any(s in true_sites for s in pred_sites[:2]):
            top2 += 1
        if any(s in true_sites for s in pred_sites[:3]):
            top3 += 1
    
    results = {
        "total": total,
        "top1": top1 / total if total > 0 else 0,
        "top2": top2 / total if total > 0 else 0,
        "top3": top3 / total if total > 0 else 0,
    }
    
    if verbose:
        print(f"\n{'═'*60}")
        print("HYDROGEN THEFT v2 - RESULTS")
        print(f"{'═'*60}")
        print(f"Molecules: {total}")
        print(f"")
        print(f"  Top-1: {results['top1']*100:.1f}%")
        print(f"  Top-2: {results['top2']*100:.1f}%")
        print(f"  Top-3: {results['top3']*100:.1f}%")
        print(f"{'═'*60}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--eval":
            evaluate_on_dataset(sys.argv[2])
        else:
            pred = HydrogenTheftPredictor()
            print(pred.predict_with_explanation(sys.argv[1]))
    else:
        # Demo
        print("\nDemo: Verapamil (CYP3A4 substrate)")
        verapamil = "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC"
        pred = HydrogenTheftPredictor()
        print(pred.predict_with_explanation(verapamil))
