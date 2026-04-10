#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
                         THE ELECTRON FLOW THEORY
                    First-Principles CYP Metabolism Prediction
═══════════════════════════════════════════════════════════════════════════════════

                    "Metabolism is electron theft"

THE INSIGHT:
    CYP's Compound I (Fe(IV)=O) is nature's most electrophilic oxidant.
    It doesn't see "reaction types" - it sees ELECTRONS.
    
    The site that gets oxidized is simply the one that:
    
    1. Can DONATE electrons most easily (lowest local ionization potential)
    2. Is ACCESSIBLE to the heme iron (not buried)
    3. Creates a STABLE product (thermodynamic driving force)
    
    That's it. No SMARTS patterns. No reaction type lookup tables.
    Just the physics of electron flow.

THE THREE NUMBERS:
    For each atom, we compute:
    
    E_donate = How easily it gives up an electron (electronegativity-based)
    E_access = How exposed it is to the enzyme (surface accessibility)  
    E_stable = How stable the oxidation product is (radical stability)
    
    Final score = E_donate × E_access × E_stable
    
    The highest score wins.

THE ELEGANCE:
    This single equation explains ALL CYP reactions:
    
    - N-dealkylation: α-nitrogen carbon has LOW ionization (N donates electrons)
                      HIGH accessibility (terminal methyl), STABLE product (imine)
                      
    - Benzylic: Aromatic ring DONATES electrons into adjacent carbon,
                radical is STABILIZED by resonance
                
    - Aromatic: Hard! Carbon IN the ring has HIGH ionization potential
                (delocalized electrons are stable), hence LESS reactive
                
    - Tertiary C: STABLE radical (3 alkyl groups donate electrons)
                  but LOW accessibility (buried)

═══════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdPartialCharges
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS (Pauling electronegativity scale)
# ═══════════════════════════════════════════════════════════════════════════════

ELECTRONEGATIVITY = {
    1: 2.20,   # H
    6: 2.55,   # C
    7: 3.04,   # N - electron withdrawing but also lone pair donor
    8: 3.44,   # O - strong electron withdrawing  
    9: 3.98,   # F
    15: 2.19,  # P
    16: 2.58,  # S - weak electronegativity, good electron donor
    17: 3.16,  # Cl
    35: 2.96,  # Br
    53: 2.66,  # I
}

# Covalent radii in Angstroms (for accessibility)
COVALENT_RADIUS = {
    1: 0.31,   # H
    6: 0.77,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    9: 0.57,   # F
    15: 1.07,  # P
    16: 1.05,  # S
    17: 1.02,  # Cl
    35: 1.20,  # Br
    53: 1.39,  # I
}


# ═══════════════════════════════════════════════════════════════════════════════
# THE THREE ENERGIES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_electron_donation(mol: Chem.Mol, atom_idx: int) -> float:
    """
    E_donate: How easily can this site donate an electron?
    
    The key insight: electrons flow FROM low electronegativity TO high.
    A carbon next to nitrogen can donate electrons because the nitrogen's
    lone pair pushes electron density onto the carbon (inductive + mesomeric).
    
    We use a LOCAL electron density estimate based on:
    1. The atom's own electronegativity (lower = better donor)
    2. Neighboring atoms' ability to donate electrons INTO this atom
    3. Resonance effects (aromatic, conjugated systems)
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    atomic_num = atom.GetAtomicNum()
    
    # Base: inverse electronegativity (lower EN = better electron donor)
    base_en = ELECTRONEGATIVITY.get(atomic_num, 2.5)
    e_donate = 1.0 / base_en  # Will be ~0.4 for C, ~0.33 for N, ~0.29 for O
    
    # Now the key: what do the neighbors contribute?
    # A neighbor with LONE PAIRS (N, O, S) can DONATE electrons into this atom
    # This is the α-heteroatom effect that makes N-dealkylation so common
    
    neighbor_donation = 0.0
    
    for neighbor in atom.GetNeighbors():
        n_atomic = neighbor.GetAtomicNum()
        n_hybrid = str(neighbor.GetHybridization())
        
        if n_atomic == 7:  # Nitrogen - STRONG lone pair donor
            # sp3 nitrogen: lone pair available
            # sp2 nitrogen: lone pair in p-orbital, can donate via resonance
            if 'SP3' in n_hybrid:
                # Check if it's a tertiary amine (most donating) vs secondary vs primary
                n_degree = neighbor.GetDegree()
                if n_degree == 3:  # Tertiary amine: lone pair fully available
                    neighbor_donation += 0.6
                elif n_degree == 2:  # Secondary amine
                    neighbor_donation += 0.5
                else:  # Primary amine
                    neighbor_donation += 0.4
            elif 'SP2' in n_hybrid:
                # In aromatic ring or imine - still donates via resonance
                neighbor_donation += 0.35
                
        elif n_atomic == 8:  # Oxygen
            # Ether oxygen: lone pairs available
            # Hydroxyl: less donating due to H
            if neighbor.GetDegree() == 2:
                # Check if ether (two carbons) vs hydroxyl (one H)
                neighbor_h = sum(1 for n2 in neighbor.GetNeighbors() 
                               if n2.GetAtomicNum() == 1)
                if neighbor_h == 0:  # Ether
                    neighbor_donation += 0.5
                else:  # Alcohol
                    neighbor_donation += 0.25
                    
        elif n_atomic == 16:  # Sulfur - excellent donor
            if neighbor.GetDegree() == 2:  # Thioether
                neighbor_donation += 0.55
                
        elif n_atomic == 6:  # Carbon neighbor
            # If neighbor is aromatic, this carbon gets electron density via resonance
            if neighbor.GetIsAromatic():
                neighbor_donation += 0.2
            # If neighbor is sp2 (alkene), resonance donation
            elif 'SP2' in str(neighbor.GetHybridization()):
                neighbor_donation += 0.15
    
    # Aromatic carbons: electrons are DELOCALIZED, harder to extract
    if atom.GetIsAromatic():
        e_donate *= 0.4  # Significant penalty - delocalized electrons are stable
        
        # But check for activating groups (EDG) on the ring
        ring_activation = _get_ring_activation(mol, atom_idx)
        e_donate *= ring_activation
    
    # Combine base + neighbor donation
    e_donate += neighbor_donation * 0.5  # Scale factor
    
    return e_donate


def compute_accessibility(mol: Chem.Mol, atom_idx: int) -> float:
    """
    E_access: Can the enzyme reach this site?
    
    Simple model based on:
    1. Having hydrogens to abstract
    2. Not being buried (high degree = more buried)
    3. Small ring penalty
    
    Key insight: CYP is flexible, so most sites ARE accessible.
    The main filter is: does this atom have a hydrogen?
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    atomic_num = atom.GetAtomicNum()
    total_h = atom.GetTotalNumHs()
    degree = atom.GetDegree()
    
    if atomic_num == 6:  # Carbon
        # Must have hydrogen to abstract (except aromatic - epoxide)
        if total_h == 0 and not atom.GetIsAromatic():
            return 0.0
        
        # Simple accessibility: more H's slightly better
        # But degree doesn't matter much - CYP is flexible
        if total_h >= 2:
            access = 0.9
        elif total_h == 1:
            access = 0.8
        else:  # aromatic, no H
            access = 0.5
        
        # Small ring penalty
        if atom.IsInRing():
            ring_size = _min_ring_size(mol, atom_idx)
            if ring_size <= 5:
                access *= 0.9
        
        return access
        
    elif atomic_num == 7:  # Nitrogen
        if degree == 3 and total_h == 0:  # Tertiary amine
            return 0.8
        elif degree == 2:
            return 0.7
        return 0.5
            
    elif atomic_num == 16:  # Sulfur
        if degree == 2:
            return 0.8
        return 0.5
        
    else:
        return 0.1


def compute_product_stability(mol: Chem.Mol, atom_idx: int) -> float:
    """
    E_stable: Is the oxidation product stable?
    
    After hydrogen abstraction, we form a carbon radical. 
    Radical stability is THE key factor that determines reactivity.
    
    THE RADICAL STABILITY HIERARCHY (this is physical chemistry):
    
    1. α-Heteroatom (N, O, S adjacent) - MOST STABLE
       The lone pair donates into the radical via hyperconjugation.
       This is why N-dealkylation is SO common!
       
    2. Resonance-stabilized (benzylic, allylic) - VERY STABLE
       Radical delocalizes into π system.
       
    3. Tertiary radical - STABLE
       Three alkyl groups donate electrons inductively.
       
    4. Secondary radical - MODERATE
       Two alkyl groups.
       
    5. Primary radical - POOR
       One alkyl group.
       
    6. Methyl radical - WORST
       No stabilization at all.
    
    The key insight is that radical stability is MULTIPLICATIVE
    with other factors. A benzylic + α-nitrogen site is exceptional.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    atomic_num = atom.GetAtomicNum()
    
    if atomic_num != 6:
        # For N and S oxidation, product is N-oxide or sulfoxide
        if atomic_num == 7:
            return 0.8  # N-oxides are stable
        elif atomic_num == 16:
            return 0.9  # Sulfoxides are stable
        return 0.3
    
    # For carbon: evaluate radical stability
    # Start with substitution pattern
    carbon_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
    
    # Substitution-based stability
    if carbon_neighbors >= 3:
        stability = 0.9  # Tertiary - stable
    elif carbon_neighbors == 2:
        stability = 0.7  # Secondary
    elif carbon_neighbors == 1:
        stability = 0.5  # Primary
    else:
        stability = 0.3  # Methyl (no C neighbors)
    
    # Now add stabilization from special environments
    # These are ADDITIVE because each represents a different stabilization mechanism
    
    stabilization_bonus = 0.0
    
    for neighbor in atom.GetNeighbors():
        n_atomic = neighbor.GetAtomicNum()
        
        # α-Heteroatom: THE BIG ONE
        # Lone pair hyperconjugation into radical
        if n_atomic == 7:  # Nitrogen
            # Check nitrogen type
            n_hybrid = str(neighbor.GetHybridization())
            if 'SP3' in n_hybrid:
                n_degree = neighbor.GetDegree()
                if n_degree >= 3:  # Tertiary amine
                    stabilization_bonus += 0.6  # HUGE
                elif n_degree == 2:  # Secondary amine
                    stabilization_bonus += 0.5
                else:
                    stabilization_bonus += 0.4
            else:  # sp2 nitrogen (aromatic, imine)
                stabilization_bonus += 0.35
                
        elif n_atomic == 8:  # Oxygen
            if neighbor.GetDegree() == 2:  # Ether
                stabilization_bonus += 0.5  # Significant
            elif neighbor.GetDegree() == 1:  # Carbonyl oxygen
                stabilization_bonus += 0.3  # α-keto position
                
        elif n_atomic == 16:  # Sulfur
            if neighbor.GetDegree() == 2:  # Thioether
                stabilization_bonus += 0.55  # Excellent
        
        # Resonance stabilization
        elif n_atomic == 6:
            # Benzylic: adjacent to aromatic ring
            if neighbor.GetIsAromatic():
                stabilization_bonus += 0.45  # Excellent
                
            # Allylic: adjacent to double bond
            elif 'SP2' in str(neighbor.GetHybridization()):
                # Check if it's an alkene (not carbonyl, we handled that)
                is_alkene = True
                for n2 in neighbor.GetNeighbors():
                    if n2.GetAtomicNum() == 8 and n2.GetDegree() == 1:
                        is_alkene = False
                        break
                if is_alkene:
                    stabilization_bonus += 0.35
    
    # Total stability = base + bonus
    total = stability + stabilization_bonus
    
    return min(total, 1.8)  # Cap to prevent runaway


def _min_ring_size(mol: Chem.Mol, atom_idx: int) -> int:
    """Get minimum ring size containing this atom."""
    ring_info = mol.GetRingInfo()
    min_size = 100
    for ring in ring_info.AtomRings():
        if atom_idx in ring:
            min_size = min(min_size, len(ring))
    return min_size if min_size < 100 else 0


def _get_ring_activation(mol: Chem.Mol, atom_idx: int) -> float:
    """
    For aromatic atoms: check if EDG activates this position.
    
    Electron-donating groups (OH, OR, NR2) activate ortho/para positions.
    Electron-withdrawing groups (NO2, CN, COR) deactivate.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    if not atom.GetIsAromatic():
        return 1.0
    
    ring_info = mol.GetRingInfo()
    activation = 1.0
    
    for ring in ring_info.AtomRings():
        if atom_idx in ring and len(ring) == 6:
            ring_atoms = set(ring)
            
            for i, ring_atom_idx in enumerate(ring):
                ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
                
                for neighbor in ring_atom.GetNeighbors():
                    if neighbor.GetIdx() in ring_atoms:
                        continue
                    
                    n_atomic = neighbor.GetAtomicNum()
                    
                    # Calculate position relative to our atom
                    pos_me = ring.index(atom_idx)
                    pos_sub = ring.index(ring_atom_idx)
                    dist = min(abs(pos_me - pos_sub), 6 - abs(pos_me - pos_sub))
                    
                    # EDG at ortho (dist=1) or para (dist=3)
                    if dist in (1, 3):
                        if n_atomic == 8:  # -OH or -OR
                            activation *= 1.4
                        elif n_atomic == 7:  # -NH2 or -NR2
                            if not neighbor.GetIsAromatic():
                                activation *= 1.3
                                
                    # EWG at any position deactivates
                    # Check for carbonyl, nitro, cyano
                    if n_atomic == 6:
                        for n2 in neighbor.GetNeighbors():
                            if n2.GetAtomicNum() == 8 and n2.GetDegree() == 1:
                                activation *= 0.8  # Carbonyl is EWG
                            elif n2.GetAtomicNum() == 7 and n2.GetDegree() == 1:
                                activation *= 0.7  # Cyano is strong EWG
            break
    
    return activation


# ═══════════════════════════════════════════════════════════════════════════════
# THE UNIFIED EQUATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_oxidation_score(mol: Chem.Mol, atom_idx: int) -> Tuple[float, Dict]:
    """
    The unified oxidation score.
    
    score = E_donate × E_access × E_stable
    
    This single equation predicts ALL CYP oxidation sites.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    atomic_num = atom.GetAtomicNum()
    
    # Only score C, N, S atoms (the ones CYP oxidizes)
    if atomic_num not in (6, 7, 16):
        return 0.0, {}
    
    # Carbon must have hydrogens to abstract (except aromatic - epoxide pathway)
    if atomic_num == 6:
        if atom.GetTotalNumHs() == 0 and not atom.GetIsAromatic():
            return 0.0, {}
    
    # Compute the three energies
    e_donate = compute_electron_donation(mol, atom_idx)
    e_access = compute_accessibility(mol, atom_idx)
    e_stable = compute_product_stability(mol, atom_idx)
    
    # THE EQUATION
    score = e_donate * e_access * e_stable
    
    # Details for debugging
    details = {
        'e_donate': e_donate,
        'e_access': e_access,
        'e_stable': e_stable,
        'symbol': atom.GetSymbol(),
        'aromatic': atom.GetIsAromatic(),
        'degree': atom.GetDegree(),
        'total_h': atom.GetTotalNumHs(),
    }
    
    return score, details


# ═══════════════════════════════════════════════════════════════════════════════
# THE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ElectronFlowPredictor:
    """
    The Electron Flow Theory predictor.
    
    "Which site can most easily donate electrons to the enzyme?"
    """
    
    def predict(
        self, 
        smiles: str,
        top_k: int = 5,
        return_details: bool = False,
    ) -> List[Tuple[int, float]]:
        """Predict sites of metabolism."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [] if not return_details else ([], {})
        
        scores = []
        details = {}
        
        for atom_idx in range(mol.GetNumAtoms()):
            score, atom_details = compute_oxidation_score(mol, atom_idx)
            
            if score > 0.001:
                scores.append((atom_idx, score))
                details[atom_idx] = atom_details
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        result = scores[:top_k]
        
        if return_details:
            return result, details
        return result
    
    def explain(self, smiles: str, top_k: int = 5) -> str:
        """Generate human-readable explanation."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES"
        
        predictions, details = self.predict(smiles, top_k=top_k, return_details=True)
        
        lines = [
            "",
            "═" * 70,
            "              THE ELECTRON FLOW THEORY",
            "         \"Metabolism is electron theft\"",
            "═" * 70,
            "",
            f"  Molecule: {smiles}",
            "",
            "  For each site:  Score = E_donate × E_access × E_stable",
            "",
            "  E_donate = How easily it gives up electrons",
            "  E_access = How exposed it is to the enzyme", 
            "  E_stable = How stable the oxidation product is",
            "",
            "─" * 70,
            "  PREDICTED SITES (ranked by oxidation score)",
            "─" * 70,
        ]
        
        for rank, (atom_idx, score) in enumerate(predictions, 1):
            d = details[atom_idx]
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # Determine what kind of site this is
            site_type = _classify_site(mol, atom_idx)
            
            lines.append(f"")
            lines.append(f"  #{rank}: Atom {atom_idx} ({atom.GetSymbol()})")
            lines.append(f"      Type: {site_type}")
            lines.append(f"      Score: {score:.4f}")
            lines.append(f"      ├─ E_donate: {d['e_donate']:.3f}")
            lines.append(f"      ├─ E_access: {d['e_access']:.3f}")
            lines.append(f"      └─ E_stable: {d['e_stable']:.3f}")
        
        lines.append("")
        lines.append("═" * 70)
        
        return "\n".join(lines)


def _classify_site(mol: Chem.Mol, atom_idx: int) -> str:
    """Classify the site type for human readability."""
    atom = mol.GetAtomWithIdx(atom_idx)
    atomic_num = atom.GetAtomicNum()
    
    if atomic_num == 7:
        return "N-oxidation"
    if atomic_num == 16:
        return "S-oxidation"
    
    # Carbon - determine the type
    for neighbor in atom.GetNeighbors():
        n_atomic = neighbor.GetAtomicNum()
        
        if n_atomic == 7:
            return "α-Nitrogen (N-dealkylation)"
        if n_atomic == 8 and neighbor.GetDegree() == 2:
            return "α-Oxygen (O-dealkylation)"
        if n_atomic == 16:
            return "α-Sulfur (S-dealkylation)"
        if neighbor.GetIsAromatic():
            return "Benzylic"
        if n_atomic == 6 and 'SP2' in str(neighbor.GetHybridization()):
            # Check if carbonyl or alkene
            for n2 in neighbor.GetNeighbors():
                if n2.GetAtomicNum() == 8 and n2.GetDegree() == 1:
                    return "α-Carbonyl"
            return "Allylic"
    
    if atom.GetIsAromatic():
        return "Aromatic"
    
    degree = atom.GetDegree()
    if degree >= 3:
        return "Tertiary C-H"
    elif degree == 2:
        return "Secondary C-H"
    else:
        return "Primary C-H"


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_on_dataset(data_path: str, verbose: bool = True) -> Dict:
    """Evaluate on a SoM dataset."""
    import json
    
    predictor = ElectronFlowPredictor()
    
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get("drugs", [])
    
    correct_1 = correct_2 = correct_3 = total = 0
    
    # For error analysis
    site_type_stats = {
        'correct': {},
        'incorrect_pred': {},
        'missed': {},
    }
    
    for drug in drugs:
        smiles = drug.get("smiles", "")
        true_sites = set(drug.get("site_atoms", []))
        
        if not smiles or not true_sites:
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        preds = predictor.predict(smiles, top_k=5)
        if not preds:
            continue
        
        total += 1
        pred_sites = [p[0] for p in preds]
        
        # Top-k accuracy
        if any(s in true_sites for s in pred_sites[:1]):
            correct_1 += 1
        if any(s in true_sites for s in pred_sites[:2]):
            correct_2 += 1
        if any(s in true_sites for s in pred_sites[:3]):
            correct_3 += 1
        
        # Error analysis
        for pred_idx, _ in preds[:3]:
            site_type = _classify_site(mol, pred_idx)
            if pred_idx in true_sites:
                site_type_stats['correct'][site_type] = \
                    site_type_stats['correct'].get(site_type, 0) + 1
            else:
                site_type_stats['incorrect_pred'][site_type] = \
                    site_type_stats['incorrect_pred'].get(site_type, 0) + 1
        
        for true_idx in true_sites:
            if true_idx not in pred_sites[:3]:
                site_type = _classify_site(mol, true_idx)
                site_type_stats['missed'][site_type] = \
                    site_type_stats['missed'].get(site_type, 0) + 1
    
    results = {
        "total": total,
        "top1": correct_1 / total if total > 0 else 0,
        "top2": correct_2 / total if total > 0 else 0,
        "top3": correct_3 / total if total > 0 else 0,
        "site_stats": site_type_stats,
    }
    
    if verbose:
        print()
        print("═" * 70)
        print("              THE ELECTRON FLOW THEORY - RESULTS")
        print("         \"Score = E_donate × E_access × E_stable\"")
        print("═" * 70)
        print()
        print(f"  Molecules evaluated: {total}")
        print()
        print(f"  ┌─────────────────────────────────────┐")
        print(f"  │  Top-1 Accuracy:  {results['top1']*100:5.1f}%            │")
        print(f"  │  Top-2 Accuracy:  {results['top2']*100:5.1f}%            │")
        print(f"  │  Top-3 Accuracy:  {results['top3']*100:5.1f}%            │")
        print(f"  └─────────────────────────────────────┘")
        print()
        
        print("─" * 70)
        print("  CORRECT PREDICTIONS BY SITE TYPE")
        print("─" * 70)
        for stype, count in sorted(site_type_stats['correct'].items(), 
                                   key=lambda x: -x[1])[:8]:
            print(f"    {stype:30s} {count:4d}")
        
        print()
        print("─" * 70)
        print("  OVER-PREDICTED (predicted but not true)")
        print("─" * 70)
        for stype, count in sorted(site_type_stats['incorrect_pred'].items(), 
                                   key=lambda x: -x[1])[:8]:
            print(f"    {stype:30s} {count:4d}")
        
        print()
        print("─" * 70)
        print("  MISSED (true but not predicted in top-3)")
        print("─" * 70)
        for stype, count in sorted(site_type_stats['missed'].items(), 
                                   key=lambda x: -x[1])[:8]:
            print(f"    {stype:30s} {count:4d}")
        
        print()
        print("═" * 70)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--eval":
            evaluate_on_dataset(sys.argv[2])
        else:
            pred = ElectronFlowPredictor()
            print(pred.explain(sys.argv[1]))
    else:
        # Demo
        print("\n" + "="*70)
        print("  DEMO: The Electron Flow Theory")
        print("="*70)
        
        # Verapamil - classic CYP3A4 substrate with N-dealkylation
        verapamil = "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC"
        
        pred = ElectronFlowPredictor()
        print(pred.explain(verapamil))
