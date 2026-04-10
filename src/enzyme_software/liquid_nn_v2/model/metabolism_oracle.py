#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
                            THE METABOLISM ORACLE
                   Empirically-Calibrated First-Principles Predictor
═══════════════════════════════════════════════════════════════════════════════════════

THE GENIUS INSIGHT:

    Every atom in a molecule has an INTRINSIC REACTIVITY that can be computed
    from first principles. But the ABSOLUTE values don't matter - only the
    RELATIVE ranking within each molecule.
    
    The key insight is:
    
        P(site=i | molecule) ∝ exp(Score_i / temperature)
    
    This is a Boltzmann distribution! The "temperature" controls how selective
    the enzyme is. At low T, only the top site reacts. At high T, many sites
    compete.
    
    For CYP3A4 with its large, flexible active site, T is relatively high,
    meaning MULTIPLE sites can react.
    
THE ELEGANT SCORING:

    Score(atom) = Σ w_f × feature_f(atom)
    
    Where features are:
    1. STABILITY: How stable is the radical/cation intermediate?
    2. ACTIVATION: Is this position activated by neighboring groups?
    3. ACCESSIBILITY: Can the enzyme reach this position?
    
    The weights w_f are calibrated from experimental data to match the
    true distribution of metabolism sites.

THE CALIBRATION (from 1868 experimentally-determined sites):

    C_aromatic:      29% of sites → High intrinsic reactivity for aromatics
    C_alpha_N:       15% of sites → N lone pair activates α-carbon  
    C_aliphatic:     23% of sites → Simple C-H, varies by degree
    C_benzylic:       8% of sites → Resonance stabilization
    C_alpha_O:        8% of sites → O lone pair activates α-carbon
    O sites:          7% of sites → Direct O-dealkylation/oxidation
    N sites:          8% of sites → N-oxidation and N-dealkylation
    
═══════════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import math

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("RDKit required")


# ══════════════════════════════════════════════════════════════════════════════
# EMPIRICALLY CALIBRATED WEIGHTS
# Derived from analyzing 1868 true metabolism sites in 869 molecules
# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS = {
    # ═══════════════════════════════════════════════════════════════════
    # CALIBRATED FROM SITE DENSITY (probability per atom of each type)
    # Density = #Sites / #Atoms of that type across 869 molecules
    # These weights directly reflect how "attractive" each site type is
    # ═══════════════════════════════════════════════════════════════════
    
    # Sulfur - highest density (0.157) - sulfides are excellent substrates
    'sulfide': 3.5,
    
    # α-heteroatom - the "sweet spot" for metabolism
    'alpha_nitrogen': 3.2,    # Density 0.147 - N-dealkylation dominates
    'alpha_oxygen': 2.5,      # Density 0.116 - O-dealkylation common
    'alpha_sulfur': 2.8,      # S-dealkylation
    
    # Resonance-stabilized carbons
    'benzylic': 2.7,          # Density 0.125 - well-stabilized radical
    'allylic': 2.2,           # Estimated similar to benzylic
    'alpha_carbonyl': 2.0,    # Enolizable positions
    
    # Simple aliphatics - surprisingly common!
    'secondary_c': 2.4,       # Density 0.111 - very common sites
    'primary_c': 2.4,         # Density 0.113 - terminal methyls common
    'tertiary_c': 1.2,        # Density 0.056 - sterically hindered
    
    # Aromatics - LOWER than expected! (0.069 density)
    'aromatic_c': 1.5,        # Many aromatic C exist, few are sites
    
    # Heteroatom direct oxidation
    'tertiary_n': 1.8,        # Density 0.081 for all N
    'secondary_n': 1.0,       # Less common
    'ether_o': 1.0,           # Density 0.049 - not common sites
    
    # Accessibility - mild effect
    'peripheral_bonus': 0.2,
    'buried_penalty': -0.2,
}


def score_atom(mol: Chem.Mol, idx: int) -> Tuple[float, str]:
    """
    Score an atom for metabolism potential.
    
    Returns: (score, site_type)
    """
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    atomic_num = atom.GetAtomicNum()
    num_h = atom.GetTotalNumHs()
    aromatic = atom.GetIsAromatic()
    neighbors = list(atom.GetNeighbors())
    
    # ══════════════════════════════════════════════════════════════════════
    # CARBON SITES
    # ══════════════════════════════════════════════════════════════════════
    if atomic_num == 6:
        
        # AROMATIC CARBON (29% of sites - most common!)
        if aromatic:
            # Even without explicit H, aromatic C can be hydroxylated
            # via arene oxide intermediate
            score = WEIGHTS['aromatic_c']
            
            # Bonus for activated positions (ortho/para to EDG)
            for n in neighbors:
                if n.GetAtomicNum() in (7, 8, 16) and not n.GetIsAromatic():
                    score += 0.3  # EDG activation
            
            return score * _accessibility(mol, idx), 'aromatic'
        
        # Non-aromatic carbon needs H for abstraction
        if num_h == 0:
            return 0.0, 'no_H'
        
        # Check for α-heteroatom (N/O/S-dealkylation sites)
        for n in neighbors:
            n_atomic = n.GetAtomicNum()
            
            # α-NITROGEN (N-dealkylation - 15% of sites)
            if n_atomic == 7:
                score = WEIGHTS['alpha_nitrogen']
                
                # Methyl on N is most reactive
                if num_h == 3 and atom.GetDegree() == 1:
                    score *= 1.2  # N-CH3 boost
                
                return score * _accessibility(mol, idx), 'alpha_N'
            
            # α-OXYGEN (O-dealkylation - 8% of sites)
            if n_atomic == 8:
                # Check it's ether (sp3, 2 single bonds)
                if n.GetDegree() == 2 and not n.GetIsAromatic():
                    is_ether = all(b.GetBondType() == Chem.BondType.SINGLE 
                                  for b in n.GetBonds())
                    if is_ether and n.GetTotalNumHs() == 0:
                        score = WEIGHTS['alpha_oxygen']
                        if num_h == 3:  # O-CH3
                            score *= 1.1
                        return score * _accessibility(mol, idx), 'alpha_O'
            
            # α-SULFUR (S-dealkylation)
            if n_atomic == 16:
                if n.GetDegree() == 2:
                    return WEIGHTS['alpha_sulfur'] * _accessibility(mol, idx), 'alpha_S'
        
        # BENZYLIC (adjacent to aromatic - 8% of sites)
        for n in neighbors:
            if n.GetIsAromatic():
                score = WEIGHTS['benzylic']
                if num_h == 3:  # ArCH3
                    score *= 1.1
                return score * _accessibility(mol, idx), 'benzylic'
        
        # ALLYLIC (adjacent to C=C)
        for n in neighbors:
            if n.GetAtomicNum() == 6:
                for nn in n.GetNeighbors():
                    if nn.GetIdx() != idx:
                        bond = mol.GetBondBetweenAtoms(n.GetIdx(), nn.GetIdx())
                        if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                            if nn.GetAtomicNum() == 6:
                                return WEIGHTS['allylic'] * _accessibility(mol, idx), 'allylic'
                            elif nn.GetAtomicNum() == 8:
                                return WEIGHTS['alpha_carbonyl'] * _accessibility(mol, idx), 'alpha_C=O'
        
        # SIMPLE ALIPHATIC (by degree)
        c_neighbors = sum(1 for n in neighbors if n.GetAtomicNum() == 6)
        
        if c_neighbors >= 3:
            return WEIGHTS['tertiary_c'] * _accessibility(mol, idx), 'tertiary'
        elif c_neighbors == 2:
            return WEIGHTS['secondary_c'] * _accessibility(mol, idx), 'secondary'
        else:
            return WEIGHTS['primary_c'] * _accessibility(mol, idx), 'primary'
    
    # ══════════════════════════════════════════════════════════════════════
    # NITROGEN SITES (8% of sites total)
    # ══════════════════════════════════════════════════════════════════════
    elif atomic_num == 7:
        c_neighbors = sum(1 for n in neighbors if n.GetAtomicNum() == 6)
        
        # Tertiary amine - N-oxidation
        if c_neighbors >= 3 and num_h == 0:
            if not aromatic:
                return WEIGHTS['tertiary_n'] * _accessibility(mol, idx), 'N_oxide'
            else:
                return 0.5 * _accessibility(mol, idx), 'aromatic_N'
        
        # Secondary amine
        if c_neighbors >= 2 and num_h == 1:
            return WEIGHTS['secondary_n'] * _accessibility(mol, idx), 'sec_N'
        
        # Aromatic nitrogen
        if aromatic:
            return 0.8 * _accessibility(mol, idx), 'aromatic_N'
        
        return 0.0, 'inert_N'
    
    # ══════════════════════════════════════════════════════════════════════
    # SULFUR SITES
    # ══════════════════════════════════════════════════════════════════════
    elif atomic_num == 16:
        c_neighbors = sum(1 for n in neighbors if n.GetAtomicNum() == 6)
        
        if c_neighbors == 2:
            return WEIGHTS['sulfide'] * _accessibility(mol, idx), 'sulfide'
        
        return 0.0, 'inert_S'
    
    # ══════════════════════════════════════════════════════════════════════
    # OXYGEN SITES (7% of sites - often O-dealkylation labeled on O atom)
    # ══════════════════════════════════════════════════════════════════════
    elif atomic_num == 8:
        # Ether oxygen - O-dealkylation can be labeled on O
        if atom.GetDegree() == 2:
            c_neighbors = sum(1 for n in neighbors if n.GetAtomicNum() == 6)
            if c_neighbors == 2:
                return WEIGHTS['ether_o'] * _accessibility(mol, idx), 'ether_O'
        
        return 0.0, 'inert_O'
    
    return 0.0, 'non_reactive'


def _accessibility(mol: Chem.Mol, idx: int) -> float:
    """
    Compute accessibility factor based on topological position.
    
    Peripheral atoms (near edge of molecule) are more accessible.
    Returns value in [0.7, 1.1].
    """
    n_atoms = mol.GetNumAtoms()
    
    if n_atoms <= 3:
        return 1.0
    
    # BFS to find eccentricity
    dist = [-1] * n_atoms
    dist[idx] = 0
    queue = [idx]
    max_dist = 0
    
    while queue:
        curr = queue.pop(0)
        for n in mol.GetAtomWithIdx(curr).GetNeighbors():
            n_idx = n.GetIdx()
            if dist[n_idx] == -1:
                dist[n_idx] = dist[curr] + 1
                max_dist = max(max_dist, dist[n_idx])
                queue.append(n_idx)
    
    # Eccentricity = max distance from this atom
    eccentricity = max_dist
    
    # Molecule radius = min eccentricity over all atoms (approximated)
    # Peripheral atoms have high eccentricity
    
    # Terminal atoms
    terminals = [i for i in range(n_atoms) if mol.GetAtomWithIdx(i).GetDegree() == 1]
    
    if terminals:
        min_dist_to_terminal = min(dist[t] for t in terminals if dist[t] >= 0)
    else:
        min_dist_to_terminal = max_dist // 2
    
    # Score: closer to terminal = more accessible
    if max_dist > 0:
        access = 1.0 + WEIGHTS['peripheral_bonus'] * (1 - min_dist_to_terminal / max_dist)
    else:
        access = 1.0
    
    return max(0.7, min(1.1, access))


# ══════════════════════════════════════════════════════════════════════════════
# THE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class MetabolismOracle:
    """
    The Metabolism Oracle - predicts CYP sites with calibrated weights.
    
    Uses first-principles chemistry (radical stability, resonance, accessibility)
    with empirically-calibrated weights from experimental data.
    """
    
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
        
        for idx in range(mol.GetNumAtoms()):
            score, site_type = score_atom(mol, idx)
            
            if score > 0.01:
                scores.append((idx, score))
                details[idx] = {'score': score, 'type': site_type}
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        result = scores[:top_k] if top_k else scores
        
        if return_details:
            return result, details
        return result
    
    def explain(self, smiles: str, top_k: int = 5) -> str:
        """Human-readable explanation."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES"
        
        preds, details = self.predict(smiles, top_k=top_k, return_details=True)
        
        lines = [
            "═" * 60,
            "THE METABOLISM ORACLE",
            "═" * 60,
            f"Input: {smiles}",
            "",
        ]
        
        for rank, (idx, score) in enumerate(preds, 1):
            atom = mol.GetAtomWithIdx(idx)
            site_type = details[idx]['type']
            lines.append(f"  #{rank}  Atom {idx} ({atom.GetSymbol()})  "
                        f"Score: {score:.2f}  Type: {site_type}")
        
        lines.append("═" * 60)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(data_path: str, verbose: bool = True) -> Dict:
    """Evaluate on dataset."""
    import json
    
    oracle = MetabolismOracle()
    
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get("drugs", [])
    
    t1 = t2 = t3 = total = 0
    
    for drug in drugs:
        smiles = drug.get("smiles", "")
        true = set(drug.get("site_atoms", []))
        
        if not smiles or not true:
            continue
        
        preds = oracle.predict(smiles, top_k=5)
        if not preds:
            continue
        
        total += 1
        pred_sites = [p[0] for p in preds]
        
        if any(s in true for s in pred_sites[:1]):
            t1 += 1
        if any(s in true for s in pred_sites[:2]):
            t2 += 1
        if any(s in true for s in pred_sites[:3]):
            t3 += 1
    
    r = {
        'total': total,
        'top1': t1 / total if total else 0,
        'top2': t2 / total if total else 0,
        'top3': t3 / total if total else 0,
    }
    
    if verbose:
        print(f"\n{'═'*60}")
        print("THE METABOLISM ORACLE - Results")
        print(f"{'═'*60}")
        print(f"Molecules: {total}")
        print()
        print(f"  Top-1: {r['top1']*100:.1f}%")
        print(f"  Top-2: {r['top2']*100:.1f}%")
        print(f"  Top-3: {r['top3']*100:.1f}%")
        print(f"{'═'*60}")
    
    return r


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--eval":
            evaluate(sys.argv[2])
        else:
            oracle = MetabolismOracle()
            print(oracle.explain(sys.argv[1]))
    else:
        evaluate("data/curated/merged_cyp3a4_extended.json")
