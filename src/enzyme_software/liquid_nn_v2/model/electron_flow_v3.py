#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
                         THE ELECTRON FLOW THEORY v3
                    First-Principles CYP Metabolism Prediction
═══════════════════════════════════════════════════════════════════════════════════

VERSION 3 KEY INSIGHT:

    The physics of electron donation is CORRECT, but it doesn't account for
    the STATISTICS of drug structures.
    
    α-Nitrogen sites are more REACTIVE (chemistry), but
    Benzylic sites are more COMMON (pharmaceutical chemistry).
    
    Solution: Instead of pure physics scoring, we use:
    
    1. PHYSICS to identify which sites are POSSIBLE candidates
    2. STATISTICS to weight how likely each type is to be the actual site
    
    This is a form of Bayesian reasoning:
    
        P(site | molecule) ∝ P(physics evidence | site) × P(site type)
        
    Where:
        - P(physics evidence | site) = our electron flow score
        - P(site type) = prior probability from real data

═══════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

ELECTRONEGATIVITY = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
}


# ═══════════════════════════════════════════════════════════════════════════════
# SITE TYPE PRIORS (from actual data distribution)
# ═══════════════════════════════════════════════════════════════════════════════

# These are the actual frequencies from the 869-molecule dataset
# They represent P(site type) in drugs
SITE_TYPE_PRIOR = {
    'BENZYLIC': 0.347,        # 34.7% of sites are benzylic
    'ALPHA_N': 0.182,         # 18.2% N-dealkylation
    'SECONDARY_CH': 0.102,    # 10.2% secondary C-H
    'N_OXIDATION': 0.077,     # 7.7% N-oxidation
    'PRIMARY_CH': 0.075,      # 7.5% primary C-H
    'ALPHA_O': 0.075,         # 7.5% O-dealkylation
    'ALLYLIC': 0.050,         # 5.0% allylic
    'ALPHA_CARBONYL': 0.044,  # 4.4% α-carbonyl
    'TERTIARY_CH': 0.021,     # 2.1% tertiary C-H
    'S_OXIDATION': 0.014,     # 1.4% S-oxidation
    'ALPHA_S': 0.012,         # 1.2% S-dealkylation
    'AROMATIC': 0.01,         # ~1% aromatic hydroxylation
    'OTHER': 0.01,
}


# ═══════════════════════════════════════════════════════════════════════════════
# SITE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_site(mol: Chem.Mol, atom_idx: int) -> str:
    """Classify site into reaction type."""
    atom = mol.GetAtomWithIdx(atom_idx)
    atomic_num = atom.GetAtomicNum()
    
    if atomic_num == 7:
        return 'N_OXIDATION'
    if atomic_num == 16:
        return 'S_OXIDATION'
    
    if atomic_num != 6:
        return 'OTHER'
    
    # For carbon, check neighbors
    has_n = has_o = has_s = has_aromatic = has_alkene = has_carbonyl = False
    
    for neighbor in atom.GetNeighbors():
        n_atomic = neighbor.GetAtomicNum()
        
        if n_atomic == 7:
            has_n = True
        elif n_atomic == 8:
            if neighbor.GetDegree() == 2:  # Ether oxygen
                has_o = True
            elif neighbor.GetDegree() == 1:  # Carbonyl oxygen - check if attached to neighbor
                pass
        elif n_atomic == 16:
            if neighbor.GetDegree() == 2:
                has_s = True
        elif n_atomic == 6:
            if neighbor.GetIsAromatic():
                has_aromatic = True
            elif 'SP2' in str(neighbor.GetHybridization()):
                # Check carbonyl vs alkene
                for n2 in neighbor.GetNeighbors():
                    if n2.GetAtomicNum() == 8 and n2.GetDegree() == 1:
                        has_carbonyl = True
                        break
                else:
                    has_alkene = True
    
    # Priority order (based on mechanistic dominance)
    if has_n:
        return 'ALPHA_N'
    if has_o:
        return 'ALPHA_O'
    if has_s:
        return 'ALPHA_S'
    if has_aromatic:
        return 'BENZYLIC'
    if has_carbonyl:
        return 'ALPHA_CARBONYL'
    if has_alkene:
        return 'ALLYLIC'
    
    if atom.GetIsAromatic():
        return 'AROMATIC'
    
    # Simple aliphatic
    carbon_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
    if carbon_neighbors >= 3:
        return 'TERTIARY_CH'
    elif carbon_neighbors == 2:
        return 'SECONDARY_CH'
    else:
        return 'PRIMARY_CH'


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS SCORING (intrinsic reactivity)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_intrinsic_reactivity(mol: Chem.Mol, atom_idx: int) -> float:
    """
    Compute intrinsic chemical reactivity using electron flow theory.
    
    This measures: "If the enzyme could reach this site equally well,
    how reactive would it be?"
    
    Score = E_donate × E_stable
    
    (Accessibility is handled separately by site counting)
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    atomic_num = atom.GetAtomicNum()
    
    # Filter: only C, N, S can be oxidized
    if atomic_num not in (6, 7, 16):
        return 0.0
    
    # Carbon must have H (except aromatic)
    if atomic_num == 6:
        if atom.GetTotalNumHs() == 0 and not atom.GetIsAromatic():
            return 0.0
    
    # === E_donate: electron availability ===
    base_en = ELECTRONEGATIVITY.get(atomic_num, 2.5)
    e_donate = 1.0 / base_en
    
    # Neighbor contributions (lone pair donation)
    for neighbor in atom.GetNeighbors():
        n_atomic = neighbor.GetAtomicNum()
        if n_atomic == 7:  # Nitrogen lone pair
            e_donate += 0.25
        elif n_atomic == 8 and neighbor.GetDegree() == 2:  # Ether oxygen
            e_donate += 0.20
        elif n_atomic == 16:  # Sulfur
            e_donate += 0.22
        elif neighbor.GetIsAromatic():  # Aromatic π-donation
            e_donate += 0.15
    
    # Aromatic penalty (delocalized = stable = hard to oxidize)
    if atom.GetIsAromatic():
        e_donate *= 0.5
    
    # === E_stable: radical/product stability ===
    carbon_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
    
    if atomic_num == 6:
        # Base: substitution pattern
        if carbon_neighbors >= 3:
            e_stable = 0.9
        elif carbon_neighbors == 2:
            e_stable = 0.7
        elif carbon_neighbors == 1:
            e_stable = 0.5
        else:
            e_stable = 0.3
        
        # Resonance stabilization
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIsAromatic():
                e_stable += 0.3
            elif neighbor.GetAtomicNum() == 7:
                e_stable += 0.35
            elif neighbor.GetAtomicNum() == 8 and neighbor.GetDegree() == 2:
                e_stable += 0.25
            elif neighbor.GetAtomicNum() == 16:
                e_stable += 0.30
            elif 'SP2' in str(neighbor.GetHybridization()):
                e_stable += 0.2
    else:
        # N-oxide and S-oxide are stable products
        e_stable = 0.8
    
    return e_donate * min(e_stable, 1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# THE BAYESIAN PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ElectronFlowPredictorV3:
    """
    Bayesian Electron Flow Predictor.
    
    Combines physics (intrinsic reactivity) with statistics (site type priors).
    
    Final score = reactivity × prior × competition_factor
    
    The competition factor accounts for: if there are many benzylic sites,
    any single one is less likely to be THE site.
    """
    
    def predict(
        self, 
        smiles: str,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Predict sites of metabolism."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # First pass: classify all sites and compute reactivity
        candidates = []
        type_counts = {}
        
        for atom_idx in range(mol.GetNumAtoms()):
            reactivity = compute_intrinsic_reactivity(mol, atom_idx)
            if reactivity > 0.01:
                site_type = classify_site(mol, atom_idx)
                candidates.append({
                    'idx': atom_idx,
                    'type': site_type,
                    'reactivity': reactivity,
                })
                type_counts[site_type] = type_counts.get(site_type, 0) + 1
        
        # Second pass: compute final scores
        scores = []
        
        for cand in candidates:
            site_type = cand['type']
            reactivity = cand['reactivity']
            
            # Prior from data distribution
            prior = SITE_TYPE_PRIOR.get(site_type, 0.01)
            
            # Competition: if many sites of same type, each is less likely
            n_competing = type_counts.get(site_type, 1)
            competition = 1.0 / math.sqrt(n_competing)  # sqrt dampening
            
            # Final score
            score = reactivity * prior * competition
            scores.append((cand['idx'], score))
        
        # Sort descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_on_dataset(data_path: str, verbose: bool = True) -> Dict:
    """Evaluate on a SoM dataset."""
    import json
    
    predictor = ElectronFlowPredictorV3()
    
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get("drugs", [])
    
    correct_1 = correct_2 = correct_3 = total = 0
    
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
        
        if any(s in true_sites for s in pred_sites[:1]):
            correct_1 += 1
        if any(s in true_sites for s in pred_sites[:2]):
            correct_2 += 1
        if any(s in true_sites for s in pred_sites[:3]):
            correct_3 += 1
    
    results = {
        "total": total,
        "top1": correct_1 / total if total > 0 else 0,
        "top2": correct_2 / total if total > 0 else 0,
        "top3": correct_3 / total if total > 0 else 0,
    }
    
    if verbose:
        print()
        print("═" * 70)
        print("              THE ELECTRON FLOW THEORY v3")
        print("       Physics × Statistics = Better Predictions")
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
        print("═" * 70)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        evaluate_on_dataset(sys.argv[2])
    else:
        print("Usage: python electron_flow_v3.py --eval <data.json>")
