#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
                              THE REACTIVITY FIELD
                        A First-Principles CYP Site Predictor
═══════════════════════════════════════════════════════════════════════════════════════

THE ONE INSIGHT:

    CYP oxidation follows the principle of least action.
    
    The enzyme will oxidize the position that requires the LEAST energy
    to reach the transition state. This is determined by THREE quantities:
    
    ΔG‡ = ΔH_bond - T·ΔS_radical + ΔG_access
    
    1. ΔH_bond    = Energy to break the C-H bond (Bond Dissociation Energy)
    2. ΔS_radical = Entropy gain from radical stabilization (resonance, hyperconjugation)
    3. ΔG_access  = Free energy cost to position the site at the heme
    
THE ELEGANT SIMPLIFICATION:

    At biological temperature and with a highly reactive oxidant (Compound I),
    the kinetics are dominated by the INTRINSIC REACTIVITY of each site:
    
        k(site) ∝ exp(-BDE/RT) × Ω_resonance × P_access
    
    Where:
        BDE         ≈ bond strength (we estimate from local environment)
        Ω_resonance = number of resonance structures for the radical
        P_access    = probability the site can reach the active site
    
THE BRILLIANT REALIZATION:

    We don't need to compute BDE explicitly!
    
    BDE is determined by the stability of the radical formed.
    Radical stability is determined by how many ways the unpaired
    electron can be delocalized.
    
    This is just COUNTING:
        - Count adjacent π systems (aromatic, C=C, C=O, lone pairs)
        - Count adjacent C-H bonds (hyperconjugation)
        - Weight by conjugation strength
    
    The SCORE is simply:
    
        Score(atom) = Σ (conjugation_weight × accessibility)
                      for each stabilizing neighbor
    
═══════════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import math

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("RDKit required")


# ══════════════════════════════════════════════════════════════════════════════
# THE CONJUGATION WEIGHTS
# These represent how effectively each structural feature stabilizes a radical
# Derived from gas-phase radical stability measurements
# ══════════════════════════════════════════════════════════════════════════════

CONJUGATION = {
    # π-conjugation (delocalization into π system)
    'aromatic': 3.0,      # Benzyl radical: ~15 kcal/mol stabilization
    'vinyl': 2.5,         # Allyl radical: ~12 kcal/mol
    'carbonyl': 2.2,      # Enolyl radical: ~10 kcal/mol
    
    # n-conjugation (lone pair donation)
    'nitrogen_sp3': 2.8,  # α-amino radical: highly stabilized
    'oxygen_sp3': 2.0,    # α-alkoxy radical: moderately stabilized  
    'sulfur': 2.5,        # α-thio radical: strong stabilization
    
    # σ-conjugation (hyperconjugation)
    'c_h_bond': 0.3,      # Each adjacent C-H: ~1.5 kcal/mol
    'c_c_bond': 0.2,      # Adjacent C-C (weaker hyperconjugation)
}


def compute_score(mol: Chem.Mol, atom_idx: int) -> Tuple[float, str, Dict]:
    """
    Compute the reactivity score for an atom.
    
    The score is the sum of all conjugation contributions,
    weighted by accessibility.
    
    Returns: (score, primary_reason, details_dict)
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    symbol = atom.GetSymbol()
    atomic_num = atom.GetAtomicNum()
    
    # ──────────────────────────────────────────────────────────────────────
    # CARBON: Score based on radical stability after H abstraction
    # ──────────────────────────────────────────────────────────────────────
    if atomic_num == 6:
        num_h = atom.GetTotalNumHs()
        
        # No H = can't undergo H abstraction (but aromatics can do epoxidation)
        if num_h == 0:
            if atom.GetIsAromatic():
                # Aromatic C without H can still undergo hydroxylation
                # via arene oxide intermediate, but it's less favored
                return 0.3, 'aromatic_no_h', {'mechanism': 'epoxidation'}
            return 0.0, 'no_hydrogen', {}
        
        # Sum up all conjugation contributions
        score = 0.0
        contributions = []
        
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            n_atomic = neighbor.GetAtomicNum()
            bond = mol.GetBondBetweenAtoms(atom_idx, n_idx)
            
            # ─── π-Conjugation ───
            if neighbor.GetIsAromatic():
                score += CONJUGATION['aromatic']
                contributions.append(('benzylic', CONJUGATION['aromatic']))
            
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                if n_atomic == 6:  # C=C
                    score += CONJUGATION['vinyl']
                    contributions.append(('allylic', CONJUGATION['vinyl']))
                elif n_atomic == 8:  # C=O
                    score += CONJUGATION['carbonyl']
                    contributions.append(('α-carbonyl', CONJUGATION['carbonyl']))
            
            # Check for C=C or C=O on the neighbor (allylic to neighbor's double bond)
            elif n_atomic == 6:
                for nn in neighbor.GetNeighbors():
                    if nn.GetIdx() == atom_idx:
                        continue
                    nn_bond = mol.GetBondBetweenAtoms(n_idx, nn.GetIdx())
                    if nn_bond and nn_bond.GetBondType() == Chem.BondType.DOUBLE:
                        if nn.GetAtomicNum() == 6:
                            score += CONJUGATION['vinyl'] * 0.8  # Slightly less direct
                            contributions.append(('allylic', CONJUGATION['vinyl'] * 0.8))
                        elif nn.GetAtomicNum() == 8:
                            score += CONJUGATION['carbonyl'] * 0.8
                            contributions.append(('α-carbonyl', CONJUGATION['carbonyl'] * 0.8))
                        break
            
            # ─── n-Conjugation (lone pairs) ───
            if n_atomic == 7:  # Nitrogen
                # Check hybridization - sp3 nitrogen donates better
                if not neighbor.GetIsAromatic():
                    score += CONJUGATION['nitrogen_sp3']
                    contributions.append(('α-nitrogen', CONJUGATION['nitrogen_sp3']))
                else:
                    score += CONJUGATION['nitrogen_sp3'] * 0.3  # Aromatic N is poor donor
                    contributions.append(('α-aromatic-N', CONJUGATION['nitrogen_sp3'] * 0.3))
            
            elif n_atomic == 8:  # Oxygen
                # Check if it's ether oxygen (sp3, 2 bonds)
                if neighbor.GetDegree() == 2 and not neighbor.GetIsAromatic():
                    # Check it's not carbonyl
                    is_ether = all(b.GetBondType() == Chem.BondType.SINGLE 
                                  for b in neighbor.GetBonds())
                    if is_ether:
                        score += CONJUGATION['oxygen_sp3']
                        contributions.append(('α-oxygen', CONJUGATION['oxygen_sp3']))
            
            elif n_atomic == 16:  # Sulfur
                if neighbor.GetDegree() == 2:
                    score += CONJUGATION['sulfur']
                    contributions.append(('α-sulfur', CONJUGATION['sulfur']))
            
            # ─── σ-Conjugation (hyperconjugation) ───
            if n_atomic == 6:  # Adjacent carbon
                n_h = neighbor.GetTotalNumHs()
                score += n_h * CONJUGATION['c_h_bond']
                if n_h > 0:
                    contributions.append(('hyperconj', n_h * CONJUGATION['c_h_bond']))
        
        # Base score for having any H (even without special stabilization)
        if score == 0:
            # Count degree to get base reactivity
            degree = atom.GetDegree()
            if degree >= 3:
                score = 0.8  # Tertiary C-H
                contributions.append(('tertiary', 0.8))
            elif degree == 2:
                score = 0.5  # Secondary C-H
                contributions.append(('secondary', 0.5))
            else:
                score = 0.3  # Primary C-H
                contributions.append(('primary', 0.3))
        
        # Apply accessibility factor
        access = _compute_accessibility(mol, atom_idx)
        final_score = score * access
        
        # Determine primary reason
        if contributions:
            primary = max(contributions, key=lambda x: x[1])[0]
        else:
            primary = 'aliphatic'
        
        return final_score, primary, {
            'base_score': score,
            'accessibility': access,
            'contributions': contributions,
        }
    
    # ──────────────────────────────────────────────────────────────────────
    # NITROGEN: N-oxidation of tertiary amines
    # ──────────────────────────────────────────────────────────────────────
    elif atomic_num == 7:
        # Only tertiary amines undergo significant N-oxidation
        c_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
        h_count = atom.GetTotalNumHs()
        
        if c_neighbors >= 3 and h_count == 0 and not atom.GetIsAromatic():
            access = _compute_accessibility(mol, atom_idx)
            # N-oxidation competes with N-dealkylation on adjacent carbons
            # It's generally less favored
            return 1.5 * access, 'N-oxidation', {'type': 'tertiary_amine'}
        
        # Secondary amines can also be oxidized but less commonly
        if c_neighbors >= 2 and h_count == 1 and not atom.GetIsAromatic():
            access = _compute_accessibility(mol, atom_idx)
            return 0.6 * access, 'N-oxidation', {'type': 'secondary_amine'}
        
        return 0.0, 'inert_N', {}
    
    # ──────────────────────────────────────────────────────────────────────
    # SULFUR: S-oxidation
    # ──────────────────────────────────────────────────────────────────────
    elif atomic_num == 16:
        c_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
        
        if c_neighbors == 2:
            access = _compute_accessibility(mol, atom_idx)
            return 2.0 * access, 'S-oxidation', {'type': 'sulfide'}
        
        return 0.0, 'inert_S', {}
    
    return 0.0, 'non_reactive', {}


def _compute_accessibility(mol: Chem.Mol, atom_idx: int) -> float:
    """
    Compute accessibility to the CYP active site.
    
    Based on topological distance from molecular periphery.
    Peripheral atoms are more accessible.
    
    Returns value in [0.5, 1.0] - most atoms are reasonably accessible.
    """
    n_atoms = mol.GetNumAtoms()
    
    if n_atoms <= 2:
        return 1.0
    
    # Find maximum distance from this atom to any other
    distances = _bfs_distances(mol, atom_idx)
    max_dist = max(d for d in distances if d >= 0)
    
    # Find terminal atoms (degree 1)
    terminals = [i for i in range(n_atoms) 
                 if mol.GetAtomWithIdx(i).GetDegree() == 1]
    
    if not terminals:
        return 0.8  # Fully cyclic molecule
    
    # Distance to nearest terminal
    min_dist = min(distances[t] for t in terminals if distances[t] >= 0)
    
    # Normalize: closer to terminal = more accessible
    if max_dist > 0:
        accessibility = 1.0 - 0.5 * (min_dist / max_dist)
    else:
        accessibility = 1.0
    
    return max(0.5, accessibility)


def _bfs_distances(mol: Chem.Mol, start: int) -> List[int]:
    """BFS to compute distances from start atom to all others."""
    n = mol.GetNumAtoms()
    dist = [-1] * n
    dist[start] = 0
    queue = [start]
    
    while queue:
        curr = queue.pop(0)
        for neighbor in mol.GetAtomWithIdx(curr).GetNeighbors():
            n_idx = neighbor.GetIdx()
            if dist[n_idx] == -1:
                dist[n_idx] = dist[curr] + 1
                queue.append(n_idx)
    
    return dist


# ══════════════════════════════════════════════════════════════════════════════
# THE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class ReactivityFieldPredictor:
    """
    Predicts CYP sites of metabolism using the Reactivity Field model.
    
    The model scores each atom based on how well the resulting radical
    would be stabilized by conjugation with neighboring π systems,
    lone pairs, and σ bonds.
    """
    
    def predict(
        self, 
        smiles: str, 
        top_k: int = 3,
        return_details: bool = False,
    ) -> List[Tuple[int, float]]:
        """
        Predict sites of metabolism.
        
        Args:
            smiles: Input molecule as SMILES
            top_k: Number of top predictions to return
            return_details: If True, return full details dict
            
        Returns:
            List of (atom_idx, score) tuples
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [] if not return_details else ([], {})
        
        scores = []
        details = {}
        
        for atom_idx in range(mol.GetNumAtoms()):
            score, reason, detail = compute_score(mol, atom_idx)
            
            if score > 0.01:
                scores.append((atom_idx, score))
                details[atom_idx] = {'score': score, 'reason': reason, **detail}
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        result = scores[:top_k] if top_k else scores
        
        if return_details:
            return result, details
        return result
    
    def explain(self, smiles: str, top_k: int = 5) -> str:
        """Generate human-readable explanation."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES"
        
        preds, details = self.predict(smiles, top_k=top_k, return_details=True)
        
        lines = [
            "═" * 65,
            "THE REACTIVITY FIELD - Site of Metabolism Prediction",
            "═" * 65,
            "",
            f"Molecule: {smiles}",
            "",
            "Principle: Score = Σ(conjugation) × accessibility",
            "",
            "─" * 65,
            "PREDICTIONS (ranked by reactivity):",
            "─" * 65,
        ]
        
        for rank, (idx, score) in enumerate(preds, 1):
            atom = mol.GetAtomWithIdx(idx)
            d = details[idx]
            
            lines.append(f"")
            lines.append(f"#{rank}  Atom {idx} ({atom.GetSymbol()})  │  Score: {score:.2f}")
            lines.append(f"    Primary factor: {d['reason']}")
            
            if 'contributions' in d and d['contributions']:
                contribs = ', '.join(f"{c[0]}:{c[1]:.1f}" for c in d['contributions'][:3])
                lines.append(f"    Contributions: {contribs}")
            
            if 'accessibility' in d:
                lines.append(f"    Accessibility: {d['accessibility']:.2f}")
        
        lines.append("")
        lines.append("═" * 65)
        
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(data_path: str, verbose: bool = True) -> Dict:
    """Evaluate on SoM dataset."""
    import json
    
    pred = ReactivityFieldPredictor()
    
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get("drugs", [])
    
    t1 = t2 = t3 = total = 0
    
    for drug in drugs:
        smiles = drug.get("smiles", "")
        true = set(drug.get("site_atoms", []))
        
        if not smiles or not true:
            continue
        
        results = pred.predict(smiles, top_k=5)
        if not results:
            continue
        
        total += 1
        predicted = [r[0] for r in results]
        
        if any(p in true for p in predicted[:1]):
            t1 += 1
        if any(p in true for p in predicted[:2]):
            t2 += 1
        if any(p in true for p in predicted[:3]):
            t3 += 1
    
    results = {
        'total': total,
        'top1': t1 / total if total else 0,
        'top2': t2 / total if total else 0,
        'top3': t3 / total if total else 0,
    }
    
    if verbose:
        print(f"\n{'═'*65}")
        print("THE REACTIVITY FIELD - Evaluation Results")
        print(f"{'═'*65}")
        print(f"Molecules evaluated: {total}")
        print()
        print(f"  Top-1 Accuracy: {results['top1']*100:.1f}%")
        print(f"  Top-2 Accuracy: {results['top2']*100:.1f}%")
        print(f"  Top-3 Accuracy: {results['top3']*100:.1f}%")
        print(f"{'═'*65}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--eval":
            evaluate(sys.argv[2])
        else:
            pred = ReactivityFieldPredictor()
            print(pred.explain(sys.argv[1]))
    else:
        # Demo
        pred = ReactivityFieldPredictor()
        
        print("\n" + "="*65)
        print("DEMO: Testosterone (classic CYP3A4 substrate)")
        print("="*65)
        testosterone = "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O"
        print(pred.explain(testosterone))
