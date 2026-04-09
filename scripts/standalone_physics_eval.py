#!/usr/bin/env python3
"""Standalone Physics Scorer Evaluation.

This script evaluates the physics-based SoM scorer WITHOUT loading the ML model.
It only needs RDKit and numpy.
"""
import json
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("RDKit not available. Install with: pip install rdkit")
    raise


# ============================================================================
# CHEMISTRY RULES (from literature)
# ============================================================================

REACTIVITY_RULES = [
    # (name, SMARTS, reactivity_score, description)
    # High reactivity (>0.8)
    ("o_demethyl_aromatic", "[CH3]O[c]",           0.95, "O-demethylation"),
    ("benzylic_ch2",        "[CH2;!R][c]",         0.92, "Benzylic CH2"),
    ("benzylic_ch3",        "[CH3][c]",            0.90, "Benzylic CH3"),
    ("n_methyl",            "[CH3][NX3]",          0.88, "N-demethylation"),
    ("allylic",             "[CH2,CH3][C]=[C]",    0.85, "Allylic"),
    ("alpha_n_ch2",         "[CH2][NX3]",          0.82, "Alpha to N"),
    ("alpha_o_ch2",         "[CH2][OX2]",          0.80, "Alpha to O"),
    # Medium reactivity (0.5-0.8)
    ("s_oxidation",         "[SX2;!$([S]=*)]",     0.78, "S-oxidation"),
    ("n_oxidation",         "[NX3;H0;!$([N+])]",   0.75, "N-oxidation"),
    ("ring_n_6",            "[NX3;r6;H0]",         0.72, "Piperidine N"),
    ("thiophene_s",         "[sX2;r5]",            0.70, "Thiophene S"),
    ("epoxidation",         "[CX3]=[CX3]",         0.68, "Epoxidation"),
    ("carbonyl_alpha",      "[CH2,CH3][CX3]=O",    0.65, "Alpha to C=O"),
    # Low reactivity (<0.5)
    ("aromatic_ch",         "[cH]",                0.20, "Aromatic CH"),
    ("halogen_adjacent",    "[CH2,CH3][F,Cl,Br,I]",0.15, "Halogen-adjacent"),
]

# Compile patterns
COMPILED_PATTERNS = []
for name, smarts, score, desc in REACTIVITY_RULES:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern:
        COMPILED_PATTERNS.append((name, pattern, score, desc))


def score_molecule(smiles):
    """Score all atoms in a molecule for SoM likelihood."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    
    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()
    
    # Initialize
    scores = np.zeros(num_atoms, dtype=np.float32)
    patterns = [""] * num_atoms
    is_heavy = np.zeros(num_atoms, dtype=np.float32)
    
    # Base scores from atom type
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()
        
        if atomic_num > 1:  # Heavy atom
            is_heavy[idx] = 1.0
            # Base reactivity
            if atomic_num == 6:  # Carbon
                scores[idx] = 0.30 + 0.1 * atom.GetTotalNumHs()  # More H = more reactive
            elif atomic_num == 7:  # Nitrogen
                scores[idx] = 0.50
            elif atomic_num == 16:  # Sulfur
                scores[idx] = 0.60
            else:
                scores[idx] = 0.10
    
    # Apply SMARTS patterns
    for name, pattern, score, desc in COMPILED_PATTERNS:
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            atom_idx = match[0]
            if score > scores[atom_idx]:
                scores[atom_idx] = score
                patterns[atom_idx] = name
    
    # Mask non-heavy atoms
    scores = scores * is_heavy
    
    return scores, patterns, is_heavy


def evaluate_physics_scorer(dataset):
    """Evaluate on dataset."""
    top1_correct = 0
    top3_correct = 0
    top6_correct = 0
    total = 0
    
    for item in dataset:
        smiles = item["smiles"]
        true_sites = set(item["site_labels"])
        
        scores, patterns, is_heavy = score_molecule(smiles)
        if scores is None:
            continue
        
        # Get heavy atom indices
        heavy_idx = np.where(is_heavy > 0.5)[0]
        if len(heavy_idx) == 0:
            continue
        
        # Sort by score
        sorted_idx = heavy_idx[np.argsort(-scores[heavy_idx])]
        
        # Check predictions
        top1 = set([int(sorted_idx[0])]) if len(sorted_idx) >= 1 else set()
        top3 = set(int(i) for i in sorted_idx[:3])
        top6 = set(int(i) for i in sorted_idx[:6])
        
        if top1 & true_sites:
            top1_correct += 1
        if top3 & true_sites:
            top3_correct += 1
        if top6 & true_sites:
            top6_correct += 1
        
        total += 1
    
    return {
        "top1": top1_correct / max(total, 1),
        "top3": top3_correct / max(total, 1),
        "top6": top6_correct / max(total, 1),
        "total": total,
    }


if __name__ == "__main__":
    # Load dataset
    dataset_path = "/content/enzyme_Software/data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json"
    
    with open(dataset_path, "r") as f:
        all_data = json.load(f)
    
    # Filter for CYP3A4
    test_data = []
    for item in all_data:
        cyp = str(item.get("cyp", "")).upper()
        if "CYP3A4" not in cyp:
            continue
        site_labels = item.get("site_labels") or item.get("som_indices") or []
        if isinstance(site_labels, list) and len(site_labels) > 0:
            sites = [int(s) for s in site_labels if isinstance(s, (int, float)) and s >= 0]
            if sites:
                test_data.append({
                    "smiles": item["smiles"],
                    "site_labels": sites,
                })
    
    print(f"Loaded {len(test_data)} CYP3A4 molecules")
    
    # Evaluate
    results = evaluate_physics_scorer(test_data)
    
    print(f"\nPhysics Scorer Results:")
    print(f"  Top-1: {results['top1']:.1%}")
    print(f"  Top-3: {results['top3']:.1%}")
    print(f"  Top-6: {results['top6']:.1%}")
    print(f"  Total: {results['total']}")
