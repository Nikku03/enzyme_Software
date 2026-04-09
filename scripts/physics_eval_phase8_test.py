#!/usr/bin/env python3
"""
Physics-Only Evaluation on Phase 8 Test Set

This evaluates the physics scorer on the SAME test set as Phase 5,
providing a baseline for comparison.

Run in Colab:
  exec(open('/content/enzyme_Software/scripts/physics_eval_phase8_test.py').read())
"""

import json
import numpy as np

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("RDKit required. Install with: pip install rdkit")

print("=" * 70)
print("PHYSICS EVALUATION ON PHASE 8 TEST SET")
print("=" * 70)

# ============================================================================
# Physics Scorer
# ============================================================================

REACTIVITY_RULES = [
    ("o_demethyl_aromatic", "[CH3]O[c]", 0.95),
    ("o_demethyl_aliphatic", "[CH3]O[C;!c]", 0.88),
    ("benzylic_ch2", "[CH2;!R][c]", 0.92),
    ("benzylic_ch3", "[CH3][c]", 0.90),
    ("n_demethyl", "[CH3][NX3]", 0.88),
    ("allylic", "[CH2,CH3][C]=[C]", 0.85),
    ("alpha_n_ch2", "[CH2][NX3]", 0.82),
    ("alpha_o_ch2", "[CH2][OX2]", 0.80),
    ("s_oxidation", "[SX2;!$([S]=*)]", 0.78),
    ("n_oxidation", "[NX3;H0;!$([N+])]", 0.75),
    ("ring_n_piperidine", "[NX3;r6;H0]", 0.72),
    ("thiophene_s", "[sX2;r5]", 0.70),
    ("hydroxylation_tert_c", "[CH;$(C(-[#6])(-[#6])-[#6])]", 0.70),
    ("epoxidation", "[CX3]=[CX3]", 0.68),
    ("carbonyl_alpha", "[CH2,CH3][CX3]=O", 0.65),
    ("omega_oxidation", "[CH3][CH2][CH2]", 0.50),
    ("aromatic_ch", "[cH]", 0.20),
]

COMPILED = [(n, Chem.MolFromSmarts(s), sc) for n, s, sc in REACTIVITY_RULES if Chem.MolFromSmarts(s)]

def physics_predict(smiles):
    """Get ranked atom predictions using physics rules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], []
    
    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    
    scores = np.zeros(n)
    patterns = [""] * n
    
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        anum = atom.GetAtomicNum()
        if anum == 6:
            scores[idx] = 0.20 + 0.08 * atom.GetTotalNumHs()
        elif anum == 7:
            scores[idx] = 0.45
        elif anum == 16:
            scores[idx] = 0.55
    
    for name, pat, sc in COMPILED:
        for match in mol.GetSubstructMatches(pat):
            if sc > scores[match[0]]:
                scores[match[0]] = sc
                patterns[match[0]] = name
    
    heavy_idx = [i for i in range(n) if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    ranking = sorted(heavy_idx, key=lambda i: -scores[i])
    
    return ranking, [scores[i] for i in ranking]

# ============================================================================
# Load test data
# ============================================================================

print("\n[1/2] Loading test data...")

test_path = "/content/enzyme_Software/data/augmented_splits/test.json"

try:
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    test_drugs = test_data.get('drugs', test_data)
    print(f"  Test molecules: {len(test_drugs)}")
except FileNotFoundError:
    print("  Test file not found. Running data preparation...")
    exec(open('/content/enzyme_Software/scripts/cyp3a4_hardcoded_data.py').read())
    exec(open('/content/enzyme_Software/scripts/merge_novel_training.py').read())
    exec(open('/content/enzyme_Software/scripts/train_augmented.py').read())
    
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    test_drugs = test_data.get('drugs', test_data)

# ============================================================================
# Evaluate
# ============================================================================

print("\n[2/2] Evaluating physics scorer...")

correct_top1 = 0
correct_top2 = 0
correct_top3 = 0
correct_top5 = 0
total = 0

results = []

for mol in test_drugs:
    smiles = mol.get('smiles', '')
    true_sites = set(mol.get('site_atoms', mol.get('metabolism_sites', [])))
    name = mol.get('name', 'unknown')
    
    if not smiles or not true_sites:
        continue
    
    ranking, scores = physics_predict(smiles)
    if not ranking:
        continue
    
    total += 1
    
    hit_top1 = ranking[0] in true_sites
    hit_top2 = any(r in true_sites for r in ranking[:2])
    hit_top3 = any(r in true_sites for r in ranking[:3])
    hit_top5 = any(r in true_sites for r in ranking[:5])
    
    if hit_top1:
        correct_top1 += 1
    if hit_top2:
        correct_top2 += 1
    if hit_top3:
        correct_top3 += 1
    if hit_top5:
        correct_top5 += 1
    
    results.append({
        'name': name,
        'smiles': smiles,
        'true_sites': list(true_sites),
        'pred_ranking': ranking[:5],
        'hit_top1': hit_top1,
        'hit_top3': hit_top3,
    })

# ============================================================================
# Results
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"""
Test set: {total} molecules (same as Phase 5)

PHYSICS SCORER PERFORMANCE:
  Top-1: {correct_top1}/{total} = {100*correct_top1/total:.1f}%
  Top-2: {correct_top2}/{total} = {100*correct_top2/total:.1f}%
  Top-3: {correct_top3}/{total} = {100*correct_top3/total:.1f}%
  Top-5: {correct_top5}/{total} = {100*correct_top5/total:.1f}%

COMPARISON:
  Phase 5 ML model: 47.4% Top-1
  Physics scorer:   {100*correct_top1/total:.1f}% Top-1
  
  The Phase 5 model is {47.4 - 100*correct_top1/total:.1f}% better than physics alone.
  
EXPECTED ENSEMBLE (Phase 5 + Physics):
  Theoretical: ~55-60% Top-1
  (Ensemble typically adds 5-15% over best single model)
""")

# Show some examples
print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

# Show failures first
failures = [r for r in results if not r['hit_top1']]
successes = [r for r in results if r['hit_top1']]

print("\nCorrect predictions:")
for r in successes[:5]:
    print(f"  {r['name'][:25]:25s} | True: {r['true_sites']} | Pred: {r['pred_ranking'][:3]}")

print("\nMissed predictions:")
for r in failures[:5]:
    print(f"  {r['name'][:25]:25s} | True: {r['true_sites']} | Pred: {r['pred_ranking'][:3]}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Current State:
  - Phase 5 ML model: 47.4% Top-1 (best achieved)
  - Physics scorer: {100*correct_top1/total:.1f}% Top-1 (baseline)
  - Simple GNN (Phase 8): 25.0% Top-1 (insufficient architecture)

Path to Higher Accuracy:
  1. Use Phase 5 + Physics ensemble → ~55-60% Top-1
  2. Train Phase 5 on augmented data → ~55-65% Top-1 (requires nexus fix)
  3. Get more experimental SoM data → 70%+ possible with 2-3x data

The augmented data (124 novel molecules) is ready for training.
To use it, the nexus module import issue needs to be resolved.
""")

print("\nDone!")
