#!/usr/bin/env python3
"""Test CPT with a simple substrate (methyl formate) that should have less steric clash."""

import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rdkit import Chem
from enzyme_software.modules.sre_atr import detect_groups
from enzyme_software.modules.sre_fragment_builder import ChemicallyAwareFragmentBuilder
from enzyme_software.cpt.geometric_cpts import EnvironmentAwareStericsCPT_Level2

# Test multiple substrates with increasing steric bulk
test_cases = [
    ("C(=O)OC", "methyl formate (HC(=O)OCH3) - no acyl substituent"),
    ("CC(=O)OC", "methyl acetate (CH3-CO-OCH3) - small acyl"),
    ("CC(=O)OCC", "ethyl acetate (CH3-CO-OCH2CH3) - original"),
    ("CC(C)(C)C(=O)OC", "methyl pivalate (tBu-CO-OCH3) - very bulky"),
]

print("="*80)
print("TESTING CPT WITH DIFFERENT SUBSTRATES")
print("="*80)

for smiles, description in test_cases:
    print(f"\n{'='*80}")
    print(f"Substrate: {description}")
    print(f"SMILES: {smiles}")
    print(f"{'='*80}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"ERROR: Could not parse SMILES: {smiles}")
        continue

    result = detect_groups(smiles)
    esters = [g for g in result.groups if g.group_type == "ester"]

    if not esters:
        print(f"ERROR: No ester groups detected in {smiles}")
        continue

    group = esters[0]
    atr = group.atoms[0].atr

    try:
        frag = ChemicallyAwareFragmentBuilder().build_from_group(atr, mol, group)
        role_to_idx = {role.value: frag.parent_uuid_to_frag_idx[atom.atom_id]
                       for role, atom in group.roles.items()}

        print(f"\nFragment: {frag.frag_smiles}")
        print(f"Atoms: {frag.frag_mol.GetNumAtoms()} heavy atoms")

        # Test with standard probe
        cpt = EnvironmentAwareStericsCPT_Level2(
            probe_radius_A=1.40,
            debug=False,
            protein_aware=False
        )

        res = cpt.run(mol3d=frag.fragment_3d.mol_3d, role_to_idx=role_to_idx)

        print(f"\nResults:")
        print(f"  Passed: {res.passed}")
        print(f"  Score: {res.score:.3f}")
        print(f"  Min clearance: {res.min_clearance_A:+.2f} Å")
        print(f"  Soft energy: {res.soft_steric_energy:.2f}")
        print(f"  Best face: {res.best_face}")
        print(f"  Best wobble: {res.best_wobble_deg:+.1f}°")

        # Interpret result
        if res.min_clearance_A > 0.5:
            print(f"  ✅ GOOD: Plenty of clearance for nucleophile approach")
        elif res.min_clearance_A > 0.0:
            print(f"  ⚠️  TIGHT: Marginal clearance, would need enzyme assistance")
        elif res.min_clearance_A > -1.0:
            print(f"  ❌ BLOCKED: Moderate steric clash")
        else:
            print(f"  ❌ SEVERELY BLOCKED: Major steric clash")

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print("""
Expected trend:
- Methyl formate (HC(=O)OR): Best clearance (no acyl substituent)
- Methyl acetate (CH3-CO-OR): Moderate clearance (small acyl)
- Ethyl acetate (CH3-CO-OCH2CH3): Poor clearance (acyl + larger alkoxy)
- Methyl pivalate (tBu-CO-OR): Worst clearance (very bulky acyl)

If this trend holds, the CPT is working correctly!
""")
