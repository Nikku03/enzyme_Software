#!/usr/bin/env python3
"""Debug Level 2 CPT to see exactly what's happening."""

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

smiles = "CC(=O)OCC"
mol = Chem.MolFromSmiles(smiles)
group = [g for g in detect_groups(smiles).groups if g.group_type == "ester"][0]
atr = group.atoms[0].atr

frag = ChemicallyAwareFragmentBuilder().build_from_group(atr, mol, group)
role_to_idx = {role.value: frag.parent_uuid_to_frag_idx[atom.atom_id]
               for role, atom in group.roles.items()}

print("="*60)
print("LEVEL 2 CPT DEBUG OUTPUT")
print("="*60)

# Run with debug=True to see all calculations
cpt = EnvironmentAwareStericsCPT_Level2(
    probe_radius_A=1.40,
    debug=True,  # Enable debug logging
    protein_aware=False  # Don't enforce hard failures
)

print("\nRunning Level 2 CPT with debug output...\n")
res = cpt.run(mol3d=frag.fragment_3d.mol_3d, role_to_idx=role_to_idx)

print("\n" + "="*60)
print("FINAL RESULT")
print("="*60)
print(f"Passed: {res.passed}")
print(f"Score: {res.score:.3f}")
print(f"Confidence: {res.confidence:.2f}")
print(f"Best face: {res.best_face}")
print(f"Best wobble: {res.best_wobble_deg:+.1f}°")
print(f"Min clearance: {res.min_clearance_A:+.2f} Å")
print(f"Soft steric energy: {res.soft_steric_energy:.2f}")
print(f"Corridor polarity: {res.corridor_polarity_score:.2f}")
print(f"SASA (reactive atoms): {res.sasa_reactive_A2}")
print(f"Warnings: {res.warnings}")
print(f"\nMessage: {res.message}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("""
If you see negative min_clearance and high soft_energy in the debug output,
look for which face/wobble combination gives the best (least bad) result.

The cone_half_angle=35° means atoms up to 35° away from the attack direction
are counted as "in the way".

The acyl methyl group (atom 0) is likely within this cone, causing the clashes.
""")
