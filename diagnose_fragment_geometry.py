#!/usr/bin/env python3
"""Diagnose fragment geometry and steric issues."""

import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rdkit import Chem
from rdkit.Chem import AllChem
from enzyme_software.modules.sre_atr import detect_groups
from enzyme_software.modules.sre_fragment_builder import ChemicallyAwareFragmentBuilder

smiles = "CC(=O)OCC"
mol = Chem.MolFromSmiles(smiles)
group = [g for g in detect_groups(smiles).groups if g.group_type == "ester"][0]
atr = group.atoms[0].atr

print("="*60)
print("FRAGMENT GEOMETRY DIAGNOSTICS")
print("="*60)

frag = ChemicallyAwareFragmentBuilder().build_from_group(atr, mol, group)

print(f"\nOriginal molecule: {smiles}")
print(f"Original atoms: {mol.GetNumAtoms()}")
print(f"Fragment atoms: {frag.frag_mol.GetNumAtoms()}")
print(f"Fragment SMILES: {frag.frag_smiles}")

# Check role mapping
print("\n" + "="*60)
print("ROLE TO INDEX MAPPING")
print("="*60)

role_to_idx = {role.value: frag.parent_uuid_to_frag_idx[atom.atom_id]
               for role, atom in group.roles.items()}

for role, idx in role_to_idx.items():
    atom = frag.fragment_3d.mol_3d.GetAtomWithIdx(idx)
    print(f"{role:20s}: idx={idx:2d}  element={atom.GetSymbol()}")

# Check 3D coordinates
print("\n" + "="*60)
print("FRAGMENT 3D COORDINATES")
print("="*60)

mol3d = frag.fragment_3d.mol_3d
conf = mol3d.GetConformer()

print(f"Conformer ID: {frag.fragment_3d.conformer_id}")
print(f"Has 3D: {frag.fragment_3d.validation.get('has_3d', False)}")

print("\nAll atoms in fragment:")
for i in range(mol3d.GetNumAtoms()):
    atom = mol3d.GetAtomWithIdx(i)
    pos = conf.GetAtomPosition(i)
    symbol = atom.GetSymbol()

    # Check if this is a key reactive atom
    role = "---"
    for r, ridx in role_to_idx.items():
        if ridx == i:
            role = r
            break

    print(f"  {i:2d} {symbol:2s} ({role:20s})  xyz: ({pos.x:7.3f}, {pos.y:7.3f}, {pos.z:7.3f})")

# Check distances between key atoms
print("\n" + "="*60)
print("KEY DISTANCES")
print("="*60)

c_idx = role_to_idx.get("carbonyl_c")
o_idx = role_to_idx.get("carbonyl_o")
x_idx = role_to_idx.get("hetero_attach")

if c_idx is not None and o_idx is not None:
    C = conf.GetAtomPosition(c_idx)
    O = conf.GetAtomPosition(o_idx)

    import math
    dist_CO = math.sqrt((O.x-C.x)**2 + (O.y-C.y)**2 + (O.z-C.z)**2)
    print(f"C=O distance: {dist_CO:.3f} Å (expect ~1.20 Å)")

if c_idx is not None and x_idx is not None:
    C = conf.GetAtomPosition(c_idx)
    X = conf.GetAtomPosition(x_idx)

    dist_CX = math.sqrt((X.x-C.x)**2 + (X.y-C.y)**2 + (X.z-C.z)**2)
    print(f"C-O(ester) distance: {dist_CX:.3f} Å (expect ~1.36 Å)")

# Check which atoms are near the carbonyl carbon
print("\n" + "="*60)
print("ATOMS NEAR CARBONYL CARBON")
print("="*60)

if c_idx is not None:
    C = conf.GetAtomPosition(c_idx)
    print(f"\nCarbonyl carbon at ({C.x:.3f}, {C.y:.3f}, {C.z:.3f})")
    print("\nAtoms within 3.0 Å:")

    for i in range(mol3d.GetNumAtoms()):
        if i == c_idx:
            continue
        atom = mol3d.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)

        dist = math.sqrt((pos.x-C.x)**2 + (pos.y-C.y)**2 + (pos.z-C.z)**2)

        if dist < 3.0:
            symbol = atom.GetSymbol()
            role = "---"
            for r, ridx in role_to_idx.items():
                if ridx == i:
                    role = r
                    break

            print(f"  Atom {i:2d} ({symbol:2s}, {role:15s}): {dist:.3f} Å")

# Check the approach vector
print("\n" + "="*60)
print("BÜRGI-DUNITZ APPROACH VECTOR")
print("="*60)

if c_idx is not None and o_idx is not None:
    C = conf.GetAtomPosition(c_idx)
    O = conf.GetAtomPosition(o_idx)

    # Unit vector C->O
    vCO = [(O.x-C.x), (O.y-C.y), (O.z-C.z)]
    n = math.sqrt(vCO[0]**2 + vCO[1]**2 + vCO[2]**2)
    vCO = [v/n for v in vCO]

    print(f"C->O unit vector: ({vCO[0]:.3f}, {vCO[1]:.3f}, {vCO[2]:.3f})")

    # Approach at 107° means cos(107°) ≈ -0.292
    theta = 107.0
    cos_theta = math.cos(math.radians(theta))

    print(f"\nBD attack angle: {theta}° (cos = {cos_theta:.3f})")
    print(f"Attack direction: opposite to C->O (negative z-component)")

    # Sample point at 2.5 Å along approximate attack vector
    # (This is simplified - real calculation includes in-plane component)
    attack_approx = [-vCO[0]*cos_theta, -vCO[1]*cos_theta, -vCO[2]*cos_theta]
    sample_point = [C.x + attack_approx[0]*2.5,
                    C.y + attack_approx[1]*2.5,
                    C.z + attack_approx[2]*2.5]

    print(f"\nApproximate sample point at 2.5 Å: ({sample_point[0]:.3f}, {sample_point[1]:.3f}, {sample_point[2]:.3f})")

    # Check which atoms are near this sample point
    print("\nAtoms within 2.0 Å of sample point (these cause clashes):")
    for i in range(mol3d.GetNumAtoms()):
        if i in {c_idx, o_idx, x_idx}:
            continue

        atom = mol3d.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)

        dist = math.sqrt((pos.x-sample_point[0])**2 +
                        (pos.y-sample_point[1])**2 +
                        (pos.z-sample_point[2])**2)

        if dist < 2.0:
            symbol = atom.GetSymbol()
            print(f"  Atom {i:2d} ({symbol:2s}): {dist:.3f} Å  <-- CLASH!")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("""
If you see atoms (especially carbons from acyl substituent) near the
sample point, that's why you're getting negative clearances and high
steric energies. This is CORRECT behavior - the fragment in isolation
has inherent steric occlusion.

To fix:
1. Test with a pre-positioned fragment (enzyme-like geometry)
2. Use a simpler substrate (e.g., formyl ester HC(=O)OR)
3. Exclude acyl substituents from steric check
""")
