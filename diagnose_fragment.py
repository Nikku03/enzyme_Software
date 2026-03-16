#!/usr/bin/env python3
"""Diagnostic script to check fragment geometry."""

import sys
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rdkit import Chem
from enzyme_software.modules.sre_atr import detect_groups, AtomicTruthRegistry, GroupRole
from enzyme_software.modules.sre_fragment_builder import ChemicallyAwareFragmentBuilder

smiles = "CC(=O)OCC"
mol = Chem.MolFromSmiles(smiles)
result = detect_groups(smiles)
esters = [g for g in result.groups if g.group_type == "ester"]
group = esters[0]
atr = group.atoms[0].atr

frag = ChemicallyAwareFragmentBuilder().build_from_group(atr, mol, group)

print("\n" + "="*60)
print("FRAGMENT DIAGNOSTICS")
print("="*60)

# Check fragment size
mol3d = frag.fragment_3d.mol_3d
print(f"\nOriginal molecule atoms: {mol.GetNumAtoms()}")
print(f"Fragment atoms: {mol3d.GetNumAtoms()}")
print(f"Fragment bonds: {mol3d.GetNumBonds()}")

# Check if fragment has 3D coords
conf = mol3d.GetConformer()
print(f"\nFragment has conformer: {conf is not None}")

# Check coordinate sanity
if conf:
    positions = []
    for i in range(mol3d.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append((pos.x, pos.y, pos.z))
        print(f"Atom {i} ({mol3d.GetAtomWithIdx(i).GetSymbol()}): ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")

    # Check for collapsed coordinates (atoms too close)
    print("\n" + "="*60)
    print("CHECKING FOR COLLAPSED COORDINATES")
    print("="*60)

    min_dist = float('inf')
    min_pair = None

    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dz = positions[i][2] - positions[j][2]
            dist = (dx*dx + dy*dy + dz*dz)**0.5

            if dist < min_dist:
                min_dist = dist
                min_pair = (i, j)

            # Flag suspiciously close atoms (< 0.5 Å is definitely wrong)
            if dist < 0.5:
                print(f"⚠️  WARNING: Atoms {i} and {j} are only {dist:.3f} Å apart!")

    print(f"\nMinimum inter-atomic distance: {min_dist:.3f} Å (atoms {min_pair})")
    if min_dist < 0.8:
        print("❌ PROBLEM: Coordinates are collapsed! Atoms are too close.")
    elif min_dist > 3.0:
        print("❌ PROBLEM: Fragment might be too large or have disconnected pieces.")
    else:
        print("✅ Inter-atomic distances look reasonable.")

# Check role mapping
print("\n" + "="*60)
print("ROLE TO INDEX MAPPING")
print("="*60)

role_to_idx = {}
for atom in group.atoms:
    role = atom.role
    uuid = atom.atr_atom_id
    frag_idx = frag.uuid_to_frag_idx.get(uuid)
    if frag_idx is not None:
        role_to_idx[role.value] = frag_idx
        print(f"{role.value}: fragment_idx={frag_idx}")
    else:
        print(f"{role.value}: NOT FOUND IN FRAGMENT (uuid={uuid})")

print(f"\nExpected roles: carbonyl_c, carbonyl_o, hetero_attach")
print(f"Found roles: {list(role_to_idx.keys())}")

# Check if required roles exist
required = ["carbonyl_c", "carbonyl_o", "hetero_attach"]
missing = [r for r in required if r not in role_to_idx]
if missing:
    print(f"❌ MISSING ROLES: {missing}")
else:
    print("✅ All required roles present")
