#!/usr/bin/env python3
"""Corrected CPT test script."""

import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rdkit import Chem
from enzyme_software.modules.sre_atr import detect_groups, AtomicTruthRegistry
from enzyme_software.modules.sre_fragment_builder import ChemicallyAwareFragmentBuilder
from enzyme_software.cpt.engine import GeometricCPTEngine

smiles = "CC(=O)OCC"
mol = Chem.MolFromSmiles(smiles)
result = detect_groups(smiles)
esters = [g for g in result.groups if g.group_type == "ester"]

if not esters:
    print("ERROR: No ester groups detected!")
    sys.exit(1)

group = esters[0]
atr = group.atoms[0].atr

print("Building fragment...")
frag = ChemicallyAwareFragmentBuilder().build_from_group(atr, mol, group)

print(f"Fragment has {frag.frag_mol.GetNumAtoms()} atoms (parent has {mol.GetNumAtoms()})")

# CORRECTED: Use parent_uuid_to_frag_idx, not uuid_to_frag_idx
# CORRECTED: Use atom_ref.atom_id, not a.atr.uuid
role_to_idx = {}
for role, atom_ref in group.roles.items():
    uuid = atom_ref.atom_id
    frag_idx = frag.parent_uuid_to_frag_idx.get(uuid)
    if frag_idx is not None:
        role_to_idx[role.value] = frag_idx
        print(f"  {role.value}: frag_idx={frag_idx}")
    else:
        print(f"  WARNING: {role.value} not found in fragment (uuid={uuid})")

print("\nBuilding 3D structure...")
if frag.fragment_3d is None:
    print("ERROR: No fragment_3d! Need to build it first.")
    sys.exit(1)

frag.fragment_3d.build()

if frag.fragment_3d.mol_3d is None:
    print("ERROR: 3D embedding failed!")
    print(f"Warnings: {frag.fragment_3d.warnings}")
    sys.exit(1)

print(f"3D structure built successfully")
print(f"Validation: {frag.fragment_3d.validation}")
print(f"Metrics: {frag.fragment_3d.metrics}")

print("\nRunning CPTs...")
engine = GeometricCPTEngine()
profiles = engine.evaluate(
    mol3d=frag.fragment_3d.mol_3d,
    role_to_idx=role_to_idx
)

for p in profiles:
    print("\n===", p.mechanism_id, "===")
    print(f"Feasibility: {p.feasibility_score:.2f} | Conf: {p.confidence:.2f} | Consistency: {p.consistency}")
    print(f"Primary constraint: {p.primary_constraint}")
    print(f"Insight: {p.key_insight}")
    for e in p.evidence:
        status = "PASS" if e.passed else "FAIL"
        print(f"- {e.cpt_id} {status} score={e.score:.2f} conf={e.confidence:.2f}")
        print(f"   {e.message}")
        if e.warnings:
            print(f"   Warnings: {e.warnings}")
