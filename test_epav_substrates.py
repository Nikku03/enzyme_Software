#!/usr/bin/env python3
"""
EP-AV CPT across substrate classes.

Expected behaviour:
  acid chloride   -> high LG, high charge  -> PASS, score ~0.8+
  thioester       -> high LG              -> PASS, score ~0.7+
  ester           -> moderate LG           -> PASS, score ~0.6+
  amide           -> low LG               -> FAIL, score ~0.4-0.5
  carbonate       -> moderate LG, low charge -> borderline
  urea            -> very low LG           -> FAIL, score < 0.4
"""

import os, sys

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rdkit import Chem
from enzyme_software.modules.sre_atr import detect_groups, GroupRole
from enzyme_software.cpt.geometric_cpts import ElectronicPropertiesAttackValidationCPT

# (smiles, group_type_to_find, expected_pass, description)
CASES = [
    ("CC(=O)Cl",       "acyl_halide",  True,  "acetyl chloride"),
    ("CC(=O)SC",       "thioester",    True,  "methyl thioacetate"),
    ("CC(=O)OCC",      "ester",        True,  "ethyl acetate"),
    ("CC(=O)OC(=O)C",  "anhydride",    True,  "acetic anhydride"),
    ("COC(=O)OC",      "carbonate",    None,  "dimethyl carbonate"),  # borderline
    ("CC(=O)NC",       "amide",        False, "N-methyl acetamide"),
    ("CNC(=O)NC",      "urea",         False, "dimethyl urea"),
]

WIDTH = 72
print("=" * WIDTH)
print("EP-AV CPT: SUBSTRATE SURVEY")
print("=" * WIDTH)

epav = ElectronicPropertiesAttackValidationCPT(debug=True)

results = []
for smiles, group_type, expected, desc in CASES:
    print(f"\n{'─'*WIDTH}")
    print(f"  {desc:24s}  SMILES: {smiles}")
    print(f"  group_type: {group_type}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  ERROR: RDKit could not parse SMILES")
        results.append((desc, None, expected))
        continue

    # Detect groups to find role indices
    det = detect_groups(smiles)
    groups = [g for g in det.groups if g.group_type == group_type]

    if not groups:
        # Fall back: try to find any carbonyl-containing group
        groups = [g for g in det.groups
                  if GroupRole.CARBONYL_C in g.roles and GroupRole.HETERO_ATTACH in g.roles]

    if not groups:
        print(f"  WARNING: no {group_type} group detected, using manual SMARTS")
        # Manual fallback via SMARTS
        patt = Chem.MolFromSmarts("[CX3](=[OX1])[!#1]")
        matches = mol.GetSubstructMatches(patt)
        if matches:
            c, o, x = matches[0]
            role_to_idx = {"carbonyl_c": c, "carbonyl_o": o, "hetero_attach": x}
        else:
            print(f"  ERROR: no carbonyl site found at all")
            results.append((desc, None, expected))
            continue
    else:
        g = groups[0]
        role_to_idx = {
            "carbonyl_c": g.roles[GroupRole.CARBONYL_C].original_index,
            "carbonyl_o": g.roles[GroupRole.CARBONYL_O].original_index,
            "hetero_attach": g.roles[GroupRole.HETERO_ATTACH].original_index,
        }

    res = epav.run(mol, role_to_idx, group_type=group_type)

    tag = "PASS" if res.passed else "FAIL"
    match = ""
    if expected is not None:
        match = " OK" if res.passed == expected else " MISMATCH!"
    print(f"  Result: {tag}  score={res.score:.3f}  dominant={res.dominant_driver}{match}")
    print(f"    charge={res.breakdown['charge']:.3f}  "
          f"lg={res.breakdown['leaving_group']:.3f}  "
          f"comp={res.breakdown['competition']:.3f}")
    if res.warnings:
        print(f"    warnings: {res.warnings}")

    results.append((desc, res, expected))

# Summary table
print(f"\n\n{'=' * WIDTH}")
print("SUMMARY")
print(f"{'=' * WIDTH}")
print(f"{'Substrate':<24s} {'Pass':>5s} {'Score':>7s} {'Chg':>6s} {'LG':>6s} {'Comp':>6s} {'Driver':<18s} {'OK?':>4s}")
print("─" * WIDTH)

for desc, res, expected in results:
    if res is None:
        print(f"{desc:<24s}   ---    ---    ---    ---    ---  parse_error")
        continue
    tag = "PASS" if res.passed else "FAIL"
    match = ""
    if expected is not None:
        match = "OK" if res.passed == expected else "BAD"
    print(
        f"{desc:<24s} {tag:>5s} {res.score:>7.3f} "
        f"{res.breakdown['charge']:>6.3f} {res.breakdown['leaving_group']:>6.3f} "
        f"{res.breakdown['competition']:>6.3f} {res.dominant_driver:<18s} {match:>4s}"
    )

# Quick assertion-style check
print(f"\n{'=' * WIDTH}")
mismatches = [d for d, r, e in results if r is not None and e is not None and r.passed != e]
if mismatches:
    print(f"MISMATCHES: {mismatches}")
else:
    print("All expected pass/fail outcomes matched.")
