#!/usr/bin/env python3
"""Compare Level 2 (vacuum) vs Level 3 (enzyme environment) CPTs."""

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
from enzyme_software.cpt.level3_env_cpts import (
    EnvContext,
    EnvPoint,
    OxyanionHoleGeometryCPT,
    TransitionStateChargeStabilizationCPT,
    SolventExposurePolarityCPT,
    Level3Orchestrator,
)

print("="*80)
print("LEVEL 2 (VACUUM) VS LEVEL 3 (ENZYME ENVIRONMENT) COMPARISON")
print("="*80)

# Test substrate
smiles = "CC(=O)OCC"
mol = Chem.MolFromSmiles(smiles)
group = [g for g in detect_groups(smiles).groups if g.group_type == "ester"][0]
atr = group.atoms[0].atr

frag = ChemicallyAwareFragmentBuilder().build_from_group(atr, mol, group)
role_to_idx = {role.value: frag.parent_uuid_to_frag_idx[atom.atom_id]
               for role, atom in group.roles.items()}

mol3d = frag.fragment_3d.mol_3d

print(f"\nSubstrate: {smiles}")
print(f"Fragment: {frag.frag_smiles}")

# ============================================================================
# LEVEL 2: VACUUM FRAGMENT ONLY
# ============================================================================

print("\n" + "="*80)
print("LEVEL 2: VACUUM (NO ENZYME ENVIRONMENT)")
print("="*80)

l2_cpt = EnvironmentAwareStericsCPT_Level2(
    probe_radius_A=1.40,
    debug=False,
    protein_aware=False,
)

l2_result = l2_cpt.run(mol3d=mol3d, role_to_idx=role_to_idx)

print(f"\nLevel 2 Results (intrinsic geometry only):")
print(f"  Passed: {l2_result.passed}")
print(f"  Score: {l2_result.score:.3f}")
print(f"  Min clearance: {l2_result.min_clearance_A:+.2f} Å")
print(f"  Soft energy: {l2_result.soft_steric_energy:.2f} kcal/mol")
print(f"  Best face: {l2_result.best_face}")
print(f"  Best wobble: {l2_result.best_wobble_deg:+.1f}°")

if l2_result.worst_blocker_atom_idx is not None:
    print(f"\n  Worst blocker: {l2_result.worst_blocker_element}{l2_result.worst_blocker_atom_idx}")
    print(f"    Side: {l2_result.worst_blocker_side}")
    print(f"    Clearance: {l2_result.worst_blocker_clearance_A:+.2f} Å")

if l2_result.warnings:
    print(f"\n  Warnings: {', '.join(l2_result.warnings)}")

print(f"\n  Interpretation:")
if l2_result.min_clearance_A < -1.0:
    print(f"    ❌ Severe steric clash - fragment geometry blocks approach")
    print(f"    → Needs enzyme to position/strain substrate")
elif l2_result.min_clearance_A < 0.0:
    print(f"    ⚠️  Moderate steric clash - tight fit")
    print(f"    → Enzyme must provide conformational flexibility")
else:
    print(f"    ✅ Sterically accessible in vacuum")
    print(f"    → Good intrinsic reactivity")

# ============================================================================
# LEVEL 3: WITH ENZYME-LIKE ENVIRONMENT
# ============================================================================

print("\n" + "="*80)
print("LEVEL 3: WITH ENZYME ENVIRONMENT")
print("="*80)

# Build simple enzyme-like environment using helper
env = EnvContext.pseudo_oxyanion_hole(
    fragment_3d=mol3d,
    role_to_idx=role_to_idx,
    donor_distance_A=2.9,
    spread_A=1.2,
)

# Add positive charge for TS stabilization
conf = mol3d.GetConformer()
o_idx = role_to_idx["carbonyl_o"]
c_idx = role_to_idx["carbonyl_c"]
O = conf.GetAtomPosition(o_idx)
C = conf.GetAtomPosition(c_idx)

import math
vOC = (C.x - O.x, C.y - O.y, C.z - O.z)
n = math.sqrt(vOC[0]**2 + vOC[1]**2 + vOC[2]**2)
outward = (-vOC[0]/n, -vOC[1]/n, -vOC[2]/n)

pos_pos = (O.x + outward[0]*4.5, O.y + outward[1]*4.5, O.z + outward[2]*4.5)
env.add(EnvPoint(pos=pos_pos, kind="pos", label="Lys", q=+1.0))

# Add hydrophobic corridor
attack_dir = (-outward[0], -outward[1], -outward[2])
for i, (d, label) in enumerate([(3.5, "Leu"), (4.2, "Val"), (4.8, "Phe")]):
    hp_pos = (C.x + attack_dir[0]*d, C.y + attack_dir[1]*d, C.z + attack_dir[2]*d)
    env.add(EnvPoint(pos=hp_pos, kind="hydrophobe", label=label, weight=1.0))

print(f"\nEnvironment constructed:")
print(f"  {len(env.donors)} donor points (oxyanion hole)")
print(f"  {len(env.charges)} charges")
print(f"  {len(env.hydrophobes)} hydrophobic residues")

# Run Level 3 CPTs
l2_best = {
    "best_face": l2_result.best_face,
    "best_wobble_deg": l2_result.best_wobble_deg,
    "min_clearance_A": l2_result.min_clearance_A,
    "attack_dir": attack_dir,
}

orchestrator = Level3Orchestrator(
    cpts=[
        OxyanionHoleGeometryCPT(),
        TransitionStateChargeStabilizationCPT(),
        SolventExposurePolarityCPT(),
    ],
    weights={
        "oxyanion_hole_geometry": 0.45,
        "ts_charge_stabilization": 0.30,
        "solvent_exposure_polarity": 0.25,
    },
)
l3_result = orchestrator.run(mol3d, role_to_idx, l2_best, env)

print(f"\nLevel 3 Results (with enzyme features):")
print(f"  Passed: {l3_result.passed}")
print(f"  Score: {l3_result.score:.3f}")
print(f"  Confidence: {l3_result.confidence:.2f}")

print(f"\n  Breakdown:")
for cpt_name, score in l3_result.breakdown.items():
    status = "✓" if score >= 0.55 else "✗"
    print(f"    {status} {cpt_name:30s}: {score:.3f}")

if l3_result.warnings:
    print(f"\n  Warnings:")
    for w in l3_result.warnings[:5]:
        print(f"    - {w}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

score_change = l3_result.score - l2_result.score

print(f"\n{'Metric':<30s} {'Level 2':>12s} {'Level 3':>12s} {'Change':>12s}")
print("-"*80)
print(f"{'Overall Score':<30s} {l2_result.score:>12.3f} {l3_result.score:>12.3f} {score_change:>+12.3f}")
print(f"{'Passed':<30s} {str(l2_result.passed):>12s} {str(l3_result.passed):>12s} {'-':>12s}")
print(f"{'Confidence':<30s} {0.75:>12.2f} {l3_result.confidence:>12.2f} {l3_result.confidence-0.75:>+12.2f}")

print("\n" + "-"*80)
print("KEY INSIGHTS")
print("-"*80)

print("\n1. LEVEL 2 (Vacuum Fragment):")
print("   - Tests INTRINSIC geometry & sterics")
print("   - Shows substrate needs enzyme assistance")
print(f"   - Score: {l2_result.score:.3f} (intrinsic favorability)")

print("\n2. LEVEL 3 (Enzyme Context):")
print("   - Tests how ENZYME FEATURES support reaction")
print("   - Oxyanion hole stabilizes TS")
print("   - Charged residues provide electrostatic assist")
print("   - Hydrophobic pocket positions substrate")
print(f"   - Score: {l3_result.score:.3f} (enzyme-assisted favorability)")

print("\n3. THE ENZYME'S JOB:")
if score_change > 0.3:
    print("   ✅ Enzyme provides MAJOR catalytic enhancement!")
    print(f"      Score improved by {score_change:.3f} with enzyme features")
    print("      → Enzyme overcomes intrinsic barriers")
elif score_change > 0.0:
    print("   ✓ Enzyme provides modest enhancement")
    print(f"      Score improved by {score_change:.3f}")
    print("      → Substrate already somewhat favorable")
else:
    print("   ⚠️  Enzyme features not sufficient")
    print(f"      Score changed by {score_change:.3f}")
    print("      → May need better substrate/enzyme match")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Use real protein structures (PDB) to extract environment points
2. Add more environment point types:
   - Backbone NH/CO for H-bonds
   - Water molecules in active site
   - Metal ions (Zn2+, Mg2+)
   - Catalytic residues (His, Asp, etc.)

3. Implement dynamic environment:
   - Sample multiple conformations
   - Test different protonation states
   - Evaluate substrate binding modes

4. Integrate with AlphaFold2:
   - Predict enzyme structure
   - Extract environment from prediction
   - Score predicted complexes
""")
