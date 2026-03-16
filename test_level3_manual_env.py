#!/usr/bin/env python3
"""
Test Level 3 CPTs with a manually constructed enzyme-like environment.

Builds:
  - 2 donor points near carbonyl O  (pseudo oxyanion hole)
  - 1 pos charge near O             (simulating Lys/Arg stabilization)
  - 5 hydrophobe points in corridor  (simulating a pocket)
"""

import os, sys, math

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rdkit import Chem
from enzyme_software.modules.sre_atr import detect_groups
from enzyme_software.modules.sre_fragment_builder import ChemicallyAwareFragmentBuilder
from enzyme_software.cpt.level3_env_cpts import (
    EnvContext,
    EnvPoint,
    OxyanionHoleGeometryCPT,
    SolventExposurePolarityCPT,
    TransitionStateChargeStabilizationCPT,
    Level3Orchestrator,
)

# ── substrate ────────────────────────────────────────────────────────────────
smiles = "CC(=O)OCC"
mol = Chem.MolFromSmiles(smiles)
group = [g for g in detect_groups(smiles).groups if g.group_type == "ester"][0]
atr = group.atoms[0].atr

frag = ChemicallyAwareFragmentBuilder().build_from_group(atr, mol, group)
role_to_idx = {
    role.value: frag.parent_uuid_to_frag_idx[atom.atom_id]
    for role, atom in group.roles.items()
}
mol3d = frag.fragment_3d.mol_3d
conf = mol3d.GetConformer()

# key positions
o_idx = role_to_idx["carbonyl_o"]
c_idx = role_to_idx["carbonyl_c"]
O = conf.GetAtomPosition(o_idx)
C = conf.GetAtomPosition(c_idx)

# local frame
vOC = (C.x - O.x, C.y - O.y, C.z - O.z)
n = math.sqrt(vOC[0]**2 + vOC[1]**2 + vOC[2]**2)
uOC = (vOC[0]/n, vOC[1]/n, vOC[2]/n)
outward = (-uOC[0], -uOC[1], -uOC[2])           # away from carbonyl

ref = (1.0, 0.0, 0.0)
if abs(outward[0]) > 0.9:
    ref = (0.0, 1.0, 0.0)
px = (
    outward[1]*ref[2] - outward[2]*ref[1],
    outward[2]*ref[0] - outward[0]*ref[2],
    outward[0]*ref[1] - outward[1]*ref[0],
)
pn = math.sqrt(px[0]**2 + px[1]**2 + px[2]**2)
perp = (px[0]/pn, px[1]/pn, px[2]/pn)

attack_dir = (-outward[0], -outward[1], -outward[2])   # toward carbonyl

# ── build EnvContext ─────────────────────────────────────────────────────────
print("="*72)
print("BUILDING MANUAL ENZYME ENVIRONMENT")
print("="*72)

# 1. Donors: two NH in oxyanion-hole positions (~2.9 Å from carbonyl O)
dd = 2.9
sp = 1.2
nh1 = (O.x + outward[0]*dd + perp[0]*sp/2,
       O.y + outward[1]*dd + perp[1]*sp/2,
       O.z + outward[2]*dd + perp[2]*sp/2)
nh2 = (O.x + outward[0]*dd - perp[0]*sp/2,
       O.y + outward[1]*dd - perp[1]*sp/2,
       O.z + outward[2]*dd - perp[2]*sp/2)

donors = [
    EnvPoint(pos=nh1, kind="donor", meta={"label": "Ser195_NH"}),
    EnvPoint(pos=nh2, kind="donor", meta={"label": "Gly193_NH"}),
]
print(f"\n1. Donors (oxyanion hole):")
for d in donors:
    print(f"   {d.meta['label']:12s}  ({d.pos[0]:+7.3f}, {d.pos[1]:+7.3f}, {d.pos[2]:+7.3f})")

# 2. Charges: one +1 Lys/Arg ~4.5 Å from carbonyl O along oxyanion direction
lys_pos = (O.x + outward[0]*4.5, O.y + outward[1]*4.5, O.z + outward[2]*4.5)

charges = [
    EnvPoint(pos=lys_pos, kind="charge", q=+1.0, meta={"label": "Lys_NH3+"}),
]
print(f"\n2. Charges (TS stabilization):")
for c in charges:
    print(f"   {c.meta['label']:12s}  q={c.q:+.1f}  ({c.pos[0]:+7.3f}, {c.pos[1]:+7.3f}, {c.pos[2]:+7.3f})")

# 3. Hydrophobes: five points lining the approach corridor
hp_specs = [
    (3.5,  +0.8, "Leu_sideA"),
    (4.0,  -0.9, "Val_sideB"),
    (4.5,  +1.1, "Phe_ring"),
    (3.8,  -1.2, "Ile_side"),
    (5.0,  +0.5, "Ala_methyl"),
]
hydrophobes = []
for dist, offset, label in hp_specs:
    hp = (C.x + attack_dir[0]*dist + perp[0]*offset,
          C.y + attack_dir[1]*dist + perp[1]*offset,
          C.z + attack_dir[2]*dist + perp[2]*offset)
    hydrophobes.append(EnvPoint(pos=hp, kind="hydrophobe", meta={"label": label}))

print(f"\n3. Hydrophobes (corridor pocket):")
for h in hydrophobes:
    print(f"   {h.meta['label']:12s}  ({h.pos[0]:+7.3f}, {h.pos[1]:+7.3f}, {h.pos[2]:+7.3f})")

env = EnvContext(donors=donors, charges=charges, hydrophobes=hydrophobes)
print(f"\nEnvContext: {len(env.donors)} donors, {len(env.charges)} charges, {len(env.hydrophobes)} hydrophobes")

# ── run Level 3 ──────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("RUNNING LEVEL 3 CPTs")
print("="*72)

l2_best = {"attack_dir": attack_dir}

cpts = [
    OxyanionHoleGeometryCPT(),
    TransitionStateChargeStabilizationCPT(),
    SolventExposurePolarityCPT(),
]
weights = {
    "oxyanion_hole_geometry": 0.45,
    "ts_charge_stabilization": 0.30,
    "solvent_exposure_polarity": 0.25,
}

# Run each CPT individually for detail
for cpt in cpts:
    out = cpt.run(mol3d, role_to_idx, l2_best, env)
    status = "PASS" if out["passed"] else "FAIL"
    print(f"\n  {cpt.name}")
    print(f"    {status}  score={out['score']:.3f}  conf={out['confidence']:.2f}")
    print(f"    driver: {out['dominant_driver']}")
    if out.get("breakdown"):
        for k, v in out["breakdown"].items():
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
    if out.get("warnings"):
        print(f"    warnings: {out['warnings']}")

# Run orchestrator
orch = Level3Orchestrator(cpts=cpts, weights=weights)
result = orch.run(mol3d, role_to_idx, l2_best, env)

print("\n" + "="*72)
print("LEVEL 3 AGGREGATE RESULT")
print("="*72)
print(f"  Passed:     {result.passed}")
print(f"  Score:      {result.score:.3f}")
print(f"  Confidence: {result.confidence:.2f}")
print(f"  Dominant:   {result.dominant_driver}")
print(f"  Breakdown:")
for k, v in result.breakdown.items():
    w = weights.get(k, 0.0)
    print(f"    {k:35s}  score={v:.3f}  weight={w:.2f}  contrib={v*w:.3f}")
if result.warnings:
    print(f"  Warnings:")
    for w in result.warnings:
        print(f"    - {w}")