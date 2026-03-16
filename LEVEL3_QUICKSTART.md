# Level 3 CPT Quick Start Guide

## Overview

Level 3 CPTs evaluate how **enzyme environment features** support catalysis. They assume Level 2 has already checked intrinsic geometry/sterics.

## Three Test Scripts

### 1. `test_level3_manual_env.py` - Full Manual Construction
**Use when**: You want complete control over environment

Builds a realistic enzyme-like active site with:
- 2 NH donors (oxyanion hole: Ser195, Gly193)
- 1 positive charge (Lys/Arg for TS stabilization)
- 5 hydrophobic residues (corridor: Leu, Val, Phe, Ile, Ala)

```bash
python test_level3_manual_env.py
```

**Expected output**:
```
LEVEL 3 RESULTS
  Passed: True
  Score: 0.750
  Confidence: 0.72

Breakdown by CPT:
  oxyanion_hole_geometry        : 1.000  (2 donors found)
  ts_charge_stabilization       : 0.825  (good pos field)
  solvent_exposure_polarity     : 0.620  (hydrophobic corridor)
```

---

### 2. `test_level2_vs_level3.py` - Comparison
**Use when**: You want to see enzyme's contribution

Shows side-by-side:
- **Level 2**: Intrinsic substrate geometry (vacuum)
- **Level 3**: With enzyme features added

```bash
python test_level2_vs_level3.py
```

**Expected output**:
```
COMPARISON SUMMARY
Metric                          Level 2      Level 3       Change
Overall Score                     0.100        0.720       +0.620
Passed                            True         True            -

KEY INSIGHT: Enzyme provides +0.62 score improvement!
```

---

### 3. Using `pseudo_oxyanion_hole` Helper (Quick)
**Use when**: You just want to test oxyanion hole

```python
from enzyme_software.cpt.level3_env_cpts import EnvContext, Level3Orchestrator

# Build minimal environment (2 donors only)
env = EnvContext.pseudo_oxyanion_hole(
    fragment_3d=mol3d,
    role_to_idx=role_to_idx,
    donor_distance_A=2.9,
    spread_A=1.2,
)

# Add more features as needed
from enzyme_software.cpt.level3_env_cpts import EnvPoint
env.add(EnvPoint(pos=(x, y, z), kind="pos", label="Lys123"))
env.add(EnvPoint(pos=(x, y, z), kind="hydrophobe", label="Phe45"))

# Run
orch = Level3Orchestrator()
result = orch.run(mol3d, role_to_idx, l2_best={}, env_context=env)
```

---

## Environment Point Types

| `kind` | Meaning | Examples |
|--------|---------|----------|
| `"donor"` | H-bond donor (NH, OH) | Ser, Gly backbone NH, Asn side chain |
| `"acceptor"` | H-bond acceptor (O, N) | Asp/Glu CO, backbone C=O |
| `"pos"` | Positive charge | Lys NH3+, Arg guanidinium |
| `"neg"` | Negative charge | Asp/Glu COO- |
| `"hydrophobe"` | Nonpolar | Leu, Val, Phe, Ala |
| `"polar"` | Polar uncharged | Ser OH, Thr, Cys |
| `"nonpolar"` | Explicitly nonpolar | Same as hydrophobe |

---

## The Three CPTs

### 1. **OxyanionHoleGeometryCPT**
Scores H-bond donors near carbonyl O

**Parameters**:
- `d_min_A` / `d_max_A`: Distance range (default 2.6-3.2 Å)
- `min_angle_deg`: Min angle for O-C...donor (default 120°)
- `require_two_donors_for_pass`: If True, needs ≥2 donors to pass

**Score**:
- 0 donors → 0.0
- 1 donor → 0.65
- 2+ donors → 1.0

---

### 2. **TransitionStateChargeStabilizationCPT**
Scores positive/negative charges near oxyanion

**Parameters**:
- `near_min_A` / `near_max_A`: Range for charges (2.5-6.0 Å)
- `align_bonus_angle_deg`: Bonus if charge aligned with O=C (45°)

**Score**:
- Positive charges → increase score (stabilize TS)
- Negative charges → decrease score (destabilize)
- `score = (net_stabilization + 1) / 2`

---

### 3. **SolventExposurePolarityCPT**
Scores corridor composition around attack trajectory

**Parameters**:
- `cone_half_angle_deg`: Corridor width (default 35°)
- `r_min_A` / `r_max_A`: Corridor depth (2.0-6.0 Å)
- `occ_target`: Ideal # of residues in corridor (6)

**Score**:
- Hydrophobic corridor (polarity < 0) → higher score
- Polar corridor (polarity > 0) → lower score
- Optimal occupancy ~ 6 residues

---

## Orchestrator Weights

Default weights (adjustable):
```python
weights = {
    "oxyanion_hole_geometry": 0.45,      # Most important
    "ts_charge_stabilization": 0.30,     # Important
    "solvent_exposure_polarity": 0.25,   # Context-dependent
}
```

**Overall score**: weighted average of 3 CPT scores
**Pass criteria**:
- Overall score ≥ 0.6 **AND**
- Oxyanion hole ≥ 0.55 **AND**
- TS charge ≥ 0.50

---

## Common Patterns

### Serine Hydrolase (e.g., Chymotrypsin)
```python
# 2-donor oxyanion hole
env.add(EnvPoint(pos=ser195_nh, kind="donor", label="Ser195_NH"))
env.add(EnvPoint(pos=gly193_nh, kind="donor", label="Gly193_NH"))

# His57 (positive at pH 7)
env.add(EnvPoint(pos=his57_pos, kind="pos", label="His57"))

# Hydrophobic S1 pocket
for pos in s1_pocket:
    env.add(EnvPoint(pos=pos, kind="hydrophobe", label="S1_pocket"))
```

**Expected**: Score ~0.80-0.90

---

### Metallo-esterase (e.g., Zn2+)
```python
# Metal coordination (acts as strong Lewis acid)
env.add(EnvPoint(pos=zn_pos, kind="pos", label="Zn2+", weight=3.0))

# Coordinating residues
env.add(EnvPoint(pos=his_pos, kind="donor", label="His_coord"))

# Water/hydroxide in active site
env.add(EnvPoint(pos=water_pos, kind="acceptor", label="Water"))
```

**Expected**: Score ~0.70-0.85

---

### Poor Active Site (Missing Features)
```python
# No donors
# No charges
# Only 1-2 hydrophobes

# Result: Score < 0.5, fails enzyme-specific checks
```

---

## Debugging Tips

### 1. Enable Debug Mode
```python
# For Level 2
l2_cpt = EnvironmentAwareStericsCPT_Level2(debug=True)

# For Level 3 - check orchestrator internals
print(result.data["cpt_results"])
```

### 2. Check Breakdown
```python
for cpt_name, score in result.breakdown.items():
    print(f"{cpt_name}: {score:.3f}")

# Identify which CPT is limiting factor
```

### 3. Inspect Environment Points
```python
print(f"Total points: {len(env.points)}")
for p in env.points:
    print(f"  {p.kind:12s} {p.label:20s} at ({p.pos[0]:.2f}, {p.pos[1]:.2f}, {p.pos[2]:.2f})")
```

### 4. Visualize (Future)
```python
# Export to PyMOL/Mol* format
# Show fragment + environment points as pseudoatoms
```

---

## What's Next?

### Immediate (Works Now)
✅ Manual environment construction
✅ Test different residue patterns
✅ Compare substrates with same environment
✅ Optimize donor/charge positions

### Short-term (Easy to add)
🔄 Extract environment from PDB structures
🔄 Add more point types (metals, waters)
🔄 Conformational sampling of environment
🔄 Visualization with PyMOL

### Long-term (Research)
🔮 AlphaFold2 integration for predictions
🔮 Machine learning environment scoring
🔮 Dynamic environment (MD snapshots)
🔮 Multi-substrate competition

---

## Running All Tests

```bash
# Test 1: Full manual construction
python test_level3_manual_env.py > output_manual.txt

# Test 2: Level 2 vs 3 comparison
python test_level2_vs_level3.py > output_comparison.txt

# Test 3: Different substrates with same environment
# (create your own based on test_level3_manual_env.py)
```

---

## Quick Checklist

Before running Level 3:
- [ ] Fragment has 3D coordinates
- [ ] role_to_idx maps carbonyl_c, carbonyl_o, hetero_attach
- [ ] Environment has at least 2-3 points
- [ ] Point positions are reasonable (2-6 Å from reactive atoms)
- [ ] Point kinds are correct ("donor", "pos", "hydrophobe", etc.)

If Level 3 fails:
- [ ] Check warnings in result.warnings
- [ ] Look at breakdown scores (which CPT failed?)
- [ ] Inspect environment points (are they positioned correctly?)
- [ ] Compare to known enzyme active site

---

## Example Output Interpretation

```
LEVEL 3 RESULTS
  Passed: True
  Score: 0.720
  Confidence: 0.72

Breakdown by CPT:
  oxyanion_hole_geometry        : 0.950  ← Good! 2 donors found
  ts_charge_stabilization       : 0.750  ← Good! Positive charge present
  solvent_exposure_polarity     : 0.450  ← Low! Corridor too polar

Warnings:
  - solvent_exposure_polarity:high_corridor_polarity
```

**Interpretation**:
- Oxyanion hole is excellent (2 well-positioned donors)
- Charge stabilization is good (Lys/Arg nearby)
- **Problem**: Corridor is too polar (needs more hydrophobes)

**Fix**: Add more hydrophobic residues around attack path
