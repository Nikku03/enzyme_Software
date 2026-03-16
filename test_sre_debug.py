#!/usr/bin/env python3
"""Test script for Module -1 SRE with debug output."""

from enzyme_software.modules.module_minus1_sre import run_module_minus1

# Test with aspirin -> salicylic acid
result = run_module_minus1(
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    target_bond="acetyl_ester_c-o",
    requested_output="salicylic acid",
    constraints={"ph_min": 7.0, "temperature_c": 25.0},
)

print("\n" + "="*60)
print("=== FINAL RESULTS ===")
print("="*60)

print(f"\n=== STATUS ===")
print(f"Status: {result['status']}")
print(f"Cache Hit: {result['cache_hit']}")
print(f"Warnings: {result['warnings']}")
print(f"Errors: {result['errors']}")

bond360 = result["bond360_profile"]
print(f"\n=== BOND360 PROFILE ===")
print(f"Bond Type: {bond360.get('bond_type')}")
print(f"Primary Role: {bond360.get('primary_role')}")
print(f"Attack Sites: {bond360.get('attack_sites')}")

print("\n=== CPT SCORES (MM Results) ===")
for key, value in result["cpt_scores"].get("mm", {}).items():
    print(f"{key}: {value}")

print("\n=== DETAILED MM RESULTS ===")
import json
print(json.dumps(result["cpt_scores"]["mm_results"], indent=2))

print("\n=== MECHANISM ELIGIBILITY ===")
for mech, status in result["mechanism_eligibility"].items():
    print(f"{mech}: {status}")

print(f"\nPrimary Constraint: {result['primary_constraint']}")
print(f"Confidence Prior: {result['confidence_prior']}")
