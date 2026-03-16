import json
from enzyme_software.modules.module_minus1_sre import run_module_minus1

result = run_module_minus1(
    smiles="CC(=O)OC",  # Methyl acetate (simple ester)
    target_bond="ester_c-o",
    requested_output=None,
    constraints={"ph_min": 7.0, "temperature_c": 25.0}
)

# Print key outputs
print("=== STATUS ===")
print(f"Status: {result['status']}")
print(f"Cache Hit: {result.get('cache_hit')}")

print("\n=== BOND360 PROFILE ===")
bond360 = result["bond360_profile"]
print(f"Bond Type: {bond360.get('bond_type')}")
print(f"Primary Role: {bond360.get('primary_role')}")
print(f"Attack Sites: {bond360.get('attack_sites')}")

print("\n=== CPT SCORES (MM Results) ===")
cpt_scores = result["cpt_scores"]
for key, value in cpt_scores.get("mm", {}).items():
    print(f"{key}: {value}")

print("\n=== MECHANISM ELIGIBILITY ===")
for mech, status in result["mechanism_eligibility"].items():
    print(f"{mech}: {status}")

print(f"\nPrimary Constraint: {result['primary_constraint']}")
print(f"Confidence Prior: {result['confidence_prior']}")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
except Exception as exc:
    print(f"\n[DEBUG] Skipping fragment visualization: {exc}")
else:
    frag_smiles = result.get("fragment", {}).get("fragment_smiles")
    if frag_smiles:
        frag_mol = Chem.MolFromSmiles(frag_smiles)
        if frag_mol is not None:
            AllChem.EmbedMolecule(frag_mol)
            print(f"\n[DEBUG] Fragment atoms: {frag_mol.GetNumAtoms()}")
            for i, atom in enumerate(frag_mol.GetAtoms()):
                atom.SetProp("atomLabel", f"{atom.GetSymbol()}{i}")
            img = Draw.MolToImage(frag_mol, size=(800, 800))
            img.save("fragment_debug.png")
            print("[DEBUG] Saved fragment_debug.png")
