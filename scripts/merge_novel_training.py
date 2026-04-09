#!/usr/bin/env python3
"""
Merge Novel CYP3A4 Data into Training Set

This script:
1. Loads existing training data (387 CYP3A4 with experimental SoM)
2. Loads novel extracted data (124 with physics-predicted SoM)
3. Merges them with appropriate weighting
4. Saves augmented training set

Run in Colab:
  exec(open('/content/enzyme_Software/scripts/merge_novel_training.py').read())
"""

import json
import os

print("=" * 70)
print("MERGING NOVEL DATA INTO TRAINING SET")
print("=" * 70)

# ============================================================================
# Load existing data
# ============================================================================

print("\n[1/4] Loading existing training data...")

existing_path = "/content/enzyme_Software/data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json"
with open(existing_path, 'r') as f:
    existing = json.load(f)

existing_drugs = existing.get('drugs', existing) if isinstance(existing, dict) else existing

# Filter to CYP3A4 only
cyp3a4_existing = []
for drug in existing_drugs:
    if not isinstance(drug, dict):
        continue
    cyp = str(drug.get('primary_cyp', '')).upper()
    if 'CYP3A4' in cyp or '3A4' in cyp:
        cyp3a4_existing.append(drug)

print(f"  Existing CYP3A4 molecules: {len(cyp3a4_existing)}")

# ============================================================================
# Load novel data
# ============================================================================

print("\n[2/4] Loading novel extracted data...")

novel_path = "/content/enzyme_Software/data/extracted/cyp3a4_hardcoded_novel.json"
with open(novel_path, 'r') as f:
    novel_data = json.load(f)

print(f"  Novel molecules: {len(novel_data)}")

# ============================================================================
# Convert novel data to training format
# ============================================================================

print("\n[3/4] Converting novel data to training format...")

def convert_to_training_format(mol):
    """Convert extracted molecule to training data format."""
    return {
        "name": mol.get("name", "unknown"),
        "smiles": mol["smiles"],
        "primary_cyp": "CYP3A4",
        "site_atoms": mol.get("predicted_som", [])[:3],  # Top 3 predicted sites
        "metabolism_sites": mol.get("predicted_som", [])[:3],
        "source": mol.get("source", "FDA_extracted"),
        "som_confidence": "physics_predicted",  # Flag as pseudo-labeled
        "sample_weight": 0.5,  # Lower weight for pseudo-labels
    }

novel_converted = [convert_to_training_format(m) for m in novel_data if m.get("predicted_som")]
print(f"  Converted {len(novel_converted)} molecules")

# ============================================================================
# Merge datasets
# ============================================================================

print("\n[4/4] Merging datasets...")

# Add sample_weight to existing data (full weight)
for mol in cyp3a4_existing:
    if "sample_weight" not in mol:
        mol["sample_weight"] = 1.0
    mol["som_confidence"] = "experimental"

# Combine
merged_cyp3a4 = cyp3a4_existing + novel_converted

print(f"  Merged dataset: {len(merged_cyp3a4)} molecules")
print(f"    - Experimental SoM: {len(cyp3a4_existing)} (weight=1.0)")
print(f"    - Physics-predicted SoM: {len(novel_converted)} (weight=0.5)")

# Effective training size
effective_size = len(cyp3a4_existing) + 0.5 * len(novel_converted)
print(f"    - Effective training size: {effective_size:.0f}")

# ============================================================================
# Save merged dataset
# ============================================================================

output_dir = "/content/enzyme_Software/data/augmented"
os.makedirs(output_dir, exist_ok=True)

# Save CYP3A4 only
cyp3a4_path = f"{output_dir}/cyp3a4_augmented.json"
with open(cyp3a4_path, 'w') as f:
    json.dump({"drugs": merged_cyp3a4}, f, indent=2)
print(f"\nSaved CYP3A4 augmented data to: {cyp3a4_path}")

# Also create full dataset with all CYPs + augmented CYP3A4
all_other_cyps = [d for d in existing_drugs if isinstance(d, dict) and 'CYP3A4' not in str(d.get('primary_cyp', '')).upper()]
full_augmented = all_other_cyps + merged_cyp3a4

full_path = f"{output_dir}/main8_augmented.json"
with open(full_path, 'w') as f:
    json.dump({"drugs": full_augmented}, f, indent=2)
print(f"Saved full augmented dataset to: {full_path}")

# ============================================================================
# Summary statistics
# ============================================================================

print("\n" + "=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

# Count by CYP
cyp_counts = {}
for drug in full_augmented:
    cyp = drug.get('primary_cyp', 'unknown')
    cyp_counts[cyp] = cyp_counts.get(cyp, 0) + 1

print("\nMolecules by CYP isoform:")
for cyp, count in sorted(cyp_counts.items()):
    marker = " ← AUGMENTED" if "3A4" in str(cyp).upper() else ""
    print(f"  {cyp}: {count}{marker}")

print(f"\nTotal molecules: {len(full_augmented)}")

# ============================================================================
# Training instructions
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING INSTRUCTIONS")
print("=" * 70)
print(f"""
The augmented dataset is ready at:
  {full_path}

To train with sample weights, modify the training script:

1. Load augmented data:
   with open('{full_path}') as f:
       data = json.load(f)

2. Use sample weights in loss:
   For each molecule, weight the loss by mol['sample_weight']
   - Experimental SoM: weight = 1.0 (full contribution)
   - Physics-predicted: weight = 0.5 (reduced contribution)

3. Or simply use the pseudo-labeled data for:
   - Data augmentation during training
   - Extended validation set
   - Ensemble diversity

EXPECTED RESULTS:
  Before: 188 CYP3A4 training → 47.4% Top-1
  After:  {len(cyp3a4_existing)} exp + {len(novel_converted)} pseudo ({effective_size:.0f} effective)
  Expected: 55-62% Top-1
""")

print("\nDone!")
