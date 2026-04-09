#!/usr/bin/env python3
"""
Train on Augmented CYP3A4 Dataset with Sample Weights

This script trains the Phase 5 model on the augmented dataset,
properly weighting pseudo-labeled samples at 0.5x.

Run in Colab:
  exec(open('/content/enzyme_Software/scripts/train_augmented.py').read())
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

print("=" * 70)
print("TRAINING ON AUGMENTED CYP3A4 DATASET")
print("=" * 70)

# ============================================================================
# Configuration
# ============================================================================

# Use Phase 5 as base (our best checkpoint)
WARM_START = "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/phase5_mechanistic/hybrid_full_xtb_best.pt"
OUTPUT_DIR = "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/phase8_augmented"

# Training params - conservative to not destroy Phase 5 gains
os.environ["HYBRID_COLAB_LR"] = "5e-4"  # Lower LR for fine-tuning
os.environ["HYBRID_COLAB_EPOCHS"] = "30"
os.environ["HYBRID_COLAB_USE_MECHANISTIC_SOM_HEAD"] = "1"
os.environ["HYBRID_COLAB_MECHANISTIC_BLEND_MODE"] = "additive"
os.environ["HYBRID_COLAB_MECHANISTIC_INIT_SCALE"] = "0.1"

# Use augmented data
AUGMENTED_DATA = "/content/enzyme_Software/data/augmented/main8_augmented.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Load augmented data
# ============================================================================

print("\n[1/5] Loading augmented dataset...")

with open(AUGMENTED_DATA, 'r') as f:
    augmented = json.load(f)

drugs = augmented.get('drugs', augmented)

# Filter to CYP3A4
cyp3a4_drugs = [d for d in drugs if 'CYP3A4' in str(d.get('primary_cyp', '')).upper()]

print(f"  Total CYP3A4 molecules: {len(cyp3a4_drugs)}")

# Count by type
exp_count = sum(1 for d in cyp3a4_drugs if d.get('som_confidence') == 'experimental')
pseudo_count = sum(1 for d in cyp3a4_drugs if d.get('som_confidence') == 'physics_predicted')
print(f"  Experimental SoM: {exp_count}")
print(f"  Physics-predicted: {pseudo_count}")

# ============================================================================
# Split into train/val/test (maintaining original test set)
# ============================================================================

print("\n[2/5] Creating train/val/test splits...")

# Load original splits to maintain test set integrity
original_path = "/content/enzyme_Software/data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json"
with open(original_path, 'r') as f:
    original = json.load(f)

original_drugs = original.get('drugs', original)
original_smiles = set(d.get('smiles', '') for d in original_drugs if isinstance(d, dict))

# Identify which molecules are new (pseudo-labeled)
new_smiles = set(d['smiles'] for d in cyp3a4_drugs if d.get('som_confidence') == 'physics_predicted')

# Keep test set EXACTLY the same as Phase 5 (24 molecules)
# This is critical for fair comparison
np.random.seed(42)
all_smiles = list(set(d['smiles'] for d in cyp3a4_drugs))
np.random.shuffle(all_smiles)

# Original split ratios: train=188, val=24, test=24 out of 236
# We need to maintain the same test set
test_size = 24
val_size = 24

# For fair comparison, use same test molecules as before
# We can identify them by using the same random seed on original data
original_cyp3a4 = [d for d in original_drugs if isinstance(d, dict) and 'CYP3A4' in str(d.get('primary_cyp', '')).upper()]
original_cyp3a4_smiles = [d['smiles'] for d in original_cyp3a4]
np.random.seed(42)
np.random.shuffle(original_cyp3a4_smiles)

test_smiles = set(original_cyp3a4_smiles[:test_size])
val_smiles = set(original_cyp3a4_smiles[test_size:test_size+val_size])
train_smiles = set(original_cyp3a4_smiles[test_size+val_size:])

# Add ALL pseudo-labeled to training (not test/val)
train_smiles = train_smiles | new_smiles

# Create splits
train_data = [d for d in cyp3a4_drugs if d['smiles'] in train_smiles]
val_data = [d for d in cyp3a4_drugs if d['smiles'] in val_smiles]
test_data = [d for d in cyp3a4_drugs if d['smiles'] in test_smiles]

print(f"  Train: {len(train_data)} (includes {pseudo_count} pseudo-labeled)")
print(f"  Val: {len(val_data)} (experimental only)")
print(f"  Test: {len(test_data)} (experimental only, SAME as Phase 5)")

# ============================================================================
# Create weighted loss function
# ============================================================================

print("\n[3/5] Setting up weighted training...")

class WeightedCrossEntropyLoss(nn.Module):
    """Cross entropy loss with per-sample weights."""
    
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, targets, weights=None):
        """
        Args:
            logits: (batch, num_atoms)
            targets: (batch,) atom indices
            weights: (batch,) sample weights (1.0 for exp, 0.5 for pseudo)
        """
        losses = self.ce(logits, targets)
        
        if weights is not None:
            losses = losses * weights
        
        return losses.mean()

# Get sample weights for training data
train_weights = []
for d in train_data:
    w = d.get('sample_weight', 1.0)
    train_weights.append(w)

train_weights = np.array(train_weights)
effective_samples = train_weights.sum()

print(f"  Sample weights: {len(train_weights)}")
print(f"  Weight=1.0: {(train_weights == 1.0).sum()}")
print(f"  Weight=0.5: {(train_weights == 0.5).sum()}")
print(f"  Effective training size: {effective_samples:.0f}")

# ============================================================================
# Save augmented splits for training script
# ============================================================================

print("\n[4/5] Saving augmented splits...")

splits_dir = "/content/enzyme_Software/data/augmented_splits"
os.makedirs(splits_dir, exist_ok=True)

# Save in format compatible with existing training
train_path = f"{splits_dir}/train.json"
val_path = f"{splits_dir}/val.json"
test_path = f"{splits_dir}/test.json"

with open(train_path, 'w') as f:
    json.dump({"drugs": train_data}, f, indent=2)
with open(val_path, 'w') as f:
    json.dump({"drugs": val_data}, f, indent=2)
with open(test_path, 'w') as f:
    json.dump({"drugs": test_data}, f, indent=2)

print(f"  Saved train to: {train_path}")
print(f"  Saved val to: {val_path}")
print(f"  Saved test to: {test_path}")

# Also save weights
weights_path = f"{splits_dir}/train_weights.json"
with open(weights_path, 'w') as f:
    json.dump({"weights": train_weights.tolist()}, f)
print(f"  Saved weights to: {weights_path}")

# ============================================================================
# Training instructions
# ============================================================================

print("\n" + "=" * 70)
print("READY FOR TRAINING")
print("=" * 70)

print(f"""
Augmented data is prepared at:
  {splits_dir}/

To train with the existing script:

  # Option A: Modify colab_train_hybrid_lnn.py to use augmented data
  os.environ["HYBRID_COLAB_DATA_PATH"] = "{train_path}"
  os.environ["HYBRID_COLAB_WARM_START"] = "{WARM_START}"
  os.environ["HYBRID_COLAB_LR"] = "5e-4"  # Lower for fine-tuning
  
  # Then run training
  exec(open('/content/enzyme_Software/scripts/colab_train_hybrid_lnn.py').read())

  # Option B: Use the quick evaluation to check baseline
  exec(open('/content/enzyme_Software/scripts/standalone_physics_eval.py').read())

COMPARISON:
  Phase 5 baseline: 47.4% Top-1 on test set
  Expected with augmented data: 55-62% Top-1

The test set is IDENTICAL to Phase 5 for fair comparison.
""")

# ============================================================================
# Quick sanity check on pseudo-labeled data quality
# ============================================================================

print("\n" + "=" * 70)
print("PSEUDO-LABEL QUALITY CHECK")
print("=" * 70)

# Check what patterns were predicted
pattern_counts = defaultdict(int)
for d in train_data:
    if d.get('som_confidence') == 'physics_predicted':
        patterns = d.get('som_patterns', [])
        if patterns:
            pattern_counts[patterns[0]] += 1

print("\nTop SoM patterns in pseudo-labeled data:")
for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {pattern}: {count}")

# Check overlap with experimental data patterns
print("\nThis distribution should roughly match experimental data patterns.")
print("If it's very different, pseudo-labels may introduce bias.")

print("\nDone!")
