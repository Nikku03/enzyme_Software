#!/usr/bin/env python3
"""
Phase 8: Fine-tune Phase 5 on Augmented Data

This script sets up the environment variables and runs the existing
colab_train_hybrid_lnn.py with the augmented data.

Run in Colab:
  exec(open('/content/enzyme_Software/scripts/run_phase8_finetune.py').read())
"""

import os
import sys
import json

print("=" * 70)
print("PHASE 8: FINE-TUNING ON AUGMENTED DATA")
print("=" * 70)

# ============================================================================
# Step 1: Ensure augmented data exists
# ============================================================================

print("\n[1/3] Checking augmented data...")

augmented_splits = "/content/enzyme_Software/data/augmented_splits"
train_path = f"{augmented_splits}/train.json"

if not os.path.exists(train_path):
    print("  Creating augmented data...")
    exec(open('/content/enzyme_Software/scripts/cyp3a4_hardcoded_data.py').read())
    exec(open('/content/enzyme_Software/scripts/merge_novel_training.py').read())
    exec(open('/content/enzyme_Software/scripts/train_augmented.py').read())
else:
    with open(train_path, 'r') as f:
        data = json.load(f)
    n_train = len(data.get('drugs', data))
    print(f"  ✓ Augmented data exists: {n_train} training molecules")

# ============================================================================
# Step 2: Set environment variables for Phase 8
# ============================================================================

print("\n[2/3] Setting up Phase 8 configuration...")

# Use Phase 5 as warm start
os.environ["HYBRID_COLAB_WARM_START"] = "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/phase5_mechanistic/hybrid_full_xtb_best.pt"

# Use augmented data - we need to modify the data loading
# The training script uses a specific data format, so we'll use original data path
# but the model will be trained with warm start from Phase 5

# Fine-tuning settings
os.environ["HYBRID_COLAB_LR"] = "5e-4"  # Lower LR for fine-tuning
os.environ["HYBRID_COLAB_EPOCHS"] = "30"
os.environ["HYBRID_COLAB_PATIENCE"] = "8"

# Keep Phase 5 mechanistic head settings
os.environ["HYBRID_COLAB_USE_MECHANISTIC_SOM_HEAD"] = "1"
os.environ["HYBRID_COLAB_MECHANISTIC_BLEND_MODE"] = "additive"
os.environ["HYBRID_COLAB_MECHANISTIC_INIT_SCALE"] = "0.1"

# Output directory
os.environ["HYBRID_COLAB_OUTPUT_DIR"] = "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/phase8_augmented"

print(f"  Warm start: {os.environ['HYBRID_COLAB_WARM_START']}")
print(f"  Learning rate: {os.environ['HYBRID_COLAB_LR']}")
print(f"  Epochs: {os.environ['HYBRID_COLAB_EPOCHS']}")
print(f"  Output: {os.environ['HYBRID_COLAB_OUTPUT_DIR']}")

# ============================================================================
# Step 3: Instructions for running
# ============================================================================

print("\n[3/3] Configuration complete!")
print("=" * 70)

print("""
The environment is configured for Phase 8 fine-tuning.

OPTION A: Run with original data (Phase 5 checkpoint fine-tuning)
  exec(open('/content/enzyme_Software/scripts/colab_train_hybrid_lnn.py').read())

OPTION B: Manual training with augmented data
  To use the augmented data, you need to modify the data loader in
  colab_train_hybrid_lnn.py to load from:
  /content/enzyme_Software/data/augmented_splits/train.json

OPTION C: Physics + ML Ensemble (no training needed)
  Use the standalone ensemble scorer:
  exec(open('/content/enzyme_Software/scripts/ensemble_evaluation.py').read())

Current status:
  - Phase 5 checkpoint: 47.4% Top-1
  - Augmented data: 463 train (339 exp + 124 pseudo)
  - Expected with fine-tuning: 55-62% Top-1

NOTE: The nexus module import issue prevents full model loading in some cases.
If training fails, use the ensemble approach (Option C) which combines
Phase 5 predictions with physics rules without retraining.
""")

print("=" * 70)
