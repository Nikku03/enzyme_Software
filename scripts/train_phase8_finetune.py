#!/usr/bin/env python3
"""
Phase 8: Fine-tune Phase 5 on Augmented Data

This script:
1. Loads Phase 5 checkpoint with its config
2. Loads augmented CYP3A4 data (463 train, 24 test)
3. Fine-tunes with sample weights
4. Evaluates on same test set

Run in Colab:
  # First prepare data
  exec(open('/content/enzyme_Software/scripts/cyp3a4_hardcoded_data.py').read())
  exec(open('/content/enzyme_Software/scripts/merge_novel_training.py').read())
  exec(open('/content/enzyme_Software/scripts/train_augmented.py').read())
  
  # Then run training
  exec(open('/content/enzyme_Software/scripts/train_phase8_finetune.py').read())
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

# Setup paths
ROOT = Path("/content/enzyme_Software")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("=" * 70)
print("PHASE 8: FINE-TUNING PHASE 5 ON AUGMENTED DATA")
print("=" * 70)

# ============================================================================
# Configuration
# ============================================================================

PHASE5_CHECKPOINT = "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/phase5_mechanistic/hybrid_full_xtb_best.pt"
OUTPUT_DIR = "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/phase8_finetuned"
AUGMENTED_TRAIN = "/content/enzyme_Software/data/augmented_splits/train.json"
AUGMENTED_VAL = "/content/enzyme_Software/data/augmented_splits/val.json"
AUGMENTED_TEST = "/content/enzyme_Software/data/augmented_splits/test.json"
WEIGHTS_PATH = "/content/enzyme_Software/data/augmented_splits/train_weights.json"

LR = 5e-4  # Lower LR for fine-tuning
EPOCHS = 30
PATIENCE = 10
BATCH_SIZE = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# Check data exists
# ============================================================================

print("\n[1/6] Checking data...")

if not os.path.exists(AUGMENTED_TRAIN):
    print("  Augmented data not found. Creating...")
    exec(open('/content/enzyme_Software/scripts/cyp3a4_hardcoded_data.py').read())
    exec(open('/content/enzyme_Software/scripts/merge_novel_training.py').read()) 
    exec(open('/content/enzyme_Software/scripts/train_augmented.py').read())

with open(AUGMENTED_TRAIN, 'r') as f:
    train_data = json.load(f).get('drugs', [])
with open(AUGMENTED_VAL, 'r') as f:
    val_data = json.load(f).get('drugs', [])
with open(AUGMENTED_TEST, 'r') as f:
    test_data = json.load(f).get('drugs', [])
with open(WEIGHTS_PATH, 'r') as f:
    train_weights = np.array(json.load(f)['weights'])

print(f"  Train: {len(train_data)} molecules (effective: {train_weights.sum():.0f})")
print(f"  Val: {len(val_data)} molecules")
print(f"  Test: {len(test_data)} molecules")

# ============================================================================
# Load Phase 5 checkpoint
# ============================================================================

print("\n[2/6] Loading Phase 5 checkpoint...")

checkpoint = torch.load(PHASE5_CHECKPOINT, map_location='cpu')
print(f"  ✓ Loaded from: {PHASE5_CHECKPOINT}")
print(f"  Phase 5 best val: {checkpoint.get('best_val_top1', 'N/A'):.1%}")
print(f"  Phase 5 best epoch: {checkpoint.get('best_epoch', 'N/A')}")

# Get config from checkpoint
ckpt_config = checkpoint.get('config', {})

# ============================================================================
# Create model with checkpoint config
# ============================================================================

print("\n[3/6] Creating model...")

from enzyme_software.liquid_nn_v2.config import ModelConfig
from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
from enzyme_software.liquid_nn_v2.model.hybrid_model import HybridLNNModel

# Create config from checkpoint
if isinstance(ckpt_config, dict):
    config = ModelConfig(**{k: v for k, v in ckpt_config.items() if hasattr(ModelConfig, k) or k in ModelConfig.__dataclass_fields__})
else:
    config = ckpt_config

# Create model
base_model = LiquidMetabolismNetV2(config)
model = HybridLNNModel(base_model)

# Load state dict
state_dict = checkpoint.get('model_state_dict', checkpoint)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"  ✓ Model created and weights loaded")
print(f"  Missing keys: {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")

model = model.to(device)

# ============================================================================
# Create data loaders using existing infrastructure
# ============================================================================

print("\n[4/6] Creating data loaders...")

try:
    from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb import (
        FullXTBHybridDataset,
        create_full_xtb_dataloaders_from_drugs,
    )
    
    # Create datasets
    train_loader, val_loader, test_loader = create_full_xtb_dataloaders_from_drugs(
        train_drugs=train_data,
        val_drugs=val_data,
        test_drugs=test_data,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    print(f"  ✓ Data loaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    USE_EXISTING_LOADER = True
    
except Exception as e:
    print(f"  Could not create existing loaders: {e}")
    print("  Using simplified evaluation...")
    USE_EXISTING_LOADER = False

# ============================================================================
# Training loop
# ============================================================================

print("\n[5/6] Training...")

if USE_EXISTING_LOADER:
    # Use existing training infrastructure
    from enzyme_software.liquid_nn_v2.training.trainer import Trainer
    
    training_config = checkpoint.get('training_config', {})
    if isinstance(training_config, dict):
        from enzyme_software.liquid_nn_v2.config import TrainingConfig
        training_config = TrainingConfig(**{k: v for k, v in training_config.items() if k in TrainingConfig.__dataclass_fields__})
    
    # Override LR for fine-tuning
    training_config.learning_rate = LR
    training_config.epochs = EPOCHS
    training_config.patience = PATIENCE
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=training_config,
        device=device,
        output_dir=OUTPUT_DIR,
    )
    
    # Train
    trainer.train()
    
    # Final evaluation
    test_metrics = trainer.evaluate(test_loader, "test")
    
else:
    # Simplified evaluation using physics scorer as proxy
    print("  Full training not available. Running physics evaluation instead.")
    
    from rdkit import Chem
    
    RULES = [
        ("[CH3]O[c]", 0.95), ("[CH2;!R][c]", 0.92), ("[CH3][c]", 0.90),
        ("[CH3][NX3]", 0.88), ("[CH2][NX3]", 0.82), ("[CH2][OX2]", 0.80),
        ("[SX2]", 0.78), ("[NX3;H0]", 0.75),
    ]
    COMPILED = [(Chem.MolFromSmarts(s), sc) for s, sc in RULES if Chem.MolFromSmarts(s)]
    
    def physics_rank(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []
        mol = Chem.AddHs(mol)
        scores = np.zeros(mol.GetNumAtoms())
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                scores[atom.GetIdx()] = 0.2 + 0.1 * atom.GetTotalNumHs()
            elif atom.GetAtomicNum() in [7, 16]:
                scores[atom.GetIdx()] = 0.5
        for pat, sc in COMPILED:
            for match in mol.GetSubstructMatches(pat):
                scores[match[0]] = max(scores[match[0]], sc)
        heavy = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
        return sorted(heavy, key=lambda i: -scores[i])
    
    correct = 0
    total = 0
    for mol in test_data:
        smiles = mol.get('smiles', '')
        sites = set(mol.get('site_atoms', mol.get('metabolism_sites', [])))
        if not sites or not smiles:
            continue
        ranking = physics_rank(smiles)
        if ranking and ranking[0] in sites:
            correct += 1
        total += 1
    
    print(f"\n  Physics baseline on test: {correct}/{total} = {100*correct/total:.1f}%")
    print("  (Phase 5 model evaluation requires full data pipeline)")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 8 COMPLETE")
print("=" * 70)

print(f"""
Checkpoint saved to: {OUTPUT_DIR}

COMPARISON:
  Phase 5 (before): 47.4% Top-1
  Phase 8 (after):  Check output above

If training didn't run, use the colab_train_hybrid_lnn.py script:

  import os
  os.environ["HYBRID_COLAB_WARM_START"] = "{PHASE5_CHECKPOINT}"
  os.environ["HYBRID_COLAB_LR"] = "5e-4"
  os.environ["HYBRID_COLAB_EPOCHS"] = "30"
  os.environ["HYBRID_COLAB_USE_MECHANISTIC_SOM_HEAD"] = "1"
  os.environ["HYBRID_COLAB_MECHANISTIC_BLEND_MODE"] = "additive"
  
  exec(open('/content/enzyme_Software/scripts/colab_train_hybrid_lnn.py').read())
""")

print("Done!")
