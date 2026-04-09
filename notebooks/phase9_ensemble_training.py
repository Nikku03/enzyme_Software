# Phase 9: Advanced Physics-ML Ensemble + NEXUS-Lite Training
# 
# This notebook trains the complete ensemble system targeting 90% Top-1 accuracy
# 
# Components:
# 1. Advanced Physics Scoring with CYP-specific rules
# 2. Learnable Ensemble Head (learns ML vs physics weights)
# 3. NEXUS-Lite: Hyperbolic embeddings + Analogical Memory + Causal Reasoning

#@title 1. Setup Environment
!pip install torch torch-geometric rdkit-pypi -q

#@title 2. Clone Repository
!rm -rf /content/enzyme_Software
!git clone https://github.com/Nikku03/enzyme_Software.git /content/enzyme_Software

#@title 3. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#@title 4. Run Phase 9 Evaluation (Physics Baseline)
%cd /content/enzyme_Software
!python scripts/phase9_ensemble.py \
    --data data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json \
    --cyp CYP3A4

#@title 5. Train Phase 9 Ensemble Model
# This trains the standalone ensemble from scratch
# For best results, integrate with Phase 5 checkpoint

import os
os.makedirs('/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/phase9_ensemble', exist_ok=True)

!python scripts/train_phase9_ensemble.py \
    --data data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json \
    --output_dir /content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/phase9_ensemble \
    --cyp CYP3A4 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001

#@title 6. Evaluate Trained Model
import torch
import json
import sys
sys.path.insert(0, '/content/enzyme_Software/src')

from scripts.train_phase9_ensemble import StandaloneEnsembleModel, SoMDataset, collate_som_batch, evaluate
from torch.utils.data import DataLoader
from enzyme_software.liquid_nn_v2.model.advanced_physics_ensemble import AdvancedPhysicsScorer

# Load best model
checkpoint = torch.load('/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/phase9_ensemble/best_model.pt')
print(f"Best model from epoch {checkpoint['epoch']}")
print(f"Validation metrics: {checkpoint['val_metrics']}")

# Load model
model = StandaloneEnsembleModel(use_nexus=checkpoint['config']['use_nexus'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

if torch.cuda.is_available():
    model.cuda()

# Evaluate on full dataset
physics_scorer = AdvancedPhysicsScorer(cyp_isoform='CYP3A4')
dataset = SoMDataset(
    '/content/enzyme_Software/data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json',
    cyp_filter='CYP3A4',
    physics_scorer=physics_scorer
)
loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_som_batch)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
metrics = evaluate(model, loader, device)

print(f"\nFinal Evaluation on Full CYP3A4 Dataset:")
print(f"  Top-1 Accuracy: {metrics['top1_accuracy']*100:.1f}%")
print(f"  Top-3 Accuracy: {metrics['top3_accuracy']*100:.1f}%")

#@title 7. Training History Visualization
import matplotlib.pyplot as plt

with open('/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/phase9_ensemble/history.json') as f:
    history = json.load(f)

epochs = [h['epoch'] for h in history]
train_top1 = [h['train']['top1_accuracy'] * 100 for h in history]
val_top1 = [h['val']['top1_accuracy'] * 100 for h in history]

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_top1, label='Train Top-1')
plt.plot(epochs, val_top1, label='Val Top-1')
plt.axhline(y=90, color='r', linestyle='--', label='Target (90%)')
plt.axhline(y=47.4, color='g', linestyle='--', label='Phase 5 Baseline (47.4%)')
plt.xlabel('Epoch')
plt.ylabel('Top-1 Accuracy (%)')
plt.title('Phase 9 Ensemble Training Progress')
plt.legend()
plt.grid(True)
plt.show()

#@title 8. Analyze Ensemble Weights
# See how the model balances ML vs Physics for different atom types
import numpy as np

model.eval()
ml_weights = []
physics_weights = []

with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch)
        ml_weights.extend(outputs['ml_weight'].cpu().numpy().tolist())
        physics_weights.extend(outputs['physics_weight'].cpu().numpy().tolist())

ml_weights = np.array(ml_weights)
physics_weights = np.array(physics_weights)

print(f"ML Weight: mean={ml_weights.mean():.3f}, std={ml_weights.std():.3f}")
print(f"Physics Weight: mean={physics_weights.mean():.3f}, std={physics_weights.std():.3f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(ml_weights, bins=50, alpha=0.7)
plt.xlabel('ML Weight')
plt.title('Distribution of Learned ML Weights')

plt.subplot(1, 2, 2)
plt.hist(physics_weights, bins=50, alpha=0.7)
plt.xlabel('Physics Weight')
plt.title('Distribution of Learned Physics Weights')
plt.tight_layout()
plt.show()
