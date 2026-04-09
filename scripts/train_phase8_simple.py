#!/usr/bin/env python3
"""
Simplified Phase 8 Training on Augmented Data

This is a standalone script that:
1. Loads the augmented CYP3A4 data
2. Uses a simplified GNN model (no nexus dependencies)
3. Trains with sample weights
4. Evaluates on the same test set as Phase 5

Run in Colab:
  exec(open('/content/enzyme_Software/scripts/train_phase8_simple.py').read())
"""

import os
import sys
import json
import time
import numpy as np

# Paths
ROOT = "/content/enzyme_Software"
sys.path.insert(0, f"{ROOT}/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print("=" * 70)
print("PHASE 8: SIMPLIFIED TRAINING ON AUGMENTED DATA")
print("=" * 70)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "train_path": f"{ROOT}/data/augmented_splits/train.json",
    "val_path": f"{ROOT}/data/augmented_splits/val.json",
    "test_path": f"{ROOT}/data/augmented_splits/test.json",
    "weights_path": f"{ROOT}/data/augmented_splits/train_weights.json",
    "output_dir": "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/phase8_simple",
    "lr": 1e-3,
    "epochs": 50,
    "batch_size": 32,
    "patience": 10,
    "hidden_dim": 128,
    "num_layers": 4,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ============================================================================
# Load RDKit and torch_geometric
# ============================================================================

print("\n[1/6] Loading libraries...")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    print("  ✓ RDKit loaded")
except ImportError:
    print("  Installing RDKit...")
    os.system("pip install rdkit -q")
    from rdkit import Chem
    from rdkit.Chem import AllChem

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, global_mean_pool
    print("  ✓ torch_geometric loaded")
except ImportError:
    print("  Installing torch_geometric...")
    os.system("pip install torch_geometric -q")
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, global_mean_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

# ============================================================================
# Load data
# ============================================================================

print("\n[2/6] Loading augmented data...")

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('drugs', data)

train_data = load_json(CONFIG["train_path"])
val_data = load_json(CONFIG["val_path"])
test_data = load_json(CONFIG["test_path"])

with open(CONFIG["weights_path"], 'r') as f:
    train_weights = np.array(json.load(f)['weights'])

print(f"  Train: {len(train_data)} (effective: {train_weights.sum():.0f})")
print(f"  Val: {len(val_data)}")
print(f"  Test: {len(test_data)}")

# ============================================================================
# Featurization
# ============================================================================

print("\n[3/6] Setting up featurization...")

# Atom features
ATOM_FEATURES = {
    'atomic_num': list(range(1, 54)),  # H to I
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'is_aromatic': [False, True],
    'num_hs': [0, 1, 2, 3, 4],
}

def one_hot(value, options):
    vec = [0] * len(options)
    if value in options:
        vec[options.index(value)] = 1
    return vec

def atom_features(atom):
    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
    features.extend(one_hot(atom.GetDegree(), ATOM_FEATURES['degree']))
    features.extend(one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    features.extend(one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
    features.extend(one_hot(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic']))
    features.extend(one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']))
    return features

ATOM_FEATURE_DIM = sum(len(v) for v in ATOM_FEATURES.values())
print(f"  Atom feature dim: {ATOM_FEATURE_DIM}")

def mol_to_graph(smiles):
    """Convert SMILES to PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)
    
    # Edge index
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    
    if len(edge_index) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index, num_nodes=mol.GetNumAtoms())

# ============================================================================
# Dataset
# ============================================================================

class SoMDataset(Dataset):
    def __init__(self, molecules, weights=None):
        self.data = []
        self.weights = weights if weights is not None else np.ones(len(molecules))
        
        for i, mol in enumerate(molecules):
            smiles = mol.get('smiles', '')
            sites = mol.get('site_atoms', mol.get('metabolism_sites', mol.get('predicted_som', [])))
            
            if not smiles or not sites:
                continue
            
            graph = mol_to_graph(smiles)
            if graph is None:
                continue
            
            # Convert sites to tensor
            sites = [s for s in sites if s < graph.num_nodes]
            if not sites:
                continue
            
            # Create target: multi-hot for all sites
            target = torch.zeros(graph.num_nodes)
            for s in sites:
                target[s] = 1.0
            
            graph.y = target
            graph.sites = torch.tensor(sites)
            graph.weight = self.weights[i] if i < len(self.weights) else 1.0
            graph.name = mol.get('name', 'unknown')
            
            self.data.append(graph)
        
        print(f"    Created {len(self.data)}/{len(molecules)} valid graphs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

print("\n[4/6] Creating datasets...")
train_dataset = SoMDataset(train_data, train_weights)
val_dataset = SoMDataset(val_data)
test_dataset = SoMDataset(test_data)

def collate_fn(batch):
    return Batch.from_data_list(batch)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)

# ============================================================================
# Model
# ============================================================================

print("\n[5/6] Creating model...")

class SoMPredictor(nn.Module):
    """Simple GNN for Site of Metabolism prediction."""
    
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Physics-inspired features
        self.physics_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Site prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        
        # Input projection
        x = self.input_proj(x)
        
        # GNN layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = F.silu(x_new)
            x_new = norm(x_new)
            x = x + x_new  # Residual
        
        # Global context
        graph_emb = global_mean_pool(x, batch.batch)
        
        # Expand graph embedding to node level
        graph_emb_expanded = graph_emb[batch.batch]
        
        # Physics branch
        physics_feat = self.physics_mlp(x)
        
        # Combine local and global features
        combined = torch.cat([x + physics_feat, graph_emb_expanded], dim=-1)
        
        # Predict site scores
        scores = self.head(combined).squeeze(-1)
        
        return scores

model = SoMPredictor(ATOM_FEATURE_DIM, CONFIG["hidden_dim"], CONFIG["num_layers"])
model = model.to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {num_params:,}")

# ============================================================================
# Training
# ============================================================================

print("\n[6/6] Training...")

optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

def compute_loss(scores, batch):
    """Weighted BCE loss with margin ranking."""
    # Get targets
    targets = batch.y
    
    # BCE loss
    bce = F.binary_cross_entropy_with_logits(scores, targets, reduction='none')
    
    # Apply sample weights
    # Group by graph and weight each graph's loss
    loss = 0
    ptr = batch.ptr.tolist()
    for i in range(len(ptr) - 1):
        start, end = ptr[i], ptr[i+1]
        graph_loss = bce[start:end].mean()
        weight = batch[i].weight if hasattr(batch[i], 'weight') else 1.0
        loss += weight * graph_loss
    
    return loss / (len(ptr) - 1)

def evaluate(loader, name=""):
    """Evaluate model on data loader."""
    model.eval()
    
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            scores = model(batch)
            
            # Process each graph in batch
            ptr = batch.ptr.tolist()
            for i in range(len(ptr) - 1):
                start, end = ptr[i], ptr[i+1]
                graph_scores = scores[start:end]
                
                # Get true sites
                true_sites = set(batch.sites[start:end].tolist()) if hasattr(batch, 'sites') else set()
                if not true_sites:
                    # Fallback to y > 0
                    true_sites = set((batch.y[start:end] > 0).nonzero().squeeze(-1).tolist())
                
                if not true_sites:
                    continue
                
                # Get predictions
                ranking = torch.argsort(graph_scores, descending=True).tolist()
                
                total += 1
                if ranking[0] in true_sites:
                    correct_top1 += 1
                if any(r in true_sites for r in ranking[:3]):
                    correct_top3 += 1
    
    return {
        'top1': correct_top1 / max(total, 1),
        'top3': correct_top3 / max(total, 1),
        'total': total
    }

# Initial evaluation
print("\n" + "-" * 50)
test_metrics = evaluate(test_loader, "Test")
print(f"Initial Test: Top-1={100*test_metrics['top1']:.1f}%, Top-3={100*test_metrics['top3']:.1f}%")
print("-" * 50)

best_val_top1 = 0
patience_counter = 0
best_epoch = 0

for epoch in range(CONFIG["epochs"]):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        scores = model(batch)
        loss = compute_loss(scores, batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Evaluate
    val_metrics = evaluate(val_loader, "Val")
    test_metrics = evaluate(test_loader, "Test")
    
    scheduler.step(val_metrics['top1'])
    
    # Check for improvement
    if val_metrics['top1'] > best_val_top1:
        best_val_top1 = val_metrics['top1']
        best_epoch = epoch + 1
        patience_counter = 0
        
        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_top1': val_metrics['top1'],
            'test_top1': test_metrics['top1'],
        }, f"{CONFIG['output_dir']}/best_model.pt")
    else:
        patience_counter += 1
    
    print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val Top-1: {100*val_metrics['top1']:.1f}% | Test Top-1: {100*test_metrics['top1']:.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    if patience_counter >= CONFIG["patience"]:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# ============================================================================
# Final evaluation
# ============================================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

# Load best model
checkpoint = torch.load(f"{CONFIG['output_dir']}/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

test_metrics = evaluate(test_loader, "Test")
val_metrics = evaluate(val_loader, "Val")

print(f"""
Best epoch: {best_epoch}
Val Top-1:  {100*val_metrics['top1']:.1f}%
Val Top-3:  {100*val_metrics['top3']:.1f}%
Test Top-1: {100*test_metrics['top1']:.1f}%
Test Top-3: {100*test_metrics['top3']:.1f}%

COMPARISON:
  Phase 5 (before augmentation): 47.4% Top-1
  Phase 8 (after augmentation):  {100*test_metrics['top1']:.1f}% Top-1
  Change: {100*test_metrics['top1'] - 47.4:+.1f}%

Saved best model to: {CONFIG['output_dir']}/best_model.pt
""")

print("Done!")
