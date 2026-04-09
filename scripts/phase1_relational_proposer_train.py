#!/usr/bin/env python3
"""
Phase 1: Relational Proposer A/B Test

This script runs a head-to-head comparison of:
  A) Baseline: ResidualFusionHead (scalar proposer)
  B) Phase 1: RelationalFusionHead (cross-atom attention + pairwise)

Uses the cleaned dataset from Phase 0.

Expected outcome: +8-12% improvement in recall@K

Usage:
    python scripts/phase1_relational_proposer_train.py --variant baseline
    python scripts/phase1_relational_proposer_train.py --variant relational

Author: Claude (Anthropic) for Naresh Chhillar
Date: 2026-04-09
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from enzyme_software.liquid_nn_v2.config import ModelConfig
from enzyme_software.liquid_nn_v2.model import LiquidMetabolismNetV2


def create_config(variant: str, **overrides) -> ModelConfig:
    """Create model config for the specified variant."""
    
    base_config = {
        # Core architecture
        "model_variant": "baseline",
        "atom_input_dim": 148,
        "hidden_dim": 128,
        "som_branch_dim": 128,
        "cyp_branch_dim": 128,
        "num_liquid_layers": 2,
        "dropout": 0.1,
        
        # Site head
        "som_head_hidden_dim": 96,
        "use_manual_engine_priors": True,
        "manual_prior_fusion_mode": "gated_add",
        
        # Training
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        
        # Disable complex features for clean comparison
        "use_nexus_bridge": False,
        "use_topk_reranker": False,
        "use_3d_branch": False,
        "use_hierarchical_pooling": False,
        "use_bde_prior": False,
        "use_cyp_site_conditioning": False,
        "disable_cyp_task": True,
    }
    
    if variant == "relational":
        base_config.update({
            "use_relational_proposer": True,
            "relational_proposer_num_heads": 4,
            "relational_proposer_num_layers": 2,
            "relational_proposer_use_pairwise": True,
            "relational_proposer_dropout": 0.1,
            "relational_proposer_residual_scale": 0.1,
        })
    else:
        base_config["use_relational_proposer"] = False
    
    base_config.update(overrides)
    return ModelConfig(**base_config)


def compute_metrics(model, dataloader, device, topk_values=[1, 3, 6, 12]):
    """Compute Top-K accuracy and recall@K metrics."""
    model.eval()
    
    results = {f"top{k}": 0 for k in topk_values}
    results.update({f"recall@{k}": 0 for k in topk_values})
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(batch)
            site_logits = outputs.get("site_logits", outputs.get("som_logits"))
            
            if site_logits is None:
                continue
            
            # Get true site indices
            true_sites = batch.get("site_labels", batch.get("som_labels"))
            mol_batch = batch["batch"]
            
            if true_sites is None:
                continue
            
            # Per-molecule evaluation
            num_mols = mol_batch.max().item() + 1
            for mol_idx in range(num_mols):
                mol_mask = (mol_batch == mol_idx)
                mol_logits = site_logits[mol_mask].squeeze(-1)
                mol_labels = true_sites[mol_mask]
                
                if mol_labels.sum() == 0:
                    continue
                
                true_indices = (mol_labels > 0.5).nonzero(as_tuple=True)[0]
                
                # Ranking
                sorted_indices = mol_logits.argsort(descending=True)
                
                for k in topk_values:
                    topk_preds = sorted_indices[:k]
                    
                    # Top-K accuracy: is any true site in top-K?
                    hit = any(t in topk_preds for t in true_indices)
                    results[f"top{k}"] += int(hit)
                    
                    # Recall@K: fraction of true sites in top-K
                    recall = sum(1 for t in true_indices if t in topk_preds) / len(true_indices)
                    results[f"recall@{k}"] += recall
                
                total += 1
    
    # Average
    for key in results:
        results[key] /= max(total, 1)
    
    results["n_molecules"] = total
    return results


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(batch)
        
        site_logits = outputs.get("site_logits", outputs.get("som_logits"))
        true_sites = batch.get("site_labels", batch.get("som_labels"))
        
        if site_logits is None or true_sites is None:
            continue
        
        loss = criterion(site_logits.squeeze(-1), true_sites.float())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Relational Proposer A/B Test")
    parser.add_argument("--variant", choices=["baseline", "relational"], default="relational",
                        help="Model variant to train")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", default="data/cleaned", help="Path to cleaned dataset")
    parser.add_argument("--output_dir", default="checkpoints/phase1", help="Output directory")
    parser.add_argument("--dry_run", action="store_true", help="Just test model construction")
    args = parser.parse_args()
    
    print(f"=" * 60)
    print(f"Phase 1: Relational Proposer A/B Test")
    print(f"Variant: {args.variant}")
    print(f"Device: {args.device}")
    print(f"=" * 60)
    
    # Create config
    config = create_config(args.variant, learning_rate=args.lr)
    
    print(f"\nConfig:")
    print(f"  use_relational_proposer: {config.use_relational_proposer}")
    if config.use_relational_proposer:
        print(f"  relational_proposer_num_heads: {config.relational_proposer_num_heads}")
        print(f"  relational_proposer_num_layers: {config.relational_proposer_num_layers}")
        print(f"  relational_proposer_use_pairwise: {config.relational_proposer_use_pairwise}")
    
    # Create model
    print(f"\nBuilding model...")
    model = LiquidMetabolismNetV2(config)
    model = model.to(args.device)
    
    # Initialize lazy modules with a dummy forward pass
    print("  Initializing lazy modules...")
    dummy_batch = {
        "x": torch.randn(20, config.atom_input_dim, device=args.device),
        "edge_index": torch.randint(0, 20, (2, 50), device=args.device),
        "batch": torch.tensor([0]*10 + [1]*10, device=args.device),
        "physics_features": {"bde_values": torch.randn(20, device=args.device)},
    }
    with torch.no_grad():
        _ = model(dummy_batch)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.numel() > 0)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad and p.numel() > 0)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Check site_head type
    site_head_type = type(model.site_head).__name__
    print(f"  site_head type: {site_head_type}")
    
    if args.dry_run:
        print("\n[DRY RUN] Model construction successful!")
        
        # Test forward pass with dummy data
        print("\nTesting forward pass with dummy data...")
        dummy_batch = {
            "x": torch.randn(20, config.atom_input_dim, device=args.device),
            "edge_index": torch.randint(0, 20, (2, 50), device=args.device),
            "batch": torch.tensor([0]*10 + [1]*10, device=args.device),
            "physics_features": {"bde_values": torch.randn(20, device=args.device)},
        }
        
        with torch.no_grad():
            outputs = model(dummy_batch)
        
        print(f"  Output keys: {list(outputs.keys())}")
        if "site_logits" in outputs:
            print(f"  site_logits shape: {outputs['site_logits'].shape}")
        
        print("\n[DRY RUN] Forward pass successful!")
        return
    
    # Load data (placeholder - you'll need to implement actual data loading)
    print("\nLoading data...")
    data_path = Path(args.data_dir) / "cyp3a4_cleaned_dataset.json"
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("Run Phase 0 cleanup first, or specify correct --data_dir")
        sys.exit(1)
    
    # TODO: Implement proper data loading with your existing data pipeline
    # For now, this is a placeholder
    print(f"  Found: {data_path}")
    print("  NOTE: Full training requires integration with your existing data pipeline")
    print("  This script demonstrates the architecture; use colab_train_hybrid_lnn.py for actual training")
    
    # Save config for reference
    output_dir = Path(args.output_dir) / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)
    print(f"\nConfig saved to: {config_path}")
    
    print("\n" + "=" * 60)
    print("To run full training:")
    print(f"  1. Use your existing training script (colab_train_hybrid_lnn.py)")
    print(f"  2. Add config: use_relational_proposer=True")
    print(f"  3. Compare metrics against baseline")
    print("=" * 60)


if __name__ == "__main__":
    main()
