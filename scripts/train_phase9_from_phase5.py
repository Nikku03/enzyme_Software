#!/usr/bin/env python3
"""
Phase 9 Ensemble Wrapper for Phase 5 Checkpoint

This script wraps the existing Phase 5 model (47.4% baseline) with:
1. Advanced Physics Scoring
2. Learnable Ensemble Head
3. NEXUS-Lite Memory and Causal Reasoning

This allows fine-tuning just the ensemble components while keeping
the Phase 5 backbone frozen.

Usage:
    python train_phase9_from_phase5.py \\
        --phase5_checkpoint /path/to/hybrid_full_xtb_best.pt \\
        --data data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json \\
        --output_dir /path/to/output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:
    print("PyTorch required")
    sys.exit(1)

try:
    from rdkit import Chem
except ImportError:
    print("RDKit required")
    sys.exit(1)

# Project imports
from enzyme_software.liquid_nn_v2.model.hybrid_model import HybridLNNModel
from enzyme_software.liquid_nn_v2.model.advanced_physics_ensemble import (
    AdvancedPhysicsScorer,
    LearnableEnsembleHead,
)
from enzyme_software.liquid_nn_v2.model.nexus_lite import (
    AnalogicalMemoryBank,
    CausalReasoningEngine,
    HyperbolicOperations,
)


class Phase5EnsembleWrapper(nn.Module):
    """
    Wraps Phase 5 model with learnable ensemble components.
    
    The Phase 5 backbone is frozen; only ensemble head and NEXUS components train.
    """
    
    def __init__(
        self,
        phase5_model: nn.Module,
        physics_scorer: AdvancedPhysicsScorer,
        hidden_dim: int = 128,
        use_nexus: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        self.phase5_model = phase5_model
        self.physics_scorer = physics_scorer
        self.use_nexus = use_nexus
        self.hidden_dim = hidden_dim
        
        if freeze_backbone:
            for param in phase5_model.parameters():
                param.requires_grad = False
        
        # Ensemble head
        self.ensemble_head = LearnableEnsembleHead(
            ml_dim=1,
            physics_dim=6,
            atom_feature_dim=hidden_dim,
            hidden_dim=64,
        )
        
        # Feature projection (Phase 5 might have different dim)
        self.feature_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # NEXUS components
        if use_nexus:
            self.hyp_proj = nn.Linear(hidden_dim, 64)
            self.memory = AnalogicalMemoryBank(
                key_dim=64,
                value_dim=hidden_dim,
                capacity=4096,
                topk=32,
            )
            self.causal = CausalReasoningEngine(
                atom_dim=hidden_dim,
                hidden_dim=64,
            )
            self.nexus_combine = nn.Linear(hidden_dim + 2, 1)
    
    def _extract_physics_features(
        self,
        smiles_list: List[str],
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """Extract physics features for batch."""
        device = batch_index.device
        num_atoms = batch_index.size(0)
        physics_features = torch.zeros(num_atoms, 6, device=device)
        
        atom_offset = 0
        for mol_idx, smiles in enumerate(smiles_list):
            mol_mask = (batch_index == mol_idx)
            mol_num = mol_mask.sum().item()
            
            try:
                result = self.physics_scorer.score_molecule(smiles)
                n = min(mol_num, len(result["final_scores"]))
                
                physics_features[atom_offset:atom_offset+n, 0] = torch.tensor(
                    result["pattern_scores"][:n], device=device, dtype=torch.float32
                )
                physics_features[atom_offset:atom_offset+n, 1] = torch.tensor(
                    result["bde_scores"][:n], device=device, dtype=torch.float32
                )
                physics_features[atom_offset:atom_offset+n, 2] = torch.tensor(
                    result["electronic_scores"][:n], device=device, dtype=torch.float32
                )
                physics_features[atom_offset:atom_offset+n, 3] = torch.tensor(
                    result["sasa_scores"][:n], device=device, dtype=torch.float32
                )
                physics_features[atom_offset:atom_offset+n, 4] = torch.tensor(
                    (result["pattern_scores"][:n] > 0.5).astype(np.float32), device=device
                )
                physics_features[atom_offset:atom_offset+n, 5] = torch.tensor(
                    result["final_scores"][:n], device=device, dtype=torch.float32
                )
            except Exception:
                pass
            
            atom_offset += mol_num
        
        return physics_features
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        update_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        
        # Run Phase 5 model (frozen)
        with torch.no_grad():
            phase5_out = self.phase5_model(batch)
        
        # Get outputs
        ml_logits = phase5_out.get("site_logits")
        atom_features = phase5_out.get("atom_features")
        
        if ml_logits is None:
            return phase5_out
        
        if atom_features is None:
            atom_features = torch.zeros(ml_logits.size(0), self.hidden_dim, device=ml_logits.device)
        
        # Project features if needed
        if atom_features.size(-1) != self.hidden_dim:
            atom_features = self.feature_proj(atom_features)
        
        # Extract physics features
        smiles_list = batch.get("smiles", [])
        batch_index = batch.get("batch", torch.zeros(ml_logits.size(0), dtype=torch.long, device=ml_logits.device))
        
        if smiles_list:
            physics_features = self._extract_physics_features(smiles_list, batch_index)
        else:
            physics_features = torch.zeros(ml_logits.size(0), 6, device=ml_logits.device)
        
        # Ensemble head
        candidate_mask = batch.get("candidate_mask")
        ensemble_result = self.ensemble_head(
            ml_logits=ml_logits,
            physics_features=physics_features,
            atom_features=atom_features,
            candidate_mask=candidate_mask,
        )
        
        site_logits = ensemble_result["ensemble_logits"]
        
        # NEXUS components
        if self.use_nexus:
            # Hyperbolic projection
            hyp = self.hyp_proj(atom_features)
            hyp = HyperbolicOperations.expmap0(hyp)
            hyp = HyperbolicOperations.project(hyp)
            
            # Memory
            memory_pred = torch.zeros_like(site_logits)
            if self.memory.size() > 0:
                mem_out = self.memory.read(hyp)
                memory_pred = (mem_out["attention"] * mem_out["retrieved_labels"]).sum(dim=-1)
            
            # Causal
            causal_out = self.causal(atom_features)
            causal_logits = causal_out["logits"]
            
            # Combine
            nexus_in = torch.cat([atom_features, memory_pred.unsqueeze(-1), causal_logits.unsqueeze(-1)], dim=-1)
            nexus_logits = self.nexus_combine(nexus_in).squeeze(-1)
            
            # Final combination
            site_logits = (
                0.6 * site_logits +  # ML + Physics ensemble
                0.2 * nexus_logits +  # NEXUS
                0.1 * causal_logits +  # Causal
                0.1 * memory_pred  # Memory
            )
            
            # Update memory
            if update_memory and "labels" in batch:
                labels = batch["labels"]
                pos_mask = labels > 0.5
                if pos_mask.any():
                    self.memory.write(
                        hyp[pos_mask].detach(),
                        atom_features[pos_mask].detach(),
                        labels[pos_mask].detach().float(),
                    )
        
        # Apply mask
        if candidate_mask is not None:
            site_logits = site_logits * candidate_mask + (-100.0) * (1 - candidate_mask)
        
        outputs = dict(phase5_out)
        outputs["site_logits"] = site_logits
        outputs["site_logits_phase5"] = ml_logits
        outputs["ml_weight"] = ensemble_result["ml_weight"]
        outputs["physics_weight"] = ensemble_result["physics_weight"]
        
        return outputs


def load_phase5_model(checkpoint_path: str, device: str = "cuda"):
    """Load Phase 5 model from checkpoint."""
    print(f"Loading Phase 5 checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    config = checkpoint.get("config", checkpoint.get("model_config", {}))
    
    # Handle different config formats
    if hasattr(config, "__dict__"):
        config_dict = config.__dict__
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = {}
    
    # Create config object
    class Config:
        pass
    
    cfg = Config()
    for k, v in config_dict.items():
        setattr(cfg, k, v)
    
    # Set defaults if missing
    if not hasattr(cfg, "hidden_dim"):
        cfg.hidden_dim = 128
    if not hasattr(cfg, "enable_nexus"):
        cfg.enable_nexus = False  # Disable in backbone, we add our own
    
    # Create model
    model = HybridLNNModel(cfg)
    
    # Load weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Handle nested state dicts
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    # Load with strict=False to handle missing/extra keys
    model.load_state_dict(state_dict, strict=False)
    
    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase5_checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cyp", type=str, default="CYP3A4")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Phase 5
    phase5_model, phase5_config = load_phase5_model(args.phase5_checkpoint, device)
    phase5_model.to(device)
    phase5_model.eval()
    
    hidden_dim = getattr(phase5_config, "hidden_dim", 128)
    
    # Create physics scorer
    physics_scorer = AdvancedPhysicsScorer(cyp_isoform=args.cyp)
    
    # Create wrapper
    wrapper = Phase5EnsembleWrapper(
        phase5_model=phase5_model,
        physics_scorer=physics_scorer,
        hidden_dim=hidden_dim,
        use_nexus=True,
        freeze_backbone=True,
    )
    wrapper.to(device)
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapper.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    
    print("\nWrapper created successfully!")
    print(f"Ready to train ensemble on {args.data}")
    print(f"Output will be saved to {args.output_dir}")
    
    # Note: Full training loop would go here
    # This script provides the wrapper architecture


if __name__ == "__main__":
    main()
