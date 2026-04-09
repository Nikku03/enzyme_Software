#!/usr/bin/env python3
"""Evaluate physics scorer and ensemble on CYP3A4 test set.

This script:
1. Loads the test set from the dataset
2. Evaluates the pure physics scorer (no ML)
3. Loads ML predictions from Phase 5 checkpoint
4. Evaluates the ensemble
5. Finds optimal ensemble weights

Run in Colab after cloning the repo.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add repo to path
repo_root = Path("/content/enzyme_Software")
sys.path.insert(0, str(repo_root / "src"))

from enzyme_software.liquid_nn_v2.model.physics_scorer import PhysicsSoMScorer
from enzyme_software.liquid_nn_v2.model.ensemble_scorer import EnsembleSoMScorer


def load_dataset(dataset_path: str, split: str = "test") -> list:
    """Load and filter dataset for evaluation."""
    with open(dataset_path, "r") as f:
        all_data = json.load(f)
    
    # Filter for CYP3A4 and site-labeled
    filtered = []
    for item in all_data:
        cyp = item.get("cyp", "").upper()
        if "CYP3A4" not in cyp:
            continue
        
        # Get site labels
        site_labels = item.get("site_labels") or item.get("som_indices") or []
        if not site_labels:
            continue
        
        # Convert to list of integers
        if isinstance(site_labels, (list, tuple)):
            sites = [int(s) for s in site_labels if isinstance(s, (int, float)) and s >= 0]
        else:
            sites = []
        
        if not sites:
            continue
        
        filtered.append({
            "smiles": item.get("smiles", ""),
            "name": item.get("name", ""),
            "site_labels": sites,
            "source": item.get("source", ""),
            "split": item.get("split", ""),
        })
    
    return filtered


def evaluate_physics_scorer(dataset: list, verbose: bool = True) -> dict:
    """Evaluate pure physics scorer."""
    scorer = PhysicsSoMScorer()
    
    top1_correct = 0
    top3_correct = 0
    top6_correct = 0
    total = 0
    errors = []
    
    for item in dataset:
        smiles = item["smiles"]
        true_sites = set(item["site_labels"])
        
        try:
            result = scorer.score_molecule(smiles)
            scores = result["final_scores"]
            is_heavy = result["is_heavy"]
            
            # Get predictions
            heavy_indices = np.where(is_heavy > 0.5)[0]
            if len(heavy_indices) == 0:
                continue
            
            sorted_indices = heavy_indices[np.argsort(-scores[heavy_indices])]
            
            top1 = set([int(sorted_indices[0])]) if len(sorted_indices) >= 1 else set()
            top3 = set(int(idx) for idx in sorted_indices[:3])
            top6 = set(int(idx) for idx in sorted_indices[:6])
            
            if top1 & true_sites:
                top1_correct += 1
            if top3 & true_sites:
                top3_correct += 1
            if top6 & true_sites:
                top6_correct += 1
            
            total += 1
            
            if verbose and not (top1 & true_sites):
                # Log misses for analysis
                top_pred = int(sorted_indices[0])
                top_pattern = result["pattern_matches"][top_pred]
                errors.append({
                    "smiles": smiles[:50],
                    "true": list(true_sites),
                    "pred": top_pred,
                    "pattern": top_pattern,
                })
                
        except Exception as e:
            if verbose:
                print(f"Error: {smiles[:30]}... - {e}")
            continue
    
    return {
        "top1_accuracy": top1_correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "top6_accuracy": top6_correct / max(total, 1),
        "total": total,
        "errors": errors[:10],  # First 10 errors
    }


def evaluate_with_ml_scores(
    dataset: list,
    ml_scores_dict: dict,
    ml_weight: float = 0.6,
    verbose: bool = True,
) -> dict:
    """Evaluate ensemble with ML scores."""
    ensemble = EnsembleSoMScorer(
        ml_weight=ml_weight,
        physics_weight=1.0 - ml_weight,
        use_rank_fusion=True,
        boost_high_confidence_physics=True,
    )
    
    top1_correct = 0
    top3_correct = 0
    total = 0
    agreements = {"full": 0, "partial": 0}
    
    for item in dataset:
        smiles = item["smiles"]
        true_sites = set(item["site_labels"])
        
        # Get ML scores
        ml_scores = ml_scores_dict.get(smiles)
        if ml_scores is None:
            continue
        
        ml_scores = np.array(ml_scores, dtype=np.float32)
        
        try:
            result = ensemble.combine_predictions(smiles, ml_scores)
            scores = result["ensemble_scores"]
            
            sorted_indices = np.argsort(-scores)
            top1 = set([int(sorted_indices[0])]) if len(sorted_indices) >= 1 else set()
            top3 = set(int(idx) for idx in sorted_indices[:3])
            
            if top1 & true_sites:
                top1_correct += 1
            if top3 & true_sites:
                top3_correct += 1
            
            agreements[result.get("agreement", "partial")] += 1
            total += 1
            
        except Exception as e:
            if verbose:
                print(f"Error: {smiles[:30]}... - {e}")
            continue
    
    return {
        "top1_accuracy": top1_correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "total": total,
        "agreements": agreements,
        "ml_weight": ml_weight,
    }


def grid_search_weights(
    dataset: list,
    ml_scores_dict: dict,
    verbose: bool = True,
) -> dict:
    """Find optimal ensemble weights."""
    best_weight = 0.5
    best_top1 = 0.0
    results = []
    
    for ml_weight in np.linspace(0.2, 0.9, 15):
        result = evaluate_with_ml_scores(
            dataset, ml_scores_dict, ml_weight=ml_weight, verbose=False
        )
        top1 = result["top1_accuracy"]
        results.append((ml_weight, top1))
        
        if verbose:
            print(f"ML weight {ml_weight:.2f}: Top-1 = {top1:.1%}")
        
        if top1 > best_top1:
            best_top1 = top1
            best_weight = ml_weight
    
    return {
        "best_ml_weight": best_weight,
        "best_top1_accuracy": best_top1,
        "all_results": results,
    }


if __name__ == "__main__":
    # This would be run in Colab
    print("Physics Scorer Evaluation")
    print("=" * 50)
    
    # Load dataset
    dataset_path = "/content/enzyme_Software/data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run this in Colab after cloning the repo.")
        sys.exit(1)
    
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} CYP3A4 molecules with site labels")
    
    # Evaluate physics scorer
    print("\n" + "=" * 50)
    print("PHYSICS SCORER RESULTS (No ML)")
    print("=" * 50)
    
    physics_results = evaluate_physics_scorer(dataset, verbose=False)
    print(f"Top-1 Accuracy: {physics_results['top1_accuracy']:.1%}")
    print(f"Top-3 Accuracy: {physics_results['top3_accuracy']:.1%}")
    print(f"Top-6 Accuracy: {physics_results['top6_accuracy']:.1%}")
    print(f"Total evaluated: {physics_results['total']}")
