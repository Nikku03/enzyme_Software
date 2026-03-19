"""
Merge two checkpoints using task arithmetic / selective weight interpolation.

Strategy:
- CYP layers (cyp_branch, cyp_head): take MORE from NEW (better CYP accuracy)
- Site layers (som_branch, site_head): take MORE from OLD (better site top1/top3)
- Shared encoder + physics/steric: 50/50 average

Usage:
    python scripts/merge_checkpoints.py \\
        --old enzyme_lnn_training/checkpoints/hybrid_full_xtb/hybrid_full_xtb_best.pt \\
        --new enzyme_lnn_training_2158/checkpoints/hybrid_full_xtb/hybrid_full_xtb_best.pt \\
        --out merged_checkpoints/hybrid_full_xtb_merged.pt \\
        --cyp-alpha 0.8 \\
        --site-alpha 0.2 \\
        --backbone-alpha 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


# Parameter group classification by key prefix
# Covers both hybrid_lnn/full_xtb keys AND meta-learner keys
CYP_KEYWORDS = ("cyp_branch", "cyp_head", "cyp_site_conditioner",
                 "cyp_attention", "cyp_refine", "cyp_conf", "cyp_temperature")
SITE_KEYWORDS = ("som_branch", "site_head",
                  "site_attention", "site_refine", "site_score", "site_temperature")
PHYSICS_KEYWORDS = ("atom_physics_residual", "mol_physics_residual", "physics_branch",
                    "bde_prior", "manual_priors", "steric_branch")


def classify_key(key: str) -> str:
    for kw in CYP_KEYWORDS:
        if kw in key:
            return "cyp"
    for kw in SITE_KEYWORDS:
        if kw in key:
            return "site"
    for kw in PHYSICS_KEYWORDS:
        if kw in key:
            return "physics"
    return "backbone"


def merge_state_dicts(
    old_sd: dict,
    new_sd: dict,
    cyp_alpha: float,
    site_alpha: float,
    backbone_alpha: float,
    physics_alpha: float,
) -> tuple[dict, dict]:
    """
    Merge two state dicts.

    alpha = fraction of NEW weights to use.
    alpha=0.0 → fully OLD, alpha=1.0 → fully NEW.

    Returns merged state dict and a summary of what was done per group.
    """
    merged = {}
    summary: dict[str, list] = {"cyp": [], "site": [], "backbone": [], "physics": [], "prior_weight": []}

    alpha_map = {
        "cyp": cyp_alpha,
        "site": site_alpha,
        "backbone": backbone_alpha,
        "physics": physics_alpha,
    }

    for key in old_sd:
        old_t = old_sd[key].float()
        if key not in new_sd:
            merged[key] = old_sd[key]
            continue
        new_t = new_sd[key].float()

        if old_t.shape != new_t.shape:
            print(f"  SHAPE MISMATCH for {key}: old={old_t.shape} new={new_t.shape} → keeping OLD")
            merged[key] = old_sd[key]
            continue

        if not old_t.is_floating_point():
            # Integer buffers (e.g. step counters) — keep old
            merged[key] = old_sd[key]
            continue

        # Special case for prior_weight_logit scalar
        if "prior_weight" in key:
            alpha = 0.5
            group = "prior_weight"
        else:
            group = classify_key(key)
            alpha = alpha_map.get(group, backbone_alpha)

        merged_t = (1.0 - alpha) * old_t + alpha * new_t
        merged[key] = merged_t.to(old_sd[key].dtype)
        summary[group].append(key)

    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two model checkpoints via task arithmetic")
    parser.add_argument("--old", required=True, help="Path to OLD checkpoint (better site accuracy)")
    parser.add_argument("--new", required=True, help="Path to NEW checkpoint (better CYP accuracy)")
    parser.add_argument("--out", required=True, help="Output path for merged checkpoint")
    parser.add_argument(
        "--cyp-alpha", type=float, default=0.8,
        help="Fraction of NEW weights to use for CYP layers (default: 0.8 = 80%% new)"
    )
    parser.add_argument(
        "--site-alpha", type=float, default=0.2,
        help="Fraction of NEW weights to use for site layers (default: 0.2 = 80%% old)"
    )
    parser.add_argument(
        "--backbone-alpha", type=float, default=0.5,
        help="Fraction of NEW weights to use for shared encoder (default: 0.5)"
    )
    parser.add_argument(
        "--physics-alpha", type=float, default=0.5,
        help="Fraction of NEW weights to use for physics/steric layers (default: 0.5)"
    )
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    out_path = Path(args.out)

    if not old_path.exists():
        raise FileNotFoundError(f"OLD checkpoint not found: {old_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"NEW checkpoint not found: {new_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading OLD: {old_path}")
    old_ckpt = torch.load(old_path, map_location="cpu", weights_only=False)
    print(f"Loading NEW: {new_path}")
    new_ckpt = torch.load(new_path, map_location="cpu", weights_only=False)

    old_sd = old_ckpt.get("model_state_dict", old_ckpt)
    new_sd = new_ckpt.get("model_state_dict", new_ckpt)

    print(f"\nMerge alphas (fraction of NEW):")
    print(f"  CYP layers    : {args.cyp_alpha:.2f}  (favor NEW → better CYP accuracy)")
    print(f"  Site layers   : {args.site_alpha:.2f}  (favor OLD → better site top1/top3)")
    print(f"  Backbone      : {args.backbone_alpha:.2f}  (balanced)")
    print(f"  Physics/steric: {args.physics_alpha:.2f}  (balanced)")

    merged_sd, summary = merge_state_dicts(
        old_sd, new_sd,
        cyp_alpha=args.cyp_alpha,
        site_alpha=args.site_alpha,
        backbone_alpha=args.backbone_alpha,
        physics_alpha=args.physics_alpha,
    )

    print("\nParameter groups merged:")
    for group, keys in summary.items():
        if keys:
            print(f"  {group}: {len(keys)} tensors")

    # Build merged checkpoint — carry over metadata from OLD
    merged_ckpt = {k: v for k, v in old_ckpt.items() if k != "model_state_dict"}
    merged_ckpt["model_state_dict"] = merged_sd
    merged_ckpt["merge_info"] = {
        "old_checkpoint": str(old_path),
        "new_checkpoint": str(new_path),
        "cyp_alpha": args.cyp_alpha,
        "site_alpha": args.site_alpha,
        "backbone_alpha": args.backbone_alpha,
        "physics_alpha": args.physics_alpha,
        "old_best_val_top1": old_ckpt.get("best_val_top1"),
        "new_best_val_top1": new_ckpt.get("best_val_top1"),
        "old_test_metrics": old_ckpt.get("test_metrics"),
        "new_test_metrics": new_ckpt.get("test_metrics"),
    }

    torch.save(merged_ckpt, out_path)
    print(f"\nSaved merged checkpoint → {out_path}")

    # Print source metrics for reference
    old_tm = old_ckpt.get("test_metrics", {})
    new_tm = new_ckpt.get("test_metrics", {})
    print("\nSource metrics (for reference):")
    print(f"  {'Metric':<25} {'OLD':>10} {'NEW':>10}")
    print(f"  {'-'*45}")
    for k in ["site_top1_acc", "site_top3_acc", "site_auc", "accuracy"]:
        label = k.replace("site_", "").replace("_acc", "").replace("accuracy", "cyp_acc")
        ov = old_tm.get(k)
        nv = new_tm.get(k)
        print(f"  {label:<25} {ov*100:>9.2f}%  {nv*100:>9.2f}%" if ov and nv else
              f"  {label:<25} {'N/A':>10} {'N/A':>10}")
    print("\nTarget: merged model should sit between OLD (site) and NEW (CYP) metrics.")
    print("Run evaluate_specialist_blend.py to verify.")


if __name__ == "__main__":
    main()
