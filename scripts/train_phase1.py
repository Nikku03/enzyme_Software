from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from enzyme_software.liquid_nn_v2.config import ModelConfig
from enzyme_software.liquid_nn_v2.data.training_drugs import TRAINING_DRUGS, validate_training_drugs
from enzyme_software.liquid_nn_v2.features.graph_builder import smiles_to_graph
from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
from enzyme_software.liquid_nn_v2.training.loss import AdaptiveLossV2
from enzyme_software.liquid_nn_v2.training.metrics import compute_cyp_metrics, compute_site_metrics_v2
from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs, move_to_device


def prepare_phase1_dataset() -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    all_drugs = list(TRAINING_DRUGS)
    random.seed(42)
    random.shuffle(all_drugs)
    n_train = int(len(all_drugs) * 0.8)
    n_val = int(len(all_drugs) * 0.1)
    train_drugs = all_drugs[:n_train]
    val_drugs = all_drugs[n_train : n_train + n_val]
    test_drugs = all_drugs[n_train + n_val :]
    print(f"Total drugs: {len(all_drugs)}")
    print(f"Split: {len(train_drugs)} train / {len(val_drugs)} val / {len(test_drugs)} test")
    return train_drugs, val_drugs, test_drugs


def drugs_to_batch(drugs: List[Dict[str, object]], device: torch.device):
    graphs = []
    for drug in drugs:
        graph = smiles_to_graph(
            str(drug["smiles"]),
            cyp_label=str(drug["primary_cyp"]),
            site_atoms=list(drug.get("site_atom_indices", [])),
        )
        if graph is not None:
            graphs.append(graph)
    if not graphs:
        return None
    return move_to_device(collate_molecule_graphs(graphs), device)


def debug_labels(drugs: List[Dict[str, object]], device: torch.device):
    batch = drugs_to_batch(drugs, device)
    if batch is None:
        print("DEBUG: No valid batch")
        return
    labels = batch["site_labels"].squeeze()
    print("\nDEBUG - Site Labels:")
    print(f"  Total atoms: {labels.shape[0]}")
    print(f"  Positive sites: {labels.sum().item():.0f}")
    print(f"  Negative sites: {(labels == 0).sum().item():.0f}")
    print(f"  Positive ratio: {labels.mean().item():.3f}")


def debug_site_scores(model: LiquidMetabolismNetV2, drugs: List[Dict[str, object]], device: torch.device):
    model.eval()
    batch = drugs_to_batch(drugs, device)
    if batch is None:
        return
    with torch.no_grad():
        outputs = model(batch)
    scores = outputs["site_scores"].squeeze()
    labels = batch["site_labels"].squeeze()
    pos_mask = labels == 1
    neg_mask = labels == 0
    print("\nDEBUG - Site Score Distribution:")
    print(f"  All scores: mean={scores.mean().item():.3f}, min={scores.min().item():.3f}, max={scores.max().item():.3f}")
    if pos_mask.sum() > 0:
        pos_scores = scores[pos_mask]
        print(
            f"  Positive sites: mean={pos_scores.mean().item():.3f}, min={pos_scores.min().item():.3f}, max={pos_scores.max().item():.3f}"
        )
    else:
        print("  Positive sites: NONE FOUND IN LABELS")
    if neg_mask.sum() > 0:
        neg_scores = scores[neg_mask]
        print(f"  Negative sites: mean={neg_scores.mean().item():.3f}")
    if scores.std(unbiased=False).item() < 0.01:
        print(f"  WARNING: Scores collapsed (std={scores.std(unbiased=False).item():.4f})")


def train_epoch(
    model: LiquidMetabolismNetV2,
    drugs: List[Dict[str, object]],
    optimizer: torch.optim.Optimizer,
    loss_fn: AdaptiveLossV2,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    batch = drugs_to_batch(drugs, device)
    if batch is None:
        return {"total_loss": float("inf")}
    optimizer.zero_grad()
    outputs = model(batch)
    loss, stats = loss_fn(
        outputs["site_logits"],
        outputs["cyp_logits"],
        batch["site_labels"],
        batch["cyp_labels"],
        batch["batch"],
        outputs.get("tau_history"),
        batch.get("tau_init"),
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return stats


def evaluate(model: LiquidMetabolismNetV2, drugs: List[Dict[str, object]], device: torch.device) -> Dict[str, float]:
    model.eval()
    batch = drugs_to_batch(drugs, device)
    if batch is None:
        return {}
    with torch.no_grad():
        outputs = model(batch)
    site_metrics = compute_site_metrics_v2(outputs["site_scores"], batch["site_labels"], batch["batch"])
    cyp_metrics = compute_cyp_metrics(outputs["cyp_logits"], batch["cyp_labels"])
    return {
        "site_precision": site_metrics["site_precision"],
        "site_recall": site_metrics["site_recall"],
        "site_f1": site_metrics["site_f1"],
        "site_auc": site_metrics["site_auc"],
        "site_top1_acc": site_metrics["site_top1_acc"],
        "site_top2_acc": site_metrics["site_top2_acc"],
        "site_top3_acc": site_metrics["site_top3_acc"],
        "cyp_accuracy": cyp_metrics["accuracy"],
        "cyp_f1_macro": cyp_metrics["f1_macro"],
    }


def analyze_tau(model: LiquidMetabolismNetV2, drugs: List[Dict[str, object]], device: torch.device) -> Dict[str, float]:
    model.eval()
    batch = drugs_to_batch(drugs, device)
    if batch is None:
        return {}
    with torch.no_grad():
        model(batch)
    tau_history = model.last_tau_history
    if not tau_history:
        return {"tau_bde_correlation": None}
    tau_init = batch["tau_init"].detach().cpu()
    tau_final = tau_history[-1].detach().cpu()
    correlation = torch.corrcoef(torch.stack([tau_init, tau_final]))[0, 1].item() if tau_init.numel() > 1 else None
    return {
        "tau_bde_correlation": correlation,
        "tau_init_mean": tau_init.mean().item(),
        "tau_init_std": tau_init.std().item(),
        "tau_final_mean": tau_final.mean().item(),
        "tau_final_std": tau_final.std().item(),
    }


def analyze_gates(model: LiquidMetabolismNetV2, drugs: List[Dict[str, object]], device: torch.device) -> Dict[str, float]:
    model.eval()
    batch = drugs_to_batch(drugs, device)
    if batch is None:
        return {}
    with torch.no_grad():
        outputs = model(batch)
    gates = outputs.get("gate_values")
    if gates is None:
        return {}
    gates = gates.detach().cpu()
    return {
        "gate_mean": gates.mean().item(),
        "gate_std": gates.std().item(),
        "gate_min": gates.min().item(),
        "gate_max": gates.max().item(),
        "pct_trust_physics": (gates < 0.5).float().mean().item(),
        "pct_trust_liquid": (gates >= 0.5).float().mean().item(),
    }


def main() -> None:
    print("=" * 60)
    print("PHASE 1: Training on 30 Validated Drugs")
    print("=" * 60)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if not validate_training_drugs(verbose=True):
        raise SystemExit("Training-drug validation failed")

    train_drugs, val_drugs, test_drugs = prepare_phase1_dataset()
    debug_labels(train_drugs, device)

    config = ModelConfig(atom_input_dim=140, hidden_dim=128, num_liquid_layers=2, ode_steps=6)
    model = LiquidMetabolismNetV2(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    loss_fn = AdaptiveLossV2().to(device)

    epochs = 100
    history = []
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_drugs, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_drugs, device)
        scheduler.step(train_loss.get("total_loss", float("inf")))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_metrics": val_metrics})
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d} | Loss: {train_loss.get('total_loss', 0):.4f} | "
                f"Site F1: {val_metrics.get('site_f1', 0):.3f} | Top1: {val_metrics.get('site_top1_acc', 0):.3f} | "
                f"CYP Acc: {val_metrics.get('cyp_accuracy', 0):.3f}"
            )

    print("-" * 60)
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    test_metrics = evaluate(model, test_drugs, device)
    print(f"\nTest Set ({len(test_drugs)} drugs):")
    print(f"  Site Precision: {test_metrics.get('site_precision', 0):.3f}")
    print(f"  Site Recall:    {test_metrics.get('site_recall', 0):.3f}")
    print(f"  Site F1:        {test_metrics.get('site_f1', 0):.3f}")
    print(f"  Site AUC:       {test_metrics.get('site_auc', 0):.3f}")
    print(f"  Site Top-1 Acc: {test_metrics.get('site_top1_acc', 0):.3f}")
    print(f"  Site Top-2 Acc: {test_metrics.get('site_top2_acc', 0):.3f}")
    print(f"  Site Top-3 Acc: {test_metrics.get('site_top3_acc', 0):.3f}")
    print(f"  CYP Accuracy:   {test_metrics.get('cyp_accuracy', 0):.3f}")
    print(f"  CYP Macro F1:   {test_metrics.get('cyp_f1_macro', 0):.3f}")

    debug_site_scores(model, train_drugs, device)

    print("\n" + "=" * 60)
    print("TAU ANALYSIS")
    print("=" * 60)
    tau_summary = analyze_tau(model, train_drugs, device)
    print(f"\nτ-BDE Correlation: {tau_summary.get('tau_bde_correlation', 'N/A')}")
    print(f"τ_init mean/std:   {tau_summary.get('tau_init_mean', 0):.3f} / {tau_summary.get('tau_init_std', 0):.3f}")
    print(f"τ_final mean/std:  {tau_summary.get('tau_final_mean', 0):.3f} / {tau_summary.get('tau_final_std', 0):.3f}")

    print("\n" + "=" * 60)
    print("GATE ANALYSIS (Physics vs Liquid Branch)")
    print("=" * 60)
    gate_summary = analyze_gates(model, train_drugs, device)
    print(f"\nGate mean:         {gate_summary.get('gate_mean', 'N/A')}")
    print(f"Gate std:          {gate_summary.get('gate_std', 'N/A')}")
    print(f"% Trust Physics:   {gate_summary.get('pct_trust_physics', 0) * 100:.1f}%")
    print(f"% Trust Liquid:    {gate_summary.get('pct_trust_liquid', 0) * 100:.1f}%")

    print("\n" + "=" * 60)
    print("SAVING")
    print("=" * 60)
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint = {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "test_metrics": test_metrics,
        "tau_analysis": tau_summary,
        "gate_analysis": gate_summary,
        "history": history,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"phase1_30drugs_{timestamp}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    report = {
        "phase": 1,
        "num_drugs": len(train_drugs) + len(val_drugs) + len(test_drugs),
        "epochs": epochs,
        "final_metrics": test_metrics,
        "tau_analysis": tau_summary,
        "gate_analysis": gate_summary,
    }
    report_path = checkpoint_dir / f"phase1_report_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved report: {report_path}")

    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    success = True
    initial_loss = history[0]["train_loss"].get("total_loss", float("inf"))
    final_loss = history[-1]["train_loss"].get("total_loss", float("inf"))
    if final_loss < initial_loss:
        print("✓ Loss decreased over training")
    else:
        print("✗ Loss did NOT decrease - check architecture")
        success = False
    tau_corr = tau_summary.get("tau_bde_correlation")
    if tau_corr is not None and tau_corr > 0.3:
        print(f"✓ τ-BDE correlation: {tau_corr:.3f} (> 0.3)")
    elif tau_corr is not None:
        print(f"⚠ τ-BDE correlation: {tau_corr:.3f} (< 0.3, may need tuning)")
    else:
        print("⚠ τ correlation not available")
    train_metrics = evaluate(model, train_drugs, device)
    if train_metrics.get("site_f1", 0) > 0.5 or train_metrics.get("site_top1_acc", 0) > 0.6:
        print(
            f"✓ Train Site F1: {train_metrics.get('site_f1', 0):.3f}, Top1: {train_metrics.get('site_top1_acc', 0):.3f}"
        )
    else:
        print(
            f"✗ Train Site F1: {train_metrics.get('site_f1', 0):.3f}, Top1: {train_metrics.get('site_top1_acc', 0):.3f}"
        )
        success = False
    gate_mean = gate_summary.get("gate_mean", 0.5)
    if 0.2 < gate_mean < 0.8:
        print(f"✓ Gate values balanced (mean: {gate_mean:.3f})")
    else:
        print("⚠ Gate values may be collapsed to one branch")
    print("\n" + "=" * 60)
    if success:
        print("✓ PHASE 1 PASSED - Architecture validated")
        print("  Proceed to Phase 2 (500 drugs)")
    else:
        print("✗ PHASE 1 FAILED - Debug issues before scaling")
    print("=" * 60)


if __name__ == "__main__":
    main()
