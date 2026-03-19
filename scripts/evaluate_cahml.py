from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from enzyme_software.cahml import CAHML, CAHMLConfig, CAHMLDataset, evaluate_cahml_predictions
from enzyme_software.liquid_nn_v2._compat import require_torch, torch


def _load_drugs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _has_site_labels(drug: dict) -> bool:
    return bool(drug.get("som") or drug.get("site_atoms") or drug.get("site_atom_indices") or drug.get("metabolism_sites"))


def _resolve_device(name: str | None):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Evaluate CAHML on the held-out split")
    parser.add_argument("--checkpoint", default="checkpoints/cahml/cahml_latest.pt")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--site-labeled-only", dest="site_labeled_only", action="store_true", default=True)
    parser.add_argument("--no-site-labeled-only", dest="site_labeled_only", action="store_false")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    config = CAHMLConfig(**{key: value for key, value in cfg.items() if key in CAHMLConfig().__dict__})
    model = CAHML(config)
    model.load_state_dict(payload.get("model_state_dict") or payload, strict=False)
    device = _resolve_device(args.device)
    model.to(device)
    model.eval()

    drugs = _load_drugs(Path(args.dataset))
    if args.site_labeled_only:
        drugs = [drug for drug in drugs if _has_site_labels(drug)]
    random.Random(args.seed).shuffle(drugs)
    n_train = int(len(drugs) * args.train_ratio)
    n_val = int(len(drugs) * args.val_ratio)
    test_drugs = drugs[n_train + n_val :]
    dataset = CAHMLDataset(args.predictions, test_drugs)

    metrics = {"site_top1": 0.0, "site_top3": 0.0, "cyp_acc": 0.0, "reaction_acc": 0.0, "count": 0, "site_count": 0, "reaction_count": 0}
    trusted = {}
    for idx in range(len(dataset)):
        row = dataset[idx]
        with torch.no_grad():
            outputs = model(
                row["mol_features_raw"].to(device),
                row["atom_features_raw"].to(device),
                row["smarts_matches"].to(device),
                row["site_scores_raw"].to(device),
                row["cyp_probs_raw"].to(device),
            )
        eval_out = evaluate_cahml_predictions(
            outputs["site_scores"],
            row["site_labels"],
            outputs["cyp_logits"],
            int(row["cyp_label"].item()),
            outputs["reaction_logits"],
            int(row["reaction_label"].item()),
        )
        metrics["cyp_acc"] += eval_out["cyp_acc"]
        if "reaction_acc" in eval_out:
            metrics["reaction_acc"] += eval_out["reaction_acc"]
            metrics["reaction_count"] += 1
        if bool(row["site_supervised"].item()):
            metrics["site_top1"] += eval_out["site_top1"]
            metrics["site_top3"] += eval_out["site_top3"]
            metrics["site_count"] += 1
        metrics["count"] += 1
        trusted_model = outputs["explanation"]["trusted_model"]
        trusted[trusted_model] = trusted.get(trusted_model, 0) + 1

    result = {
        "site_top1": metrics["site_top1"] / max(1, metrics["site_count"]),
        "site_top3": metrics["site_top3"] / max(1, metrics["site_count"]),
        "cyp_acc": metrics["cyp_acc"] / max(1, metrics["count"]),
        "reaction_acc": metrics["reaction_acc"] / max(1, metrics["reaction_count"]),
        "count": metrics["count"],
        "site_count": metrics["site_count"],
        "reaction_count": metrics["reaction_count"],
        "trusted_model_counts": trusted,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
