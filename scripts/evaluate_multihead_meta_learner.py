from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.meta_learner import MetaLearnerDataset
from enzyme_software.meta_learner.meta_evaluator import evaluate_meta_predictions
from enzyme_software.meta_learner.multi_head_meta_model import MultiHeadMetaLearner


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


def _load_prediction_metadata(path: Path) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    predictions = payload.get("predictions") or payload
    if not predictions:
        raise ValueError(f"No predictions found in {path}")
    first = next(iter(predictions.values()))
    return int(first["site_scores_raw"].shape[1]), int(first["atom_features"].shape[1]), int(first["global_features"].shape[0])


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Evaluate the multi-head meta learner on the held-out split")
    parser.add_argument("--checkpoint", default="checkpoints/meta_learner_multihead/multihead_meta_learner_latest.pt")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--site-labeled-only", dest="site_labeled_only", action="store_true", default=True)
    parser.add_argument("--no-site-labeled-only", dest="site_labeled_only", action="store_false")
    args = parser.parse_args()

    n_models, atom_feature_dim, global_feature_dim = _load_prediction_metadata(Path(args.predictions))

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    model = MultiHeadMetaLearner(
        n_models=n_models,
        n_cyp=5,
        atom_feature_dim=atom_feature_dim,
        global_feature_dim=global_feature_dim,
        hidden_dim=int(cfg.get("hidden_dim", 32)),
    )
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
    dataset = MetaLearnerDataset(args.predictions, test_drugs)

    metrics = {"site_top1": 0.0, "site_top3": 0.0, "cyp_acc": 0.0, "count": 0, "site_count": 0}
    site_attn_sum = None
    cyp_attn_sum = None
    for idx in range(len(dataset)):
        row = dataset[idx]
        atom_features = row["atom_features"].to(device)
        global_features = row["global_features"].to(device)
        site_scores_raw = row["site_scores_raw"].to(device)
        cyp_probs_raw = row["cyp_probs_raw"].to(device)
        with torch.no_grad():
            site_logits, cyp_logits, stats = model(atom_features, global_features, site_scores_raw, cyp_probs_raw)
        metrics["cyp_acc"] += float(int(torch.argmax(cyp_logits.detach()).item()) == int(row["cyp_label"].item()))
        if bool(torch.any(row["site_labels"] > 0.5)):
            out = evaluate_meta_predictions(site_logits, row["site_labels"], cyp_logits, int(row["cyp_label"].item()))
            metrics["site_top1"] += out["site_top1"]
            metrics["site_top3"] += out["site_top3"]
            metrics["site_count"] += 1
        metrics["count"] += 1
        site_attn_sum = stats["site_attention"].detach().cpu() if site_attn_sum is None else site_attn_sum + stats["site_attention"].detach().cpu()
        cyp_attn_sum = stats["cyp_attention"].detach().cpu() if cyp_attn_sum is None else cyp_attn_sum + stats["cyp_attention"].detach().cpu()
    denom = max(1, metrics["count"])
    result = {
        "site_top1": metrics["site_top1"] / max(1, metrics["site_count"]),
        "site_top3": metrics["site_top3"] / max(1, metrics["site_count"]),
        "cyp_acc": metrics["cyp_acc"] / denom,
        "count": metrics["count"],
        "site_count": metrics["site_count"],
    }
    if metrics["count"] > 0:
        result["site_attention"] = (site_attn_sum / float(metrics["count"])).tolist()
        result["cyp_attention"] = (cyp_attn_sum / float(metrics["count"])).tolist()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
