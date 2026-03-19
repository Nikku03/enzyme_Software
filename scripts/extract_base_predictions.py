from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from enzyme_software.cahml import CAHML, CAHMLConfig
from enzyme_software.cahml.components.chemistry_encoder import ChemistryFeatureExtractor
from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.meta_learner import FeatureStacker, MultiModelPredictor, load_default_model_specs


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


def _load_cahml(checkpoint_path: str | Path, device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    allowed = set(CAHMLConfig().__dict__.keys())
    config = CAHMLConfig(**{key: value for key, value in cfg.items() if key in allowed})
    model = CAHML(config)
    model.load_state_dict(payload.get("model_state_dict") or payload, strict=False)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Extract stacked base-model predictions for the meta learner")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="cache/meta_learner/base_predictions.pt")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site-labeled-only", action="store_true")
    parser.add_argument("--model-names", nargs="*", default=["hybrid_lnn", "hybrid_full_xtb", "micropattern_xtb"])
    parser.add_argument("--cahml-checkpoint", default=None, help="Optional CAHML checkpoint to append as a fourth expert")
    args = parser.parse_args()

    drugs = _load_drugs(Path(args.dataset))
    if args.site_labeled_only:
        drugs = [drug for drug in drugs if _has_site_labels(drug)]
    random.Random(args.seed).shuffle(drugs)

    specs = load_default_model_specs(args.model_names)
    predictor = MultiModelPredictor(specs, device=args.device, structure_sdf=args.structure_sdf)
    base_model_names = [spec.name for spec in specs if spec.name in predictor.wrappers]
    base_stacker = FeatureStacker(base_model_names)
    final_model_names = list(base_model_names)
    cahml_model = None
    chemistry_extractor = None
    device = _resolve_device(args.device)
    if args.cahml_checkpoint:
        cahml_model = _load_cahml(args.cahml_checkpoint, device)
        chemistry_extractor = ChemistryFeatureExtractor()
        final_model_names.append("cahml")
    stacker = FeatureStacker(final_model_names)

    predictions = {}
    failures = {}
    for idx, drug in enumerate(drugs, start=1):
        smiles = str(drug.get("smiles", "")).strip()
        if not smiles:
            failures[f"missing_smiles_{idx}"] = "missing_smiles"
            continue
        try:
            raw_predictions = predictor.predict_all(drug)
            if cahml_model is not None and chemistry_extractor is not None:
                chemistry = chemistry_extractor.extract(smiles)
                if chemistry is None:
                    raise ValueError("Failed to extract chemistry features for CAHML")
                base_stacked = base_stacker.stack(raw_predictions)
                with torch.no_grad():
                    cahml_out = cahml_model(
                        chemistry.mol_features_raw.to(device),
                        chemistry.atom_features_raw.to(device),
                        chemistry.smarts_matches.to(device),
                        base_stacked["site_scores_raw"].to(device),
                        base_stacked["cyp_probs_raw"].to(device),
                    )
                raw_predictions["cahml"] = {
                    "site_scores": torch.sigmoid(cahml_out["site_scores"]).detach().cpu(),
                    "cyp_probs": torch.softmax(cahml_out["cyp_logits"], dim=-1).detach().cpu(),
                    "site_labels": base_stacked["site_labels"].detach().cpu(),
                    "cyp_label": int(base_stacked["cyp_label"].item()),
                    "num_atoms": int(base_stacked["num_atoms"].item()),
                    "reaction_type": cahml_out["reaction_type"],
                    "trusted_model": cahml_out["explanation"]["trusted_model"],
                }
            stacked = stacker.stack(raw_predictions)
            stacked["model_predictions"] = {
                name: {
                    key: value
                    for key, value in (pred or {}).items()
                    if key in {"site_scores", "cyp_probs", "base_site_scores", "stats", "num_atoms", "cyp_label", "reaction_type", "trusted_model"}
                }
                for name, pred in raw_predictions.items()
            }
            predictions[smiles] = stacked
        except Exception as exc:
            failures[smiles] = str(exc)
        if idx % 25 == 0 or idx == len(drugs):
            print(f"{idx}/{len(drugs)} processed | ok={len(predictions)} | failed={len(failures)}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "predictions": predictions,
            "failures": failures,
            "dataset": str(args.dataset),
            "model_names": final_model_names,
        },
        out_path,
    )
    print(f"Saved predictions: {out_path}", flush=True)
    print(f"Successful drugs: {len(predictions)}", flush=True)
    print(f"Failed drugs: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
