from __future__ import annotations

import argparse
import json
from pathlib import Path

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig
from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.arbitration.hybrid_arbitrator import HybridArbitrator
from enzyme_software.liquid_nn_v2.features.graph_builder import smiles_to_graph
from enzyme_software.liquid_nn_v2.features.manual_engine_features import attach_manual_engine_features_to_graph, extract_module_minus1_features
from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs, move_to_device
from enzyme_software.liquid_nn_v2.utils.mol_provenance import mol_provenance_context


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
    parser = argparse.ArgumentParser(description="Run a single hybrid LNN + manual-engine prediction")
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--target-bond", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    base_model = LiquidMetabolismNetV2(
        ModelConfig.light_advanced(
            use_manual_engine_priors=True,
            return_intermediate_stats=True,
        )
    )
    model = HybridLNNModel(base_model)
    device = _resolve_device(args.device)
    model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=False)

    with mol_provenance_context(
        caller_module="run_hybrid_prediction",
        module_triggered="run_hybrid_prediction",
        source_category="evaluation script",
        original_smiles=args.smiles,
        drug_name=args.smiles,
        drug_id="cli_single_molecule",
    ):
        graph = smiles_to_graph(args.smiles)
        graph = attach_manual_engine_features_to_graph(graph, target_bond=args.target_bond)
        batch = move_to_device(collate_molecule_graphs([graph]), device)

    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    site_scores = outputs["site_scores"].detach().cpu().reshape(-1)
    top_sites = torch.argsort(site_scores, descending=True)[:3].tolist()
    cyp_probs = torch.softmax(outputs["cyp_logits"], dim=-1).detach().cpu()
    cyp_idx = int(torch.argmax(cyp_probs, dim=-1)[0].item())
    cyp_name = list(base_model.config.cyp_names)[cyp_idx]
    lnn_output = {
        "site_atoms": [int(v) for v in top_sites],
        "site_scores": [float(v) for v in site_scores.tolist()],
        "cyp": cyp_name,
        "cyp_confidence": float(cyp_probs[0, cyp_idx].item()),
    }

    with mol_provenance_context(
        caller_module="run_hybrid_prediction",
        module_triggered="run_hybrid_prediction",
        source_category="evaluation script",
        original_smiles=args.smiles,
        drug_name=args.smiles,
        drug_id="cli_single_molecule",
    ):
        manual_features = extract_module_minus1_features(args.smiles, target_bond=args.target_bond) or {}
    manual_output = {
        "predicted_sites": [
            int(item.get("heavy_atom_index"))
            for item in (manual_features.get("candidate_sites") or [])
            if isinstance(item, dict) and isinstance(item.get("heavy_atom_index"), int)
        ][:3],
        "route": manual_features.get("selected_route") or "unknown",
        "route_confidence": float(manual_features.get("route_confidence") or 0.0),
    }

    prediction = HybridArbitrator().arbitrate(lnn_output, manual_output)
    result = {
        "prediction": prediction.__dict__,
        "lnn_output": lnn_output,
        "manual_output": manual_output,
        "model_diagnostics": outputs.get("diagnostics") or {},
        "hybrid_prior": outputs.get("hybrid_manual_prior") or {},
    }

    print(json.dumps(result, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
