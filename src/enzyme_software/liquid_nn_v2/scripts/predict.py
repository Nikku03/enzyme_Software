from __future__ import annotations

import argparse

from enzyme_software.liquid_nn_v2._compat import require_torch
from enzyme_software.liquid_nn_v2.config import ModelConfig
from enzyme_software.liquid_nn_v2.features.graph_builder import smiles_to_graph
from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs
from enzyme_software.liquid_nn_v2.utils.mol_provenance import mol_provenance_context


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Predict metabolism sites and CYP class for one molecule")
    parser.add_argument("smiles")
    args = parser.parse_args()
    with mol_provenance_context(
        caller_module="prediction script",
        module_triggered="prediction script",
        source_category="evaluation script",
        original_smiles=args.smiles,
        drug_name=args.smiles,
        drug_id="cli_single_molecule",
    ):
        graph = smiles_to_graph(args.smiles)
    batch = collate_molecule_graphs([graph])
    model = LiquidMetabolismNetV2(ModelConfig())
    outputs = model(batch)
    site_scores = outputs["site_scores"].detach().cpu().numpy().reshape(-1)
    cyp_logits = outputs["cyp_logits"].detach().cpu().numpy()[0]
    ranked_atoms = sorted(range(len(site_scores)), key=lambda idx: float(site_scores[idx]), reverse=True)
    class_names = list(getattr(model.config, "cyp_names", [f"CYP_{idx}" for idx in range(len(cyp_logits))]))
    ranked_cyp = sorted(zip(class_names, cyp_logits), key=lambda row: float(row[1]), reverse=True)
    print({"top_atoms": ranked_atoms[:5], "top_cyp": ranked_cyp[:3]})


if __name__ == "__main__":
    main()
