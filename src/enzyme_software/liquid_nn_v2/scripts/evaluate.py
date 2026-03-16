from __future__ import annotations

import argparse

from enzyme_software.liquid_nn_v2._compat import require_torch
from enzyme_software.liquid_nn_v2.config import ModelConfig
from enzyme_software.liquid_nn_v2.data.drug_database import DRUG_DATABASE
from enzyme_software.liquid_nn_v2.features.graph_builder import smiles_to_graph
from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
from enzyme_software.liquid_nn_v2.training.trainer import Trainer
from enzyme_software.liquid_nn_v2.utils.mol_provenance import mol_provenance_context


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Evaluate LiquidMetabolismNetV2 on the curated drug set")
    parser.parse_args()
    graphs = []
    for drug_id, entry in DRUG_DATABASE.items():
        with mol_provenance_context(
            caller_module="evaluation script",
            module_triggered="evaluation script",
            source_category="evaluation script",
            original_smiles=entry["smiles"],
            drug_name=str(entry.get("name", drug_id)),
            drug_id=str(drug_id),
        ):
            graphs.append(smiles_to_graph(entry["smiles"], cyp_label=entry["primary_cyp"], site_atoms=entry["site_atoms"]))
    trainer = Trainer(model=LiquidMetabolismNetV2(ModelConfig()))
    print(trainer.evaluate(graphs))


if __name__ == "__main__":
    main()
