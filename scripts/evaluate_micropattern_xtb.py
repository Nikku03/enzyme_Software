from __future__ import annotations

import argparse
import json
from pathlib import Path

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.config import MicroPatternXTBConfig
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.dataset import (
    create_micropattern_dataloaders_from_drugs,
    filter_site_labeled_drugs,
    split_drugs,
)
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.model import (
    MicroPatternXTBHybridModel,
    load_base_hybrid_checkpoint,
)
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.trainer import MicroPatternTrainer


def _load_drugs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _without_xtb(loader):
    for batch in loader:
        if batch is None:
            yield batch
            continue
        copied = dict(batch)
        if copied.get("xtb_atom_features") is not None:
            copied["xtb_atom_features"] = torch.zeros_like(copied["xtb_atom_features"])
        if copied.get("xtb_atom_valid_mask") is not None:
            copied["xtb_atom_valid_mask"] = torch.zeros_like(copied["xtb_atom_valid_mask"])
        if copied.get("xtb_mol_valid") is not None:
            copied["xtb_mol_valid"] = torch.zeros_like(copied["xtb_mol_valid"])
        yield copied


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
    parser = argparse.ArgumentParser(description="Evaluate base vs micropattern xTB reranker")
    parser.add_argument("--dataset", default="data/drugbank_standardized.json")
    parser.add_argument("--supercyp-dataset", default=None)
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--base-checkpoint", default="checkpoints/hybrid_lnn_latest.pt")
    parser.add_argument("--reranker-checkpoint", default="checkpoints/micropattern_xtb/micropattern_xtb_latest.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site-labeled-only", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.68)
    parser.add_argument("--val-ratio", type=float, default=0.16)
    parser.add_argument("--xtb-cache-dir", default="cache/micropattern_xtb")
    parser.add_argument("--device", default=None)
    parser.add_argument("--without-xtb", action="store_true")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    config = MicroPatternXTBConfig.default(
        base_checkpoint=args.base_checkpoint,
        xtb_cache_dir=args.xtb_cache_dir,
        compute_xtb_if_missing=False,
    )
    primary = _load_drugs(Path(args.dataset))
    if args.supercyp_dataset:
        primary.extend(_load_drugs(Path(args.supercyp_dataset)))
    if args.site_labeled_only:
        primary = filter_site_labeled_drugs(primary)
    train_drugs, val_drugs, test_drugs = split_drugs(
        primary,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    _, _, test_loader = create_micropattern_dataloaders_from_drugs(
        train_drugs,
        val_drugs,
        test_drugs,
        structure_sdf=args.structure_sdf,
        batch_size=args.batch_size,
        xtb_cache_dir=config.xtb_cache_dir,
        compute_xtb_if_missing=False,
    )

    base_model = load_base_hybrid_checkpoint(config.base_checkpoint, device=device)
    model = MicroPatternXTBHybridModel(base_model, config=config)
    reranker_payload = torch.load(args.reranker_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(reranker_payload.get("model_state_dict") or reranker_payload, strict=False)
    trainer = MicroPatternTrainer(model=model, config=config, device=device)
    eval_loader = _without_xtb(test_loader) if args.without_xtb else test_loader
    metrics = trainer.evaluate(eval_loader)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
