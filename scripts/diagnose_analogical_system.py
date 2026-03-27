"""
scripts/diagnose_analogical_system.py

Serious diagnostic harness for the NEXUS analogical reasoning system.

It answers the questions that the training log does not:
    - Is retrieval using the intended embedding space?
    - Is exact-query masking actually working?
    - Which transport backend is winning in practice?
    - How often does transport succeed?
    - What is the analogical branch's own top-1 / top-2 accuracy before the
      live scan head dilutes the picture?
    - Does the full trainer gate ever trust analogy on real forward passes?

Examples
--------
python scripts/diagnose_analogical_system.py
python scripts/diagnose_analogical_system.py --bank-mode continuous --max-mols 16
python scripts/diagnose_analogical_system.py --bank-scope all-isoforms --full-forward 4
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from rdkit import Chem

from nexus.data.metabolic_dataset import ZaretzkiMetabolicDataset, geometric_collate_fn
from nexus.reasoning.baseline_memory import _extract_som_idx
from nexus.training.causal_trainer import Metabolic_Causal_Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sdf", default=str(_REPO / "data/ATTNSOM/cyp_dataset/3A4.sdf"))
    p.add_argument(
        "--bank-scope",
        choices=("dataset", "all-isoforms"),
        default="all-isoforms",
        help="Populate the analogical bank from just the query dataset or all CYP isoforms.",
    )
    p.add_argument(
        "--bank-mode",
        choices=("fingerprint", "continuous"),
        default="continuous",
        help="Use Morgan-fingerprint retrieval or continuous projected retrieval.",
    )
    p.add_argument("--max-mols", type=int, default=32, help="Maximum query molecules to diagnose.")
    p.add_argument(
        "--bank-max-mols",
        type=int,
        default=0,
        help="Optional cap per bank SDF (0 = all available molecules).",
    )
    p.add_argument(
        "--full-forward",
        type=int,
        default=0,
        help="Run the real trainer forward path on the first N query molecules for gate diagnostics.",
    )
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--out-json", default=None, help="Optional path for the full diagnostic report JSON.")
    args, _ = p.parse_known_args()
    return args


def _canonical_smiles(mol) -> str | None:
    try:
        base = Chem.RemoveHs(Chem.Mol(mol))
        return Chem.MolToSmiles(base, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _pick_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _configure_trainer(device: torch.device) -> Metabolic_Causal_Trainer:
    trainer = Metabolic_Causal_Trainer(
        dynamics_steps=1,
        dynamics_dt=0.001,
        checkpoint_dynamics=False,
        enable_wsd_scheduler=True,
        low_memory_train_mode=True,
        enable_static_compile=False,
        use_galore=False,
    ).to(device)

    # Keep the field/scanner cheap enough for diagnostics.
    qe = trainer.model.module1.field_engine.quantum_enforcer
    qe.integration_resolution = 8
    qe.integration_chunk_size = 32

    se = trainer.model.module1.field_engine.query_engine
    se.n_points = 8
    se.radius = 1.0
    se.query_chunk_size = 2
    se.shell_fractions = (0.5, 1.0)
    se.refine_steps = 0
    se.create_approach_graph = False

    trainer.memory_bank.device = str(device)
    if trainer.memory_bank.pgw is not None:
        trainer.memory_bank.pgw.device = str(device)
    trainer.eval()
    return trainer


def _iter_bank_sources(bank_scope: str, query_sdf: Path) -> Iterable[Path]:
    if bank_scope == "dataset":
        yield query_sdf
        return
    cyp_dir = query_sdf.parent
    for sdf in sorted(cyp_dir.glob("*.sdf")):
        yield sdf


def _load_bank_mols(query_sdf: Path, bank_scope: str, bank_max_mols: int) -> List:
    bank_mols: List = []
    for sdf in _iter_bank_sources(bank_scope, query_sdf):
        ds = ZaretzkiMetabolicDataset(str(sdf), max_molecules=bank_max_mols)
        for mol in ds.mols:
            mol.SetProp("_DIAG_SOURCE", sdf.stem)
            bank_mols.append(mol)
    return bank_mols


def _topk_hit(pred: torch.Tensor, target_idx: int, k: int) -> bool:
    if pred.numel() == 0 or target_idx < 0 or target_idx >= pred.numel():
        return False
    kk = min(k, int(pred.numel()))
    topk = torch.topk(pred, kk).indices.tolist()
    return int(target_idx) in topk


def _mean_or_zero(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _counter_dict(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def _retrieval_diagnostic(
    trainer: Metabolic_Causal_Trainer,
    dataset: ZaretzkiMetabolicDataset,
    *,
    bank_mode: str,
) -> Dict[str, Any]:
    bank = trainer.memory_bank
    query_records: List[Dict[str, Any]] = []
    embedding_spaces: Counter = Counter()
    transport_backends: Counter = Counter()
    retrieved_sources: Counter = Counter()

    top1_hits = 0
    top2_hits = 0
    transport_hits = 0
    self_leaks = 0
    zero_pred = 0
    confidences: List[float] = []
    support_sizes: List[float] = []
    transported_masses: List[float] = []

    for idx, mol in enumerate(dataset.mols):
        true_som = _extract_som_idx(mol)
        if true_som is None:
            continue

        query_embedding = None
        query_multivectors = None
        if bank_mode == "continuous":
            encoded = trainer.encode_mol_for_memory_bank(mol)
            query_embedding = encoded.get("graph_embedding")
            query_multivectors = encoded.get("node_multivectors")

        result = bank.retrieve_and_transport(
            mol,
            query_smiles=_canonical_smiles(mol),
            mechanism_encoder=trainer.gated_loss.mechanism_encoder,
            query_embedding=query_embedding,
            query_multivectors=query_multivectors,
        )

        pred = result.analogical_pred.detach().float().cpu().view(-1)
        pred_idx = int(pred.argmax().item()) if pred.numel() > 0 and float(pred.max().item()) > 0.0 else None
        top1 = _topk_hit(pred, true_som, 1)
        top2 = _topk_hit(pred, true_som, 2)
        retrieved_source = (
            result.retrieved_mol.GetProp("_DIAG_SOURCE")
            if result.retrieved_mol.HasProp("_DIAG_SOURCE")
            else "unknown"
        )
        query_key = _canonical_smiles(mol)
        retrieved_key = _canonical_smiles(result.retrieved_mol)
        same_query = bool(query_key is not None and query_key == retrieved_key)

        top1_hits += int(top1)
        top2_hits += int(top2)
        transport_hits += int(result.transport_succeeded)
        self_leaks += int(same_query)
        zero_pred += int(pred_idx is None)
        confidences.append(float(result.confidence))
        support_sizes.append(float(result.transport_support_size))
        transported_masses.append(float(result.transported_mass))
        embedding_spaces[result.embedding_space] += 1
        transport_backends[result.transport_backend] += 1
        retrieved_sources[retrieved_source] += 1

        if len(query_records) < 12:
            query_records.append(
                {
                    "query_index": idx,
                    "query_atoms": mol.GetNumAtoms(),
                    "query_source": Path(dataset.sdf_file_path).stem,
                    "true_som": int(true_som),
                    "predicted_atom": pred_idx,
                    "top1": bool(top1),
                    "top2": bool(top2),
                    "confidence": float(result.confidence),
                    "embedding_space": result.embedding_space,
                    "transport_backend": result.transport_backend,
                    "transport_succeeded": bool(result.transport_succeeded),
                    "transport_support_size": int(result.transport_support_size),
                    "transported_mass": float(result.transported_mass),
                    "retrieved_source": retrieved_source,
                    "retrieved_som": int(result.retrieved_som_idx),
                    "retrieved_same_query": bool(same_query),
                }
            )

    projected = (
        int(bank.memory_projected_mask.sum().item())
        if bank.memory_projected_mask is not None
        else 0
    )
    bank_size = len(bank.historical_mols)

    return {
        "query_count": len(dataset.mols),
        "bank_size": bank_size,
        "bank_projected_count": projected,
        "bank_projected_fraction": (projected / bank_size) if bank_size else 0.0,
        "retrieval_embedding_space_counts": _counter_dict(embedding_spaces),
        "transport_backend_counts": _counter_dict(transport_backends),
        "retrieved_source_counts": _counter_dict(retrieved_sources),
        "transport_success_rate": transport_hits / max(len(dataset.mols), 1),
        "exact_query_leak_rate": self_leaks / max(len(dataset.mols), 1),
        "analogical_zero_prediction_rate": zero_pred / max(len(dataset.mols), 1),
        "analogical_top1": top1_hits / max(len(dataset.mols), 1),
        "analogical_top2": top2_hits / max(len(dataset.mols), 1),
        "mean_confidence": _mean_or_zero(confidences),
        "median_confidence": float(statistics.median(confidences)) if confidences else 0.0,
        "mean_support_size": _mean_or_zero(support_sizes),
        "mean_transported_mass": _mean_or_zero(transported_masses),
        "sample_queries": query_records,
    }


def _full_forward_diagnostic(
    trainer: Metabolic_Causal_Trainer,
    dataset: ZaretzkiMetabolicDataset,
    *,
    count: int,
    device: torch.device,
) -> Dict[str, Any]:
    if count <= 0:
        return {}

    metrics_rows: List[Dict[str, float]] = []
    failures: List[str] = []
    for idx in range(min(count, len(dataset))):
        try:
            batch = geometric_collate_fn([dataset[idx]])
            batch = trainer._move_to_device(batch)
            result = trainer.forward_batch(batch)
            row = {"index": float(idx)}
            for key in (
                "som_top1",
                "som_top2",
                "ana_loss_total",
                "ana_gate_open",
                "ana_confidence",
                "ana_transport_ok",
                "ana_peak",
                "ana_watson_agreement",
                "ana_encoder_loss",
                "physics_cache_hit",
            ):
                value = result.metrics.get(key)
                if value is None:
                    continue
                row[key] = float(torch.as_tensor(value).detach().float().cpu().item())
            metrics_rows.append(row)
        except Exception as exc:
            failures.append(f"idx={idx}: {type(exc).__name__}: {exc}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary: Dict[str, Any] = {"samples": metrics_rows}
    if metrics_rows:
        scalar_keys = sorted({k for row in metrics_rows for k in row.keys() if k != "index"})
        summary["means"] = {
            key: _mean_or_zero([row[key] for row in metrics_rows if key in row])
            for key in scalar_keys
        }
    if failures:
        summary["failures"] = failures
    return summary


def _print_summary(report: Dict[str, Any]) -> None:
    cfg = report["config"]
    retrieval = report["retrieval"]
    print(f"Device           : {cfg['device']}")
    print(f"Query dataset    : {cfg['sdf']}")
    print(f"Bank scope/mode  : {cfg['bank_scope']} / {cfg['bank_mode']}")
    print(f"Queries analysed : {retrieval['query_count']}")
    print(f"Bank size        : {retrieval['bank_size']}")
    print(
        f"Projected bank   : {retrieval['bank_projected_count']}/{retrieval['bank_size']} "
        f"({retrieval['bank_projected_fraction']:.1%})"
    )
    print(f"Retrieval space  : {retrieval['retrieval_embedding_space_counts']}")
    print(f"Transport modes  : {retrieval['transport_backend_counts']}")
    print(f"Retrieved source : {retrieval['retrieved_source_counts']}")
    print(f"Transport OK     : {retrieval['transport_success_rate']:.1%}")
    print(f"Self leakage     : {retrieval['exact_query_leak_rate']:.1%}")
    print(f"Ana zero-pred    : {retrieval['analogical_zero_prediction_rate']:.1%}")
    print(f"Ana top1/top2    : {retrieval['analogical_top1']:.1%} / {retrieval['analogical_top2']:.1%}")
    print(
        f"Confidence       : mean={retrieval['mean_confidence']:.4f}  "
        f"median={retrieval['median_confidence']:.4f}"
    )
    print(
        f"Transport stats  : support={retrieval['mean_support_size']:.2f}  "
        f"mass={retrieval['mean_transported_mass']:.4f}"
    )

    if report.get("forward"):
        means = report["forward"].get("means", {})
        if means:
            print("Forward means    :", means)
        failures = report["forward"].get("failures", [])
        if failures:
            print("Forward failures :")
            for item in failures:
                print(f"  {item}")


def main() -> None:
    args = parse_args()
    device = _pick_device(args.device)
    query_sdf = Path(args.sdf)

    print("Starting analogical-system diagnostic...")
    print(f"Device: {device}")

    dataset = ZaretzkiMetabolicDataset(str(query_sdf), max_molecules=args.max_mols)
    print(f"Loaded query dataset: {len(dataset)} molecules from {query_sdf}")

    trainer = _configure_trainer(device)

    bank_mols = _load_bank_mols(query_sdf, args.bank_scope, args.bank_max_mols)
    print(f"Loaded bank molecules: {len(bank_mols)}")

    continuous_encoder = trainer.encode_mol_for_memory_bank if args.bank_mode == "continuous" else None
    trainer.memory_bank.populate_from_mols(bank_mols, continuous_encoder=continuous_encoder)
    print(f"Memory bank ready: {len(trainer.memory_bank.historical_mols)} molecules")

    retrieval = _retrieval_diagnostic(trainer, dataset, bank_mode=args.bank_mode)
    forward = _full_forward_diagnostic(
        trainer,
        dataset,
        count=args.full_forward,
        device=device,
    )

    report = {
        "config": {
            "device": str(device),
            "sdf": str(query_sdf),
            "bank_scope": args.bank_scope,
            "bank_mode": args.bank_mode,
            "max_mols": int(args.max_mols),
            "bank_max_mols": int(args.bank_max_mols),
            "full_forward": int(args.full_forward),
        },
        "retrieval": retrieval,
        "forward": forward,
    }

    _print_summary(report)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"Report saved → {out}")


if __name__ == "__main__":
    main()
