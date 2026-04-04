from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig, TrainingConfig
from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb import (
    create_full_xtb_dataloaders_from_drugs,
    load_full_xtb_warm_start,
    split_drugs,
)
from enzyme_software.liquid_nn_v2.features.xtb_features import FULL_XTB_FEATURE_DIM
from enzyme_software.liquid_nn_v2.training.episode_logger import EpisodeLogger
from enzyme_software.liquid_nn_v2.training.trainer import Trainer


def _initialized_state_dict(model) -> dict:
    state = {}
    uninitialized_type = getattr(torch.nn.parameter, "UninitializedParameter", ())
    for key, value in model.state_dict().items():
        if isinstance(value, uninitialized_type):
            continue
        state[key] = value.detach().cpu() if hasattr(value, "detach") else value
    return state


def _resolve_device(name: str | None):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return float(raw)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return int(raw)


def _collect_model_overrides() -> dict[str, int | float]:
    mapping = {
        "HYBRID_COLAB_NEXUS_WAVE_HIDDEN_DIM": (_env_int, "nexus_wave_hidden_dim"),
        "HYBRID_COLAB_NEXUS_GRAPH_DIM": (_env_int, "nexus_graph_dim"),
        "HYBRID_COLAB_NEXUS_MEMORY_CAPACITY": (_env_int, "nexus_memory_capacity"),
        "HYBRID_COLAB_NEXUS_MEMORY_TOPK": (_env_int, "nexus_memory_topk"),
        "HYBRID_COLAB_NEXUS_WAVE_AUX_WEIGHT": (_env_float, "nexus_wave_aux_weight"),
        "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT": (_env_float, "nexus_analogical_aux_weight"),
        "HYBRID_COLAB_NEXUS_WAVE_SITE_INIT": (_env_float, "nexus_wave_site_init"),
        "HYBRID_COLAB_NEXUS_ANALOGICAL_SITE_INIT": (_env_float, "nexus_analogical_site_init"),
        "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_INIT": (_env_float, "nexus_analogical_cyp_init"),
        "HYBRID_COLAB_NEXUS_SITE_ARBITER_HIDDEN_DIM": (_env_int, "nexus_site_arbiter_hidden_dim"),
        "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT": (_env_float, "nexus_site_arbiter_dropout"),
        "HYBRID_COLAB_NEXUS_LNN_VOTE_AUX_WEIGHT": (_env_float, "nexus_lnn_vote_aux_weight"),
        "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT": (_env_float, "nexus_wave_vote_aux_weight"),
        "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT": (_env_float, "nexus_analogical_vote_aux_weight"),
        "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT": (_env_float, "nexus_board_entropy_weight"),
        "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE": (_env_float, "nexus_vote_logit_scale"),
        "HYBRID_COLAB_NEXUS_LIVE_WAVE_VOTE_GRAD_SCALE": (_env_float, "nexus_live_wave_vote_grad_scale"),
        "HYBRID_COLAB_NEXUS_LIVE_ANALOGICAL_VOTE_GRAD_SCALE": (_env_float, "nexus_live_analogical_vote_grad_scale"),
        "HYBRID_COLAB_NEXUS_WAVE_SIDEINFO_AUX_WEIGHT": (_env_float, "nexus_wave_sideinfo_aux_weight"),
        "HYBRID_COLAB_NEXUS_ANALOGICAL_SIDEINFO_AUX_WEIGHT": (_env_float, "nexus_analogical_sideinfo_aux_weight"),
        "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE": (_env_float, "nexus_analogical_cyp_aux_scale"),
        "HYBRID_COLAB_NEXUS_SIDEINFO_HIDDEN_DIM": (_env_int, "nexus_sideinfo_hidden_dim"),
        "HYBRID_COLAB_NEXUS_SIDEINFO_DROPOUT": (_env_float, "nexus_sideinfo_dropout"),
        "HYBRID_COLAB_NEXUS_SIDEINFO_INIT_SCALE": (_env_float, "nexus_sideinfo_init_scale"),
        "HYBRID_COLAB_SITE_RANKING_WEIGHT": (_env_float, "site_ranking_weight"),
        "HYBRID_COLAB_SITE_HARD_NEGATIVE_FRACTION": (_env_float, "site_hard_negative_fraction"),
        "HYBRID_COLAB_SITE_TOP1_MARGIN_TOPK": (_env_int, "site_top1_margin_topk"),
        "HYBRID_COLAB_SITE_TOP1_MARGIN_DECAY": (_env_float, "site_top1_margin_decay"),
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DEFAULT": (_env_float, "site_source_weight_default"),
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DRUGBANK": (_env_float, "site_source_weight_drugbank"),
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_AZ120": (_env_float, "site_source_weight_az120"),
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_METXBIODB": (_env_float, "site_source_weight_metxbiodb"),
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_ATTNSOM": (_env_float, "site_source_weight_attnsom"),
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_CYP_DBS_EXTERNAL": (_env_float, "site_source_weight_cyp_dbs_external"),
        "HYBRID_COLAB_DOMAIN_ADV_WEIGHT": (_env_float, "domain_adv_weight"),
        "HYBRID_COLAB_DOMAIN_ADV_GRAD_SCALE": (_env_float, "domain_adv_grad_scale"),
        "HYBRID_COLAB_DOMAIN_ADV_HIDDEN_DIM": (_env_int, "domain_adv_hidden_dim"),
        "HYBRID_COLAB_SOURCE_ALIGN_WEIGHT": (_env_float, "source_align_weight"),
        "HYBRID_COLAB_SOURCE_ALIGN_COV_WEIGHT": (_env_float, "source_align_cov_weight"),
    }
    overrides: dict[str, int | float] = {}
    for env_name, (parser, field_name) in mapping.items():
        value = parser(env_name)
        if value is not None:
            overrides[field_name] = value
    return overrides


def _load_drugs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _has_site_labels(drug: dict) -> bool:
    return bool(drug.get("som") or drug.get("site_atoms") or drug.get("site_atom_indices") or drug.get("metabolism_sites"))


def _primary_cyp(drug: dict) -> str:
    value = str(drug.get("cyp") or drug.get("primary_cyp") or "").strip()
    if value:
        return value
    all_cyps = list(drug.get("all_cyps", []) or [])
    return str(all_cyps[0]).strip() if all_cyps else ""


def _parse_csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in str(raw or "").split(",") if token.strip()]


def _normalize_source_name(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _site_atom_indices(drug: dict) -> list[int]:
    site_atoms: list[int] = []
    if drug.get("som"):
        for som in drug["som"]:
            atom_idx = som.get("atom_idx", som) if isinstance(som, dict) else som
            if isinstance(atom_idx, int):
                site_atoms.append(int(atom_idx))
    elif drug.get("site_atoms"):
        site_atoms = [int(v) for v in drug.get("site_atoms", []) if isinstance(v, int)]
    elif drug.get("site_atom_indices"):
        site_atoms = [int(v) for v in drug.get("site_atom_indices", []) if isinstance(v, int)]
    elif drug.get("metabolism_sites"):
        site_atoms = [int(v) for v in drug.get("metabolism_sites", []) if isinstance(v, int)]
    return sorted(set(site_atoms))


def _canonical_smiles_key(smiles: str) -> str:
    text = " ".join(str(smiles or "").strip().split())
    if not text:
        return ""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            return str(Chem.MolToSmiles(mol, canonical=True))
    except Exception:
        pass
    return text


def _safe_num_atoms(drug: dict) -> int:
    value = drug.get("num_atoms")
    if isinstance(value, int) and value > 0:
        return int(value)
    smiles = _canonical_smiles_key(drug.get("smiles", ""))
    if smiles:
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return int(mol.GetNumAtoms())
        except Exception:
            pass
    return 0


def _atom_bucket(drug: dict) -> str:
    num_atoms = _safe_num_atoms(drug)
    if num_atoms <= 0:
        return "unknown"
    if num_atoms <= 15:
        return "<=15"
    if num_atoms <= 25:
        return "16-25"
    if num_atoms <= 40:
        return "26-40"
    if num_atoms <= 60:
        return "41-60"
    return "61+"


def _site_count_bucket(drug: dict) -> str:
    count = len(_site_atom_indices(drug))
    if count <= 0:
        return "none"
    if count == 1:
        return "single"
    return "multi"


def _near_duplicate_summary(items: list[dict]) -> dict[str, int]:
    smiles_counts = Counter(_canonical_smiles_key(d.get("smiles", "")) for d in items)
    nonempty = {key: value for key, value in smiles_counts.items() if key}
    return {
        "duplicate_rows": int(sum(value - 1 for value in nonempty.values() if value > 1)),
        "duplicate_keys": int(sum(1 for value in nonempty.values() if value > 1)),
        "unique_smiles": int(len(nonempty)),
    }


def _split_summary(items: list[dict]) -> dict[str, object]:
    return {
        "total": int(len(items)),
        "site_supervised": int(sum(1 for d in items if _has_site_labels(d))),
        "sources": dict(Counter(str(d.get("source", "DrugBank")) for d in items)),
        "atom_buckets": dict(Counter(_atom_bucket(d) for d in items)),
        "site_count_buckets": dict(Counter(_site_count_bucket(d) for d in items)),
        "near_duplicates": _near_duplicate_summary(items),
    }


def _filter_by_sources(items: list[dict], allowlist: list[str]) -> list[dict]:
    if not allowlist:
        return list(items)
    allowed = {_normalize_source_name(token) for token in allowlist}
    return [drug for drug in items if _normalize_source_name(str(drug.get("source", "DrugBank"))) in allowed]


def _load_xenosite_aux_entries(manifest_path: Path, *, topk: int = 1, per_file_limit: int = 0) -> list[dict]:
    from rdkit import Chem

    payload = json.loads(manifest_path.read_text())
    datasets = list(payload.get("datasets", []))
    root = manifest_path.parent
    merged: list[dict] = []
    seen_smiles: set[str] = set()
    topk = max(1, int(topk))
    per_file_limit = max(0, int(per_file_limit))
    for meta in datasets:
        rel = str(meta.get("file", "")).strip()
        if not rel:
            continue
        data_path = root / rel
        if not data_path.exists():
            continue
        data = json.loads(data_path.read_text())
        entries = list(data.get("entries", []))
        if per_file_limit > 0:
            entries = entries[:per_file_limit]
        for entry in entries:
            smiles = _canonical_smiles_key(entry.get("smiles", ""))
            if not smiles or smiles in seen_smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            num_atoms = int(mol.GetNumAtoms())
            pairs = list(entry.get("xenosite_score_pairs", []))
            site_atoms = []
            for pair in pairs[:topk]:
                try:
                    site_atoms.append(int(pair.get("atom_index")))
                except Exception:
                    continue
            if not site_atoms:
                top_atoms = entry.get("top_atoms") or []
                site_atoms = [int(v) for v in top_atoms[:topk] if isinstance(v, int)]
            site_atoms = sorted(set(idx for idx in site_atoms if 0 <= int(idx) < num_atoms))
            if not site_atoms:
                continue
            seen_smiles.add(smiles)
            merged.append(
                {
                    "id": f"xenosite:{entry.get('source', 'aux')}:{entry.get('mol_index', len(merged))}",
                    "name": entry.get("name") or f"xenosite_{len(merged)}",
                    "smiles": smiles,
                    "primary_cyp": "",
                    "all_cyps": [],
                    "reactions": [],
                    "site_atoms": sorted(set(site_atoms)),
                    "site_source": f"{entry.get('source', 'xenosite')}_top{topk}",
                    "source": "XenoSiteAux",
                    "confidence": "low",
                    "full_xtb_status": "external_uncomputed",
                    "auxiliary_site_only": True,
                    "xenosite_dense_scores": entry.get("xenosite_dense_scores"),
                }
            )
    return merged


def _summarize_xtb_validity(drugs: list[dict], cache_dir: Path) -> dict[str, object]:
    from enzyme_software.liquid_nn_v2.features.xtb_features import load_or_compute_full_xtb_features, payload_xtb_validity_summary

    strict_true_valid = 0
    cached_valid = 0
    training_usable_valid = 0
    statuses: dict[str, int] = {}
    source_kinds: dict[str, int] = {}
    for drug in drugs:
        smiles = str(drug.get("smiles", "")).strip()
        if not smiles:
            continue
        payload = load_or_compute_full_xtb_features(smiles, cache_dir=cache_dir, compute_if_missing=False)
        summary = payload_xtb_validity_summary(payload)
        if bool(summary["strict_true_xtb_valid"]):
            strict_true_valid += 1
        if bool(summary["cached_xtb_valid"]):
            cached_valid += 1
        if bool(summary["training_usable_xtb_valid"]):
            training_usable_valid += 1
        status = str(summary["status"] or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        source_kind = str(summary["source_kind"] or "unknown")
        source_kinds[source_kind] = source_kinds.get(source_kind, 0) + 1
    return {
        "total_molecules": int(len(drugs)),
        "strict_true_xtb_valid_molecules": int(strict_true_valid),
        "cached_xtb_valid_molecules": int(cached_valid),
        "training_usable_xtb_valid_molecules": int(training_usable_valid),
        "statuses": dict(sorted(statuses.items())),
        "source_kinds": dict(sorted(source_kinds.items())),
        "definitions": {
            "strict_true_xtb_valid_molecules": "Benchmark-grade true xTB provenance only.",
            "cached_xtb_valid_molecules": "Any cached xTB payload marked xtb_valid, including lookup/manual-backed payloads.",
            "training_usable_xtb_valid_molecules": "Training-usable xTB payloads with either strict true-xTB validity or at least one valid cached atom feature.",
        },
    }


def _best_history_entry(history: list[dict], metric_name: str) -> dict | None:
    if not history:
        return None
    return max(history, key=lambda row: float((row.get("val") or {}).get(metric_name, float("-inf"))))


def _nexus_diagnosis(history: list[dict]) -> dict[str, object]:
    best_site_top1 = _best_history_entry(history, "site_top1_acc")
    train_stats = dict((best_site_top1 or {}).get("train") or {})
    diagnosis: dict[str, object] = {
        "wave": {},
        "analogical": {},
        "summary": [],
    }
    wave_valid_mol = float(train_stats.get("nexus_wave_valid_mol_fraction", 0.0))
    wave_valid_atom = float(train_stats.get("nexus_wave_valid_atom_fraction", 0.0))
    wave_reliability = float(train_stats.get("nexus_wave_reliability_mean", 0.0))
    diagnosis["wave"] = {
        "valid_molecule_fraction": wave_valid_mol,
        "valid_atom_fraction": wave_valid_atom,
        "reliability_mean": wave_reliability,
        "assessment": (
            "weak_due_to_low_validity_and_low_reliability"
            if wave_valid_mol < 0.5 or wave_reliability < 0.15
            else "potentially_usable"
        ),
    }
    analogical_margin = float(train_stats.get("nexus_analogical_margin_mean", 0.0))
    analogical_concentration = float(train_stats.get("nexus_analogical_concentration_mean", 0.0))
    analogical_gate = float(train_stats.get("nexus_analogical_gate_mean", 0.0))
    precedent_size = float(train_stats.get("nexus_precedent_logbook_size", 0.0))
    diagnosis["analogical"] = {
        "confidence_mean": float(train_stats.get("nexus_analogical_confidence_mean", 0.0)),
        "gate_mean": analogical_gate,
        "margin_mean": analogical_margin,
        "concentration_mean": analogical_concentration,
        "selectivity_mean": float(train_stats.get("nexus_analogical_selectivity_mean", 0.0)),
        "precedent_logbook_size": precedent_size,
        "assessment": (
            "weak_due_to_diffuse_memory_and_missing_precedents"
            if analogical_margin < 0.03 or analogical_concentration < 0.02 or precedent_size <= 0.0
            else "potentially_usable"
        ),
    }
    summary: list[str] = []
    if wave_valid_mol < 0.5:
        summary.append("Wave is data-limited: fewer than half the molecules are training-usable for wave supervision.")
    if wave_reliability < 0.15:
        summary.append("Wave is reliability-limited: even valid molecules produce weak trusted wave signal.")
    if precedent_size <= 0.0:
        summary.append("Analogical is precedent-limited: no curated precedent logbook was loaded.")
    if analogical_margin < 0.03 or analogical_concentration < 0.02:
        summary.append("Analogical is retrieval-limited: memory matches are diffuse, with tiny support margin/concentration.")
    diagnosis["summary"] = summary
    return diagnosis


def _build_loaders_with_fallback(
    train_drugs: list[dict],
    val_drugs: list[dict],
    test_drugs: list[dict],
    *,
    args,
):
    common = dict(
        batch_size=args.batch_size,
        structure_sdf=args.structure_sdf,
        manual_target_bond=args.manual_target_bond,
        manual_feature_cache_dir=args.manual_feature_cache_dir,
        full_xtb_cache_dir=str(Path(args.xtb_cache_dir)),
        compute_full_xtb_if_missing=args.compute_xtb_if_missing,
        use_candidate_mask=bool(getattr(args, "use_candidate_mask", False)),
        candidate_cyp=str(getattr(args, "target_cyp", "") or "").strip() or None,
        balance_train_sources=bool(getattr(args, "balance_train_sources", False)),
        drop_failed=True,
    )
    try:
        loaders = create_full_xtb_dataloaders_from_drugs(
            train_drugs,
            val_drugs,
            test_drugs,
            use_manual_engine_features=True,
            **common,
        )
        return loaders, True
    except RuntimeError as exc:
        message = str(exc)
        if "zero valid graphs" not in message:
            raise
        print(
            "Full-xTB loader produced zero valid graphs with manual-engine features enabled. "
            "Retrying without manual-engine features.",
            flush=True,
        )
        print(f"Loader failure: {message}", flush=True)
        loaders = create_full_xtb_dataloaders_from_drugs(
            train_drugs,
            val_drugs,
            test_drugs,
            use_manual_engine_features=False,
            **common,
        )
        return loaders, False


def _resolve_precedent_logbook(path_arg: str, artifact_dir: Path) -> Path | None:
    if str(path_arg or "").strip():
        path = Path(path_arg)
        return path if path.exists() else None
    return None


def _attach_effective_split_summary(split_summary: dict[str, object], loaders: dict[str, object]) -> dict[str, object]:
    updated = {name: dict(summary) for name, summary in split_summary.items()}
    for split_name, loader in loaders.items():
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            continue
        summary = updated.get(split_name, {})
        valid_count = int(getattr(dataset, "_valid_count", 0))
        invalid_reasons = dict(getattr(dataset, "_invalid_reasons", {}) or {})
        total = int(summary.get("total", valid_count))
        summary["effective_total"] = valid_count
        summary["invalid_count"] = max(0, total - valid_count)
        summary["invalid_reasons"] = invalid_reasons
        updated[split_name] = summary
    return updated


def _effective_split_summary(split_summary: dict[str, object]) -> dict[str, dict[str, object]]:
    compact: dict[str, dict[str, object]] = {}
    for split_name, summary in split_summary.items():
        compact[split_name] = {
            "total": int(summary.get("effective_total", summary.get("total", 0))),
            "invalid_count": int(summary.get("invalid_count", 0)),
            "invalid_reasons": dict(summary.get("invalid_reasons", {}) or {}),
        }
    return compact


def _save_training_state(
    *,
    model,
    output_dir: Path,
    artifact_dir: Path,
    args,
    history,
    best_val_top1: float,
    best_val_monitor: float,
    best_state,
    base_config,
    xtb_cache_dir: Path,
    xtb_validity_summary: dict[str, object],
    split_mode: str,
    split_summary: dict[str, object],
    episode_log_path: Path | None = None,
    test_metrics=None,
    status: str = "running",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = output_dir / "hybrid_full_xtb_latest.pt"
    best_path = output_dir / "hybrid_full_xtb_best.pt"
    archive_path = output_dir / f"hybrid_full_xtb_{timestamp}.pt"
    report_path = artifact_dir / f"hybrid_full_xtb_report_{timestamp}.json"
    effective_split_summary = _effective_split_summary(split_summary)
    history_len = int(len(history))
    final_epoch = int(history[-1]["epoch"]) if history else 0
    best_site_top1_entry = _best_history_entry(history, "site_top1_acc")
    best_monitor_entry = _best_history_entry(history, args.early_stopping_metric)
    best_epoch = int((best_site_top1_entry or {}).get("epoch") or 0)
    best_monitor_epoch = int((best_monitor_entry or {}).get("epoch") or 0)
    last_train_metrics = dict(history[-1].get("train") or {}) if history else {}
    last_val_metrics = dict(history[-1].get("val") or {}) if history else {}
    best_val_metrics = dict((best_site_top1_entry or {}).get("val") or {})
    best_train_metrics = dict((best_site_top1_entry or {}).get("train") or {})
    nexus_diagnosis = _nexus_diagnosis(history)
    checkpoint = {
        "model_state_dict": _initialized_state_dict(model),
        "config": {
            "base_model": base_config.__dict__,
            "hybrid_wrapper": {"prior_weight": float(torch.sigmoid(model.prior_weight_logit).detach().item())},
        },
        "training_config": TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
        ).__dict__,
        "best_val_top1": best_val_top1,
        "best_val_monitor": best_val_monitor,
        "best_epoch": best_epoch,
        "best_monitor_epoch": best_monitor_epoch,
        "history_len": history_len,
        "final_epoch": final_epoch,
        "early_stopping_metric": args.early_stopping_metric,
        "test_metrics": test_metrics,
        "history": history,
        "xtb_feature_dim": FULL_XTB_FEATURE_DIM,
        "xtb_cache_dir": str(xtb_cache_dir),
        "xtb_validity": xtb_validity_summary,
        "status": status,
        "split_mode": split_mode,
        "target_cyp": str(getattr(args, "target_cyp", "") or ""),
        "confidence_allowlist": _parse_csv_tokens(str(getattr(args, "confidence_allowlist", "") or "")),
        "train_source_allowlist": _parse_csv_tokens(str(getattr(args, "train_source_allowlist", "") or "")),
        "base_lnn_first": bool(getattr(args, "base_lnn_first", False)),
        "nexus_sideinfo_only": bool(getattr(args, "nexus_sideinfo_only", False)),
        "use_candidate_mask": bool(getattr(args, "use_candidate_mask", False)),
        "candidate_mask_mode": str(getattr(args, "candidate_mask_mode", "hard") or "hard"),
        "candidate_mask_logit_bias": float(getattr(args, "candidate_mask_logit_bias", 2.0)),
        "balance_train_sources": bool(getattr(args, "balance_train_sources", False)),
        "freeze_base_modules": _parse_csv_tokens(str(getattr(args, "freeze_base_modules", "") or "")),
        "backbone_thaw_lr_scale": float(getattr(args, "backbone_thaw_lr_scale", 0.1)),
        "site_only_target_cyp": bool(getattr(args, "site_only_target_cyp", False)),
        "split_summary": split_summary,
        "effective_split_summary": effective_split_summary,
        "last_train_metrics": last_train_metrics,
        "last_val_metrics": last_val_metrics,
        "best_val_metrics": best_val_metrics,
        "best_train_metrics": best_train_metrics,
        "nexus_diagnosis": nexus_diagnosis,
    }
    torch.save(checkpoint, latest_path)
    torch.save(checkpoint, archive_path)
    if best_state is not None:
        best_checkpoint = dict(checkpoint)
        best_checkpoint["model_state_dict"] = best_state
        best_checkpoint["status"] = f"{status}_best"
        torch.save(best_checkpoint, best_path)
    report_path.write_text(
        json.dumps(
            {
                "status": status,
                "best_val_top1": best_val_top1,
                "best_val_monitor": best_val_monitor,
                "best_epoch": best_epoch,
                "best_monitor_epoch": best_monitor_epoch,
                "history_len": history_len,
                "final_epoch": final_epoch,
                "early_stopping_metric": args.early_stopping_metric,
                "test_metrics": test_metrics,
                "xtb_feature_dim": FULL_XTB_FEATURE_DIM,
                "xtb_validity": xtb_validity_summary,
                "split_mode": split_mode,
                "target_cyp": str(getattr(args, "target_cyp", "") or ""),
                "confidence_allowlist": _parse_csv_tokens(str(getattr(args, "confidence_allowlist", "") or "")),
                "train_source_allowlist": _parse_csv_tokens(str(getattr(args, "train_source_allowlist", "") or "")),
                "base_lnn_first": bool(getattr(args, "base_lnn_first", False)),
                "nexus_sideinfo_only": bool(getattr(args, "nexus_sideinfo_only", False)),
                "use_candidate_mask": bool(getattr(args, "use_candidate_mask", False)),
                "candidate_mask_mode": str(getattr(args, "candidate_mask_mode", "hard") or "hard"),
                "candidate_mask_logit_bias": float(getattr(args, "candidate_mask_logit_bias", 2.0)),
                "balance_train_sources": bool(getattr(args, "balance_train_sources", False)),
                "freeze_base_modules": _parse_csv_tokens(str(getattr(args, "freeze_base_modules", "") or "")),
                "backbone_thaw_lr_scale": float(getattr(args, "backbone_thaw_lr_scale", 0.1)),
                "site_only_target_cyp": bool(getattr(args, "site_only_target_cyp", False)),
                "split_summary": split_summary,
                "effective_split_summary": effective_split_summary,
                "episode_log_path": str(episode_log_path) if episode_log_path is not None else None,
                "last_train_metrics": last_train_metrics,
                "last_val_metrics": last_val_metrics,
                "best_val_metrics": best_val_metrics,
                "best_train_metrics": best_train_metrics,
                "nexus_diagnosis": nexus_diagnosis,
                "history": history,
            },
            indent=2,
        )
    )
    return latest_path, best_path, archive_path, report_path


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Train hybrid model with full xTB manual priors")
    parser.add_argument("--dataset", default="data/training_dataset_drugbank.json")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--checkpoint", default="checkpoints/hybrid_lnn_latest.pt")
    parser.add_argument("--xtb-cache-dir", default="cache/full_xtb")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--manual-target-bond", default=None)
    parser.add_argument("--manual-feature-cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-metric", choices=("site_top1", "site_top3"), default="site_top3")
    parser.add_argument("--output-dir", default="checkpoints/hybrid_full_xtb")
    parser.add_argument("--artifact-dir", default="artifacts/hybrid_full_xtb")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--split-mode",
        choices=("random", "scaffold_source", "scaffold_source_size"),
        default="scaffold_source_size",
    )
    parser.add_argument("--site-labeled-only", action="store_true")
    parser.add_argument("--compute-xtb-if-missing", action="store_true")
    parser.add_argument("--disable-nexus-bridge", action="store_true")
    parser.add_argument("--base-lnn-first", action="store_true")
    parser.add_argument("--nexus-sideinfo-only", action="store_true")
    parser.add_argument("--freeze-nexus-memory", action="store_true")
    parser.add_argument("--skip-nexus-memory-rebuild", action="store_true")
    parser.add_argument("--backbone-freeze-epochs", type=int, default=0,
                        help="Freeze base_lnn backbone for this many epochs; only train hybrid heads."
                             " After thaw, backbone trains at 0.1x LR via a separate param group.")
    parser.add_argument("--xenosite-manifest", default="")
    parser.add_argument("--xenosite-topk", type=int, default=1)
    parser.add_argument("--xenosite-per-file-limit", type=int, default=0)
    parser.add_argument("--episode-log", default="")
    parser.add_argument("--disable-episode-log", action="store_true")
    parser.add_argument("--precedent-logbook", default="")
    parser.add_argument("--disable-precedent-logbook", action="store_true")
    parser.add_argument("--target-cyp", default="")
    parser.add_argument("--confidence-allowlist", default="")
    parser.add_argument("--use-candidate-mask", action="store_true")
    parser.add_argument("--candidate-mask-mode", default="hard")
    parser.add_argument("--candidate-mask-logit-bias", type=float, default=2.0)
    parser.add_argument("--balance-train-sources", action="store_true")
    parser.add_argument("--train-source-allowlist", default="")
    parser.add_argument("--freeze-base-modules", default="")
    parser.add_argument("--backbone-thaw-lr-scale", type=float, default=0.1)
    parser.add_argument("--site-only-target-cyp", action="store_true")
    args = parser.parse_args()
    freeze_base_modules = _parse_csv_tokens(args.freeze_base_modules)
    early_stopping_patience = int(args.early_stopping_patience)
    early_stopping_enabled = early_stopping_patience > 0

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    artifact_dir = Path(args.artifact_dir)
    xtb_cache_dir = Path(args.xtb_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    xtb_cache_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    episode_log_path = (
        None
        if args.disable_episode_log
        else Path(args.episode_log) if args.episode_log else artifact_dir / f"hybrid_full_xtb_episode_log_{timestamp}.jsonl"
    )
    episode_logger = EpisodeLogger(episode_log_path, run_id=timestamp) if episode_log_path is not None else None

    print("=" * 60, flush=True)
    print("HYBRID LNN: FULL XTB MANUAL PRIORS", flush=True)
    print("=" * 60, flush=True)
    print(f"Using device: {device}", flush=True)
    if episode_log_path is not None:
        print(f"Episode log: {episode_log_path}", flush=True)

    drugs = _load_drugs(dataset_path)
    print(f"Loaded {len(drugs)} drugs", flush=True)
    if args.nexus_sideinfo_only:
        args.disable_nexus_bridge = False
        args.base_lnn_first = False
        print("nexus_sideinfo_only=1 | NEXUS enabled as feature sidecar only", flush=True)
    if args.base_lnn_first:
        args.disable_nexus_bridge = True
        args.freeze_nexus_memory = True
        args.skip_nexus_memory_rebuild = True
        print("base_lnn_first=1 | NEXUS bridge disabled for this run", flush=True)
    if str(args.target_cyp or "").strip():
        target_cyp = str(args.target_cyp).strip()
        drugs = [drug for drug in drugs if _primary_cyp(drug) == target_cyp]
        print(f"Filtered target_cyp={target_cyp}: {len(drugs)}", flush=True)
        if target_cyp.upper() == "CYP3A4" and args.base_lnn_first and not args.use_candidate_mask:
            args.use_candidate_mask = True
            print("Auto-enabled CYP3A4 candidate masking for base_lnn_first run", flush=True)
    confidence_allowlist = _parse_csv_tokens(args.confidence_allowlist)
    if confidence_allowlist:
        allowed = {token.lower() for token in confidence_allowlist}
        drugs = [drug for drug in drugs if str(drug.get("confidence") or "").strip().lower() in allowed]
        print(f"Filtered confidence_allowlist={confidence_allowlist}: {len(drugs)}", flush=True)
    if args.site_labeled_only:
        drugs = [drug for drug in drugs if _has_site_labels(drug)]
        print(f"Site-labeled: {len(drugs)}", flush=True)
    if args.limit is not None:
        drugs = drugs[: int(args.limit)]
        print(f"Limited to: {len(drugs)}", flush=True)
    if not drugs:
        raise RuntimeError("No training drugs remain after target/confidence/site filters")

    train_drugs, val_drugs, test_drugs = split_drugs(
        drugs,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        mode=args.split_mode,
    )
    print(f"Split mode: {args.split_mode}", flush=True)
    train_source_allowlist = _parse_csv_tokens(args.train_source_allowlist)
    if train_source_allowlist:
        train_drugs = _filter_by_sources(train_drugs, train_source_allowlist)
        print(
            f"Filtered train_source_allowlist={train_source_allowlist}: {len(train_drugs)}",
            flush=True,
        )
        if not train_drugs:
            raise RuntimeError("No train drugs remain after train_source_allowlist filter")
    xenosite_added = 0
    if args.xenosite_manifest:
        manifest_path = Path(args.xenosite_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"XenoSite manifest not found: {manifest_path}")
        xenosite_entries = _load_xenosite_aux_entries(
            manifest_path,
            topk=args.xenosite_topk,
            per_file_limit=args.xenosite_per_file_limit,
        )
        if xenosite_entries:
            existing = {_canonical_smiles_key(d.get("smiles", "")) for d in train_drugs}
            xenosite_entries = [d for d in xenosite_entries if _canonical_smiles_key(d.get("smiles", "")) not in existing]
            train_drugs.extend(xenosite_entries)
            xenosite_added = len(xenosite_entries)
            print(
                f"Added XenoSite auxiliary train entries: {xenosite_added} "
                f"(topk={max(1, int(args.xenosite_topk))})",
                flush=True,
            )
    split_summary = {
        "train": _split_summary(train_drugs),
        "val": _split_summary(val_drugs),
        "test": _split_summary(test_drugs),
    }
    for split_name, split_items in (("train", train_drugs), ("val", val_drugs), ("test", test_drugs)):
        summary = split_summary[split_name]
        print(
            f"{split_name}: total={summary['total']} | site_supervised={summary['site_supervised']} | "
            f"sources={summary['sources']} | atom_buckets={summary['atom_buckets']} | "
            f"site_count_buckets={summary['site_count_buckets']} | near_duplicates={summary['near_duplicates']}",
            flush=True,
        )
    if args.use_candidate_mask:
        print(
            "candidate_mask=1 | "
            f"candidate_cyp={str(args.target_cyp or '').strip() or 'generic'} | "
            f"mode={str(args.candidate_mask_mode or 'hard').strip().lower() or 'hard'} | "
            f"logit_bias={float(args.candidate_mask_logit_bias):.3f}",
            flush=True,
        )
    if args.balance_train_sources:
        print("balance_train_sources=1", flush=True)
    if train_source_allowlist:
        print(f"train_source_allowlist={train_source_allowlist}", flush=True)
    if freeze_base_modules:
        print(f"freeze_base_modules={freeze_base_modules}", flush=True)
    if args.nexus_sideinfo_only:
        print("nexus_sideinfo_only=1 | side engines feed features into LNN without votes", flush=True)
    if args.site_only_target_cyp and str(args.target_cyp or "").strip():
        print(f"site_only_target_cyp=1 | disabling CYP task for {str(args.target_cyp).strip()}", flush=True)
    fixed_cyp_index = -1
    if args.site_only_target_cyp and str(args.target_cyp or "").strip():
        target = str(args.target_cyp).strip().upper()
        try:
            fixed_cyp_index = list(ModelConfig().cyp_names).index(target)
            print(f"fixed_cyp_context=1 | cyp={target} | cyp_index={fixed_cyp_index}", flush=True)
        except ValueError:
            print(f"fixed_cyp_context=0 | target_cyp={target} not in model cyp_names", flush=True)

    xtb_validity_summary = _summarize_xtb_validity(drugs, xtb_cache_dir)
    print(
        "xTB validity: "
        f"strict_true={xtb_validity_summary['strict_true_xtb_valid_molecules']}/{len(drugs)} | "
        f"training_usable={xtb_validity_summary['training_usable_xtb_valid_molecules']}/{len(drugs)} | "
        f"cached={xtb_validity_summary['cached_xtb_valid_molecules']}/{len(drugs)} | "
        f"statuses={xtb_validity_summary['statuses']}",
        flush=True,
    )

    (train_loader, val_loader, test_loader), manual_engine_enabled = _build_loaders_with_fallback(
        train_drugs,
        val_drugs,
        test_drugs,
        args=args,
    )
    split_summary = _attach_effective_split_summary(
        split_summary,
        {"train": train_loader, "val": val_loader, "test": test_loader},
    )
    for split_name in ("train", "val", "test"):
        summary = split_summary[split_name]
        print(
            f"{split_name} effective: total={summary.get('effective_total', summary.get('total'))} | "
            f"invalid={summary.get('invalid_count', 0)} | invalid_reasons={summary.get('invalid_reasons', {})}",
            flush=True,
        )

    manual_atom_feature_dim = (32 if manual_engine_enabled else 0) + FULL_XTB_FEATURE_DIM
    # Step 1 atom_input_dim = 146 = 140 base graph features + 6 standard XTB dims.
    # Step 2 appends FULL_XTB_FEATURE_DIM (8) instead of 6, so atom_input_dim = 140 + 8 = 148.
    _BASE_GRAPH_ATOM_DIM = 140
    full_xtb_atom_input_dim = _BASE_GRAPH_ATOM_DIM + FULL_XTB_FEATURE_DIM
    live_wave_vote_inputs = _env_flag("HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS", "1")
    live_analogical_vote_inputs = _env_flag("HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS", "1")
    model_overrides = _collect_model_overrides()
    base_config = ModelConfig.light_advanced(
        use_manual_engine_priors=manual_engine_enabled,
        use_3d_branch=True,
        use_nexus_bridge=not bool(args.disable_nexus_bridge),
        use_nexus_site_arbiter=not bool(args.nexus_sideinfo_only),
        use_nexus_sideinfo_features=bool(args.nexus_sideinfo_only),
        use_cyp_site_conditioning=not bool(args.site_only_target_cyp and str(args.target_cyp or "").strip()),
        disable_cyp_task=bool(args.site_only_target_cyp and str(args.target_cyp or "").strip()),
        fixed_cyp_index=int(fixed_cyp_index),
        candidate_mask_mode=str(args.candidate_mask_mode or "hard").strip().lower() or "hard",
        candidate_mask_logit_bias=float(args.candidate_mask_logit_bias),
        nexus_memory_frozen=bool(args.freeze_nexus_memory),
        nexus_rebuild_memory_before_train=not bool(args.skip_nexus_memory_rebuild),
        return_intermediate_stats=True,
        manual_atom_feature_dim=manual_atom_feature_dim,
        atom_input_dim=full_xtb_atom_input_dim,
        nexus_live_wave_vote_inputs=live_wave_vote_inputs,
        nexus_live_analogical_vote_inputs=live_analogical_vote_inputs,
        **model_overrides,
    )
    base_model = LiquidMetabolismNetV2(base_config)
    model = HybridLNNModel(base_model)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        load_report = load_full_xtb_warm_start(
            model,
            checkpoint_path,
            device=device,
            new_manual_atom_dim=manual_atom_feature_dim,
            new_atom_input_dim=full_xtb_atom_input_dim,
        )
        print(f"Loaded warm-start checkpoint: {checkpoint_path}", flush=True)
        print(
            "Warm-start load summary: "
            f"loaded={load_report.get('loaded', 0)} "
            f"missing={load_report.get('missing', 0)} "
            f"mismatch={load_report.get('mismatch', 0)} "
            f"nonfinite={load_report.get('nonfinite', 0)}",
            flush=True,
        )
    else:
        print(f"No warm-start checkpoint found at {checkpoint_path}; starting from current initialization", flush=True)

    precedent_logbook = None if args.disable_precedent_logbook else _resolve_precedent_logbook(args.precedent_logbook, artifact_dir)
    if precedent_logbook is not None and precedent_logbook.exists():
        precedent_stats = model.load_nexus_precedent_logbook(
            str(precedent_logbook),
            cyp_names=list(getattr(base_config, "cyp_names", ())),
        )
        print(
            f"Loaded precedent logbook: {precedent_logbook} | "
            f"cases={int(precedent_stats.get('cases', 0.0))} "
            f"episodes={int(precedent_stats.get('episodes', 0.0))}",
            flush=True,
        )
    else:
        if args.disable_precedent_logbook:
            print("Precedent logbook loading disabled; analogical precedent briefs will remain empty for this run", flush=True)
        else:
            print(
                "No explicit precedent logbook provided; analogical precedent briefs will remain empty for this run",
                flush=True,
            )

    model.to(device)

    if (
        getattr(base_config, "use_nexus_bridge", False)
        and getattr(base_config, "nexus_rebuild_memory_before_train", False)
        and getattr(model, "nexus_bridge", None) is not None
    ):
        memory_stats = model.rebuild_nexus_memory(train_loader, device=device)
        print(
            f"Built NEXUS memory: size={int(memory_stats.get('memory_size', 0.0))} "
            f"from_batches={int(memory_stats.get('batches', 0.0))} "
            f"frozen={'yes' if base_config.nexus_memory_frozen else 'no'}",
            flush=True,
        )
    print(
        f"Live sidecar vote inputs: wave={'yes' if live_wave_vote_inputs else 'no'} "
        f"analogical={'yes' if live_analogical_vote_inputs else 'no'}",
        flush=True,
    )
    if model_overrides:
        print(f"Model overrides: {model_overrides}", flush=True)

    trainer = Trainer(
        model=model,
        config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping_patience=max(0, early_stopping_patience),
        ),
        device=device,
        episode_logger=episode_logger,
    )

    history = []
    best_val_top1 = -1.0
    best_val_monitor = -1.0
    best_state = None
    epochs_without_improvement = 0
    train_start = time.perf_counter()
    backbone_freeze_epochs = max(0, int(args.backbone_freeze_epochs))
    backbone_thaw_lr_scale = min(max(float(args.backbone_thaw_lr_scale), 0.0), 1.0)
    _backbone_frozen = False

    def _resolve_base_predictor():
        base = getattr(model, "base_lnn", None) or getattr(model, "_base_lnn", None)
        if base is None:
            wrapper = getattr(model, "nexus_wrapper", None) or model
            base = getattr(wrapper, "base_lnn", None)
        return getattr(base, "impl", base)

    base_predictor = _resolve_base_predictor()
    frozen_named_modules: list[tuple[str, object]] = []
    if bool(args.site_only_target_cyp and str(args.target_cyp or "").strip()):
        for module_name in ("cyp_branch", "cyp_head"):
            if module_name not in freeze_base_modules:
                freeze_base_modules.append(module_name)
    if base_predictor is not None and freeze_base_modules:
        available_modules = dict(base_predictor.named_children())
        for module_name in freeze_base_modules:
            module = available_modules.get(module_name)
            if module is not None:
                frozen_named_modules.append((module_name, module))
        if frozen_named_modules:
            for _, module in frozen_named_modules:
                for param in module.parameters():
                    param.requires_grad = False
            print(
                "Frozen base modules: " + ",".join(name for name, _ in frozen_named_modules),
                flush=True,
            )
        else:
            print(
                "Requested freeze_base_modules but no matching base modules were found; "
                f"available={sorted(available_modules)}",
                flush=True,
            )

    def _set_backbone_frozen(frozen: bool) -> None:
        """Freeze or unfreeze base_lnn backbone. When unfreezing, add a low-LR param group."""
        base = getattr(model, "base_lnn", None) or getattr(model, "_base_lnn", None)
        if base is None:
            wrapper = getattr(model, "nexus_wrapper", None) or model
            base = getattr(wrapper, "base_lnn", None)
        if base is None:
            return
        for param in base.parameters():
            param.requires_grad = not frozen
        for _, module in frozen_named_modules:
            for param in module.parameters():
                param.requires_grad = False
        if not frozen:
            # Add backbone params as a separate lower-LR group if not already added
            backbone_params = [p for p in base.parameters() if p.requires_grad]
            existing_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
            new_backbone = [p for p in backbone_params if id(p) not in existing_ids]
            if new_backbone:
                backbone_lr = args.learning_rate * backbone_thaw_lr_scale
                # Copy betas/eps from first group so _ManualAdamW.step() doesn't KeyError
                ref_group = trainer.optimizer.param_groups[0]
                trainer.optimizer.param_groups.append({
                    "params": new_backbone,
                    "lr": backbone_lr,
                    "weight_decay": args.weight_decay,
                    "betas": ref_group.get("betas", (0.9, 0.999)),
                    "eps": ref_group.get("eps", 1e-8),
                })
                print(f"Backbone unfrozen: added {len(new_backbone)} params at lr={backbone_lr:.2e}", flush=True)

    if backbone_freeze_epochs > 0:
        _set_backbone_frozen(True)
        _backbone_frozen = True
        print(f"Backbone frozen for first {backbone_freeze_epochs} epochs.", flush=True)

    try:
        for epoch in range(args.epochs):
            if _backbone_frozen and epoch >= backbone_freeze_epochs:
                _set_backbone_frozen(False)
                _backbone_frozen = False
                print(f"Epoch {epoch + 1}: backbone unfrozen (thaw at 0.1x LR).", flush=True)
            epoch_start = time.perf_counter()
            setattr(train_loader, "_current_epoch", epoch)
            setattr(train_loader, "_split_name", "train")
            train_stats = trainer.train_loader_epoch(train_loader)

            # Refresh analogical memory after each epoch so every stored key is
            # encoded by the same up-to-date network (not a mid-epoch mix).
            if (
                getattr(model, "nexus_bridge", None) is not None
                and not getattr(base_config, "nexus_memory_frozen", False)
                and getattr(model, "refresh_nexus_memory", None) is not None
            ):
                try:
                    _ingested = model.refresh_nexus_memory(train_loader, device=device)
                    print(f"  [memory refreshed: {_ingested} atoms in buffer]", flush=True)
                except Exception as _mem_err:
                    print(f"  [memory refresh skipped: {_mem_err}]", flush=True)

            setattr(val_loader, "_current_epoch", epoch)
            setattr(val_loader, "_split_name", "val")
            val_metrics = trainer.evaluate_loader(val_loader)
            epoch_seconds = time.perf_counter() - epoch_start
            elapsed_seconds = time.perf_counter() - train_start
            history.append({"epoch": epoch + 1, "train": train_stats, "val": val_metrics})

            val_top1 = float(val_metrics.get("site_top1_acc", 0.0))
            val_top3 = float(val_metrics.get("site_top3_acc", 0.0))
            monitor_name = args.early_stopping_metric
            monitor_value = val_top1 if monitor_name == "site_top1" else val_top3
            trainer.step_scheduler(monitor_value)
            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
            if monitor_value > best_val_monitor:
                best_val_monitor = monitor_value
                best_state = _initialized_state_dict(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % max(1, int(args.log_every)) == 0 or epoch == 0:
                avg_epoch_seconds = elapsed_seconds / float(epoch + 1)
                eta_seconds = avg_epoch_seconds * max(0, args.epochs - (epoch + 1))
                print(
                    f"Epoch {epoch + 1:3d} | loss={train_stats.get('total_loss', float('nan')):.4f} | "
                    f"site_loss={train_stats.get('site_loss', float('nan')):.4f} | "
                    f"cyp_loss={train_stats.get('cyp_loss', float('nan')):.4f} | "
                    f"site_top1={val_metrics.get('site_top1_acc', 0.0):.3f} | "
                    f"site_top3={val_metrics.get('site_top3_acc', 0.0):.3f} | "
                    f"cyp_acc={val_metrics.get('accuracy', 0.0):.3f} | "
                    f"cyp_f1={val_metrics.get('f1_macro', 0.0):.3f} | "
                    f"physics_gate={train_stats.get('physics_gate_mean', 0.0):.3f} | "
                    f"epoch_time={epoch_seconds:.1f}s | "
                    f"elapsed={elapsed_seconds / 60.0:.1f}m | "
                    f"eta={eta_seconds / 60.0:.1f}m",
                    flush=True,
                )

            latest_path, best_path, _, report_path = _save_training_state(
                model=model,
                output_dir=output_dir,
                artifact_dir=artifact_dir,
                args=args,
                history=history,
                best_val_top1=best_val_top1,
                best_val_monitor=best_val_monitor,
                best_state=best_state,
                base_config=base_config,
                xtb_cache_dir=xtb_cache_dir,
                xtb_validity_summary=xtb_validity_summary,
                split_mode=args.split_mode,
                split_summary=split_summary,
                episode_log_path=episode_log_path,
                test_metrics=None,
                status="running",
            )

            if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping after epoch {epoch + 1}: no {monitor_name} improvement for "
                    f"{early_stopping_patience} epochs.",
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Saving current hybrid_full_xtb progress...", flush=True)
        latest_path, best_path, _, report_path = _save_training_state(
            model=model,
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            args=args,
            history=history,
            best_val_top1=best_val_top1,
            best_val_monitor=best_val_monitor,
            best_state=best_state,
            base_config=base_config,
            xtb_cache_dir=xtb_cache_dir,
            xtb_validity_summary=xtb_validity_summary,
            split_mode=args.split_mode,
            split_summary=split_summary,
            episode_log_path=episode_log_path,
            test_metrics=None,
            status="interrupted",
        )
        print(f"Saved latest checkpoint: {latest_path}", flush=True)
        print(f"Saved best checkpoint: {best_path}", flush=True)
        print(f"Saved report: {report_path}", flush=True)
        return

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    print("\n" + "=" * 60, flush=True)
    print("TEST SET EVALUATION", flush=True)
    print("=" * 60, flush=True)
    setattr(test_loader, "_current_epoch", max(0, len(history) - 1))
    setattr(test_loader, "_split_name", "test")
    test_metrics = trainer.evaluate_loader(test_loader)
    print(json.dumps(test_metrics, indent=2), flush=True)

    latest_path, best_path, archive_path, report_path = _save_training_state(
        model=model,
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        args=args,
        history=history,
        best_val_top1=best_val_top1,
        best_val_monitor=best_val_monitor,
        best_state=best_state,
        base_config=base_config,
        xtb_cache_dir=xtb_cache_dir,
        xtb_validity_summary=xtb_validity_summary,
        split_mode=args.split_mode,
        split_summary=split_summary,
        episode_log_path=episode_log_path,
        test_metrics=test_metrics,
        status="completed",
    )
    print(f"\nSaved checkpoint: {archive_path}", flush=True)
    print(f"Saved latest checkpoint: {latest_path}", flush=True)
    print(f"Saved best checkpoint: {best_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
