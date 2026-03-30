from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, torch


def _to_serializable(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if TORCH_AVAILABLE and hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return float(value)
    except Exception:
        return str(value)


class EpisodeLogger:
    def __init__(self, path: str | Path, *, run_id: Optional[str] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

    def _write(self, record: Dict[str, object]) -> None:
        payload = dict(record)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("logged_at_utc", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def log_step(
        self,
        *,
        split: str,
        epoch: Optional[int],
        batch_idx: int,
        batch,
        stats: Optional[Dict[str, float]] = None,
        outputs: Optional[Dict[str, object]] = None,
    ) -> None:
        record = {
            "record_type": "step",
            "split": str(split),
            "epoch": None if epoch is None else int(epoch) + 1,
            "batch_idx": int(batch_idx),
            "num_graphs": int(batch.get("num_graphs", 0)),
            "graph_names": list(batch.get("graph_names", [])),
            "graph_smiles": list(batch.get("canonical_smiles", [])),
            "stats": _to_serializable(stats or {}),
        }
        if outputs is not None:
            diagnostics = outputs.get("diagnostics") or {}
            record["diagnostics"] = _to_serializable(diagnostics)
        self._write(record)

    def log_examples(
        self,
        *,
        split: str,
        epoch: Optional[int],
        batch_idx: int,
        batch,
        outputs: Dict[str, object],
        stats: Optional[Dict[str, float]] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            return
        graph_num_atoms = [int(v) for v in list(batch.get("graph_num_atoms", []))]
        if not graph_num_atoms:
            return
        batch_metadata = list(batch.get("graph_metadata", []))
        graph_names = list(batch.get("graph_names", []))
        graph_confidences = list(batch.get("graph_confidences", []))
        canonical_smiles = list(batch.get("canonical_smiles", []))
        parsing_status = list(batch.get("parsing_status", []))
        repaired = list(batch.get("repaired", []))
        aggressive_repair = list(batch.get("aggressive_repair", []))
        xtb_status = list(batch.get("xtb_feature_status", []))

        site_logits = outputs["site_logits"].detach().cpu().view(-1)
        site_scores = outputs.get("site_scores")
        if site_scores is None:
            site_scores = torch.sigmoid(outputs["site_logits"])
        site_scores = site_scores.detach().cpu().view(-1)
        site_labels = batch.get("site_labels")
        site_labels_cpu = site_labels.detach().cpu().view(-1) if site_labels is not None else None
        site_mask = batch.get("site_supervision_mask")
        site_mask_cpu = (
            site_mask.detach().cpu().view(-1)
            if site_mask is not None
            else torch.ones_like(site_scores, dtype=torch.float32)
        )
        cyp_logits = outputs.get("cyp_logits")
        cyp_logits_cpu = cyp_logits.detach().cpu() if cyp_logits is not None else None
        cyp_probs_cpu = torch.softmax(cyp_logits_cpu, dim=-1) if cyp_logits_cpu is not None else None
        cyp_labels = batch.get("cyp_labels")
        cyp_labels_cpu = cyp_labels.detach().cpu() if cyp_labels is not None else None
        cyp_mask = batch.get("cyp_supervision_mask")
        cyp_mask_cpu = cyp_mask.detach().cpu() if cyp_mask is not None else None

        vote_heads = outputs.get("site_vote_heads") or {}
        vote_heads_cpu = {
            key: value.detach().cpu().view(-1)
            for key, value in vote_heads.items()
            if value is not None
        }

        bridge = outputs.get("nexus_bridge_outputs") or {}
        atom_multivectors = bridge.get("atom_multivectors")
        atom_multivectors_cpu = atom_multivectors.detach().cpu() if atom_multivectors is not None else None
        wave_predictions = bridge.get("wave_predictions") or {}
        wave_predictions_cpu = {
            key: value.detach().cpu()
            for key, value in wave_predictions.items()
            if value is not None
        }
        wave_field = bridge.get("wave_field") or {}
        wave_field_cpu = {
            key: value.detach().cpu()
            for key, value in wave_field.items()
            if value is not None
        }
        analogical_site_prior = bridge.get("analogical_site_prior")
        analogical_site_prior_cpu = analogical_site_prior.detach().cpu() if analogical_site_prior is not None else None
        analogical_cyp_prior = bridge.get("analogical_cyp_prior")
        analogical_cyp_prior_cpu = analogical_cyp_prior.detach().cpu() if analogical_cyp_prior is not None else None
        analogical_confidence = bridge.get("analogical_confidence")
        analogical_confidence_cpu = analogical_confidence.detach().cpu() if analogical_confidence is not None else None
        analogical_site_bias = bridge.get("analogical_site_bias")
        analogical_site_bias_cpu = analogical_site_bias.detach().cpu() if analogical_site_bias is not None else None
        analogical_cyp_bias = bridge.get("analogical_cyp_bias")
        analogical_cyp_bias_cpu = analogical_cyp_bias.detach().cpu() if analogical_cyp_bias is not None else None
        continuous_reasoning = bridge.get("continuous_reasoning_features")
        continuous_reasoning_cpu = continuous_reasoning.detach().cpu() if continuous_reasoning is not None else None
        bridge_metrics = _to_serializable(bridge.get("metrics") or {})

        offset = 0
        for graph_idx, num_atoms in enumerate(graph_num_atoms):
            start = offset
            end = offset + num_atoms
            offset = end

            meta = dict(batch_metadata[graph_idx]) if graph_idx < len(batch_metadata) else {}
            if "site_atoms" not in meta:
                meta["site_atoms"] = []
            predicted_order = torch.argsort(site_scores[start:end], descending=True)
            top5_count = min(5, int(num_atoms))
            top5 = [int(v) for v in predicted_order[:top5_count].tolist()]
            top3 = top5[: min(3, len(top5))]
            top1 = top5[0] if top5 else None
            true_site_atoms = list(meta.get("site_atoms", []))
            if site_labels_cpu is not None:
                true_site_atoms = [
                    int(local_idx)
                    for local_idx in range(num_atoms)
                    if float(site_mask_cpu[start + local_idx].item()) > 0.0
                    and float(site_labels_cpu[start + local_idx].item()) > 0.5
                ]
            true_site_set = set(true_site_atoms)
            top1_hit = bool(top1 in true_site_set) if top1 is not None else False
            top3_hit = bool(true_site_set.intersection(top3))
            top5_hit = bool(true_site_set.intersection(top5))

            record = {
                "record_type": "episode",
                "split": str(split),
                "epoch": None if epoch is None else int(epoch) + 1,
                "batch_idx": int(batch_idx),
                "graph_idx": int(graph_idx),
                "input": {
                    "id": meta.get("id", ""),
                    "name": meta.get("name", graph_names[graph_idx] if graph_idx < len(graph_names) else ""),
                    "smiles": meta.get(
                        "smiles",
                        canonical_smiles[graph_idx] if graph_idx < len(canonical_smiles) else "",
                    ),
                    "canonical_smiles": canonical_smiles[graph_idx] if graph_idx < len(canonical_smiles) else "",
                    "source": meta.get("source", ""),
                    "confidence": meta.get(
                        "confidence",
                        graph_confidences[graph_idx] if graph_idx < len(graph_confidences) else "",
                    ),
                    "primary_cyp": meta.get("primary_cyp", ""),
                    "all_cyps": meta.get("all_cyps", []),
                    "site_atoms": true_site_atoms,
                    "site_source": meta.get("site_source", ""),
                    "auxiliary_site_only": bool(meta.get("auxiliary_site_only", False)),
                    "num_atoms": int(num_atoms),
                    "parsing_status": parsing_status[graph_idx] if graph_idx < len(parsing_status) else "",
                    "repaired": bool(repaired[graph_idx]) if graph_idx < len(repaired) else False,
                    "aggressive_repair": bool(aggressive_repair[graph_idx]) if graph_idx < len(aggressive_repair) else False,
                    "xtb_feature_status": xtb_status[graph_idx] if graph_idx < len(xtb_status) else "",
                    "site_supervision": bool(true_site_atoms),
                    "cyp_supervision": bool(
                        cyp_mask_cpu is None or float(cyp_mask_cpu[graph_idx].item()) > 0.0
                    ),
                },
                "output": {
                    "site_logits": _to_serializable(site_logits[start:end]),
                    "site_scores": _to_serializable(site_scores[start:end]),
                    "cyp_logits": _to_serializable(cyp_logits_cpu[graph_idx]) if cyp_logits_cpu is not None else None,
                    "cyp_probs": _to_serializable(cyp_probs_cpu[graph_idx]) if cyp_probs_cpu is not None else None,
                    "true_cyp_label": int(cyp_labels_cpu[graph_idx].item()) if cyp_labels_cpu is not None else None,
                },
                "votes": {
                    "lnn_vote": _to_serializable(vote_heads_cpu.get("lnn_vote", site_logits)[start:end]),
                    "lnn_conf": _to_serializable(vote_heads_cpu.get("lnn_conf")[start:end]) if "lnn_conf" in vote_heads_cpu else None,
                    "wave_vote": _to_serializable(vote_heads_cpu.get("wave_vote")[start:end]) if "wave_vote" in vote_heads_cpu else None,
                    "wave_conf": _to_serializable(vote_heads_cpu.get("wave_conf")[start:end]) if "wave_conf" in vote_heads_cpu else None,
                    "analogical_vote": _to_serializable(vote_heads_cpu.get("analogical_vote")[start:end]) if "analogical_vote" in vote_heads_cpu else None,
                    "analogical_conf": _to_serializable(vote_heads_cpu.get("analogical_conf")[start:end]) if "analogical_conf" in vote_heads_cpu else None,
                    "council_logit": _to_serializable(vote_heads_cpu.get("council_logit")[start:end]) if "council_logit" in vote_heads_cpu else None,
                },
                "wave": {
                    "atom_multivectors": _to_serializable(atom_multivectors_cpu[start:end]) if atom_multivectors_cpu is not None else None,
                    "predicted_charges": _to_serializable(wave_predictions_cpu.get("predicted_charges")[start:end]) if "predicted_charges" in wave_predictions_cpu else None,
                    "predicted_fukui": _to_serializable(wave_predictions_cpu.get("predicted_fukui")[start:end]) if "predicted_fukui" in wave_predictions_cpu else None,
                    "predicted_gap": _to_serializable(wave_predictions_cpu.get("predicted_gap")[graph_idx]) if "predicted_gap" in wave_predictions_cpu else None,
                    "atom_field_features": _to_serializable(wave_field_cpu.get("atom_field_features")[start:end]) if "atom_field_features" in wave_field_cpu else None,
                    "global_density": _to_serializable(wave_field_cpu.get("global_density")[graph_idx]) if "global_density" in wave_field_cpu else None,
                    "global_gap_proxy": _to_serializable(wave_field_cpu.get("global_gap_proxy")[graph_idx]) if "global_gap_proxy" in wave_field_cpu else None,
                },
                "analogical": {
                    "site_prior": _to_serializable(analogical_site_prior_cpu[start:end]) if analogical_site_prior_cpu is not None else None,
                    "site_bias": _to_serializable(analogical_site_bias_cpu[start:end]) if analogical_site_bias_cpu is not None else None,
                    "confidence": _to_serializable(analogical_confidence_cpu[start:end]) if analogical_confidence_cpu is not None else None,
                    "continuous_reasoning_features": _to_serializable(continuous_reasoning_cpu[start:end]) if continuous_reasoning_cpu is not None else None,
                    "cyp_prior": _to_serializable(analogical_cyp_prior_cpu[graph_idx]) if analogical_cyp_prior_cpu is not None else None,
                    "cyp_bias": _to_serializable(analogical_cyp_bias_cpu[graph_idx]) if analogical_cyp_bias_cpu is not None else None,
                    "bridge_metrics": bridge_metrics,
                },
                "decision": {
                    "top1_atom": top1,
                    "top3_atoms": top3,
                    "top5_atoms": top5,
                    "top1_score": float(site_scores[start + top1].item()) if top1 is not None else None,
                    "predicted_cyp_idx": int(torch.argmax(cyp_logits_cpu[graph_idx]).item()) if cyp_logits_cpu is not None else None,
                    "predicted_cyp_prob": float(torch.max(cyp_probs_cpu[graph_idx]).item()) if cyp_probs_cpu is not None else None,
                },
                "outcome": {
                    "true_site_atoms": true_site_atoms,
                    "top1_hit": top1_hit,
                    "top3_hit": top3_hit,
                    "top5_hit": top5_hit,
                    "positive_site_count": int(len(true_site_atoms)),
                },
                "step_stats": _to_serializable(stats or {}),
            }
            self._write(record)
