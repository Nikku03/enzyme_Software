from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import DataLoader, Dataset

from enzyme_software.cahml.components.chemistry_encoder import ChemistryFeatureExtractor
from enzyme_software.cahml.config import CAHMLConfig, REACTION_TYPE_TO_INDEX, REACTION_TYPES
from enzyme_software.cahml.evaluator import evaluate_cahml_predictions
from enzyme_software.losses import CombinedSiteRankingLoss, SiteRankingLossV2
from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES


def _first_present_site_list(drug: Dict[str, object]) -> List[int]:
    for key in ("site_atoms", "site_atom_indices", "metabolism_sites"):
        raw = drug.get(key)
        if isinstance(raw, list) and raw:
            out = []
            for item in raw:
                if isinstance(item, int):
                    out.append(int(item))
                elif isinstance(item, dict):
                    atom_idx = item.get("atom_index", item.get("index", item.get("atom_idx", -1)))
                    if isinstance(atom_idx, int) and atom_idx >= 0:
                        out.append(int(atom_idx))
            if out:
                return out
    som = drug.get("som")
    if isinstance(som, list) and som:
        out = []
        for item in som:
            if isinstance(item, dict) and item.get("atom_idx") is not None:
                out.append(int(item["atom_idx"]))
        if out:
            return out
    return []


if TORCH_AVAILABLE:
    class CAHMLDataset(Dataset):
        def __init__(self, predictions_path: str | Path, drugs: List[Dict[str, object]]):
            payload = torch.load(predictions_path, map_location="cpu", weights_only=False)
            self.predictions = payload.get("predictions") or payload
            self.drugs = list(drugs)
            self.extractor = ChemistryFeatureExtractor()
            self._features: Dict[str, object] = {}
            self.rows: List[Dict[str, object]] = []
            for drug in self.drugs:
                smiles = str(drug.get("smiles", "")).strip()
                if not smiles or smiles not in self.predictions:
                    continue
                chemistry = self.extractor.extract(smiles)
                if chemistry is None:
                    continue
                self._features[smiles] = chemistry
                self.rows.append(drug)

        def __len__(self) -> int:
            return len(self.rows)

        @staticmethod
        def _align_vector(values: torch.Tensor, size: int) -> torch.Tensor:
            values = values.view(-1).float()
            if values.shape[0] == size:
                return values
            out = torch.zeros((size,), dtype=torch.float32)
            out[: min(size, values.shape[0])] = values[: min(size, values.shape[0])]
            return out

        @staticmethod
        def _align_matrix(values: torch.Tensor, rows: int, cols: Optional[int] = None) -> torch.Tensor:
            matrix = values.float()
            if matrix.ndim == 1:
                matrix = matrix.unsqueeze(-1)
            target_cols = cols or matrix.shape[1]
            out = torch.zeros((rows, target_cols), dtype=torch.float32)
            out[: min(rows, matrix.shape[0]), : min(target_cols, matrix.shape[1])] = matrix[: min(rows, matrix.shape[0]), : min(target_cols, matrix.shape[1])]
            return out

        def _infer_cyp_label(self, drug: Dict[str, object], record: Dict[str, object]) -> int:
            cyp_label = record.get("cyp_label")
            if cyp_label is not None:
                return int(cyp_label)
            cyp = str(drug.get("primary_cyp") or drug.get("cyp") or "")
            return MAJOR_CYP_CLASSES.index(cyp) if cyp in MAJOR_CYP_CLASSES else 0

        def _infer_reaction_label(self, chemistry, site_labels: torch.Tensor) -> int:
            positives = torch.nonzero(site_labels > 0.5, as_tuple=False).view(-1).tolist()
            if not positives:
                return -1
            for idx in positives:
                if idx >= len(chemistry.site_types):
                    continue
                site_type = chemistry.site_types[idx]
                atomic_num = int(round(float(chemistry.atom_features_raw[idx, 0].item() * 20.0)))
                is_aromatic = bool(chemistry.atom_features_raw[idx, 1].item() > 0.5)
                # Specific demethylation handles — highest priority
                if site_type == "o_methyl_aromatic":
                    return REACTION_TYPE_TO_INDEX["o_demethylation"]
                if site_type == "n_methyl":
                    return REACTION_TYPE_TO_INDEX["n_demethylation"]
                # Heteroatom oxidation — differentiated by atom type and site pattern
                if site_type in ("s_oxidation", "thiophene_s") or atomic_num == 16:
                    return REACTION_TYPE_TO_INDEX["s_oxidation"]
                if site_type in ("n_oxidation", "ring_nitrogen_6", "primary_aro_amine") or (atomic_num == 7 and site_type not in ("alpha_to_nitrogen",)):
                    return REACTION_TYPE_TO_INDEX["n_oxidation"]
                if atomic_num == 8:
                    return REACTION_TYPE_TO_INDEX["oxidation"]
                # Epoxidation
                if site_type == "alkene_epoxidation":
                    return REACTION_TYPE_TO_INDEX["epoxidation"]
                # C-H oxidation
                if site_type == "benzylic":
                    return REACTION_TYPE_TO_INDEX["benzylic_hydroxylation"]
                if is_aromatic:
                    return REACTION_TYPE_TO_INDEX["aromatic_hydroxylation"]
            return REACTION_TYPE_TO_INDEX["aliphatic_hydroxylation"]

        def __getitem__(self, idx: int) -> Dict[str, object]:
            drug = self.rows[idx]
            smiles = str(drug.get("smiles", "")).strip()
            chemistry = self._features[smiles]
            record = self.predictions[smiles]
            num_atoms = int(chemistry.num_atoms)
            site_scores_raw = self._align_matrix(record["site_scores_raw"], num_atoms, 3)
            cyp_probs_raw = record.get("cyp_probs_raw")
            if cyp_probs_raw is None:
                global_features = record["global_features"].float()
                cyp_probs_raw = global_features[: 15].view(3, 5)
            cyp_probs_raw = cyp_probs_raw.float()
            site_labels = self._align_vector(record.get("site_labels", torch.zeros(num_atoms)), num_atoms)
            if not bool(torch.any(site_labels > 0.5)):
                site_atoms = _first_present_site_list(drug)
                if site_atoms:
                    site_labels = torch.zeros((num_atoms,), dtype=torch.float32)
                    for atom_idx in site_atoms:
                        if 0 <= atom_idx < num_atoms:
                            site_labels[atom_idx] = 1.0
            reaction_label = self._infer_reaction_label(chemistry, site_labels)
            return {
                "smiles": smiles,
                "mol_features_raw": chemistry.mol_features_raw.float(),
                "atom_features_raw": chemistry.atom_features_raw.float(),
                "smarts_matches": chemistry.smarts_matches.float(),
                "site_scores_raw": site_scores_raw.float(),
                "cyp_probs_raw": cyp_probs_raw.float(),
                "site_labels": site_labels.float(),
                "site_supervised": torch.tensor(bool(torch.any(site_labels > 0.5)), dtype=torch.bool),
                "cyp_label": torch.tensor(self._infer_cyp_label(drug, record), dtype=torch.long),
                "reaction_label": torch.tensor(reaction_label, dtype=torch.long),
                "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
                "name": str(drug.get("name", "")),
            }


    def collate_cahml_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
        return batch[0]


    @dataclass
    class CAHMLTrainer:
        model: object
        train_dataset: CAHMLDataset
        val_dataset: CAHMLDataset
        config: CAHMLConfig
        device: Optional[torch.device] = None

        def __post_init__(self):
            self.device = self.device or (
                torch.device("mps")
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.model.to(self.device)
            self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, collate_fn=collate_cahml_batch)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, collate_fn=collate_cahml_batch)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=5)
            self.best_val_top1 = -1.0
            self.history: List[Dict[str, object]] = []
            if float(self.config.listmle_weight) > 0.0 or float(self.config.focal_weight) > 0.0:
                self.site_loss_fn = CombinedSiteRankingLoss(
                    mirank_weight=self.config.mirank_weight,
                    listmle_weight=self.config.listmle_weight,
                    bce_weight=self.config.bce_weight,
                    focal_weight=self.config.focal_weight,
                    margin=self.config.ranking_margin,
                    hard_negative_fraction=self.config.hard_negative_fraction,
                )
            else:
                self.site_loss_fn = SiteRankingLossV2(
                    mirank_weight=self.config.mirank_weight,
                    bce_weight=self.config.bce_weight,
                    margin=self.config.ranking_margin,
                    hard_negative_fraction=self.config.hard_negative_fraction,
                )

        def _move(self, batch: Dict[str, object]) -> Dict[str, object]:
            out: Dict[str, object] = {}
            for key, value in batch.items():
                out[key] = value.to(self.device) if hasattr(value, "to") else value
            return out

        def train_epoch(self) -> Dict[str, float]:
            self.model.train()
            totals = {"loss": 0.0, "site_loss": 0.0, "cyp_loss": 0.0, "reaction_loss": 0.0}
            n = 0
            site_loss_stats = {"mirank": 0.0, "bce": 0.0, "total": 0.0}
            for raw in self.train_loader:
                batch = self._move(raw)
                outputs = self.model(
                    batch["mol_features_raw"],
                    batch["atom_features_raw"],
                    batch["smarts_matches"],
                    batch["site_scores_raw"],
                    batch["cyp_probs_raw"],
                )
                if bool(batch["site_supervised"].item()):
                    site_loss, site_loss_stats = self.site_loss_fn(outputs["site_scores"], batch["site_labels"])
                else:
                    site_loss = outputs["site_scores"].sum() * 0.0
                    site_loss_stats = {"mirank": 0.0, "bce": 0.0, "total": 0.0}
                cyp_loss = torch.nn.functional.cross_entropy(outputs["cyp_logits"].unsqueeze(0), batch["cyp_label"].view(1))
                if int(batch["reaction_label"].item()) >= 0:
                    reaction_loss = torch.nn.functional.cross_entropy(outputs["reaction_logits"].unsqueeze(0), batch["reaction_label"].view(1))
                else:
                    reaction_loss = outputs["reaction_logits"].sum() * 0.0
                loss = (
                    self.config.site_loss_weight * site_loss
                    + self.config.cyp_loss_weight * cyp_loss
                    + self.config.reaction_loss_weight * reaction_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                totals["loss"] += float(loss.item())
                totals["site_loss"] += float(site_loss.item())
                totals["cyp_loss"] += float(cyp_loss.item())
                totals["reaction_loss"] += float(reaction_loss.item())
                n += 1
            denom = max(1, n)
            return {
                **{k: v / denom for k, v in totals.items()},
                "site_mirank": float(site_loss_stats.get("mirank", 0.0)),
                "site_bce": float(site_loss_stats.get("bce", 0.0)),
            }

        def validate(self) -> Dict[str, object]:
            self.model.eval()
            metrics = {"site_top1": 0.0, "site_top3": 0.0, "cyp_acc": 0.0, "reaction_acc": 0.0}
            trusted_models = {name: 0 for name in self.config.model_names}
            n = 0
            n_site = 0
            n_reaction = 0
            with torch.no_grad():
                for raw in self.val_loader:
                    batch = self._move(raw)
                    outputs = self.model(
                        batch["mol_features_raw"],
                        batch["atom_features_raw"],
                        batch["smarts_matches"],
                        batch["site_scores_raw"],
                        batch["cyp_probs_raw"],
                    )
                    eval_out = evaluate_cahml_predictions(
                        outputs["site_scores"],
                        batch["site_labels"],
                        outputs["cyp_logits"],
                        int(batch["cyp_label"].item()),
                        outputs["reaction_logits"],
                        int(batch["reaction_label"].item()),
                    )
                    metrics["cyp_acc"] += eval_out["cyp_acc"]
                    if "reaction_acc" in eval_out:
                        metrics["reaction_acc"] += eval_out["reaction_acc"]
                        n_reaction += 1
                    if bool(batch["site_supervised"].item()):
                        metrics["site_top1"] += eval_out["site_top1"]
                        metrics["site_top3"] += eval_out["site_top3"]
                        n_site += 1
                    trusted_models[outputs["explanation"]["trusted_model"]] += 1
                    n += 1
            return {
                "site_top1": metrics["site_top1"] / max(1, n_site),
                "site_top3": metrics["site_top3"] / max(1, n_site),
                "cyp_acc": metrics["cyp_acc"] / max(1, n),
                "reaction_acc": metrics["reaction_acc"] / max(1, n_reaction),
                "n_samples": float(n),
                "n_site_samples": float(n_site),
                "trusted_model_counts": trusted_models,
            }

        def _save_progress(self, *, best_state, last_val: Optional[Dict[str, object]] = None, status: str = "running") -> Dict[str, object]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            latest_path = Path(self.config.checkpoint_dir) / "cahml_latest.pt"
            best_path = Path(self.config.checkpoint_dir) / "cahml_best.pt"
            archive_path = Path(self.config.checkpoint_dir) / f"cahml_{timestamp}.pt"
            report_path = Path(self.config.artifact_dir) / f"cahml_report_{timestamp}.json"
            payload = {
                "model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "config": self.config.__dict__,
                "best_val_top1": self.best_val_top1,
                "history": self.history,
                "final_validation": last_val or {},
                "status": status,
            }
            torch.save(payload, latest_path)
            if best_state is not None:
                best_payload = dict(payload)
                best_payload["model_state_dict"] = best_state
                best_payload["status"] = f"{status}_best"
                torch.save(best_payload, best_path)
            torch.save(payload, archive_path)
            report_path.write_text(json.dumps({"best_val_top1": self.best_val_top1, "history": self.history, "final_validation": last_val or {}, "status": status}, indent=2))
            return {
                "payload": payload,
                "latest_path": str(latest_path),
                "best_path": str(best_path),
                "archive_path": str(archive_path),
                "report_path": str(report_path),
            }

        def train(self) -> Dict[str, object]:
            self.config.ensure_dirs()
            patience_left = self.config.patience
            best_state = None
            last_val = {}
            try:
                for epoch in range(1, int(self.config.epochs) + 1):
                    train_metrics = self.train_epoch()
                    val_metrics = self.validate()
                    last_val = val_metrics
                    self.scheduler.step(float(val_metrics["site_top1"]))
                    self.history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
                    print(
                        f"Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} | "
                        f"site_top1={float(val_metrics['site_top1']):.3f} | site_top3={float(val_metrics['site_top3']):.3f} | "
                        f"cyp={float(val_metrics['cyp_acc']):.3f} | rxn={float(val_metrics['reaction_acc']):.3f} | "
                        f"trusted={val_metrics['trusted_model_counts']}",
                        flush=True,
                    )
                    if float(val_metrics["site_top1"]) > self.best_val_top1:
                        self.best_val_top1 = float(val_metrics["site_top1"])
                        best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                        patience_left = self.config.patience
                    else:
                        patience_left -= 1
                    self._save_progress(best_state=best_state, last_val=last_val, status="running")
                    if patience_left <= 0:
                        break
            except KeyboardInterrupt:
                saved = self._save_progress(best_state=best_state, last_val=last_val, status="interrupted")
                print(f"Interrupted. Saved latest checkpoint: {saved['latest_path']}", flush=True)
                print(f"Saved best checkpoint: {saved['best_path']}", flush=True)
                print(f"Saved report: {saved['report_path']}", flush=True)
                return saved["payload"]
            if best_state is not None:
                self.model.load_state_dict(best_state, strict=False)
            saved = self._save_progress(best_state=best_state, last_val=last_val, status="completed")
            return saved["payload"]
else:  # pragma: no cover
    class CAHMLDataset:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class CAHMLTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
