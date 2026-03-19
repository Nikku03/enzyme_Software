from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None

from .metabolism_simulator import MetabolismSimulator, MetabolismType
from .utils import extract_site_indices


@dataclass
class MetabolicStep:
    step_number: int
    parent_smiles: str
    metabolite_smiles: str
    site_atom_idx: int
    metabolism_type: str
    supervision_source: str
    source_weight: float
    candidate_rank: int = 0


@dataclass
class MetabolicPathway:
    drug_smiles: str
    drug_name: str
    steps: List[MetabolicStep] = field(default_factory=list)
    terminal_metabolite: Optional[str] = None
    total_steps: int = 0
    ground_truth_sites: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "drug_smiles": self.drug_smiles,
            "drug_name": self.drug_name,
            "steps": [
                {
                    **asdict(step),
                    "source": step.supervision_source,
                }
                for step in self.steps
            ],
            "terminal_metabolite": self.terminal_metabolite,
            "total_steps": int(self.total_steps),
            "ground_truth_sites": list(self.ground_truth_sites),
        }


class PathwayGenerator:
    def __init__(
        self,
        *,
        max_steps: int = 6,
        min_heavy_atoms: int = 5,
        ground_truth_first_only: bool = True,
        pseudo_step_weight: float = 0.35,
        ground_truth_step_weight: float = 1.0,
    ):
        self.max_steps = int(max_steps)
        self.min_heavy_atoms = int(min_heavy_atoms)
        self.ground_truth_first_only = bool(ground_truth_first_only)
        self.pseudo_step_weight = float(pseudo_step_weight)
        self.ground_truth_step_weight = float(ground_truth_step_weight)
        self.simulator = MetabolismSimulator()

    def _pick_site(
        self,
        current_smiles: str,
        *,
        step: int,
        ground_truth_sites: List[int],
        predictor_fn: Optional[Callable[[str], int | Dict[str, object]]],
    ):
        if step == 0 and ground_truth_sites:
            return {
                "site_atom_idx": int(ground_truth_sites[0]),
                "metabolism_type": None,
                "supervision_source": "ground_truth",
                "source_weight": self.ground_truth_step_weight,
                "candidate_rank": 0,
            }

        if predictor_fn is not None:
            predicted = predictor_fn(current_smiles)
            if isinstance(predicted, dict):
                site = predicted.get("site_atom_idx", predicted.get("site"))
                if site is not None:
                    return {
                        "site_atom_idx": int(site),
                        "metabolism_type": predicted.get("metabolism_type"),
                        "supervision_source": "predicted",
                        "source_weight": self.pseudo_step_weight,
                        "candidate_rank": int(predicted.get("candidate_rank", 0)),
                    }
            else:
                return {
                    "site_atom_idx": int(predicted),
                    "metabolism_type": None,
                    "supervision_source": "predicted",
                    "source_weight": self.pseudo_step_weight,
                    "candidate_rank": 0,
                }

        candidate_sites = self.simulator.get_all_metabolism_sites(current_smiles)
        if not candidate_sites:
            return None
        best = candidate_sites[0]
        return {
            "site_atom_idx": int(best["atom_idx"]),
            "metabolism_type": best.get("metabolism_type"),
            "supervision_source": "heuristic",
            "source_weight": self.pseudo_step_weight,
            "candidate_rank": 0,
        }

    def generate(
        self,
        *,
        smiles: str,
        drug_name: str = "unknown",
        ground_truth_sites: Optional[List[int]] = None,
        predictor_fn: Optional[Callable[[str], int | Dict[str, object]]] = None,
    ) -> MetabolicPathway:
        pathway = MetabolicPathway(
            drug_smiles=str(smiles),
            drug_name=str(drug_name),
            ground_truth_sites=list(ground_truth_sites or []),
        )
        if Chem is None:
            pathway.terminal_metabolite = smiles
            return pathway

        seen_smiles = set()
        current_smiles = str(smiles)
        ordered_truth = list(ground_truth_sites or [])
        for step in range(self.max_steps):
            prep = Chem.MolFromSmiles(current_smiles)
            if prep is None or prep.GetNumHeavyAtoms() < self.min_heavy_atoms:
                break
            if current_smiles in seen_smiles:
                break
            seen_smiles.add(current_smiles)

            site_payload = self._pick_site(
                current_smiles,
                step=step,
                ground_truth_sites=ordered_truth,
                predictor_fn=predictor_fn,
            )
            if site_payload is None:
                break
            metabolism_type = site_payload.get("metabolism_type")
            if isinstance(metabolism_type, str):
                try:
                    metabolism_type = MetabolismType(metabolism_type)
                except Exception:
                    metabolism_type = None
            result = self.simulator.metabolize(
                current_smiles,
                int(site_payload["site_atom_idx"]),
                metabolism_type=metabolism_type,
            )
            if not result.success or not result.metabolite_smiles:
                if step == 0 and ordered_truth:
                    heuristic = self._pick_site(
                        current_smiles,
                        step=1,
                        ground_truth_sites=[],
                        predictor_fn=predictor_fn,
                    )
                    if heuristic is None:
                        break
                    result = self.simulator.metabolize(
                        current_smiles,
                        int(heuristic["site_atom_idx"]),
                        metabolism_type=heuristic.get("metabolism_type"),
                    )
                    if not result.success or not result.metabolite_smiles:
                        break
                    site_payload = heuristic
                else:
                    break

            pathway.steps.append(
                MetabolicStep(
                    step_number=step,
                    parent_smiles=current_smiles,
                    metabolite_smiles=str(result.metabolite_smiles),
                    site_atom_idx=int(site_payload["site_atom_idx"]),
                    metabolism_type=str(result.metabolism_type.value),
                    supervision_source=str(site_payload["supervision_source"]),
                    source_weight=float(site_payload["source_weight"]),
                    candidate_rank=int(site_payload.get("candidate_rank", 0)),
                )
            )
            current_smiles = str(result.metabolite_smiles)
        pathway.terminal_metabolite = current_smiles
        pathway.total_steps = len(pathway.steps)
        return pathway

    def generate_dataset(
        self,
        drugs: List[Dict[str, object]],
        *,
        output_path: str | Path | None = None,
        predictor_fn: Optional[Callable[[str], int | Dict[str, object]]] = None,
    ) -> List[MetabolicPathway]:
        pathways: list[MetabolicPathway] = []
        for idx, drug in enumerate(drugs, start=1):
            smiles = str(drug.get("smiles", "")).strip()
            if not smiles:
                continue
            pathway = self.generate(
                smiles=smiles,
                drug_name=str(drug.get("name", f"drug_{idx}")),
                ground_truth_sites=extract_site_indices(drug),
                predictor_fn=predictor_fn,
            )
            pathways.append(pathway)
            if idx % 50 == 0 or idx == len(drugs):
                print(f"Generated {idx}/{len(drugs)} pathways", flush=True)
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                __import__("json").dumps([pathway.to_dict() for pathway in pathways], indent=2),
                encoding="utf-8",
            )
        return pathways
