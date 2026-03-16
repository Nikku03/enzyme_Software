from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from .benchmark_schema import BenchmarkCase, BenchmarkTolerances


def _normalize_case(payload: dict) -> BenchmarkCase:
    tolerances = payload.get("tolerances") or {}
    return BenchmarkCase(
        case_id=str(payload["case_id"]),
        smiles=str(payload["smiles"]),
        target_bond=payload["target_bond"],
        expected_reactive_sites=payload.get("expected_reactive_sites"),
        expected_route_family=payload.get("expected_route_family"),
        expected_mechanism_tags=list(payload.get("expected_mechanism_tags") or []),
        expected_scaffold_tags=list(payload.get("expected_scaffold_tags") or []),
        expected_variant_tags=list(payload.get("expected_variant_tags") or []),
        expected_experiment_tags=list(payload.get("expected_experiment_tags") or []),
        difficulty=str(payload.get("difficulty") or "medium"),
        notes=str(payload.get("notes") or ""),
        metadata=dict(payload.get("metadata") or {}),
        tolerances=BenchmarkTolerances(**tolerances),
    )


def demo_benchmark_cases() -> List[BenchmarkCase]:
    raw = [
        {
            "case_id": "demo_benzylic_toluene",
            "smiles": "Cc1ccccc1",
            "target_bond": "C-H",
            "expected_route_family": "p450",
            "expected_mechanism_tags": ["C-H activation", "benzylic"],
            "difficulty": "easy",
            "notes": "Strong benzylic oxidation prior; good route sanity check.",
        },
        {
            "case_id": "demo_ester_ethyl_acetate",
            "smiles": "CC(=O)OCC",
            "target_bond": "ester__acyl_o",
            "expected_route_family": "serine_hydrolase",
            "expected_mechanism_tags": ["hydrolysis"],
            "expected_variant_tags": ["oxyanion", "anchor"],
            "difficulty": "easy",
        },
        {
            "case_id": "demo_amide_nma",
            "smiles": "CC(=O)NC",
            "target_bond": "amide",
            "expected_route_family": "amidase",
            "expected_mechanism_tags": ["amide hydrolysis"],
            "difficulty": "medium",
        },
        {
            "case_id": "demo_aliphatic_butane",
            "smiles": "CCCC",
            "target_bond": "C-H",
            "expected_route_family": "p450",
            "expected_mechanism_tags": ["C-H activation", "aliphatic"],
            "difficulty": "medium",
        },
        {
            "case_id": "demo_hetero_ethanol",
            "smiles": "CCO",
            "target_bond": "O-H",
            "expected_mechanism_tags": ["oxidoreductase"],
            "difficulty": "medium",
            "notes": "Weak expectation: mainly structural validity and route plausibility.",
        },
        {
            "case_id": "demo_alkyl_halide",
            "smiles": "CCCl",
            "target_bond": "C-Cl",
            "expected_route_family": "haloalkane_dehalogenase",
            "expected_mechanism_tags": ["sn2_displacement"],
            "difficulty": "medium",
        },
        {
            "case_id": "demo_ambiguous_cysteine_like",
            "smiles": "O=C(O)C(N)CS",
            "target_bond": "C-H",
            "expected_mechanism_tags": ["alpha_hetero"],
            "difficulty": "hard",
            "notes": "Ambiguous multi-site case; evaluate disambiguation and calibration, not strict truth.",
        },
        {
            "case_id": "demo_benzene_edge",
            "smiles": "c1ccccc1",
            "target_bond": "C-H",
            "expected_mechanism_tags": ["aryl"],
            "difficulty": "edge",
            "notes": "Aromatic C-H is hard; weak expectation only.",
        },
        {
            "case_id": "demo_fluoroform_edge",
            "smiles": "FC(F)F",
            "target_bond": "C-H",
            "expected_mechanism_tags": ["fluorinated"],
            "difficulty": "edge",
            "notes": "Used to probe overconfidence on difficult radical chemistry.",
        },
        {
            "case_id": "demo_perfluoro_no_ch",
            "smiles": "FC(F)(F)C(F)(F)F",
            "target_bond": "C-H",
            "difficulty": "edge",
            "notes": "Expected graceful failure because no C-H exists.",
        },
    ]
    return [_normalize_case(item) for item in raw]


def load_benchmark_cases(path: Optional[str] = None, *, max_cases: Optional[int] = None) -> List[BenchmarkCase]:
    if path is None:
        cases = demo_benchmark_cases()
        return cases[:max_cases] if max_cases is not None else cases

    benchmark_path = Path(path)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    if benchmark_path.suffix.lower() == ".jsonl":
        with benchmark_path.open() as handle:
            raw_cases = [json.loads(line) for line in handle if line.strip()]
    else:
        payload = json.loads(benchmark_path.read_text())
        raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_cases, list):
            raise ValueError("Benchmark JSON must be a list or contain a top-level 'cases' list")

    cases = [_normalize_case(item) for item in raw_cases]
    return cases[:max_cases] if max_cases is not None else cases
