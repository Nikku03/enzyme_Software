#!/usr/bin/env python3
"""Reusable manual pipeline audit runner.

Runs a curated set of substrate/target cases through the full pipeline,
prints compact checkpoints, and optionally writes full ctx.data JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


PRIORITY_RUNS: List[Tuple[str, str, str]] = [
    ("amide", "CC(=O)NC", "amide"),
    ("aliphatic_ch", "CCCC", "C-H"),
    ("ester_with_cf3", "CC(=O)OCC(F)(F)F", "ester__acyl_o"),
    ("benzene_ch", "c1ccccc1", "C-H"),
    ("no_ch_available", "FC(F)(F)C(F)(F)F", "C-H"),
    ("cysteine_hetero_rich", "O=C(O)C(N)CS", "C-H"),
    ("alkyl_halide", "CCCl", "C-Cl"),
    ("oh_bond", "CCO", "O-H"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual pipeline checkpoint audit")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case id to run (repeatable). Default: all priority cases.",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Write full ctx.data for each case to output dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/manual_audit",
        help="Output directory for JSON artifacts.",
    )
    parser.add_argument(
        "--single-json",
        default=None,
        help=(
            "Write one combined JSON file containing all case outputs. "
            "Example: artifacts/manual_audit/all_cases.json"
        ),
    )
    return parser.parse_args()


def _load_pipeline():
    # Ensure src/ is importable when script is run from project root.
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from enzyme_software.pipeline import run_pipeline  # noqa: WPS433

    return run_pipeline


def _select_cases(requested: List[str]) -> List[Tuple[str, str, str]]:
    if not requested:
        return PRIORITY_RUNS
    allowed = set(requested)
    selected = [row for row in PRIORITY_RUNS if row[0] in allowed]
    missing = sorted(allowed - {row[0] for row in selected})
    if missing:
        raise SystemExit(f"Unknown case id(s): {', '.join(missing)}")
    return selected


def _top_candidate(resolved: Dict[str, Any]) -> Dict[str, Any]:
    candidates = resolved.get("candidate_bonds") or []
    if not candidates:
        return {}
    c0 = candidates[0]
    keys = [
        "rank",
        "atom_indices",
        "subclass",
        "bond_class",
        "bde_kj_mol",
        "radical_stability",
        "resolution_policy",
    ]
    return {k: c0.get(k) for k in keys}


def _print_case_header(case_id: str, smiles: str, target_bond: str) -> None:
    print("\n" + "=" * 96)
    print(f"{case_id} | {smiles} | target={target_bond}")


def _print_case_summary(ctx_data: Dict[str, Any]) -> None:
    job_card = ctx_data.get("job_card") or {}
    shared = ctx_data.get("shared_io") or {}
    m1 = (shared.get("outputs") or {}).get("module_minus1") or {}
    m1_result = m1.get("result") or {}
    m1_cpt = m1.get("cpt") or {}
    resolved = m1_result.get("resolved_target") or {}
    route = (job_card.get("mechanism_route") or {}).get("primary")
    energy = job_card.get("energy_ledger") or {}
    physics = job_card.get("physics_layer") or {}
    route_post = job_card.get("route_posteriors") or []

    print("decision:", job_card.get("decision"), "| halt:", job_card.get("pipeline_halt_reason"))
    print("route:", route)
    print("module-1:", m1_result.get("status"), "| cpt.track:", m1_cpt.get("track"))
    print(
        "resolved:",
        "match_count=",
        resolved.get("match_count"),
        "| next_input_required=",
        resolved.get("next_input_required"),
    )
    print(
        "resolution:",
        resolved.get("resolution_policy"),
        "| confidence=",
        resolved.get("resolution_confidence"),
    )
    print("resolution_note:", resolved.get("resolution_note"))
    print("top_candidate:", _top_candidate(resolved))
    print(
        "energy:",
        f"ΔG‡={energy.get('deltaG_dagger_kJ')}",
        f"k_eff={energy.get('k_eff_s_inv')}",
        f"p_success={energy.get('p_success_horizon')}",
    )
    print(
        "physics_layer:",
        f"bond_length_A={physics.get('bond_length_A')}",
        f"coulomb={physics.get('coulomb_energy_kj_mol')}",
        f"polarization={physics.get('polarization_ratio')}",
    )
    top_routes = [(r.get("route"), r.get("posterior")) for r in route_post[:4]]
    print("route_posteriors:", top_routes)


def main() -> None:
    args = _parse_args()
    run_pipeline = _load_pipeline()
    selected = _select_cases(args.case)
    out_dir = Path(args.output_dir)
    combined_path = Path(args.single_json) if args.single_json else None
    if args.dump_json:
        out_dir.mkdir(parents=True, exist_ok=True)
    if combined_path is not None:
        combined_path.parent.mkdir(parents=True, exist_ok=True)

    combined: Dict[str, Any] = {
        "schema_version": "manual_pipeline_audit.v1",
        "cases_requested": [case_id for case_id, _, _ in selected],
        "cases": {},
        "failures": [],
    }
    failures: List[str] = []
    for case_id, smiles, target_bond in selected:
        try:
            ctx = run_pipeline(smiles, target_bond)
            _print_case_header(case_id, smiles, target_bond)
            _print_case_summary(ctx.data)
            combined["cases"][case_id] = {
                "smiles": smiles,
                "target_bond": target_bond,
                "ctx_data": ctx.data,
            }
            if args.dump_json:
                payload = json.dumps(ctx.data, indent=2, default=str)
                (out_dir / f"{case_id}.json").write_text(payload, encoding="utf-8")
        except Exception as exc:  # pragma: no cover - manual tool path
            msg = f"{case_id}: {type(exc).__name__}: {exc}"
            failures.append(msg)
            combined["failures"].append(msg)

    if args.dump_json:
        print(f"\nJSON artifacts written to: {out_dir}")
    if combined_path is not None:
        combined_json = json.dumps(combined, indent=2, default=str)
        combined_path.write_text(combined_json, encoding="utf-8")
        print(f"Combined JSON written to: {combined_path}")

    if failures:
        print("\nFAILURES:")
        for item in failures:
            print("-", item)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
