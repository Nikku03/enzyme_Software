from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from enzyme_software.evidence_store import (
    add_outcome,
    init_db as init_evidence_db,
    load_literature_evidence,
    save_run,
)
from enzyme_software.literature_evidence import EvidenceRecord
from enzyme_software.unity_schema import (
    BondContext,
    ConditionProfile,
    Module0Out,
    Module3Out,
    PhysicsAudit,
    UnityRecord,
    build_features,
)
from enzyme_software.physicscore import eyring_rate_constant, get_baseline_barrier


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_revision_hash(rows: List[Tuple[Any, ...]]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        line = json.dumps(row, sort_keys=True, default=str).encode("utf-8")
        hasher.update(line)
    return hasher.hexdigest()


def _load_reactions(path: str) -> List[Tuple[Any, ...]]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, source, source_reaction_id, reactants_smiles, products_smiles,
                   conditions_json, yield_value, metadata_json
            FROM reactions
            ORDER BY id ASC
            """
        ).fetchall()
        return [tuple(row) for row in rows]


def _parse_conditions(payload: Optional[str]) -> ConditionProfile:
    if not payload:
        return ConditionProfile()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ConditionProfile()
    return ConditionProfile(
        pH=data.get("pH"),
        temperature_C=data.get("temperature_C"),
        ionic_strength=data.get("ionic_strength"),
        solvent=data.get("solvent"),
        cofactors=list(data.get("cofactors") or []),
    )


def _split_smiles(smiles: Optional[str]) -> List[str]:
    if not smiles:
        return []
    return [chunk for chunk in smiles.split(".") if chunk]


def _infer_target_bond(reactants: List[str], products: List[str]) -> str:
    reactant_str = ".".join(reactants)
    product_str = ".".join(products)
    patterns = [
        ("amide__C_N", "C(=O)N"),
        ("ester__acyl_O", "C(=O)O"),
        ("thioester__acyl_S", "C(=O)S"),
        ("aryl_halide__C_X", "CBr"),
        ("aryl_halide__C_X", "CCl"),
        ("aryl_halide__C_X", "CI"),
        ("phosphate_ester__P_O", "P(=O)O"),
    ]
    for token, pattern in patterns:
        reactant_hits = reactant_str.count(pattern)
        product_hits = product_str.count(pattern)
        if reactant_hits > product_hits:
            return token
    for token, pattern in patterns:
        if pattern in reactant_str:
            return token
    return "unknown"


def _bond_context_from_smiles(smiles: str, bond_class: str) -> BondContext:
    atoms = 0
    hetero = 0
    rings = 0
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        if ch.isdigit():
            rings += 1
        if ch.isalpha():
            symbol = ch
            if i + 1 < len(smiles) and smiles[i + 1].islower():
                symbol += smiles[i + 1]
                i += 1
            atoms += 1
            if symbol not in {"C", "H"}:
                hetero += 1
        i += 1
    polarity = "polar" if bond_class in {"ester", "amide", "thioester", "phosphate"} else "nonpolar"
    return BondContext(
        bond_role=bond_class,
        bond_role_confidence=0.6 if bond_class != "unknown" else 0.3,
        bond_class=bond_class,
        polarity=polarity,
        atom_count=atoms,
        hetero_atoms=hetero,
        ring_count=rings,
    )


def _route_for_bond_class(bond_class: str) -> str:
    if bond_class.startswith("ester"):
        return "serine_hydrolase"
    if bond_class.startswith("amide"):
        return "metallo_esterase"
    return "other"


def _normalize_yield(yield_value: Optional[float]) -> Optional[float]:
    if yield_value is None:
        return None
    try:
        value = float(yield_value)
    except (TypeError, ValueError):
        return None
    if value > 1.0:
        if value <= 100.0:
            return value / 100.0
        return 1.0
    return max(0.0, min(1.0, value))


def _record_from_evidence(evidence: EvidenceRecord, run_id: str) -> UnityRecord:
    condition_profile = ConditionProfile(
        pH=evidence.conditions.get("pH"),
        temperature_C=evidence.conditions.get("temperature_C"),
        ionic_strength=evidence.conditions.get("ionic_strength"),
        solvent=evidence.conditions.get("solvent"),
        cofactors=list(evidence.conditions.get("cofactors") or []),
    )
    bond_class = "unknown"
    if evidence.reaction_family == "hydrolysis":
        bond_class = "ester"
    bond_context = _bond_context_from_smiles(evidence.substrate_smiles, bond_class)
    route_family = _route_for_bond_class(bond_class)
    barrier = get_baseline_barrier(bond_class, route_family)
    delta_g_j = barrier.deltaG_dagger_kJ * 1000.0
    temp_k = condition_profile.temperature_K or 298.15
    k_eyring = eyring_rate_constant(delta_g_j, temp_k)
    physics_audit = PhysicsAudit(
        deltaG_dagger_kJ_per_mol=barrier.deltaG_dagger_kJ,
        eyring_k_s_inv=k_eyring,
        k_eff_s_inv=k_eyring,
        temperature_K=temp_k,
        horizon_s=3600.0,
        notes=[f"baseline_barrier_source={barrier.source}"],
    )
    return UnityRecord(
        run_id=run_id,
        smiles=evidence.substrate_smiles,
        target_bond=bond_class,
        requested_output=None,
        condition_profile=condition_profile,
        bond_context=bond_context,
        physics_audit=physics_audit,
        module0=Module0Out(route_family=route_family),
        module3=Module3Out(status="LITERATURE", batch_id=evidence.source_id),
        created_at=_now_iso(),
        updated_at=_now_iso(),
    )


def _iter_training_lines(
    dataset_db: str,
    evidence_db: str,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    rows = _load_reactions(dataset_db)
    dataset_hash = _dataset_revision_hash(rows) if rows else "no_dataset"
    literature = load_literature_evidence(evidence_db)
    lines: List[Dict[str, Any]] = []

    for entry in literature:
        if entry.confidence < 0.5 or entry.reaction_family in {"unknown", "bioactivity"}:
            continue
        run_id = f"lit_{entry.source}_{entry.source_id}"
        record = _record_from_evidence(entry, run_id)
        features = build_features(record)
        lines.append(
            {
                "features": features,
                "label": 1 if entry.outcome_label else 0,
                "provenance": entry.provenance,
            }
        )

    count = 0
    for row in rows:
        if limit is not None and count >= limit:
            break
        (
            reaction_id,
            source,
            source_reaction_id,
            reactants_smiles,
            products_smiles,
            conditions_json,
            yield_value,
            metadata_json,
        ) = row
        reactants = _split_smiles(reactants_smiles)
        products = _split_smiles(products_smiles)
        if not reactants:
            continue
        smiles = reactants[0]
        target_bond_token = _infer_target_bond(reactants, products)
        bond_class = target_bond_token.split("__")[0] if "__" in target_bond_token else target_bond_token
        condition_profile = _parse_conditions(conditions_json)
        bond_context = _bond_context_from_smiles(smiles, bond_class)
        barrier = get_baseline_barrier(bond_class, _route_for_bond_class(bond_class))
        temp_k = condition_profile.temperature_K or 298.15
        k_eyring = eyring_rate_constant(barrier.deltaG_dagger_kJ * 1000.0, temp_k)
        record = UnityRecord(
            run_id=f"synth_{source}_{reaction_id}_{dataset_hash[:8]}",
            smiles=smiles,
            target_bond=target_bond_token,
            requested_output=products[0] if products else None,
            condition_profile=condition_profile,
            bond_context=bond_context,
            physics_audit=PhysicsAudit(
                deltaG_dagger_kJ_per_mol=barrier.deltaG_dagger_kJ,
                eyring_k_s_inv=k_eyring,
                k_eff_s_inv=k_eyring,
                temperature_K=temp_k,
                horizon_s=3600.0,
                notes=[f"baseline_barrier_source={barrier.source}"],
            ),
            module0=Module0Out(route_family=_route_for_bond_class(bond_class)),
            module3=Module3Out(status="SYNTHETIC", batch_id=dataset_hash),
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        conversion = _normalize_yield(yield_value)
        if conversion is None:
            continue
        features = build_features(record)
        lines.append(
            {
                "features": features,
                "label": 1 if conversion >= 0.2 else 0,
                "provenance": {
                    "source": source,
                    "source_reaction_id": source_reaction_id,
                    "dataset_hash": dataset_hash,
                },
            }
        )
        count += 1
    return lines


def build_corpus(
    dataset_db: str,
    evidence_db: str,
    limit: Optional[int],
    out_path: str,
    write_evidence: bool,
) -> int:
    lines = _iter_training_lines(dataset_db, evidence_db, limit)
    if not lines:
        raise ValueError("No evidence lines generated for training corpus.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")

    if write_evidence:
        dataset_rows = _load_reactions(dataset_db)
        dataset_hash = _dataset_revision_hash(dataset_rows) if dataset_rows else "no_dataset"
        init_evidence_db(evidence_db)
        count = 0
        for row in dataset_rows:
            if limit is not None and count >= limit:
                break
            (
                reaction_id,
                source,
                source_reaction_id,
                reactants_smiles,
                products_smiles,
                conditions_json,
                yield_value,
                metadata_json,
            ) = row
            reactants = _split_smiles(reactants_smiles)
            products = _split_smiles(products_smiles)
            if not reactants:
                continue
            smiles = reactants[0]
            requested_output = products[0] if products else None
            target_bond_token = _infer_target_bond(reactants, products)
            bond_class = target_bond_token.split("__")[0] if "__" in target_bond_token else target_bond_token
            condition_profile = _parse_conditions(conditions_json)
            bond_context = _bond_context_from_smiles(smiles, bond_class)
            run_id = f"synth_{source}_{reaction_id}_{dataset_hash[:8]}"
            record = UnityRecord(
                run_id=run_id,
                smiles=smiles,
                target_bond=target_bond_token,
                requested_output=requested_output,
                condition_profile=condition_profile,
                bond_context=bond_context,
                module0=Module0Out(route_family=_route_for_bond_class(bond_class)),
                module3=Module3Out(status="SYNTHETIC", batch_id=dataset_hash),
                created_at=_now_iso(),
                updated_at=_now_iso(),
            )
            save_run(evidence_db, record)
            add_outcome(
                evidence_db,
                run_id=run_id,
                arm_id="SYNTH",
                any_activity=True,
                target_match=True,
                conversion=_normalize_yield(yield_value),
                failure_mode=None,
                notes=(
                    "synthetic_from_dataset "
                    f"source={source} source_reaction_id={source_reaction_id} "
                    f"dataset_hash={dataset_hash}"
                ),
            )
            count += 1
        print(f"Wrote {count} synthetic runs. dataset_hash={dataset_hash}")

    print(f"Wrote training corpus: {out_path} (lines={len(lines)})")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build training corpus from dataset store.")
    parser.add_argument("--dataset-db", required=True, help="Path to dataset store DB.")
    parser.add_argument("--evidence-db", required=True, help="Path to evidence store DB.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of runs.")
    parser.add_argument(
        "--out",
        default="artifacts/training_corpus.jsonl",
        help="Output JSONL training corpus path.",
    )
    parser.add_argument(
        "--write-evidence",
        action="store_true",
        help="Also write synthetic runs to the evidence DB (legacy behavior).",
    )
    args = parser.parse_args(argv)
    try:
        return build_corpus(
            args.dataset_db,
            args.evidence_db,
            args.limit,
            args.out,
            args.write_evidence,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
