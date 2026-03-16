from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional

from enzyme_software.evidence_store import load_datapoints


def _filter_datapoints(
    datapoints: Iterable[Dict[str, Any]],
    module_id: Optional[int],
    item_type: Optional[str],
    scaffold_id: Optional[str],
    variant_id: Optional[str],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for point in datapoints:
        if module_id is not None and point.get("module_id") != module_id:
            continue
        if item_type and point.get("item_type") != item_type:
            continue
        if scaffold_id and point.get("scaffold_id") != scaffold_id:
            continue
        if variant_id and point.get("variant_id") != variant_id:
            continue
        results.append(point)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Query EvidenceStore datapoints by run_id.")
    parser.add_argument("--db", required=True, help="Path to EvidenceStore SQLite DB.")
    parser.add_argument("--run-id", required=True, help="run_id to inspect.")
    parser.add_argument("--limit", type=int, default=None, help="Max datapoints to return.")
    parser.add_argument("--module-id", type=int, default=None, help="Filter by module id.")
    parser.add_argument("--item-type", default=None, help="Filter by item type.")
    parser.add_argument("--scaffold-id", default=None, help="Filter by scaffold id.")
    parser.add_argument("--variant-id", default=None, help="Filter by variant id.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of a short summary.",
    )
    args = parser.parse_args(argv)

    datapoints = load_datapoints(args.db, args.run_id, limit=args.limit)
    filtered = _filter_datapoints(
        datapoints,
        module_id=args.module_id,
        item_type=args.item_type,
        scaffold_id=args.scaffold_id,
        variant_id=args.variant_id,
    )

    if args.json:
        print(json.dumps(filtered, indent=2))
        return 0

    print(f"run_id: {args.run_id}")
    print(f"datapoints: {len(filtered)} (filtered from {len(datapoints)})")
    for point in filtered[:25]:
        module_id = point.get("module_id")
        item_type = point.get("item_type")
        scaffold = point.get("scaffold_id") or "-"
        variant = point.get("variant_id") or "-"
        reasons = point.get("reasons")
        reason_text = ""
        if isinstance(reasons, list) and reasons:
            reason_text = f" | reasons: {', '.join(str(r) for r in reasons[:3])}"
        print(f"- M{module_id} {item_type} scaffold={scaffold} variant={variant}{reason_text}")
    if len(filtered) > 25:
        print(f"... {len(filtered) - 25} more (use --json or --limit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
