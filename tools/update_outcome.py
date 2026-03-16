from __future__ import annotations

import argparse
import sys
from typing import Optional

from enzyme_software.evidence_store import add_outcome

ALLOWED_FAILURE_MODES = {
    "conditions_limited",
    "access_limited",
    "reach_limited",
    "retention_limited",
    "mechanism_mismatch",
    "unknown",
}


def _parse_bool(value: Optional[str], field: str) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{field} must be a boolean (true/false).")


def _parse_float(value: Optional[str], field: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number.") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update wet-lab outcomes for a run.")
    parser.add_argument("--db", required=True, help="Path to SQLite evidence DB.")
    parser.add_argument("--run-id", required=True, help="Run ID to attach the outcome.")
    parser.add_argument("--arm-id", required=True, help="Experiment arm ID (e.g., A1).")
    parser.add_argument(
        "--any-activity",
        required=True,
        help="Boolean: true/false.",
    )
    parser.add_argument(
        "--target-match",
        default=None,
        help="Boolean: true/false (optional).",
    )
    parser.add_argument(
        "--conversion",
        default=None,
        help="Observed conversion fraction (0-1) or percent; numeric.",
    )
    parser.add_argument(
        "--failure-mode",
        default=None,
        help="Failure mode label (optional).",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Free text notes (optional).",
    )
    args = parser.parse_args(argv)

    try:
        any_activity = _parse_bool(args.any_activity, "--any-activity")
        target_match = _parse_bool(args.target_match, "--target-match")
        conversion = _parse_float(args.conversion, "--conversion")
        failure_mode = args.failure_mode.strip() if args.failure_mode else None
        if failure_mode and failure_mode not in ALLOWED_FAILURE_MODES:
            allowed = ", ".join(sorted(ALLOWED_FAILURE_MODES))
            raise ValueError(f"--failure-mode must be one of: {allowed}.")
        add_outcome(
            args.db,
            args.run_id,
            args.arm_id,
            bool(any_activity),
            target_match,
            conversion,
            failure_mode,
            args.notes,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
