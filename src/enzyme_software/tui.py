from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from enzyme_software.context import OperationalConstraints
from enzyme_software.input_parsing import ParsedEntry, parse_input_payload
from enzyme_software.pipeline import run_pipeline


def run_terminal_ui() -> int:
    print("BondBreak v1 - Strategy Router UI")
    print("Provide a SMILES/SDF/MOL file path or paste a SMILES string.")
    try:
        smiles, warnings = _prompt_smiles_payload()
        for warning in warnings:
            print(f"Warning: {warning}")
        target_bond = _prompt_nonempty("Target bond (e.g., C-O or 3-7): ")
        requested_output = _prompt_optional(
            "Requested output (e.g., -CF3, CF3 radical) [optional]: "
        )
        trap_target = None
        if requested_output:
            trap_target = _prompt_optional(
                "Trap/acceptor target for reagent generation [optional]: "
            )
        constraints = _prompt_constraints()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130

    ctx = run_pipeline(
        smiles,
        target_bond,
        requested_output=requested_output,
        trap_target=trap_target,
        constraints=constraints,
    )
    print(json.dumps(ctx.to_dict(), indent=2))
    return 0


def _prompt_smiles_payload() -> Tuple[str, List[str]]:
    while True:
        path = input("Path to SMILES/SDF/MOL file (blank to paste SMILES): ").strip()
        if not path:
            smiles = _prompt_nonempty("Paste SMILES string: ")
            return smiles, []
        try:
            file_path = Path(path).expanduser()
            if not file_path.is_file():
                raise FileNotFoundError(f"File not found: {file_path}")
            payload = file_path.read_text(encoding="utf-8", errors="replace")
            entries, warnings = parse_input_payload(payload, filename=file_path.name)
            if len(entries) == 1:
                return entries[0].payload, warnings
            index = _select_entry(entries)
            return entries[index].payload, warnings
        except FileNotFoundError as exc:
            print(str(exc))
        except ValueError as exc:
            print(f"Could not read file: {exc}")


def _prompt_constraints() -> OperationalConstraints:
    ph_min = _prompt_float_optional("Minimum pH (blank to skip): ")
    ph_max = _prompt_float_optional("Maximum pH (blank to skip): ")
    temperature_c = _prompt_float_optional("Temperature C (blank to skip): ")
    metals_allowed = _prompt_bool_optional("Allow metal cofactors? (y/n/blank): ")
    oxidation_allowed = _prompt_bool_optional("Allow oxidation? (y/n/blank): ")
    host = _prompt_optional("Host organism (blank to skip): ")
    return OperationalConstraints(
        ph_min=ph_min,
        ph_max=ph_max,
        temperature_c=temperature_c,
        metals_allowed=metals_allowed,
        oxidation_allowed=oxidation_allowed,
        host=host,
    )


def _prompt_nonempty(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Value required.")


def _prompt_optional(prompt: str) -> Optional[str]:
    value = input(prompt).strip()
    return value or None


def _prompt_float_optional(prompt: str) -> Optional[float]:
    while True:
        value = input(prompt).strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            print("Enter a numeric value or leave blank.")


def _prompt_bool_optional(prompt: str) -> Optional[bool]:
    while True:
        value = input(prompt).strip().lower()
        if not value:
            return None
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Enter y, n, or leave blank.")


def _select_entry(entries: Sequence[ParsedEntry]) -> int:
    print(f"Found {len(entries)} entries.")
    _preview_entries([entry.label for entry in entries])
    return _prompt_index(len(entries))


def _preview_entries(labels: Sequence[str], limit: int = 10) -> None:
    for idx, label in enumerate(labels[:limit], start=1):
        print(f"{idx:>3}: {label}")
    if len(labels) > limit:
        print(f"... {len(labels) - limit} more not shown")


def _prompt_index(max_index: int) -> int:
    while True:
        value = input(f"Select entry (1-{max_index}): ").strip().lower()
        if value in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if value.isdigit():
            index = int(value)
            if 1 <= index <= max_index:
                return index - 1
        print("Invalid selection.")
