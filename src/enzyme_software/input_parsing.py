from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional, Tuple


@dataclass
class ParsedEntry:
    payload: str
    label: str
    kind: str


def load_smiles_from_file(path: str, select_index: Optional[int] = None) -> Tuple[str, List[str]]:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    payload = file_path.read_text(encoding="utf-8", errors="replace")
    entries, warnings = parse_input_payload(payload, filename=file_path.name)
    if not entries:
        raise ValueError("No structure records or SMILES entries found.")

    if select_index is None:
        if len(entries) > 1:
            warnings.append("Multiple entries found; using the first.")
        return entries[0].payload, warnings

    if select_index < 0 or select_index >= len(entries):
        raise IndexError("Selected entry index is out of range.")
    return entries[select_index].payload, warnings


def parse_input_payload(payload: str, filename: Optional[str] = None) -> Tuple[List[ParsedEntry], List[str]]:
    warnings: List[str] = []
    suffix = Path(filename).suffix.lower() if filename else ""
    treat_as_structure = suffix in {".sdf", ".mol"} or looks_like_structure_block(payload)

    if treat_as_structure:
        entries = split_structure_blocks(payload)
        if entries:
            return entries, warnings
        if suffix in {".sdf", ".mol"}:
            warnings.append("No SDF/MOL records found; falling back to SMILES parsing.")

    smiles_entries = parse_smiles_entries(payload)
    if smiles_entries:
        return smiles_entries, warnings

    raise ValueError("No structure records or SMILES entries found.")


def looks_like_structure_block(payload: str) -> bool:
    markers = ("M  END", "V2000", "V3000", "$$$$")
    return any(marker in payload for marker in markers)


def split_structure_blocks(payload: str) -> List[ParsedEntry]:
    blocks = payload.split("$$$$") if "$$$$" in payload else [payload]
    entries: List[ParsedEntry] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if not looks_like_structure_block(block):
            continue
        title = block.splitlines()[0].strip() if block.splitlines() else "Untitled"
        entries.append(
            ParsedEntry(
                payload=block + "\n",
                label=title or "Untitled",
                kind="molblock",
            )
        )
    return entries


def parse_smiles_entries(payload: str) -> List[ParsedEntry]:
    entries: List[ParsedEntry] = []
    for line in payload.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        smiles = parts[0]
        name = " ".join(parts[1:]) if len(parts) > 1 else None
        label = f"{name} ({smiles})" if name else smiles
        entries.append(ParsedEntry(payload=smiles, label=label, kind="smiles"))
    return entries


def parse_bond_indices(text: str) -> Optional[List[int]]:
    if text is None:
        return None
    match = re.match(r"^\s*\[?\s*(\d+)\s*[,;:-]\s*(\d+)\s*\]?\s*$", text)
    if not match:
        return None
    return [int(match.group(1)), int(match.group(2))]
