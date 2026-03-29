"""
Offline GFN2-xTB distillation for local quantum targets.

This script builds a per-molecule cache of xTB-derived atomwise quantities and
then packages the dataset into a single `.pt` payload keyed by canonical
SMILES. It is intended as Phase 1 fuel for a future wave/equivariant analogical
engine and deliberately stays outside the main training loop.

Targets captured when available:
  - partial charges
  - Wiberg bond-order summaries per atom
  - vertical Fukui indices f(+), f(-), f(0)
  - HOMO/LUMO gap (parsed from xTB stdout)

Example:
    python scripts/distill_quantum_wave_targets.py \
        --sdf data/ATTNSOM/cyp_dataset/3A4.sdf \
        --output artifacts/quantum/nexus_3a4_quantum_features.pt \
        --cache-dir cache/quantum_wave_xtb
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional dependency
    linear_sum_assignment = None

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception as exc:  # pragma: no cover - environment issue
    raise RuntimeError("RDKit is required for quantum distillation") from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol


_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")
DISTILL_CACHE_VERSION = 2


def _cache_key(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:24]


def _canonical_smiles(mol) -> str:
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), canonical=True, isomericSmiles=True)
    except Exception:
        return Chem.MolToSmiles(Chem.Mol(mol), canonical=True, isomericSmiles=True)


def _ensure_conformer(mol):
    work = Chem.Mol(mol)
    if work.GetNumConformers() > 0:
        return work
    work = Chem.AddHs(work, addCoords=True)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = False
    if int(AllChem.EmbedMolecule(work, params)) != 0:
        if int(AllChem.EmbedMolecule(work, randomSeed=42)) != 0:
            fallback = AllChem.ETKDGv3()
            fallback.randomSeed = 0
            fallback.useRandomCoords = True
            if int(AllChem.EmbedMolecule(work, fallback)) != 0:
                raise RuntimeError("RDKit embedding failed")
    try:
        AllChem.MMFFOptimizeMolecule(work)
    except Exception:
        pass
    return work


def _mol_to_xyz_path(mol, path: Path) -> None:
    xyz_block = Chem.MolToXYZBlock(mol)
    path.write_text(xyz_block, encoding="utf-8")


def _parse_gap_from_stdout(stdout: str) -> float:
    for line in stdout.splitlines():
        if "HOMO-LUMO GAP" not in line.upper():
            continue
        floats = _FLOAT_RE.findall(line)
        if not floats:
            continue
        try:
            return float(floats[-1])
        except ValueError:
            continue
    return 0.0


def _atomic_numbers_and_coords(mol) -> tuple[np.ndarray, np.ndarray]:
    conf = mol.GetConformer()
    atom_numbers = np.asarray([int(atom.GetAtomicNum()) for atom in mol.GetAtoms()], dtype=np.int64)
    coords = np.asarray(conf.GetPositions(), dtype=np.float32)
    return atom_numbers, coords


def _parse_xyz_atoms(path: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        return None
    symbols: list[int] = []
    coords: list[list[float]] = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        atom = Chem.GetPeriodicTable().GetAtomicNumber(parts[0])
        if atom <= 0:
            continue
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            continue
        symbols.append(atom)
        coords.append(xyz)
    if not coords:
        return None
    return np.asarray(symbols, dtype=np.int64), np.asarray(coords, dtype=np.float32)


def _align_xtb_to_rdkit(
    rdkit_atomic_numbers: np.ndarray,
    rdkit_coords: np.ndarray,
    xtb_atomic_numbers: np.ndarray,
    xtb_coords: np.ndarray,
    *,
    threshold: float = 1.0e-2,
) -> tuple[np.ndarray, float, bool]:
    if rdkit_coords.shape != xtb_coords.shape:
        raise ValueError("RDKit/xTB coordinate arrays have different shapes")
    if rdkit_atomic_numbers.shape != xtb_atomic_numbers.shape:
        raise ValueError("RDKit/xTB atom-number arrays have different shapes")

    if np.array_equal(rdkit_atomic_numbers, xtb_atomic_numbers):
        direct = np.linalg.norm(rdkit_coords - xtb_coords, axis=1)
        max_direct = float(direct.max(initial=0.0))
        if max_direct <= threshold:
            identity = np.arange(rdkit_coords.shape[0], dtype=np.int64)
            return identity, max_direct, False

    n_atoms = int(rdkit_coords.shape[0])
    dist = np.linalg.norm(
        rdkit_coords[:, None, :] - xtb_coords[None, :, :],
        axis=-1,
    )
    atom_match = rdkit_atomic_numbers[:, None] == xtb_atomic_numbers[None, :]
    penalty = np.where(atom_match, 0.0, 1.0e6)
    cost = dist + penalty
    assignment = np.full(n_atoms, -1, dtype=np.int64)
    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost)
        if len(row_ind) != n_atoms:
            raise ValueError("Could not build a full RDKit/xTB atom assignment")
        assignment[row_ind] = col_ind
    else:
        used_cols: set[int] = set()
        flat_pairs = sorted(
            (
                (float(cost[i, j]), i, j)
                for i in range(n_atoms)
                for j in range(n_atoms)
                if atom_match[i, j]
            ),
            key=lambda item: item[0],
        )
        for _, i, j in flat_pairs:
            if assignment[i] >= 0 or j in used_cols:
                continue
            assignment[i] = j
            used_cols.add(j)
    if np.any(assignment < 0):
        raise ValueError("Incomplete RDKit/xTB atom assignment")
    assigned_dist = dist[np.arange(n_atoms), assignment]
    if np.any(~atom_match[np.arange(n_atoms), assignment]):
        raise ValueError("RDKit/xTB assignment violated atomic-number consistency")
    max_dist = float(assigned_dist.max(initial=0.0))
    if max_dist > threshold:
        raise ValueError(
            f"RDKit/xTB coordinate alignment exceeded threshold ({max_dist:.6f} > {threshold:.6f})"
        )
    remapped = not np.array_equal(assignment, np.arange(n_atoms, dtype=np.int64))
    return assignment, max_dist, remapped


def _parse_vector_file(path: Path, expected_len: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    values: list[float] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            values.append(float(parts[-1]))
        except ValueError:
            continue
    if len(values) < expected_len:
        return None
    return np.asarray(values[:expected_len], dtype=np.float32)


def _parse_wbo(path: Path, atom_count: int) -> Optional[dict[str, np.ndarray]]:
    if not path.exists():
        return None
    wbo_sum = np.zeros(atom_count, dtype=np.float32)
    wbo_max = np.zeros(atom_count, dtype=np.float32)
    any_found = False
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            a = int(parts[0]) - 1
            b = int(parts[1]) - 1
            value = float(parts[2])
        except ValueError:
            continue
        if 0 <= a < atom_count and 0 <= b < atom_count:
            any_found = True
            wbo_sum[a] += value
            wbo_sum[b] += value
            wbo_max[a] = max(wbo_max[a], value)
            wbo_max[b] = max(wbo_max[b], value)
    if not any_found:
        return None
    return {
        "wbo_sum": wbo_sum,
        "wbo_max": wbo_max,
    }


def _candidate_fukui_files(tmpdir: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in ("fukui", "fukui.out", "vfukui", "vfukui.out", "*fukui*"):
        for path in tmpdir.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def _parse_fukui_table(path: Path, atom_count: int) -> Optional[dict[str, np.ndarray]]:
    lines = path.read_text().splitlines()
    rows: list[tuple[float, float, float]] = []
    in_table = False
    for line in lines:
        if ("f(+)" in line and "f(-)" in line) or ("fukui" in line.lower() and "#" in line):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.strip():
            if rows:
                break
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        floats = []
        for token in parts[1:]:
            try:
                floats.append(float(token))
            except ValueError:
                continue
        if len(floats) >= 3:
            rows.append((floats[-3], floats[-2], floats[-1]))
    if len(rows) < atom_count:
        return None
    arr = np.asarray(rows[:atom_count], dtype=np.float32)
    return {
        "fukui_f_plus": arr[:, 0],
        "fukui_f_minus": arr[:, 1],
        "fukui_f_zero": arr[:, 2],
    }


def _parse_any_fukui(tmpdir: Path, atom_count: int) -> Optional[dict[str, np.ndarray]]:
    for path in _candidate_fukui_files(tmpdir):
        parsed = _parse_fukui_table(path, atom_count)
        if parsed is not None:
            return parsed
    return None


def _invalid_payload(canonical_smiles: str, atom_count: int, *, status: str, error: Optional[str]) -> Dict[str, Any]:
    zeros = np.zeros(atom_count, dtype=np.float32)
    return {
        "cache_version": DISTILL_CACHE_VERSION,
        "canonical_smiles": canonical_smiles,
        "xtb_valid": False,
        "status": status,
        "error": error,
        "atom_count": int(atom_count),
        "atom_numbers": np.zeros(atom_count, dtype=np.int64),
        "xtb_to_rdkit_map": np.arange(atom_count, dtype=np.int64),
        "alignment_max_distance": 0.0,
        "alignment_remapped": False,
        "charges": zeros,
        "wbo_sum": zeros,
        "wbo_max": zeros,
        "fukui_f_plus": zeros,
        "fukui_f_minus": zeros,
        "fukui_f_zero": zeros,
        "homo_lumo_gap_ev": 0.0,
    }


def compute_quantum_wave_payload(
    mol,
    *,
    xtb_path: str = "xtb",
    timeout_s: int = 900,
    solvent: str = "water",
) -> Dict[str, Any]:
    prepared = _ensure_conformer(mol)
    canonical = _canonical_smiles(prepared)
    atom_count = prepared.GetNumAtoms()
    rdkit_atomic_numbers, rdkit_coords = _atomic_numbers_and_coords(prepared)
    if shutil.which(xtb_path) is None:
        return _invalid_payload(
            canonical,
            atom_count,
            status="xtb_unavailable",
            error=f"{xtb_path!r} not found in PATH",
        )

    charge = int(sum(int(atom.GetFormalCharge()) for atom in prepared.GetAtoms()))
    with tempfile.TemporaryDirectory(prefix="quantum_wave_xtb_") as tmp:
        tmpdir = Path(tmp)
        xyz_path = tmpdir / "input.xyz"
        _mol_to_xyz_path(prepared, xyz_path)
        cmd = [
            xtb_path,
            str(xyz_path),
            "--gfn",
            "2",
            "--chrg",
            str(charge),
            "--uhf",
            "0",
            "--vfukui",
            "--wbo",
            "--pop",
        ]
        if solvent:
            cmd.extend(["--alpb", str(solvent)])
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except Exception as exc:
            return _invalid_payload(canonical, atom_count, status="xtb_error", error=str(exc))

        if proc.returncode != 0:
            return _invalid_payload(
                canonical,
                atom_count,
                status="xtb_failed",
                error=(proc.stderr.strip() or proc.stdout.strip() or f"returncode={proc.returncode}")[:2000],
            )

        charges = _parse_vector_file(tmpdir / "charges", atom_count)
        wbo = _parse_wbo(tmpdir / "wbo", atom_count)
        fukui = _parse_any_fukui(tmpdir, atom_count)
        gap = _parse_gap_from_stdout(proc.stdout)
        xtb_xyz = _parse_xyz_atoms(tmpdir / "xtbopt.xyz")
        if xtb_xyz is None:
            xtb_xyz = _parse_xyz_atoms(xyz_path)

    if charges is None:
        return _invalid_payload(
            canonical,
            atom_count,
            status="charges_missing",
            error="xTB completed but charges file was missing or truncated",
        )
    if xtb_xyz is None:
        return _invalid_payload(
            canonical,
            atom_count,
            status="coords_missing",
            error="xTB completed but no coordinate file could be parsed",
        )

    try:
        xtb_atomic_numbers, xtb_coords = xtb_xyz
        xtb_to_rdkit_map, alignment_max_distance, alignment_remapped = _align_xtb_to_rdkit(
            rdkit_atomic_numbers,
            rdkit_coords,
            xtb_atomic_numbers,
            xtb_coords,
        )
    except Exception as exc:
        return _invalid_payload(
            canonical,
            atom_count,
            status="alignment_failed",
            error=str(exc)[:2000],
        )

    charges = np.asarray(charges, dtype=np.float32)[xtb_to_rdkit_map]
    if wbo is not None:
        wbo = {
            "wbo_sum": np.asarray(wbo["wbo_sum"], dtype=np.float32)[xtb_to_rdkit_map],
            "wbo_max": np.asarray(wbo["wbo_max"], dtype=np.float32)[xtb_to_rdkit_map],
        }
    if fukui is not None:
        fukui = {
            key: np.asarray(value, dtype=np.float32)[xtb_to_rdkit_map]
            for key, value in fukui.items()
        }

    payload = {
        "cache_version": DISTILL_CACHE_VERSION,
        "canonical_smiles": canonical,
        "xtb_valid": True,
        "status": "ok",
        "error": None,
        "atom_count": int(atom_count),
        "atom_numbers": rdkit_atomic_numbers.astype(np.int64),
        "xtb_to_rdkit_map": xtb_to_rdkit_map.astype(np.int64),
        "alignment_max_distance": float(alignment_max_distance),
        "alignment_remapped": bool(alignment_remapped),
        "charges": charges,
        "wbo_sum": np.zeros(atom_count, dtype=np.float32) if wbo is None else wbo["wbo_sum"],
        "wbo_max": np.zeros(atom_count, dtype=np.float32) if wbo is None else wbo["wbo_max"],
        "fukui_f_plus": np.zeros(atom_count, dtype=np.float32) if fukui is None else fukui["fukui_f_plus"],
        "fukui_f_minus": np.zeros(atom_count, dtype=np.float32) if fukui is None else fukui["fukui_f_minus"],
        "fukui_f_zero": np.zeros(atom_count, dtype=np.float32) if fukui is None else fukui["fukui_f_zero"],
        "homo_lumo_gap_ev": float(gap),
    }
    return payload


def _json_ready(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _tensorize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cache_version": int(payload.get("cache_version", 0)),
        "canonical_smiles": payload["canonical_smiles"],
        "xtb_valid": bool(payload["xtb_valid"]),
        "status": payload["status"],
        "error": payload["error"],
        "atom_count": int(payload["atom_count"]),
        "atom_numbers": torch.tensor(payload.get("atom_numbers", []), dtype=torch.long),
        "xtb_to_rdkit_map": torch.tensor(payload.get("xtb_to_rdkit_map", []), dtype=torch.long),
        "alignment_max_distance": torch.tensor(float(payload.get("alignment_max_distance", 0.0)), dtype=torch.float32),
        "alignment_remapped": bool(payload.get("alignment_remapped", False)),
        "charges": torch.tensor(payload["charges"], dtype=torch.float32),
        "wbo_sum": torch.tensor(payload["wbo_sum"], dtype=torch.float32),
        "wbo_max": torch.tensor(payload["wbo_max"], dtype=torch.float32),
        "fukui_f_plus": torch.tensor(payload["fukui_f_plus"], dtype=torch.float32),
        "fukui_f_minus": torch.tensor(payload["fukui_f_minus"], dtype=torch.float32),
        "fukui_f_zero": torch.tensor(payload["fukui_f_zero"], dtype=torch.float32),
        "homo_lumo_gap_ev": torch.tensor(float(payload["homo_lumo_gap_ev"]), dtype=torch.float32),
    }


def _iter_supplier(sdf_path: Path):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for idx, mol in enumerate(supplier):
        yield idx, mol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", required=True, help="Input SDF file")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--cache-dir", default="cache/quantum_wave_xtb")
    parser.add_argument("--xtb-path", default="xtb")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--solvent", default="water")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    sdf_path = Path(args.sdf)
    output_path = Path(args.output)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not sdf_path.exists():
        raise FileNotFoundError(
            f"SDF not found at {sdf_path}. "
            "If running on Colab, bootstrap ATTNSOM assets first with "
            "scripts/setup_colab_nexus.sh."
        )

    print(f"Quantum distillation target: {sdf_path}")
    print(f"Output: {output_path}")
    print(f"Per-molecule cache: {cache_dir}")
    print(f"xTB binary: {args.xtb_path}")

    quantum_dataset: Dict[str, Dict[str, Any]] = {}
    ok = skipped = failed = 0

    iterator = _iter_supplier(sdf_path)
    if tqdm is not None:
        iterator = tqdm(iterator)

    for idx, mol in iterator:
        if args.limit > 0 and idx >= args.limit:
            break
        if mol is None:
            failed += 1
            continue

        try:
            canonical = _canonical_smiles(mol)
        except Exception as exc:
            failed += 1
            print(f"[{idx}] canonicalisation failed: {exc}")
            continue

        cache_path = cache_dir / f"{_cache_key(canonical)}.json"
        payload = None
        if args.skip_existing and cache_path.exists():
            cached_payload = json.loads(cache_path.read_text())
            if int(cached_payload.get("cache_version", 0)) == DISTILL_CACHE_VERSION:
                payload = cached_payload
                skipped += 1
        if payload is None:
            payload = _json_ready(
                compute_quantum_wave_payload(
                    mol,
                    xtb_path=args.xtb_path,
                    timeout_s=args.timeout_s,
                    solvent=args.solvent,
                )
            )
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        quantum_dataset[canonical] = _tensorize_payload(payload)
        if payload.get("xtb_valid"):
            ok += 1
        else:
            failed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantum_dataset, output_path)
    print(f"\nSaved {len(quantum_dataset)} molecules to {output_path}")
    print(f"ok={ok}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
