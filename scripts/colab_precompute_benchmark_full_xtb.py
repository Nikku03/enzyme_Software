"""
Colab helper for precomputing full xTB cache for the external benchmark sets.

Run from a Colab cell with:

    exec(open('/content/enzyme_Software/scripts/colab_precompute_benchmark_full_xtb.py').read(), {'__name__': '__main__'})
"""
from __future__ import annotations

import importlib
import json
import os
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_DIR = Path("/content/enzyme_Software")
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _setdefault_env(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


def _ensure_rdkit() -> None:
    try:
        from rdkit import Chem  # noqa: F401
        return
    except Exception:
        pass
    print("RDKit not found. Installing rdkit-pypi for this Colab runtime...", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "rdkit-pypi"],
        cwd=str(REPO_DIR),
    )
    importlib.invalidate_caches()
    try:
        from rdkit import Chem  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "RDKit installation completed but import still failed in the current runtime."
        ) from exc


def _clear_repo_python_caches() -> None:
    subprocess.run(
        ["find", str(REPO_DIR / "src"), "-name", "*.pyc", "-delete"],
        check=False,
    )
    subprocess.run(
        [
            "find",
            str(REPO_DIR / "src"),
            "-name",
            "__pycache__",
            "-type",
            "d",
            "-exec",
            "rm",
            "-rf",
            "{}",
            "+",
        ],
        check=False,
    )
    stale_modules = [
        name
        for name in list(sys.modules)
        if name == "enzyme_software"
        or name.startswith("enzyme_software.")
        or name.startswith("scripts.precompute_full_xtb")
    ]
    for name in stale_modules:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    print("Cleared repo bytecode and module caches.", flush=True)


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return list(payload.get("drugs", []))
    return list(payload)


def _canonical_smiles(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return " ".join(smiles.split())
    return Chem.MolToSmiles(mol, canonical=True)


def _resolve_dataset_path(path_str: str, *, benchmark_dir: Path) -> Path:
    raw = Path(path_str)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.append(REPO_DIR / raw)
        candidates.append(benchmark_dir / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _maybe_build_benchmark_sets(*, benchmark_dir: Path, holdout_path: Path) -> None:
    builder = REPO_DIR / "scripts" / "build_main8_benchmark_sets.py"
    if not builder.exists() or not holdout_path.exists():
        return
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    print(f"Benchmark datasets missing. Building A/B/C from holdout: {holdout_path}", flush=True)
    sys.argv = [
        str(builder),
        "--input",
        str(holdout_path),
        "--output-dir",
        str(benchmark_dir),
    ]
    runpy.run_path(str(builder), run_name="__main__")
    print()


def main() -> None:
    os.chdir(REPO_DIR)
    _clear_repo_python_caches()
    _ensure_rdkit()

    from enzyme_software.liquid_nn_v2.features.xtb_features import load_or_compute_full_xtb_features

    _setdefault_env("HYBRID_COLAB_BENCHMARK_XTB_CACHE_DIR", "/content/drive/MyDrive/enzyme_hybrid_lnn/cache/full_xtb")
    _setdefault_env("HYBRID_COLAB_BENCHMARK_DIR", "/content/drive/MyDrive/enzyme_hybrid_lnn/benchmarks")
    _setdefault_env(
        "HYBRID_COLAB_BENCHMARK_HOLDOUT",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/benchmarks/main8_benchmark_holdout_singlecyp.json",
    )
    _setdefault_env(
        "HYBRID_COLAB_BENCHMARK_XTB_DATASETS",
        ",".join(
            [
                "/content/drive/MyDrive/enzyme_hybrid_lnn/benchmarks/main8_benchmark_a_row_level_singlecyp.json",
                "/content/drive/MyDrive/enzyme_hybrid_lnn/benchmarks/main8_benchmark_b_unique_molecules.json",
                "/content/drive/MyDrive/enzyme_hybrid_lnn/benchmarks/main8_benchmark_c_high_confidence.json",
            ]
        ),
    )
    _setdefault_env(
        "HYBRID_COLAB_BENCHMARK_XTB_SUMMARY_JSON",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/artifacts/hybrid_full_xtb/benchmark_full_xtb_summary.json",
    )

    cache_dir = Path(os.environ["HYBRID_COLAB_BENCHMARK_XTB_CACHE_DIR"])
    benchmark_dir = Path(os.environ["HYBRID_COLAB_BENCHMARK_DIR"])
    holdout_path = Path(os.environ["HYBRID_COLAB_BENCHMARK_HOLDOUT"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(os.environ["HYBRID_COLAB_BENCHMARK_XTB_SUMMARY_JSON"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    requested_paths = [part.strip() for part in os.environ["HYBRID_COLAB_BENCHMARK_XTB_DATASETS"].split(",") if part.strip()]
    dataset_paths = [_resolve_dataset_path(part, benchmark_dir=benchmark_dir) for part in requested_paths]
    if any(not path.exists() for path in dataset_paths):
        _maybe_build_benchmark_sets(benchmark_dir=benchmark_dir, holdout_path=holdout_path)
        dataset_paths = [_resolve_dataset_path(part, benchmark_dir=benchmark_dir) for part in requested_paths]
    missing = [path for path in dataset_paths if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Benchmark dataset(s) not found. "
            f"Missing: {missing_str}. "
            f"Upload the benchmark JSONs to {benchmark_dir} or upload the row-level holdout to {holdout_path} "
            "so the helper can derive A/B/C automatically."
        )

    print("Hybrid Benchmark full-xTB precompute helper", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"benchmark_dir={benchmark_dir}", flush=True)
    print(f"holdout_path={holdout_path}", flush=True)
    print(f"summary_json={summary_path}", flush=True)
    print("datasets:", flush=True)
    for path in dataset_paths:
        print(f"  {path}", flush=True)
    print()

    unique_smiles_seen: set[str] = set()
    dataset_summaries: dict[str, dict[str, object]] = {}
    global_statuses: Counter[str] = Counter()
    total_unique_ok = 0
    total_unique_failed = 0

    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        rows = _load_rows(dataset_path)
        dataset_statuses: Counter[str] = Counter()
        dataset_unique_smiles: list[str] = []
        for row in rows:
            smiles = str(row.get("smiles", "")).strip()
            if not smiles:
                continue
            canonical = _canonical_smiles(smiles)
            if canonical not in dataset_unique_smiles:
                dataset_unique_smiles.append(canonical)

        dataset_ok = 0
        dataset_failed = 0
        print(f"Precomputing {dataset_path} | rows={len(rows)} | unique_smiles={len(dataset_unique_smiles)}", flush=True)
        for index, smiles in enumerate(dataset_unique_smiles, start=1):
            payload = load_or_compute_full_xtb_features(
                smiles,
                cache_dir=cache_dir,
                compute_if_missing=True,
            )
            status = str(payload.get("status") or "unknown")
            dataset_statuses[status] += 1
            if smiles not in unique_smiles_seen:
                unique_smiles_seen.add(smiles)
                global_statuses[status] += 1
                if bool(payload.get("true_xtb_valid", payload.get("xtb_valid"))):
                    total_unique_ok += 1
                else:
                    total_unique_failed += 1
            if bool(payload.get("true_xtb_valid", payload.get("xtb_valid"))):
                dataset_ok += 1
            else:
                dataset_failed += 1
            if index % 10 == 0 or index == len(dataset_unique_smiles):
                print(
                    f"  {index}/{len(dataset_unique_smiles)} processed | ok={dataset_ok} | failed={dataset_failed}",
                    flush=True,
                )

        dataset_summaries[str(dataset_path)] = {
            "rows": len(rows),
            "unique_smiles": len(dataset_unique_smiles),
            "ok": dataset_ok,
            "failed": dataset_failed,
            "statuses": dict(sorted(dataset_statuses.items())),
        }
        print(f"  status summary: {dict(sorted(dataset_statuses.items()))}", flush=True)
        print()

    summary = {
        "cache_dir": str(cache_dir),
        "datasets": dataset_summaries,
        "global_unique_smiles": len(unique_smiles_seen),
        "global_ok": total_unique_ok,
        "global_failed": total_unique_failed,
        "global_statuses": dict(sorted(global_statuses.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60, flush=True)
    print("BENCHMARK FULL-XTB PRECOMPUTE COMPLETE", flush=True)
    print("=" * 60, flush=True)
    print(f"Global unique smiles: {summary['global_unique_smiles']}", flush=True)
    print(f"Global ok: {summary['global_ok']} | failed: {summary['global_failed']}", flush=True)
    print(f"Global statuses: {summary['global_statuses']}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
