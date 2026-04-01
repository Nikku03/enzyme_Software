"""
Dedicated Colab entrypoint for the hybrid LNN with both imported NEXUS parts
running on the full-xTB hybrid path.

This is the correct Colab route for:
- analogical memory bank
- wave / quantum bridge

because it feeds real xTB atom features into the bridge instead of starving the
wave side on the plain baseline loader.

Run from a Colab cell with:

    exec(open('/content/enzyme_Software/scripts/colab_train_hybrid_lnn.py').read())
"""
from __future__ import annotations

import os
import runpy
import subprocess
import sys
import importlib
import json
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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


LOCKED_PRESET_KEYS = {
    "HYBRID_COLAB_DATASET",
    "HYBRID_COLAB_STRUCTURE_SDF",
    "HYBRID_COLAB_EPOCHS",
    "HYBRID_COLAB_BATCH_SIZE",
    "HYBRID_COLAB_LR",
    "HYBRID_COLAB_WD",
    "HYBRID_COLAB_SPLIT_MODE",
    "HYBRID_COLAB_LIMIT",
    "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING",
    "HYBRID_COLAB_SITE_LABELED_ONLY",
    "HYBRID_COLAB_FREEZE_NEXUS_MEMORY",
    "HYBRID_COLAB_EARLY_STOPPING_PATIENCE",
    "HYBRID_COLAB_EARLY_STOPPING_METRIC",
    "HYBRID_COLAB_INCLUDE_XENOSITE",
    "HYBRID_COLAB_XENOSITE_TOPK",
    "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK",
    "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS",
    "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS",
    "HYBRID_COLAB_SEED",
    "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS",
    "HYBRID_COLAB_TRAIN_RATIO",
    "HYBRID_COLAB_VAL_RATIO",
}


def _apply_preset(preset_values: dict[str, str]) -> list[str]:
    locked = _env_flag("HYBRID_COLAB_LOCK_PRESET_POLICY", True)
    overridden: list[str] = []
    for key, value in preset_values.items():
        if locked and key in LOCKED_PRESET_KEYS:
            previous = os.environ.get(key)
            os.environ[key] = value
            if previous not in {None, "", value}:
                overridden.append(key)
            continue
        _setdefault_env(key, value)
    return overridden


def _resolve_warm_start_from_report(report_path: str, output_dir: str) -> str:
    report_file = Path(report_path)
    if not report_file.exists():
        return ""
    try:
        with report_file.open() as handle:
            report = json.load(handle)
    except Exception:
        return ""

    explicit_keys = (
        "best_checkpoint_path",
        "latest_checkpoint_path",
        "checkpoint_path",
    )
    for key in explicit_keys:
        value = (report.get(key) or "").strip()
        if value and Path(value).exists():
            return value

    candidates: list[Path] = []
    report_parent = report_file.parent
    if report_parent.name:
        attempt_name = report_parent.name
        candidates.extend(
            [
                Path("/content/drive/MyDrive/enzyme_hybrid_lnn/revisit/checkpoints") / attempt_name / "hybrid_full_xtb_best.pt",
                Path("/content/drive/MyDrive/enzyme_hybrid_lnn/revisit/checkpoints") / attempt_name / "hybrid_full_xtb_latest.pt",
                report_parent / "hybrid_full_xtb_best.pt",
                report_parent / "hybrid_full_xtb_latest.pt",
            ]
        )

    episode_log_path = (report.get("episode_log_path") or "").strip()
    if episode_log_path:
        episode_parent = Path(episode_log_path).parent
        attempt_name = episode_parent.name
        candidates.extend(
            [
                Path("/content/drive/MyDrive/enzyme_hybrid_lnn/revisit/checkpoints") / attempt_name / "hybrid_full_xtb_best.pt",
                Path("/content/drive/MyDrive/enzyme_hybrid_lnn/revisit/checkpoints") / attempt_name / "hybrid_full_xtb_latest.pt",
            ]
        )

    out = Path(output_dir)
    candidates.extend(
        [
            out / "hybrid_full_xtb_best.pt",
            out / "hybrid_full_xtb_latest.pt",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def _resolve_warm_start(output_dir: str) -> str:
    explicit = os.environ.get("HYBRID_COLAB_WARM_START", "").strip()
    if explicit:
        return explicit
    report_path = os.environ.get("HYBRID_COLAB_WARM_START_REPORT", "").strip()
    if report_path:
        resolved = _resolve_warm_start_from_report(report_path, output_dir)
        if resolved:
            return resolved
    mode = os.environ.get("HYBRID_COLAB_WARM_START_MODE", "best").strip().lower() or "best"
    out = Path(output_dir)
    best = out / "hybrid_full_xtb_best.pt"
    latest = out / "hybrid_full_xtb_latest.pt"
    if mode == "latest":
        return str(latest)
    if best.exists():
        return str(best)
    return str(latest)


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
    import importlib

    importlib.invalidate_caches()
    try:
        from rdkit import Chem  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime bootstrap failure
        raise RuntimeError(
            "RDKit installation completed but import still failed in the current runtime."
        ) from exc


def _clear_repo_python_caches() -> None:
    for pattern in ("*.pyc",):
        subprocess.run(
            ["find", str(REPO_DIR / "src"), "-name", pattern, "-delete"],
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
        or name == "train_hybrid_full_xtb"
        or name.startswith("scripts.train_hybrid_full_xtb")
    ]
    for name in stale_modules:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    print("Cleared repo bytecode and module caches.", flush=True)


PRESETS: dict[str, dict[str, str]] = {
    "fast": {
        # main7 = 703 molecules (2.5x main6); adds expanded_metx_test + multi-CYP primary rows + AZ120
        "HYBRID_COLAB_DATASET": "data/prepared_training/main7_site_conservative_singlecyp_clean_symm.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "3",
        "HYBRID_COLAB_BATCH_SIZE": "12",
        "HYBRID_COLAB_LR": "2e-4",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "128",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "1",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "0",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "1",
        "HYBRID_COLAB_XENOSITE_TOPK": "1",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "1",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "1",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
    },
    "balanced": {
        # main8 = 933 molecules (main7 703 + CYP_DBs novel 230)
        # Strategy: backbone frozen ALL 50 epochs (prevents scaffold memorisation).
        # Only hybrid heads (council, arbiter, wave, analogical) train.
        # No early stopping: run the full fixed schedule for controlled comparison.
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "50",
        "HYBRID_COLAB_BATCH_SIZE": "16",
        "HYBRID_COLAB_LR": "2e-4",
        "HYBRID_COLAB_WD": "5e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "1",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "0",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "0",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "50",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "1",
        "HYBRID_COLAB_XENOSITE_TOPK": "1",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "1",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "1",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
    },
    "full": {
        "HYBRID_COLAB_DATASET": "data/combined_drugbank_supercyp_full_xtb_valid_site_labeled.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "25",
        "HYBRID_COLAB_BATCH_SIZE": "24",
        "HYBRID_COLAB_LR": "2e-4",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "1",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "1",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "0",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "1",
        "HYBRID_COLAB_XENOSITE_TOPK": "1",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "1",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "1",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
    },
}


def main() -> None:
    os.chdir(REPO_DIR)
    _clear_repo_python_caches()
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("HYBRID_FORCE_MANUAL_OPTIMIZER", "1")
    os.environ["PYTHONPATH"] = f"{SRC_DIR}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    _ensure_rdkit()
    preset = os.environ.get("HYBRID_COLAB_PRESET", "balanced").strip().lower() or "balanced"
    if preset not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown HYBRID_COLAB_PRESET={preset!r}. Valid presets: {valid}")

    overridden_keys = _apply_preset(PRESETS[preset])

    requested_dataset = Path(os.environ["HYBRID_COLAB_DATASET"])
    if not requested_dataset.exists():
        fallback_candidates = [
            REPO_DIR / "data" / "prepared_training" / "main7_site_conservative_singlecyp_clean_symm.json",
            REPO_DIR / "data" / "prepared_training" / "main6_site_conservative_singlecyp_clean_symm.json",
            REPO_DIR / "data" / "prepared_training" / "main5_site_conservative_singlecyp_clean.json",
            REPO_DIR / "data" / "combined_drugbank_supercyp_full_xtb_valid_site_labeled.json",
            REPO_DIR / "data" / "training_dataset_580.json",
        ]
        fallback = next((path for path in fallback_candidates if path.exists()), None)
        if fallback is None:
            raise FileNotFoundError(
                f"Requested dataset not found: {requested_dataset}. No tracked fallback dataset found in repo."
            )
        print(f"Requested dataset not found: {requested_dataset}")
        print(f"Falling back to tracked dataset: {fallback.relative_to(REPO_DIR)}")
        os.environ["HYBRID_COLAB_DATASET"] = str(fallback.relative_to(REPO_DIR))

    output_dir = os.environ.get(
        "HYBRID_COLAB_OUTPUT_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/hybrid_full_xtb",
    )
    artifact_dir = os.environ.get(
        "HYBRID_COLAB_ARTIFACT_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/artifacts/hybrid_full_xtb",
    )
    manual_cache_dir = os.environ.get(
        "HYBRID_COLAB_MANUAL_CACHE_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/cache/manual_engine_full",
    )
    xtb_cache_dir = os.environ.get(
        "HYBRID_COLAB_XTB_CACHE_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/cache/full_xtb",
    )
    checkpoint = _resolve_warm_start(output_dir)
    xenosite_manifest = os.environ.get(
        "HYBRID_COLAB_XENOSITE_MANIFEST",
        "data/xenosite_suppl/manifest.json",
    )

    argv = [
        str(REPO_DIR / "scripts" / "train_hybrid_full_xtb.py"),
        "--dataset",
        os.environ["HYBRID_COLAB_DATASET"],
        "--structure-sdf",
        os.environ["HYBRID_COLAB_STRUCTURE_SDF"],
        "--checkpoint",
        checkpoint,
        "--xtb-cache-dir",
        xtb_cache_dir,
        "--epochs",
        os.environ["HYBRID_COLAB_EPOCHS"],
        "--batch-size",
        os.environ["HYBRID_COLAB_BATCH_SIZE"],
        "--learning-rate",
        os.environ["HYBRID_COLAB_LR"],
        "--weight-decay",
        os.environ["HYBRID_COLAB_WD"],
        "--split-mode",
        os.environ["HYBRID_COLAB_SPLIT_MODE"],
        "--seed",
        os.environ["HYBRID_COLAB_SEED"],
        "--output-dir",
        output_dir,
        "--artifact-dir",
        artifact_dir,
        "--manual-feature-cache-dir",
        manual_cache_dir,
        "--early-stopping-patience",
        os.environ["HYBRID_COLAB_EARLY_STOPPING_PATIENCE"],
        "--early-stopping-metric",
        os.environ["HYBRID_COLAB_EARLY_STOPPING_METRIC"],
        "--train-ratio",
        os.environ.get("HYBRID_COLAB_TRAIN_RATIO", "0.80"),
        "--val-ratio",
        os.environ.get("HYBRID_COLAB_VAL_RATIO", "0.10"),
    ]

    limit = int(os.environ.get("HYBRID_COLAB_LIMIT", "0") or "0")
    if limit > 0:
        argv.extend(["--limit", str(limit)])
    if os.environ.get("HYBRID_COLAB_SITE_LABELED_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--site-labeled-only")
    if os.environ.get("HYBRID_COLAB_COMPUTE_XTB_IF_MISSING", "0").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--compute-xtb-if-missing")
    if os.environ.get("HYBRID_COLAB_FREEZE_NEXUS_MEMORY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--freeze-nexus-memory")
    backbone_freeze = int(os.environ.get("HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS", "0") or "0")
    if backbone_freeze > 0:
        argv.extend(["--backbone-freeze-epochs", str(backbone_freeze)])
    if os.environ.get("HYBRID_COLAB_INCLUDE_XENOSITE", "1").strip().lower() in {"1", "true", "yes", "on"}:
        argv.extend(["--xenosite-manifest", xenosite_manifest])
        argv.extend(["--xenosite-topk", os.environ["HYBRID_COLAB_XENOSITE_TOPK"]])
    if os.environ.get("HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK", "1").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--disable-precedent-logbook")
    precedent_logbook = os.environ.get("HYBRID_COLAB_PRECEDENT_LOGBOOK", "").strip()
    if precedent_logbook:
        argv.extend(["--precedent-logbook", precedent_logbook])

    print("Hybrid LNN Colab wrapper")
    print(f"preset={preset}")
    print(f"output_dir={output_dir}")
    print(f"artifact_dir={artifact_dir}")
    print(f"manual_cache_dir={manual_cache_dir}")
    print(f"xtb_cache_dir={xtb_cache_dir}")
    print(f"warm_start={checkpoint}")
    if os.environ.get("HYBRID_COLAB_WARM_START_REPORT", "").strip():
        print(f"warm_start_report={os.environ['HYBRID_COLAB_WARM_START_REPORT']}")
    print(f"warm_start_mode={os.environ.get('HYBRID_COLAB_WARM_START_MODE', 'best')}")
    print(f"split_mode={os.environ.get('HYBRID_COLAB_SPLIT_MODE', 'scaffold_source_size')}")
    print(f"lock_preset_policy={int(_env_flag('HYBRID_COLAB_LOCK_PRESET_POLICY', True))}")
    print(f"disable_precedent_logbook={os.environ.get('HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK', '1')}")
    print(f"live_wave_vote_inputs={os.environ.get('HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS', '0')}")
    print(f"live_analogical_vote_inputs={os.environ.get('HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS', '0')}")
    print(f"TORCHDYNAMO_DISABLE={os.environ.get('TORCHDYNAMO_DISABLE', '')}")
    print(f"TORCH_COMPILE_DISABLE={os.environ.get('TORCH_COMPILE_DISABLE', '')}")
    print(f"HYBRID_FORCE_MANUAL_OPTIMIZER={os.environ.get('HYBRID_FORCE_MANUAL_OPTIMIZER', '')}")
    if overridden_keys:
        print("preset_policy_overrode=" + ",".join(sorted(overridden_keys)))
    if precedent_logbook:
        print(f"precedent_logbook={precedent_logbook}")
    for key in sorted(PRESETS[preset]):
        print(f"{key}={os.environ[key]}")
    print()

    sys.argv = argv
    runpy.run_path(str(REPO_DIR / "scripts" / "train_hybrid_full_xtb.py"), run_name="__main__")


if __name__ == "__main__":
    main()
