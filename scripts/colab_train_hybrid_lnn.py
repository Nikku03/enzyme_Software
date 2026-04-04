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
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


REPO_DIR = Path("/content/enzyme_Software")
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
SCRIPT_DIR = REPO_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parse_hybrid_console_log import parse_console_line


def _setdefault_env(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class _TeeStream:
    def __init__(self, *streams, jsonl_handle=None, stream_name="stdout"):
        self._streams = streams
        self._jsonl_handle = jsonl_handle
        self._stream_name = stream_name
        self._buffer = ""

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        if self._jsonl_handle is not None and data:
            self._buffer += data
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                record = parse_console_line(line + "\n")
                record["stream"] = self._stream_name
                self._jsonl_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return len(data)

    def flush(self):
        if self._jsonl_handle is not None and self._buffer:
            record = parse_console_line(self._buffer)
            record["stream"] = self._stream_name
            self._jsonl_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            self._buffer = ""
        for stream in self._streams:
            stream.flush()
        if self._jsonl_handle is not None:
            self._jsonl_handle.flush()

    def isatty(self):
        primary = self._streams[0]
        return bool(getattr(primary, "isatty", lambda: False)())

    @property
    def encoding(self):
        primary = self._streams[0]
        return getattr(primary, "encoding", "utf-8")


@contextmanager
def _capture_console_log(log_path: Path, jsonl_path: Path | None = None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8", buffering=1) as handle, (
        jsonl_path.open("a", encoding="utf-8", buffering=1) if jsonl_path is not None else open(os.devnull, "w", encoding="utf-8")
    ) as jsonl_handle:
        active_jsonl = None if jsonl_path is None else jsonl_handle
        sys.stdout = _TeeStream(original_stdout, handle, jsonl_handle=active_jsonl, stream_name="stdout")
        sys.stderr = _TeeStream(original_stderr, handle, jsonl_handle=active_jsonl, stream_name="stderr")
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


LOCKED_PRESET_KEYS = {
    "HYBRID_COLAB_DATASET",
    "HYBRID_COLAB_STRUCTURE_SDF",
    "HYBRID_COLAB_SPLIT_MODE",
    "HYBRID_COLAB_LIMIT",
    "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING",
    "HYBRID_COLAB_SITE_LABELED_ONLY",
    "HYBRID_COLAB_INCLUDE_XENOSITE",
    "HYBRID_COLAB_XENOSITE_TOPK",
    "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK",
    "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS",
    "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS",
    "HYBRID_COLAB_SEED",
    "HYBRID_COLAB_TRAIN_RATIO",
    "HYBRID_COLAB_VAL_RATIO",
    "HYBRID_COLAB_TARGET_CYP",
    "HYBRID_COLAB_CONFIDENCE_ALLOWLIST",
    "HYBRID_COLAB_TRAIN_SOURCE_ALLOWLIST",
    "HYBRID_COLAB_BASE_LNN_FIRST",
    "HYBRID_COLAB_NEXUS_SIDEINFO_ONLY",
    "HYBRID_COLAB_USE_CANDIDATE_MASK",
    "HYBRID_COLAB_CANDIDATE_MASK_MODE",
    "HYBRID_COLAB_CANDIDATE_MASK_LOGIT_BIAS",
    "HYBRID_COLAB_BALANCE_TRAIN_SOURCES",
    "HYBRID_COLAB_FREEZE_BASE_MODULES",
    "HYBRID_COLAB_SITE_ONLY_TARGET_CYP",
    "HYBRID_COLAB_USE_TOPK_RERANKER",
    "HYBRID_COLAB_TOPK_RERANKER_K",
    "HYBRID_COLAB_TOPK_RERANKER_HIDDEN_DIM",
    "HYBRID_COLAB_TOPK_RERANKER_HEADS",
    "HYBRID_COLAB_TOPK_RERANKER_LAYERS",
    "HYBRID_COLAB_TOPK_RERANKER_DROPOUT",
    "HYBRID_COLAB_TOPK_RERANKER_RESIDUAL_SCALE",
    "HYBRID_COLAB_TOPK_RERANKER_LR_SCALE",
    "HYBRID_COLAB_TOPK_RERANKER_HEADSTART_EPOCHS",
    "HYBRID_COLAB_BENCHMARK_DATASETS",
    "HYBRID_COLAB_BENCHMARK_BATCH_SIZE",
    "HYBRID_COLAB_BENCHMARK_EVERY",
    "HYBRID_COLAB_BENCHMARK_SELECTION_METRIC",
    "HYBRID_COLAB_BENCHMARK_SELECTION_WEIGHT",
    "HYBRID_COLAB_USE_SOURCE_SITE_HEADS",
    "HYBRID_COLAB_SOURCE_SITE_AUX_WEIGHT",
    "HYBRID_COLAB_SOURCE_SITE_BLEND_WEIGHT",
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
    argv_backup = list(sys.argv)
    try:
        sys.argv = [
            str(builder),
            "--input",
            str(holdout_path),
            "--output-dir",
            str(benchmark_dir),
        ]
        runpy.run_path(str(builder), run_name="__main__")
    finally:
        sys.argv = argv_backup
    print()


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
        # Strategy: short backbone freeze for stabilization, then full-model
        # finetuning on the scaffold/source/size split.
        # Use early stopping because recent runs overfit after thaw.
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "40",
        "HYBRID_COLAB_BATCH_SIZE": "16",
        "HYBRID_COLAB_LR": "3e-5",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "0",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "8",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "5",
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
    "cyp3a4_base": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "35",
        "HYBRID_COLAB_BATCH_SIZE": "12",
        "HYBRID_COLAB_LR": "2e-5",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "1",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "8",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "3",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "0",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "0",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "0",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
        "HYBRID_COLAB_TARGET_CYP": "CYP3A4",
        "HYBRID_COLAB_CONFIDENCE_ALLOWLIST": "high,validated,validated_gold,validated_literature,curated",
        "HYBRID_COLAB_BASE_LNN_FIRST": "1",
        "HYBRID_COLAB_USE_CANDIDATE_MASK": "1",
        "HYBRID_COLAB_BALANCE_TRAIN_SOURCES": "1",
    },
    "cyp3a4_sideinfo": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "35",
        "HYBRID_COLAB_BATCH_SIZE": "12",
        "HYBRID_COLAB_LR": "2e-5",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "0",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "8",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "3",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "0",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "0",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "0",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
        "HYBRID_COLAB_TARGET_CYP": "CYP3A4",
        "HYBRID_COLAB_CONFIDENCE_ALLOWLIST": "high,validated,validated_gold,validated_literature,curated",
        "HYBRID_COLAB_NEXUS_SIDEINFO_ONLY": "1",
        "HYBRID_COLAB_USE_CANDIDATE_MASK": "1",
        "HYBRID_COLAB_CANDIDATE_MASK_MODE": "soft",
        "HYBRID_COLAB_CANDIDATE_MASK_LOGIT_BIAS": "1.50",
        "HYBRID_COLAB_BALANCE_TRAIN_SOURCES": "0",
        "HYBRID_COLAB_SITE_ONLY_TARGET_CYP": "1",
    },
    "cyp3a4_sideinfo_augmented": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_cyp3a4_augmented.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "35",
        "HYBRID_COLAB_BATCH_SIZE": "12",
        "HYBRID_COLAB_LR": "2e-5",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "0",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "8",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "3",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "0",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "0",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "0",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
        "HYBRID_COLAB_TARGET_CYP": "CYP3A4",
        "HYBRID_COLAB_NEXUS_SIDEINFO_ONLY": "1",
        "HYBRID_COLAB_USE_CANDIDATE_MASK": "1",
        "HYBRID_COLAB_CANDIDATE_MASK_MODE": "soft",
        "HYBRID_COLAB_CANDIDATE_MASK_LOGIT_BIAS": "1.35",
        "HYBRID_COLAB_BALANCE_TRAIN_SOURCES": "0",
        "HYBRID_COLAB_SITE_ONLY_TARGET_CYP": "1",
        "HYBRID_COLAB_SITE_RANKING_WEIGHT": "0.85",
        "HYBRID_COLAB_SITE_HARD_NEGATIVE_FRACTION": "0.75",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_TOPK": "3",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_DECAY": "0.70",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DEFAULT": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DRUGBANK": "0.95",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_AZ120": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_METXBIODB": "1.15",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_ATTNSOM": "1.35",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_CYP_DBS_EXTERNAL": "1.75",
        "HYBRID_COLAB_DOMAIN_ADV_WEIGHT": "0.06",
        "HYBRID_COLAB_DOMAIN_ADV_GRAD_SCALE": "0.15",
        "HYBRID_COLAB_DOMAIN_ADV_HIDDEN_DIM": "96",
    },
    "multicyp_sideinfo_fullrank_pretrain": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_multicyp_attnsom_sourceaware.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "20",
        "HYBRID_COLAB_BATCH_SIZE": "16",
        "HYBRID_COLAB_LR": "2e-5",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "0",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "6",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1_all",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "2",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "0",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "0",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "0",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
        "HYBRID_COLAB_NEXUS_SIDEINFO_ONLY": "1",
        "HYBRID_COLAB_USE_CANDIDATE_MASK": "0",
        "HYBRID_COLAB_BALANCE_TRAIN_SOURCES": "1",
        "HYBRID_COLAB_SITE_ONLY_TARGET_CYP": "0",
        "HYBRID_COLAB_SITE_RANKING_WEIGHT": "0.80",
        "HYBRID_COLAB_SITE_HARD_NEGATIVE_FRACTION": "0.70",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_TOPK": "2",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_DECAY": "0.85",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DEFAULT": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DRUGBANK": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_AZ120": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_METXBIODB": "1.05",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_ATTNSOM": "1.10",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_CYP_DBS_EXTERNAL": "1.20",
        "HYBRID_COLAB_DOMAIN_ADV_WEIGHT": "0.0",
        "HYBRID_COLAB_SOURCE_ALIGN_WEIGHT": "0.0",
        "HYBRID_COLAB_SOURCE_ALIGN_COV_WEIGHT": "0.0",
        "HYBRID_COLAB_NEXUS_LIVE_WAVE_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_LIVE_ANALOGICAL_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_WAVE_SIDEINFO_AUX_WEIGHT": "0.03",
        "HYBRID_COLAB_NEXUS_ANALOGICAL_SIDEINFO_AUX_WEIGHT": "0.05",
    },
    "multicyp_sideinfo_fullrank_benchmark_ready": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_multicyp_attnsom_sourceaware.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "20",
        "HYBRID_COLAB_BATCH_SIZE": "16",
        "HYBRID_COLAB_LR": "2e-5",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "0",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "6",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1_all",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "2",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "0",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "0",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "0",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
        "HYBRID_COLAB_NEXUS_SIDEINFO_ONLY": "1",
        "HYBRID_COLAB_USE_CANDIDATE_MASK": "0",
        "HYBRID_COLAB_BALANCE_TRAIN_SOURCES": "1",
        "HYBRID_COLAB_SITE_ONLY_TARGET_CYP": "0",
        "HYBRID_COLAB_SITE_RANKING_WEIGHT": "0.80",
        "HYBRID_COLAB_SITE_HARD_NEGATIVE_FRACTION": "0.70",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_TOPK": "2",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_DECAY": "0.85",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DEFAULT": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DRUGBANK": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_AZ120": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_METXBIODB": "1.05",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_ATTNSOM": "1.10",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_CYP_DBS_EXTERNAL": "1.20",
        "HYBRID_COLAB_DOMAIN_ADV_WEIGHT": "0.0",
        "HYBRID_COLAB_SOURCE_ALIGN_WEIGHT": "0.0",
        "HYBRID_COLAB_SOURCE_ALIGN_COV_WEIGHT": "0.0",
        "HYBRID_COLAB_NEXUS_LIVE_WAVE_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_LIVE_ANALOGICAL_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_WAVE_SIDEINFO_AUX_WEIGHT": "0.03",
        "HYBRID_COLAB_NEXUS_ANALOGICAL_SIDEINFO_AUX_WEIGHT": "0.05",
        "HYBRID_COLAB_USE_SOURCE_SITE_HEADS": "1",
        "HYBRID_COLAB_SOURCE_SITE_AUX_WEIGHT": "0.20",
        "HYBRID_COLAB_SOURCE_SITE_BLEND_WEIGHT": "0.30",
        "HYBRID_COLAB_BENCHMARK_DATASETS": "data/prepared_training/main8_benchmark_a_row_level_singlecyp.json,data/prepared_training/main8_benchmark_b_unique_molecules.json,data/prepared_training/main8_benchmark_c_high_confidence.json",
        "HYBRID_COLAB_BENCHMARK_BATCH_SIZE": "16",
        "HYBRID_COLAB_BENCHMARK_EVERY": "1",
        "HYBRID_COLAB_BENCHMARK_SELECTION_METRIC": "site_top1_acc_all_molecules",
        "HYBRID_COLAB_BENCHMARK_SELECTION_WEIGHT": "0.75",
    },
    "cyp3a4_sideinfo_fullrank_from_multicyp": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_cyp3a4_augmented.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "20",
        "HYBRID_COLAB_BATCH_SIZE": "12",
        "HYBRID_COLAB_LR": "1.2e-5",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "0",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "6",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1_all",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "2",
        "HYBRID_COLAB_BACKBONE_THAW_LR_SCALE": "0.05",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "0",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "0",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "0",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
        "HYBRID_COLAB_TARGET_CYP": "CYP3A4",
        "HYBRID_COLAB_NEXUS_SIDEINFO_ONLY": "1",
        "HYBRID_COLAB_USE_CANDIDATE_MASK": "0",
        "HYBRID_COLAB_BALANCE_TRAIN_SOURCES": "1",
        "HYBRID_COLAB_SITE_ONLY_TARGET_CYP": "1",
        "HYBRID_COLAB_SITE_RANKING_WEIGHT": "0.80",
        "HYBRID_COLAB_SITE_HARD_NEGATIVE_FRACTION": "0.70",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_TOPK": "2",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_DECAY": "0.85",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DEFAULT": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DRUGBANK": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_AZ120": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_METXBIODB": "1.10",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_ATTNSOM": "1.20",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_CYP_DBS_EXTERNAL": "1.35",
        "HYBRID_COLAB_DOMAIN_ADV_WEIGHT": "0.0",
        "HYBRID_COLAB_SOURCE_ALIGN_WEIGHT": "0.0",
        "HYBRID_COLAB_SOURCE_ALIGN_COV_WEIGHT": "0.0",
        "HYBRID_COLAB_NEXUS_LIVE_WAVE_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_LIVE_ANALOGICAL_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_WAVE_SIDEINFO_AUX_WEIGHT": "0.03",
        "HYBRID_COLAB_NEXUS_ANALOGICAL_SIDEINFO_AUX_WEIGHT": "0.05",
    },
    "cyp3a4_sideinfo_fullrank": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_cyp3a4_augmented.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "35",
        "HYBRID_COLAB_BATCH_SIZE": "12",
        "HYBRID_COLAB_LR": "2e-5",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "0",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "8",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1_all",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "3",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "0",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "0",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "0",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
        "HYBRID_COLAB_TARGET_CYP": "CYP3A4",
        "HYBRID_COLAB_NEXUS_SIDEINFO_ONLY": "1",
        "HYBRID_COLAB_USE_CANDIDATE_MASK": "0",
        "HYBRID_COLAB_BALANCE_TRAIN_SOURCES": "1",
        "HYBRID_COLAB_SITE_ONLY_TARGET_CYP": "1",
        "HYBRID_COLAB_SITE_RANKING_WEIGHT": "0.80",
        "HYBRID_COLAB_SITE_HARD_NEGATIVE_FRACTION": "0.70",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_TOPK": "2",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_DECAY": "0.85",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DEFAULT": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DRUGBANK": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_AZ120": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_METXBIODB": "1.10",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_ATTNSOM": "1.20",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_CYP_DBS_EXTERNAL": "1.35",
        "HYBRID_COLAB_DOMAIN_ADV_WEIGHT": "0.0",
        "HYBRID_COLAB_SOURCE_ALIGN_WEIGHT": "0.01",
        "HYBRID_COLAB_SOURCE_ALIGN_COV_WEIGHT": "0.35",
        "HYBRID_COLAB_NEXUS_LIVE_WAVE_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_LIVE_ANALOGICAL_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_WAVE_SIDEINFO_AUX_WEIGHT": "0.03",
        "HYBRID_COLAB_NEXUS_ANALOGICAL_SIDEINFO_AUX_WEIGHT": "0.05",
    },
    "cyp3a4_sideinfo_fullrank_topk": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_cyp3a4_augmented.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "35",
        "HYBRID_COLAB_BATCH_SIZE": "12",
        "HYBRID_COLAB_LR": "2e-5",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "0",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "8",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1_all",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "3",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "0",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "0",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "0",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
        "HYBRID_COLAB_TARGET_CYP": "CYP3A4",
        "HYBRID_COLAB_NEXUS_SIDEINFO_ONLY": "1",
        "HYBRID_COLAB_USE_CANDIDATE_MASK": "0",
        "HYBRID_COLAB_BALANCE_TRAIN_SOURCES": "1",
        "HYBRID_COLAB_SITE_ONLY_TARGET_CYP": "1",
        "HYBRID_COLAB_USE_TOPK_RERANKER": "1",
        "HYBRID_COLAB_TOPK_RERANKER_K": "8",
        "HYBRID_COLAB_TOPK_RERANKER_HIDDEN_DIM": "128",
        "HYBRID_COLAB_TOPK_RERANKER_HEADS": "4",
        "HYBRID_COLAB_TOPK_RERANKER_LAYERS": "2",
        "HYBRID_COLAB_TOPK_RERANKER_DROPOUT": "0.10",
        "HYBRID_COLAB_TOPK_RERANKER_RESIDUAL_SCALE": "0.85",
        "HYBRID_COLAB_TOPK_RERANKER_LR_SCALE": "8.0",
        "HYBRID_COLAB_TOPK_RERANKER_HEADSTART_EPOCHS": "6",
        "HYBRID_COLAB_BACKBONE_THAW_LR_SCALE": "0.05",
        "HYBRID_COLAB_SITE_RANKING_WEIGHT": "0.85",
        "HYBRID_COLAB_SITE_HARD_NEGATIVE_FRACTION": "0.80",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_TOPK": "3",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_DECAY": "0.75",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DEFAULT": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DRUGBANK": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_AZ120": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_METXBIODB": "1.10",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_ATTNSOM": "1.20",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_CYP_DBS_EXTERNAL": "1.35",
        "HYBRID_COLAB_DOMAIN_ADV_WEIGHT": "0.0",
        "HYBRID_COLAB_SOURCE_ALIGN_WEIGHT": "0.0",
        "HYBRID_COLAB_SOURCE_ALIGN_COV_WEIGHT": "0.0",
        "HYBRID_COLAB_NEXUS_LIVE_WAVE_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_LIVE_ANALOGICAL_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_WAVE_SIDEINFO_AUX_WEIGHT": "0.03",
        "HYBRID_COLAB_NEXUS_ANALOGICAL_SIDEINFO_AUX_WEIGHT": "0.05",
    },
    "cyp3a4_sideinfo_fullrank_strict": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_cyp3a4_augmented_strict.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "35",
        "HYBRID_COLAB_BATCH_SIZE": "12",
        "HYBRID_COLAB_LR": "2e-5",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "0",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "8",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1_all",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "3",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "0",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "0",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "0",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
        "HYBRID_COLAB_TARGET_CYP": "CYP3A4",
        "HYBRID_COLAB_NEXUS_SIDEINFO_ONLY": "1",
        "HYBRID_COLAB_USE_CANDIDATE_MASK": "0",
        "HYBRID_COLAB_BALANCE_TRAIN_SOURCES": "1",
        "HYBRID_COLAB_SITE_ONLY_TARGET_CYP": "1",
        "HYBRID_COLAB_SITE_RANKING_WEIGHT": "0.80",
        "HYBRID_COLAB_SITE_HARD_NEGATIVE_FRACTION": "0.70",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_TOPK": "2",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_DECAY": "0.85",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DEFAULT": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DRUGBANK": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_AZ120": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_METXBIODB": "1.10",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_ATTNSOM": "1.20",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_CYP_DBS_EXTERNAL": "1.35",
        "HYBRID_COLAB_DOMAIN_ADV_WEIGHT": "0.0",
        "HYBRID_COLAB_SOURCE_ALIGN_WEIGHT": "0.01",
        "HYBRID_COLAB_SOURCE_ALIGN_COV_WEIGHT": "0.35",
        "HYBRID_COLAB_NEXUS_LIVE_WAVE_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_LIVE_ANALOGICAL_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_WAVE_SIDEINFO_AUX_WEIGHT": "0.03",
        "HYBRID_COLAB_NEXUS_ANALOGICAL_SIDEINFO_AUX_WEIGHT": "0.05",
    },
    "cyp3a4_sideinfo_fullrank_sourceaware": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_cyp3a4_augmented_sourceaware.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "35",
        "HYBRID_COLAB_BATCH_SIZE": "12",
        "HYBRID_COLAB_LR": "2e-5",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "0",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "8",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1_all",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "3",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "0",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "0",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "0",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
        "HYBRID_COLAB_TARGET_CYP": "CYP3A4",
        "HYBRID_COLAB_NEXUS_SIDEINFO_ONLY": "1",
        "HYBRID_COLAB_USE_CANDIDATE_MASK": "0",
        "HYBRID_COLAB_BALANCE_TRAIN_SOURCES": "1",
        "HYBRID_COLAB_SITE_ONLY_TARGET_CYP": "1",
        "HYBRID_COLAB_SITE_RANKING_WEIGHT": "0.80",
        "HYBRID_COLAB_SITE_HARD_NEGATIVE_FRACTION": "0.70",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_TOPK": "2",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_DECAY": "0.85",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DEFAULT": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DRUGBANK": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_AZ120": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_METXBIODB": "1.10",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_ATTNSOM": "1.20",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_CYP_DBS_EXTERNAL": "1.35",
        "HYBRID_COLAB_DOMAIN_ADV_WEIGHT": "0.0",
        "HYBRID_COLAB_SOURCE_ALIGN_WEIGHT": "0.01",
        "HYBRID_COLAB_SOURCE_ALIGN_COV_WEIGHT": "0.35",
        "HYBRID_COLAB_NEXUS_LIVE_WAVE_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_LIVE_ANALOGICAL_VOTE_GRAD_SCALE": "0.10",
        "HYBRID_COLAB_NEXUS_WAVE_SIDEINFO_AUX_WEIGHT": "0.03",
        "HYBRID_COLAB_NEXUS_ANALOGICAL_SIDEINFO_AUX_WEIGHT": "0.05",
    },
    "cyp3a4_sideinfo_transfer": {
        "HYBRID_COLAB_DATASET": "data/prepared_training/main8_cyp3a4_augmented.json",
        "HYBRID_COLAB_STRUCTURE_SDF": "3D structures.sdf",
        "HYBRID_COLAB_EPOCHS": "8",
        "HYBRID_COLAB_BATCH_SIZE": "10",
        "HYBRID_COLAB_LR": "7e-6",
        "HYBRID_COLAB_WD": "1e-4",
        "HYBRID_COLAB_SPLIT_MODE": "scaffold_source_size",
        "HYBRID_COLAB_LIMIT": "0",
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": "0",
        "HYBRID_COLAB_SITE_LABELED_ONLY": "1",
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": "0",
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "4",
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top1",
        "HYBRID_COLAB_BACKBONE_FREEZE_EPOCHS": "2",
        "HYBRID_COLAB_BACKBONE_THAW_LR_SCALE": "0.05",
        "HYBRID_COLAB_INCLUDE_XENOSITE": "0",
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "1",
        "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS": "0",
        "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "0",
        "HYBRID_COLAB_SEED": "42",
        "HYBRID_COLAB_TRAIN_RATIO": "0.80",
        "HYBRID_COLAB_VAL_RATIO": "0.10",
        "HYBRID_COLAB_TARGET_CYP": "CYP3A4",
        "HYBRID_COLAB_NEXUS_SIDEINFO_ONLY": "1",
        "HYBRID_COLAB_USE_CANDIDATE_MASK": "1",
        "HYBRID_COLAB_BALANCE_TRAIN_SOURCES": "0",
        "HYBRID_COLAB_TRAIN_SOURCE_ALLOWLIST": "ATTNSOM,CYP_DBs_external",
        "HYBRID_COLAB_FREEZE_BASE_MODULES": "shared_encoder,physics_branch,steric_branch,manual_priors,cyp_branch,cyp_head",
        "HYBRID_COLAB_SITE_RANKING_WEIGHT": "0.90",
        "HYBRID_COLAB_SITE_HARD_NEGATIVE_FRACTION": "0.80",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_TOPK": "3",
        "HYBRID_COLAB_SITE_TOP1_MARGIN_DECAY": "0.65",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DEFAULT": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_DRUGBANK": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_AZ120": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_METXBIODB": "1.0",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_ATTNSOM": "1.4",
        "HYBRID_COLAB_SITE_SOURCE_WEIGHT_CYP_DBS_EXTERNAL": "1.8",
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
        raise FileNotFoundError(
            "Requested dataset not found: "
            f"{requested_dataset}. Set HYBRID_COLAB_DATASET to an existing path before launching training."
        )

    output_dir = os.environ.get(
        "HYBRID_COLAB_OUTPUT_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/hybrid_full_xtb",
    )
    artifact_dir = os.environ.get(
        "HYBRID_COLAB_ARTIFACT_DIR",
        "/content/drive/MyDrive/enzyme_hybrid_lnn/artifacts/hybrid_full_xtb",
    )
    console_log_path = Path(
        os.environ.get(
            "HYBRID_COLAB_CONSOLE_LOG",
            str(Path(artifact_dir) / f"hybrid_full_xtb_console_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        )
    )
    console_jsonl_path = Path(
        os.environ.get(
            "HYBRID_COLAB_CONSOLE_JSONL",
            str(console_log_path.with_suffix(".jsonl")),
        )
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
    benchmark_dir = Path(
        os.environ.get(
            "HYBRID_COLAB_BENCHMARK_DIR",
            "/content/drive/MyDrive/enzyme_hybrid_lnn/benchmarks",
        )
    )
    benchmark_holdout = Path(
        os.environ.get(
            "HYBRID_COLAB_BENCHMARK_HOLDOUT",
            str(benchmark_dir / "main8_benchmark_holdout_singlecyp.json"),
        )
    )
    benchmark_datasets_env = os.environ.get("HYBRID_COLAB_BENCHMARK_DATASETS", "").strip()
    if benchmark_datasets_env:
        requested_benchmarks = [part.strip() for part in benchmark_datasets_env.split(",") if part.strip()]
        resolved_benchmarks = [_resolve_dataset_path(part, benchmark_dir=benchmark_dir) for part in requested_benchmarks]
        if any(not path.exists() for path in resolved_benchmarks):
            _maybe_build_benchmark_sets(benchmark_dir=benchmark_dir, holdout_path=benchmark_holdout)
            resolved_benchmarks = [_resolve_dataset_path(part, benchmark_dir=benchmark_dir) for part in requested_benchmarks]
        missing = [str(path) for path in resolved_benchmarks if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Benchmark dataset(s) not found. "
                f"Missing: {', '.join(missing)}. "
                f"Upload benchmark JSONs to {benchmark_dir} or upload the holdout to {benchmark_holdout}."
            )
        os.environ["HYBRID_COLAB_BENCHMARK_DATASETS"] = ",".join(str(path) for path in resolved_benchmarks)

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
    backbone_thaw_lr_scale = os.environ.get("HYBRID_COLAB_BACKBONE_THAW_LR_SCALE", "").strip()
    if backbone_thaw_lr_scale:
        argv.extend(["--backbone-thaw-lr-scale", backbone_thaw_lr_scale])
    if os.environ.get("HYBRID_COLAB_INCLUDE_XENOSITE", "1").strip().lower() in {"1", "true", "yes", "on"}:
        argv.extend(["--xenosite-manifest", xenosite_manifest])
        argv.extend(["--xenosite-topk", os.environ["HYBRID_COLAB_XENOSITE_TOPK"]])
    precedent_logbook = os.environ.get("HYBRID_COLAB_PRECEDENT_LOGBOOK", "").strip()
    disable_precedent_logbook = os.environ.get("HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # An explicit logbook path should win over the preset default disable flag.
    if disable_precedent_logbook and not precedent_logbook:
        argv.append("--disable-precedent-logbook")
    target_cyp = os.environ.get("HYBRID_COLAB_TARGET_CYP", "").strip()
    if target_cyp:
        argv.extend(["--target-cyp", target_cyp])
    confidence_allowlist = os.environ.get("HYBRID_COLAB_CONFIDENCE_ALLOWLIST", "").strip()
    if confidence_allowlist:
        argv.extend(["--confidence-allowlist", confidence_allowlist])
    train_source_allowlist = os.environ.get("HYBRID_COLAB_TRAIN_SOURCE_ALLOWLIST", "").strip()
    if train_source_allowlist:
        argv.extend(["--train-source-allowlist", train_source_allowlist])
    if os.environ.get("HYBRID_COLAB_BASE_LNN_FIRST", "0").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--base-lnn-first")
    if os.environ.get("HYBRID_COLAB_NEXUS_SIDEINFO_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--nexus-sideinfo-only")
    if os.environ.get("HYBRID_COLAB_USE_CANDIDATE_MASK", "0").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--use-candidate-mask")
    candidate_mask_mode = os.environ.get("HYBRID_COLAB_CANDIDATE_MASK_MODE", "").strip()
    if candidate_mask_mode:
        argv.extend(["--candidate-mask-mode", candidate_mask_mode])
    candidate_mask_logit_bias = os.environ.get("HYBRID_COLAB_CANDIDATE_MASK_LOGIT_BIAS", "").strip()
    if candidate_mask_logit_bias:
        argv.extend(["--candidate-mask-logit-bias", candidate_mask_logit_bias])
    if os.environ.get("HYBRID_COLAB_BALANCE_TRAIN_SOURCES", "0").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--balance-train-sources")
    if os.environ.get("HYBRID_COLAB_SITE_ONLY_TARGET_CYP", "0").strip().lower() in {"1", "true", "yes", "on"}:
        argv.append("--site-only-target-cyp")
    freeze_base_modules = os.environ.get("HYBRID_COLAB_FREEZE_BASE_MODULES", "").strip()
    if freeze_base_modules:
        argv.extend(["--freeze-base-modules", freeze_base_modules])
    benchmark_datasets = os.environ.get("HYBRID_COLAB_BENCHMARK_DATASETS", "").strip()
    if benchmark_datasets:
        argv.extend(["--benchmark-datasets", benchmark_datasets])
    benchmark_batch_size = os.environ.get("HYBRID_COLAB_BENCHMARK_BATCH_SIZE", "").strip()
    if benchmark_batch_size:
        argv.extend(["--benchmark-batch-size", benchmark_batch_size])
    benchmark_every = os.environ.get("HYBRID_COLAB_BENCHMARK_EVERY", "").strip()
    if benchmark_every:
        argv.extend(["--benchmark-every", benchmark_every])
    benchmark_selection_metric = os.environ.get("HYBRID_COLAB_BENCHMARK_SELECTION_METRIC", "").strip()
    if benchmark_selection_metric:
        argv.extend(["--benchmark-selection-metric", benchmark_selection_metric])
    benchmark_selection_weight = os.environ.get("HYBRID_COLAB_BENCHMARK_SELECTION_WEIGHT", "").strip()
    if benchmark_selection_weight:
        argv.extend(["--benchmark-selection-weight", benchmark_selection_weight])
    if precedent_logbook:
        argv.extend(["--precedent-logbook", precedent_logbook])

    with _capture_console_log(console_log_path, console_jsonl_path):
        print("Hybrid LNN Colab wrapper")
        print(f"console_log={console_log_path}")
        print(f"console_jsonl={console_jsonl_path}")
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
        effective_disable_precedent_logbook = "0" if precedent_logbook else os.environ.get("HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK", "1")
        print(f"disable_precedent_logbook={effective_disable_precedent_logbook}")
        print(f"live_wave_vote_inputs={os.environ.get('HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS', '0')}")
        print(f"live_analogical_vote_inputs={os.environ.get('HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS', '0')}")
        if benchmark_datasets:
            print(f"benchmark_datasets={benchmark_datasets}")
            print(f"benchmark_selection_metric={benchmark_selection_metric or 'site_top1_acc_all_molecules'}")
            print(f"benchmark_selection_weight={benchmark_selection_weight or '0'}")
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
