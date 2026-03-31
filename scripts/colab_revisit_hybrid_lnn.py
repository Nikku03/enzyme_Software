"""
Iterative Colab revisit loop for the hybrid full-xTB council model.

This script:
1. Finds the latest N reports/logs.
2. Builds a baseline from the best recent run.
3. Proposes several council-training strategy paths from the report/log evidence.
4. Runs train -> analyze -> compare in a loop.
5. Promotes the best run metadata when it beats the current baseline.

Run from Colab or locally:

    exec(open('/content/enzyme_Software/scripts/colab_revisit_hybrid_lnn.py').read())
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_analyzer():
    path = ROOT / "scripts" / "analyze_hybrid_episode_log.py"
    spec = importlib.util.spec_from_file_location("_hybrid_episode_analyzer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load analyzer module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze


ANALYZE_EPISODE_LOG = _load_analyzer()


def _ensure_rdkit() -> None:
    try:
        from rdkit import Chem  # noqa: F401
        return
    except Exception:
        pass
    print("RDKit not found. Installing rdkit-pypi...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rdkit-pypi"], cwd=str(ROOT))


def _setdefault_env(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


def _score_metrics(metrics: dict[str, Any]) -> float:
    if not metrics:
        return float("-inf")
    top1 = float(metrics.get("site_top1_acc", 0.0))
    top3 = float(metrics.get("site_top3_acc", 0.0))
    auc = float(metrics.get("site_auc", 0.0))
    precision = float(metrics.get("site_precision", 0.0))
    fp = float(metrics.get("fp", 0.0))
    return (0.40 * top3) + (0.30 * auc) + (0.20 * top1) + (0.10 * precision) - (0.0005 * fp)


def _compare_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        _score_metrics(metrics),
        float(metrics.get("site_top3_acc", 0.0)),
        float(metrics.get("site_auc", 0.0)),
        float(metrics.get("site_top1_acc", 0.0)),
        -float(metrics.get("fp", 0.0)),
    )


def _resolve_existing_path(raw_path: str | None, search_dirs: list[Path]) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    direct = Path(text)
    if direct.exists():
        return direct
    name = Path(text).name
    for directory in search_dirs:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _latest_paths(pattern: str, search_dirs: list[Path], limit: int) -> list[Path]:
    items: dict[Path, float] = {}
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            try:
                items[path.resolve()] = path.stat().st_mtime
            except FileNotFoundError:
                continue
    ranked = sorted(items.items(), key=lambda item: item[1], reverse=True)
    return [path for path, _ in ranked[:limit]]


@dataclass
class ReviewedRun:
    report_path: Path
    log_path: Path | None
    test_analysis: dict[str, Any] | None
    val_analysis: dict[str, Any] | None
    report: dict[str, Any]
    score: float


@dataclass
class Strategy:
    name: str
    rationale: str
    train_overrides: dict[str, str]
    model_env: dict[str, str]


def _load_run(report_path: Path, analysis_dir: Path, search_dirs: list[Path]) -> ReviewedRun:
    report = json.loads(report_path.read_text())
    log_path = _resolve_existing_path(report.get("episode_log_path"), search_dirs)
    test_analysis = None
    val_analysis = None
    if log_path is not None and log_path.exists():
        analysis_dir.mkdir(parents=True, exist_ok=True)
        test_json = analysis_dir / f"{log_path.stem}_test_analysis.json"
        val_json = analysis_dir / f"{log_path.stem}_val_analysis.json"
        test_analysis = ANALYZE_EPISODE_LOG(log_path, "test", 10)
        val_analysis = ANALYZE_EPISODE_LOG(log_path, "val", 10)
        test_json.write_text(json.dumps(test_analysis, indent=2), encoding="utf-8")
        val_json.write_text(json.dumps(val_analysis, indent=2), encoding="utf-8")
    score = _score_metrics(report.get("test_metrics") or {})
    return ReviewedRun(
        report_path=report_path,
        log_path=log_path,
        test_analysis=test_analysis,
        val_analysis=val_analysis,
        report=report,
        score=score,
    )


def _summarize_findings(run: ReviewedRun) -> dict[str, bool | float]:
    test_analysis = run.test_analysis or {}
    winner_counts = test_analysis.get("winner_counts") or {}
    winner_hits = test_analysis.get("winner_hits") or {}
    board = test_analysis.get("board_weight_summary") or {}
    metrics = run.report.get("test_metrics") or {}
    analogical_wins = float(winner_counts.get("analogical", 0))
    lnn_wins = float(winner_counts.get("lnn", 0))
    analogical_hit_rate = float(winner_hits.get("analogical", 0)) / max(1.0, analogical_wins)
    lnn_hit_rate = float(winner_hits.get("lnn", 0)) / max(1.0, lnn_wins)
    wave_hits = float(winner_hits.get("wave", 0))
    return {
        "analogical_overtrust": analogical_wins > lnn_wins and analogical_hit_rate < lnn_hit_rate,
        "wave_dead": wave_hits <= 0.0,
        "ranking_gap": float(metrics.get("site_top3_acc", 0.0)) - float(metrics.get("site_top1_acc", 0.0)),
        "top3": float(metrics.get("site_top3_acc", 0.0)),
        "top1": float(metrics.get("site_top1_acc", 0.0)),
        "auc": float(metrics.get("site_auc", 0.0)),
        "precision": float(metrics.get("site_precision", 0.0)),
        "fp": float(metrics.get("fp", 0.0)),
        "analogical_hit_rate": analogical_hit_rate,
        "lnn_hit_rate": lnn_hit_rate,
        "wave_weight": float(board.get("wave_board_weight", 0.0)),
        "analogical_weight": float(board.get("analogical_board_weight", 0.0)),
    }


def _build_strategies(run: ReviewedRun, iterations: int) -> list[Strategy]:
    findings = _summarize_findings(run)
    baseline_log = str(run.log_path) if run.log_path is not None else ""
    strategies: list[Strategy] = [
        Strategy(
            name="patience5_conservative",
            rationale="Keep the current stable council, run longer, and let site_top3 recover before stopping.",
            train_overrides={
                "HYBRID_COLAB_EPOCHS": "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "5",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top3",
                "HYBRID_COLAB_LR": "5e-5",
            },
            model_env={},
        ),
        Strategy(
            name="analogical_rebalance",
            rationale="Analogical is winning often; reduce its training pressure and CYP pull so it stops overreaching.",
            train_overrides={
                "HYBRID_COLAB_EPOCHS": "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "5",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top3",
                "HYBRID_COLAB_LR": "5e-5",
            },
            model_env={
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT": "0.03",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE": "0.06",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT": "0.005",
            },
        ),
        Strategy(
            name="wave_support",
            rationale="Wave is active in the board but still weak as a voter; give it a slightly stronger supervised push.",
            train_overrides={
                "HYBRID_COLAB_EPOCHS": "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "5",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top3",
                "HYBRID_COLAB_LR": "5e-5",
            },
            model_env={
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT": "0.05",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT": "0.03",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE": "1.8",
            },
        ),
        Strategy(
            name="precedent_enabled",
            rationale="Enable audited analogical precedent replay from the best known episode log to see if analogical becomes more selective.",
            train_overrides={
                "HYBRID_COLAB_EPOCHS": "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "5",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top3",
                "HYBRID_COLAB_LR": "5e-5",
                "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "0",
                "HYBRID_COLAB_PRECEDENT_LOGBOOK": baseline_log,
            },
            model_env={
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT": "0.03",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE": "0.06",
            },
        ),
        Strategy(
            name="longer_patient_low_lr",
            rationale="Lower the step size slightly and allow more epochs before stopping to test a slower path.",
            train_overrides={
                "HYBRID_COLAB_EPOCHS": "60",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": "6",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC": "site_top3",
                "HYBRID_COLAB_LR": "4e-5",
                "HYBRID_COLAB_WD": "2e-4",
            },
            model_env={
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT": "0.005",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT": "0.15",
            },
        ),
    ]
    if findings.get("wave_dead"):
        strategies[0], strategies[2] = strategies[2], strategies[0]
    if findings.get("analogical_overtrust"):
        strategies[0], strategies[1] = strategies[1], strategies[0]
    if not baseline_log:
        strategies = [s for s in strategies if s.name != "precedent_enabled"]
    return strategies[:iterations]


def _build_train_command(args, checkpoint_path: Path, output_dir: Path, artifact_dir: Path, settings: dict[str, str]) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_hybrid_full_xtb.py"),
        "--dataset",
        settings["HYBRID_COLAB_DATASET"],
        "--structure-sdf",
        settings["HYBRID_COLAB_STRUCTURE_SDF"],
        "--checkpoint",
        str(checkpoint_path),
        "--xtb-cache-dir",
        settings["HYBRID_COLAB_XTB_CACHE_DIR"],
        "--epochs",
        settings["HYBRID_COLAB_EPOCHS"],
        "--batch-size",
        settings["HYBRID_COLAB_BATCH_SIZE"],
        "--learning-rate",
        settings["HYBRID_COLAB_LR"],
        "--weight-decay",
        settings["HYBRID_COLAB_WD"],
        "--seed",
        settings["HYBRID_COLAB_SEED"],
        "--output-dir",
        str(output_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--manual-feature-cache-dir",
        settings["HYBRID_COLAB_MANUAL_CACHE_DIR"],
        "--early-stopping-patience",
        settings["HYBRID_COLAB_EARLY_STOPPING_PATIENCE"],
        "--early-stopping-metric",
        settings["HYBRID_COLAB_EARLY_STOPPING_METRIC"],
    ]
    if int(settings.get("HYBRID_COLAB_LIMIT", "0") or "0") > 0:
        cmd.extend(["--limit", settings["HYBRID_COLAB_LIMIT"]])
    if settings.get("HYBRID_COLAB_SITE_LABELED_ONLY", "1") == "1":
        cmd.append("--site-labeled-only")
    if settings.get("HYBRID_COLAB_COMPUTE_XTB_IF_MISSING", "0") == "1":
        cmd.append("--compute-xtb-if-missing")
    if settings.get("HYBRID_COLAB_FREEZE_NEXUS_MEMORY", "1") == "1":
        cmd.append("--freeze-nexus-memory")
    if settings.get("HYBRID_COLAB_INCLUDE_XENOSITE", "1") == "1":
        cmd.extend(["--xenosite-manifest", settings["HYBRID_COLAB_XENOSITE_MANIFEST"]])
        cmd.extend(["--xenosite-topk", settings["HYBRID_COLAB_XENOSITE_TOPK"]])
    if settings.get("HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK", "1") == "1":
        cmd.append("--disable-precedent-logbook")
    elif settings.get("HYBRID_COLAB_PRECEDENT_LOGBOOK"):
        cmd.extend(["--precedent-logbook", settings["HYBRID_COLAB_PRECEDENT_LOGBOOK"]])
    return cmd


def _resolve_attempt_artifacts(artifact_dir: Path, output_dir: Path) -> tuple[Path, Path | None, Path | None]:
    reports = sorted(artifact_dir.glob("hybrid_full_xtb_report_*.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        raise FileNotFoundError(f"No report produced in {artifact_dir}")
    report_path = reports[-1]
    report = json.loads(report_path.read_text())
    log_path = _resolve_existing_path(report.get("episode_log_path"), [artifact_dir, ROOT])
    best_checkpoint = output_dir / "hybrid_full_xtb_best.pt"
    if not best_checkpoint.exists():
        best_checkpoint = output_dir / "hybrid_full_xtb_latest.pt"
    return report_path, log_path, best_checkpoint if best_checkpoint.exists() else None


def _default_train_settings() -> dict[str, str]:
    return {
        "HYBRID_COLAB_DATASET": os.environ.get("HYBRID_COLAB_DATASET", "data/prepared_training/main5_site_conservative_singlecyp_clean.json"),
        "HYBRID_COLAB_STRUCTURE_SDF": os.environ.get("HYBRID_COLAB_STRUCTURE_SDF", "3D structures.sdf"),
        "HYBRID_COLAB_XTB_CACHE_DIR": os.environ.get("HYBRID_COLAB_XTB_CACHE_DIR", "/content/drive/MyDrive/enzyme_hybrid_lnn/cache/full_xtb"),
        "HYBRID_COLAB_MANUAL_CACHE_DIR": os.environ.get("HYBRID_COLAB_MANUAL_CACHE_DIR", "/content/drive/MyDrive/enzyme_hybrid_lnn/cache/manual_engine_full"),
        "HYBRID_COLAB_XENOSITE_MANIFEST": os.environ.get("HYBRID_COLAB_XENOSITE_MANIFEST", "data/xenosite_suppl/manifest.json"),
        "HYBRID_COLAB_EPOCHS": os.environ.get("HYBRID_COLAB_EPOCHS", "50"),
        "HYBRID_COLAB_BATCH_SIZE": os.environ.get("HYBRID_COLAB_BATCH_SIZE", "16"),
        "HYBRID_COLAB_LR": os.environ.get("HYBRID_COLAB_LR", "5e-5"),
        "HYBRID_COLAB_WD": os.environ.get("HYBRID_COLAB_WD", "1e-4"),
        "HYBRID_COLAB_SEED": os.environ.get("HYBRID_COLAB_SEED", "42"),
        "HYBRID_COLAB_LIMIT": os.environ.get("HYBRID_COLAB_LIMIT", "0"),
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": os.environ.get("HYBRID_COLAB_EARLY_STOPPING_PATIENCE", "5"),
        "HYBRID_COLAB_EARLY_STOPPING_METRIC": os.environ.get("HYBRID_COLAB_EARLY_STOPPING_METRIC", "site_top3"),
        "HYBRID_COLAB_SITE_LABELED_ONLY": os.environ.get("HYBRID_COLAB_SITE_LABELED_ONLY", "1"),
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING": os.environ.get("HYBRID_COLAB_COMPUTE_XTB_IF_MISSING", "0"),
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY": os.environ.get("HYBRID_COLAB_FREEZE_NEXUS_MEMORY", "1"),
        "HYBRID_COLAB_INCLUDE_XENOSITE": os.environ.get("HYBRID_COLAB_INCLUDE_XENOSITE", "1"),
        "HYBRID_COLAB_XENOSITE_TOPK": os.environ.get("HYBRID_COLAB_XENOSITE_TOPK", "1"),
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": os.environ.get("HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK", "1"),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative revisit loops for hybrid council training.")
    parser.add_argument("--recent-reports", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--baseline-report", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--output-root", default=os.environ.get("HYBRID_COLAB_REVISIT_OUTPUT_ROOT", "/content/drive/MyDrive/enzyme_hybrid_lnn/revisit/checkpoints"))
    parser.add_argument("--artifact-root", default=os.environ.get("HYBRID_COLAB_REVISIT_ARTIFACT_ROOT", "/content/drive/MyDrive/enzyme_hybrid_lnn/revisit/artifacts"))
    parser.add_argument("--search-dir", action="append", default=[])
    args = parser.parse_args()

    os.chdir(ROOT)
    _ensure_rdkit()
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("HYBRID_FORCE_MANUAL_OPTIMIZER", "1")

    search_dirs = [ROOT]
    for raw in args.search_dir:
        path = Path(raw)
        if path.exists():
            search_dirs.append(path)
    default_artifact_dir = Path(os.environ.get("HYBRID_COLAB_ARTIFACT_DIR", "/content/drive/MyDrive/enzyme_hybrid_lnn/artifacts/hybrid_full_xtb"))
    if default_artifact_dir.exists():
        search_dirs.append(default_artifact_dir)
    output_root = Path(args.output_root)
    artifact_root = Path(args.artifact_root)
    analysis_dir = artifact_root / "analysis_cache"
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    if args.baseline_report:
        recent_report_paths = [Path(args.baseline_report)]
    else:
        recent_report_paths = _latest_paths("**/hybrid_full_xtb_report_*.json", search_dirs, args.recent_reports)
    if not recent_report_paths:
        raise FileNotFoundError("No recent hybrid_full_xtb reports found for revisit bootstrap.")

    reviewed_runs = [_load_run(path, analysis_dir, search_dirs) for path in recent_report_paths]
    baseline = max(reviewed_runs, key=lambda run: _compare_key(run.report.get("test_metrics") or {}))

    if args.baseline_checkpoint:
        baseline_checkpoint = Path(args.baseline_checkpoint)
    else:
        default_output_dir = Path(os.environ.get("HYBRID_COLAB_OUTPUT_DIR", "/content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/hybrid_full_xtb"))
        best_candidate = default_output_dir / "hybrid_full_xtb_best.pt"
        latest_candidate = default_output_dir / "hybrid_full_xtb_latest.pt"
        baseline_checkpoint = best_candidate if best_candidate.exists() else latest_candidate
    if not baseline_checkpoint.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_checkpoint}")

    print("Hybrid revisit bootstrap", flush=True)
    print(f"baseline_report={baseline.report_path}", flush=True)
    print(f"baseline_checkpoint={baseline_checkpoint}", flush=True)
    print(f"baseline_score={baseline.score:.6f}", flush=True)

    summary_payload = {
        "bootstrap_reports": [str(path) for path in recent_report_paths],
        "baseline_report": str(baseline.report_path),
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_score": baseline.score,
        "attempts": [],
    }
    _write_json(artifact_root / "revisit_bootstrap.json", summary_payload)

    strategies = _build_strategies(baseline, args.iterations)
    base_settings = _default_train_settings()

    current_baseline = baseline
    current_checkpoint = baseline_checkpoint

    for attempt_idx, strategy in enumerate(strategies, start=1):
        attempt_name = f"attempt_{attempt_idx:02d}_{strategy.name}"
        attempt_output_dir = output_root / attempt_name
        attempt_artifact_dir = artifact_root / attempt_name
        attempt_output_dir.mkdir(parents=True, exist_ok=True)
        attempt_artifact_dir.mkdir(parents=True, exist_ok=True)

        settings = dict(base_settings)
        settings.update(strategy.train_overrides)
        env = os.environ.copy()
        env.update(settings)
        env["PYTHONPATH"] = f"{SRC}:{env.get('PYTHONPATH', '')}".rstrip(":")
        env["HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS"] = env.get("HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS", "0")
        env["HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS"] = env.get("HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS", "0")
        env["HYBRID_COLAB_LOCK_PRESET_POLICY"] = "1"
        env.update(strategy.model_env)

        print(f"\n=== Revisit {attempt_idx}/{len(strategies)}: {strategy.name} ===", flush=True)
        print(strategy.rationale, flush=True)
        print(f"warm_start={current_checkpoint}", flush=True)
        print(f"train_overrides={settings}", flush=True)
        if strategy.model_env:
            print(f"model_env={strategy.model_env}", flush=True)

        cmd = _build_train_command(
            args=args,
            checkpoint_path=current_checkpoint,
            output_dir=attempt_output_dir,
            artifact_dir=attempt_artifact_dir,
            settings=settings,
        )
        subprocess.check_call(cmd, cwd=str(ROOT), env=env)

        report_path, log_path, produced_checkpoint = _resolve_attempt_artifacts(attempt_artifact_dir, attempt_output_dir)
        reviewed = _load_run(report_path, analysis_dir, [attempt_artifact_dir, attempt_output_dir, ROOT])
        improved = _compare_key(reviewed.report.get("test_metrics") or {}) > _compare_key(current_baseline.report.get("test_metrics") or {})

        attempt_record = {
            "attempt": attempt_idx,
            "name": strategy.name,
            "rationale": strategy.rationale,
            "report_path": str(report_path),
            "log_path": str(log_path) if log_path is not None else None,
            "checkpoint_path": str(produced_checkpoint) if produced_checkpoint is not None else None,
            "score": reviewed.score,
            "improved": improved,
            "test_metrics": reviewed.report.get("test_metrics"),
            "test_analysis_summary": reviewed.test_analysis.get("summary") if reviewed.test_analysis else None,
        }
        summary_payload["attempts"].append(attempt_record)

        if improved:
            current_baseline = reviewed
            if produced_checkpoint is not None:
                current_checkpoint = produced_checkpoint
            summary_payload["baseline_report"] = str(report_path)
            summary_payload["baseline_checkpoint"] = str(current_checkpoint)
            summary_payload["baseline_score"] = reviewed.score
            print(f"Promoted new baseline: {report_path}", flush=True)
        else:
            print("No improvement over current baseline.", flush=True)

        _write_json(artifact_root / "revisit_bootstrap.json", summary_payload)

    _write_json(
        artifact_root / "revisit_best.json",
        {
            "baseline_report": summary_payload["baseline_report"],
            "baseline_checkpoint": summary_payload["baseline_checkpoint"],
            "baseline_score": summary_payload["baseline_score"],
            "attempt_count": len(summary_payload["attempts"]),
        },
    )
    print("\nRevisit loop complete.", flush=True)
    print(f"best_report={summary_payload['baseline_report']}", flush=True)
    print(f"best_checkpoint={summary_payload['baseline_checkpoint']}", flush=True)
    print(f"best_score={summary_payload['baseline_score']:.6f}", flush=True)


main()
