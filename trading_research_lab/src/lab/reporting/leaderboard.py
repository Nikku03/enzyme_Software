from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from lab.backtest.metrics import compute_score


def load_leaderboard(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_leaderboard(path: Path, entries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def build_entry(
    metrics_payload: Dict[str, Any],
    metrics_path: Path,
    scoring_cfg: Dict[str, float],
) -> Dict[str, Any]:
    metrics = metrics_payload["metrics"]
    score = compute_score(metrics, scoring_cfg)
    return {
        "run_id": metrics_payload.get("timestamp", datetime.utcnow().isoformat()),
        "strategy": metrics_payload.get("strategy"),
        "dataset_id": metrics_payload.get("dataset_id"),
        "metrics_path": str(metrics_path),
        "score": score,
        "failure_tags": metrics_payload.get("failure_tags", []),
    }


def update_leaderboard(
    path: Path,
    metrics_payload: Dict[str, Any],
    metrics_path: Path,
    scoring_cfg: Dict[str, float],
) -> List[Dict[str, Any]]:
    entries = load_leaderboard(path)
    entries.append(build_entry(metrics_payload, metrics_path, scoring_cfg))
    entries.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    write_leaderboard(path, entries)
    return entries


def top_entries(entries: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    return entries[:top_n]
