from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def write_research_note(
    path: Path,
    iteration: int,
    dataset_id: Optional[str],
    source_url: Optional[str],
    search_urls: List[str],
    evaluator_json_path: Optional[Path],
    notes: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Research Note {datetime.utcnow().date()} Iter {iteration}",
        "",
        f"Dataset ID: {dataset_id or 'UNKNOWN'}",
        f"Source URL: {source_url or 'UNKNOWN'}",
        "",
        "## Search URLs",
    ]
    if search_urls:
        lines.extend([f"- {url}" for url in search_urls])
    else:
        lines.append("- UNKNOWN")

    lines.append("")
    lines.append("## Evaluator JSON")
    if evaluator_json_path:
        lines.append(f"- {evaluator_json_path}")
    else:
        lines.append("- UNKNOWN")

    if notes:
        lines.append("")
        lines.append("## Notes")
        lines.extend([f"- {note}" for note in notes])

    if tasks:
        lines.append("")
        lines.append("## Fetch/Verify Tasks")
        lines.extend([f"- {task}" for task in tasks])

    path.write_text("\n".join(lines), encoding="utf-8")


def render_leaderboard(entries: List[Dict[str, str]]) -> str:
    lines = ["# Leaderboard", "", "| Rank | Strategy | Dataset | Score |", "| --- | --- | --- | --- |"]
    for idx, entry in enumerate(entries, start=1):
        score = entry.get("score")
        score_str = f"{score:.4f}" if score is not None else "UNKNOWN"
        lines.append(
            f"| {idx} | {entry.get('strategy')} | {entry.get('dataset_id')} | {score_str} |"
        )
    return "\n".join(lines)
