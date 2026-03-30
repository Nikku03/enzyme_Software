from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def _scalar_at(value, idx: int | None):
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return float(value)
    if idx is None:
        return None
    try:
        item = value[idx]
    except Exception:
        return None
    if isinstance(item, list):
        if not item:
            return None
        item = item[0]
    try:
        return float(item)
    except Exception:
        return None


def _brief(record: dict) -> dict:
    idx = record.get("decision", {}).get("top1_atom")
    votes = record.get("votes") or {}
    analogical = record.get("analogical") or {}
    wave = record.get("wave") or {}
    return {
        "smiles": record.get("input", {}).get("smiles"),
        "source": record.get("input", {}).get("source"),
        "name": record.get("input", {}).get("name"),
        "top1_atom": idx,
        "top1_score": record.get("decision", {}).get("top1_score"),
        "top3_atoms": record.get("decision", {}).get("top3_atoms"),
        "true_site_atoms": record.get("outcome", {}).get("true_site_atoms"),
        "top1_hit": bool(record.get("outcome", {}).get("top1_hit")),
        "top3_hit": bool(record.get("outcome", {}).get("top3_hit")),
        "votes": {
            "lnn_vote": _scalar_at(votes.get("lnn_vote"), idx),
            "lnn_conf": _scalar_at(votes.get("lnn_conf"), idx),
            "wave_vote": _scalar_at(votes.get("wave_vote"), idx),
            "wave_conf": _scalar_at(votes.get("wave_conf"), idx),
            "analogical_vote": _scalar_at(votes.get("analogical_vote"), idx),
            "analogical_conf": _scalar_at(votes.get("analogical_conf"), idx),
            "council_logit": _scalar_at(votes.get("council_logit"), idx),
            "board_weights": None if votes.get("board_weights") is None or idx is None else votes.get("board_weights")[idx],
        },
        "analogical_top1": {
            "confidence": _scalar_at(analogical.get("confidence"), idx),
            "site_prior": _scalar_at(analogical.get("site_prior"), idx),
            "site_bias": _scalar_at(analogical.get("site_bias"), idx),
            "precedent_brief": None if analogical.get("precedent_brief") is None else analogical.get("precedent_brief")[idx],
        },
        "wave_top1": {
            "predicted_gap": wave.get("predicted_gap"),
            "global_density": wave.get("global_density"),
            "global_gap_proxy": wave.get("global_gap_proxy"),
        },
    }


def _winner(vote_block: dict, idx: int | None):
    stream_votes = {}
    for stream in ("lnn", "wave", "analogical"):
        value = _scalar_at(vote_block.get(f"{stream}_vote"), idx)
        if value is not None:
            stream_votes[stream] = value
    if len(stream_votes) != 3:
        return None, stream_votes
    return max(stream_votes, key=stream_votes.get), stream_votes


def analyze(path: Path, split: str, top_n: int) -> dict:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if record.get("record_type") != "episode":
                continue
            if split and record.get("split") != split:
                continue
            rows.append(record)

    source_counts = Counter()
    winner_counts = Counter()
    winner_hits = Counter()
    source_hits = Counter()
    stats = defaultdict(list)
    misses = []
    rescued = []
    hits = []

    for row in rows:
        source = str(row.get("input", {}).get("source") or "unknown")
        source_counts[source] += 1
        if row.get("outcome", {}).get("top1_hit"):
            source_hits[source] += 1

        idx = row.get("decision", {}).get("top1_atom")
        votes = row.get("votes") or {}
        top1_hit = bool(row.get("outcome", {}).get("top1_hit"))

        for stream in ("lnn", "wave", "analogical"):
            vote = _scalar_at(votes.get(f"{stream}_vote"), idx)
            conf = _scalar_at(votes.get(f"{stream}_conf"), idx)
            if vote is not None:
                stats[f"{stream}_vote"].append(vote)
                stats[f"{stream}_vote_hit" if top1_hit else f"{stream}_vote_miss"].append(vote)
            if conf is not None:
                stats[f"{stream}_conf"].append(conf)
                stats[f"{stream}_conf_hit" if top1_hit else f"{stream}_conf_miss"].append(conf)

        winner, stream_votes = _winner(votes, idx)
        if winner is not None:
            winner_counts[winner] += 1
            if top1_hit:
                winner_hits[winner] += 1

        brief = _brief(row)
        if top1_hit:
            hits.append(brief)
        else:
            misses.append(brief)
            if bool(row.get("outcome", {}).get("top3_hit")):
                rescued.append(brief)

    misses.sort(key=lambda item: float(item.get("top1_score") or 0.0), reverse=True)
    hits.sort(key=lambda item: float(item.get("top1_score") or 0.0), reverse=True)
    rescued.sort(key=lambda item: float(item.get("top1_score") or 0.0), reverse=True)

    split_summary = {
        "episodes": len(rows),
        "top1_acc": sum(1 for row in rows if row.get("outcome", {}).get("top1_hit")) / max(1, len(rows)),
        "top3_acc": sum(1 for row in rows if row.get("outcome", {}).get("top3_hit")) / max(1, len(rows)),
        "top5_acc": sum(1 for row in rows if row.get("outcome", {}).get("top5_hit")) / max(1, len(rows)),
    }

    source_summary = []
    for source, count in source_counts.most_common():
        source_summary.append({
            "source": source,
            "episodes": count,
            "top1_hits": source_hits[source],
            "top1_acc": source_hits[source] / max(1, count),
        })

    stat_summary = {key: mean(values) for key, values in stats.items() if values}
    board_weight_summary = {}
    for stream, col_idx in (("lnn", 0), ("wave", 1), ("analogical", 2)):
        values = []
        hit_values = []
        miss_values = []
        for row in rows:
            idx = row.get("decision", {}).get("top1_atom")
            board_weights = (row.get("votes") or {}).get("board_weights")
            if idx is None or board_weights is None:
                continue
            try:
                value = float(board_weights[idx][col_idx])
            except Exception:
                continue
            values.append(value)
            if bool(row.get("outcome", {}).get("top1_hit")):
                hit_values.append(value)
            else:
                miss_values.append(value)
        if values:
            board_weight_summary[f"{stream}_board_weight"] = mean(values)
        if hit_values:
            board_weight_summary[f"{stream}_board_weight_hit"] = mean(hit_values)
        if miss_values:
            board_weight_summary[f"{stream}_board_weight_miss"] = mean(miss_values)

    return {
        "log_path": str(path),
        "split": split,
        "summary": split_summary,
        "source_summary": source_summary,
        "winner_counts": dict(winner_counts),
        "winner_hits": dict(winner_hits),
        "vote_summary": stat_summary,
        "board_weight_summary": board_weight_summary,
        "top_confident_misses": misses[:top_n],
        "top_confident_hits": hits[:top_n],
        "top3_rescued_cases": rescued[:top_n],
    }


def _write_markdown(payload: dict, path: Path) -> None:
    lines = []
    lines.append(f"# Hybrid Episode Analysis")
    lines.append("")
    lines.append(f"- log: `{payload['log_path']}`")
    lines.append(f"- split: `{payload['split']}`")
    summary = payload["summary"]
    lines.append(f"- episodes: `{summary['episodes']}`")
    lines.append(f"- top1_acc: `{summary['top1_acc']:.4f}`")
    lines.append(f"- top3_acc: `{summary['top3_acc']:.4f}`")
    lines.append(f"- top5_acc: `{summary['top5_acc']:.4f}`")
    lines.append("")
    lines.append("## Source Summary")
    lines.append("")
    for row in payload["source_summary"]:
        lines.append(
            f"- `{row['source']}`: episodes=`{row['episodes']}` top1_hits=`{row['top1_hits']}` top1_acc=`{row['top1_acc']:.4f}`"
        )
    lines.append("")
    lines.append("## Winner Counts")
    lines.append("")
    for key, value in sorted(payload["winner_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Winner Hits")
    lines.append("")
    for key, value in sorted(payload["winner_hits"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Vote Summary")
    lines.append("")
    for key, value in sorted(payload["vote_summary"].items()):
        lines.append(f"- `{key}`: `{value:.6f}`")
    lines.append("")
    lines.append("## Board Weight Summary")
    lines.append("")
    for key, value in sorted(payload.get("board_weight_summary", {}).items()):
        lines.append(f"- `{key}`: `{value:.6f}`")
    lines.append("")
    for section in ("top_confident_misses", "top_confident_hits", "top3_rescued_cases"):
        lines.append(f"## {section.replace('_', ' ').title()}")
        lines.append("")
        for item in payload[section]:
            lines.append(f"- smiles: `{item['smiles']}`")
            lines.append(f"  top1_atom=`{item['top1_atom']}` top1_score=`{item['top1_score']}` true_site_atoms=`{item['true_site_atoms']}`")
            lines.append(f"  votes={json.dumps(item['votes'])}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hybrid episode JSONL logs without loading them into a database.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    payload = analyze(log_path, args.split, args.top_n)

    output_json = Path(args.output_json) if args.output_json else log_path.with_name(f"{log_path.stem}_{args.split}_analysis.json")
    output_md = Path(args.output_md) if args.output_md else log_path.with_name(f"{log_path.stem}_{args.split}_analysis.md")

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, output_md)

    print(f"Wrote JSON analysis: {output_json}")
    print(f"Wrote Markdown analysis: {output_md}")


if __name__ == "__main__":
    main()
