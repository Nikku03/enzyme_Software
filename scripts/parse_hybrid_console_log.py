from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


_EPOCH_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+)\s+\|\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+\|\s+"
    r"site_loss=(?P<site_loss>[-+0-9.eE]+)\s+\|\s+"
    r"cyp_loss=(?P<cyp_loss>[-+0-9.eE]+)\s+\|\s+"
    r"site_top1=(?P<site_top1>[-+0-9.eE]+)\s+\|\s+"
    r"site_top3=(?P<site_top3>[-+0-9.eE]+)\s+\|\s+"
    r"cyp_acc=(?P<cyp_acc>[-+0-9.eE]+)\s+\|\s+"
    r"cyp_f1=(?P<cyp_f1>[-+0-9.eE]+)\s+\|\s+"
    r"physics_gate=(?P<physics_gate>[-+0-9.eE]+)\s+\|\s+"
    r"epoch_time=(?P<epoch_time_s>[-+0-9.eE]+)s\s+\|\s+"
    r"elapsed=(?P<elapsed_m>[-+0-9.eE]+)m\s+\|\s+"
    r"eta=(?P<eta_m>[-+0-9.eE]+)m$"
)
_MEMORY_RE = re.compile(r"^\s*\[memory refreshed:\s+(?P<atoms>\d+)\s+atoms in buffer\]\s*$")
_BACKBONE_FROZEN_RE = re.compile(r"^Backbone frozen for first\s+(?P<epochs>\d+)\s+epochs\.$")
_BACKBONE_UNFROZEN_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+): backbone unfrozen \(thaw at (?P<lr_scale>[-+0-9.eE]+)x LR\)\.$"
)
_EARLY_STOP_RE = re.compile(
    r"^Early stopping after epoch\s+(?P<epoch>\d+): no\s+(?P<metric>[a-zA-Z0-9_]+)\s+improvement for\s+(?P<patience>\d+)\s+epochs\.$"
)
_WARM_START_SUMMARY_RE = re.compile(
    r"^Warm-start load summary:\s+loaded=(?P<loaded>\d+)\s+missing=(?P<missing>\d+)\s+mismatch=(?P<mismatch>\d+)\s+nonfinite=(?P<nonfinite>\d+)$"
)
_NEXUS_MEMORY_RE = re.compile(
    r"^Built NEXUS memory:\s+size=(?P<size>\d+)\s+from_batches=(?P<batches>\d+)\s+frozen=(?P<frozen>yes|no)$"
)
_XTB_CACHE_RE = re.compile(
    r"^xTB cache valid molecules:\s+(?P<valid>\d+)/(?P<total>\d+)\s+\|\s+statuses=(?P<statuses>.+)$"
)
_LOADED_DRUGS_RE = re.compile(r"^Loaded\s+(?P<count>\d+)\s+drugs$")
_LOADED_SPLIT_RE = re.compile(r"^Loaded\s+(?P<count>\d+)\s+drugs for\s+(?P<split>[a-z]+)\s+split$")
_EFFECTIVE_SPLIT_RE = re.compile(
    r"^(?P<split>train|val|test)\s+effective:\s+total=(?P<total>\d+)\s+\|\s+invalid=(?P<invalid>\d+)\s+\|\s+invalid_reasons=(?P<reasons>.+)$"
)


def _to_float_dict(match: re.Match[str], keys: Iterable[str]) -> Dict[str, float]:
    return {key: float(match.group(key)) for key in keys}


def _parse_kv_line(line: str) -> Dict[str, Any] | None:
    if "=" not in line:
        return None
    if line.startswith("Epoch "):
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return {"event": "kv", "key": key, "value": value.strip()}


def parse_console_line(line: str) -> Dict[str, Any]:
    line = line.rstrip("\n")
    record: Dict[str, Any] = {"raw": line, "event": "raw"}

    if not line.strip():
        record["event"] = "blank"
        return record

    if line == "Hybrid LNN Colab wrapper":
        record["event"] = "wrapper_start"
        return record

    if line.startswith("Loaded warm-start checkpoint: "):
        record["event"] = "warm_start_checkpoint"
        record["path"] = line.split(": ", 1)[1].strip()
        return record

    if line.startswith("Episode log: "):
        record["event"] = "episode_log"
        record["path"] = line.split(": ", 1)[1].strip()
        return record

    if line.startswith("Precedent logbook loading disabled"):
        record["event"] = "precedent_logbook_disabled"
        return record

    if line.startswith("Live sidecar vote inputs: "):
        record["event"] = "live_sidecar_vote_inputs"
        tail = line.split(": ", 1)[1]
        parts = dict(part.split("=") for part in tail.split() if "=" in part)
        record.update(parts)
        return record

    if line.startswith("Using device: "):
        record["event"] = "device"
        record["device"] = line.split(": ", 1)[1].strip()
        return record

    if line.startswith("Using ManualAdamW"):
        record["event"] = "optimizer"
        record["optimizer"] = "ManualAdamW"
        record["reason"] = line
        return record

    if line.startswith("Split mode: "):
        record["event"] = "split_mode"
        record["split_mode"] = line.split(": ", 1)[1].strip()
        return record

    m = _LOADED_DRUGS_RE.match(line)
    if m:
        record["event"] = "loaded_drugs"
        record["count"] = int(m.group("count"))
        return record

    m = _LOADED_SPLIT_RE.match(line)
    if m:
        record["event"] = "loaded_split"
        record["split"] = m.group("split")
        record["count"] = int(m.group("count"))
        return record

    m = _WARM_START_SUMMARY_RE.match(line)
    if m:
        record["event"] = "warm_start_summary"
        for key in ("loaded", "missing", "mismatch", "nonfinite"):
            record[key] = int(m.group(key))
        return record

    m = _NEXUS_MEMORY_RE.match(line)
    if m:
        record["event"] = "nexus_memory_built"
        record["size"] = int(m.group("size"))
        record["batches"] = int(m.group("batches"))
        record["frozen"] = m.group("frozen") == "yes"
        return record

    m = _XTB_CACHE_RE.match(line)
    if m:
        record["event"] = "xtb_cache_summary"
        record["valid"] = int(m.group("valid"))
        record["total"] = int(m.group("total"))
        try:
            record["statuses"] = ast.literal_eval(m.group("statuses"))
        except Exception:
            record["statuses"] = m.group("statuses")
        return record

    m = _BACKBONE_FROZEN_RE.match(line)
    if m:
        record["event"] = "backbone_frozen"
        record["epochs"] = int(m.group("epochs"))
        return record

    m = _BACKBONE_UNFROZEN_RE.match(line)
    if m:
        record["event"] = "backbone_unfrozen"
        record["epoch"] = int(m.group("epoch"))
        record["lr_scale"] = float(m.group("lr_scale"))
        return record

    m = _MEMORY_RE.match(line)
    if m:
        record["event"] = "memory_refreshed"
        record["atoms"] = int(m.group("atoms"))
        return record

    m = _EPOCH_RE.match(line)
    if m:
        record["event"] = "epoch"
        record["epoch"] = int(m.group("epoch"))
        record.update(
            _to_float_dict(
                m,
                (
                    "loss",
                    "site_loss",
                    "cyp_loss",
                    "site_top1",
                    "site_top3",
                    "cyp_acc",
                    "cyp_f1",
                    "physics_gate",
                    "epoch_time_s",
                    "elapsed_m",
                    "eta_m",
                ),
            )
        )
        return record

    m = _EARLY_STOP_RE.match(line)
    if m:
        record["event"] = "early_stopping"
        record["epoch"] = int(m.group("epoch"))
        record["metric"] = m.group("metric")
        record["patience"] = int(m.group("patience"))
        return record

    m = _EFFECTIVE_SPLIT_RE.match(line)
    if m:
        record["event"] = "effective_split"
        record["split"] = m.group("split")
        record["total"] = int(m.group("total"))
        record["invalid"] = int(m.group("invalid"))
        try:
            record["invalid_reasons"] = ast.literal_eval(m.group("reasons"))
        except Exception:
            record["invalid_reasons"] = m.group("reasons")
        return record

    kv = _parse_kv_line(line)
    if kv is not None:
        return {"raw": line, **kv}

    return record


def convert_console_log_to_jsonl(log_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with log_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            record = parse_console_line(line)
            record["line_number"] = line_number
            dst.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert hybrid console log into structured JSONL.")
    parser.add_argument("log_path", help="Path to the plain-text console log.")
    parser.add_argument("--output-jsonl", help="Output JSONL path. Defaults next to the log.")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Console log not found: {log_path}")
    output_path = Path(args.output_jsonl) if args.output_jsonl else log_path.with_suffix(".jsonl")
    count = convert_console_log_to_jsonl(log_path, output_path)
    print(f"Wrote {count} JSONL events to {output_path}")


if __name__ == "__main__":
    main()
