from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict


TELEMETRY_PATH = Path(__file__).resolve().parents[3] / "cache" / "telemetry.json"
MAX_EVENTS = 200


def record_event(event: Dict[str, Any]) -> None:
    payload = dict(event)
    payload.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    payload.setdefault("schema_version", "v1")

    events = []
    if TELEMETRY_PATH.exists():
        try:
            with TELEMETRY_PATH.open("r", encoding="utf-8") as handle:
                events = json.load(handle) or []
        except (json.JSONDecodeError, OSError):
            events = []

    events.append(payload)
    if len(events) > MAX_EVENTS:
        events = events[-MAX_EVENTS:]

    try:
        TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TELEMETRY_PATH.open("w", encoding="utf-8") as handle:
            json.dump(events, handle, indent=2)
    except OSError:
        return
