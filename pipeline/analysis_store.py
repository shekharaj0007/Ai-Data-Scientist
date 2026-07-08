"""Persist latest analysis run so it survives server restarts."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

STORE_PATH = "monitoring/latest_analysis.json"


def save_analysis(analysis: dict) -> None:
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "analysis": analysis,
    }
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_analysis() -> dict | None:
    if not os.path.exists(STORE_PATH):
        return None
    try:
        with open(STORE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("analysis")
    except (json.JSONDecodeError, OSError):
        return None
