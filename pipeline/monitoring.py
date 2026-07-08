"""
Lightweight post-deployment monitoring helpers.
Stores prediction inputs and flags simple distribution drift vs. training baseline.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


PREDICTION_LOG = "monitoring/predictions.jsonl"
BASELINE_PATH = "monitoring/baseline.json"


def ensure_monitoring_dir() -> None:
    os.makedirs("monitoring", exist_ok=True)


def save_training_baseline(df: pd.DataFrame) -> None:
    """Persist numeric column means/stds from the training set for drift checks."""
    ensure_monitoring_dir()
    numeric = df.select_dtypes(include=[np.number])
    baseline = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "columns": {
            col: {"mean": float(numeric[col].mean()), "std": float(numeric[col].std())}
            for col in numeric.columns
            if numeric[col].std() > 0
        },
    }
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)


def log_prediction(records: list[dict], predictions: list) -> None:
    ensure_monitoring_dir()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_records": len(records),
        "predictions": predictions,
        "input_means": _numeric_means(records),
    }
    with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def check_drift(records: list[dict], z_threshold: float = 3.0) -> list[str]:
    """
    Compare incoming batch means to training baseline.
    Returns human-readable warnings when a feature mean drifts beyond z_threshold std devs.
    """
    if not os.path.exists(BASELINE_PATH):
        return []

    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)

    incoming = _numeric_means(records)
    warnings = []
    for col, stats in baseline["columns"].items():
        if col not in incoming:
            continue
        z = abs(incoming[col] - stats["mean"]) / stats["std"]
        if z > z_threshold:
            warnings.append(
                f"{col}: incoming mean {incoming[col]:.3f} deviates {z:.1f}σ from training ({stats['mean']:.3f})"
            )
    return warnings


def _numeric_means(records: list[dict]) -> dict[str, float]:
    if not records:
        return {}
    df = pd.DataFrame(records)
    numeric = df.select_dtypes(include=[np.number])
    return {col: float(numeric[col].mean()) for col in numeric.columns}
