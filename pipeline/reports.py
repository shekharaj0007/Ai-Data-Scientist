"""
Export prediction results as CSV reports for business teams.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd


def build_report_csv(records: list[dict], result: dict, use_case: dict | None = None) -> str:
    """Combine input records with model outputs into a CSV string."""
    prob_label = (use_case or {}).get("probability_label", "probability")
    rows = []

    for i, record in enumerate(records):
        row = dict(record)
        row["prediction"] = result["predictions"][i]

        prob = None
        if result.get("probabilities"):
            prob = result["probabilities"][i]
        elif result.get("probability_scores"):
            prob = result["probability_scores"][i]

        if prob is not None:
            row[prob_label] = round(float(prob), 4)
            row[f"{prob_label}_pct"] = round(float(prob) * 100, 1)

        if result.get("risk_tier"):
            row["risk_tier"] = result["risk_tier"][i]
        if result.get("recommended_action"):
            row["recommended_action"] = result["recommended_action"][i]

        rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def report_filename(use_case: dict | None = None) -> str:
    prefix = (use_case or {}).get("report_prefix", "prediction_report")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}.csv"
