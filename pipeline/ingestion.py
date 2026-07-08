"""
Ingestion & Profiling
Loads a CSV, infers schema, and produces a data quality report.
"""
import pandas as pd
import numpy as np


def load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def profile_data(df: pd.DataFrame) -> dict:
    """Generate a structured profile of the dataset."""
    profile = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": {}
    }

    for col in df.columns:
        col_data = df[col]
        col_profile = {
            "dtype": str(col_data.dtype),
            "missing_count": int(col_data.isna().sum()),
            "missing_pct": round(float(col_data.isna().mean() * 100), 2),
            "unique_count": int(col_data.nunique()),
        }

        if pd.api.types.is_numeric_dtype(col_data):
            col_profile.update({
                "mean": float(col_data.mean()) if not col_data.isna().all() else None,
                "std": float(col_data.std()) if not col_data.isna().all() else None,
                "min": float(col_data.min()) if not col_data.isna().all() else None,
                "max": float(col_data.max()) if not col_data.isna().all() else None,
            })
        else:
            top_values = col_data.value_counts().head(5).to_dict()
            col_profile["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        profile["columns"][col] = col_profile

    return profile


def detect_datetime_columns(df: pd.DataFrame) -> list:
    """Heuristically detect columns that are likely datetime."""
    candidates = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(50)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            success_rate = parsed.notna().mean()
            if success_rate < 0.8:
                continue
            years = parsed.dt.year.dropna()
            if years.empty or years.min() < 1900 or years.max() > 2100:
                continue
            candidates.append(col)
    return candidates
