"""
Problem Type Detection
Infers whether the task is regression, classification, clustering, or time-series
based on the target column (if any) and data characteristics.
"""
import pandas as pd
import numpy as np

ID_LIKE_NAMES = {"id", "uuid", "index", "key", "customer_id", "lead_id", "parentid"}


def validate_target_column(df: pd.DataFrame, target_col: str | None) -> list[str]:
    """
    Validate that the chosen target can be used for supervised learning.
    Raises ValueError with a plain-English message when invalid.
    """
    if not target_col:
        return []

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataset.")

    target = df[target_col].dropna()
    if len(target) < 10:
        raise ValueError(
            f"Target column '{target_col}' has too few non-empty rows ({len(target)}). "
            "Need at least 10 rows to train a model."
        )

    n_unique = int(target.nunique())
    ratio = n_unique / len(target)

    if n_unique <= 1:
        raise ValueError(
            f"Target column '{target_col}' has only one unique value. "
            "Pick a column with variation (e.g. Yes/No, 0/1, or numeric outcomes)."
        )

    name = target_col.lower().replace(" ", "_")
    looks_like_id = name in ID_LIKE_NAMES or name.endswith("_id") or name.endswith("id")

    if n_unique == len(target) and n_unique > 20:
        raise ValueError(
            f"'{target_col}' is a unique identifier (every row has a different value). "
            "That column is an ID, not something to predict. "
            "Choose your outcome column (e.g. Churn, Converted) or leave target blank for clustering."
        )

    if looks_like_id and ratio > 0.85:
        raise ValueError(
            f"'{target_col}' looks like an ID column ({n_unique} unique values). "
            "The target must be what you want the model to predict, not a row identifier."
        )

    if n_unique > 500 and ratio > 0.8:
        raise ValueError(
            f"Target '{target_col}' has too many unique values ({n_unique}) for classification. "
            "Use a categorical outcome (e.g. Yes/No) or leave blank for unsupervised clustering."
        )

    warnings = []
    if ratio > 0.5 and n_unique > 50:
        warnings.append(
            f"Target '{target_col}' has many unique values — results may be unreliable."
        )
    return warnings


def detect_problem_type(df: pd.DataFrame, target_col: str = None, datetime_cols: list = None) -> str:
    datetime_cols = datetime_cols or []

    if target_col is None or target_col not in df.columns:
        return "clustering"

    target = df[target_col].dropna()

    if len(datetime_cols) > 0 and pd.api.types.is_numeric_dtype(target):
        # Strong time-series signal: a datetime column present alongside a numeric target
        return "time_series"

    if pd.api.types.is_numeric_dtype(target):
        n_unique = target.nunique()
        # Heuristic: small integer-like value set relative to data size -> classification
        looks_categorical = n_unique <= 20 and (n_unique / len(target) < 0.3 or n_unique <= 10)
        is_int_like = (target == target.round()).all()
        if looks_categorical and is_int_like:
            return "classification"
        return "regression"
    else:
        return "classification"
