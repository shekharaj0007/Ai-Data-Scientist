"""
Feature Engineering
Encoding, scaling, datetime feature extraction, and LLM-suggested features.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder


def extract_datetime_features(df: pd.DataFrame, datetime_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in datetime_cols:
        dt = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = dt.dt.year
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_day"] = dt.dt.day
        df[f"{col}_dayofweek"] = dt.dt.dayofweek
        df[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        df = df.drop(columns=[col])
    return df


def encode_categoricals(df: pd.DataFrame, target_col: str = None, high_cardinality_threshold: int = 15) -> pd.DataFrame:
    """
    One-hot encodes low-cardinality categoricals, target-encodes high-cardinality ones.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    categorical_cols = [c for c in categorical_cols if c != target_col]

    low_card = [c for c in categorical_cols if df[c].nunique() <= high_cardinality_threshold]
    high_card = [c for c in categorical_cols if df[c].nunique() > high_cardinality_threshold]

    if low_card:
        df = pd.get_dummies(df, columns=low_card, drop_first=True)

    if high_card and target_col is not None:
        encoder = TargetEncoder()
        df[high_card] = encoder.fit_transform(df[high_card], df[target_col])
    elif high_card:
        # No target available (e.g. clustering) — fall back to frequency encoding
        for col in high_card:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)

    return df


def scale_numeric(df: pd.DataFrame, exclude: list = None) -> pd.DataFrame:
    exclude = exclude or []
    df = df.copy()
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def suggest_features_with_llm(column_names: list, sample_rows: list, call_llm_fn) -> list:
    """
    Delegates to an LLM call (injected as call_llm_fn) to propose domain-relevant
    derived features beyond generic statistical transforms.

    call_llm_fn: a function(prompt: str) -> str, e.g. wrapping the Anthropic API.
    Returns a list of plain-text feature suggestions for a human/pipeline to review.
    """
    prompt = f"""Given a dataset with columns: {column_names}
and sample rows: {sample_rows[:3]}

Suggest 3-5 domain-relevant derived features that a generic AutoML pipeline
would likely miss. Return each suggestion as a short bullet with the
transformation logic in plain English."""

    response = call_llm_fn(prompt)
    if "LLM narrative unavailable" in response:
        return []
    return [line.strip("- ").strip() for line in response.split("\n") if line.strip()]
