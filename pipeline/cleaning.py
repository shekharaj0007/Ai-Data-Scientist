"""
Cleaning
Handles missing value imputation, duplicate removal, and anomaly detection.
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest


def handle_missing_values(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    strategy: 'auto' picks median/mode per column type,
              'knn' uses KNN imputation for numeric columns.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    if strategy == "knn" and len(numeric_cols) > 0:
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    else:
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        mode = df[col].mode()
        fill_value = mode[0] if not mode.empty else "Unknown"
        df[col] = df[col].fillna(fill_value)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """
    Flags anomalous rows using Isolation Forest on numeric columns.
    Adds an 'is_anomaly' column (1 = anomaly, 0 = normal).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        df["is_anomaly"] = 0
        return df

    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(numeric_df)
    df = df.copy()
    df["is_anomaly"] = (preds == -1).astype(int)
    return df


def detect_data_leakage(df: pd.DataFrame, target_col: str, threshold: float = 0.98) -> list:
    """
    Flags features that are suspiciously highly correlated with the target,
    a common sign of leakage.
    """
    if target_col not in df.columns:
        return []
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        return []

    correlations = numeric_df.corr()[target_col].abs().drop(target_col)
    suspicious = correlations[correlations > threshold].index.tolist()
    return suspicious
