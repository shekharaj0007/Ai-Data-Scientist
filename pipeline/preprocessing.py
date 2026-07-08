"""
Production preprocessing pipeline — fit on training data, replay at inference time.
Ensures raw CSV columns can be sent to /predict without manual feature engineering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, TargetEncoder


def extract_datetime_features(df: pd.DataFrame, datetime_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in datetime_cols:
        if col not in df.columns:
            continue
        dt = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = dt.dt.year
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_day"] = dt.dt.day
        df[f"{col}_dayofweek"] = dt.dt.dayofweek
        df[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        df = df.drop(columns=[col])
    return df


INTERNAL_COLUMNS = {"is_anomaly"}


class ProductionPreprocessor:
    """Serializable preprocessing: datetime features, categoricals, scaling."""

    def __init__(self):
        self.datetime_cols: list[str] = []
        self.target_col: str | None = None
        self.exclude_scale: list[str] = []
        self.low_card_cols: list[str] = []
        self.high_card_cols: list[str] = []
        self.dummy_columns: list[str] = []
        self.freq_maps: dict[str, dict] = {}
        self.target_encoder: TargetEncoder | None = None
        self.target_encoder_cols: list[str] = []
        self.scaler: StandardScaler | None = None
        self.scale_cols: list[str] = []
        self.feature_columns: list[str] = []
        self.raw_input_columns: list[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
        datetime_cols: list | None = None,
        exclude_columns: list | None = None,
    ) -> pd.DataFrame:
        df = df.copy()
        self.target_col = target_col
        self.datetime_cols = [c for c in (datetime_cols or []) if c and c != target_col]
        skip = INTERNAL_COLUMNS | set(exclude_columns or [])
        self.raw_input_columns = [
            c for c in df.columns if c != target_col and c not in skip
        ]
        self.exclude_scale = [c for c in (target_col, "is_anomaly") if c and c in df.columns]

        if self.datetime_cols:
            df = extract_datetime_features(df, self.datetime_cols)

        df = self._fit_encode_categoricals(df)
        df = self._fit_scale(df)
        self.feature_columns = [c for c in df.columns if c not in self.exclude_scale]
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing = [c for c in self.raw_input_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Required: {self.raw_input_columns}")

        if self.datetime_cols:
            df = extract_datetime_features(df, self.datetime_cols)

        for col in INTERNAL_COLUMNS:
            if col in self.feature_columns and col not in df.columns:
                df[col] = 0

        df = self._transform_encode_categoricals(df)
        df = self._transform_scale(df)
        return df.reindex(columns=self.feature_columns, fill_value=0)

    def _fit_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c != self.target_col]

        self.low_card_cols = [c for c in categorical_cols if df[c].nunique() <= 15]
        self.high_card_cols = [c for c in categorical_cols if df[c].nunique() > 15]

        if self.low_card_cols:
            df = pd.get_dummies(df, columns=self.low_card_cols, drop_first=True)

        if self.high_card_cols and self.target_col and self.target_col in df.columns:
            target_nunique = df[self.target_col].nunique()
            use_target_encoder = target_nunique <= 50 and target_nunique < len(df) * 0.5
            if use_target_encoder:
                self.target_encoder = TargetEncoder()
                self.target_encoder_cols = self.high_card_cols.copy()
                df[self.high_card_cols] = self.target_encoder.fit_transform(
                    df[self.high_card_cols], df[self.target_col]
                )
            else:
                for col in self.high_card_cols:
                    freq = df[col].value_counts(normalize=True)
                    self.freq_maps[col] = freq.to_dict()
                    df[col] = df[col].map(self.freq_maps[col]).fillna(0)
        elif self.high_card_cols:
            for col in self.high_card_cols:
                freq = df[col].value_counts(normalize=True)
                self.freq_maps[col] = freq.to_dict()
                df[col] = df[col].map(self.freq_maps[col]).fillna(0)

        self.dummy_columns = [c for c in df.columns if c != self.target_col]
        return df

    def _transform_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.low_card_cols:
            df = pd.get_dummies(df, columns=self.low_card_cols, drop_first=True)

        if self.target_encoder is not None and self.target_encoder_cols:
            present = [c for c in self.target_encoder_cols if c in df.columns]
            if present:
                df[present] = self.target_encoder.transform(df[present])
        elif self.freq_maps:
            for col, freq in self.freq_maps.items():
                if col in df.columns:
                    df[col] = df[col].map(freq).fillna(0)

        for col in self.dummy_columns:
            if col not in df.columns:
                df[col] = 0
        return df

    def _fit_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        self.scale_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in self.exclude_scale
        ]
        if self.scale_cols:
            self.scaler = StandardScaler()
            df[self.scale_cols] = self.scaler.fit_transform(df[self.scale_cols])
        return df

    def _transform_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler and self.scale_cols:
            present = [c for c in self.scale_cols if c in df.columns]
            if present:
                df[present] = self.scaler.transform(df[present])
        return df
