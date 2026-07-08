"""
Detects data modalities: tabular, text, image, time_series, multimodal.
"""
import os
import re

import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv"}


def detect_modalities(df: pd.DataFrame, filepath: str = "", target_col: str | None = None) -> dict:
    text_cols = _detect_text_columns(df, target_col)
    image_cols = _detect_image_columns(df, target_col)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        c for c in df.select_dtypes(exclude=[np.number]).columns
        if c not in text_cols and c != target_col
    ]

    has_tabular = len(numeric_cols) > 0 or len(categorical_cols) > 0
    has_text = len(text_cols) > 0
    has_image = len(image_cols) > 0

    modalities = []
    if has_tabular:
        modalities.append("tabular")
    if has_text:
        modalities.append("text")
    if has_image:
        modalities.append("image")

    primary = _primary_modality(has_tabular, has_text, has_image, len(df))

    return {
        "primary_modality": primary,
        "modalities": modalities,
        "text_columns": text_cols,
        "image_columns": image_cols,
        "numeric_columns": [c for c in numeric_cols if c != target_col],
        "categorical_columns": categorical_cols,
        "is_multimodal": len(modalities) > 1,
        "file_type": os.path.splitext(filepath)[1].lower() if filepath else ".csv",
    }


def _detect_text_columns(df: pd.DataFrame, target_col: str | None) -> list[str]:
    cols = []
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        avg_len = series.str.len().mean()
        avg_words = series.str.split().str.len().mean()
        unique_ratio = series.nunique() / len(series)
        if (
            avg_len > 35
            or (avg_len > 12 and unique_ratio > 0.3)
            or (avg_len > 8 and series.nunique() > 10)
            or (avg_words >= 3 and avg_len > 15)
        ):
            cols.append(col)
    return cols


def _detect_image_columns(df: pd.DataFrame, target_col: str | None) -> list[str]:
    cols = []
    path_pattern = re.compile(r"\.(jpg|jpeg|png|gif|bmp|webp|tiff|tif)$", re.I)
    for col in df.columns:
        if col == target_col:
            continue
        sample = df[col].dropna().astype(str).head(50)
        if sample.empty:
            continue
        hits = sample.str.contains(path_pattern).mean()
        if hits >= 0.6:
            cols.append(col)
    return cols


def _primary_modality(has_tabular: bool, has_text: bool, has_image: bool, n_rows: int) -> str:
    if has_image and has_text:
        return "multimodal"
    if has_image and has_tabular:
        return "multimodal"
    if has_image:
        return "image"
    if has_text and not has_tabular:
        return "text"
    if has_text and has_tabular:
        return "multimodal"
    return "tabular"
