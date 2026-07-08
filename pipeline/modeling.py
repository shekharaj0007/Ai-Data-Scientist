"""
Modeling
Trains and compares a broad catalog of algorithms per problem type and data modality.
"""
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import joblib

from .model_registry import (
    get_classification_models,
    get_regression_models,
    get_text_classification_models,
    get_clustering_models,
)

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def train_and_compare(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    model_dir: str = "models",
    text_columns: list | None = None,
    image_columns: list | None = None,
    modality_info: dict | None = None,
    raw_df: pd.DataFrame | None = None,
    preprocessor=None,
    use_case_id: str | None = None,
):
    modality_info = modality_info or {}
    text_columns = text_columns or modality_info.get("text_columns", [])
    image_columns = image_columns or modality_info.get("image_columns", [])
    source_df = raw_df if raw_df is not None else df

    if problem_type == "clustering":
        return _run_clustering(df, model_dir, preprocessor=preprocessor, use_case_id=use_case_id)

    if problem_type == "time_series":
        return _run_time_series(df, target_col, model_dir)

    if image_columns and problem_type == "classification":
        image_result = _run_image_classification(source_df, target_col, image_columns, model_dir)
        if image_result is not None:
            return image_result

    if text_columns and problem_type == "classification":
        text_result = _run_text_classification(source_df, target_col, text_columns, model_dir)
        if text_result is not None:
            return text_result

    tabular_result = _run_tabular_supervised(
        df, target_col, problem_type, model_dir, preprocessor=preprocessor, use_case_id=use_case_id
    )
    tabular_result = _append_deep_to_tabular(source_df, target_col, text_columns, tabular_result, model_dir)
    return tabular_result


def _run_tabular_supervised(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    model_dir: str,
    preprocessor=None,
    use_case_id: str | None = None,
):
    drop_cols = [target_col]
    if "is_anomaly" in df.columns:
        drop_cols.append("is_anomaly")

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    label_encoder = None
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_bank = (
        get_classification_models() if problem_type == "classification" else get_regression_models()
    )
    scoring = "accuracy" if problem_type == "classification" else "r2"
    cv_folds = 3 if len(model_bank) > 10 else 5

    leaderboard, fitted_models = _evaluate_model_bank(
        model_bank, X_train, y_train, X_test, y_test, scoring, cv_folds
    )

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("test_score", ascending=False)

    if use_case_id and problem_type == "classification":
        proba_rows = [
            row for row in leaderboard if hasattr(fitted_models[row["model"]], "predict_proba")
        ]
        if proba_rows:
            leaderboard_df = pd.DataFrame(proba_rows).sort_values("test_score", ascending=False)

    best_model_name = leaderboard_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    artifact = {
        "model": best_model,
        "label_encoder": label_encoder,
        "model_type": "tabular",
        "feature_columns": list(X_train.columns),
        "raw_input_columns": preprocessor.raw_input_columns if preprocessor else list(X_train.columns),
        "preprocessor": preprocessor,
        "use_case_id": use_case_id,
        "problem_type": problem_type,
        "target_col": target_col,
    }
    model_path = f"{model_dir}/best_model.pkl"
    joblib.dump(artifact, model_path)

    return {
        "leaderboard": leaderboard_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test,
        "model_path": model_path,
        "label_encoder": label_encoder,
        "models_trained": len(leaderboard),
    }


def _run_text_classification(df: pd.DataFrame, target_col: str, text_columns: list, model_dir: str):
    """Train TF-IDF + optional PyTorch text models on combined text columns."""
    from . import deep_learning

    text_cols = [c for c in text_columns if c in df.columns and c != target_col]
    if not text_cols:
        return None

    work = df[text_cols + [target_col]].dropna()
    if len(work) < 16:
        return None

    work = work.copy()
    texts = work[text_cols].astype(str).agg(" ".join, axis=1).tolist()
    y = work[target_col]
    work["_combined_text"] = texts
    X = work[["_combined_text"]]

    label_encoder = None
    if not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_bank = get_text_classification_models()

    leaderboard, fitted_models = _evaluate_model_bank(
        model_bank, X_train, y_train, X_test, y_test, "accuracy", 3, text_mode=True
    )

    best_sklearn_name = None
    best_sklearn_model = None
    best_sklearn_score = -1.0
    if leaderboard:
        best_row = max(leaderboard, key=lambda r: r["test_score"])
        best_sklearn_name = best_row["model"]
        best_sklearn_score = best_row["test_score"]
        best_sklearn_model = fitted_models[best_sklearn_name]

    deep_rows, deep_artifact, deep_name = deep_learning.train_text_models(
        texts, work[target_col].values, model_dir, text_cols
    )
    leaderboard.extend(deep_rows)

    if not leaderboard:
        return None

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("test_score", ascending=False)
    best_model_name = leaderboard_df.iloc[0]["model"]
    model_path = f"{model_dir}/best_model.pkl"

    if deep_artifact and best_model_name.startswith("pytorch_"):
        deep_learning.save_best_deep_artifact(deep_artifact, model_path)
        best_model = deep_artifact
        label_encoder = deep_artifact.get("label_encoder", label_encoder)
    else:
        joblib.dump(
            {
                "model": best_sklearn_model,
                "label_encoder": label_encoder,
                "model_type": "text",
                "text_columns": text_cols,
                "feature_columns": ["_combined_text"],
            },
            model_path,
        )
        best_model = best_sklearn_model

    return {
        "leaderboard": leaderboard_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test,
        "model_path": model_path,
        "label_encoder": label_encoder,
        "models_trained": len(leaderboard),
        "text_mode": True,
        "pytorch_used": bool(deep_rows),
    }


def _run_image_classification(df: pd.DataFrame, target_col: str, image_columns: list, model_dir: str):
    from . import deep_learning

    if not deep_learning.is_available():
        return None

    img_col = image_columns[0]
    work = df[[img_col, target_col]].dropna()
    if len(work) < 16:
        return None

    paths = work[img_col].astype(str).tolist()
    labels = work[target_col].values

    deep_rows, deep_artifact, deep_name = deep_learning.train_image_models(
        paths, labels, model_dir, image_columns
    )
    if not deep_rows or deep_artifact is None:
        return None

    model_path = f"{model_dir}/best_model.pkl"
    deep_learning.save_best_deep_artifact(deep_artifact, model_path)

    return {
        "leaderboard": pd.DataFrame(deep_rows),
        "best_model_name": deep_name,
        "best_model": deep_artifact,
        "model_path": model_path,
        "label_encoder": deep_artifact.get("label_encoder"),
        "models_trained": len(deep_rows),
        "pytorch_used": True,
    }


def _append_deep_to_tabular(df, target_col, text_columns, tabular_result, model_dir):
    """When tabular data also has text columns, append PyTorch text models to leaderboard."""
    from . import deep_learning

    if not text_columns or not deep_learning.is_available():
        return tabular_result

    text_cols = [c for c in text_columns if c in df.columns and c != target_col]
    if not text_cols:
        return tabular_result

    work = df[text_cols + [target_col]].dropna()
    if len(work) < 30:
        return tabular_result

    texts = work[text_cols].astype(str).agg(" ".join, axis=1).tolist()
    deep_rows, deep_artifact, _ = deep_learning.train_text_models(
        texts, work[target_col].values, model_dir, text_cols
    )
    if not deep_rows:
        return tabular_result

    lb = tabular_result["leaderboard"].to_dict(orient="records")
    lb.extend(deep_rows)
    leaderboard_df = pd.DataFrame(lb).sort_values("test_score", ascending=False)
    best_name = leaderboard_df.iloc[0]["model"]

    if best_name.startswith("pytorch_") and deep_artifact:
        model_path = tabular_result["model_path"]
        deep_learning.save_best_deep_artifact(deep_artifact, model_path)
        tabular_result["best_model"] = deep_artifact
        tabular_result["label_encoder"] = deep_artifact.get("label_encoder")

    tabular_result["leaderboard"] = leaderboard_df
    tabular_result["best_model_name"] = best_name
    tabular_result["models_trained"] = len(leaderboard_df)
    tabular_result["pytorch_used"] = True
    return tabular_result


def _evaluate_model_bank(model_bank, X_train, y_train, X_test, y_test, scoring, cv_folds, text_mode=False):
    leaderboard = []
    fitted_models = {}

    for name, model in model_bank.items():
        try:
            if not text_mode and len(X_train) > 5000 and name in ("svm_rbf", "svr_rbf"):
                continue  # skip slow kernels on large data
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1)
            model.fit(X_train, y_train)
            test_score = model.score(X_test, y_test)
            fitted_models[name] = model
            leaderboard.append({
                "model": name,
                "family": _model_family(name),
                "cv_mean_score": round(float(scores.mean()), 4),
                "cv_std": round(float(scores.std()), 4),
                "test_score": round(float(test_score), 4),
            })
        except Exception:
            continue

    return leaderboard, fitted_models


def _model_family(name: str) -> str:
    families = {
        "mlp_ann": "ann",
        "xgboost": "tree_ensemble",
        "lightgbm": "tree_ensemble",
        "catboost": "tree_ensemble",
        "random_forest": "tree_ensemble",
        "extra_trees": "tree_ensemble",
        "gradient_boosting": "tree_ensemble",
        "ada_boost": "tree_ensemble",
        "decision_tree": "tree_ensemble",
        "linear_svc": "linear",
        "linear_svr": "linear",
        "logistic_regression": "linear",
        "ridge_classifier": "linear",
        "linear_regression": "linear",
        "ridge": "linear",
        "lasso": "linear",
        "elastic_net": "linear",
        "svm_rbf": "kernel",
        "svr_rbf": "kernel",
        "knn": "instance_based",
        "gaussian_nb": "probabilistic",
        "tfidf_logistic": "text_linear",
        "tfidf_linear_svc": "text_linear",
        "tfidf_complement_nb": "text_probabilistic",
        "tfidf_random_forest": "text_ensemble",
        "pytorch_text_cnn": "cnn",
        "pytorch_text_lstm": "rnn",
        "pytorch_image_cnn": "cnn",
    }
    return families.get(name, "other")


def _run_clustering(df: pd.DataFrame, model_dir: str, preprocessor=None, use_case_id: str | None = None):
    numeric_df = df.select_dtypes(include=[np.number])
    if "is_anomaly" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["is_anomaly"])

    results = []
    best_score = -1
    best_model = None
    best_name = None

    # KMeans sweep
    for k in range(2, min(10, max(3, len(numeric_df) // 5))):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(numeric_df)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(numeric_df, labels)
        results.append({"model": f"kmeans_k{k}", "family": "clustering", "silhouette_score": round(float(score), 4)})
        if score > best_score:
            best_score = score
            best_model = model
            best_name = f"kmeans_k{k}"

    # Other clustering algorithms
    for name, model in get_clustering_models().items():
        try:
            labels = model.fit_predict(numeric_df)
            if len(set(labels)) < 2 or (labels == -1).mean() > 0.5:
                continue
            score = silhouette_score(numeric_df, labels)
            results.append({"model": name, "family": "clustering", "silhouette_score": round(float(score), 4)})
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        except Exception:
            continue

    model_path = f"{model_dir}/best_model.pkl"
    joblib.dump(
        {
            "model": best_model,
            "label_encoder": None,
            "model_type": "clustering",
            "feature_columns": list(numeric_df.columns),
            "raw_input_columns": preprocessor.raw_input_columns if preprocessor else list(numeric_df.columns),
            "preprocessor": preprocessor,
            "use_case_id": use_case_id,
            "problem_type": "clustering",
        },
        model_path,
    )

    return {
        "leaderboard": pd.DataFrame(results).sort_values("silhouette_score", ascending=False),
        "best_model_name": best_name or "kmeans",
        "best_model": best_model,
        "model_path": model_path,
        "models_trained": len(results),
    }


def _run_time_series(df: pd.DataFrame, target_col: str, model_dir: str):
    if ARIMA is None:
        raise ImportError("statsmodels is required for time series forecasting")

    series = df[target_col].dropna()
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    results = []
    best_model = None
    best_name = None
    best_mae = float("inf")

    for order, label in [((1, 1, 1), "ARIMA(1,1,1)"), ((2, 1, 1), "ARIMA(2,1,1)"), ((1, 1, 2), "ARIMA(1,1,2)")]:
        try:
            model = ARIMA(train, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=len(test))
            mae = float(np.mean(np.abs(forecast.values - test.values)))
            results.append({"model": label, "family": "statistical", "test_mae": round(mae, 4)})
            if mae < best_mae:
                best_mae = mae
                best_model = fitted
                best_name = label
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("All time series models failed to fit")

    model_path = f"{model_dir}/best_model.pkl"
    joblib.dump({"model": best_model, "label_encoder": None, "model_type": "time_series"}, model_path)

    return {
        "leaderboard": pd.DataFrame(results).sort_values("test_mae"),
        "best_model_name": best_name,
        "best_model": best_model,
        "model_path": model_path,
        "models_trained": len(results),
    }
