"""
Explainability — SHAP, permutation importance, metrics, cluster profiles,
text saliency, and LLM narratives for every problem type.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None


def build_explanation_report(
    problem_type: str,
    results: dict,
    target_col: str | None = None,
    df: pd.DataFrame | None = None,
    call_llm_fn=None,
    top_n: int = 10,
) -> dict:
    """Unified explainability entry point for all model types."""
    best_name = results.get("best_model_name", "model")
    best_model = results.get("best_model")

    report = {
        "method": "none",
        "global_importance": [],
        "metrics": {},
        "confusion_matrix": None,
        "class_labels": [],
        "cluster_profiles": None,
        "forecast_summary": None,
        "text_tokens": None,
        "narrative": None,
        "warnings": [],
    }

    if _is_deep_text_artifact(best_model) or (
        results.get("text_mode") and isinstance(best_model, dict)
    ):
        report.update(_explain_deep_text_from_artifact(best_model, top_n))
    elif problem_type in ("classification", "regression") and results.get("X_test") is not None:
        if isinstance(best_model, dict):
            report["warnings"].append("Best model is a deep-learning artifact; using text saliency if available.")
            if best_model.get("model_type") == "deep_text":
                report.update(_explain_deep_text_from_artifact(best_model, top_n))
        else:
            report.update(_explain_supervised(results, problem_type, top_n))
    elif problem_type == "clustering" and df is not None and not isinstance(best_model, dict):
        report.update(_explain_clustering(best_model, df, top_n))
    elif problem_type == "time_series" and target_col and df is not None and not isinstance(best_model, dict):
        report.update(_explain_time_series(best_model, df, target_col))
    elif results.get("best_model_name", "").startswith("pytorch_text"):
        report.update(_explain_deep_text(results, top_n))

    if call_llm_fn and report["global_importance"]:
        report["narrative"] = explain_results_with_llm(
            pd.DataFrame(report["global_importance"]),
            problem_type,
            best_name,
            call_llm_fn,
            report.get("metrics"),
        )
    elif call_llm_fn and report.get("cluster_profiles"):
        report["narrative"] = _narrate_clusters(call_llm_fn, best_name, report["cluster_profiles"])
    elif call_llm_fn and not report["global_importance"] and not report.get("narrative"):
        report["narrative"] = call_llm_fn(
            f"A {best_name} model was trained for {problem_type}. "
            f"Metrics: {report.get('metrics', {})}. "
            "Write 2-3 plain-English sentences summarizing model quality for a stakeholder."
        )

    return report


def _is_deep_text_artifact(model) -> bool:
    return isinstance(model, dict) and model.get("model_type") == "deep_text"


def _explain_supervised(results: dict, problem_type: str, top_n: int) -> dict:
    model = results["best_model"]
    X_test = results["X_test"]
    y_test = results["y_test"]
    feature_names = list(X_test.columns)

    out = {"metrics": {}, "warnings": []}

    # Try SHAP first for tree/linear models; fall back to permutation importance
    importance_df, method = _compute_global_importance(model, X_test, y_test, feature_names, top_n)
    out["method"] = method
    out["global_importance"] = importance_df.to_dict(orient="records")

    y_pred = model.predict(X_test)
    if problem_type == "classification":
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

        out["metrics"] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        }
        cm = confusion_matrix(y_test, y_pred)
        out["confusion_matrix"] = cm.tolist()
        out["class_labels"] = sorted(pd.Series(y_test).unique().astype(str).tolist())
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        out["metrics"]["classification_report"] = {
            k: v for k, v in report_dict.items() if isinstance(v, dict)
        }
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        out["metrics"] = {
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        }

    return out


def _compute_global_importance(model, X_test, y_test, feature_names, top_n):
    if shap is not None:
        try:
            shap_values = _safe_shap(model, X_test)
            importance_df = top_feature_importance(shap_values, feature_names, top_n)
            importance_df = importance_df.rename(columns={"mean_abs_shap": "importance"})
            importance_df["method_detail"] = "SHAP (Shapley additive values)"
            return importance_df, "shap"
        except Exception:
            pass

    from sklearn.inspection import permutation_importance

    try:
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": perm.importances_mean,
            "std": perm.importances_std,
        }).sort_values("importance", ascending=False).head(top_n)
        importance_df["method_detail"] = "Permutation importance"
        return importance_df, "permutation"
    except Exception:
        pass

    # Last resort: native feature_importances_ if available
    est = _unwrap_estimator(model)
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names[: len(imp)],
            "importance": imp,
        }).sort_values("importance", ascending=False).head(top_n)
        importance_df["method_detail"] = "Model-native feature importance"
        return importance_df, "native"

    return pd.DataFrame(columns=["feature", "importance"]), "none"


def _safe_shap(model, X_sample: pd.DataFrame):
    est = _unwrap_estimator(model)
    sample = X_sample.iloc[: min(200, len(X_sample))]

    if shap is None:
        raise ImportError("shap not installed")

    try:
        explainer = shap.TreeExplainer(est)
        return explainer.shap_values(sample)
    except Exception:
        pass

    try:
        background = shap.sample(sample, min(50, len(sample)))
        explainer = shap.Explainer(est.predict, background)
        return explainer(sample).values
    except Exception:
        background = shap.sample(sample, min(30, len(sample)))
        explainer = shap.KernelExplainer(est.predict, background)
        return explainer.shap_values(sample)


def _unwrap_estimator(model):
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def top_feature_importance(shap_values, feature_names: list, top_n: int = 10) -> pd.DataFrame:
    values = shap_values
    if isinstance(values, list):
        values = np.mean([np.abs(v) for v in values], axis=0)
    else:
        values = np.abs(values)

    if values.ndim > 2:
        values = values.mean(axis=-1)

    mean_abs = np.asarray(values).mean(axis=0)
    if len(mean_abs) != len(feature_names):
        feature_names = feature_names[: len(mean_abs)]

    return pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n)


def _explain_clustering(model, df: pd.DataFrame, top_n: int) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    if "is_anomaly" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["is_anomaly"])

    labels = model.predict(numeric_df)
    profile = numeric_df.groupby(labels).mean().round(4)
    sizes = pd.Series(labels).value_counts().sort_index()

    profiles = []
    for cluster_id in profile.index:
        row = profile.loc[cluster_id]
        top_feats = row.abs().sort_values(ascending=False).head(top_n)
        profiles.append({
            "cluster": int(cluster_id),
            "size": int(sizes[cluster_id]),
            "top_features": [
                {"feature": f, "mean_value": float(row[f])} for f in top_feats.index
            ],
        })

    global_imp = []
    variance_between = profile.var(axis=0).sort_values(ascending=False).head(top_n)
    for feat, var in variance_between.items():
        global_imp.append({"feature": feat, "importance": float(var), "method_detail": "Between-cluster variance"})

    return {
        "method": "cluster_profile",
        "global_importance": global_imp,
        "cluster_profiles": profiles,
        "metrics": {"n_clusters": int(len(profile)), "total_points": int(len(labels))},
    }


def _explain_time_series(model, df: pd.DataFrame, target_col: str) -> dict:
    series = df[target_col].dropna()
    train_size = int(len(series) * 0.8)
    test = series[train_size:]
    try:
        forecast = model.forecast(steps=len(test))
        mae = float(np.mean(np.abs(forecast.values - test.values)))
        rmse = float(np.sqrt(np.mean((forecast.values - test.values) ** 2)))
    except Exception:
        mae = rmse = None

    return {
        "method": "forecast_analysis",
        "global_importance": [],
        "forecast_summary": {
            "train_size": train_size,
            "test_size": len(test),
            "test_mae": round(mae, 4) if mae else None,
            "test_rmse": round(rmse, 4) if rmse else None,
        },
        "metrics": {"test_mae": round(mae, 4) if mae else None},
    }


def _explain_deep_text(results: dict, top_n: int) -> dict:
    artifact = results.get("best_model")
    if isinstance(artifact, dict) and artifact.get("model_type") == "deep_text":
        return _explain_deep_text_from_artifact(artifact, top_n)
    return {"method": "text_model", "global_importance": [], "warnings": ["Deep text explainability requires PyTorch artifact."]}


def _explain_deep_text_from_artifact(artifact: dict, top_n: int) -> dict:
    try:
        from . import deep_learning

        tokens = deep_learning.explain_text_tokens(artifact, top_n=top_n)
        return {
            "method": "text_saliency",
            "text_tokens": tokens,
            "global_importance": [
                {"feature": t["token"], "importance": t["score"], "method_detail": "Gradient saliency"}
                for t in tokens
            ],
            "metrics": {"architecture": artifact.get("architecture", "pytorch_text")},
        }
    except Exception as exc:
        return {"method": "text_saliency", "global_importance": [], "warnings": [str(exc)]}


def explain_results_with_llm(
    importance_df: pd.DataFrame,
    problem_type: str,
    model_name: str,
    call_llm_fn,
    metrics: dict | None = None,
) -> str:
    feature_summary = importance_df.to_string(index=False)
    metrics_block = ""
    if metrics:
        metrics_block = f"\nModel metrics: {metrics}\n"

    prompt = f"""A {model_name} model was trained for a {problem_type} task.
{metrics_block}
Top features by importance:

{feature_summary}

Write a 4-5 sentence plain-English explanation for a non-technical stakeholder:
1) What drives predictions most
2) How reliable the model appears based on metrics
3) Any caveats (avoid jargon like 'SHAP')"""

    return call_llm_fn(prompt)


def _narrate_clusters(call_llm_fn, model_name: str, profiles: list) -> str:
    summary = "\n".join(
        f"Cluster {p['cluster']}: {p['size']} points, top features: "
        + ", ".join(f"{t['feature']}={t['mean_value']}" for t in p["top_features"][:3])
        for p in profiles
    )
    return call_llm_fn(
        f"Clustering model {model_name} found these groups:\n{summary}\n"
        "Explain in plain English what distinguishes each cluster and how to use this for business decisions."
    )


# Legacy helpers kept for compatibility
def compute_shap_values(model, X_sample: pd.DataFrame):
    return None, _safe_shap(model, X_sample)
