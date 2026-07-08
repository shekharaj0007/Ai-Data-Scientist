"""
Experiment tracking via MLflow.
Logs dataset metadata, metrics, and model artifacts per pipeline run.
"""
import os
from typing import Any

import mlflow
import pandas as pd


def log_pipeline_run(
    filename: str,
    problem_type: str,
    leaderboard: pd.DataFrame,
    best_model_name: str,
    model_path: str,
    leakage_warnings: list,
    profile: dict | None = None,
) -> str:
    mlflow.set_experiment("ai_data_scientist")
    with mlflow.start_run(run_name=filename) as run:
        mlflow.log_param("source_file", filename)
        mlflow.log_param("problem_type", problem_type)
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("leakage_warnings", ",".join(leakage_warnings) if leakage_warnings else "none")

        if profile:
            mlflow.log_metric("n_rows", profile.get("n_rows", 0))
            mlflow.log_metric("n_cols", profile.get("n_cols", 0))

        if not leaderboard.empty:
            top = leaderboard.iloc[0]
            for col in leaderboard.columns:
                value = top[col]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(f"best_{col}", float(value))

        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="model")

        return run.info.run_id


def get_run_metrics(run_id: str) -> dict[str, Any]:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return {
        "run_id": run_id,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "status": run.info.status,
    }
