"""
Orchestrator
Runs the full pipeline end-to-end: ingestion -> cleaning -> feature engineering
-> problem detection -> modeling -> explainability. This is the layer an LLM
agent would call as tools, deciding parameters and interpreting outputs.
"""
import os
import shutil

import pandas as pd

from . import (
    ingestion,
    cleaning,
    feature_engineering,
    problem_detection,
    modeling,
    explainability,
    tracking,
    monitoring,
    modality_detection,
    architecture_advisor,
    preprocessing,
    use_cases,
)


def run_pipeline(
    filepath: str,
    target_col: str = None,
    model_dir: str = "models",
    call_llm_fn=None,
    log_to_mlflow: bool = True,
    filename: str | None = None,
    use_case_id: str | None = None,
) -> dict:
    filename = filename or os.path.basename(filepath)
    use_case = use_cases.get_use_case(use_case_id) if use_case_id else None
    if use_case and not target_col:
        target_col = use_case.get("target_col")

    id_columns = use_case.get("id_columns", []) if use_case else []

    # 1. Ingest & profile
    df = ingestion.load_csv(filepath)
    profile = ingestion.profile_data(df)
    datetime_cols = ingestion.detect_datetime_columns(df)
    modality_info = modality_detection.detect_modalities(df, filepath, target_col)

    # 2. Clean
    df = cleaning.remove_duplicates(df)
    df = cleaning.handle_missing_values(df, strategy="auto")
    df = cleaning.detect_anomalies(df)
    if id_columns:
        df = df.drop(columns=[c for c in id_columns if c in df.columns], errors="ignore")

    leakage_warnings = []
    target_warnings = []
    if target_col:
        target_warnings = problem_detection.validate_target_column(df, target_col)
        leakage_warnings = cleaning.detect_data_leakage(df, target_col)

    # 3. Detect problem type (before heavy feature engineering, since it affects encoding choices)
    problem_type = problem_detection.detect_problem_type(df, target_col, datetime_cols)

    n_classes = None
    if target_col and target_col in df.columns and problem_type == "classification":
        n_classes = int(df[target_col].nunique())

    architecture_plan = architecture_advisor.recommend_architectures(
        problem_type, modality_info, n_rows=len(df), n_classes=n_classes
    )

    # 4. Feature engineering (serializable for production inference)
    feature_suggestions = []
    if datetime_cols:
        datetime_cols = [c for c in datetime_cols if c != target_col]

    preprocessor = preprocessing.ProductionPreprocessor()
    df = preprocessor.fit_transform(
        df,
        target_col=target_col,
        datetime_cols=datetime_cols,
        exclude_columns=id_columns,
    )

    if call_llm_fn:
        sample_rows = df.head(3).to_dict(orient="records")
        feature_suggestions = feature_engineering.suggest_features_with_llm(
            list(df.columns), sample_rows, call_llm_fn
        )

    monitoring.save_training_baseline(df)

    # 5. Train & compare models (use raw df for text/image deep learning)
    df_raw = ingestion.load_csv(filepath)
    df_raw = cleaning.handle_missing_values(df_raw, strategy="auto")
    results = modeling.train_and_compare(
        df,
        target_col,
        problem_type,
        model_dir=model_dir,
        text_columns=modality_info.get("text_columns"),
        image_columns=modality_info.get("image_columns"),
        modality_info=modality_info,
        raw_df=df_raw,
        preprocessor=preprocessor,
        use_case_id=use_case_id,
    )

    run_id = None
    if log_to_mlflow:
        run_id = tracking.log_pipeline_run(
            filename=filename,
            problem_type=problem_type,
            leaderboard=results["leaderboard"],
            best_model_name=results["best_model_name"],
            model_path=results.get("model_path", f"{model_dir}/best_model.pkl"),
            leakage_warnings=leakage_warnings,
            profile=profile,
        )

    # 6. Explainability — all problem types
    explanation = explainability.build_explanation_report(
        problem_type=problem_type,
        results=results,
        target_col=target_col,
        df=df,
        call_llm_fn=call_llm_fn,
        top_n=10,
    )

    importance_df = explanation.get("global_importance")
    if importance_df:
        importance_df = pd.DataFrame(importance_df)
    else:
        importance_df = None

    return {
        "run_id": run_id,
        "profile": profile,
        "problem_type": problem_type,
        "modality_info": modality_info,
        "architecture_plan": architecture_plan,
        "leakage_warnings": leakage_warnings,
        "target_warnings": target_warnings,
        "feature_suggestions": feature_suggestions,
        "leaderboard": results["leaderboard"],
        "best_model_name": results["best_model_name"],
        "models_trained": results.get("models_trained", 0),
        "feature_importance": importance_df,
        "explanation": explanation,
        "narrative": explanation.get("narrative"),
        "model_path": results.get("model_path", f"{model_dir}/best_model.pkl"),
        "use_case_id": use_case_id,
        "use_case": use_case,
    }


def approve_model(staging_dir: str = "models/staging", production_dir: str = "models") -> str:
    """
    Human review checkpoint: promote a staged model to production only after approval.
    """
    staged = os.path.join(staging_dir, "best_model.pkl")
    if not os.path.exists(staged):
        raise FileNotFoundError("No staged model found. Run /analyze first.")

    os.makedirs(production_dir, exist_ok=True)
    dest = os.path.join(production_dir, "best_model.pkl")
    shutil.copy2(staged, dest)
    return dest
