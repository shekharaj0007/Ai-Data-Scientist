"""
FastAPI entry point for the AI Data Scientist service.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

_APP_DIR = Path(__file__).resolve().parent
for _env_path in (_APP_DIR / ".env", Path.cwd() / ".env"):
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
        break

import shutil
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import (
    ingestion,
    orchestrator,
    monitoring,
    modality_detection,
    use_cases,
    reports,
    analysis_store,
)

try:
    from pipeline import deep_learning
except ImportError:
    deep_learning = None

app = FastAPI(
    title="AI Data Scientist API",
    description=(
        "Automated ML pipeline: ingest CSV data, train and compare models, "
        "generate SHAP explainability, and deploy with human review."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
STAGING_DIR = "models/staging"
PRODUCTION_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(PRODUCTION_DIR, exist_ok=True)

# Latest analysis — persisted to disk so it survives restarts
_latest_analysis: dict | None = analysis_store.load_analysis()


def _anthropic_api_key() -> str | None:
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip().strip('"').strip("'")
    return key or None


def call_llm_fn(prompt: str) -> str:
    """
    Wraps a call to the Anthropic API for narrative generation.
    Falls back to a stub when ANTHROPIC_API_KEY is not set.
    """
    api_key = _anthropic_api_key()
    if not api_key:
        return (
            "LLM narrative unavailable (set ANTHROPIC_API_KEY in .env and restart start.bat). "
            "Review the feature_importance table for model drivers."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _prepare_features(artifact: dict, records: list[dict]) -> pd.DataFrame:
    """Transform raw business records into model-ready features."""
    input_df = pd.DataFrame(records)
    preprocessor = artifact.get("preprocessor")
    raw_columns = artifact.get("raw_input_columns", [])
    feature_columns = artifact.get("feature_columns", [])

    if preprocessor is not None:
        return preprocessor.transform(input_df)

    if raw_columns:
        missing = [c for c in raw_columns if c not in input_df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Required: {raw_columns}")
        input_df = input_df[raw_columns]

    if feature_columns:
        missing = [c for c in feature_columns if c not in input_df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Required: {feature_columns}")
        return input_df[feature_columns]

    return input_df


def _extract_probabilities(model, features_df: pd.DataFrame, artifact: dict) -> list | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_df)
        pos_idx = _positive_class_index(artifact, model)
        if pos_idx is not None:
            return proba[:, pos_idx].tolist()
        return proba.max(axis=1).tolist()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(features_df)
        if getattr(scores, "ndim", 1) == 1:
            probs = 1 / (1 + np.exp(-scores))
            return probs.tolist()
    return None


def _positive_class_index(artifact: dict, model) -> int | None:
    use_case = use_cases.get_use_case(artifact.get("use_case_id", ""))
    if not use_case or artifact.get("problem_type") != "classification":
        return None

    positive = use_case.get("positive_class")
    label_encoder = artifact.get("label_encoder")
    if label_encoder is not None:
        classes = list(label_encoder.classes_)
        if positive in classes:
            return classes.index(positive)
        return 1 if len(classes) > 1 else 0

    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if positive in classes:
            return classes.index(positive)
        numeric_pos = 1 if len(classes) > 1 else 0
        return numeric_pos
    return 1


def _build_business_output(artifact: dict, model, predictions, probabilities: list | None):
    use_case = use_cases.get_use_case(artifact.get("use_case_id", ""))
    if not use_case or artifact.get("problem_type") != "classification":
        return {}

    thresholds = use_case.get("risk_thresholds")
    prob_label = use_case.get("probability_label", "probability")
    risk_tiers = []
    actions = []
    prob_scores = []

    for i, pred in enumerate(predictions):
        prob = probabilities[i] if probabilities else None
        if prob is None:
            risk_tiers.append("unknown")
            actions.append("Review prediction manually.")
            prob_scores.append(None)
            continue
        tier = use_cases.risk_tier(prob, thresholds)
        risk_tiers.append(tier)
        actions.append(use_cases.recommended_action(use_case, tier))
        prob_scores.append(round(prob, 4))

    output = {
        "probability_label": prob_label,
        "probability_scores": prob_scores,
        "high_risk_label": use_case.get("high_risk_label", "Score"),
        "risk_tier": risk_tiers,
        "recommended_action": actions,
        "use_case": use_case["name"],
        "use_case_id": use_case["id"],
    }
    if use_case["id"] == "customer_churn":
        output["churn_probability"] = prob_scores
    return output


def _load_production_artifact():
    model_path = os.path.join(PRODUCTION_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="No production model available. Approve a staged model first.")
    return joblib.load(model_path)


def _run_production_inference(records: list[dict]) -> dict:
    artifact = _load_production_artifact()

    if isinstance(artifact, dict):
        model = artifact["model"]
        label_encoder = artifact.get("label_encoder")
        model_type = artifact.get("model_type", "tabular")
        text_columns = artifact.get("text_columns", [])
    else:
        model = artifact
        label_encoder = None
        model_type = "tabular"
        text_columns = []

    if model_type == "time_series":
        raise HTTPException(
            status_code=400,
            detail="Time-series models do not support batch record prediction.",
        )

    input_df = pd.DataFrame(records)
    probabilities = None

    try:
        if model_type in ("deep_text", "deep_image"):
            if deep_learning is None or not deep_learning.is_available():
                raise HTTPException(status_code=500, detail="PyTorch required for this model but not installed.")
            predictions = deep_learning.predict_deep(artifact, records)
        elif model_type == "text" and text_columns:
            missing = [c for c in text_columns if c not in input_df.columns]
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing columns: {missing}. Required: {text_columns}")
            input_df = pd.DataFrame({
                "_combined_text": input_df[text_columns].astype(str).agg(" ".join, axis=1)
            })
            predictions = model.predict(input_df)
        else:
            features_df = _prepare_features(artifact, records)
            predictions = model.predict(features_df)
            probabilities = _extract_probabilities(model, features_df, artifact)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    if label_encoder is not None and model_type not in ("deep_text", "deep_image"):
        predictions = label_encoder.inverse_transform(predictions.astype(int))

    drift_warnings = monitoring.check_drift(records)
    monitoring.log_prediction(records, predictions.tolist())

    response = {
        "predictions": predictions.tolist(),
        "drift_warnings": drift_warnings,
    }
    if probabilities is not None:
        response["probabilities"] = probabilities
    response.update(_build_business_output(artifact, model, response["predictions"], probabilities))
    return response


@app.get("/demo/{use_case_id}/sample-batch")
async def demo_sample_batch(use_case_id: str):
    """Sample records for batch inference and report export demos."""
    if not use_cases.get_use_case(use_case_id):
        raise HTTPException(status_code=404, detail=f"Unknown use case: {use_case_id}")
    records = use_cases.get_sample_batch(use_case_id)
    return {"use_case_id": use_case_id, "records": records}


@app.get("/use-cases")
async def list_use_cases():
    return {"use_cases": use_cases.list_use_cases()}


@app.post("/demo/{use_case_id}")
async def load_demo(use_case_id: str):
    try:
        demo = use_cases.load_demo_dataset(use_case_id, UPLOAD_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    df = ingestion.load_csv(demo["filepath"])
    profile = ingestion.profile_data(df)
    modality_info = modality_detection.detect_modalities(df, demo["filepath"], demo["target_col"])

    return {
        "filename": demo["filename"],
        "columns": list(df.columns),
        "profile": profile,
        "modality_info": modality_info,
        "use_case_id": demo["use_case_id"],
        "target_col": demo["target_col"],
        "use_case": {
            "id": demo["use_case"]["id"],
            "name": demo["use_case"]["name"],
            "description": demo["use_case"]["description"],
            "business_goal": demo["use_case"]["business_goal"],
            "industry": demo["use_case"]["industry"],
        },
    }


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    df = ingestion.load_csv(filepath)
    profile = ingestion.profile_data(df)
    modality_info = modality_detection.detect_modalities(df, filepath)

    return {
        "filename": file.filename,
        "columns": list(df.columns),
        "profile": profile,
        "modality_info": modality_info,
    }


class AnalyzeRequest(BaseModel):
    filename: str
    target_col: str | None = None
    use_llm: bool = True
    use_case_id: str | None = None


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    global _latest_analysis

    filepath = os.path.join(UPLOAD_DIR, request.filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found. Upload it first via /upload.")

    llm_fn = call_llm_fn if request.use_llm and _anthropic_api_key() else None
    try:
        results = orchestrator.run_pipeline(
            filepath=filepath,
            target_col=request.target_col,
            model_dir=STAGING_DIR,
            call_llm_fn=llm_fn,
            filename=request.filename,
            use_case_id=request.use_case_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Pipeline failed: {exc}") from exc

    response = {
        "status": "pending_review",
        "filename": request.filename,
        "run_id": results["run_id"],
        "problem_type": results["problem_type"],
        "modality_info": results["modality_info"],
        "architecture_plan": results["architecture_plan"],
        "leakage_warnings": results["leakage_warnings"],
        "target_warnings": results.get("target_warnings", []),
        "feature_suggestions": results["feature_suggestions"],
        "leaderboard": results["leaderboard"].to_dict(orient="records"),
        "best_model_name": results["best_model_name"],
        "models_trained": results.get("models_trained", 0),
        "narrative": results["narrative"],
        "use_case_id": results.get("use_case_id"),
        "use_case": results.get("use_case"),
        "llm_enabled": llm_fn is not None,
        "message": "Model staged. Call POST /approve to deploy to production.",
    }
    if results["feature_importance"] is not None:
        response["feature_importance"] = results["feature_importance"].to_dict(orient="records")
    if results.get("explanation"):
        response["explanation"] = results["explanation"]

    _latest_analysis = response
    analysis_store.save_analysis(response)
    return response


@app.get("/analysis/latest")
async def latest_analysis():
    if _latest_analysis is None:
        raise HTTPException(status_code=404, detail="No analysis run yet.")
    return _latest_analysis


class ApproveRequest(BaseModel):
    confirmed: bool = True


@app.post("/approve")
async def approve_model(request: ApproveRequest):
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="Approval not confirmed.")

    try:
        dest = orchestrator.approve_model(STAGING_DIR, PRODUCTION_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"status": "deployed", "model_path": dest}


@app.get("/model/download")
async def download_model():
    model_path = os.path.join(PRODUCTION_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="No production model available yet.")
    return FileResponse(model_path, filename="best_model.pkl")


class PredictRequest(BaseModel):
    records: list[dict]


@app.get("/model/info")
async def model_info():
    model_path = os.path.join(PRODUCTION_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="No production model available yet.")

    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        model = artifact.get("model")
        model_type = artifact.get("model_type", "tabular")
        use_case = use_cases.get_use_case(artifact.get("use_case_id", ""))
    else:
        model = artifact
        model_type = "tabular"
        use_case = None

    raw_columns = artifact.get("raw_input_columns", []) if isinstance(artifact, dict) else []
    accepts_raw = bool(artifact.get("preprocessor")) if isinstance(artifact, dict) else False

    if model_type == "deep_text":
        required = artifact.get("text_columns", [])
        sample = [{col: "your text here" for col in required}]
    elif model_type == "deep_image":
        required = artifact.get("image_columns", [])
        sample = [{col: "C:/path/to/image.jpg" for col in required}]
    elif model_type == "text":
        required = artifact.get("text_columns", [])
        sample = [{col: "your text here" for col in required}]
    elif model_type == "time_series":
        required = []
        sample = []
    elif raw_columns:
        required = raw_columns
        sample = [_demo_sample_row(raw_columns, use_case)]
    else:
        required = artifact.get("feature_columns", []) if isinstance(artifact, dict) else []
        if not required and model is not None and hasattr(model, "feature_names_in_"):
            required = list(model.feature_names_in_)
        sample = [{col: 0 for col in required}] if required else [{}]

    payload = {
        "model_type": model_type,
        "problem_type": artifact.get("problem_type") if isinstance(artifact, dict) else None,
        "required_columns": required,
        "raw_input_columns": raw_columns,
        "accepts_raw_columns": accepts_raw,
        "sample_payload": sample,
        "use_case_id": artifact.get("use_case_id") if isinstance(artifact, dict) else None,
    }
    if use_case:
        payload["use_case"] = {
            "name": use_case["name"],
            "description": use_case["description"],
            "business_goal": use_case["business_goal"],
            "positive_class": use_case.get("positive_class"),
        }
    return payload


def _demo_sample_row(columns: list[str], use_case: dict | None) -> dict:
    if use_case:
        row = use_cases.get_demo_sample_row(use_case.get("id", ""))
        if row:
            return row
    return {col: 0 for col in columns}


@app.post("/predict")
async def predict(request: PredictRequest):
    return _run_production_inference(request.records)


@app.post("/reports/export")
async def export_report(request: PredictRequest):
    """Run inference and download results as a CSV report for business teams."""
    result = _run_production_inference(request.records)
    artifact = _load_production_artifact()
    use_case = None
    if isinstance(artifact, dict) and artifact.get("use_case_id"):
        use_case = use_cases.get_use_case(artifact["use_case_id"])

    csv_content = reports.build_report_csv(request.records, result, use_case)
    filename = reports.report_filename(use_case)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/health")
async def health():
    production_ready = os.path.exists(os.path.join(PRODUCTION_DIR, "best_model.pkl"))
    staged_ready = os.path.exists(os.path.join(STAGING_DIR, "best_model.pkl"))
    pytorch = deep_learning.get_device_info() if deep_learning else {"available": False}
    return {
        "status": "ok",
        "service": "ai-data-scientist",
        "production_model_ready": production_ready,
        "staged_model_ready": staged_ready,
        "llm_ready": _anthropic_api_key() is not None,
        "use_cases_available": len(use_cases.list_use_cases()),
        "analysis_saved": _latest_analysis is not None,
        "pytorch": pytorch,
    }


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def serve_app():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
