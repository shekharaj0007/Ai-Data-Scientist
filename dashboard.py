"""
AI Data Scientist — professional dashboard.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import requests
import streamlit as st

from ui.components import (
    empty_state,
    leaderboard_chart,
    leaderboard_radar,
    narrative_block,
    panel,
    profile_summary,
    profile_table,
    render_header,
    render_pipeline_steps,
    render_sidebar,
    shap_chart,
)
from ui.theme import inject_styles

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Data Scientist",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
render_header()
page = render_sidebar(API_URL)


def _post(path: str, **kwargs):
    resp = requests.post(f"{API_URL}{path}", timeout=600, **kwargs)
    resp.raise_for_status()
    return resp.json()


def _get(path: str):
    resp = requests.get(f"{API_URL}{path}", timeout=120)
    resp.raise_for_status()
    return resp.json()


if page == "overview":
    render_pipeline_steps("upload")
    panel(
        "How it works",
        "This platform runs a full ML pipeline on your CSV: profiling, cleaning, feature "
        "engineering, model comparison, SHAP explainability, and optional LLM summaries. "
        "Models are staged for review before production deployment.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        panel("1 · Data Studio", "Upload a dataset and preview schema quality before training.")
    with c2:
        panel("2 · Model Lab", "Compare algorithms, inspect leakage warnings, and read SHAP insights.")
    with c3:
        panel("3 · Deployment", "Approve the best model, then run batch inference with drift alerts.")

    analysis = st.session_state.get("analysis")
    if analysis:
        st.markdown("#### Latest analysis snapshot")
        cols = st.columns(4)
        cols[0].metric("Problem", analysis["problem_type"].replace("_", " ").title())
        cols[1].metric("Best model", analysis["best_model_name"])
        cols[2].metric("Run ID", (analysis.get("run_id") or "—")[:8])
        cols[3].metric("Status", (st.session_state.get("deploy_status") or analysis["status"]).replace("_", " ").title())
    else:
        empty_state("No runs yet", "Start in Data Studio by uploading a CSV file.")

elif page == "upload":
    render_pipeline_steps("analyze")
    panel(
        "Data Studio",
        "Upload a CSV and specify your target column for supervised learning. "
        "Leave the target blank to run unsupervised clustering.",
    )

    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        uploaded = st.file_uploader("Dataset (CSV)", type=["csv"])
        columns = st.session_state.get("upload_columns", [])

        if uploaded:
            with st.spinner("Profiling dataset…"):
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                upload_resp = _post("/upload", files=files)
                columns = upload_resp["columns"]
                st.session_state["upload_profile"] = upload_resp["profile"]
                st.session_state["upload_columns"] = columns
                st.session_state["upload_filename"] = uploaded.name

        target_options = ["— Unsupervised (clustering) —", *columns]
        target_col = st.selectbox(
            "Target column",
            options=target_options if columns else ["— Unsupervised (clustering) —"],
            help="The column you want to predict.",
        )
        use_llm = st.toggle("Enable LLM insights", value=True, help="Feature ideas and plain-English SHAP summary.")

        run_disabled = uploaded is None
        if st.button("Run full pipeline", type="primary", disabled=run_disabled, use_container_width=True):
            filename = st.session_state.get("upload_filename", uploaded.name)
            target = None if target_col.startswith("—") else target_col
            payload = {"filename": filename, "target_col": target, "use_llm": use_llm}
            with st.spinner("Training models and generating explainability…"):
                analysis = _post("/analyze", json=payload)
            st.session_state["analysis"] = analysis
            st.session_state["analysis"]["filename"] = filename
            st.session_state.pop("deploy_status", None)
            st.success("Pipeline complete. Open Model Lab to review results.")
            st.rerun()

    with col_right:
        profile = st.session_state.get("upload_profile")
        if profile:
            st.markdown("#### Dataset profile")
            profile_summary(profile)
            profile_table(profile)
        else:
            empty_state("Awaiting upload", "Drop a CSV on the left to see row counts, missing values, and column types.")

elif page == "review":
    render_pipeline_steps("review")
    analysis = st.session_state.get("analysis")

    if not analysis:
        empty_state("No analysis available", "Run the pipeline from Data Studio first.")
    else:
        panel(
            "Model Lab",
            "Review leaderboard scores, leakage warnings, and explainability before approving deployment.",
        )

        cols = st.columns(4)
        cols[0].metric("Problem type", analysis["problem_type"].replace("_", " ").title())
        cols[1].metric("Best model", analysis["best_model_name"])
        cols[2].metric("Status", analysis["status"].replace("_", " ").title())
        cols[3].metric("Dataset", analysis.get("filename", "—"))

        if analysis.get("leakage_warnings"):
            st.warning("Possible data leakage detected: " + ", ".join(analysis["leakage_warnings"]))

        tab_scores, tab_explain, tab_suggest = st.tabs(["Leaderboard", "Explainability", "Suggestions"])

        with tab_scores:
            chart_cols = st.columns([1.2, 1])
            with chart_cols[0]:
                leaderboard_chart(analysis["leaderboard"])
            with chart_cols[1]:
                leaderboard_radar(analysis["leaderboard"])
            st.dataframe(pd.DataFrame(analysis["leaderboard"]), use_container_width=True, hide_index=True)

        with tab_explain:
            if analysis.get("feature_importance"):
                shap_chart(analysis["feature_importance"])
            if analysis.get("narrative"):
                st.markdown("#### Executive summary")
                narrative_block(analysis["narrative"])
            elif not analysis.get("feature_importance"):
                empty_state(
                    "Explainability not available",
                    "SHAP summaries are generated for classification and regression tasks when LLM mode is enabled.",
                )

        with tab_suggest:
            suggestions = analysis.get("feature_suggestions") or []
            if suggestions:
                for item in suggestions:
                    st.markdown(f"- {item}")
            else:
                empty_state("No suggestions", "Enable LLM insights during analysis to get domain-specific feature ideas.")

elif page == "deploy":
    render_pipeline_steps("deploy")
    analysis = st.session_state.get("analysis")

    if not analysis:
        empty_state("Nothing to deploy", "Complete an analysis run before promoting a model to production.")
    else:
        panel(
            "Deployment gate",
            "Production models require explicit approval. The staged artifact stays isolated until you confirm.",
        )

        c1, c2 = st.columns(2)
        c1.metric("Candidate model", analysis["best_model_name"])
        c2.metric("Review status", analysis["status"].replace("_", " ").title())

        st.markdown("#### Pre-flight checklist")
        checks = [
            ("Leaderboard reviewed", True),
            ("Leakage warnings acknowledged", not analysis.get("leakage_warnings")),
            ("Explainability reviewed", bool(analysis.get("feature_importance") or analysis.get("narrative"))),
        ]
        for label, ok in checks:
            st.markdown(f"- {'✅' if ok else '⚠️'} {label}")

        confirm = st.checkbox("I confirm this model is ready for production use.")
        if st.button("Approve and deploy", type="primary", disabled=not confirm):
            deploy = _post("/approve", json={"confirmed": True})
            st.session_state["deploy_status"] = "deployed"
            st.success("Model deployed successfully.")
            st.caption(deploy.get("model_path", ""))

        if st.session_state.get("deploy_status") == "deployed":
            try:
                model_bytes = requests.get(f"{API_URL}/model/download", timeout=30).content
                st.download_button(
                    "Download production artifact",
                    data=model_bytes,
                    file_name="best_model.pkl",
                    mime="application/octet-stream",
                )
            except Exception:
                pass

elif page == "predict":
    render_pipeline_steps("predict")
    panel(
        "Inference",
        "Send batch records to the production model. Responses include predictions and simple drift warnings.",
    )

    if st.session_state.get("deploy_status") != "deployed":
        st.info("Deploy a model first to enable live inference.")

    default_payload = '[{"age": 35, "tenure": 24, "support_tickets": 1}]'
    sample_json = st.text_area("Request payload (JSON array of records)", value=default_payload, height=140)

    if st.button("Run prediction", type="primary"):
        try:
            records = json.loads(sample_json)
            if not isinstance(records, list):
                raise ValueError("Payload must be a JSON list of objects.")
            result = _post("/predict", json={"records": records})
        except json.JSONDecodeError:
            st.error("Invalid JSON. Please check your payload format.")
        except requests.HTTPError as exc:
            st.error(f"Prediction failed: {exc.response.text if exc.response else exc}")
        except Exception as exc:
            st.error(str(exc))
        else:
            st.markdown("#### Results")
            result_df = pd.DataFrame({"prediction": result["predictions"]})
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            if result.get("drift_warnings"):
                st.warning("Drift warnings:\n\n" + "\n".join(f"- {w}" for w in result["drift_warnings"]))
            else:
                st.success("No drift detected against the training baseline.")
