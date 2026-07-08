"""Reusable dashboard layout components."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.theme import CHART_LAYOUT, COLORS


def render_header() -> None:
    st.markdown(
        """
        <div class="brand-badge">Automated ML Platform</div>
        <h1 class="hero-title">AI Data Scientist</h1>
        <p class="hero-subtitle">
            Upload raw data, train and compare models automatically, review explainability,
            and deploy with a human-in-the-loop approval step.
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_steps(current: str) -> None:
    steps = [
        ("upload", "01", "Upload"),
        ("analyze", "02", "Analyze"),
        ("review", "03", "Review"),
        ("deploy", "04", "Deploy"),
        ("predict", "05", "Predict"),
    ]
    order = [s[0] for s in steps]
    current_idx = order.index(current) if current in order else 0

    chips = []
    for idx, (key, number, label) in enumerate(steps):
        state = "active" if key == current else ("done" if idx < current_idx else "")
        chips.append(
            f'<div class="step-chip {state}">'
            f'<div class="step-number">{number}</div>'
            f'<div class="step-label">{label}</div>'
            f"</div>"
        )
    st.markdown(f'<div class="step-row">{"".join(chips)}</div>', unsafe_allow_html=True)


def render_sidebar(api_url: str) -> str:
    with st.sidebar:
        st.markdown("### Workspace")
        page = st.radio(
            "Navigation",
            ["Overview", "Data Studio", "Model Lab", "Deployment", "Inference"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**System status**")
        try:
            import requests

            health = requests.get(f"{api_url}/health", timeout=5).json()
            st.success(f"API online — {health.get('status', 'ok')}")
        except Exception:
            st.error("API unreachable")
            st.caption(f"Expected at `{api_url}`")

        analysis = st.session_state.get("analysis")
        deploy_status = st.session_state.get("deploy_status")
        if analysis:
            st.markdown("**Latest run**")
            st.caption(analysis.get("filename", "Unknown file"))
            status = deploy_status or analysis.get("status", "idle")
            css = "status-deployed" if status == "deployed" else "status-pending"
            st.markdown(
                f'<span class="status-pill {css}">{status.replace("_", " ").title()}</span>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<p class="footer-note">Deterministic ML core · LLM for narrative only</p>',
            unsafe_allow_html=True,
        )

    page_map = {
        "Overview": "overview",
        "Data Studio": "upload",
        "Model Lab": "review",
        "Deployment": "deploy",
        "Inference": "predict",
    }
    return page_map[page]


def panel(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">{title}</div>
            <p class="panel-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def profile_summary(profile: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Rows", f"{profile.get('n_rows', 0):,}")
    cols[1].metric("Columns", profile.get("n_cols", 0))
    missing_cols = sum(
        1 for c in profile.get("columns", {}).values() if c.get("missing_pct", 0) > 0
    )
    cols[2].metric("Cols w/ missing", missing_cols)
    high_card = sum(
        1 for c in profile.get("columns", {}).values() if c.get("unique_count", 0) > 50
    )
    cols[3].metric("High cardinality", high_card)


def profile_table(profile: dict) -> None:
    rows = []
    for name, stats in profile.get("columns", {}).items():
        rows.append(
            {
                "Column": name,
                "Type": stats.get("dtype", ""),
                "Missing %": stats.get("missing_pct", 0),
                "Unique": stats.get("unique_count", 0),
            }
        )
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )


def leaderboard_chart(leaderboard: list[dict]) -> None:
    df = pd.DataFrame(leaderboard)
    if df.empty:
        return

    score_col = "test_score" if "test_score" in df.columns else df.columns[-1]
    fig = px.bar(
        df.sort_values(score_col, ascending=True),
        x=score_col,
        y="model" if "model" in df.columns else df.columns[0],
        orientation="h",
        color=score_col,
        color_continuous_scale=["#475569", "#6366F1", "#818CF8"],
        title="Cross-validated model comparison",
    )
    fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def shap_chart(feature_importance: list[dict]) -> None:
    imp = pd.DataFrame(feature_importance)
    fig = px.bar(
        imp.sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="mean_abs_shap",
        color_continuous_scale=["#334155", "#38BDF8", "#6366F1"],
        title="Top drivers (mean |SHAP|)",
    )
    fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def leaderboard_radar(leaderboard: list[dict]) -> None:
    df = pd.DataFrame(leaderboard)
    if "cv_mean_score" not in df.columns or "test_score" not in df.columns:
        return

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row["cv_mean_score"], row["test_score"], row.get("cv_std", 0)],
                theta=["CV mean", "Test score", "CV std"],
                fill="toself",
                name=str(row.get("model", "model")),
                opacity=0.65,
            )
        )
    fig.update_layout(
        **CHART_LAYOUT,
        polar=dict(radialaxis=dict(visible=True, gridcolor=COLORS["border"])),
        showlegend=True,
        title="Model score profile",
    )
    st.plotly_chart(fig, use_container_width=True)


def narrative_block(text: str) -> None:
    st.markdown(f'<div class="narrative-box">{text}</div>', unsafe_allow_html=True)


def empty_state(title: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">{title}</div>
            <p class="panel-copy">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
