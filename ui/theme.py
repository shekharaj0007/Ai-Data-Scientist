"""Brand palette and global CSS for the dashboard."""

COLORS = {
    "bg": "#0F172A",
    "surface": "#1E293B",
    "surface_alt": "#334155",
    "border": "#475569",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "accent": "#6366F1",
    "accent_soft": "#818CF8",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#38BDF8",
}

CHART_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": COLORS["text"], "family": "Inter, system-ui, sans-serif"},
    "margin": {"l": 20, "r": 20, "t": 40, "b": 20},
}


def inject_styles() -> None:
    import streamlit as st

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, sans-serif;
        }}

        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }}

        header[data-testid="stHeader"] {{
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(12px);
        }}

        div[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #111827 0%, #0F172A 100%);
            border-right: 1px solid {COLORS["border"]};
        }}

        div[data-testid="stMetric"] {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 14px;
            padding: 0.85rem 1rem;
        }}

        div[data-testid="stMetric"] label {{
            color: {COLORS["muted"]} !important;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            color: {COLORS["text"]};
            font-weight: 700;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: transparent;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: {COLORS["surface"]};
            border-radius: 10px;
            border: 1px solid {COLORS["border"]};
            padding: 0.5rem 1rem;
            color: {COLORS["muted"]};
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {COLORS["accent"]}, {COLORS["accent_soft"]}) !important;
            color: white !important;
            border-color: transparent !important;
        }}

        .hero-title {{
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin: 0;
            color: {COLORS["text"]};
        }}

        .hero-subtitle {{
            color: {COLORS["muted"]};
            font-size: 1rem;
            margin-top: 0.35rem;
            line-height: 1.6;
        }}

        .brand-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.35);
            color: {COLORS["accent_soft"]};
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}

        .panel-card {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 16px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
        }}

        .panel-title {{
            font-size: 0.95rem;
            font-weight: 600;
            color: {COLORS["text"]};
            margin-bottom: 0.35rem;
        }}

        .panel-copy {{
            color: {COLORS["muted"]};
            font-size: 0.9rem;
            line-height: 1.55;
            margin: 0;
        }}

        .step-row {{
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin: 1rem 0 1.5rem 0;
        }}

        .step-chip {{
            flex: 1;
            min-width: 140px;
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
            padding: 0.85rem 1rem;
        }}

        .step-chip.active {{
            border-color: {COLORS["accent"]};
            box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.35);
        }}

        .step-chip.done {{
            border-color: rgba(16, 185, 129, 0.45);
        }}

        .step-number {{
            color: {COLORS["accent_soft"]};
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .step-label {{
            color: {COLORS["text"]};
            font-weight: 600;
            margin-top: 0.25rem;
        }}

        .status-pill {{
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .status-pending {{ background: rgba(245, 158, 11, 0.15); color: {COLORS["warning"]}; }}
        .status-deployed {{ background: rgba(16, 185, 129, 0.15); color: {COLORS["success"]}; }}
        .status-idle {{ background: rgba(148, 163, 184, 0.15); color: {COLORS["muted"]}; }}

        .narrative-box {{
            background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(56,189,248,0.08));
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            color: {COLORS["text"]};
            line-height: 1.7;
        }}

        .footer-note {{
            color: {COLORS["muted"]};
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid {COLORS["border"]};
        }}

        div[data-testid="stFileUploader"] section {{
            background: {COLORS["surface"]};
            border: 1px dashed {COLORS["border"]};
            border-radius: 14px;
            padding: 0.5rem;
        }}

        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {COLORS["accent"]}, {COLORS["accent_soft"]});
            border: none;
            border-radius: 10px;
            font-weight: 600;
        }}

        .stButton > button[kind="secondary"] {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            color: {COLORS["text"]};
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
