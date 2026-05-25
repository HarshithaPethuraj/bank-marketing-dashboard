"""
Portuguese Bank Marketing — Subscription Prediction Dashboard
Main entry point. Sidebar navigation is auto-built from the pages/ folder.
Mobile-responsive, bold-text, light-mode-locked styling.
"""

import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


# ──────────────────────────────────────────────────────────
# PAGE CONFIG (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Marketing — Subscription Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────
# GLOBAL CSS — corporate polish + mobile-friendly + bold text + forced light mode
# ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Force browser to treat this page as LIGHT ──
       Stops mobile Chrome / iOS Safari from auto-darkening inputs and text */
    :root { color-scheme: light !important; }
    html, body, .stApp { color-scheme: light !important; background: #ffffff !important; }

    /* Base text — bolder + darker for higher contrast on mobile */
    html, body, [class*="css"] {
        color: #0f172a !important;
        font-weight: 500;
    }

    p, span, label, div, li {
        color: #0f172a !important;
    }

    /* Container — responsive padding for mobile */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        max-width: 1400px;
    }

    /* Headings — bold and high-contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    h1 {
        font-size: clamp(1.6rem, 4vw, 2.4rem) !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    h2 {
        font-size: clamp(1.3rem, 3vw, 1.7rem) !important;
        border-bottom: 2px solid #cbd5e1;
        padding-bottom: 0.5rem;
        margin-top: 1.8rem;
    }
    h3 {
        font-size: clamp(1.1rem, 2.5vw, 1.4rem) !important;
        color: #1e40af !important;
        font-weight: 700 !important;
    }

    /* Markdown body text — bolder */
    .stMarkdown p, .stMarkdown li {
        color: #0f172a !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        line-height: 1.6;
    }

    /* Captions — readable but secondary */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #475569 !important;
        font-weight: 500 !important;
    }

    /* Metric cards — bolder and mobile-stacking friendly */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
        transition: transform 0.15s, box-shadow 0.15s;
        margin-bottom: 8px;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.1);
        border-color: #94a3b8;
    }
    div[data-testid="stMetric"] label {
        color: #475569 !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: clamp(1.3rem, 3vw, 1.8rem) !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }

    /* Buttons — larger touch target for mobile + bolder */
    .stButton > button {
        border-radius: 8px;
        font-weight: 700 !important;
        transition: all 0.15s;
        border: 1px solid #cbd5e1;
        padding: 0.5rem 1.25rem;
        min-height: 44px;  /* iOS recommended touch target */
        font-size: 0.95rem;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.18);
        border-color: #1e40af;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
        border: none;
        color: white !important;
        font-weight: 700 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%);
    }

    /* Form inputs — force light + bolder */
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    div[data-testid="stDateInput"] > div > div,
    div[data-testid="stTimeInput"] > div > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
    }
    [data-baseweb="select"] *, [data-baseweb="input"] *,
    [data-baseweb="popover"] *, [role="listbox"] *,
    li[role="option"], li[role="option"] * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background-color: #ffffff !important;
    }
    li[role="option"]:hover { background-color: #f1f5f9 !important; }
    [data-baseweb="select"] svg { fill: #64748b !important; color: #64748b !important; }

    /* Slider — clearer visibility on mobile */
    [data-testid="stSlider"] label {
        font-weight: 600 !important;
        color: #0f172a !important;
    }

    /* Number input */
    [data-testid="stNumberInput"] label {
        font-weight: 600 !important;
        color: #0f172a !important;
    }
    [data-testid="stNumberInput"] input {
        background: #ffffff !important;
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Info / success / warning / error boxes — bolder */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border-left-width: 5px;
        padding: 1rem 1.25rem;
        font-weight: 500;
    }
    div[data-testid="stAlert"] p {
        font-weight: 500 !important;
        color: #0f172a !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #cbd5e1;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: #475569 !important;
        font-weight: 500 !important;
    }

    /* Tables — cleaner and bolder */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #cbd5e1;
    }
    .dataframe th {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .dataframe td {
        color: #0f172a !important;
        font-weight: 500 !important;
    }

    /* Markdown tables */
    table {
        border-collapse: collapse !important;
        width: 100% !important;
    }
    table th {
        background: #f1f5f9 !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        padding: 10px !important;
        border: 1px solid #cbd5e1 !important;
    }
    table td {
        color: #0f172a !important;
        font-weight: 500 !important;
        padding: 10px !important;
        border: 1px solid #e2e8f0 !important;
    }

    /* Expander — bolder */
    div[data-testid="stExpander"] {
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        background: #ffffff;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Divider — subtler */
    hr {
        border-color: #cbd5e1;
        margin: 1.5rem 0;
    }

    /* Hide Streamlit menu/footer for cleaner look */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── MOBILE-SPECIFIC FIXES ── */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            padding-top: 1rem !important;
        }

        /* Stack metric cards on mobile */
        div[data-testid="stMetric"] {
            padding: 14px 16px;
            margin-bottom: 8px;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
        }

        /* Larger tap targets on mobile */
        .stButton > button {
            min-height: 48px;
            font-size: 1rem;
            padding: 0.75rem 1rem;
        }

        /* Slim down heading sizes */
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }

        /* Markdown tables — make them scroll horizontally on mobile */
        table {
            display: block;
            overflow-x: auto;
            white-space: nowrap;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────
# DATA + MODEL LOADERS
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifact():
    path = Path("models/bank_marketing_pipeline.pkl")
    if not path.exists():
        st.error(f"❌ Model file not found at {path}.")
        st.stop()
    return joblib.load(path)


@st.cache_data
def load_data():
    path = Path("data/bank-additional-full.csv")
    if not path.exists():
        st.error(f"❌ Data file not found at {path}.")
        st.stop()
    return pd.read_csv(path, sep=";")


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Bank Marketing")
    st.caption("Subscription Prediction · v1.0")
    st.divider()

    st.markdown(
        """
        <div style='padding: 14px; background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
                    border-radius: 10px; color: white !important; margin-bottom: 14px;'>
            <div style='font-size: 0.75rem; font-weight: 700; opacity: 0.95;
                        letter-spacing: 0.06em; color: white !important;'>FINAL MODEL</div>
            <div style='font-size: 1.35rem; font-weight: 800; margin: 4px 0; color: white !important;'>
                LightGBM Tuned
            </div>
            <div style='font-size: 0.8rem; font-weight: 500; opacity: 0.95;
                        line-height: 1.5; color: white !important;'>
                ROC-AUC: <b>0.8140</b><br>
                Top-20% capture: <b>66.4%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📍 Pages")
    st.markdown(
        """
        **1.** Campaign Overview

        **2.** Predict Customer

        **3.** Why This Prediction?

        **4.** Model Insights

        **5.** Recommendations
        """
    )

    st.divider()
    st.caption("Built with Streamlit · LightGBM · SHAP")


# ──────────────────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────────────────
banner_path = Path("assets/banner.png")
if banner_path.exists():
    st.image(str(banner_path), use_container_width=True)

st.title("Portuguese Bank Marketing — Subscription Prediction")
st.markdown(
    "**Predicting which customers will subscribe to a term deposit, explaining *why*, "
    "and recommending data-driven marketing actions.**"
)

st.divider()

# ── Smoke test: confirm artifact + data load ──
try:
    artifact = load_artifact()
    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "LightGBM (Tuned)")
    col2.metric("Test ROC-AUC", f"{artifact['metrics']['roc_auc']:.4f}")
    col3.metric("F1 Score", f"{artifact['metrics']['f1']:.4f}")
    col4.metric("Dataset Rows", f"{len(df):,}")

    st.success("**Model and data loaded successfully — dashboard ready.**")

    with st.expander("🔧 Artifact contents (debug view)"):
        st.write({
            "threshold": artifact["threshold"],
            "metrics": artifact["metrics"],
        })

except Exception as e:
    st.error(f"❌ Error loading artifact or data: {e}")

st.divider()

# ──────────────────────────────────────────────────────────
# DASHBOARD SECTIONS OVERVIEW
# ──────────────────────────────────────────────────────────
st.markdown("### 📍 Dashboard Sections")

st.markdown(
    """
| Section | What it does |
|---|---|
| **Campaign Overview** | Slice-and-dice conversion analytics by demographic, channel, and timing |
| **Predict Customer** | Enter a customer profile → get subscription probability + call/skip recommendation |
| **Why This Prediction?** | Per-customer SHAP explanation — see exactly which features drove the score |
| **Model Insights** | Global SHAP, lift chart, threshold slider for capacity planning |
| **Recommendations** | Data-backed action items for the marketing team |
"""
)

st.divider()

# ──────────────────────────────────────────────────────────
# PROJECT HIGHLIGHTS
# ──────────────────────────────────────────────────────────
st.markdown("### Project Highlights")

h1, h2, h3 = st.columns(3)

with h1:
    st.markdown(
        """
        <div style='padding: 20px; background: #f1f5f9; border-radius: 12px;
                    border-left: 5px solid #1e40af; height: 100%;'>
            <div style='font-size: 2rem;'>🎯</div>
            <div style='font-weight: 800; font-size: 1.1rem; margin: 8px 0; color: #0f172a;'>
                Top-20% Targeting
            </div>
            <div style='color: #1e293b; font-size: 0.95rem; font-weight: 500; line-height: 1.6;'>
                Calling the top 20% of leads by model score captures
                <b>66.4%</b> of all subscribers — at one-fifth the cost.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with h2:
    st.markdown(
        """
        <div style='padding: 20px; background: #f1f5f9; border-radius: 12px;
                    border-left: 5px solid #10b981; height: 100%;'>
            <div style='font-size: 2rem;'>🧠</div>
            <div style='font-weight: 800; font-size: 1.1rem; margin: 8px 0; color: #0f172a;'>
                Explainable AI
            </div>
            <div style='color: #1e293b; font-size: 0.95rem; font-weight: 500; line-height: 1.6;'>
                Every prediction comes with a SHAP waterfall — defensible
                under audit, useful for agent talking points.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with h3:
    st.markdown(
        """
        <div style='padding: 20px; background: #f1f5f9; border-radius: 12px;
                    border-left: 5px solid #f59e0b; height: 100%;'>
            <div style='font-size: 2rem;'>📈</div>
            <div style='font-weight: 800; font-size: 1.1rem; margin: 8px 0; color: #0f172a;'>
                Production Ready
            </div>
            <div style='color: #1e293b; font-size: 0.95rem; font-weight: 500; line-height: 1.6;'>
                ROC-AUC <b>0.8140</b> (matches the published research benchmark)
                after 100 Optuna trials. No data leakage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.caption(
    " **Dataset:** UCI Bank Marketing (Portuguese banking institution, May 2008 – Nov 2010) · "
    "41,188 customer contacts · 20 features · Binary target (subscribed term deposit?)"
)