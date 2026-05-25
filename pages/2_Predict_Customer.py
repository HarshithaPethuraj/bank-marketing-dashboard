"""
Predict Customer tab — Enter a customer profile and get the subscription probability +
a Call / Skip recommendation based on the tuned model's threshold.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Predict Customer", page_icon="🔮", layout="wide")


# ──────────────────────────────────────────────────────────
# LOAD MODEL + DATA (cached so they only load once per session)
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifact():
    return joblib.load(Path("models/bank_marketing_pipeline.pkl"))


@st.cache_data
def load_data():
    df = pd.read_csv(Path("data/bank-additional-full.csv"), sep=";")
    return df


artifact = load_artifact()
df = load_data()

# Use shap_pipeline if available (tuned LightGBM, cleaner predictions); else fall back to main pipeline
pipeline = artifact.get("shap_pipeline", artifact["pipeline"])
threshold = artifact["threshold"]


# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
st.title("Predict Customer")
st.caption("Enter a customer profile to get the subscription probability and call/skip recommendation")

st.divider()

# ──────────────────────────────────────────────────────────
# QUICK PROFILES (one-click prefill examples)
# ──────────────────────────────────────────────────────────
st.markdown("### Quick Profiles")
st.caption("Click a profile to auto-fill the form with example values")

profiles = {
    " Ideal Customer (high probability)": {
        "age": 70, "job": "retired", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no",
        "contact": "cellular", "month": "mar", "day_of_week": "tue",
        "campaign": 1, "pdays": 6, "previous": 2, "poutcome": "success",
        "emp_var_rate": -1.8, "cons_price_idx": 92.893, "cons_conf_idx": -46.2,
        "euribor3m": 1.299, "nr_employed": 5099.1,
    },
    " Average Customer": {
        "age": 40, "job": "admin.", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no",
        "contact": "cellular", "month": "may", "day_of_week": "thu",
        "campaign": 2, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
        "emp_var_rate": 1.1, "cons_price_idx": 93.994, "cons_conf_idx": -36.4,
        "euribor3m": 4.857, "nr_employed": 5191.0,
    },
    "❌ Unlikely Customer (low probability)": {
        "age": 35, "job": "blue-collar", "marital": "married", "education": "basic.9y",
        "default": "unknown", "housing": "no", "loan": "yes",
        "contact": "telephone", "month": "may", "day_of_week": "mon",
        "campaign": 5, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
        "emp_var_rate": 1.4, "cons_price_idx": 93.918, "cons_conf_idx": -42.7,
        "euribor3m": 4.962, "nr_employed": 5228.1,
    },
}

profile_cols = st.columns(3)
for i, (name, _) in enumerate(profiles.items()):
    if profile_cols[i].button(name, use_container_width=True):
        st.session_state.profile = profiles[name]

# Default values (used if no profile clicked yet)
if "profile" not in st.session_state:
    st.session_state.profile = profiles["Average Customer"]

p = st.session_state.profile

st.divider()

# ──────────────────────────────────────────────────────────
# INPUT FORM
# ──────────────────────────────────────────────────────────
st.markdown("### Customer Details")

with st.form("predict_form"):
    # Demographics
    st.markdown("#### 👤 Demographics")
    c1, c2, c3, c4 = st.columns(4)
    age = c1.slider("Age", 17, 98, p["age"])
    job = c2.selectbox("Job", sorted(df["job"].unique()), index=sorted(df["job"].unique()).index(p["job"]))
    marital = c3.selectbox("Marital status", sorted(df["marital"].unique()),
                            index=sorted(df["marital"].unique()).index(p["marital"]))
    education = c4.selectbox("Education", sorted(df["education"].unique()),
                              index=sorted(df["education"].unique()).index(p["education"]))

    # Financial situation
    st.markdown("#### Financial")
    f1, f2, f3 = st.columns(3)
    default = f1.selectbox("Has credit in default?", ["no", "yes", "unknown"],
                            index=["no", "yes", "unknown"].index(p["default"]))
    housing = f2.selectbox("Has housing loan?", ["no", "yes", "unknown"],
                            index=["no", "yes", "unknown"].index(p["housing"]))
    loan = f3.selectbox("Has personal loan?", ["no", "yes", "unknown"],
                         index=["no", "yes", "unknown"].index(p["loan"]))

    # Contact information
    st.markdown("####  Contact")
    co1, co2, co3 = st.columns(3)
    contact = co1.selectbox("Contact method", ["cellular", "telephone"],
                             index=["cellular", "telephone"].index(p["contact"]))
    month = co2.selectbox("Last contact month",
                          ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                          index=["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"].index(p["month"]))
    day_of_week = co3.selectbox("Day of week", ["mon", "tue", "wed", "thu", "fri"],
                                 index=["mon", "tue", "wed", "thu", "fri"].index(p["day_of_week"]))

    # Campaign history
    st.markdown("#### Campaign History")
    h1, h2, h3, h4 = st.columns(4)
    campaign = h1.slider("Contacts this campaign", 1, 50, p["campaign"])
    pdays_raw = h2.slider("Days since last contact (999 = never)", 0, 999, p["pdays"])
    previous = h3.slider("Previous contacts", 0, 10, p["previous"])
    poutcome = h4.selectbox("Previous outcome", ["nonexistent", "failure", "success"],
                             index=["nonexistent", "failure", "success"].index(p["poutcome"]))

    # Macro indicators
    st.markdown("####  Macro Indicators (at time of contact)")
    st.caption("These reflect the economic context — the model uses them heavily")
    m1, m2, m3, m4, m5 = st.columns(5)
    emp_var_rate = m1.number_input("Employment variation rate", -3.5, 1.5, float(p["emp_var_rate"]), step=0.1)
    cons_price_idx = m2.number_input("Consumer price index", 92.0, 95.0, float(p["cons_price_idx"]), step=0.001, format="%.3f")
    cons_conf_idx = m3.number_input("Consumer confidence", -55.0, -25.0, float(p["cons_conf_idx"]), step=0.1)
    euribor3m = m4.number_input("Euribor 3-month rate", 0.5, 5.5, float(p["euribor3m"]), step=0.01, format="%.3f")
    nr_employed = m5.number_input("Number of employees (k)", 4960.0, 5230.0, float(p["nr_employed"]), step=0.1, format="%.1f")

    submitted = st.form_submit_button(" Predict Subscription Probability", use_container_width=True, type="primary")


# ──────────────────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────────────────
if submitted:
    # Mirror the feature engineering done in the notebook
    was_contacted_before = 0 if pdays_raw == 999 else 1
    pdays = 0 if pdays_raw == 999 else pdays_raw

    input_dict = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        "was_contacted_before": was_contacted_before,
    }
    input_df = pd.DataFrame([input_dict])

    # Ensure column order matches training
    feature_cols = artifact.get("feature_columns")
    if feature_cols:
        # Add any missing columns with default values (shouldn't happen but safe)
        for c in feature_cols:
            if c not in input_df.columns:
                input_df[c] = 0
        input_df = input_df[feature_cols]

    # Save input to session state for the SHAP tab
    st.session_state["last_input"] = input_df.copy()

    try:
        proba = pipeline.predict_proba(input_df)[0, 1]
        prediction = int(proba >= threshold)

        # ──────────────────────────────────────────────
        # RESULT DISPLAY
        # ──────────────────────────────────────────────
        st.divider()
        st.markdown("##  Prediction Result")

        # Three-column result layout
        r1, r2, r3 = st.columns([1, 1, 2])

        r1.metric("Subscription probability", f"{proba * 100:.1f}%")
        r2.metric("Decision threshold", f"{threshold * 100:.1f}%", "F1-optimal")

        with r3:
            if proba >= threshold:
                lift = (proba * 100) / 11.27
                st.success(f"### RECOMMENDATION: **CALL THIS CUSTOMER**")
                st.markdown(f"**Confidence:** {proba * 100:.1f}% probability — about **{lift:.1f}× the base rate** of 11.27%")
            else:
                gap = (threshold - proba) * 100
                st.error(f"### ⏭ RECOMMENDATION: **SKIP THIS CUSTOMER**")
                st.markdown(f"Probability is {gap:.1f} pp below the threshold. Allocate this slot to a higher-scoring lead.")

        # Visual probability gauge
        st.markdown("##### Probability scale")
        gauge_col1, gauge_col2 = st.columns([5, 1])
        with gauge_col1:
            st.progress(min(proba, 1.0), text=f"{proba * 100:.1f}%")
        with gauge_col2:
            st.caption(f"Threshold: {threshold * 100:.1f}%")

        # Decile estimate based on probability
        st.divider()
        st.markdown("### Where does this customer land?")
        if proba >= 0.7:
            decile_text = "**Top 10%** — premium lead, highest priority"
            decile_color = "success"
        elif proba >= 0.5:
            decile_text = "**Top 20%** — strong lead, in the recommended call zone"
            decile_color = "success"
        elif proba >= 0.35:
            decile_text = "**Top 30–40%** — borderline; call if capacity allows"
            decile_color = "warning"
        elif proba >= 0.2:
            decile_text = "**Middle pack (40–60%)** — weak signal; deprioritize"
            decile_color = "warning"
        else:
            decile_text = "**Bottom 40%** — below random baseline; skip"
            decile_color = "error"

        getattr(st, decile_color)(decile_text)

        # Quick interpretation hints
        st.markdown("### 🧠 Interpretation hints")
        hints = []
        if poutcome == "success":
            hints.append("Strong positive: previous campaign succeeded — this is the **#1 single predictor** in SHAP")
        if contact == "cellular":
            hints.append("Positive: cellular contacts convert ~3× more than telephone")
        else:
            hints.append("Negative: telephone (landline) converts ~3× less than cellular")
        if month in ["mar", "sep", "oct", "dec"]:
            hints.append(f"Positive: **{month}** is a golden month (40–50% conversion historically)")
        if month in ["may", "jun", "jul", "aug"]:
            hints.append(f"Negative: **{month}** is a weak month (~6–10% conversion historically)")
        if euribor3m < 1.5:
            hints.append("Positive: low Euribor — customers seek safer investments during low-rate periods")
        elif euribor3m > 4.0:
            hints.append("Negative: high Euribor — customers tend to wait for better deals")
        if campaign > 3:
            hints.append(f"Negative: already contacted {campaign} times — diminishing returns kick in after 3 attempts")
        if age >= 65 or age < 25:
            hints.append(f"Positive: age {age} is in the high-conversion U-curve extreme")

        for hint in hints:
            st.markdown(hint)

        st.divider()
        st.info(
            " **Want to see exactly why the model gave this score?** Open the **🧠 Why This Prediction?** tab "
            "in the sidebar — it shows the SHAP waterfall for the input you just submitted."
        )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.exception(e)