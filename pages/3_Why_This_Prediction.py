"""
Why This Prediction? tab — Per-customer SHAP explanation.

Shows the SHAP waterfall plot for the most recently submitted customer
from the Predict Customer tab. Lets the marketing agent see exactly which
features drove the model's score for THIS specific customer.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from pathlib import Path

st.set_page_config(page_title="Why This Prediction?", page_icon="🧠", layout="wide")


# ──────────────────────────────────────────────────────────
# LOAD ARTIFACT
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifact():
    return joblib.load(Path("models/bank_marketing_pipeline.pkl"))


@st.cache_resource
def get_explainer(_pipeline):
    """Build and cache the SHAP TreeExplainer for the LightGBM model."""
    model = _pipeline.named_steps["model"]
    return shap.TreeExplainer(model)


artifact = load_artifact()
pipeline = artifact.get("shap_pipeline", artifact["pipeline"])
threshold = artifact["threshold"]
explainer = get_explainer(pipeline)


# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
st.title("🧠 Why This Prediction?")
st.caption("Per-customer SHAP explanation — see exactly which features drove the model's decision")

st.divider()

# ──────────────────────────────────────────────────────────
# GET INPUT — either from session state (Predict tab) or from a default
# ──────────────────────────────────────────────────────────
if "last_input" not in st.session_state:
    st.warning(
        " **No customer prediction yet.** Go to the **🔮 Predict Customer** tab first, "
        "fill in a customer profile, and click 'Predict'. Then come back here to see the SHAP explanation."
    )
    st.info(
        " **Or click the button below to load a sample customer** so you can see what this tab does without leaving."
    )

    if st.button(" Load 'Ideal Customer' sample", type="primary"):
        sample_input = pd.DataFrame([{
            "age": 70, "job": "retired", "marital": "married", "education": "university.degree",
            "default": "no", "housing": "yes", "loan": "no",
            "contact": "cellular", "month": "mar", "day_of_week": "tue",
            "campaign": 1, "pdays": 6, "previous": 2, "poutcome": "success",
            "emp.var.rate": -1.8, "cons.price.idx": 92.893, "cons.conf.idx": -46.2,
            "euribor3m": 1.299, "nr.employed": 5099.1, "was_contacted_before": 1,
        }])
        # Match column order to training
        feature_cols = artifact.get("feature_columns")
        if feature_cols:
            for c in feature_cols:
                if c not in sample_input.columns:
                    sample_input[c] = 0
            sample_input = sample_input[feature_cols]
        st.session_state["last_input"] = sample_input
        st.rerun()

    st.stop()

input_df = st.session_state["last_input"]

# ──────────────────────────────────────────────────────────
# PREDICT + EXPLAIN
# ──────────────────────────────────────────────────────────
proba = pipeline.predict_proba(input_df)[0, 1]
prediction = int(proba >= threshold)

# Transform input through the preprocessor
preprocessor = pipeline.named_steps["preprocessor"]
input_transformed = preprocessor.transform(input_df)

# Get feature names after one-hot encoding
ohe = preprocessor.named_transformers_["cat"]
cat_cols = preprocessor.transformers_[1][2]
num_cols = preprocessor.transformers_[0][2]
cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
feature_names = list(num_cols) + cat_feature_names

# Compute SHAP values for this single customer
shap_values_raw = explainer.shap_values(input_transformed)
if isinstance(shap_values_raw, list):
    # Old SHAP API — list per class
    shap_values_pos = shap_values_raw[1][0]
    base_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )
elif shap_values_raw.ndim == 3:
    # Newer SHAP API — 3D array (samples, features, classes)
    shap_values_pos = shap_values_raw[0, :, 1]
    base_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )
else:
    # 2D array (samples, features)
    shap_values_pos = shap_values_raw[0]
    base_value = (
        explainer.expected_value
        if not isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value[0]
    )

# ──────────────────────────────────────────────────────────
# SUMMARY ROW
# ──────────────────────────────────────────────────────────
s1, s2, s3 = st.columns([1, 1, 2])
s1.metric("Predicted probability", f"{proba * 100:.1f}%")
s2.metric("Decision threshold", f"{threshold * 100:.1f}%")

with s3:
    if prediction == 1:
        st.success(f"###  Decision: **CALL** — probability above threshold")
    else:
        st.error(f"### ⏭ Decision: **SKIP** — probability below threshold")

st.divider()

# ──────────────────────────────────────────────────────────
# CUSTOMER PROFILE DISPLAY
# ──────────────────────────────────────────────────────────
with st.expander(" View customer profile being explained", expanded=False):
    display_df = input_df.T.reset_index()
    display_df.columns = ["Feature", "Value"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

# ──────────────────────────────────────────────────────────
# SHAP WATERFALL PLOT
# ──────────────────────────────────────────────────────────
st.header(" SHAP Waterfall — Feature Contributions")
st.caption(
    "Each bar shows how a feature pushed the prediction up (red, toward 'yes') or down (blue, toward 'no'), "
    "starting from the model's base value and landing at the final prediction."
)

# Build the waterfall explanation object
explanation = shap.Explanation(
    values=shap_values_pos,
    base_values=base_value,
    data=input_transformed[0],
    feature_names=feature_names,
)

# Render waterfall using matplotlib
fig, ax = plt.subplots(figsize=(10, 7))
shap.plots.waterfall(explanation, max_display=12, show=False)
plt.tight_layout()
st.pyplot(fig, clear_figure=True)

st.info(
    f" **How to read this:**\n\n"
    f"- The waterfall starts at `E[f(X)]` (the average prediction across all customers) "
    f"and adds/subtracts feature contributions to reach `f(x) = {explanation.values.sum() + base_value:.2f}` (this customer's logit)\n"
    f"- **Red bars push toward 'yes'** (higher probability), **blue bars push toward 'no'**\n"
    f"- Bars are sorted by absolute contribution — biggest drivers on top\n"
    f"- Values shown on each bar are SHAP values in log-odds units"
)

st.divider()

# ──────────────────────────────────────────────────────────
# TOP DRIVERS — TABULAR BREAKDOWN
# ──────────────────────────────────────────────────────────
st.header(" Top Drivers — Talking Points for the Agent")
st.caption("The 8 features that most influenced this prediction, in plain language")

# Build a clean drivers table
drivers_df = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_values_pos,
    "feature_value": input_transformed[0],
}).assign(abs_shap=lambda d: d["shap_value"].abs()).sort_values("abs_shap", ascending=False).head(8)

drivers_df["direction"] = drivers_df["shap_value"].apply(lambda x: "↑ Pushes YES" if x > 0 else "↓ Pushes NO")
drivers_df["magnitude"] = drivers_df["abs_shap"].apply(
    lambda x: " Strong" if x > 0.3 else " Medium" if x > 0.1 else "🟢 Weak"
)

display_cols = drivers_df[["feature", "direction", "magnitude", "shap_value"]].copy()
display_cols["shap_value"] = display_cols["shap_value"].round(3)
display_cols.columns = ["Feature", "Direction", "Magnitude", "SHAP value"]

st.dataframe(display_cols, use_container_width=True, hide_index=True)

# Build a narrative summary
top_positive = drivers_df[drivers_df["shap_value"] > 0].head(3)
top_negative = drivers_df[drivers_df["shap_value"] < 0].head(3)

st.markdown("###  Agent Talking Points")

if len(top_positive) > 0:
    st.success(
        "**Why this customer is likely to subscribe (lead with these):**\n\n"
        + "\n".join(f"- **{row['feature']}** — strong positive signal (SHAP +{row['shap_value']:.2f})"
                    for _, row in top_positive.iterrows())
    )

if len(top_negative) > 0:
    st.error(
        "**Watch out for these (objections / risk factors):**\n\n"
        + "\n".join(f"- **{row['feature']}** — pushes against subscribing (SHAP {row['shap_value']:.2f})"
                    for _, row in top_negative.iterrows())
    )

st.divider()

# ──────────────────────────────────────────────────────────
# WHY THIS MATTERS — FOR COMPLIANCE
# ──────────────────────────────────────────────────────────
with st.expander(" Why does the bank need this? (compliance + regulatory context)"):
    st.markdown("""
    In regulated industries like banking, **"the model said so" is not an acceptable answer.** When a customer is
    targeted or excluded by an algorithmic system, banks must be able to show:

    - **Which features drove the decision** — SHAP provides exact contributions
    - **Whether the model uses protected attributes inappropriately** — visible in feature lists
    - **Whether the decision can be reproduced** — SHAP values are deterministic given the same model + input

    This tab is what makes the model **defensible under audit**. Without per-customer explanations, the model
    is a black box and would fail any regulatory review.

    **Use this tab:**
    - Before high-value calls — give agents personalized talking points
    - During compliance reviews — show the exact reasoning behind any contested decision
    - For customer-facing explanations — turn SHAP into plain language as shown in the talking points above
    """)

st.divider()
st.caption(
    "📌 To explain a different customer, go back to the ** Predict Customer** tab, change inputs, "
    "click Predict, then return here. The most recent submitted profile is always shown."
)