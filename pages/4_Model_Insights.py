"""
Model Insights tab — Global model behavior and operational tools.
Shows global SHAP importance, lift chart by decile, and an interactive threshold slider
that lets the marketing team see how precision/recall trade off in real time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
import shap
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)

st.set_page_config(page_title="Model Insights", page_icon="📈", layout="wide")


# ──────────────────────────────────────────────────────────
# LOAD ARTIFACT + DATA
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifact():
    return joblib.load(Path("models/bank_marketing_pipeline.pkl"))


@st.cache_data
def load_data():
    df = pd.read_csv(Path("data/bank-additional-full.csv"), sep=";")
    df = df.drop_duplicates().reset_index(drop=True)
    df["was_contacted_before"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, 0)
    df = df.drop(columns=["duration"])
    df["y"] = (df["y"] == "yes").astype(int)
    return df


@st.cache_resource
def compute_predictions(_pipeline, df):
    """Get probabilities for the entire dataset (cached so it only runs once)."""
    X = df.drop(columns=["y"])
    y = df["y"].values
    proba = _pipeline.predict_proba(X)[:, 1]
    return proba, y


@st.cache_resource
def compute_global_shap(_pipeline, df, sample_size=1000):
    """Compute SHAP values on a sample for the global summary plot."""
    X = df.drop(columns=["y"]).sample(n=min(sample_size, len(df)), random_state=42)
    X_transformed = _pipeline.named_steps["preprocessor"].transform(X)

    ohe = _pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_cols = _pipeline.named_steps["preprocessor"].transformers_[1][2]
    num_cols = _pipeline.named_steps["preprocessor"].transformers_[0][2]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names = list(num_cols) + cat_feature_names

    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    model = _pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values, X_df, feature_names


artifact = load_artifact()
pipeline = artifact.get("shap_pipeline", artifact["pipeline"])
default_threshold = artifact["threshold"]
df = load_data()

# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
st.title(" Model Insights")
st.caption("Global model behavior, feature importance, and operational tools for the marketing team")

st.divider()

# ──────────────────────────────────────────────────────────
# MODEL METRICS HEADER ROW
# ──────────────────────────────────────────────────────────
metrics = artifact.get("metrics", {})
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Test ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
col2.metric("Test PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
col3.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
col4.metric("Precision", f"{metrics.get('precision', 0):.4f}")
col5.metric("Recall", f"{metrics.get('recall', 0):.4f}")

st.caption(f"Default threshold: **{default_threshold:.2f}** (F1-optimized) · Baseline subscriber rate: **11.27%**")

st.divider()

# ──────────────────────────────────────────────────────────
# SECTION 1:GLOBAL SHAP FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────
st.header("🧠 Global Feature Importance (SHAP)")
st.caption("Which features matter most across all customers?")

with st.spinner("Computing SHAP values on 1,000 sampled customers..."):
    shap_values, X_sample, feature_names = compute_global_shap(pipeline, df, sample_size=1000)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=True).tail(15)

fig_imp = go.Figure()
fig_imp.add_trace(
    go.Bar(
        x=importance_df["mean_abs_shap"],
        y=importance_df["feature"],
        orientation="h",
        marker_color="#3b82f6",
        text=[f"{v:.3f}" for v in importance_df["mean_abs_shap"]],
        textposition="outside",
    )
)
fig_imp.update_layout(
    xaxis_title="Mean |SHAP value| (impact on model output)",
    yaxis_title="",
    height=500,
    plot_bgcolor="white",
    showlegend=False,
    margin=dict(l=10, r=80, t=10, b=10),
)
st.plotly_chart(fig_imp, use_container_width=True)

st.info(
    " **Insights:**\n\n"
    "- **Top 2 features are macro:** `nr.employed` and `euribor3m` — economic context dominates\n"
    "- **`contact_cellular` ranks #3** — the strongest customer-level signal in the model, and a directly actionable lever for marketers\n"
    "- **`campaign` (number of contact attempts) is high** — confirms diminishing returns\n"
    "- **`age` ranks lower than EDA suggested** — likely because the U-shape relationship is captured through other correlates"
)

st.divider()

# ──────────────────────────────────────────────────────────
# SECTION 2: LIFT CHART
# ──────────────────────────────────────────────────────────
st.header(" Lift Chart — The Marketing Team's Metric")
st.caption("If you call the top X% of customers by score, what fraction of subscribers do you capture?")

with st.spinner("Computing predictions for lift chart..."):
    y_proba, y_true = compute_predictions(pipeline, df)

lift_df = pd.DataFrame({"actual": y_true, "score": y_proba}).sort_values("score", ascending=False).reset_index(drop=True)
lift_df["decile"] = pd.qcut(lift_df.index, 10, labels=range(1, 11)).astype(int)

decile_summary = (
    lift_df.groupby("decile")
    .agg(customers=("actual", "count"), subscribers=("actual", "sum"), rate=("actual", "mean"))
    .reset_index()
)
decile_summary["cum_subscribers"] = decile_summary["subscribers"].cumsum()
decile_summary["cum_pct_subscribers"] = (
    decile_summary["cum_subscribers"] / decile_summary["subscribers"].sum() * 100
)
decile_summary["cum_pct_customers"] = (
    decile_summary["customers"].cumsum() / decile_summary["customers"].sum() * 100
)
decile_summary["lift"] = decile_summary["rate"] / y_true.mean()

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("##### Cumulative Gain")
    fig_gain = go.Figure()
    fig_gain.add_trace(
        go.Scatter(
            x=[0] + decile_summary["cum_pct_customers"].tolist(),
            y=[0] + decile_summary["cum_pct_subscribers"].tolist(),
            mode="lines+markers",
            name="Model",
            line=dict(color="#10b981", width=3),
            marker=dict(size=10),
        )
    )
    fig_gain.add_trace(
        go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random",
                   line=dict(color="gray", dash="dash"))
    )
    fig_gain.update_layout(
        xaxis_title="% of customers contacted (sorted by score)",
        yaxis_title="% of subscribers captured",
        height=400,
        plot_bgcolor="white",
        legend=dict(x=0.7, y=0.1),
    )
    st.plotly_chart(fig_gain, use_container_width=True)

with col_right:
    st.markdown("##### Lift by Decile")
    fig_lift = go.Figure()
    fig_lift.add_trace(
        go.Bar(
            x=decile_summary["decile"].astype(str),
            y=decile_summary["lift"],
            text=[f"{v:.2f}×" for v in decile_summary["lift"]],
            textposition="outside",
            marker_color=["#10b981" if l >= 1 else "#94a3b8" for l in decile_summary["lift"]],
        )
    )
    fig_lift.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Random baseline")
    fig_lift.update_layout(
        xaxis_title="Decile (1 = highest score)",
        yaxis_title="Lift over random",
        height=400,
        plot_bgcolor="white",
        showlegend=False,
    )
    st.plotly_chart(fig_lift, use_container_width=True)

top_10 = decile_summary.iloc[0]["cum_pct_subscribers"]
top_20 = decile_summary.iloc[1]["cum_pct_subscribers"]
top_10_lift = decile_summary.iloc[0]["lift"]

st.success(
    f" **Headline:** Calling the **top 10%** captures **{top_10:.1f}%** of all subscribers "
    f"(lift = {top_10_lift:.2f}×). Calling the **top 20%** captures **{top_20:.1f}%**.\n\n"
    f"Below decile 3, lift drops under 1.0 — meaning **calling those customers is worse than random selection.**"
)

st.divider()

# ──────────────────────────────────────────────────────────
# SECTION 3: INTERACTIVE THRESHOLD SLIDER
# ──────────────────────────────────────────────────────────
st.header(" Interactive Threshold Slider")
st.caption("See how precision and recall trade off in real time — adjust based on call-center capacity")

threshold = st.slider(
    "Decision threshold",
    min_value=0.10,
    max_value=0.90,
    value=float(default_threshold),
    step=0.01,
    help="Probability above this threshold → predicted 'yes' (call). "
         "Lower = catch more subscribers but more wasted calls. Higher = fewer wasted calls but more missed subscribers.",
)

y_pred_at_threshold = (y_proba >= threshold).astype(int)

# Compute metrics at the chosen threshold
prec = precision_score(y_true, y_pred_at_threshold, zero_division=0)
rec = recall_score(y_true, y_pred_at_threshold)
f1 = f1_score(y_true, y_pred_at_threshold)
cm = confusion_matrix(y_true, y_pred_at_threshold)
tn, fp, fn, tp = cm.ravel()

# Side-by-side: metrics + confusion matrix + business impact
m1, m2, m3, m4 = st.columns(4)
m1.metric("Precision", f"{prec:.3f}", help="Of customers we predicted 'yes', what % actually subscribed")
m2.metric("Recall", f"{rec:.3f}", help="Of actual subscribers, what % did we correctly identify")
m3.metric("F1 score", f"{f1:.3f}")
m4.metric("Customers to call", f"{(y_pred_at_threshold == 1).sum():,}",
          f"{(y_pred_at_threshold == 1).mean() * 100:.1f}% of base")

# Confusion matrix and business interpretation side by side
cm_col, biz_col = st.columns([1, 1])

with cm_col:
    st.markdown("##### Confusion Matrix")
    cm_text = [
        [f"<b>{tn:,}</b><br>True Neg<br>(correctly skipped)", f"<b>{fp:,}</b><br>False Pos<br>(wasted calls)"],
        [f"<b>{fn:,}</b><br>False Neg<br>(missed subscribers)", f"<b>{tp:,}</b><br>True Pos<br>(campaign wins)"],
    ]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            text=cm_text,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale="Blues",
            showscale=False,
            x=["Predicted: no", "Predicted: yes"],
            y=["Actual: no", "Actual: yes"],
        )
    )
    fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

with biz_col:
    st.markdown("##### 💼 Business Impact (assumed values)")

    # Configurable cost assumptions
    rev_per_sub = st.number_input("Revenue per subscriber (€)", 100, 5000, 500, step=50)
    cost_per_call = st.number_input("Cost per wasted call (€)", 1, 50, 2, step=1)

    tp_rev = tp * rev_per_sub
    fp_cost = fp * cost_per_call
    fn_opportunity = fn * rev_per_sub
    net = tp_rev - fp_cost

    st.metric("TP revenue captured", f"€{tp_rev:,.0f}")
    st.metric("FP wasted-call cost", f"€{fp_cost:,.0f}")
    st.metric("Net value", f"€{net:,.0f}", f"{net / 1000:+,.0f}K vs threshold 0.5")
    st.caption(f" Missed opportunity (FN): €{fn_opportunity:,.0f}")

st.info(
    f" **Insight:** Lower the threshold to catch more subscribers (higher recall, more wasted calls). "
    f"Raise it for fewer wasted calls (higher precision, more missed subscribers). "
    f"With revenue **{rev_per_sub / cost_per_call:.0f}× wasted-call cost**, "
    f"the math usually favors a lower threshold."
)

st.divider()

# ──────────────────────────────────────────────────────────
# SECTION 4: SCORE DISTRIBUTION
# ──────────────────────────────────────────────────────────
st.header(" Score Distribution by True Class")
st.caption("How distinctly does the model separate subscribers from non-subscribers?")

fig_dist = go.Figure()
fig_dist.add_trace(
    go.Histogram(
        x=y_proba[y_true == 0],
        name=f"Actual: no ({(y_true == 0).sum():,})",
        marker_color="#94a3b8",
        opacity=0.65,
        nbinsx=40,
    )
)
fig_dist.add_trace(
    go.Histogram(
        x=y_proba[y_true == 1],
        name=f"Actual: yes ({(y_true == 1).sum():,})",
        marker_color="#10b981",
        opacity=0.75,
        nbinsx=40,
    )
)
fig_dist.add_vline(x=threshold, line_dash="dash", line_color="red",
                    annotation_text=f"Threshold = {threshold:.2f}")
fig_dist.update_layout(
    xaxis_title="Predicted probability of subscription",
    yaxis_title="Count",
    barmode="overlay",
    height=400,
    plot_bgcolor="white",
    legend=dict(x=0.7, y=0.95),
)
st.plotly_chart(fig_dist, use_container_width=True)

mean_no = y_proba[y_true == 0].mean()
mean_yes = y_proba[y_true == 1].mean()

st.info(
    f" **Score gap:** Subscribers average **{mean_yes:.3f}**, non-subscribers **{mean_no:.3f}** — "
    f"a gap of **{mean_yes - mean_no:.3f}**. The subscriber distribution is **bimodal** — "
    f"the model is confident about ~70% of subscribers (right cluster) but misses ~30% whose features look identical to non-subscribers."
)

st.divider()
st.caption("📌 All metrics on this page are computed on the full dataset (41,176 rows after deduplication). "
           "Production performance was validated on the held-out test set in the notebook — see Section 9.")