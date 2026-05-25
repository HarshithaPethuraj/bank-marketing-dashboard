"""
Campaign Overview tab — Slice-and-dice conversion analytics by demographic, channel, and timing.
Marketers can filter the data and see how conversion rates shift across segments.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Campaign Overview", page_icon="📊", layout="wide")


# ──────────────────────────────────────────────────────────
# LOAD DATA (cached)
# ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = Path("data/bank-additional-full.csv")
    df = pd.read_csv(path, sep=";")
    df["y_bin"] = (df["y"] == "yes").astype(int)
    return df


df = load_data()

# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
st.title(" Campaign Overview")
st.caption("Interactive analytics on the Portuguese Bank's direct marketing campaign — May 2008 to November 2010")

# ──────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")
st.sidebar.caption("Slice the data to explore specific segments")

# Month filter
months_order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
available_months = [m for m in months_order if m in df["month"].unique()]
selected_months = st.sidebar.multiselect("Month", available_months, default=available_months)

# Contact method
contacts = sorted(df["contact"].unique())
selected_contacts = st.sidebar.multiselect("Contact method", contacts, default=contacts)

# Job filter
jobs = sorted(df["job"].unique())
selected_jobs = st.sidebar.multiselect("Job category", jobs, default=jobs)

# Age range slider
age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

# Apply filters
filtered = df[
    (df["month"].isin(selected_months))
    & (df["contact"].isin(selected_contacts))
    & (df["job"].isin(selected_jobs))
    & (df["age"].between(age_range[0], age_range[1]))
]

if len(filtered) == 0:
    st.warning(" No customers match the current filter combination. Try widening your filters.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.metric("Filtered customers", f"{len(filtered):,}", f"{len(filtered) / len(df) * 100:.1f}% of total")

# ──────────────────────────────────────────────────────────
# KPI ROW
# ──────────────────────────────────────────────────────────
n_total = len(filtered)
n_subscribers = filtered["y_bin"].sum()
conv_rate = filtered["y_bin"].mean() * 100
baseline_rate = df["y_bin"].mean() * 100
lift_vs_baseline = (conv_rate / baseline_rate) if baseline_rate > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers in segment", f"{n_total:,}")
col2.metric("Subscribers", f"{n_subscribers:,}")
col3.metric("Conversion rate", f"{conv_rate:.2f}%", f"{conv_rate - baseline_rate:+.2f} pp vs overall")
col4.metric("Lift vs random", f"{lift_vs_baseline:.2f}×")

st.divider()

# ──────────────────────────────────────────────────────────
# CHART 1: CONVERSION BY MONTH
# ──────────────────────────────────────────────────────────
st.subheader("Conversion Rate by Month")
st.caption("Identifies which months produce the best campaign returns")

month_stats = (
    filtered.groupby("month")
    .agg(customers=("y_bin", "count"), subscribers=("y_bin", "sum"), conv_rate=("y_bin", "mean"))
    .reset_index()
)
month_stats["conv_rate"] *= 100
month_stats["month"] = pd.Categorical(month_stats["month"], categories=months_order, ordered=True)
month_stats = month_stats.sort_values("month")

fig_month = go.Figure()
fig_month.add_trace(
    go.Bar(
        x=month_stats["month"],
        y=month_stats["conv_rate"],
        text=[f"{v:.1f}%" for v in month_stats["conv_rate"]],
        textposition="outside",
        marker_color=["#10b981" if v >= baseline_rate else "#94a3b8" for v in month_stats["conv_rate"]],
        hovertemplate="<b>%{x}</b><br>Conversion: %{y:.2f}%<br>Customers: %{customdata[0]:,}<br>Subscribers: %{customdata[1]:,}<extra></extra>",
        customdata=month_stats[["customers", "subscribers"]].values,
    )
)
fig_month.add_hline(y=baseline_rate, line_dash="dash", line_color="red",
                     annotation_text=f"Overall avg: {baseline_rate:.1f}%", annotation_position="right")
fig_month.update_layout(
    yaxis_title="Conversion Rate (%)", xaxis_title="Month", height=400,
    plot_bgcolor="white", showlegend=False,
)
st.plotly_chart(fig_month, use_container_width=True)

st.info(
    " **Insight:** March, September, October, and December are the **golden months** "
    "with 40–50%+ conversion rates. May has the highest contact volume but the lowest conversion."
)

st.divider()

# ──────────────────────────────────────────────────────────
# CHART 2: CONVERSION BY JOB
# ──────────────────────────────────────────────────────────
st.subheader(" Conversion Rate by Job Category")
st.caption("Which professions are most receptive to term deposits?")

job_stats = (
    filtered.groupby("job")
    .agg(customers=("y_bin", "count"), subscribers=("y_bin", "sum"), conv_rate=("y_bin", "mean"))
    .reset_index()
    .sort_values("conv_rate", ascending=True)
)
job_stats["conv_rate"] *= 100

fig_job = go.Figure()
fig_job.add_trace(
    go.Bar(
        y=job_stats["job"],
        x=job_stats["conv_rate"],
        orientation="h",
        text=[f"{v:.1f}%" for v in job_stats["conv_rate"]],
        textposition="outside",
        marker_color=["#10b981" if v >= baseline_rate else "#94a3b8" for v in job_stats["conv_rate"]],
        hovertemplate="<b>%{y}</b><br>Conversion: %{x:.2f}%<br>Customers: %{customdata[0]:,}<extra></extra>",
        customdata=job_stats[["customers"]].values,
    )
)
fig_job.add_vline(x=baseline_rate, line_dash="dash", line_color="red",
                   annotation_text=f"Avg: {baseline_rate:.1f}%")
fig_job.update_layout(
    xaxis_title="Conversion Rate (%)", yaxis_title="Job", height=450,
    plot_bgcolor="white", showlegend=False,
)
st.plotly_chart(fig_job, use_container_width=True)

st.info(
    " **Insight:** Students and retired customers convert at **2–4× the population average**. "
    "Blue-collar and services customers convert below average and represent the bulk of the contact volume."
)

st.divider()

# ──────────────────────────────────────────────────────────
# CHART 3: CONVERSION BY AGE BAND
# ──────────────────────────────────────────────────────────
st.subheader(" Conversion Rate by Age Band")
st.caption("Both extremes outperform the middle — the 'age U-curve'")

filtered_age = filtered.copy()
filtered_age["age_band"] = pd.cut(
    filtered_age["age"],
    bins=[0, 25, 35, 45, 55, 65, 100],
    labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
)
age_stats = filtered_age.groupby("age_band", observed=True).agg(
    customers=("y_bin", "count"), conv_rate=("y_bin", "mean")
).reset_index()
age_stats["conv_rate"] *= 100

fig_age = go.Figure()
fig_age.add_trace(
    go.Bar(
        x=age_stats["age_band"].astype(str),
        y=age_stats["conv_rate"],
        text=[f"{v:.1f}%" for v in age_stats["conv_rate"]],
        textposition="outside",
        marker_color=["#10b981" if v >= baseline_rate else "#94a3b8" for v in age_stats["conv_rate"]],
        hovertemplate="<b>%{x}</b><br>Conversion: %{y:.2f}%<br>Customers: %{customdata[0]:,}<extra></extra>",
        customdata=age_stats[["customers"]].values,
    )
)
fig_age.add_hline(y=baseline_rate, line_dash="dash", line_color="red",
                   annotation_text=f"Avg: {baseline_rate:.1f}%")
fig_age.update_layout(
    yaxis_title="Conversion Rate (%)", xaxis_title="Age Band", height=400,
    plot_bgcolor="white", showlegend=False,
)
st.plotly_chart(fig_age, use_container_width=True)

st.info(
    "💡 **Insight:** Customers under 25 (~21%) and over 65 (~47%) convert dramatically better than "
    "the 25–54 middle (8–12%). This is a textbook example of why linear models fail — the relationship is **U-shaped**, not monotonic."
)

st.divider()

# ──────────────────────────────────────────────────────────
# CHART 4: CONTACT METHOD COMPARISON
# ──────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader(" Contact Channel")
    contact_stats = filtered.groupby("contact").agg(
        customers=("y_bin", "count"), conv_rate=("y_bin", "mean")
    ).reset_index()
    contact_stats["conv_rate"] *= 100

    fig_contact = px.bar(
        contact_stats,
        x="contact",
        y="conv_rate",
        text=[f"{v:.1f}%" for v in contact_stats["conv_rate"]],
        color="contact",
        color_discrete_map={"cellular": "#10b981", "telephone": "#94a3b8"},
    )
    fig_contact.update_traces(textposition="outside")
    fig_contact.update_layout(
        yaxis_title="Conversion Rate (%)", xaxis_title="", height=350,
        plot_bgcolor="white", showlegend=False,
    )
    st.plotly_chart(fig_contact, use_container_width=True)

with col_right:
    st.subheader(" Previous Outcome")
    if "poutcome" in filtered.columns:
        pout_stats = filtered.groupby("poutcome").agg(
            customers=("y_bin", "count"), conv_rate=("y_bin", "mean")
        ).reset_index()
        pout_stats["conv_rate"] *= 100
        pout_stats = pout_stats.sort_values("conv_rate", ascending=False)

        fig_pout = px.bar(
            pout_stats,
            x="poutcome",
            y="conv_rate",
            text=[f"{v:.1f}%" for v in pout_stats["conv_rate"]],
            color="poutcome",
            color_discrete_map={"success": "#10b981", "failure": "#ef4444", "nonexistent": "#94a3b8"},
        )
        fig_pout.update_traces(textposition="outside")
        fig_pout.update_layout(
            yaxis_title="Conversion Rate (%)", xaxis_title="", height=350,
            plot_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(fig_pout, use_container_width=True)

st.info(
    " **Insight:** Cellular contacts convert at **~3× the rate** of telephone. "
    "Previous campaign success (`poutcome=success`) is the strongest single positive predictor — "
    "past converters are warm leads worth re-targeting."
)

st.divider()

# ──────────────────────────────────────────────────────────
# CHART 5: CAMPAIGN ATTEMPTS — DIMINISHING RETURNS
# ─────────────────────────────────────────────────────────
st.subheader(" Diminishing Returns by Contact Attempts")
st.caption("Each additional contact attempt yields lower conversion")

camp_filtered = filtered.copy()
camp_filtered["campaign_clipped"] = camp_filtered["campaign"].clip(upper=10)
camp_stats = camp_filtered.groupby("campaign_clipped").agg(
    customers=("y_bin", "count"), conv_rate=("y_bin", "mean")
).reset_index()
camp_stats["conv_rate"] *= 100
camp_stats = camp_stats[camp_stats["customers"] >= 20]  # hide noisy low-sample bars

fig_camp = go.Figure()
fig_camp.add_trace(
    go.Bar(
        x=camp_stats["campaign_clipped"].astype(str),
        y=camp_stats["conv_rate"],
        text=[f"{v:.1f}%" for v in camp_stats["conv_rate"]],
        textposition="outside",
        marker_color=["#10b981" if v >= baseline_rate else "#94a3b8" for v in camp_stats["conv_rate"]],
        hovertemplate="<b>%{x} contacts</b><br>Conversion: %{y:.2f}%<br>Customers: %{customdata[0]:,}<extra></extra>",
        customdata=camp_stats[["customers"]].values,
    )
)
fig_camp.add_hline(y=baseline_rate, line_dash="dash", line_color="red",
                    annotation_text=f"Avg: {baseline_rate:.1f}%")
fig_camp.update_layout(
    yaxis_title="Conversion Rate (%)",
    xaxis_title="Number of Contacts in Campaign (10 = 10+)",
    height=400,
    plot_bgcolor="white",
    showlegend=False,
)
st.plotly_chart(fig_camp, use_container_width=True)

st.info(
    " **Insight:** First contact converts best (~13%). By the 4th contact, conversion drops below 10%. "
    "**Recommended cap: 3–4 contact attempts per customer per campaign.**"
)

st.divider()
st.caption("📌 All charts respect the filters in the sidebar. Adjust filters to explore specific segments.")