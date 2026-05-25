"""
Recommendations tab — Data-backed action items for the marketing team.
Pure markdown content. No model loading needed.
"""

import streamlit as st

st.set_page_config(page_title="Recommendations", page_icon="💡", layout="wide")

st.title(" Recommendations to the Marketing Team")
st.caption("Data-backed action items derived from EDA, SHAP analysis, and lift chart")

st.divider()

# ──────────────────────────────────────────────────────────
# HEADLINE METRIC
# ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Top-20% Lead Capture", "66.4%", "of all subscribers")
col2.metric("Top-Decile Lift", "4.7×", "vs random baseline")
col3.metric("Production Model", "LightGBM", "ROC-AUC 0.8140")

st.divider()

# ──────────────────────────────────────────────────────────
# TARGETING STRATEGY
# ──────────────────────────────────────────────────────────
st.header(" Targeting Strategy")

st.markdown("""
**1. Call the top 20% of customers by model score** — captures **66.4% of all subscribers** at one-fifth the contact volume. This is the single most actionable headline of the project.

**2. Sharp cutoff at the top 20%** — customers in deciles 3–10 have lift below 1.0 (below random baseline). **The model only adds value for the top 20% of leads.** Everything below should be deprioritized or skipped entirely.

**3. Re-target past converters first.** `poutcome=success` customers convert at ~65% — over **5× the population average**. These are the warmest leads in the dataset.

**4. Prioritize age extremes, not the middle:**
- **65+ customers** convert at **46.8%** (4× population average)
- **Under-25 customers** convert at **20.9%** (1.9× average)
- The 35–54 middle segment converts at only 8.5–8.7% despite being the largest contact group
""")

with st.expander(" Tier-based segment priority list"):
    st.markdown("""
    | Tier | Segments | Why |
    |---|---|---|
    | **Tier 1 — Gold** | Students (~32%), Retired (~25%) | Highest conversion rates by far |
    | **Tier 2 — Strong** | Unemployed (~14%), Admin/Mgmt + university degree | Above-average converters |
    | **Tier 3 — Standard** | White-collar workers | At or near average |
    | **Tier 4 — Deprioritize** | Blue-collar, Services | Below-average conversion |
    """)

st.divider()

# ──────────────────────────────────────────────────────────
# CHANNEL & TIMING
# ──────────────────────────────────────────────────────────
st.header(" Channel & Timing")

st.markdown("""
**5. Use cellular, not telephone.** Cellular contacts convert at **14.7%**, telephone at **5.2%** — a 2.8× lift. SHAP analysis confirms `contact_cellular` is the **#3 most important feature** in the entire model, behind only macro indicators.

**6. Reallocate campaign volume away from May.** May receives ~2,500–2,900 contacts per day (the most of any month) but converts at only 5.3–7.2% — the worst of any month. Move budget to **golden months** that currently receive 20–80× fewer contacts:
""")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("March", "50.6%", "+39 pp vs May")
col_b.metric("December", "48.9%", "+38 pp vs May")
col_c.metric("September", "44.9%", "+34 pp vs May")
col_d.metric("October", "~45%", "+34 pp vs May")

st.markdown("""
**7. Cap contact attempts at 3–4 per customer per campaign.** Conversion drops below the population average after 3 contacts (10.7%). Beyond 4 attempts (9.4%), agents should pivot to fresh prospects.

**8. Avoid Mondays in golden months.** Within Mar/Sep/Oct/Dec, Tuesday and Wednesday hit 50–60% conversion; Mondays drop to 30–40%.
""")

st.divider()

# ──────────────────────────────────────────────────────────
# MACRO SIGNAL
# ──────────────────────────────────────────────────────────
st.header(" Macro Signal")

st.markdown("""
**9. Trigger opportunistic campaigns on Euribor drops.** When `euribor3m` falls below ~1.5%, conversion jumps to 45–50% (vs 6–10% at high rates). This is the cleanest causal macro signal in the model.

**10. Use macro features as a leading indicator** — `nr.employed`, `euribor3m`, and `emp.var.rate` are the top 3 SHAP features. Campaigns timed with ECB rate-cut announcements can compound the natural targeting gains.
""")

st.divider()

# ──────────────────────────────────────────────────────────
# OPERATIONAL USE & CAVEATS
# ──────────────────────────────────────────────────────────
st.header("🧠 Operational Use & Honest Caveats")

st.markdown("""
**11. Manual review for `'unknown' default` customers** — they're 21% of the dataset and the model has less signal on them.

**12. Use the SHAP dashboard tab before every high-value call** — agents get a per-customer "why this prediction" breakdown they can turn into a personalized pitch.

**13. Trust the model unevenly across job segments.** It's most reliable on housemaid (AUC 0.90), unemployed (0.87), admin (0.85), retired (0.82) — and weakest on the two largest segments: blue-collar (0.69) and services (0.70). **Weight agent intuition more heavily on those calls.**

**14. The model has a "hidden subscriber" blind spot.** Score-distribution analysis shows ~30% of actual subscribers have features that look identical to non-subscribers — the model will miss them no matter how it's tuned. **Closing this gap requires new features** (transaction history, web behavior, branch interactions), not better algorithms.
""")

st.divider()

# ──────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────
st.header(" Project Deliverables Summary")

st.markdown("""
-  **EDA report** — 13 charts with verified, business-specific insights
-  **Predictive model** — Tuned LightGBM via 100 Optuna trials, **Test ROC-AUC = 0.8140**
-  **Model evaluation** — Full diagnostics: threshold tuning, confusion matrix, ROC/PR curves, lift, calibration, per-segment AUC, score distribution, learning curve
-  **Explainable AI** — SHAP global + local (per-customer waterfall) explanations
-  **Recommendations** — 14 data-backed action items, each tied to a specific chart or metric
-  **Saved pipeline** — Ready for Streamlit dashboard + SHAP tab
-  **No data leakage** — `duration` excluded from production model
-  **Honest about limitations** — Documented model overconfidence (calibration), bimodal subscriber distribution, and segment fairness gaps
""")

st.success("**Bottom line:** Calling the top 20% of customers ranked by model score captures **66.4% of all subscribers** at one-fifth the contact effort — or equivalently, **multiplies effective conversion from 11.3% to ~37%** at the same budget.")