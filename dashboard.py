import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="India Volatility & Regime Dashboard", layout="wide")

st.title(" India Market Volatility Dashboard 📈")
st.markdown("Nifty 50 | GARCH/EGARCH | XGBoost | LSTM | HMM Regime Detection")

@st.cache_data  
def load_data():
    raw = pd.read_csv("raw_market_data.csv", index_col="Date", parse_dates=True)
    regimes = pd.read_csv("hmm_regime_labeled.csv", index_col="Date", parse_dates=True)
    comparison = pd.read_csv("master_comparison_results.csv")
    return raw, regimes, comparison

raw_data, regime_data, comparison_data = load_data()

st.subheader("📅 Select Date Range")

min_date = raw_data.index.min().date()
max_date = raw_data.index.max().date()

col1, col2 = st.columns(2)

with col1:
    start = st.date_input(
        "Start date",
        value=max_date - pd.Timedelta(days=365),
        min_value=min_date,
        max_value=max_date
    )

with col2:
    end = st.date_input(
        "End date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

if start > end:
    st.error("⚠️ Start date must be before end date. Please adjust your selection.")
    st.stop()  
    
filtered_raw = raw_data.loc[str(start):str(end)]
filtered_regime = regime_data.loc[str(start):str(end)]
st.subheader("Nifty 50 Price")
st.line_chart(filtered_raw["NSEI"])

st.subheader("India VIX (Implied Volatility)")
st.line_chart(filtered_raw["INDIAVIX"])

st.write("Dashboard skeleton loaded successfully. Building further sections next.")
st.markdown("---")
st.subheader("📊 Volatility Forecasting Model Comparison")
st.markdown("RMSE/MAE for each model, evaluated on the same out-of-sample test period (last 20% of dates, chronologically).")

pivot_table = comparison_data.pivot(index="Model", columns="Target", values="RMSE")
st.dataframe(comparison_data.style.highlight_min(subset=["RMSE", "MAE"], color="lightgreen"))

st.markdown("""
**Key takeaway:** XGBoost achieves the lowest error on both targets, with LSTM close behind on
Realized Volatility but noticeably behind on VIX. EGARCH, while interpretable and grounded in
econometric theory, is outperformed by both ML models — though it remains valuable for its
transparent structure and statistically significant leverage effect finding.
""")

st.markdown("---")
st.subheader("🔍 Market Regime Detection (Hidden Markov Model)")

regime_colors = {
    "Bull (Strong Uptrend)": "#2ecc71",
    "Low-Momentum / Sideways": "#e74c3c",
    "High Volatility / Crisis": "#f39c12"
}

fig, ax = plt.subplots(figsize=(14, 5))
for label, color in regime_colors.items():
    mask = filtered_regime["regime_label"] == label
    ax.scatter(filtered_regime.index[mask], filtered_regime["NSEI"][mask],
               color=color, label=label, s=8)

ax.set_title("Nifty 50 Price Colored by Detected Regime")
ax.legend()
st.pyplot(fig)

if len(filtered_regime) > 0:
    latest_regime = filtered_regime["regime_label"].iloc[-1]
    latest_date = filtered_regime.index[-1].date()
    st.info(f"**Most recent detected regime** (as of {latest_date}): **{latest_regime}**")

st.subheader("Regime Distribution (selected period)")
regime_counts = filtered_regime["regime_label"].value_counts()
st.bar_chart(regime_counts)