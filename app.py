from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import ARTIFACT_DIR, PROFILES, load_artifacts, save_artifacts


st.set_page_config(page_title="NIFTY-50 Investment Intelligence", layout="wide")


@st.cache_data(show_spinner=False)
def cached_artifacts():
    return load_artifacts()


if not (ARTIFACT_DIR / "stock_scores.csv").exists():
    with st.spinner("Generating local analytics artifacts from provided datasets..."):
        save_artifacts()

data = cached_artifacts()
summary = data["summary"]
features = data["features"]
scores = data["scores"]
portfolios = data["portfolios"]
holdings = data["holdings"]
anomalies = data["anomalies"]
importance = data["importance"]
metrics = summary["model_metrics"]

st.title("NIFTY-50 AI Investment Intelligence Platform")
st.caption("Built only from the provided historical NIFTY-50, NSE index, and metadata datasets. No live data, APIs, news, or sentiment feeds are used.")

with st.sidebar:
    st.header("Controls")
    page = st.radio(
        "View",
        ["Market Overview", "Stock Intelligence", "Portfolio Builder", "Risk Lab", "Anomaly Monitor", "Methodology"],
    )
    profile = st.selectbox("Investor profile", list(PROFILES.keys()), index=1)
    symbol = st.selectbox(
        "Stock",
        scores.sort_values("quality_score", ascending=False)["source_file_symbol"].tolist(),
        index=0,
    )


def fmt_pct(x):
    if pd.isna(x):
        return "n/a"
    return f"{x:.2%}"


if page == "Market Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Investable stocks", summary["stocks"])
    c2.metric("Rows processed", f"{summary['rows']:,}")
    c3.metric("History", f"{summary['date_start']} to {summary['date_end']}")
    c4.metric("Industries", summary["industries"])

    st.subheader("Model Validation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE, 21-day return", fmt_pct(metrics["mae_21d_return"]))
    m2.metric("RMSE, 21-day return", fmt_pct(metrics["rmse_21d_return"]))
    m3.metric("Directional accuracy", fmt_pct(metrics["directional_accuracy_regression"]))
    m4.metric("Classifier direction accuracy", fmt_pct(metrics["directional_accuracy_classifier"]))
    st.caption(f"Validation uses a strict time split: train before 2020-01-01, test from {metrics['test_start']} to {metrics['test_end']}.")

    top = scores.head(15).copy()
    top["forecast_return_21d_pct"] = top["forecast_return_21d"] * 100
    fig = px.bar(
        top,
        x="source_file_symbol",
        y="quality_score",
        color="industry",
        hover_data=["company", "forecast_return_21d_pct", "sharpe", "volatility"],
        title="Top AI-ranked opportunities",
    )
    st.plotly_chart(fig, use_container_width=True)

    imp = importance.head(12)
    st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h", title="Global model explainability: permutation importance"), use_container_width=True)

elif page == "Stock Intelligence":
    row = scores[scores["source_file_symbol"].eq(symbol)].iloc[0]
    hist = features[features["source_file_symbol"].eq(symbol)].sort_values("date").copy()
    st.subheader(f"{symbol}: {row['company']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("21-day forecast", fmt_pct(row["forecast_return_21d"]))
    c2.metric("Up probability", fmt_pct(row["forecast_up_probability"]))
    c3.metric("Recent Sharpe", f"{row['sharpe']:.2f}")
    c4.metric("Volatility", fmt_pct(row["volatility"]))
    c5.metric("Max drawdown", fmt_pct(row["max_drawdown"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"], name="Close"))
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"].rolling(50).mean(), name="50D average"))
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"].rolling(200).mean(), name="200D average"))
    fig.update_layout(title="Price trend and moving averages", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Actionable Rationale")
    rationale = []
    if row["forecast_return_21d"] > scores["forecast_return_21d"].median():
        rationale.append("Model forecast is above the NIFTY-50 stock median for the next 21 trading days.")
    else:
        rationale.append("Model forecast is below the current universe median, so position sizing should be conservative.")
    if row["sharpe"] > scores["sharpe"].median():
        rationale.append("Recent risk-adjusted return is stronger than peers.")
    else:
        rationale.append("Recent risk-adjusted return is weaker than peers.")
    if row["volatility"] > scores["volatility"].quantile(0.75):
        rationale.append("Volatility is in the upper quartile, increasing allocation risk.")
    if row["drawdown"] < -0.2:
        rationale.append("The stock remains materially below its historical peak, indicating unresolved drawdown risk or recovery potential.")
    for item in rationale:
        st.write(f"- {item}")

    latest_features = hist.tail(1)[["rsi_14", "sma20_gap", "sma50_gap", "bollinger_z", "beta_252", "vix_close"]].T
    latest_features.columns = ["latest"]
    st.dataframe(latest_features.style.format("{:.3f}"), use_container_width=True)

elif page == "Portfolio Builder":
    p = portfolios[portfolios["profile"].eq(profile)].iloc[0]
    h = holdings[holdings["profile"].eq(profile)].sort_values("weight", ascending=False)
    st.subheader(f"{profile} Portfolio")
    st.caption(p["description"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Expected 21D return", fmt_pct(p["expected_21d_return"]))
    c2.metric("Expected annualized", fmt_pct(p["expected_annualized_return"]))
    c3.metric("Historical volatility", fmt_pct(p["volatility"]))
    c4.metric("Sharpe", f"{p['sharpe']:.2f}")
    c5.metric("Max drawdown", fmt_pct(p["max_drawdown"]))

    fig = px.pie(h, names="source_file_symbol", values="weight", title="Recommended allocation")
    st.plotly_chart(fig, use_container_width=True)

    sector = h.groupby("industry", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
    st.plotly_chart(px.bar(sector, x="industry", y="weight", title="Sector diversification"), use_container_width=True)

    st.subheader("Holdings and Justification")
    display = h[["source_file_symbol", "company", "industry", "weight", "forecast_return_21d", "forecast_up_probability", "sharpe", "volatility", "max_drawdown"]].copy()
    st.dataframe(display.style.format({"weight":"{:.2%}", "forecast_return_21d":"{:.2%}", "forecast_up_probability":"{:.2%}", "sharpe":"{:.2f}", "volatility":"{:.2%}", "max_drawdown":"{:.2%}"}), use_container_width=True)

elif page == "Risk Lab":
    st.subheader("Cross-sectional Risk Map")
    plot = scores.copy()
    plot["forecast_return_21d_pct"] = plot["forecast_return_21d"] * 100
    fig = px.scatter(
        plot,
        x="volatility",
        y="forecast_return_21d",
        size="forecast_up_probability",
        color="industry",
        hover_name="source_file_symbol",
        hover_data=["company", "sharpe", "max_drawdown"],
        title="Forecast return versus historical volatility",
    )
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Table")
    cols = ["source_file_symbol", "company", "industry", "forecast_return_21d", "forecast_up_probability", "annual_return", "volatility", "sharpe", "sortino", "max_drawdown", "calmar", "beta_252"]
    st.dataframe(scores[cols].sort_values("sharpe", ascending=False).style.format({"forecast_return_21d":"{:.2%}", "forecast_up_probability":"{:.2%}", "annual_return":"{:.2%}", "volatility":"{:.2%}", "sharpe":"{:.2f}", "sortino":"{:.2f}", "max_drawdown":"{:.2%}", "calmar":"{:.2f}", "beta_252":"{:.2f}"}), use_container_width=True)

elif page == "Anomaly Monitor":
    st.subheader("Recent Historical Anomalies")
    if anomalies.empty:
        st.info("No anomalies found with the current thresholds.")
    else:
        display = anomalies.copy()
        display["signal"] = np.select(
            [display["volume_anomaly"], display["volatility_anomaly"], display["drawdown_anomaly"]],
            ["Unusual volume", "Volatility spike", "Extreme drawdown"],
            default="Mixed",
        )
        st.dataframe(display[["date", "source_file_symbol", "company", "industry", "signal", "ret_1d", "volume_z_20", "abs_return_z", "drawdown"]].style.format({"ret_1d":"{:.2%}", "volume_z_20":"{:.2f}", "abs_return_z":"{:.2f}", "drawdown":"{:.2%}"}), use_container_width=True)
        fig = px.scatter(display, x="date", y="source_file_symbol", color="signal", size=display["ret_1d"].abs() + 0.001, hover_data=["company", "ret_1d", "volume_z_20", "abs_return_z", "drawdown"], title="Anomaly timeline")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("Methodology")
    st.write("The platform transforms local historical prices into technical, volatility, drawdown, beta, volume, and benchmark-regime features. A time-split gradient boosting model forecasts 21-trading-day returns, while a random forest estimates the probability of positive forward return.")
    st.write("Portfolio construction is long-only. Each investor profile applies a different risk aversion, volatility tolerance, and maximum position weight. Holdings are selected from model-ranked stocks and scored using forecast return, probability of upside, Sharpe ratio, and volatility penalties.")
    st.write("Explainability is provided through global permutation importance, visible feature values, stock-level rationales, portfolio weight evidence, and anomaly labels. The approach is intentionally auditable and uses only the provided datasets.")
    st.json(metrics)
