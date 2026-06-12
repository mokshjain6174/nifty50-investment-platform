from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics import ARTIFACT_DIR, ROOT, load_artifacts, save_artifacts

REPORT_PATH = ROOT / "reports" / "Technical_Report.pdf"


def wrap(text, width=95):
    words = str(text).split()
    lines, line = [], []
    for word in words:
        if sum(len(w) for w in line) + len(line) + len(word) > width:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def text_page(pdf, title, sections):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    y = 0.95
    fig.text(0.08, y, title, fontsize=18, weight="bold", va="top")
    y -= 0.055
    for heading, body in sections:
        fig.text(0.08, y, heading, fontsize=12, weight="bold", va="top")
        y -= 0.026
        fig.text(0.08, y, wrap(body), fontsize=9.3, va="top", linespacing=1.35)
        y -= 0.037 + 0.018 * (wrap(body).count("\n") + 1)
        if y < 0.10:
            break
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def table_page(pdf, title, df, note=""):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", loc="left", pad=16)
    if note:
        fig.text(0.04, 0.90, wrap(note, 145), fontsize=9, va="top")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#273746")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f6f7")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    if not (ARTIFACT_DIR / "stock_scores.csv").exists():
        save_artifacts()
    data = load_artifacts()
    summary = data["summary"]
    scores = data["scores"]
    portfolios = data["portfolios"]
    holdings = data["holdings"]
    anomalies = data["anomalies"]
    importance = data["importance"]
    metrics = summary["model_metrics"]
    REPORT_PATH.parent.mkdir(exist_ok=True)

    with PdfPages(REPORT_PATH) as pdf:
        text_page(
            pdf,
            "AI-Powered Investment Intelligence Using NIFTY-50 Market Data",
            [
                ("Objective", "This submission builds a local investment decision-support platform using only the provided NIFTY-50 stock archive, stock metadata, and NSE index datasets. The emphasis is portfolio construction, risk assessment, explainability, anomaly monitoring, and actionable investment support rather than standalone price prediction."),
                ("Dataset", f"The pipeline processed {summary['rows']:,} equity rows across {summary['stocks']} current NIFTY-50 stocks from {summary['date_start']} to {summary['date_end']}. Metadata contributes company and industry labels. NIFTY 50 and INDIA VIX files from the supplied second archive provide benchmark return, beta, and market-regime context."),
                ("Architecture", "A reproducible Python pipeline loads CSVs, engineers features, trains time-split models, builds portfolio profiles, detects anomalies, and writes deterministic artifacts. A Streamlit application reads those artifacts for interactive stock intelligence, portfolio construction, risk analysis, and explanation views."),
                ("Constraint Compliance", "No live market data, financial APIs, news, sentiment, proprietary feeds, or alternative external datasets are used. All calculations are derived from local files supplied in the project directory."),
            ],
        )
        text_page(
            pdf,
            "EDA and Feature Engineering",
            [
                ("Exploratory Findings", "The stock universe spans multiple industries including financial services, energy, automobiles, consumer goods, IT, metals, pharma, cement, telecom, services, construction, fertilizers, and media. The long history supports cycle-aware risk estimation and post-2020 out-of-sample validation."),
                ("Features", "Daily close, high, low, VWAP, volume, turnover, and benchmark series are transformed into return horizons, annualized volatility, moving-average gaps, EMA spread, MACD, RSI, Bollinger z-score, ATR, volume and turnover z-scores, drawdown, rolling beta to NIFTY 50, and VIX regime level."),
                ("Targets", "The predictive target is 21-trading-day forward return. A second binary target captures whether the 21-day forward return is positive, supporting directional decision-making."),
                ("Validation Design", "Training uses observations before 2020-01-01. Validation uses 2020 onward, preserving temporal order and avoiding look-ahead leakage."),
            ],
        )
        text_page(
            pdf,
            "Modeling Methodology and Explainability",
            [
                ("Predictor Engine", "A HistGradientBoostingRegressor forecasts 21-day return. A RandomForestClassifier estimates probability of positive 21-day return. Both use engineered technical, risk, liquidity, benchmark, and regime features."),
                ("Evaluation", f"Out-of-sample MAE is {metrics['mae_21d_return']:.2%}, RMSE is {metrics['rmse_21d_return']:.2%}, R2 is {metrics['r2_21d_return']:.3f}, regression directional accuracy is {metrics['directional_accuracy_regression']:.2%}, and classifier directional accuracy is {metrics['directional_accuracy_classifier']:.2%}."),
                ("Explainability", "The platform exposes global permutation importance, latest stock-level feature values, transparent stock rationales, risk metrics, anomaly tags, and portfolio weight evidence. This creates auditable recommendations rather than black-box buy lists."),
                ("Limitations", "The models learn from historical technical and index-derived behavior only. They do not include earnings, macro releases, news, analyst revisions, transaction costs, taxes, or live liquidity changes. Recommendations are decision-support outputs, not financial advice."),
            ],
        )

        top = scores.head(12)[["source_file_symbol", "industry", "quality_score", "forecast_return_21d", "forecast_up_probability", "sharpe", "volatility", "max_drawdown"]].copy()
        for col in ["quality_score", "forecast_return_21d", "forecast_up_probability", "volatility", "max_drawdown"]:
            top[col] = top[col].map(lambda x: f"{x:.2%}")
        top["sharpe"] = top["sharpe"].map(lambda x: f"{x:.2f}")
        table_page(pdf, "Top Ranked Stock Opportunities", top, "Ranking combines 21-day forecast, probability of upside, recent Sharpe ratio, and volatility penalty.")

        p = portfolios.copy()
        pct_cols = ["expected_21d_return", "expected_annualized_return", "annual_return", "volatility", "max_drawdown"]
        for col in pct_cols:
            p[col] = p[col].map(lambda x: f"{x:.2%}")
        for col in ["sharpe", "sortino", "calmar"]:
            p[col] = p[col].map(lambda x: f"{x:.2f}")
        table_page(pdf, "Portfolio Profiles and Risk Metrics", p[["profile", "expected_21d_return", "expected_annualized_return", "volatility", "sharpe", "sortino", "max_drawdown", "holdings"]])

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        top_imp = importance.head(12).sort_values("importance")
        ax.barh(top_imp["feature"], top_imp["importance"], color="#2874a6")
        ax.set_title("Global Explainability: Permutation Importance", loc="left", fontsize=16, weight="bold")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.scatter(scores["volatility"], scores["forecast_return_21d"], s=(scores["forecast_up_probability"] * 180).clip(20, 180), alpha=0.72)
        for _, row in scores.head(8).iterrows():
            ax.annotate(row["source_file_symbol"], (row["volatility"], row["forecast_return_21d"]), fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Investment Map: Forecast Return vs Historical Volatility", loc="left", fontsize=16, weight="bold")
        ax.set_xlabel("Recent annualized volatility")
        ax.set_ylabel("Forecast 21-day return")
        ax.grid(alpha=0.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        text_page(
            pdf,
            "Portfolio Construction, Risk, and Insights",
            [
                ("Construction Logic", "The system creates Conservative, Balanced, and Aggressive long-only portfolios. Each profile applies different volatility tolerance, maximum position size, and risk-aversion penalty. Candidate stocks must pass model-quality filters and are weighted by expected return, probability of upside, Sharpe ratio, and volatility risk."),
                ("Risk Assessment", "Every stock and portfolio is evaluated using annualized return, volatility, Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, and beta where relevant. Portfolio risk is calculated from the historical weighted return stream of selected holdings."),
                ("Anomaly Detection", "The anomaly module flags unusual volume, volatility spikes, and extreme drawdown states using rolling z-scores and historical drawdown thresholds. These signals help users separate ordinary ranking changes from unusual market behavior."),
                ("Key Insight", "The most useful decision signal is not a single forecast. The platform combines forecast direction, return magnitude, risk-adjusted performance, volatility, drawdown, and diversification to produce actionable but explainable investment choices."),
            ],
        )

    print(REPORT_PATH)


if __name__ == "__main__":
    main()
