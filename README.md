# NIFTY-50 AI Investment Intelligence Platform

A complete, reproducible investment decision-support platform built only from the provided NIFTY-50 stock market datasets in this workspace.

## Deliverables

- `app.py`: Streamlit working prototype.
- `reports/Technical_Report.pdf`: submission-ready technical report.
- `src/analytics.py`: data loading, feature engineering, modeling, portfolio construction, risk analytics, explainability, and anomaly detection.
- `scripts/build_artifacts.py`: reproducibly generates model and analytics artifacts.
- `scripts/generate_report.py`: regenerates the PDF report from local artifacts.
- `artifacts/`: generated model metrics, scores, portfolio allocations, anomalies, feature importance, and model pickle.

## Dataset Use and Constraints

The solution uses only local files supplied in:

- `archive/`: NIFTY-50 stock OHLCV files and `stock_metadata.csv`.
- `archive (2)/Datasets/INDEX/`: NIFTY 50 and INDIA VIX index context.

No live market data, APIs, news, social media sentiment, proprietary feeds, or external financial datasets are used.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the required packages are already installed globally, the virtual environment is optional.

## Reproduce Results

Generate all artifacts:

```bash
python3 scripts/build_artifacts.py
```

Generate the technical report:

```bash
python3 scripts/generate_report.py
```

Run the working prototype:

```bash
streamlit run app.py
```

The app will also generate missing artifacts automatically on first launch.

## Methodology Summary

1. Load current NIFTY-50 stock CSV files and metadata from the provided archive.
2. Add benchmark and volatility-regime context from supplied NIFTY 50 and INDIA VIX index files.
3. Engineer technical, liquidity, volatility, momentum, drawdown, beta, and VIX features.
4. Train a 21-trading-day return forecasting model using a strict temporal split.
5. Train a classifier for probability of positive 21-day return.
6. Rank stocks using forecast return, up probability, Sharpe ratio, and volatility penalty.
7. Construct Conservative, Balanced, and Aggressive long-only portfolios with position caps and risk-aversion penalties.
8. Explain recommendations with feature importance, visible feature values, quantitative rationales, and risk metrics.
9. Detect unusual volume, volatility spikes, and extreme drawdowns from historical rolling statistics.

## Outputs

Key generated files:

- `artifacts/model_metrics.json`: validation metrics.
- `artifacts/stock_scores.csv`: latest stock intelligence scores and forecasts.
- `artifacts/portfolio_profiles.csv`: profile-level portfolio metrics.
- `artifacts/portfolio_holdings.csv`: recommended allocations.
- `artifacts/feature_importance.csv`: global explainability.
- `artifacts/anomalies.csv`: recent anomaly flags.
- `artifacts/models.pkl`: trained local models.

## Notes

This is an investment intelligence and decision-support system, not financial advice. Historical technical data cannot capture all risks, including earnings surprises, macro shocks, transaction costs, taxes, or changing liquidity.
