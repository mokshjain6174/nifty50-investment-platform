# 📈 Nifty50 Investment Intelligence Platform

An end-to-end quantitative investment analysis platform designed to evaluate, rank, and recommend NIFTY-50 stocks using machine learning, portfolio optimization, risk analytics, and explainable AI techniques.

The platform transforms historical market data into actionable investment insights by combining predictive modeling, anomaly detection, feature engineering, and portfolio construction within a fully reproducible workflow.

---

## 🚀 Key Features

### Market Intelligence Engine

* Processes historical OHLCV data for NIFTY-50 constituents.
* Generates momentum, volatility, liquidity, drawdown, and trend-based indicators.
* Incorporates benchmark and volatility regime information using NIFTY-50 and INDIA VIX datasets.

### Predictive Analytics

* Forecasts future stock performance using machine learning models.
* Estimates probability of positive returns over a forward investment horizon.
* Ranks stocks using a composite scoring framework.

### Portfolio Optimization

* Builds investment portfolios for multiple risk profiles:

  * Conservative
  * Balanced
  * Aggressive
* Applies allocation constraints and diversification controls.
* Computes portfolio-level return and risk statistics.

### Explainable AI

* Provides feature importance analysis.
* Generates transparent stock recommendations.
* Highlights key drivers influencing model predictions.

### Risk Monitoring

* Detects abnormal market behavior.
* Identifies unusual trading volume spikes.
* Flags volatility shocks and drawdown events.
* Generates anomaly reports for investment review.

---

## 📂 Project Structure

```text
.
├── app.py
├── archive/
├── archive (2)/
├── artifacts/
├── reports/
├── scripts/
├── src/
└── requirements.txt
```

### Core Components

| Component                    | Description                                 |
| ---------------------------- | ------------------------------------------- |
| `app.py`                     | Streamlit-based interactive dashboard       |
| `src/analytics.py`           | Core analytics and modeling pipeline        |
| `scripts/build_artifacts.py` | Generates all analytical outputs            |
| `scripts/generate_report.py` | Produces technical report                   |
| `artifacts/`                 | Model outputs and portfolio recommendations |
| `reports/`                   | Technical documentation                     |

---

## 🛠️ Technology Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Matplotlib
* ReportLab

---

## ⚙️ Installation

```bash
git clone <repository-url>
cd nifty50-investment-platform

pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Generate Analytical Artifacts

```bash
python scripts/build_artifacts.py
```

### Generate Technical Report

```bash
python scripts/generate_report.py
```

### Launch Dashboard

```bash
streamlit run app.py
```

---

## 📊 Generated Outputs

The platform automatically creates:

* Stock ranking scores
* Portfolio recommendations
* Portfolio allocation weights
* Risk metrics
* Model evaluation statistics
* Feature importance analysis
* Market anomaly reports

Important outputs are stored in:

```text
artifacts/
├── stock_scores.csv
├── portfolio_holdings.csv
├── portfolio_profiles.csv
├── feature_importance.csv
├── anomalies.csv
├── model_metrics.json
└── models.pkl
```

---

## 🔬 Methodology

1. Data ingestion and validation
2. Feature engineering
3. Volatility regime analysis
4. Return forecasting
5. Classification modeling
6. Stock scoring and ranking
7. Portfolio construction
8. Explainability analysis
9. Risk and anomaly detection

---

## 📈 Investment Profiles

### Conservative Portfolio

Focuses on stability, lower volatility, and risk-adjusted returns.

### Balanced Portfolio

Balances growth opportunities with controlled risk exposure.

### Aggressive Portfolio

Prioritizes return maximization while accepting higher volatility.

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. It does not constitute investment advice, financial recommendations, or portfolio management services. Investment decisions should always involve independent research and professional consultation.
