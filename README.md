# 📈 Nifty50 Investment Intelligence Platform

### Machine Learning-Based Stock Analysis, Forecasting & Portfolio Optimization

An end-to-end quantitative investment analysis platform designed to evaluate, rank, and recommend NIFTY-50 stocks using machine learning, portfolio optimization, risk analytics, and explainable AI techniques.

The platform transforms historical market data into actionable investment insights by combining predictive modeling, anomaly detection, feature engineering, and portfolio construction within a fully reproducible workflow.

---

## 🎯 Overview

The Nifty50 Investment Intelligence Platform is a data-driven decision-support system that analyzes historical stock market data to generate investment insights and portfolio recommendations.

The system leverages machine learning models, technical indicators, risk analytics, and explainability techniques to identify promising investment opportunities while accounting for market volatility and risk.

---

## 🚀 Key Features

### 📊 Market Intelligence Engine

* Processes historical OHLCV data for NIFTY-50 constituents.
* Generates momentum, volatility, liquidity, drawdown, and trend-based indicators.
* Incorporates benchmark and volatility-regime information using NIFTY-50 and INDIA VIX datasets.
* Produces comprehensive stock-level analytics.

### 🤖 Predictive Analytics

* Forecasts future stock performance using machine learning models.
* Estimates probability of positive returns over a forward investment horizon.
* Generates stock ranking scores using multiple performance factors.
* Evaluates model performance through validation metrics.

### 💼 Portfolio Optimization

* Builds investment portfolios for multiple risk profiles:

  * Conservative
  * Balanced
  * Aggressive
* Applies allocation constraints and diversification controls.
* Computes portfolio-level return and risk statistics.
* Generates optimized stock allocations.

### 🔍 Explainable AI

* Provides feature importance analysis.
* Generates transparent stock recommendations.
* Highlights key drivers influencing model predictions.
* Improves interpretability of investment decisions.

### ⚠️ Risk Monitoring

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

| Component                    | Description                                  |
| ---------------------------- | -------------------------------------------- |
| `app.py`                     | Streamlit-based interactive dashboard        |
| `src/analytics.py`           | Core analytics and machine learning pipeline |
| `scripts/build_artifacts.py` | Generates analytical outputs and models      |
| `scripts/generate_report.py` | Produces technical report                    |
| `artifacts/`                 | Model outputs and portfolio recommendations  |
| `reports/`                   | Technical documentation                      |
| `archive/`                   | Historical stock market datasets             |

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

### Clone the Repository

```bash
git clone https://github.com/mokshjain6174/nifty50-investment-platform.git
cd nifty50-investment-platform
```

### Install Dependencies

```bash
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

After launching, open the local URL displayed in the terminal (typically `http://localhost:8501`).

---

## 📊 Results

The platform automatically generates:

* Stock ranking and recommendation scores
* Future return forecasts
* Probability of positive returns
* Portfolio allocation recommendations
* Risk-adjusted portfolio metrics
* Feature importance analysis
* Market anomaly detection reports

---

## 📁 Generated Outputs

Important outputs are stored in the `artifacts/` directory:

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

### Output Descriptions

| File                     | Description                         |
| ------------------------ | ----------------------------------- |
| `stock_scores.csv`       | Stock rankings and forecast scores  |
| `portfolio_holdings.csv` | Recommended stock allocations       |
| `portfolio_profiles.csv` | Portfolio-level performance metrics |
| `feature_importance.csv` | Model explainability metrics        |
| `anomalies.csv`          | Detected market anomalies           |
| `model_metrics.json`     | Validation and evaluation metrics   |
| `models.pkl`             | Trained machine learning models     |

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

## 🔒 Dataset Constraints

This project uses only the datasets provided within the repository and does not rely on:

* Live market feeds
* External APIs
* News sentiment data
* Proprietary financial databases
* Third-party investment signals

This ensures complete reproducibility of all results.

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. It does not constitute investment advice, financial recommendations, or portfolio management services.

Investment decisions should always involve independent research, risk assessment, and consultation with qualified financial professionals.

---

## 👨‍💻 Author

**Moksh Jain**
IIT Roorkee

Machine Learning • Quantitative Finance • Data Analytics
