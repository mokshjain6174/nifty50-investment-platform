from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "archive"
INDEX_DIR = ROOT / "archive (2)" / "Datasets" / "INDEX"
ARTIFACT_DIR = ROOT / "artifacts"

TRADING_DAYS = 252
HORIZON_DAYS = 21
FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "vol_20d",
    "vol_60d",
    "sma20_gap",
    "sma50_gap",
    "ema12_26_gap",
    "rsi_14",
    "macd",
    "bollinger_z",
    "atr_14",
    "volume_z_20",
    "turnover_z_20",
    "drawdown",
    "beta_252",
    "vix_close",
    "symbol_code",
    "industry_code",
]


@dataclass(frozen=True)
class Profile:
    name: str
    max_weight: float
    min_score_quantile: float
    risk_aversion: float
    max_volatility: float
    description: str


PROFILES = {
    "Conservative": Profile(
        "Conservative",
        max_weight=0.12,
        min_score_quantile=0.45,
        risk_aversion=8.0,
        max_volatility=0.28,
        description="Prioritizes drawdown control, diversification, and stable risk-adjusted returns.",
    ),
    "Balanced": Profile(
        "Balanced",
        max_weight=0.16,
        min_score_quantile=0.35,
        risk_aversion=4.5,
        max_volatility=0.36,
        description="Balances expected return, Sharpe ratio, volatility, and sector diversification.",
    ),
    "Aggressive": Profile(
        "Aggressive",
        max_weight=0.22,
        min_score_quantile=0.25,
        risk_aversion=2.2,
        max_volatility=0.50,
        description="Accepts higher volatility in exchange for stronger model-ranked upside.",
    ),
}


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(DATA_DIR / "stock_metadata.csv")
    return meta.rename(columns={"Company Name": "company", "Industry": "industry"})


def _stock_files() -> List[Path]:
    excluded = {"NIFTY50_all.csv", "stock_metadata.csv"}
    return sorted(p for p in DATA_DIR.glob("*.csv") if p.name not in excluded)


def load_stock_data() -> pd.DataFrame:
    frames = []
    for path in _stock_files():
        df = pd.read_csv(path)
        df["source_file_symbol"] = path.stem
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data[data["Series"].eq("EQ")].copy()
    numeric = [
        "Prev Close",
        "Open",
        "High",
        "Low",
        "Last",
        "Close",
        "VWAP",
        "Volume",
        "Turnover",
        "Trades",
        "Deliverable Volume",
        "%Deliverble",
    ]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.rename(
        columns={
            "Date": "date",
            "Symbol": "symbol",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "VWAP": "vwap",
            "Volume": "volume",
            "Turnover": "turnover",
            "Trades": "trades",
            "Deliverable Volume": "deliverable_volume",
            "%Deliverble": "deliverable_pct",
        }
    )
    meta = load_metadata()[["Symbol", "company", "industry"]].rename(columns={"Symbol": "source_file_symbol"})
    data = data.merge(meta, on="source_file_symbol", how="left")
    data["company"] = data["company"].fillna(data["source_file_symbol"])
    data["industry"] = data["industry"].fillna("UNKNOWN")
    return data.sort_values(["source_file_symbol", "date"]).reset_index(drop=True)


def load_index_context() -> pd.DataFrame:
    idx = pd.read_csv(INDEX_DIR / "NIFTY 50.csv", parse_dates=["Date"])
    idx = idx.rename(columns={"Date": "date", "Close": "nifty_close"})[["date", "nifty_close"]]
    idx["nifty_ret_1d"] = idx["nifty_close"].pct_change()
    vix_path = INDEX_DIR / "INDIA VIX.csv"
    if vix_path.exists():
        vix = pd.read_csv(vix_path, parse_dates=["Date"]).rename(columns={"Date": "date", "Close": "vix_close"})
        idx = idx.merge(vix[["date", "vix_close"]], on="date", how="left")
    else:
        idx["vix_close"] = np.nan
    idx["vix_close"] = idx["vix_close"].ffill()
    return idx


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def engineer_features(raw: pd.DataFrame, index_context: pd.DataFrame) -> pd.DataFrame:
    df = raw.merge(index_context, on="date", how="left")
    df["nifty_ret_1d"] = df["nifty_ret_1d"].fillna(0)
    df["vix_close"] = df["vix_close"].ffill().fillna(df["vix_close"].median())
    out = []
    for symbol, g in df.groupby("source_file_symbol", sort=False):
        g = g.sort_values("date").copy()
        g["ret_1d"] = g["close"].pct_change()
        g["ret_5d"] = g["close"].pct_change(5)
        g["ret_21d"] = g["close"].pct_change(21)
        g["ret_63d"] = g["close"].pct_change(63)
        log_ret = np.log(g["close"]).diff()
        g["vol_20d"] = log_ret.rolling(20).std() * math.sqrt(TRADING_DAYS)
        g["vol_60d"] = log_ret.rolling(60).std() * math.sqrt(TRADING_DAYS)
        sma20 = g["close"].rolling(20).mean()
        sma50 = g["close"].rolling(50).mean()
        g["sma20_gap"] = g["close"] / sma20 - 1
        g["sma50_gap"] = g["close"] / sma50 - 1
        ema12 = g["close"].ewm(span=12, adjust=False).mean()
        ema26 = g["close"].ewm(span=26, adjust=False).mean()
        g["ema12_26_gap"] = ema12 / ema26 - 1
        g["macd"] = ema12 - ema26
        g["rsi_14"] = rsi(g["close"], 14)
        rolling_mean = g["close"].rolling(20).mean()
        rolling_std = g["close"].rolling(20).std()
        g["bollinger_z"] = (g["close"] - rolling_mean) / rolling_std.replace(0, np.nan)
        true_range = pd.concat(
            [
                g["high"] - g["low"],
                (g["high"] - g["close"].shift()).abs(),
                (g["low"] - g["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        g["atr_14"] = true_range.rolling(14).mean() / g["close"]
        g["volume_z_20"] = (g["volume"] - g["volume"].rolling(20).mean()) / g["volume"].rolling(20).std()
        g["turnover_z_20"] = (g["turnover"] - g["turnover"].rolling(20).mean()) / g["turnover"].rolling(20).std()
        g["drawdown"] = g["close"] / g["close"].cummax() - 1
        cov = g["ret_1d"].rolling(252).cov(g["nifty_ret_1d"])
        var = g["nifty_ret_1d"].rolling(252).var()
        g["beta_252"] = cov / var.replace(0, np.nan)
        g["target_return_21d"] = g["close"].shift(-HORIZON_DAYS) / g["close"] - 1
        g["target_up_21d"] = (g["target_return_21d"] > 0).astype(int)
        out.append(g)
    features = pd.concat(out, ignore_index=True)
    features["symbol_code"] = features["source_file_symbol"].astype("category").cat.codes
    features["industry_code"] = features["industry"].astype("category").cat.codes
    features[FEATURE_COLUMNS] = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return features


def risk_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {k: float("nan") for k in ["annual_return", "volatility", "sharpe", "sortino", "max_drawdown", "calmar"]}
    ann_return = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
    vol = r.std() * math.sqrt(TRADING_DAYS)
    downside = r[r < 0].std() * math.sqrt(TRADING_DAYS)
    sharpe = (ann_return - risk_free_rate) / vol if vol and not np.isnan(vol) else np.nan
    sortino = (ann_return - risk_free_rate) / downside if downside and not np.isnan(downside) else np.nan
    curve = (1 + r).cumprod()
    drawdown = curve / curve.cummax() - 1
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "annual_return": ann_return,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def build_model_dataset(features: pd.DataFrame) -> pd.DataFrame:
    model_df = features.dropna(subset=FEATURE_COLUMNS + ["target_return_21d"]).copy()
    return model_df[model_df["date"] <= pd.Timestamp("2021-03-31")].copy()


def train_models(model_df: pd.DataFrame) -> Tuple[object, object, Dict[str, float], pd.DataFrame]:
    train = model_df[model_df["date"] < pd.Timestamp("2020-01-01")]
    test = model_df[model_df["date"] >= pd.Timestamp("2020-01-01")]
    X_train, y_train = train[FEATURE_COLUMNS], train["target_return_21d"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target_return_21d"]
    reg = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.055, l2_regularization=0.05, random_state=42)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    clf = RandomForestClassifier(
        n_estimators=140,
        max_depth=8,
        min_samples_leaf=80,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, train["target_up_21d"])
    cls_pred = clf.predict(X_test)

    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae_21d_return": float(mean_absolute_error(y_test, pred)),
        "rmse_21d_return": float(math.sqrt(mean_squared_error(y_test, pred))),
        "r2_21d_return": float(r2_score(y_test, pred)),
        "directional_accuracy_regression": float(accuracy_score(y_test > 0, pred > 0)),
        "directional_accuracy_classifier": float(accuracy_score(test["target_up_21d"], cls_pred)),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
    }
    try:
        sample = test.sample(min(5000, len(test)), random_state=42)
        perm = permutation_importance(reg, sample[FEATURE_COLUMNS], sample["target_return_21d"], n_repeats=5, random_state=42)
        importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": perm.importances_mean})
    except Exception:
        importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": np.nan})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    return reg, clf, metrics, importance


def latest_stock_snapshot(features: pd.DataFrame, reg, clf) -> pd.DataFrame:
    latest = (
        features.dropna(subset=FEATURE_COLUMNS)
        .sort_values("date")
        .groupby("source_file_symbol", as_index=False)
        .tail(1)
        .copy()
    )
    latest["forecast_return_21d"] = reg.predict(latest[FEATURE_COLUMNS])
    latest["forecast_up_probability"] = clf.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest["forecast_annualized"] = (1 + latest["forecast_return_21d"]).pow(TRADING_DAYS / HORIZON_DAYS) - 1
    metrics = []
    for sym, g in features.groupby("source_file_symbol"):
        recent = g.sort_values("date").tail(252)
        m = risk_metrics(recent["ret_1d"])
        m["source_file_symbol"] = sym
        metrics.append(m)
    risk = pd.DataFrame(metrics)
    latest = latest.merge(risk, on="source_file_symbol", how="left", suffixes=("", "_recent"))
    latest["quality_score"] = (
        latest["forecast_return_21d"].rank(pct=True) * 0.35
        + latest["forecast_up_probability"].rank(pct=True) * 0.25
        + latest["sharpe"].rank(pct=True) * 0.25
        + (1 - latest["volatility"].rank(pct=True)) * 0.15
    )
    return latest.sort_values("quality_score", ascending=False)


def stock_return_matrix(features: pd.DataFrame, days: int = 756) -> pd.DataFrame:
    pivot = features.pivot_table(index="date", columns="source_file_symbol", values="ret_1d")
    return pivot.tail(days).dropna(axis=1, thresh=int(days * 0.65)).fillna(0)


def _cap_and_normalize(scores: pd.Series, max_weight: float, max_iter: int = 20) -> pd.Series:
    weights = scores.clip(lower=0).copy()
    if weights.sum() <= 0:
        weights[:] = 1 / len(weights)
    else:
        weights = weights / weights.sum()
    for _ in range(max_iter):
        over = weights > max_weight
        if not over.any():
            break
        excess = (weights[over] - max_weight).sum()
        weights[over] = max_weight
        under = ~over
        if weights[under].sum() <= 0:
            break
        weights[under] += excess * weights[under] / weights[under].sum()
    return weights / weights.sum()


def construct_portfolios(latest: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    returns = stock_return_matrix(features)
    portfolios = []
    holdings = []
    for profile in PROFILES.values():
        candidates = latest[latest["volatility"].le(profile.max_volatility)].copy()
        if len(candidates) < 8:
            candidates = latest.copy()
        threshold = candidates["quality_score"].quantile(profile.min_score_quantile)
        candidates = candidates[candidates["quality_score"] >= threshold].copy()
        candidates = candidates[candidates["source_file_symbol"].isin(returns.columns)]
        candidates["raw_score"] = (
            candidates["forecast_annualized"].clip(lower=-0.5, upper=1.5)
            + 0.35 * candidates["sharpe"].fillna(0)
            - profile.risk_aversion * candidates["volatility"].fillna(candidates["volatility"].median()) ** 2
            + 0.20 * candidates["forecast_up_probability"]
        )
        candidates = candidates.sort_values("raw_score", ascending=False).head(18)
        if candidates.empty:
            continue
        base = candidates.set_index("source_file_symbol")["raw_score"]
        base = base - base.min() + 0.01
        weights = _cap_and_normalize(base, profile.max_weight)
        profile_returns = returns[weights.index].mul(weights, axis=1).sum(axis=1)
        m = risk_metrics(profile_returns)
        expected_21d = float((candidates.set_index("source_file_symbol")["forecast_return_21d"] * weights).sum())
        row = {
            "profile": profile.name,
            "description": profile.description,
            "expected_21d_return": expected_21d,
            "expected_annualized_return": (1 + expected_21d) ** (TRADING_DAYS / HORIZON_DAYS) - 1,
            **m,
            "holdings": int(len(weights)),
        }
        portfolios.append(row)
        h = candidates.set_index("source_file_symbol").loc[weights.index].copy()
        h["weight"] = weights
        h["profile"] = profile.name
        holdings.append(h.reset_index())
    return pd.DataFrame(portfolios), pd.concat(holdings, ignore_index=True)


def detect_anomalies(features: pd.DataFrame) -> pd.DataFrame:
    latest = []
    for sym, g in features.groupby("source_file_symbol"):
        g = g.sort_values("date").tail(252).copy()
        if g.empty:
            continue
        g["abs_return_z"] = (g["ret_1d"].abs() - g["ret_1d"].abs().rolling(60).mean()) / g["ret_1d"].abs().rolling(60).std()
        g["volume_anomaly"] = g["volume_z_20"] > 3
        g["volatility_anomaly"] = g["abs_return_z"] > 3
        g["drawdown_anomaly"] = g["drawdown"] < g["drawdown"].rolling(252).quantile(0.05)
        latest.append(g[g[["volume_anomaly", "volatility_anomaly", "drawdown_anomaly"]].any(axis=1)].tail(5))
    if not latest:
        return pd.DataFrame()
    cols = [
        "date",
        "source_file_symbol",
        "company",
        "industry",
        "close",
        "ret_1d",
        "volume_z_20",
        "abs_return_z",
        "drawdown",
        "volume_anomaly",
        "volatility_anomaly",
        "drawdown_anomaly",
    ]
    return pd.concat(latest, ignore_index=True)[cols].sort_values("date", ascending=False)


def write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def save_artifacts() -> Dict[str, object]:
    ensure_dirs()
    raw = load_stock_data()
    index_context = load_index_context()
    features = engineer_features(raw, index_context)
    model_df = build_model_dataset(features)
    reg, clf, metrics, importance = train_models(model_df)
    latest = latest_stock_snapshot(features, reg, clf)
    portfolios, holdings = construct_portfolios(latest, features)
    anomalies = detect_anomalies(features)

    features.to_pickle(ARTIFACT_DIR / "engineered_features.pkl")
    latest.to_csv(ARTIFACT_DIR / "stock_scores.csv", index=False)
    portfolios.to_csv(ARTIFACT_DIR / "portfolio_profiles.csv", index=False)
    holdings.to_csv(ARTIFACT_DIR / "portfolio_holdings.csv", index=False)
    anomalies.to_csv(ARTIFACT_DIR / "anomalies.csv", index=False)
    importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    write_json(ARTIFACT_DIR / "model_metrics.json", metrics)
    with (ARTIFACT_DIR / "models.pkl").open("wb") as f:
        pickle.dump({"regressor": reg, "classifier": clf, "features": FEATURE_COLUMNS}, f)

    summary = {
        "stocks": int(raw["source_file_symbol"].nunique()),
        "rows": int(len(raw)),
        "date_start": str(raw["date"].min().date()),
        "date_end": str(raw["date"].max().date()),
        "industries": int(raw["industry"].nunique()),
        "model_metrics": metrics,
        "top_stocks": latest.head(10)[
            ["source_file_symbol", "company", "industry", "quality_score", "forecast_return_21d", "sharpe", "volatility"]
        ].to_dict(orient="records"),
    }
    write_json(ARTIFACT_DIR / "summary.json", summary)
    return summary


def load_artifacts() -> Dict[str, object]:
    if not (ARTIFACT_DIR / "stock_scores.csv").exists():
        save_artifacts()
    with (ARTIFACT_DIR / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return {
        "summary": summary,
        "features": pd.read_pickle(ARTIFACT_DIR / "engineered_features.pkl"),
        "scores": pd.read_csv(ARTIFACT_DIR / "stock_scores.csv", parse_dates=["date"]),
        "portfolios": pd.read_csv(ARTIFACT_DIR / "portfolio_profiles.csv"),
        "holdings": pd.read_csv(ARTIFACT_DIR / "portfolio_holdings.csv"),
        "anomalies": pd.read_csv(ARTIFACT_DIR / "anomalies.csv", parse_dates=["date"]),
        "importance": pd.read_csv(ARTIFACT_DIR / "feature_importance.csv"),
    }
