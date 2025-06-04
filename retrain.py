import os
import joblib
import pandas as pd
import numpy as np

from datetime import datetime, date, time, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import pandas_ta as ta
from bot import detect_regime  # Import regime detection

# Regime model file map
MODEL_FILES = {
    "bull": {"rf": "model_bull_rf.pkl", "xgb": "model_bull_xgb.pkl", "lgb": "model_bull_lgb.pkl"},
    "bear": {"rf": "model_bear_rf.pkl", "xgb": "model_bear_xgb.pkl", "lgb": "model_bear_lgb.pkl"},
    "chop": {"rf": "model_chop_rf.pkl", "xgb": "model_chop_xgb.pkl", "lgb": "model_chop_lgb.pkl"},
}

# Full indicator preparation
def prepare_indicators(df: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    df = df.copy()
    df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    df["vwap"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    df["rsi"] = ta.rsi(df["Close"], length=14)
    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    try:
        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        df["macd"] = macd["MACD_12_26_9"]
        df["macds"] = macd["MACDs_12_26_9"]
    except:
        df["macd"], df["macds"] = np.nan, np.nan

    try:
        bb = ta.bbands(df["Close"], length=20)
        df["bb_upper"], df["bb_lower"], df["bb_percent"] = bb["BBU_20_2.0"], bb["BBL_20_2.0"], bb["BBP_20_2.0"]
    except:
        df["bb_upper"], df["bb_lower"], df["bb_percent"] = np.nan, np.nan, np.nan

    try:
        adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        df["adx"], df["dmp"], df["dmn"] = adx["ADX_14"], adx["DMP_14"], adx["DMN_14"]
    except:
        df["adx"], df["dmp"], df["dmn"] = np.nan, np.nan, np.nan

    try:
        df["cci"] = ta.cci(df["High"], df["Low"], df["Close"], length=20)
    except:
        df["cci"] = np.nan

    try:
        df["mfi"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)
    except:
        df["mfi"] = np.nan

    try:
        df["tema"] = ta.tema(df["Close"], length=10)
    except:
        df["tema"] = np.nan

    try:
        df["willr"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)
    except:
        df["willr"] = np.nan

    try:
        psar = ta.psar(df["High"], df["Low"], df["Close"])
        df["psar_long"], df["psar_short"] = psar["PSARl_0.02_0.2"], psar["PSARs_0.02_0.2"]
    except:
        df["psar_long"], df["psar_short"] = np.nan, np.nan

    try:
        ich = ta.ichimoku(high=df["High"], low=df["Low"], close=df["Close"])
        conv = ich[0] if isinstance(ich, tuple) else ich.iloc[:, 0]
        base = ich[1] if isinstance(ich, tuple) else ich.iloc[:, 1]
        df["ichimoku_conv"], df["ichimoku_base"] = conv.iloc[:, 0], base.iloc[:, 0]
    except:
        df["ichimoku_conv"], df["ichimoku_base"] = np.nan, np.nan

    try:
        st = ta.stochrsi(df["Close"])
        df["stochrsi"] = st["STOCHRSIk_14_14_3_3"]
    except:
        df["stochrsi"] = np.nan

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    required = ["vwap", "rsi", "atr", "macd", "macds", "ichimoku_conv", "ichimoku_base", "stochrsi"]
    if freq == "daily":
        df["sma_50"] = ta.sma(df["Close"], length=50)
        df["sma_200"] = ta.sma(df["Close"], length=200)
        required += ["sma_50", "sma_200"]
        df.dropna(subset=required, inplace=True)
    else:
        df.dropna(subset=required, how="all", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df

# Download minute data per ticker
def gather_minute_data(ctx, symbols, lookback_days=5) -> dict[str, pd.DataFrame]:
    raw_store = {}
    end_dt, start_dt = date.today(), date.today() - timedelta(days=lookback_days)
    for sym in symbols:
        try:
            bars = ctx.data_fetcher.get_historical_minute(ctx, sym, start_dt, end_dt)
            if bars is not None and not bars.empty:
                raw_store[sym] = bars
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
    return raw_store

# Build training dataset with regime labels
def build_feature_label_df(raw_store, Δ_minutes=30, threshold_pct=0.002):
    rows = []
    for sym, raw in raw_store.items():
        if raw.shape[0] < (Δ_minutes + 1): continue
        feat = prepare_indicators(raw, freq="intraday")
        regime_series = detect_regime(raw)
        if isinstance(regime_series, pd.Series):
            regimes = regime_series.reset_index(drop=True)
        else:
            regimes = pd.Series([regime_series] * len(feat))

        closes = raw["Close"].values
        for i in range(len(feat) - Δ_minutes):
            buy_fill = closes[i] * (1 + 0.0005)
            sell_fill = closes[i + Δ_minutes] * (1 - 0.0005)
            ret_pct = (sell_fill / buy_fill) - 1.0
            label = 1 if ret_pct >= threshold_pct else 0

            row = feat.iloc[i].copy().to_dict()
            row["regime"] = regimes.iloc[i]
            row["label"] = label
            rows.append(row)

    df_all = pd.DataFrame(rows).dropna()
    return df_all

# Main retraining logic
def retrain_meta_learner(ctx, symbols, lookback_days=5, Δ_minutes=30, threshold_pct=0.002, force=False):
    now = datetime.now()
    if not force and (now.weekday() >= 5 or not time(9, 30) <= now.time() <= time(16, 0)):
        print("Skipping retrain (weekend or outside hours)")
        return False

    raw_store = gather_minute_data(ctx, symbols, lookback_days)
    if not raw_store: return False

    df_all = build_feature_label_df(raw_store, Δ_minutes, threshold_pct)
    if df_all.empty: return False

    trained_any = False
    for regime, subset in df_all.groupby("regime"):
        if regime not in MODEL_FILES or subset.empty:
            continue

        X = subset.drop(columns=["label", "regime"])
        y = subset["label"]
        split_idx = int(len(subset) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        pos_ratio = y_train.mean()
        scoring = "f1" if 0.4 <= pos_ratio <= 0.6 else "roc_auc"
        print(f"Training {regime} regime model...")

        models = {
            "rf": (RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1), {
                "n_estimators": [100, 200, 300], "max_depth": [None, 4, 8]
            }),
            "xgb": (XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42, n_jobs=-1), {
                "n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.01, 0.05]
            }),
            "lgb": (LGBMClassifier(objective="binary", random_state=42, n_jobs=-1), {
                "n_estimators": [100, 200], "num_leaves": [31, 63], "max_depth": [-1, 8], "learning_rate": [0.01, 0.05]
            })
        }

        tscv = TimeSeriesSplit(n_splits=3)
        for model_key, (base_model, param_grid) in models.items():
            search = RandomizedSearchCV(base_model, param_grid, n_iter=10, scoring=scoring, cv=tscv, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            from sklearn.metrics import f1_score, roc_auc_score
            if scoring == "f1":
                score = f1_score(y_val, best_model.predict(X_val))
            else:
                score = roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])

            print(f"{regime} {model_key} validation {scoring.upper()}: {score:.4f}")
            model_path = MODEL_FILES[regime][model_key]
            joblib.dump(best_model, model_path)
            print(f"Saved {model_path}")

        trained_any = True

    return trained_any
