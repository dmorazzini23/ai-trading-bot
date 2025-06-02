import os
import joblib
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas_ta as ta

# ─── COPY&PASTE of prepare_indicators (unchanged) ─────────────────
def prepare_indicators(df: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    df = df.copy()
    if df.index.name:
        df = df.reset_index().rename(columns={df.index.name: "Date"})
    else:
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Calculate basic TA indicators
    df["vwap"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    df["rsi"] = ta.rsi(df["Close"], length=14)
    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    try:
        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        df["macd"] = macd["MACD_12_26_9"]
        df["macds"] = macd["MACDs_12_26_9"]
    except Exception:
        df["macd"] = np.nan
        df["macds"] = np.nan

    try:
        ich = ta.ichimoku(high=df["High"], low=df["Low"], close=df["Close"])
        conv = ich[0] if isinstance(ich, tuple) else ich.iloc[:, 0]
        base = ich[1] if isinstance(ich, tuple) else ich.iloc[:, 1]
        df["ichimoku_conv"] = conv.iloc[:, 0] if hasattr(conv, "iloc") else conv
        df["ichimoku_base"] = base.iloc[:, 0] if hasattr(base, "iloc") else base
    except Exception:
        df["ichimoku_conv"] = np.nan
        df["ichimoku_base"] = np.nan

    try:
        st = ta.stochrsi(df["Close"])
        df["stochrsi"] = st["STOCHRSIk_14_14_3_3"]
    except Exception:
        df["stochrsi"] = np.nan

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    required = ["vwap", "rsi", "atr", "macd", "macds", "ichimoku_conv", "ichimoku_base", "stochrsi"]
    if freq == "daily":
        df["sma_50"] = ta.sma(df["Close"], length=50)
        df["sma_200"] = ta.sma(df["Close"], length=200)
        required += ["sma_50", "sma_200"]
        df.dropna(subset=required, how="any", inplace=True)
    else:  # intraday
        df.dropna(subset=required, how="all", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df
# ──────────────────────────────────────────────────────────────────────────────

# Read the same MODEL_PATH from the environment (just like bot.py does):
MODEL_PATH = os.getenv("MODEL_PATH", "meta_model.pkl")


def gather_minute_data(ctx, symbols, lookback_days=5):
    """
    1) Try to fetch up to the last 5 calendar days of minute bars via get_minute_df().
    2) If that comes back empty for a given symbol, do a single-day Alpaca fallback
       (yesterday’s date) via get_historical_minute().
    We then trim to `lookback_days` and return whatever is non-empty.
    """
    raw_store: dict[str, pd.DataFrame] = {}
    cutoff_ts = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)

    for sym in symbols:
        # ── PRIMARY: pull up to 5 days in one shot ──────────────────────────
        raw = ctx.data_fetcher.get_minute_df(ctx, sym)
        if raw is not None and not raw.empty:
            # Keep only the last `lookback_days` worth of rows:
            recent = raw.loc[raw.index >= cutoff_ts]
            if not recent.empty:
                raw_store[sym] = recent
                continue  # next symbol

        # ── FALLBACK: try exactly “yesterday” via get_historical_minute(…) ──
        yesterday = (date.today() - timedelta(days=1))
        bars1d = ctx.data_fetcher.get_historical_minute(sym, yesterday, yesterday)
        if bars1d is None or bars1d.empty:
            # no minute bars at all
            continue

        # Keep only rows within lookback_days (this is just 1 day, so OK)
        bars1d = bars1d.loc[bars1d.index >= cutoff_ts]
        if not bars1d.empty:
            raw_store[sym] = bars1d

    return raw_store


def build_feature_label_df(raw_store, Δ_minutes=30, threshold_pct=0.002):
    """
    Build a single DataFrame of (feature_vector, label) for each minute‐slice.
    Label = 1 if price change over next Δ_minutes ≥ threshold_pct,
            0 otherwise.
    """
    rows = []
    for sym, raw in raw_store.items():
        feat = prepare_indicators(raw, freq="intraday")
        if feat.empty:
            continue

        closes = raw["Close"].values
        n = len(feat)
        for i in range(n - Δ_minutes):
            base_price = closes[i]
            future_price = closes[i + Δ_minutes]
            ret_pct = future_price / base_price - 1.0
            label = 1 if ret_pct >= threshold_pct else 0

            row = feat.iloc[i].copy().to_dict()
            row["label"] = label
            rows.append(row)

    df_all = pd.DataFrame(rows).dropna()
    return df_all


def retrain_meta_learner(ctx, symbols, lookback_days=5, Δ_minutes=30, threshold_pct=0.002):
    """
    1. Gather minute data for each symbol over the last `lookback_days` (via get_minute_df).
    2. Build features & labels using a Δ‐minute horizon.
    3. Train a RandomForestClassifier on (X, y).
    4. Save the new model to MODEL_PATH.
    Returns True if training succeeded, False otherwise.
    """
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ▶ Starting meta‐learner retraining...")

    raw_store = gather_minute_data(ctx, symbols, lookback_days=lookback_days)
    if not raw_store:
        print("  ⚠️ No minute data fetched; skipping retrain.")
        return False

    df_all = build_feature_label_df(raw_store, Δ_minutes=Δ_minutes, threshold_pct=threshold_pct)
    if df_all.empty:
        print("  ⚠️ Feature/label DataFrame is empty; skipping retrain.")
        return False

    X = df_all.drop(columns=["label"])
    y = df_all["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # (Optional) report validation AUC:
    try:
        from sklearn.metrics import roc_auc_score
        val_probs = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        print(f"  ✔ Validation AUC = {auc:.4f}")
    except Exception:
        pass

    joblib.dump(clf, MODEL_PATH)
    print(f"  ✔ Saved new meta‐learner to {MODEL_PATH}")
    return True
