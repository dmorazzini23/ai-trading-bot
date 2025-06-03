import os
import joblib
import pandas as pd
import numpy as np

from datetime import datetime, date, timedelta
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

    required = [
        "vwap", "rsi", "atr", "macd", "macds",
        "ichimoku_conv", "ichimoku_base", "stochrsi"
    ]
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

MODEL_PATH = os.getenv("MODEL_PATH", "meta_model.pkl")


def gather_minute_data(ctx, symbols, lookback_days: int = 10):
    """
    1) Try to grab minute‐cache via get_minute_df(ctx, sym).
    2) If that is empty, fetch day‐by‐day via get_historical_minute(ctx, sym,...).
    Prints one line per symbol so you know exactly what succeeded/failed.
    """
    raw_store: dict[str, pd.DataFrame] = {}
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=lookback_days)
    any_symbol_had_data = False

    for sym in symbols:
        bars = None

        # 1) first attempt: get whatever intraday minutes are already cached
        try:
            bars = ctx.data_fetcher.get_minute_df(ctx, sym)
        except Exception:
            bars = None

        # 2) if no intraday cache, fetch day-by-day historical
        if bars is None or bars.empty:
            try:
                bars = ctx.data_fetcher.get_historical_minute(ctx, sym, start_dt, end_dt)
            except Exception:
                bars = None

        if bars is None or bars.empty:
            print(f"[gather_minute_data] → {sym} returned NO minute rows ({start_dt}→{end_dt})")
            continue

        # At this point we have at least some rows for 'sym'
        print(f"[gather_minute_data] → {sym} fetched {len(bars)} rows of minute data")
        raw_store[sym] = bars
        any_symbol_had_data = True

    # If absolutely no symbol returned bars, raw_store will be empty
    return raw_store


def build_feature_label_df(raw_store, Δ_minutes=30, threshold_pct=0.002):
    """
    Build a single DataFrame of (feature_vector, label) for each minute-slice.
    Label = 1 if price change over next Δ_minutes ≥ threshold_pct, else 0.
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


def retrain_meta_learner(ctx, symbols, lookback_days=10, Δ_minutes=30, threshold_pct=0.002):
    """
    1. Gather minute data for each symbol over the last `lookback_days`.
       - tries get_minute_df() first, then get_historical_minute()
    2. If ANY symbol did produce minute rows, build features/labels and train.
    3. If ZERO symbols produced minute rows, fallback to downloading ONE‐DAY Daily-OHLC and train on daily features.
    4. Save new model to MODEL_PATH.
    Returns True if training succeeded; False otherwise.
    """
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ▶ Starting meta-learner retraining…")

    raw_store = gather_minute_data(ctx, symbols, lookback_days=lookback_days)
    if raw_store:
        # Build minute‐based features/labels
        df_all = build_feature_label_df(raw_store, Δ_minutes=Δ_minutes, threshold_pct=threshold_pct)
        if df_all.empty:
            print("  ⚠️ Feature/label DataFrame (intraday) is empty; skipping minute-train.")
        else:
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
            try:
                from sklearn.metrics import roc_auc_score
                val_probs = clf.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, val_probs)
                print(f"  ✔ Validation AUC (intraday) = {auc:.4f}")
            except Exception:
                pass

            joblib.dump(clf, MODEL_PATH)
            print(f"  ✔ Saved new intraday meta-learner to {MODEL_PATH}")
            return True

        # If we reach here, intraday features/labels existed but df_all was empty
        # fall through below to daily fallback


    # If raw_store was empty OR intraday df_all was empty, do a daily fallback:
    print("  ‼️ No valid intraday training data found. Falling back to a 1-day daily dataset…")
    # We will grab “yesterday’s” daily OHLC for each symbol,
    # compute the same indicators at daily frequency, and train.

    # 1) Build a tiny DataFrame for each symbol’s single daily OHLC:
    daily_rows = []
    from_date = date.today() - timedelta(days=1)
    to_date   = date.today() - timedelta(days=1)
    for sym in symbols:
        try:
            df_daily = ctx.data_fetcher.get_daily_df(ctx, sym)
        except Exception:
            df_daily = None

        if df_daily is None or df_daily.empty or from_date not in df_daily.index.date:
            print(f"[daily_fallback] → {sym} missing daily row for {from_date}")
            continue

        # Extract just that one day’s OHLCV
        one_day_row = df_daily.loc[df_daily.index.date == from_date].iloc[0]
        daily_rows.append({
            "Open": one_day_row["open"],
            "High": one_day_row["high"],
            "Low":  one_day_row["low"],
            "Close": one_day_row["close"],
            "Volume": one_day_row["volume"],
            "symbol": sym
        })

    if not daily_rows:
        print("  ❌ Even daily fallback failed; skipping retrain entirely.")
        return False

    # Build a single DataFrame (each row is one symbol’s daily OHLC)
    df_daily_all = pd.DataFrame(daily_rows)
    # Compute indicators on a 1-row DataFrame by re-indexing as if it were a 1-day series
    # (we’ll just treat each symbol independently for training.)
    feature_rows = []
    for row in daily_rows:
        temp_df = pd.DataFrame([row], index=[from_date])
        temp_feat = prepare_indicators(
            temp_df.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume":"volume"
            }),
            freq="daily"
        )
        if temp_feat.empty:
            continue
        feat_dict = temp_feat.iloc[0].to_dict()
        feat_dict["label"] = 0  # label=0 (no future minute data to compute a “future Δ”)
        feature_rows.append(feat_dict)

    if not feature_rows:
        print("  ❌ Daily-fallback indicators empty; skipping retrain entirely.")
        return False

    df_features = pd.DataFrame(feature_rows).dropna()
    X = df_features.drop(columns=["label"])
    y = df_features["label"].astype(int)

    if X.empty:
        print("  ❌ Daily fallback X is empty; skipping retrain entirely.")
        return False

    # Just train/test‐split on these few rows:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2 if len(y) > 1 else 0.5,
        random_state=42, stratify=y if len(set(y)) > 1 else None
    )
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("  ✔ Trained fallback daily-model on", len(X_train), "rows.")

    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y_val)) > 1:
            val_probs = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_probs)
            print(f"  ✔ Validation AUC (daily fallback) = {auc:.4f}")
    except Exception:
        pass

    joblib.dump(clf, MODEL_PATH)
    print(f"  ✔ Saved new daily-fallback meta-learner to {MODEL_PATH}")
    return True


