import os
import joblib
import pandas as pd
import numpy as np

from datetime import datetime, date, time, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

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

    # Additional indicators for richer ML features
    try:
        bb = ta.bbands(df["Close"], length=20)
        df["bb_upper"]   = bb["BBU_20_2.0"]
        df["bb_lower"]   = bb["BBL_20_2.0"]
        df["bb_percent"] = bb["BBP_20_2.0"]
    except Exception:
        df["bb_upper"] = np.nan
        df["bb_lower"] = np.nan
        df["bb_percent"] = np.nan

    try:
        adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        df["adx"] = adx["ADX_14"]
        df["dmp"] = adx["DMP_14"]
        df["dmn"] = adx["DMN_14"]
    except Exception:
        df["adx"] = np.nan
        df["dmp"] = np.nan
        df["dmn"] = np.nan

    try:
        df["cci"] = ta.cci(df["High"], df["Low"], df["Close"], length=20)
    except Exception:
        df["cci"] = np.nan

    try:
        df["mfi"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)
    except Exception:
        df["mfi"] = np.nan

    try:
        df["tema"] = ta.tema(df["Close"], length=10)
    except Exception:
        df["tema"] = np.nan

    try:
        df["willr"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)
    except Exception:
        df["willr"] = np.nan

    try:
        psar = ta.psar(df["High"], df["Low"], df["Close"])
        df["psar_long"]  = psar["PSARl_0.02_0.2"]
        df["psar_short"] = psar["PSARs_0.02_0.2"]
    except Exception:
        df["psar_long"] = np.nan
        df["psar_short"] = np.nan

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

def gather_minute_data(ctx, symbols, lookback_days: int = 5) -> dict[str, pd.DataFrame]:
    """
    For each symbol, fetch minute bars from Alpaca day‐by‐day over the last `lookback_days`.
    Prints per‐symbol row‐counts, and only stores those with ≥ 1 row.
    """
    raw_store: dict[str, pd.DataFrame] = {}
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=lookback_days)

    print(f"[gather_minute_data] start={start_dt}, end={end_dt}, symbols={symbols}")
    for sym in symbols:
        bars = None
        try:
            bars = ctx.data_fetcher.get_historical_minute(ctx, sym, start_dt, end_dt)
        except Exception as e:
            bars = None
            print(f"[gather_minute_data] ✗ {sym} → exception {start_dt}→{end_dt}: {e}")

        if bars is None or bars.empty:
            print(f"[gather_minute_data] – {sym} → 0 rows")
        else:
            print(f"[gather_minute_data] ✓ {sym} → {len(bars)} rows")
            raw_store[sym] = bars

    return raw_store

def build_feature_label_df(
    raw_store: dict[str, pd.DataFrame],
    Δ_minutes: int = 30,
    threshold_pct: float = 0.002
) -> pd.DataFrame:
    """
    Build a combined DataFrame of (feature_vector, label) for each minute‐slice.
    If a symbol has fewer than Δ_minutes+1 rows, it is skipped with a notice.
    """
    rows = []
    for sym, raw in raw_store.items():
        if raw.shape[0] < (Δ_minutes + 1):
            print(f"[build_feature_label_df] – skipping {sym}, only {raw.shape[0]} < {Δ_minutes + 1}")
            continue

        feat = prepare_indicators(raw, freq="intraday")
        if feat.empty:
            print(f"[build_feature_label_df] – {sym} indicators empty after dropna")
            continue

        closes = raw["Close"].values
        n = len(feat)
        for i in range(n - Δ_minutes):
            base_price = closes[i]
            future_price = closes[i + Δ_minutes]
            ret_pct = (future_price / base_price) - 1.0
            label = 1 if ret_pct >= threshold_pct else 0

            row = feat.iloc[i].copy().to_dict()
            row["label"] = label
            rows.append(row)

    df_all = pd.DataFrame(rows).dropna()
    return df_all

def retrain_meta_learner(
    ctx,
    symbols,
    lookback_days: int = 5,
    Δ_minutes: int = 30,
    threshold_pct: float = 0.002,
    force: bool = False,
) -> bool:
    """
    1) Skip retrain on weekends or outside market hours.
    2) Call gather_minute_data()—prints per‐symbol counts.
    3) If at least one symbol has ≥ Δ_minutes+1 bars, build features & train.
    4) Otherwise, skip retrain with a clear message.
    """
    now = datetime.now()
    if not force:
        # Skip on weekends
        if now.weekday() >= 5:
            print(
                f"[retrain_meta_learner] Weekend detected (weekday={now.weekday()}). Skipping retrain."
            )
            return False

        # Skip outside market hours (9:30 AM–4:00 PM)
        market_open = time(9, 30)
        market_close = time(16, 0)
        if not (market_open <= now.time() <= market_close):
            print(
                f"[retrain_meta_learner] Outside market hours ({now.time().strftime('%H:%M')}). Skipping retrain."
            )
            return False

    print(f"[{now:%Y-%m-%d %H:%M:%S}] ▶ Starting meta‐learner retraining…")
    raw_store = gather_minute_data(ctx, symbols, lookback_days=lookback_days)
    if not raw_store:
        print("  ⚠️ No symbol returned any minute bars → skipping retrain.")
        return False

    df_all = build_feature_label_df(raw_store, Δ_minutes=Δ_minutes, threshold_pct=threshold_pct)
    if df_all.empty:
        print("  ⚠️ No usable rows after building (Δ, threshold) → skipping retrain.")
        return False

    X = df_all.drop(columns=["label"])
    y = df_all["label"]

    split_idx = int(len(df_all) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    pos_ratio = y_train.mean()
    scoring = "f1" if 0.4 <= pos_ratio <= 0.6 else "roc_auc"
    print(f"  ✔ Using scoring='{scoring}' (positive ratio={pos_ratio:.3f})")

    base_clf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 4, 6, 8, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=20,
        scoring=scoring,
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    clf = search.best_estimator_
    print(f"  ✔ Best params: {search.best_params_}")

    from sklearn.metrics import f1_score, roc_auc_score
    if scoring == "f1":
        val_pred = clf.predict(X_val)
        metric = f1_score(y_val, val_pred)
        print(f"  ✔ Holdout F1 = {metric:.4f}")
    else:
        val_probs = clf.predict_proba(X_val)[:, 1]
        metric = roc_auc_score(y_val, val_probs)
        print(f"  ✔ Holdout AUC = {metric:.4f}")

    importances = pd.Series(clf.feature_importances_, index=X.columns)
    print("  ✔ Top feature importances:")
    print(importances.sort_values(ascending=False).head(10))

    # ─── Feature Selection ────────────────────────────────────────────────────
    selected = importances[importances >= 0.005].index.tolist()
    if len(selected) == 0:
        print("  ⚠️ No features met the importance threshold; using all features.")
        selected = list(X.columns)

    print(f"  ✔ Selected {len(selected)} features with importance >= 0.005")
    print("  ✔ Final feature list:")
    print(selected)

    # Retrain final model using only the selected features
    X_train_sel = X_train[selected]
    X_val_sel = X_val[selected]

    final_clf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        **search.best_params_,
    )
    final_clf.fit(X_train_sel, y_train)

    if scoring == "f1":
        val_pred = final_clf.predict(X_val_sel)
        metric = f1_score(y_val, val_pred)
        print(f"  ✔ Final Holdout F1 = {metric:.4f}")
    else:
        val_probs = final_clf.predict_proba(X_val_sel)[:, 1]
        metric = roc_auc_score(y_val, val_probs)
        print(f"  ✔ Final Holdout AUC = {metric:.4f}")

    joblib.dump(final_clf, MODEL_PATH)
    print(f"  ✔ Saved new meta‐learner to {MODEL_PATH}")
    return True


