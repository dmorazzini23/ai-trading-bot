# retrain.py

import joblib
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from bot import prepare_indicators   # or wherever that function lives
# from bot import CONF_THRESHOLD, BUY_THRESHOLD  # if needed elsewhere
# from bot import ctx  # if you keep a global context; otherwise pass in ctx

def gather_minute_data(ctx, symbols, lookback_days=30):
    """
    Fetch raw minute bars for each symbol over the past `lookback_days` calendar days.
    Returns a dict: { symbol: DataFrame_of_minute_bars }.
    """
    end_dt = datetime.now(ctx.TIMECONVERTION_ZONE).date()
    start_dt = end_dt - timedelta(days=lookback_days)
    raw_store = {}
    for sym in symbols:
        raw = ctx.data_fetcher.get_historical_minute(
            ctx, sym, start=start_dt, end=end_dt
        )
        if raw is None or raw.empty:
            continue
        raw_store[sym] = raw
    return raw_store

def build_feature_label_df(raw_store, Δ_minutes=30, threshold_pct=0.002):
    """
    For each minute‐bar in each raw_store[sym], compute indicators (feat_df)
    and a binary label: 1 if price in Δ_minutes ahead ≥ threshold_pct above current.
    Returns a single DataFrame with columns [rsi, macd, atr,…, label].
    """
    examples = []
    for sym, raw in raw_store.items():
        # raw must have a DateTimeIndex and a "Close" column
        feat = prepare_indicators(raw, freq="intraday")
        if feat.empty:
            continue

        closes = raw["Close"].values
        n = len(feat)
        for i in range(n - Δ_minutes):
            base_price = closes[i]
            future_price = closes[i + Δ_minutes]
            label = 1 if (future_price / base_price - 1) >= threshold_pct else 0

            row = feat.iloc[i].copy().to_dict()
            row["label"] = label
            examples.append(row)

    df_all = pd.DataFrame(examples).dropna()
    return df_all

def retrain_meta_learner(ctx, symbols, lookback_days=30, Δ_minutes=30, threshold_pct=0.002):
    """
    1. Gather minute data for each symbol over lookback_days
    2. Build features+labels
    3. Train a fresh RandomForestClassifier
    4. Persist to disk at ctx.MODEL_PATH (e.g. "models/meta_learner.pkl")
    """
    print(f"[{datetime.now()}] ▶ Starting meta‐learner retraining over last {lookback_days} days…")

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

    # Stratified split to preserve 0/1 ratios
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

    # Optional: print out validation AUC so you can log it
    try:
        from sklearn.metrics import roc_auc_score
        val_probs = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        print(f"  ✔ Validation AUC = {auc:.4f}")
        print(f"  ✔ Positive‐prediction rate = {(val_probs >= 0.5).mean():.3%}")
    except Exception:
        pass

    # Save the model to disk
    model_path = ctx.MODEL_PATH  # e.g. "models/meta_learner.pkl"
    joblib.dump(clf, model_path)
    print(f"  ✔ Saved new meta‐learner to {model_path}")
    return True
