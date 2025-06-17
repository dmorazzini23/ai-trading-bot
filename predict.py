import argparse
import json
import logging
import os
import warnings

import joblib
import pandas as pd
import requests

import config
from retrain import prepare_indicators

config.reload_env()
from config import NEWS_API_KEY

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INACTIVE_FEATURES_FILE = os.path.join(BASE_DIR, "inactive_features.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def fetch_sentiment(symbol: str) -> float:
    if not NEWS_API_KEY:
        return 0.0
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&pageSize=5&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        arts = resp.json().get("articles", [])
        if not arts:
            return 0.0
        score = sum(1 for a in arts if "positive" in (a.get("title") or "").lower()) / len(arts)
        return float(score)
    except Exception as e:
        logger.error("fetch_sentiment failed for %s: %s", symbol, e)
        return 0.0


def detect_regime(df: pd.DataFrame) -> str:
    if df is None or df.empty or "close" not in df:
        return "chop"
    close = df["close"].astype(float)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    if sma50.iloc[-1] > sma200.iloc[-1]:
        return "bull"
    if sma50.iloc[-1] < sma200.iloc[-1]:
        return "bear"
    return "chop"


MODEL_FILES = {
    "bull": os.path.join(MODELS_DIR, "model_bull.pkl"),
    "bear": os.path.join(MODELS_DIR, "model_bear.pkl"),
    "chop": os.path.join(MODELS_DIR, "model_chop.pkl"),
}


def load_model(regime: str):
    path = MODEL_FILES.get(regime)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model for regime '{regime}' not found: {path}")
    return joblib.load(path)


def predict(csv_path: str, freq: str = "intraday"):
    df = pd.read_csv(csv_path)
    feat = prepare_indicators(df, freq=freq)
    if os.path.exists(INACTIVE_FEATURES_FILE):
        try:
            inactive = set(json.load(open(INACTIVE_FEATURES_FILE)))
            feat = feat.drop(columns=[c for c in inactive if c in feat.columns], errors="ignore")
        except Exception as e:
            logger.warning("Failed loading inactive features: %s", e)
    symbol = os.path.splitext(os.path.basename(csv_path))[0]
    feat["sentiment"] = fetch_sentiment(symbol)
    regime = detect_regime(df)
    if isinstance(regime, pd.Series):
        regime = regime.iloc[-1]
    model = load_model(regime)
    expected_features = list(model.feature_names_in_)
    missing = set(expected_features) - set(feat.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = feat[expected_features].iloc[-1:].astype({col: "float64" for col in expected_features})
    try:
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][pred]
    except (ValueError, TypeError) as e:
        logger.error(f"Prediction failed for {symbol}: {e}")
        return None, None
    print(f"Regime: {regime}, Prediction: {pred}, Probability: {proba:.4f}")
    return pred, proba


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict trade signal")
    parser.add_argument("csv", help="CSV file with OHLCV data")
    parser.add_argument("--freq", choices=["daily", "intraday"], default="intraday")
    args = parser.parse_args()
    predict(args.csv, args.freq)
