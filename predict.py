import argparse
import os
import pandas as pd
import joblib
from retrain import prepare_indicators


def detect_regime(df: pd.DataFrame) -> str:
    if df is None or df.empty or "Close" not in df:
        return "chop"
    close = df["Close"].astype(float)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    if sma50.iloc[-1] > sma200.iloc[-1]:
        return "bull"
    if sma50.iloc[-1] < sma200.iloc[-1]:
        return "bear"
    return "chop"

MODEL_FILES = {
    "bull": "model_bull.pkl",
    "bear": "model_bear.pkl",
    "chop": "model_chop.pkl",
}


def load_model(regime: str):
    path = MODEL_FILES.get(regime)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model for regime '{regime}' not found: {path}")
    return joblib.load(path)


def predict(csv_path: str, freq: str = "intraday"):
    df = pd.read_csv(csv_path)
    feat = prepare_indicators(df, freq=freq)
    regime = detect_regime(df)
    if isinstance(regime, pd.Series):
        regime = regime.iloc[-1]
    model = load_model(regime)
    X = feat[model.feature_names_in_].iloc[-1].values.reshape(1, -1)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][pred]
    print(f"Regime: {regime}, Prediction: {pred}, Probability: {proba:.4f}")
    return pred, proba


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict trade signal")
    parser.add_argument("csv", help="CSV file with OHLCV data")
    parser.add_argument("--freq", choices=["daily", "intraday"], default="intraday")
    args = parser.parse_args()
    predict(args.csv, args.freq)
