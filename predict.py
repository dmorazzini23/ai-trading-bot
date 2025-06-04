import argparse
import os
import joblib
import pandas as pd

from bot import detect_regime  # type: ignore
from retrain import prepare_indicators, MODEL_FILES


def load_model(regime: str):
    path = MODEL_FILES.get(regime)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Model file for regime '{regime}' not found")
    return joblib.load(path)


def predict(csv_file: str):
    raw = pd.read_csv(csv_file)
    feat = prepare_indicators(raw, freq="intraday")
    regime = detect_regime(raw)
    if isinstance(regime, pd.Series):
        regime = regime.iloc[-1]
    model = load_model(regime)
    X = feat[model.feature_names_in_]
    preds = model.predict(X)
    return preds, regime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict trade signals")
    parser.add_argument("csv", help="CSV of OHLCV data")
    args = parser.parse_args()

    predictions, regime = predict(args.csv)
    print(f"Regime: {regime}")
    print(predictions)


