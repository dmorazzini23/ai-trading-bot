import numpy as np
import pandas as pd


def load_data(file: str) -> pd.DataFrame:
    """Load a CSV with OHLCV data, filling missing columns."""
    df = pd.read_csv(file)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            print(f"Warning: {file} missing {col}, filling with last value or ffill.")
            df[col] = np.nan
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    return df
