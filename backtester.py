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


def execute_backtest():
    """Placeholder backtest executor returning empty metrics."""
    # AI-AGENT-REF: simplified backtest stub for grid tuning
    return {}


def run_backtest(
    volume_spike_threshold: float,
    ml_confidence_threshold: float,
    pyramid_levels: dict,
) -> dict:
    """Run backtest with overridable parameters."""
    from config import set_runtime_config

    set_runtime_config(volume_spike_threshold, ml_confidence_threshold, pyramid_levels)
    results = execute_backtest()
    return results

