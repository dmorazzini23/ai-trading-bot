import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    return df


def compute_macds(df: pd.DataFrame) -> pd.DataFrame:
    df['macds'] = df['macd'].ewm(span=9, adjust=False).mean()
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=period).mean()
    return df


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    q = df['volume']
    p = (df['high'] + df['low'] + df['close']) / 3
    df['cum_vol'] = q.cumsum()
    df['cum_pv'] = (p * q).cumsum()
    df['vwap'] = df['cum_pv'] / df['cum_vol']
    return df


def ensure_columns(df: pd.DataFrame, columns: list[str], symbol: str | None = None) -> pd.DataFrame:
    """Ensure required indicator columns exist, filling with 0.0 if missing."""
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
            if symbol:
                logger.warning(f"Column {col} was missing for {symbol}, filled with 0.0.")
            else:
                logger.warning(f"Column {col} was missing, filled with 0.0.")
    return df


def build_features_pipeline(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    try:
        logger.debug(f"Starting feature pipeline for {symbol}. Initial shape: {df.shape}")
        df = compute_macd(df)
        df = compute_atr(df)
        df = compute_vwap(df)
        df = compute_macds(df)
        required_cols = ['macd', 'atr', 'vwap', 'macds']
        df = ensure_columns(df, required_cols, symbol)
        logger.debug(f"Feature pipeline complete for {symbol}. Last rows:\n{df.tail(3)}")
    except Exception as e:
        logger.exception(f"Feature pipeline failed for {symbol}: {e}")
    return df
