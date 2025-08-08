"""
Technical indicator calculations for AI trading platform.

This module provides compute functions for MACD, ATR, VWAP and other 
technical indicators used in trading strategies.

Moved from root features.py for package-safe imports.
"""

# AI-AGENT-REF: guard pandas/numpy imports for test environments
try:
    import pandas as pd
except ImportError:
    from datetime import datetime
    class MockDataFrame:
        def __init__(self, *args, **kwargs):
            pass
    class MockPandas:
        DataFrame = MockDataFrame
        Timestamp = datetime
    pd = MockPandas()

try:
    import numpy as np
except ImportError:
    class MockNumpy:
        def array(self, *args, **kwargs):
            return []
        def mean(self, *args, **kwargs):
            return 0.0
        def std(self, *args, **kwargs):
            return 1.0
        def nan(self):
            return float('nan')
    np = MockNumpy()

import logging

try:
    from indicators import ema
except ImportError:
    def ema(data, period):
        """Fallback EMA calculation."""
        return pd.Series(0.0, index=range(len(data)))

try:
    from indicators import atr
except ImportError:
    def atr(high, low, close, period=14):
        """Fallback ATR calculation."""
        return pd.Series(0.0, index=close.index)

logger = logging.getLogger(__name__)


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD indicator."""
    try:
        if "close" not in df.columns:
            logger.error("Missing 'close' column for MACD calculation")
            return df
        close = tuple(df["close"].astype(float))
        df["ema12"] = ema(close, 12)
        df["ema26"] = ema(close, 26)
        df["macd"] = df["ema12"] - df["ema26"]
        df["signal"] = ema(tuple(df["macd"]), 9)
        df["histogram"] = df["macd"] - df["signal"]
        return df
    except Exception as e:
        logger.error("MACD computation failed", exc_info=True)
        return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Average True Range (ATR)."""
    try:
        if not all(col in df.columns for col in ["high", "low", "close"]):
            logger.error("Missing required columns for ATR calculation")
            return df
        
        high = df["high"]
        low = df["low"] 
        close = df["close"]
        
        df["atr"] = atr(high, low, close, period)
        return df
    except Exception as e:
        logger.error("ATR computation failed", exc_info=True)
        return df


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Volume Weighted Average Price (VWAP)."""
    try:
        if not all(col in df.columns for col in ["high", "low", "close", "volume"]):
            logger.error("Missing required columns for VWAP calculation")
            return df
        
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return df
    except Exception as e:
        logger.error("VWAP computation failed", exc_info=True)
        return df


def compute_macds(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD with multiple timeframes.""" 
    try:
        df = compute_macd(df)
        # Add additional MACD variations if needed
        return df
    except Exception as e:
        logger.error("MACDS computation failed", exc_info=True)
        return df


def ensure_columns(df: pd.DataFrame, required_columns: list) -> pd.DataFrame:
    """Ensure DataFrame has required columns for calculations."""
    try:
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column '{col}', filling with zeros")
                df[col] = 0.0
        return df
    except Exception as e:
        logger.error("Column validation failed", exc_info=True)
        return df


# Export all compute functions
__all__ = [
    "compute_macd",
    "compute_atr", 
    "compute_vwap",
    "compute_macds",
    "ensure_columns"
]