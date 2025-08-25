"""
Technical indicator calculations for AI trading platform.

This module provides compute functions for MACD, ATR, VWAP and other
technical indicators used in trading strategies.

Moved from root features.py for package-safe imports.
"""
from __future__ import annotations
import logging
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.indicators import atr, ema
logger = logging.getLogger(__name__)

# Lazy pandas proxy
pd = load_pandas()

def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD indicator."""
    try:
        if 'close' not in df.columns:
            logger.error("Missing 'close' column for MACD calculation")
            return df
        close = tuple(df['close'].astype(float))
        df['ema12'] = ema(close, 12)
        df['ema26'] = ema(close, 26)
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = ema(tuple(df['macd']), 9)
        df['histogram'] = df['macd'] - df['signal']
        return df
    except (KeyError, ValueError, TypeError):
        logger.error('MACD computation failed', exc_info=True)
        return df

def compute_atr(df: pd.DataFrame, period: int=14) -> pd.DataFrame:
    """Compute Average True Range (ATR)."""
    try:
        if not all((col in df.columns for col in ['high', 'low', 'close'])):
            logger.error('Missing required columns for ATR calculation')
            return df
        high = df['high']
        low = df['low']
        close = df['close']
        df['atr'] = atr(high, low, close, period)
        return df
    except (KeyError, ValueError, TypeError):
        logger.error('ATR computation failed', exc_info=True)
        return df

def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Volume Weighted Average Price (VWAP)."""
    try:
        if not all((col in df.columns for col in ['high', 'low', 'close', 'volume'])):
            logger.error('Missing required columns for VWAP calculation')
            return df
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return df
    except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        logger.error('VWAP computation failed', exc_info=True)
        return df

def compute_macds(df: pd.DataFrame) -> pd.DataFrame:
    """Map MACD signal to 'macds' column if available."""
    try:
        if 'macds' not in df.columns:
            if 'signal' in df.columns:
                df['macds'] = df['signal']
            else:
                logger.warning("Missing column 'macds' and 'signal', filling with zeros")
                df['macds'] = 0.0
        return df
    except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        logger.error('MACDS computation failed', exc_info=True)
        return df

def ensure_columns(df: pd.DataFrame, required: list[str] | None=None, symbol: str | None=None) -> pd.DataFrame:
    """Ensure DataFrame has required columns for calculations."""
    required = required or ['open', 'high', 'low', 'close', 'volume']
    try:
        for col in required:
            if col not in df.columns:
                if symbol:
                    logger.warning(f"Missing column '{col}' for {symbol}, filling with zeros")
                else:
                    logger.warning(f"Missing column '{col}', filling with zeros")
                df[col] = 0.0
        return df
    except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        logger.error('Column validation failed', exc_info=True)
        return df
__all__ = ['compute_macd', 'compute_atr', 'compute_vwap', 'compute_macds', 'ensure_columns']
