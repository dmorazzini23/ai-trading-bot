# AI-AGENT-REF: guard pandas/numpy imports for test environments
import logging

import pandas as pd
from ai_trading.indicators import atr, ema

logger = logging.getLogger(__name__)


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if "close" not in df.columns:
            logger.error("Missing 'close' column for MACD calculation")
            return df
        close = tuple(df["close"].astype(float))
        df["ema12"] = ema(close, 12)
        df["ema26"] = ema(close, 26)
        df["macd"] = df["ema12"] - df["ema26"]
    except Exception as e:
        logger.error("MACD calculation failed: %s", e)
    return df


def compute_macds(df: pd.DataFrame) -> pd.DataFrame:
    df["macds"] = ema(tuple(df["macd"].astype(float)), 9)
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df["atr"] = atr(df["high"], df["low"], df["close"], period)
    return df


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    q = df["volume"]
    p = (df["high"] + df["low"] + df["close"]) / 3
    df["cum_vol"] = q.cumsum()
    df["cum_pv"] = (p * q).cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"]
    return df


def ensure_columns(
    df: pd.DataFrame, columns: list[str], symbol: str | None = None
) -> pd.DataFrame:
    """Ensure required indicator columns exist, filling with 0.0 if missing."""
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
            if symbol:
                logger.warning(
                    f"Column {col} was missing for {symbol}, filled with 0.0."
                )
            else:
                logger.warning(f"Column {col} was missing, filled with 0.0.")
    return df


def build_features_pipeline(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    try:
        logger.debug(
            f"Starting feature pipeline for {symbol}. Initial shape: {df.shape}"
        )
        df = compute_macd(df)
        logger.debug(f"[{symbol}] Post MACD: last closes:\n{df[['close']].tail(5)}")
        df = compute_atr(df)
        logger.debug(f"[{symbol}] Post ATR: last closes:\n{df[['close']].tail(5)}")
        df = compute_vwap(df)
        logger.debug(f"[{symbol}] Post VWAP: last closes:\n{df[['close']].tail(5)}")
        df = compute_macds(df)
        logger.debug(f"[{symbol}] Post MACDS: last closes:\n{df[['close']].tail(5)}")
        required_cols = ["macd", "atr", "vwap", "macds"]
        df = ensure_columns(df, required_cols, symbol)
        logger.debug(
            f"Feature pipeline complete for {symbol}. Last rows:\n{df.tail(3)}"
        )
    except Exception as e:
        logger.exception(f"Feature pipeline failed for {symbol}: {e}")
    return df
