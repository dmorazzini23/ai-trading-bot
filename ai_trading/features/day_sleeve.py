"""Canonical runtime feature construction for the five-minute day model."""

from __future__ import annotations

from typing import Any

import numpy as np

from ai_trading.features.indicators import (
    compute_atr,
    compute_macd,
    compute_macds,
    compute_sma,
    compute_vwap,
)
from ai_trading.indicators import rsi as rsi_indicator
from ai_trading.models.contracts import DAY_SLEEVE_ML_FEATURE_COLUMNS
from ai_trading.utils.lazy_imports import load_pandas


def build_day_sleeve_features(bars: Any) -> Any:
    """Return one ordered feature row for the latest finalized five-minute bar.

    ``bars`` must contain sufficient finalized OHLCV history. Missing inputs or
    non-finite values on the latest feature row fail closed instead of being
    replaced with synthetic zero values.
    """

    pd = load_pandas()
    if pd is None or not hasattr(pd, "DataFrame"):
        raise RuntimeError("pandas is required for day-sleeve feature construction")
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise ValueError("Day-sleeve feature bars must be a non-empty DataFrame")

    frame = bars.copy()
    normalized_columns = [
        str(column).strip().lower() for column in frame.columns
    ]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("Day-sleeve feature bars contain duplicate columns")
    frame.columns = normalized_columns
    required_raw = ("open", "high", "low", "close", "volume")
    missing_raw = [column for column in required_raw if column not in frame.columns]
    if missing_raw:
        raise ValueError(f"Day-sleeve feature bars missing columns: {missing_raw}")
    frame = frame.sort_index()
    for column in required_raw:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    raw_values = frame.loc[:, required_raw].to_numpy(dtype=float)
    if not bool(np.isfinite(raw_values).all()):
        raise ValueError("Day-sleeve feature bars contain non-finite OHLCV values")

    frame = compute_macd(frame)
    frame = compute_macds(frame)
    frame = compute_atr(frame)
    frame = compute_vwap(frame)
    frame = compute_sma(frame, windows=(50, 200))

    close_values = frame["close"].astype(float).to_numpy()
    rsi_values = np.asarray(rsi_indicator(tuple(close_values.tolist()), 14), dtype=float)
    if rsi_values.size != close_values.size:
        raise ValueError("Day-sleeve RSI output is not aligned to input bars")
    frame["rsi"] = rsi_values

    close = pd.to_numeric(frame["close"], errors="coerce")
    close_abs = close.abs().replace(0.0, np.nan)
    atr = pd.to_numeric(frame.get("atr"), errors="coerce")
    vwap = pd.to_numeric(frame.get("vwap"), errors="coerce").replace(0.0, np.nan)
    sma_50 = pd.to_numeric(frame.get("sma_50"), errors="coerce")
    sma_200 = pd.to_numeric(frame.get("sma_200"), errors="coerce")
    macd = pd.to_numeric(frame.get("macd"), errors="coerce")
    rsi = pd.to_numeric(frame.get("rsi"), errors="coerce")
    signal_source = frame.get("signal", frame.get("macds"))
    signal = pd.to_numeric(signal_source, errors="coerce")

    frame["signal"] = signal
    frame["atr_pct"] = (atr / close_abs) * 100.0
    frame["vwap_distance"] = (close / vwap) - 1.0
    frame["sma_spread"] = (sma_50 - sma_200) / close_abs
    frame["macd_signal_gap"] = macd - signal
    frame["rsi_centered"] = (rsi - 50.0) / 50.0

    missing_features = [
        column for column in DAY_SLEEVE_ML_FEATURE_COLUMNS if column not in frame.columns
    ]
    if missing_features:
        raise ValueError(f"Day-sleeve features missing columns: {missing_features}")
    feature_row = frame.loc[frame.index[-1:], list(DAY_SLEEVE_ML_FEATURE_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = feature_row.to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()):
        invalid = [
            column
            for column, value in feature_row.iloc[-1].items()
            if not np.isfinite(float(value))
        ]
        raise ValueError(f"Day-sleeve latest feature row is non-finite: {invalid}")
    return feature_row


__all__ = ["build_day_sleeve_features"]
