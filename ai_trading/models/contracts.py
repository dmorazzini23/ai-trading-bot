"""Shared model feature and bar-timeframe contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np

MODEL_FEATURE_CONTRACT_VERSION = "ml_feature_contract_v1"
LIVE_ML_BAR_TIMEFRAME = "1Min"
AFTER_HOURS_ML_BAR_TIMEFRAME = "1Day"
DAY_SLEEVE_ML_FEATURE_CONTRACT_VERSION = "day_sleeve_ml_feature_contract_v1"
DAY_SLEEVE_ML_BAR_TIMEFRAME = "5Min"

LIVE_ML_FEATURE_COLUMNS: tuple[str, ...] = (
    "rsi",
    "macd",
    "atr",
    "vwap",
    "sma_50",
    "sma_200",
)

DAY_SLEEVE_ML_FEATURE_COLUMNS: tuple[str, ...] = LIVE_ML_FEATURE_COLUMNS + (
    "signal",
    "atr_pct",
    "vwap_distance",
    "sma_spread",
    "macd_signal_gap",
    "rsi_centered",
)


def infer_day_sleeve_regimes(close_values: object) -> tuple[str, ...]:
    """Infer past-only regimes shared by day-sleeve training and serving."""

    close = np.asarray(close_values, dtype=float)
    if close.size == 0:
        return ()
    returns = np.zeros_like(close, dtype=float)
    returns[1:] = np.diff(close) / np.maximum(close[:-1], 1e-9)
    vol = np.full_like(returns, np.nan, dtype=float)
    trend = np.full_like(returns, np.nan, dtype=float)
    window = 20
    for idx in range(window, len(returns)):
        segment = returns[idx - window : idx]
        vol[idx] = float(np.std(segment))
        trend[idx] = float((close[idx] / close[idx - window]) - 1.0)
    labels: list[str] = []
    threshold_window = max(window * 3, window)
    for idx in range(len(close)):
        threshold_slice = vol[max(0, idx - threshold_window + 1) : idx + 1]
        finite_thresholds = threshold_slice[np.isfinite(threshold_slice)]
        vol_threshold = (
            float(np.nanpercentile(finite_thresholds, 70))
            if finite_thresholds.size
            else 0.02
        )
        if np.isfinite(vol[idx]) and vol[idx] >= vol_threshold:
            labels.append("volatile")
        elif np.isfinite(trend[idx]) and trend[idx] >= 0.02:
            labels.append("uptrend")
        elif np.isfinite(trend[idx]) and trend[idx] <= -0.02:
            labels.append("downtrend")
        else:
            labels.append("sideways")
    return tuple(labels)


def normalize_bar_timeframe(value: object) -> str:
    """Return a stable timeframe label for contract comparisons."""

    raw = str(value or "").strip()
    lowered = raw.lower().replace("_", "").replace("-", "")
    aliases = {
        "1m": LIVE_ML_BAR_TIMEFRAME,
        "1min": LIVE_ML_BAR_TIMEFRAME,
        "1minute": LIVE_ML_BAR_TIMEFRAME,
        "minute": LIVE_ML_BAR_TIMEFRAME,
        "5m": DAY_SLEEVE_ML_BAR_TIMEFRAME,
        "5min": DAY_SLEEVE_ML_BAR_TIMEFRAME,
        "5minute": DAY_SLEEVE_ML_BAR_TIMEFRAME,
        "1d": AFTER_HOURS_ML_BAR_TIMEFRAME,
        "1day": AFTER_HOURS_ML_BAR_TIMEFRAME,
        "day": AFTER_HOURS_ML_BAR_TIMEFRAME,
        "daily": AFTER_HOURS_ML_BAR_TIMEFRAME,
    }
    return aliases.get(lowered, raw)


def model_feature_contract_hash(
    feature_columns: Sequence[str],
    *,
    bar_timeframe: str,
    contract_version: str = MODEL_FEATURE_CONTRACT_VERSION,
) -> str:
    payload = {
        "bar_timeframe": normalize_bar_timeframe(bar_timeframe),
        "feature_columns": [str(column) for column in feature_columns],
        "version": str(contract_version),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
