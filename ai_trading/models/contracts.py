"""Shared model feature and bar-timeframe contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

MODEL_FEATURE_CONTRACT_VERSION = "ml_feature_contract_v1"
LIVE_ML_BAR_TIMEFRAME = "1Min"
AFTER_HOURS_ML_BAR_TIMEFRAME = "1Day"

LIVE_ML_FEATURE_COLUMNS: tuple[str, ...] = (
    "rsi",
    "macd",
    "atr",
    "vwap",
    "sma_50",
    "sma_200",
)

AFTER_HOURS_ML_FEATURE_COLUMNS: tuple[str, ...] = LIVE_ML_FEATURE_COLUMNS + (
    "signal",
    "atr_pct",
    "vwap_distance",
    "sma_spread",
    "macd_signal_gap",
    "rsi_centered",
)


def normalize_bar_timeframe(value: object) -> str:
    """Return a stable timeframe label for contract comparisons."""

    raw = str(value or "").strip()
    lowered = raw.lower().replace("_", "").replace("-", "")
    aliases = {
        "1m": LIVE_ML_BAR_TIMEFRAME,
        "1min": LIVE_ML_BAR_TIMEFRAME,
        "1minute": LIVE_ML_BAR_TIMEFRAME,
        "minute": LIVE_ML_BAR_TIMEFRAME,
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
) -> str:
    payload = {
        "bar_timeframe": normalize_bar_timeframe(bar_timeframe),
        "feature_columns": [str(column) for column in feature_columns],
        "version": MODEL_FEATURE_CONTRACT_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
