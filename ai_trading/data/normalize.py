"""Utilities for normalizing OHLCV data returned by third-party providers."""

from __future__ import annotations

import pandas as pd
import numpy as np


def ensure_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return a normalized OHLCV frame or ``None`` when the data is unusable."""

    if df is None or df.empty:
        return None

    cols = {col: str(col).strip().lower() for col in df.columns}
    df = df.rename(columns=cols)

    if "adj close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj close"]

    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return None

    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required, how="any")
    if df.empty:
        return None

    numeric = df.loc[:, required]
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    positive_ohlc = (numeric.loc[:, ["open", "high", "low", "close"]] > 0).all(axis=1)
    valid_volume = numeric["volume"] >= 0
    consistent_ohlc = (
        (numeric["high"] >= numeric[["open", "low", "close"]].max(axis=1))
        & (numeric["low"] <= numeric[["open", "high", "close"]].min(axis=1))
    )
    df = df.loc[finite_mask & positive_ohlc & valid_volume & consistent_ohlc]
    if df.empty:
        return None
    df["volume"] = df["volume"].astype("int64")
    return df[required]


__all__ = ["ensure_ohlcv"]
