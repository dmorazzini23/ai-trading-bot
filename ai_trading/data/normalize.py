"""Utilities for normalizing OHLCV data returned by third-party providers."""

from __future__ import annotations

import pandas as pd


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

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    return df[required]


__all__ = ["ensure_ohlcv"]

