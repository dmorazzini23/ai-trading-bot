from __future__ import annotations

from typing import TYPE_CHECKING

from ai_trading.utils.lazy_imports import load_pandas

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import pandas as _pd

pd = load_pandas()

REQUIRED = ("open", "high", "low", "close", "volume")


def normalize_ohlcv_df(df: "_pd.DataFrame | None") -> "_pd.DataFrame":
    """Return a normalized OHLCV dataframe with canonical columns."""

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=REQUIRED)

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(levels[0]).lower() for levels in df.columns]
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    else:
        df = df.copy()
        df.columns = [str(col).lower().strip() for col in df.columns]

    if "adj close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj close"]
    if "close" not in df.columns and "value" in df.columns:
        df["close"] = df["value"]

    for column in REQUIRED:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    subset = [column for column in REQUIRED if column in df.columns]
    if subset:
        df = df.dropna(subset=subset)

    index = df.index
    if isinstance(index, pd.DatetimeIndex):
        if index.tz is None:
            df.index = index.tz_localize("UTC")
    elif hasattr(index, "tz") and getattr(index, "tz", None) is None:
        try:
            df.index = index.tz_localize("UTC")  # type: ignore[attr-defined]
        except Exception:
            pass

    df = df.sort_index()

    return df[[column for column in REQUIRED if column in df.columns]]
