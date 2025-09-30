from __future__ import annotations

from typing import TYPE_CHECKING

from ai_trading.utils.lazy_imports import load_pandas

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import pandas as _pd

pd = load_pandas()

REQUIRED = ("open", "high", "low", "close", "volume")

_COLUMN_CANONICAL_MAP = {
    "t": "timestamp",
    "time": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "totalvolume": "volume",
    "volumetotal": "volume",
    "volume_total": "volume",
    "total_volume": "volume",
}


def _empty_frame() -> "_pd.DataFrame":
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    base = pd.DataFrame(columns=REQUIRED, index=idx)
    base.insert(0, "timestamp", idx)
    return base


def normalize_ohlcv_df(df: "_pd.DataFrame | None") -> "_pd.DataFrame":
    """Return a normalized OHLCV dataframe with canonical columns."""

    if df is None or len(df) == 0:
        return _empty_frame()

    attrs: dict[str, object] = {}
    try:
        attrs = dict(getattr(df, "attrs", {}) or {})
    except (AttributeError, TypeError):  # pragma: no cover - metadata optional
        attrs = {}

    if isinstance(df.columns, pd.MultiIndex):
        frame = df.copy()
        frame.columns = [str(levels[0]).lower().strip() for levels in frame.columns]
    else:
        frame = df.copy()
        frame.columns = [str(col).lower().strip() for col in frame.columns]

    if _COLUMN_CANONICAL_MAP:
        frame = frame.rename(columns=_COLUMN_CANONICAL_MAP)

    frame = frame.loc[:, ~pd.Index(frame.columns).duplicated()]

    had_timestamp_column = "timestamp" in frame.columns
    if "adj close" in frame.columns and "close" not in frame.columns:
        frame["close"] = frame["adj close"]
    if "close" not in frame.columns and "value" in frame.columns:
        frame["close"] = frame["value"]

    for column in REQUIRED:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.drop(columns=["timestamp"])
        frame.index = ts

    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        except (TypeError, ValueError, AttributeError):
            return _empty_frame()

    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")

    valid_index = ~frame.index.isna()
    if not valid_index.any():
        return _empty_frame()
    if not valid_index.all():
        frame = frame.loc[valid_index]

    subset = [column for column in REQUIRED if column in frame.columns]
    if subset:
        frame = frame.dropna(subset=subset, how="any")

    if frame.empty:
        return _empty_frame()

    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.sort_index()
    frame.index.rename("timestamp", inplace=True)

    cols = [column for column in REQUIRED if column in frame.columns]
    if not cols:
        return _empty_frame()
    normalized = frame[cols].copy()
    if normalized.index.tz is None:
        normalized.index = normalized.index.tz_localize("UTC")
    normalized.index.rename("timestamp", inplace=True)
    if had_timestamp_column or getattr(df.index, "name", None) == "timestamp":
        if "timestamp" in normalized.columns:
            normalized = normalized.drop(columns=["timestamp"])
        normalized.insert(0, "timestamp", normalized.index)
    if attrs:
        try:
            normalized.attrs.update(attrs)
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - metadata optional
            pass
    return normalized
