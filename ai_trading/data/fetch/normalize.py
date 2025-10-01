from __future__ import annotations

from typing import TYPE_CHECKING
import re

from ai_trading.utils.lazy_imports import load_pandas

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import pandas as _pd

pd = load_pandas()

REQUIRED = ("open", "high", "low", "close", "volume")


def _normalize_column_name(value: object) -> str:
    token = str(value).strip()
    token = re.sub(r"(?<=[A-Za-z0-9])(?=[A-Z])", "_", token)
    token = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", token)
    lowered = token.lower().replace(" ", "_").replace("-", "_")
    while "__" in lowered:
        lowered = lowered.replace("__", "_")
    return lowered

_COLUMN_CANONICAL_MAP = {
    "t": "timestamp",
    "time": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",
    "o": "open",
    "open_price": "open",
    "opening_price": "open",
    "session_open": "open",
    "session_open_price": "open",
    "start_price": "open",
    "starting_price": "open",
    "h": "high",
    "high_price": "high",
    "maximum_price": "high",
    "session_high": "high",
    "session_high_price": "high",
    "peak_price": "high",
    "l": "low",
    "low_price": "low",
    "minimum_price": "low",
    "session_low": "low",
    "session_low_price": "low",
    "floor_price": "low",
    "c": "close",
    "close_price": "close",
    "latest_price": "close",
    "latest_value": "close",
    "market_price": "close",
    "official_price": "close",
    "ending_price": "close",
    "end_price": "close",
    "final_value": "close",
    "session_close": "close",
    "session_close_price": "close",
    "v": "volume",
    "volume_total": "volume",
    "total_volume": "volume",
    "totalvolume": "volume",
    "volumetotal": "volume",
    "session_volume": "volume",
    "share_count": "volume",
    "shares_traded": "volume",
    "shares": "volume",
    "adjclose": "adj close",
    "adj_close": "adj close",
    "closeadj": "adj close",
    "close_adj": "adj close",
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
        frame.columns = [_normalize_column_name(levels[0]) for levels in frame.columns]
    else:
        frame = df.copy()
        frame.columns = [_normalize_column_name(col) for col in frame.columns]

    if _COLUMN_CANONICAL_MAP:
        frame = frame.rename(columns=_COLUMN_CANONICAL_MAP)

    frame = frame.loc[:, ~pd.Index(frame.columns).duplicated()]

    had_timestamp_column = "timestamp" in frame.columns
    if "adj close" in frame.columns and "close" not in frame.columns:
        frame["close"] = frame["adj close"]
    if "close" not in frame.columns and "value" in frame.columns:
        frame["close"] = frame["value"]
    if "close" not in frame.columns:
        for candidate in (
            "latest_price",
            "latest_value",
            "market_price",
            "official_price",
            "ending_price",
            "end_price",
            "final_value",
            "final_price",
            "last_value",
            "last_price",
            "price",
        ):
            if candidate in frame.columns:
                frame["close"] = frame[candidate]
                break

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
