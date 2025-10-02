from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
import re

from ai_trading.utils.lazy_imports import load_pandas

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import pandas as _pd

pd = load_pandas()

REQUIRED = ("open", "high", "low", "close", "volume")


_ACRONYM_TOKEN_PATTERN = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+"
)


def _normalize_column_name(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"[\s\-\.]+", "_", text)
    parts = [part for part in text.split("_") if part]
    tokens: list[str] = []
    for part in parts:
        matches = _ACRONYM_TOKEN_PATTERN.findall(part)
        if not matches:
            tokens.append(part)
            continue
        merged: list[str] = []
        for segment in matches:
            if segment.isdigit() and merged:
                merged[-1] = f"{merged[-1]}{segment}"
            else:
                merged.append(segment)
        tokens.extend(merged)
    if not tokens:
        return ""
    normalized = "_".join(tokens).lower()
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized

_COLUMN_CANONICAL_MAP = {
    "t": "timestamp",
    "time": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",
    "ts": "timestamp",
    "timestamp_utc": "timestamp",
    "bars_t": "timestamp",
    "bar_t": "timestamp",
    "bars_time": "timestamp",
    "bar_time": "timestamp",
    "o": "open",
    "open_price": "open",
    "opening_price": "open",
    "session_open": "open",
    "session_open_price": "open",
    "start_price": "open",
    "starting_price": "open",
    "bars_open": "open",
    "bar_open": "open",
    "h": "high",
    "high_price": "high",
    "maximum_price": "high",
    "session_high": "high",
    "session_high_price": "high",
    "peak_price": "high",
    "bars_high": "high",
    "bar_high": "high",
    "l": "low",
    "low_price": "low",
    "minimum_price": "low",
    "session_low": "low",
    "session_low_price": "low",
    "floor_price": "low",
    "bars_low": "low",
    "bar_low": "low",
    "c": "close",
    "close_price": "close",
    "closing_price": "close",
    "latest_price": "close",
    "latest_value": "close",
    "market_price": "close",
    "official_price": "close",
    "ending_price": "close",
    "end_price": "close",
    "final_price": "close",
    "final_value": "close",
    "session_close": "close",
    "session_close_price": "close",
    "bars_close": "close",
    "bar_close": "close",
    "v": "volume",
    "volume_total": "volume",
    "total_volume": "volume",
    "totalvolume": "volume",
    "volumetotal": "volume",
    "session_volume": "volume",
    "share_count": "volume",
    "shares_traded": "volume",
    "shares": "volume",
    "bars_volume": "volume",
    "bar_volume": "volume",
    "adjclose": "adj close",
    "adj_close": "adj close",
    "closeadj": "adj close",
    "close_adj": "adj close",
}


def _empty_frame(include_timestamp: bool = False) -> "_pd.DataFrame":
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    base_columns = list(REQUIRED)
    frame = pd.DataFrame(columns=base_columns, index=idx)
    if include_timestamp:
        frame.insert(0, "timestamp", idx)
    return frame


def normalize_ohlcv_df(
    df: "_pd.DataFrame | None",
    *,
    include_columns: Iterable[object] | None = None,
) -> "_pd.DataFrame":
    """Return a normalized OHLCV dataframe with canonical columns.

    Parameters
    ----------
    df:
        Input OHLCV-like dataframe. May be ``None`` or empty.
    include_columns:
        Optional iterable of column names to retain in the normalized output.
        Recognized values include ``"timestamp"`` to reintroduce the datetime
        index as a column and ``"trade_count"`` to preserve the source field
        when provided.
    """

    include_list: list[str] = []
    include_lookup: set[str] = set()
    for value in include_columns or []:
        text = str(value).strip()
        if not text:
            continue
        column_name = _normalize_column_name(value)
        if column_name in include_lookup:
            continue
        include_lookup.add(column_name)
        include_list.append(column_name)
    include_timestamp = "timestamp" in include_lookup

    if df is None or len(df) == 0:
        return _empty_frame(include_timestamp=include_timestamp)

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

    had_trade_count_column = "trade_count" in frame.columns
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

    optional_columns = [
        column
        for column in include_list
        if column not in {"timestamp", *REQUIRED}
        and column in frame.columns
    ]
    cols = [column for column in REQUIRED if column in frame.columns]
    if not cols:
        return _empty_frame(include_timestamp=include_timestamp)
    selected_cols = cols + [col for col in optional_columns if col not in cols]
    normalized = frame[selected_cols].copy()
    if normalized.index.tz is None:
        normalized.index = normalized.index.tz_localize("UTC")
    normalized.index.rename("timestamp", inplace=True)
    if include_timestamp:
        if "timestamp" in normalized.columns:
            normalized = normalized.drop(columns=["timestamp"])
        normalized.insert(0, "timestamp", normalized.index)
    if attrs:
        try:
            normalized.attrs.update(attrs)
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - metadata optional
            pass
    if "trade_count" in normalized.columns and not had_trade_count_column:
        normalized = normalized.drop(columns=["trade_count"])
    return normalized
