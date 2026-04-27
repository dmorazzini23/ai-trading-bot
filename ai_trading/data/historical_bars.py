"""Shared CSV loader for deterministic historical OHLCV workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


_REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


@dataclass(frozen=True)
class HistoricalBarLoadReport:
    path: str
    rows_read: int
    rows_after_cleanup: int
    timestamp_column: str | None
    inferred_range_index: bool
    missing_volume_filled: bool
    invalid_timestamp_rows: int
    duplicate_timestamp_rows: int
    non_numeric_rows_dropped: int
    non_positive_rows_dropped: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_timestamp_series(series: pd.Series) -> pd.Index:
    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except TypeError:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed


def load_historical_bars(
    csv_path: str | Path,
    *,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, HistoricalBarLoadReport]:
    path = Path(csv_path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")

    rows_read = int(len(df))
    idx: pd.Index | None = None
    lower_map = {str(col).lower(): col for col in df.columns}
    resolved_ts_col = lower_map.get(timestamp_col.lower()) or lower_map.get("timestamp")
    inferred_range_index = False

    if resolved_ts_col is not None:
        idx = _parse_timestamp_series(df[resolved_ts_col])
        df = df.drop(columns=[resolved_ts_col])
    else:
        first_col = df.columns[0]
        first_series = df[first_col]
        if pd.api.types.is_numeric_dtype(first_series):
            idx = pd.RangeIndex(start=0, stop=len(df), step=1)
            inferred_range_index = True
        else:
            candidate = _parse_timestamp_series(first_series)
            parse_ratio = float(candidate.notna().mean())
            if parse_ratio >= 0.95:
                idx = candidate
                df = df.drop(columns=[first_col])
                resolved_ts_col = str(first_col)
            else:
                idx = pd.RangeIndex(start=0, stop=len(df), step=1)
                inferred_range_index = True

    df = df.rename(columns={col: str(col).lower() for col in df.columns})
    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    missing_volume_filled = "volume" not in df.columns
    if missing_volume_filled:
        df["volume"] = 0.0

    out = df[list(_REQUIRED_COLUMNS) + ["volume"]].copy()
    out.index = idx
    for col in list(_REQUIRED_COLUMNS) + ["volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([float("inf"), float("-inf")], pd.NA)

    invalid_timestamp_rows = 0
    duplicate_timestamp_rows = 0
    if isinstance(out.index, pd.DatetimeIndex):
        valid_mask = ~out.index.isna()
        invalid_timestamp_rows = int((~valid_mask).sum())
        if invalid_timestamp_rows > 0:
            out = out.loc[valid_mask]
        out = out.sort_index(kind="stable")
        duplicate_timestamp_rows = int(out.index.duplicated(keep=False).sum())
        if duplicate_timestamp_rows > 0:
            out = out.loc[~out.index.duplicated(keep="last")]
    else:
        out = out.sort_index(kind="stable")

    before_numeric_drop = int(len(out))
    out = out.dropna(subset=list(_REQUIRED_COLUMNS))
    non_numeric_rows_dropped = before_numeric_drop - int(len(out))

    before_positive_drop = int(len(out))
    positive_mask = (out[list(_REQUIRED_COLUMNS)] > 0.0).all(axis=1)
    out = out.loc[positive_mask]
    non_positive_rows_dropped = before_positive_drop - int(len(out))

    if out.empty:
        raise ValueError(f"{path} has no valid OHLC rows after cleanup")

    report = HistoricalBarLoadReport(
        path=str(path),
        rows_read=rows_read,
        rows_after_cleanup=int(len(out)),
        timestamp_column=(str(resolved_ts_col) if resolved_ts_col is not None else None),
        inferred_range_index=bool(inferred_range_index),
        missing_volume_filled=bool(missing_volume_filled),
        invalid_timestamp_rows=invalid_timestamp_rows,
        duplicate_timestamp_rows=duplicate_timestamp_rows,
        non_numeric_rows_dropped=non_numeric_rows_dropped,
        non_positive_rows_dropped=non_positive_rows_dropped,
    )
    return out, report


def filter_historical_bars_window(
    frame: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    if (start is not None or end is not None) and not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("date filters require timestamped historical bars; CSV did not provide a datetime index")
    start_ts = pd.to_datetime(str(start), errors="coerce", utc=True) if start is not None else pd.NaT
    end_ts = pd.to_datetime(str(end), errors="coerce", utc=True) if end is not None else pd.NaT
    filtered = frame
    if not pd.isna(start_ts):
        filtered = filtered[filtered.index >= start_ts]
    if not pd.isna(end_ts):
        filtered = filtered[filtered.index <= end_ts]
    return filtered


__all__ = [
    "HistoricalBarLoadReport",
    "filter_historical_bars_window",
    "load_historical_bars",
]
