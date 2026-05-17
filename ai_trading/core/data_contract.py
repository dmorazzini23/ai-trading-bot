"""Market data contract normalization and validation."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from ai_trading.logging import get_logger
from ai_trading.utils.lazy_imports import load_pandas

logger = get_logger(__name__)
pd = load_pandas()

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
_NY_TZ = ZoneInfo("America/New_York")
_MAX_FUTURE_BAR_SKEW_SECONDS = 5.0


@dataclass
class DataContractResult:
    ok: bool
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def _coerce_datetime_index(df: Any) -> Any:
    if df is None or df.empty:
        return df
    if not hasattr(df, "index"):
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    for candidate in ("timestamp", "time", "date", "datetime"):
        if candidate in df.columns:
            try:
                df = df.set_index(candidate)
                break
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                continue
    try:
        converted = pd.to_datetime(df.index, errors="coerce")
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        return df
    if isinstance(converted, pd.DatetimeIndex) and converted.tz is None:
        raise ValueError("naive timestamps are not allowed")
    df.index = converted
    return df


def normalize_bars(
    df: Any,
    timeframe: str,
    tz: ZoneInfo | None = None,
    rth_only: bool = True,
) -> Any:
    """Return normalized bar DataFrame with standard columns and UTC index."""
    if str(timeframe).lower() in {"1day", "day", "1d"}:
        rth_only = False
    if df is None:
        return df
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = _coerce_datetime_index(df)
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            raise ValueError("naive timestamps are not allowed")
        df.index = df.index.tz_convert(UTC)
    if rth_only and isinstance(df.index, pd.DatetimeIndex):
        eastern = df.index.tz_convert(_NY_TZ)
        is_weekday = eastern.weekday < 5
        is_rth = (
            (eastern.hour > 9) | ((eastern.hour == 9) & (eastern.minute >= 30))
        ) & (eastern.hour < 16)
        df = df[is_weekday & is_rth]
    return df


def validate_bars(
    df: Any,
    timeframe: str,
    freshness_seconds: int,
    rth_only: bool = True,
) -> DataContractResult:
    """Validate bar data for completeness and freshness."""
    if str(timeframe).lower() in {"1day", "day", "1d"}:
        rth_only = False
    if df is None or getattr(df, "empty", True):
        return DataContractResult(False, "EMPTY_BARS")
    if not set(REQUIRED_COLUMNS).issubset(df.columns):
        missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
        return DataContractResult(False, "MISSING_COLUMNS", {"missing": missing})
    if not isinstance(df.index, pd.DatetimeIndex):
        return DataContractResult(False, "INVALID_INDEX")
    if df.index.tz is None:
        return DataContractResult(False, "NAIVE_INDEX")
    if df.index.has_duplicates:
        return DataContractResult(False, "DUPLICATE_BARS")
    if not df.index.is_monotonic_increasing:
        return DataContractResult(False, "NON_MONOTONIC")
    last_row = df.iloc[-1]
    for col in REQUIRED_COLUMNS:
        try:
            if pd.isna(last_row[col]):
                return DataContractResult(False, "NAN_LAST_BAR", {"column": col})
        except AI_TRADING_FALLBACK_EXCEPTIONS:
            return DataContractResult(False, "NAN_LAST_BAR", {"column": col})
    last_ts = df.index[-1]
    if last_ts.tzinfo is None:
        return DataContractResult(False, "NAIVE_INDEX")
    now = datetime.now(UTC)
    age_seconds = (now - last_ts.astimezone(UTC)).total_seconds()
    if age_seconds < -_MAX_FUTURE_BAR_SKEW_SECONDS:
        return DataContractResult(
            False,
            "FUTURE_BAR",
            {
                "future_skew_seconds": abs(age_seconds),
                "allowed_future_skew_seconds": _MAX_FUTURE_BAR_SKEW_SECONDS,
            },
        )
    age_seconds = max(0.0, age_seconds)
    if freshness_seconds >= 0 and age_seconds > freshness_seconds:
        return DataContractResult(False, "STALE_BAR", {"age_seconds": age_seconds})
    if rth_only:
        eastern = last_ts.astimezone(_NY_TZ)
        if eastern.weekday() >= 5:
            return DataContractResult(False, "OUT_OF_SESSION", {"session": "weekend"})
        if not (
            (eastern.hour > 9 or (eastern.hour == 9 and eastern.minute >= 30))
            and eastern.hour < 16
        ):
            return DataContractResult(False, "OUT_OF_SESSION", {"session": "extended"})
    return DataContractResult(True)
