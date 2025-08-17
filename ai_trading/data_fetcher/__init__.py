"""Lightweight data fetching helpers with patchable client."""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

try:  # pragma: no cover - pandas optional in some tests
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

__all__ = [
    "ensure_datetime",
    "ensure_utc",
    "rfc3339",
    "get_bars",
    "get_minute_df",
    "get_historical_data",
    "_DATA_CLIENT",
    "get_bars_batch",
    "get_minute_bars",
    "get_minute_bars_batch",
    "warmup_cache",
    "get_cached_minute_timestamp",
    "last_minute_bar_age_seconds",
    "_MINUTE_CACHE",
]

_DATA_CLIENT: Any | None = None
_MINUTE_CACHE: dict[str, tuple[Any, datetime]] = {}


def ensure_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def ensure_datetime(dt: Any) -> datetime:
    """Normalize various datetime inputs to aware UTC ``datetime``."""
    if isinstance(dt, datetime):
        return ensure_utc(dt)
    try:  # pandas Timestamp support
        if pd is not None and isinstance(dt, pd.Timestamp):
            return ensure_utc(dt.to_pydatetime())
    except Exception:  # pragma: no cover - defensive
        pass
    if isinstance(dt, str):
        try:
            parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            return ensure_utc(parsed)
        except Exception:  # pragma: no cover - simple parser
            pass
    raise TypeError(f"Unsupported datetime value: {type(dt)!r}")


def rfc3339(dt: datetime | str) -> str:
    """Return an RFC3339 UTC timestamp string."""
    return ensure_datetime(dt).isoformat().replace("+00:00", "Z")


def _require_client():  # pragma: no cover - simple guard
    if _DATA_CLIENT is None:
        raise RuntimeError(
            "Data client not configured; patch ai_trading.data_fetcher._DATA_CLIENT"
        )
    return _DATA_CLIENT


def _call_bars(
    symbol: str, start: datetime, end: datetime, timeframe: str, limit: int | None
):
    client = _require_client()
    fn = getattr(client, "get_stock_bars", None) or getattr(client, "get_bars", None)
    if fn is None:
        raise RuntimeError("Data client missing get_stock_bars/get_bars")
    return fn(
        symbol=symbol,
        start=rfc3339(start),
        end=rfc3339(end),
        timeframe=timeframe,
        limit=limit,
    )


def get_bars(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    timeframe: str = "1Min",
    limit: int | None = None,
    *,
    max_retries: int = 3,
    retry_sleep_s: float = 0.25,
):
    """Fetch OHLCV bars with bounded retries."""
    start_dt, end_dt = ensure_datetime(start), ensure_datetime(end)
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return _call_bars(symbol, start_dt, end_dt, timeframe, limit)
        except Exception as e:  # pragma: no cover - network mocked in tests
            last_exc = e
            msg = str(e)
            if "Invalid format for parameter start" in msg and attempt >= max_retries:
                raise
            if attempt < max_retries:
                time.sleep(retry_sleep_s)
                continue
            raise
    raise last_exc or RuntimeError("get_bars failed unexpectedly")


def get_minute_df(symbol: str, start: datetime | str, end: datetime | str, **kw):
    """Fetch minute-level bars."""
    return get_bars(symbol, start, end, timeframe="1Min", **kw)


def get_historical_data(
    symbols: Iterable[str],
    start: datetime | str,
    end: datetime | str,
    timeframe: str = "1Day",
    **kw,
):
    """Fetch bars for multiple symbols and return mapping of symbol->DataFrame."""
    out: dict[str, Any] = {}
    for sym in symbols:
        out[sym] = get_bars(sym, start, end, timeframe=timeframe, **kw)
    return out


def get_bars_batch(*args, **kwargs):  # pragma: no cover - legacy stub
    raise NotImplementedError("get_bars_batch is no longer implemented")


def get_minute_bars(*args, **kwargs):  # pragma: no cover - legacy stub
    raise NotImplementedError("get_minute_bars is no longer implemented")


def get_minute_bars_batch(*args, **kwargs):  # pragma: no cover - legacy stub
    raise NotImplementedError("get_minute_bars_batch is no longer implemented")


def warmup_cache(*args, **kwargs):  # pragma: no cover - legacy stub
    return None


def get_cached_minute_timestamp(symbol: str):
    item = _MINUTE_CACHE.get(symbol)
    if not item:
        return None
    ts = item[1]
    if isinstance(ts, pd.Timestamp):
        return ts
    return pd.Timestamp(ts, tz="UTC") if pd is not None else ts


def last_minute_bar_age_seconds(symbol: str, now: datetime | None = None):
    ts = get_cached_minute_timestamp(symbol)
    if ts is None:
        return None
    now_dt = now or datetime.now(UTC)
    if isinstance(ts, pd.Timestamp):
        ts_dt = ts.to_pydatetime()
    else:
        ts_dt = ts
    return (now_dt - ts_dt).total_seconds()
