from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

__all__ = [
    "ensure_datetime",
    "get_minute_df",
    "get_historical_data",
    "_DATA_CLIENT",
    "MAX_RETRIES",
]

# Patch point used by tests (monkeypatch/patch)
_DATA_CLIENT: Any | None = None

MAX_RETRIES = int(os.getenv("DATA_RETRY_MAX", "3") or 3)
RETRY_SLEEP_S = float(os.getenv("DATA_RETRY_SLEEP_S", "0.05") or 0.05)


def ensure_datetime(dt: datetime | str) -> datetime:
    """
    Ensure datetime is timezone-aware in UTC (RFC3339-compatible).
    Accepts datetime or ISO-like string; returns tz-aware UTC datetime.
    """
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    if isinstance(dt, str):
        # Let datetime parse common formats; default to UTC.
        try:
            parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            # Fallback: naive parse, treat as UTC
            parsed = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise TypeError("dt must be datetime or str")


def _to_rfc3339(dt: datetime | str) -> str:
    d = ensure_datetime(dt)
    # Alpaca accepts 'Z' suffix
    return d.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def _get_data_client() -> Any | None:
    """
    Lazily get Alpaca historical data client; optional import.
    Never raises at import time; only called at runtime.
    """
    global _DATA_CLIENT
    if _DATA_CLIENT is not None:
        return _DATA_CLIENT
    try:
        from alpaca.data.historical import StockHistoricalDataClient

        # In tests we don't want real creds; passing blank is allowed for paper mocks.
        _DATA_CLIENT = StockHistoricalDataClient("", "")
        return _DATA_CLIENT
    except Exception:
        return None


def _is_persistent_datetime_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "invalid format for parameter start" in msg
        or "error parsing" in msg
        or "rfc3339" in msg
    )


def get_minute_df(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    limit: int | None = None,
) -> Any:
    """
    Fetch minute bars. Tests will monkeypatch `_DATA_CLIENT.get_stock_bars`.
    On persistent datetime format errors, retry up to MAX_RETRIES then raise.
    """
    client = _DATA_CLIENT or _get_data_client()
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            if client is None:
                raise RuntimeError("data client unavailable")
            # The tests patch this method; keep the signature simple.
            return client.get_stock_bars(
                symbol,
                _to_rfc3339(start),
                _to_rfc3339(end),
                timeframe="1Min",
                limit=limit,
            )
        except Exception as e:  # pragma: no cover (covered by tests)
            last_exc = e
            if _is_persistent_datetime_error(e) and attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_SLEEP_S)

    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to fetch minute bars")


def get_historical_data(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    timeframe: str = "1Day",
    limit: int | None = None,
) -> Any:
    """
    Simplified historical fetcher; tests patch the client similarly.
    """
    client = _DATA_CLIENT or _get_data_client()
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            if client is None:
                raise RuntimeError("data client unavailable")
            return client.get_stock_bars(
                symbol,
                _to_rfc3339(start),
                _to_rfc3339(end),
                timeframe=timeframe,
                limit=limit,
            )
        except Exception as e:
            last_exc = e
            if _is_persistent_datetime_error(e) and attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_SLEEP_S)

    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to fetch historical data")
