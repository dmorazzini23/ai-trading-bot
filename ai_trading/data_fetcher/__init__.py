"""Lightweight data fetching helpers with patchable client."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from datetime import UTC, date, datetime
from typing import Any

try:  # pragma: no cover - pandas optional in some tests
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

__all__ = [
    "ensure_datetime",
    "ensure_rfc3339",
    "get_bars",
    "get_historical_data",
    "get_minute_df",
    "set_data_client",
    "_DATA_CLIENT",
]

_DATA_CLIENT: Any | None = None

# AI-AGENT-REF: legacy import alias
sys.modules.setdefault("data_fetcher", sys.modules[__name__])


def set_data_client(client: Any) -> None:
    """Set the global data client used by helpers."""  # AI-AGENT-REF: patchable client
    global _DATA_CLIENT
    _DATA_CLIENT = client


def ensure_datetime(dt: datetime | date | str) -> datetime:
    """Return an aware ``datetime`` in UTC."""  # AI-AGENT-REF: normalize inputs
    if isinstance(dt, datetime):
        return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    if isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day, tzinfo=UTC)
    if isinstance(dt, str):
        parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    raise TypeError(f"Unsupported datetime value: {dt!r}")


def ensure_rfc3339(dt: datetime | date | str) -> str:
    """Return RFC3339 formatted timestamp in UTC."""
    return ensure_datetime(dt).isoformat().replace("+00:00", "Z")


def _client() -> Any:
    if _DATA_CLIENT is None:
        raise RuntimeError("DATA_CLIENT not configured")
    return _DATA_CLIENT


def get_bars(
    symbols: Iterable[str] | str,
    timeframe: str,
    start: datetime | date | str,
    end: datetime | date | str | None = None,
    *,
    limit: int | None = None,
    retries: int = 3,
    backoff_s: float = 0.2,
) -> dict[str, pd.DataFrame]:
    """Fetch bars for ``symbols`` and return mapping of symbol to DataFrame."""
    sym_list = [symbols] if isinstance(symbols, str) else list(symbols)
    start_s = ensure_rfc3339(start)
    end_s = ensure_rfc3339(end) if end is not None else None
    client = _client()
    fn = getattr(client, "get_stock_bars", None) or getattr(client, "get_bars", None)
    if fn is None:
        raise RuntimeError("DATA_CLIENT missing get_stock_bars/get_bars")
    out: dict[str, pd.DataFrame] = {}
    for sym in sym_list:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                out[sym] = fn(
                    symbol=sym,
                    timeframe=timeframe,
                    start=start_s,
                    end=end_s,
                    limit=limit,
                )
                break
            except Exception as e:  # pragma: no cover - network mocked in tests
                last_exc = e
                if attempt >= retries:
                    raise
                time.sleep(backoff_s)
        else:
            if last_exc:
                raise last_exc
    return out


def get_historical_data(
    symbol: str,
    start: datetime | date | str,
    end: datetime | date | str,
    *,
    timeframe: str = "1Min",
    limit: int | None = 1000,
    retries: int = 3,
    backoff_s: float = 0.2,
) -> pd.DataFrame:
    """Fetch historical bars for a single ``symbol``."""
    return get_bars(
        [symbol],
        timeframe,
        start,
        end,
        limit=limit,
        retries=retries,
        backoff_s=backoff_s,
    )[symbol]


def get_minute_df(
    symbol: str,
    start_dt: datetime | date | str,
    end_dt: datetime | date | str,
    *,
    retries: int = 3,
    backoff_s: float = 0.2,
) -> pd.DataFrame:
    """Fetch minute-level bars for ``symbol``."""  # AI-AGENT-REF: bounded retry
    start_s = ensure_rfc3339(start_dt)
    end_s = ensure_rfc3339(end_dt)
    client = _client()
    fn = getattr(client, "get_stock_bars", None) or getattr(client, "get_bars", None)
    if fn is None:
        raise RuntimeError("DATA_CLIENT missing get_stock_bars/get_bars")
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn(symbol=symbol, timeframe="1Min", start=start_s, end=end_s, limit=None)
        except Exception as e:  # pragma: no cover - network mocked
            last_exc = e
            if "Invalid format for parameter" in str(e) or attempt >= retries:
                raise
            time.sleep(backoff_s)
    raise last_exc or RuntimeError("get_minute_df failed")
