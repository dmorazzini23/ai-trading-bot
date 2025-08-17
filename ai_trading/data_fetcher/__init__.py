"""Unified data fetcher API with patchable client and UTC helpers."""

from __future__ import annotations

import sys as _sys
import time
from datetime import UTC, datetime
from typing import Any

try:  # pragma: no cover - pandas optional
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

__all__ = [
    "ensure_datetime",
    "rfc3339",
    "get_bars",
    "get_minute_df",
    "get_historical_data",
    "set_data_client",
    "_DATA_CLIENT",
]

_DATA_CLIENT: Any | None = None
_CACHE: dict[tuple[str, datetime, datetime], pd.DataFrame] = {}


def set_data_client(client: Any) -> None:
    """Set the global data client used for fetching bars."""
    global _DATA_CLIENT  # noqa: PLW0603
    _DATA_CLIENT = client


def ensure_datetime(dt: datetime | str) -> datetime:
    """Return a timezone-aware UTC ``datetime`` for ``dt``."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    elif hasattr(dt, "to_pydatetime"):
        dt = dt.to_pydatetime()  # type: ignore[assignment]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def rfc3339(dt: datetime | str) -> str:
    """Return an RFC3339 string in UTC for ``dt``."""
    return ensure_datetime(dt).isoformat().replace("+00:00", "Z")


def get_minute_df(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    *,
    retries: int = 3,
    backoff_s: float = 0.5,
) -> pd.DataFrame:
    """Fetch minute bars for ``symbol`` with bounded retries."""
    if _DATA_CLIENT is None:
        raise RuntimeError("DATA_CLIENT not configured")
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    key = (symbol, start_dt, end_dt)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            df = _DATA_CLIENT.get_stock_bars(symbol, start=start_dt, end=end_dt)
            _CACHE[key] = df
            return df
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc)
            if (
                "Invalid format for parameter start" not in msg and "error parsing" not in msg
            ) or attempt >= retries - 1:
                raise
            time.sleep(backoff_s * (2**attempt))
    raise last_exc  # pragma: no cover - unreachable


def get_historical_data(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Backward compatible wrapper for :func:`get_minute_df`."""
    return get_minute_df(symbol, start, end, **kwargs)


get_bars = get_minute_df

_sys.modules.setdefault("data_fetcher", _sys.modules[__name__])
