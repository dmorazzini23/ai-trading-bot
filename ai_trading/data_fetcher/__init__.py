"""Lightweight data fetching helpers with patchable client."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any

try:  # pragma: no cover - pandas optional in some tests
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ai_trading.utils import clamp_timeout

__all__ = [
    "ensure_datetime",
    "to_rfc3339",
    "_DATA_CLIENT",
    "get_bars",
    "get_minute_df",
    "get_historical_data",
]

# Patch point used by tests
_DATA_CLIENT: Any | None = None
MAX_RETRIES = int(os.getenv("DATA_RETRY_MAX", "3") or 3)
RETRY_SLEEP_S = float(os.getenv("DATA_RETRY_SLEEP_S", "0.05") or 0.05)


def ensure_datetime(dt: datetime | str) -> datetime:
    """Return timezone-aware UTC datetime for inputs."""
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    parsed = datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def to_rfc3339(dt: datetime | str) -> str:
    """Return RFC3339 timestamp string in UTC."""
    return ensure_datetime(dt).isoformat().replace("+00:00", "Z")


def _require_client():
    if _DATA_CLIENT is None:
        raise RuntimeError("DATA_CLIENT not configured")
    return _DATA_CLIENT


def _is_persistent_datetime_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "invalid format for parameter start" in msg and "error parsing" in msg


def get_bars(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    *,
    timeframe: str = "1Min",
    limit: int | None = None,
    adjust: str = "raw",
    include_extended: bool = False,
    timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    """Fetch OHLCV bars for ``symbol`` between ``start`` and ``end``."""
    client = _require_client()
    start_s = to_rfc3339(start)
    end_s = to_rfc3339(end)
    tf_map = {
        "1Min": "1Min",
        "5Min": "5Min",
        "15Min": "15Min",
        "1Hour": "1Hour",
        "1Day": "1Day",
    }
    tf = tf_map.get(timeframe, timeframe)
    timeout_v = clamp_timeout(timeout)
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            fn = getattr(client, "get_stock_bars", None) or client.get_bars
            params = {
                "timeframe": tf,
                "limit": limit,
                "adjust": adjust,
                "include_extended": include_extended,
            }
            params.update(kwargs)
            try:
                resp = fn(
                    symbol,
                    start_s,
                    end_s,
                    timeout=timeout_v,
                    **{k: v for k, v in params.items() if v is not None},
                )
            except TypeError:
                resp = fn(
                    symbol,
                    start_s,
                    end_s,
                    **{k: v for k, v in params.items() if v is not None},
                )
            if pd is not None and isinstance(resp, pd.DataFrame):
                if resp.index.tz is None:
                    resp.index = resp.index.tz_localize("UTC")
                else:
                    resp.index = resp.index.tz_convert("UTC")
            return resp
        except Exception as e:  # pragma: no cover - network errors mocked in tests
            last_exc = e
            if _is_persistent_datetime_error(e) and attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_SLEEP_S)
    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to fetch bars")


def get_minute_df(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    limit: int | None = None,
    **kwargs: Any,
) -> Any:
    """Fetch 1-minute bars."""
    return get_bars(
        symbol,
        start,
        end,
        timeframe="1Min",
        limit=limit,
        **kwargs,
    )


def get_historical_data(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    timeframe: str = "1Day",
    limit: int | None = None,
    **kwargs: Any,
) -> Any:
    """Fetch historical bars using the patchable client."""
    return get_bars(
        symbol,
        start,
        end,
        timeframe=timeframe,
        limit=limit,
        **kwargs,
    )
