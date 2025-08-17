from __future__ import annotations

import importlib.util as _iu
import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime

import pandas as pd

# Robust availability checks (lazy, no import side effects)
FINNHUB_AVAILABLE = _iu.find_spec("finnhub") is not None
YFIN_AVAILABLE = _iu.find_spec("yfinance") is not None


def ensure_datetime(dt) -> datetime:
    """Return a timezone-aware UTC datetime for provider calls."""  # AI-AGENT-REF
    if isinstance(dt, datetime):
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    try:
        return datetime.fromtimestamp(float(dt), tz=UTC)
    except Exception:  # noqa: BLE001
        return datetime.now(UTC)


def get_bars(
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    feed=None,
    client=None,
) -> pd.DataFrame:
    """Safe fetcher returning empty frame if provider missing."""  # AI-AGENT-REF
    if start is not None:
        start = ensure_datetime(start)
    if end is not None:
        end = ensure_datetime(end)
    try:
        if client and hasattr(client, "get_bars"):
            return (
                client.get_bars(symbol, timeframe, start, end, feed=feed)
                or pd.DataFrame()
            )
    except Exception:  # noqa: BLE001
        pass
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return pd.DataFrame(columns=cols)


def get_bars_batch(
    symbols: Iterable[str],
    timeframe: str,
    start: datetime | None,
    end: datetime | None,
    *,
    feed=None,
    client=None,
) -> dict[str, pd.DataFrame]:
    """Compatibility batch fetcher used by legacy code; never raises."""  # AI-AGENT-REF
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[str(sym)] = get_bars(sym, timeframe, start, end, feed=feed, client=client)
    return out


# --- minute-bar cache (thread-safe, import-safe) -------------------------
_MINUTE_CACHE_LOCK = threading.RLock()
# symbol -> (last_minute_epoch_seconds, inserted_epoch_seconds)
_MINUTE_CACHE: dict[str, tuple[int, float]] = {}


def get_cached_minute_timestamp(symbol: str) -> int | None:
    with _MINUTE_CACHE_LOCK:
        tup = _MINUTE_CACHE.get(symbol.upper())
        return tup[0] if tup else None


def set_cached_minute_timestamp(symbol: str, ts: int) -> None:
    with _MINUTE_CACHE_LOCK:
        _MINUTE_CACHE[symbol.upper()] = (int(ts), time.time())


def clear_minute_cache(symbol: str | None = None) -> None:
    with _MINUTE_CACHE_LOCK:
        if symbol is None:
            _MINUTE_CACHE.clear()
        else:
            _MINUTE_CACHE.pop(symbol.upper(), None)


def age_minute_cache(max_age_seconds: int) -> int:
    cutoff = time.time() - int(max_age_seconds)
    removed = 0
    with _MINUTE_CACHE_LOCK:
        for k in list(_MINUTE_CACHE.keys()):
            _, inserted = _MINUTE_CACHE[k]
            if inserted < cutoff:
                _MINUTE_CACHE.pop(k, None)
                removed += 1
    return removed


# Backwards compat aliases -------------------------------------------------
def clear_cached_minute_timestamp(
    symbol: str | None = None,
) -> None:  # pragma: no cover - legacy
    clear_minute_cache(symbol)


def age_cached_minute_timestamps(
    max_age_seconds: int,
) -> int:  # pragma: no cover - legacy
    return age_minute_cache(max_age_seconds)


# --- simple retry helper used by get_minute_df ---------------------------
def _retry(n: int, delay: float, fn, *args, **kwargs):
    last_err = None
    for _ in range(max(1, n)):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(delay)
    raise last_err  # type: ignore[misc]


retry = _retry  # AI-AGENT-REF: export retry helper


# --- bar fetcher (minimal, patchable) ------------------------------------
def get_minute_df(
    symbol: str,
    start: datetime | str | int | float | None = None,
    end: datetime | str | int | float | None = None,
    retries: int = 2,
    delay: float = 0.25,
):
    """Minimal, patchable fetcher used by tests."""
    s = ensure_datetime(start)
    e = ensure_datetime(end)
    return _retry(retries, delay, _fetch_bars_impl, symbol, s, e)


def _fetch_bars_impl(symbol: str, start: datetime, end: datetime):
    # Placeholder implementation; in production replaced/monkeypatched by tests.
    raise RuntimeError("No data provider configured")


__all__ = [
    "FINNHUB_AVAILABLE",
    "YFIN_AVAILABLE",
    "ensure_datetime",
    "get_bars",
    "get_bars_batch",
    "get_minute_df",
    "get_cached_minute_timestamp",
    "set_cached_minute_timestamp",
    "clear_minute_cache",
    "age_minute_cache",
    "retry",
]
