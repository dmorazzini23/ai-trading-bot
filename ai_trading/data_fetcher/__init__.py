from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime

import pandas as pd

# Availability flags -------------------------------------------------------
try:  # pragma: no cover - optional provider
    import finnhub  # type: ignore  # noqa: F401

    FINNHUB_AVAILABLE = True
except Exception:  # noqa: BLE001, pragma: no cover
    FINNHUB_AVAILABLE = False


def ensure_datetime(dt_like) -> datetime:
    """Return a timezone-aware (UTC) datetime."""
    if isinstance(dt_like, datetime):
        return dt_like if dt_like.tzinfo else dt_like.replace(tzinfo=UTC)
    try:
        dt = pd.to_datetime(dt_like, utc=True).to_pydatetime()
        if isinstance(dt, datetime):
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except Exception:  # noqa: BLE001
        pass
    raise ValueError(f"Unsupported datetime-like value: {dt_like!r}")


def get_bars(
    symbol: str,
    timeframe: str,
    start,
    end,
    feed: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV bars for ``symbol``. Safe default returns empty DataFrame."""
    _ = ensure_datetime(start), ensure_datetime(end)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return pd.DataFrame(columns=cols)


def get_bars_batch(
    symbols: Iterable[str],
    timeframe: str,
    start,
    end,
    feed: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Batch fetch bars -> {symbol: DataFrame}."""
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            out[str(s)] = get_bars(str(s), timeframe, start, end, feed=feed)
        except Exception:  # noqa: BLE001
            out[str(s)] = pd.DataFrame()
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
