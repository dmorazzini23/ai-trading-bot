from __future__ import annotations
import time as _time_module
from datetime import UTC, datetime, timedelta, tzinfo, date as _date
from time import time as _time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_trading.utils.lazy_imports import load_pandas

# Simple day-scoped cache for last_market_session to avoid repeated calendar work
_LAST_SESSION_CACHE: dict[str, "SessionWindow | None"] = {}
_LAST_MONOTONIC_VALUE: float | None = None

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import pandas as pd

def safe_utcnow(tz: tzinfo | None = UTC) -> datetime:
    """Return an aware UTC ``datetime`` resilient to Freezegun edge cases.

    Freezegun can raise ``IndexError`` when multiple threads call
    :func:`datetime.now` while the clock is frozen.  This helper catches that
    specific failure mode and falls back to building a timestamp from
    ``time.time()`` so worker threads keep running.
    """

    try:
        now = datetime.now(UTC)
    except IndexError:  # pragma: no cover - depends on freezegun behaviour
        now = datetime.fromtimestamp(_time(), UTC)
    return now if tz in (UTC, None) else now.astimezone(tz)


def utcnow(tz: tzinfo | None = UTC) -> datetime:
    """Repository-standard aware now with optional timezone."""

    return safe_utcnow(tz)

now_utc = utcnow

@dataclass
class SessionWindow:
    open: pd.Timestamp
    close: pd.Timestamp

def last_market_session(now: pd.Timestamp) -> SessionWindow | None:
    """Return previous market session window for NYSE.

    Returns ``None`` if pandas or market calendars are unavailable.
    """
    pd = load_pandas()
    if pd is None:
        return None
    try:
        from ai_trading.market.calendars import get_calendar_registry
    except ImportError:  # calendars package missing
        return None
    cal = get_calendar_registry()
    current: _date = now.tz_convert('UTC').date()
    key = current.isoformat()
    cached = _LAST_SESSION_CACHE.get(key)
    if cached is not None:
        return cached
    for _ in range(10):
        start, end = cal.get_session_bounds('SPY', current)
        if start and end and (end <= now.to_pydatetime()):
            win = SessionWindow(pd.Timestamp(start).tz_convert('UTC'), pd.Timestamp(end).tz_convert('UTC'))
            _LAST_SESSION_CACHE[key] = win
            return win
        current -= timedelta(days=1)
    _LAST_SESSION_CACHE[key] = None
    return None

def monotonic_time() -> float:
    """Return a monotonic timestamp with a realtime fallback.

    Some minimal Python runtimes (or constrained testing environments) may lack
    :func:`time.monotonic`.  This helper performs a best-effort lookup each
    call, falling back to :func:`time.time` when ``monotonic`` is unavailable or
    raises ``RuntimeError`` (possible when the underlying clock is not ready).
    """

    global _LAST_MONOTONIC_VALUE

    monotonic = getattr(_time_module, "monotonic", None)
    if monotonic is not None:
        try:
            value = float(monotonic())
        except StopIteration:  # pragma: no cover - patched generators may exhaust
            value = None
        except RuntimeError:  # pragma: no cover - platform specific or patched generator stop
            value = None
        except Exception:  # pragma: no cover - defensive: patched clocks may raise other errors
            value = None
        else:
            _LAST_MONOTONIC_VALUE = value
            return value

    fallback = float(_time_module.time())
    if _LAST_MONOTONIC_VALUE is not None and fallback < _LAST_MONOTONIC_VALUE:
        return _LAST_MONOTONIC_VALUE

    _LAST_MONOTONIC_VALUE = fallback
    return fallback


__all__ = ['safe_utcnow', 'utcnow', 'now_utc', 'SessionWindow', 'last_market_session', 'monotonic_time']
