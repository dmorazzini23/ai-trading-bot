from __future__ import annotations
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

@dataclass
class _State:
    count: int = 0
    last: datetime = datetime.min.replace(tzinfo=UTC)

_TTL_SECONDS = 60
_store: dict[tuple[str, str, str, str, str], _State] = defaultdict(_State)

def classify(is_market_open: bool) -> int:
    """Return logging level for empty bars."""
    return logging.WARNING if is_market_open else logging.INFO

def should_emit(key: tuple[str, str, str, str, str], now: datetime, ttl: int = _TTL_SECONDS) -> bool:
    """Return True when the throttle window has passed."""
    st = _store[key]
    return now - st.last >= timedelta(seconds=ttl)

def record(key: tuple[str, str, str, str, str], now: datetime) -> int:
    """Record an occurrence and update state."""
    st = _store[key]
    st.count += 1
    st.last = now
    return st.count

__all__ = ["classify", "should_emit", "record"]
