from __future__ import annotations

# Rate-limit and classify empty bar events.  # AI-AGENT-REF
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass
class _State:  # AI-AGENT-REF: track duplicate events
    count: int = 0
    last: datetime = datetime.min.replace(tzinfo=UTC)


_TTL_SECONDS = 60  # suppress duplicates per key within 60s
_store: dict[tuple[str, str, str, str, str], _State] = defaultdict(_State)


def classify(is_market_open: bool) -> int:
    """Return logging level for empty bars."""  # AI-AGENT-REF

    return logging.WARNING if is_market_open else logging.INFO


def should_emit(key: tuple[str, str, str, str, str], now: datetime) -> bool:
    """Determine if the event should be logged."""  # AI-AGENT-REF

    st = _store[key]
    if now - st.last >= timedelta(seconds=_TTL_SECONDS):
        return True
    return False


def record(key: tuple[str, str, str, str, str], now: datetime) -> int:
    """Record an occurrence and update state."""  # AI-AGENT-REF

    st = _store[key]
    st.count += 1
    st.last = now
    return st.count

