from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Repository-standard UTC now (timezone-aware)."""  # AI-AGENT-REF
    return datetime.now(UTC)


# Back-compat alias
now_utc = utcnow
