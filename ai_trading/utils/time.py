"""Time utilities for timezone-aware datetime operations."""

from datetime import datetime, timezone


def now_utc():
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)