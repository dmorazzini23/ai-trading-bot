"""
UTC timestamp formatting utilities.

This module provides standardized UTC timestamp formatting to fix the double "Z"
issue and ensure consistent ISO-8601 format compliance throughout the application.
"""
from datetime import UTC, datetime

def utc_now_iso_from(now: datetime) -> str:
    """Format provided datetime as ISO-8601 UTC string."""
    dt = now.astimezone(UTC)
    return dt.isoformat().replace('+00:00', 'Z')

def utc_now_iso() -> str:
    """
    Generate a properly formatted UTC timestamp in ISO-8601 format.

    Returns a timestamp string with a single trailing 'Z' to indicate UTC timezone.
    This fixes the issue where double "ZZ" suffixes were being generated.

    Returns:
        str: ISO-8601 formatted UTC timestamp with single 'Z' suffix

    Examples:
        >>> timestamp = utc_now_iso()
        >>> timestamp.endswith('Z')
        True
        >>> timestamp.count('Z')
        1
    """
    now = datetime.now(UTC)
    return now.isoformat().replace('+00:00', 'Z')

