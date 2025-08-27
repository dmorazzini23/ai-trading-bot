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

def format_datetime_utc(dt: datetime) -> str:
    """
    Format a datetime object as ISO-8601 UTC string.

    Args:
        dt: datetime object to format (will be converted to UTC if needed)

    Returns:
        str: ISO-8601 formatted UTC timestamp with single 'Z' suffix

    Examples:
        >>> from datetime import datetime, timezone
        >>> dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        >>> format_datetime_utc(dt)
        '2024-01-01T12:00:00Z'
    """
    if dt is None:
        return utc_now_iso()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    elif dt.tzinfo != UTC:
        dt = dt.astimezone(UTC)
    return dt.isoformat().replace('+00:00', 'Z')


def isoformat_z(dt: datetime) -> str:
    """Return RFC3339 timestamp with trailing ``Z``.

    Thin wrapper around :func:`format_datetime_utc` offering a clearer helper
    name for callers that expect the ``Z``-suffixed UTC representation.
    """
    return format_datetime_utc(dt)

def parse_iso_utc(timestamp_str: str) -> datetime | None:
    """
    Parse an ISO-8601 UTC timestamp string.

    Args:
        timestamp_str: ISO timestamp string to parse

    Returns:
        datetime: UTC datetime object, or None if parsing fails

    Examples:
        >>> dt = parse_iso_utc('2024-01-01T12:00:00Z')
        >>> dt.year
        2024
        >>> dt.tzinfo == timezone.utc
        True
    """
    if not timestamp_str:
        return None
    try:
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        return datetime.fromisoformat(timestamp_str).astimezone(UTC)
    except (ValueError, AttributeError):
        return None

def ensure_utc_format(timestamp: str) -> str:
    """
    Ensure a timestamp string has proper UTC formatting.

    This function cleans up timestamps that may have double Z suffixes
    or other formatting issues.

    Args:
        timestamp: Timestamp string to clean up

    Returns:
        str: Cleaned timestamp with single Z suffix

    Examples:
        >>> ensure_utc_format('2024-01-01T12:00:00ZZ')
        '2024-01-01T12:00:00Z'
        >>> ensure_utc_format('2024-01-01T12:00:00+00:00')
        '2024-01-01T12:00:00Z'
    """
    if not timestamp:
        return utc_now_iso()
    while timestamp.endswith('Z'):
        timestamp = timestamp[:-1]
    if timestamp.endswith('+00:00'):
        timestamp = timestamp[:-6]
    return timestamp + 'Z'