"""
Test UTC timestamp formatting utilities.

Tests the ai_trading.utils.timefmt module to ensure proper UTC timestamp
formatting without double "Z" suffixes and ISO-8601 compliance.
"""

import pytest
from datetime import datetime, timezone, timedelta
from ai_trading.utils.timefmt import (
    utc_now_iso,
    format_datetime_utc,
    parse_iso_utc,
    ensure_utc_format
)


class TestUTCTimestampFormatting:
    """Test UTC timestamp formatting functions."""
    
    def test_utc_now_iso_format(self):
        """Test that utc_now_iso() returns properly formatted timestamp."""
        timestamp = utc_now_iso()
        
        # Should end with single Z
        assert timestamp.endswith('Z'), "Timestamp should end with Z"
        assert timestamp.count('Z') == 1, "Timestamp should have exactly one Z"
        
        # Should be parseable
        parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        assert parsed.tzinfo == timezone.utc
    
    def test_utc_now_iso_is_recent(self):
        """Test that utc_now_iso() returns a recent timestamp."""
        before = datetime.now(timezone.utc)
        timestamp_str = utc_now_iso()
        after = datetime.now(timezone.utc)
        
        # Parse the timestamp
        timestamp = parse_iso_utc(timestamp_str)
        assert timestamp is not None
        
        # Should be between before and after
        assert before <= timestamp <= after
    
    def test_format_datetime_utc_with_utc_datetime(self):
        """Test formatting UTC datetime."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = format_datetime_utc(dt)
        
        assert result == "2024-01-01T12:00:00Z"
        assert result.count('Z') == 1
    
    def test_format_datetime_utc_with_naive_datetime(self):
        """Test formatting naive datetime (assumed UTC)."""
        dt = datetime(2024, 1, 1, 12, 0, 0)  # No timezone
        result = format_datetime_utc(dt)
        
        assert result == "2024-01-01T12:00:00Z"
        assert result.count('Z') == 1
    
    def test_format_datetime_utc_with_non_utc_datetime(self):
        """Test formatting datetime with non-UTC timezone."""
        # Create EST timezone (UTC-5)
        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 1, 7, 0, 0, tzinfo=est)  # 7 AM EST = 12 PM UTC
        result = format_datetime_utc(dt)
        
        assert result == "2024-01-01T12:00:00Z"
        assert result.count('Z') == 1
    
    def test_format_datetime_utc_with_none(self):
        """Test formatting None datetime returns current time."""
        result = format_datetime_utc(None)
        
        # Should be a valid timestamp
        assert result.endswith('Z')
        assert result.count('Z') == 1
        
        # Should be recent
        parsed = parse_iso_utc(result)
        now = datetime.now(timezone.utc)
        assert abs((now - parsed).total_seconds()) < 5  # Within 5 seconds
    
    def test_parse_iso_utc_with_z_suffix(self):
        """Test parsing ISO timestamp with Z suffix."""
        timestamp_str = "2024-01-01T12:00:00Z"
        result = parse_iso_utc(timestamp_str)
        
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12
        assert result.minute == 0
        assert result.second == 0
        assert result.tzinfo == timezone.utc
    
    def test_parse_iso_utc_with_offset(self):
        """Test parsing ISO timestamp with +00:00 offset."""
        timestamp_str = "2024-01-01T12:00:00+00:00"
        result = parse_iso_utc(timestamp_str)
        
        assert result is not None
        assert result.year == 2024
        assert result.tzinfo == timezone.utc
    
    def test_parse_iso_utc_with_invalid_format(self):
        """Test parsing invalid timestamp returns None."""
        assert parse_iso_utc("invalid") is None
        assert parse_iso_utc("") is None
        assert parse_iso_utc(None) is None
    
    def test_ensure_utc_format_fixes_double_z(self):
        """Test that ensure_utc_format fixes double Z suffixes."""
        result = ensure_utc_format("2024-01-01T12:00:00ZZ")
        assert result == "2024-01-01T12:00:00Z"
        assert result.count('Z') == 1
    
    def test_ensure_utc_format_fixes_offset(self):
        """Test that ensure_utc_format converts +00:00 to Z."""
        result = ensure_utc_format("2024-01-01T12:00:00+00:00")
        assert result == "2024-01-01T12:00:00Z"
        assert result.count('Z') == 1
    
    def test_ensure_utc_format_with_multiple_z(self):
        """Test that ensure_utc_format handles multiple Z suffixes."""
        result = ensure_utc_format("2024-01-01T12:00:00ZZZZ")
        assert result == "2024-01-01T12:00:00Z"
        assert result.count('Z') == 1
    
    def test_ensure_utc_format_with_empty_string(self):
        """Test that ensure_utc_format handles empty string."""
        result = ensure_utc_format("")
        
        # Should return current time
        assert result.endswith('Z')
        assert result.count('Z') == 1
    
    def test_roundtrip_formatting(self):
        """Test that format->parse->format is consistent."""
        original_dt = datetime(2024, 6, 15, 14, 30, 45, tzinfo=timezone.utc)
        
        # Format to string
        formatted = format_datetime_utc(original_dt)
        
        # Parse back to datetime
        parsed = parse_iso_utc(formatted)
        
        # Format again
        reformatted = format_datetime_utc(parsed)
        
        # Should be identical
        assert formatted == reformatted
        assert parsed == original_dt
    
    def test_no_double_z_in_any_function(self):
        """Comprehensive test that no function produces double Z."""
        # Test utc_now_iso
        now_iso = utc_now_iso()
        assert 'ZZ' not in now_iso
        assert now_iso.count('Z') == 1
        
        # Test format_datetime_utc with various inputs
        dt_utc = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        formatted_utc = format_datetime_utc(dt_utc)
        assert 'ZZ' not in formatted_utc
        assert formatted_utc.count('Z') == 1
        
        dt_naive = datetime(2024, 1, 1, 12, 0, 0)
        formatted_naive = format_datetime_utc(dt_naive)
        assert 'ZZ' not in formatted_naive
        assert formatted_naive.count('Z') == 1
        
        # Test ensure_utc_format with problematic input
        cleaned = ensure_utc_format("2024-01-01T12:00:00ZZ")
        assert 'ZZ' not in cleaned
        assert cleaned.count('Z') == 1
    
    def test_microseconds_handling(self):
        """Test handling of microseconds in timestamps."""
        dt = datetime(2024, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
        formatted = format_datetime_utc(dt)
        
        # Should include microseconds
        assert ".123456" in formatted
        assert formatted.endswith('Z')
        assert formatted.count('Z') == 1
        
        # Should be parseable
        parsed = parse_iso_utc(formatted)
        assert parsed.microsecond == 123456


if __name__ == "__main__":
    pytest.main([__file__])