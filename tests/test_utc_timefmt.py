"""
Test UTC timestamp formatting utilities.

Tests the ai_trading.utils.timefmt module to ensure proper UTC timestamp
formatting without double "Z" suffixes and ISO-8601 compliance.
"""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from ai_trading.utils.timefmt import utc_now_iso


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
        assert parsed.tzinfo == UTC

















if __name__ == "__main__":
    pytest.main([__file__])
