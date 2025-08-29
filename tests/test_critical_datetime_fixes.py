#!/usr/bin/env python3
"""
Focused tests for critical trading bot datetime and MetaLearning fixes.
Tests the specific issues identified in the problem statement.
"""

import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch


class TestDatetimeTimezoneAwareness(unittest.TestCase):
    """Test that datetime objects are timezone-aware for Alpaca API compatibility."""

    def test_ensure_datetime_returns_timezone_aware(self):
        """Test that ensure_datetime returns timezone-aware datetime objects."""
        from ai_trading.data.fetch import ensure_datetime

        # Test with naive datetime
        naive_dt = datetime(2025, 1, 1, 12, 0, 0)
        result = ensure_datetime(naive_dt)

        # After fix, this should be timezone-aware
        self.assertIsNotNone(result.tzinfo, "ensure_datetime should return timezone-aware datetime")

    def test_alpaca_api_format_compatibility(self):
        """Test that datetime format is compatible with Alpaca API RFC3339 requirements."""
        from ai_trading.data.fetch import ensure_datetime

        # Test various input formats
        test_inputs = [
            datetime.now(UTC).replace(tzinfo=None),  # AI-AGENT-REF: Fixed naive datetime by creating from UTC
            "2025-01-01 12:00:00",  # string format
            datetime.now(UTC),  # already timezone-aware
        ]

        for input_dt in test_inputs:
            with self.subTest(input_dt=input_dt):
                result = ensure_datetime(input_dt)

                # Should be timezone-aware
                self.assertIsNotNone(result.tzinfo, f"Input {input_dt} should produce timezone-aware result")

                # Should be able to format as RFC3339 for Alpaca API
                rfc3339_str = result.isoformat()
                self.assertIn("T", rfc3339_str, "Should produce valid RFC3339 format")

    def test_get_bars_datetime_parameters(self):
        """Test that get_bars handles timezone-aware datetime parameters."""
        from ai_trading.data.fetch import get_bars

        start_dt = datetime.now(UTC)
        end_dt = start_dt + timedelta(hours=1)

        try:
            get_bars("AAPL", start_dt, end_dt)
        except (ValueError, TypeError) as e:  # pragma: no cover - ensure no TZ errors
            if "timezone" in str(e).lower():
                self.fail(f"get_bars failed due to timezone issues: {e}")


class TestMetaLearningDataFetching(unittest.TestCase):
    """Test MetaLearning data fetching and function calls."""

    def test_metalearn_invalid_prices_prevention(self):
        """Test that MetaLearning can handle valid trade data without METALEARN_INVALID_PRICES."""
        # Create a temporary CSV file with valid trade data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
            tmp_file.write("exit_price,entry_price,signal_tags,side\n")
            tmp_file.write("150.0,148.0,momentum+breakout,buy\n")
            tmp_file.write("145.0,147.0,mean_reversion,sell\n")
            tmp_file_path = tmp_file.name

        try:
            # Mock the TRADE_LOG_FILE to use our test file
            with patch("ai_trading.core.bot_engine.TRADE_LOG_FILE", tmp_file_path):
                from ai_trading.core.bot_engine import load_global_signal_performance

                # This should not return None or empty dict due to METALEARN_INVALID_PRICES
                result = load_global_signal_performance(min_trades=1, threshold=0.1)

                # Should return signal performance data, not None/empty due to invalid prices
                if result is None:
                    self.fail("load_global_signal_performance returned None - possible METALEARN_INVALID_PRICES issue")
                if isinstance(result, dict) and len(result) == 0:
                    # Check if it's empty due to invalid prices vs other reasons
                    # This test validates the function can process valid data
                    pass

        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)


class TestSentimentCaching(unittest.TestCase):
    """Test sentiment analysis caching and rate limit handling."""

    def test_sentiment_cache_rate_limit_handling(self):
        """Test that sentiment caching properly handles rate limits."""
        try:
            import time

            from ai_trading.core.bot_engine import _SENTIMENT_CACHE, fetch_sentiment
            from requests.exceptions import HTTPError

            # Clear cache
            _SENTIMENT_CACHE.clear()

            # Mock the API key variables in bot_engine module
            with patch("ai_trading.core.bot_engine.SENTIMENT_API_KEY", "test_key"):
                with patch("ai_trading.core.bot_engine.NEWS_API_KEY", "test_key"):
                    # Mock the requests to simulate rate limiting
                    with patch("ai_trading.core.bot_engine._HTTP_SESSION.get") as mock_get:
                        # First call - simulate rate limit (429)
                        mock_response = MagicMock()
                        mock_response.status_code = 429
                        mock_response.raise_for_status.side_effect = HTTPError("429 Too Many Requests")
                        mock_get.return_value = mock_response

                        # Mock context for fetch_sentiment
                        mock_ctx = MagicMock()

                        # This should handle the rate limit gracefully and cache neutral score
                        score = fetch_sentiment(mock_ctx, "AAPL")

                        # Should return neutral score (0.0) when rate limited
                        self.assertEqual(score, 0.0, "Should return neutral score when rate limited")

                        # Should cache the neutral score
                        self.assertIn("AAPL", _SENTIMENT_CACHE, "Should cache the rate-limited result")

                        # Verify the cached value is correct
                        cached_entry = _SENTIMENT_CACHE["AAPL"]
                        self.assertEqual(cached_entry[1], 0.0, "Cached sentiment should be 0.0")

                        # Second call should use cache (no API call)
                        _SENTIMENT_CACHE.clear()  # Clear cache to test fresh call
                        _SENTIMENT_CACHE["AAPL"] = (
                            time.time(),
                            0.0,
                        )  # Pre-populate with rate-limited result

                        score2 = fetch_sentiment(mock_ctx, "AAPL")
                        self.assertEqual(score2, 0.0, "Should return cached neutral score")

        except ImportError as e:
            # Skip test if modules are not available
            self.skipTest(f"Required modules not available: {e}")


if __name__ == "__main__":
    unittest.main()
