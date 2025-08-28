"""Tests for data staleness guard functionality."""
import datetime
from unittest.mock import Mock, patch

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.guards.staleness import ensure_data_fresh, _ensure_data_fresh


class TestStalenessGuard:
    """Test data staleness validation functionality."""

    def test_staleness_guard_fresh_data(self):
        """Fresh data should not raise."""
        now = datetime.datetime.now(datetime.UTC)
        fresh_ts = now - datetime.timedelta(seconds=30)
        df = pd.DataFrame({"timestamp": [fresh_ts], "close": [100.0]})
        mock_fetcher = Mock()
        mock_fetcher.get_minute_df = Mock(return_value=df)
        ensure_data_fresh(mock_fetcher, ["AAPL"], max_age_seconds=300)

    def test_staleness_guard_stale_data(self):
        """Stale data should raise runtime error."""
        now = datetime.datetime.now(datetime.UTC)
        stale_ts = now - datetime.timedelta(seconds=600)
        df = pd.DataFrame({"timestamp": [stale_ts], "close": [100.0]})
        mock_fetcher = Mock()
        mock_fetcher.get_minute_df = Mock(return_value=df)
        with pytest.raises(RuntimeError, match="Stale data for symbols"):
            ensure_data_fresh(mock_fetcher, ["AAPL"], max_age_seconds=300)

    def test_staleness_guard_no_data(self):
        """None from fetcher should raise."""
        mock_fetcher = Mock()
        mock_fetcher.get_minute_df = Mock(return_value=None)
        with pytest.raises(RuntimeError, match="Stale data for symbols"):
            ensure_data_fresh(mock_fetcher, ["AAPL"], max_age_seconds=300)

    def test_staleness_guard_empty_dataframe(self):
        """Empty dataframe should raise."""
        mock_fetcher = Mock()
        mock_fetcher.get_minute_df = Mock(return_value=pd.DataFrame())
        with pytest.raises(RuntimeError, match="Stale data for symbols"):
            ensure_data_fresh(mock_fetcher, ["AAPL"], max_age_seconds=300)

    def test_staleness_guard_multiple_symbols(self):
        """Mix of fresh, stale, and missing symbols should report all."""
        now = datetime.datetime.now(datetime.UTC)

        def mock_get_minute_df(symbol, start, end):
            if symbol == "AAPL":
                fresh_ts = now - datetime.timedelta(seconds=30)
                return pd.DataFrame({"timestamp": [fresh_ts], "close": [150.0]})
            if symbol == "MSFT":
                stale_ts = now - datetime.timedelta(seconds=600)
                return pd.DataFrame({"timestamp": [stale_ts], "close": [300.0]})
            return None

        mock_fetcher = Mock()
        mock_fetcher.get_minute_df = Mock(side_effect=mock_get_minute_df)
        with pytest.raises(RuntimeError) as exc_info:
            ensure_data_fresh(
                mock_fetcher,
                ["AAPL", "MSFT", "GOOGL"],
                max_age_seconds=300,
            )
        msg = str(exc_info.value)
        assert "MSFT" in msg
        assert "GOOGL" in msg

    def test_staleness_guard_utc_logging(self):
        """Ensure logging uses UTC timestamps."""
        with patch("ai_trading.guards.staleness.logger") as mock_logger:
            now = datetime.datetime.now(datetime.UTC)
            fresh_ts = now - datetime.timedelta(seconds=30)
            df = pd.DataFrame({"timestamp": [fresh_ts], "close": [100.0]})
            _ensure_data_fresh(df, 300, symbol="AAPL", now=now)
            mock_logger.debug.assert_called()
            args = mock_logger.debug.call_args[0]
            assert "UTC now=" in args[0]
            assert "T" in args[1]

    def test_staleness_guard_timezone_handling(self):
        """Handle naive and aware timestamps."""
        now = datetime.datetime.now(datetime.UTC)
        naive_ts = (now - datetime.timedelta(seconds=30)).replace(tzinfo=None)
        aware_ts = now - datetime.timedelta(seconds=30)
        df_naive = pd.DataFrame({"timestamp": [naive_ts], "close": [100.0]})
        df_aware = pd.DataFrame({"timestamp": [aware_ts], "close": [100.0]})
        for df in (df_naive, df_aware):
            _ensure_data_fresh(df, 300, symbol="AAPL", now=now)

    def test_staleness_guard_error_handling(self):
        """Fetcher exceptions should propagate details."""
        mock_fetcher = Mock()
        mock_fetcher.get_minute_df = Mock(side_effect=Exception("Network error"))
        with pytest.raises(RuntimeError) as exc_info:
            ensure_data_fresh(mock_fetcher, ["AAPL"], max_age_seconds=300)
        assert "error=" in str(exc_info.value)
