import datetime as dt
from zoneinfo import ZoneInfo
import pytest

from ai_trading import data_fetcher


class TestDataFetcherAPI:
    def test_data_fetch_error_alias(self):
        """DataFetchError is exposed and matches the legacy alias."""
        assert data_fetcher.DataFetchException is data_fetcher.DataFetchError

    def test_ensure_datetime_handles_callable(self):
        """ensure_datetime accepts callables returning datetimes."""
        expected = dt.datetime(2024, 1, 1, 12, tzinfo=ZoneInfo("UTC"))

        def factory():
            return expected

        result = data_fetcher.ensure_datetime(factory)
        assert result == expected

    def test_ensure_datetime_coerces_naive(self):
        """Naive datetimes are treated as America/New_York before UTC conversion."""
        naive = dt.datetime(2024, 1, 1, 9, 30)
        result = data_fetcher.ensure_datetime(naive)
        expected = naive.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(ZoneInfo("UTC"))
        assert result == expected

    def test_ensure_datetime_invalid_input(self):
        """Non-datetime values raise TypeError."""
        with pytest.raises(TypeError):
            data_fetcher.ensure_datetime("not-a-date")
