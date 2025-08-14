import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_trading.data_fetcher as data_fetcher


@pytest.mark.parametrize("value,expected", [
    (
        datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    ),
    ("2024-01-01", datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
    ("2024-01-01 05:30:00", datetime(2024, 1, 1, 5, 30, 0, tzinfo=timezone.utc)),
    ("20240101", datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
    ("2024-01-01T12:00:00Z", datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)),
])
def test_ensure_datetime_valid(value, expected):
    result = data_fetcher.ensure_datetime(value)
    assert result == expected


def test_ensure_datetime_none():
    with pytest.raises(ValueError):
        data_fetcher.ensure_datetime(None)


def test_ensure_datetime_empty_str():
    with pytest.raises(ValueError):
        data_fetcher.ensure_datetime("")


def test_ensure_datetime_invalid_str():
    with pytest.raises(ValueError):
        data_fetcher.ensure_datetime("notadate")


def test_ensure_datetime_pandas_timestamp():
    ts = pd.Timestamp("2024-02-02T15:00:00", tz="UTC")
    result = data_fetcher.ensure_datetime(ts)
    assert result == ts.to_pydatetime()


def test_ensure_datetime_nat():
    with pytest.raises(ValueError):
        data_fetcher.ensure_datetime(pd.NaT)


def test_ensure_datetime_bad_type():
    with pytest.raises(TypeError):
        data_fetcher.ensure_datetime(123)
