from datetime import datetime, date, timezone

import pytest

from ai_trading.data.timeutils import ensure_utc_datetime


def test_reject_callable():
    with pytest.raises(TypeError):
        ensure_utc_datetime(lambda: None)


def test_datetime_naive_to_utc():
    dt = ensure_utc_datetime(datetime(2025, 8, 20, 12, 0, 0))
    assert dt.tzinfo == timezone.utc


def test_date_to_utc_midnight():
    dt = ensure_utc_datetime(date(2025, 8, 20))
    assert dt.tzinfo == timezone.utc and dt.hour == 0
