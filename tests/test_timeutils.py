from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from ai_trading.data.fetch import ensure_datetime
from ai_trading.data.timeutils import ensure_utc_datetime






def test_mixed_naive_datetime_semantics_are_intentional():
    naive = datetime(2025, 8, 20, 9, 30)

    assert ensure_utc_datetime(naive) == naive.replace(tzinfo=UTC)
    assert ensure_datetime(naive) == naive.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)
