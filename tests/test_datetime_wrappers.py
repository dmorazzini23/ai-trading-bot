from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.data.fetch import ensure_datetime


def test_naive_et_is_converted_to_utc():
    # AI-AGENT-REF: naive ET → UTC conversion
    et_naive = datetime(2025, 8, 20, 9, 30)  # intended ET naive
    dt_utc = ensure_datetime(et_naive)
    assert dt_utc.tzinfo is UTC
    assert dt_utc.hour == 13 and dt_utc.minute == 30


def test_callable_is_unwrapped():
    # AI-AGENT-REF: callable handling
    dt_utc = ensure_datetime(lambda: datetime(2025, 8, 20, 9, 30))
    assert dt_utc.tzinfo is UTC

