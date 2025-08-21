from __future__ import annotations

# Tests for empty bars logging policy.  # AI-AGENT-REF
from datetime import UTC, datetime

from ai_trading.logging.empty_policy import classify, record, should_emit


def test_classify_levels() -> None:
    assert classify(False) == 20  # INFO
    assert classify(True) == 30   # WARNING


def test_rate_limit_window() -> None:
    now = datetime(2025, 8, 20, 20, 0, tzinfo=UTC)
    key = ("SPY", "DAILY", now.date().isoformat(), "iex", "1Min")
    assert should_emit(key, now) is True
    c1 = record(key, now)
    assert c1 == 1
    assert should_emit(key, now) is False

