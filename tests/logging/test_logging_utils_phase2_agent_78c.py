from __future__ import annotations

import datetime as dt
import io
import itertools
import json
import logging
import sys
from enum import Enum
from types import MappingProxyType, SimpleNamespace
from uuid import UUID

import pytest

import ai_trading.logging as logmod
import ai_trading.utils.base as base


class _Side(Enum):
    BUY = "buy"


class _FakeLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[object, ...], dict[str, object]]] = []

    def debug(self, msg: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("debug", msg, args, kwargs))

    def info(self, msg: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("info", msg, args, kwargs))

    def warning(self, msg: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("warning", msg, args, kwargs))

    def error(self, msg: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("error", msg, args, kwargs))

    def critical(self, msg: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("critical", msg, args, kwargs))

    def log(self, level: int, msg: str, *args: object, **kwargs: object) -> None:
        self.calls.append((logging.getLevelName(level).lower(), msg, args, kwargs))


def _record(
    name: str = "ai_trading.tests.logging",
    msg: str = "event",
    *,
    level: int = logging.INFO,
) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, msg, (), None)


def test_sanitize_adapter_formatter_and_emit_once(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = logmod.sanitize_extra(
        {
            "name": "reserved",
            "api_key": "abc123",
            "has_secret": True,
            "nested_secret": "hidden",
            "DATABASE_URL": "postgresql://user:pass@example/db",
            "authorization": "Bearer abc123",
            "connection_string": "postgresql://user:pass@example/db",
            "dsn": "postgresql://user:pass@example/db",
            "plain": 7,
        }
    )
    assert payload == {
        "x_name": "reserved",
        "api_key": logmod._ENV_MASK,
        "has_secret": True,
        "nested_secret": logmod._ENV_MASK,
        "DATABASE_URL": logmod._ENV_MASK,
        "authorization": logmod._ENV_MASK,
        "connection_string": logmod._ENV_MASK,
        "dsn": logmod._ENV_MASK,
        "plain": 7,
    }

    adapter = logmod.SanitizingLoggerAdapter(logging.getLogger("ai_trading.tests.adapter"), {})
    msg, kwargs = adapter.process("hello", {"extra": {"msg": "reserved", "secret": "hidden"}})
    assert msg == "hello"
    assert kwargs["extra"]["x_msg"] == "reserved"
    assert kwargs["extra"]["secret"] == logmod._ENV_MASK
    assert adapter.handlers == (adapter.logger.handlers or logging.getLogger().handlers)

    record = _record(msg="compact")
    record.bot_phase = "TEST"
    record.present = True
    record.timestamp = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    record.exc_info = (ValueError, ValueError("bad"), None)
    formatted = json.loads(logmod.CompactJsonFormatter("%Y-%m-%dT%H:%M:%SZ").format(record))
    assert formatted["msg"] == "compact"
    assert formatted["bot_phase"] == "TEST"
    assert formatted["present"] is True
    assert formatted["exc"] == "ValueError: bad"

    fake = _FakeLogger()
    once = logmod.EmitOnceLogger(fake)
    days = iter(
        [
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 2),
        ]
    )
    monkeypatch.setattr(logmod, "_utc_today", lambda: next(days))
    once.info("same", key="k")
    once.info("same", key="k")
    once.info("same", key="k")
    once.debug("debug")
    once.warning("warning")
    once.error("error")
    once.critical("critical")
    assert [call[0] for call in fake.calls] == [
        "info",
        "info",
        "debug",
        "warning",
        "error",
        "critical",
    ]


def test_message_and_rate_limit_throttling_paths(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    throttle = logmod.MessageThrottleFilter(throttle_seconds=10)
    times = iter([0.0, 1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(throttle, "_now", lambda: next(times))

    first = _record("ai_trading.tests.throttle", "STAGE_TIMING")
    first.stage = "fetch"
    first.elapsed_ms = "12.8"
    assert throttle.filter(first) is True
    assert first.getMessage() == "STAGE_TIMING | stage=fetch ms=12.8"

    for _ in range(3):
        repeat = _record("ai_trading.tests.throttle", "STAGE_TIMING")
        repeat.stage = "fetch"
        repeat.elapsed_ms = "12.8"
        assert throttle.filter(repeat) is False

    with caplog.at_level(logging.INFO, logger="ai_trading.tests.throttle"):
        throttle.flush_cycle(namespace="ai_trading.tests")
    assert any("LOG_THROTTLE_SUMMARY" in rec.getMessage() for rec in caplog.records)
    assert logmod.MessageThrottleFilter._quote_message('a "quote"') == '"a \\"quote\\""'
    assert logmod.MessageThrottleFilter(throttle_seconds=0).filter(_record()) is True

    tracker = logmod.RateLimitedEventTracker(window_seconds=2)
    monotonic = itertools.chain([10.0, 10.5, 11.0, 12.1], itertools.repeat(12.1))
    monkeypatch.setattr(logmod, "_monotonic_time", lambda: next(monotonic))
    assert tracker.record("Fetch attempt, noisy", logger_name="ai_trading.tests.rate")[0] is True
    assert tracker.record(
        "Fetch attempt, noisy",
        logger_name="ai_trading.tests.rate",
        extra={"symbol": "AAPL", "timeframe": "1Min"},
    )[0] is False
    assert tracker.record("Fetch attempt, noisy", logger_name="ai_trading.tests.rate")[0] is False
    should_log, summaries = tracker.record("Fetch attempt, noisy", logger_name="ai_trading.tests.rate")
    assert should_log is True
    assert summaries[0].key == "FETCH_ATTEMPT"
    assert summaries[0].suppressed == 2
    assert tracker.flush(force=True) == []

    logmod._SUMMARY_LAST_EMIT.clear()
    summary = logmod.RateLimitedSummary(
        key="manual key",
        suppressed=3,
        logger_name="ai_trading.tests.rate",
        window_s=2.0,
        sample_symbol="MSFT",
        sample_feed="1Day",
    )
    with caplog.at_level(logging.INFO, logger="ai_trading.tests.rate"):
        logmod._emit_rate_limit_summaries([summary])
        logmod._emit_rate_limit_summaries([summary])
    summary_messages = [rec.getMessage() for rec in caplog.records if "manual key" in rec.getMessage()]
    assert len(summary_messages) == 1
    assert 'sample_symbol="MSFT"' in summary_messages[0]










