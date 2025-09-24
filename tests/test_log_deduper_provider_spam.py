from __future__ import annotations

import logging

import pytest

from ai_trading.logging import LogDeduper, dedupe_ttl_s, get_logger


@pytest.fixture(autouse=True)
def _reset_deduper() -> None:
    """Ensure deduper state is isolated between tests."""

    LogDeduper.reset()
    yield
    LogDeduper.reset()


def test_provider_logs_deduped_and_summarized() -> None:
    """Duplicate provider logs within a cycle are suppressed and summarized."""

    logger = get_logger("ai_trading.test.log_deduper")

    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
            records.append(record)

    handler = _Collector(level=logging.INFO)
    logger.logger.addHandler(handler)
    try:
        LogDeduper.begin_cycle("cycle-1")

        key = "USING_BACKUP_PROVIDER:test:AAPL:1Min"
        assert LogDeduper.should_log(key, dedupe_ttl_s)
        logger.info("USING_BACKUP_PROVIDER", extra={"provider": "test", "symbol": "AAPL"})

        assert not LogDeduper.should_log(key, dedupe_ttl_s)
        assert not LogDeduper.should_log(key, dedupe_ttl_s)

        LogDeduper.emit_summaries(logger)

        using_backup = [rec for rec in records if rec.getMessage() == "USING_BACKUP_PROVIDER"]
        summaries = [rec.getMessage() for rec in records if rec.getMessage().startswith("LOG_THROTTLE_SUMMARY")]

        assert len(using_backup) == 1
        assert summaries == ['LOG_THROTTLE_SUMMARY | message="USING_BACKUP_PROVIDER" suppressed=2']

        records.clear()
        LogDeduper.emit_summaries(logger)
        assert not records
    finally:
        logger.logger.removeHandler(handler)
