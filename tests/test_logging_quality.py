"""Quality gates for structured logging utilities."""

from __future__ import annotations

import logging

import pytest

from ai_trading.logging import get_logger


def test_stage_timing_throttled(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Repeated stage timing logs are throttled and normalized."""

    times = [100.0]

    def fake_monotonic() -> float:
        return times[0]

    monkeypatch.setattr("ai_trading.logging.__init__.time.monotonic", fake_monotonic)

    caplog.set_level(logging.INFO)
    logger = get_logger("ai_trading.test.logging_quality")

    caplog.clear()
    logger.info("STAGE_TIMING", extra={"stage": "bootstrap", "elapsed_ms": 7})
    times[0] += 1.0
    logger.info("STAGE_TIMING", extra={"stage": "bootstrap", "elapsed_ms": 7})
    times[0] += 1.0
    logger.info("STAGE_TIMING", extra={"stage": "bootstrap", "elapsed_ms": 7})

    stage_messages = [record.getMessage() for record in caplog.records if "STAGE_TIMING" in record.getMessage()]
    assert stage_messages == ["STAGE_TIMING | stage=bootstrap ms=7"]
