"""Quality gates for structured logging utilities."""

from __future__ import annotations

import io
import logging

import pytest

from ai_trading.logging import get_logger


def test_stage_timing_throttled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated stage timing logs are throttled and normalized."""

    times = [100.0]

    def fake_monotonic() -> float:
        return times[0]

    monkeypatch.setattr("ai_trading.logging.time.monotonic", fake_monotonic)
    monkeypatch.setenv("AI_TRADING_LOG_TIMINGS_LEVEL", "INFO")

    logger = get_logger("ai_trading.test.logging_quality")

    from ai_trading import logging as logging_mod

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    handler.addFilter(logging_mod._THROTTLE_FILTER)

    original_handlers = list(logger.handlers)
    logger.handlers = [handler]

    try:
        logger.info("STAGE_TIMING", extra={"stage": "bootstrap", "elapsed_ms": 7})
        times[0] += 1.0
        logger.info("STAGE_TIMING", extra={"stage": "bootstrap", "elapsed_ms": 7})
        times[0] += 1.0
        logger.info("STAGE_TIMING", extra={"stage": "bootstrap", "elapsed_ms": 7})
        handler.flush()
    finally:
        logger.handlers = original_handlers

    stream.seek(0)
    lines = [line for line in stream.getvalue().splitlines() if "STAGE_TIMING" in line]
    assert lines == ["STAGE_TIMING | stage=bootstrap ms=7"]
