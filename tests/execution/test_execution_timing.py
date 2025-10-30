import logging
import time

import pytest

from ai_trading.execution import timing as execution_timing
from ai_trading.utils.prof import StageTimer


def test_execution_span_records_non_zero_duration(caplog):
    execution_timing.reset_cycle()
    logger = logging.getLogger("ai_trading.execution.test")

    with caplog.at_level(logging.INFO):
        with execution_timing.execution_span(logger, symbol="AAPL", side="buy", qty=10):
            time.sleep(0.002)

    total = execution_timing.cycle_seconds()
    assert total > 0.0

    messages = [record.message for record in caplog.records]
    assert "EXECUTE_TIMING_START" in messages
    assert "EXECUTE_TIMING_END" in messages

    with caplog.at_level(logging.DEBUG):
        with StageTimer(logger, "CYCLE_EXECUTE", override_ms=total * 1000):
            pass

    stage_records = [
        record
        for record in caplog.records
        if "STAGE_TIMING" in str(record.message) and getattr(record, "stage", None) == "CYCLE_EXECUTE"
    ]
    assert stage_records
    assert any(getattr(record, "elapsed_ms", 0) >= int(total * 1000) for record in stage_records)


def test_stage_timer_minimum_elapsed_ms():
    logger = logging.getLogger("ai_trading.utils.prof.test")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    logger.addHandler(handler)
    try:
        logger.setLevel(logging.DEBUG)
        with StageTimer(logger, "TINY", override_ms=0.2):
            pass
    finally:
        logger.removeHandler(handler)

    tiny = [record for record in records if getattr(record, "stage", None) == "TINY"]
    assert tiny
    assert tiny[0].elapsed_ms == 1
