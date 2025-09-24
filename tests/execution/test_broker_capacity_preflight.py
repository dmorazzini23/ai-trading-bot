"""Capacity exhaustion should raise a non-retryable broker error."""

from __future__ import annotations

import logging

from ai_trading.execution import live_trading as lt


class DummyAPIError(lt.APIError):
    def __init__(self, status_code=403, code="40310000", message="insufficient day trading buying power"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _engine() -> lt.ExecutionEngine:
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine.stats = {"capacity_skips": 0, "skipped_orders": 0, "retry_count": 0}
    engine.logger = lt.logger
    return engine


def test_nonretryable_capacity_error(monkeypatch, caplog):
    engine = _engine()
    caplog.set_level(logging.WARNING)

    err = DummyAPIError()
    result = engine._handle_nonretryable_api_error(err, {"symbol": "AAPL"})

    assert isinstance(result, lt.NonRetryableBrokerError)
    assert engine.stats["capacity_skips"] == 1
    assert engine.stats["skipped_orders"] == 1
    messages = [rec.message for rec in caplog.records]
    assert "BROKER_CAPACITY_EXCEEDED" in messages


def test_capacity_error_passthrough_for_other_codes():
    engine = _engine()
    err = DummyAPIError(status_code=500, code="other", message="different")
    assert engine._handle_nonretryable_api_error(err, {"symbol": "AAPL"}) is None
    assert engine.stats["capacity_skips"] == 0
