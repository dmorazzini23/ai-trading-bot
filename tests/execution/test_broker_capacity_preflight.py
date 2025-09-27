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


def test_skip_shorting_when_asset_not_shortable(monkeypatch, caplog):
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine._refresh_settings = lambda: None
    engine._ensure_initialized = lambda: True
    engine._pre_execution_checks = lambda: True
    engine.is_initialized = True
    engine.shadow_mode = False
    engine.stats = {}

    class DummyClient:
        def __init__(self):
            self.symbols = []

        def get_asset(self, symbol):
            self.symbols.append(symbol)
            return type("Asset", (), {"shortable": False})()

    client = DummyClient()
    engine.trading_client = client

    called = {"preflight": False}

    def fake_preflight(symbol, side, price_hint, quantity, trading_client):
        called["preflight"] = True
        return lt.CapacityCheck(True, quantity, None)

    monkeypatch.setattr(lt, "preflight_capacity", fake_preflight)

    caplog.set_level(logging.WARNING)

    result = engine.submit_market_order("AAPL", "sell", 1)

    assert result is None
    assert called["preflight"] is False
    assert client.symbols == ["AAPL"]
    messages = [record.getMessage() for record in caplog.records]
    assert any("ORDER_SKIPPED_NONRETRYABLE" in msg for msg in messages)
    assert any("shorting_disabled" in msg for msg in messages)
