from __future__ import annotations

import logging

import pytest

from ai_trading.execution import live_trading as lt


class DummyAPIError(lt.APIError):
    def __init__(self, message: str, *, code: str | None = None, status_code: int = 403):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


def _engine() -> lt.ExecutionEngine:
    engine = lt.ExecutionEngine.__new__(lt.ExecutionEngine)
    engine.stats = {"capacity_skips": 0, "skipped_orders": 0}
    return engine


@pytest.mark.parametrize(
    "message, expected",
    [
        ("insufficient buying power to submit order", "insufficient_buying_power"),
        ("Insufficient Day Trading Buying Power", "insufficient_day_trading_buying_power"),
        ("Not enough equity in account", "not_enough_equity"),
    ],
)
def test_capacity_error_tokens(message, expected, caplog):
    engine = _engine()
    caplog.set_level(logging.DEBUG)
    err = DummyAPIError(message)

    result = engine._handle_nonretryable_api_error(err, {"symbol": "TSLA"})

    assert isinstance(result, lt.NonRetryableBrokerError)
    assert result.args[0] == expected
    assert result.detail == message
    assert engine.stats["capacity_skips"] == 1
    assert engine.stats["skipped_orders"] == 1
    info_records = [rec for rec in caplog.records if rec.getMessage() == "BROKER_CAPACITY_EXCEEDED"]
    assert info_records and info_records[0].levelno == logging.INFO
    assert getattr(info_records[0], "reason", None) == expected
    debug_records = [rec for rec in caplog.records if rec.getMessage() == "BROKER_CAPACITY_EXCEEDED_DETAIL"]
    assert debug_records and getattr(debug_records[0], "detail", "") == message


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Shorting is not permitted on this instrument", "shorting_not_permitted"),
        ("No shares available to short right now", "no_shares_available"),
        ("Cannot open short positions for this asset", "short_open_blocked"),
    ],
)
def test_short_restriction_tokens(message, expected, caplog):
    engine = _engine()
    caplog.set_level(logging.DEBUG)
    err = DummyAPIError(message)

    result = engine._handle_nonretryable_api_error(err, {"symbol": "TSLA"})

    assert isinstance(result, lt.NonRetryableBrokerError)
    assert result.args[0] == expected
    assert engine.stats["skipped_orders"] == 1
    assert engine.stats.get("capacity_skips", 0) == 0
    info_records = [rec for rec in caplog.records if rec.getMessage() == "ORDER_REJECTED_SHORT_RESTRICTION"]
    assert info_records and info_records[0].levelno == logging.INFO
    assert getattr(info_records[0], "reason", None) == expected
    debug_records = [rec for rec in caplog.records if rec.getMessage() == "ORDER_REJECTED_SHORT_RESTRICTION_DETAIL"]
    assert debug_records and getattr(debug_records[0], "detail", "") == message


def test_day_trade_code_maps_without_phrase(caplog):
    engine = _engine()
    caplog.set_level(logging.DEBUG)
    err = DummyAPIError("Custom broker message", code="40310000")

    result = engine._handle_nonretryable_api_error(err, {"symbol": "AAPL"})

    assert isinstance(result, lt.NonRetryableBrokerError)
    assert result.args[0] == "insufficient_day_trading_buying_power"
    info_records = [rec for rec in caplog.records if rec.getMessage() == "BROKER_CAPACITY_EXCEEDED"]
    assert info_records and getattr(info_records[0], "reason", None) == "insufficient_day_trading_buying_power"
