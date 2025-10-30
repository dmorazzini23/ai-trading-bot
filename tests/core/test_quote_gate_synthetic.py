from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ai_trading.config.runtime import reload_trading_config
from ai_trading.core import bot_engine
from ai_trading.telemetry import runtime_state


@pytest.fixture(autouse=True)
def _reset_runtime_state():
    runtime_state.update_quote_status(allowed=False, reason=None, status="reset", synthetic=False)


def test_ensure_executable_quote_requires_override_for_synthetic(monkeypatch):
    monkeypatch.setenv("EXECUTION_ALLOW_LAST_CLOSE", "1")
    reload_trading_config()
    ctx = SimpleNamespace(data_client=object())
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "_fetch_quote", lambda *_args, **_kwargs: None)

    decision = bot_engine._ensure_executable_quote(
        ctx,
        "AAPL",
        reference_price=150.0,
    )

    assert not decision
    assert decision.reason == "missing_bid_ask"
    quote_state = runtime_state.observe_quote_status()
    assert quote_state["allowed"] is False
    assert quote_state["reason"] == "missing_bid_ask"

    with_override = bot_engine._ensure_executable_quote(
        ctx,
        "AAPL",
        reference_price=150.0,
        allow_reference_fallback=True,
    )

    assert with_override
    assert with_override.details["synthetic"] is True
    assert math.isclose(with_override.details["reference_price"], 150.0)
    assert with_override.details["bid"] < with_override.details["ask"]
    quote_state = runtime_state.observe_quote_status()
    assert quote_state["allowed"] is True
    assert quote_state["synthetic"] is True

    monkeypatch.delenv("EXECUTION_ALLOW_LAST_CLOSE", raising=False)
    reload_trading_config()


def test_ensure_executable_quote_requires_bid_ask_by_default(monkeypatch):
    reload_trading_config()
    ctx = SimpleNamespace(data_client=object())
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "_fetch_quote", lambda *_args, **_kwargs: None)

    decision = bot_engine._ensure_executable_quote(
        ctx,
        "MSFT",
        reference_price=200.0,
    )

    assert not decision
    assert decision.reason == "missing_bid_ask"


def test_allow_reference_fallback_override(monkeypatch):
    monkeypatch.delenv("EXECUTION_ALLOW_LAST_CLOSE", raising=False)
    reload_trading_config()
    ctx = SimpleNamespace(data_client=object())
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "_fetch_quote", lambda *_args, **_kwargs: None)

    decision = bot_engine._ensure_executable_quote(
        ctx,
        "GOOG",
        reference_price=123.45,
        allow_reference_fallback=True,
    )

    assert decision
    assert decision.details["synthetic"] is True
    assert decision.details["fallback_reason"] == "missing_bid_ask"


def test_gap_ratio_exceeded_uses_synthetic(monkeypatch):
    class Quote:
        bid_price = 120.0
        ask_price = 120.4
        timestamp = datetime.now(UTC)

    monkeypatch.setenv("EXECUTION_ALLOW_LAST_CLOSE", "1")
    monkeypatch.setenv("GAP_RATIO_LIMIT", "0.01")
    reload_trading_config()
    ctx = SimpleNamespace(data_client=object())
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "_fetch_quote", lambda *_args, **_kwargs: Quote())

    decision = bot_engine._ensure_executable_quote(
        ctx,
        "NFLX",
        reference_price=100.0,
        allow_reference_fallback=True,
    )

    assert decision
    assert decision.details["synthetic"] is True
    assert decision.details["fallback_reason"] == "gap_ratio_exceeded"

    monkeypatch.delenv("EXECUTION_ALLOW_LAST_CLOSE", raising=False)
    monkeypatch.delenv("GAP_RATIO_LIMIT", raising=False)
    reload_trading_config()


def test_pdt_limit_exhausted_helper():
    ctx = SimpleNamespace(
        _pdt_last_context={
            "pattern_day_trader": True,
            "daytrade_limit": 3,
            "daytrade_count": 4,
            "daytrade_limit_enforced": True,
        }
    )
    blocked, context = bot_engine._pdt_limit_exhausted(ctx)
    assert blocked is True
    assert context is not None


def test_pdt_limit_not_enforced():
    ctx = SimpleNamespace(
        _pdt_last_context={
            "pattern_day_trader": True,
            "daytrade_limit": 3,
            "daytrade_count": 1,
            "daytrade_limit_enforced": True,
        }
    )
    blocked, context = bot_engine._pdt_limit_exhausted(ctx)
    assert blocked is False
    assert context is not None
