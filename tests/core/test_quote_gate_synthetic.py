from __future__ import annotations

import math
from types import SimpleNamespace

from ai_trading.config.runtime import reload_trading_config
from ai_trading.core import bot_engine


def test_ensure_executable_quote_synthetic_when_last_close_allowed(monkeypatch):
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

    assert decision
    assert decision.details["synthetic"] is True
    assert math.isclose(decision.details["reference_price"], 150.0)
    assert decision.details["bid"] < decision.details["ask"]

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
