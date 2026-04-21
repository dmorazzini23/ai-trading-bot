from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai_trading.config.runtime import reload_trading_config
from ai_trading.core import bot_engine
from ai_trading.telemetry import runtime_state


@pytest.fixture(autouse=True)
def _reset_runtime_state():
    runtime_state.update_quote_status(allowed=False, reason=None, status="reset", synthetic=False)


def test_ensure_executable_quote_requires_override_for_synthetic(monkeypatch):
    monkeypatch.setenv("EXECUTION_ALLOW_LAST_CLOSE", "1")
    monkeypatch.setenv("EXECUTION_ALLOW_FALLBACK_PRICE", "0")
    reload_trading_config()
    ctx = SimpleNamespace(data_client=object())
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "_fetch_quote", lambda *_args, **_kwargs: None)

    decision = bot_engine._ensure_executable_quote(
        cast(Any, ctx),
        "AAPL",
        reference_price=150.0,
    )

    assert not decision
    assert decision.reason == "missing_bid_ask"
    quote_state = runtime_state.observe_quote_status()
    assert quote_state["allowed"] is False
    assert quote_state["reason"] == "missing_bid_ask"

    with_override = bot_engine._ensure_executable_quote(
        cast(Any, ctx),
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
    monkeypatch.delenv("EXECUTION_ALLOW_FALLBACK_PRICE", raising=False)
    reload_trading_config()


def test_ensure_executable_quote_requires_bid_ask_by_default(monkeypatch):
    monkeypatch.setenv("EXECUTION_ALLOW_FALLBACK_PRICE", "0")
    reload_trading_config()
    ctx = SimpleNamespace(data_client=object())
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "_fetch_quote", lambda *_args, **_kwargs: None)

    decision = bot_engine._ensure_executable_quote(
        cast(Any, ctx),
        "MSFT",
        reference_price=200.0,
    )

    assert not decision
    assert decision.reason == "missing_bid_ask"

    monkeypatch.delenv("EXECUTION_ALLOW_FALLBACK_PRICE", raising=False)
    reload_trading_config()


def test_allow_reference_fallback_override(monkeypatch):
    monkeypatch.delenv("EXECUTION_ALLOW_LAST_CLOSE", raising=False)
    reload_trading_config()
    ctx = SimpleNamespace(data_client=object())
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "_fetch_quote", lambda *_args, **_kwargs: None)

    decision = bot_engine._ensure_executable_quote(
        cast(Any, ctx),
        "GOOG",
        reference_price=123.45,
        allow_reference_fallback=True,
    )

    assert decision
    assert decision.details["synthetic"] is True
    assert decision.details["fallback_reason"] == "missing_bid_ask"


def test_reference_fallback_enabled_by_config(monkeypatch):
    monkeypatch.setenv("EXECUTION_ALLOW_FALLBACK_PRICE", "1")
    reload_trading_config()
    ctx = SimpleNamespace(data_client=object())
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "_fetch_quote", lambda *_args, **_kwargs: None)

    decision = bot_engine._ensure_executable_quote(
        cast(Any, ctx),
        "IBM",
        reference_price=111.11,
    )

    assert decision
    assert decision.details["synthetic"] is True
    assert decision.details.get("reference_price") == 111.11
    assert decision.details.get("fallback_reason") == "missing_bid_ask"

    monkeypatch.delenv("EXECUTION_ALLOW_FALLBACK_PRICE", raising=False)
    reload_trading_config()


def test_fetch_quote_unwraps_symbol_keyed_mapping(monkeypatch):
    class _Quote:
        bid_price = 100.0
        ask_price = 100.2

    class _Request:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    payload = {"AAPL": _Quote()}
    ctx = SimpleNamespace(
        data_client=SimpleNamespace(get_stock_latest_quote=lambda _req: payload)
    )
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "StockLatestQuoteRequest", _Request, raising=False)

    resolved = bot_engine._fetch_quote(cast(Any, ctx), "AAPL", feed="iex")
    assert resolved is payload["AAPL"]
    assert resolved.bid_price == 100.0
    assert resolved.ask_price == 100.2


def test_fetch_quote_unwraps_quotes_mapping_container(monkeypatch):
    class _Quote:
        bid_price = 200.0
        ask_price = 200.4

    class _Request:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    payload = SimpleNamespace(quotes={"MSFT": _Quote()})
    ctx = SimpleNamespace(
        data_client=SimpleNamespace(get_stock_latest_quote=lambda _req: payload)
    )
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "StockLatestQuoteRequest", _Request, raising=False)

    resolved = bot_engine._fetch_quote(cast(Any, ctx), "MSFT")
    assert resolved is payload.quotes["MSFT"]
    assert resolved.bid_price == 200.0
    assert resolved.ask_price == 200.4


def test_fetch_quote_falls_back_to_global_data_client(monkeypatch):
    class _Quote:
        bid_price = 150.0
        ask_price = 150.3

    class _Request:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _GlobalClient:
        def get_stock_latest_quote(self, _req):
            return {"AAPL": _Quote()}

    ctx = SimpleNamespace(data_client=None)
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "StockLatestQuoteRequest", _Request, raising=False)
    monkeypatch.setattr(bot_engine, "data_client", _GlobalClient(), raising=False)

    resolved = bot_engine._fetch_quote(cast(Any, ctx), "AAPL", feed="iex")
    assert resolved is not None
    assert resolved.bid_price == 150.0
    assert resolved.ask_price == 150.3


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
        cast(Any, ctx),
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


def test_live_quote_gate_fails_closed_when_quote_support_unavailable(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "live")
    reload_trading_config()
    ctx = SimpleNamespace(data_client=None, execution_mode="live")
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: False)

    decision = bot_engine._ensure_executable_quote(
        cast(Any, ctx),
        "AAPL",
        reference_price=150.0,
    )

    assert not decision
    assert decision.reason == "quote_support_unavailable"
    quote_state = runtime_state.observe_quote_status()
    assert quote_state["allowed"] is False
    assert quote_state["reason"] == "quote_support_unavailable"

    monkeypatch.delenv("EXECUTION_MODE", raising=False)
    reload_trading_config()


def test_live_quote_gate_blocks_synthetic_reference_fallback(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("EXECUTION_ALLOW_FALLBACK_PRICE", "1")
    monkeypatch.delenv("ALLOW_EXECUTION_ON_FALLBACK_QUOTES", raising=False)
    reload_trading_config()
    ctx = SimpleNamespace(data_client=object(), execution_mode="live")
    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(bot_engine, "_fetch_quote", lambda *_args, **_kwargs: None)

    decision = bot_engine._ensure_executable_quote(
        cast(Any, ctx),
        "AAPL",
        reference_price=150.0,
        allow_reference_fallback=True,
    )

    assert not decision
    assert decision.reason == "missing_bid_ask"
    quote_state = runtime_state.observe_quote_status()
    assert quote_state["allowed"] is False
    assert quote_state["synthetic"] is False

    monkeypatch.delenv("EXECUTION_MODE", raising=False)
    monkeypatch.delenv("EXECUTION_ALLOW_FALLBACK_PRICE", raising=False)
    reload_trading_config()
