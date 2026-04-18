"""Tests for the lightweight context singleton."""

from types import SimpleNamespace

import ai_trading.core.context as context_module
from ai_trading.core.context import get_context


def test_get_context_is_singleton():
    ctx1 = get_context()
    ctx2 = get_context()
    assert ctx1 is ctx2


def test_context_has_default_attributes():
    ctx = get_context()
    assert hasattr(ctx, "alpaca_data_feed")
    assert hasattr(ctx, "alpaca_execution_feed")
    assert hasattr(ctx, "alpaca_reference_feed")
    assert hasattr(ctx, "log_market_fetch")
    assert hasattr(ctx, "testing")


def test_context_tracks_execution_and_reference_feeds(monkeypatch):
    monkeypatch.setattr(
        context_module,
        "get_settings",
        lambda: SimpleNamespace(
            alpaca_execution_feed="sip",
            alpaca_reference_feed="delayed_sip",
            log_market_fetch=True,
            testing=False,
            alpaca_api_key=None,
            alpaca_base_url="https://paper-api.alpaca.markets",
        ),
    )
    monkeypatch.setattr(context_module, "get_alpaca_secret_key_plain", lambda: None)
    monkeypatch.setattr(context_module, "_CTX", None, raising=False)
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_SIP_UNAUTHORIZED", "0")

    ctx = context_module.get_context()

    assert ctx.alpaca_data_feed == "sip"
    assert ctx.alpaca_execution_feed == "sip"
    assert ctx.alpaca_reference_feed == "delayed_sip"
    monkeypatch.setattr(context_module, "_CTX", None, raising=False)
