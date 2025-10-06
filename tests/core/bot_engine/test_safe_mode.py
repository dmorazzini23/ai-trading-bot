from __future__ import annotations

import types

import ai_trading.core.bot_engine as bot_engine


def test_enter_long_blocks_when_primary_provider_disabled(monkeypatch):
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: False,
        raising=False,
    )
    state = types.SimpleNamespace(degraded_providers=set())
    ctx = types.SimpleNamespace()
    blocked = bot_engine._enter_long(
        ctx,
        state,
        "AAPL",
        1000.0,
        object(),
        1.0,
        0.8,
        "test",
    )
    assert blocked is True


def test_enter_long_blocks_when_safe_mode_active(monkeypatch):
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "is_primary_provider_enabled",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(bot_engine, "is_safe_mode_active", lambda: True)
    monkeypatch.setattr(bot_engine, "safe_mode_reason", lambda: "provider_safe_mode")
    state = types.SimpleNamespace(degraded_providers=set())
    ctx = types.SimpleNamespace()
    blocked = bot_engine._enter_long(
        ctx,
        state,
        "AAPL",
        1000.0,
        object(),
        1.0,
        0.8,
        "test",
    )
    assert blocked is True


def test_should_skip_order_for_fallback_price(monkeypatch):
    state = types.SimpleNamespace(auth_skipped_symbols=set())
    assert bot_engine._should_skip_order_for_alpaca_unavailable(state, "AAPL", "yahoo")
