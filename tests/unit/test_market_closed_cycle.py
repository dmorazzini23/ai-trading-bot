"""Ensure trading cycle is skipped when market is closed."""

import types

from ai_trading.core import bot_engine


def test_cycle_skipped_when_market_closed(monkeypatch):
    """schedule_run_all_trades should short-circuit when market closed."""

    runtime = types.SimpleNamespace(api=object())
    calls: list[tuple] = []

    monkeypatch.setattr(bot_engine, "run_all_trades_worker", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(bot_engine, "ensure_alpaca_attached", lambda *_: None)
    monkeypatch.setattr(bot_engine, "_validate_trading_api", lambda *_: True)
    monkeypatch.setattr(bot_engine, "_is_market_open_base", lambda: False)

    bot_engine.schedule_run_all_trades(runtime)

    assert calls == []

