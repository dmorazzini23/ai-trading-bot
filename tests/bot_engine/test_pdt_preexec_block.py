"""Regression tests for PDT suppression behavior in the bot engine."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ai_trading.core import bot_engine


def _pdt_account(pattern_trader: bool = True, count: int = 6, limit: int = 3) -> SimpleNamespace:
    """Create a minimal Alpaca account stub for PDT checks."""

    return SimpleNamespace(
        equity="30000",
        buying_power="150000",
        pattern_day_trader=pattern_trader,
        daytrade_count=str(count),
        daytrade_limit=str(limit),
        pattern_day_trades=str(count),
        pattern_day_trades_count=str(count),
    )


def test_check_pdt_rule_blocks_pattern_day_traders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure engine-side PDT gate mirrors execution semantics."""

    runtime = SimpleNamespace(api=SimpleNamespace())
    account = _pdt_account()
    monkeypatch.setattr(bot_engine, "ensure_alpaca_attached", lambda _rt: None)
    monkeypatch.setattr(bot_engine, "safe_alpaca_get_account", lambda _rt: account)
    monkeypatch.setattr(bot_engine, "PDT_DAY_TRADE_LIMIT", 3, raising=False)

    blocked = bot_engine.check_pdt_rule(runtime)

    assert blocked is True
    context = getattr(runtime, "_pdt_last_context")
    assert context["pattern_day_trader"] is True
    assert context["daytrade_count"] == 6
    assert context["daytrade_limit"] == 3


def test_run_all_trades_logs_single_pdt_suppression(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """run_all_trades_worker should emit a single suppression log when PDT blocks."""

    runtime = SimpleNamespace(
        api=SimpleNamespace(),
        risk_engine=SimpleNamespace(wait_for_exposure_update=lambda *_a, **_k: None),
        tickers=["AAPL", "MSFT"],
    )
    state = bot_engine.BotState()

    account = _pdt_account()

    monkeypatch.setattr(bot_engine, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(bot_engine, "_ALPACA_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(bot_engine, "_init_metrics", lambda: None)
    monkeypatch.setattr(bot_engine, "_ensure_execution_engine", lambda _rt: None)
    monkeypatch.setattr(bot_engine, "get_trade_cooldown_min", lambda: 0)
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(bot_engine, "monotonic_time", lambda: 0.0)
    monkeypatch.setattr(bot_engine, "ensure_alpaca_attached", lambda _rt: None)
    monkeypatch.setattr(bot_engine, "_validate_trading_api", lambda _api: True)
    monkeypatch.setattr(bot_engine, "_log_loop_heartbeat", lambda *_a, **_k: None)
    monkeypatch.setattr(bot_engine, "flush_log_throttle_summaries", lambda: None)
    monkeypatch.setattr(bot_engine, "_check_runtime_stops", lambda _rt: None)
    monkeypatch.setattr(bot_engine, "MEMORY_OPTIMIZATION_AVAILABLE", False, raising=False)
    monkeypatch.setattr(bot_engine, "safe_alpaca_get_account", lambda _rt: account)
    monkeypatch.setattr(bot_engine, "PDT_DAY_TRADE_LIMIT", 3, raising=False)

    with caplog.at_level(logging.INFO):
        bot_engine.run_all_trades_worker(state, runtime)

    assert state.pdt_blocked is True
    records = [record for record in caplog.records if record.msg == "ORDERS_SUPPRESSED_BY_PDT"]
    assert len(records) == 1
    record = records[0]
    assert record.pattern_day_trader is True
    assert record.daytrade_count == 6
    assert record.daytrade_limit == 3
    assert record.symbol_count == 2
