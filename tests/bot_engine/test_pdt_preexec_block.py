"""Regression tests for PDT suppression behavior in the bot engine."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ai_trading.core import bot_engine


def _pdt_account(
    pattern_trader: bool = True,
    count: int = 6,
    limit: int = 3,
    *,
    equity: str = "30000",
    buying_power: str = "150000",
    daytrading_buying_power: str | None = None,
) -> SimpleNamespace:
    """Create a minimal Alpaca account stub for PDT checks."""

    return SimpleNamespace(
        equity=equity,
        buying_power=buying_power,
        daytrading_buying_power=daytrading_buying_power
        if daytrading_buying_power is not None
        else buying_power,
        pattern_day_trader=pattern_trader,
        daytrade_count=str(count),
        daytrade_limit=str(limit),
        pattern_day_trades=str(count),
        pattern_day_trades_count=str(count),
    )


def test_check_pdt_rule_allows_high_equity_pattern_day_trader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High-equity PDT accounts should pass when buying power remains."""

    runtime = SimpleNamespace(api=SimpleNamespace())
    account = _pdt_account()
    monkeypatch.setattr(bot_engine, "ensure_alpaca_attached", lambda _rt: None)
    monkeypatch.setattr(bot_engine, "safe_alpaca_get_account", lambda _rt: account)
    monkeypatch.setattr(bot_engine, "PDT_DAY_TRADE_LIMIT", 3, raising=False)

    blocked = bot_engine.check_pdt_rule(runtime)

    assert blocked is False
    context = getattr(runtime, "_pdt_last_context")
    assert context["pattern_day_trader"] is True
    assert context["daytrade_count"] == 6
    assert context["daytrade_limit"] == 3
    assert context.get("block_reason") is None


def test_check_pdt_rule_warns_when_dtbp_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below-threshold equity with no DTBP should warn-only."""

    runtime = SimpleNamespace(api=SimpleNamespace())
    account = _pdt_account(
        equity="20000",
        buying_power="0",
        daytrading_buying_power="0",
    )
    monkeypatch.setattr(bot_engine, "ensure_alpaca_attached", lambda _rt: None)
    monkeypatch.setattr(bot_engine, "safe_alpaca_get_account", lambda _rt: account)
    monkeypatch.setattr(bot_engine, "PDT_DAY_TRADE_LIMIT", 3, raising=False)

    blocked = bot_engine.check_pdt_rule(runtime)

    assert blocked is False
    context = getattr(runtime, "_pdt_last_context")
    assert context.get("block_enforced") is False
    assert "dtbp_exhausted" in context.get("warn_reasons", ())
    assert context["equity"] == pytest.approx(20000.0)
    assert context["daytrading_buying_power"] == pytest.approx(0.0)


def test_run_all_trades_logs_single_pdt_suppression(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """run_all_trades_worker should emit a single suppression log when PDT blocks."""

    runtime = SimpleNamespace(
        api=SimpleNamespace(
            list_orders=lambda **_kw: [],
            get_all_positions=lambda: [],
        ),
        risk_engine=SimpleNamespace(wait_for_exposure_update=lambda *_a, **_k: None),
        tickers=["AAPL", "MSFT"],
    )
    state = bot_engine.BotState()

    account = _pdt_account(
        equity="20000",
        buying_power="0",
        daytrading_buying_power="0",
    )

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

    assert state.pdt_blocked is False
    records = [record for record in caplog.records if record.msg == "ORDERS_SUPPRESSED_BY_PDT"]
    assert not records
    warn_records = [record for record in caplog.records if record.msg == "PDT_NO_DTBP_WARN_ONLY"]
    assert warn_records
