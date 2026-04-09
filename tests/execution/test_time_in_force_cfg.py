from __future__ import annotations

from types import SimpleNamespace

from ai_trading.config.runtime import reload_trading_config
from ai_trading.execution import live_trading as lt
from ai_trading.execution.live_trading import LiveTradingExecutionEngine


def test_tif_from_trading_config(monkeypatch):
    monkeypatch.setenv("EXECUTION_TIME_IN_FORCE", "IOC")
    monkeypatch.setenv("AI_TRADING_EXECUTION_ALLOW_IMMEDIATE_TIF_OUTSIDE_MARKET_HOURS", "1")
    reload_trading_config()

    engine = LiveTradingExecutionEngine(ctx=SimpleNamespace())
    engine._refresh_settings()

    resolved = engine._resolve_time_in_force(None)
    assert resolved == "ioc"

    monkeypatch.delenv("EXECUTION_TIME_IN_FORCE", raising=False)
    reload_trading_config()


def test_tif_defaults_to_day_when_unset(monkeypatch):
    monkeypatch.delenv("EXECUTION_TIME_IN_FORCE", raising=False)
    monkeypatch.delenv("ALPACA_TIME_IN_FORCE", raising=False)
    reload_trading_config()

    engine = LiveTradingExecutionEngine(ctx=SimpleNamespace())
    engine._refresh_settings()

    resolved = engine._resolve_time_in_force(None)
    assert resolved == "day"


def test_tif_uses_runtime_execution_time_in_force(monkeypatch):
    monkeypatch.setenv("AI_TRADING_EXECUTION_ALLOW_IMMEDIATE_TIF_OUTSIDE_MARKET_HOURS", "1")
    monkeypatch.setattr(
        lt,
        "get_trading_config",
        lambda: SimpleNamespace(execution_time_in_force="FOK"),
    )
    engine = LiveTradingExecutionEngine(ctx=SimpleNamespace())
    engine.settings = None
    engine.config = None

    resolved = engine._resolve_time_in_force(None)
    assert resolved == "fok"


def test_tif_downgrades_immediate_outside_market_hours(monkeypatch):
    monkeypatch.setattr(lt, "_market_is_open_now", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("EXECUTION_TIME_IN_FORCE", "IOC")
    monkeypatch.setenv("AI_TRADING_EXECUTION_IMMEDIATE_TIF_MARKET_HOURS_ONLY", "1")
    monkeypatch.setenv("AI_TRADING_EXECUTION_ALLOW_IMMEDIATE_TIF_OUTSIDE_MARKET_HOURS", "0")
    monkeypatch.setenv("AI_TRADING_EXECUTION_IMMEDIATE_TIF_OFF_HOURS_FALLBACK", "day")
    reload_trading_config()

    engine = LiveTradingExecutionEngine(ctx=SimpleNamespace())
    engine._refresh_settings()

    resolved = engine._resolve_time_in_force(None)
    assert resolved == "day"
