from __future__ import annotations

from types import SimpleNamespace

from ai_trading.config.runtime import reload_trading_config
from ai_trading.execution import live_trading as lt
from ai_trading.execution.live_trading import LiveTradingExecutionEngine


def test_tif_from_trading_config(monkeypatch):
    monkeypatch.setenv("EXECUTION_TIME_IN_FORCE", "IOC")
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
