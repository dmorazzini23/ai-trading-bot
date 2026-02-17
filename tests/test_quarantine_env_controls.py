from __future__ import annotations

from datetime import UTC, datetime

from ai_trading.core import bot_engine
from ai_trading.runtime.quarantine import QuarantineManager


def test_quarantine_targets_from_env_supports_both(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_QUARANTINE_APPLIES_TO", "both")
    apply_sleeve, apply_symbol = bot_engine._quarantine_targets_from_env()
    assert apply_sleeve is True
    assert apply_symbol is True


def test_trigger_quarantine_respects_symbol_only(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_QUARANTINE_APPLIES_TO", "symbol")
    manager = QuarantineManager()
    bot_engine._trigger_quarantine(
        manager=manager,
        symbol="AAPL",
        sleeve="day",
        reason="UNIT_TEST",
        metrics_snapshot={"k": "v"},
    )
    is_symbol, _ = manager.is_quarantined(symbol="AAPL", now=datetime.now(UTC))
    is_sleeve, _ = manager.is_quarantined(sleeve="day", now=datetime.now(UTC))
    assert is_symbol is True
    assert is_sleeve is False
