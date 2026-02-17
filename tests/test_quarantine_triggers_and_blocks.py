from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trading.core import bot_engine
from ai_trading.runtime.quarantine import (
    QuarantineManager,
    load_quarantine_state,
    save_quarantine_state,
)


def test_quarantine_triggers_and_blocks(tmp_path: Path) -> None:
    manager = QuarantineManager()
    manager.quarantine_sleeve(
        "day",
        duration=timedelta(hours=1),
        trigger_reason="SLEEVE_QUARANTINED",
        metrics_snapshot={"reject_rate": 0.2},
    )
    active, reason = manager.is_quarantined(sleeve="day", now=datetime.now(UTC))
    assert active is True
    assert reason == "SLEEVE_QUARANTINED"

    manager.quarantine_symbol(
        "AAPL",
        duration=timedelta(hours=1),
        trigger_reason="SYMBOL_QUARANTINED",
        metrics_snapshot={"expectancy": -0.01},
    )
    active_symbol, reason_symbol = manager.is_quarantined(symbol="AAPL", now=datetime.now(UTC))
    assert active_symbol is True
    assert reason_symbol == "SYMBOL_QUARANTINED"

    path = tmp_path / "quarantine.json"
    save_quarantine_state(str(path), manager)
    loaded = load_quarantine_state(str(path))
    active_loaded, _ = loaded.is_quarantined(sleeve="day", now=datetime.now(UTC))
    assert active_loaded is True


def test_quarantine_state_relative_path_uses_data_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(data_dir))
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)

    manager = QuarantineManager()
    manager.quarantine_symbol(
        "AAPL",
        duration=timedelta(hours=1),
        trigger_reason="SYMBOL_QUARANTINED",
        metrics_snapshot={"expectancy": -0.01},
    )

    save_quarantine_state("runtime/quarantine_state.json", manager)
    expected = (data_dir / "runtime" / "quarantine_state.json").resolve()
    assert expected.exists()

    loaded = load_quarantine_state("runtime/quarantine_state.json")
    active_loaded, reason = loaded.is_quarantined(symbol="AAPL", now=datetime.now(UTC))
    assert active_loaded is True
    assert reason == "SYMBOL_QUARANTINED"


def test_persist_quarantine_manager_fail_soft(monkeypatch, caplog) -> None:
    state = bot_engine.BotState()
    setattr(state, "_quarantine_manager", QuarantineManager())

    def _fail(path: str, manager: QuarantineManager) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(bot_engine, "save_quarantine_state", _fail)
    with caplog.at_level(logging.WARNING):
        bot_engine._persist_quarantine_manager(state)

    assert any(record.getMessage() == "QUARANTINE_STATE_WRITE_FAILED" for record in caplog.records)
