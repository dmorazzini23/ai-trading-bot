from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

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
