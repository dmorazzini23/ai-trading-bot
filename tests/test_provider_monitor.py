from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from ai_trading.data.provider_monitor import ProviderMonitor


def test_primary_dwell_prevents_immediate_switchback(caplog):
    monitor = ProviderMonitor(threshold=1, cooldown=30, primary_dwell_seconds=600)
    monitor.decision_window_seconds = 0
    primary = "alpaca"
    backup = "yahoo"

    caplog.set_level(logging.INFO)

    first = monitor.update_data_health(primary, backup, healthy=False, reason="timeout", severity="degraded")
    assert first == backup

    state = monitor._pair_states[(primary, backup)]
    state["consecutive_passes"] = monitor.recovery_passes_required
    state["cooldown"] = 0
    monitor.min_recovery_seconds = 0

    caplog.clear()
    second = monitor.update_data_health(primary, backup, healthy=True, reason="recovered", severity="good")
    assert second == backup
    assert any("primary_dwell" in record.getMessage() for record in caplog.records)

    state["last_switch"] = datetime.now(UTC) - timedelta(seconds=monitor.primary_dwell_seconds + 5)
    state["consecutive_passes"] = monitor.recovery_passes_required
    state["cooldown"] = 0

    caplog.clear()
    third = monitor.update_data_health(primary, backup, healthy=True, reason="recovered", severity="good")
    assert third == primary
    assert any("DATA_PROVIDER_SWITCHOVER" in record.getMessage() for record in caplog.records)
