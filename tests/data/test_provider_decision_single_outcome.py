"""Provider health updates should yield a single active source."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from ai_trading.data import provider_monitor as monitor_mod


def _fresh_monitor() -> monitor_mod.ProviderMonitor:
    mon = monitor_mod.ProviderMonitor(
        threshold=3,
        cooldown=10,
        switchover_threshold=2,
        backoff_factor=2.0,
        max_cooldown=120,
    )
    mon.decision_window_seconds = 0
    return mon


def test_update_data_health_tracks_single_active_provider(monkeypatch, caplog):
    mon = _fresh_monitor()

    # Avoid interference from global instance
    monkeypatch.setattr(monitor_mod, "provider_monitor", mon)
    monkeypatch.setattr(monitor_mod, "get_env", lambda *_, **__: "0")
    monkeypatch.setattr(monitor_mod.logger, "info", lambda *a, **k: None)
    monkeypatch.setattr(monitor_mod.logger, "warning", lambda *a, **k: None)

    primary = "alpaca_primary"
    backup = "yahoo"

    assert (
        mon.update_data_health(primary, backup, healthy=False, reason="empty", severity="degraded")
        == backup
    )
    # First two healthy passes keep backup active while accumulating health
    assert (
        mon.update_data_health(primary, backup, healthy=True, reason="recovering", severity="good")
        == backup
    )
    assert (
        mon.update_data_health(primary, backup, healthy=True, reason="stabilizing", severity="good")
        == backup
    )

    # Third healthy pass after dwell & cooldown switches back
    state = mon._pair_states[(primary, backup)]
    state["last_switch"] = datetime.now(UTC) - timedelta(seconds=monitor_mod._MIN_RECOVERY_SECONDS + 5)
    state["cooldown"] = 0
    caplog.set_level(logging.INFO)
    decision = mon.update_data_health(primary, backup, healthy=True, reason="stable", severity="good")
    assert decision == primary

    # Subsequent healthy updates stick with primary
    assert (
        mon.update_data_health(primary, backup, healthy=True, reason="stable", severity="good")
        == primary
    )
