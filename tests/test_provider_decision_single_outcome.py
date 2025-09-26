from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from ai_trading.data import provider_monitor as monitor_mod
from ai_trading.data.provider_monitor import (
    ProviderAction,
    ProviderMonitor,
    decide_provider_action,
)

_TOKENS = (
    "DATA_PROVIDER_SWITCHOVER",
    "DATA_PROVIDER_STAY",
    "DATA_PROVIDER_DISABLED",
)


def _extract_tokens(records: list[logging.LogRecord]) -> list[str]:
    hits: list[str] = []
    for record in records:
        message = record.getMessage()
        for token in _TOKENS:
            if token in message:
                hits.append(token)
    return hits


def test_provider_decision_single_outcome(caplog):
    monitor = ProviderMonitor(threshold=1, cooldown=120)
    monitor.decision_window_seconds = 0
    primary = "alpaca_iex"
    backup = "yahoo"

    caplog.set_level(logging.INFO)

    caplog.clear()
    active = monitor.update_data_health(
        primary,
        backup,
        healthy=False,
        reason="gap_ratio=2.5%",
        severity="degraded",
    )
    assert active == backup
    assert _extract_tokens(caplog.records) == ["DATA_PROVIDER_SWITCHOVER"]

    caplog.clear()
    active = monitor.update_data_health(
        primary,
        backup,
        healthy=False,
        reason="gap_ratio=2.7%",
        severity="degraded",
    )
    assert active == backup
    assert _extract_tokens(caplog.records) == ["DATA_PROVIDER_STAY"]

    caplog.clear()
    active = monitor.update_data_health(
        primary,
        backup,
        healthy=True,
        reason="gap_ratio=0.4%",
        severity="good",
    )
    assert active == backup
    assert _extract_tokens(caplog.records) == ["DATA_PROVIDER_STAY"]

    active = monitor.update_data_health(
        primary,
        backup,
        healthy=True,
        reason="gap_ratio=0.3%",
        severity="good",
    )
    assert active == backup

    state = monitor._pair_states[(primary, backup)]
    state["last_switch"] = datetime.now(UTC) - timedelta(seconds=monitor_mod._MIN_RECOVERY_SECONDS + 5)
    state["cooldown"] = 0

    caplog.clear()
    active = monitor.update_data_health(
        primary,
        backup,
        healthy=True,
        reason="gap_ratio=0.2%",
        severity="good",
    )
    assert active == primary
    assert _extract_tokens(caplog.records) == ["DATA_PROVIDER_SWITCHOVER"]


def test_provider_decision_same_provider_logs_stay(caplog):
    monitor = ProviderMonitor(threshold=1, cooldown=60)
    monitor.decision_window_seconds = 0
    primary = "alpaca_yahoo"
    backup = "yahoo"

    caplog.set_level(logging.INFO)
    caplog.clear()

    active = monitor.update_data_health(
        primary,
        backup,
        healthy=False,
        reason="gap_ratio=4.2%",
        severity="degraded",
    )

    assert active == primary
    tokens = _extract_tokens(caplog.records)
    assert tokens == ["DATA_PROVIDER_STAY"]
    assert any("redundant_request" in record.getMessage() for record in caplog.records)


def test_decide_provider_action_disable():
    action = decide_provider_action(
        {"is_healthy": False, "using_backup": True},
        cooldown_ok=False,
        consecutive_switches=5,
        policy={"disable_after": 5},
    )
    assert action is ProviderAction.DISABLE


def test_provider_decision_window_prevents_thrashing():
    monitor = ProviderMonitor(threshold=1, cooldown=30)
    monitor.decision_window_seconds = 90
    primary = "alpaca_iex"
    backup = "yahoo"

    first = monitor.update_data_health(
        primary,
        backup,
        healthy=False,
        reason="gap_ratio=5.0%",
        severity="degraded",
    )
    assert first == backup

    second = monitor.update_data_health(
        primary,
        backup,
        healthy=False,
        reason="gap_ratio=8.0%",
        severity="degraded",
    )
    assert second == backup

    third = monitor.update_data_health(
        primary,
        backup,
        healthy=True,
        reason="gap_ratio=0.4%",
        severity="good",
    )
    assert third == backup

    state = monitor._pair_states[(primary, backup)]
    state["decision_until"] = datetime.now(UTC) - timedelta(seconds=1)
    dwell = max(monitor.decision_window_seconds, monitor_mod._MIN_RECOVERY_SECONDS)
    state["last_switch"] = datetime.now(UTC) - timedelta(seconds=dwell + 5)
    state["consecutive_passes"] = monitor_mod._MIN_RECOVERY_PASSES - 1
    state["cooldown"] = 0

    recovered = monitor.update_data_health(
        primary,
        backup,
        healthy=True,
        reason="gap_ratio=0.3%",
        severity="good",
    )
    assert recovered == primary
