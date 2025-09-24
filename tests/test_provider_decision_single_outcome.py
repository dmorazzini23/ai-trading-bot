from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

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
    primary = "alpaca_iex"
    backup = "yahoo"

    caplog.set_level(logging.INFO)

    caplog.clear()
    active = monitor.update_data_health(primary, backup, healthy=False, reason="gap_ratio=2.5%")
    assert active == backup
    assert _extract_tokens(caplog.records) == ["DATA_PROVIDER_SWITCHOVER"]

    caplog.clear()
    active = monitor.update_data_health(primary, backup, healthy=False, reason="gap_ratio=2.7%")
    assert active == backup
    assert _extract_tokens(caplog.records) == ["DATA_PROVIDER_STAY"]

    caplog.clear()
    active = monitor.update_data_health(primary, backup, healthy=True, reason="gap_ratio=0.4%")
    assert active == backup
    assert _extract_tokens(caplog.records) == ["DATA_PROVIDER_STAY"]

    state = monitor._pair_states[(primary, backup)]
    state["last_switch"] = datetime.now(UTC) - timedelta(seconds=300)
    state["cooldown"] = 120

    caplog.clear()
    active = monitor.update_data_health(primary, backup, healthy=True, reason="gap_ratio=0.2%")
    assert active == primary
    assert _extract_tokens(caplog.records) == ["DATA_PROVIDER_SWITCHOVER"]


def test_decide_provider_action_disable():
    action = decide_provider_action(
        {"is_healthy": False, "using_backup": True},
        cooldown_ok=False,
        consecutive_switches=5,
        policy={"disable_after": 5},
    )
    assert action is ProviderAction.DISABLE
