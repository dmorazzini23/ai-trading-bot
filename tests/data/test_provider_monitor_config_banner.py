"""Tests for provider monitor configuration banner logging."""

from __future__ import annotations

import logging

import pytest

from ai_trading.data import provider_monitor


def test_provider_monitor_logs_config_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The monitor should emit a single configuration banner per process."""

    monkeypatch.setattr(provider_monitor, "_PROVIDER_CONFIG_LOGGED", False, raising=False)
    monkeypatch.setenv("AI_TRADING_PROVIDER_DECISION_SECS", "300")
    monkeypatch.setenv("AI_TRADING_PROVIDER_SWITCH_COOLDOWN_SEC", "600")
    monkeypatch.setenv("DATA_PROVIDER_MAX_COOLDOWN", "900")

    with caplog.at_level(logging.INFO):
        provider_monitor.ProviderMonitor(cooldown=600, max_cooldown=900)
        provider_monitor.ProviderMonitor(cooldown=600, max_cooldown=900)

    records = [record for record in caplog.records if record.msg == "PROVIDER_MONITOR_CONFIG"]
    assert len(records) == 1
    banner = records[0]
    assert banner.decision_window_secs == 300
    assert banner.switch_cooldown_secs == 600
    assert banner.max_cooldown_secs == 900
