from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _reset_provider_env(monkeypatch):
    for key in (
        "AI_TRADING_PROVIDER_DECISION_SECS",
        "AI_TRADING_PROVIDER_SWITCH_COOLDOWN_SEC",
        "AI_TRADING_PROVIDER_HEALTH_PASSES_REQUIRED",
        "AI_TRADING_PROVIDER_PRIMARY_RECOVERY_BIAS_ENABLED",
        "AI_TRADING_PROVIDER_PRIMARY_RECOVERY_MIN_PASSES",
        "AI_TRADING_PROVIDER_PRIMARY_RECOVERY_COOLDOWN_SCALE",
        "DATA_COOLDOWN_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


def test_provider_monitor_respects_env_knobs(monkeypatch):
    from ai_trading.data import provider_monitor as monitor_mod

    def raise_settings_error(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("no settings")

    monkeypatch.setattr(monitor_mod, "get_settings", raise_settings_error)
    monkeypatch.setenv("AI_TRADING_PROVIDER_DECISION_SECS", "180")
    monkeypatch.setenv("AI_TRADING_PROVIDER_SWITCH_COOLDOWN_SEC", "900")
    monkeypatch.setenv("AI_TRADING_PROVIDER_HEALTH_PASSES_REQUIRED", "4")

    monitor = monitor_mod.ProviderMonitor()

    assert monitor.decision_window_seconds == 180
    assert monitor.min_recovery_seconds == 900
    assert monitor.recovery_passes_required == 4


def test_provider_monitor_primary_recovery_bias_switches_back_faster(monkeypatch):
    from ai_trading.data import provider_monitor as monitor_mod

    def raise_settings_error(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("no settings")

    monkeypatch.setattr(monitor_mod, "get_settings", raise_settings_error)
    monkeypatch.setenv("AI_TRADING_PROVIDER_DECISION_SECS", "0")
    monkeypatch.setenv("AI_TRADING_PROVIDER_SWITCH_COOLDOWN_SEC", "900")
    monkeypatch.setenv("AI_TRADING_PROVIDER_HEALTH_PASSES_REQUIRED", "4")
    monkeypatch.setenv("DATA_COOLDOWN_SECONDS", "120")
    monkeypatch.setenv("AI_TRADING_PROVIDER_PRIMARY_RECOVERY_BIAS_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PROVIDER_PRIMARY_RECOVERY_MIN_PASSES", "1")
    monkeypatch.setenv("AI_TRADING_PROVIDER_PRIMARY_RECOVERY_COOLDOWN_SCALE", "0")

    monitor = monitor_mod.ProviderMonitor()
    switched_to_backup = monitor.update_data_health(
        "alpaca",
        "yahoo",
        healthy=False,
        reason="upstream_unavailable",
        severity="hard_fail",
    )
    assert switched_to_backup == "yahoo"

    switched_to_primary = monitor.update_data_health(
        "alpaca",
        "yahoo",
        healthy=True,
        reason="primary_recovered",
        severity="good",
    )
    assert switched_to_primary == "alpaca"
