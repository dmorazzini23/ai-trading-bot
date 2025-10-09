from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _reset_provider_env(monkeypatch):
    for key in (
        "AI_TRADING_PROVIDER_DECISION_SECS",
        "AI_TRADING_PROVIDER_SWITCH_COOLDOWN_SEC",
        "AI_TRADING_PROVIDER_HEALTH_PASSES_REQUIRED",
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
