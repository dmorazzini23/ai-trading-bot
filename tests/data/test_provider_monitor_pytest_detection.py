from __future__ import annotations

from ai_trading.data import provider_monitor


def test_record_switchover_pytest_detection(monkeypatch):
    # Ensure detection relies on module inspection rather than env flags.
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    # Construct a fresh monitor to avoid global state from other tests.
    monitor = provider_monitor.ProviderMonitor(cooldown=1, max_cooldown=1)
    monitor.record_switchover("alpaca", "yahoo")
