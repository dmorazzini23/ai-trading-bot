import logging
from datetime import UTC, datetime

import pytest

from ai_trading.config.settings import get_settings
from ai_trading.data.provider_monitor import ProviderMonitor


def test_provider_quiet_period_respects_config(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PROVIDER_SWITCH_QUIET_SECONDS", "45")
    monkeypatch.setenv("PROVIDER_MAX_COOLDOWN_SECONDS", "180")
    get_settings.cache_clear()

    monitor = ProviderMonitor(cooldown=10, switchover_threshold=3)
    assert monitor.max_cooldown == pytest.approx(180.0)

    caplog.set_level(logging.WARNING)
    for _ in range(3):
        monitor.record_switchover("alpaca_iex", "yahoo")

    blocked = [record for record in caplog.records if record.message == "DATA_PROVIDER_SWITCHOVER_BLOCKED"]
    assert blocked
    blocked_record = blocked[-1]
    assert blocked_record.window_seconds == 45

    disabled_until = monitor.disabled_until.get("alpaca_iex")
    assert disabled_until is not None
    remaining = (disabled_until - datetime.now(UTC)).total_seconds()
    assert remaining == pytest.approx(180.0, rel=0.1)

    get_settings.cache_clear()
