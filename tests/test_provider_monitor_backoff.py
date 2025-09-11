import logging
from datetime import UTC, datetime, timedelta

from ai_trading.data import provider_monitor as pm


class DummyAlerts:
    def __init__(self):
        self.calls: list[tuple[tuple, dict]] = []

    def create_alert(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_exponential_backoff_extends_cooldown(monkeypatch):
    base = datetime(2024, 1, 1, tzinfo=UTC)

    class FakeDT(datetime):
        current = base

        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return cls.current

    monkeypatch.setattr(pm, "datetime", FakeDT)
    monitor = pm.ProviderMonitor(cooldown=10, backoff_factor=2, max_cooldown=60)

    monitor.disable("alpaca")
    assert monitor.disabled_until["alpaca"] == base + timedelta(seconds=10)

    FakeDT.current = base + timedelta(seconds=15)
    monitor.disable("alpaca")
    assert monitor.disabled_until["alpaca"] == FakeDT.current + timedelta(seconds=20)

    FakeDT.current = base + timedelta(seconds=40)
    monitor.disable("alpaca")
    # Third disable should backoff to 40s but capped below max
    assert monitor.disabled_until["alpaca"] == FakeDT.current + timedelta(seconds=40)


def test_outage_logging_and_alert(monkeypatch, caplog):
    base = datetime(2024, 1, 1, tzinfo=UTC)

    class FakeDT(datetime):
        current = base

        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return cls.current

    monkeypatch.setattr(pm, "datetime", FakeDT)
    alerts = DummyAlerts()
    monitor = pm.ProviderMonitor(cooldown=10, alert_manager=alerts)

    monitor.disable("alpaca")
    FakeDT.current = base + timedelta(seconds=15)
    monitor.disable("alpaca")

    FakeDT.current = base + timedelta(seconds=40)
    with caplog.at_level(logging.INFO):
        assert not monitor.is_disabled("alpaca")

    assert alerts.calls
    args, kwargs = alerts.calls[0]
    assert kwargs["metadata"]["disable_count"] == 2
    assert kwargs["metadata"]["duration"] == 40.0
    assert any(r.message == "DATA_PROVIDER_RECOVERED" for r in caplog.records)

