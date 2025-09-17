import logging
from datetime import UTC, datetime, timedelta

from ai_trading.data import provider_monitor as pm
from ai_trading.data.metrics import provider_failure_duration_seconds


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


def test_record_switchover_backoff(monkeypatch):
    base = datetime(2024, 1, 1, tzinfo=UTC)

    class FakeDT(datetime):
        current = base

        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return cls.current

    monkeypatch.setattr(pm, "datetime", FakeDT)
    alerts = DummyAlerts()
    monitor = pm.ProviderMonitor(
        cooldown=10,
        switchover_threshold=2,
        backoff_factor=2,
        max_cooldown=40,
        alert_manager=alerts,
    )

    monitor.record_switchover("a", "b")
    assert monitor.consecutive_switches == 1
    assert monitor.consecutive_switches_by_provider["a"] == 1
    assert monitor._current_switch_cooldowns["a"] == 10

    FakeDT.current = base + timedelta(seconds=5)
    monitor.record_switchover("b", "a")
    assert monitor.consecutive_switches == 1
    assert monitor.consecutive_switches_by_provider["b"] == 1
    assert monitor._current_switch_cooldowns["b"] == 10
    assert not alerts.calls

    FakeDT.current = base + timedelta(seconds=8)
    monitor.record_switchover("a", "b")
    assert monitor.consecutive_switches == 2
    assert monitor.consecutive_switches_by_provider["a"] == 2
    assert monitor._current_switch_cooldowns["a"] == 20
    assert len(alerts.calls) == 1

    FakeDT.current = base + timedelta(seconds=60)
    monitor.record_switchover("b", "a")
    assert monitor.consecutive_switches == 1
    assert monitor._current_switch_cooldowns["b"] == 10


def test_partial_coverage_reset_prevents_disable(monkeypatch):
    base = datetime(2024, 1, 1, tzinfo=UTC)

    class FakeDT(datetime):
        current = base

        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return cls.current

    monkeypatch.setattr(pm, "datetime", FakeDT)
    alerts = DummyAlerts()
    monitor = pm.ProviderMonitor(
        cooldown=10,
        switchover_threshold=2,
        backoff_factor=2,
        max_cooldown=60,
        alert_manager=alerts,
    )
    disabled: list[float] = []
    monitor.register_disable_callback("alpaca_iex", lambda duration: disabled.append(duration.total_seconds()))

    monitor.record_switchover("alpaca_iex", "yahoo")
    assert monitor.consecutive_switches == 1

    monitor.record_success("alpaca_iex")
    assert monitor.consecutive_switches == 0

    FakeDT.current = base + timedelta(seconds=2)
    monitor.record_switchover("alpaca_iex", "yahoo")
    assert monitor.consecutive_switches == 1
    assert not disabled
    assert not alerts.calls


def test_failure_duration_metric(monkeypatch, caplog):
    base = datetime(2024, 1, 1, tzinfo=UTC)

    class FakeDT(datetime):
        current = base

        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return cls.current

    monkeypatch.setattr(pm, "datetime", FakeDT)
    monitor = pm.ProviderMonitor()
    monitor.disabled_since["a"] = base - timedelta(seconds=30)

    with caplog.at_level(logging.INFO):
        monitor.record_switchover("a", "b")

    metric = provider_failure_duration_seconds.labels(provider="a")
    if hasattr(metric, "_value"):
        assert metric._value.get() == 30
    assert any(r.message == "DATA_PROVIDER_FAILURE_DURATION" for r in caplog.records)

