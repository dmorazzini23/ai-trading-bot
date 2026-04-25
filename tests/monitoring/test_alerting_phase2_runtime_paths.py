from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ai_trading.monitoring import alerting


def test_email_alerter_success_builds_message_and_uses_tls(monkeypatch: pytest.MonkeyPatch) -> None:
    sent_messages: list[Any] = []
    events: list[str] = []

    class SMTP:
        def __init__(self, server: str, port: int) -> None:
            assert server == "smtp.example.com"
            assert port == 2525

        def __enter__(self) -> SMTP:
            events.append("enter")
            return self

        def __exit__(self, *_args: object) -> None:
            events.append("exit")

        def starttls(self) -> None:
            events.append("tls")

        def login(self, username: str, password: str) -> None:
            events.append(f"login:{username}:{password}")

        def send_message(self, message: Any) -> None:
            sent_messages.append(message)

    monkeypatch.setattr(alerting.smtplib, "SMTP", SMTP)
    alerter = alerting.EmailAlerter(
        smtp_server="smtp.example.com",
        smtp_port=2525,
        username="bot@example.com",
        password="secret",
    )
    alert = alerting.Alert(
        "Risk limit",
        "Drawdown breached",
        alerting.AlertSeverity.CRITICAL,
        source="RiskEngine",
        metadata={"symbol": "SPY"},
    )

    assert alerter.send_alert(alert, ["ops@example.com"]) is True
    assert events == ["enter", "tls", "login:bot@example.com:secret", "exit"]
    assert sent_messages[0]["Subject"] == "[CRITICAL] Risk limit"
    assert "ops@example.com" in sent_messages[0]["To"]


def test_email_alerter_disabled_and_missing_recipients_do_not_send() -> None:
    alert = alerting.Alert("Title", "Message", alerting.AlertSeverity.INFO)

    assert alerting.EmailAlerter().send_alert(alert, ["ops@example.com"]) is False
    assert (
        alerting.EmailAlerter("smtp.example.com", 2525, "bot", "secret").send_alert(
            alert,
            [],
        )
        is False
    )


def test_slack_alerter_posts_payload_with_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class Response:
        def raise_for_status(self) -> None:
            calls.append({"raised": False})

    def fake_post(url: str, *, json: dict[str, Any], timeout: float) -> Response:
        calls.append({"url": url, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(alerting.http, "post", fake_post)
    alerter = alerting.SlackAlerter("https://hooks.example/slack", "#ops")
    alert = alerting.Alert(
        "Broker stale",
        "Open-order snapshot is stale",
        alerting.AlertSeverity.WARNING,
        source="OMS",
        metadata={"stale_seconds": 90},
    )

    assert alerter.send_alert(alert) is True
    payload = calls[0]["json"]
    fields = payload["attachments"][0]["fields"]
    assert payload["channel"] == "#ops"
    assert payload["attachments"][0]["color"] == "#ff9500"
    assert {"title": "stale_seconds", "value": "90", "short": True} in fields


def test_slack_alerter_handles_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            raise alerting.RequestException("no route")

    monkeypatch.setattr(
        alerting.http,
        "post",
        lambda *_args, **_kwargs: Response(),
    )

    alerter = alerting.SlackAlerter("https://hooks.example/slack")

    assert alerter.send_alert(alerting.Alert("Title", "Message", alerting.AlertSeverity.INFO)) is False


def test_alert_manager_rate_limits_history_and_forced_alerts() -> None:
    manager = alerting.AlertManager()
    manager.max_history_size = 2

    first = manager.send_alert("Provider down", "IEX unavailable", alerting.AlertSeverity.WARNING)
    limited = manager.send_alert("Provider down", "IEX unavailable", alerting.AlertSeverity.WARNING)
    forced = manager.send_alert(
        "Provider down",
        "IEX unavailable",
        alerting.AlertSeverity.WARNING,
        force=True,
    )

    assert first.startswith("alert_")
    assert limited.startswith("alert_")
    assert forced.startswith("alert_")
    assert manager.alert_queue.qsize() == 2
    assert len(manager.alert_history) == 2


def test_alert_manager_processes_channels_and_custom_handler() -> None:
    manager = alerting.AlertManager()
    delivered: list[tuple[str, alerting.AlertSeverity]] = []
    handled: list[str] = []
    alert = alerting.Alert("System started", "Ready", alerting.AlertSeverity.INFO)

    manager.is_running = True
    manager.alert_queue.put(alert)
    manager.custom_handlers[alerting.AlertSeverity.INFO] = lambda item: handled.append(item.title)

    def fake_send_to_channel(
        alert: alerting.Alert,
        channel: alerting.AlertChannel,
    ) -> bool:
        delivered.append((channel.value, alert.severity))
        manager.is_running = False
        return True

    manager._send_to_channel = fake_send_to_channel  # type: ignore[method-assign]

    manager._process_alerts()

    assert delivered == [("slack", alerting.AlertSeverity.INFO)]
    assert handled == ["System started"]
    assert alert.channels_sent == [alerting.AlertChannel.SLACK]


def test_alert_manager_builds_typed_alert_messages_and_stats() -> None:
    manager = alerting.AlertManager()
    manager.send_trading_alert(
        "OrderRejected",
        "SPY",
        {"reason": "min_qty"},
        alerting.AlertSeverity.CRITICAL,
    )
    manager.send_system_alert("OMS", "STALE", "lagging", alerting.AlertSeverity.WARNING)
    manager.send_performance_alert("cycle_seconds", 82, 81)
    old = alerting.Alert("Old", "Old", alerting.AlertSeverity.INFO)
    old.timestamp = datetime.now(UTC) - timedelta(days=2)
    manager.alert_history.append(old)

    stats = manager.get_alert_stats()

    assert stats["total_alerts"] == 4
    assert stats["alerts_last_day"] == 3
    assert stats["severity_counts"]["critical"] == 1
    assert stats["severity_counts"]["warning"] == 2
