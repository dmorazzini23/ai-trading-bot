from __future__ import annotations

from pathlib import Path
from typing import cast

from tools import mcp_oncall_alerts_server as oncall_srv


def test_jsm_auth_headers_prefers_api_key(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_JSM_OPS_API_KEY", "abc123")
    monkeypatch.setenv("AI_TRADING_JSM_OPS_EMAIL", "ops@example.com")
    monkeypatch.setenv("AI_TRADING_JSM_OPS_API_TOKEN", "token")
    headers = oncall_srv._jsm_auth_headers({})
    assert headers["Authorization"] == "GenieKey abc123"


def test_notify_oncall_skips_when_no_trigger(monkeypatch) -> None:
    monkeypatch.setattr(
        oncall_srv,
        "_slack_runtime_incident_snapshot",
        lambda args: {
            "snapshot": {"health_status": "healthy"},
            "triggers": [],
            "fingerprint": "fp0",
            "should_alert": False,
        },
    )
    payload = oncall_srv.tool_notify_oncall_incident({})
    assert payload["sent"] is False
    assert payload["reason"] == "no_incident_triggered"


def test_notify_oncall_sends_and_dedupes(monkeypatch, tmp_path: Path) -> None:
    sent_urls: list[str] = []

    monkeypatch.setattr(
        oncall_srv,
        "_slack_runtime_incident_snapshot",
        lambda args: {
            "snapshot": {
                "health_status": "degraded",
                "health_reason": "broker_status_unknown",
                "broker_status": "disconnected",
                "execution_capture_ratio": 0.04,
                "slippage_drag_bps": 9.8,
                "go_no_go_failed_checks": ["profit_factor"],
                "timestamp": "2026-03-29T04:00:00Z",
            },
            "triggers": ["go_no_go_failed", "health_degraded", "broker_disconnected"],
            "fingerprint": "fp1",
            "should_alert": True,
        },
    )

    def _fake_post_json(*, url: str, payload, headers=None, timeout_s: float = 8.0):
        _ = payload, headers, timeout_s
        sent_urls.append(url)
        return 202, '{"status":"ok"}'

    monkeypatch.setattr(oncall_srv, "_post_json", _fake_post_json)
    state_path = tmp_path / "oncall_state.json"

    first = oncall_srv.tool_notify_oncall_incident(
        {
            "state_path": str(state_path),
            "providers": "jsm_ops",
            "jsm_ops_cloud_id": "cloud-123",
            "jsm_ops_email": "ops@example.com",
            "jsm_ops_api_token": "token",
            "on_change_only": True,
        }
    )
    second = oncall_srv.tool_notify_oncall_incident(
        {
            "state_path": str(state_path),
            "providers": "jsm_ops",
            "jsm_ops_cloud_id": "cloud-123",
            "jsm_ops_email": "ops@example.com",
            "jsm_ops_api_token": "token",
            "on_change_only": True,
        }
    )

    assert first["sent"] is True
    assert first["severity"] == "critical"
    assert len(first["deliveries"]) == 1
    assert any("/jsm/ops/api/cloud-123/v1/alerts" in url for url in sent_urls)
    assert second["sent"] is False
    assert second["reason"] == "duplicate_fingerprint"


def test_notify_oncall_jsm_ticket_provider(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        oncall_srv,
        "_slack_runtime_incident_snapshot",
        lambda args: {
            "snapshot": {
                "health_status": "degraded",
                "health_reason": "broker_status_unknown",
                "broker_status": "disconnected",
                "execution_capture_ratio": 0.04,
                "slippage_drag_bps": 9.8,
                "go_no_go_failed_checks": ["profit_factor"],
                "provider_active": "alpaca",
                "provider_status": "degraded",
                "timestamp": "2026-03-29T04:00:00Z",
            },
            "triggers": ["go_no_go_failed", "health_degraded", "broker_disconnected"],
            "fingerprint": "fp-ticket-1",
            "should_alert": True,
        },
    )

    def _fake_post_json(*, url: str, payload, headers=None, timeout_s: float = 8.0):
        observed["url"] = url
        observed["payload"] = payload
        observed["headers"] = headers
        observed["timeout_s"] = timeout_s
        return 201, '{"id":"10001","key":"OPS-42"}'

    monkeypatch.setattr(oncall_srv, "_post_json", _fake_post_json)
    state_path = tmp_path / "oncall_state_ticket.json"
    payload = oncall_srv.tool_notify_oncall_incident(
        {
            "state_path": str(state_path),
            "providers": "jsm_ticket",
            "jsm_site_url": "https://example.atlassian.net",
            "jsm_ticket_project_key": "OPS",
            "jsm_ticket_issue_type": "Task",
            "jsm_ops_email": "ops@example.com",
            "jsm_ops_api_token": "token",
            "on_change_only": True,
        }
    )

    assert payload["sent"] is True
    assert payload["severity"] == "critical"
    assert len(payload["deliveries"]) == 1
    assert payload["deliveries"][0]["provider"] == "jsm_ticket"
    assert payload["deliveries"][0]["issue_key"] == "OPS-42"
    assert observed["url"] == "https://example.atlassian.net/rest/api/3/issue"
    assert "Basic " in str(cast(dict[str, str], observed["headers"])["Authorization"])


def test_resolve_providers_prefers_jsm_ticket_toggle(monkeypatch) -> None:
    monkeypatch.delenv("AI_TRADING_ONCALL_PROVIDERS", raising=False)
    monkeypatch.setenv("AI_TRADING_CONNECTOR_JSM_TICKET_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_CONNECTOR_JSM_OPS_ENABLED", "0")
    assert oncall_srv._resolve_providers({}) == ["jsm_ticket"]


def test_notify_oncall_jsm_ticket_respects_min_severity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        oncall_srv,
        "_slack_runtime_incident_snapshot",
        lambda args: {
            "snapshot": {
                "health_status": "healthy",
                "health_reason": "runtime_health_ok",
                "broker_status": "connected",
                "execution_capture_ratio": 0.22,
                "slippage_drag_bps": 7.1,
                "go_no_go_failed_checks": [],
                "provider_active": "alpaca",
                "provider_status": "healthy",
                "timestamp": "2026-03-30T15:00:00Z",
            },
            "triggers": ["soft_watch"],
            "fingerprint": "fp-ticket-min-sev",
            "should_alert": True,
        },
    )

    def _never_called(*, url: str, payload, headers=None, timeout_s: float = 8.0):
        _ = url, payload, headers, timeout_s
        raise AssertionError("_post_json should not be called when severity is below minimum")

    monkeypatch.setattr(oncall_srv, "_post_json", _never_called)
    payload = oncall_srv.tool_notify_oncall_incident(
        {
            "state_path": str(tmp_path / "oncall_state_ticket_min_sev.json"),
            "providers": "jsm_ticket",
            "jsm_site_url": "https://example.atlassian.net",
            "jsm_ticket_project_key": "OPS",
            "jsm_ops_email": "ops@example.com",
            "jsm_ops_api_token": "token",
            "jsm_ticket_min_severity": "critical",
            "on_change_only": False,
        }
    )

    assert payload["sent"] is False
    assert payload["severity"] == "warning"
    assert payload["reason"] == "severity_below_minimum"
    assert payload["deliveries"][0]["provider"] == "jsm_ticket"
    assert payload["deliveries"][0]["reason"] == "severity_below_minimum"
    assert payload["deliveries"][0]["min_severity"] == "critical"


def test_notify_oncall_jsm_ticket_force_bypasses_min_severity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        oncall_srv,
        "_slack_runtime_incident_snapshot",
        lambda args: {
            "snapshot": {
                "health_status": "healthy",
                "health_reason": "runtime_health_ok",
                "broker_status": "connected",
                "execution_capture_ratio": None,
                "slippage_drag_bps": None,
                "go_no_go_failed_checks": [],
                "provider_active": "alpaca",
                "provider_status": "healthy",
                "timestamp": "2026-03-30T15:01:00Z",
            },
            "triggers": [],
            "fingerprint": "fp-ticket-force-min-sev",
            "should_alert": False,
        },
    )

    called: dict[str, object] = {}

    def _fake_post_json(*, url: str, payload, headers=None, timeout_s: float = 8.0):
        called["url"] = url
        _ = payload, headers, timeout_s
        return 201, '{"id":"10011","key":"OPS-99"}'

    monkeypatch.setattr(oncall_srv, "_post_json", _fake_post_json)
    payload = oncall_srv.tool_notify_oncall_incident(
        {
            "state_path": str(tmp_path / "oncall_state_ticket_force_min_sev.json"),
            "providers": "jsm_ticket",
            "jsm_site_url": "https://example.atlassian.net",
            "jsm_ticket_project_key": "OPS",
            "jsm_ops_email": "ops@example.com",
            "jsm_ops_api_token": "token",
            "jsm_ticket_min_severity": "critical",
            "on_change_only": False,
            "force": True,
        }
    )

    assert payload["sent"] is True
    assert payload["severity"] == "info"
    assert payload["deliveries"][0]["issue_key"] == "OPS-99"
    assert called["url"] == "https://example.atlassian.net/rest/api/3/issue"
