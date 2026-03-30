from __future__ import annotations

from pathlib import Path

from tools import mcp_oncall_alerts_server as oncall_srv


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

    def _fake_post_json(*, url: str, payload, headers=None, timeout_s: float = 6.0):
        _ = payload, headers, timeout_s
        sent_urls.append(url)
        return 202, '{"status":"ok"}'

    monkeypatch.setattr(oncall_srv, "_post_json", _fake_post_json)
    state_path = tmp_path / "oncall_state.json"

    first = oncall_srv.tool_notify_oncall_incident(
        {
            "state_path": str(state_path),
            "providers": "pagerduty,opsgenie",
            "pagerduty_routing_key": "pd_key",
            "opsgenie_api_key": "og_key",
            "on_change_only": True,
        }
    )
    second = oncall_srv.tool_notify_oncall_incident(
        {
            "state_path": str(state_path),
            "providers": "pagerduty,opsgenie",
            "pagerduty_routing_key": "pd_key",
            "opsgenie_api_key": "og_key",
            "on_change_only": True,
        }
    )

    assert first["sent"] is True
    assert first["severity"] == "critical"
    assert len(first["deliveries"]) == 2
    assert any("pagerduty.com" in url for url in sent_urls)
    assert any("opsgenie.com" in url for url in sent_urls)
    assert second["sent"] is False
    assert second["reason"] == "duplicate_fingerprint"
