from __future__ import annotations

from pathlib import Path
from typing import Any

from tools import mcp_slack_alerts_server as slack_srv


def test_evaluate_incident_triggers_catches_regressions() -> None:
    snapshot = {
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["profit_factor"],
        "execution_capture_ratio": 0.03,
        "slippage_drag_bps": 8.1,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "runtime_gate_failed",
        "provider_status": "degraded",
        "provider_active": "alpaca",
        "provider_reason": "runtime_gate_failed",
        "using_backup": True,
        "broker_status": "disconnected",
    }
    triggers = slack_srv._evaluate_incident_triggers(snapshot, {"min_capture_ratio": 0.08})
    assert "go_no_go_failed" in triggers
    assert "go_no_go_failed_checks" in triggers
    assert "execution_capture_ratio_low" in triggers
    assert "health_degraded" in triggers
    assert "data_provider_backup_active" in triggers
    assert "broker_disconnected" in triggers


def test_evaluate_incident_triggers_treats_reachable_broker_as_healthy() -> None:
    snapshot = {
        "go_no_go_gate_passed": None,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": None,
        "slippage_drag_bps": None,
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "runtime_health_ok",
        "provider_status": "healthy",
        "provider_active": "alpaca",
        "provider_reason": "data_available_netting",
        "using_backup": False,
        "broker_status": "reachable",
    }
    triggers = slack_srv._evaluate_incident_triggers(snapshot, {"min_capture_ratio": 0.08})
    assert "broker_disconnected" not in triggers


def test_evaluate_incident_triggers_suppresses_startup_warmup_health() -> None:
    snapshot = {
        "go_no_go_gate_passed": None,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": None,
        "slippage_drag_bps": None,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "warmup_cycle",
        "provider_status": "warming_up",
        "provider_active": "alpaca",
        "provider_reason": "market_closed",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(snapshot, {"min_capture_ratio": 0.08})
    assert "health_degraded" not in triggers


def test_evaluate_incident_triggers_can_disable_startup_warmup_suppression() -> None:
    snapshot = {
        "go_no_go_gate_passed": None,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": None,
        "slippage_drag_bps": None,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "warmup_cycle",
        "provider_status": "warming_up",
        "provider_active": "alpaca",
        "provider_reason": "market_closed",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {
            "min_capture_ratio": 0.08,
            "suppress_startup_warmup_health_alerts": False,
        },
    )
    assert "health_degraded" in triggers


def test_notify_incident_channel_dedupes(monkeypatch, tmp_path: Path) -> None:
    snapshot = {
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["win_rate"],
        "execution_capture_ratio": 0.05,
        "slippage_drag_bps": 7.0,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "runtime_gate_failed",
        "provider_status": "degraded",
        "provider_active": "alpaca",
        "provider_reason": "runtime_gate_failed",
        "using_backup": False,
        "broker_status": "connected",
        "timestamp": "2026-03-28T20:00:00Z",
    }

    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_runtime_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)

    state_path = tmp_path / "slack_state.json"
    args = {
        "state_path": str(state_path),
        "webhook_url": "https://hooks.slack.test/example",
    }
    first = slack_srv.tool_notify_incident_channel(args)
    second = slack_srv.tool_notify_incident_channel(args)

    assert first["sent"] is True
    assert second["sent"] is False
    assert second["reason"] == "duplicate_fingerprint"
    assert len(posts) == 1


def test_notify_eod_summary_sends_once_per_report_date(monkeypatch, tmp_path: Path) -> None:
    snapshot: dict[str, Any] = {
        "report_date": "2026-03-28",
        "go_no_go_gate_passed": True,
        "go_no_go_failed_checks": [],
        "net_pnl": 123.45,
        "profit_factor": 1.2,
        "win_rate": 0.57,
        "closed_trades": 42,
        "execution_capture_ratio": 0.18,
        "slippage_drag_bps": 7.5,
        "order_reject_rate_pct": 0.0,
        "open_position_reconciliation_mismatch_count": 0,
        "open_position_reconciliation_max_abs_delta_qty": 0.0,
        "health_status": "healthy",
        "health_reason": "market_closed",
        "provider_status": "healthy",
        "provider_active": "alpaca",
        "using_backup": False,
        "broker_status": "connected",
        "top_loss_symbols": [],
        "timestamp": "2026-03-29T01:00:00Z",
        "learning": {
            "after_hours": {"model_name": "xgb", "model_id": "m123"},
            "execution_learning": {"samples": 20},
            "execution_autotune": {"enabled": True},
            "model_liveness": {},
        },
    }

    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_eod_summary_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)

    state_path = tmp_path / "slack_eod_state.json"
    args = {
        "state_path": str(state_path),
        "webhook_url": "https://hooks.slack.test/example",
        "require_after_hours_training": False,
    }
    first = slack_srv.tool_notify_eod_summary(args)
    second = slack_srv.tool_notify_eod_summary(args)

    assert first["sent"] is True
    assert second["sent"] is False
    assert second["reason"] == "already_sent_for_report_date"
    assert len(posts) == 1
    payload = posts[0]["payload"]
    assert isinstance(payload, dict)
    text = payload.get("text")
    assert isinstance(text, str)
    assert "ai-trading EOD summary" in text


def test_notify_eod_summary_respects_market_closed_gate(monkeypatch) -> None:
    snapshot: dict[str, Any] = {
        "report_date": "2026-03-28",
        "health_reason": "runtime_health_ok",
        "learning": {},
    }

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    monkeypatch.setattr(slack_srv, "_collect_eod_summary_snapshot", _fake_collect)
    result = slack_srv.tool_notify_eod_summary(
        {
            "webhook_url": "https://hooks.slack.test/example",
            "require_market_closed": True,
            "require_after_hours_training": False,
        }
    )
    assert result["sent"] is False
    assert result["reason"] == "market_not_closed"


def test_notify_eod_summary_sends_when_training_not_complete_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot: dict[str, Any] = {
        "report_date": "2026-03-28",
        "health_reason": "market_closed",
        "learning": {},
    }
    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_eod_summary_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)
    result = slack_srv.tool_notify_eod_summary(
        {
            "webhook_url": "https://hooks.slack.test/example",
            "state_path": str(tmp_path / "slack_eod_state.json"),
            "after_hours_training_marker_path": str(tmp_path / "missing_marker.json"),
        }
    )
    assert result["sent"] is True
    assert len(posts) == 1
    gate = result.get("training_gate")
    assert isinstance(gate, dict)
    assert gate.get("required") is True
    assert gate.get("ready") is False


def test_notify_eod_summary_blocks_until_training_complete_when_configured(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot: dict[str, Any] = {
        "report_date": "2026-03-28",
        "health_reason": "market_closed",
        "learning": {},
    }

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    monkeypatch.setattr(slack_srv, "_collect_eod_summary_snapshot", _fake_collect)
    result = slack_srv.tool_notify_eod_summary(
        {
            "webhook_url": "https://hooks.slack.test/example",
            "block_on_training_gate": True,
            "after_hours_training_marker_path": str(tmp_path / "missing_marker.json"),
        }
    )
    assert result["sent"] is False
    assert result["reason"] == "after_hours_training_not_complete"
    gate = result.get("training_gate")
    assert isinstance(gate, dict)
    assert gate.get("required") is True
    assert gate.get("ready") is False


def test_notify_eod_summary_allows_send_when_training_marker_matches(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot: dict[str, Any] = {
        "report_date": "2026-03-28",
        "health_reason": "market_closed",
        "learning": {},
    }
    marker_path = tmp_path / "after_hours_training.marker.json"
    marker_path.write_text(
        '{"date":"2026-03-28","status":"trained","model_id":"m1"}\n',
        encoding="utf-8",
    )
    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_eod_summary_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)

    result = slack_srv.tool_notify_eod_summary(
        {
            "webhook_url": "https://hooks.slack.test/example",
            "state_path": str(tmp_path / "slack_eod_state.json"),
            "after_hours_training_marker_path": str(marker_path),
        }
    )
    assert result["sent"] is True
    gate = result.get("training_gate")
    assert isinstance(gate, dict)
    assert gate.get("ready") is True
    assert len(posts) == 1


def test_collect_eod_summary_uses_runtime_file_fallback(monkeypatch) -> None:
    def _fake_runtime_report() -> dict[str, Any]:
        return {
            "go_no_go": {"gate_passed": None, "failed_checks": []},
            "execution_vs_alpha": {"daily": []},
            "trade_history": {"daily_trade_stats": []},
        }

    def _fake_read_json_object(_path: Path) -> dict[str, Any]:
        return {
            "go_no_go": {
                "gate_passed": True,
                "failed_checks": [],
                "observed": {"net_pnl": 10.0},
            },
            "execution_vs_alpha": {
                "execution_capture_ratio": 0.2,
                "slippage_drag_bps": 5.0,
                "daily": [{"date": "2026-03-27"}],
            },
            "trade_history": {"daily_trade_stats": [{"date": "2026-03-27"}]},
        }

    def _fake_health_payload(*, port: int, timeout_s: float) -> dict[str, Any]:
        assert port == 8081
        assert timeout_s == 2.0
        return {
            "status": "healthy",
            "reason": "market_closed",
            "data_provider": {"status": "healthy", "active": "alpaca", "using_backup": False},
            "broker": {"status": "connected"},
            "model_liveness": {},
            "timestamp": "2026-03-29T01:00:00Z",
        }

    monkeypatch.setattr(slack_srv, "_runtime_report_payload", _fake_runtime_report)
    monkeypatch.setattr(slack_srv, "_read_json_object", _fake_read_json_object)
    monkeypatch.setattr(slack_srv, "_health_payload", _fake_health_payload)

    snapshot = slack_srv._collect_eod_summary_snapshot({})
    assert snapshot["report_date"] == "2026-03-27"
    assert snapshot["go_no_go_gate_passed"] is True
    assert snapshot["execution_capture_ratio"] == 0.2
