from __future__ import annotations

from pathlib import Path

from tools import mcp_linear_issues_server as linear_srv


def test_runtime_regression_snapshot_detects_gate_failure(monkeypatch) -> None:
    def _fake_report() -> dict[str, object]:
        return {
            "go_no_go": {"gate_passed": False, "failed_checks": ["profit_factor"]},
            "execution_vs_alpha": {"execution_capture_ratio": 0.04, "slippage_drag_bps": 9.2},
        }

    def _fake_health(port: int, timeout_s: float) -> dict[str, object]:
        return {
            "ok": False,
            "status": "degraded",
            "reason": "runtime_gate_failed",
            "data_provider": {"status": "degraded", "active": "alpaca"},
            "broker": {"status": "connected"},
            "timestamp": "2026-03-28T20:00:00Z",
        }

    monkeypatch.setattr(linear_srv, "_runtime_report_payload", _fake_report)
    monkeypatch.setattr(linear_srv, "_health_payload", _fake_health)

    payload = linear_srv.tool_runtime_regression_snapshot({"min_capture_ratio": 0.08})
    assert payload["regression_detected"] is True
    triggers = payload["triggers"]
    assert "go_no_go_failed" in triggers
    assert "go_no_go_failed_checks" in triggers
    assert "health_degraded" in triggers
    assert "provider_degraded" in triggers
    assert "execution_capture_ratio_low" in triggers


def test_runtime_regression_snapshot_treats_reachable_as_healthy(monkeypatch) -> None:
    def _fake_report() -> dict[str, object]:
        return {
            "go_no_go": {"gate_passed": None, "failed_checks": []},
            "execution_vs_alpha": {"execution_capture_ratio": None, "slippage_drag_bps": None},
        }

    def _fake_health(port: int, timeout_s: float) -> dict[str, object]:
        return {
            "ok": True,
            "status": "healthy",
            "reason": "runtime_health_ok",
            "data_provider": {"status": "healthy", "active": "alpaca"},
            "broker": {"status": "reachable"},
            "timestamp": "2026-04-01T15:36:26.501084Z",
        }

    monkeypatch.setattr(linear_srv, "_runtime_report_payload", _fake_report)
    monkeypatch.setattr(linear_srv, "_health_payload", _fake_health)

    payload = linear_srv.tool_runtime_regression_snapshot({})
    assert "broker_disconnected" not in payload["triggers"]
    assert payload["regression_detected"] is False


def test_create_regression_issue_dry_run(monkeypatch, tmp_path: Path) -> None:
    snapshot = {
        "regression_detected": True,
        "triggers": ["go_no_go_failed"],
        "fingerprint": "abc123",
        "snapshot": {
            "gate_passed": False,
            "failed_checks": ["profit_factor"],
            "execution_capture_ratio": 0.04,
            "slippage_drag_bps": 8.4,
            "health_status": "degraded",
            "health_reason": "runtime_gate_failed",
            "provider_status": "degraded",
            "broker_status": "connected",
        },
    }

    monkeypatch.setattr(linear_srv, "_runtime_regression_snapshot", lambda _args: snapshot)
    result = linear_srv.tool_create_regression_issue(
        {
            "state_path": str(tmp_path / "linear_state.json"),
            "team_id": "team_123",
            "dry_run": True,
        }
    )
    assert result["dry_run"] is True
    assert result["created"] is False
    assert result["issue_input"]["teamId"] == "team_123"
    assert "go/no-go failed" in result["issue_input"]["title"].lower()


def test_create_regression_issue_dedupes_state(monkeypatch, tmp_path: Path) -> None:
    snapshot = {
        "regression_detected": True,
        "triggers": ["go_no_go_failed"],
        "fingerprint": "dup123",
        "snapshot": {"gate_passed": False, "failed_checks": ["win_rate"]},
    }
    monkeypatch.setattr(linear_srv, "_runtime_regression_snapshot", lambda _args: snapshot)
    monkeypatch.setattr(linear_srv, "_linear_graphql", lambda **kwargs: {"issueCreate": {"success": True, "issue": {"id": "issue1", "identifier": "OPS-1", "title": "t", "url": "https://linear.app/x"}}})

    state_path = tmp_path / "linear_state.json"
    create_args = {
        "state_path": str(state_path),
        "team_id": "team_123",
        "api_key": "lin_api_key",
    }
    first = linear_srv.tool_create_regression_issue(create_args)
    second = linear_srv.tool_create_regression_issue(create_args)
    assert first["created"] is True
    assert second["created"] is False
    assert second["reason"] == "duplicate_fingerprint"


def test_create_regression_issue_rolling_session_reuses_issue(monkeypatch, tmp_path: Path) -> None:
    snapshots = [
        {
            "regression_detected": True,
            "triggers": ["health_degraded", "provider_degraded"],
            "fingerprint": "fp-1",
            "snapshot": {
                "gate_passed": None,
                "failed_checks": [],
                "execution_capture_ratio": None,
                "slippage_drag_bps": None,
                "health_status": "degraded",
                "health_reason": "data_available_via_backup",
                "provider_status": "degraded",
                "provider_active": "yahoo",
                "broker_status": "connected",
                "timestamp": "2026-04-01T14:05:05.626855Z",
            },
        },
        {
            "regression_detected": True,
            "triggers": ["health_degraded", "provider_degraded"],
            "fingerprint": "fp-2",
            "snapshot": {
                "gate_passed": None,
                "failed_checks": [],
                "execution_capture_ratio": None,
                "slippage_drag_bps": None,
                "health_status": "degraded",
                "health_reason": "data_available_via_backup",
                "provider_status": "degraded",
                "provider_active": "yahoo",
                "broker_status": "connected",
                "timestamp": "2026-04-01T14:10:19.025362Z",
            },
        },
    ]
    calls = {"issue_create": 0, "comment_create": 0}

    def _fake_snapshot(_args: dict[str, object]) -> dict[str, object]:
        if snapshots:
            return snapshots.pop(0)
        return {
            "regression_detected": True,
            "triggers": ["health_degraded", "provider_degraded"],
            "fingerprint": "fp-2",
            "snapshot": {},
        }

    def _fake_linear_graphql(**kwargs: object) -> dict[str, object]:
        query = str(kwargs.get("query") or "")
        if "issueCreate" in query:
            calls["issue_create"] += 1
            return {
                "issueCreate": {
                    "success": True,
                    "issue": {"id": "issue-rollup-1", "identifier": "OPS-99", "title": "t", "url": "https://linear.app/x"},
                }
            }
        if "commentCreate" in query:
            calls["comment_create"] += 1
            return {"commentCreate": {"success": True, "comment": {"id": "comment-1", "body": "ok"}}}
        raise AssertionError("unexpected mutation")

    monkeypatch.setattr(linear_srv, "_runtime_regression_snapshot", _fake_snapshot)
    monkeypatch.setattr(linear_srv, "_linear_graphql", _fake_linear_graphql)

    state_path = tmp_path / "linear_state_rolling.json"
    args = {
        "state_path": str(state_path),
        "team_id": "team_123",
        "api_key": "lin_api_key",
    }

    first = linear_srv.tool_create_regression_issue(args)
    second = linear_srv.tool_create_regression_issue(args)

    assert first["created"] is True
    assert first["rolling_key"] == "2026-04-01:provider_health"
    assert second["created"] is False
    assert second["reason"] == "rolling_session_existing_issue"
    assert second["rolling_key"] == "2026-04-01:provider_health"
    assert second["event_count"] == 2
    assert calls["issue_create"] == 1
    assert calls["comment_create"] == 1
