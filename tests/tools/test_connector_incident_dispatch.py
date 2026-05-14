from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "connector_incident_dispatch.py"
_SPEC = importlib.util.spec_from_file_location("connector_incident_dispatch", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
dispatch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dispatch)


def test_run_dispatch_calls_active_connectors() -> None:
    calls: dict[str, dict[str, Any]] = {}

    def _slack(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack"] = args
        return {"sent": True}

    def _slack_eod(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack_eod"] = args
        return {"sent": False, "reason": "market_not_closed"}

    def _openclaw(args: dict[str, Any]) -> dict[str, Any]:
        calls["openclaw"] = args
        return {"sent": False, "reason": "no_incident_triggered"}

    def _incident_snapshot(args: dict[str, Any]) -> dict[str, Any]:
        calls["incident_snapshot"] = args
        return {
            "should_alert": False,
            "fingerprint": "fp-1",
            "snapshot": {"health_status": "ready"},
            "triggers": [],
        }

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc",
        },
        slack_notifier=_slack,
        slack_eod_notifier=_slack_eod,
        openclaw_notifier=_openclaw,
        incident_snapshot_builder=_incident_snapshot,
    )
    assert payload["ok"] is True
    assert payload["slack"]["attempted"] is True
    assert payload["slack_eod"]["attempted"] is True
    assert payload["openclaw"]["attempted"] is True
    assert calls["slack"]["webhook_url"] == "https://hooks.slack.test/abc"
    assert calls["slack"]["health_port"] == 9001
    assert calls["slack_eod"]["webhook_url"] == "https://hooks.slack.test/abc"
    assert calls["slack_eod"]["health_port"] == 9001
    assert calls["slack_eod"]["require_after_hours_training"] is True
    assert calls["slack_eod"]["block_on_training_gate"] is False
    assert calls["incident_snapshot"]["repeat_cooldown_minutes"] == 45
    assert calls["incident_snapshot"]["health_port"] == 9001
    assert "health_timeout_s" not in calls["incident_snapshot"]


def test_run_dispatch_hydrates_slack_webhook_from_managed_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, dict[str, Any]] = {}

    from ai_trading.config import managed_secrets

    def _fake_fetch(secret_id: str, *, region: str, profile: str) -> dict[str, str]:
        assert secret_id == "ai-trading/prod"
        assert region == "us-west-2"
        assert profile == ""
        return {"AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/aws"}

    monkeypatch.setattr(managed_secrets, "fetch_aws_secret_payload", _fake_fetch)

    def _slack(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack"] = args
        return {"sent": True}

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_CONNECTOR_OPENCLAW_ENABLED": "0",
            "AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED": "0",
            "AI_TRADING_SECRETS_BACKEND": "aws-secrets-manager",
            "AI_TRADING_AWS_SECRET_ID": "ai-trading/prod",
            "AI_TRADING_AWS_REGION": "us-west-2",
            "AI_TRADING_MANAGED_SECRET_KEYS": "AI_TRADING_SLACK_WEBHOOK_URL",
        },
        slack_notifier=_slack,
        slack_eod_notifier=lambda args: {"unused": args},
        openclaw_notifier=lambda args: {"unused": args},
        incident_snapshot_builder=lambda args: {
            "should_alert": False,
            "fingerprint": "fp-1",
            "snapshot": {"health_status": "ready"},
            "triggers": [],
        },
    )

    assert payload["ok"] is True
    assert payload["managed_connector_secrets"]["hydrated"] == 1
    assert payload["slack"]["attempted"] is True
    assert calls["slack"]["webhook_url"] == "https://hooks.slack.test/aws"


def test_resolve_openclaw_runtime_target_accepts_explicit_url_with_config_token(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(
        dispatch.json.dumps({"hooks": {"token": "cfg-token", "path": "/custom-hooks"}}),
        encoding="utf-8",
    )

    target = dispatch._resolve_openclaw_runtime_target(
        {
            "AI_TRADING_OPENCLAW_RUNTIME_WEBHOOK_URL": "http://runtime.test/override",
            "AI_TRADING_OPENCLAW_CONFIG_PATH": str(config_path),
        }
    )

    assert target == {
        "url": "http://runtime.test/override",
        "token": "cfg-token",
    }


def test_resolve_openclaw_runtime_target_accepts_explicit_token_with_config_path(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(
        dispatch.json.dumps({"hooks": {"token": "cfg-token", "path": "custom-hooks"}}),
        encoding="utf-8",
    )

    target = dispatch._resolve_openclaw_runtime_target(
        {
            "AI_TRADING_OPENCLAW_HOOK_TOKEN": "explicit-token",
            "AI_TRADING_OPENCLAW_GATEWAY_URL": "http://gateway.test/",
            "AI_TRADING_OPENCLAW_CONFIG_PATH": str(config_path),
        }
    )

    assert target == {
        "url": "http://gateway.test/custom-hooks/runtime",
        "token": "explicit-token",
    }


def test_resolve_openclaw_runtime_target_accepts_explicit_token_without_config(
    tmp_path: Path,
) -> None:
    target = dispatch._resolve_openclaw_runtime_target(
        {
            "AI_TRADING_OPENCLAW_HOOK_TOKEN": "explicit-token",
            "AI_TRADING_OPENCLAW_GATEWAY_URL": "http://gateway.test",
            "AI_TRADING_OPENCLAW_CONFIG_PATH": str(tmp_path / "missing.json"),
        }
    )

    assert target == {
        "url": "http://gateway.test/hooks/ai-trading-bot/runtime",
        "token": "explicit-token",
    }


def test_run_dispatch_skips_missing_credentials() -> None:
    payload = dispatch.run_dispatch(
        env={},
        slack_notifier=lambda args: {"unused": args},
        slack_eod_notifier=lambda args: {"unused": args},
        openclaw_notifier=lambda args: {
            "sent": False,
            "reason": "missing_openclaw_target",
        },
        incident_snapshot_builder=lambda args: {
            "should_alert": True,
            "fingerprint": "fp-missing-target",
            "snapshot": {},
            "triggers": ["health_degraded"],
        },
    )
    assert payload["ok"] is True
    assert payload["slack"]["attempted"] is False
    assert payload["slack"]["skipped_reason"] == "missing_webhook"
    assert payload["slack_eod"]["attempted"] is False
    assert payload["slack_eod"]["skipped_reason"] == "missing_webhook"
    assert payload["openclaw"]["attempted"] is True
    assert payload["openclaw"]["result"]["reason"] == "missing_openclaw_target"


def test_run_dispatch_captures_connector_errors() -> None:
    def _broken_slack(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("slack broken")

    payload = dispatch.run_dispatch(
        env={"AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc"},
        slack_notifier=_broken_slack,
        slack_eod_notifier=lambda args: {"sent": False, "reason": "market_not_closed"},
        openclaw_notifier=lambda args: {"sent": False, "reason": "no_incident_triggered"},
        incident_snapshot_builder=lambda args: {
            "should_alert": False,
            "fingerprint": "fp-1",
            "snapshot": {},
            "triggers": [],
        },
    )
    assert payload["ok"] is False
    assert len(payload["errors"]) == 1
    assert payload["errors"][0]["connector"] == "slack"


def test_run_dispatch_forwards_incident_realism_and_concentration_thresholds() -> None:
    slack_args: dict[str, Any] = {}
    incident_snapshot_args: dict[str, Any] = {}
    openclaw_args: dict[str, Any] = {}

    def _slack(args: dict[str, Any]) -> dict[str, Any]:
        slack_args.update(args)
        return {"sent": False, "reason": "no_incident_triggered"}

    def _incident_snapshot(args: dict[str, Any]) -> dict[str, Any]:
        incident_snapshot_args.update(args)
        return {"should_alert": False, "fingerprint": "fp-1", "snapshot": {}, "triggers": []}

    def _openclaw(args: dict[str, Any]) -> dict[str, Any]:
        openclaw_args.update(args)
        return {"sent": False, "reason": "no_incident_triggered"}

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc",
            "AI_TRADING_INCIDENT_MIN_EDGE_REALISM_RATIO": "0.40",
            "AI_TRADING_INCIDENT_MIN_EXPECTED_EDGE_BPS_FOR_REALISM": "0.8",
            "AI_TRADING_INCIDENT_MAX_REJECTION_CONCENTRATION_RATIO": "0.70",
            "AI_TRADING_INCIDENT_MIN_REJECTED_RECORDS_FOR_CONCENTRATION": "25",
        },
        slack_notifier=_slack,
        slack_eod_notifier=lambda args: {"unused": args},
        openclaw_notifier=_openclaw,
        incident_snapshot_builder=_incident_snapshot,
    )
    assert payload["ok"] is True
    assert slack_args["min_edge_realism_ratio"] == 0.40
    assert slack_args["min_expected_edge_bps_for_realism"] == 0.8
    assert slack_args["max_rejection_concentration_ratio"] == 0.70
    assert slack_args["min_rejected_records_for_concentration"] == 25
    assert incident_snapshot_args["min_edge_realism_ratio"] == 0.40
    assert incident_snapshot_args["min_expected_edge_bps_for_realism"] == 0.8
    assert incident_snapshot_args["max_rejection_concentration_ratio"] == 0.70
    assert incident_snapshot_args["min_rejected_records_for_concentration"] == 25
    assert openclaw_args["snapshot_result"]["fingerprint"] == "fp-1"


def test_run_dispatch_forwards_shared_health_timeout() -> None:
    calls: dict[str, dict[str, Any]] = {}

    def _slack(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack"] = args
        return {"sent": False, "reason": "no_incident_triggered"}

    def _slack_eod(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack_eod"] = args
        return {"sent": False, "reason": "market_not_closed"}

    def _openclaw(args: dict[str, Any]) -> dict[str, Any]:
        calls["openclaw"] = args
        return {"sent": False, "reason": "no_incident_triggered"}

    def _incident_snapshot(args: dict[str, Any]) -> dict[str, Any]:
        calls["incident_snapshot"] = args
        return {
            "should_alert": False,
            "fingerprint": "fp-1",
            "snapshot": {},
            "triggers": [],
        }

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc",
            "AI_TRADING_CONNECTOR_HEALTH_TIMEOUT_S": "5.5",
        },
        slack_notifier=_slack,
        slack_eod_notifier=_slack_eod,
        openclaw_notifier=_openclaw,
        incident_snapshot_builder=_incident_snapshot,
    )
    assert payload["ok"] is True
    assert calls["slack"]["health_timeout_s"] == 5.5
    assert calls["slack_eod"]["health_timeout_s"] == 5.5
    assert calls["incident_snapshot"]["health_timeout_s"] == 5.5


def test_run_dispatch_forwards_packaged_connector_health_port_default() -> None:
    calls: dict[str, dict[str, Any]] = {}

    def _slack(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack"] = args
        return {"sent": False, "reason": "no_incident_triggered"}

    def _slack_eod(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack_eod"] = args
        return {"sent": False, "reason": "market_not_closed"}

    def _incident_snapshot(args: dict[str, Any]) -> dict[str, Any]:
        calls["incident_snapshot"] = args
        return {"should_alert": False, "fingerprint": "fp-1", "snapshot": {}, "triggers": []}

    payload = dispatch.run_dispatch(
        env={"AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc"},
        slack_notifier=_slack,
        slack_eod_notifier=_slack_eod,
        openclaw_notifier=lambda args: {"sent": False, "reason": "no_incident_triggered"},
        incident_snapshot_builder=_incident_snapshot,
    )

    assert payload["ok"] is True
    assert calls["slack"]["health_port"] == 9001
    assert calls["slack_eod"]["health_port"] == 9001
    assert calls["incident_snapshot"]["health_port"] == 9001


def test_run_dispatch_allows_explicit_connector_health_port_override() -> None:
    calls: dict[str, dict[str, Any]] = {}

    def _incident_snapshot(args: dict[str, Any]) -> dict[str, Any]:
        calls["incident_snapshot"] = args
        return {"should_alert": False, "fingerprint": "fp-1", "snapshot": {}, "triggers": []}

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_CONNECTOR_OPENCLAW_ENABLED": "1",
            "AI_TRADING_CONNECTOR_SLACK_ENABLED": "0",
            "AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED": "0",
            "AI_TRADING_CONNECTOR_HEALTH_PORT": "19001",
        },
        slack_notifier=lambda args: {"unused": args},
        slack_eod_notifier=lambda args: {"unused": args},
        openclaw_notifier=lambda args: {"sent": False, "reason": "no_incident_triggered"},
        incident_snapshot_builder=_incident_snapshot,
    )

    assert payload["ok"] is True
    assert calls["incident_snapshot"]["health_port"] == 19001


def test_run_dispatch_forwards_slack_eod_block_on_training_gate() -> None:
    calls: dict[str, dict[str, Any]] = {}

    def _slack_eod(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack_eod"] = args
        return {"sent": False, "reason": "market_not_closed"}

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc",
            "AI_TRADING_SLACK_EOD_BLOCK_ON_TRAINING_GATE": "1",
        },
        slack_notifier=lambda args: {"sent": False, "reason": "no_incident_triggered"},
        slack_eod_notifier=_slack_eod,
        openclaw_notifier=lambda args: {"sent": False, "reason": "no_incident_triggered"},
        incident_snapshot_builder=lambda args: {
            "should_alert": False,
            "fingerprint": "fp-1",
            "snapshot": {},
            "triggers": [],
        },
    )

    assert payload["ok"] is True
    assert calls["slack_eod"]["block_on_training_gate"] is True


def test_run_dispatch_openclaw_alert_uses_snapshot_builder() -> None:
    calls: dict[str, dict[str, Any]] = {}

    def _incident_snapshot(args: dict[str, Any]) -> dict[str, Any]:
        calls["incident_snapshot"] = args
        return {
            "should_alert": True,
            "fingerprint": "fp-openclaw",
            "incident_signature": "sig-openclaw",
            "snapshot": {"health_status": "degraded", "health_ok": False},
            "triggers": ["health_degraded"],
        }

    def _openclaw(args: dict[str, Any]) -> dict[str, Any]:
        calls["openclaw"] = args
        return {"sent": True, "status_code": 202}

    payload = dispatch.run_dispatch(
        env={"AI_TRADING_CONNECTOR_OPENCLAW_ENABLED": "1"},
        slack_notifier=lambda args: {"unused": args},
        slack_eod_notifier=lambda args: {"unused": args},
        openclaw_notifier=_openclaw,
        incident_snapshot_builder=_incident_snapshot,
    )

    assert payload["ok"] is True
    assert payload["openclaw"]["attempted"] is True
    assert payload["openclaw"]["result"]["sent"] is True
    assert calls["openclaw"]["snapshot_result"]["fingerprint"] == "fp-openclaw"
    assert calls["openclaw"]["env"]["AI_TRADING_CONNECTOR_OPENCLAW_ENABLED"] == "1"


def test_build_openclaw_runtime_payload_includes_operator_policy() -> None:
    payload = dispatch._build_openclaw_runtime_payload(
        {
            "should_alert": True,
            "fingerprint": "fp-policy",
            "snapshot": {
                "health_status": "degraded",
                "health_ok": False,
                "health_reason": "runtime_gonogo_failed",
            },
            "triggers": ["go_no_go_failed"],
        }
    )

    policy = payload["operatorAssistantPolicy"]
    assert policy["default_mode"] == "fast_read_only_artifact_summary"
    assert policy["code_change_path"] == "produce_codex_goal"
    assert policy["critical_alert_route"] == "#all-beatwallstreet"
    assert "broad_validation" in policy["disallowed_from_slack"]
    assert "code_patches" in policy["disallowed_from_slack"]
    assert payload["details"]["operator_assistant_policy"] == policy


def test_notify_openclaw_incident_suppresses_duplicate_without_send(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "openclaw_incident_state.json"
    state_path.write_text(
        dispatch.json.dumps(
            {
                "fingerprint": "old-fingerprint",
                "incident_signature": "stable-health-degraded",
                "sent_at": dispatch._utc_now_iso(),
            }
        ),
        encoding="utf-8",
    )

    def _unexpected_post(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("duplicate incident should not post to OpenClaw")

    monkeypatch.setattr(dispatch, "_post_openclaw_runtime_event", _unexpected_post)

    result = dispatch._notify_openclaw_incident(
        {
            "state_path": str(state_path),
            "env": {
                "AI_TRADING_OPENCLAW_RUNTIME_WEBHOOK_URL": "https://openclaw.test/runtime",
                "AI_TRADING_OPENCLAW_HOOK_TOKEN": "token",
            },
            "snapshot_result": {
                "should_alert": True,
                "fingerprint": "new-metric-fingerprint",
                "incident_signature": "stable-health-degraded",
                "snapshot": {"health_status": "unhealthy", "health_ok": False},
                "triggers": ["health_degraded"],
            },
        }
    )

    assert result["sent"] is False
    assert result["reason"] == "repeat_cooldown_active"
    assert result["incident_signature"] == "stable-health-degraded"


def test_notify_openclaw_incident_records_clear_state_without_send(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "openclaw_incident_state.json"
    state_path.write_text(
        dispatch.json.dumps(
            {
                "fingerprint": "old-fingerprint",
                "incident_signature": "old-incident",
                "sent_at": "2026-05-04T17:47:34.706472Z",
                "triggers": ["go_no_go_failed"],
            }
        ),
        encoding="utf-8",
    )

    def _unexpected_post(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("clear OpenClaw state must not post")

    monkeypatch.setattr(dispatch, "_post_openclaw_runtime_event", _unexpected_post)

    result = dispatch._notify_openclaw_incident(
        {
            "state_path": str(state_path),
            "env": {
                "AI_TRADING_OPENCLAW_RUNTIME_WEBHOOK_URL": "https://openclaw.test/runtime",
                "AI_TRADING_OPENCLAW_HOOK_TOKEN": "token",
            },
            "snapshot_result": {
                "should_alert": False,
                "fingerprint": "clear-fingerprint",
                "incident_signature": "clear-signature",
                "snapshot": {
                    "health_status": "ready",
                    "health_ok": True,
                    "broker_status": "connected",
                },
                "triggers": [],
            },
        }
    )

    state = dispatch.json.loads(state_path.read_text(encoding="utf-8"))
    assert result["sent"] is False
    assert result["reason"] == "no_incident_triggered"
    assert result["state_path"] == str(state_path)
    assert state["status"] == "clear"
    assert state["sent"] is False
    assert state["sent_at"] is None
    assert state["fingerprint"] == "clear-fingerprint"
    assert state["triggers"] == []


def test_run_dispatch_records_openclaw_clear_state_below_min_severity(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "openclaw_incident_state.json"

    def _unexpected_openclaw(args: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("below-min-severity OpenClaw snapshot should not post")

    def _noop_slack(args: dict[str, Any]) -> dict[str, Any]:
        return {"sent": False, "reason": "disabled"}

    def _incident_snapshot(args: dict[str, Any]) -> dict[str, Any]:
        return {
            "should_alert": False,
            "fingerprint": "below-min-fingerprint",
            "incident_signature": "below-min-signature",
            "snapshot": {
                "health_status": "ready",
                "health_ok": True,
                "broker_status": "connected",
            },
            "triggers": [],
        }

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_CONNECTOR_SLACK_ENABLED": "0",
            "AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED": "0",
            "AI_TRADING_CONNECTOR_OPENCLAW_ENABLED": "1",
            "AI_TRADING_CONNECTOR_OPENCLAW_MIN_SEVERITY": "error",
            "AI_TRADING_OPENCLAW_INCIDENT_STATE_PATH": str(state_path),
        },
        slack_notifier=_noop_slack,
        slack_eod_notifier=_noop_slack,
        openclaw_notifier=_unexpected_openclaw,
        incident_snapshot_builder=_incident_snapshot,
    )

    state = dispatch.json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["openclaw"]["result"]["reason"] == "below_min_severity"
    assert payload["openclaw"]["result"]["state_path"] == str(state_path)
    assert state["status"] == "clear"
    assert state["fingerprint"] == "below-min-fingerprint"


def test_notify_openclaw_incident_min_interval_allows_severity_escalation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "openclaw_incident_state.json"
    state_path.write_text(
        dispatch.json.dumps(
            {
                "fingerprint": "old-fingerprint",
                "incident_signature": "stable-runtime-incident",
                "sent_at": dispatch._utc_now_iso(),
                "severity": "error",
                "triggers": ["health_degraded"],
                "snapshot": {
                    "health_status": "degraded",
                    "health_ok": False,
                    "broker_status": "connected",
                },
            }
        ),
        encoding="utf-8",
    )
    posts: list[dict[str, Any]] = []

    def _fake_post(**kwargs: Any) -> dict[str, Any]:
        posts.append(kwargs)
        return {"status_code": 202}

    monkeypatch.setattr(dispatch, "_post_openclaw_runtime_event", _fake_post)

    result = dispatch._notify_openclaw_incident(
        {
            "state_path": str(state_path),
            "env": {
                "AI_TRADING_OPENCLAW_RUNTIME_WEBHOOK_URL": "https://openclaw.test/runtime",
                "AI_TRADING_OPENCLAW_HOOK_TOKEN": "token",
            },
            "min_interval_minutes": 60,
            "repeat_cooldown_minutes": 60,
            "snapshot_result": {
                "should_alert": True,
                "fingerprint": "new-critical-fingerprint",
                "incident_signature": "stable-runtime-incident",
                "snapshot": {
                    "health_status": "degraded",
                    "health_ok": False,
                    "broker_status": "disconnected",
                },
                "triggers": ["health_degraded"],
            },
        }
    )

    assert result["sent"] is True
    assert result["severity"] == "critical"
    assert posts


def test_notify_openclaw_incident_min_interval_allows_blocker_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "openclaw_incident_state.json"
    state_path.write_text(
        dispatch.json.dumps(
            {
                "fingerprint": "old-fingerprint",
                "incident_signature": "stable-runtime-incident",
                "sent_at": dispatch._utc_now_iso(),
                "severity": "error",
                "triggers": ["rejection_concentration_high"],
                "snapshot": {
                    "health_status": "healthy",
                    "health_ok": True,
                    "broker_status": "connected",
                    "top_rejection_concentration_gate": "OLD_GATE",
                },
            }
        ),
        encoding="utf-8",
    )
    posts: list[dict[str, Any]] = []

    def _fake_post(**kwargs: Any) -> dict[str, Any]:
        posts.append(kwargs)
        return {"status_code": 202}

    monkeypatch.setattr(dispatch, "_post_openclaw_runtime_event", _fake_post)

    result = dispatch._notify_openclaw_incident(
        {
            "state_path": str(state_path),
            "env": {
                "AI_TRADING_OPENCLAW_RUNTIME_WEBHOOK_URL": "https://openclaw.test/runtime",
                "AI_TRADING_OPENCLAW_HOOK_TOKEN": "token",
            },
            "min_interval_minutes": 60,
            "repeat_cooldown_minutes": 60,
            "snapshot_result": {
                "should_alert": True,
                "fingerprint": "new-blocker-fingerprint",
                "incident_signature": "stable-runtime-incident",
                "snapshot": {
                    "health_status": "healthy",
                    "health_ok": True,
                    "broker_status": "connected",
                    "top_rejection_concentration_gate": "NEW_GATE",
                },
                "triggers": ["rejection_concentration_high"],
            },
        }
    )

    assert result["sent"] is True
    assert result["material"]["top_rejection_concentration_gate"] == "NEW_GATE"
    assert posts


def test_run_dispatch_openclaw_min_severity_suppresses_noncritical() -> None:
    calls: dict[str, dict[str, Any]] = {}

    def _incident_snapshot(args: dict[str, Any]) -> dict[str, Any]:
        calls["incident_snapshot"] = args
        return {
            "should_alert": True,
            "fingerprint": "fp-openclaw-error",
            "incident_signature": "sig-openclaw-error",
            "snapshot": {
                "health_status": "degraded",
                "health_ok": False,
                "broker_status": "connected",
            },
            "triggers": ["health_degraded"],
        }

    def _openclaw(args: dict[str, Any]) -> dict[str, Any]:
        calls["openclaw"] = args
        return {"sent": True}

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_CONNECTOR_OPENCLAW_ENABLED": "1",
            "AI_TRADING_CONNECTOR_OPENCLAW_MIN_SEVERITY": "critical",
            "AI_TRADING_CONNECTOR_SLACK_ENABLED": "0",
            "AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED": "0",
        },
        slack_notifier=lambda args: {"unused": args},
        slack_eod_notifier=lambda args: {"unused": args},
        openclaw_notifier=_openclaw,
        incident_snapshot_builder=_incident_snapshot,
    )

    assert payload["ok"] is True
    assert payload["openclaw"]["attempted"] is True
    assert payload["openclaw"]["min_severity"] == "critical"
    assert payload["openclaw"]["severity"] == "error"
    assert payload["openclaw"]["result"]["sent"] is False
    assert payload["openclaw"]["result"]["reason"] == "below_min_severity"
    assert "openclaw" not in calls


def test_run_dispatch_openclaw_min_severity_allows_critical() -> None:
    calls: dict[str, dict[str, Any]] = {}

    def _incident_snapshot(args: dict[str, Any]) -> dict[str, Any]:
        calls["incident_snapshot"] = args
        return {
            "should_alert": True,
            "fingerprint": "fp-openclaw-critical",
            "incident_signature": "sig-openclaw-critical",
            "snapshot": {
                "health_status": "degraded",
                "health_ok": False,
                "broker_status": "disconnected",
            },
            "triggers": ["broker_disconnected"],
        }

    def _openclaw(args: dict[str, Any]) -> dict[str, Any]:
        calls["openclaw"] = args
        return {"sent": True, "status_code": 202}

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_CONNECTOR_OPENCLAW_ENABLED": "1",
            "AI_TRADING_CONNECTOR_OPENCLAW_MIN_SEVERITY": "critical",
            "AI_TRADING_CONNECTOR_SLACK_ENABLED": "0",
            "AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED": "0",
        },
        slack_notifier=lambda args: {"unused": args},
        slack_eod_notifier=lambda args: {"unused": args},
        openclaw_notifier=_openclaw,
        incident_snapshot_builder=_incident_snapshot,
    )

    assert payload["ok"] is True
    assert payload["openclaw"]["min_severity"] == "critical"
    assert payload["openclaw"]["severity"] == "critical"
    assert payload["openclaw"]["result"]["sent"] is True
    assert calls["openclaw"]["snapshot_result"]["fingerprint"] == "fp-openclaw-critical"


def test_run_dispatch_openclaw_model_readiness_notifies_when_ready(
    tmp_path: Path,
) -> None:
    calls: dict[str, dict[str, Any]] = {}
    report_path = tmp_path / "after_hours_training_latest.json"
    report_path.write_text(
        dispatch.json.dumps(
            {
                "updated_at": "2026-04-25T00:00:00Z",
                "report_path": str(report_path),
                "report": {
                    "model": {"model_id": "ml-ready-1"},
                    "score_orientation": "direct",
                    "label_quality": {
                        "warnings": [],
                        "timestamp_order_violations": 0,
                        "duplicate_symbol_timestamp_rows": 0,
                    },
                    "runtime_performance_gate": {
                        "checks": {
                            "live_samples_sufficient": True,
                            "open_position_reconciliation_consistent": True,
                        }
                    },
                    "candidate_metrics": [
                        {
                            "selected": True,
                            "name": "histgb",
                            "mean_expectancy_bps": 1.2,
                            "mean_hit_rate": 0.51,
                            "profitable_fold_count": 1,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    def _readiness(args: dict[str, Any]) -> dict[str, Any]:
        calls["readiness"] = args
        return {"sent": True, "fingerprint": args["readiness_result"]["fingerprint"]}

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_CONNECTOR_OPENCLAW_MODEL_READINESS_ENABLED": "1",
            "AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED": "0",
            "AI_TRADING_AFTER_HOURS_REPORT_LATEST_PATH": str(report_path),
            "AI_TRADING_CONNECTOR_OPENCLAW_ENABLED": "0",
            "AI_TRADING_CONNECTOR_SLACK_ENABLED": "0",
            "AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED": "0",
        },
        slack_notifier=lambda args: {"unused": args},
        slack_eod_notifier=lambda args: {"unused": args},
        openclaw_notifier=lambda args: {"unused": args},
        incident_snapshot_builder=lambda args: {
            "should_alert": False,
            "fingerprint": "fp-unused",
            "snapshot": {},
            "triggers": [],
        },
        openclaw_model_readiness_notifier=_readiness,
    )

    assert payload["ok"] is True
    assert payload["openclaw_model_readiness"]["attempted"] is True
    readiness = calls["readiness"]["readiness_result"]
    assert readiness["ready"] is True
    assert readiness["observed"]["model_id"] == "ml-ready-1"
    assert readiness["observed"]["mean_expectancy_bps"] == 1.2
    assert readiness["observed"]["profitable_fold_count"] == 1


def test_run_dispatch_openclaw_model_readiness_blocks_unclean_candidate(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "after_hours_training_latest.json"
    report_path.write_text(
        dispatch.json.dumps(
            {
                "report": {
                    "model": {"model_id": "ml-not-ready-1"},
                    "score_orientation": "inverse",
                    "label_quality": {"warnings": ["score_orientation_inverse"]},
                    "runtime_performance_gate": {
                        "checks": {
                            "live_samples_sufficient": False,
                            "open_position_reconciliation_consistent": True,
                        }
                    },
                    "candidate_metrics": [
                        {
                            "selected": True,
                            "name": "histgb",
                            "mean_expectancy_bps": -0.5,
                            "mean_hit_rate": 0.10,
                            "profitable_fold_count": 0,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    calls: dict[str, dict[str, Any]] = {}

    def _readiness(args: dict[str, Any]) -> dict[str, Any]:
        calls["readiness"] = args
        return {
            "sent": False,
            "reason": args["readiness_result"]["reason"],
            "readiness": args["readiness_result"],
        }

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_CONNECTOR_OPENCLAW_MODEL_READINESS_ENABLED": "1",
            "AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED": "0",
            "AI_TRADING_AFTER_HOURS_REPORT_LATEST_PATH": str(report_path),
            "AI_TRADING_CONNECTOR_OPENCLAW_ENABLED": "0",
            "AI_TRADING_CONNECTOR_SLACK_ENABLED": "0",
            "AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED": "0",
        },
        slack_notifier=lambda args: {"unused": args},
        slack_eod_notifier=lambda args: {"unused": args},
        openclaw_notifier=lambda args: {"unused": args},
        incident_snapshot_builder=lambda args: {
            "should_alert": False,
            "fingerprint": "fp-unused",
            "snapshot": {},
            "triggers": [],
        },
        openclaw_model_readiness_notifier=_readiness,
    )

    assert payload["ok"] is True
    readiness = calls["readiness"]["readiness_result"]
    assert readiness["ready"] is False
    assert "expectancy_not_positive" in readiness["failed_reasons"]
    assert "profitable_folds_insufficient" in readiness["failed_reasons"]
    assert "score_orientation_not_direct" in readiness["failed_reasons"]
    assert "label_quality_warnings_present" in readiness["failed_reasons"]
    assert "runtime_data_quality_checks_failed" in readiness["failed_reasons"]


def test_load_runtime_env_defaults_populates_missing_values(tmp_path: Path) -> None:
    runtime_env = tmp_path / "runtime" / "ai-trading-runtime.env"
    runtime_env.parent.mkdir()
    runtime_env.write_text(
        "AI_TRADING_SLACK_WEBHOOK_URL=https://hooks.slack.test/from-runtime\n"
        "AI_TRADING_OPENCLAW_GATEWAY_URL=http://127.0.0.1:18789\n",
        encoding="utf-8",
    )
    env: dict[str, str] = {}

    summary = dispatch._load_runtime_env_defaults(env=env, repo_root=tmp_path)

    assert summary["loaded"] is True
    assert summary["applied"] == 2
    assert env["AI_TRADING_SLACK_WEBHOOK_URL"] == "https://hooks.slack.test/from-runtime"
    assert env["AI_TRADING_OPENCLAW_GATEWAY_URL"] == "http://127.0.0.1:18789"


def test_load_runtime_env_defaults_does_not_override_existing(tmp_path: Path) -> None:
    runtime_env = tmp_path / "runtime" / "ai-trading-runtime.env"
    runtime_env.parent.mkdir()
    runtime_env.write_text(
        "AI_TRADING_SLACK_WEBHOOK_URL=https://hooks.slack.test/from-runtime\n",
        encoding="utf-8",
    )
    env: dict[str, str] = {"AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/already-set"}

    summary = dispatch._load_runtime_env_defaults(env=env, repo_root=tmp_path)

    assert summary["loaded"] is True
    assert summary["applied"] == 0
    assert env["AI_TRADING_SLACK_WEBHOOK_URL"] == "https://hooks.slack.test/already-set"
