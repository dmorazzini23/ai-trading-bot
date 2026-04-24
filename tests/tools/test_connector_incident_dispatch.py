from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


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
    assert calls["slack_eod"]["webhook_url"] == "https://hooks.slack.test/abc"
    assert calls["slack_eod"]["require_after_hours_training"] is True
    assert "health_timeout_s" not in calls["incident_snapshot"]


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

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc",
            "AI_TRADING_CONNECTOR_HEALTH_TIMEOUT_S": "5.5",
        },
        slack_notifier=_slack,
        slack_eod_notifier=_slack_eod,
        openclaw_notifier=lambda args: (
            calls.__setitem__("openclaw", args) or {"sent": False, "reason": "no_incident_triggered"}
        ),
        incident_snapshot_builder=lambda args: (
            calls.__setitem__("incident_snapshot", args)
            or {
                "should_alert": False,
                "fingerprint": "fp-1",
                "snapshot": {},
                "triggers": [],
            }
        ),
    )
    assert payload["ok"] is True
    assert calls["slack"]["health_timeout_s"] == 5.5
    assert calls["slack_eod"]["health_timeout_s"] == 5.5
    assert calls["incident_snapshot"]["health_timeout_s"] == 5.5


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


def test_load_runtime_env_defaults_populates_missing_values(tmp_path: Path) -> None:
    runtime_env = tmp_path / ".env.runtime"
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
    runtime_env = tmp_path / ".env.runtime"
    runtime_env.write_text(
        "AI_TRADING_SLACK_WEBHOOK_URL=https://hooks.slack.test/from-runtime\n",
        encoding="utf-8",
    )
    env: dict[str, str] = {"AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/already-set"}

    summary = dispatch._load_runtime_env_defaults(env=env, repo_root=tmp_path)

    assert summary["loaded"] is True
    assert summary["applied"] == 0
    assert env["AI_TRADING_SLACK_WEBHOOK_URL"] == "https://hooks.slack.test/already-set"
