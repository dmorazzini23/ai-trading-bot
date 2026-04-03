from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "connector_incident_dispatch.py"
_SPEC = importlib.util.spec_from_file_location("connector_incident_dispatch", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
dispatch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dispatch)


def test_run_dispatch_calls_both_connectors() -> None:
    calls: dict[str, dict[str, Any]] = {}

    def _slack(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack"] = args
        return {"sent": True}

    def _linear(args: dict[str, Any]) -> dict[str, Any]:
        calls["linear"] = args
        return {"created": False, "reason": "no_runtime_regression_detected"}

    def _slack_eod(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack_eod"] = args
        return {"sent": False, "reason": "market_not_closed"}

    def _oncall(args: dict[str, Any]) -> dict[str, Any]:
        calls["oncall"] = args
        return {"sent": False, "reason": "no_incident_triggered"}

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc",
            "AI_TRADING_LINEAR_API_KEY": "lin_key",
            "AI_TRADING_LINEAR_TEAM_ID": "team123",
            "AI_TRADING_CONNECTOR_ONCALL_ENABLED": "1",
        },
        slack_notifier=_slack,
        slack_eod_notifier=_slack_eod,
        linear_creator=_linear,
        oncall_notifier=_oncall,
    )
    assert payload["ok"] is True
    assert payload["slack"]["attempted"] is True
    assert payload["slack_eod"]["attempted"] is True
    assert payload["linear"]["attempted"] is True
    assert payload["oncall"]["attempted"] is True
    assert calls["slack"]["webhook_url"] == "https://hooks.slack.test/abc"
    assert calls["slack_eod"]["webhook_url"] == "https://hooks.slack.test/abc"
    assert calls["slack_eod"]["require_after_hours_training"] is True
    assert calls["linear"]["team_id"] == "team123"


def test_run_dispatch_skips_missing_credentials() -> None:
    payload = dispatch.run_dispatch(
        env={},
        slack_notifier=lambda args: {"unused": args},
        slack_eod_notifier=lambda args: {"unused": args},
        linear_creator=lambda args: {"unused": args},
        oncall_notifier=lambda args: {"unused": args},
    )
    assert payload["ok"] is True
    assert payload["slack"]["attempted"] is False
    assert payload["slack"]["skipped_reason"] == "missing_webhook"
    assert payload["slack_eod"]["attempted"] is False
    assert payload["slack_eod"]["skipped_reason"] == "missing_webhook"
    assert payload["linear"]["attempted"] is False
    assert payload["linear"]["skipped_reason"] == "missing_api_key_or_team_id"
    assert payload["oncall"]["attempted"] is False
    assert payload["oncall"]["skipped_reason"] == "disabled"


def test_run_dispatch_captures_connector_errors() -> None:
    def _broken_slack(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("slack broken")

    payload = dispatch.run_dispatch(
        env={"AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc"},
        slack_notifier=_broken_slack,
        slack_eod_notifier=lambda args: {"sent": False, "reason": "market_not_closed"},
        linear_creator=lambda args: {"created": False},
        oncall_notifier=lambda args: {"sent": False, "reason": "disabled"},
    )
    assert payload["ok"] is False
    assert len(payload["errors"]) == 1
    assert payload["errors"][0]["connector"] == "slack"


def test_run_dispatch_forwards_oncall_jsm_ticket_env() -> None:
    captured: dict[str, Any] = {}

    def _oncall(args: dict[str, Any]) -> dict[str, Any]:
        captured.update(args)
        return {"sent": False, "reason": "no_incident_triggered"}

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_CONNECTOR_ONCALL_ENABLED": "1",
            "AI_TRADING_ONCALL_PROVIDERS": "jsm_ticket",
            "AI_TRADING_JSM_SITE_URL": "https://example.atlassian.net",
            "AI_TRADING_JSM_TICKET_PROJECT_KEY": "OPS",
            "AI_TRADING_JSM_TICKET_ISSUE_TYPE": "Task",
            "AI_TRADING_JSM_TICKET_LABELS": "ai-trading,runtime-incident",
        },
        slack_notifier=lambda args: {"unused": args},
        slack_eod_notifier=lambda args: {"unused": args},
        linear_creator=lambda args: {"unused": args},
        oncall_notifier=_oncall,
    )
    assert payload["ok"] is True
    assert payload["oncall"]["attempted"] is True
    assert captured["providers"] == "jsm_ticket"
    assert captured["jsm_site_url"] == "https://example.atlassian.net"
    assert captured["jsm_ticket_project_key"] == "OPS"


def test_run_dispatch_forwards_incident_realism_and_concentration_thresholds() -> None:
    slack_args: dict[str, Any] = {}
    oncall_args: dict[str, Any] = {}

    def _slack(args: dict[str, Any]) -> dict[str, Any]:
        slack_args.update(args)
        return {"sent": False, "reason": "no_incident_triggered"}

    def _oncall(args: dict[str, Any]) -> dict[str, Any]:
        oncall_args.update(args)
        return {"sent": False, "reason": "no_incident_triggered"}

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc",
            "AI_TRADING_CONNECTOR_ONCALL_ENABLED": "1",
            "AI_TRADING_INCIDENT_MIN_EDGE_REALISM_RATIO": "0.40",
            "AI_TRADING_INCIDENT_MIN_EXPECTED_EDGE_BPS_FOR_REALISM": "0.8",
            "AI_TRADING_INCIDENT_MAX_REJECTION_CONCENTRATION_RATIO": "0.70",
            "AI_TRADING_INCIDENT_MIN_REJECTED_RECORDS_FOR_CONCENTRATION": "25",
        },
        slack_notifier=_slack,
        slack_eod_notifier=lambda args: {"unused": args},
        linear_creator=lambda args: {"unused": args},
        oncall_notifier=_oncall,
    )
    assert payload["ok"] is True
    assert slack_args["min_edge_realism_ratio"] == 0.40
    assert slack_args["min_expected_edge_bps_for_realism"] == 0.8
    assert slack_args["max_rejection_concentration_ratio"] == 0.70
    assert slack_args["min_rejected_records_for_concentration"] == 25
    assert oncall_args["min_edge_realism_ratio"] == 0.40
    assert oncall_args["min_expected_edge_bps_for_realism"] == 0.8
    assert oncall_args["max_rejection_concentration_ratio"] == 0.70
    assert oncall_args["min_rejected_records_for_concentration"] == 25


def test_load_runtime_env_defaults_populates_missing_values(tmp_path: Path) -> None:
    runtime_env = tmp_path / ".env.runtime"
    runtime_env.write_text(
        "AI_TRADING_SLACK_WEBHOOK_URL=https://hooks.slack.test/from-runtime\n"
        "AI_TRADING_LINEAR_API_KEY=lin_from_runtime\n",
        encoding="utf-8",
    )
    env: dict[str, str] = {}

    summary = dispatch._load_runtime_env_defaults(env=env, repo_root=tmp_path)

    assert summary["loaded"] is True
    assert summary["applied"] == 2
    assert env["AI_TRADING_SLACK_WEBHOOK_URL"] == "https://hooks.slack.test/from-runtime"
    assert env["AI_TRADING_LINEAR_API_KEY"] == "lin_from_runtime"


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
