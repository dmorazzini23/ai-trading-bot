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

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc",
        },
        slack_notifier=_slack,
        slack_eod_notifier=_slack_eod,
    )
    assert payload["ok"] is True
    assert payload["slack"]["attempted"] is True
    assert payload["slack_eod"]["attempted"] is True
    assert calls["slack"]["webhook_url"] == "https://hooks.slack.test/abc"
    assert calls["slack"]["health_port"] == 9001
    assert calls["slack_eod"]["webhook_url"] == "https://hooks.slack.test/abc"
    assert calls["slack_eod"]["health_port"] == 9001
    assert calls["slack_eod"]["require_after_hours_training"] is True
    assert calls["slack_eod"]["block_on_training_gate"] is False


def test_retired_openclaw_settings_do_not_dispatch() -> None:
    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc",
            "AI_TRADING_CONNECTOR_OPENCLAW_ENABLED": "1",
            "AI_TRADING_CONNECTOR_OPENCLAW_MODEL_READINESS_ENABLED": "1",
        },
        slack_notifier=lambda args: {"sent": False, "reason": "no_incident_triggered"},
        slack_eod_notifier=lambda args: {"sent": False, "reason": "market_not_closed"},
    )
    assert payload["ok"] is True
    assert not any("openclaw" in key for key in payload)


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
            "AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED": "0",
            "AI_TRADING_SECRETS_BACKEND": "aws-secrets-manager",
            "AI_TRADING_AWS_SECRET_ID": "ai-trading/prod",
            "AI_TRADING_AWS_REGION": "us-west-2",
            "AI_TRADING_MANAGED_SECRET_KEYS": "AI_TRADING_SLACK_WEBHOOK_URL",
        },
        slack_notifier=_slack,
        slack_eod_notifier=lambda args: {"unused": args},
    )

    assert payload["ok"] is True
    assert payload["managed_connector_secrets"]["hydrated"] == 1
    assert payload["slack"]["attempted"] is True
    assert calls["slack"]["webhook_url"] == "https://hooks.slack.test/aws"


def test_run_dispatch_skips_missing_credentials() -> None:
    payload = dispatch.run_dispatch(
        env={},
        slack_notifier=lambda args: {"unused": args},
        slack_eod_notifier=lambda args: {"unused": args},
    )
    assert payload["ok"] is True
    assert payload["slack"]["attempted"] is False
    assert payload["slack"]["skipped_reason"] == "missing_webhook"
    assert payload["slack_eod"]["attempted"] is False
    assert payload["slack_eod"]["skipped_reason"] == "missing_webhook"


def test_run_dispatch_captures_connector_errors() -> None:
    def _broken_slack(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("slack broken")

    payload = dispatch.run_dispatch(
        env={"AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc"},
        slack_notifier=_broken_slack,
        slack_eod_notifier=lambda args: {"sent": False, "reason": "market_not_closed"},
    )
    assert payload["ok"] is False
    assert len(payload["errors"]) == 1
    assert payload["errors"][0]["connector"] == "slack"


def test_run_dispatch_forwards_incident_realism_and_concentration_thresholds() -> None:
    slack_args: dict[str, Any] = {}

    def _slack(args: dict[str, Any]) -> dict[str, Any]:
        slack_args.update(args)
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
    )
    assert payload["ok"] is True
    assert slack_args["min_edge_realism_ratio"] == 0.40
    assert slack_args["min_expected_edge_bps_for_realism"] == 0.8
    assert slack_args["max_rejection_concentration_ratio"] == 0.70
    assert slack_args["min_rejected_records_for_concentration"] == 25


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
    )
    assert payload["ok"] is True
    assert calls["slack"]["health_timeout_s"] == 5.5
    assert calls["slack_eod"]["health_timeout_s"] == 5.5


def test_run_dispatch_forwards_packaged_connector_health_port_default() -> None:
    calls: dict[str, dict[str, Any]] = {}

    def _slack(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack"] = args
        return {"sent": False, "reason": "no_incident_triggered"}

    def _slack_eod(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack_eod"] = args
        return {"sent": False, "reason": "market_not_closed"}

    payload = dispatch.run_dispatch(
        env={"AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc"},
        slack_notifier=_slack,
        slack_eod_notifier=_slack_eod,
    )

    assert payload["ok"] is True
    assert calls["slack"]["health_port"] == 9001
    assert calls["slack_eod"]["health_port"] == 9001


def test_run_dispatch_allows_explicit_connector_health_port_override() -> None:
    calls: dict[str, dict[str, Any]] = {}

    def _slack(args: dict[str, Any]) -> dict[str, Any]:
        calls["slack"] = args
        return {"sent": False, "reason": "no_incident_triggered"}

    payload = dispatch.run_dispatch(
        env={
            "AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc",
            "AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED": "0",
            "AI_TRADING_CONNECTOR_HEALTH_PORT": "19001",
        },
        slack_notifier=_slack,
        slack_eod_notifier=lambda args: {"unused": args},
    )

    assert payload["ok"] is True
    assert calls["slack"]["health_port"] == 19001


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
    )

    assert payload["ok"] is True
    assert calls["slack_eod"]["block_on_training_gate"] is True


def test_load_runtime_env_defaults_populates_missing_values(tmp_path: Path) -> None:
    runtime_env = tmp_path / "runtime" / "ai-trading-runtime.env"
    runtime_env.parent.mkdir()
    runtime_env.write_text(
        "AI_TRADING_SLACK_WEBHOOK_URL=https://hooks.slack.test/from-runtime\n"
        "AI_TRADING_SLACK_CHANNEL=#ops\n",
        encoding="utf-8",
    )
    env: dict[str, str] = {}

    summary = dispatch._load_runtime_env_defaults(env=env, repo_root=tmp_path)

    assert summary["loaded"] is True
    assert summary["applied"] == 2
    assert env["AI_TRADING_SLACK_WEBHOOK_URL"] == "https://hooks.slack.test/from-runtime"
    assert env["AI_TRADING_SLACK_CHANNEL"] == "#ops"


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


def test_format_preflight_report_summarizes_health() -> None:
    text = dispatch.format_preflight_report(
        {
            "report_status": "complete",
            "readiness_verdict": "blocked",
            "status": "degraded",
            "http_status": 503,
            "reason": "required_model_stale",
            "attention_flags": ["required_model_stale"],
            "service_state": {"status": "ready", "phase": "active"},
            "broker": {"status": "connected", "positions_count": 0, "open_orders_count": 0},
            "data_provider": {"active": "alpaca", "status": "healthy"},
        }
    )
    assert text.startswith("*Market preflight: blocked*")
    assert "0 positions, 0 open orders" in text
    assert "required_model_stale" in text


def test_format_preflight_report_flags_unreadable_health() -> None:
    text = dispatch.format_preflight_report(
        {"report_status": "failed", "error_type": "URLError", "error": "refused"}
    )
    assert "FAILED" in text and "URLError: refused" in text


def test_run_scheduled_report_posts_text_through_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(dispatch, "build_scheduled_report", lambda kind: f"report:{kind}")

    result = dispatch.run_scheduled_report(
        "close-recap",
        env={"AI_TRADING_SLACK_WEBHOOK_URL": "https://hooks.slack.test/abc"},
        poster=lambda url, payload: posted.append((url, payload)) or 200,
    )

    assert result["sent"] is True
    assert posted == [("https://hooks.slack.test/abc", {"text": "report:close-recap"})]


def test_run_scheduled_report_skips_without_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch, "build_scheduled_report", lambda kind: pytest.fail("built without webhook"))
    result = dispatch.run_scheduled_report("preflight", env={}, poster=lambda *a: pytest.fail("posted"))
    assert result == {"report": "preflight", "managed_connector_secrets": result["managed_connector_secrets"],
                      "sent": False, "skipped_reason": "missing_webhook"}


def test_main_rejects_unknown_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dispatch, "_load_runtime_env_defaults", lambda *a, **k: {})
    assert dispatch.main(["--report", "weekly-audit"]) == 2
