#!/usr/bin/env python3
"""Dispatch runtime incident connectors (Slack + Linear) as a periodic job."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SlackIncidentNotifier = Callable[[dict[str, Any]], dict[str, Any]]
SlackEodNotifier = Callable[[dict[str, Any]], dict[str, Any]]
LinearCreator = Callable[[dict[str, Any]], dict[str, Any]]


def _bool_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _load_connector_callables() -> tuple[SlackIncidentNotifier, SlackEodNotifier, LinearCreator]:
    from tools import mcp_linear_issues_server as linear_srv
    from tools import mcp_slack_alerts_server as slack_srv

    return (
        slack_srv.tool_notify_incident_channel,
        slack_srv.tool_notify_eod_summary,
        linear_srv.tool_create_regression_issue,
    )


def run_dispatch(
    *,
    env: Mapping[str, str] | None = None,
    slack_notifier: SlackIncidentNotifier,
    slack_eod_notifier: SlackEodNotifier,
    linear_creator: LinearCreator,
) -> dict[str, Any]:
    env_map = os.environ if env is None else env
    summary: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "slack": {"enabled": False, "attempted": False},
        "slack_eod": {"enabled": False, "attempted": False},
        "linear": {"enabled": False, "attempted": False},
        "errors": [],
    }

    webhook = (env_map.get("AI_TRADING_SLACK_WEBHOOK_URL") or env_map.get("SLACK_WEBHOOK_URL") or "").strip()

    slack_enabled = _bool_env(env_map.get("AI_TRADING_CONNECTOR_SLACK_ENABLED"), default=True)
    summary["slack"]["enabled"] = slack_enabled
    if slack_enabled:
        if webhook:
            slack_args: dict[str, Any] = {
                "webhook_url": webhook,
                "on_change_only": _bool_env(
                    env_map.get("AI_TRADING_CONNECTOR_SLACK_ON_CHANGE_ONLY"),
                    default=True,
                ),
            }
            channel = (env_map.get("AI_TRADING_SLACK_CHANNEL") or "").strip()
            if channel:
                slack_args["channel"] = channel
            min_capture = (env_map.get("AI_TRADING_INCIDENT_MIN_CAPTURE_RATIO") or "").strip()
            if min_capture:
                try:
                    slack_args["min_capture_ratio"] = float(min_capture)
                except ValueError:
                    pass
            state_path = (env_map.get("AI_TRADING_SLACK_INCIDENT_STATE_PATH") or "").strip()
            if state_path:
                slack_args["state_path"] = state_path

            summary["slack"]["attempted"] = True
            try:
                summary["slack"]["result"] = slack_notifier(slack_args)
            except Exception as exc:  # pragma: no cover - runtime guard
                summary["slack"]["error"] = str(exc)
                summary["errors"].append(
                    {
                        "connector": "slack",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        else:
            summary["slack"]["skipped_reason"] = "missing_webhook"
    else:
        summary["slack"]["skipped_reason"] = "disabled"

    slack_eod_enabled = _bool_env(
        env_map.get("AI_TRADING_CONNECTOR_SLACK_EOD_ENABLED"),
        default=True,
    )
    summary["slack_eod"]["enabled"] = slack_eod_enabled
    if slack_eod_enabled:
        if webhook:
            slack_eod_args: dict[str, Any] = {
                "webhook_url": webhook,
                "require_market_closed": _bool_env(
                    env_map.get("AI_TRADING_SLACK_EOD_REQUIRE_MARKET_CLOSED"),
                    default=True,
                ),
                "require_after_hours_training": _bool_env(
                    env_map.get("AI_TRADING_SLACK_EOD_REQUIRE_AFTER_HOURS_TRAINING"),
                    default=True,
                ),
            }
            eod_channel = (
                env_map.get("AI_TRADING_SLACK_EOD_CHANNEL")
                or env_map.get("AI_TRADING_SLACK_CHANNEL")
                or ""
            ).strip()
            if eod_channel:
                slack_eod_args["channel"] = eod_channel
            eod_state_path = (env_map.get("AI_TRADING_SLACK_EOD_STATE_PATH") or "").strip()
            if eod_state_path:
                slack_eod_args["state_path"] = eod_state_path
            if _bool_env(
                env_map.get("AI_TRADING_CONNECTOR_SLACK_EOD_FORCE"),
                default=False,
            ):
                slack_eod_args["force"] = True

            summary["slack_eod"]["attempted"] = True
            try:
                summary["slack_eod"]["result"] = slack_eod_notifier(slack_eod_args)
            except Exception as exc:  # pragma: no cover - runtime guard
                summary["slack_eod"]["error"] = str(exc)
                summary["errors"].append(
                    {
                        "connector": "slack_eod",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        else:
            summary["slack_eod"]["skipped_reason"] = "missing_webhook"
    else:
        summary["slack_eod"]["skipped_reason"] = "disabled"

    linear_enabled = _bool_env(env_map.get("AI_TRADING_CONNECTOR_LINEAR_ENABLED"), default=True)
    summary["linear"]["enabled"] = linear_enabled
    if linear_enabled:
        linear_key = (
            env_map.get("AI_TRADING_LINEAR_API_KEY") or env_map.get("LINEAR_API_KEY") or ""
        ).strip()
        linear_team = (
            env_map.get("AI_TRADING_LINEAR_TEAM_ID") or env_map.get("LINEAR_TEAM_ID") or ""
        ).strip()
        if linear_key and linear_team:
            linear_args: dict[str, Any] = {
                "api_key": linear_key,
                "team_id": linear_team,
                "dedupe": True,
                "dry_run": _bool_env(
                    env_map.get("AI_TRADING_CONNECTOR_LINEAR_DRY_RUN"),
                    default=False,
                ),
            }
            linear_state_path = (
                env_map.get("AI_TRADING_LINEAR_REGRESSION_STATE_PATH") or ""
            ).strip()
            if linear_state_path:
                linear_args["state_path"] = linear_state_path
            linear_labels = (env_map.get("AI_TRADING_LINEAR_LABEL_IDS") or "").strip()
            if linear_labels:
                linear_args["label_ids"] = linear_labels

            priority = (env_map.get("AI_TRADING_LINEAR_PRIORITY") or "").strip()
            if priority:
                try:
                    linear_args["priority"] = int(priority)
                except ValueError:
                    pass

            summary["linear"]["attempted"] = True
            try:
                summary["linear"]["result"] = linear_creator(linear_args)
            except Exception as exc:  # pragma: no cover - runtime guard
                summary["linear"]["error"] = str(exc)
                summary["errors"].append(
                    {
                        "connector": "linear",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        else:
            summary["linear"]["skipped_reason"] = "missing_api_key_or_team_id"
    else:
        summary["linear"]["skipped_reason"] = "disabled"

    summary["finished_at"] = datetime.now(UTC).isoformat()
    summary["ok"] = len(summary["errors"]) == 0
    return summary


def main(argv: list[str] | None = None) -> int:
    _ = argv
    slack_notifier, slack_eod_notifier, linear_creator = _load_connector_callables()
    summary = run_dispatch(
        slack_notifier=slack_notifier,
        slack_eod_notifier=slack_eod_notifier,
        linear_creator=linear_creator,
    )
    print(json.dumps(summary, sort_keys=True))

    fail_on_error = _bool_env(
        os.environ.get("AI_TRADING_CONNECTOR_FAIL_ON_ERROR"),
        default=False,
    )
    if fail_on_error and not bool(summary.get("ok", False)):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
