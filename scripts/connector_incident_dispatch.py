#!/usr/bin/env python3
"""Dispatch runtime incident connectors (Slack + Linear + On-call) as a periodic job."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SlackIncidentNotifier = Callable[[dict[str, Any]], dict[str, Any]]
SlackEodNotifier = Callable[[dict[str, Any]], dict[str, Any]]
LinearCreator = Callable[[dict[str, Any]], dict[str, Any]]
OncallNotifier = Callable[[dict[str, Any]], dict[str, Any]]


def _bool_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_runtime_env_path(*, env: Mapping[str, str], repo_root: Path) -> Path:
    raw = str(env.get("AI_TRADING_RUNTIME_ENV_PATH") or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        return candidate
    return (repo_root / ".env.runtime").resolve()


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    parsed: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in raw_line:
            continue
        key_raw, value_raw = raw_line.split("=", 1)
        key = key_raw.strip()
        if not _ENV_KEY_RE.match(key):
            continue
        value = value_raw.strip()
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        parsed[key] = value
    return parsed


def _load_runtime_env_defaults(
    *,
    env: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    target_env = os.environ if env is None else env
    root = REPO_ROOT if repo_root is None else repo_root
    runtime_env_path = _resolve_runtime_env_path(env=target_env, repo_root=root)
    parsed = _parse_env_file(runtime_env_path)
    applied = 0
    for key, value in parsed.items():
        if not target_env.get(key):
            target_env[key] = value
            applied += 1
    return {
        "path": str(runtime_env_path),
        "loaded": bool(parsed),
        "entries": int(len(parsed)),
        "applied": int(applied),
    }


def _load_connector_callables() -> tuple[
    SlackIncidentNotifier,
    SlackEodNotifier,
    LinearCreator,
    OncallNotifier,
]:
    from tools import mcp_linear_issues_server as linear_srv
    from tools import mcp_oncall_alerts_server as oncall_srv
    from tools import mcp_slack_alerts_server as slack_srv

    return (
        slack_srv.tool_notify_incident_channel,
        slack_srv.tool_notify_eod_summary,
        linear_srv.tool_create_regression_issue,
        oncall_srv.tool_notify_oncall_incident,
    )


def run_dispatch(
    *,
    env: Mapping[str, str] | None = None,
    slack_notifier: SlackIncidentNotifier,
    slack_eod_notifier: SlackEodNotifier,
    linear_creator: LinearCreator,
    oncall_notifier: OncallNotifier,
) -> dict[str, Any]:
    env_map = os.environ if env is None else env
    summary: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "slack": {"enabled": False, "attempted": False},
        "slack_eod": {"enabled": False, "attempted": False},
        "linear": {"enabled": False, "attempted": False},
        "oncall": {"enabled": False, "attempted": False},
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
            for arg_name, env_name in {
                "min_edge_realism_ratio": "AI_TRADING_INCIDENT_MIN_EDGE_REALISM_RATIO",
                "min_expected_edge_bps_for_realism": "AI_TRADING_INCIDENT_MIN_EXPECTED_EDGE_BPS_FOR_REALISM",
                "max_rejection_concentration_ratio": "AI_TRADING_INCIDENT_MAX_REJECTION_CONCENTRATION_RATIO",
            }.items():
                raw = (env_map.get(env_name) or "").strip()
                if not raw:
                    continue
                try:
                    slack_args[arg_name] = float(raw)
                except ValueError:
                    continue
            min_rejected = (
                env_map.get("AI_TRADING_INCIDENT_MIN_REJECTED_RECORDS_FOR_CONCENTRATION")
                or ""
            ).strip()
            if min_rejected:
                try:
                    slack_args["min_rejected_records_for_concentration"] = int(min_rejected)
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

    oncall_enabled = _bool_env(env_map.get("AI_TRADING_CONNECTOR_ONCALL_ENABLED"), default=False)
    summary["oncall"]["enabled"] = oncall_enabled
    if oncall_enabled:
        oncall_args: dict[str, Any] = {
            "on_change_only": _bool_env(
                env_map.get("AI_TRADING_ONCALL_ON_CHANGE_ONLY"),
                default=True,
            ),
        }
        oncall_forward_map = {
            "jsm_ops_base_url": "AI_TRADING_JSM_OPS_BASE_URL",
            "jsm_ops_cloud_id": "AI_TRADING_JSM_OPS_CLOUD_ID",
            "jsm_ops_api_key": "AI_TRADING_JSM_OPS_API_KEY",
            "jsm_ops_email": "AI_TRADING_JSM_OPS_EMAIL",
            "jsm_ops_api_token": "AI_TRADING_JSM_OPS_API_TOKEN",
            "jsm_ops_bearer_token": "AI_TRADING_JSM_OPS_BEARER_TOKEN",
            "jsm_site_url": "AI_TRADING_JSM_SITE_URL",
            "jsm_ticket_project_key": "AI_TRADING_JSM_TICKET_PROJECT_KEY",
            "jsm_ticket_issue_type": "AI_TRADING_JSM_TICKET_ISSUE_TYPE",
            "jsm_ticket_labels": "AI_TRADING_JSM_TICKET_LABELS",
        }
        for arg_name, env_name in oncall_forward_map.items():
            raw_value = (env_map.get(env_name) or "").strip()
            if raw_value:
                oncall_args[arg_name] = raw_value
        providers = (env_map.get("AI_TRADING_ONCALL_PROVIDERS") or "").strip()
        if providers:
            oncall_args["providers"] = providers
        state_path = (env_map.get("AI_TRADING_ONCALL_STATE_PATH") or "").strip()
        if state_path:
            oncall_args["state_path"] = state_path
        min_capture = (env_map.get("AI_TRADING_INCIDENT_MIN_CAPTURE_RATIO") or "").strip()
        if min_capture:
            try:
                oncall_args["min_capture_ratio"] = float(min_capture)
            except ValueError:
                pass
        for arg_name, env_name in {
            "min_edge_realism_ratio": "AI_TRADING_INCIDENT_MIN_EDGE_REALISM_RATIO",
            "min_expected_edge_bps_for_realism": "AI_TRADING_INCIDENT_MIN_EXPECTED_EDGE_BPS_FOR_REALISM",
            "max_rejection_concentration_ratio": "AI_TRADING_INCIDENT_MAX_REJECTION_CONCENTRATION_RATIO",
        }.items():
            raw = (env_map.get(env_name) or "").strip()
            if not raw:
                continue
            try:
                oncall_args[arg_name] = float(raw)
            except ValueError:
                continue
        min_rejected = (
            env_map.get("AI_TRADING_INCIDENT_MIN_REJECTED_RECORDS_FOR_CONCENTRATION")
            or ""
        ).strip()
        if min_rejected:
            try:
                oncall_args["min_rejected_records_for_concentration"] = int(min_rejected)
            except ValueError:
                pass
        if _bool_env(env_map.get("AI_TRADING_CONNECTOR_ONCALL_FORCE"), default=False):
            oncall_args["force"] = True

        summary["oncall"]["attempted"] = True
        try:
            summary["oncall"]["result"] = oncall_notifier(oncall_args)
        except Exception as exc:  # pragma: no cover - runtime guard
            summary["oncall"]["error"] = str(exc)
            summary["errors"].append(
                {
                    "connector": "oncall",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    else:
        summary["oncall"]["skipped_reason"] = "disabled"

    summary["finished_at"] = datetime.now(UTC).isoformat()
    summary["ok"] = len(summary["errors"]) == 0
    return summary


def main(argv: list[str] | None = None) -> int:
    _ = argv
    _load_runtime_env_defaults()
    slack_notifier, slack_eod_notifier, linear_creator, oncall_notifier = _load_connector_callables()
    summary = run_dispatch(
        slack_notifier=slack_notifier,
        slack_eod_notifier=slack_eod_notifier,
        linear_creator=linear_creator,
        oncall_notifier=oncall_notifier,
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
