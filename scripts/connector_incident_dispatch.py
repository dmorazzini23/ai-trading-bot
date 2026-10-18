#!/usr/bin/env python3
"""Dispatch runtime incident connectors (Slack) as a periodic job."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error as urllib_error

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SlackIncidentNotifier = Callable[[dict[str, Any]], dict[str, Any]]
SlackEodNotifier = Callable[[dict[str, Any]], dict[str, Any]]

def _bool_env(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _float_env(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int_env(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


_DEFAULT_CONNECTOR_HEALTH_PORT = 9001

def _connector_health_port(env_map: Mapping[str, str]) -> int:
    for key in (
        "AI_TRADING_CONNECTOR_HEALTH_PORT",
        "HEALTHCHECK_PORT",
        "API_PORT",
    ):
        parsed = _int_env(env_map.get(key))
        if parsed is not None and parsed > 0:
            return parsed
    return _DEFAULT_CONNECTOR_HEALTH_PORT


def _resolve_runtime_env_path(*, env: Mapping[str, str], repo_root: Path) -> Path:
    raw = str(env.get("AI_TRADING_RUNTIME_ENV_PATH") or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        return candidate
    repo_runtime = (repo_root / "runtime" / "ai-trading-runtime.env").resolve()
    if repo_root != REPO_ROOT and repo_runtime.exists():
        return repo_runtime
    packaged_runtime = Path("/run/ai-trading-bot/ai-trading-runtime.env")
    if packaged_runtime.exists():
        return packaged_runtime
    return repo_runtime


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


def _csv_env_keys(raw: str | None) -> set[str]:
    return {
        item.strip().upper()
        for item in str(raw or "").split(",")
        if item.strip()
    }


def _load_managed_connector_secret_defaults(env: dict[str, str]) -> dict[str, Any]:
    """Hydrate connector-only secret values from the configured backend in memory."""

    backend = str(env.get("AI_TRADING_SECRETS_BACKEND", "none") or "none").strip().lower()
    if backend in {"", "none", "off", "disabled"}:
        return {"secrets_backend": backend or "none", "hydrated": 0}
    if backend not in {"aws-secrets-manager", "aws_sm", "aws"}:
        return {"secrets_backend": backend, "hydrated": 0, "error": "unsupported_backend"}

    secret_id = str(env.get("AI_TRADING_AWS_SECRET_ID") or "").strip()
    if not secret_id:
        return {"secrets_backend": backend, "hydrated": 0, "error": "missing_secret_id"}

    explicit_keys = _csv_env_keys(env.get("AI_TRADING_MANAGED_SECRET_KEYS"))
    excluded_keys = _csv_env_keys(env.get("AI_TRADING_EXCLUDED_MANAGED_SECRET_KEYS"))
    candidate_keys = {
        "AI_TRADING_SLACK_WEBHOOK_URL",
        "SLACK_WEBHOOK_URL",
    }
    if explicit_keys:
        candidate_keys &= explicit_keys
    candidate_keys -= excluded_keys
    missing_keys = {
        key for key in candidate_keys if not str(env.get(key) or "").strip()
    }
    if not missing_keys:
        return {"secrets_backend": backend, "hydrated": 0}

    try:
        from ai_trading.config.managed_secrets import fetch_aws_secret_payload

        payload = fetch_aws_secret_payload(
            secret_id,
            region=str(env.get("AI_TRADING_AWS_REGION") or "").strip(),
            profile=str(env.get("AI_TRADING_AWS_PROFILE") or "").strip(),
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        return {
            "secrets_backend": backend,
            "hydrated": 0,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }

    hydrated = 0
    for key in sorted(missing_keys):
        value = str(payload.get(key) or "").strip()
        if not value:
            continue
        env[key] = value
        hydrated += 1
    return {"secrets_backend": backend, "hydrated": hydrated}


def _load_connector_callables() -> tuple[SlackIncidentNotifier, SlackEodNotifier]:
    from tools import mcp_slack_alerts_server as slack_srv

    return (
        slack_srv.tool_notify_incident_channel,
        slack_srv.tool_notify_eod_summary,
    )


def run_dispatch(
    *,
    env: Mapping[str, str] | None = None,
    slack_notifier: SlackIncidentNotifier,
    slack_eod_notifier: SlackEodNotifier,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    managed_secret_summary = _load_managed_connector_secret_defaults(env_map)
    summary: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "slack": {"enabled": False, "attempted": False},
        "slack_eod": {"enabled": False, "attempted": False},
        "managed_connector_secrets": managed_secret_summary,
        "errors": [],
    }

    webhook = (env_map.get("AI_TRADING_SLACK_WEBHOOK_URL") or env_map.get("SLACK_WEBHOOK_URL") or "").strip()
    health_port = _connector_health_port(env_map)

    slack_enabled = _bool_env(env_map.get("AI_TRADING_CONNECTOR_SLACK_ENABLED"), default=True)
    summary["slack"]["enabled"] = slack_enabled
    if slack_enabled:
        if webhook:
            slack_args: dict[str, Any] = {
                "webhook_url": webhook,
                "health_port": health_port,
                "on_change_only": _bool_env(
                    env_map.get("AI_TRADING_CONNECTOR_SLACK_ON_CHANGE_ONLY"),
                    default=True,
                ),
            }
            health_timeout_s = _float_env(
                env_map.get("AI_TRADING_CONNECTOR_HEALTH_TIMEOUT_S")
            )
            if health_timeout_s is not None:
                slack_args["health_timeout_s"] = health_timeout_s
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
                "health_port": health_port,
                "require_market_closed": _bool_env(
                    env_map.get("AI_TRADING_SLACK_EOD_REQUIRE_MARKET_CLOSED"),
                    default=True,
                ),
                "require_after_hours_training": _bool_env(
                    env_map.get("AI_TRADING_SLACK_EOD_REQUIRE_AFTER_HOURS_TRAINING"),
                    default=True,
                ),
                "block_on_training_gate": _bool_env(
                    env_map.get("AI_TRADING_SLACK_EOD_BLOCK_ON_TRAINING_GATE"),
                    default=False,
                ),
            }
            health_timeout_s = _float_env(
                env_map.get("AI_TRADING_CONNECTOR_HEALTH_TIMEOUT_S")
            )
            if health_timeout_s is not None:
                slack_eod_args["health_timeout_s"] = health_timeout_s
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

    summary["finished_at"] = datetime.now(UTC).isoformat()
    summary["ok"] = len(summary["errors"]) == 0
    return summary


SCHEDULED_REPORTS = ("preflight", "close-recap")


_PREFLIGHT_VERDICT_SEVERITY = {"ready": "OK", "blocked": "WARNING"}


def format_preflight_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Render the market_preflight health report as a Slack Block Kit payload."""
    from ai_trading.monitoring import slack_format as fmt

    if report.get("report_status") != "complete":
        title = "Market preflight failed — health endpoint unreadable"
        detail = f"{report.get('error_type')}: {report.get('error')}"
        return fmt.message(
            fmt.fallback_text(fmt.Severity.CRITICAL, title, detail),
            [
                fmt.header_block(fmt.Severity.CRITICAL, title),
                fmt.section_block(f"*What happened*\n{fmt.truncate(detail, 500)}"),
                fmt.section_block("*Action*\n• Check `ai-trading.service` before the open."),
            ],
        )
    broker = report.get("broker") if isinstance(report.get("broker"), Mapping) else {}
    provider = report.get("data_provider") if isinstance(report.get("data_provider"), Mapping) else {}
    service = report.get("service_state") if isinstance(report.get("service_state"), Mapping) else {}
    verdict = str(report.get("readiness_verdict") or "unknown").strip().lower()
    severity = fmt.Severity[_PREFLIGHT_VERDICT_SEVERITY.get(verdict, "WARNING")]
    flags = [fmt.humanize(flag) for flag in report.get("attention_flags") or []]
    reason = fmt.humanize(report.get("reason"))
    title = f"Market preflight · {verdict}"
    return fmt.message(
        fmt.fallback_text(severity, title, reason, f"{len(flags)} flags"),
        [
            fmt.header_block(severity, title),
            fmt.fields_block(
                [
                    (
                        "Health",
                        f"{fmt.capitalized(report.get('status'))} (HTTP {report.get('http_status')}) — {reason}",
                    ),
                    ("Service", f"{fmt.humanize(service.get('status'))} ({fmt.humanize(service.get('phase'))})"),
                    (
                        "Broker",
                        f"{fmt.humanize(broker.get('status'))} — {fmt.fmt_count(broker.get('positions_count'))} "
                        f"positions, {fmt.fmt_count(broker.get('open_orders_count'))} open orders",
                    ),
                    (
                        "Data provider",
                        f"{fmt.capitalized(provider.get('active') or 'n/a')} — {fmt.humanize(provider.get('status'))}",
                    ),
                ]
            ),
            fmt.section_block(f"*Open flags*\n{fmt.bullets(flags)}"),
            fmt.context_block(
                "A blocked verdict reflects model/replay gates, not service liveness.",
                fmt.now_label(),
            ),
        ],
    )


def build_scheduled_report(kind: str) -> dict[str, Any]:
    """Build a scheduled report as a Slack webhook payload ({"text", "blocks"})."""
    if kind == "preflight":
        from ai_trading.tools import market_preflight

        try:
            report: dict[str, Any] = market_preflight.collect_health()
        except (urllib_error.URLError, OSError, ValueError) as exc:
            report = {"report_status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        return format_preflight_report(report)
    if kind == "close-recap":
        spec = importlib.util.spec_from_file_location(
            "market_close_recap", REPO_ROOT / "scripts" / "market_close_recap.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("market_close_recap.py not found")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return dict(module.build_recap_payload())
    raise ValueError(f"unknown scheduled report: {kind}")


def run_scheduled_report(
    kind: str,
    *,
    env: Mapping[str, str] | None = None,
    poster: Callable[[str, dict[str, Any]], int] | None = None,
) -> dict[str, Any]:
    """Build a scheduled report and post it through the Slack incident webhook."""
    env_map = dict(os.environ if env is None else env)
    secrets = _load_managed_connector_secret_defaults(env_map)
    webhook = (env_map.get("AI_TRADING_SLACK_WEBHOOK_URL") or env_map.get("SLACK_WEBHOOK_URL") or "").strip()
    summary: dict[str, Any] = {"report": kind, "managed_connector_secrets": secrets, "sent": False}
    if not webhook:
        summary["skipped_reason"] = "missing_webhook"
        return summary
    payload = build_scheduled_report(kind)
    if poster is None:
        from tools import mcp_slack_alerts_server as slack_srv

        poster = slack_srv._post_slack_message
    summary["status_code"] = poster(webhook, payload)
    summary["sent"] = True
    return summary


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _load_runtime_env_defaults()
    if args[:1] == ["--report"]:
        kind = args[1] if len(args) > 1 else ""
        if kind not in SCHEDULED_REPORTS:
            print(json.dumps({"error": f"--report must be one of {', '.join(SCHEDULED_REPORTS)}"}))
            return 2
        result = run_scheduled_report(kind)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("sent") or result.get("skipped_reason") else 1
    slack_notifier, slack_eod_notifier = _load_connector_callables()
    summary = run_dispatch(
        slack_notifier=slack_notifier,
        slack_eod_notifier=slack_eod_notifier,
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
