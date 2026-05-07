#!/usr/bin/env python3
"""Dispatch runtime incident connectors (Slack + OpenClaw) as a periodic job."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SlackIncidentNotifier = Callable[[dict[str, Any]], dict[str, Any]]
SlackEodNotifier = Callable[[dict[str, Any]], dict[str, Any]]
OpenClawIncidentNotifier = Callable[[dict[str, Any]], dict[str, Any]]
OpenClawModelReadinessNotifier = Callable[[dict[str, Any]], dict[str, Any]]
IncidentSnapshotBuilder = Callable[[dict[str, Any]], dict[str, Any]]
_DEFAULT_INCIDENT_REPEAT_COOLDOWN_MINUTES = 45


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


def _float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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


def _int_value(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_SEVERITY_RANKS = {
    "info": 0,
    "warning": 1,
    "error": 2,
    "critical": 3,
}
_DEFAULT_CONNECTOR_HEALTH_PORT = 9001


def _normalize_severity(value: Any, *, default: str = "warning") -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _SEVERITY_RANKS else default


def _severity_at_least(severity: str, minimum: str) -> bool:
    return _SEVERITY_RANKS[severity] >= _SEVERITY_RANKS[minimum]


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


def _load_connector_callables() -> tuple[
    SlackIncidentNotifier,
    SlackEodNotifier,
    OpenClawIncidentNotifier,
    OpenClawModelReadinessNotifier,
    IncidentSnapshotBuilder,
]:
    from tools import mcp_slack_alerts_server as slack_srv

    return (
        slack_srv.tool_notify_incident_channel,
        slack_srv.tool_notify_eod_summary,
        _notify_openclaw_incident,
        _notify_openclaw_rl_overlay_readiness,
        slack_srv.tool_runtime_incident_snapshot,
    )


def _parse_iso_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _load_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_openclaw_runtime_target(env_map: Mapping[str, str]) -> dict[str, str] | None:
    explicit_url = (
        env_map.get("AI_TRADING_OPENCLAW_RUNTIME_WEBHOOK_URL")
        or env_map.get("OPENCLAW_RUNTIME_WEBHOOK_URL")
        or ""
    ).strip()
    explicit_token = (
        env_map.get("AI_TRADING_OPENCLAW_HOOK_TOKEN")
        or env_map.get("OPENCLAW_HOOK_TOKEN")
        or ""
    ).strip()
    if explicit_url and explicit_token:
        return {"url": explicit_url, "token": explicit_token}

    raw_cfg_path = (
        env_map.get("AI_TRADING_OPENCLAW_CONFIG_PATH")
        or env_map.get("OPENCLAW_CONFIG_PATH")
        or "~/.openclaw/openclaw.json"
    ).strip()
    cfg_path = Path(raw_cfg_path).expanduser()
    cfg = _read_json_file(cfg_path)
    hooks = cfg.get("hooks") if isinstance(cfg.get("hooks"), dict) else {}
    token = explicit_token or str(hooks.get("token") or "").strip()
    hook_path = str(hooks.get("path") or "/hooks").strip() or "/hooks"
    if not token:
        return None
    if explicit_url:
        return {"url": explicit_url, "token": token}
    hook_path = "/" + hook_path.strip("/ ")
    gateway_url = (
        env_map.get("AI_TRADING_OPENCLAW_GATEWAY_URL")
        or env_map.get("OPENCLAW_GATEWAY_URL")
        or "http://127.0.0.1:18789"
    ).strip().rstrip("/")
    return {
        "url": f"{gateway_url}{hook_path}/runtime",
        "token": token,
    }


def _post_openclaw_runtime_event(
    *,
    url: str,
    token: str,
    payload: dict[str, Any],
    timeout_s: float,
    idempotency_key: str | None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            **(
                {"Idempotency-Key": idempotency_key}
                if idempotency_key
                else {}
            ),
        },
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout_s) as response:
            raw_body = response.read().decode("utf-8", errors="replace").strip()
            parsed: Any = None
            if raw_body:
                try:
                    parsed = json.loads(raw_body)
                except json.JSONDecodeError:
                    parsed = raw_body
            return {
                "status_code": int(response.getcode()),
                "body": parsed,
            }
    except urllib_error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"OpenClaw webhook returned HTTP {exc.code}: {details or exc.reason}"
        ) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"OpenClaw webhook request failed: {exc.reason}") from exc


def _build_openclaw_runtime_payload(snapshot_result: dict[str, Any]) -> dict[str, Any]:
    snapshot = snapshot_result.get("snapshot") if isinstance(snapshot_result.get("snapshot"), dict) else {}
    triggers_raw = snapshot_result.get("triggers")
    triggers = [str(item).strip() for item in triggers_raw if str(item).strip()] if isinstance(triggers_raw, list) else []
    health_status = str(snapshot.get("health_status") or "unknown").strip()
    health_reason = str(snapshot.get("health_reason") or "unknown").strip()
    provider_status = str(snapshot.get("provider_status") or "unknown").strip()
    broker_status = str(snapshot.get("broker_status") or "unknown").strip()
    failed_checks = snapshot.get("go_no_go_failed_checks")
    failed_checks_list = [str(item).strip() for item in failed_checks if str(item).strip()] if isinstance(failed_checks, list) else []
    severity = "warning"
    if not bool(snapshot.get("health_ok", False)) or failed_checks_list:
        severity = "error"
    if health_status.lower() not in {"ready", "ok", "healthy"} and health_status:
        severity = "error"
    if broker_status.lower() in {"down", "disconnected", "unreachable"}:
        severity = "critical"
    summary_parts = []
    if triggers:
        summary_parts.append(f"Incident triggers: {', '.join(triggers[:3])}")
    else:
        summary_parts.append("Runtime incident connector forced an alert.")
    summary_parts.append(f"health={health_status} ({health_reason})")
    summary = " | ".join(summary_parts)
    suggested_action = "Run /triage immediately."
    if severity == "warning":
        suggested_action = "Run /triage soon and confirm whether intervention is needed."
    details = {
        "triggers": triggers,
        "health_status": health_status,
        "health_reason": health_reason,
        "provider_status": provider_status,
        "provider_reason": snapshot.get("provider_reason"),
        "broker_status": broker_status,
        "go_no_go_failed_checks": failed_checks_list,
        "top_rejection_concentration_gate": snapshot.get("top_rejection_concentration_gate"),
        "top_rejection_concentration_ratio": snapshot.get("top_rejection_concentration_ratio"),
        "fingerprint": snapshot_result.get("fingerprint"),
        "snapshot_timestamp": snapshot.get("timestamp"),
    }
    return {
        "source": "runtime",
        "service": "ai-trading.service",
        "severity": severity,
        "summary": summary,
        "suggestedAction": suggested_action,
        "details": details,
    }


def _openclaw_incident_severity(snapshot_result: dict[str, Any]) -> str:
    return _normalize_severity(
        _build_openclaw_runtime_payload(snapshot_result).get("severity"),
        default="warning",
    )


def _notify_openclaw_incident(args: dict[str, Any]) -> dict[str, Any]:
    snapshot_result = args.get("snapshot_result")
    if not isinstance(snapshot_result, dict):
        raise RuntimeError("snapshot_result is required")
    env_map = args.get("env")
    if not isinstance(env_map, Mapping):
        raise RuntimeError("env is required")
    force = bool(args.get("force"))
    triggers_raw = snapshot_result.get("triggers")
    triggers = [str(item).strip() for item in triggers_raw if str(item).strip()] if isinstance(triggers_raw, list) else []
    should_alert = bool(snapshot_result.get("should_alert")) or force
    fingerprint = str(snapshot_result.get("fingerprint") or "").strip()
    signature = str(snapshot_result.get("incident_signature") or fingerprint)
    snapshot = snapshot_result.get("snapshot") if isinstance(snapshot_result.get("snapshot"), dict) else {}
    if not should_alert:
        return {
            "sent": False,
            "reason": "no_incident_triggered",
            "fingerprint": fingerprint,
            "incident_signature": signature,
            "snapshot": snapshot,
            "triggers": triggers,
        }

    state_path_raw = str(args.get("state_path") or "").strip()
    state_path = Path(state_path_raw) if state_path_raw else Path("/var/lib/ai-trading-bot/runtime/openclaw_incident_state.json")
    prior = _load_state(state_path)
    prior_fp = str(prior.get("fingerprint") or "").strip()
    prior_signature = str(prior.get("incident_signature") or "").strip()
    prior_sent_at = _parse_iso_ts(prior.get("sent_at"))
    on_change_only = bool(args.get("on_change_only", True))
    repeat_cooldown_minutes = int(
        args.get("repeat_cooldown_minutes") or _DEFAULT_INCIDENT_REPEAT_COOLDOWN_MINUTES
    )
    min_interval_minutes = int(args.get("min_interval_minutes") or 0)
    now = datetime.now(UTC)
    if (
        min_interval_minutes > 0
        and prior_sent_at is not None
        and not force
    ):
        elapsed = now - prior_sent_at
        min_interval = timedelta(minutes=min_interval_minutes)
        if elapsed < min_interval:
            return {
                "sent": False,
                "reason": "min_interval_active",
                "fingerprint": fingerprint,
                "incident_signature": signature,
                "snapshot": snapshot,
                "triggers": triggers,
                "next_eligible_at": (prior_sent_at + min_interval).isoformat().replace("+00:00", "Z"),
            }
    if on_change_only and prior_signature and prior_signature == signature and not force:
        if repeat_cooldown_minutes > 0 and prior_sent_at is not None:
            elapsed = now - prior_sent_at
            cooldown = timedelta(minutes=repeat_cooldown_minutes)
            if elapsed < cooldown:
                return {
                    "sent": False,
                    "reason": "repeat_cooldown_active",
                    "fingerprint": fingerprint,
                    "incident_signature": signature,
                    "snapshot": snapshot,
                    "triggers": triggers,
                    "next_eligible_at": (prior_sent_at + cooldown).isoformat().replace("+00:00", "Z"),
                }
        else:
            return {
                "sent": False,
                "reason": "duplicate_signature",
                "fingerprint": fingerprint,
                "incident_signature": signature,
                "snapshot": snapshot,
                "triggers": triggers,
            }
    if on_change_only and prior_fp == fingerprint and not force:
        return {
            "sent": False,
            "reason": "duplicate_fingerprint",
            "fingerprint": fingerprint,
            "incident_signature": signature,
            "snapshot": snapshot,
            "triggers": triggers,
        }

    target = _resolve_openclaw_runtime_target(env_map)
    if target is None:
        return {
            "sent": False,
            "reason": "missing_openclaw_target",
            "fingerprint": fingerprint,
            "incident_signature": signature,
            "snapshot": snapshot,
            "triggers": triggers,
        }
    timeout_s = float(args.get("timeout_s") or 5.0)
    post_result = _post_openclaw_runtime_event(
        url=target["url"],
        token=target["token"],
        payload=_build_openclaw_runtime_payload(snapshot_result),
        timeout_s=timeout_s,
        idempotency_key=fingerprint or None,
    )
    state_payload = {
        "fingerprint": fingerprint,
        "incident_signature": signature,
        "sent_at": _utc_now_iso(),
        "triggers": triggers,
        "snapshot": snapshot,
        "status_code": post_result["status_code"],
    }
    _save_state(state_path, state_payload)
    return {
        "sent": True,
        "fingerprint": fingerprint,
        "incident_signature": signature,
        "snapshot": snapshot,
        "triggers": triggers,
        "status_code": post_result["status_code"],
        "state_path": str(state_path),
        "target_url": target["url"],
    }


def _resolve_after_hours_latest_path(env_map: Mapping[str, str]) -> Path:
    raw = str(
        env_map.get("AI_TRADING_AFTER_HOURS_REPORT_LATEST_PATH")
        or "/var/lib/ai-trading-bot/runtime/research_reports/after_hours_training_latest.json"
    ).strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _after_hours_report_bundle(env_map: Mapping[str, str]) -> dict[str, Any]:
    latest_path = _resolve_after_hours_latest_path(env_map)
    payload = _read_json_file(latest_path)
    report_path = latest_path
    if not payload:
        state_path = Path(
            str(
                env_map.get("AI_TRADING_AFTER_HOURS_STATE_PATH")
                or "/var/lib/ai-trading-bot/runtime/after_hours_training_state.json"
            ).strip()
        ).expanduser()
        if not state_path.is_absolute():
            state_path = (REPO_ROOT / state_path).resolve()
        state = _read_json_file(state_path)
        state_report_path = str(state.get("report_path") or "").strip()
        if state_report_path:
            report_path = Path(state_report_path).expanduser()
            payload = _read_json_file(report_path)
    report_obj = payload.get("report") if isinstance(payload.get("report"), dict) else payload
    return {
        "latest_path": str(latest_path),
        "report_path": str(payload.get("report_path") or report_path),
        "daily_report_path": payload.get("daily_report_path"),
        "updated_at": payload.get("updated_at") or report_obj.get("ts"),
        "report": report_obj if isinstance(report_obj, dict) else {},
    }


def _selected_after_hours_candidate(report: Mapping[str, Any]) -> dict[str, Any]:
    candidates = report.get("candidate_metrics")
    if isinstance(candidates, list):
        dict_candidates = [item for item in candidates if isinstance(item, dict)]
        for item in dict_candidates:
            if bool(item.get("selected", False)):
                return dict(item)
        if dict_candidates:
            return dict(
                sorted(
                    dict_candidates,
                    key=lambda item: int(item.get("rank") or 9999),
                )[0]
            )
    metrics = report.get("metrics")
    return dict(metrics) if isinstance(metrics, dict) else {}


def _readiness_required_runtime_checks(env_map: Mapping[str, str]) -> list[str]:
    raw = str(
        env_map.get("AI_TRADING_OPENCLAW_RL_READINESS_REQUIRED_RUNTIME_CHECKS")
        or "open_position_reconciliation_consistent,live_samples_sufficient"
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


def _evaluate_rl_overlay_readiness(env_map: Mapping[str, str]) -> dict[str, Any]:
    bundle = _after_hours_report_bundle(env_map)
    report = bundle["report"] if isinstance(bundle.get("report"), dict) else {}
    candidate = _selected_after_hours_candidate(report)
    reasons: list[str] = []

    if not report:
        reasons.append("after_hours_report_missing")

    min_expectancy_bps = _float_env(
        env_map.get("AI_TRADING_OPENCLAW_RL_READINESS_MIN_EXPECTANCY_BPS")
    )
    if min_expectancy_bps is None:
        min_expectancy_bps = 0.0
    min_profitable_folds = _int_env(
        env_map.get("AI_TRADING_OPENCLAW_RL_READINESS_MIN_PROFITABLE_FOLDS")
    )
    if min_profitable_folds is None:
        min_profitable_folds = 1

    expectancy_bps = _float_value(candidate.get("mean_expectancy_bps"))
    profitable_folds = _int_value(candidate.get("profitable_fold_count"))
    if expectancy_bps is None or expectancy_bps <= min_expectancy_bps:
        reasons.append("expectancy_not_positive")
    if profitable_folds is None or profitable_folds < min_profitable_folds:
        reasons.append("profitable_folds_insufficient")

    orientation = str(
        report.get("score_orientation")
        or candidate.get("score_orientation")
        or ""
    ).strip().lower()
    if orientation != "direct":
        reasons.append("score_orientation_not_direct")

    label_quality = report.get("label_quality")
    require_clean_labels = _bool_env(
        env_map.get("AI_TRADING_OPENCLAW_RL_READINESS_REQUIRE_CLEAN_LABELS"),
        default=True,
    )
    label_warnings: list[str] = []
    timestamp_violations = 0
    duplicate_symbol_ts_rows = 0
    if isinstance(label_quality, Mapping):
        raw_warnings = label_quality.get("warnings")
        if isinstance(raw_warnings, list):
            label_warnings = [str(item) for item in raw_warnings if str(item).strip()]
        timestamp_violations = int(label_quality.get("timestamp_order_violations") or 0)
        duplicate_symbol_ts_rows = int(
            label_quality.get("duplicate_symbol_timestamp_rows") or 0
        )
    elif require_clean_labels:
        reasons.append("label_quality_missing")
    if require_clean_labels and label_warnings:
        reasons.append("label_quality_warnings_present")
    if require_clean_labels and timestamp_violations > 0:
        reasons.append("label_timestamp_order_violations")
    if require_clean_labels and duplicate_symbol_ts_rows > 0:
        reasons.append("label_duplicate_symbol_timestamp_rows")

    runtime_gate = report.get("runtime_performance_gate")
    runtime_checks = (
        runtime_gate.get("checks")
        if isinstance(runtime_gate, Mapping) and isinstance(runtime_gate.get("checks"), Mapping)
        else {}
    )
    failed_required_runtime_checks: list[str] = []
    for check_name in _readiness_required_runtime_checks(env_map):
        if runtime_checks.get(check_name) is False:
            failed_required_runtime_checks.append(check_name)
    if failed_required_runtime_checks:
        reasons.append("runtime_data_quality_checks_failed")

    rl_overlay_enabled = _bool_env(
        env_map.get("AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED"),
        default=False,
    )
    only_when_disabled = _bool_env(
        env_map.get("AI_TRADING_OPENCLAW_RL_READINESS_ONLY_WHEN_RL_OVERLAY_DISABLED"),
        default=True,
    )
    if only_when_disabled and rl_overlay_enabled:
        reasons.append("rl_overlay_already_enabled")

    model_payload = report.get("model") if isinstance(report.get("model"), Mapping) else {}
    observed = {
        "model_id": model_payload.get("model_id")
        or model_payload.get("id")
        or report.get("model_id")
        or candidate.get("model_id")
        or "",
        "model_name": candidate.get("name") or report.get("model_name") or "",
        "report_path": bundle.get("report_path"),
        "updated_at": bundle.get("updated_at"),
        "mean_expectancy_bps": expectancy_bps,
        "min_expectancy_bps": float(min_expectancy_bps),
        "profitable_fold_count": profitable_folds,
        "min_profitable_folds": int(min_profitable_folds),
        "mean_hit_rate": _float_value(candidate.get("mean_hit_rate")),
        "score_orientation": orientation or None,
        "label_quality_warnings": label_warnings,
        "timestamp_order_violations": timestamp_violations,
        "duplicate_symbol_timestamp_rows": duplicate_symbol_ts_rows,
        "failed_required_runtime_checks": failed_required_runtime_checks,
        "rl_overlay_enabled": rl_overlay_enabled,
        "use_rl_agent": _bool_env(env_map.get("USE_RL_AGENT"), default=False),
    }
    fingerprint_material = {
        "model_id": observed["model_id"],
        "report_path": observed["report_path"],
        "criteria": {
            "min_expectancy_bps": observed["min_expectancy_bps"],
            "min_profitable_folds": observed["min_profitable_folds"],
            "required_runtime_checks": _readiness_required_runtime_checks(env_map),
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_material, sort_keys=True).encode("utf-8")
    ).hexdigest()
    ready = not reasons
    return {
        "ready": ready,
        "should_alert": ready,
        "reason": "criteria_met" if ready else "criteria_not_met",
        "failed_reasons": reasons,
        "fingerprint": fingerprint,
        "observed": observed,
    }


def _build_openclaw_rl_overlay_readiness_payload(
    readiness_result: Mapping[str, Any],
) -> dict[str, Any]:
    observed = (
        readiness_result.get("observed")
        if isinstance(readiness_result.get("observed"), Mapping)
        else {}
    )
    summary = (
        "After-hours ML candidate is ready to consider re-enabling RL overlay training."
    )
    return {
        "source": "runtime",
        "service": "ai-trading.service",
        "severity": "info",
        "summary": summary,
        "suggestedAction": (
            "Review the report, then set AI_TRADING_AFTER_HOURS_RL_OVERLAY_ENABLED=1 "
            "if you want RL overlay training to resume. Keep USE_RL_AGENT=0 unless "
            "live RL is separately approved."
        ),
        "details": {
            "event_type": "after_hours_rl_overlay_readiness",
            "reason": readiness_result.get("reason"),
            "failed_reasons": list(readiness_result.get("failed_reasons") or []),
            "model_id": observed.get("model_id"),
            "model_name": observed.get("model_name"),
            "report_path": observed.get("report_path"),
            "mean_expectancy_bps": observed.get("mean_expectancy_bps"),
            "profitable_fold_count": observed.get("profitable_fold_count"),
            "mean_hit_rate": observed.get("mean_hit_rate"),
            "score_orientation": observed.get("score_orientation"),
            "label_quality_warnings": observed.get("label_quality_warnings"),
            "failed_required_runtime_checks": observed.get(
                "failed_required_runtime_checks"
            ),
            "rl_overlay_enabled": observed.get("rl_overlay_enabled"),
            "use_rl_agent": observed.get("use_rl_agent"),
            "fingerprint": readiness_result.get("fingerprint"),
        },
    }


def _notify_openclaw_rl_overlay_readiness(args: dict[str, Any]) -> dict[str, Any]:
    readiness_result = args.get("readiness_result")
    if not isinstance(readiness_result, dict):
        raise RuntimeError("readiness_result is required")
    env_map = args.get("env")
    if not isinstance(env_map, Mapping):
        raise RuntimeError("env is required")

    fingerprint = str(readiness_result.get("fingerprint") or "").strip()
    should_alert = bool(readiness_result.get("should_alert", False)) or bool(
        args.get("force")
    )
    if not should_alert:
        return {
            "sent": False,
            "reason": readiness_result.get("reason") or "criteria_not_met",
            "fingerprint": fingerprint,
            "readiness": readiness_result,
        }

    state_path_raw = str(
        args.get("state_path")
        or env_map.get("AI_TRADING_OPENCLAW_RL_READINESS_STATE_PATH")
        or "/var/lib/ai-trading-bot/runtime/openclaw_rl_overlay_readiness_state.json"
    ).strip()
    state_path = Path(state_path_raw).expanduser()
    prior = _load_state(state_path)
    if (
        bool(args.get("on_change_only", True))
        and str(prior.get("fingerprint") or "").strip() == fingerprint
        and not bool(args.get("force"))
    ):
        return {
            "sent": False,
            "reason": "duplicate_fingerprint",
            "fingerprint": fingerprint,
            "state_path": str(state_path),
            "readiness": readiness_result,
        }

    target = _resolve_openclaw_runtime_target(env_map)
    if target is None:
        return {
            "sent": False,
            "reason": "missing_openclaw_target",
            "fingerprint": fingerprint,
            "readiness": readiness_result,
        }

    timeout_s = float(args.get("timeout_s") or 5.0)
    post_result = _post_openclaw_runtime_event(
        url=target["url"],
        token=target["token"],
        payload=_build_openclaw_rl_overlay_readiness_payload(readiness_result),
        timeout_s=timeout_s,
        idempotency_key=fingerprint or None,
    )
    _save_state(
        state_path,
        {
            "fingerprint": fingerprint,
            "sent_at": _utc_now_iso(),
            "readiness": readiness_result,
            "status_code": post_result["status_code"],
        },
    )
    return {
        "sent": True,
        "fingerprint": fingerprint,
        "status_code": post_result["status_code"],
        "state_path": str(state_path),
        "target_url": target["url"],
        "readiness": readiness_result,
    }


def run_dispatch(
    *,
    env: Mapping[str, str] | None = None,
    slack_notifier: SlackIncidentNotifier,
    slack_eod_notifier: SlackEodNotifier,
    openclaw_notifier: OpenClawIncidentNotifier,
    incident_snapshot_builder: IncidentSnapshotBuilder,
    openclaw_model_readiness_notifier: OpenClawModelReadinessNotifier | None = None,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    managed_secret_summary = _load_managed_connector_secret_defaults(env_map)
    summary: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "slack": {"enabled": False, "attempted": False},
        "slack_eod": {"enabled": False, "attempted": False},
        "openclaw": {"enabled": False, "attempted": False},
        "openclaw_model_readiness": {"enabled": False, "attempted": False},
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

    openclaw_enabled = _bool_env(
        env_map.get("AI_TRADING_CONNECTOR_OPENCLAW_ENABLED"),
        default=True,
    )
    summary["openclaw"]["enabled"] = openclaw_enabled
    if openclaw_enabled:
        openclaw_min_severity = _normalize_severity(
            env_map.get("AI_TRADING_CONNECTOR_OPENCLAW_MIN_SEVERITY"),
            default="warning",
        )
        summary["openclaw"]["min_severity"] = openclaw_min_severity
        openclaw_args: dict[str, Any] = {
            "on_change_only": _bool_env(
                env_map.get("AI_TRADING_CONNECTOR_OPENCLAW_ON_CHANGE_ONLY"),
                default=True,
            ),
            "repeat_cooldown_minutes": _DEFAULT_INCIDENT_REPEAT_COOLDOWN_MINUTES,
            "health_port": health_port,
            "env": env_map,
        }
        health_timeout_s = _float_env(
            env_map.get("AI_TRADING_CONNECTOR_HEALTH_TIMEOUT_S")
        )
        if health_timeout_s is not None:
            openclaw_args["health_timeout_s"] = health_timeout_s
            openclaw_args["timeout_s"] = health_timeout_s
        min_capture = (env_map.get("AI_TRADING_INCIDENT_MIN_CAPTURE_RATIO") or "").strip()
        if min_capture:
            try:
                openclaw_args["min_capture_ratio"] = float(min_capture)
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
                openclaw_args[arg_name] = float(raw)
            except ValueError:
                continue
        min_rejected = (
            env_map.get("AI_TRADING_INCIDENT_MIN_REJECTED_RECORDS_FOR_CONCENTRATION")
            or ""
        ).strip()
        if min_rejected:
            try:
                openclaw_args["min_rejected_records_for_concentration"] = int(min_rejected)
            except ValueError:
                pass
        state_path = (env_map.get("AI_TRADING_OPENCLAW_INCIDENT_STATE_PATH") or "").strip()
        if state_path:
            openclaw_args["state_path"] = state_path
        repeat_cooldown_minutes = _int_env(
            env_map.get("AI_TRADING_OPENCLAW_INCIDENT_REPEAT_COOLDOWN_MINUTES")
            or env_map.get("AI_TRADING_SLACK_INCIDENT_REPEAT_COOLDOWN_MINUTES")
        )
        if repeat_cooldown_minutes is not None:
            openclaw_args["repeat_cooldown_minutes"] = repeat_cooldown_minutes
        min_interval_minutes = _int_env(
            env_map.get("AI_TRADING_OPENCLAW_INCIDENT_MIN_INTERVAL_MINUTES")
            or env_map.get("AI_TRADING_SLACK_INCIDENT_MIN_INTERVAL_MINUTES")
        )
        if min_interval_minutes is not None:
            openclaw_args["min_interval_minutes"] = min_interval_minutes
        if _bool_env(
            env_map.get("AI_TRADING_CONNECTOR_OPENCLAW_FORCE"),
            default=False,
        ):
            openclaw_args["force"] = True

        summary["openclaw"]["attempted"] = True
        try:
            snapshot_result = incident_snapshot_builder(openclaw_args)
            openclaw_args["snapshot_result"] = snapshot_result
            incident_severity = _openclaw_incident_severity(snapshot_result)
            summary["openclaw"]["severity"] = incident_severity
            if _severity_at_least(incident_severity, openclaw_min_severity):
                summary["openclaw"]["result"] = openclaw_notifier(openclaw_args)
            else:
                summary["openclaw"]["result"] = {
                    "sent": False,
                    "reason": "below_min_severity",
                    "severity": incident_severity,
                    "min_severity": openclaw_min_severity,
                    "fingerprint": snapshot_result.get("fingerprint"),
                    "incident_signature": snapshot_result.get("incident_signature")
                    or snapshot_result.get("fingerprint"),
                    "triggers": snapshot_result.get("triggers") or [],
                }
        except Exception as exc:  # pragma: no cover - runtime guard
            summary["openclaw"]["error"] = str(exc)
            summary["errors"].append(
                {
                    "connector": "openclaw",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    else:
        summary["openclaw"]["skipped_reason"] = "disabled"

    model_readiness_enabled = _bool_env(
        env_map.get("AI_TRADING_CONNECTOR_OPENCLAW_MODEL_READINESS_ENABLED"),
        default=False,
    )
    summary["openclaw_model_readiness"]["enabled"] = model_readiness_enabled
    if model_readiness_enabled:
        model_readiness_args: dict[str, Any] = {
            "env": env_map,
            "on_change_only": _bool_env(
                env_map.get("AI_TRADING_CONNECTOR_OPENCLAW_MODEL_READINESS_ON_CHANGE_ONLY"),
                default=True,
            ),
        }
        health_timeout_s = _float_env(
            env_map.get("AI_TRADING_CONNECTOR_HEALTH_TIMEOUT_S")
        )
        if health_timeout_s is not None:
            model_readiness_args["timeout_s"] = health_timeout_s
        state_path = (
            env_map.get("AI_TRADING_OPENCLAW_RL_READINESS_STATE_PATH") or ""
        ).strip()
        if state_path:
            model_readiness_args["state_path"] = state_path
        if _bool_env(
            env_map.get("AI_TRADING_CONNECTOR_OPENCLAW_MODEL_READINESS_FORCE"),
            default=False,
        ):
            model_readiness_args["force"] = True

        summary["openclaw_model_readiness"]["attempted"] = True
        try:
            readiness_result = _evaluate_rl_overlay_readiness(env_map)
            model_readiness_args["readiness_result"] = readiness_result
            notifier = (
                openclaw_model_readiness_notifier
                or _notify_openclaw_rl_overlay_readiness
            )
            summary["openclaw_model_readiness"]["result"] = notifier(
                model_readiness_args
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            summary["openclaw_model_readiness"]["error"] = str(exc)
            summary["errors"].append(
                {
                    "connector": "openclaw_model_readiness",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    else:
        summary["openclaw_model_readiness"]["skipped_reason"] = "disabled"

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


def main(argv: list[str] | None = None) -> int:
    _ = argv
    _load_runtime_env_defaults()
    (
        slack_notifier,
        slack_eod_notifier,
        openclaw_notifier,
        openclaw_model_readiness_notifier,
        incident_snapshot_builder,
    ) = _load_connector_callables()
    summary = run_dispatch(
        slack_notifier=slack_notifier,
        slack_eod_notifier=slack_eod_notifier,
        openclaw_notifier=openclaw_notifier,
        incident_snapshot_builder=incident_snapshot_builder,
        openclaw_model_readiness_notifier=openclaw_model_readiness_notifier,
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
