"""On-call MCP server for Jira Service Management incident escalation."""

from __future__ import annotations

import base64
import importlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.mcp_common import ToolSpec

_mcp_common_mod = importlib.import_module(
    "tools.mcp_common" if __package__ == "tools" else "mcp_common"
)
run_tool_server = cast(Callable[..., int], getattr(_mcp_common_mod, "run_tool_server"))
utc_now_iso = cast(Callable[[], str], getattr(_mcp_common_mod, "utc_now_iso"))

_slack_mod = importlib.import_module(
    "tools.mcp_slack_alerts_server"
    if __package__ == "tools"
    else "mcp_slack_alerts_server"
)
_slack_runtime_incident_snapshot = cast(
    Callable[[dict[str, Any]], dict[str, Any]],
    getattr(_slack_mod, "tool_runtime_incident_snapshot"),
)

_DEFAULT_RUNTIME_ROOT = Path("/var/lib/ai-trading-bot/runtime")
_DEFAULT_STATE_PATH = _DEFAULT_RUNTIME_ROOT / "oncall_incident_state.json"
_SEVERITY_RANK: dict[str, int] = {
    "info": 0,
    "warning": 1,
    "error": 2,
    "critical": 3,
}


def _bool_arg(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _state_path(args: dict[str, Any]) -> Path:
    raw = (
        str(args.get("state_path") or "").strip()
        or os.getenv("AI_TRADING_ONCALL_STATE_PATH", "").strip()
    )
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_STATE_PATH


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def _post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout_s: float = 8.0,
) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        method="POST",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            return int(response.status), response_body
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request to {url} failed: {exc.reason}") from exc


def _incident_severity(triggers: list[str]) -> str:
    values = set(triggers)
    if "go_no_go_failed" in values or "broker_disconnected" in values:
        return "critical"
    if "health_degraded" in values:
        return "error"
    if values:
        return "warning"
    return "info"


def _normalized_severity(value: Any, *, default: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in _SEVERITY_RANK:
        return candidate
    return default


def _provider_min_severity(provider: str, args: dict[str, Any]) -> str:
    if provider == "jsm_ticket":
        return _normalized_severity(
            args.get("jsm_ticket_min_severity")
            or os.getenv("AI_TRADING_JSM_TICKET_MIN_SEVERITY", "").strip()
            or os.getenv("AI_TRADING_ONCALL_MIN_SEVERITY", "").strip(),
            default="warning",
        )
    return _normalized_severity(
        args.get("jsm_ops_min_severity")
        or os.getenv("AI_TRADING_JSM_OPS_MIN_SEVERITY", "").strip()
        or os.getenv("AI_TRADING_ONCALL_MIN_SEVERITY", "").strip(),
        default="info",
    )


def _meets_min_severity(*, severity: str, min_severity: str) -> bool:
    return _SEVERITY_RANK[severity] >= _SEVERITY_RANK[min_severity]


def _jsm_priority(severity: str) -> str:
    sev = severity.strip().lower()
    if sev == "critical":
        return "P1"
    if sev == "error":
        return "P2"
    if sev == "warning":
        return "P3"
    return "P4"


def _incident_summary(snapshot: dict[str, Any], triggers: list[str]) -> str:
    checks = ", ".join(list(snapshot.get("go_no_go_failed_checks") or [])) or "none"
    return (
        "ai-trading incident: "
        f"triggers={','.join(triggers) if triggers else 'none'} | "
        f"health={snapshot.get('health_status')}({snapshot.get('health_reason')}) | "
        f"broker={snapshot.get('broker_status')} | "
        f"capture={snapshot.get('execution_capture_ratio')} | "
        f"failed_checks={checks}"
    )


def _resolve_providers(args: dict[str, Any]) -> list[str]:
    known = {"jsm_ops", "jsm_ticket"}

    raw = args.get("providers")
    if isinstance(raw, str):
        requested = [chunk.strip().lower() for chunk in raw.split(",") if chunk.strip()]
    elif isinstance(raw, list):
        requested = [str(chunk).strip().lower() for chunk in raw if str(chunk).strip()]
    else:
        requested = []

    requested = [provider for provider in requested if provider in known]
    if requested:
        return sorted(set(requested))

    env_providers = str(os.getenv("AI_TRADING_ONCALL_PROVIDERS", "")).strip()
    if env_providers:
        from_env = [chunk.strip().lower() for chunk in env_providers.split(",") if chunk.strip()]
        filtered = [provider for provider in from_env if provider in known]
        if filtered:
            return sorted(set(filtered))

    providers: list[str] = []
    if _bool_arg(os.getenv("AI_TRADING_CONNECTOR_JSM_TICKET_ENABLED"), default=False):
        providers.append("jsm_ticket")
    if _bool_arg(os.getenv("AI_TRADING_CONNECTOR_JSM_OPS_ENABLED"), default=True):
        providers.append("jsm_ops")
    return sorted(set(providers))


def _jsm_alert_url(args: dict[str, Any]) -> str:
    cloud_id = (
        str(args.get("jsm_ops_cloud_id") or "").strip()
        or os.getenv("AI_TRADING_JSM_OPS_CLOUD_ID", "").strip()
    )
    base = (
        str(args.get("jsm_ops_base_url") or "").strip()
        or os.getenv("AI_TRADING_JSM_OPS_BASE_URL", "").strip()
    )
    if base:
        resolved = base.replace("{cloudId}", cloud_id) if "{cloudId}" in base else base
        normalized = resolved.rstrip("/")
        if normalized.endswith("/alerts"):
            return normalized
        return f"{normalized}/alerts"
    if not cloud_id:
        raise RuntimeError(
            "missing JSM Ops cloud id (jsm_ops_cloud_id arg or AI_TRADING_JSM_OPS_CLOUD_ID)"
        )
    return f"https://api.atlassian.com/jsm/ops/api/{cloud_id}/v1/alerts"


def _jsm_auth_headers(args: dict[str, Any]) -> dict[str, str]:
    api_key = (
        str(args.get("jsm_ops_api_key") or "").strip()
        or os.getenv("AI_TRADING_JSM_OPS_API_KEY", "").strip()
    )
    if api_key:
        return {"Authorization": f"GenieKey {api_key}"}

    email = (
        str(args.get("jsm_ops_email") or "").strip()
        or os.getenv("AI_TRADING_JSM_OPS_EMAIL", "").strip()
    )
    api_token = (
        str(args.get("jsm_ops_api_token") or "").strip()
        or os.getenv("AI_TRADING_JSM_OPS_API_TOKEN", "").strip()
    )
    if email and api_token:
        basic = base64.b64encode(f"{email}:{api_token}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {basic}"}

    bearer = (
        str(args.get("jsm_ops_bearer_token") or "").strip()
        or os.getenv("AI_TRADING_JSM_OPS_BEARER_TOKEN", "").strip()
    )
    if bearer:
        return {"Authorization": f"Bearer {bearer}"}

    raise RuntimeError(
        "missing JSM Ops auth (set AI_TRADING_JSM_OPS_EMAIL + AI_TRADING_JSM_OPS_API_TOKEN, "
        "or AI_TRADING_JSM_OPS_BEARER_TOKEN)"
    )


def _jsm_ticket_auth_headers(args: dict[str, Any]) -> dict[str, str]:
    email = (
        str(args.get("jsm_ops_email") or "").strip()
        or os.getenv("AI_TRADING_JSM_OPS_EMAIL", "").strip()
    )
    api_token = (
        str(args.get("jsm_ops_api_token") or "").strip()
        or os.getenv("AI_TRADING_JSM_OPS_API_TOKEN", "").strip()
    )
    if email and api_token:
        basic = base64.b64encode(f"{email}:{api_token}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {basic}"}

    bearer = (
        str(args.get("jsm_ops_bearer_token") or "").strip()
        or os.getenv("AI_TRADING_JSM_OPS_BEARER_TOKEN", "").strip()
    )
    if bearer:
        return {"Authorization": f"Bearer {bearer}"}

    raise RuntimeError(
        "missing JSM ticket auth (set AI_TRADING_JSM_OPS_EMAIL + AI_TRADING_JSM_OPS_API_TOKEN, "
        "or AI_TRADING_JSM_OPS_BEARER_TOKEN)"
    )


def _jsm_ticket_url(args: dict[str, Any]) -> str:
    site_url = (
        str(args.get("jsm_site_url") or "").strip()
        or os.getenv("AI_TRADING_JSM_SITE_URL", "").strip()
    )
    if not site_url:
        raise RuntimeError("missing JSM site URL (jsm_site_url arg or AI_TRADING_JSM_SITE_URL)")
    return f"{site_url.rstrip('/')}/rest/api/3/issue"


def _parse_csv(raw: str) -> list[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _issue_description_adf(
    *,
    snapshot: dict[str, Any],
    triggers: list[str],
    fingerprint: str,
    severity: str,
) -> dict[str, Any]:
    checks = ", ".join(list(snapshot.get("go_no_go_failed_checks") or [])) or "none"
    text = (
        f"ai-trading runtime incident\n"
        f"severity={severity}\n"
        f"fingerprint={fingerprint}\n"
        f"triggers={','.join(triggers) if triggers else 'none'}\n"
        f"health={snapshot.get('health_status')} ({snapshot.get('health_reason')})\n"
        f"broker={snapshot.get('broker_status')}\n"
        f"provider={snapshot.get('provider_active')} ({snapshot.get('provider_status')})\n"
        f"capture={snapshot.get('execution_capture_ratio')}\n"
        f"slippage_drag_bps={snapshot.get('slippage_drag_bps')}\n"
        f"failed_checks={checks}\n"
        f"timestamp={snapshot.get('timestamp')}"
    )
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


def _notify_jsm_ticket(
    *,
    args: dict[str, Any],
    fingerprint: str,
    severity: str,
    snapshot: dict[str, Any],
    triggers: list[str],
) -> dict[str, Any]:
    issue_url = _jsm_ticket_url(args)
    headers = _jsm_ticket_auth_headers(args)
    project_key = (
        str(args.get("jsm_ticket_project_key") or "").strip()
        or os.getenv("AI_TRADING_JSM_TICKET_PROJECT_KEY", "").strip()
    )
    if not project_key:
        raise RuntimeError(
            "missing JSM ticket project key (jsm_ticket_project_key arg or AI_TRADING_JSM_TICKET_PROJECT_KEY)"
        )
    issue_type = (
        str(args.get("jsm_ticket_issue_type") or "").strip()
        or os.getenv("AI_TRADING_JSM_TICKET_ISSUE_TYPE", "").strip()
        or "Task"
    )
    labels_raw = (
        str(args.get("jsm_ticket_labels") or "").strip()
        or os.getenv("AI_TRADING_JSM_TICKET_LABELS", "").strip()
        or "ai-trading,runtime-incident"
    )
    labels = sorted(set(_parse_csv(labels_raw) + [f"severity-{severity.lower()}"]))
    summary = f"[ai-trading] {severity.upper()} runtime incident ({fingerprint[:8]})"
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": _issue_description_adf(
                snapshot=snapshot,
                triggers=triggers,
                fingerprint=fingerprint,
                severity=severity,
            ),
            "issuetype": {"name": issue_type},
            "labels": labels,
        }
    }
    timeout_s = float(args.get("timeout_s") or 8.0)
    status_code, body = _post_json(url=issue_url, payload=payload, headers=headers, timeout_s=timeout_s)
    issue_key: str | None = None
    issue_id: str | None = None
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            maybe_key = parsed.get("key")
            maybe_id = parsed.get("id")
            issue_key = str(maybe_key) if maybe_key is not None else None
            issue_id = str(maybe_id) if maybe_id is not None else None
    except json.JSONDecodeError:
        pass
    return {
        "provider": "jsm_ticket",
        "sent": True,
        "status_code": status_code,
        "issue_key": issue_key,
        "issue_id": issue_id,
        "response": body,
    }


def _notify_jsm_ops(
    *,
    args: dict[str, Any],
    fingerprint: str,
    severity: str,
    snapshot: dict[str, Any],
    triggers: list[str],
) -> dict[str, Any]:
    url = _jsm_alert_url(args)
    headers = _jsm_auth_headers(args)
    using_custom_base = bool(
        str(args.get("jsm_ops_base_url") or "").strip()
        or os.getenv("AI_TRADING_JSM_OPS_BASE_URL", "").strip()
    )

    source = str(args.get("source") or os.getenv("AI_TRADING_ONCALL_SOURCE", "ai-trading")).strip()
    alias_prefix = str(args.get("alias_prefix") or "ai-trading").strip()
    alias = f"{alias_prefix}:{fingerprint}"
    payload = {
        "message": _incident_summary(snapshot, triggers)[:130],
        "description": _incident_summary(snapshot, triggers),
        "alias": alias,
        "priority": _jsm_priority(severity),
        "source": source,
        "tags": ["ai-trading", "runtime"] + sorted(set(triggers)),
        "details": {
            "fingerprint": fingerprint,
            "triggers": ",".join(triggers),
            "health_status": snapshot.get("health_status"),
            "health_reason": snapshot.get("health_reason"),
            "broker_status": snapshot.get("broker_status"),
            "execution_capture_ratio": snapshot.get("execution_capture_ratio"),
            "slippage_drag_bps": snapshot.get("slippage_drag_bps"),
        },
    }
    timeout_s = float(args.get("timeout_s") or 8.0)
    try:
        status_code, body = _post_json(url=url, payload=payload, headers=headers, timeout_s=timeout_s)
    except RuntimeError as exc:
        if "HTTP 404" in str(exc) and not using_custom_base:
            raise RuntimeError(
                "JSM Ops cloud alerts endpoint returned 404. "
                "Set AI_TRADING_JSM_OPS_BASE_URL to your JSM Ops alert API integration URL "
                "(often from Integrations/API) and authenticate via AI_TRADING_JSM_OPS_API_KEY."
            ) from exc
        raise
    return {"provider": "jsm_ops", "sent": True, "status_code": status_code, "response": body}


def tool_runtime_incident_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    return _slack_runtime_incident_snapshot(args)


def _notify_oncall(args: dict[str, Any], providers: list[str] | None = None) -> dict[str, Any]:
    snapshot_payload = _slack_runtime_incident_snapshot(args)
    snapshot = cast(dict[str, Any], snapshot_payload.get("snapshot") or {})
    triggers = cast(list[str], snapshot_payload.get("triggers") or [])
    fingerprint = str(snapshot_payload.get("fingerprint") or "")

    force = _bool_arg(args.get("force"), default=False)
    on_change_only = _bool_arg(
        args.get("on_change_only"),
        default=_bool_arg(os.getenv("AI_TRADING_ONCALL_ON_CHANGE_ONLY"), default=True),
    )
    should_alert = bool(triggers) or force
    if not should_alert:
        return {
            "sent": False,
            "reason": "no_incident_triggered",
            "triggers": triggers,
            "fingerprint": fingerprint,
            "snapshot": snapshot,
        }

    state_path = _state_path(args)
    prior = _load_state(state_path)
    prior_fp = str(prior.get("fingerprint") or "")
    if on_change_only and prior_fp == fingerprint and not force:
        return {
            "sent": False,
            "reason": "duplicate_fingerprint",
            "fingerprint": fingerprint,
            "state_path": str(state_path),
            "triggers": triggers,
            "snapshot": snapshot,
        }

    selected = providers if providers is not None else _resolve_providers(args)
    if not selected:
        return {
            "sent": False,
            "reason": "no_oncall_provider_enabled",
            "triggers": triggers,
            "fingerprint": fingerprint,
            "snapshot": snapshot,
        }

    severity = _incident_severity(triggers)
    deliveries: list[dict[str, Any]] = []
    for provider in selected:
        min_severity = _provider_min_severity(provider, args)
        if not force and not _meets_min_severity(severity=severity, min_severity=min_severity):
            deliveries.append(
                {
                    "provider": provider,
                    "sent": False,
                    "reason": "severity_below_minimum",
                    "severity": severity,
                    "min_severity": min_severity,
                }
            )
            continue
        if provider == "jsm_ops":
            deliveries.append(
                _notify_jsm_ops(
                    args=args,
                    fingerprint=fingerprint,
                    severity=severity,
                    snapshot=snapshot,
                    triggers=triggers,
                )
            )
        elif provider == "jsm_ticket":
            deliveries.append(
                _notify_jsm_ticket(
                    args=args,
                    fingerprint=fingerprint,
                    severity=severity,
                    snapshot=snapshot,
                    triggers=triggers,
                )
            )

    sent_any = any(bool(item.get("sent")) for item in deliveries)
    if sent_any:
        _save_state(
            state_path,
            {
                "fingerprint": fingerprint,
                "sent_at": utc_now_iso(),
                "triggers": triggers,
                "severity": severity,
                "deliveries": deliveries,
                "snapshot": snapshot,
            },
        )
    response = {
        "sent": sent_any,
        "providers": selected,
        "severity": severity,
        "fingerprint": fingerprint,
        "triggers": triggers,
        "deliveries": deliveries,
        "state_path": str(state_path),
        "snapshot": snapshot,
    }
    if not sent_any:
        unsent_reasons = sorted(
            {
                str(item.get("reason") or "").strip()
                for item in deliveries
                if not bool(item.get("sent")) and str(item.get("reason") or "").strip()
            }
        )
        if unsent_reasons:
            response["reason"] = unsent_reasons[0] if len(unsent_reasons) == 1 else ",".join(unsent_reasons)
    return response


def tool_notify_oncall_incident(args: dict[str, Any]) -> dict[str, Any]:
    return _notify_oncall(args)


def tool_notify_jsm_ops_incident(args: dict[str, Any]) -> dict[str, Any]:
    return _notify_oncall(args, providers=["jsm_ops"])


def tool_notify_jsm_ticket_issue(args: dict[str, Any]) -> dict[str, Any]:
    return _notify_oncall(args, providers=["jsm_ticket"])


def tool_clear_oncall_state(args: dict[str, Any]) -> dict[str, Any]:
    path = _state_path(args)
    existed = path.exists()
    if existed:
        path.unlink()
    return {"cleared": existed, "state_path": str(path)}


TOOLS = {
    "runtime_incident_snapshot": tool_runtime_incident_snapshot,
    "notify_oncall_incident": tool_notify_oncall_incident,
    "notify_jsm_ops_incident": tool_notify_jsm_ops_incident,
    "notify_jsm_ticket_issue": tool_notify_jsm_ticket_issue,
    "clear_oncall_state": tool_clear_oncall_state,
}

TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "runtime_incident_snapshot",
        "description": "Build runtime incident snapshot + triggers used for escalation.",
    },
    {
        "name": "notify_oncall_incident",
        "description": "Escalate incident to enabled on-call providers (JSM Ops/JSM ticket fallback).",
    },
    {
        "name": "notify_jsm_ops_incident",
        "description": "Create incident alert in Jira Service Management (Ops).",
    },
    {
        "name": "notify_jsm_ticket_issue",
        "description": "Create Jira issue fallback for runtime incidents (JSM project ticket).",
    },
    {"name": "clear_oncall_state", "description": "Clear on-call dedupe state."},
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="On-call escalation MCP server (JSM Ops)",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_oncall_alerts",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
