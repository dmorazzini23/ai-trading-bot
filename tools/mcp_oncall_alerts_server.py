"""On-call MCP server for PagerDuty/Opsgenie incident escalation."""

from __future__ import annotations

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

_PAGERDUTY_EVENTS_V2 = "https://events.pagerduty.com/v2/enqueue"
_OPSGENIE_DEFAULT_URL = "https://api.opsgenie.com/v2/alerts"


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
    timeout_s: float = 6.0,
) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        method="POST",
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
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


def _opsgenie_priority(severity: str) -> str:
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
    raw = args.get("providers")
    if isinstance(raw, str):
        requested = [chunk.strip().lower() for chunk in raw.split(",") if chunk.strip()]
    elif isinstance(raw, list):
        requested = [str(chunk).strip().lower() for chunk in raw if str(chunk).strip()]
    else:
        requested = []

    known = {"pagerduty", "opsgenie"}
    requested = [provider for provider in requested if provider in known]
    if requested:
        return sorted(set(requested))

    providers: list[str] = []
    if _bool_arg(os.getenv("AI_TRADING_CONNECTOR_PAGERDUTY_ENABLED"), default=False):
        providers.append("pagerduty")
    if _bool_arg(os.getenv("AI_TRADING_CONNECTOR_OPSGENIE_ENABLED"), default=False):
        providers.append("opsgenie")
    return sorted(set(providers))


def _notify_pagerduty(
    *,
    args: dict[str, Any],
    fingerprint: str,
    severity: str,
    snapshot: dict[str, Any],
    triggers: list[str],
) -> dict[str, Any]:
    routing_key = (
        str(args.get("pagerduty_routing_key") or "").strip()
        or os.getenv("AI_TRADING_PAGERDUTY_ROUTING_KEY", "").strip()
    )
    if not routing_key:
        return {"provider": "pagerduty", "sent": False, "reason": "missing_routing_key"}

    source = str(args.get("source") or os.getenv("AI_TRADING_ONCALL_SOURCE", "ai-trading")).strip()
    component = str(args.get("component") or "runtime").strip()
    group = str(args.get("group") or "trading-bot").strip()

    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "dedup_key": fingerprint,
        "payload": {
            "summary": _incident_summary(snapshot, triggers),
            "source": source,
            "severity": severity,
            "component": component,
            "group": group,
            "class": "runtime_incident",
            "timestamp": str(snapshot.get("timestamp") or utc_now_iso()),
            "custom_details": {
                "snapshot": snapshot,
                "triggers": triggers,
                "fingerprint": fingerprint,
            },
        },
    }
    timeout_s = float(args.get("timeout_s") or 6.0)
    status_code, body = _post_json(url=_PAGERDUTY_EVENTS_V2, payload=payload, timeout_s=timeout_s)
    return {"provider": "pagerduty", "sent": True, "status_code": status_code, "response": body}


def _notify_opsgenie(
    *,
    args: dict[str, Any],
    fingerprint: str,
    severity: str,
    snapshot: dict[str, Any],
    triggers: list[str],
) -> dict[str, Any]:
    api_key = (
        str(args.get("opsgenie_api_key") or "").strip()
        or os.getenv("AI_TRADING_OPSGENIE_API_KEY", "").strip()
    )
    if not api_key:
        return {"provider": "opsgenie", "sent": False, "reason": "missing_api_key"}

    endpoint = str(args.get("opsgenie_url") or os.getenv("AI_TRADING_OPSGENIE_ALERT_URL", "")).strip()
    if not endpoint:
        endpoint = _OPSGENIE_DEFAULT_URL

    alias_prefix = str(args.get("alias_prefix") or "ai-trading").strip()
    alias = f"{alias_prefix}:{fingerprint}"
    source = str(args.get("source") or os.getenv("AI_TRADING_ONCALL_SOURCE", "ai-trading")).strip()
    tags = ["ai-trading", "runtime"] + sorted(set(triggers))
    responders = []
    team = str(args.get("opsgenie_team") or os.getenv("AI_TRADING_OPSGENIE_TEAM", "")).strip()
    if team:
        responders = [{"name": team, "type": "team"}]

    payload = {
        "message": _incident_summary(snapshot, triggers)[:130],
        "alias": alias,
        "description": _incident_summary(snapshot, triggers),
        "priority": _opsgenie_priority(severity),
        "source": source,
        "tags": tags,
        "responders": responders,
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

    timeout_s = float(args.get("timeout_s") or 6.0)
    headers = {"Authorization": f"GenieKey {api_key}"}
    status_code, body = _post_json(
        url=endpoint,
        payload=payload,
        headers=headers,
        timeout_s=timeout_s,
    )
    return {"provider": "opsgenie", "sent": True, "status_code": status_code, "response": body}


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
        if provider == "pagerduty":
            deliveries.append(
                _notify_pagerduty(
                    args=args,
                    fingerprint=fingerprint,
                    severity=severity,
                    snapshot=snapshot,
                    triggers=triggers,
                )
            )
        elif provider == "opsgenie":
            deliveries.append(
                _notify_opsgenie(
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
    return {
        "sent": sent_any,
        "providers": selected,
        "severity": severity,
        "fingerprint": fingerprint,
        "triggers": triggers,
        "deliveries": deliveries,
        "state_path": str(state_path),
        "snapshot": snapshot,
    }


def tool_notify_oncall_incident(args: dict[str, Any]) -> dict[str, Any]:
    return _notify_oncall(args)


def tool_notify_pagerduty_incident(args: dict[str, Any]) -> dict[str, Any]:
    return _notify_oncall(args, providers=["pagerduty"])


def tool_notify_opsgenie_alert(args: dict[str, Any]) -> dict[str, Any]:
    return _notify_oncall(args, providers=["opsgenie"])


def tool_clear_oncall_state(args: dict[str, Any]) -> dict[str, Any]:
    path = _state_path(args)
    existed = path.exists()
    if existed:
        path.unlink()
    return {"cleared": existed, "state_path": str(path)}


TOOLS = {
    "runtime_incident_snapshot": tool_runtime_incident_snapshot,
    "notify_oncall_incident": tool_notify_oncall_incident,
    "notify_pagerduty_incident": tool_notify_pagerduty_incident,
    "notify_opsgenie_alert": tool_notify_opsgenie_alert,
    "clear_oncall_state": tool_clear_oncall_state,
}

TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "runtime_incident_snapshot",
        "description": "Build runtime incident snapshot + triggers used for escalation.",
    },
    {
        "name": "notify_oncall_incident",
        "description": "Escalate incident to enabled on-call providers (PagerDuty/Opsgenie).",
    },
    {
        "name": "notify_pagerduty_incident",
        "description": "Send incident event to PagerDuty Events API v2.",
    },
    {
        "name": "notify_opsgenie_alert",
        "description": "Create incident alert in Opsgenie Alerts API.",
    },
    {"name": "clear_oncall_state", "description": "Clear on-call dedupe state."},
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="On-call escalation MCP server",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_oncall_alerts",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
