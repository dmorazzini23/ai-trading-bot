"""Linear issue MCP server for runtime regression tracking."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.mcp_common import ToolSpec

_mcp_common_mod = importlib.import_module(
    "tools.mcp_common" if __package__ == "tools" else "mcp_common"
)
run_tool_server = cast(Callable[..., int], getattr(_mcp_common_mod, "run_tool_server"))
utc_now_iso = cast(Callable[[], str], getattr(_mcp_common_mod, "utc_now_iso"))

_runtime_data_mod = importlib.import_module(
    "tools.mcp_runtime_data_server"
    if __package__ == "tools"
    else "mcp_runtime_data_server"
)
_run_module_json = cast(
    Callable[[str, list[str]], dict[str, Any]],
    getattr(_runtime_data_mod, "_run_module_json"),
)

_DEFAULT_RUNTIME_ROOT = Path("/var/lib/ai-trading-bot/runtime")
_DEFAULT_STATE_PATH = _DEFAULT_RUNTIME_ROOT / "linear_regression_state.json"


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


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _health_payload(port: int, timeout_s: float) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/healthz"
    request = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise RuntimeError("health payload was not a JSON object")
        return payload


def _runtime_report_payload() -> dict[str, Any]:
    return _run_module_json(
        "ai_trading.tools.runtime_performance_report",
        ["--json", "--go-no-go"],
    )


def _regression_state_path(args: dict[str, Any]) -> Path:
    raw = (
        str(args.get("state_path") or "").strip()
        or os.getenv("AI_TRADING_LINEAR_REGRESSION_STATE_PATH", "").strip()
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


def _capture_ratio_threshold(args: dict[str, Any]) -> float:
    raw = args.get("min_capture_ratio")
    if raw is None:
        raw = os.getenv("AI_TRADING_INCIDENT_MIN_CAPTURE_RATIO", "0.08")
    return max(0.0, float(raw))


def _parse_utc(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _regression_family(triggers: list[str]) -> str:
    trigger_set = set(triggers)
    if "go_no_go_failed" in trigger_set or "go_no_go_failed_checks" in trigger_set:
        return "go_no_go"
    if "broker_disconnected" in trigger_set:
        return "broker_connectivity"
    if "execution_capture_ratio_low" in trigger_set:
        return "execution_quality"
    if "provider_degraded" in trigger_set or "health_degraded" in trigger_set:
        return "provider_health"
    return "runtime_regression"


def _session_date(snapshot: dict[str, Any]) -> str:
    parsed = _parse_utc(str(snapshot.get("timestamp") or ""))
    if parsed is None:
        return datetime.now(tz=timezone.utc).date().isoformat()
    return parsed.date().isoformat()


def _rolling_issue_key(snapshot_payload: dict[str, Any]) -> str:
    snapshot = cast(dict[str, Any], snapshot_payload.get("snapshot") or {})
    triggers = cast(list[str], snapshot_payload.get("triggers") or [])
    return f"{_session_date(snapshot)}:{_regression_family(triggers)}"


def _rolling_comment_enabled(args: dict[str, Any]) -> bool:
    return _bool_arg(
        args.get("rolling_comment_updates"),
        default=_bool_arg(os.getenv("AI_TRADING_LINEAR_ROLLING_COMMENT_UPDATES"), default=True),
    )


def _rolling_throttle_seconds(args: dict[str, Any]) -> int:
    raw = args.get("rolling_comment_interval_minutes")
    if raw is None:
        raw = os.getenv("AI_TRADING_LINEAR_ROLLING_COMMENT_INTERVAL_MINUTES", "30")
    try:
        minutes = max(1, int(raw))
    except (TypeError, ValueError):
        minutes = 30
    return minutes * 60


def _rolling_comment_body(snapshot_payload: dict[str, Any], *, event_count: int) -> str:
    snapshot = cast(dict[str, Any], snapshot_payload.get("snapshot") or {})
    triggers = cast(list[str], snapshot_payload.get("triggers") or [])
    failed_checks = cast(list[str], snapshot.get("failed_checks") or [])
    checks_text = ", ".join(failed_checks) if failed_checks else "none"
    return "\n".join(
        [
            "Automated recurring runtime regression update.",
            "",
            f"- observed_at: {utc_now_iso()}",
            f"- event_count_this_session: {event_count}",
            f"- triggers: {', '.join(triggers) if triggers else 'none'}",
            f"- go_no_go_gate_passed: {snapshot.get('gate_passed')}",
            f"- go_no_go_failed_checks: {checks_text}",
            f"- execution_capture_ratio: {snapshot.get('execution_capture_ratio')}",
            f"- slippage_drag_bps: {snapshot.get('slippage_drag_bps')}",
            f"- health_status: {snapshot.get('health_status')}",
            f"- health_reason: {snapshot.get('health_reason')}",
            f"- provider_status: {snapshot.get('provider_status')}",
            f"- broker_status: {snapshot.get('broker_status')}",
        ]
    )


def _create_linear_comment(
    *,
    issue_id: str,
    body: str,
    token: str,
    endpoint: str,
    timeout_s: float,
) -> dict[str, Any]:
    mutation = """
mutation CommentCreate($input: CommentCreateInput!) {
  commentCreate(input: $input) {
    success
    comment {
      id
      body
    }
  }
}
""".strip()
    data = _linear_graphql(
        query=mutation,
        variables={"input": {"issueId": issue_id, "body": body}},
        token=token,
        endpoint=endpoint,
        timeout_s=timeout_s,
    )
    result = cast(dict[str, Any], data.get("commentCreate") or {})
    if not bool(result.get("success", False)):
        raise RuntimeError("Linear commentCreate did not succeed")
    return cast(dict[str, Any], result.get("comment") or {})


def _runtime_regression_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    report = _runtime_report_payload()
    go_no_go = report.get("go_no_go") or {}
    execution = report.get("execution_vs_alpha") or {}

    health_port = int(args.get("health_port") or os.getenv("HEALTHCHECK_PORT", "8081"))
    timeout_s = float(args.get("health_timeout_s") or 2.0)
    health = _health_payload(port=health_port, timeout_s=timeout_s)
    data_provider = health.get("data_provider") or {}
    broker = health.get("broker") or {}

    gate_passed = go_no_go.get("gate_passed")
    failed_checks = list(go_no_go.get("failed_checks") or [])
    capture_ratio = _float_or_none(execution.get("execution_capture_ratio"))
    slippage_drag = _float_or_none(execution.get("slippage_drag_bps"))

    triggers: list[str] = []
    if gate_passed is False:
        triggers.append("go_no_go_failed")
    if failed_checks:
        triggers.append("go_no_go_failed_checks")

    health_ok = bool(health.get("ok", False))
    health_status = str(health.get("status") or "unknown").lower()
    if not health_ok or health_status in {"degraded", "down", "unhealthy"}:
        triggers.append("health_degraded")

    provider_status = str(data_provider.get("status") or "unknown").lower()
    if provider_status in {"degraded", "down"}:
        triggers.append("provider_degraded")

    broker_status = str(broker.get("status") or "unknown").lower()
    if broker_status not in {"connected", "reachable", "unknown"}:
        triggers.append("broker_disconnected")

    min_capture = _capture_ratio_threshold(args)
    if capture_ratio is not None and capture_ratio < min_capture:
        triggers.append("execution_capture_ratio_low")

    snapshot = {
        "gate_passed": gate_passed,
        "failed_checks": failed_checks,
        "execution_capture_ratio": capture_ratio,
        "slippage_drag_bps": slippage_drag,
        "health_ok": health_ok,
        "health_status": str(health.get("status") or "unknown"),
        "health_reason": str(health.get("reason") or "unknown"),
        "provider_status": str(data_provider.get("status") or "unknown"),
        "provider_active": str(data_provider.get("active") or "unknown"),
        "broker_status": str(broker.get("status") or "unknown"),
        "timestamp": str(health.get("timestamp") or utc_now_iso()),
    }
    capture_rounded = round(capture_ratio, 6) if capture_ratio is not None else None
    slippage_rounded = round(slippage_drag, 6) if slippage_drag is not None else None
    fingerprint_material = {
        "triggers": sorted(set(triggers)),
        "gate_passed": gate_passed,
        "failed_checks": failed_checks,
        "execution_capture_ratio": capture_rounded,
        "slippage_drag_bps": slippage_rounded,
        "health_ok": health_ok,
        "health_status": snapshot["health_status"],
        "health_reason": snapshot["health_reason"],
        "provider_status": snapshot["provider_status"],
        "provider_active": snapshot["provider_active"],
        "broker_status": snapshot["broker_status"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_material, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {
        "regression_detected": bool(triggers),
        "triggers": sorted(set(triggers)),
        "fingerprint": fingerprint,
        "snapshot": snapshot,
    }


def _linear_token(args: dict[str, Any]) -> str:
    token = (
        str(args.get("api_key") or "").strip()
        or os.getenv("AI_TRADING_LINEAR_API_KEY", "").strip()
        or os.getenv("LINEAR_API_KEY", "").strip()
    )
    if token:
        return token
    raise RuntimeError("missing Linear API key (api_key arg or AI_TRADING_LINEAR_API_KEY)")


def _linear_team_id(args: dict[str, Any]) -> str:
    team_id = (
        str(args.get("team_id") or "").strip()
        or os.getenv("AI_TRADING_LINEAR_TEAM_ID", "").strip()
        or os.getenv("LINEAR_TEAM_ID", "").strip()
    )
    if team_id:
        return team_id
    raise RuntimeError("missing Linear team ID (team_id arg or AI_TRADING_LINEAR_TEAM_ID)")


def _linear_endpoint(args: dict[str, Any]) -> str:
    return str(args.get("endpoint") or os.getenv("AI_TRADING_LINEAR_ENDPOINT", "https://api.linear.app/graphql"))


def _linear_graphql(*, query: str, variables: dict[str, Any], token: str, endpoint: str, timeout_s: float) -> dict[str, Any]:
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = urllib.request.Request(
        url=endpoint,
        method="POST",
        data=body,
        headers={
            "Authorization": token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("Linear response was not a JSON object")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Linear API HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Linear API request failed: {exc.reason}") from exc

    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0] if isinstance(errors[0], dict) else {"message": str(errors[0])}
        raise RuntimeError(str(first.get("message") or "Linear GraphQL error"))

    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Linear response missing data payload")
    return data


def _default_issue_title(snapshot_payload: dict[str, Any]) -> str:
    triggers = list(snapshot_payload.get("triggers") or [])
    snapshot = snapshot_payload.get("snapshot") or {}
    health_status = snapshot.get("health_status")
    gate_passed = snapshot.get("gate_passed")
    if "go_no_go_failed" in triggers:
        return f"Runtime regression: go/no-go failed ({len(snapshot.get('failed_checks') or [])} checks)"
    if "health_degraded" in triggers:
        return f"Runtime regression: health degraded ({health_status})"
    if gate_passed is True and not triggers:
        return "Runtime event: manual tracking"
    return "Runtime regression detected by MCP monitor"


def _default_issue_description(snapshot_payload: dict[str, Any]) -> str:
    snapshot = cast(dict[str, Any], snapshot_payload.get("snapshot") or {})
    triggers = cast(list[str], snapshot_payload.get("triggers") or [])
    failed_checks = cast(list[str], snapshot.get("failed_checks") or [])
    checks_text = ", ".join(failed_checks) if failed_checks else "none"
    return "\n".join(
        [
            "Automated runtime regression capture from MCP connector.",
            "",
            f"- detected_at: {utc_now_iso()}",
            f"- triggers: {', '.join(triggers) if triggers else 'none'}",
            f"- go_no_go_gate_passed: {snapshot.get('gate_passed')}",
            f"- go_no_go_failed_checks: {checks_text}",
            f"- execution_capture_ratio: {snapshot.get('execution_capture_ratio')}",
            f"- slippage_drag_bps: {snapshot.get('slippage_drag_bps')}",
            f"- health_status: {snapshot.get('health_status')}",
            f"- health_reason: {snapshot.get('health_reason')}",
            f"- provider_status: {snapshot.get('provider_status')}",
            f"- broker_status: {snapshot.get('broker_status')}",
        ]
    )


def tool_runtime_regression_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    return _runtime_regression_snapshot(args)


def tool_create_regression_issue(args: dict[str, Any]) -> dict[str, Any]:
    snapshot_payload = _runtime_regression_snapshot(args)
    regression = bool(snapshot_payload.get("regression_detected", False))
    force = _bool_arg(args.get("force"), default=False)
    if not regression and not force:
        return {
            "created": False,
            "reason": "no_runtime_regression_detected",
            "snapshot": snapshot_payload,
        }

    state_path = _regression_state_path(args)
    dedupe = _bool_arg(args.get("dedupe"), default=True)
    fingerprint = str(snapshot_payload.get("fingerprint") or "")
    prior = _load_state(state_path)
    rollups_raw = prior.get("rollups")
    rollups = cast(dict[str, dict[str, Any]], rollups_raw if isinstance(rollups_raw, dict) else {})
    if dedupe and prior.get("fingerprint") == fingerprint:
        return {
            "created": False,
            "reason": "duplicate_fingerprint",
            "state_path": str(state_path),
            "existing_issue": prior.get("issue"),
            "snapshot": snapshot_payload,
        }

    rolling_enabled = _bool_arg(
        args.get("rolling_session_enabled"),
        default=_bool_arg(os.getenv("AI_TRADING_LINEAR_ROLLING_SESSION_ENABLED"), default=True),
    )
    rolling_key = _rolling_issue_key(snapshot_payload)
    rolling_entry = cast(dict[str, Any], rollups.get(rolling_key) or {})
    rolling_issue = cast(dict[str, Any], rolling_entry.get("issue") or {})
    rolling_issue_id = str(rolling_issue.get("id") or "").strip()
    if rolling_enabled and rolling_issue_id and not force:
        event_count = int(rolling_entry.get("event_count") or 0) + 1
        comment_created = False
        comment: dict[str, Any] = {}
        now = datetime.now(tz=timezone.utc)
        maybe_last = _parse_utc(str(rolling_entry.get("last_commented_at") or ""))
        should_comment = _rolling_comment_enabled(args)
        if should_comment:
            throttle_seconds = _rolling_throttle_seconds(args)
            elapsed_ok = maybe_last is None or (now - maybe_last).total_seconds() >= throttle_seconds
            if elapsed_ok:
                token = _linear_token(args)
                endpoint = _linear_endpoint(args)
                timeout_s = float(args.get("timeout_s") or 8.0)
                comment = _create_linear_comment(
                    issue_id=rolling_issue_id,
                    body=_rolling_comment_body(snapshot_payload, event_count=event_count),
                    token=token,
                    endpoint=endpoint,
                    timeout_s=timeout_s,
                )
                comment_created = True

        updated_rollups: dict[str, dict[str, Any]] = dict(rollups)
        next_entry: dict[str, Any] = dict(rolling_entry)
        next_entry["event_count"] = event_count
        next_entry["last_seen_at"] = utc_now_iso()
        next_entry["last_fingerprint"] = fingerprint
        next_entry["snapshot"] = snapshot_payload
        if comment_created:
            next_entry["last_commented_at"] = utc_now_iso()
        updated_rollups[rolling_key] = next_entry
        _save_state(
            state_path,
            {
                "fingerprint": fingerprint,
                "saved_at": utc_now_iso(),
                "issue": rolling_issue,
                "snapshot": snapshot_payload,
                "rolling_key": rolling_key,
                "rollups": updated_rollups,
            },
        )
        return {
            "created": False,
            "reason": "rolling_session_existing_issue",
            "state_path": str(state_path),
            "issue": rolling_issue,
            "rolling_key": rolling_key,
            "event_count": event_count,
            "comment_created": comment_created,
            "comment": comment if comment_created else None,
            "snapshot": snapshot_payload,
        }

    issue_title = str(args.get("title") or "").strip() or _default_issue_title(snapshot_payload)
    issue_description = str(args.get("description") or "").strip() or _default_issue_description(
        snapshot_payload
    )
    team_id = _linear_team_id(args)
    priority_value = args.get("priority")
    priority = int(priority_value) if priority_value is not None else None
    label_ids_raw = args.get("label_ids") or os.getenv("AI_TRADING_LINEAR_LABEL_IDS", "")
    label_ids: list[str] = []
    if isinstance(label_ids_raw, str):
        label_ids = [chunk.strip() for chunk in label_ids_raw.split(",") if chunk.strip()]
    elif isinstance(label_ids_raw, list):
        label_ids = [str(chunk).strip() for chunk in label_ids_raw if str(chunk).strip()]

    issue_input: dict[str, Any] = {
        "teamId": team_id,
        "title": issue_title,
        "description": issue_description,
    }
    if priority is not None:
        issue_input["priority"] = priority
    if label_ids:
        issue_input["labelIds"] = label_ids

    dry_run = _bool_arg(args.get("dry_run"), default=False)
    if dry_run:
        return {
            "created": False,
            "dry_run": True,
            "issue_input": issue_input,
            "snapshot": snapshot_payload,
        }

    token = _linear_token(args)
    endpoint = _linear_endpoint(args)
    timeout_s = float(args.get("timeout_s") or 8.0)
    mutation = """
mutation IssueCreate($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue {
      id
      identifier
      title
      url
    }
  }
}
""".strip()
    data = _linear_graphql(
        query=mutation,
        variables={"input": issue_input},
        token=token,
        endpoint=endpoint,
        timeout_s=timeout_s,
    )
    result = cast(dict[str, Any], data.get("issueCreate") or {})
    issue = cast(dict[str, Any], result.get("issue") or {})
    if not bool(result.get("success", False)) or not issue:
        raise RuntimeError("Linear issueCreate did not return a created issue")

    new_rollups: dict[str, dict[str, Any]] = dict(rollups)
    new_rollups[rolling_key] = {
        "issue": issue,
        "first_seen_at": utc_now_iso(),
        "last_seen_at": utc_now_iso(),
        "last_commented_at": None,
        "event_count": 1,
        "last_fingerprint": fingerprint,
        "snapshot": snapshot_payload,
    }
    state_payload = {
        "fingerprint": fingerprint,
        "saved_at": utc_now_iso(),
        "issue": issue,
        "snapshot": snapshot_payload,
        "rolling_key": rolling_key,
        "rollups": new_rollups,
    }
    _save_state(state_path, state_payload)
    return {
        "created": True,
        "issue": issue,
        "state_path": str(state_path),
        "rolling_key": rolling_key,
        "snapshot": snapshot_payload,
    }


def tool_clear_regression_state(args: dict[str, Any]) -> dict[str, Any]:
    path = _regression_state_path(args)
    existed = path.exists()
    if existed:
        path.unlink()
    return {"cleared": existed, "state_path": str(path)}


TOOLS = {
    "runtime_regression_snapshot": tool_runtime_regression_snapshot,
    "create_regression_issue": tool_create_regression_issue,
    "clear_regression_state": tool_clear_regression_state,
}

TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "runtime_regression_snapshot",
        "description": "Detect runtime regressions from go/no-go and health state.",
    },
    {
        "name": "create_regression_issue",
        "description": "Create Linear issue from runtime regression snapshot (dedupe-aware).",
    },
    {
        "name": "clear_regression_state",
        "description": "Clear saved Linear regression dedupe state.",
    },
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="Linear regression issue MCP server",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_linear_issues",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
