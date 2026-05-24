"""Slack MCP server for runtime incident and end-of-day summaries."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
import hashlib
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
_DEFAULT_INCIDENT_STATE = _DEFAULT_RUNTIME_ROOT / "slack_incident_state.json"
_DEFAULT_EOD_STATE = _DEFAULT_RUNTIME_ROOT / "slack_eod_state.json"
_DEFAULT_INCIDENT_REPEAT_COOLDOWN_MINUTES = 45
_DEFAULT_HEALTH_PORT = 9001
_STARTUP_WARMUP_HEALTH_REASONS = frozenset(
    {
        "startup",
        "warmup_cycle",
        "startup_complete_pending_runtime_health",
        "startup_pending_reconcile",
        "startup_pending_reconcile_complete",
    }
)
_NON_BLOCKING_REJECTION_GATES = frozenset(
    {
        "OK_TRADE",
        "EXPECTED_CAPTURE_MODEL_LEARNED",
        "EXPECTED_CAPTURE_OPTIMIZER",
        "EXECUTION_LEARNING_DEWEIGHT",
        "REJECTION_CONCENTRATION_DEWEIGHT",
        "PORTFOLIO_LOG_GROWTH",
        "COUNTERFACTUAL_DR",
        "COUNTERFACTUAL_DR_SHADOW",
        "EDGE_MODEL_V2",
        "EDGE_MODEL_V2_REGIME_BLEND",
        "REPLAY_QUALITY_UPLIFT",
        "REPLAY_QUALITY_DEWEIGHT",
        "RANK_DOWNSIDE_OVERLAP_CAP",
    }
)
_NON_BLOCKING_REJECTION_PREFIXES = (
    "BANDIT_",
)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _int_arg(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _is_non_blocking_rejection_gate(gate_name: str) -> bool:
    token = str(gate_name or "").strip().upper()
    if not token:
        return False
    if token in _NON_BLOCKING_REJECTION_GATES:
        return True
    return any(token.startswith(prefix) for prefix in _NON_BLOCKING_REJECTION_PREFIXES)


def _runtime_report_payload() -> dict[str, Any]:
    report_path_raw = str(
        os.getenv("AI_TRADING_RUNTIME_PERF_REPORT_LATEST_PATH")
        or (_DEFAULT_RUNTIME_ROOT / "runtime_performance_report_latest.json")
    ).strip()
    report_path = Path(report_path_raw).expanduser()
    if not report_path.is_absolute():
        report_path = (_DEFAULT_RUNTIME_ROOT / report_path).resolve()
    if report_path.exists():
        cached = _read_json_object(report_path)
        if cached:
            return cached
    return _run_module_json(
        "ai_trading.tools.runtime_performance_report",
        ["--json", "--go-no-go"],
    )


def _normalized_runtime_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical report body while preserving top-level runtime gates."""

    nested = payload.get("report")
    if not isinstance(nested, dict):
        return payload
    normalized = dict(nested)
    for key in (
        "generated_at",
        "go_no_go",
        "edge_realism_gap_ratio",
        "edge_realism_report",
        "expected_edge_clip",
        "expected_edge_clip_bps",
        "expected_edge_for_realism_bps",
        "expected_edge_per_filled_trade_bps",
        "realization_gap_bps",
        "realized_net_edge_bps",
        "source",
    ):
        if key in payload and key not in normalized:
            normalized[key] = payload[key]
    return normalized


def _health_unavailable_payload(*, port: int, url: str, error: str) -> dict[str, Any]:
    reason = "health_payload_unavailable"
    return {
        "ok": False,
        "status": "degraded",
        "reason": reason,
        "timestamp": utc_now_iso(),
        "health_port": int(port),
        "health_url": url,
        "health_error": str(error)[:500],
        "data_provider": {
            "status": "unknown",
            "active": "unknown",
            "reason": reason,
            "using_backup": False,
        },
        "broker": {"status": "unknown"},
        "model_liveness": {},
    }


def _health_payload(port: int, timeout_s: float) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/healthz"
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise RuntimeError("health payload was not a JSON object")
            return payload
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
        return _health_unavailable_payload(
            port=port,
            url=url,
            error=f"HTTP {exc.code}: {body[:300]}",
        )
    except Exception as exc:
        return _health_unavailable_payload(port=port, url=url, error=str(exc))


def _safe_health_payload(port: int, timeout_s: float) -> dict[str, Any]:
    try:
        return _health_payload(port=port, timeout_s=timeout_s)
    except Exception as exc:
        url = f"http://127.0.0.1:{port}/healthz"
        return _health_unavailable_payload(port=port, url=url, error=str(exc))


def _incident_state_path(args: dict[str, Any]) -> Path:
    raw = (
        str(args.get("state_path") or "").strip()
        or os.getenv("AI_TRADING_SLACK_INCIDENT_STATE_PATH", "").strip()
    )
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_INCIDENT_STATE


def _eod_state_path(args: dict[str, Any]) -> Path:
    raw = (
        str(args.get("state_path") or "").strip()
        or os.getenv("AI_TRADING_SLACK_EOD_STATE_PATH", "").strip()
    )
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_EOD_STATE


def _runtime_root(args: dict[str, Any]) -> Path:
    raw = str(args.get("runtime_root") or os.getenv("AI_TRADING_RUNTIME_ROOT", "")).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_RUNTIME_ROOT


def _after_hours_training_marker_path(args: dict[str, Any]) -> Path:
    raw = str(
        args.get("after_hours_training_marker_path")
        or os.getenv("AI_TRADING_AFTER_HOURS_TRAINING_MARKER_PATH", "")
    ).strip()
    if not raw:
        return (_runtime_root(args) / "after_hours_training.marker.json").resolve()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    if raw.startswith("runtime/"):
        relative = raw.removeprefix("runtime/").strip("/")
        return (_runtime_root(args) / relative).resolve()
    return path.resolve()


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _parse_iso_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _tail_jsonl_rows(path: Path, *, max_rows: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines: deque[str] = deque(maxlen=max(1, int(max_rows)))
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    lines.append(text)
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for text in lines:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _runtime_event_path(args: dict[str, Any], *, env_key: str, default_relative: str) -> Path:
    raw = str(os.getenv(env_key, "")).strip() or default_relative
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    if raw.startswith("runtime/"):
        relative = raw.removeprefix("runtime/").strip("/")
    else:
        relative = raw.strip("/")
    return (_runtime_root(args) / relative).resolve()


def _collect_execution_window_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    window_minutes = _int_arg(
        args.get("execution_window_minutes")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_EXEC_WINDOW_MINUTES"),
        default=30,
    )
    window_minutes = max(5, min(window_minutes, 24 * 60))
    max_rows = _int_arg(
        args.get("execution_window_max_rows")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_EXEC_WINDOW_MAX_ROWS"),
        default=5000,
    )
    max_rows = max(250, min(max_rows, 100_000))
    cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)

    order_events_path = _runtime_event_path(
        args,
        env_key="AI_TRADING_ORDER_EVENTS_PATH",
        default_relative="runtime/order_events.jsonl",
    )
    quality_events_path = _runtime_event_path(
        args,
        env_key="AI_TRADING_EXEC_QUALITY_EVENTS_PATH",
        default_relative="runtime/execution_quality_events.jsonl",
    )
    order_rows = _tail_jsonl_rows(order_events_path, max_rows=max_rows)
    quality_rows = _tail_jsonl_rows(quality_events_path, max_rows=max_rows)

    final_by_order: dict[str, tuple[datetime | None, str]] = {}
    for row in order_rows:
        if str(row.get("event") or "").strip().lower() != "final_state":
            continue
        ts = _parse_iso_ts(row.get("ts") or row.get("timestamp"))
        if ts is not None and ts < cutoff:
            continue
        order_id = str(row.get("order_id") or row.get("client_order_id") or "").strip()
        if not order_id:
            continue
        status = str(row.get("status") or row.get("new_status") or "").strip().lower()
        previous = final_by_order.get(order_id)
        if previous is not None:
            previous_ts, _ = previous
            if previous_ts is not None and ts is not None and ts <= previous_ts:
                continue
        final_by_order[order_id] = (ts, status)

    fill_statuses = {"filled", "partially_filled"}
    fill_ratio_samples = len(final_by_order)
    fill_ratio_filled = sum(
        1 for _order_id, (_ts, status) in final_by_order.items() if status in fill_statuses
    )
    fill_ratio = (
        float(fill_ratio_filled) / float(fill_ratio_samples)
        if fill_ratio_samples > 0
        else None
    )

    skipped_count = 0
    precheck_failure_count = 0
    precheck_failure_detail_counts: dict[str, int] = {}
    for row in quality_rows:
        ts = _parse_iso_ts(row.get("ts") or row.get("timestamp"))
        if ts is not None and ts < cutoff:
            continue
        status = str(row.get("status") or "").strip().lower()
        if status != "skipped":
            continue
        skipped_count += 1
        reason = str(row.get("reason") or "").strip().lower()
        if reason == "pre_execution_order_checks_failed":
            precheck_failure_count += 1
            detail = str(row.get("detail") or "").strip().lower()
            if not detail:
                context_raw = row.get("context")
                if isinstance(context_raw, dict):
                    detail = str(
                        context_raw.get("reason")
                        or context_raw.get("detail")
                        or ""
                    ).strip().lower()
            if not detail:
                detail = "unspecified"
            precheck_failure_detail_counts[detail] = (
                precheck_failure_detail_counts.get(detail, 0) + 1
            )
    precheck_failure_ratio = (
        float(precheck_failure_count) / float(skipped_count) if skipped_count > 0 else None
    )
    top_precheck_failure_details = [
        {"detail": detail, "count": int(count)}
        for detail, count in sorted(
            precheck_failure_detail_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
    ]
    top_precheck_failure_actionable_details = [
        item for item in top_precheck_failure_details if item.get("detail") != "unspecified"
    ][:5]

    return {
        "execution_window_minutes": int(window_minutes),
        "execution_fill_ratio": fill_ratio,
        "execution_fill_ratio_samples": int(fill_ratio_samples),
        "execution_fill_ratio_filled": int(fill_ratio_filled),
        "execution_skipped_count": int(skipped_count),
        "precheck_failure_count": int(precheck_failure_count),
        "precheck_failure_ratio": precheck_failure_ratio,
        "precheck_failure_top_details": top_precheck_failure_details,
        "precheck_failure_top_actionable_details": top_precheck_failure_actionable_details,
        "order_events_path": str(order_events_path),
        "exec_quality_events_path": str(quality_events_path),
    }


def _collect_gate_window_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    window_minutes = _int_arg(
        args.get("gate_window_minutes")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_GATE_WINDOW_MINUTES"),
        default=60,
    )
    window_minutes = max(5, min(window_minutes, 24 * 60))
    max_rows = _int_arg(
        args.get("gate_window_max_rows")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_GATE_WINDOW_MAX_ROWS"),
        default=500,
    )
    max_rows = max(100, min(max_rows, 10_000))
    cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)

    gate_events_path = _runtime_event_path(
        args,
        env_key="AI_TRADING_GATE_EFFECTIVENESS_LOG_PATH",
        default_relative="runtime/gate_effectiveness.jsonl",
    )
    rows = _tail_jsonl_rows(gate_events_path, max_rows=max_rows)
    gate_stats: dict[str, dict[str, float]] = {}
    rejected_records_total = 0
    sampled_rows = 0
    for row in rows:
        ts = _parse_iso_ts(row.get("ts") or row.get("timestamp"))
        if ts is not None and ts < cutoff:
            continue
        sampled_rows += 1
        rejected_records_total += max(0, _int_arg(row.get("rejected_records"), default=0))
        gate_attribution = row.get("gate_attribution")
        if not isinstance(gate_attribution, dict):
            continue
        for gate_name, payload in gate_attribution.items():
            if not isinstance(payload, dict):
                continue
            blocked = _float_or_none(payload.get("blocked_records"))
            if blocked is None or blocked <= 0.0:
                continue
            accepted = _float_or_none(payload.get("accepted_records"))
            key = str(gate_name or "").strip()
            if not key:
                continue
            gate_entry = gate_stats.setdefault(
                key,
                {
                    "blocked_records": 0.0,
                    "accepted_records": 0.0,
                },
            )
            gate_entry["blocked_records"] = float(gate_entry["blocked_records"]) + float(blocked)
            gate_entry["accepted_records"] = float(gate_entry["accepted_records"]) + float(
                accepted if accepted is not None else 0.0
            )

    top_gate = ""
    top_blocked = 0.0
    blocking_only_stats = {
        gate_name: stats
        for gate_name, stats in gate_stats.items()
        if float(stats.get("blocked_records", 0.0) or 0.0) > 0.0
        and float(stats.get("accepted_records", 0.0) or 0.0) <= 0.0
        and not _is_non_blocking_rejection_gate(gate_name)
    }
    if blocking_only_stats:
        top_gate, top_blocked = max(
            (
                (gate_name, float(stats.get("blocked_records", 0.0) or 0.0))
                for gate_name, stats in blocking_only_stats.items()
            ),
            key=lambda item: float(item[1]),
        )
    top_ratio = (
        (float(top_blocked) / float(rejected_records_total))
        if rejected_records_total > 0 and top_blocked > 0.0
        else None
    )
    return {
        "gate_window_minutes": int(window_minutes),
        "gate_window_rows": int(sampled_rows),
        "gate_window_events_path": str(gate_events_path),
        "gate_rejected_records": int(rejected_records_total),
        "top_rejection_concentration_gate": top_gate or "",
        "top_rejection_concentration_ratio": top_ratio,
        "top_rejection_concentration_blocking_gate_found": bool(top_gate),
    }


def _collect_runtime_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    report = _normalized_runtime_report_payload(_runtime_report_payload())
    go_no_go = report.get("go_no_go") or {}
    execution = report.get("execution_vs_alpha") or {}
    gate_effectiveness = report.get("gate_effectiveness") or {}

    port = int(
        args.get("health_port")
        or os.getenv("HEALTHCHECK_PORT", str(_DEFAULT_HEALTH_PORT))
    )
    timeout_s = float(args.get("health_timeout_s") or 2.0)
    health = _safe_health_payload(port=port, timeout_s=timeout_s)
    data_provider = health.get("data_provider") or {}
    broker = health.get("broker") or {}
    service_state = health.get("service_state") or {}
    execution_window = _collect_execution_window_snapshot(args)
    gate_window = _collect_gate_window_snapshot(args)

    gate_rejected_records = _int_arg(gate_window.get("gate_rejected_records"), default=0)
    if gate_rejected_records <= 0:
        gate_rejected_records = _int_arg(
            gate_effectiveness.get("rejected_records"), default=0
        )
    gate_window_rows = _int_arg(gate_window.get("gate_window_rows"), default=0)
    gate_window_blocking_found = _bool_arg(
        gate_window.get("top_rejection_concentration_blocking_gate_found"),
        default=False,
    )
    if gate_window_rows > 0:
        if gate_window_blocking_found:
            top_rejection_gate = str(
                gate_window.get("top_rejection_concentration_gate") or ""
            ).strip()
            top_rejection_ratio = _float_or_none(
                gate_window.get("top_rejection_concentration_ratio")
            )
            if _is_non_blocking_rejection_gate(top_rejection_gate):
                top_rejection_gate = ""
                top_rejection_ratio = None
        else:
            top_rejection_gate = ""
            top_rejection_ratio = None
    else:
        top_rejection_gate = str(
            gate_effectiveness.get("top_rejection_concentration_gate") or ""
        ).strip()
        top_rejection_ratio = _float_or_none(
            gate_effectiveness.get("top_rejection_concentration_ratio")
        )
        if _is_non_blocking_rejection_gate(top_rejection_gate):
            top_rejection_gate = ""
            top_rejection_ratio = None

    return {
        "runtime_gonogo_block_openings_enabled": _runtime_gonogo_block_openings_enabled(args),
        "go_no_go_gate_passed": go_no_go.get("gate_passed"),
        "go_no_go_failed_checks": list(go_no_go.get("failed_checks") or []),
        "execution_capture_ratio": _float_or_none(execution.get("execution_capture_ratio")),
        "slippage_drag_bps": _float_or_none(execution.get("slippage_drag_bps")),
        "expected_edge_per_accept_bps": _float_or_none(
            execution.get("expected_edge_per_accept_bps")
        ),
        "realization_gap_bps": _float_or_none(execution.get("realization_gap_bps")),
        "edge_realism_gap_ratio": _float_or_none(
            execution.get("edge_realism_gap_ratio")
        ),
        "gate_rejected_records": int(gate_rejected_records),
        "top_rejection_concentration_gate": top_rejection_gate,
        "top_rejection_concentration_ratio": top_rejection_ratio,
        "gate_window_minutes": _int_arg(gate_window.get("gate_window_minutes"), default=60),
        "gate_window_rows": int(gate_window_rows),
        "gate_window_events_path": str(
            gate_window.get("gate_window_events_path") or ""
        ),
        "execution_fill_ratio": _float_or_none(execution_window.get("execution_fill_ratio")),
        "execution_fill_ratio_samples": _int_arg(
            execution_window.get("execution_fill_ratio_samples"), default=0
        ),
        "execution_fill_ratio_filled": _int_arg(
            execution_window.get("execution_fill_ratio_filled"), default=0
        ),
        "execution_window_minutes": _int_arg(
            execution_window.get("execution_window_minutes"), default=30
        ),
        "execution_skipped_count": _int_arg(
            execution_window.get("execution_skipped_count"), default=0
        ),
        "precheck_failure_count": _int_arg(
            execution_window.get("precheck_failure_count"), default=0
        ),
        "precheck_failure_ratio": _float_or_none(
            execution_window.get("precheck_failure_ratio")
        ),
        "precheck_failure_top_details": list(
            execution_window.get("precheck_failure_top_details") or []
        ),
        "precheck_failure_top_actionable_details": list(
            execution_window.get("precheck_failure_top_actionable_details") or []
        ),
        "health_ok": bool(health.get("ok", False)),
        "health_status": str(health.get("status") or "unknown"),
        "health_reason": str(health.get("reason") or "unknown"),
        "provider_status": str(data_provider.get("status") or "unknown"),
        "provider_active": str(data_provider.get("active") or "unknown"),
        "provider_reason": str(data_provider.get("reason") or "unknown"),
        "using_backup": bool(data_provider.get("using_backup", False)),
        "broker_status": str(broker.get("status") or "unknown"),
        "service_status": str(service_state.get("status") or "unknown"),
        "service_phase": str(service_state.get("phase") or "unknown"),
        "service_reason": str(service_state.get("reason") or "unknown"),
        "service_phase_since": str(service_state.get("phase_since") or ""),
        "service_updated": str(service_state.get("updated") or ""),
        "timestamp": str(health.get("timestamp") or utc_now_iso()),
    }


def _extract_report_date(report: dict[str, Any]) -> str:
    candidates: list[str] = []

    def _append_date(value: Any) -> None:
        day = str(value or "").strip()
        if len(day) == 10:
            try:
                datetime.fromisoformat(day)
            except ValueError:
                return
            candidates.append(day)

    go_no_go = report.get("go_no_go")
    if isinstance(go_no_go, dict):
        observed = go_no_go.get("observed")
        if isinstance(observed, dict):
            for scope_key in ("trade_metric_scope", "gate_metric_scope"):
                scope = observed.get(scope_key)
                if isinstance(scope, dict):
                    _append_date(scope.get("end_date"))

    execution = report.get("execution_vs_alpha")
    if isinstance(execution, dict):
        daily = execution.get("daily")
        if isinstance(daily, list):
            for row in daily:
                if not isinstance(row, dict):
                    continue
                _append_date(row.get("date"))

    trade_history = report.get("trade_history")
    if isinstance(trade_history, dict):
        for key in ("daily_trade_stats", "daily_expectancy_live"):
            rows = trade_history.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                _append_date(row.get("date"))

    return max(candidates) if candidates else ""


def _daily_trade_row_for_report_date(
    trade_history: dict[str, Any], report_date: str
) -> dict[str, Any]:
    rows = trade_history.get("daily_trade_stats")
    if not isinstance(rows, list):
        return {}
    if report_date:
        for row in reversed(rows):
            if not isinstance(row, dict):
                continue
            if str(row.get("date") or "").strip() == report_date:
                return row
        return {}
    for row in reversed(rows):
        if isinstance(row, dict):
            return row
    return {}


def _same_day_fill_round_trip_summary(args: dict[str, Any], report_date: str) -> dict[str, Any]:
    if not report_date:
        return {"status": "missing_report_date"}
    try:
        report_day = datetime.fromisoformat(report_date).date()
    except ValueError:
        return {"status": "invalid_report_date", "report_date": report_date}
    fill_events_path = _runtime_event_path(
        args,
        env_key="AI_TRADING_FILL_EVENTS_PATH",
        default_relative="runtime/fill_events.jsonl",
    )
    max_rows = _int_arg(args.get("fill_event_max_rows"), default=20000)
    rows = _tail_jsonl_rows(fill_events_path, max_rows=max_rows)
    if not rows:
        return {
            "status": "no_fill_events",
            "path": str(fill_events_path),
            "report_date": report_date,
        }

    events: list[tuple[datetime, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        ts = _parse_iso_ts(row.get("ts") or row.get("entry_time") or row.get("timestamp"))
        if ts is None or ts.date() != report_day:
            continue
        side = str(row.get("side") or "").strip().lower()
        if side not in {"buy", "sell", "sell_short"}:
            continue
        qty = _float_or_none(row.get("fill_qty") or row.get("qty") or row.get("quantity"))
        price = _float_or_none(row.get("fill_price") or row.get("entry_price") or row.get("price"))
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or qty is None or price is None or qty <= 0 or price <= 0:
            continue
        events.append(
            (
                ts,
                index,
                {
                    "symbol": symbol,
                    "side": "sell" if side == "sell_short" else side,
                    "qty": float(qty),
                    "price": float(price),
                },
            )
        )
    events.sort(key=lambda item: (item[0], item[1]))
    if not events:
        return {
            "status": "no_same_day_fills",
            "path": str(fill_events_path),
            "report_date": report_date,
            "fill_rows_seen": int(len(rows)),
        }

    books: dict[str, list[dict[str, float | str]]] = {}
    pnl_by_symbol: dict[str, float] = {}
    closed_trades = 0
    for _ts, _index, event in events:
        symbol = str(event["symbol"])
        side = str(event["side"])
        remaining = float(event["qty"])
        price = float(event["price"])
        book = books.setdefault(symbol, [])
        while remaining > 0 and book and str(book[0].get("side")) != side:
            lot = book[0]
            lot_qty = float(lot.get("qty") or 0.0)
            lot_price = float(lot.get("price") or 0.0)
            close_qty = min(remaining, lot_qty)
            if str(lot.get("side")) == "buy" and side == "sell":
                pnl = (price - lot_price) * close_qty
            else:
                pnl = (lot_price - price) * close_qty
            pnl_by_symbol[symbol] = pnl_by_symbol.get(symbol, 0.0) + pnl
            closed_trades += 1
            remaining -= close_qty
            lot["qty"] = lot_qty - close_qty
            if float(lot["qty"]) <= 1e-9:
                book.pop(0)
        if remaining > 1e-9:
            book.append({"side": side, "qty": remaining, "price": price})

    open_qty_by_symbol: dict[str, float] = {}
    for symbol, lots in books.items():
        net_qty = 0.0
        for lot in lots:
            qty = float(lot.get("qty") or 0.0)
            net_qty += qty if str(lot.get("side")) == "buy" else -qty
        if abs(net_qty) > 1e-9:
            open_qty_by_symbol[symbol] = net_qty
    top_losses = [
        {"symbol": symbol, "net_pnl": pnl}
        for symbol, pnl in sorted(pnl_by_symbol.items(), key=lambda item: (item[1], item[0]))
        if pnl < 0
    ][:3]
    return {
        "status": "ready",
        "path": str(fill_events_path),
        "report_date": report_date,
        "fill_rows_seen": int(len(rows)),
        "same_day_fill_events": int(len(events)),
        "closed_trades": int(closed_trades),
        "net_pnl": float(sum(pnl_by_symbol.values())),
        "open_qty_by_symbol": open_qty_by_symbol,
        "top_loss_symbols": top_losses,
    }


def _collect_learning_snapshot(args: dict[str, Any], health: dict[str, Any]) -> dict[str, Any]:
    runtime_root = _runtime_root(args)
    after_hours = _read_json_object(runtime_root / "after_hours_training_state.json")
    execution_learning = _read_json_object(runtime_root / "execution_learning_state.json")
    execution_autotune = _read_json_object(runtime_root / "execution_autotune.json")

    learning_global = execution_learning.get("global")
    learning_global_obj = learning_global if isinstance(learning_global, dict) else {}
    model_liveness = health.get("model_liveness")
    model_liveness_obj = model_liveness if isinstance(model_liveness, dict) else {}

    return {
        "after_hours": {
            "updated_at": after_hours.get("updated_at"),
            "model_name": after_hours.get("model_name"),
            "model_id": after_hours.get("model_id"),
            "rows": after_hours.get("rows"),
            "governance_status": after_hours.get("governance_status"),
            "promotion_gate_passed": after_hours.get("promotion_gate_passed"),
            "promotion_consecutive_passes": after_hours.get("promotion_consecutive_passes"),
            "promotion_confidence_enabled": after_hours.get("promotion_confidence_enabled"),
            "promotion_confidence_gate_passed": after_hours.get(
                "promotion_confidence_gate_passed"
            ),
            "promotion_confidence_reason": after_hours.get("promotion_confidence_reason"),
            "promotion_confidence_observed": after_hours.get("promotion_confidence_observed"),
            "report_path": after_hours.get("report_path"),
        },
        "execution_learning": {
            "updated_at": execution_learning.get("updated_at"),
            "samples": learning_global_obj.get("samples"),
            "fill_rate": learning_global_obj.get("fill_rate"),
            "mean_slippage_bps": learning_global_obj.get("mean_slippage_bps"),
            "mean_net_edge_bps": learning_global_obj.get("mean_net_edge_bps"),
        },
        "execution_autotune": {
            "generated_at": execution_autotune.get("generated_at"),
            "enabled": execution_autotune.get("enabled"),
            "active": execution_autotune.get("active"),
            "profile_bias": execution_autotune.get("profile_bias"),
            "sample_count": execution_autotune.get("sample_count"),
            "mean_slippage_bps": execution_autotune.get("mean_slippage_bps"),
            "fill_rate": execution_autotune.get("fill_rate"),
            "reason": execution_autotune.get("reason"),
        },
        "model_liveness": {
            "last_ml_signal_ts": model_liveness_obj.get("last_ml_signal_ts"),
            "last_rl_signal_ts": model_liveness_obj.get("last_rl_signal_ts"),
            "ml_age_s": model_liveness_obj.get("ml_age_s"),
            "rl_age_s": model_liveness_obj.get("rl_age_s"),
            "ml_since_start_s": model_liveness_obj.get("ml_since_start_s"),
            "rl_since_start_s": model_liveness_obj.get("rl_since_start_s"),
        },
    }


def _collect_eod_summary_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    report = _normalized_runtime_report_payload(_runtime_report_payload())
    report_date = _extract_report_date(report)
    if not report_date:
        fallback_report = _read_json_object(_runtime_root(args) / "runtime_performance_report_latest.json")
        if fallback_report:
            report = _normalized_runtime_report_payload(fallback_report)
            report_date = _extract_report_date(report)
    go_no_go = report.get("go_no_go")
    go_no_go_obj = go_no_go if isinstance(go_no_go, dict) else {}
    observed = go_no_go_obj.get("observed")
    observed_obj = observed if isinstance(observed, dict) else {}
    trade_metric_scope = observed_obj.get("trade_metric_scope")
    trade_metric_scope_obj = trade_metric_scope if isinstance(trade_metric_scope, dict) else {}
    trade_metric_scope_end_date = str(
        trade_metric_scope_obj.get("end_date") or ""
    ).strip()
    observed_trade_kpis_current = (
        not report_date
        or not trade_metric_scope_end_date
        or trade_metric_scope_end_date == report_date
    )
    execution = report.get("execution_vs_alpha")
    execution_obj = execution if isinstance(execution, dict) else {}
    trade_history = report.get("trade_history")
    trade_history_obj = trade_history if isinstance(trade_history, dict) else {}

    port = int(
        args.get("health_port")
        or os.getenv("HEALTHCHECK_PORT", str(_DEFAULT_HEALTH_PORT))
    )
    timeout_s = float(args.get("health_timeout_s") or 2.0)
    health = _safe_health_payload(port=port, timeout_s=timeout_s)
    data_provider = health.get("data_provider")
    data_provider_obj = data_provider if isinstance(data_provider, dict) else {}
    broker = health.get("broker")
    broker_obj = broker if isinstance(broker, dict) else {}

    top_losses = (
        (trade_history_obj.get("top_loss_drivers") or {}).get("symbols")
        if isinstance(trade_history_obj.get("top_loss_drivers"), dict)
        else []
    )
    top_loss_items: list[dict[str, Any]] = []
    if isinstance(top_losses, list):
        for row in top_losses[:3]:
            if not isinstance(row, dict):
                continue
            top_loss_items.append(
                {
                    "symbol": row.get("name"),
                    "net_pnl": _float_or_none(row.get("net_pnl")),
                }
            )

    slippage_drag_bps = _float_or_none(observed_obj.get("slippage_drag_bps"))
    if slippage_drag_bps is None:
        slippage_drag_bps = _float_or_none(execution_obj.get("slippage_drag_bps"))

    execution_capture_ratio = _float_or_none(observed_obj.get("execution_capture_ratio"))
    if execution_capture_ratio is None:
        execution_capture_ratio = _float_or_none(execution_obj.get("execution_capture_ratio"))

    daily_trade_row = _daily_trade_row_for_report_date(trade_history_obj, report_date)
    net_pnl = _float_or_none(daily_trade_row.get("net_pnl"))
    if net_pnl is None and observed_trade_kpis_current:
        net_pnl = _float_or_none(observed_obj.get("net_pnl"))
    profit_factor = _float_or_none(daily_trade_row.get("profit_factor"))
    if profit_factor is None and observed_trade_kpis_current:
        profit_factor = _float_or_none(observed_obj.get("profit_factor"))
    win_rate = _float_or_none(daily_trade_row.get("win_rate"))
    if win_rate is None and observed_trade_kpis_current:
        win_rate = _float_or_none(observed_obj.get("win_rate"))
    closed_trades = daily_trade_row.get("trades")
    if closed_trades is None and observed_trade_kpis_current:
        closed_trades = observed_obj.get("closed_trades")
    same_day_fill_summary = _same_day_fill_round_trip_summary(args, report_date)
    same_day_net_pnl = _float_or_none(same_day_fill_summary.get("net_pnl"))
    pnl_discrepancy: dict[str, Any] | None = None
    if net_pnl is not None and same_day_net_pnl is not None:
        delta = float(net_pnl) - float(same_day_net_pnl)
        pnl_discrepancy = {
            "status": "mismatch" if abs(delta) > 0.01 else "matched",
            "accounting_net_pnl": float(net_pnl),
            "same_day_fill_net_pnl": float(same_day_net_pnl),
            "delta": float(delta),
        }

    snapshot = {
        "report_date": report_date,
        "go_no_go_gate_passed": go_no_go_obj.get("gate_passed"),
        "go_no_go_failed_checks": list(go_no_go_obj.get("failed_checks") or []),
        "net_pnl": net_pnl,
        "pnl_basis": "fifo_accounting_daily_trade_stats",
        "same_day_fill_summary": same_day_fill_summary,
        "pnl_discrepancy": pnl_discrepancy,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "closed_trades": closed_trades,
        "execution_capture_ratio": execution_capture_ratio,
        "slippage_drag_bps": slippage_drag_bps,
        "order_reject_rate_pct": _float_or_none(observed_obj.get("order_reject_rate_pct")),
        "open_position_reconciliation_mismatch_count": observed_obj.get(
            "open_position_reconciliation_mismatch_count"
        ),
        "open_position_reconciliation_max_abs_delta_qty": _float_or_none(
            observed_obj.get("open_position_reconciliation_max_abs_delta_qty")
        ),
        "health_status": str(health.get("status") or "unknown"),
        "health_reason": str(health.get("reason") or "unknown"),
        "provider_status": str(data_provider_obj.get("status") or "unknown"),
        "provider_active": str(data_provider_obj.get("active") or "unknown"),
        "using_backup": bool(data_provider_obj.get("using_backup", False)),
        "broker_status": str(broker_obj.get("status") or "unknown"),
        "top_loss_symbols": top_loss_items,
        "timestamp": str(health.get("timestamp") or utc_now_iso()),
    }
    snapshot["learning"] = _collect_learning_snapshot(args, health)
    return snapshot


def _after_hours_training_gate(report_date: str, args: dict[str, Any]) -> dict[str, Any]:
    required = _bool_arg(
        args.get("require_after_hours_training"),
        default=_bool_arg(
            os.getenv("AI_TRADING_SLACK_EOD_REQUIRE_AFTER_HOURS_TRAINING"),
            default=True,
        ),
    )
    marker_path = _after_hours_training_marker_path(args)
    marker = _read_json_object(marker_path)
    marker_date = str(marker.get("date") or "").strip()
    marker_status = str(marker.get("status") or "").strip().lower()
    ready = True
    reason = "not_required"
    if required:
        ready = bool(report_date) and marker_date == report_date and marker_status == "trained"
        reason = "trained_for_report_date" if ready else "training_not_complete_for_report_date"
    return {
        "required": required,
        "ready": ready,
        "reason": reason,
        "marker_path": str(marker_path),
        "marker_date": marker_date or None,
        "marker_status": marker_status or None,
        "marker_updated_at": marker.get("updated_at"),
        "marker_model_id": marker.get("model_id"),
        "marker_model_name": marker.get("model_name"),
    }


def _capture_ratio_threshold(args: dict[str, Any]) -> float:
    raw = args.get("min_capture_ratio")
    if raw is None:
        raw = os.getenv("AI_TRADING_INCIDENT_MIN_CAPTURE_RATIO", "0.08")
    return max(0.0, float(raw))


def _runtime_gonogo_block_openings_enabled(args: dict[str, Any]) -> bool:
    raw = args.get("runtime_gonogo_block_openings_enabled")
    if raw in (None, ""):
        raw = os.getenv("AI_TRADING_EXECUTION_RUNTIME_GONOGO_BLOCK_OPENINGS_ENABLED")
    if raw in (None, ""):
        raw = os.getenv("AI_TRADING_RUNTIME_GONOGO_BLOCK_OPENINGS_ENABLED")
    return _bool_arg(raw, default=True)


def _block_eod_on_training_gate(args: dict[str, Any]) -> bool:
    return _bool_arg(
        args.get("block_on_training_gate"),
        default=_bool_arg(os.getenv("AI_TRADING_SLACK_EOD_BLOCK_ON_TRAINING_GATE"), default=False),
    )


def _should_suppress_startup_warmup_health_alert(
    snapshot: dict[str, Any], args: dict[str, Any]
) -> bool:
    enabled = _bool_arg(
        args.get("suppress_startup_warmup_health_alerts"),
        default=_bool_arg(
            os.getenv("AI_TRADING_SLACK_SUPPRESS_STARTUP_WARMUP_HEALTH_ALERTS"),
            default=True,
        ),
    )
    if not enabled:
        return False

    health_reason = str(snapshot.get("health_reason") or "").strip().lower()
    provider_reason = str(snapshot.get("provider_reason") or "").strip().lower()
    provider_status = str(snapshot.get("provider_status") or "").strip().lower()
    if health_reason in _STARTUP_WARMUP_HEALTH_REASONS and provider_status == "warming_up":
        return True
    if health_reason in _STARTUP_WARMUP_HEALTH_REASONS:
        return provider_reason in {"", "unknown", "market_closed", "warmup_cycle", "startup"}

    if not _within_incident_startup_grace(snapshot, args):
        return False
    if provider_status == "warming_up":
        return True
    service_phase = str(snapshot.get("service_phase") or "").strip().lower()
    service_reason = str(snapshot.get("service_reason") or "").strip().lower()
    service_starting = service_phase in {"bootstrap", "starting", "startup"} or service_reason in {
        "startup",
        "bootstrap",
        "starting",
    }
    if service_starting and health_reason in {
        "provider_status_unknown",
        "provider_state_unknown",
        "data_provider_unknown",
        "health_payload_unavailable",
    }:
        return True
    return health_reason in {
        "broker_status_unknown",
        "startup_config_resolved",
    } or provider_reason in {
        "startup_config_resolved",
        "warmup_cycle",
        "startup",
    }


def _incident_startup_grace_seconds(args: dict[str, Any]) -> int:
    raw = args.get("startup_grace_seconds")
    if raw in (None, ""):
        raw = os.getenv("AI_TRADING_SLACK_INCIDENT_STARTUP_GRACE_SECONDS")
    return max(0, _int_arg(raw, default=300))


def _snapshot_service_age_seconds(snapshot: dict[str, Any]) -> float | None:
    phase_since = _parse_iso_ts(snapshot.get("service_phase_since"))
    if phase_since is None:
        return None
    observed_at = _parse_iso_ts(snapshot.get("timestamp")) or datetime.now(UTC)
    return max(0.0, (observed_at - phase_since).total_seconds())


def _within_incident_startup_grace(
    snapshot: dict[str, Any], args: dict[str, Any]
) -> bool:
    grace_seconds = _incident_startup_grace_seconds(args)
    if grace_seconds <= 0:
        return False
    service_age_seconds = _snapshot_service_age_seconds(snapshot)
    return service_age_seconds is not None and service_age_seconds <= float(grace_seconds)


def _market_closed_snapshot(snapshot: dict[str, Any]) -> bool:
    health_reason = str(snapshot.get("health_reason") or "").strip().lower()
    provider_reason = str(snapshot.get("provider_reason") or "").strip().lower()
    return health_reason == "market_closed" or provider_reason == "market_closed"


def _should_suppress_gonogo_incident(
    snapshot: dict[str, Any], args: dict[str, Any]
) -> bool:
    if _within_incident_startup_grace(snapshot, args):
        return True
    enabled = _bool_arg(
        args.get("suppress_market_closed_gonogo_alerts"),
        default=_bool_arg(
            os.getenv("AI_TRADING_SLACK_SUPPRESS_MARKET_CLOSED_GONOGO_ALERTS"),
            default=True,
        ),
    )
    return bool(enabled and _market_closed_snapshot(snapshot))


def _should_suppress_market_closed_health_alert(
    snapshot: dict[str, Any], args: dict[str, Any]
) -> bool:
    enabled = _bool_arg(
        args.get("suppress_market_closed_health_alerts"),
        default=_bool_arg(
            os.getenv("AI_TRADING_SLACK_SUPPRESS_MARKET_CLOSED_HEALTH_ALERTS"),
            default=True,
        ),
    )
    if not enabled or not _market_closed_snapshot(snapshot):
        return False
    health_status = str(snapshot.get("health_status") or "").strip().lower()
    provider_status = str(snapshot.get("provider_status") or "").strip().lower()
    provider_reason = str(snapshot.get("provider_reason") or "").strip().lower()
    broker_status = str(snapshot.get("broker_status") or "").strip().lower()
    using_backup = bool(snapshot.get("using_backup", False))
    return (
        health_status in {"healthy", "ready"}
        and provider_status in {"warming_up", "healthy", "ready", ""}
        and provider_reason in {"market_closed", ""}
        and broker_status in {"connected", "reachable", "unknown", ""}
        and not using_backup
    )


def _evaluate_incident_triggers(snapshot: dict[str, Any], args: dict[str, Any]) -> list[str]:
    triggers: list[str] = []
    health_reason = str(snapshot.get("health_reason") or "").strip().lower()
    startup_or_warmup = health_reason in _STARTUP_WARMUP_HEALTH_REASONS
    startup_grace_active = _within_incident_startup_grace(snapshot, args)
    market_closed = _market_closed_snapshot(snapshot)

    runtime_gonogo_block_openings_enabled = _bool_arg(
        snapshot.get("runtime_gonogo_block_openings_enabled"),
        default=True,
    )
    gate_passed = snapshot.get("go_no_go_gate_passed")
    failed_checks = list(snapshot.get("go_no_go_failed_checks") or [])
    availability_only_checks = {
        "closed_trades",
        "gate_used_days",
        "live_samples_sufficient",
        "open_position_reconciliation_available",
        "pnl_available",
        "trade_used_days",
    }
    material_failed_checks = [
        str(check)
        for check in failed_checks
        if str(check) not in availability_only_checks
    ]
    if (
        runtime_gonogo_block_openings_enabled
        and not _should_suppress_gonogo_incident(snapshot, args)
    ):
        if gate_passed is False and (not failed_checks or bool(material_failed_checks)):
            triggers.append("go_no_go_failed")
        if material_failed_checks:
            triggers.append("go_no_go_failed_checks")

    capture_ratio = _float_or_none(snapshot.get("execution_capture_ratio"))
    min_capture = _capture_ratio_threshold(args)
    if (
        capture_ratio is not None
        and capture_ratio < min_capture
        and not startup_or_warmup
        and not startup_grace_active
        and not market_closed
    ):
        triggers.append("execution_capture_ratio_low")
    fill_ratio = _float_or_none(snapshot.get("execution_fill_ratio"))
    fill_ratio_samples = _int_arg(snapshot.get("execution_fill_ratio_samples"), default=0)
    min_fill_ratio = _float_or_none(
        args.get("min_fill_ratio")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_MIN_FILL_RATIO")
    )
    min_fill_ratio_samples = _int_arg(
        args.get("min_fill_ratio_samples")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_FILL_RATIO_MIN_SAMPLES"),
        default=20,
    )
    min_fill_ratio_samples = max(1, min_fill_ratio_samples)
    if min_fill_ratio is not None:
        min_fill_ratio = max(0.0, min(float(min_fill_ratio), 1.0))
        if (
            fill_ratio is not None
            and fill_ratio_samples >= min_fill_ratio_samples
            and fill_ratio < min_fill_ratio
            and not startup_or_warmup
            and not market_closed
        ):
            triggers.append("execution_fill_ratio_low")
    fill_ratio_healthy = bool(
        min_fill_ratio is not None
        and fill_ratio is not None
        and fill_ratio_samples >= min_fill_ratio_samples
        and fill_ratio >= min_fill_ratio
    )

    precheck_failure_count = _int_arg(snapshot.get("precheck_failure_count"), default=0)
    execution_skipped_count = _int_arg(snapshot.get("execution_skipped_count"), default=0)
    precheck_failure_ratio = _float_or_none(snapshot.get("precheck_failure_ratio"))
    precheck_spike_min_count = _int_arg(
        args.get("precheck_spike_min_count")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_PRECHECK_SPIKE_MIN_COUNT"),
        default=10,
    )
    precheck_spike_min_ratio = _float_or_none(
        args.get("precheck_spike_min_ratio")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_PRECHECK_SPIKE_MIN_RATIO")
    )
    if precheck_spike_min_ratio is None:
        precheck_spike_min_ratio = 0.60
    precheck_spike_min_ratio = max(0.0, min(float(precheck_spike_min_ratio), 1.0))
    precheck_spike_min_skips = _int_arg(
        args.get("precheck_spike_min_skipped")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_PRECHECK_SPIKE_MIN_SKIPPED"),
        default=12,
    )
    if (
        precheck_failure_count >= max(1, precheck_spike_min_count)
        and execution_skipped_count >= max(1, precheck_spike_min_skips)
        and precheck_failure_ratio is not None
        and precheck_failure_ratio >= precheck_spike_min_ratio
        and not fill_ratio_healthy
        and not startup_or_warmup
        and not startup_grace_active
        and not market_closed
    ):
        triggers.append("pre_execution_checks_spike")

    top_rejection_concentration_ratio = _float_or_none(
        snapshot.get("top_rejection_concentration_ratio")
    )
    gate_rejected_records = _int_arg(snapshot.get("gate_rejected_records"), default=0)
    max_rejection_concentration_ratio = _float_or_none(
        args.get("max_rejection_concentration_ratio")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_MAX_REJECTION_CONCENTRATION_RATIO")
    )
    if max_rejection_concentration_ratio is None:
        max_rejection_concentration_ratio = 0.65
    max_rejection_concentration_ratio = max(
        0.0,
        min(float(max_rejection_concentration_ratio), 1.0),
    )
    min_rejected_records_for_concentration = _int_arg(
        args.get("min_rejected_records_for_concentration")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_MIN_REJECTED_RECORDS_FOR_CONCENTRATION"),
        default=20,
    )
    if (
        top_rejection_concentration_ratio is not None
        and gate_rejected_records >= max(1, min_rejected_records_for_concentration)
        and top_rejection_concentration_ratio >= max_rejection_concentration_ratio
        and not startup_or_warmup
        and not startup_grace_active
        and not market_closed
    ):
        triggers.append("rejection_concentration_high")

    expected_edge_per_accept_bps = _float_or_none(
        snapshot.get("expected_edge_per_accept_bps")
    )
    edge_realism_gap_ratio = _float_or_none(snapshot.get("edge_realism_gap_ratio"))
    min_expected_edge_bps_for_realism = _float_or_none(
        args.get("min_expected_edge_bps_for_realism")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_MIN_EXPECTED_EDGE_BPS_FOR_REALISM")
    )
    if min_expected_edge_bps_for_realism is None:
        min_expected_edge_bps_for_realism = 0.5
    min_edge_realism_ratio = _float_or_none(
        args.get("min_edge_realism_ratio")
        or os.getenv("AI_TRADING_SLACK_INCIDENT_MIN_EDGE_REALISM_RATIO")
    )
    if min_edge_realism_ratio is None:
        min_edge_realism_ratio = 0.35
    if (
        expected_edge_per_accept_bps is not None
        and edge_realism_gap_ratio is not None
        and expected_edge_per_accept_bps >= float(min_expected_edge_bps_for_realism)
        and edge_realism_gap_ratio < float(min_edge_realism_ratio)
        and not startup_or_warmup
        and not startup_grace_active
        and not market_closed
    ):
        triggers.append("edge_realism_gap_high")

    health_ok = bool(snapshot.get("health_ok", False))
    health_status = str(snapshot.get("health_status") or "unknown").lower()
    health_degraded = not health_ok or health_status in {"degraded", "down", "unhealthy"}
    if (
        health_degraded
        and not _should_suppress_startup_warmup_health_alert(snapshot, args)
        and not _should_suppress_market_closed_health_alert(snapshot, args)
    ):
        triggers.append("health_degraded")

    if bool(snapshot.get("using_backup", False)) and not startup_grace_active:
        triggers.append("data_provider_backup_active")

    broker_status = str(snapshot.get("broker_status") or "unknown").lower()
    if broker_status not in {"connected", "reachable", "unknown"}:
        triggers.append("broker_disconnected")

    return sorted(set(triggers))


def _incident_fingerprint(snapshot: dict[str, Any], triggers: list[str]) -> str:
    capture_ratio = _float_or_none(snapshot.get("execution_capture_ratio"))
    slippage_drag = _float_or_none(snapshot.get("slippage_drag_bps"))
    fill_ratio = _float_or_none(snapshot.get("execution_fill_ratio"))
    precheck_failure_ratio = _float_or_none(snapshot.get("precheck_failure_ratio"))
    top_rejection_concentration_ratio = _float_or_none(
        snapshot.get("top_rejection_concentration_ratio")
    )
    edge_realism_gap_ratio = _float_or_none(snapshot.get("edge_realism_gap_ratio"))
    material = {
        "triggers": sorted(triggers),
        "go_no_go_gate_passed": snapshot.get("go_no_go_gate_passed"),
        "go_no_go_failed_checks": list(snapshot.get("go_no_go_failed_checks") or []),
        "execution_capture_ratio": round(capture_ratio, 6) if capture_ratio is not None else None,
        "slippage_drag_bps": round(slippage_drag, 6) if slippage_drag is not None else None,
        "execution_fill_ratio": round(fill_ratio, 6) if fill_ratio is not None else None,
        "execution_fill_ratio_samples": _int_arg(
            snapshot.get("execution_fill_ratio_samples"), default=0
        ),
        "expected_edge_per_accept_bps": _float_or_none(
            snapshot.get("expected_edge_per_accept_bps")
        ),
        "realization_gap_bps": _float_or_none(snapshot.get("realization_gap_bps")),
        "edge_realism_gap_ratio": (
            round(edge_realism_gap_ratio, 6)
            if edge_realism_gap_ratio is not None
            else None
        ),
        "gate_rejected_records": _int_arg(snapshot.get("gate_rejected_records"), default=0),
        "top_rejection_concentration_gate": snapshot.get("top_rejection_concentration_gate"),
        "top_rejection_concentration_ratio": (
            round(top_rejection_concentration_ratio, 6)
            if top_rejection_concentration_ratio is not None
            else None
        ),
        "precheck_failure_count": _int_arg(snapshot.get("precheck_failure_count"), default=0),
        "precheck_failure_ratio": (
            round(precheck_failure_ratio, 6)
            if precheck_failure_ratio is not None
            else None
        ),
        "health_status": snapshot.get("health_status"),
        "health_reason": snapshot.get("health_reason"),
        "provider_status": snapshot.get("provider_status"),
        "provider_reason": snapshot.get("provider_reason"),
        "broker_status": snapshot.get("broker_status"),
        "using_backup": bool(snapshot.get("using_backup", False)),
    }
    encoded = json.dumps(material, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _incident_signature(snapshot: dict[str, Any], triggers: list[str]) -> str:
    """Stable trigger signature for anti-spam dedupe across metric drift."""
    material = {
        "triggers": _material_incident_triggers_for_dedupe(triggers),
        "go_no_go_failed_checks": sorted(set(snapshot.get("go_no_go_failed_checks") or [])),
    }
    encoded = json.dumps(material, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _material_incident_triggers_for_dedupe(triggers: list[str] | set[str]) -> list[str]:
    """Collapse secondary execution-noise triggers under the same Go/No-Go incident."""

    material = {
        str(trigger).strip()
        for trigger in triggers
        if str(trigger).strip()
    }
    if material.intersection({"go_no_go_failed", "go_no_go_failed_checks"}):
        material.discard("pre_execution_checks_spike")
    return sorted(material)


def _incident_repeat_cooldown_minutes(args: dict[str, Any]) -> int:
    raw = args.get("repeat_cooldown_minutes")
    if raw is None:
        raw = os.getenv("AI_TRADING_SLACK_INCIDENT_REPEAT_COOLDOWN_MINUTES")
    return max(0, _int_arg(raw, default=_DEFAULT_INCIDENT_REPEAT_COOLDOWN_MINUTES))


def _incident_min_interval_minutes(args: dict[str, Any]) -> int:
    raw = args.get("min_interval_minutes")
    if raw is None:
        raw = os.getenv("AI_TRADING_SLACK_INCIDENT_MIN_INTERVAL_MINUTES")
    return max(0, _int_arg(raw, default=0))


def _confirm_health_unavailable_before_alert(args: dict[str, Any]) -> bool:
    return _bool_arg(
        args.get("confirm_health_unavailable_before_alert"),
        default=_bool_arg(
            os.getenv("AI_TRADING_SLACK_CONFIRM_HEALTH_UNAVAILABLE_BEFORE_ALERT"),
            default=True,
        ),
    )


def _requires_health_unavailable_confirmation(
    snapshot: dict[str, Any], triggers: list[str], args: dict[str, Any]
) -> bool:
    if not _confirm_health_unavailable_before_alert(args):
        return False
    health_reason = str(snapshot.get("health_reason") or "").strip().lower()
    if health_reason != "health_payload_unavailable":
        return False
    return "health_degraded" in set(triggers)


def _slack_webhook_url(args: dict[str, Any]) -> str:
    direct = str(args.get("webhook_url") or "").strip()
    if direct:
        return direct
    env_url = (
        os.getenv("AI_TRADING_SLACK_WEBHOOK_URL", "").strip()
        or os.getenv("SLACK_WEBHOOK_URL", "").strip()
    )
    if env_url:
        return env_url
    raise RuntimeError("missing Slack webhook URL (webhook_url arg or env var)")


def _post_slack_message(webhook_url: str, payload: dict[str, Any], timeout_s: float = 5.0) -> int:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=webhook_url,
        method="POST",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Slack webhook returned HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Slack webhook request failed: {exc.reason}") from exc


def _slack_plain_text(text: str, *, max_len: int = 150) -> dict[str, Any]:
    value = str(text or "").strip()
    if len(value) > max_len:
        value = f"{value[: max_len - 1]}…"
    return {"type": "plain_text", "text": value or "n/a", "emoji": True}


def _slack_mrkdwn(text: str, *, max_len: int = 3000) -> dict[str, Any]:
    value = str(text or "").strip()
    if len(value) > max_len:
        value = f"{value[: max_len - 1]}…"
    return {"type": "mrkdwn", "text": value or "n/a"}


def _slack_field(label: str, value: Any) -> dict[str, Any]:
    return _slack_mrkdwn(f"*{label}*\n{value}", max_len=2000)


def _slack_section(
    text: str | None = None,
    *,
    fields: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "section"}
    if text is not None:
        block["text"] = _slack_mrkdwn(text)
    if fields:
        block["fields"] = fields[:10]
    return block


def _slack_context(*items: str) -> dict[str, Any]:
    return {"type": "context", "elements": [_slack_mrkdwn(item, max_len=300) for item in items[:10]]}


def _incident_trigger_labels(triggers: list[str]) -> list[str]:
    trigger_map = {
        "go_no_go_failed": "Go/no-go gate failed",
        "go_no_go_failed_checks": "Go/no-go reported failed checks",
        "execution_capture_ratio_low": "Execution capture ratio is below target",
        "execution_fill_ratio_low": "Recent fill ratio is below target",
        "pre_execution_checks_spike": "Pre-execution order-check failures are spiking",
        "rejection_concentration_high": "One rejection reason is dominating decisions",
        "edge_realism_gap_high": "Live realized edge is lagging expected edge",
        "health_degraded": "Runtime health is degraded",
        "data_provider_backup_active": "Data provider is running on backup feed",
        "broker_disconnected": "Broker connection is not healthy",
    }
    return [trigger_map.get(trigger, trigger) for trigger in triggers]


def _incident_message_text(snapshot: dict[str, Any], triggers: list[str]) -> str:
    failed_checks = list(snapshot.get("go_no_go_failed_checks") or [])
    checks_text = ", ".join(failed_checks) if failed_checks else "none"
    window_minutes = _int_arg(snapshot.get("execution_window_minutes"), default=30)
    fill_samples = _int_arg(snapshot.get("execution_fill_ratio_samples"), default=0)
    fill_filled = _int_arg(snapshot.get("execution_fill_ratio_filled"), default=0)
    precheck_fail_count = _int_arg(snapshot.get("precheck_failure_count"), default=0)
    execution_skipped_count = _int_arg(snapshot.get("execution_skipped_count"), default=0)
    precheck_top_details_raw = list(
        snapshot.get("precheck_failure_top_actionable_details")
        or snapshot.get("precheck_failure_top_details")
        or []
    )
    precheck_top_details: list[str] = []
    for item in precheck_top_details_raw[:3]:
        if not isinstance(item, dict):
            continue
        detail = str(item.get("detail") or "").strip()
        if not detail:
            continue
        count = _int_arg(item.get("count"), default=0)
        precheck_top_details.append(f"{detail.replace('_', ' ')}={count}")
    precheck_top_details_text = ", ".join(precheck_top_details) if precheck_top_details else "n/a"
    top_concentration_gate = str(snapshot.get("top_rejection_concentration_gate") or "").strip()
    top_concentration_ratio = _float_or_none(snapshot.get("top_rejection_concentration_ratio"))
    rejected_records = _int_arg(snapshot.get("gate_rejected_records"), default=0)
    trigger_lines = [f"- {label}" for label in _incident_trigger_labels(triggers)] or ["- None"]
    return "\n".join(
        [
            "🚨 ai-trading incident update",
            "",
            "⚠️ What triggered this alert:",
            *trigger_lines,
            "",
            "📊 Current status:",
            f"- Go/No-Go: {_fmt_gate_status(snapshot.get('go_no_go_gate_passed'))}",
            f"- Failed checks: {checks_text}",
            f"- Execution capture ratio: {_fmt_num(snapshot.get('execution_capture_ratio'), digits=3)}",
            (
                f"- Fill ratio ({window_minutes}m): "
                f"{_fmt_num(snapshot.get('execution_fill_ratio'), digits=3)} "
                f"(filled={fill_filled}, samples={fill_samples})"
            ),
            (
                f"- Pre-check failure spike ({window_minutes}m): "
                f"count={precheck_fail_count}, "
                f"ratio={_fmt_num(snapshot.get('precheck_failure_ratio'), digits=3)}, "
                f"skipped={execution_skipped_count}"
            ),
            f"- Top pre-check blockers ({window_minutes}m): {precheck_top_details_text}",
            f"- Slippage drag: {_fmt_num(snapshot.get('slippage_drag_bps'), digits=3)} bps",
            (
                "- Edge realism gap: "
                f"ratio={_fmt_num(snapshot.get('edge_realism_gap_ratio'), digits=3)} "
                f"(realized/expected), "
                f"realization_gap_bps={_fmt_num(snapshot.get('realization_gap_bps'), digits=3)}"
            ),
            (
                "- Rejection concentration: "
                f"{top_concentration_gate or 'n/a'} "
                f"({_fmt_pct(top_concentration_ratio, digits=1, assume_ratio=True)} "
                f"of rejected, rejected_records={rejected_records})"
            ),
            f"- Health: {snapshot.get('health_status')} ({snapshot.get('health_reason')})",
            (
                f"- Data provider: {snapshot.get('provider_active')} / {snapshot.get('provider_status')} "
                f"(reason={snapshot.get('provider_reason')}, backup={_fmt_bool(snapshot.get('using_backup'))})"
            ),
            f"- Broker: {snapshot.get('broker_status')}",
            "",
            f"🕒 Timestamp: {snapshot.get('timestamp')}",
        ]
    )


def _incident_message_blocks(snapshot: dict[str, Any], triggers: list[str]) -> list[dict[str, Any]]:
    failed_checks = list(snapshot.get("go_no_go_failed_checks") or [])
    checks_text = ", ".join(failed_checks) if failed_checks else "none"
    window_minutes = _int_arg(snapshot.get("execution_window_minutes"), default=30)
    fill_samples = _int_arg(snapshot.get("execution_fill_ratio_samples"), default=0)
    fill_filled = _int_arg(snapshot.get("execution_fill_ratio_filled"), default=0)
    precheck_fail_count = _int_arg(snapshot.get("precheck_failure_count"), default=0)
    execution_skipped_count = _int_arg(snapshot.get("execution_skipped_count"), default=0)
    trigger_labels = _incident_trigger_labels(triggers)
    trigger_text = "\n".join(f"• {label}" for label in trigger_labels) if trigger_labels else "• None"
    top_concentration_gate = str(snapshot.get("top_rejection_concentration_gate") or "").strip()
    top_concentration_ratio = _float_or_none(snapshot.get("top_rejection_concentration_ratio"))
    rejected_records = _int_arg(snapshot.get("gate_rejected_records"), default=0)

    precheck_top_details_raw = list(
        snapshot.get("precheck_failure_top_actionable_details")
        or snapshot.get("precheck_failure_top_details")
        or []
    )
    blockers: list[str] = []
    for item in precheck_top_details_raw[:3]:
        if not isinstance(item, dict):
            continue
        detail = str(item.get("detail") or "").strip()
        if not detail:
            continue
        blockers.append(f"{detail.replace('_', ' ')}={_int_arg(item.get('count'), default=0)}")
    blockers_text = ", ".join(blockers) if blockers else "n/a"

    return [
        {"type": "header", "text": _slack_plain_text("🚨 ai-trading incident update")},
        _slack_section(
            fields=[
                _slack_field("Go/No-Go", _fmt_gate_status(snapshot.get("go_no_go_gate_passed"))),
                _slack_field("Failed checks", checks_text),
                _slack_field(
                    "Health",
                    f"{snapshot.get('health_status') or 'unknown'} ({snapshot.get('health_reason') or 'n/a'})",
                ),
                _slack_field(
                    "Provider",
                    (
                        f"{snapshot.get('provider_active') or 'n/a'} / "
                        f"{snapshot.get('provider_status') or 'unknown'}"
                    ),
                ),
            ],
        ),
        {"type": "divider"},
        _slack_section(f"*Triggered by*\n{trigger_text}"),
        _slack_section(
            fields=[
                _slack_field("Capture ratio", _fmt_num(snapshot.get("execution_capture_ratio"), digits=3)),
                _slack_field(
                    f"Fill ratio ({window_minutes}m)",
                    (
                        f"{_fmt_num(snapshot.get('execution_fill_ratio'), digits=3)} "
                        f"({fill_filled}/{fill_samples})"
                    ),
                ),
                _slack_field(
                    f"Pre-check failures ({window_minutes}m)",
                    f"{precheck_fail_count} failures, {execution_skipped_count} skipped",
                ),
                _slack_field("Top blockers", blockers_text),
                _slack_field("Slippage drag", f"{_fmt_num(snapshot.get('slippage_drag_bps'), digits=3)} bps"),
                _slack_field(
                    "Rejection concentration",
                    (
                        f"{top_concentration_gate or 'n/a'} "
                        f"({_fmt_pct(top_concentration_ratio, digits=1, assume_ratio=True)}, "
                        f"records={rejected_records})"
                    ),
                ),
            ],
        ),
        _slack_context(
            f"Broker: {snapshot.get('broker_status') or 'unknown'}",
            f"Backup feed: {_fmt_bool(snapshot.get('using_backup'))}",
            f"Timestamp: {snapshot.get('timestamp') or 'n/a'}",
        ),
    ]


def _eod_fingerprint(snapshot: dict[str, Any]) -> str:
    material = {
        "report_date": snapshot.get("report_date"),
        "go_no_go_gate_passed": snapshot.get("go_no_go_gate_passed"),
        "go_no_go_failed_checks": list(snapshot.get("go_no_go_failed_checks") or []),
        "net_pnl": _float_or_none(snapshot.get("net_pnl")),
        "profit_factor": _float_or_none(snapshot.get("profit_factor")),
        "win_rate": _float_or_none(snapshot.get("win_rate")),
        "execution_capture_ratio": _float_or_none(snapshot.get("execution_capture_ratio")),
        "slippage_drag_bps": _float_or_none(snapshot.get("slippage_drag_bps")),
        "same_day_fill_summary": snapshot.get("same_day_fill_summary"),
        "pnl_discrepancy": snapshot.get("pnl_discrepancy"),
        "learning": snapshot.get("learning"),
    }
    encoded = json.dumps(material, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fmt_num(value: Any, *, digits: int = 3) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def _fmt_gate_status(value: Any) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "UNKNOWN"


def _fmt_pct(value: Any, *, digits: int = 1, assume_ratio: bool = False) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    if assume_ratio:
        number *= 100.0
    return f"{number:.{digits}f}%"


def _fmt_currency(value: Any, *, digits: int = 2) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return f"${number:,.{digits}f}"


def _fmt_count(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return str(int(number))


def _fmt_duration_s(value: Any, *, digits: int = 1) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}s"


def _eod_message_text(snapshot: dict[str, Any]) -> str:
    failed_checks = list(snapshot.get("go_no_go_failed_checks") or [])
    checks_text = ", ".join(failed_checks) if failed_checks else "none"
    training_gate = snapshot.get("training_gate")
    training_gate_obj = training_gate if isinstance(training_gate, dict) else {}
    losses = snapshot.get("top_loss_symbols")
    losses_rows = losses if isinstance(losses, list) else []
    losses_text = ", ".join(
        f"{str(item.get('symbol') or 'n/a')} ({_fmt_currency(item.get('net_pnl'), digits=1)})"
        for item in losses_rows
        if isinstance(item, dict)
    )
    if not losses_text:
        losses_text = "none"
    same_day_fill = snapshot.get("same_day_fill_summary")
    same_day_fill_obj = same_day_fill if isinstance(same_day_fill, dict) else {}
    pnl_discrepancy = snapshot.get("pnl_discrepancy")
    pnl_discrepancy_obj = pnl_discrepancy if isinstance(pnl_discrepancy, dict) else {}
    pnl_check = str(pnl_discrepancy_obj.get("status") or same_day_fill_obj.get("status") or "n/a")
    if pnl_discrepancy_obj.get("delta") is not None:
        pnl_check = f"{pnl_check}, delta={_fmt_currency(pnl_discrepancy_obj.get('delta'))}"

    learning = snapshot.get("learning")
    learning_obj = learning if isinstance(learning, dict) else {}
    after_hours = learning_obj.get("after_hours")
    after_hours_obj = after_hours if isinstance(after_hours, dict) else {}
    exec_learning = learning_obj.get("execution_learning")
    exec_learning_obj = exec_learning if isinstance(exec_learning, dict) else {}
    exec_autotune = learning_obj.get("execution_autotune")
    exec_autotune_obj = exec_autotune if isinstance(exec_autotune, dict) else {}
    model_liveness = learning_obj.get("model_liveness")
    model_liveness_obj = model_liveness if isinstance(model_liveness, dict) else {}

    return "\n".join(
        [
            "📘 ai-trading EOD summary",
            f"Date: {snapshot.get('report_date') or 'unknown'}",
            "",
            (
                f"✅ Go/No-Go: {_fmt_gate_status(snapshot.get('go_no_go_gate_passed'))} "
                f"(failed checks: {checks_text})"
            ),
            "",
            "💰 Day performance:",
            f"- Accounting net PnL: {_fmt_currency(snapshot.get('net_pnl'))}",
            f"- Same-day fill PnL: {_fmt_currency(same_day_fill_obj.get('net_pnl'))}",
            f"- PnL check: {pnl_check}",
            f"- Profit factor: {_fmt_num(snapshot.get('profit_factor'), digits=3)}",
            f"- Win rate: {_fmt_pct(snapshot.get('win_rate'), digits=1, assume_ratio=True)}",
            f"- Closed trades: {_fmt_count(snapshot.get('closed_trades'))}",
            "",
            "⚙️ Execution quality:",
            f"- Capture ratio: {_fmt_num(snapshot.get('execution_capture_ratio'), digits=3)}",
            f"- Slippage drag: {_fmt_num(snapshot.get('slippage_drag_bps'), digits=3)} bps",
            f"- Reject rate: {_fmt_pct(snapshot.get('order_reject_rate_pct'), digits=2)}",
            (
                f"- Reconciliation: mismatches={snapshot.get('open_position_reconciliation_mismatch_count')}, "
                f"max abs delta qty={_fmt_num(snapshot.get('open_position_reconciliation_max_abs_delta_qty'), digits=3)}"
            ),
            "",
            "🩺 Runtime health:",
            f"- Health: {snapshot.get('health_status')} ({snapshot.get('health_reason')})",
            (
                f"- Data provider: {snapshot.get('provider_active')} / {snapshot.get('provider_status')} "
                f"(backup={_fmt_bool(snapshot.get('using_backup'))})"
            ),
            f"- Broker: {snapshot.get('broker_status')}",
            "",
            "🧠 Learning models:",
            (
                f"- Training completion gate: ready={_fmt_bool(training_gate_obj.get('ready'))}, "
                f"reason={training_gate_obj.get('reason') or 'n/a'}, "
                f"marker_date={training_gate_obj.get('marker_date') or 'n/a'}"
            ),
            (
                f"- After-hours model: {after_hours_obj.get('model_name') or 'n/a'} "
                f"[{after_hours_obj.get('governance_status') or 'n/a'}], "
                f"promotion gate={after_hours_obj.get('promotion_gate_passed')}, "
                f"confidence gate={after_hours_obj.get('promotion_confidence_gate_passed')}, "
                f"rows={after_hours_obj.get('rows')}, "
                f"model_id={after_hours_obj.get('model_id') or 'n/a'}"
            ),
            (
                f"- Promotion confidence: enabled={_fmt_bool(after_hours_obj.get('promotion_confidence_enabled'))}, "
                f"reason={after_hours_obj.get('promotion_confidence_reason') or 'n/a'}"
            ),
            (
                f"- Execution learning: samples={exec_learning_obj.get('samples')}, "
                f"fill_rate={_fmt_pct(exec_learning_obj.get('fill_rate'), digits=1, assume_ratio=True)}, "
                f"mean_slippage={_fmt_num(exec_learning_obj.get('mean_slippage_bps'), digits=3)} bps, "
                f"mean_net_edge={_fmt_num(exec_learning_obj.get('mean_net_edge_bps'), digits=3)} bps"
            ),
            (
                f"- Execution autotune: enabled={_fmt_bool(exec_autotune_obj.get('enabled'))}, "
                f"active={_fmt_bool(exec_autotune_obj.get('active'))}, "
                f"bias={exec_autotune_obj.get('profile_bias') or 'n/a'}, "
                f"samples={exec_autotune_obj.get('sample_count')}, "
                f"reason={exec_autotune_obj.get('reason') or 'n/a'}"
            ),
            (
                f"- Model liveness: ml_age={_fmt_duration_s(model_liveness_obj.get('ml_age_s'))}, "
                f"rl_age={_fmt_duration_s(model_liveness_obj.get('rl_age_s'))}, "
                f"last_ml={model_liveness_obj.get('last_ml_signal_ts') or 'n/a'}, "
                f"last_rl={model_liveness_obj.get('last_rl_signal_ts') or 'n/a'}"
            ),
            "",
            f"📉 Top loss symbols: {losses_text}",
            f"🕒 Timestamp: {snapshot.get('timestamp')}",
        ]
    )


def _eod_message_blocks(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    failed_checks = list(snapshot.get("go_no_go_failed_checks") or [])
    checks_text = ", ".join(failed_checks) if failed_checks else "none"
    training_gate = snapshot.get("training_gate")
    training_gate_obj = training_gate if isinstance(training_gate, dict) else {}
    losses = snapshot.get("top_loss_symbols")
    losses_rows = losses if isinstance(losses, list) else []
    losses_text = ", ".join(
        f"{str(item.get('symbol') or 'n/a')} ({_fmt_currency(item.get('net_pnl'), digits=1)})"
        for item in losses_rows
        if isinstance(item, dict)
    )
    if not losses_text:
        losses_text = "none"
    same_day_fill = snapshot.get("same_day_fill_summary")
    same_day_fill_obj = same_day_fill if isinstance(same_day_fill, dict) else {}
    pnl_discrepancy = snapshot.get("pnl_discrepancy")
    pnl_discrepancy_obj = pnl_discrepancy if isinstance(pnl_discrepancy, dict) else {}
    pnl_check = str(pnl_discrepancy_obj.get("status") or same_day_fill_obj.get("status") or "n/a")
    if pnl_discrepancy_obj.get("delta") is not None:
        pnl_check = f"{pnl_check}, delta={_fmt_currency(pnl_discrepancy_obj.get('delta'))}"

    learning = snapshot.get("learning")
    learning_obj = learning if isinstance(learning, dict) else {}
    after_hours = learning_obj.get("after_hours")
    after_hours_obj = after_hours if isinstance(after_hours, dict) else {}
    exec_learning = learning_obj.get("execution_learning")
    exec_learning_obj = exec_learning if isinstance(exec_learning, dict) else {}
    exec_autotune = learning_obj.get("execution_autotune")
    exec_autotune_obj = exec_autotune if isinstance(exec_autotune, dict) else {}
    model_liveness = learning_obj.get("model_liveness")
    model_liveness_obj = model_liveness if isinstance(model_liveness, dict) else {}

    return [
        {"type": "header", "text": _slack_plain_text("📘 ai-trading EOD summary")},
        _slack_section(
            (
                f"*Date:* {snapshot.get('report_date') or 'unknown'}\n"
                f"*Readiness Go/No-Go:* {_fmt_gate_status(snapshot.get('go_no_go_gate_passed'))} "
                f"(failed checks: {checks_text})"
            )
        ),
        {"type": "divider"},
        _slack_section(
            "*Day performance*",
            fields=[
                _slack_field("Accounting net PnL", _fmt_currency(snapshot.get("net_pnl"))),
                _slack_field("Same-day fill PnL", _fmt_currency(same_day_fill_obj.get("net_pnl"))),
                _slack_field("PnL check", pnl_check),
                _slack_field("Closed trades", _fmt_count(snapshot.get("closed_trades"))),
                _slack_field("Win rate", _fmt_pct(snapshot.get("win_rate"), digits=1, assume_ratio=True)),
                _slack_field("Profit factor", _fmt_num(snapshot.get("profit_factor"), digits=3)),
                _slack_field("Top loss symbols", losses_text),
            ],
        ),
        _slack_section(
            "*Execution quality*",
            fields=[
                _slack_field("Capture ratio", _fmt_num(snapshot.get("execution_capture_ratio"), digits=3)),
                _slack_field("Slippage drag", f"{_fmt_num(snapshot.get('slippage_drag_bps'), digits=3)} bps"),
                _slack_field("Reject rate", _fmt_pct(snapshot.get("order_reject_rate_pct"), digits=2)),
                _slack_field(
                    "Reconciliation",
                    (
                        f"mismatches={snapshot.get('open_position_reconciliation_mismatch_count')}, "
                        f"max_delta={_fmt_num(snapshot.get('open_position_reconciliation_max_abs_delta_qty'), digits=3)}"
                    ),
                ),
            ],
        ),
        _slack_section(
            "*Runtime health*",
            fields=[
                _slack_field(
                    "Health",
                    f"{snapshot.get('health_status') or 'unknown'} ({snapshot.get('health_reason') or 'n/a'})",
                ),
                _slack_field(
                    "Provider",
                    (
                        f"{snapshot.get('provider_active') or 'n/a'} / "
                        f"{snapshot.get('provider_status') or 'unknown'} "
                        f"(backup={_fmt_bool(snapshot.get('using_backup'))})"
                    ),
                ),
                _slack_field("Broker", snapshot.get("broker_status") or "unknown"),
            ],
        ),
        _slack_section(
            "*Learning models*",
            fields=[
                _slack_field(
                    "Training gate",
                    (
                        f"ready={_fmt_bool(training_gate_obj.get('ready'))}, "
                        f"reason={training_gate_obj.get('reason') or 'n/a'}"
                    ),
                ),
                _slack_field(
                    "After-hours",
                    (
                        f"{after_hours_obj.get('model_name') or 'n/a'} "
                        f"[{after_hours_obj.get('governance_status') or 'n/a'}], "
                        f"rows={after_hours_obj.get('rows') or 'n/a'}"
                    ),
                ),
                _slack_field(
                    "Execution learning",
                    (
                        f"samples={exec_learning_obj.get('samples') or 'n/a'}, "
                        f"fill_rate={_fmt_pct(exec_learning_obj.get('fill_rate'), digits=1, assume_ratio=True)}, "
                        f"slippage={_fmt_num(exec_learning_obj.get('mean_slippage_bps'), digits=3)} bps"
                    ),
                ),
                _slack_field(
                    "Autotune",
                    (
                        f"enabled={_fmt_bool(exec_autotune_obj.get('enabled'))}, "
                        f"bias={exec_autotune_obj.get('profile_bias') or 'n/a'}, "
                        f"samples={exec_autotune_obj.get('sample_count') or 'n/a'}"
                    ),
                ),
                _slack_field(
                    "Liveness",
                    (
                        f"ml_age={_fmt_duration_s(model_liveness_obj.get('ml_age_s'))}, "
                        f"rl_age={_fmt_duration_s(model_liveness_obj.get('rl_age_s'))}"
                    ),
                ),
            ],
        ),
        _slack_context(f"Timestamp: {snapshot.get('timestamp') or 'n/a'}"),
    ]


def tool_runtime_incident_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    snapshot = _collect_runtime_snapshot(args)
    triggers = _evaluate_incident_triggers(snapshot, args)
    return {
        "snapshot": snapshot,
        "triggers": triggers,
        "should_alert": bool(triggers),
        "fingerprint": _incident_fingerprint(snapshot, triggers),
        "incident_signature": _incident_signature(snapshot, triggers),
    }


def tool_notify_incident_channel(args: dict[str, Any]) -> dict[str, Any]:
    snapshot = _collect_runtime_snapshot(args)
    triggers = _evaluate_incident_triggers(snapshot, args)
    fingerprint = _incident_fingerprint(snapshot, triggers)
    signature = _incident_signature(snapshot, triggers)
    force = _bool_arg(args.get("force"), default=False)
    on_change_only = _bool_arg(args.get("on_change_only"), default=True)
    repeat_cooldown_minutes = _incident_repeat_cooldown_minutes(args)
    min_interval_minutes = _incident_min_interval_minutes(args)
    should_alert = bool(triggers) or force
    if not should_alert:
        return {
            "sent": False,
            "reason": "no_incident_triggered",
            "triggers": triggers,
            "fingerprint": fingerprint,
            "snapshot": snapshot,
        }

    state_path = _incident_state_path(args)
    prior = _load_state(state_path)
    prior_fp = str(prior.get("fingerprint") or "")
    prior_signature = str(
        prior.get("incident_signature")
        or prior.get("trigger_signature")
        or ""
    )
    prior_triggers_raw = prior.get("triggers")
    if isinstance(prior_triggers_raw, list):
        prior_triggers = set(_material_incident_triggers_for_dedupe({
            str(item).strip()
            for item in prior_triggers_raw
            if str(item).strip()
        }))
    else:
        prior_triggers = set()
    current_triggers = set(_material_incident_triggers_for_dedupe({
        str(item).strip()
        for item in triggers
        if str(item).strip()
    }))
    trigger_set_changed = current_triggers != prior_triggers
    prior_sent_at = _parse_iso_ts(prior.get("sent_at"))
    now = datetime.now(UTC)
    if (
        not force
        and _requires_health_unavailable_confirmation(snapshot, triggers, args)
    ):
        pending_signature = str(prior.get("pending_health_unavailable_signature") or "")
        pending_seen_at = _parse_iso_ts(prior.get("pending_health_unavailable_seen_at"))
        if pending_signature != signature or pending_seen_at is None:
            pending_payload = dict(prior)
            pending_payload.update(
                {
                    "pending_health_unavailable_signature": signature,
                    "pending_health_unavailable_seen_at": utc_now_iso(),
                    "pending_health_unavailable_triggers": triggers,
                    "pending_health_unavailable_snapshot": snapshot,
                }
            )
            _save_state(state_path, pending_payload)
            return {
                "sent": False,
                "reason": "health_unavailable_confirmation_pending",
                "fingerprint": fingerprint,
                "incident_signature": signature,
                "state_path": str(state_path),
                "triggers": triggers,
                "snapshot": snapshot,
            }

    if on_change_only and prior_sent_at is not None and not force and not trigger_set_changed:
        if repeat_cooldown_minutes > 0:
            elapsed = now - prior_sent_at
            cooldown = timedelta(minutes=repeat_cooldown_minutes)
            if elapsed < cooldown:
                next_eligible_at = (prior_sent_at + cooldown).isoformat().replace("+00:00", "Z")
                return {
                    "sent": False,
                    "reason": "repeat_cooldown_active",
                    "fingerprint": fingerprint,
                    "incident_signature": signature,
                    "state_path": str(state_path),
                    "triggers": triggers,
                    "snapshot": snapshot,
                    "next_eligible_at": next_eligible_at,
                }
    if (
        min_interval_minutes > 0
        and prior_sent_at is not None
        and not force
        and not trigger_set_changed
    ):
        elapsed = now - prior_sent_at
        min_interval = timedelta(minutes=min_interval_minutes)
        if elapsed < min_interval:
            next_eligible_at = (prior_sent_at + min_interval).isoformat().replace("+00:00", "Z")
            return {
                "sent": False,
                "reason": "min_interval_active",
                "fingerprint": fingerprint,
                "incident_signature": signature,
                "state_path": str(state_path),
                "triggers": triggers,
                "snapshot": snapshot,
                "next_eligible_at": next_eligible_at,
                "min_interval_minutes": int(min_interval_minutes),
            }
    if on_change_only and prior_signature and prior_signature == signature and not force:
        if repeat_cooldown_minutes > 0 and prior_sent_at is not None:
            elapsed = now - prior_sent_at
            cooldown = timedelta(minutes=repeat_cooldown_minutes)
            if elapsed < cooldown:
                next_eligible_at = (prior_sent_at + cooldown).isoformat().replace("+00:00", "Z")
                return {
                    "sent": False,
                    "reason": "repeat_cooldown_active",
                    "fingerprint": fingerprint,
                    "incident_signature": signature,
                    "state_path": str(state_path),
                    "triggers": triggers,
                    "snapshot": snapshot,
                    "next_eligible_at": next_eligible_at,
                }
        else:
            return {
                "sent": False,
                "reason": "duplicate_signature",
                "fingerprint": fingerprint,
                "incident_signature": signature,
                "state_path": str(state_path),
                "triggers": triggers,
                "snapshot": snapshot,
            }
    if on_change_only and prior_fp == fingerprint and not force:
        return {
            "sent": False,
            "reason": "duplicate_fingerprint",
            "fingerprint": fingerprint,
            "incident_signature": signature,
            "state_path": str(state_path),
            "triggers": triggers,
            "snapshot": snapshot,
        }

    webhook_url = _slack_webhook_url(args)
    channel = str(args.get("channel") or "").strip()
    payload: dict[str, Any] = {
        "text": _incident_message_text(snapshot, triggers),
        "blocks": _incident_message_blocks(snapshot, triggers),
    }
    if channel:
        payload["channel"] = channel
    timeout_s = float(args.get("timeout_s") or 5.0)
    status_code = _post_slack_message(webhook_url, payload, timeout_s=timeout_s)

    state_payload = {
        "fingerprint": fingerprint,
        "incident_signature": signature,
        "sent_at": utc_now_iso(),
        "triggers": triggers,
        "snapshot": snapshot,
        "status_code": status_code,
    }
    _save_state(state_path, state_payload)
    return {
        "sent": True,
        "status_code": status_code,
        "fingerprint": fingerprint,
        "incident_signature": signature,
        "state_path": str(state_path),
        "triggers": triggers,
        "snapshot": snapshot,
    }


def tool_runtime_eod_summary_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    snapshot = _collect_eod_summary_snapshot(args)
    require_market_closed = _bool_arg(
        args.get("require_market_closed"),
        default=_bool_arg(os.getenv("AI_TRADING_SLACK_EOD_REQUIRE_MARKET_CLOSED"), default=True),
    )
    health_reason = str(snapshot.get("health_reason") or "").strip().lower()
    market_closed = health_reason == "market_closed"
    report_date = str(snapshot.get("report_date") or "").strip()
    training_gate = _after_hours_training_gate(report_date, args)
    block_on_training_gate = _block_eod_on_training_gate(args)
    training_ready = bool(training_gate.get("ready", False))
    snapshot["training_gate"] = training_gate
    return {
        "snapshot": snapshot,
        "fingerprint": _eod_fingerprint(snapshot),
        "market_closed": market_closed,
        "report_date": report_date,
        "training_gate": training_gate,
        "block_on_training_gate": block_on_training_gate,
        "eligible": (
            (not require_market_closed or market_closed)
            and bool(report_date)
            and (training_ready or not block_on_training_gate)
        ),
    }


def tool_notify_eod_summary(args: dict[str, Any]) -> dict[str, Any]:
    snapshot = _collect_eod_summary_snapshot(args)
    report_date = str(snapshot.get("report_date") or "").strip()
    force = _bool_arg(args.get("force"), default=False)
    require_market_closed = _bool_arg(
        args.get("require_market_closed"),
        default=_bool_arg(os.getenv("AI_TRADING_SLACK_EOD_REQUIRE_MARKET_CLOSED"), default=True),
    )
    health_reason = str(snapshot.get("health_reason") or "").strip().lower()
    market_closed = health_reason == "market_closed"
    training_gate = _after_hours_training_gate(report_date, args)
    block_on_training_gate = _block_eod_on_training_gate(args)
    snapshot["training_gate"] = training_gate

    if require_market_closed and not market_closed and not force:
        return {
            "sent": False,
            "reason": "market_not_closed",
            "report_date": report_date,
            "block_on_training_gate": block_on_training_gate,
            "training_gate": training_gate,
            "snapshot": snapshot,
        }

    if not report_date and not force:
        return {
            "sent": False,
            "reason": "missing_report_date",
            "block_on_training_gate": block_on_training_gate,
            "training_gate": training_gate,
            "snapshot": snapshot,
        }

    if block_on_training_gate and not bool(training_gate.get("ready", False)) and not force:
        return {
            "sent": False,
            "reason": "after_hours_training_not_complete",
            "report_date": report_date,
            "block_on_training_gate": block_on_training_gate,
            "training_gate": training_gate,
            "snapshot": snapshot,
        }

    state_path = _eod_state_path(args)
    prior = _load_state(state_path)
    prior_report_date = str(prior.get("report_date") or "").strip()
    if prior_report_date == report_date and not force:
        return {
            "sent": False,
            "reason": "already_sent_for_report_date",
            "report_date": report_date,
            "state_path": str(state_path),
            "block_on_training_gate": block_on_training_gate,
            "training_gate": training_gate,
            "snapshot": snapshot,
        }

    webhook_url = _slack_webhook_url(args)
    channel = str(args.get("channel") or os.getenv("AI_TRADING_SLACK_EOD_CHANNEL", "")).strip()
    payload: dict[str, Any] = {
        "text": _eod_message_text(snapshot),
        "blocks": _eod_message_blocks(snapshot),
    }
    if channel:
        payload["channel"] = channel
    timeout_s = float(args.get("timeout_s") or 5.0)
    status_code = _post_slack_message(webhook_url, payload, timeout_s=timeout_s)

    fingerprint = _eod_fingerprint(snapshot)
    _save_state(
        state_path,
        {
            "fingerprint": fingerprint,
            "report_date": report_date,
            "sent_at": utc_now_iso(),
            "snapshot": snapshot,
            "status_code": status_code,
        },
    )
    return {
        "sent": True,
        "status_code": status_code,
        "fingerprint": fingerprint,
        "report_date": report_date,
        "state_path": str(state_path),
        "block_on_training_gate": block_on_training_gate,
        "training_gate": training_gate,
        "snapshot": snapshot,
    }


def tool_clear_incident_state(args: dict[str, Any]) -> dict[str, Any]:
    path = _incident_state_path(args)
    existed = path.exists()
    if existed:
        path.unlink()
    return {"cleared": existed, "state_path": str(path)}


def tool_clear_eod_summary_state(args: dict[str, Any]) -> dict[str, Any]:
    path = _eod_state_path(args)
    existed = path.exists()
    if existed:
        path.unlink()
    return {"cleared": existed, "state_path": str(path)}


TOOLS = {
    "runtime_incident_snapshot": tool_runtime_incident_snapshot,
    "notify_incident_channel": tool_notify_incident_channel,
    "clear_incident_state": tool_clear_incident_state,
    "runtime_eod_summary_snapshot": tool_runtime_eod_summary_snapshot,
    "notify_eod_summary": tool_notify_eod_summary,
    "clear_eod_summary_state": tool_clear_eod_summary_state,
}

TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "runtime_incident_snapshot",
        "description": "Build go/no-go + execution-capture + health snapshot and incident triggers.",
    },
    {
        "name": "notify_incident_channel",
        "description": "Push incident snapshot to Slack webhook with dedupe-aware alerting.",
    },
    {
        "name": "clear_incident_state",
        "description": "Clear saved Slack incident dedupe state.",
    },
    {
        "name": "runtime_eod_summary_snapshot",
        "description": "Build end-of-day KPI + learning-model snapshot for Slack delivery.",
    },
    {
        "name": "notify_eod_summary",
        "description": "Send one Slack end-of-day summary message per report_date.",
    },
    {
        "name": "clear_eod_summary_state",
        "description": "Clear saved Slack EOD dedupe state.",
    },
]


def main(argv: list[str] | None = None) -> int:
    result = run_tool_server(
        argv=argv,
        description="Slack incident MCP server",
        tools=TOOLS,
        specs=TOOL_SPECS,
        server_name="trading_slack_alerts",
    )
    return int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
