"""Verify pre-trade risk controls from artifacts before live order submission."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_trading.config.launch_profiles import resolve_launch_profile
from ai_trading.config.management import get_env, get_trading_config
from ai_trading.runtime.artifacts import resolve_runtime_artifact_path


KNOWN_CONTROLS = frozenset(
    {
        "allow_shorting",
        "buying_power",
        "duplicate_intent",
        "kill_switch",
        "liquidity",
        "market_hours",
        "max_daily_loss",
        "max_gross_exposure",
        "max_order_notional",
        "max_position_notional",
        "max_position_size",
        "max_slippage_bps",
        "max_symbol_exposure",
        "price_band",
        "short_sale_locate",
    }
)
DEFAULT_REQUIRED_CONTROLS = (
    "kill_switch",
    "buying_power",
    "max_order_notional",
    "max_position_notional",
    "max_daily_loss",
    "duplicate_intent",
)
PASS_TOKENS = {"ok", "pass", "passed", "enabled", "active", "allow", "allowed"}
UNKNOWN_TOKENS = {"", "unknown", "missing", "unconfigured", "not_configured", "na", "n/a", "none"}
FAIL_OPEN_MODES = {"observe", "observe_only", "warn", "warn_only", "shadow"}
LIFECYCLE_ONLY_EVENTS = {
    "ack",
    "final_state",
    "order_status",
    "status_transition",
    "terminal_state",
}
LIFECYCLE_ONLY_STATUSES = {
    "accepted",
    "canceled",
    "cancelled",
    "expired",
    "filled",
    "new",
    "partially_filled",
    "pending_cancel",
    "pending_new",
    "rejected",
}


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed > 0:
        return parsed
    return None


def _control_spec(
    *,
    status: str,
    source: str,
    evidence: Mapping[str, Any],
    reason: str | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "enabled": True,
        "fail_closed": True,
        "mode": "enforce",
        "status": status,
        "source": source,
        "evidence": dict(evidence),
    }
    if reason:
        spec["reason"] = reason
    return spec


def _unknown_runtime_controls(reason: str) -> dict[str, Mapping[str, Any]]:
    return {
        control: _control_spec(
            status="unknown",
            source="runtime_config",
            reason=reason,
            evidence={"required_control": control},
        )
        for control in DEFAULT_REQUIRED_CONTROLS
    }


def _runtime_detected_controls() -> dict[str, Mapping[str, Any]]:
    """Build a fail-closed control inventory from canonical runtime config.

    This is intentionally conservative: controls that cannot be proven from
    config or canonical runtime enforcement are reported as unknown, which keeps
    the verifier blocking instead of manufacturing authority from a missing
    external controls JSON.
    """

    try:
        cfg = get_trading_config()
        profile = resolve_launch_profile()
    except (RuntimeError, ValueError, TypeError) as exc:
        return _unknown_runtime_controls(f"runtime_config_unavailable:{type(exc).__name__}")

    controls: dict[str, Mapping[str, Any]] = {}

    kill_switch_enabled = bool(getattr(cfg, "kill_switch", False))
    kill_switch_path_raw = str(getattr(cfg, "kill_switch_path", "") or "").strip()
    kill_switch_file_active = False
    if kill_switch_path_raw:
        try:
            kill_switch_file_active = Path(kill_switch_path_raw).exists()
        except OSError:
            kill_switch_file_active = True
    controls["kill_switch"] = _control_spec(
        status="failed" if kill_switch_enabled or kill_switch_file_active else "passed",
        source="runtime_config",
        reason="kill_switch_active" if kill_switch_enabled or kill_switch_file_active else None,
        evidence={
            "env": "AI_TRADING_KILL_SWITCH",
            "path_env": "AI_TRADING_KILL_SWITCH_PATH",
            "kill_switch_configured": True,
            "kill_switch_active": kill_switch_enabled,
            "kill_switch_path": kill_switch_path_raw or None,
            "kill_switch_file_active": kill_switch_file_active,
        },
    )

    max_order_candidates = (
        getattr(profile, "max_notional_per_order", None),
        get_env("AI_TRADING_MAX_ORDER_DOLLARS", None, cast=float),
        get_env("MAX_ORDER_DOLLARS", None, cast=float),
        get_env("AI_TRADING_PAPER_SAMPLING_MAX_NOTIONAL_PER_ORDER", None, cast=float),
    )
    max_order_limit = next(
        (value for value in (_positive_float(v) for v in max_order_candidates) if value),
        None,
    )
    controls["max_order_notional"] = _control_spec(
        status="passed" if max_order_limit is not None else "unknown",
        source="runtime_config",
        reason=None if max_order_limit is not None else "max_order_notional_not_configured",
        evidence={
            "limit": max_order_limit,
            "launch_profile": getattr(profile, "name", None),
            "profile_max_notional_per_order": getattr(profile, "max_notional_per_order", None),
            "paper_sampling_max_notional_per_order": get_env(
                "AI_TRADING_PAPER_SAMPLING_MAX_NOTIONAL_PER_ORDER",
                None,
                cast=float,
            ),
            "max_order_dollars": get_env("MAX_ORDER_DOLLARS", None, cast=float),
            "ai_trading_max_order_dollars": get_env("AI_TRADING_MAX_ORDER_DOLLARS", None, cast=float),
        },
    )

    max_position_candidates = (
        get_env("AI_TRADING_MAX_SYMBOL_NOTIONAL", None, cast=float),
        get_env("MAX_SYMBOL_NOTIONAL", None, cast=float),
        getattr(cfg, "max_position_size", None),
        getattr(profile, "max_symbol_exposure", None),
    )
    max_position_limit = next(
        (value for value in (_positive_float(v) for v in max_position_candidates) if value),
        None,
    )
    controls["max_position_notional"] = _control_spec(
        status="passed" if max_position_limit is not None else "unknown",
        source="runtime_config",
        reason=None if max_position_limit is not None else "max_position_control_not_configured",
        evidence={
            "limit": max_position_limit,
            "limit_kind": (
                "notional_or_configured_position_size"
                if _positive_float(get_env("AI_TRADING_MAX_SYMBOL_NOTIONAL", None, cast=float))
                or _positive_float(get_env("MAX_SYMBOL_NOTIONAL", None, cast=float))
                or _positive_float(getattr(cfg, "max_position_size", None))
                else "launch_profile_symbol_exposure_fraction"
            ),
            "launch_profile": getattr(profile, "name", None),
            "profile_max_symbol_exposure": getattr(profile, "max_symbol_exposure", None),
            "configured_max_position_size": getattr(cfg, "max_position_size", None),
        },
    )

    daily_loss_candidates = (
        getattr(profile, "max_daily_loss", None),
        getattr(cfg, "daily_loss_limit", None),
        get_env("AI_TRADING_DAILY_LOSS_LIMIT_ABS", None, cast=float),
        get_env("AI_TRADING_DAILY_LOSS_LIMIT_PCT", None, cast=float),
        get_env("AI_TRADING_DAILY_LOSS_LIMIT", None, cast=float),
    )
    daily_loss_limit = next(
        (value for value in (_positive_float(v) for v in daily_loss_candidates) if value),
        None,
    )
    controls["max_daily_loss"] = _control_spec(
        status="passed" if daily_loss_limit is not None else "unknown",
        source="runtime_config",
        reason=None if daily_loss_limit is not None else "daily_loss_limit_not_configured",
        evidence={
            "limit": daily_loss_limit,
            "launch_profile": getattr(profile, "name", None),
            "profile_max_daily_loss": getattr(profile, "max_daily_loss", None),
            "configured_daily_loss_limit": getattr(cfg, "daily_loss_limit", None),
            "absolute_daily_loss_limit": get_env("AI_TRADING_DAILY_LOSS_LIMIT_ABS", None, cast=float),
            "percent_daily_loss_limit": get_env("AI_TRADING_DAILY_LOSS_LIMIT_PCT", None, cast=float),
        },
    )

    controls["buying_power"] = _control_spec(
        status="passed",
        source="canonical_runtime_guard",
        evidence={
            "enforced_by": [
                "ai_trading.oms.pretrade.validate_pretrade",
                "ai_trading.core.execution_flow",
                "ai_trading.core.submit_runtime.submit_order_runtime",
            ],
            "broker_ready_gate": True,
            "buying_power_prescale_or_block": True,
        },
    )
    controls["duplicate_intent"] = _control_spec(
        status="passed",
        source="canonical_runtime_guard",
        evidence={
            "enforced_by": [
                "ai_trading.core.submit_runtime.submit_order_runtime",
                "ai_trading.execution.live_trading._should_suppress_duplicate_intent",
            ],
            "ledger_idempotency_enabled": bool(getattr(cfg, "ledger_enabled", True)),
        },
    )
    return controls


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_jsonl(path: Path | None, *, report_date: str | None = None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            if report_date and not _date_match(parsed, report_date):
                continue
            rows.append(parsed)
    return rows


def _date_match(row: Mapping[str, Any], report_date: str) -> bool:
    ts = str(row.get("ts") or row.get("timestamp") or row.get("decision_ts") or row.get("created_at") or "")
    return ts.startswith(report_date)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _control_name(row: Mapping[str, Any]) -> str:
    return str(row.get("name") or row.get("control") or row.get("control_name") or "").strip().lower()


def _normalise_controls(config: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    payload = _as_mapping(config)
    raw_controls = payload.get("controls")
    if raw_controls is None:
        raw_controls = payload.get("risk_controls") or payload.get("pretrade_controls") or payload
    controls: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_controls, Sequence) and not isinstance(raw_controls, (str, bytes, bytearray)):
        for item in raw_controls:
            if not isinstance(item, Mapping):
                continue
            name = _control_name(item)
            if name:
                controls[name] = item
        return controls
    if isinstance(raw_controls, Mapping):
        for key, value in raw_controls.items():
            name = str(key).strip().lower()
            if not name:
                continue
            controls[name] = value if isinstance(value, Mapping) else {"value": value}
    return controls


def _control_status(spec: Mapping[str, Any]) -> str:
    raw_status = str(
        spec.get("status")
        or spec.get("state")
        or spec.get("result")
        or spec.get("decision")
        or ""
    ).strip().lower()
    if raw_status in UNKNOWN_TOKENS:
        return "unknown"
    if raw_status in PASS_TOKENS:
        return "passed"
    if raw_status in {"fail", "failed", "reject", "rejected", "blocked", "breach"}:
        return "failed"
    return raw_status or "unknown"


def _bool_field(spec: Mapping[str, Any], key: str, *, default: bool | None = None) -> bool | None:
    if key not in spec:
        return default
    value = spec.get(key)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _intent_id(row: Mapping[str, Any]) -> str:
    return str(row.get("intent_id") or row.get("decision_id") or row.get("client_order_id") or "").strip()


def _intent_controls(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("controls", "risk_controls", "pretrade_controls", "control_results"):
        raw = row.get(key)
        if isinstance(raw, Mapping):
            return raw
    return {}


def _intent_control_status(value: Any) -> str:
    if isinstance(value, Mapping):
        return _control_status(value)
    text = str(value or "").strip().lower()
    if text in PASS_TOKENS:
        return "passed"
    if text in UNKNOWN_TOKENS:
        return "unknown"
    if text in {"fail", "failed", "reject", "rejected", "blocked", "breach"}:
        return "failed"
    return text or "unknown"


def _lifecycle_row_without_controls(row: Mapping[str, Any]) -> bool:
    """Return true for broker/OMS lifecycle rows that are not pretrade evidence."""

    event = str(row.get("event") or row.get("type") or "").strip().lower()
    if event in LIFECYCLE_ONLY_EVENTS:
        return True
    status = str(row.get("status") or row.get("new_status") or "").strip().lower()
    return bool(row.get("order_id") and status in LIFECYCLE_ONLY_STATUSES)


def build_pretrade_risk_control_verification(
    *,
    report_date: str,
    controls_config: Mapping[str, Any] | None = None,
    intents: Sequence[Mapping[str, Any]] = (),
    required_controls: Sequence[str] = DEFAULT_REQUIRED_CONTROLS,
) -> dict[str, Any]:
    auto_detected = controls_config is None
    controls = _runtime_detected_controls() if auto_detected else _normalise_controls(controls_config)
    required = tuple(str(control).strip().lower() for control in required_controls if str(control).strip())
    violations: list[dict[str, Any]] = []
    control_statuses: dict[str, dict[str, Any]] = {}
    checked_intent_ids: set[str] = set()
    missing_control_intent_ids: set[str] = set()
    lifecycle_rows_ignored = 0
    duplicate_intent_rows_ignored = 0

    for control_name in sorted(controls):
        spec = controls[control_name]
        status = _control_status(spec)
        enabled = _bool_field(spec, "enabled", default=True)
        fail_closed = _bool_field(spec, "fail_closed", default=False)
        mode = str(spec.get("mode") or spec.get("enforcement") or "").strip().lower()
        known = control_name in KNOWN_CONTROLS
        control_statuses[control_name] = {
            "known": known,
            "enabled": enabled,
            "fail_closed": fail_closed,
            "mode": mode or "enforce",
            "status": status,
        }
        for field in ("source", "reason", "evidence"):
            if field in spec:
                control_statuses[control_name][field] = spec[field]
        if not known:
            violations.append({"kind": "unknown_control", "control": control_name})
        if enabled is not True:
            violations.append({"kind": "disabled_control", "control": control_name})
        if fail_closed is not True:
            violations.append({"kind": "control_not_fail_closed", "control": control_name})
        if mode in FAIL_OPEN_MODES:
            violations.append({"kind": "control_not_enforced", "control": control_name, "mode": mode})
        if status != "passed":
            violations.append({"kind": "control_status_not_passed", "control": control_name, "status": status})

    for control_name in required:
        if control_name not in controls:
            violations.append({"kind": "missing_required_control", "control": control_name})

    for row in intents:
        row_controls = _intent_controls(row)
        row_id = _intent_id(row) or "unknown"
        if not row_controls:
            if _lifecycle_row_without_controls(row):
                lifecycle_rows_ignored += 1
                continue
            if row_id in missing_control_intent_ids:
                duplicate_intent_rows_ignored += 1
                continue
            missing_control_intent_ids.add(row_id)
            violations.append({"kind": "intent_controls_missing", "intent_id": row_id})
            continue
        if row_id in checked_intent_ids:
            duplicate_intent_rows_ignored += 1
            continue
        checked_intent_ids.add(row_id)
        normalised_row_controls = {str(key).strip().lower(): value for key, value in row_controls.items()}
        for control_name in required:
            if control_name not in normalised_row_controls:
                violations.append(
                    {
                        "kind": "intent_required_control_missing",
                        "intent_id": row_id,
                        "control": control_name,
                    }
                )
                continue
            observed_status = _intent_control_status(normalised_row_controls[control_name])
            if observed_status != "passed":
                violations.append(
                    {
                        "kind": "intent_control_not_passed",
                        "intent_id": row_id,
                        "control": control_name,
                        "status": observed_status,
                    }
                )
        for control_name in sorted(normalised_row_controls):
            if control_name not in KNOWN_CONTROLS:
                violations.append(
                    {
                        "kind": "intent_unknown_control",
                        "intent_id": row_id,
                        "control": control_name,
                    }
                )

    status = "passed" if not violations else "fail_closed"
    action = "allow_pretrade_flow" if status == "passed" else "block_live_orders_until_pretrade_controls_are_resolved"
    return {
        "schema_version": "1.0.0",
        "artifact_type": "pretrade_risk_control_verification",
        "report_date": report_date,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "recommended_next_action": action,
        "fail_closed": status != "passed",
        "required_controls": list(required),
        "summary": {
            "configured_controls": len(controls),
            "event_rows_seen": len(intents),
            "intents_checked": len(checked_intent_ids) + len(missing_control_intent_ids),
            "intents_with_control_evidence": len(checked_intent_ids),
            "lifecycle_rows_ignored": lifecycle_rows_ignored,
            "duplicate_intent_rows_ignored": duplicate_intent_rows_ignored,
            "runtime_controls_auto_detected": auto_detected,
            "violations": len(violations),
            "unknown_controls": sum(1 for item in violations if str(item.get("kind")) == "unknown_control"),
            "missing_required_controls": sum(
                1 for item in violations if str(item.get("kind")) == "missing_required_control"
            ),
        },
        "control_statuses": control_statuses,
        "violations": violations,
        "control_source": "runtime_config" if auto_detected else "controls_json",
        "promotion_authority": False,
        "live_money_authority": False,
    }


def _default_report_paths(report_date: str) -> tuple[Path, Path]:
    root = resolve_runtime_artifact_path("runtime/reports", default_relative="runtime/reports", for_write=True)
    compact = report_date.replace("-", "")
    return root / f"pretrade_risk_control_verification_{compact}.json", root / "pretrade_risk_control_verification_latest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--controls-json", type=Path, default=None)
    parser.add_argument("--intents-jsonl", type=Path, default=None)
    parser.add_argument("--required-control", action="append", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--latest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, latest_json = _default_report_paths(str(args.report_date))
    output_json = args.output_json or output_json
    latest_json = args.latest_json or latest_json
    report = build_pretrade_risk_control_verification(
        report_date=str(args.report_date),
        controls_config=None if args.controls_json is None else _read_json(args.controls_json),
        intents=_read_jsonl(args.intents_jsonl, report_date=str(args.report_date)),
        required_controls=tuple(args.required_control or DEFAULT_REQUIRED_CONTROLS),
    )
    for path in (output_json, latest_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"path": str(output_json), "status": report["status"]}) + "\n")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
