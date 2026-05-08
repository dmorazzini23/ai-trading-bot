"""Verify pre-trade risk controls from artifacts before live order submission."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

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


def build_pretrade_risk_control_verification(
    *,
    report_date: str,
    controls_config: Mapping[str, Any] | None = None,
    intents: Sequence[Mapping[str, Any]] = (),
    required_controls: Sequence[str] = DEFAULT_REQUIRED_CONTROLS,
) -> dict[str, Any]:
    controls = _normalise_controls(controls_config)
    required = tuple(str(control).strip().lower() for control in required_controls if str(control).strip())
    violations: list[dict[str, Any]] = []
    control_statuses: dict[str, dict[str, Any]] = {}

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
            violations.append({"kind": "intent_controls_missing", "intent_id": row_id})
            continue
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
            "intents_checked": len(intents),
            "violations": len(violations),
            "unknown_controls": sum(1 for item in violations if str(item.get("kind")) == "unknown_control"),
            "missing_required_controls": sum(
                1 for item in violations if str(item.get("kind")) == "missing_required_control"
            ),
        },
        "control_statuses": control_statuses,
        "violations": violations,
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
        controls_config=_read_json(args.controls_json),
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
